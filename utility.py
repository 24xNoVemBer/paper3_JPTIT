from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


UtilityMode = Literal[
    "mean_sinr",
    "success_rate",
    "hybrid",
]


@dataclass
class ChannelUtilityEstimator:
    """
    Maintain long-term SINR observations for each channel
    and compute channel utility values.

    The observation history is stored using a sliding window.
    """

    num_channels: int
    sinr_threshold_linear: float
    window_size: int = 500
    hybrid_weight: float = 0.5

    _sinr_history: list[deque[float]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.num_channels <= 0:
            raise ValueError(
                "num_channels must be greater than zero."
            )

        if self.sinr_threshold_linear <= 0:
            raise ValueError(
                "sinr_threshold_linear must be positive."
            )

        if self.window_size <= 0:
            raise ValueError(
                "window_size must be greater than zero."
            )

        if not 0.0 <= self.hybrid_weight <= 1.0:
            raise ValueError(
                "hybrid_weight must be between 0 and 1."
            )

        self._sinr_history = [
            deque(maxlen=self.window_size)
            for _ in range(self.num_channels)
        ]

    def update(
        self,
        channel: int,
        sinr_linear: float,
    ) -> None:
        """
        Store one SINR observation for a channel.
        """
        if not 0 <= channel < self.num_channels:
            raise IndexError(
                "channel index is out of range."
            )

        if sinr_linear < 0:
            raise ValueError(
                "sinr_linear cannot be negative."
            )

        if not np.isfinite(sinr_linear):
            raise ValueError(
                "sinr_linear must be finite."
            )

        self._sinr_history[channel].append(
            float(sinr_linear)
        )

    def observation_counts(self) -> np.ndarray:
        """
        Return the number of observations available
        for each channel.
        """
        return np.asarray(
            [
                len(history)
                for history in self._sinr_history
            ],
            dtype=int,
        )

    def mean_sinr(self) -> np.ndarray:
        """
        Compute the mean linear SINR for each channel.

        Channels without observations return NaN.
        """
        values = []

        for history in self._sinr_history:
            if history:
                values.append(float(np.mean(history)))
            else:
                values.append(np.nan)

        return np.asarray(values, dtype=float)

    def success_rates(self) -> np.ndarray:
        """
        Compute the successful transmission ratio
        for each channel.

        A transmission is successful when:

            SINR >= threshold
        """
        rates = []

        for history in self._sinr_history:
            if not history:
                rates.append(np.nan)
                continue

            samples = np.asarray(
                history,
                dtype=float,
            )

            success_rate = np.mean(
                samples >= self.sinr_threshold_linear
            )

            rates.append(float(success_rate))

        return np.asarray(rates, dtype=float)

    def outage_rates(self) -> np.ndarray:
        """
        Compute outage ratio for every channel.
        """
        success = self.success_rates()

        return np.where(
            np.isfinite(success),
            1.0 - success,
            np.nan,
        )

    @staticmethod
    def _min_max_normalize(
        values: np.ndarray,
        neutral_value: float = 0.5,
    ) -> np.ndarray:
        """
        Normalize finite values to [0, 1].

        Channels without observations receive a neutral
        score so they are not completely ignored.
        """
        values = np.asarray(values, dtype=float)

        normalized = np.full_like(
            values,
            fill_value=neutral_value,
            dtype=float,
        )

        finite_mask = np.isfinite(values)

        if not np.any(finite_mask):
            return normalized

        finite_values = values[finite_mask]

        minimum = np.min(finite_values)
        maximum = np.max(finite_values)

        if np.isclose(maximum, minimum):
            normalized[finite_mask] = neutral_value
            return normalized

        normalized[finite_mask] = (
            finite_values - minimum
        ) / (
            maximum - minimum
        )

        return normalized

    def utilities(
        self,
        mode: UtilityMode = "hybrid",
    ) -> np.ndarray:
        """
        Compute one utility value for each channel.

        Modes
        -----
        mean_sinr:
            Utility based only on average SINR.

        success_rate:
            Utility based only on successful transmission
            probability.

        hybrid:
            Weighted combination of normalized mean SINR
            and successful transmission probability.
        """
        mean_sinr_values = self.mean_sinr()
        success_rate_values = self.success_rates()

        # log1p reduces the effect of extremely large
        # linear SINR values.
        sinr_scores = self._min_max_normalize(
            np.log1p(mean_sinr_values)
        )

        success_scores = np.where(
            np.isfinite(success_rate_values),
            success_rate_values,
            0.5,
        )

        if mode == "mean_sinr":
            return sinr_scores

        if mode == "success_rate":
            return success_scores

        if mode == "hybrid":
            return (
                self.hybrid_weight * sinr_scores
                + (1.0 - self.hybrid_weight)
                * success_scores
            )

        raise ValueError(
            f"Unknown utility mode: {mode}"
        )

    def reset(self) -> None:
        """
        Remove all stored observations.
        """
        for history in self._sinr_history:
            history.clear()