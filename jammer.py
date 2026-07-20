from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from channel import sample_jammer_reader_power


def validate_probability_vector(probabilities: np.ndarray) -> None:
    """Validate a discrete probability vector."""
    if probabilities.ndim != 1:
        raise ValueError("Probability vector must be one-dimensional.")

    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative.")

    if not np.isclose(np.sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1.")


@dataclass
class ProbabilisticJammer:
    """
    Jammer selects one channel according to probability vector q.

    The legitimate transmission is jammed only when the jammer
    selects the same channel as the backscatter tag.
    """

    power_watt: float
    channel_probabilities: np.ndarray
    jammer_reader_path_loss: float = 1.0

    def __post_init__(self) -> None:
        self.channel_probabilities = np.asarray(
            self.channel_probabilities,
            dtype=float,
        )

        validate_probability_vector(self.channel_probabilities)

        if self.power_watt < 0:
            raise ValueError("Jammer power cannot be negative.")

        if self.jammer_reader_path_loss <= 0:
            raise ValueError(
                "jammer_reader_path_loss must be positive."
            )

    def generate_interference(
        self,
        rng: Generator,
        selected_channel: int,
    ) -> tuple[float, bool, int]:
        """
        Return:
            interference power,
            whether the legitimate channel was jammed,
            jammer-selected channel.
        """
        num_channels = len(self.channel_probabilities)

        if not 0 <= selected_channel < num_channels:
            raise ValueError("selected_channel is out of range.")

        jammer_channel = int(
            rng.choice(
                num_channels,
                p=self.channel_probabilities,
            )
        )

        was_jammed = jammer_channel == selected_channel

        if not was_jammed:
            return 0.0, False, jammer_channel

        channel_power_gain = sample_jammer_reader_power(
            rng=rng,
            jammer_reader_path_loss=self.jammer_reader_path_loss,
        )

        interference = self.power_watt * channel_power_gain

        return float(interference), True, jammer_channel


@dataclass
class ReactiveJammer:
    """
    Reactive jammer detects the legitimate transmission
    with probability Pd and jams the currently selected channel.
    """

    power_watt: float
    detection_probability: float
    jammer_reader_path_loss: float = 1.0

    def __post_init__(self) -> None:
        if self.power_watt < 0:
            raise ValueError("Jammer power cannot be negative.")

        if not 0 <= self.detection_probability <= 1:
            raise ValueError(
                "detection_probability must be between 0 and 1."
            )

        if self.jammer_reader_path_loss <= 0:
            raise ValueError(
                "jammer_reader_path_loss must be positive."
            )

    def generate_interference(
        self,
        rng: Generator,
    ) -> tuple[float, bool]:
        """
        Return:
            interference power,
            whether the jammer detected and attacked.
        """
        detected = (
            rng.random() < self.detection_probability
        )

        if not detected:
            return 0.0, False

        channel_power_gain = sample_jammer_reader_power(
            rng=rng,
            jammer_reader_path_loss=self.jammer_reader_path_loss,
        )

        interference = self.power_watt * channel_power_gain

        return float(interference), True