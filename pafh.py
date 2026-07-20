from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator


def validate_probability_vector(
    probabilities: np.ndarray,
) -> None:
    """Validate a discrete probability distribution."""
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 1:
        raise ValueError(
            "Probability vector must be one-dimensional."
        )

    if probabilities.size == 0:
        raise ValueError(
            "Probability vector cannot be empty."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Probability vector must contain finite values."
        )

    if np.any(probabilities < 0):
        raise ValueError(
            "Probabilities cannot be negative."
        )

    if not np.isclose(
        np.sum(probabilities),
        1.0,
    ):
        raise ValueError(
            "Probabilities must sum to 1."
        )


def softmax_probabilities(
    utilities: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Convert channel utilities into probabilities:

        p_k = exp(beta * U_k)
              -----------------
              sum_i exp(beta * U_i)

    beta = 0:
        Uniform random hopping.

    Large beta:
        Strong preference for high-utility channels.
    """
    utilities = np.asarray(
        utilities,
        dtype=float,
    )

    if utilities.ndim != 1:
        raise ValueError(
            "utilities must be one-dimensional."
        )

    if utilities.size == 0:
        raise ValueError(
            "utilities cannot be empty."
        )

    if not np.all(np.isfinite(utilities)):
        raise ValueError(
            "utilities must contain finite values."
        )

    if beta < 0:
        raise ValueError(
            "beta cannot be negative."
        )

    scaled_utilities = beta * utilities

    # Numerical stabilization:
    # exp(x - max(x)) avoids overflow.
    scaled_utilities -= np.max(scaled_utilities)

    exponentials = np.exp(scaled_utilities)

    probabilities = exponentials / np.sum(
        exponentials
    )

    return probabilities


def normalized_utility_probabilities(
    utilities: np.ndarray,
) -> np.ndarray:
    """
    Direct utility normalization baseline:

        p_k = U_k / sum_i U_i
    """
    utilities = np.asarray(
        utilities,
        dtype=float,
    )

    if utilities.ndim != 1:
        raise ValueError(
            "utilities must be one-dimensional."
        )

    if not np.all(np.isfinite(utilities)):
        raise ValueError(
            "utilities must be finite."
        )

    if np.any(utilities < 0):
        raise ValueError(
            "utilities cannot be negative."
        )

    total_utility = np.sum(utilities)

    if np.isclose(total_utility, 0.0):
        return np.ones_like(utilities) / len(
            utilities
        )

    return utilities / total_utility


def distribution_entropy(
    probabilities: np.ndarray,
    normalized: bool = False,
) -> float:
    """
    Compute Shannon entropy:

        H(p) = -sum_k p_k log(p_k)

    If normalized=True, divide by log(K), making entropy
    lie between 0 and 1.
    """
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    validate_probability_vector(probabilities)

    positive_probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        positive_probabilities
        * np.log(positive_probabilities)
    )

    if normalized:
        num_channels = len(probabilities)

        if num_channels == 1:
            return 0.0

        entropy /= np.log(num_channels)

    return float(entropy)


@dataclass
class PAFHPolicy:
    """
    Probabilistic Anti-Jamming Frequency Hopping policy.

    The policy stores one probability for every channel.
    Probabilities are updated from long-term utilities
    using entropy-regularized softmax.
    """

    num_channels: int
    beta: float = 3.0

    probabilities: np.ndarray = field(
        init=False,
    )

    def __post_init__(self) -> None:
        if self.num_channels <= 0:
            raise ValueError(
                "num_channels must be greater than zero."
            )

        if self.beta < 0:
            raise ValueError(
                "beta cannot be negative."
            )

        self.reset()

    def update(
        self,
        utilities: np.ndarray,
    ) -> np.ndarray:
        """
        Recalculate hopping probabilities from utilities.
        """
        utilities = np.asarray(
            utilities,
            dtype=float,
        )

        if utilities.shape != (
            self.num_channels,
        ):
            raise ValueError(
                "utilities must have one value "
                "for every channel."
            )

        self.probabilities = (
            softmax_probabilities(
                utilities=utilities,
                beta=self.beta,
            )
        )

        return self.probabilities.copy()

    def select_channel(
        self,
        rng: Generator,
    ) -> int:
        """
        Sample one hopping channel according to the
        current probability distribution.
        """
        return int(
            rng.choice(
                self.num_channels,
                p=self.probabilities,
            )
        )

    def entropy(
        self,
        normalized: bool = True,
    ) -> float:
        """Return entropy of the current hopping policy."""
        return distribution_entropy(
            probabilities=self.probabilities,
            normalized=normalized,
        )

    def reset(self) -> None:
        """Reset policy to uniform random hopping."""
        self.probabilities = (
            np.ones(self.num_channels)
            / self.num_channels
        )