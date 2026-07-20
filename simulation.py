from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from channel import sample_dyadic_backscatter_power
from config import (
    SimulationConfig,
    db_to_linear,
)
from jammer import ProbabilisticJammer
from metrics import (
    achievable_rate,
    average_throughput,
    outage_probability,
    successful_throughput,
)
from pafh import PAFHPolicy, distribution_entropy
from utility import ChannelUtilityEstimator


SchemeName = Literal["RCH", "PAFH"]


@dataclass
class SimulationResult:
    scheme: str

    average_throughput_bit_s: float
    successful_throughput_bit_s: float
    outage_probability: float

    selection_frequencies: np.ndarray
    final_probabilities: np.ndarray
    normalized_entropy: float

    sinr_samples: np.ndarray
    rate_samples_bit_s: np.ndarray


def validate_channel_parameters(
    channel_quality: np.ndarray,
    jammer_probabilities: np.ndarray,
    num_channels: int,
) -> None:
    """Validate channel-quality and jammer vectors."""
    if channel_quality.shape != (num_channels,):
        raise ValueError(
            "channel_quality must contain one value "
            "for every channel."
        )

    if jammer_probabilities.shape != (num_channels,):
        raise ValueError(
            "jammer_probabilities must contain one value "
            "for every channel."
        )

    if np.any(channel_quality <= 0):
        raise ValueError(
            "All channel-quality values must be positive."
        )

    if np.any(jammer_probabilities < 0):
        raise ValueError(
            "Jammer probabilities cannot be negative."
        )

    if not np.isclose(
        np.sum(jammer_probabilities),
        1.0,
    ):
        raise ValueError(
            "Jammer probabilities must sum to 1."
        )


def run_simulation(
    config: SimulationConfig,
    scheme: SchemeName,
    snr_db: float,
    jnr_db: float,
    channel_quality: np.ndarray,
    jammer_probabilities: np.ndarray,
    random_seed: int | None = None,
) -> SimulationResult:
    """
    Run one Monte Carlo experiment.

    Parameters
    ----------
    snr_db:
        Average received signal-to-noise ratio before
        small-scale fading.

    jnr_db:
        Average jammer-to-noise ratio when the jammer
        selects the same channel as the tag.

    channel_quality:
        Long-term quality multiplier for each channel.

    jammer_probabilities:
        Probability that the jammer selects each channel.
    """
    if scheme not in ("RCH", "PAFH"):
        raise ValueError(
            "scheme must be either 'RCH' or 'PAFH'."
        )

    channel_quality = np.asarray(
        channel_quality,
        dtype=float,
    )

    jammer_probabilities = np.asarray(
        jammer_probabilities,
        dtype=float,
    )

    validate_channel_parameters(
        channel_quality=channel_quality,
        jammer_probabilities=jammer_probabilities,
        num_channels=config.num_channels,
    )

    seed = (
        config.random_seed
        if random_seed is None
        else random_seed
    )

    rng = np.random.default_rng(seed)

    # Normalized average received signal and jammer powers.
    average_signal_power = (
        config.noise_power_watt
        * db_to_linear(snr_db)
    )

    average_jammer_power = (
        config.noise_power_watt
        * db_to_linear(jnr_db)
    )

    jammer = ProbabilisticJammer(
        power_watt=average_jammer_power,
        channel_probabilities=jammer_probabilities,
    )

    policy = PAFHPolicy(
        num_channels=config.num_channels,
        beta=config.beta,
    )

    utility_estimator = ChannelUtilityEstimator(
        num_channels=config.num_channels,
        sinr_threshold_linear=(
            config.sinr_threshold_linear
        ),
        window_size=config.utility_window_size,
        hybrid_weight=0.5,
    )

    sinr_samples = np.empty(
        config.num_slots,
        dtype=float,
    )

    rate_samples = np.empty(
        config.num_slots,
        dtype=float,
    )

    selection_counts = np.zeros(
        config.num_channels,
        dtype=int,
    )

    uniform_probabilities = (
        np.ones(config.num_channels)
        / config.num_channels
    )

    for slot in range(config.num_slots):
        # -------------------------
        # 1. Select hopping channel
        # -------------------------

        if scheme == "RCH":
            selected_channel = int(
                rng.choice(
                    config.num_channels,
                    p=uniform_probabilities,
                )
            )

        elif (
            slot
            < config.initial_exploration_slots
        ):
            # Ensure every channel is initially observed.
            selected_channel = (
                slot % config.num_channels
            )

        else:
            selected_channel = (
                policy.select_channel(rng)
            )

        selection_counts[selected_channel] += 1

        # -------------------------
        # 2. Generate dyadic fading
        # -------------------------

        dyadic_power_gain = (
            sample_dyadic_backscatter_power(
                rng=rng,
            )
        )

        received_signal_power = (
            average_signal_power
            * channel_quality[selected_channel]
            * dyadic_power_gain
        )

        # -------------------------
        # 3. Generate interference
        # -------------------------

        (
            interference_power,
            _was_jammed,
            _jammer_channel,
        ) = jammer.generate_interference(
            rng=rng,
            selected_channel=selected_channel,
        )

        # -------------------------
        # 4. Compute SINR
        # -------------------------

        sinr = (
            received_signal_power
            / (
                config.noise_power_watt
                + interference_power
            )
        )

        rate = achievable_rate(
            sinr_linear=sinr,
            bandwidth_hz=config.bandwidth_hz,
        )

        sinr_samples[slot] = sinr
        rate_samples[slot] = rate

        # -------------------------
        # 5. Update PA-FH utility
        # -------------------------

        if scheme == "PAFH":
            utility_estimator.update(
                channel=selected_channel,
                sinr_linear=sinr,
            )

            should_update = (
                (slot + 1)
                % config.probability_update_interval
                == 0
            )

            if should_update:
                utilities = (
                    utility_estimator.utilities(
                        mode="hybrid",
                    )
                )

                policy.update(utilities)

    # -------------------------
    # 6. Aggregate metrics
    # -------------------------

    mean_throughput = average_throughput(
        rate_samples_bit_s=rate_samples,
    )

    goodput = successful_throughput(
        sinr_samples=sinr_samples,
        bandwidth_hz=config.bandwidth_hz,
        sinr_threshold_linear=(
            config.sinr_threshold_linear
        ),
    )

    outage = outage_probability(
        sinr_samples=sinr_samples,
        sinr_threshold_linear=(
            config.sinr_threshold_linear
        ),
    )

    selection_frequencies = (
        selection_counts / config.num_slots
    )

    if scheme == "RCH":
        final_probabilities = (
            uniform_probabilities.copy()
        )
    else:
        final_probabilities = (
            policy.probabilities.copy()
        )

    entropy = distribution_entropy(
        probabilities=final_probabilities,
        normalized=True,
    )

    return SimulationResult(
        scheme=scheme,
        average_throughput_bit_s=(
            mean_throughput
        ),
        successful_throughput_bit_s=goodput,
        outage_probability=outage,
        selection_frequencies=(
            selection_frequencies
        ),
        final_probabilities=(
            final_probabilities
        ),
        normalized_entropy=entropy,
        sinr_samples=sinr_samples,
        rate_samples_bit_s=rate_samples,
    )