import numpy as np

from config import SimulationConfig
from utility import ChannelUtilityEstimator


def main() -> None:
    config = SimulationConfig()

    estimator = ChannelUtilityEstimator(
        num_channels=config.num_channels,
        sinr_threshold_linear=(
            config.sinr_threshold_linear
        ),
        window_size=5,
        hybrid_weight=0.5,
    )

    # Channel 0: high-quality channel
    for sinr in [10.0, 12.0, 8.0, 15.0, 9.0]:
        estimator.update(
            channel=0,
            sinr_linear=sinr,
        )

    # Channel 1: medium-quality channel
    for sinr in [3.0, 2.5, 2.2, 1.8, 3.5]:
        estimator.update(
            channel=1,
            sinr_linear=sinr,
        )

    # Channel 2: poor-quality channel
    for sinr in [0.5, 0.7, 1.0, 0.4, 0.8]:
        estimator.update(
            channel=2,
            sinr_linear=sinr,
        )

    print("=== Channel statistics ===")

    print(
        "Observation counts:",
        estimator.observation_counts(),
    )

    print(
        "Mean SINR:",
        np.round(
            estimator.mean_sinr(),
            3,
        ),
    )

    print(
        "Success rates:",
        np.round(
            estimator.success_rates(),
            3,
        ),
    )

    print(
        "Outage rates:",
        np.round(
            estimator.outage_rates(),
            3,
        ),
    )

    print(
        "Mean-SINR utilities:",
        np.round(
            estimator.utilities("mean_sinr"),
            3,
        ),
    )

    print(
        "Success-rate utilities:",
        np.round(
            estimator.utilities("success_rate"),
            3,
        ),
    )

    print(
        "Hybrid utilities:",
        np.round(
            estimator.utilities("hybrid"),
            3,
        ),
    )


if __name__ == "__main__":
    main()