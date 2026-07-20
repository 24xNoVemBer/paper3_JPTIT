from dataclasses import replace

import numpy as np

from config import SimulationConfig
from experiments_beta import (
    run_beta_sweep,
    save_beta_sweep,
)


def main() -> None:
    base_config = SimulationConfig()

    # Cấu hình debug để chạy nhanh.
    config = replace(
        base_config,
        num_slots=100_000,
    )

    channel_quality = np.array(
        [
            1.40,
            1.25,
            1.10,
            0.95,
            0.80,
            0.65,
            0.55,
            0.45,
        ]
    )

    jammer_probabilities = np.array(
        [
            0.05,
            0.05,
            0.08,
            0.10,
            0.15,
            0.17,
            0.18,
            0.22,
        ]
    )

    beta_values = np.array(
        [
            0.0,
            0.5,
            1.0,
            2.0,
            3.0,
            5.0,
            8.0,
            12.0,
        ]
    )

    result = run_beta_sweep(
        base_config=config,
        beta_values=beta_values,
        channel_quality=channel_quality,
        jammer_probabilities=(
            jammer_probabilities
        ),
        repetitions=10,
    )

    save_beta_sweep(result)


if __name__ == "__main__":
    main()