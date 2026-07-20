import numpy as np

from channel import (
    path_loss,
    sample_dyadic_backscatter_power,
    sample_rayleigh_power,
)
from config import SimulationConfig


def main() -> None:
    config = SimulationConfig()
    rng = np.random.default_rng(config.random_seed)

    num_samples = 100_000

    rayleigh_samples = sample_rayleigh_power(
        rng=rng,
        size=num_samples,
    )

    dyadic_samples = sample_dyadic_backscatter_power(
        rng=rng,
        size=num_samples,
    )

    print("=== Configuration ===")
    print(f"Channels: {config.num_channels}")
    print(f"Bandwidth: {config.bandwidth_hz / 1e3:.0f} kHz")
    print(f"Noise power: {config.noise_power_watt:.3e} W")
    print(f"Jammer power: {config.jammer_power_watt:.3e} W")

    print("\n=== Channel validation ===")
    print(
        "Mean Rayleigh power |h|^2:",
        f"{np.mean(rayleigh_samples):.4f}",
    )
    print(
        "Mean dyadic power |h_st|^2|h_tr|^2:",
        f"{np.mean(dyadic_samples):.4f}",
    )

    example_path_loss = path_loss(
        distance_m=5.0,
        path_loss_exponent=2.7,
    )

    print(
        "Example path loss at 5 m:",
        f"{example_path_loss:.6e}",
    )


if __name__ == "__main__":
    main()