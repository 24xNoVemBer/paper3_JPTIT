import numpy as np

from config import SimulationConfig
from metrics import (
    achievable_rate,
    average_throughput,
    compute_sinr,
    hopping_gain,
    linear_to_db,
    outage_probability,
    successful_throughput,
)


def main() -> None:
    config = SimulationConfig()

    # Example received backscatter signal power
    signal_power = 1e-9

    # Case 1: no jammer
    sinr_no_jammer = compute_sinr(
        signal_power_watt=signal_power,
        noise_power_watt=config.noise_power_watt,
        interference_power_watt=0.0,
    )

    # Case 2: jammer interference is present
    jammer_interference = 9.9e-11

    sinr_with_jammer = compute_sinr(
        signal_power_watt=signal_power,
        noise_power_watt=config.noise_power_watt,
        interference_power_watt=jammer_interference,
    )

    rate_no_jammer = achievable_rate(
        sinr_linear=sinr_no_jammer,
        bandwidth_hz=config.bandwidth_hz,
    )

    rate_with_jammer = achievable_rate(
        sinr_linear=sinr_with_jammer,
        bandwidth_hz=config.bandwidth_hz,
    )

    print("=== Single-slot metrics ===")
    print(
        "SINR without jammer:",
        f"{sinr_no_jammer:.4f}",
        f"({linear_to_db(sinr_no_jammer):.2f} dB)",
    )
    print(
        "SINR with jammer:",
        f"{sinr_with_jammer:.4f}",
        f"({linear_to_db(sinr_with_jammer):.2f} dB)",
    )
    print(
        "Rate without jammer:",
        f"{rate_no_jammer / 1e3:.2f} kbit/s",
    )
    print(
        "Rate with jammer:",
        f"{rate_with_jammer / 1e3:.2f} kbit/s",
    )

    # Monte Carlo-like sample validation
    sinr_samples = np.array(
        [10.0, 4.0, 2.5, 1.5, 1.0, 0.5]
    )

    rate_samples = achievable_rate(
        sinr_linear=sinr_samples,
        bandwidth_hz=config.bandwidth_hz,
    )

    outage = outage_probability(
        sinr_samples=sinr_samples,
        sinr_threshold_linear=config.sinr_threshold_linear,
    )

    mean_throughput = average_throughput(rate_samples)

    goodput = successful_throughput(
        sinr_samples=sinr_samples,
        bandwidth_hz=config.bandwidth_hz,
        sinr_threshold_linear=config.sinr_threshold_linear,
    )

    print("\n=== Multiple-slot metrics ===")
    print(
        "SINR threshold:",
        f"{config.sinr_threshold_db:.2f} dB",
    )
    print(
        "Outage probability:",
        f"{outage:.4f}",
    )
    print(
        "Average throughput:",
        f"{mean_throughput / 1e3:.2f} kbit/s",
    )
    print(
        "Successful throughput:",
        f"{goodput / 1e3:.2f} kbit/s",
    )

    example_gain = hopping_gain(
        pafh_throughput_bit_s=454e3,
        rch_throughput_bit_s=409e3,
    )

    print(
        "Example hopping gain:",
        f"{example_gain:.4f}",
    )


if __name__ == "__main__":
    main()