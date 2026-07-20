import numpy as np

from config import SimulationConfig
from metrics import hopping_gain
from simulation import run_simulation


def print_result(result) -> None:
    print(f"\n=== {result.scheme} ===")

    print(
        "Average throughput:",
        f"{result.average_throughput_bit_s / 1e3:.2f}",
        "kbit/s",
    )

    print(
        "Successful throughput:",
        f"{result.successful_throughput_bit_s / 1e3:.2f}",
        "kbit/s",
    )

    print(
        "Outage probability:",
        f"{result.outage_probability:.4f}",
    )

    print(
        "Final probabilities:",
        np.round(
            result.final_probabilities,
            4,
        ),
    )

    print(
        "Selection frequencies:",
        np.round(
            result.selection_frequencies,
            4,
        ),
    )

    print(
        "Normalized entropy:",
        f"{result.normalized_entropy:.4f}",
    )


def main() -> None:
    config = SimulationConfig()

    # Kênh đầu có chất lượng tốt hơn,
    # các kênh cuối có chất lượng thấp hơn.
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

    # Jammer tập trung nhiều hơn vào các kênh cuối.
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

    rch_result = run_simulation(
        config=config,
        scheme="RCH",
        snr_db=config.default_snr_db,
        jnr_db=config.default_jnr_db,
        channel_quality=channel_quality,
        jammer_probabilities=(
            jammer_probabilities
        ),
        random_seed=42,
    )

    pafh_result = run_simulation(
        config=config,
        scheme="PAFH",
        snr_db=config.default_snr_db,
        jnr_db=config.default_jnr_db,
        channel_quality=channel_quality,
        jammer_probabilities=(
            jammer_probabilities
        ),
        random_seed=43,
    )

    print_result(rch_result)
    print_result(pafh_result)

    gain = hopping_gain(
        pafh_throughput_bit_s=(
            pafh_result.average_throughput_bit_s
        ),
        rch_throughput_bit_s=(
            rch_result.average_throughput_bit_s
        ),
    )

    print(
        "\nPA-FH hopping gain over RCH:",
        f"{gain:.4f}",
    )


if __name__ == "__main__":
    main()