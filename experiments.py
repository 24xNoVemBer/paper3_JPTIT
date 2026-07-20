from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import SimulationConfig
from simulation import run_simulation


@dataclass
class SNRSweepResult:
    snr_db: np.ndarray

    rch_throughput_kbit_s: np.ndarray
    pafh_throughput_kbit_s: np.ndarray

    rch_outage: np.ndarray
    pafh_outage: np.ndarray

    hopping_gain: np.ndarray


def run_snr_sweep(
    config: SimulationConfig,
    channel_quality: np.ndarray,
    jammer_probabilities: np.ndarray,
    repetitions: int = 3,
) -> SNRSweepResult:
    """
    Run RCH and PA-FH over the configured SNR range.

    Results are averaged across multiple independent
    Monte Carlo repetitions.
    """
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    snr_values = np.asarray(
        config.snr_db_values,
        dtype=float,
    )

    rch_throughput = []
    pafh_throughput = []

    rch_outage = []
    pafh_outage = []

    for snr_index, snr_db in enumerate(snr_values):
        rch_throughput_runs = []
        pafh_throughput_runs = []

        rch_outage_runs = []
        pafh_outage_runs = []

        for repetition in range(repetitions):
            base_seed = (
                config.random_seed
                + 1000 * snr_index
                + 10 * repetition
            )

            rch_result = run_simulation(
                config=config,
                scheme="RCH",
                snr_db=float(snr_db),
                jnr_db=config.default_jnr_db,
                channel_quality=channel_quality,
                jammer_probabilities=jammer_probabilities,
                random_seed=base_seed,
            )

            pafh_result = run_simulation(
                config=config,
                scheme="PAFH",
                snr_db=float(snr_db),
                jnr_db=config.default_jnr_db,
                channel_quality=channel_quality,
                jammer_probabilities=jammer_probabilities,
                random_seed=base_seed + 1,
            )

            rch_throughput_runs.append(
                rch_result.average_throughput_bit_s / 1e3
            )

            pafh_throughput_runs.append(
                pafh_result.average_throughput_bit_s / 1e3
            )

            rch_outage_runs.append(
                rch_result.outage_probability
            )

            pafh_outage_runs.append(
                pafh_result.outage_probability
            )

        rch_throughput.append(
            np.mean(rch_throughput_runs)
        )

        pafh_throughput.append(
            np.mean(pafh_throughput_runs)
        )

        rch_outage.append(
            np.mean(rch_outage_runs)
        )

        pafh_outage.append(
            np.mean(pafh_outage_runs)
        )

        print(
            f"SNR={snr_db:>4.1f} dB | "
            f"RCH={rch_throughput[-1]:>7.2f} kbit/s | "
            f"PA-FH={pafh_throughput[-1]:>7.2f} kbit/s"
        )

    rch_throughput_array = np.asarray(
        rch_throughput
    )

    pafh_throughput_array = np.asarray(
        pafh_throughput
    )

    gains = (
        pafh_throughput_array
        / rch_throughput_array
    )

    return SNRSweepResult(
        snr_db=snr_values,
        rch_throughput_kbit_s=(
            rch_throughput_array
        ),
        pafh_throughput_kbit_s=(
            pafh_throughput_array
        ),
        rch_outage=np.asarray(rch_outage),
        pafh_outage=np.asarray(pafh_outage),
        hopping_gain=gains,
    )


def save_snr_sweep(
    result: SNRSweepResult,
    output_directory: str = "results",
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_path / "snr_sweep.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "snr_db",
                "rch_throughput_kbit_s",
                "pafh_throughput_kbit_s",
                "rch_outage",
                "pafh_outage",
                "hopping_gain",
            ]
        )

        for index in range(len(result.snr_db)):
            writer.writerow(
                [
                    result.snr_db[index],
                    result.rch_throughput_kbit_s[index],
                    result.pafh_throughput_kbit_s[index],
                    result.rch_outage[index],
                    result.pafh_outage[index],
                    result.hopping_gain[index],
                ]
            )

    plt.figure(figsize=(8, 5))

    plt.plot(
        result.snr_db,
        result.rch_throughput_kbit_s,
        marker="o",
        label="RCH",
    )

    plt.plot(
        result.snr_db,
        result.pafh_throughput_kbit_s,
        marker="s",
        label="PA-FH",
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Average Throughput (kbit/s)")
    plt.title("Average Throughput versus SNR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path / "throughput_vs_snr.png",
        dpi=300,
    )

    plt.show()

    plt.figure(figsize=(8, 5))

    plt.plot(
        result.snr_db,
        result.rch_outage,
        marker="o",
        label="RCH",
    )

    plt.plot(
        result.snr_db,
        result.pafh_outage,
        marker="s",
        label="PA-FH",
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Outage Probability")
    plt.title("Outage Probability versus SNR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path / "outage_vs_snr.png",
        dpi=300,
    )

    plt.show()