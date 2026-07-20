from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import SimulationConfig
from simulation import run_simulation


@dataclass
class JNRSweepResult:
    """
    Store the results of a jammer-to-noise ratio sweep.
    """

    jnr_db: np.ndarray

    rch_throughput_mean_kbit_s: np.ndarray
    pafh_throughput_mean_kbit_s: np.ndarray

    rch_throughput_ci95_kbit_s: np.ndarray
    pafh_throughput_ci95_kbit_s: np.ndarray

    rch_outage_mean: np.ndarray
    pafh_outage_mean: np.ndarray

    rch_outage_ci95: np.ndarray
    pafh_outage_ci95: np.ndarray

    hopping_gain: np.ndarray


def mean_and_ci95(
    values: list[float],
) -> tuple[float, float]:
    """
    Return sample mean and approximate 95% confidence interval.

    CI95 = 1.96 * standard_error
    """
    samples = np.asarray(values, dtype=float)

    if samples.size == 0:
        raise ValueError("values cannot be empty.")

    mean_value = float(np.mean(samples))

    if samples.size == 1:
        return mean_value, 0.0

    standard_error = (
        np.std(samples, ddof=1)
        / np.sqrt(samples.size)
    )

    confidence_interval = 1.96 * standard_error

    return mean_value, float(confidence_interval)


def run_jnr_sweep(
    config: SimulationConfig,
    channel_quality: np.ndarray,
    jammer_probabilities: np.ndarray,
    jnr_db_values: np.ndarray,
    repetitions: int = 10,
) -> JNRSweepResult:
    """
    Compare RCH and PA-FH over different JNR values.

    SNR is fixed at config.default_snr_db.
    """
    if repetitions <= 0:
        raise ValueError(
            "repetitions must be greater than zero."
        )

    jnr_values = np.asarray(
        jnr_db_values,
        dtype=float,
    )

    if jnr_values.ndim != 1:
        raise ValueError(
            "jnr_db_values must be one-dimensional."
        )

    rch_throughput_means = []
    pafh_throughput_means = []

    rch_throughput_ci95 = []
    pafh_throughput_ci95 = []

    rch_outage_means = []
    pafh_outage_means = []

    rch_outage_ci95 = []
    pafh_outage_ci95 = []

    for jnr_index, jnr_db in enumerate(jnr_values):
        rch_throughput_runs = []
        pafh_throughput_runs = []

        rch_outage_runs = []
        pafh_outage_runs = []

        for repetition in range(repetitions):
            base_seed = (
                config.random_seed
                + 10_000 * jnr_index
                + 100 * repetition
            )

            rch_result = run_simulation(
                config=config,
                scheme="RCH",
                snr_db=config.default_snr_db,
                jnr_db=float(jnr_db),
                channel_quality=channel_quality,
                jammer_probabilities=(
                    jammer_probabilities
                ),
                random_seed=base_seed,
            )

            pafh_result = run_simulation(
                config=config,
                scheme="PAFH",
                snr_db=config.default_snr_db,
                jnr_db=float(jnr_db),
                channel_quality=channel_quality,
                jammer_probabilities=(
                    jammer_probabilities
                ),
                random_seed=base_seed + 1,
            )

            rch_throughput_runs.append(
                rch_result.average_throughput_bit_s
                / 1e3
            )

            pafh_throughput_runs.append(
                pafh_result.average_throughput_bit_s
                / 1e3
            )

            rch_outage_runs.append(
                rch_result.outage_probability
            )

            pafh_outage_runs.append(
                pafh_result.outage_probability
            )

        (
            rch_throughput_mean,
            rch_throughput_interval,
        ) = mean_and_ci95(
            rch_throughput_runs
        )

        (
            pafh_throughput_mean,
            pafh_throughput_interval,
        ) = mean_and_ci95(
            pafh_throughput_runs
        )

        (
            rch_outage_mean,
            rch_outage_interval,
        ) = mean_and_ci95(
            rch_outage_runs
        )

        (
            pafh_outage_mean,
            pafh_outage_interval,
        ) = mean_and_ci95(
            pafh_outage_runs
        )

        rch_throughput_means.append(
            rch_throughput_mean
        )

        pafh_throughput_means.append(
            pafh_throughput_mean
        )

        rch_throughput_ci95.append(
            rch_throughput_interval
        )

        pafh_throughput_ci95.append(
            pafh_throughput_interval
        )

        rch_outage_means.append(
            rch_outage_mean
        )

        pafh_outage_means.append(
            pafh_outage_mean
        )

        rch_outage_ci95.append(
            rch_outage_interval
        )

        pafh_outage_ci95.append(
            pafh_outage_interval
        )

        print(
            f"JNR={jnr_db:>5.1f} dB | "
            f"RCH outage={rch_outage_mean:.4f} | "
            f"PA-FH outage={pafh_outage_mean:.4f} | "
            f"RCH throughput={rch_throughput_mean:.2f} kbit/s | "
            f"PA-FH throughput={pafh_throughput_mean:.2f} kbit/s"
        )

    rch_throughput_array = np.asarray(
        rch_throughput_means,
        dtype=float,
    )

    pafh_throughput_array = np.asarray(
        pafh_throughput_means,
        dtype=float,
    )

    hopping_gain = (
        pafh_throughput_array
        / rch_throughput_array
    )

    return JNRSweepResult(
        jnr_db=jnr_values,

        rch_throughput_mean_kbit_s=(
            rch_throughput_array
        ),

        pafh_throughput_mean_kbit_s=(
            pafh_throughput_array
        ),

        rch_throughput_ci95_kbit_s=np.asarray(
            rch_throughput_ci95,
            dtype=float,
        ),

        pafh_throughput_ci95_kbit_s=np.asarray(
            pafh_throughput_ci95,
            dtype=float,
        ),

        rch_outage_mean=np.asarray(
            rch_outage_means,
            dtype=float,
        ),

        pafh_outage_mean=np.asarray(
            pafh_outage_means,
            dtype=float,
        ),

        rch_outage_ci95=np.asarray(
            rch_outage_ci95,
            dtype=float,
        ),

        pafh_outage_ci95=np.asarray(
            pafh_outage_ci95,
            dtype=float,
        ),

        hopping_gain=hopping_gain,
    )


def save_jnr_sweep(
    result: JNRSweepResult,
    output_directory: str = "results/jnr_sweep",
) -> None:
    """
    Save JNR sweep results as CSV and PNG files.
    """
    output_path = Path(output_directory)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_path / "jnr_sweep.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "jnr_db",
                "rch_throughput_mean_kbit_s",
                "rch_throughput_ci95_kbit_s",
                "pafh_throughput_mean_kbit_s",
                "pafh_throughput_ci95_kbit_s",
                "rch_outage_mean",
                "rch_outage_ci95",
                "pafh_outage_mean",
                "pafh_outage_ci95",
                "hopping_gain",
            ]
        )

        for index in range(len(result.jnr_db)):
            writer.writerow(
                [
                    result.jnr_db[index],
                    result.rch_throughput_mean_kbit_s[
                        index
                    ],
                    result.rch_throughput_ci95_kbit_s[
                        index
                    ],
                    result.pafh_throughput_mean_kbit_s[
                        index
                    ],
                    result.pafh_throughput_ci95_kbit_s[
                        index
                    ],
                    result.rch_outage_mean[index],
                    result.rch_outage_ci95[index],
                    result.pafh_outage_mean[index],
                    result.pafh_outage_ci95[index],
                    result.hopping_gain[index],
                ]
            )

    plot_outage_vs_jnr(
        result=result,
        output_path=output_path,
    )

    plot_throughput_vs_jnr(
        result=result,
        output_path=output_path,
    )

    plot_hopping_gain_vs_jnr(
        result=result,
        output_path=output_path,
    )


def plot_outage_vs_jnr(
    result: JNRSweepResult,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        result.jnr_db,
        result.rch_outage_mean,
        yerr=result.rch_outage_ci95,
        marker="o",
        capsize=4,
        label="RCH",
    )

    plt.errorbar(
        result.jnr_db,
        result.pafh_outage_mean,
        yerr=result.pafh_outage_ci95,
        marker="s",
        capsize=4,
        label="PA-FH",
    )

    plt.xlabel("Jammer-to-Noise Ratio, JNR (dB)")
    plt.ylabel("Outage Probability")
    plt.title("Outage Probability versus JNR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path / "outage_vs_jnr.png",
        dpi=300,
    )

    plt.show()


def plot_throughput_vs_jnr(
    result: JNRSweepResult,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        result.jnr_db,
        result.rch_throughput_mean_kbit_s,
        yerr=result.rch_throughput_ci95_kbit_s,
        marker="o",
        capsize=4,
        label="RCH",
    )

    plt.errorbar(
        result.jnr_db,
        result.pafh_throughput_mean_kbit_s,
        yerr=result.pafh_throughput_ci95_kbit_s,
        marker="s",
        capsize=4,
        label="PA-FH",
    )

    plt.xlabel("Jammer-to-Noise Ratio, JNR (dB)")
    plt.ylabel("Average Throughput (kbit/s)")
    plt.title("Average Throughput versus JNR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path / "throughput_vs_jnr.png",
        dpi=300,
    )

    plt.show()


def plot_hopping_gain_vs_jnr(
    result: JNRSweepResult,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.plot(
        result.jnr_db,
        result.hopping_gain,
        marker="o",
        label="PA-FH / RCH",
    )

    plt.axhline(
        y=1.0,
        linestyle="--",
        label="No gain",
    )

    plt.xlabel("Jammer-to-Noise Ratio, JNR (dB)")
    plt.ylabel("Hopping Gain")
    plt.title("PA-FH Hopping Gain versus JNR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path / "hopping_gain_vs_jnr.png",
        dpi=300,
    )

    plt.show()