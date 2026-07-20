from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import SimulationConfig
from simulation import run_simulation


@dataclass
class BetaSweepResult:
    beta_values: np.ndarray

    throughput_mean_kbit_s: np.ndarray
    throughput_ci95_kbit_s: np.ndarray

    outage_mean: np.ndarray
    outage_ci95: np.ndarray

    entropy_mean: np.ndarray
    entropy_ci95: np.ndarray

    max_probability_mean: np.ndarray
    max_probability_ci95: np.ndarray


def mean_and_ci95(
    values: list[float],
) -> tuple[float, float]:
    """
    Compute sample mean and approximate 95% confidence interval.

    CI95 = 1.96 * standard error
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

    ci95 = 1.96 * standard_error

    return mean_value, float(ci95)


def run_beta_sweep(
    base_config: SimulationConfig,
    beta_values: np.ndarray,
    channel_quality: np.ndarray,
    jammer_probabilities: np.ndarray,
    repetitions: int = 5,
) -> BetaSweepResult:
    """
    Evaluate PA-FH under different beta values.

    SNR and JNR remain fixed while beta changes.
    """
    if repetitions <= 0:
        raise ValueError(
            "repetitions must be greater than zero."
        )

    beta_array = np.asarray(
        beta_values,
        dtype=float,
    )

    if beta_array.ndim != 1:
        raise ValueError(
            "beta_values must be one-dimensional."
        )

    if np.any(beta_array < 0):
        raise ValueError(
            "beta values cannot be negative."
        )

    throughput_means = []
    throughput_ci95_values = []

    outage_means = []
    outage_ci95_values = []

    entropy_means = []
    entropy_ci95_values = []

    max_probability_means = []
    max_probability_ci95_values = []

    for beta_index, beta in enumerate(beta_array):
        throughput_runs = []
        outage_runs = []
        entropy_runs = []
        max_probability_runs = []

        beta_config = replace(
            base_config,
            beta=float(beta),
        )

        for repetition in range(repetitions):
            seed = (
                base_config.random_seed
                + beta_index * 10_000
                + repetition * 100
            )

            result = run_simulation(
                config=beta_config,
                scheme="PAFH",
                snr_db=base_config.default_snr_db,
                jnr_db=base_config.default_jnr_db,
                channel_quality=channel_quality,
                jammer_probabilities=(
                    jammer_probabilities
                ),
                random_seed=seed,
            )

            throughput_runs.append(
                result.average_throughput_bit_s
                / 1e3
            )

            outage_runs.append(
                result.outage_probability
            )

            entropy_runs.append(
                result.normalized_entropy
            )

            max_probability_runs.append(
                float(
                    np.max(
                        result.final_probabilities
                    )
                )
            )

        throughput_mean, throughput_ci95 = (
            mean_and_ci95(
                throughput_runs
            )
        )

        outage_mean, outage_ci95 = (
            mean_and_ci95(
                outage_runs
            )
        )

        entropy_mean, entropy_ci95 = (
            mean_and_ci95(
                entropy_runs
            )
        )

        (
            max_probability_mean,
            max_probability_ci95,
        ) = mean_and_ci95(
            max_probability_runs
        )

        throughput_means.append(
            throughput_mean
        )

        throughput_ci95_values.append(
            throughput_ci95
        )

        outage_means.append(
            outage_mean
        )

        outage_ci95_values.append(
            outage_ci95
        )

        entropy_means.append(
            entropy_mean
        )

        entropy_ci95_values.append(
            entropy_ci95
        )

        max_probability_means.append(
            max_probability_mean
        )

        max_probability_ci95_values.append(
            max_probability_ci95
        )

        print(
            f"beta={beta:>5.2f} | "
            f"throughput={throughput_mean:>7.2f} kbit/s | "
            f"outage={outage_mean:.4f} | "
            f"entropy={entropy_mean:.4f} | "
            f"max_p={max_probability_mean:.4f}"
        )

    return BetaSweepResult(
        beta_values=beta_array,

        throughput_mean_kbit_s=np.asarray(
            throughput_means,
            dtype=float,
        ),

        throughput_ci95_kbit_s=np.asarray(
            throughput_ci95_values,
            dtype=float,
        ),

        outage_mean=np.asarray(
            outage_means,
            dtype=float,
        ),

        outage_ci95=np.asarray(
            outage_ci95_values,
            dtype=float,
        ),

        entropy_mean=np.asarray(
            entropy_means,
            dtype=float,
        ),

        entropy_ci95=np.asarray(
            entropy_ci95_values,
            dtype=float,
        ),

        max_probability_mean=np.asarray(
            max_probability_means,
            dtype=float,
        ),

        max_probability_ci95=np.asarray(
            max_probability_ci95_values,
            dtype=float,
        ),
    )


def save_beta_sweep(
    result: BetaSweepResult,
    output_directory: str = "results/beta_sweep",
) -> None:
    """
    Save beta sweep results to CSV and PNG files.
    """
    output_path = Path(output_directory)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_beta_csv(
        result=result,
        output_path=output_path,
    )

    plot_throughput_vs_beta(
        result=result,
        output_path=output_path,
    )

    plot_outage_vs_beta(
        result=result,
        output_path=output_path,
    )

    plot_entropy_vs_beta(
        result=result,
        output_path=output_path,
    )

    plot_max_probability_vs_beta(
        result=result,
        output_path=output_path,
    )


def save_beta_csv(
    result: BetaSweepResult,
    output_path: Path,
) -> None:
    csv_path = output_path / "beta_sweep.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "beta",
                "throughput_mean_kbit_s",
                "throughput_ci95_kbit_s",
                "outage_mean",
                "outage_ci95",
                "entropy_mean",
                "entropy_ci95",
                "max_probability_mean",
                "max_probability_ci95",
            ]
        )

        for index in range(
            len(result.beta_values)
        ):
            writer.writerow(
                [
                    result.beta_values[index],
                    result.throughput_mean_kbit_s[
                        index
                    ],
                    result.throughput_ci95_kbit_s[
                        index
                    ],
                    result.outage_mean[index],
                    result.outage_ci95[index],
                    result.entropy_mean[index],
                    result.entropy_ci95[index],
                    result.max_probability_mean[
                        index
                    ],
                    result.max_probability_ci95[
                        index
                    ],
                ]
            )


def plot_throughput_vs_beta(
    result: BetaSweepResult,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        result.beta_values,
        result.throughput_mean_kbit_s,
        yerr=result.throughput_ci95_kbit_s,
        marker="o",
        capsize=4,
        label="PA-FH",
    )

    plt.xlabel("Softmax Parameter, Beta")
    plt.ylabel("Average Throughput (kbit/s)")
    plt.title("Average Throughput versus Beta")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path / "throughput_vs_beta.png",
        dpi=300,
    )

    plt.show()


def plot_outage_vs_beta(
    result: BetaSweepResult,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        result.beta_values,
        result.outage_mean,
        yerr=result.outage_ci95,
        marker="o",
        capsize=4,
        label="PA-FH",
    )

    plt.xlabel("Softmax Parameter, Beta")
    plt.ylabel("Outage Probability")
    plt.title("Outage Probability versus Beta")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path / "outage_vs_beta.png",
        dpi=300,
    )

    plt.show()


def plot_entropy_vs_beta(
    result: BetaSweepResult,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        result.beta_values,
        result.entropy_mean,
        yerr=result.entropy_ci95,
        marker="o",
        capsize=4,
        label="Normalized Entropy",
    )

    plt.xlabel("Softmax Parameter, Beta")
    plt.ylabel("Normalized Entropy")
    plt.title("Hopping Entropy versus Beta")
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path / "entropy_vs_beta.png",
        dpi=300,
    )

    plt.show()


def plot_max_probability_vs_beta(
    result: BetaSweepResult,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        result.beta_values,
        result.max_probability_mean,
        yerr=result.max_probability_ci95,
        marker="o",
        capsize=4,
        label="Maximum Channel Probability",
    )

    plt.xlabel("Softmax Parameter, Beta")
    plt.ylabel("Maximum Selection Probability")
    plt.title(
        "Maximum Channel Probability versus Beta"
    )
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path
        / "max_probability_vs_beta.png",
        dpi=300,
    )

    plt.show()