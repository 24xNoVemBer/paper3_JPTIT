import numpy as np


def compute_sinr(
    signal_power_watt: float | np.ndarray,
    noise_power_watt: float,
    interference_power_watt: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """
    Compute the linear SINR:

        SINR = signal_power / (noise_power + interference_power)

    Parameters
    ----------
    signal_power_watt:
        Received backscatter signal power at the reader.
    noise_power_watt:
        Receiver noise power.
    interference_power_watt:
        Jammer interference power at the reader.

    Returns
    -------
    float or numpy.ndarray
        SINR in linear scale, not dB.
    """
    signal_power = np.asarray(signal_power_watt, dtype=float)
    interference_power = np.asarray(
        interference_power_watt,
        dtype=float,
    )

    if np.any(signal_power < 0):
        raise ValueError("signal_power_watt cannot be negative.")

    if noise_power_watt <= 0:
        raise ValueError("noise_power_watt must be positive.")

    if np.any(interference_power < 0):
        raise ValueError(
            "interference_power_watt cannot be negative."
        )

    sinr = signal_power / (
        noise_power_watt + interference_power
    )

    if sinr.ndim == 0:
        return float(sinr)

    return sinr


def linear_to_db(value: float | np.ndarray) -> float | np.ndarray:
    """Convert a positive linear value to dB."""
    value_array = np.asarray(value, dtype=float)

    if np.any(value_array <= 0):
        raise ValueError("Linear value must be greater than zero.")

    value_db = 10.0 * np.log10(value_array)

    if value_db.ndim == 0:
        return float(value_db)

    return value_db


def achievable_rate(
    sinr_linear: float | np.ndarray,
    bandwidth_hz: float,
) -> float | np.ndarray:
    """
    Shannon achievable rate:

        R = B * log2(1 + SINR)

    Returns the rate in bit/s.
    """
    sinr = np.asarray(sinr_linear, dtype=float)

    if np.any(sinr < 0):
        raise ValueError("SINR cannot be negative.")

    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive.")

    rate = bandwidth_hz * np.log2(1.0 + sinr)

    if rate.ndim == 0:
        return float(rate)

    return rate


def is_outage(
    sinr_linear: float,
    sinr_threshold_linear: float,
) -> bool:
    """
    A transmission is in outage when:

        SINR < SINR_threshold
    """
    if sinr_linear < 0:
        raise ValueError("SINR cannot be negative.")

    if sinr_threshold_linear <= 0:
        raise ValueError(
            "sinr_threshold_linear must be positive."
        )

    return sinr_linear < sinr_threshold_linear


def outage_probability(
    sinr_samples: np.ndarray,
    sinr_threshold_linear: float,
) -> float:
    """
    Estimate outage probability from Monte Carlo samples.
    """
    samples = np.asarray(sinr_samples, dtype=float)

    if samples.size == 0:
        raise ValueError("sinr_samples cannot be empty.")

    if np.any(samples < 0):
        raise ValueError("SINR samples cannot be negative.")

    if sinr_threshold_linear <= 0:
        raise ValueError(
            "sinr_threshold_linear must be positive."
        )

    return float(
        np.mean(samples < sinr_threshold_linear)
    )


def average_throughput(
    rate_samples_bit_s: np.ndarray,
) -> float:
    """
    Average Shannon throughput in bit/s.
    """
    rates = np.asarray(rate_samples_bit_s, dtype=float)

    if rates.size == 0:
        raise ValueError("rate_samples_bit_s cannot be empty.")

    if np.any(rates < 0):
        raise ValueError("Rate samples cannot be negative.")

    return float(np.mean(rates))


def successful_throughput(
    sinr_samples: np.ndarray,
    bandwidth_hz: float,
    sinr_threshold_linear: float,
) -> float:
    """
    Average successfully delivered rate.

    Slots whose SINR is below the decoding threshold
    contribute zero throughput.

    This metric is also known as goodput.
    """
    sinr = np.asarray(sinr_samples, dtype=float)

    rates = achievable_rate(
        sinr_linear=sinr,
        bandwidth_hz=bandwidth_hz,
    )

    successful_rates = np.where(
        sinr >= sinr_threshold_linear,
        rates,
        0.0,
    )

    return float(np.mean(successful_rates))


def hopping_gain(
    pafh_throughput_bit_s: float,
    rch_throughput_bit_s: float,
) -> float:
    """
    Hopping gain defined in the paper as:

        G_H = eta_PA-FH / eta_RCH
    """
    if pafh_throughput_bit_s < 0:
        raise ValueError(
            "PA-FH throughput cannot be negative."
        )

    if rch_throughput_bit_s <= 0:
        raise ValueError(
            "RCH throughput must be greater than zero."
        )

    return pafh_throughput_bit_s / rch_throughput_bit_s