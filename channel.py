import numpy as np
from numpy.random import Generator


def path_loss(
    distance_m: float,
    reference_gain: float = 1.0,
    path_loss_exponent: float = 2.7,
) -> float:
    """
    Distance-based path-loss model:

        L(d) = K0 * d^(-alpha)

    Parameters
    ----------
    distance_m:
        Distance between transmitter and receiver in meters.
    reference_gain:
        Path-loss constant K0.
    path_loss_exponent:
        Path-loss exponent alpha.
    """
    if distance_m <= 0:
        raise ValueError("distance_m must be greater than zero.")

    if reference_gain <= 0:
        raise ValueError("reference_gain must be greater than zero.")

    if path_loss_exponent <= 0:
        raise ValueError("path_loss_exponent must be greater than zero.")

    return reference_gain * distance_m ** (-path_loss_exponent)


def sample_rayleigh_channel(
    rng: Generator,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | complex:
    """
    Generate a complex Rayleigh fading coefficient:

        h ~ CN(0, 1)

    Therefore |h|^2 follows an exponential distribution
    with unit mean.
    """
    real = rng.normal(0.0, 1.0, size)
    imaginary = rng.normal(0.0, 1.0, size)

    return (real + 1j * imaginary) / np.sqrt(2.0)


def sample_rayleigh_power(
    rng: Generator,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Generate the Rayleigh fading power gain |h|^2."""
    channel = sample_rayleigh_channel(rng, size)
    return np.abs(channel) ** 2


def sample_dyadic_backscatter_power(
    rng: Generator,
    size: int | tuple[int, ...] | None = None,
    source_tag_path_loss: float = 1.0,
    tag_reader_path_loss: float = 1.0,
) -> np.ndarray | float:
    """
    Generate the effective power gain of the dyadic
    backscatter channel:

        h_bs = h_st * h_tr

        |h_bs|^2 = |h_st|^2 * |h_tr|^2

    Path loss is also applied to both transmission stages.
    """
    if source_tag_path_loss <= 0:
        raise ValueError("source_tag_path_loss must be positive.")

    if tag_reader_path_loss <= 0:
        raise ValueError("tag_reader_path_loss must be positive.")

    h_st_power = sample_rayleigh_power(rng, size)
    h_tr_power = sample_rayleigh_power(rng, size)

    return (
        source_tag_path_loss
        * tag_reader_path_loss
        * h_st_power
        * h_tr_power
    )


def sample_jammer_reader_power(
    rng: Generator,
    size: int | tuple[int, ...] | None = None,
    jammer_reader_path_loss: float = 1.0,
) -> np.ndarray | float:
    """Generate the jammer-to-reader channel power gain."""
    if jammer_reader_path_loss <= 0:
        raise ValueError("jammer_reader_path_loss must be positive.")

    return jammer_reader_path_loss * sample_rayleigh_power(rng, size)