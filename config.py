from dataclasses import dataclass


def dbm_to_watt(power_dbm: float) -> float:
    """Convert power from dBm to watt."""
    return 10 ** ((power_dbm - 30.0) / 10.0)


def db_to_linear(value_db: float) -> float:
    """Convert a value from dB to linear scale."""
    return 10 ** (value_db / 10.0)


@dataclass(frozen=True)
class SimulationConfig:
    """Default parameters based on the paper."""

    num_channels: int = 8
    bandwidth_hz: float = 100e3
    num_slots: int = 100_000

    source_power_dbm: float = 30.0
    noise_power_dbm: float = -90.0
    jammer_power_dbm: float = 25.0

    sinr_threshold_db: float = 3.0
    beta: float = 3.0

    jammer_activation_probability: float = 0.5
    random_seed: int = 42

    snr_db_start: int = 0
    snr_db_stop: int = 30
    snr_db_step: int = 5

    @property
    def source_power_watt(self) -> float:
        return dbm_to_watt(self.source_power_dbm)

    @property
    def noise_power_watt(self) -> float:
        return dbm_to_watt(self.noise_power_dbm)

    @property
    def jammer_power_watt(self) -> float:
        return dbm_to_watt(self.jammer_power_dbm)

    @property
    def sinr_threshold_linear(self) -> float:
        return db_to_linear(self.sinr_threshold_db)

    @property
    def snr_db_values(self) -> list[int]:
        return list(
            range(
                self.snr_db_start,
                self.snr_db_stop + 1,
                self.snr_db_step,
            )
        )