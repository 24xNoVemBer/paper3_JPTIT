import numpy as np

from config import SimulationConfig
from jammer import ProbabilisticJammer, ReactiveJammer


def main() -> None:
    config = SimulationConfig()
    rng = np.random.default_rng(config.random_seed)

    num_trials = 100_000

    # Uniform probabilistic jammer
    q = np.ones(config.num_channels) / config.num_channels

    probabilistic_jammer = ProbabilisticJammer(
        power_watt=config.jammer_power_watt,
        channel_probabilities=q,
    )

    selected_channel = 0
    probabilistic_jam_count = 0
    probabilistic_interference_sum = 0.0

    for _ in range(num_trials):
        interference, was_jammed, jammer_channel = (
            probabilistic_jammer.generate_interference(
                rng=rng,
                selected_channel=selected_channel,
            )
        )

        if was_jammed:
            probabilistic_jam_count += 1
            probabilistic_interference_sum += interference

    probabilistic_hit_rate = (
        probabilistic_jam_count / num_trials
    )

    print("=== Probabilistic jammer ===")
    print(
        "Expected hit probability:",
        f"{1 / config.num_channels:.4f}",
    )
    print(
        "Simulated hit probability:",
        f"{probabilistic_hit_rate:.4f}",
    )

    if probabilistic_jam_count > 0:
        print(
            "Mean interference when jammed:",
            f"{probabilistic_interference_sum / probabilistic_jam_count:.4e} W",
        )

    print()

    # Reactive jammer
    detection_probability = 0.7

    reactive_jammer = ReactiveJammer(
        power_watt=config.jammer_power_watt,
        detection_probability=detection_probability,
    )

    reactive_jam_count = 0
    reactive_interference_sum = 0.0

    for _ in range(num_trials):
        interference, was_jammed = (
            reactive_jammer.generate_interference(rng)
        )

        if was_jammed:
            reactive_jam_count += 1
            reactive_interference_sum += interference

    reactive_hit_rate = reactive_jam_count / num_trials

    print("=== Reactive jammer ===")
    print(
        "Expected detection probability:",
        f"{detection_probability:.4f}",
    )
    print(
        "Simulated detection probability:",
        f"{reactive_hit_rate:.4f}",
    )

    if reactive_jam_count > 0:
        print(
            "Mean interference when jammed:",
            f"{reactive_interference_sum / reactive_jam_count:.4e} W",
        )


if __name__ == "__main__":
    main()