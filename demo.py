import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Simulation parameters
# =========================

K = 8                       # number of channels
B = 100e3                   # bandwidth = 100 kHz
N_slots = 100_000           # Monte Carlo slots
noise_dBm = -90
Pj_dBm = 25                 # jammer power
gamma_th_dB = 3             # outage threshold
beta = 3                    # softmax parameter

noise = 10 ** ((noise_dBm - 30) / 10)
Pj = 10 ** ((Pj_dBm - 30) / 10)
gamma_th = 10 ** (gamma_th_dB / 10)

snr_dB_range = np.arange(0, 31, 5)


# =========================
# 2. Helper functions
# =========================

def softmax(u, beta):
    x = beta * u
    x = x - np.max(x)       # avoid overflow
    e = np.exp(x)
    return e / np.sum(e)


def simulate_scheme(snr_dB, scheme="RCH"):
    Ps = noise * 10 ** (snr_dB / 10)

    # long-term channel quality / utility
    # giả lập vài kênh tốt, vài kênh xấu
    avg_channel_quality = np.array([1.4, 1.3, 1.1, 1.0, 0.7, 0.6, 0.5, 0.4])

    if scheme == "RCH":
        p = np.ones(K) / K

    elif scheme == "PAFH":
        utility = avg_channel_quality
        p = softmax(utility, beta)

    else:
        raise ValueError("Unknown scheme")

    throughput = []
    outages = 0

    for _ in range(N_slots):
        # chọn channel theo xác suất p
        k = np.random.choice(K, p=p)

        # Rayleigh fading: |h|^2 ~ exponential(1)
        h_st = np.random.exponential(1)
        h_tr = np.random.exponential(1)
        h_jr = np.random.exponential(1)

        # dyadic backscatter channel
        h_bs_power = h_st * h_tr

        # jammer activity: giả sử jammer active với xác suất 0.5
        jammer_active = np.random.rand() < 0.5

        interference = Pj * h_jr if jammer_active else 0

        sinr = (Ps * avg_channel_quality[k] * h_bs_power) / (noise + interference)

        rate = B * np.log2(1 + sinr)
        throughput.append(rate)

        if sinr < gamma_th:
            outages += 1

    avg_throughput = np.mean(throughput)
    outage_prob = outages / N_slots

    return avg_throughput, outage_prob, p


# =========================
# 3. Run simulation
# =========================

throughput_rch = []
throughput_pafh = []

outage_rch = []
outage_pafh = []

for snr_dB in snr_dB_range:
    th_rch, out_rch, _ = simulate_scheme(snr_dB, "RCH")
    th_pafh, out_pafh, p_pafh = simulate_scheme(snr_dB, "PAFH")

    throughput_rch.append(th_rch / 1e3)      # kbit/s
    throughput_pafh.append(th_pafh / 1e3)

    outage_rch.append(out_rch)
    outage_pafh.append(out_pafh)


print("PA-FH probability vector:")
print(np.round(p_pafh, 3))


# =========================
# 4. Plot throughput
# =========================

plt.figure()
plt.plot(snr_dB_range, throughput_rch, marker="o", label="RCH")
plt.plot(snr_dB_range, throughput_pafh, marker="s", label="PA-FH")
plt.xlabel("SNR (dB)")
plt.ylabel("Average Throughput (kbit/s)")
plt.title("Throughput vs SNR")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 5. Plot outage
# =========================

plt.figure()
plt.plot(snr_dB_range, outage_rch, marker="o", label="RCH")
plt.plot(snr_dB_range, outage_pafh, marker="s", label="PA-FH")
plt.xlabel("SNR (dB)")
plt.ylabel("Outage Probability")
plt.title("Outage vs SNR")
plt.legend()
plt.grid(True)
plt.show()