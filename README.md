# PA-FH Simulation for Backscatter IoT Networks

A Python implementation of the paper:

> **Probabilistic Analysis of Anti-Jamming Frequency Hopping in Backscatter IoT Networks**
> Le Hoang Hiep, Journal of Science and Technology on Information and Communications, 2026.

This project aims to **reproduce**, **analyze**, and **extend** the Probabilistic Anti-Jamming Frequency Hopping (PA-FH) algorithm for Backscatter IoT Networks.

---

## 📖 Overview

Backscatter communication enables ultra-low-power IoT devices by allowing tags to transmit data through RF signal reflection instead of generating their own RF signals.

However, because the reflected signal is extremely weak, Backscatter systems are highly vulnerable to intentional jamming attacks.

This project implements the **PA-FH (Probabilistic Anti-Jamming Frequency Hopping)** algorithm proposed in the paper, where channel selection probabilities are dynamically assigned based on channel utility and interference statistics.

---

## 🎯 Objectives

- Reproduce the simulation results reported in the paper.
- Understand the mathematical model behind PA-FH.
- Build a modular simulation framework in Python.
- Provide an extensible platform for future anti-jamming research.

---

## 📂 Project Structure

```
paper3/
│
├── config.py          # Simulation parameters
├── channel.py         # Channel and fading model
├── jammer.py          # Jammer model
├── utility.py         # Utility estimation
├── pafh.py            # PA-FH algorithm
├── metrics.py         # Throughput, Outage, Energy
├── simulation.py      # Monte Carlo simulation
├── main.py            # Main program
│
├── figures/
│
├── docs/
│
└── README.md
```

---

## 📡 System Model

The simulated network consists of:

- RF Source
- Backscatter Tag
- Reader
- Jammer

The Backscatter communication follows a **dyadic channel model**:

```
RF Source
     │
     ▼
    Tag
     │
     ▼
   Reader

     ▲
  Jammer
```

---

## 🔬 Implemented Algorithms

- No Anti-Jamming (NAJ)
- Static Channel Selection (SCS)
- Random Channel Hopping (RCH)
- Probabilistic Anti-Jamming Frequency Hopping (PA-FH)

---

## 📊 Performance Metrics

The simulator evaluates:

- Average Throughput
- Outage Probability
- Hopping Gain
- Energy Efficiency

---

## ⚙️ Simulation Parameters

Default parameters follow the paper.

| Parameter | Value |
|------------|--------|
| Number of Channels | 8 |
| Bandwidth | 100 kHz |
| RF Source Power | 30 dBm |
| Noise Power | -90 dBm |
| Jammer Power | 20–30 dBm |
| Monte Carlo Slots | 100000 |
| Fading Model | Rayleigh |

---

## 🚀 Running

Clone the repository

```bash
git clone https://github.com/yourname/pafh-backscatter.git
cd pafh-backscatter
```

Install dependencies

```bash
pip install numpy matplotlib
```

Run

```bash
python main.py
```

---

## 📈 Current Progress

- [x] Basic Monte Carlo simulation
- [x] Throughput evaluation
- [x] Outage evaluation
- [ ] Utility update mechanism
- [ ] Full Algorithm 1 implementation
- [ ] Energy efficiency
- [ ] Hopping gain
- [ ] Reactive jammer
- [ ] Intelligent jammer
- [ ] Multi-tag extension

---

## 🔭 Future Work

This repository will be extended with:

- Dynamic utility update
- Reactive jammer model
- DRL-based anti-jamming
- Multi-tag Backscatter network
- RIS-assisted communication
- Hardware-oriented implementation
- Performance comparison with recent anti-jamming approaches

---

## 📚 Reference

Le Hoang Hiep,
**Probabilistic Analysis of Anti-Jamming Frequency Hopping in Backscatter IoT Networks**,
Journal of Science and Technology on Information and Communications,
Vol. 11, No. 1, 2026.

---

## 📌 Disclaimer

This project is an independent educational and research implementation inspired by the published paper.
It is intended for learning, reproducibility, and academic research purposes only.
