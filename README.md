# 🛡️ IoT Network Intrusion Detection & Zero-Trust Defense Framework

> **Research Focus**: Analyzing multi-class IoT network cyberattacks, mitigating extreme class imbalance, evaluating adversarial model vulnerabilities (FGSM/PGD), and architecting a defense-in-depth Context-Aware Zero-Trust Policy Engine.

---

## 📌 Project Overview

Internet of Things (IoT) ecosystems are increasingly targeted by sophisticated network attacks such as Distributed Denial of Service (DDoS), Botnets, Port Scans, and Brute Force intrusions. Traditional Machine Learning (ML) based Intrusion Detection Systems (IDS) suffer from three critical challenges:
1. **Severe Data Imbalance**: Benign traffic overwhelmingly dominates network captures, causing models to miss rare but high-impact attack categories.
2. **Adversarial Evasion Vulnerability**: Attackers can craft subtle, gradient-based feature perturbations (e.g., Fast Gradient Sign Method - FGSM, Projected Gradient Descent - PGD) that fool ML classifiers into marking malicious flows as benign.
3. **Single-Point-of-Failure Risk**: Relying solely on ML risk scores leaves networks unprotected when an adversarial payload bypasses the detection layer.

### 🎯 Research Objective
This project addresses these challenges by introducing an end-to-end framework combining **hybrid data balancing (SMOTE + Tomek Links)**, **generative synthetic augmentation (Conditional GAN)**, **multiclass PyTorch deep learning classification**, **adversarial benchmarking**, and a **prioritized Context-Aware Zero-Trust Policy Engine**.

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    subgraph COMPLETED ["Completed Milestones (Phases 1 - 6 & Phase 10)"]
        A[Raw IoT Flow Data 76 Features] --> B[Phase 1: Stratified Data Split]
        B --> C[Phase 2: Hybrid Balancing SMOTE + Tomek Links]
        B --> D[Untouched Isolated Test Set]
        
        C --> E[Phase 2: UMAP 2D Feature Reduction & Visual Analysis]
        C --> F[Phase 3: Deep Multiclass Neural Classifier Training]
        C --> G[Phase 4: Conditional GAN Synthetic Augmentation]
        
        F --> H[Phase 5: Adversarial Evasion Benchmarking FGSM & PGD]
        C --> H1[Phase 10: Denoising Autoencoder Feature Pre-Filtering]
        C --> H2[Phase 10: Adversarially Robust Classifier Retraining]
        
        H --> I{Adversarial Perturbation Injected?}
        I --> J1[Sanitization via DAE Pre-Filter]
        J1 --> J2[Classification via Adversarially-Trained Model]
        
        J2 -- Risk Triggered / Bypassed --> K[Phase 6: Context-Aware Zero-Trust Engine]
        K --> L{Context Signal Evaluation Device Trust, Geo Risk, Time, Identity}
        L -- Risk Triggered --> M[Final Interception by Zero-Trust Layer DENY]
        L -- Pass --> N[Access Granted ALLOW]
    end

    subgraph UPCOMING ["Upcoming Research Roadmap (Phases 7 - 9 & 11 - 12)"]
        O[Phase 7: Live eBPF / Scapy Streaming] --> Q[Phase 8: Adaptive RL Policy Engine]
        Q --> R[Phase 9: Federated Edge Learning FedAvg + DP]
        R --> S[Phase 11: Hardware Edge Deployment Raspberry Pi / Jetson]
        S --> T[Phase 12: Automated SOAR & iptables Quarantine]
    end

    M -. Live Feed .-> O
```

---

## 🔍 IoT Attack Taxonomy & Dataset Structure

The framework processes **76 network traffic flow features** (e.g., Flow Duration, Packet Length Statistics, Inter-Arrival Times, Flag Counts) across **10 distinct traffic categories**:

| Label | Category | Description | Primary Attack Vectors | Status |
| :---: | :--- | :--- | :--- | :---: |
| `0` | **Benign** | Normal, uncompromised IoT device traffic | Standard MQTT, HTTP, DNS, telemetry | Analyzed |
| `1` | **DoS / DDoS** | Flood of packet volumes targeting resource exhaustion | SYN Flood, UDP Burst, ICMP Flood | Analyzed |
| `2` | **PortScan** | Network reconnaissance & vulnerability probing | Nmap SYN/ACK scans, IP sweeps | Analyzed |
| `3` | **Botnet** | Command & Control (C2) communication vectors | Mirai, Gafgyt, Reaper heartbeats | Augmented via cGAN |
| `4` | **Brute Force** | Automated credential harvesting attempts | SSH / Telnet dictionary attacks | Augmented via cGAN |
| `5` | **Web Attack** | Exploitation of web interfaces on IoT gateways | SQLi, XSS, Command Injection | Augmented via cGAN |
| `6` | **Infiltration** | Post-exploitation lateral movement & privilege escalation | Internal pivoting, backdoor staging | Analyzed |
| `7` | **Heartbleed / Exploits** | OpenSSL buffer over-read exploitation | Memory leakage attacks | Analyzed |
| `8` | **Volumetric Scans** | Service enumeration & banner grabbing | Vulnerability discovery probes | Augmented via cGAN |
| `9` | **Advanced Persistent Threat** | Multi-stage persistent access attempts | Stealthy exfiltration, custom C2 | Augmented via cGAN |

---

## 🗺️ Detailed Roadmap & Execution Plan

### 🟩 Part 1: Completed Milestones (What We Have Built So Far)

#### 📍 Phase 1: Data Ingestion & Stratified Split `[COMPLETED ✅]`
- **Objective**: Isolate held-out testing data prior to any preprocessing to eliminate data leakage.
- **Implementation**: [data_split.py](file:///c:/Users/Harshita/IOT_Research/data_split.py)
- **Mechanism**: Stratified 80/20 train-test split preserving identical 10-class proportions in `X_test_isolated.csv` and `y_test_isolated.csv`.

#### 📍 Phase 2: Hybrid Class Imbalance Resolution & UMAP Visualization `[COMPLETED ✅]`
- **Objective**: Balance minority attack samples while removing noisy boundary overlaps, followed by high-dimensional visualization.
- **Implementation**: [balance.py](file:///c:/Users/Harshita/IOT_Research/balance.py) & [reduction.py](file:///c:/Users/Harshita/IOT_Research/reduction.py)
- **Mechanism**:
  1. *SMOTE (Synthetic Minority Over-sampling Technique)* synthesizes minority attack vectors.
  2. *Tomek Links* removes ambiguous border instances between classes to clean decision boundaries.
  3. *UMAP (Uniform Manifold Approximation and Projection)* projects flow features into 2D embeddings (`umap_embedding.csv`, `umap_plot.png`) for visual cluster separation analysis.

#### 📍 Phase 3: Multiclass Deep Neural Network Classifier `[COMPLETED ✅]`
- **Objective**: Build and train a robust deep neural network for multiclass intrusion detection.
- **Implementation**: [src/risk_engine/Network_classifier.py](file:///c:/Users/Harshita/IOT_Research/src/risk_engine/Network_classifier.py) & [src/risk_engine/new_train_baseline.py](file:///c:/Users/Harshita/IOT_Research/src/risk_engine/new_train_baseline.py)
- **Architecture**:
  - `Linear(76 -> 128)` ➔ `BatchNorm1d` ➔ `ReLU` ➔ `Dropout(0.3)`
  - `Linear(128 -> 64)` ➔ `BatchNorm1d` ➔ `ReLU` ➔ `Dropout(0.3)`
  - `Linear(64 -> 32)` ➔ `ReLU`
  - `Linear(32 -> 10)` (Output Logits over 10 classes)
- **Loss & Optimization**: Class-weighted `CrossEntropyLoss` with majority undersampling (`MAJORITY_CAP=40,000`) and Adam optimizer (`lr=0.001`). Saved weights: `src/risk_engine/models/network_risk_classifier_multiclass.pth`.

#### 📍 Phase 4: Synthetic Minority Augmentation via Conditional GAN (cGAN) `[COMPLETED ✅]`
- **Objective**: Supplement rare/underperforming attack classes with realistic synthetic flow samples.
- **Implementation**: [src/risk_engine/gan_augmentation.py](file:///c:/Users/Harshita/IOT_Research/src/risk_engine/gan_augmentation.py)
- **Mechanism**: Generator conditions on random noise vector (`NOISE_DIM=32`) + target attack class label. Output features are inverse-transformed and clipped to physical min/max bounds.

#### 📍 Phase 5: Adversarial Evasion Benchmarking (FGSM, PGD & HopSkipJump) `[COMPLETED ✅]`
- **Objective**: Measure the classifier's vulnerability to gradient-based (white-box) and decision-based (black-box) adversarial evasion tactics across fixed epsilons and continuous perturbation sweeps.
- **Implementation**: [src/risk_engine/adversial_attack.py](file:///c:/Users/Harshita/IOT_Research/src/risk_engine/adversial_attack.py), [src/risk_engine/blackbox.py](file:///c:/Users/Harshita/IOT_Research/src/risk_engine/blackbox.py) & [src/risk_engine/sweep.py](file:///c:/Users/Harshita/IOT_Research/src/risk_engine/sweep.py)
- **Attacks Evaluated**: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD, 10-step), and HopSkipJump Black-Box Attack.

#### 📍 Phase 6: Context-Aware Zero-Trust Policy Engine (Defense-in-Depth) `[COMPLETED ✅]`
- **Objective**: Ensure secondary defense mechanisms block attacks that succeed in fooling the ML classifier.
- **Implementation**: [src/risk_engine/zero_trust_engine.py](file:///c:/Users/Harshita/IOT_Research/src/risk_engine/zero_trust_engine.py)
- **Mechanism**: 8-rule prioritized policy chain incorporating ML risk score (`1 - P(benign)`) alongside device trust, geo-risk score, time-of-day, and identity verification.

#### 📍 Phase 10: Advanced Adversarial Robustness & Denoising Autoencoders `[COMPLETED ✅]`
- **Objective**: Harden the deep neural network against adversarial noise prior to and during inference.
- **Implementation**: [src/risk_engine/denoising_autoencoder.py](file:///c:/Users/Harshita/IOT_Research/src/risk_engine/denoising_autoencoder.py), [src/risk_engine/adversarial_training.py](file:///c:/Users/Harshita/IOT_Research/src/risk_engine/adversarial_training.py), [src/risk_engine/evaluate_phase10.py](file:///c:/Users/Harshita/IOT_Research/src/risk_engine/evaluate_phase10.py) & [phase10_benchmark.py](file:///c:/Users/Harshita/IOT_Research/phase10_benchmark.py)
- **Mechanism**:
  1. **Denoising Autoencoder (DAE)**: Symmetric autoencoder (`76 -> 64 -> 32 -> 16 -> 32 -> 64 -> 76`) trained with composite Gaussian/uniform perturbation and feature dropout to project perturbed vectors back onto the natural flow manifold (Val MSE: **0.08655**).
  2. **Adversarial Retraining (AT)**: Retrains `NetworkRiskClassifier` with composite 50% clean + 50% FGSM adversarial minibatches.
  3. **Multi-Tier Defense Evaluation**: Evaluates 4 defense configurations across continuous FGSM epsilon sweeps and 10-step PGD attacks.

---

### 🟦 Part 2: Upcoming Research Roadmap (What We Will Continue To Build)

#### 📍 Phase 7: Live Packet Streaming & Real-Time eBPF Pipeline `[UPCOMING 🚀]`
- **Goal**: Transition from static offline CSV dataset processing to live packet stream ingestion.
- **Planned Work**:
  - Implement eBPF/XDP kernel probes and Scapy-based network sniffers on Linux IoT gateways.
  - Dynamically calculate sliding window flow metrics (Flow Duration, Bytes/sec, Packet Length Variances) to feed directly into the PyTorch classifier.

#### 📍 Phase 8: Dynamic & Adaptive Policy Engine via Reinforcement Learning (RL) `[UPCOMING 🚀]`
- **Goal**: Replace static policy thresholds with an adaptive online learning agent.
- **Planned Work**:
  - Implement a Deep Q-Network (DQN) or Contextual Multi-Armed Bandit agent to adjust rule priorities and threshold weights dynamically based on real-time feedback.
  - Adapt policy sensitivity automatically when detecting novel threat vectors or shifting contextual risk patterns.

#### 📍 Phase 9: Privacy-Preserving Federated Edge Learning (FedAvg + DP) `[UPCOMING 🚀]`
- **Goal**: Train decentralized intrusion models across multiple isolated IoT edge nodes without aggregating raw packet payloads centrally.
- **Planned Work**:
  - Deploy Federated Averaging (`FedAvg`) across distributed IoT gateways.
  - Integrate Differential Privacy (DP) noise injection to prevent membership inference attacks while preserving global model accuracy.

#### 📍 Phase 11: Physical Edge Hardware Deployment & Benchmarking `[UPCOMING 🚀]`
- **Goal**: Benchmark real-world performance, latency, and resource footprint on physical IoT edge hardware.
- **Planned Work**:
  - Export PyTorch models to ONNX / TensorRT format for optimized edge runtime.
  - Deploy onto physical testbeds (Raspberry Pi 4, NVIDIA Jetson Nano, ESP32 gateways).
  - Measure packet processing latency (ms), throughput (Mpps), memory overhead (RAM), and battery consumption under live DDoS simulations.

#### 📍 Phase 12: Automated Incident Response & SOAR Integration `[UPCOMING 🚀]`
- **Goal**: Translate Zero-Trust policy decisions into instant automated network defenses.
- **Planned Work**:
  - Automatically script `iptables` / `nftables` firewall drop rules upon a `DENY` decision.
  - Integrate with SIEM/SOAR platforms (Elastic Stack, Splunk, Webhooks) for instant security operation alerts and automated micro-segmentation quarantine.

---

## 📊 Key Results & Multi-Tier Defense Performance

Empirical benchmark across 1,000 correctly identified attack flows under adversarial perturbation:

| Defense Tier / Configuration | FGSM ($\epsilon=0.15$) | FGSM ($\epsilon=0.30$) | PGD (10-Step, $\epsilon=0.15$) | Zero-Trust Interception Rate | Defense Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tier 1: Baseline Classifier** (No Defense) | **87.0%** | **93.0%** | **95.0%** | Vulnerable to Evasion | Benchmark Established ✅ |
| **Tier 2: DAE Pre-Filter Sanitizer** | **46.1%** | **58.4%** | **38.2%** | Moderate (~57% Evasion Drop) | Implemented & Verified ✅ |
| **Tier 3: Adversarially-Trained Model** | **31.3%** | **48.5%** | **31.6%** | High (~63% Evasion Drop) | Implemented & Verified ✅ |
| **Tier 4: Full Defense-in-Depth (DAE + AT + ZT)** | **0.0%** | **0.0%** | **0.0%** | **100.0% Intercepted** | Implemented & Verified ✅ |

---

## 📂 Repository Directory Structure

```
IOT_Research/
├── README.md                           # Main Project Overview & Architecture Guide
├── Data.csv                            # Raw IoT Network Flow Features (76 cols)
├── Label.csv                           # Raw Multiclass Traffic Labels (0-9)
├── data_split.py                       # Phase 1: Stratified Train/Test Data Isolation [COMPLETED]
├── balance.py                          # Phase 2: SMOTE + Tomek Links Hybrid Balancing [COMPLETED]
├── reduction.py                        # Phase 2: UMAP 2D Feature Reduction & Plotting [COMPLETED]
├── verify.py                           # Dataset Proportions & Integrity Verification [COMPLETED]
├── blackbox.py                         # Root entrypoint for HopSkipJump Black-Box Attack [COMPLETED]
├── sweep.py                            # Root entrypoint for Multi-Epsilon Robustness Sweep [COMPLETED]
├── denoising_autoencoder.py            # Root entrypoint for Denoising Autoencoder Training [COMPLETED]
├── adversarial_training.py             # Root entrypoint for Adversarial Retraining Pipeline [COMPLETED]
├── phase10_benchmark.py                # Root entrypoint for Comprehensive Phase 10 Evaluation [COMPLETED]
└── src/
    └── risk_engine/
        ├── Network_classifier.py       # Multiclass PyTorch Neural Network Definition [COMPLETED]
        ├── new_train_baseline.py       # Phase 3: Classifier Training Pipeline [COMPLETED]
        ├── gan_augmentation.py         # Phase 4: Conditional GAN Generator & Discriminator [COMPLETED]
        ├── zero_trust_engine.py        # Phase 6: 8-Rule Prioritized Zero-Trust Policy Engine [COMPLETED]
        ├── adversial_attack.py         # Phase 5 & 6: FGSM/PGD Attack & Defense Benchmark [COMPLETED]
        ├── blackbox.py                 # Phase 5 & 6: ART HopSkipJump Black-Box Attack Benchmark [COMPLETED]
        ├── sweep.py                    # Phase 5 & 6: Epsilon Sweep Execution Module [COMPLETED]
        ├── denoising_autoencoder.py    # Phase 10: Deep DAE Pre-Filtering Sanitizer [COMPLETED]
        ├── adversarial_training.py     # Phase 10: Adversarially Robust Classifier Retraining [COMPLETED]
        ├── evaluate_phase10.py         # Phase 10: Comprehensive Multi-Tier Defense Benchmark [COMPLETED]
        ├── phase10_results.csv         # Numerical Multi-Tier Evasion & Interception Metrics [COMPLETED]
        ├── phase10_defense_comparison.png # Multi-Curve Evasion vs. Perturbation Plot [COMPLETED]
        └── models/
            ├── network_risk_classifier_multiclass.pth  # Baseline Classifier Checkpoint [COMPLETED]
            ├── denoising_autoencoder.pth               # DAE Feature Sanitizer Checkpoint [COMPLETED]
            └── network_risk_classifier_adversarial.pth # Adversarially-Trained Classifier [COMPLETED]
```

---

## 🚀 Quickstart & Execution Guide

### 1. Prerequisites & Environment Setup
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### 2. Train Models (Phases 3 & 10)
```bash
# Train Baseline Classifier
python src/risk_engine/new_train_baseline.py

# Train Denoising Autoencoder (DAE) Sanitizer
python denoising_autoencoder.py

# Train Adversarially-Robust Classifier
python adversarial_training.py
```

### 3. Run Phase 10 Multi-Tier Defense Benchmark
```bash
python phase10_benchmark.py
```
*(Generates `phase10_results.csv` and `phase10_defense_comparison.png`)*

---

## 🔬 Research Significance & Contributions

1. **Defense-in-Depth Pre-Filtering**: Placing a Denoising Autoencoder ahead of the neural classifier projects adversarial perturbations back onto the natural data manifold, slashing PGD bypass from 95% down to 38.2%.
2. **Adversarial Retraining Hardening**: Incorporating FGSM minibatches during training directly smooths the classifier's decision boundaries, reducing PGD vulnerability to 31.6%.
3. **100% Zero-Trust Defense Boundary**: Even when high-strength adversarial payloads fool both the DAE and the classifier, the prioritized Context-Aware Zero-Trust Policy Engine intercepts 100% of evasion attempts.

