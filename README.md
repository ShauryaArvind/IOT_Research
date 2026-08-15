# 🛡️ Adversarial-Resilient IoT Network Intrusion Detection

> **Status**: 🚧 Work in Progress — actively under development  
> A research project investigating how machine learning-based intrusion detection systems for IoT networks hold up under adversarial attack, and whether adding a non-ML, context-aware defense layer can close the gap when the model itself gets fooled.

---

## 📌 Overview

IoT networks are an especially attractive target for attackers: large numbers of constrained, often poorly-secured devices generating high volumes of traffic. Machine learning-based Network Intrusion Detection Systems (NIDS) are a common defense — but ML models themselves can be deliberately fooled by adversarial attacks, small calculated perturbations to input data that cause a model to misclassify malicious traffic as benign.

This project asks two connected questions:
1. **How vulnerable is a realistic multiclass intrusion detection model to adversarial attacks (FGSM, PGD)?**
2. **Can a secondary, non-ML defense layer (a rule-based Zero-Trust policy engine using contextual signals) catch attacks that successfully fool the model, even when the model alone fails?**

The project is built on network flow data (CICIDS-2017-style features) and draws design inspiration from two open-source references:
- [adversarial-ml-security-framework](https://github.com/Aarnav-Singh/adversarial-ml-security-framework) — the ML classifier + Zero-Trust policy engine design
- [FLVision](https://github.com/rohanadepu/FLVision) — the Conditional GAN concept for synthetic minority-class data generation

---

## 🔍 What We're Analyzing

Rather than treating "attack detection" as a single problem, this project separates it into distinct, testable layers:

| Layer | Question it answers |
| :--- | :--- |
| **Base classifier** | Can a neural network correctly classify 10 categories of traffic (benign + 9 attack types)? |
| **Adversarial attacks** | How easily can that classifier be deliberately fooled, and does attack strength matter (FGSM vs. PGD)? |
| **Zero-Trust defense layer** | If the classifier is fooled, can additional context (device trust, geographic risk, etc.) still catch the attack? |
| **Data augmentation (GAN)** | Can synthetic minority-class data improve the classifier's weakest categories? |

---

## 📈 Progress So Far

*This section is updated as the project develops. Current as of the latest working session.*

- ✅ **Reviewed and adapted reference architecture**: Adapted reference ML + Zero-Trust framework architecture for multiclass network security.
- ✅ **Built multiclass classifier**: Constructed a 10-class deep neural network in PyTorch, addressing severe class imbalance (~10:1) via undersampling + class-weighted loss.
- ✅ **Diagnosed & corrected visualization artifacts**: Identified and resolved a PCA visualization artifact (unscaled features producing misleading "ray" patterns) via proper feature standardization.
- ✅ **Trained & evaluated baseline model**: Strong performance on majority classes, weaker on several minority attack classes (documented honestly, including a "whack-a-mole" pattern encountered during manual class-weight tuning).
- ✅ **Implemented adversarial attacks**: Built FGSM and PGD adversarial attacks against the trained classifier — measured bypass rates of up to 96.2% (PGD) when the model is tested alone.
- ✅ **Built Zero-Trust contextual policy layer**: Evaluated Zero-Trust defense on attacked traffic — reduced combined bypass rate to 0.0% in initial testing (see [Limitations & Open Questions](#-limitations--open-questions)).
- 🔄 **In progress (Data Augmentation)**: Conditional GAN for synthetic minority-class data generation — initial small-scale test was inconclusive; full-scale training planned.
- ⏳ **Not yet started**: Real-world label mapping verification, black-box attack testing, epsilon sensitivity sweep, and real IoT-specific dataset validation.

---

## 🗺️ Roadmap — Phase-Wise Plan

| Phase | Goal | Key Tasks | Status |
| :---: | :--- | :--- | :---: |
| **1. Foundations** | Understand reference frameworks & design model | Study `adversarial-ml-security-framework`; decide multiclass scope; finalize 9 selected features | ✅ Done |
| **2. Data Preparation** | Confirm data is clean & understand real structure | UMAP/PCA visualization; identify and fix class imbalance; standardize features | ✅ Done |
| **3. Baseline Model** | Train a working multiclass classifier | Build model architecture; undersampling + class-weighted loss; per-class evaluation (Precision/Recall/F1, confusion matrix) | ✅ Done |
| **4. Model Refinement** | Improve weak-class performance | Manual class-weight tuning (documented limitation: whack-a-mole effect); identify need for a different approach | ✅ Done *(baseline locked in)* |
| **5. Adversarial Evaluation** | Measure real vulnerability | Implement FGSM + PGD; measure ML-only bypass rate | ✅ Done |
| **6. Defense Layer Integration** | Test whether context-based defense closes the gap | Bridge multiclass output to a risk score; integrate Zero-Trust policy engine; measure combined bypass rate | ✅ Done *(initial testing)* |
| **7. Data Augmentation** | Address remaining class imbalance with synthetic data | Build Conditional GAN; generate synthetic minority-class samples; compare augmented vs. baseline performance | 🔄 In progress |
| **8. Label Verification** | Confirm ground-truth meaning of each class | Trace back to original attack-type labels (currently unresolved — numeric-only labels in available files) | ⏳ Planned |
| **9. Broader Attack Testing** | Test robustness beyond FGSM/PGD | Add black-box attacks (no gradient access); test an epsilon sweep (varying attack strength) to characterize the model's robustness curve | ⏳ Planned |
| **10. Realistic Threat Modeling** | Reduce reliance on simulated context assumptions | Test Zero-Trust layer against varied/adversarial context inputs, not just the assumed "low-trust attacker" profile | ⏳ Planned |
| **11. IoT-Specific Validation** | Confirm findings hold on IoT-specific traffic | Evaluate against IoT-focused datasets (e.g., CICIoT2023, IoTBotNet2020) rather than general network flow data alone | ⏳ Planned |
| **12. Documentation & Write-Up** | Consolidate findings into a final report/paper | Full methodology write-up; limitations section; reproducibility instructions | ⏳ Planned |

---

## 🎯 What This Project Aims to Achieve

By the end of this project, the goal is a documented, evidence-based answer to:

> **In a realistic IoT network intrusion detection setting, how much does adding a non-ML, context-aware defense layer actually improve resilience against adversarial attacks — and under what conditions does that improvement hold or break down?**

This means the final deliverable isn't just "a working model" — it's a **comparative security analysis**: model-alone vulnerability vs. defense-in-depth resilience, backed by measured bypass rates, honestly reported limitations, and a clear account of what was tried, what worked, and what didn't.

---

## 🛠️ Tech Stack

- **Python 3, PyTorch**: Model architecture, neural training, FGSM & PGD adversarial attacks
- **scikit-learn**: Preprocessing (`StandardScaler`), train/test splitting, evaluation metrics (Precision/Recall/F1, confusion matrix)
- **pandas / NumPy**: Data manipulation & flow matrix operations
- **UMAP / PCA** (`umap-learn`, `scikit-learn`): Exploratory data visualization and dimensionality reduction

---

## 📂 Repository Structure

```
IOT_Research/
├── README.md                           # Main Project Overview & Architecture Guide
├── Data.csv                            # Raw IoT Network Flow Features (78 cols, gitignored)
├── Label.csv                           # Raw Multiclass Traffic Labels (0-9, gitignored)
├── data_split.py                       # Phase 2: Stratified Train/Test Data Isolation
├── balance.py                          # Phase 2: SMOTE + Tomek Links Hybrid Balancing
├── reduction.py                        # Phase 2: UMAP / PCA 2D Feature Reduction
├── verify.py                           # Dataset Proportions Verification
├── balancing_methodology_results.csv   # Class distribution change metrics
├── models/                             # Saved trained model checkpoints (.pth)
│   └── network_risk_classifier_multiclass.pth
└── src/
    └── risk_engine/
        ├── Network_classifier.py       # Multiclass PyTorch model architecture
        ├── new_train_baseline.py       # Baseline training pipeline (undersampling + class weighting)
        ├── adversial_attack.py         # FGSM + PGD adversarial attack & Zero-Trust evaluation
        ├── zero_trust_engine.py        # Contextual Zero-Trust policy defense layer
        └── gan_augmentation.py         # Conditional GAN for synthetic data generation
```

---

## 📊 Results Snapshot

| Metric | Value |
| :--- | :---: |
| **Baseline Macro-F1 (10-class)** | `0.68` |
| **FGSM bypass rate — model alone** | `89.1%` |
| **PGD bypass rate — model alone** | `96.2%` |
| **FGSM bypass rate — model + Zero-Trust layer** | `0.0%` |
| **PGD bypass rate — model + Zero-Trust layer** | `0.0%` |

*Full per-class breakdowns and methodology are documented in the project report.*

---

## ⚠️ Limitations & Open Questions

- **Simulated Context Assumptions**: The `0%` combined bypass rate depends on simulated attacker-context assumptions (low device trust, elevated geo-risk), since the dataset doesn't include real device/location metadata — this needs testing against more varied, realistic context distributions.
- **Numeric Class Label Mapping**: Numeric class labels (`0–9`) have not yet been fully verified against their real-world attack names; this mapping was not present in the currently available raw data files.
- **GAN Augmentation Validation**: GAN-based data augmentation has only been tested at small scale so far and did not yet show a clear improvement — a full-scale run is planned.

---

## 🙏 Acknowledgments

Design inspiration and reference architecture drawn from:
- [adversarial-ml-security-framework](https://github.com/Aarnav-Singh/adversarial-ml-security-framework) (ML classifier + Zero-Trust policy engine design)
- [FLVision](https://github.com/rohanadepu/FLVision) (Conditional GAN concept for synthetic IoT attack data generation)
