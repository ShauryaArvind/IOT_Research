"""
Phase 10 Comprehensive Defense Benchmark Suite
==============================================
Evaluates and compares 4 defense architectures against adversarial attacks:
  1. Baseline Classifier (No Defense)
  2. Denoising Autoencoder (DAE Pre-Filter Sanitization)
  3. Adversarially-Trained Classifier (AT Hardening)
  4. Full Defense-in-Depth (DAE Sanitizer + AT Classifier + Context-Aware Zero-Trust Engine)

Attacks Tested:
  - Multi-Epsilon Fast Gradient Sign Method (FGSM, eps in [0.0, 0.30])
  - Iterative Projected Gradient Descent (PGD, 10 steps, eps=0.15)

Outputs:
  - Formatted terminal benchmark tables
  - Results CSV: `phase10_results.csv`
  - Multi-curve comparison visualization: `phase10_defense_comparison.png`
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from Network_classifier import NetworkRiskClassifier
from denoising_autoencoder import DenoisingAutoencoder
from zero_trust_engine import ZeroTrustEngine

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
X_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'X_train_final_balanced.csv')
Y_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'y_train_final_balanced.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, 'network_risk_classifier_multiclass.pth')
ADV_MODEL_PATH = os.path.join(MODELS_DIR, 'network_risk_classifier_adversarial.pth')
DAE_MODEL_PATH = os.path.join(MODELS_DIR, 'denoising_autoencoder.pth')

RESULTS_CSV_PATH = os.path.join(os.path.dirname(__file__), 'phase10_results.csv')
PLOT_PATH = os.path.join(os.path.dirname(__file__), 'phase10_defense_comparison.png')

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_EVAL_SAMPLES = 1000  # Number of correctly-classified test samples to evaluate
EPSILON_SWEEP = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def pprint(*args, **kwargs):
    """Print wrapper forcing immediate stdout flush for real-time logging."""
    print(*args, **kwargs)
    sys.stdout.flush()


def load_data_and_split():
    """Load dataset and prepare standardized train/test split."""
    pprint("  Loading tabular dataset...")
    X = pd.read_csv(X_CSV_PATH).replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    y = pd.read_csv(Y_CSV_PATH).iloc[:, 0].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    return X_test_scaled, y_test, X_train.shape[1], int(y.max()) + 1


def fgsm_attack(model, X, y, epsilon):
    """Untargeted FGSM adversarial attack."""
    if epsilon == 0.0:
        return X.clone().detach()

    X_adv = X.clone().detach().requires_grad_(True)
    outputs = model(X_adv)
    loss = nn.CrossEntropyLoss()(outputs, y)
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        X_adv = X_adv + epsilon * X_adv.grad.sign()

    return X_adv.detach()


def pgd_attack(model, X, y, epsilon=0.15, alpha=None, num_steps=10):
    """Untargeted PGD 10-step iterative adversarial attack."""
    if alpha is None:
        alpha = epsilon / 4.0

    X_orig = X.clone().detach()
    X_adv = X.clone().detach()

    for _ in range(num_steps):
        X_adv.requires_grad_(True)
        outputs = model(X_adv)
        loss = nn.CrossEntropyLoss()(outputs, y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            perturbation = torch.clamp(X_adv - X_orig, min=-epsilon, max=epsilon)
            X_adv = X_orig + perturbation

        X_adv = X_adv.detach()

    return X_adv


def simulate_attacker_context(n_samples):
    """
    Simulate contextual signals for attacker flows based on threat model:
    Adversarial actors typically operate from untrusted devices and anomalous geographies.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    contexts = []
    for _ in range(n_samples):
        ctx = {
            'device_trust': float(rng.beta(2, 8)),        # skew low (mean ~0.20)
            'geo_risk': float(rng.beta(8, 2)),            # skew high (mean ~0.80)
            'time_of_day': int(rng.choice([1, 2, 3, 22, 23])), # off-hours
            'identity_verified': bool(rng.random() < 0.15), # mostly unverified
            'resource_sensitivity': float(rng.uniform(0.4, 0.95)),
        }
        contexts.append(ctx)
    return contexts


def evaluate_phase10_suite():
    pprint("=" * 80)
    pprint("PHASE 10: ADVANCED ADVERSARIAL ROBUSTNESS & DENOISING AUTOENCODER BENCHMARK")
    pprint("=" * 80)

    # 1. Load Data
    X_test_scaled, y_test, input_dim, num_classes = load_data_and_split()
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.LongTensor(y_test)

    # 2. Load Models
    pprint("\n[1/5] Loading trained models...")
    
    # Baseline Model
    baseline_model = NetworkRiskClassifier(input_dim=input_dim, num_classes=num_classes)
    if os.path.exists(BASELINE_MODEL_PATH):
        baseline_model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location='cpu'))
        baseline_model.eval()
        pprint("  [+] Baseline Classifier loaded successfully.")
    else:
        raise FileNotFoundError(f"Baseline model not found at {BASELINE_MODEL_PATH}. Please train baseline first.")

    # Adversarially-Trained Model
    adv_model = NetworkRiskClassifier(input_dim=input_dim, num_classes=num_classes)
    if os.path.exists(ADV_MODEL_PATH):
        adv_model.load_state_dict(torch.load(ADV_MODEL_PATH, map_location='cpu'))
        adv_model.eval()
        pprint("  [+] Adversarially-Trained Classifier loaded successfully.")
    else:
        raise FileNotFoundError(f"Adversarial model not found at {ADV_MODEL_PATH}. Please run adversarial training first.")

    # Denoising Autoencoder
    dae_model = DenoisingAutoencoder(input_dim=input_dim, latent_dim=16)
    if os.path.exists(DAE_MODEL_PATH):
        dae_model.load_state_dict(torch.load(DAE_MODEL_PATH, map_location='cpu'))
        dae_model.eval()
        pprint("  [+] Denoising Autoencoder (DAE) loaded successfully.")
    else:
        raise FileNotFoundError(f"DAE model not found at {DAE_MODEL_PATH}. Please run DAE training first.")

    # Zero-Trust Engine
    zt_engine = ZeroTrustEngine()

    # 3. Clean Performance Baseline
    pprint("\n[2/5] Evaluating clean test performance...")
    with torch.no_grad():
        baseline_preds = torch.argmax(baseline_model(X_test_t), dim=1).numpy()
        adv_preds = torch.argmax(adv_model(X_test_t), dim=1).numpy()

    b_acc = accuracy_score(y_test, baseline_preds)
    _, _, b_f1, _ = precision_recall_fscore_support(y_test, baseline_preds, labels=list(range(num_classes)), zero_division=0)
    
    a_acc = accuracy_score(y_test, adv_preds)
    _, _, a_f1, _ = precision_recall_fscore_support(y_test, adv_preds, labels=list(range(num_classes)), zero_division=0)

    pprint(f"  Baseline Model     -> Clean Accuracy: {b_acc*100:.2f}% | Macro-F1: {b_f1.mean():.4f}")
    pprint(f"  Adversarial Model  -> Clean Accuracy: {a_acc*100:.2f}% | Macro-F1: {a_f1.mean():.4f}")

    # Select correctly-classified evaluation subset (where true label != 0 for attack evasion test)
    correct_mask = (baseline_preds == y_test) & (y_test != 0)
    correct_indices = np.where(correct_mask)[0]
    
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples = min(N_EVAL_SAMPLES, len(correct_indices))
    eval_indices = rng.choice(correct_indices, size=n_samples, replace=False)

    X_eval = X_test_t[eval_indices]
    y_eval = y_test_t[eval_indices]
    attacker_contexts = simulate_attacker_context(n_samples)

    pprint(f"  Evaluation subset: {n_samples} correctly-classified attack flows.")

    # 4. Multi-Epsilon Sweep Across 4 Defense Configurations
    pprint("\n[3/5] Executing Multi-Epsilon FGSM Robustness Sweep...")
    pprint("-" * 95)
    pprint(f"{'Epsilon':<9} | {'1. Baseline (Raw)':<18} | {'2. DAE Sanitizer':<18} | {'3. Adv-Trained':<18} | {'4. Full Defense (ZT)':<20}")
    pprint("-" * 95)

    sweep_records = []

    for eps in EPSILON_SWEEP:
        # Generate FGSM perturbations targeting Baseline model
        X_adv_fgsm = fgsm_attack(baseline_model, X_eval, y_eval, eps)

        # Config 1: Baseline Model (Raw perturbed input)
        with torch.no_grad():
            preds_raw = torch.argmax(baseline_model(X_adv_fgsm), dim=1)
            raw_bypass = (preds_raw != y_eval).float().mean().item() * 100.0

        # Config 2: DAE Sanitizer -> Baseline Model
        with torch.no_grad():
            X_sanitized = dae_model(X_adv_fgsm)
            preds_dae = torch.argmax(baseline_model(X_sanitized), dim=1)
            dae_bypass = (preds_dae != y_eval).float().mean().item() * 100.0

        # Config 3: Adversarially-Trained Model (Direct on perturbed input)
        with torch.no_grad():
            preds_adv = torch.argmax(adv_model(X_adv_fgsm), dim=1)
            adv_bypass = (preds_adv != y_eval).float().mean().item() * 100.0

        # Config 4: Full Defense-in-Depth (DAE Sanitizer -> AT Model -> Zero-Trust Engine)
        with torch.no_grad():
            X_full_sanitized = dae_model(X_adv_fgsm)
            probs_full = torch.softmax(adv_model(X_full_sanitized), dim=1)
            # ML risk score = 1 - P(Class 0 / Benign)
            risk_scores = (1.0 - probs_full[:, 0]).cpu().numpy()

        denied_count = 0
        for i in range(n_samples):
            decision = zt_engine.evaluate(ml_risk_score=float(risk_scores[i]), context=attacker_contexts[i])
            if decision.decision == 'DENY':
                denied_count += 1

        full_bypass = (1.0 - (denied_count / n_samples)) * 100.0

        sweep_records.append({
            'epsilon': eps,
            'baseline_bypass': raw_bypass,
            'dae_bypass': dae_bypass,
            'adv_model_bypass': adv_bypass,
            'full_defense_bypass': full_bypass,
            'zt_interception_rate': (denied_count / n_samples) * 100.0
        })

        pprint(f"{eps:<9.2f} | {raw_bypass:>16.1f}% | {dae_bypass:>16.1f}% | {adv_bypass:>16.1f}% | {full_bypass:>18.1f}%")

    # 5. PGD-10 Attack Evaluation
    pprint("\n[4/5] Evaluating Multi-Step PGD Attack (10 steps, eps=0.15)...")
    X_adv_pgd = pgd_attack(baseline_model, X_eval, y_eval, epsilon=0.15, num_steps=10)

    with torch.no_grad():
        pgd_raw_preds = torch.argmax(baseline_model(X_adv_pgd), dim=1)
        pgd_raw_bypass = (pgd_raw_preds != y_eval).float().mean().item() * 100.0

        X_pgd_sanitized = dae_model(X_adv_pgd)
        pgd_dae_preds = torch.argmax(baseline_model(X_pgd_sanitized), dim=1)
        pgd_dae_bypass = (pgd_dae_preds != y_eval).float().mean().item() * 100.0

        pgd_adv_preds = torch.argmax(adv_model(X_adv_pgd), dim=1)
        pgd_adv_bypass = (pgd_adv_preds != y_eval).float().mean().item() * 100.0

        X_pgd_full = dae_model(X_adv_pgd)
        pgd_full_probs = torch.softmax(adv_model(X_pgd_full), dim=1)
        pgd_risk_scores = (1.0 - pgd_full_probs[:, 0]).cpu().numpy()

    pgd_denied_count = 0
    for i in range(n_samples):
        decision = zt_engine.evaluate(ml_risk_score=float(pgd_risk_scores[i]), context=attacker_contexts[i])
        if decision.decision == 'DENY':
            pgd_denied_count += 1

    pgd_full_bypass = (1.0 - (pgd_denied_count / n_samples)) * 100.0

    pprint("-" * 95)
    pprint(f"PGD-10 (eps=0.15) | {pgd_raw_bypass:>16.1f}% | {pgd_dae_bypass:>16.1f}% | {pgd_adv_bypass:>16.1f}% | {pgd_full_bypass:>18.1f}%")
    pprint("=" * 95)

    # 6. Save Results to CSV
    df_results = pd.DataFrame(sweep_records)
    df_results.to_csv(RESULTS_CSV_PATH, index=False)
    pprint(f"\n[5/5] Saved numerical results to: {RESULTS_CSV_PATH}")

    # 7. Generate High-Quality Publication Plot
    generate_comparison_plot(df_results, pgd_results={
        'raw': pgd_raw_bypass,
        'dae': pgd_dae_bypass,
        'adv': pgd_adv_bypass,
        'full': pgd_full_bypass
    })

    return df_results


def generate_comparison_plot(df: pd.DataFrame, pgd_results: dict):
    """Generates a multi-line comparison plot of all 4 defense configurations."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [2.2, 1]})

    # --- Plot 1: FGSM Epsilon Sweep Curve ---
    epsilons = df['epsilon'].values
    ax1.plot(epsilons, df['baseline_bypass'], 'r-o', linewidth=2.2, markersize=6, label='1. Baseline (No Defense)')
    ax1.plot(epsilons, df['dae_bypass'], 'm--s', linewidth=2.0, markersize=5, label='2. DAE Sanitizer Pre-Filter')
    ax1.plot(epsilons, df['adv_model_bypass'], 'b-.^', linewidth=2.0, markersize=5, label='3. Adversarially-Trained Model')
    ax1.plot(epsilons, df['full_defense_bypass'], 'g-D', linewidth=2.5, markersize=7, label='4. Full Defense-in-Depth (DAE + AT + ZT)')

    ax1.set_title('Adversarial Evasion Bypass vs. Perturbation Strength (FGSM)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel(r'Perturbation Magnitude $\epsilon$ (Standardized Units)', fontsize=11)
    ax1.set_ylabel('Evasion Bypass Rate (%)', fontsize=11)
    ax1.set_ylim(-2, 102)
    ax1.set_xlim(-0.01, 0.31)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='upper left')

    # --- Plot 2: PGD-10 (eps=0.15) Bar Comparison ---
    categories = ['Baseline', 'DAE', 'Adv-Train', 'Full ZT']
    values = [pgd_results['raw'], pgd_results['dae'], pgd_results['adv'], pgd_results['full']]
    colors = ['#d9534f', '#f0ad4e', '#0275d8', '#5cb85c']

    bars = ax2.bar(categories, values, color=colors, width=0.55, edgecolor='black', linewidth=1)
    ax2.set_title('10-Step PGD Attack (eps=0.15)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel('Bypass Rate (%)', fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    pprint(f"  [+] Saved defense comparison visualization plot to: {PLOT_PATH}")


if __name__ == '__main__':
    evaluate_phase10_suite()
