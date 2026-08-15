"""
Adversarial Attack + Zero-Trust Defense Evaluation
====================================================
Tests how easily the trained multiclass NetworkRiskClassifier can be fooled
by an FGSM adversarial attack, then checks whether a Zero-Trust contextual
policy layer catches the fooled samples anyway.

This bridges two things that don't naturally fit together:
  - Your model outputs 10 class probabilities (multiclass)
  - The repo's ZeroTrustEngine expects a single 0-1 "risk score" (binary-style)

Bridge used: risk_score = 1 - P(predicted class == 0 / benign)
i.e. "how much probability mass the model puts on ANY attack class."

Steps:
  1. Recreate the exact same train/test split + scaling used in training
     (same random seed), so preprocessing matches the saved model.
  2. Load the trained model.
  3. Take a batch of TEST samples the model currently classifies correctly.
  4. Run untargeted FGSM: nudge each sample to try to flip the prediction.
  5. Measure "ML-only bypass rate": how often the attack succeeds.
  6. Simulate Zero-Trust context signals for these (attacker-controlled)
     samples per the repo's documented threat model: low device trust,
     high geo-risk (see zero_trust_engine.py docstring).
  7. Run the Zero-Trust engine on the SAME fooled samples and measure the
     "combined bypass rate" -- how many actually get through both layers.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from Network_classifier import NetworkRiskClassifier
from zero_trust_engine import ZeroTrustEngine

# ---------------------------------------------------------------------------
# CONFIG -- keep in sync with new_train_baseline.py so preprocessing matches
# ---------------------------------------------------------------------------
X_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'X_train_final_balanced.csv')
Y_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'y_train_final_balanced.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'network_risk_classifier_multiclass.pth')

TEST_SIZE = 0.2
RANDOM_STATE = 42
EPSILON = 0.15          # attack strength (in standardized feature units)
N_ATTACK_SAMPLES = 2000  # how many correctly-classified test samples to attack


def load_and_prepare_data():
    """Recreate the exact same split + scaling used during training."""
    X = pd.read_csv(X_CSV_PATH)
    y = pd.read_csv(Y_CSV_PATH).iloc[:, 0]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.values.astype(np.float32)
    y = y.values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # NOTE: this refits the scaler on the FULL (non-undersampled) train split,
    # which is a slight approximation of the exact scaler used in training
    # (that one was fit on the undersampled set). Feature-wise means/stds are
    # very close either way since undersampling only removed Class-0 rows,
    # but for a perfect match, save/load the actual scaler.fit() object from
    # training instead of refitting here.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, y_test


def fgsm_attack(model, X, y, epsilon):
    """Untargeted FGSM: nudge inputs to INCREASE loss on the true label,
    pushing the model toward misclassifying them. One single step."""
    X = X.clone().detach().requires_grad_(True)
    outputs = model(X)
    loss = nn.CrossEntropyLoss()(outputs, y)
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        X_adv = X + epsilon * X.grad.sign()

    return X_adv.detach()


def pgd_attack(model, X, y, epsilon, alpha=None, num_steps=10):
    """Untargeted PGD: like FGSM, but takes several small steps instead of
    one big jump, re-checking the gradient each time and clipping back into
    an epsilon-ball around the original input. Generally a STRONGER attack
    than FGSM since it can find better perturbations through iteration.

    Args:
        epsilon: max total perturbation allowed per feature (same "budget"
            as FGSM, for a fair comparison).
        alpha: step size per iteration. Defaults to epsilon / 4.
        num_steps: number of gradient steps to take within the epsilon-ball.
    """
    if alpha is None:
        alpha = epsilon / 4

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
            # project back into the epsilon-ball around the original input
            perturbation = torch.clamp(X_adv - X_orig, min=-epsilon, max=epsilon)
            X_adv = X_orig + perturbation

        X_adv = X_adv.detach()

    return X_adv


def run_attack_and_defense(model, X_attack, y_attack, attack_fn, attack_name, epsilon):
    """Runs one attack, then evaluates the Zero-Trust layer on whatever
    fools the model. Returns a dict of summary numbers for easy comparison."""
    print(f"\n--- {attack_name} (epsilon={epsilon}) ---")

    X_adv = attack_fn(model, X_attack, y_attack, epsilon)

    with torch.no_grad():
        adv_logits = model(X_adv)
        adv_preds = torch.argmax(adv_logits, dim=1)
        adv_probs = torch.softmax(adv_logits, dim=1).numpy()

    fooled_mask = (adv_preds != y_attack).numpy()
    n_attack = len(y_attack)
    ml_bypass_rate = fooled_mask.mean()
    print(f"  ML-only bypass rate: {ml_bypass_rate*100:.1f}% "
          f"({fooled_mask.sum()}/{n_attack} attacks succeeded)")

    fooled_indices = np.where(fooled_mask)[0]
    n_fooled = len(fooled_indices)

    if n_fooled == 0:
        print("  No samples were fooled -- nothing for Zero-Trust to catch.")
        return {
            'attack': attack_name, 'ml_bypass_rate': ml_bypass_rate,
            'combined_bypass_rate': 0.0,
        }

    risk_scores = 1.0 - adv_probs[fooled_indices, 0]
    contexts = simulate_attacker_context(n_fooled)

    full_engine = ZeroTrustEngine()
    full_decisions = full_engine.evaluate_batch(risk_scores, contexts)
    full_denied = sum(1 for d in full_decisions if d.decision == 'DENY')
    full_final_bypass = 1 - (full_denied / n_fooled)

    print(f"  Of the {n_fooled} fooled samples, Zero-Trust denies: "
          f"{full_denied}/{n_fooled} ({full_denied/n_fooled*100:.1f}%)")

    combined_bypass_rate = ml_bypass_rate * full_final_bypass
    print(f"  Combined (ML + Zero-Trust) bypass rate: {combined_bypass_rate*100:.1f}%")

    return {
        'attack': attack_name,
        'ml_bypass_rate': ml_bypass_rate,
        'combined_bypass_rate': combined_bypass_rate,
    }


def simulate_attacker_context(n_samples, seed=RANDOM_STATE):
    """Simulate Zero-Trust context signals for attack traffic, per the
    repo's documented threat model: attackers crafting adversarial flows
    are assumed to run from low-trust devices and anomalous locations."""
    rng = np.random.default_rng(seed)
    contexts = []
    for _ in range(n_samples):
        contexts.append({
            'device_trust': float(rng.uniform(0.0, 0.4)),       # low trust
            'geo_risk': float(rng.uniform(0.5, 1.0)),            # elevated risk
            'time_of_day': int(rng.integers(0, 24)),
            'identity_verified': bool(rng.random() > 0.6),       # often unverified
            'resource_sensitivity': float(rng.uniform(0.3, 1.0)),
        })
    return contexts


def main():
    print("=" * 70)
    print("Adversarial Attack + Zero-Trust Defense Evaluation")
    print("=" * 70)

    print("\n[1/5] Rebuilding test set with matching preprocessing...")
    X_test, y_test = load_and_prepare_data()
    num_classes = int(y_test.max()) + 1
    print(f"  Test set: {X_test.shape}, classes: {num_classes}")

    print("\n[2/5] Loading trained model...")
    model = NetworkRiskClassifier(input_dim=X_test.shape[1], num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    print("\n[3/5] Selecting correctly-classified samples to attack...")
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    with torch.no_grad():
        preds = torch.argmax(model(X_test_t), dim=1)
    correct_mask = (preds == y_test_t).numpy()
    correct_idx = np.where(correct_mask)[0]

    n_attack = min(N_ATTACK_SAMPLES, len(correct_idx))
    rng = np.random.default_rng(RANDOM_STATE)
    attack_idx = rng.choice(correct_idx, size=n_attack, replace=False)
    print(f"  Attacking {n_attack} samples the model currently gets RIGHT")

    X_attack = X_test_t[attack_idx]
    y_attack = y_test_t[attack_idx]

    print(f"\n[4/5] Running attacks (epsilon={EPSILON})...")
    fgsm_result = run_attack_and_defense(
        model, X_attack, y_attack, fgsm_attack, "FGSM (single-step)", EPSILON
    )
    pgd_result = run_attack_and_defense(
        model, X_attack, y_attack, pgd_attack, "PGD (10-step iterative)", EPSILON
    )

    print("\n" + "=" * 70)
    print("SUMMARY -- FGSM vs PGD")
    
    print("=" * 70)
    print(f"{'Attack':<25}{'ML-only bypass':>18}{'ML+ZeroTrust bypass':>22}")
    for r in (fgsm_result, pgd_result):
        print(f"{r['attack']:<25}{r['ml_bypass_rate']*100:>17.1f}%{r['combined_bypass_rate']*100:>21.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()