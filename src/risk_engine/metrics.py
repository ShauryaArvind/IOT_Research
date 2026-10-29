"""
Metrics used across the evaluation scripts.

Clean metrics are always reported on the full isolated test set. Robust
metrics are always reported next to the clean numbers for the same model,
because a defence that lowers attack success by misclassifying everything is
not a defence.

Two adversarial success notions are kept separate:
  evasion_rate     attack flow -> predicted BENIGN. This is what matters for an IDS.
  misclass_rate    attack flow -> any wrong class (attack A -> attack B counts).
                   This is what the legacy scripts called "bypass".
"""

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support,
)

from data import BENIGN, NUM_CLASSES


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(y_true, y_pred, fn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of fn(y_true, y_pred), resampling within each
    class so rare classes (49 test rows for class 9) are always present."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    by_class = [np.where(y_true == c)[0] for c in np.unique(y_true)]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_class])
        stats[b] = fn(y_true[idx], y_pred[idx])
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def expected_calibration_error(y_true, proba, n_bins: int = 15) -> float:
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0)


def clean_metrics(y_true, proba, n_boot: int = 1000, seed: int = 0) -> Dict:
    """Full clean-test report. `proba` has shape (N, NUM_CLASSES)."""
    y_true = np.asarray(y_true)
    y_pred = proba.argmax(axis=1)
    labels = list(range(NUM_CLASSES))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pr_auc = []
    for c in labels:
        pos = (y_true == c)
        pr_auc.append(float(average_precision_score(pos, proba[:, c])) if pos.any() else np.nan)

    is_attack = y_true != BENIGN
    flagged = y_pred != BENIGN
    out = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1.mean(),
        'weighted_f1': f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'macro_pr_auc': float(np.nanmean(pr_auc)),
        'ece': expected_calibration_error(y_true, proba),
        # binary IDS view: attack (any class) vs benign
        'attack_detection_rate': float(flagged[is_attack].mean()) if is_attack.any() else np.nan,
        'benign_fpr': float(flagged[~is_attack].mean()) if (~is_attack).any() else np.nan,
        'per_class': {
            'precision': p.tolist(), 'recall': r.tolist(), 'f1': f1.tolist(),
            'pr_auc': pr_auc, 'support': sup.tolist(),
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if n_boot:
        out['macro_f1_ci'] = bootstrap_ci(y_true, y_pred, macro_f1, n_boot, seed)
        out['mcc_ci'] = bootstrap_ci(y_true, y_pred, matthews_corrcoef, n_boot, seed)
    return out


def robust_metrics(y_true, pred_clean, pred_adv, flagged_adv: Optional[np.ndarray] = None) -> Dict:
    """Adversarial metrics on a set of ATTACK rows (y_true != BENIGN).

    `flagged_adv` (optional) marks rows a detector rejected; a rejected row is
    never an evasion regardless of the classifier output.
    """
    y_true = np.asarray(y_true)
    assert (y_true != BENIGN).all(), "robust_metrics expects attack rows only"
    pred_benign = pred_adv == BENIGN
    if flagged_adv is not None:
        pred_benign = pred_benign & ~flagged_adv
    n = len(y_true)
    k_ev = int(pred_benign.sum())
    k_mis = int((pred_adv != y_true).sum())
    per_class = {}
    for c in np.unique(y_true):
        m = y_true == c
        kc = int(pred_benign[m].sum())
        per_class[int(c)] = {'n': int(m.sum()), 'evasion_rate': kc / m.sum(),
                             'evasion_ci': wilson_ci(kc, int(m.sum())),
                             'clean_evasion_rate': float((pred_clean[m] == BENIGN).mean())}
    return {
        'n': n,
        'clean_evasion_rate': float((pred_clean == BENIGN).mean()),
        'evasion_rate': k_ev / n, 'evasion_ci': wilson_ci(k_ev, n),
        'misclass_rate': k_mis / n, 'misclass_ci': wilson_ci(k_mis, n),
        'robust_accuracy': float((pred_adv == y_true).mean()),
        'robust_detection_rate': 1.0 - k_ev / n,
        'per_class': per_class,
    }


def summarise_seeds(values: Sequence[float]) -> str:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return 'n/a'
    if len(v) == 1:
        return f'{v[0]:.4f}'
    return f'{v.mean():.4f} ± {v.std(ddof=1):.4f} (n={len(v)})'
