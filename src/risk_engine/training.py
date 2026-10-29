"""
Classifier training under the shared protocol (data.py).

Changes from the legacy training scripts:
  * Checkpoints are selected on the VALIDATION split, never on test.
  * One imbalance mechanism per run (named config below) instead of stacking
    SMOTE+Tomek + benign cap + class weights + a hand-tuned 0.5 multiplier on
    class 9. The stacked legacy recipe is kept as 'legacy_stacked' (without the
    multiplier, which was tuned on the test split) so it can be compared.
  * Adversarial training: PGD-AT (Madry) and TRADES, with FGSM-AT kept only as a
    comparison point (it is prone to catastrophic overfitting). For robust
    runs the selection metric includes PGD robustness on a validation
    subset, which also catches catastrophic overfitting.
  * Each run is saved to models/runs/<name>/ with its config, data info and a
    pointer to the single saved preprocessor.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from attacks import pgd
from data import BENIGN, NUM_CLASSES, RUNS_DIR, Preprocessor, Splits, load_splits, scaler_path
from metrics import clean_metrics, macro_f1
from Network_classifier import NetworkRiskClassifier

IMBALANCE_CONFIGS: Dict[str, Dict] = {
    'none':             dict(resampling='none',            loss='ce',          sampler='shuffle'),
    'class_weight':     dict(resampling='none',            loss='weighted_ce', sampler='shuffle'),
    'focal':            dict(resampling='none',            loss='focal',       sampler='shuffle'),
    'balanced_sampler': dict(resampling='none',            loss='ce',          sampler='balanced'),
    'smote':            dict(resampling='smote',           loss='ce',          sampler='shuffle'),
    'smote_tomek':      dict(resampling='smote_tomek',     loss='ce',          sampler='shuffle'),
    'cgan':             dict(resampling='cgan',            loss='ce',          sampler='shuffle'),
    # Uses the committed X_train_final_balanced.csv; the only config that runs without X_train_raw.csv.
    'legacy_balanced':  dict(resampling='legacy_balanced', loss='ce',          sampler='shuffle'),
    'legacy_stacked':   dict(resampling='legacy_balanced', loss='weighted_ce', sampler='shuffle', benign_cap=40_000),
}
DEFAULT_CONFIG = 'legacy_balanced'
DEFENSES = ('none', 'fgsm_at', 'pgd_at', 'trades')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_name(config: str, defense: str, seed: int) -> str:
    return f'{config}__{defense}__seed{seed}'


def run_dir(config: str, defense: str, seed: int) -> Path:
    return RUNS_DIR / run_name(config, defense, seed)


def splits_for_config(config: str, seed: int, skip_resampling: bool = False, **kw) -> Splits:
    cfg = IMBALANCE_CONFIGS[config]
    return load_splits(resampling=cfg['resampling'], seed=seed, benign_cap=cfg.get('benign_cap'),
                       skip_resampling=skip_resampling, **kw)


# ---------------------------------------------------------------------------
# Losses / samplers
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits, y):
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, y[:, None]).squeeze(1)
        loss = -((1 - logp_t.exp()) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight[y]
        return loss.mean()


def inverse_frequency_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.maximum(np.bincount(y, minlength=NUM_CLASSES), 1).astype(np.float64)
    w = 1.0 / counts
    return torch.tensor(w * NUM_CLASSES / w.sum(), dtype=torch.float32)


def make_criterion(kind: str, y_train: np.ndarray) -> nn.Module:
    if kind == 'ce':
        return nn.CrossEntropyLoss()
    if kind == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=inverse_frequency_weights(y_train))
    if kind == 'focal':
        return FocalLoss(gamma=2.0)
    raise ValueError(kind)


def make_loader(X, y, batch_size: int, sampler: str, seed: int) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X), torch.as_tensor(y))
    g = torch.Generator().manual_seed(seed)
    if sampler == 'balanced':
        w = (1.0 / np.maximum(np.bincount(y, minlength=NUM_CLASSES), 1))[y]
        s = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(y),
                                  replacement=True, generator=g)
        return DataLoader(ds, batch_size=batch_size, sampler=s)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_proba(model: nn.Module, X, batch_size: int = 16384) -> np.ndarray:
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    out = [torch.softmax(model(X[i:i + batch_size]), dim=1) for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


def val_robust_detection(model, X_val, y_val, eps, n=2000, seed=0) -> float:
    """Fraction of validation attack rows still flagged as an attack after PGD-10
    evasion. Used for robust model selection."""
    idx = np.where(y_val != BENIGN)[0]
    idx = np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False)
    X = torch.as_tensor(X_val[idx])
    y = torch.as_tensor(y_val[idx])
    X_adv = pgd(model, X, y, eps, steps=10, restarts=1, loss='benign_margin')
    with torch.no_grad():
        return float((model(X_adv).argmax(1) != BENIGN).float().mean())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _adv_batch(model, xb, yb, defense, eps, steps):
    if defense == 'fgsm_at':
        return pgd(model, xb, yb, eps, steps=1, alpha=eps, random_start=False, loss='ce')
    return pgd(model, xb, yb, eps, steps=steps, restarts=1, random_start=True, loss='ce')


def _trades_adv(model, xb, eps, steps):
    was = model.training
    model.eval()
    with torch.no_grad():
        p_clean = F.softmax(model(xb), dim=1)
    x = xb + 0.001 * torch.randn_like(xb)
    alpha = 2.5 * eps / steps
    for _ in range(steps):
        x.requires_grad_(True)
        kl = F.kl_div(F.log_softmax(model(x), dim=1), p_clean, reduction='sum')
        g, = torch.autograd.grad(kl, x)
        with torch.no_grad():
            x = xb + torch.clamp(x + alpha * g.sign() - xb, -eps, eps)
    model.train(was)
    return x.detach()


def train_classifier(splits: Splits, loss: str = 'ce', sampler: str = 'shuffle', defense: str = 'none',
                     eps: float = 0.1, at_steps: int = 7, trades_beta: float = 6.0,
                     epochs: int = 40, patience: int = 6, lr: float = 1e-3, batch_size: int = 512,
                     seed: int = 42, verbose: bool = True):
    """Train and return (best_model, history). Best = highest validation selection score."""
    assert defense in DEFENSES, defense
    set_seed(seed)
    model = NetworkRiskClassifier(input_dim=splits.input_dim, num_classes=NUM_CLASSES)
    criterion = make_criterion(loss, splits.y_train)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = make_loader(splits.X_train, splits.y_train, batch_size, sampler, seed)

    best_score, best_state, best_epoch, bad = -np.inf, None, -1, 0
    history = []
    for epoch in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for xb, yb in loader:
            if defense == 'none':
                out = criterion(model(xb), yb)
            elif defense == 'trades':
                x_adv = _trades_adv(model, xb, eps, at_steps)
                logits = model(xb)
                kl = F.kl_div(F.log_softmax(model(x_adv), dim=1), F.softmax(logits, dim=1),
                              reduction='batchmean')
                out = criterion(logits, yb) + trades_beta * kl
            else:
                x_adv = _adv_batch(model, xb, yb, defense, eps, at_steps)
                model.train()
                out = 0.5 * criterion(model(xb), yb) + 0.5 * criterion(model(x_adv), yb)
            opt.zero_grad()
            out.backward()
            opt.step()
            tot += out.item()

        val_pred = predict_proba(model, splits.X_val).argmax(1)
        f1 = macro_f1(splits.y_val, val_pred)
        rec = {'epoch': epoch + 1, 'loss': tot / len(loader), 'val_macro_f1': f1, 'sec': time.time() - t0}
        score = f1
        if defense != 'none':
            rob = val_robust_detection(model, splits.X_val, splits.y_val, eps, seed=seed)
            rec['val_pgd10_detection'] = rob
            score = 0.5 * (f1 + rob)
        rec['selection_score'] = score
        history.append(rec)
        if verbose:
            extra = f" | val PGD-10 detection {rec['val_pgd10_detection']:.4f}" if defense != 'none' else ''
            print(f"  epoch {epoch+1:2d}/{epochs} | loss {rec['loss']:.4f} | val macro-F1 {f1:.4f}{extra}"
                  f" | {rec['sec']:.1f}s")
        if score > best_score:
            best_score, best_epoch, bad = score, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop (no val improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {'history': history, 'best_epoch': best_epoch, 'best_val_score': best_score}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_run(path: Path, model: nn.Module, splits: Splits, config: Dict, train_info: Dict,
             test_report: Optional[Dict] = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / 'classifier.pth')
    meta = {
        'config': config, 'data': splits.info, 'train': train_info,
        'input_dim': splits.input_dim, 'num_classes': NUM_CLASSES,
        'feature_names': splits.prep.feature_names,
        'scaler_path': str(scaler_path(splits.prep.source, splits.prep.seed)),
    }
    if test_report is not None:
        meta['test'] = test_report
    with open(path / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=_json_default)


def load_run(path: Path):
    path = Path(path)
    with open(path / 'meta.json') as f:
        meta = json.load(f)
    model = NetworkRiskClassifier(input_dim=meta['input_dim'], num_classes=meta['num_classes'])
    model.load_state_dict(torch.load(path / 'classifier.pth', map_location='cpu'))
    model.eval()
    prep = Preprocessor.load(meta['scaler_path'])
    return model, meta, prep


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def train_and_save(config: str = DEFAULT_CONFIG, defense: str = 'none', seed: int = 42,
                   splits: Optional[Splits] = None, **train_kw):
    cfg = IMBALANCE_CONFIGS[config]
    if splits is None:
        splits = splits_for_config(config, seed)
    print(f"\n=== train {run_name(config, defense, seed)} ===")
    model, info = train_classifier(splits, loss=cfg['loss'], sampler=cfg['sampler'], defense=defense,
                                   seed=seed, **train_kw)
    report = clean_metrics(splits.y_test, predict_proba(model, splits.X_test), seed=seed)
    print(f"  isolated test: macro-F1 {report['macro_f1']:.4f} "
          f"[{report['macro_f1_ci'][0]:.4f}, {report['macro_f1_ci'][1]:.4f}] | "
          f"MCC {report['mcc']:.4f} | macro PR-AUC {report['macro_pr_auc']:.4f} | "
          f"benign FPR {report['benign_fpr']:.4f}")
    print(f"  per-class F1: {np.round(report['per_class']['f1'], 3).tolist()}")
    out = run_dir(config, defense, seed)
    save_run(out, model, splits, {'imbalance_config': config, **cfg, 'defense': defense, 'seed': seed,
                                  **train_kw}, info, report)
    print(f"  saved -> {out}")
    return model, splits, report
