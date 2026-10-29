"""
Tree-based baselines on the same leakage-free splits as the MLP.

Models: RandomForest and HistGradientBoosting (always available), plus
XGBoost / LightGBM when those packages are installed. Trees are the standard
strong baseline for CIC-style flow features; without them, the MLP numbers
have no reference point.

Also reports:
  * transfer robustness: PGD-50 evasion examples crafted on the MLP baseline
    (white-box) and replayed against each tree (black-box transfer), with and
    without problem-space constraints;
  * a probability-averaging ensemble of the MLP and the best tree, on clean data
    and on the transferred examples.

    python src/risk_engine/tree_baselines.py --config legacy_balanced --seed 42
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from attacks import FeatureConstraints, run_attack
from data import NUM_CLASSES, RESULTS_DIR
from evaluate_robustness import select_attack_rows
from metrics import clean_metrics, robust_metrics
from training import DEFAULT_CONFIG, IMBALANCE_CONFIGS, load_run, predict_proba, run_dir, splits_for_config


class FullProba:
    """Wrap a fitted sklearn-style model so predict_proba always has NUM_CLASSES columns."""

    def __init__(self, model):
        self.model = model
        self.cols = np.asarray(model.classes_, dtype=int)

    def predict_proba(self, X):
        p = np.zeros((len(X), NUM_CLASSES))
        p[:, self.cols] = self.model.predict_proba(np.asarray(X))
        return p


def make_models(seed, weighted):
    cw = 'balanced' if weighted else None
    models = {
        'random_forest': RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed,
                                                class_weight=cw, min_samples_leaf=1),
        'hist_gb': HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True,
                                                  validation_fraction=0.1, random_state=seed, class_weight=cw),
    }
    try:
        from xgboost import XGBClassifier
        models['xgboost'] = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, tree_method='hist',
                                          random_state=seed, n_jobs=-1, eval_metric='mlogloss')
    except ImportError:
        print("[trees] xgboost not installed; skipping")
    try:
        from lightgbm import LGBMClassifier
        models['lightgbm'] = LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05,
                                            random_state=seed, class_weight=cw, verbose=-1)
    except ImportError:
        print("[trees] lightgbm not installed; skipping")
    return models


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eps', type=float, nargs='+', default=[0.05, 0.1, 0.3])
    ap.add_argument('--per-class-cap', type=int, default=1000)
    a = ap.parse_args(argv)

    splits = splits_for_config(a.config, a.seed)
    weighted = IMBALANCE_CONFIGS[a.config]['loss'] == 'weighted_ce'
    X_tr, y_tr = splits.X_train, splits.y_train

    mlp_dir = run_dir(a.config, 'none', a.seed)
    mlp = load_run(mlp_dir)[0] if (mlp_dir / 'classifier.pth').exists() else None
    if mlp is None:
        print(f"[trees] no MLP baseline at {mlp_dir}; transfer/ensemble evaluation skipped")

    clean_rows, robust_rows = [], []
    fitted = {}
    for name, m in make_models(a.seed, weighted).items():
        print(f"\n[trees] fitting {name} on {len(y_tr)} rows...")
        m.fit(X_tr, y_tr)
        fm = FullProba(m)
        fitted[name] = fm
        cm = clean_metrics(splits.y_test, fm.predict_proba(splits.X_test), n_boot=500, seed=a.seed)
        val_f1 = clean_metrics(splits.y_val, fm.predict_proba(splits.X_val), n_boot=0)['macro_f1']
        print(f"  val macro-F1 {val_f1:.4f} | test macro-F1 {cm['macro_f1']:.4f} {np.round(cm['macro_f1_ci'], 4)} "
              f"| MCC {cm['mcc']:.4f} | benign FPR {cm['benign_fpr']:.4f}")
        print(f"  per-class F1: {np.round(cm['per_class']['f1'], 3).tolist()}")
        clean_rows.append({'model': name, 'val_macro_f1': val_f1, 'macro_f1': cm['macro_f1'],
                           'macro_f1_ci': cm['macro_f1_ci'], 'mcc': cm['mcc'], 'macro_pr_auc': cm['macro_pr_auc'],
                           'benign_fpr': cm['benign_fpr'], 'per_class_f1': cm['per_class']['f1']})

    if mlp is not None:
        cm = clean_metrics(splits.y_test, predict_proba(mlp, splits.X_test), n_boot=500, seed=a.seed)
        clean_rows.append({'model': 'mlp', 'macro_f1': cm['macro_f1'], 'macro_f1_ci': cm['macro_f1_ci'],
                           'mcc': cm['mcc'], 'macro_pr_auc': cm['macro_pr_auc'], 'benign_fpr': cm['benign_fpr'],
                           'per_class_f1': cm['per_class']['f1']})
        best = max((r for r in clean_rows if r['model'] in fitted), key=lambda r: r['val_macro_f1'])['model']
        tree = fitted[best]

        def ens_proba(X):
            return 0.5 * predict_proba(mlp, X) + 0.5 * tree.predict_proba(X)

        cm = clean_metrics(splits.y_test, ens_proba(splits.X_test), n_boot=500, seed=a.seed)
        clean_rows.append({'model': f'ensemble_mlp+{best}', 'macro_f1': cm['macro_f1'],
                           'macro_f1_ci': cm['macro_f1_ci'], 'mcc': cm['mcc'], 'macro_pr_auc': cm['macro_pr_auc'],
                           'benign_fpr': cm['benign_fpr'], 'per_class_f1': cm['per_class']['f1']})
        print(f"\n[trees] ensemble MLP+{best}: test macro-F1 {cm['macro_f1']:.4f} | benign FPR {cm['benign_fpr']:.4f}")

        rows = select_attack_rows(splits.y_test, a.per_class_cap, a.seed)
        Xa = torch.as_tensor(splits.X_test[rows])
        ya = torch.as_tensor(splits.y_test[rows])
        cons = {'none': None,
                'attacker': FeatureConstraints(splits.prep, splits.prep.inverse(splits.X_val), mask='attacker',
                                               verbose=False)}
        targets = {**{n: fm.predict_proba for n, fm in fitted.items()},
                   'mlp (white-box)': lambda X: predict_proba(mlp, X), f'ensemble_mlp+{best}': ens_proba}
        for mode, c in cons.items():
            for eps in a.eps:
                X_adv = run_attack('pgd50_evasion', mlp, Xa, ya, eps, constraints=c).numpy()
                for tname, pf in targets.items():
                    rm = robust_metrics(ya.numpy(), pf(Xa.numpy()).argmax(1), pf(X_adv).argmax(1))
                    robust_rows.append({'target': tname, 'constraints': mode, 'eps': eps,
                                        'evasion_rate': rm['evasion_rate'], 'evasion_ci': rm['evasion_ci'],
                                        'clean_evasion_rate': rm['clean_evasion_rate'],
                                        'misclass_rate': rm['misclass_rate']})
                    print(f"  transfer {mode:<8} eps={eps:<5} -> {tname:<26} evasion {rm['evasion_rate']:.3f} "
                          f"(clean {rm['clean_evasion_rate']:.3f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clean_rows).to_csv(RESULTS_DIR / f'trees_clean_{a.config}_seed{a.seed}.csv', index=False)
    if robust_rows:
        pd.DataFrame(robust_rows).to_csv(RESULTS_DIR / f'trees_transfer_{a.config}_seed{a.seed}.csv', index=False)
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
