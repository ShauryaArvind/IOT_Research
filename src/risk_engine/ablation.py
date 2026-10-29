"""
Balancing ablation: does the imbalance-handling method change per-class
adversarial robustness?

For each imbalance config x seed it trains a baseline MLP (or reuses an
existing run), then records on the isolated test set:
  * clean macro-F1 / MCC / per-class F1 and PR-AUC,
  * per-class PGD-50 evasion (attack -> benign) at each eps, unconstrained and
    problem-space constrained,
  * the per-class synthetic fraction of the training data.

The hypothesis to test: classes whose training data is mostly synthetic
(1, 2, 9 have ~300x SMOTE oversampling: 77 / 90 / 49 isolated-test rows) are
the least robust. The output joins synthetic fraction against per-class
evasion, so the correlation can be read off directly.

    python src/risk_engine/ablation.py --seeds 42 43 44
    python src/risk_engine/ablation.py --configs legacy_balanced legacy_stacked --seeds 42

Every config except legacy_* needs X_train_raw.csv (data_split.py).
"""

import argparse

import pandas as pd
import torch

from attacks import FeatureConstraints, run_attack
from data import BENIGN, NUM_CLASSES, RESULTS_DIR, raw_train_available
from evaluate_robustness import select_attack_rows
from metrics import clean_metrics, robust_metrics
from training import IMBALANCE_CONFIGS, load_run, predict_proba, run_dir, splits_for_config, train_and_save


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', nargs='+', default=list(IMBALANCE_CONFIGS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--eps', type=float, nargs='+', default=[0.05, 0.1, 0.3])
    ap.add_argument('--per-class-cap', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--retrain', action='store_true', help='retrain even if a run already exists')
    a = ap.parse_args(argv)

    configs = a.configs
    if not raw_train_available():
        skipped = [c for c in configs if not c.startswith('legacy')]
        configs = [c for c in configs if c.startswith('legacy')]
        if skipped:
            print(f"[ablation] X_train_raw.csv not found: skipping {skipped}. Regenerate it with data_split.py.")

    rows = []
    for config in configs:
        for seed in a.seeds:
            rd = run_dir(config, 'none', seed)
            splits = splits_for_config(config, seed)
            if a.retrain or not (rd / 'classifier.pth').exists():
                model, _, _ = train_and_save(config, 'none', seed, splits=splits, epochs=a.epochs)
            else:
                model = load_run(rd)[0]
            cm = clean_metrics(splits.y_test, predict_proba(model, splits.X_test), n_boot=0)
            synth = splits.info['train_synthetic_fraction']

            idx = select_attack_rows(splits.y_test, a.per_class_cap, seed)
            Xa = torch.as_tensor(splits.X_test[idx])
            ya = torch.as_tensor(splits.y_test[idx])
            cons = {'none': None, 'attacker': FeatureConstraints(
                splits.prep, splits.prep.inverse(splits.X_val), mask='attacker', verbose=False)}
            per_eps = {}
            for mode, c in cons.items():
                for eps in a.eps:
                    X_adv = run_attack('pgd50_evasion', model, Xa, ya, eps, constraints=c)
                    with torch.no_grad():
                        pa = model(X_adv).argmax(1).numpy()
                        pc = model(Xa).argmax(1).numpy()
                    per_eps[(mode, eps)] = robust_metrics(ya.numpy(), pc, pa)['per_class']

            for cls in range(NUM_CLASSES):
                rec = {'config': config, 'seed': seed, 'class': cls,
                       'train_count': splits.info['train_counts'][cls],
                       'test_count': splits.info['test_counts'][cls],
                       'synthetic_fraction': synth[cls], 'synthetic_fraction_note': splits.info['synthetic_fraction_note'],
                       'clean_f1': cm['per_class']['f1'][cls], 'clean_recall': cm['per_class']['recall'][cls],
                       'clean_pr_auc': cm['per_class']['pr_auc'][cls],
                       'macro_f1': cm['macro_f1'], 'mcc': cm['mcc'], 'benign_fpr': cm['benign_fpr']}
                if cls != BENIGN:
                    for (mode, eps), pcs in per_eps.items():
                        if cls in pcs:
                            rec[f'evasion_{mode}_eps{eps}'] = pcs[cls]['evasion_rate']
                rows.append(rec)
            print(f"[ablation] {config} seed={seed}: macro-F1 {cm['macro_f1']:.4f}, MCC {cm['mcc']:.4f}")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / 'ablation_balancing.csv'
    df.to_csv(out, index=False)

    print("\nPer-config clean metrics (mean over seeds):")
    print(df.groupby('config')[['macro_f1', 'mcc', 'benign_fpr']].mean().round(4).to_string())
    ev_cols = [c for c in df.columns if c.startswith('evasion_')]
    if ev_cols:
        print("\nPer-class evasion, mean over seeds (rows: config, class):")
        print(df[df['class'] != BENIGN].groupby(['config', 'class'])[['synthetic_fraction'] + ev_cols]
              .mean().round(3).to_string())
        att = df[df['class'] != BENIGN]
        for col in ev_cols:
            r = att[['synthetic_fraction', col]].corr(method='spearman').iloc[0, 1]
            print(f"Spearman(synthetic_fraction, {col}) = {r:.3f}")
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
