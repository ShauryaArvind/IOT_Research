"""
Leave-one-attack-out (open-set) evaluation.

For each attack class k, train a baseline MLP with class k removed from the
training and validation data, then measure on the isolated test set how
often unseen class-k flows are still flagged as an attack (predicted
anything but benign). This is the realistic question for an IDS facing a
new attack family. The script also reports the benign FPR of each
leave-one-out model.

    python src/risk_engine/open_set_eval.py --config legacy_balanced --seed 42 --epochs 15
"""

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from data import BENIGN, NUM_CLASSES, RESULTS_DIR
from metrics import wilson_ci
from training import DEFAULT_CONFIG, IMBALANCE_CONFIGS, predict_proba, splits_for_config, train_classifier


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--classes', type=int, nargs='+', default=list(range(1, NUM_CLASSES)))
    a = ap.parse_args(argv)

    cfg = IMBALANCE_CONFIGS[a.config]
    base = splits_for_config(a.config, a.seed)
    rows = []
    for k in a.classes:
        tr, va = base.y_train != k, base.y_val != k
        s = replace(base, X_train=base.X_train[tr], y_train=base.y_train[tr],
                    X_val=base.X_val[va], y_val=base.y_val[va], train_is_synthetic=base.train_is_synthetic[tr])
        print(f"\n[open-set] holding out class {k}")
        model, _ = train_classifier(s, loss=cfg['loss'], sampler=cfg['sampler'], seed=a.seed,
                                    epochs=a.epochs, verbose=False)
        pred = predict_proba(model, base.X_test).argmax(1)
        m = base.y_test == k
        flagged = int((pred[m] != BENIGN).sum())
        ci = wilson_ci(flagged, int(m.sum()))
        fpr = float((pred[base.y_test == BENIGN] != BENIGN).mean())
        print(f"  unseen class {k}: {flagged}/{m.sum()} flagged as attack ({flagged/m.sum():.3f}, "
              f"CI [{ci[0]:.3f}, {ci[1]:.3f}]) | benign FPR {fpr:.4f}")
        print(f"  predicted as: {np.bincount(pred[m], minlength=NUM_CLASSES).tolist()}")
        rows.append({'held_out_class': k, 'n_test': int(m.sum()), 'unseen_detection_rate': flagged / m.sum(),
                     'ci_lo': ci[0], 'ci_hi': ci[1], 'benign_fpr': fpr,
                     'pred_distribution': np.bincount(pred[m], minlength=NUM_CLASSES).tolist()})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f'open_set_{a.config}_seed{a.seed}.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
