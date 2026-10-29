"""
Train classifiers under the leakage-free protocol.

    # baseline on the committed balanced file (works without X_train_raw.csv)
    python src/risk_engine/train_classifier.py --config legacy_balanced --defense none --seeds 42 43 44

    # adversarially trained variants
    python src/risk_engine/train_classifier.py --defense pgd_at --eps 0.1 --seeds 42 43 44
    python src/risk_engine/train_classifier.py --defense trades --eps 0.1 --seeds 42 43 44

Imbalance configs: see training.IMBALANCE_CONFIGS. All except legacy_* need
X_train_raw.csv / y_train_raw.csv (regenerate with data_split.py).
Each run is written to src/risk_engine/models/runs/<config>__<defense>__seed<k>/.
"""

import argparse

from training import DEFAULT_CONFIG, DEFENSES, IMBALANCE_CONFIGS, splits_for_config, train_and_save


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_CONFIG, choices=list(IMBALANCE_CONFIGS))
    ap.add_argument('--defense', default='none', choices=DEFENSES)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--eps', type=float, default=0.1, help='training eps (standardised units) for AT/TRADES')
    ap.add_argument('--at-steps', type=int, default=7)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    a = ap.parse_args(argv)

    for seed in a.seeds:
        splits = splits_for_config(a.config, seed)
        train_and_save(a.config, a.defense, seed, splits=splits, eps=a.eps, at_steps=a.at_steps,
                       epochs=a.epochs, patience=a.patience, batch_size=a.batch_size, lr=a.lr)


if __name__ == '__main__':
    main()
