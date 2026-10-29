"""
Train the multiclass baseline classifier (Phase 3).

Kept for backwards compatibility; the implementation lives in training.py and
train_classifier.py. Differences from the original version of this script:
  * evaluation is on the isolated test set (X_test_isolated.csv), not on an
    80/20 re-split of the SMOTE+Tomek output;
  * the checkpoint is selected on a validation split, not on test;
  * no benign cap / class weights / class-9 multiplier stacked on top of
    SMOTE+Tomek by default (use --config legacy_stacked to compare);
  * the preprocessor is saved once and reused by every other script.

    python src/risk_engine/new_train_baseline.py [--config ...] [--seeds ...]
"""

import sys

from train_classifier import main as _main


def train_network_classifier(argv=None):
    _main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    train_network_classifier()
