"""
Phase 10 defence benchmark (legacy entry point).

The original version of this script re-split X_train_final_balanced.csv,
crafted FGSM/PGD against the bare baseline and only then applied the DAE
(a transfer attack), counted any misclassification as a "bypass", and
reported Zero-Trust interception on simulated attacker contexts. Those
numbers (phase10_results.csv) are not valid and are kept only for history.

This wrapper now runs evaluate_robustness.py, which evaluates the same
defence tiers on the isolated test set with adaptive attacks. For the
Zero-Trust layer, run zero_trust_eval.py.

    python src/risk_engine/evaluate_phase10.py [evaluate_robustness.py args]
"""

import sys

from evaluate_robustness import main as _main


def evaluate_phase10_suite(argv=None):
    sys.argv = [sys.argv[0]] + list(sys.argv[1:] if argv is None else argv)
    _main()


if __name__ == '__main__':
    evaluate_phase10_suite()
