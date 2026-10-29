"""
Adversarial training (Phase 10).

Kept for backwards compatibility; delegates to train_classifier.py.
The default is now PGD-AT (Madry et al.). The original FGSM-AT
(--defense fgsm_at) is kept for comparison only: single-step AT is prone to
catastrophic overfitting, where robustness to FGSM rises while robustness to
multi-step PGD collapses. Robust checkpoints are selected on validation
clean macro-F1 averaged with validation PGD-10 detection, which exposes that
failure mode.

    python src/risk_engine/adversarial_training.py --defense pgd_at --eps 0.1 --seeds 42 43 44
    python src/risk_engine/adversarial_training.py --defense trades --eps 0.1
"""

import sys

from train_classifier import main as _main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--defense' not in argv:
        argv += ['--defense', 'pgd_at']
    _main(argv)


if __name__ == "__main__":
    main()
