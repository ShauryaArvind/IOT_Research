"""
FGSM / PGD attack on the baseline (legacy entry point, Phase 5).

Superseded by evaluate_robustness.py. This wrapper runs it at eps=0.15 (the
legacy setting) without randomized smoothing: untargeted FGSM / PGD-10 for
comparison with the old numbers, plus FGSM and PGD-50 evasion attacks, on
every trained system that exists for the given seeds.

    python src/risk_engine/adversial_attack.py [--config ...] [--seeds ...]
"""

import sys

from evaluate_robustness import main as _main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    defaults = {'--eps': ['0.15'], '--smoothing-sigma': ['0'], '--tag': ['phase5']}
    for k, v in defaults.items():
        if k not in argv:
            argv += [k, *v]
    sys.argv = [sys.argv[0]] + argv
    _main()


if __name__ == "__main__":
    main()
