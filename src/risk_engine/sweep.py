"""
Epsilon sweep (legacy entry point).

Runs evaluate_robustness.py over a dense eps grid with FGSM (untargeted and
evasion) only, which is the legacy sweep setting, on the isolated test set.
The old "ML + Zero-Trust bypass ~ 0%" curve came from simulated attacker
contexts that trigger context-only DENY rules regardless of the ML score;
see zero_trust_eval.py for the spoofing-aware evaluation instead.

    python src/risk_engine/sweep.py [--config ...] [--seeds ...]
"""

import sys

from evaluate_robustness import main as _main

SWEEP_EPS = ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3']


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--eps' not in argv:
        argv += ['--eps', *SWEEP_EPS]
    if '--attacks' not in argv:
        argv += ['--attacks', 'fgsm_untargeted', 'fgsm_evasion']
    if '--tag' not in argv:
        argv += ['--tag', 'sweep']
    if '--smoothing-sigma' not in argv:
        argv += ['--smoothing-sigma', '0']
    sys.argv = [sys.argv[0]] + argv
    _main()


if __name__ == "__main__":
    main()
