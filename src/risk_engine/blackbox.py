"""
Decision-based black-box attack (HopSkipJump, via ART) on the baseline classifier.

The attacker only sees predicted labels. Runs on attack rows of the isolated
test set with the saved preprocessor, and reports evasion (attack ->
benign) separately from misclassification. A lower black-box success rate
mostly reflects the small query budget (max_iter / max_eval). It does not
show the model is robust; report the budget alongside the number.

Changes from the original script: no re-split of the SMOTE output, no
hardcoded white-box reference numbers (run evaluate_robustness.py for
those), and no Zero-Trust step on simulated attacker contexts (see
zero_trust_eval.py).

    python src/risk_engine/blackbox.py --config legacy_balanced --seed 42 --per-class 50
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

from data import BENIGN, NUM_CLASSES
from metrics import robust_metrics
from training import DEFAULT_CONFIG, load_run, run_dir, splits_for_config


def pprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class Float32ModelWrapper(nn.Module):
    """Casts ART's float64 inputs to float32."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.float())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--defense', default='none')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--per-class', type=int, default=50, help='attack rows per attack class')
    ap.add_argument('--max-iter', type=int, default=15)
    ap.add_argument('--max-eval', type=int, default=100)
    ap.add_argument('--targeted-benign', action='store_true',
                    help='targeted HSJ toward benign (needs benign starting points)')
    a = ap.parse_args(argv)

    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import PyTorchClassifier

    splits = splits_for_config(a.config, a.seed, skip_resampling=True)
    model, _, _ = load_run(run_dir(a.config, a.defense, a.seed))
    clf = PyTorchClassifier(model=Float32ModelWrapper(model), loss=nn.CrossEntropyLoss(),
                            input_shape=(splits.input_dim,), nb_classes=NUM_CLASSES, clip_values=None)

    rng = np.random.default_rng(a.seed)
    idx = np.concatenate([rng.choice(ix, size=min(a.per_class, len(ix)), replace=False)
                          for c in range(1, NUM_CLASSES) if len(ix := np.where(splits.y_test == c)[0])])
    X, y = splits.X_test[idx], splits.y_test[idx]
    pred_clean = model.predict_batch(X)

    attack = HopSkipJump(classifier=clf, targeted=a.targeted_benign, norm=np.inf, max_iter=a.max_iter,
                         max_eval=a.max_eval, init_eval=10, verbose=False)
    pprint(f"HopSkipJump (L-inf, max_iter={a.max_iter}, max_eval={a.max_eval}, "
           f"targeted={'benign' if a.targeted_benign else 'no'}) on {len(y)} attack rows...")
    if a.targeted_benign:
        x_init = splits.X_test[np.where(splits.y_test == BENIGN)[0][:len(y)]]
        X_adv = attack.generate(x=X, y=np.eye(NUM_CLASSES)[np.zeros(len(y), int)], x_adv_init=x_init)
    else:
        X_adv = attack.generate(x=X)
    pred_adv = model.predict_batch(torch.FloatTensor(X_adv))
    rm = robust_metrics(y, pred_clean, pred_adv)
    linf = np.abs(X_adv - X).max(axis=1)

    pprint(f"  evasion (attack -> benign): {rm['evasion_rate']:.3f} CI {np.round(rm['evasion_ci'], 3)}")
    pprint(f"  misclassification:          {rm['misclass_rate']:.3f} CI {np.round(rm['misclass_ci'], 3)}")
    pprint(f"  L-inf perturbation (scaled units): median {np.median(linf):.3f}, mean {linf.mean():.3f}")
    pprint("  per-class evasion: " + ", ".join(f"c{c}={v['evasion_rate']:.2f}" for c, v in rm['per_class'].items()))
    return rm


if __name__ == "__main__":
    main()
