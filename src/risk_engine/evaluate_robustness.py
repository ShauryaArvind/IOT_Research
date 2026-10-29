"""
Robustness evaluation under the leakage-free protocol.

Replaces the legacy phase-10 benchmark / sweep / adversial_attack scripts.
What is different:
  * Everything is measured on the isolated test set, with the saved
    preprocessor (no refitting).
  * Clean metrics (macro-F1, MCC, benign FPR) are printed next to the robust
    numbers for every system.
  * Primary metric is EVASION (attack -> predicted benign). Misclassification
    (the legacy "bypass") is reported alongside it.
  * Attacks: FGSM and PGD-50 with 5 random restarts and a CW-style margin loss
    toward benign. Legacy untargeted FGSM / PGD-10 are kept for comparison.
  * Defences are attacked adaptively: gradients flow through the DAE, the
    detector's reconstruction error is part of the attack objective, and
    randomized smoothing is attacked with EOT. The legacy "attack the baseline,
    then denoise" setting is kept as `dae_denoiser_transfer` so the gap is visible.
  * Black-box transfer: perturbations crafted on a surrogate model trained with
    a different seed.
  * Constraint modes: none (legacy), domain (bounds + integers + dependencies on
    all features), attacker (domain + attacker-controllable mask + monotonicity).
  * Several seeds; per-class evasion with Wilson CIs; per-feature perturbation
    attribution for successful constrained evasions.

Prerequisites (per seed): a baseline run (train_classifier.py --defense none);
optionally robust runs (--defense pgd_at / trades / fgsm_at), a DAE
(denoising_autoencoder.py), and baseline runs for other seeds (surrogates).

Usage:
    python src/risk_engine/evaluate_robustness.py --config legacy_balanced --seeds 42 43 44
"""

import argparse
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch

from attacks import ATTACKS, FeatureConstraints, run_attack
from data import BENIGN, NUM_CLASSES, RESULTS_DIR
from defenses import DAEPipeline, ReconstructionDetector, SmoothedClassifier
from metrics import clean_metrics, robust_metrics, summarise_seeds
from training import DEFAULT_CONFIG, load_run, predict_proba, run_dir, splits_for_config

DEFAULT_EPS = [0.05, 0.1, 0.15, 0.3]
DEFAULT_ATTACKS = ['fgsm_untargeted', 'pgd10_untargeted', 'fgsm_evasion', 'pgd50_evasion']
CONSTRAINT_MODES = ['none', 'domain', 'attacker']


@dataclass
class System:
    name: str
    attack_model: torch.nn.Module          # model the attacker differentiates
    predict: Callable                      # X (tensor) -> np.ndarray of class predictions
    flag: Optional[Callable] = None        # X -> bool np.ndarray (detector rejects)
    eot: int = 1
    extra_loss: Optional[Callable] = None
    row_cap: Optional[int] = None          # per-class cap for expensive systems
    clean_proba: Optional[Callable] = None


def _argmax_predict(model):
    def f(X):
        with torch.no_grad():
            model.eval()
            return model(X).argmax(1).numpy()
    return f


def build_systems(config: str, seed: int, surrogate_seed: Optional[int], smoothing_sigma: float,
                  smoothing_cap: int, detector_pct: str) -> List[System]:
    base, _, _ = load_run(run_dir(config, 'none', seed))
    systems = [System('baseline', base, _argmax_predict(base),
                      clean_proba=lambda X: predict_proba(base, X))]

    robust = {}
    for d in ('pgd_at', 'trades', 'fgsm_at'):
        p = run_dir(config, d, seed)
        if (p / 'classifier.pth').exists():
            m, _, _ = load_run(p)
            robust[d] = m
            systems.append(System(d, m, _argmax_predict(m), clean_proba=lambda X, m=m: predict_proba(m, X)))

    dae_dir = run_dir(config, 'dae', seed)
    if (dae_dir / 'dae.pth').exists():
        from denoising_autoencoder import load_dae
        dae, dmeta = load_dae(config, seed)
        pipe = DAEPipeline(dae, base).eval()
        systems.append(System('dae_denoiser', pipe, _argmax_predict(pipe),
                              clean_proba=lambda X: predict_proba(pipe, X)))
        # legacy (non-adaptive) setting: perturb against the bare baseline, then denoise
        systems.append(System('dae_denoiser_transfer', base, _argmax_predict(pipe),
                              clean_proba=lambda X: predict_proba(pipe, X)))
        det = ReconstructionDetector(dae, dmeta[f'threshold_{detector_pct}'])
        systems.append(System(f'dae_detector_{detector_pct}', base, _argmax_predict(base), flag=det.flag,
                              extra_loss=det.adaptive_penalty(),
                              clean_proba=lambda X: predict_proba(base, X)))
        for d, m in robust.items():
            p2 = DAEPipeline(dae, m).eval()
            systems.append(System(f'dae+{d}', p2, _argmax_predict(p2),
                                  clean_proba=lambda X, p2=p2: predict_proba(p2, X)))

    if smoothing_sigma > 0:
        sm = SmoothedClassifier(base, sigma=smoothing_sigma, n_train_draws=1)
        systems.append(System(f'smoothing_s{smoothing_sigma}', sm, lambda X: sm.predict(X, n=100),
                              eot=8, row_cap=smoothing_cap,
                              clean_proba=None))

    if surrogate_seed is not None:
        sp = run_dir(config, 'none', surrogate_seed)
        if (sp / 'classifier.pth').exists():
            sur, _, _ = load_run(sp)
            systems.append(System(f'transfer_from_seed{surrogate_seed}', sur, _argmax_predict(base),
                                  clean_proba=lambda X: predict_proba(base, X)))
        else:
            print(f"[eval] no surrogate run for seed {surrogate_seed}; skipping transfer attack")
    return systems


def select_attack_rows(y_test: np.ndarray, per_class_cap: Optional[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(NUM_CLASSES):
        if c == BENIGN:
            continue
        ix = np.where(y_test == c)[0]
        if per_class_cap is not None and len(ix) > per_class_cap:
            ix = rng.choice(ix, size=per_class_cap, replace=False)
        idx.append(np.sort(ix))
    return np.concatenate(idx)


def evaluate_seed(config: str, seed: int, eps_list, attacks, modes, per_class_cap, surrogate_seed,
                  smoothing_sigma, smoothing_cap, detector_pct):
    splits = splits_for_config(config, seed, skip_resampling=True, verbose=True)
    prep = splits.prep
    X_real_raw = prep.inverse(splits.X_val)   # for validating feature dependencies
    constraints = {
        'none': None,
        'domain': FeatureConstraints(prep, X_real_raw, mask='all', monotone=False, verbose=False),
        'attacker': FeatureConstraints(prep, X_real_raw, mask='attacker', verbose=True),
    }

    X_test = torch.as_tensor(splits.X_test)
    y_test = splits.y_test
    systems = build_systems(config, seed, surrogate_seed, smoothing_sigma, smoothing_cap, detector_pct)
    rows_all = select_attack_rows(y_test, per_class_cap, seed)

    records, clean_records, attribution = [], [], []
    for sys_ in systems:
        print(f"\n[eval] seed={seed} system={sys_.name}")
        # ---- clean metrics on the full isolated test set ----
        if sys_.clean_proba is not None:
            proba = sys_.clean_proba(X_test)
            cm = clean_metrics(y_test, proba, n_boot=500, seed=seed)
            pred_full = proba.argmax(1)
        else:
            pred_full = np.concatenate([sys_.predict(X_test[i:i + 4096]) for i in range(0, len(X_test), 4096)])
            onehot = np.eye(NUM_CLASSES)[pred_full]
            cm = clean_metrics(y_test, onehot, n_boot=500, seed=seed)
        benign_fpr = cm['benign_fpr']
        if sys_.flag is not None:
            fl = sys_.flag(X_test)
            benign = y_test == BENIGN
            benign_fpr = float(((pred_full != BENIGN) | fl)[benign].mean())
        clean_records.append({'seed': seed, 'system': sys_.name, 'macro_f1': cm['macro_f1'],
                              'macro_f1_ci': cm['macro_f1_ci'], 'mcc': cm['mcc'],
                              'macro_pr_auc': cm['macro_pr_auc'], 'accuracy': cm['accuracy'],
                              'benign_fpr': benign_fpr, 'attack_detection_rate': cm['attack_detection_rate'],
                              'ece': cm['ece'], 'per_class_f1': cm['per_class']['f1']})
        print(f"  clean: macro-F1 {cm['macro_f1']:.4f} | MCC {cm['mcc']:.4f} | benign FPR {benign_fpr:.4f}")

        rows = rows_all
        if sys_.row_cap is not None:
            rows = select_attack_rows(y_test, sys_.row_cap, seed)
        Xa, ya = X_test[rows], torch.as_tensor(y_test[rows])
        pred_clean = sys_.predict(Xa)

        for mode in modes:
            for atk in attacks:
                for eps in eps_list:
                    X_adv = run_attack(atk, sys_.attack_model, Xa, ya, eps, constraints=constraints[mode],
                                       eot=sys_.eot, extra_loss=sys_.extra_loss)
                    pred_adv = sys_.predict(X_adv)
                    flagged = sys_.flag(X_adv) if sys_.flag is not None else None
                    rm = robust_metrics(ya.numpy(), pred_clean, pred_adv, flagged)
                    rec = {'seed': seed, 'system': sys_.name, 'attack': atk, 'constraints': mode, 'eps': eps,
                           'n': rm['n'], 'clean_macro_f1': cm['macro_f1'], 'clean_benign_fpr': benign_fpr,
                           'clean_evasion_rate': rm['clean_evasion_rate'],
                           'evasion_rate': rm['evasion_rate'], 'evasion_ci_lo': rm['evasion_ci'][0],
                           'evasion_ci_hi': rm['evasion_ci'][1], 'misclass_rate': rm['misclass_rate'],
                           'robust_accuracy': rm['robust_accuracy']}
                    for c, pc in rm['per_class'].items():
                        rec[f'evasion_c{c}'] = pc['evasion_rate']
                    records.append(rec)
                    print(f"  {mode:<8} {atk:<17} eps={eps:<5} evasion {rm['evasion_rate']:.3f} "
                          f"[{rm['evasion_ci'][0]:.3f},{rm['evasion_ci'][1]:.3f}] | misclass {rm['misclass_rate']:.3f}")

                    if sys_.name == 'baseline' and atk == 'pgd50_evasion' and mode == 'attacker':
                        ok = (pred_adv == BENIGN) & (pred_clean != BENIGN)
                        if ok.any():
                            d = np.abs(prep.inverse(X_adv[ok].numpy()) - prep.inverse(Xa[ok].numpy()))
                            attribution.append({'seed': seed, 'eps': eps, 'n_success': int(ok.sum()),
                                                **dict(zip(prep.feature_names, (d / prep.scale).mean(0)))})
    return records, clean_records, attribution


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--eps', type=float, nargs='+', default=DEFAULT_EPS)
    ap.add_argument('--attacks', nargs='+', default=DEFAULT_ATTACKS, choices=list(ATTACKS))
    ap.add_argument('--constraints', nargs='+', default=CONSTRAINT_MODES, choices=CONSTRAINT_MODES)
    ap.add_argument('--per-class-cap', type=int, default=1000,
                    help='max attack rows per class (all rows of small classes are always used)')
    ap.add_argument('--surrogate-seed', type=int, default=None,
                    help='seed of a baseline run used as black-box surrogate (default: next seed in --seeds)')
    ap.add_argument('--smoothing-sigma', type=float, default=0.25, help='0 disables randomized smoothing')
    ap.add_argument('--smoothing-cap', type=int, default=200)
    ap.add_argument('--detector-pct', default='p99', choices=['p95', 'p99'])
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    records, clean, attrib = [], [], []
    for i, s in enumerate(a.seeds):
        sur = a.surrogate_seed
        if sur is None and len(a.seeds) > 1:
            sur = a.seeds[(i + 1) % len(a.seeds)]
        r, c, at = evaluate_seed(a.config, s, a.eps, a.attacks, a.constraints, a.per_class_cap, sur,
                                 a.smoothing_sigma, a.smoothing_cap, a.detector_pct)
        records += r
        clean += c
        attrib += at

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f'{a.config}{("_" + a.tag) if a.tag else ""}'
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / f'robustness_{tag}.csv', index=False)
    pd.DataFrame(clean).to_csv(RESULTS_DIR / f'clean_{tag}.csv', index=False)
    if attrib:
        pd.DataFrame(attrib).to_csv(RESULTS_DIR / f'attribution_{tag}.csv', index=False)

    print("\n" + "=" * 100)
    print("SUMMARY (mean ± std over seeds). Clean metrics on the full isolated test set;")
    print("evasion = attack rows predicted benign (or not rejected by a detector).")
    print("=" * 100)
    cdf = pd.DataFrame(clean)
    for name, g in cdf.groupby('system', sort=False):
        print(f"\n{name}: clean macro-F1 {summarise_seeds(g['macro_f1'])} | MCC {summarise_seeds(g['mcc'])} "
              f"| benign FPR {summarise_seeds(g['benign_fpr'])}")
        sub = df[(df.system == name) & (df.attack == 'pgd50_evasion')]
        for (mode, eps), h in sub.groupby(['constraints', 'eps']):
            print(f"    PGD-50 evasion  constraints={mode:<8} eps={eps:<5} {summarise_seeds(h['evasion_rate'])}"
                  f"   (misclass {summarise_seeds(h['misclass_rate'])})")
    print(f"\nSaved: {RESULTS_DIR}/robustness_{tag}.csv, clean_{tag}.csv"
          + (f", attribution_{tag}.csv" if attrib else ''))


if __name__ == '__main__':
    main()
