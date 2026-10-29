"""
Zero-Trust layer evaluation with spoofable context and benign false positives.

Why the legacy "100% interception" is not a result: every attacker context
was drawn with device_trust ~ U(0, 0.4) and geo_risk ~ U(0.5, 1). Rules 1, 3, 4
and 6 use context only, so they DENY these flows whatever the ML score is,
and nobody measured what the same rules do to benign traffic.

This script:
  * scores REAL isolated-test flows with a trained system: benign rows clean,
    attack rows clean and after a PGD-50 evasion attack (optionally
    problem-space constrained);
  * gives benign rows contexts from the benign profile, and attack rows
    contexts that are spoofed (drawn from the benign profile) with probability p;
  * sweeps p in [0, 1] and reports attack interception (DENY rate on attack
    rows), benign FPR (DENY rate on benign rows), the ML-only engine for
    reference, and which rules made the decisions (context-only rules vs
    rules that use the ML score).

    python src/risk_engine/zero_trust_eval.py --config legacy_balanced --seed 42 --eps 0.1
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch

from attacks import FeatureConstraints, run_attack
from context_profiles import PROFILE_DESCRIPTION, sample_attacker, sample_benign
from data import BENIGN, RESULTS_DIR
from evaluate_robustness import select_attack_rows
from training import DEFAULT_CONFIG, load_run, predict_proba, run_dir, splits_for_config
from zero_trust_engine import ZeroTrustEngine

CONTEXT_ONLY_RULES = {'Critical Resource Microsegment', 'Device Trust Threshold', 'Geo-Risk Threshold',
                      'Off-Hours Device Anomaly'}


def deny_stats(engine, risk, contexts):
    decisions = engine.evaluate_batch(risk, contexts)
    deny = np.array([d.decision == 'DENY' for d in decisions])
    rules = Counter(d.rule_fired for d in decisions if d.decision == 'DENY')
    ctx_only = sum(v for k, v in rules.items() if k in CONTEXT_ONLY_RULES)
    return deny, rules, ctx_only


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--defense', default='none', help='which trained classifier provides the ML score')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eps', type=float, default=0.1)
    ap.add_argument('--constraints', default='attacker', choices=['none', 'attacker'])
    ap.add_argument('--per-class-cap', type=int, default=1000)
    ap.add_argument('--benign-rows', type=int, default=20000)
    ap.add_argument('--spoof-probs', type=float, nargs='+', default=[0, .1, .25, .5, .75, .9, 1.0])
    a = ap.parse_args(argv)

    splits = splits_for_config(a.config, a.seed, skip_resampling=True)
    model, _, prep = load_run(run_dir(a.config, a.defense, a.seed))
    rng = np.random.default_rng(a.seed)

    ben_idx = np.where(splits.y_test == BENIGN)[0]
    ben_idx = rng.choice(ben_idx, size=min(a.benign_rows, len(ben_idx)), replace=False)
    risk_benign = 1.0 - predict_proba(model, splits.X_test[ben_idx])[:, BENIGN]

    atk_idx = select_attack_rows(splits.y_test, a.per_class_cap, a.seed)
    Xa = torch.as_tensor(splits.X_test[atk_idx])
    ya = torch.as_tensor(splits.y_test[atk_idx])
    cons = FeatureConstraints(prep, prep.inverse(splits.X_val), mask='attacker', verbose=False) \
        if a.constraints == 'attacker' else None
    X_adv = run_attack('pgd50_evasion', model, Xa, ya, a.eps, constraints=cons)
    risk_attack = {
        'clean': 1.0 - predict_proba(model, Xa)[:, BENIGN],
        f'pgd50_eps{a.eps}_{a.constraints}': 1.0 - predict_proba(model, X_adv)[:, BENIGN],
    }

    full, ml_only = ZeroTrustEngine(), ZeroTrustEngine(enabled_factors=set())
    ben_ctx = sample_benign(len(ben_idx), a.seed)
    ben_deny, ben_rules, ben_ctx_only = deny_stats(full, risk_benign, ben_ctx)
    ml_ben_deny, _, _ = deny_stats(ml_only, risk_benign, ben_ctx)

    print(f"\nContext profiles (assumptions): {PROFILE_DESCRIPTION}")
    print(f"\nBenign rows: {len(ben_idx)}")
    print(f"  FPR full engine: {ben_deny.mean():.4f}  (ML-only engine: {ml_ben_deny.mean():.4f})")
    print(f"  benign DENY reasons: {dict(ben_rules)}")

    records = []
    for variant, risk in risk_attack.items():
        print(f"\nAttack rows: {len(atk_idx)}  [{variant}]")
        ml_deny, _, _ = deny_stats(ml_only, risk, [{}] * len(risk))
        print(f"  ML-only engine interception: {ml_deny.mean():.4f}")
        print(f"  {'spoof p':>8} | {'interception':>12} | {'by context-only rules':>22} | {'benign FPR':>10}")
        for p in a.spoof_probs:
            ctx = sample_attacker(len(risk), p, a.seed + 1)
            deny, rules, ctx_only = deny_stats(full, risk, ctx)
            frac_ctx = ctx_only / max(deny.sum(), 1)
            print(f"  {p:>8.2f} | {deny.mean():>12.4f} | {frac_ctx:>22.4f} | {ben_deny.mean():>10.4f}")
            records.append({'seed': a.seed, 'defense': a.defense, 'variant': variant, 'spoof_prob': p,
                            'attack_interception': deny.mean(), 'ml_only_interception': ml_deny.mean(),
                            'share_denied_by_context_only_rules': frac_ctx,
                            'benign_fpr': ben_deny.mean(), 'benign_fpr_ml_only': ml_ben_deny.mean(),
                            'deny_rules': dict(rules)})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f'zero_trust_{a.config}_{a.defense}_seed{a.seed}.csv'
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
