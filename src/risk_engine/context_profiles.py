"""
Context distributions for the Zero-Trust evaluation.

These are MODELLING ASSUMPTIONS, not measurements: the dataset has no device,
geo or identity signals. The Zero-Trust results depend directly on them, so
they live in one place and are reported with the results.

  benign    enrolled devices, mostly low geo-risk, mostly business hours,
            mostly verified. Not perfectly clean: some legitimate users travel,
            work late or use new devices, which drives the benign FPR.
  attacker  the legacy threat model (unregistered device, anomalous location).
            The legacy scripts drew every attacker context from this, so
            context-only rules (device_trust < 0.3, geo_risk > 0.8, off-hours
            AND device_trust < 0.5) denied attacks whatever the ML score was.
  spoofed   with probability p an attacker's context is drawn from the benign
            profile instead (compromised enrolled device / stolen session /
            residential proxy). Sweeping p from 0 to 1 shows how much of the
            interception depends on the attacker not controlling context.
"""

from typing import Dict, List

import numpy as np


def benign_context(rng: np.random.Generator) -> Dict:
    business = rng.random() < 0.85
    return {
        'device_trust': float(rng.beta(8, 2)),                  # mean 0.80
        'geo_risk': float(rng.beta(2, 8)),                      # mean 0.20
        'time_of_day': int(rng.integers(8, 19) if business else rng.integers(0, 24)),
        'identity_verified': bool(rng.random() < 0.95),
        'resource_sensitivity': float(rng.uniform(0.0, 1.0)),
    }


def attacker_context(rng: np.random.Generator) -> Dict:
    return {
        'device_trust': float(rng.uniform(0.0, 0.4)),
        'geo_risk': float(rng.uniform(0.5, 1.0)),
        'time_of_day': int(rng.integers(0, 24)),
        'identity_verified': bool(rng.random() > 0.6),
        'resource_sensitivity': float(rng.uniform(0.3, 1.0)),
    }


def sample_benign(n: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    return [benign_context(rng) for _ in range(n)]


def sample_attacker(n: int, spoof_prob: float, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    return [benign_context(rng) if rng.random() < spoof_prob else attacker_context(rng) for _ in range(n)]


PROFILE_DESCRIPTION = {
    'benign': 'device_trust~Beta(8,2), geo_risk~Beta(2,8), 85% business hours, 95% verified',
    'attacker': 'device_trust~U(0,0.4), geo_risk~U(0.5,1), uniform hour, 40% verified',
    'spoofed': 'benign profile with probability p',
}
