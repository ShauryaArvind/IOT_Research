"""
White-box attacks in standardised feature space, with optional problem-space
constraints.

Unconstrained attacks (the legacy setting) perturb all 76 standardised
features by up to eps, including flags, integer counts and derived features,
so the resulting vectors are usually not realisable flows. `FeatureConstraints`
narrows this to what a flow-level attacker can plausibly do:

  * mask         only attacker-controllable features move (forward-direction
                 packet sizes / counts / IATs, flow duration and flow IATs).
                 Backward-direction (server response) features, TCP flags,
                 header lengths and window sizes are left untouched.
  * monotone     counts, byte totals, max packet length and duration can only
                 grow (padding, delays and dummy packets add; they cannot
                 remove).
  * domain       every feature stays inside the [min, max] seen in real
                 training rows, and integer-valued features are rounded.
  * dependencies derived features (Flow Bytes/s, Packets/s, mean packet length,
                 average packet size, min <= mean <= max) are recomputed from
                 their parents. A relation is only enforced if it actually holds
                 on the real training rows, so a naming mismatch cannot inject
                 a wrong formula.

The eps budget applies to the controllable features. Recomputed derived
features follow from them and may move further than eps.

The projection is written in torch, so PGD differentiates through it: the
gradient reaches the controllable features through the derived features that
depend on them (e.g. Flow Bytes/s). Only rounding uses a straight-through
estimator.
"""

import re
import warnings
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import BENIGN, Preprocessor


# ---------------------------------------------------------------------------
# Feature-name matching (CICFlowMeter naming varies between releases)
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


ALIASES = {
    'duration': ['flowduration'],
    'fwd_pkts': ['totalfwdpackets', 'totalfwdpacket', 'totfwdpkts', 'fwdpackets'],
    'bwd_pkts': ['totalbackwardpackets', 'totalbwdpackets', 'totalbwdpacket', 'totbwdpkts', 'bwdpackets'],
    'fwd_bytes': ['totallengthoffwdpackets', 'totallengthoffwdpacket', 'totlenfwdpkts', 'fwdpacketslengthtotal'],
    'bwd_bytes': ['totallengthofbwdpackets', 'totallengthofbwdpacket', 'totlenbwdpkts', 'bwdpacketslengthtotal'],
    'flow_bytes_s': ['flowbytess', 'flowbytss'],
    'flow_pkts_s': ['flowpacketss', 'flowpktss'],
    'fwd_pkts_s': ['fwdpacketss', 'fwdpktss'],
    'bwd_pkts_s': ['bwdpacketss', 'bwdpktss'],
    'fwd_len_max': ['fwdpacketlengthmax', 'fwdpktlenmax'],
    'fwd_len_min': ['fwdpacketlengthmin', 'fwdpktlenmin'],
    'fwd_len_mean': ['fwdpacketlengthmean', 'fwdpktlenmean'],
    'bwd_len_mean': ['bwdpacketlengthmean', 'bwdpktlenmean'],
    'avg_pkt_size': ['averagepacketsize', 'avgpacketsize', 'pktsizeavg'],
    'fwd_seg_avg': ['avgfwdsegmentsize', 'fwdsegsizeavg', 'fwdsegmentsizeavg'],
    'bwd_seg_avg': ['avgbwdsegmentsize', 'bwdsegsizeavg', 'bwdsegmentsizeavg'],
    'subflow_fwd_pkts': ['subflowfwdpackets', 'subflowfwdpkts'],
    'subflow_fwd_bytes': ['subflowfwdbytes', 'subflowfwdbyts'],
}

# (min, mean, max) triples; we enforce min <= mean <= max after a step.
ORDER_GROUPS = [
    ('fwdpacketlengthmin', 'fwdpacketlengthmean', 'fwdpacketlengthmax'),
    ('bwdpacketlengthmin', 'bwdpacketlengthmean', 'bwdpacketlengthmax'),
    ('packetlengthmin', 'packetlengthmean', 'packetlengthmax'),
    ('minpacketlength', 'packetlengthmean', 'maxpacketlength'),
    ('flowiatmin', 'flowiatmean', 'flowiatmax'),
    ('fwdiatmin', 'fwdiatmean', 'fwdiatmax'),
    ('bwdiatmin', 'bwdiatmean', 'bwdiatmax'),
    ('activemin', 'activemean', 'activemax'),
    ('idlemin', 'idlemean', 'idlemax'),
]

MONOTONE_KEYS = ['duration', 'fwd_pkts', 'fwd_bytes', 'fwd_len_max', 'subflow_fwd_pkts', 'subflow_fwd_bytes']


def _is_attacker_controllable(n: str) -> bool:
    if any(t in n for t in ('flag', 'header', 'init', 'win', 'bulk', 'urg', 'psh', 'bwd', 'backward')):
        return False
    if n.startswith('flowduration') or n.startswith('flowiat'):
        return True
    if 'fwd' in n and any(t in n for t in ('packet', 'pkt', 'iat', 'seg', 'byte', 'byts', 'len')):
        return True
    return False


class FeatureConstraints:
    """Problem-space constraint set. See module docstring."""

    def __init__(self, prep: Preprocessor, X_train_real_raw: Optional[np.ndarray] = None,
                 mask: str = 'attacker', monotone: bool = True, integer: bool = True,
                 dependencies: bool = True, verbose: bool = True):
        self.prep = prep
        names = [_norm(n) for n in prep.feature_names]
        self.name_to_idx = {n: i for i, n in enumerate(names)}
        self.key_idx = {}
        for key, aliases in ALIASES.items():
            for a in aliases:
                if a in self.name_to_idx:
                    self.key_idx[key] = self.name_to_idx[a]
                    break

        d = prep.n_features
        variable = prep.raw_max > prep.raw_min
        if mask == 'all':
            m = variable.copy()
        elif mask == 'attacker':
            m = np.array([_is_attacker_controllable(n) for n in names]) & variable
            if m.sum() < 3:
                warnings.warn("Feature names do not look like CICFlowMeter output; the attacker "
                              "mask matched <3 features, falling back to all variable features.")
                m = variable.copy()
        else:
            raise ValueError(mask)

        self.derived = {}
        self.order_groups = []
        if dependencies:
            self._setup_dependencies(X_train_real_raw)
            for t in self.derived:
                m[t] = False  # derived features are recomputed, not attacked directly

        self.mask = torch.tensor(m, dtype=torch.float32)
        self.mask_np = m
        self.monotone = np.zeros(d, dtype=bool)
        if monotone:
            for k in MONOTONE_KEYS:
                if k in self.key_idx:
                    self.monotone[self.key_idx[k]] = True
        self.integer = prep.integer_mask.copy() if integer else np.zeros(d, dtype=bool)
        # torch copies of the preprocessor statistics in float64 (rates reach ~1e9)
        self._t = {n: torch.as_tensor(getattr(prep, n), dtype=torch.float64)
                   for n in ('mean', 'scale', 'raw_min', 'raw_max')}
        if verbose:
            print(self.summary())

    # -- dependency set-up --------------------------------------------------

    def _setup_dependencies(self, X_real: Optional[np.ndarray]):
        k = self.key_idx

        def rate(num_keys, den_key):
            def f(X):
                num = sum(X[:, k[n]] for n in num_keys)
                dur = X[:, k[den_key]] / 1e6  # Flow Duration is in microseconds
                return torch.where(dur > 0, num / dur.clamp_min(1e-12), torch.zeros_like(num))
            return f

        def ratio(num_keys, den_keys):
            def f(X):
                num = sum(X[:, k[n]] for n in num_keys)
                den = sum(X[:, k[n]] for n in den_keys)
                return torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num))
            return f

        candidates = [
            ('flow_bytes_s', ['fwd_bytes', 'bwd_bytes', 'duration'], rate(['fwd_bytes', 'bwd_bytes'], 'duration')),
            ('flow_pkts_s', ['fwd_pkts', 'bwd_pkts', 'duration'], rate(['fwd_pkts', 'bwd_pkts'], 'duration')),
            ('fwd_pkts_s', ['fwd_pkts', 'duration'], rate(['fwd_pkts'], 'duration')),
            ('bwd_pkts_s', ['bwd_pkts', 'duration'], rate(['bwd_pkts'], 'duration')),
            ('fwd_len_mean', ['fwd_bytes', 'fwd_pkts'], ratio(['fwd_bytes'], ['fwd_pkts'])),
            ('fwd_seg_avg', ['fwd_bytes', 'fwd_pkts'], ratio(['fwd_bytes'], ['fwd_pkts'])),
            ('avg_pkt_size', ['fwd_bytes', 'bwd_bytes', 'fwd_pkts', 'bwd_pkts'],
             ratio(['fwd_bytes', 'bwd_bytes'], ['fwd_pkts', 'bwd_pkts'])),
            ('subflow_fwd_pkts', ['fwd_pkts'], lambda X: X[:, k['fwd_pkts']]),
            ('subflow_fwd_bytes', ['fwd_bytes'], lambda X: X[:, k['fwd_bytes']]),
        ]
        self.dependency_report = []
        for target, parents, fn in candidates:
            if target not in k or not all(p in k for p in parents):
                continue
            if X_real is None:
                ok, err = False, np.nan
            else:
                sub = X_real[np.random.default_rng(0).choice(len(X_real), min(20000, len(X_real)), replace=False)]
                pred = fn(torch.as_tensor(sub, dtype=torch.float64)).numpy()
                true = sub[:, k[target]]
                rel = np.abs(pred - true) / np.maximum(np.abs(true), 1.0)
                err = float(np.median(rel))
                ok = err < 0.05
            self.dependency_report.append((target, err, ok))
            if ok:
                self.derived[k[target]] = fn

        for lo, mid, hi in ORDER_GROUPS:
            if all(n in self.name_to_idx for n in (lo, mid, hi)):
                self.order_groups.append(tuple(self.name_to_idx[n] for n in (lo, mid, hi)))

    def summary(self) -> str:
        names = self.prep.feature_names
        lines = [f"[constraints] {int(self.mask_np.sum())}/{len(names)} features attacker-controllable",
                 f"[constraints] controllable: {[names[i] for i in np.where(self.mask_np)[0]]}",
                 f"[constraints] monotone-increasing: {[names[i] for i in np.where(self.monotone)[0]]}",
                 f"[constraints] integer-valued: {int(self.integer.sum())} features"]
        for target, err, ok in getattr(self, 'dependency_report', []):
            lines.append(f"[constraints] dependency {target}: median rel. err on real rows "
                         f"{err:.3g} -> {'enforced' if ok else 'NOT enforced'}")
        lines.append(f"[constraints] min<=mean<=max groups enforced: {len(self.order_groups)}")
        return '\n'.join(lines)

    # -- projection ---------------------------------------------------------

    def project(self, X_adv: torch.Tensor, X_orig: torch.Tensor) -> torch.Tensor:
        """Map a scaled candidate onto the feasible set (scaled in, scaled out).
        Differentiable except for rounding (straight-through)."""
        t = self._t
        mask = torch.as_tensor(self.mask_np)
        mono = torch.as_tensor(self.monotone)
        integer = torch.as_tensor(self.integer)
        raw = X_adv.double() * t['scale'] + t['mean']
        raw0 = X_orig.detach().double() * t['scale'] + t['mean']
        raw = torch.where(mask, raw, raw0)
        raw = torch.where(mono, torch.maximum(raw, raw0), raw)
        raw = torch.minimum(torch.maximum(raw, t['raw_min']), t['raw_max'])
        raw = torch.where(integer, raw + (torch.round(raw) - raw).detach(), raw)
        cols = list(raw.unbind(1))
        for tgt, fn in self.derived.items():
            cols[tgt] = fn(torch.stack(cols, 1))
        for lo, mid, hi in self.order_groups:
            cols[hi] = torch.maximum(cols[hi], cols[mid])
            cols[lo] = torch.minimum(cols[lo], cols[mid])
        raw = torch.stack(cols, 1)
        return ((raw - t['mean']) / t['scale']).to(X_adv.dtype)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def attack_loss(logits: torch.Tensor, y: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-sample loss the attacker MAXIMISES.

    ce             untargeted: push away from the true class.
    benign_ce      targeted evasion: pull toward BENIGN (-CE to class 0).
    benign_margin  CW-style evasion: z_benign - max_{j != benign} z_j.
    """
    if kind == 'ce':
        return F.cross_entropy(logits, y, reduction='none')
    if kind == 'benign_ce':
        return -F.cross_entropy(logits, torch.full_like(y, BENIGN), reduction='none')
    if kind == 'benign_margin':
        z_b = logits[:, BENIGN]
        other = logits.clone()
        other[:, BENIGN] = -torch.inf
        return z_b - other.max(dim=1).values
    raise ValueError(kind)


def success(logits: torch.Tensor, y: torch.Tensor, kind: str) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    if kind == 'ce':
        return pred != y
    return pred == BENIGN


# ---------------------------------------------------------------------------
# PGD / FGSM
# ---------------------------------------------------------------------------

def _forward(model: nn.Module, X: torch.Tensor, eot: int) -> torch.Tensor:
    """Logits; averaged over `eot` passes when the model is stochastic (EOT)."""
    if eot <= 1 or not getattr(model, 'stochastic', False):
        return model(X)
    return torch.stack([model(X) for _ in range(eot)]).mean(0)


def pgd(model: nn.Module, X: torch.Tensor, y: torch.Tensor, eps: float, steps: int = 50,
        alpha: Optional[float] = None, restarts: int = 1, random_start: bool = True,
        loss: str = 'benign_margin', constraints: Optional[FeatureConstraints] = None,
        eot: int = 1, extra_loss: Optional[Callable] = None, batch_size: int = 4096) -> torch.Tensor:
    """L-inf PGD with random restarts. Returns, per sample, the restart that
    succeeded (or had the highest loss if none did).

    `extra_loss(X_eval) -> per-sample tensor` is added to the objective. The
    adaptive attack against the DAE detector uses it.
    FGSM is pgd(steps=1, alpha=eps, random_start=False, restarts=1).
    """
    if eps == 0:
        return X.clone()
    if alpha is None:
        alpha = 2.5 * eps / steps if steps > 1 else eps
    was_training = model.training
    model.eval()
    outs = []
    for s in range(0, len(X), batch_size):
        outs.append(_pgd_batch(model, X[s:s + batch_size], y[s:s + batch_size], eps, steps, alpha,
                               restarts, random_start, loss, constraints, eot, extra_loss))
    model.train(was_training)
    return torch.cat(outs)


def _pgd_batch(model, X, y, eps, steps, alpha, restarts, random_start, loss, constraints, eot, extra_loss):
    mask = constraints.mask.to(X.device) if constraints is not None else torch.ones(X.shape[1], device=X.device)
    X0 = X.detach()
    best = X0.clone()
    best_score = torch.full((len(X),), -torch.inf, device=X.device)
    best_succ = torch.zeros(len(X), dtype=torch.bool, device=X.device)

    def evaluate_point(Xc):
        return Xc if constraints is None else constraints.project(Xc, X0)

    for r in range(restarts):
        Xc = X0.clone()
        if random_start:
            Xc = Xc + mask * torch.empty_like(Xc).uniform_(-eps, eps)
        for _ in range(steps):
            Xc.requires_grad_(True)
            Xe = evaluate_point(Xc)
            obj = attack_loss(_forward(model, Xe, eot), y, loss)
            if extra_loss is not None:
                obj = obj + extra_loss(Xe)
            g, = torch.autograd.grad(obj.sum(), Xc)
            with torch.no_grad():
                Xc = Xc + alpha * mask * g.sign()
                Xc = X0 + torch.clamp(Xc - X0, -eps, eps)
            Xc = Xc.detach()
        with torch.no_grad():
            Xf = constraints.project(Xc, X0) if constraints is not None else Xc
            logits = _forward(model, Xf, eot)
            score = attack_loss(logits, y, loss)
            if extra_loss is not None:
                score = score + extra_loss(Xf)
            succ = success(logits, y, loss)
            better = (succ & ~best_succ) | ((succ == best_succ) & (score > best_score))
            best[better] = Xf[better]
            best_score[better] = score[better]
            best_succ |= succ
    return best.detach()


def fgsm(model, X, y, eps, loss='ce', constraints=None, **kw):
    return pgd(model, X, y, eps, steps=1, alpha=eps, restarts=1, random_start=False,
               loss=loss, constraints=constraints, **kw)


ATTACKS: Dict[str, Dict] = {
    # legacy settings, kept for comparison with the old numbers
    'fgsm_untargeted': dict(steps=1, restarts=1, random_start=False, loss='ce'),
    'pgd10_untargeted': dict(steps=10, restarts=1, random_start=False, loss='ce'),
    # evasion-to-benign attacks used for the headline numbers
    'fgsm_evasion': dict(steps=1, restarts=1, random_start=False, loss='benign_margin'),
    'pgd50_evasion': dict(steps=50, restarts=5, random_start=True, loss='benign_margin'),
}


def run_attack(name: str, model, X, y, eps, constraints=None, eot: int = 1, extra_loss=None):
    cfg = dict(ATTACKS[name])
    if cfg['steps'] == 1:
        cfg['alpha'] = eps
    return pgd(model, X, y, eps, constraints=constraints, eot=eot, extra_loss=extra_loss, **cfg)
