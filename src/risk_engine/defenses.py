"""
Defence wrappers evaluated by evaluate_robustness.py.

Every wrapper is an nn.Module mapping scaled features -> logits, so the
attacks in attacks.py can differentiate straight through it. Attacking the
wrapper (not the bare classifier) is the adaptive white-box setting. The
legacy phase-10 benchmark crafted perturbations on the baseline classifier
and only then passed them through the DAE, which is a transfer attack and
overstates the defence.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import beta as beta_dist, norm


class DAEPipeline(nn.Module):
    """x -> DAE(x) -> classifier. Fully differentiable, so plain PGD is the adaptive attack."""

    def __init__(self, dae: nn.Module, clf: nn.Module):
        super().__init__()
        self.dae, self.clf = dae, clf

    def forward(self, x):
        return self.clf(self.dae(x))


class ReconstructionDetector:
    """Reject flows whose DAE reconstruction error exceeds `threshold`.
    A rejected flow is treated as an attack (DENY), never as benign."""

    def __init__(self, dae: nn.Module, threshold: float):
        self.dae, self.threshold = dae, threshold

    @torch.no_grad()
    def flag(self, X: torch.Tensor) -> np.ndarray:
        self.dae.eval()
        return (self.dae.reconstruction_error(X) > self.threshold).cpu().numpy()

    def adaptive_penalty(self, weight: float = 10.0):
        """extra_loss for pgd(): the attacker also keeps the reconstruction error
        below the threshold (hinge), i.e. evades classifier AND detector."""
        def f(X):
            return -weight * torch.relu(self.dae.reconstruction_error(X) - 0.9 * self.threshold)
        return f


class SmoothedClassifier(nn.Module):
    """Randomized smoothing (Cohen et al., 2019) with Gaussian noise in scaled space.

    forward() returns log of the Monte-Carlo-averaged softmax for ONE noise draw
    per call; mark stochastic=True so the attacks average gradients over
    several calls (EOT). predict()/certify() use the standard majority vote.
    """
    stochastic = True

    def __init__(self, clf: nn.Module, sigma: float = 0.25, n_train_draws: int = 1):
        super().__init__()
        self.clf, self.sigma, self.n = clf, sigma, n_train_draws

    def forward(self, x):
        probs = torch.stack([torch.softmax(self.clf(x + self.sigma * torch.randn_like(x)), 1)
                             for _ in range(self.n)]).mean(0)
        return torch.log(probs.clamp_min(1e-12))

    @torch.no_grad()
    def _counts(self, x, n, batch=64):
        self.clf.eval()
        counts = torch.zeros(len(x), self.clf.network[-1].out_features, dtype=torch.long)
        remaining = n
        while remaining > 0:
            k = min(batch, remaining)
            xr = x.repeat_interleave(k, 0)
            pred = self.clf(xr + self.sigma * torch.randn_like(xr)).argmax(1).view(len(x), k)
            counts += torch.nn.functional.one_hot(pred, counts.shape[1]).sum(1)
            remaining -= k
        return counts

    @torch.no_grad()
    def predict(self, x, n=200):
        return self._counts(x, n).argmax(1).numpy()

    @torch.no_grad()
    def certify(self, x, n0=100, n=1000, alpha=0.001):
        """Return (prediction, certified L2 radius in scaled units); prediction -1 = abstain."""
        top = self._counts(x, n0).argmax(1)
        counts = self._counts(x, n)
        nA = counts.gather(1, top[:, None]).squeeze(1).numpy()
        pA = beta_dist.ppf(alpha, nA, n - nA + 1)  # Clopper-Pearson lower bound
        pA = np.nan_to_num(pA, nan=0.0)
        radius = self.sigma * norm.ppf(np.clip(pA, 1e-12, 1 - 1e-12))
        pred = np.where(pA > 0.5, top.numpy(), -1)
        return pred, np.where(pA > 0.5, radius, 0.0)


class AveragingEnsemble(nn.Module):
    """Average class probabilities of the MLP and a fixed (non-differentiable)
    tree-model probability table. Only used for transfer evaluation."""

    def __init__(self, clf: nn.Module, tree_model, weight: float = 0.5):
        super().__init__()
        self.clf, self.tree, self.w = clf, tree_model, weight

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.clf.eval()
        p_mlp = torch.softmax(self.clf(torch.as_tensor(X, dtype=torch.float32)), 1).numpy()
        p_tree = self.tree.predict_proba(X)
        return self.w * p_mlp + (1 - self.w) * p_tree
