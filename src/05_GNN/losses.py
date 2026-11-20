"""
Losses. Alle Korrelations-Losses haben die Signatur  forward(z, rho) -> Skalar
    z   : (B, d) L2-normalisierte Embeddings
    rho : (B, B) Zielkorrelation (NaN = ignorieren)

Wichtig: Die Losses haben teils eigene Parameter (a, b bzw. log-Varianzen).
-> beim Optimizer: list(model.parameters()) + list(criterion.parameters())
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from targets import corr_from_residuals


def _offdiag_valid(rho: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(rho.size(0), dtype=torch.bool, device=rho.device)
    return (~eye) & torch.isfinite(rho)


class CorrelationRegressionLoss(nn.Module):
    """Variante 1: rho_hat = a*cos + b (a, b lernbar) gegen rho; optional im Fisher-z-Raum."""

    def __init__(self, fisher: bool = True, learn_affine: bool = True, eps: float = 1e-4):
        super().__init__()
        self.fisher, self.eps = fisher, eps
        self.a = nn.Parameter(torch.tensor(1.0), requires_grad=learn_affine)
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=learn_affine)

    def forward(self, z, rho):
        valid = _offdiag_valid(rho)
        pred = self.a * (z @ z.T) + self.b
        if self.fisher:
            lim = 1.0 - self.eps
            pred = torch.atanh(pred.clamp(-lim, lim))
            rho = torch.atanh(torch.nan_to_num(rho).clamp(-lim, lim))
        return F.mse_loss(pred[valid], torch.nan_to_num(rho)[valid])


class ListwiseRankingLoss(nn.Module):
    """Variante 2: Zeilen-Softmax der Modell-Aehnlichkeiten gegen Zeilen-Softmax der Zielkorrelationen
    (Kreuzentropie = KL + Konstante). Unempfindlich gegen konstanten Offset (Marktniveau) je Zeile."""

    def __init__(self, tau: float = 0.1, tau_target: float = 0.1):
        super().__init__()
        self.tau, self.tau_t = tau, tau_target

    def forward(self, z, rho):
        valid = _offdiag_valid(rho)
        neg = torch.finfo(z.dtype).min
        logits = ((z @ z.T) / self.tau).masked_fill(~valid, neg)
        logp = torch.log_softmax(logits, dim=-1)
        tgt = (torch.nan_to_num(rho) / self.tau_t).masked_fill(~valid, neg)
        q = torch.softmax(tgt, dim=-1) * valid
        ce = -(q * torch.where(valid, logp, torch.zeros_like(logp))).sum(-1)
        rows = valid.any(-1)
        return ce[rows].mean()


class ReturnLoss(nn.Module):
    """Hilfsziel: (Querschnitts-)Return je Firma. Huber optional (Returns haben schwere Raender)."""

    def __init__(self, huber_delta: Optional[float] = None):
        super().__init__()
        self.delta = huber_delta

    def forward(self, pred, target):
        ok = torch.isfinite(target)
        if not ok.any():
            return pred.sum() * 0.0
        if self.delta is None:
            return F.mse_loss(pred[ok], target[ok])
        return F.huber_loss(pred[ok], target[ok], delta=self.delta)


class MultiTaskLoss(nn.Module):
    """L = L_corr + lambda * L_ret   (statisch)   oder   Uncertainty Weighting nach Kendall et al."""

    def __init__(self, corr_loss: nn.Module, ret_loss: Optional[nn.Module] = None,
                 ret_weight: float = 0.1, uncertainty_weighting: bool = False,
                 min_obs: int = 8):
        super().__init__()
        self.corr_loss, self.ret_loss = corr_loss, ret_loss
        self.ret_weight, self.uw, self.min_obs = ret_weight, uncertainty_weighting, min_obs
        if uncertainty_weighting and ret_loss is not None:
            self.log_var = nn.Parameter(torch.zeros(2))

    def forward(self, out: Dict[str, torch.Tensor], batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        rho = corr_from_residuals(batch.fwd_resid, self.min_obs)
        l_corr = self.corr_loss(out["z"], rho)
        logs = {"corr": float(l_corr)}
        if self.ret_loss is None or out.get("ret_pred") is None or batch.fwd_ret is None:
            return l_corr, logs
        l_ret = self.ret_loss(out["ret_pred"], batch.fwd_ret)
        logs["ret"] = float(l_ret)
        if self.uw:
            total = (torch.exp(-self.log_var[0]) * l_corr + self.log_var[0]
                     + torch.exp(-self.log_var[1]) * l_ret + self.log_var[1])
        else:
            total = l_corr + self.ret_weight * l_ret
        logs["total"] = float(total)
        return total, logs
