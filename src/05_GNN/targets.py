"""
Zielgroessen: Marktbereinigung und Korrelationsmatrix aus Residual-Returns.

Ablauf pro Zielfenster (t, t+h]:
    1. R (T, N_universe) -> residualize_returns(R, method)   # einmal pro Fenster, ueber das ganze Universum
    2. Spalten der Batch-Firmen auswaehlen -> Batch.fwd_resid (T, B)
    3. corr_from_residuals(fwd_resid)                        # (B, B), on the fly im Loss

So muss nie eine 6200x6200-Matrix pro Fenster gespeichert werden.
Evaluation (Diversification Ratio) immer auf den UNBEREINIGTEN Returns.
"""
from typing import Literal, Optional

import torch

Method = Literal["none", "ew_market", "pc"]


def cross_sectional_demean(r: torch.Tensor) -> torch.Tensor:
    """Hilfsziel-Returns: Querschnittsmittel (NaN-sicher) abziehen."""
    return r - torch.nanmean(r, dim=-1, keepdim=True)


def _demean(R: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    n = valid.sum(0).clamp(min=1)
    mean = (torch.nan_to_num(R) * valid).sum(0) / n
    return (torch.nan_to_num(R) - mean) * valid


@torch.no_grad()
def residualize_returns(R: torch.Tensor, method: Method = "ew_market",
                        n_factors: int = 1) -> torch.Tensor:
    """
    R: (T, N) Returns, NaN = fehlend. Gibt Residuen (T, N) zurueck (NaN bleibt NaN).

    "none"      : nur zentrieren
    "ew_market" : gleichgewichteter Marktreturn (Querschnittsmittel je Zeitpunkt) als Faktor
    "pc"        : erste n_factors Hauptkomponenten der standardisierten Returns (SVD);
                  n_factors=1 entspricht dem Entfernen des "market mode"
    Hinweis: fuer mehrere Regionen den Faktor je Region/Boerse separat anwenden, falls gewuenscht.
    """
    valid = torch.isfinite(R)
    Rc = _demean(R, valid)                                       # (T,N), fehlend = 0
    if method == "none":
        resid = Rc
    else:
        if method == "ew_market":
            f = torch.nanmean(R, dim=1, keepdim=True)            # (T,1)
            Fm = f - f.mean(0, keepdim=True)
        elif method == "pc":
            std = (Rc ** 2).sum(0).div(valid.sum(0).clamp(min=1)).sqrt() + 1e-8
            U, S, _ = torch.linalg.svd(Rc / std, full_matrices=False)
            Fm = U[:, :n_factors] * S[:n_factors]                # (T,k)
        else:
            raise ValueError(method)
        beta = torch.linalg.lstsq(Fm, Rc).solution               # (k,N)
        resid = (Rc - Fm @ beta) * valid
    return resid.masked_fill(~valid, float("nan"))


def corr_from_residuals(E: torch.Tensor, min_obs: int = 20) -> torch.Tensor:
    """
    E: (T, B) Residual-Returns (NaN erlaubt). Gibt (B, B) Korrelation zurueck;
    Paare mit weniger als min_obs Beobachtungen (je Firma) werden NaN.
    Naeherung bei fehlenden Werten: Standardisierung je Spalte, Nullauffuellung.
    """
    valid = torch.isfinite(E)
    n = valid.sum(0)
    E0 = torch.nan_to_num(E)
    mean = (E0 * valid).sum(0) / n.clamp(min=1)
    Ec = (E0 - mean) * valid
    std = (Ec ** 2).sum(0).div(n.clamp(min=1)).sqrt() + 1e-8
    Z = Ec / std
    C = (Z.T @ Z) / torch.sqrt(n[:, None].clamp(min=1) * n[None, :].clamp(min=1))
    C = C.clamp(-1.0, 1.0)
    ok = (n >= min_obs)
    C = C.masked_fill(~(ok[:, None] & ok[None, :]), float("nan"))
    C.fill_diagonal_(1.0)
    return C
