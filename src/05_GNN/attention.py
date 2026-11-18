"""
Austauschbare Firma->Kontext-Attention-Module.

Gemeinsame Schnittstelle (alle Module):
    forward(q, kv, mask) -> (out, attn)
        q    : (B, d)     Firmenzustand (Query)
        kv   : (B, N, d)  Kontextknoten (Rohstoffe, Makro, Zinsen) je Firma
        mask : (B, N)     True = Knoten hat gueltige Daten
        out  : (B, d)     "Nachricht", die im Block per Residual auf q addiert wird
        attn : (B, H, N)  Attention-Gewichte (None bei Modulen ohne Attention)

Neues Modul hinzufuegen:
    @register_attention("mein_name")
    class MeinModul(CrossAttentionBase): ...  # scores() ueberschreiben
"""
import math
from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_attention(name: str):
    def deco(cls):
        ATTENTION_REGISTRY[name] = cls
        return cls
    return deco


def build_attention(name: str, d_model: int, n_heads: int = 4,
                    dropout: float = 0.1, topk: Optional[int] = None) -> nn.Module:
    if name not in ATTENTION_REGISTRY:
        raise KeyError(f"Unbekannte Attention '{name}'. Verfuegbar: {list(ATTENTION_REGISTRY)}")
    return ATTENTION_REGISTRY[name](d_model, n_heads=n_heads, dropout=dropout, topk=topk)


# --------------------------------------------------------------------------
# Basisklasse: Maskierung, optionales Top-k, Softmax, Value-Aggregation
# --------------------------------------------------------------------------
class CrossAttentionBase(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1,
                 topk: Optional[int] = None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model muss durch n_heads teilbar sein"
        self.d, self.h, self.dh = d_model, n_heads, d_model // n_heads
        self.topk = topk
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def scores(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Rohscores (B, H, N). Hier unterscheiden sich die Varianten."""
        raise NotImplementedError

    def forward(self, q, kv, mask=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = kv.shape
        s = self.scores(q, kv)                                   # (B,H,N)
        neg = torch.finfo(s.dtype).min
        if mask is not None:
            s = s.masked_fill(~mask[:, None, :], neg)
        if self.topk is not None and self.topk < N:
            kth = s.topk(self.topk, dim=-1).values[..., -1:]
            s = s.masked_fill(s < kth, neg)
        a = torch.softmax(s, dim=-1)
        if mask is not None:                                     # komplett maskierte Zeilen -> 0
            a = a * mask[:, None, :]
        a = self.attn_drop(a)
        v = self.v_proj(kv).view(B, N, self.h, self.dh)
        out = torch.einsum("bhn,bnhd->bhd", a, v).reshape(B, self.d)
        return self.out_proj(out), a


# --------------------------------------------------------------------------
# Varianten
# --------------------------------------------------------------------------
@register_attention("dot")
class DotProductCrossAttention(CrossAttentionBase):
    """Transformer-Cross-Attention (Skalarprodukt). Dynamisch."""

    def __init__(self, d_model, n_heads=4, dropout=0.1, topk=None):
        super().__init__(d_model, n_heads, dropout, topk)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

    def scores(self, q, kv):
        B, N, _ = kv.shape
        qh = self.q_proj(q).view(B, self.h, self.dh)
        kh = self.k_proj(kv).view(B, N, self.h, self.dh)
        return torch.einsum("bhd,bnhd->bhn", qh, kh) / math.sqrt(self.dh)


@register_attention("gatv2")
class GATv2CrossAttention(CrossAttentionBase):
    """GATv2 (Brody et al.): a^T LeakyReLU(W_q q + W_k k). Dynamische Attention."""

    def __init__(self, d_model, n_heads=4, dropout=0.1, topk=None):
        super().__init__(d_model, n_heads, dropout, topk)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.att = nn.Parameter(torch.empty(self.h, self.dh))
        nn.init.xavier_uniform_(self.att)

    def scores(self, q, kv):
        B, N, _ = kv.shape
        qh = self.w_q(q).view(B, 1, self.h, self.dh)
        kh = self.w_k(kv).view(B, N, self.h, self.dh)
        e = F.leaky_relu(qh + kh, 0.2)                           # (B,N,H,dh)
        return (e * self.att).sum(-1).permute(0, 2, 1)           # (B,H,N)


@register_attention("gatv1")
class GATv1CrossAttention(CrossAttentionBase):
    """Original-GAT: LeakyReLU(a_q.q + a_k.k). STATISCH: das Ranking der Knoten ist fuer
    alle Firmen identisch (nur als Ablation gedacht, siehe Brody et al.)."""

    def __init__(self, d_model, n_heads=4, dropout=0.1, topk=None):
        super().__init__(d_model, n_heads, dropout, topk)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.a_q = nn.Parameter(torch.empty(self.h, self.dh))
        self.a_k = nn.Parameter(torch.empty(self.h, self.dh))
        nn.init.xavier_uniform_(self.a_q)
        nn.init.xavier_uniform_(self.a_k)

    def scores(self, q, kv):
        B, N, _ = kv.shape
        sq = (self.w_q(q).view(B, self.h, self.dh) * self.a_q).sum(-1)          # (B,H)
        sk = (self.w_k(kv).view(B, N, self.h, self.dh) * self.a_k).sum(-1)      # (B,N,H)
        return F.leaky_relu(sq[:, :, None] + sk.permute(0, 2, 1), 0.2)          # (B,H,N)


# --------------------------------------------------------------------------
# Baselines ohne Attention (gleiche Schnittstelle)
# --------------------------------------------------------------------------
def _masked_mean(kv: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return kv.mean(1)
    m = mask.unsqueeze(-1).to(kv.dtype)
    return (kv * m).sum(1) / m.sum(1).clamp(min=1.0)


@register_attention("film")
class FiLMConditioning(nn.Module):
    """Kontext wird zu einem Vektor gepoolt, der die Firma moduliert: gamma(c)*q + beta(c).
    Im Block wirkt das als (1+gamma)*q + beta. Keine firmenspezifische Auswahl."""

    def __init__(self, d_model, n_heads=4, dropout=0.1, topk=None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q, kv, mask=None):
        gamma, beta = self.net(_masked_mean(kv, mask)).chunk(2, dim=-1)
        return gamma * q + beta, None


@register_attention("meanpool")
class MeanPoolContext(nn.Module):
    """Einfachste Baseline: gemittelter Kontextvektor -> Linear."""

    def __init__(self, d_model, n_heads=4, dropout=0.1, topk=None):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, q, kv, mask=None):
        return self.proj(_masked_mean(kv, mask)), None
