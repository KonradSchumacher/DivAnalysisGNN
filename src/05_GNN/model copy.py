"""
Modell: Firma (64d-VAE + Preisfeatures) wird per Attention mit Kontextknoten (Rohstoffe, Makro, Zinsen)
angereichert. Kein Firma-Firma-Austausch; Nachrichten laufen nur Kontext -> Firma (bipartit).

    firm_x ──FirmEncoder──► h ──[EnrichmentBlock x L]──► emb ──normalize──► z   (fuer Clustering/Korrelationsloss)
                                       ▲                    └─Linear──► ret_pred (Hilfsziel)
    ctx_x ──ContextEncoder──► ctx ─────┘

Attention-Typ per Config: "dot", "gatv2", "gatv1", "film", "meanpool" (oder eigene via register_attention).
"""
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import build_attention
from .data import Batch


@dataclass
class ModelConfig:
    firm_in_dim: int                 # z. B. 64 (VAE) + Anzahl Preisfeatures
    ctx_in_dim: int = 8              # Preis, Ret 1/3/6/12/24M, Vola, Z-Score
    n_node_types: int = 3            # Rohstoff, Makro, Zins
    n_entities: int = 1              # Rohstoff-IDs, Makrovariablen, Zinsnamen (Index 0 = "keine")
    n_countries: int = 1             # ISO3-IDs (Index 0 = "kein Land", z. B. Rohstoffe)
    d_model: int = 64
    emb_dim: int = 32
    attn: str = "gatv2"
    n_heads: int = 4
    n_layers: int = 1
    topk: Optional[int] = None       # nur die k staerksten Kontextknoten behalten
    dropout: float = 0.1
    ctx_self_layers: int = 0         # optional: Self-Attention zwischen Kontextknoten
    ret_head: bool = True


class ContextEncoder(nn.Module):
    """
    Typspezifische Projektion + kompositionelle Knotenidentitaet:
        h_j = W_typ(j) x_j + e_typ(j) + e_entity(j) + e_country(j)
    e_country teilt Parameter ueber alle Variablen eines Landes, e_entity ueber alle Laender.
    """

    def __init__(self, cfg: ModelConfig, node_type, node_entity, node_country):
        super().__init__()
        d = cfg.d_model
        self.register_buffer("node_type", node_type.long())
        self.register_buffer("node_entity", node_entity.long())
        self.register_buffer("node_country", node_country.long())
        self.proj = nn.ModuleList([nn.Linear(cfg.ctx_in_dim, d) for _ in range(cfg.n_node_types)])
        self.e_type = nn.Embedding(cfg.n_node_types, d)
        self.e_entity = nn.Embedding(cfg.n_entities, d, padding_idx=0)
        self.e_country = nn.Embedding(cfg.n_countries, d, padding_idx=0)
        for e in (self.e_type, self.e_entity, self.e_country):
            nn.init.normal_(e.weight, std=0.02)
            if e.padding_idx is not None:
                with torch.no_grad():
                    e.weight[e.padding_idx].zero_()
        self.norm = nn.LayerNorm(d)
        self.self_attn = None
        if cfg.ctx_self_layers > 0:
            layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 2 * d, cfg.dropout,
                                               batch_first=True, norm_first=True)
            self.self_attn = nn.TransformerEncoder(layer, cfg.ctx_self_layers,
                                                   enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (M,N,F), mask: (M,N) -> (M,N,d)"""
        x = torch.nan_to_num(x)
        M, N, _ = x.shape
        h = x.new_zeros(M, N, self.e_type.embedding_dim)
        for t, lin in enumerate(self.proj):
            idx = self.node_type == t
            if idx.any():
                h[:, idx] = lin(x[:, idx])
        h = h + (self.e_type(self.node_type) + self.e_entity(self.node_entity)
                 + self.e_country(self.node_country))
        h = self.norm(h) * mask.unsqueeze(-1)
        if self.self_attn is not None:
            h = self.self_attn(h, src_key_padding_mask=~mask)
            h = h * mask.unsqueeze(-1)
        return h


class FirmEncoder(nn.Module):
    def __init__(self, in_dim: int, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, x):
        return self.net(torch.nan_to_num(x))


class EnrichmentBlock(nn.Module):
    """Pre-Norm: h <- h + Attn(h, ctx);  h <- h + FFN(h)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.norm_q, self.norm_kv, self.norm_ffn = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = build_attention(cfg.attn, d, cfg.n_heads, cfg.dropout, cfg.topk)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(cfg.dropout),
                                 nn.Linear(2 * d, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h, ctx, mask):
        msg, attn = self.attn(self.norm_q(h), self.norm_kv(ctx), mask)
        h = h + self.drop(msg)
        h = h + self.drop(self.ffn(self.norm_ffn(h)))
        return h, attn


class ContextEnrichedEmbedder(nn.Module):
    def __init__(self, cfg: ModelConfig, node_type, node_entity, node_country):
        super().__init__()
        self.cfg = cfg
        self.firm_enc = FirmEncoder(cfg.firm_in_dim, cfg.d_model, cfg.dropout)
        self.ctx_enc = ContextEncoder(cfg, node_type, node_entity, node_country)
        self.blocks = nn.ModuleList([EnrichmentBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.emb_dim)
        self.ret_head = nn.Linear(cfg.emb_dim, 1) if cfg.ret_head else None   # linear, siehe Diskussion

    def forward(self, batch: Batch) -> Dict[str, Optional[torch.Tensor]]:
        h = self.firm_enc(batch.firm_x)                                        # (B,d)
        ctx = self.ctx_enc(batch.ctx_x, batch.ctx_mask)                        # (M,N,d)
        if ctx.size(0) == 1:
            ctx_b = ctx.expand(h.size(0), -1, -1)
            mask_b = batch.ctx_mask.expand(h.size(0), -1)
        else:
            ctx_b = ctx[batch.firm_to_ctx]
            mask_b = batch.ctx_mask[batch.firm_to_ctx]
        attn = None
        for blk in self.blocks:
            h, attn = blk(h, ctx_b, mask_b)
        emb = self.out_proj(self.out_norm(h))                                  # (B,emb_dim)
        return {
            "emb": emb,
            "z": F.normalize(emb, dim=-1),                                     # fuer Loss + Clustering
            "ret_pred": self.ret_head(emb).squeeze(-1) if self.ret_head is not None else None,
            "attn": attn,                                                      # (B,H,N) der letzten Schicht
        }

    @torch.no_grad()
    def embed(self, batch: Batch) -> torch.Tensor:
        """Normalisierte Embeddings fuer KMeans/Agglomerative (euklidisch ~ Kosinus)."""
        was_training = self.training
        self.eval()
        z = self.forward(batch)["z"]
        self.train(was_training)
        return z
