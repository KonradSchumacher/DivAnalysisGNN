"""
Minimalbeispiel mit synthetischen Daten. Zeigt:
  - Aufbau der Kontextknoten (Typ / Entitaet / Land)
  - Austausch von Attention-Typ und Korrelations-Loss per String
  - einen Trainingsschritt inkl. Multitask-Loss

Ausfuehren:  python -m ctxemb.example
"""
import itertools

import torch

from .data import Batch
from .losses import CorrelationRegressionLoss, ListwiseRankingLoss, MultiTaskLoss, ReturnLoss
from .model import ContextEnrichedEmbedder, ModelConfig
from .targets import cross_sectional_demean, residualize_returns

CORR_LOSSES = {"regression": CorrelationRegressionLoss, "ranking": ListwiseRankingLoss}


def make_synthetic(B=96, T=120, n_com=20, n_countries=6, n_macro_vars=3, firm_dim=72, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Kontextknoten: Rohstoffe (Typ 0), Makro (Typ 1: Land x Variable), Zinsen (Typ 2: je Land)
    types, ents, ctrs = [], [], []
    for i in range(n_com):
        types.append(0); ents.append(1 + i); ctrs.append(0)
    for c in range(n_countries):
        for v in range(n_macro_vars):
            types.append(1); ents.append(1 + n_com + v); ctrs.append(1 + c)
        types.append(2); ents.append(1 + n_com + n_macro_vars); ctrs.append(1 + c)
    node_type, node_entity, node_country = map(torch.tensor, (types, ents, ctrs))
    N = len(types)
    # Faktorstruktur: Firmen haengen an latenten Faktoren -> Korrelationen sind lernbar
    n_fac = 4
    load = torch.randn(B, n_fac, generator=g)
    fac = torch.randn(T, n_fac, generator=g)
    R = fac @ load.T * 0.01 + 0.01 * torch.randn(T, B, generator=g) + 0.005 * torch.randn(T, 1, generator=g)
    firm_x = torch.cat([load + 0.1 * torch.randn_like(load),
                        torch.randn(B, firm_dim - n_fac, generator=g)], dim=1)
    ctx_x = torch.randn(1, N, 8, generator=g)
    ctx_mask = torch.rand(1, N, generator=g) > 0.1
    batch = Batch(
        firm_x=firm_x, ctx_x=ctx_x, ctx_mask=ctx_mask,
        firm_to_ctx=torch.zeros(B, dtype=torch.long),
        fwd_resid=residualize_returns(R, "ew_market"),
        fwd_ret=cross_sectional_demean(R.sum(0)),
    )
    cfg_kwargs = dict(firm_in_dim=firm_dim, n_entities=2 + n_com + n_macro_vars,
                      n_countries=1 + n_countries)
    return batch, (node_type, node_entity, node_country), cfg_kwargs


def build(attn: str, corr_loss: str, cfg_kwargs, nodes, **overrides):
    cfg = ModelConfig(attn=attn, **cfg_kwargs, **overrides)
    model = ContextEnrichedEmbedder(cfg, *nodes)
    criterion = MultiTaskLoss(CORR_LOSSES[corr_loss](), ReturnLoss(), ret_weight=0.1)
    return model, criterion


def train_step(model, criterion, optimizer, batch):
    model.train()
    optimizer.zero_grad()
    out = model(batch)
    loss, logs = criterion(out, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return logs


if __name__ == "__main__":
    batch, nodes, cfg_kwargs = make_synthetic()
    for attn, closs in itertools.product(["dot", "gatv2", "gatv1", "film", "meanpool"],
                                         ["regression", "ranking"]):
        torch.manual_seed(0)
        model, criterion = build(attn, closs, cfg_kwargs, nodes)
        opt = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=1e-3,
                                weight_decay=1e-2)
        first = last = None
        for step in range(60):
            logs = train_step(model, criterion, opt, batch)
            first = first or logs["corr"]
            last = logs["corr"]
        z = model.embed(batch)
        print(f"{attn:9s} {closs:10s} corr-loss {first:.4f} -> {last:.4f}   z {tuple(z.shape)}")
    # Optionen: Top-k, Kontext-Self-Attention, 2 Schichten
    model, criterion = build("gatv2", "ranking", cfg_kwargs, nodes, topk=8, ctx_self_layers=1, n_layers=2)
    out = model(batch)
    print("topk/self/2-layer ok:", tuple(out["z"].shape), tuple(out["attn"].shape))
