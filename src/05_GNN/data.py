"""
Batch-Container. Ein Batch ist ein Querschnitt von B Firmen; Kontext-Snapshots werden je Monat
einmal gehalten (M Monate im Batch, meist M=1) und per `firm_to_ctx` den Firmen zugeordnet.

Zeitliche Disziplin (wichtig gegen Look-ahead):
    firm_x, ctx_x   : nur Information bis Stichtag t (Makro mit Veroeffentlichungsverzug!)
    fwd_resid       : Returns im Folgefenster (t, t+h], bereits marktbereinigt (targets.py)
    fwd_ret         : Querschnitts-demeanter Return im Folgefenster (Hilfsziel)
"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Batch:
    firm_x: torch.Tensor                      # (B, F_firm)  64d-VAE-Embedding + Preisfeatures
    ctx_x: torch.Tensor                       # (M, N, F_ctx) Kontext-Features je Monat
    ctx_mask: torch.Tensor                    # (M, N) bool, True = Daten vorhanden
    firm_to_ctx: torch.Tensor                 # (B,) long, Index in [0, M)
    fwd_resid: Optional[torch.Tensor] = None  # (T_fwd, B) Returns fuer Zielkorrelation (NaN erlaubt)
    fwd_ret: Optional[torch.Tensor] = None    # (B,) Hilfsziel (NaN erlaubt)

    def to(self, device) -> "Batch":
        return Batch(**{k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in self.__dict__.items()})
