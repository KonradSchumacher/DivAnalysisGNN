"""
Modell: Firma (64d-VAE + Preisfeatures) wird per Attention mit Kontextknoten (Rohstoffe, Makro, Zinsen)
angereichert. Kein Firma-Firma-Austausch; Nachrichten laufen nur Kontext -> Firma (bipartit).

    firm_x ──FirmEncoder──► h ──[EnrichmentBlock x L]──► emb ──normalize──► z   (fuer Clustering/Korrelationsloss)
                                       ▲                    └─Linear──► ret_pred (Hilfsziel)
    ctx_x ──ContextEncoder──► ctx ─────┘

Attention-Typ per Config: "dot", "gatv2", "gatv1", "film", "meanpool" (oder eigene via register_attention).
"""
from dataclasses import dataclass #Hilfsfunktion, damit man der Klasse wie bei einer Funktion einfach die Parameter übergeben kann und nicht eine eigene Init funktion dafür tippen muss. Liefert auch automatisch printfunktionen usw.
from typing import Dict, Optional

#Standard Pytorch Setup
import torch 
import torch.nn as nn # Linear und Layernorm
import torch.nn.functional as F # Normalize usw

from attention import build_attention # import der verschiedenen Attention Module
from data import Batch # Import der Batchmodule 


@dataclass
class ModelConfig:
    firm_in_dim: int                 # Legt die Dimension der Firmennodes fest: z. B. 64 (VAE) + Anzahl Preisfeatures
    ctx_in_dim: int = 8             # Legt die Dimension der Kontextnodes fest:Preis, Ret 1/3/6/12/24M, Vola, Z-Score
    n_node_types: int = 3            # Wie viele verschiedene Kontextnodetypen gibt es: Rohstoff, Makro, Zins
    n_entities: int = 1              # Wie viele verscheidene Variablennamen gibt es über alle Typen gibt es: Rohstoff-IDs, Makrovariablen, Zinsnamen (Index 0 = "keine")
    n_countries: int = 1            # Wie viele verschiedene Länder gibt es: ISO3-IDs (Index 0 = "kein Land", z. B. Rohstoffe)
    d_model: int = 64                # Mit welcher Dimension soll innerhalb des Modells gearbeitet werden. Muss durch die Anzahl der Attentionheads (n_heads) teilbar sein
    emb_dim: int = 32                # Die Dimension des Output Embeddings, welches dann weiter verwendet wird.
    attn: str = "gatv2"              # Welcher Attention Mechanismus wird genutzt 
    n_heads: int = 4                 # Anzahl der Attention heads. Also in wie viele Teile wird das Embedding geteilt um mit einem Attention Head es zu vergleichen
    n_layers: int = 1                # Wie viel Message Passing soll durchgeführt werden. Da nur Kontext zu Firma sollte einmal reichen.
    topk: Optional[int] = None       # nur die k staerksten Kontextknoten behalten
    dropout: float = 0.1
    ctx_self_layers: int = 0         # Lässt makroknoten untereinander Message Passing machen: optional: Self-Attention zwischen Kontextknoten
    ret_head: bool = True            # Schält einen linearen head ein wo die rendite der Firma vorhersagt.


class ContextEncoder(nn.Module):
    """
    Hier werden die Kontextvektoren auf die entsprechende Modellgröße projeziert und mit einem Nodetyp, Entity und Länderembedding angereichert, 
    damit die Zeitreihen ihren Kontext erhalten z.B. Short-Term Loan, bekommt das Typ Zins Embedding addiert, 
    ein Embedding für das spezifische Feture Short-Term Loan und dann noch einen für das Land z.B. USA
    Typspezifische Projektion + kompositionelle Knotenidentitaet:
        h_j = W_typ(j) x_j + e_typ(j) + e_entity(j) + e_country(j)
    e_country teilt Parameter ueber alle Variablen eines Landes, e_entity ueber alle Laender.
    """

    def __init__(self, cfg: ModelConfig, node_type, node_entity, node_country): # node_type, node_entity, node_country sind 3 Tensorlisten die jeweils für alle Kontextknoten (Länder_Makro, Länder_Zins, Rohstoff) bennen was es genau ist. 
        super().__init__() # Basisklasse für jedes Neuronale Netz worüber die Vererbung läuft. Wird benötigt um alle Parameter sauber durch die ganze Architektur zu tracken.
        d = cfg.d_model # Holt sich die Modell Dimension aus der ModelConfig Klasse
        self.register_buffer("node_type", node_type.long()) # Hängt die ID listen an das Modell als fixe Look up Table an, die kann also nicht verändert werden.
        self.register_buffer("node_entity", node_entity.long())
        self.register_buffer("node_country", node_country.long())
        self.proj = nn.ModuleList([nn.Linear(cfg.ctx_in_dim, d) for _ in range(cfg.n_node_types)]) # Hier wird für jeden Kontextknoten eine Lineare Layer angelegt, also für Rohstoff, Zins und Makro
        self.e_type = nn.Embedding(cfg.n_node_types, d) # Hier wird ein Embedding für den Knotentyp angelegt was dann nachher auf die lineare Projektion der Zeitreihe addiert wird
        self.e_entity = nn.Embedding(cfg.n_entities, d, padding_idx=0)
        self.e_country = nn.Embedding(cfg.n_countries, d, padding_idx=0)
        for e in (self.e_type, self.e_entity, self.e_country): # initialisierung mit kleinen Werte der Embeddings, damit sie die Zahlen der Zeitreihe zu beginn nciht direkt komplett überschrieben.
            nn.init.normal_(e.weight, std=0.02)
            if e.padding_idx is not None:
                with torch.no_grad():
                    e.weight[e.padding_idx].zero_() # Hier wird sichergestellt, das Werte die nicht vorhanden sind wie z.B. Land bei Rohstoff dann auf 0 gesetzt sind.
        self.norm = nn.LayerNorm(d) # Init der Layernormalisierung, damit die Werte nicht explodieren  
        self.self_attn = None
        if cfg.ctx_self_layers > 0: # Falls message Passing unter den Kontextknoten aktiviert ist.
            layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 2 * d, cfg.dropout,
                                               batch_first=True, norm_first=True)
            self.self_attn = nn.TransformerEncoder(layer, cfg.ctx_self_layers,
                                                   enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (M,N,F), mask: (M,N) -> (M,N,d)
            M=Anzahl der Monate, N= Anzahl der Kontextknoten, F=Anzahl der Features pro Knoten
        """
        x = torch.nan_to_num(x) # Falls noch irgendwo NaN Werte drin sind werden sie heir einfach auf 0 gesetzt, damit NaN nicht durchpropaiert wird
        M, N, _ = x.shape 
        h = x.new_zeros(M, N, self.e_type.embedding_dim) # Hier wird der Zielvektor erstellt mit Nullen erstellt.
        for t, lin in enumerate(self.proj): # Hier werden die Knotentypen also Rohstoff, Zins und Makro rausgefiltert und durch die dafür vorgesehenen Linearen Schichten gedrückt, damit die Zeitreihen Embeddings heruaskommen
            idx = self.node_type == t
            if idx.any():
                h[:, idx] = lin(x[:, idx]).to(h.dtype)
        h = h + (self.e_type(self.node_type) + self.e_entity(self.node_entity) + self.e_country(self.node_country)) # Hier wird auf das Zeitreihenembedding dann die NodeTyp, EntityTyp und Land Embeddings addiert.
        h = self.norm(h) * mask.unsqueeze(-1) # Setzt alle Werte wo eine NaN Waren auf 0 damit sie in einer optionalen message passing innerhalb der Kontexnodes nicht weitergeleitet wird.
        if self.self_attn is not None:
            h = self.self_attn(h, src_key_padding_mask=~mask)
            h = h * mask.unsqueeze(-1)
        return h


class FirmEncoder(nn.Module):
    def __init__(self, in_dim: int, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d)) # Bildet ein Netz wo die Firmeninputs, also VAE Embedding und Kurszeitreihe verarbeitet und dann ein Firmenembedding der gleichen Dimension ausgibt wie die Kontextknoten.
        #LayerNorm bringt VAE Embedding und Kursinfo in einen gleichen Wertebereich. die erste Lineare Schicht bringt den Input auf die richtige Dimension und verbinet Bilanz und Kurs. Dann Aktivierungsfunktion für nichtlineare Zusammenhänge und dann wieder eine lienare Schicht um weitere features zu extrahieren.
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
        msg, attn = self.attn(self.norm_q(h), self.norm_kv(ctx), mask) # Hier wird die Messages aus den Makroknote berechnet und zusammen mit den attention scores zurückgegeben
        h = h + self.drop(msg) # Hier wird dann die messages auf das ursprüngliche Firmenembedding addiert und somit kontextualisiert.
        h = h + self.drop(self.ffn(self.norm_ffn(h))) # Das Kontexttualisierte Firmenembedding wird dann nochmal durch ein FFN geschickt umd die Zusammenhänge rauszuarbeiten. Die überarbeitete version wird dann wieder auf die kontextualisierte addiert.
        return h, attn # Gibt nun das kontextualisierte Embedding und die Attention Scores zurück.


class ContextEnrichedEmbedder(nn.Module):
    def __init__(self, cfg: ModelConfig, node_type, node_entity, node_country):
        super().__init__()
        self.cfg = cfg
        self.firm_enc = FirmEncoder(cfg.firm_in_dim, cfg.d_model, cfg.dropout)
        self.ctx_enc = ContextEncoder(cfg, node_type, node_entity, node_country)
        self.blocks = nn.ModuleList([EnrichmentBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model) #Normalisiert nochmal die fertigen Embeddings
        self.out_proj = nn.Linear(cfg.d_model, cfg.emb_dim) # Die fertigen Embeddings werden nochmals in eine linearen Transformation auf eine finale Embedding size transformiert
        self.ret_head = nn.Linear(cfg.emb_dim, 1) if cfg.ret_head else None   #Macht die Return prediction falls diese auf True gesetzt wurde.

    def forward(self, batch: Batch) -> Dict[str, Optional[torch.Tensor]]:
        h = self.firm_enc(batch.firm_x)                                        # (B,d)
        ctx = self.ctx_enc(batch.ctx_x, batch.ctx_mask)                        # (M,N,d)
        if ctx.size(0) == 1: # Ist irgendwie ein Speichereffizient ding
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
    def embed(self, batch: Batch) -> torch.Tensor: # Hilfsmethode, damit man einfach nur die Embeddings rausbekommt aus dem trainierten Modell
        """Normalisierte Embeddings fuer KMeans/Agglomerative (euklidisch ~ Kosinus)."""
        was_training = self.training
        self.eval()
        z = self.forward(batch)["z"]
        self.train(was_training)
        return z
