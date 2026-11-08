# %%
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# %%
def data_split(data: pd.DataFrame, test_cutoff_date: str = '2023-12-31'):
    # 1. Daten auf Stichtag begrenzen (Kein Blick in die Zukunft für den VAE!)
    df_historical = data[data['Date'] <= test_cutoff_date].copy()
    # 2. Unique Firmen (Entities) splitten
    unique_entities = df_historical['EntityCode'].unique()
    train_entities, temp_entities = train_test_split(unique_entities, train_size=0.70, random_state=42, shuffle=True)
    val_entities, test_entities = train_test_split(temp_entities, test_size=0.50, random_state=42, shuffle=True)
    # 3. Train & Val (Panel-Daten: alle historischen Bilanzen <= Cutoff)
    train = df_historical[df_historical['EntityCode'].isin(train_entities)].reset_index(drop=True)
    val   = df_historical[df_historical['EntityCode'].isin(val_entities)].reset_index(drop=True)
    # 4. Test (Exakt die aktuellste Bilanz pro Firma bis zum Cutoff-Datum)
    test = (df_historical[df_historical['EntityCode'].isin(test_entities)].sort_values('Date').groupby('EntityCode').last().reset_index())
    #Hier soll verhindert werden, dass ganz alte Fälle aus 2021 oderso im Test landen
    test = test[test['Date'] > '2023-01-01']
    print(f"Train Rows: {len(train)} ({len(train_entities)} Firmen)")
    print(f"Val Rows:   {len(val)} ({len(val_entities)} Firmen)")
    print(f"Test Rows:  {len(test)} ({len(test_entities)} Firmen)")
    return train, val, test

def robust_scaling(dataset: pd.DataFrame, scaler: Optional[RobustScaler] = None):
    if scaler is None:
        scaler = RobustScaler()
        scaled_array = scaler.fit_transform(dataset)
    else:
        scaled_array = scaler.transform(dataset)
    return scaled_array, scaler

def transformation(dataset: pd.DataFrame, usable_features: list, scaler):
    arcsinh = np.arcsinh(dataset[usable_features])
    mask = (~arcsinh.isna()).astype(np.float32).values
    scaled, scaler = robust_scaling(arcsinh, scaler)
    scaled_array = np.nan_to_num(scaled, nan=0.0).astype(np.float32)
    return scaled_array, mask, scaler

def full_data_prep():
    data = joblib.load("../../data/03_gold/full_data.pkl")
    df = data['data']
    usable_features = data['usable_features']
    train, val, test = data_split(df, test_cutoff_date='2023-12-31')
    test_entities = test['EntityCode'].values
    train, train_mask, scaler = transformation(train, usable_features, None)
    val, val_mask, _ = transformation(val, usable_features, scaler)
    test, test_mask, _ = transformation(test, usable_features, scaler)
    data_preped = {"train": train, "train_mask": train_mask, "val": val, "val_mask": val_mask, "test": test, "test_mask": test_mask, "test_entities": test_entities, "usable_features": usable_features, "scaler": scaler}
    return data_preped

# %%
data = full_data_prep()
X_train_scaled = data['train']
X_train_mask = data['train_mask']
X_val_scaled = data['val']
X_val_mask = data['val_mask']
X_test_scaled = data['test']
X_test_mask = data['test_mask']
test_entities = data['test_entities']
usable_features = data['usable_features']
scaler = data['scaler']

dataset_dict = {
    "X_test_scaled": X_test_scaled, 
    "X_test_mask": X_test_mask,
    "X_test_EntityCodes": test_entities,        
    "usable_features": usable_features
}

joblib.dump(dataset_dict, "../../data/03_gold/testset_scaled.pkl")

# %%
"""
VAE mit gelernten Missing-Value-Embeddings für finanzielle Panel-Daten.

Erwartete Inputs (aus der vorherigen Datenvorbereitung):
- X_train, X_val, X_test: np.ndarray, shape (N, D)
    skalierte Feature-Werte, fehlende Werte mit 0.0 gefüllt
    (Skalierung: mean/std NUR aus X_train berechnet, siehe vorheriger Schritt)
- M_train, M_val, M_test: np.ndarray, shape (N, D)
    Missingness-Maske, 1 = beobachtet, 0 = fehlend

Idee:
- Jedes Feature bekommt eine eigene lineare "Werte-Embedding"-Projektion.
- Für fehlende Werte (mask == 0) wird diese Projektion durch einen gelernten,
  pro Feature geteilten "Missing"-Embedding-Vektor ersetzt (analog zu einem
  Mask-Token in Transformern) statt den künstlichen 0-Platzhalter zu embedden.
- Der Reconstruction-Loss wird nur über tatsächlich beobachtete Zellen berechnet
  (maskierter Loss) - der Decoder wird also nie dafür bestraft, dass er
  ursprünglich fehlende Werte "falsch" rekonstruiert.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#Datenvorbereitungsklasse, kann auch direkt durch TensorDataset() PyTorch Klasse gemacht werden, aber so kann man später noch Anpassungen machen oder die Transformation noch dort reinpacken. 
class PanelDataset(Dataset):
    def __init__(self, X, M): # Objekt kann nur mit Features und Maske initialisiert werden
        self.X = torch.as_tensor(X, dtype=torch.float32) #Wandelt in Tensoren mit 32Bit Floats um. Tensoren sind im Grunde das gleiche wie Arrays aber sie können auf GPU rechnen und PyTorch zeichnet alle mathematischen Operationen in dynamischen Rechengraph auf Tensoren auf -> Es kann somit besser Gradienten berechnen  
        self.M = torch.as_tensor(M, dtype=torch.float32)

    def __len__(self):
        return len(self.X) #Für Batchsize usw braucht man die Länge des Datensatzes

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx] # Damit man sicher sein kann, dass Beispiel und Maske immer zusammenpassen


class FeatureEmbedding(nn.Module):
    """Pro Feature eine Werte-Embedding-Projektion + ein gelerntes Missing-Embedding."""

    def __init__(self, num_features: int, emb_dim: int):
        super().__init__() #Das ist der Konstruktoraufruf für die nn.Module Elternklasse. Hier werden z.B. die Parameter in geeigneten Datenstrukturen für den Optimuzer gespeichert.
        self.num_features = num_features #Übergabe der Anzahl der Features das braucht man um gemeinsam mit der Embedding Dimension die Anzahl der Parameter zu bestimmen die für die Erstellung des Input Embeddings benötig werden 
        self.emb_dim = emb_dim # Hier legt man die Embedding Dimension fest also in was für einer Dimension soll das Feature dargestellt werden.

        # Pro Feature eigene lineare Projektion: x_d -> R^emb_dim
        self.value_weight = nn.Parameter(torch.randn(num_features, emb_dim) * 0.1) # nn.Parameters baut die Gewichtsmatrizen auf die dann später für die Transformation der Input Embeddings verwendet wird. nn.Parameter ist dafür nötig weil der Tensor sonst nicht als Gewichtsmatrix erkannt wird und vom Modell einfach ignoriert würde. 0.1 dient dazu das die Werte möglichst nahe bei 0 initialisiert werden für stabiles Training
        self.value_bias = nn.Parameter(torch.zeros(num_features, emb_dim)) # Genau das gleiche wie value_weight nur für die Biasmatrix. Hier wird jedoch direkt alles mit 0 initialisiert, da der Bias ja addiert wird.

        # Ein gelernter Embedding-Vektor pro Feature für "fehlt"
        self.missing_embedding = nn.Parameter(torch.randn(num_features, emb_dim) * 0.1) # Hier dann nochmal für die Missing Value Embeddings. Der Shape ist immer num_features x emb_dim, damit jedes Feature einen Vektor der Embedding Dimension hat.

    def forward(self, x, mask):
        # x, mask: (batch, num_features)
        value_emb = x.unsqueeze(-1) * self.value_weight + self.value_bias  # (batch, D, emb_dim)  Das unsqueeze fügt hinten eine Dimension an. Dann wird durch das Broadcasting Verfahren die Tensoren von recht nach links angelichen sodass erkannt wird, dass es keine Batch Dimension gibt und diese deshalb passend zur Batchsize kopiert wird, damit jedes Sample die Gewichtsmatrix hat. Dann wird die Skalar Inputs mit der Embedding Transformation multipliziert und somit kommt dann die emb_dim dabei heraus.  
        mask_exp = mask.unsqueeze(-1)  # (batch, D, 1) #Gleiches beim Mask Tensor es wird eine Dimension hinten angehängt
        combined = mask_exp * value_emb + (1 - mask_exp) * self.missing_embedding # Hier werden dann als erste ein Tensor erstellt der alle tatsächlich enthaltetenen Werte transformiert und die nicht Missing auf 0 setzt und dann im zweiten Schritte werden die Missing Embeddings an die entsprechenden Stellen aufaddiert.
        return combined.flatten(start_dim=1)  # (batch, D * emb_dim) Zurückgegeben wird dann wieder ein 2D Tensor, da die Embeddings nun einfach nacheinander aufgereiht werden, damit sie als einzelne Inputs für eine Lineare Schicht genutzt werden können.


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate):
        super().__init__() #Wieder der Konstruktoraufruf für das nn.Module zur Vererbung und initialisierung wichtiger Parts
        layers = [] # Speicher in dem die einzelnen Schichten, also jeweils die Lineare Transformation mit Inputgröße und Outpuntdimension, sowie die nicht-lineare Aktivierungsfunktion
        prev = input_dim # Hier wird die Inputdimension initialisiert. Wird dann immer auf die Zieldimension der linearen Transformation gesetzt
        for h in hidden_dims: # Durchlauf durch alle Zieldimensionen der linearen Transformationen 
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout_rate)] # Bildung der Schicht mit linearer Transformation(Inputgröße, Outputgröße) und Wahl der Aktivierungsfunktion
            prev = h # Outputdimension der vorherigen Sicht wird als Input für die nächste gesetzt
        self.net = nn.Sequential(*layers) # Sequential baut nun die Schichten als sequenzielle Pipeline zusammen, damit man beim Forwardpass einfach nurnoch den Input mit self.net(x) aufrufen muss und es durch Netz propagiert wird anstatt selbst den Input durch die einzelnen Schichten jagen zu müssen
        self.mu = nn.Linear(prev, latent_dim) # Nach der Featureextraktion durch das Netz wird nun eine eigene lineare Transformation genutzt um den Mittelwert vorherzusagen. Hier ist die Zieloutputdimension die festgelegte Dimension des Latenten Raums.
        self.logvar = nn.Linear(prev, latent_dim) # Gleiches geht für die Varianz, da Varianzen jedoch immer positiv sein müssen, aber eine lineare Transformation auch negative Zahlen ausspucken kann wird immer die logvar verwendet. Das Netz lernt das dann implizit weil im Reparametrisierungstrick die logvar in eine Standardabweichung umgewandelt wird zum Samples ziehen.

    def forward(self, x):
        h = self.net(x) # Hier wird dann einfach nur noch der Input durch das Netz geschoben, weil net ja die ganze Netsequenz enthält und der h-Vektor enthält dann die extrahierten Features, damit man sie in einer finalen auf das Target ausgerichteten Transformation verwenden kann
        return self.mu(h), self.logvar(h) # Nutzt die vorher initierte lineare Transformation um einen Mittelwertvektor und Logvar Vektor für das Sampling zu predicten. 


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, dropout_rate):
        super().__init__() #Wieder der Konstruktoraufruf für das nn.Module zur Vererbung und initialisierung wichtiger Parts
        layers = [] # Gleich wie Encoder
        prev = latent_dim # Bekommt ein Sample aus dem Latenten Raum als Input 
        for h in hidden_dims: # Hidden Dims ist genau die umgedrehte Reihenfolge wie vom Encoder
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),nn.SiLU(), nn.Dropout(dropout_rate)] #Gleich wie Encoder
            prev = h # Gleich wie Encoder
        layers += [nn.Linear(prev, output_dim)] # Zur finalen Rekonstruktion der Firma wird auch einfach eine lineare Transformation mit Zieldimension Anzahl der Features verwendet um den Input zu rekonstruieren
        self.net = nn.Sequential(*layers) # Decoder wird in einem Sequential Objekt als Netz gespeichert

    def forward(self, z):
        return self.net(z) # Schickt das Latente Embedding einmal durch alle Decoder Schichten und Spuckt die Rekonstruktion dann wieder aus. Hier wird direkt der Skalierte Input und nicht die Embeddings rekonstruiert. 

# Hier wird nun der Gesamte VAE mit den Feature Embeddings, dem Encoder, Decoder und der Reparametrisierung zusammengebaut.
class MissingValueVAE(nn.Module):
    def __init__(self, num_features, emb_dim=2, hidden_dims=(512, 256, 128, 64), latent_dim=12, dropout_rate=0.1): # Übergabe der Parameter Anzahl der Features, Dimension des Input Embeddings, die Dimensionen der Encoder/Decoder und die Dimension des Latenten Raums
        super().__init__() # Initialisierung der Mutterklasse nn.Module
        self.feature_embedding = FeatureEmbedding(num_features, emb_dim) #Erstellt ein Objekt, was die Linearen Transformationen zur Erstellung der Feature Embeddings enthält
        input_dim = num_features * emb_dim # Berechnung der Inputdimension für den Encoder, da die Input Embeddings geflattet und aufgereiht als Input übergeben werden.
        self.encoder = Encoder(input_dim, list(hidden_dims), latent_dim, dropout_rate) #Initialisiert den Encoder
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), num_features, dropout_rate) # Initialisiert den Decoder
# Die Reparametrisierung wird benötigt, weil man eine Zufallsprozess wie das samplen nicht ableiten kann, weil da ja keine Variablen durchgereicht werden können. Deshalb wird das Sampling als mü + eps * Std dargestellt, somit hat man eine ableitbare Form und der Zufallsprozess steckt in einer Konstanten eps, damit es differenzierbar bleibt.
    def reparameterize(self, mu, logvar): # Ohne diese Reparametrisierung und das damit einhergehende Sampling wäre der VAE nur ein normaler Autoencoder. Durch das Rauschen wird der Embeddingraum gefüllt und ermöglicht es so auch neue Daten zu generieren
        std = torch.exp(0.5 * logvar) # Umwandlung der log Varianz in eine Standardabweichung, damit mit Hilfe dieser ein Input für den Decoder berechnet werden kann. 
        eps = torch.randn_like(std) # Ziehung eines zufälligen eps damit ein Vektor 
        return mu + eps * std # Hier wird dann der tatsächliche Input für den Decoder berechnet. 

    def forward(self, x, mask):
        emb = self.feature_embedding(x, mask) #Hier wird dann die Feature Embedding Klasse verwendet um aus den Inputdaten mit Hilfe der Maske Input Embeddings zu machen. Fehlende Werte werden einfach durch das Missing Embedding ersetzt
        mu, logvar = self.encoder(emb) # Die Input Embeddings werden durch den Encoder geschoben, welcher einen Mittelwertvektor und einen Logvar Vektor zurückgeben
        z = self.reparameterize(mu, logvar) #Durch die Reparametrisierung wird mit Hilfe von Mü und Std ein Sample gezogen, was dann der Input für den Decoder ist.
        x_hat = self.decoder(z) # z wird durch den Decoder geschoben und gibt einen Rekonstruierten Input zurück 
        return x_hat, mu, logvar # Die Rekonstruktion, der Mittelwert, sowie die Logvar werden zurückgegeben um den Loss zu berechnen.


def masked_vae_loss(x, x_hat, mask, mu, logvar, beta=2.0, free_bits=0.0):
    # Reconstruction: nur über beobachtete Zellen (mask == 1)
    sq_error = (x - x_hat) ** 2 * mask # Hier wird der Squared Error berechnet also die quadrierte Differenz zwischen Rekonstruktion und tatsächlichem Input für jedes Element im Vektor. Anschließend wird die Maske multiplziert um den Loss Werte die durch ein Missing Embedding aufgefüllt wurden auf 0 zu setzen.
    recon_loss = sq_error.sum() / mask.sum().clamp(min=1.0) # Hier wird dann aus dem Squared Error der MSE weil die Summe der einzelnen Fehler durch die Anzahl der tatsächlichen Beaobachtungen geteilt wird. Das clamp verhinder eine Division durch 0.

    #  Hier wird für jede Dimension des Latenten Raums und jedes Beispiel die KL-Div zur Standardnormalverteilung berechnet. (batch, latent_dim)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # logvar verhindert, dass eine zu kleine oder negative Varianz entsteht, da sie für kleine Werte gegen -inf geht. -mü^2 ist am kleinsten wenn mü=0, -exp(logvar) sorgt dafür das zu große Varianzen massiv bestraft werden. +1 ist ein Ausgleich, damit Loss genau 0 wenn mü=0 und std=1  
    if free_bits > 0:
        # "Free bits": jede Dimension darf mindestens so viel KL "kostenlos" haben,
        # bevor der Gradient sie weiter Richtung 0 drückt -> verhindert Posterior Collapse
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits) # Die KL-Divergenz pro Dimension kann nicht unter die Free Bits Grenze gehen, sonst würde die KL-Divergenz immer richtung 0 und somit einfach immer Mittelwert 0 vorhersagen, dann wäre eine Rekonstruktion nicht mehr möglich.
    kl = kl_per_dim.sum(dim=1).mean()

    return recon_loss + beta * kl, recon_loss.item(), kl.item() # Hier werden als erstes der Tensor mit dem gesamten Rechengraphen übergeben, damit der Gradient berechnet werden kann. Mit der Item() funktion wird ein einzelnes Skalar als Loss zum Logging übergeben. Da der Rechengraph wo an den Tensoren hängt sehr speicherintensiv ist.

def relative_error(x_scaled, x_hat_scaled, mask, scaler, eps=1.0):
    """
    Berechnet den relativen Fehler stabil gegen Nullwerte und Exponenten-Explosionen.
    """
    device = x_scaled.device
    center = torch.tensor(scaler.center_, dtype=torch.float32, device=device)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    
    # 1. Zurück in den echten Wertebereich
    # Extreme Werte vor dem sinh clampen, um torch.sinh() vor 'Inf' zu schützen
    x_orig = torch.clamp((x_scaled * scale) + center, min=-80.0, max=80.0)
    x_hat_orig = torch.clamp((x_hat_scaled * scale) + center, min=-80.0, max=80.0)
    
    x_real = torch.sinh(x_orig)
    x_hat_real = torch.sinh(x_hat_orig)
    
    # 2. Absoluter Fehler
    abs_err = torch.abs(x_real - x_hat_real)
    
    # 3. Stabilisierter Nennwert: 
    # SMAPE-Korrektur oder eps=1.0 (damit 1 Euro Abweichung bei 0 Euro nicht explodiert)
    denominator = torch.abs(x_real) + torch.abs(x_hat_real) + eps
    
    # Relativer Fehler (SMAPE-basiert: skaliert sauber zwischen 0% und 200%)
    rel_err = (2.0 * abs_err) / denominator
    
    # 4. Nur vorhandene Maskenwerte berücksichtigen
    return rel_err * mask 
    

def train_vae(
    X_train, M_train, X_val, M_val,
    emb_dim=2, hidden_dims=(512, 256, 128, 64), latent_dim=8,
    batch_size=256, lr=1e-3, epochs=100, beta=1.0, device=None,
    print_every=5, eval_every=10, feature_names=None,
    kl_warmup_epochs=0, free_bits=0.0, scaler=scaler,
):
    """
    eval_every: alle wie viele Epochen zusätzlich der Imputations-Check
        (evaluate_imputation_quality) auf den Val-Daten mitläuft und ausgegeben
        wird. 0/None deaktiviert das (spart etwas Zeit bei sehr großen Daten).
    kl_warmup_epochs: linear von beta=0 auf beta=<beta> über die ersten N Epochen
        hochfahren, statt von Anfang an den vollen KL-Druck anzuwenden. Hilft
        gegen Posterior Collapse (Encoder ignoriert den Input, KL fällt sofort
        auf ~0). 0 = kein Warmup.
    free_bits: Mindest-KL pro Latent-Dimension (z.B. 0.5), unterhalb dessen kein
        Gradient mehr Richtung "noch kleiner" wirkt. Zusätzliche Absicherung
        gegen Posterior Collapse, kombinierbar mit kl_warmup_epochs.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PanelDataset(X_train, M_train)
    val_ds = PanelDataset(X_val, M_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MissingValueVAE(
        num_features=X_train.shape[1], emb_dim=emb_dim,
        hidden_dims=hidden_dims, latent_dim=latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_recon": [], "train_kl": [],
        "val_loss": [], "val_recon": [], "val_kl": [],
        "val_imputation_mse": [],  # Liste von (epoch, mse)
        "val_mape": [],
        "beta": [],
    }

    for epoch in range(1, epochs + 1):
        current_beta = beta * min(1.0, epoch / kl_warmup_epochs) if kl_warmup_epochs > 0 else beta
        history["beta"].append(current_beta)

        model.train()
        train_loss = train_recon = train_kl = 0.0
        for x, m in train_loader:
            x, m = x.to(device), m.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x, m)
            loss, recon, kl = masked_vae_loss(x, x_hat, m, mu, logvar, beta=current_beta, free_bits=free_bits)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_recon += recon * x.size(0)
            train_kl += kl * x.size(0)
        n_train = len(train_ds)
        train_loss /= n_train
        train_recon /= n_train
        train_kl /= n_train

        model.eval()
        val_loss = val_recon = val_kl = 0.0
        val_rel_err_sum = 0.0
        val_valid_points = 0
        with torch.no_grad():
            for x, m in val_loader:
                x, m = x.to(device), m.to(device)
                x_hat, mu, logvar = model(x, m)
                loss, recon, kl = masked_vae_loss(x, x_hat, m, mu, logvar, beta=current_beta, free_bits=free_bits)
                # Relativer Fehler berechnen & akkumulieren
                rel_err_matrix = relative_error(x, x_hat, m, scaler=scaler)
                val_rel_err_sum += rel_err_matrix.sum().item()
                val_valid_points += m.sum().item()

                val_loss += loss.item() * x.size(0)
                val_recon += recon * x.size(0)
                val_kl += kl * x.size(0)
        n_val = len(val_ds)
        val_loss /= n_val
        val_recon /= n_val
        val_kl /= n_val
        val_mape = val_rel_err_sum / max(val_valid_points, 1)

        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        if print_every and (epoch % print_every == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d} | "
                f"train: loss={train_loss:.4f} recon={train_recon:.4f} kl={train_kl:.4f} | "
                f"val: loss={val_loss:.4f} recon={val_recon:.4f} kl={val_kl:.4f}"
                f"val_rel_err={val_mape:.2%}"
            )

        if eval_every and epoch % eval_every == 0:
            imp_mse, _, _ = evaluate_imputation_quality(model, X_val, M_val, device=device)
            history["val_imputation_mse"].append((epoch, imp_mse))
            print(f"           -> Imputation-Check (val, künstlich versteckt): MSE={imp_mse:.4f}")
            

    print_final_summary(model, X_val, M_val, feature_names=feature_names, device=device)

    return model, history


def print_final_summary(model, X_val, M_val, feature_names=None, device=None, top_n=10):
    """Druckt eine Zusammenfassung aller Kernmetriken nach Trainingsende."""
    res = evaluate_reconstruction(model, X_val, M_val, device=device, feature_names=feature_names)
    print("\n=== Finale Evaluationsmetriken (Val-Set) ===")
    print(f"Overall Reconstruction-MSE (beobachtete Zellen): {res['overall_mse']:.4f}")

    if feature_names is not None:
        sorted_items = sorted(res["per_feature_mse"].items(), key=lambda kv: kv[1], reverse=True)
        print(f"Schlechteste {top_n} Features (höchster MSE):")
        for name, mse in sorted_items[:top_n]:
            print(f"  {name}: {mse:.4f}")

    imp_mse, true_vals, pred_vals = evaluate_imputation_quality(model, X_val, M_val, device=device)
    print(f"Imputation-Check MSE (künstlich versteckte, aber bekannte Werte): {imp_mse:.4f}")
    print("=============================================\n")


@torch.no_grad()
def evaluate_reconstruction(model, X, M, device=None, feature_names=None):
    """
    Maskierter Reconstruction-Error (MSE) auf tatsächlich BEOBACHTETEN Werten.
    Das ist die einzige Stelle, an der man direkt gegen Ground Truth prüfen kann -
    für echte Missing-Werte gibt es ja keinen wahren Wert zum Vergleichen.
    Gibt overall MSE + MSE pro Feature zurück (zeigt, welche Items gut/schlecht
    modelliert werden).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    x = torch.as_tensor(X, dtype=torch.float32).to(device)
    m = torch.as_tensor(M, dtype=torch.float32).to(device)

    x_hat, mu, logvar = model(x, m)
    sq_error = (x - x_hat) ** 2 * m

    overall_mse = (sq_error.sum() / m.sum().clamp(min=1.0)).item()
    per_feature_mse = (sq_error.sum(dim=0) / m.sum(dim=0).clamp(min=1.0)).cpu().numpy()

    result = {"overall_mse": overall_mse, "per_feature_mse": per_feature_mse}
    if feature_names is not None:
        result["per_feature_mse"] = dict(zip(feature_names, per_feature_mse))
    return result


@torch.no_grad()
def get_latent_embeddings(model, X, M, device=None):
    """
    Deterministische Embeddings (mu, nicht das gesampelte z) pro Zeile -
    das ist das, was du später z.B. als Node-Features für den GAT nehmen würdest.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    x = torch.as_tensor(X, dtype=torch.float32).to(device)
    m = torch.as_tensor(M, dtype=torch.float32).to(device)

    emb = model.feature_embedding(x, m)
    mu, _ = model.encoder(emb)
    return mu.cpu().numpy()


def evaluate_imputation_quality(model, X, M, device=None, mask_fraction=0.1, seed=0):
    """
    Versteckt zusätzlich einen Teil der ohnehin beobachteten Werte künstlich und
    prüft, wie gut das Modell genau diese (bekannten!) Werte rekonstruiert.
    Das ist der einzige Weg, echte Imputationsqualität mit Ground Truth zu messen -
    denn bei "natürlich" fehlenden Werten weißt du nie, was der wahre Wert war.
    """
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    observed_idx = np.argwhere(M == 1)
    n_to_hide = int(len(observed_idx) * mask_fraction)
    hide_idx = observed_idx[rng.choice(len(observed_idx), n_to_hide, replace=False)]

    M_eval = M.copy()
    M_eval[hide_idx[:, 0], hide_idx[:, 1]] = 0  # zusätzlich verstecken (Modell sieht diese nicht)

    model.eval().to(device)
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32).to(device)
        m_eval = torch.as_tensor(M_eval, dtype=torch.float32).to(device)
        x_hat, _, _ = model(x, m_eval)

    true_vals = X[hide_idx[:, 0], hide_idx[:, 1]]
    pred_vals = x_hat.cpu().numpy()[hide_idx[:, 0], hide_idx[:, 1]]
    mse = float(np.mean((true_vals - pred_vals) ** 2))
    return mse, true_vals, pred_vals

