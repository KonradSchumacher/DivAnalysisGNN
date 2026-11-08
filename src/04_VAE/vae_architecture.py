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
    def __init__(self, input_dim, hidden_dims, latent_dim, activation_fun, dropout_rate):
        super().__init__() #Wieder der Konstruktoraufruf für das nn.Module zur Vererbung und initialisierung wichtiger Parts
        layers = [] # Speicher in dem die einzelnen Schichten, also jeweils die Lineare Transformation mit Inputgröße und Outpuntdimension, sowie die nicht-lineare Aktivierungsfunktion
        prev = input_dim # Hier wird die Inputdimension initialisiert. Wird dann immer auf die Zieldimension der linearen Transformation gesetzt
        for h in hidden_dims: # Durchlauf durch alle Zieldimensionen der linearen Transformationen 
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), activation_fun, nn.Dropout(dropout_rate)] # Bildung der Schicht mit linearer Transformation(Inputgröße, Outputgröße) und Wahl der Aktivierungsfunktion
            prev = h # Outputdimension der vorherigen Sicht wird als Input für die nächste gesetzt
        self.net = nn.Sequential(*layers) # Sequential baut nun die Schichten als sequenzielle Pipeline zusammen, damit man beim Forwardpass einfach nurnoch den Input mit self.net(x) aufrufen muss und es durch Netz propagiert wird anstatt selbst den Input durch die einzelnen Schichten jagen zu müssen
        self.mu = nn.Linear(prev, latent_dim) # Nach der Featureextraktion durch das Netz wird nun eine eigene lineare Transformation genutzt um den Mittelwert vorherzusagen. Hier ist die Zieloutputdimension die festgelegte Dimension des Latenten Raums.
        self.logvar = nn.Linear(prev, latent_dim) # Gleiches geht für die Varianz, da Varianzen jedoch immer positiv sein müssen, aber eine lineare Transformation auch negative Zahlen ausspucken kann wird immer die logvar verwendet. Das Netz lernt das dann implizit weil im Reparametrisierungstrick die logvar in eine Standardabweichung umgewandelt wird zum Samples ziehen.

    def forward(self, x):
        h = self.net(x) # Hier wird dann einfach nur noch der Input durch das Netz geschoben, weil net ja die ganze Netsequenz enthält und der h-Vektor enthält dann die extrahierten Features, damit man sie in einer finalen auf das Target ausgerichteten Transformation verwenden kann
        return self.mu(h), self.logvar(h) # Nutzt die vorher initierte lineare Transformation um einen Mittelwertvektor und Logvar Vektor für das Sampling zu predicten. 


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, activation_fun, dropout_rate):
        super().__init__() #Wieder der Konstruktoraufruf für das nn.Module zur Vererbung und initialisierung wichtiger Parts
        layers = [] # Gleich wie Encoder
        prev = latent_dim # Bekommt ein Sample aus dem Latenten Raum als Input 
        for h in hidden_dims: # Hidden Dims ist genau die umgedrehte Reihenfolge wie vom Encoder
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), activation_fun, nn.Dropout(dropout_rate)] #Gleich wie Encoder
            prev = h # Gleich wie Encoder
        layers += [nn.Linear(prev, output_dim)] # Zur finalen Rekonstruktion der Firma wird auch einfach eine lineare Transformation mit Zieldimension Anzahl der Features verwendet um den Input zu rekonstruieren
        self.net = nn.Sequential(*layers) # Decoder wird in einem Sequential Objekt als Netz gespeichert

    def forward(self, z):
        return self.net(z) # Schickt das Latente Embedding einmal durch alle Decoder Schichten und Spuckt die Rekonstruktion dann wieder aus. Hier wird direkt der Skalierte Input und nicht die Embeddings rekonstruiert. 

# Hier wird nun der Gesamte VAE mit den Feature Embeddings, dem Encoder, Decoder und der Reparametrisierung zusammengebaut.
class MissingValueVAE(nn.Module):
    def __init__(self, num_features, emb_dim, hidden_dims, latent_dim, activation_fun, dropout_rate): # Übergabe der Parameter Anzahl der Features, Dimension des Input Embeddings, die Dimensionen der Encoder/Decoder und die Dimension des Latenten Raums
        super().__init__() # Initialisierung der Mutterklasse nn.Module
        activation_map = {
                "elu": nn.ELU(),
                "gelu": nn.GELU(),
                "leaky_relu": nn.LeakyReLU(),
                "silu": nn.SiLU(),
            }
        activation_fun = activation_map[activation_fun]
        self.feature_embedding = FeatureEmbedding(num_features, emb_dim) #Erstellt ein Objekt, was die Linearen Transformationen zur Erstellung der Feature Embeddings enthält
        input_dim = num_features * emb_dim # Berechnung der Inputdimension für den Encoder, da die Input Embeddings geflattet und aufgereiht als Input übergeben werden.
        self.encoder = Encoder(input_dim, list(hidden_dims), latent_dim, activation_fun,dropout_rate) #Initialisiert den Encoder
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), num_features, activation_fun, dropout_rate) # Initialisiert den Decoder
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