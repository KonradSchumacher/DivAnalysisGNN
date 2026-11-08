from vae_architecture import PanelDataset, MissingValueVAE, DataLoader
from reconstruction_eval import ReconstructionEval, ImputationEval
import torch
import numpy as np
import pandas as pd
import copy

class Modeltraining:
    @staticmethod
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
    
    @staticmethod
    def train_vae(
        X_train, M_train, X_val, M_val,
        emb_dim, hidden_dims, activation_fun ,latent_dim, dropout_rate,
        batch_size, lr, epochs, beta, device,
        print_every, eval_every, feature_names,
        kl_warmup_epochs, free_bits, scaler,
        early_stopping_patience=None,        # NEU: None = kein Early Stopping, altes Verhalten
        early_stopping_min_delta=0.0,        # NEU: Mindestverbesserung, um als "besser" zu zählen
        early_stopping_metric="val_loss",   # NEU: "val_recon" oder "val_mape" - NIE "val_loss"
        early_stopping_start_epoch=None,     # NEU: None -> automatisch kl_warmup_epochs + 1
    ):
        """
        ... (bestehender Docstring unverändert) ...

        early_stopping_patience: NEU - Anzahl Epochen ohne Verbesserung, nach denen
            abgebrochen wird. None deaktiviert Early Stopping komplett.
        early_stopping_metric: NEU - "val_recon" oder "val_mape". Bewusst NICHT
            "val_loss" erlaubt, da der Gesamt-Loss durch das sich ändernde beta
            während/kurz nach dem Warmup nicht über Epochen hinweg vergleichbar ist.
        early_stopping_start_epoch: NEU - erste Epoche, ab der überhaupt auf
            Verbesserung geprüft wird. None -> automatisch kl_warmup_epochs + 1,
            damit die Übergangsphase des Warmups nicht fälschlich als "bestes
            Modell" gewertet wird.
        """
        if early_stopping_metric not in ("val_recon", "val_mape", "val_loss"):
            raise ValueError(
                "early_stopping_metric muss 'val_recon' oder 'val_mape' sein - "
                "'val_loss' ist während/kurz nach dem KL-Warmup nicht über Epochen "
                "hinweg vergleichbar, da beta sich noch ändert."
            )
        if early_stopping_start_epoch is None:
            early_stopping_start_epoch = (kl_warmup_epochs + 1) if kl_warmup_epochs > 0 else 1

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        train_ds = PanelDataset(X_train, M_train)
        val_ds = PanelDataset(X_val, M_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = MissingValueVAE(
            num_features=X_train.shape[1], emb_dim=emb_dim,
            hidden_dims=hidden_dims, latent_dim=latent_dim, activation_fun=activation_fun ,dropout_rate=dropout_rate
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = {
            "train_loss": [], "train_recon": [], "train_kl": [],
            "val_loss": [], "val_recon": [], "val_kl": [],
            "val_imputation_mse": [], "val_imputation_mae": [], "val_imputation_nmae_std": [],
            "val_mape": [],
            "beta": [],
        }

        # NEU: Early-Stopping-Zustand
        best_metric = float("inf")
        best_epoch = None
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            current_beta = beta * min(1.0, epoch / kl_warmup_epochs) if kl_warmup_epochs > 0 else beta
            history["beta"].append(current_beta)

            model.train()
            train_loss = train_recon = train_kl = 0.0
            for x, m in train_loader:
                x, m = x.to(device), m.to(device)
                optimizer.zero_grad()
                x_hat, mu, logvar = model(x, m)
                loss, recon, kl = Modeltraining.masked_vae_loss(x, x_hat, m, mu, logvar, beta=current_beta, free_bits=free_bits)
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
                    loss, recon, kl = Modeltraining.masked_vae_loss(x, x_hat, m, mu, logvar, beta=current_beta, free_bits=free_bits)
                    rel_err_matrix = ReconstructionEval.relative_error(x, x_hat, m, scaler=scaler)
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
            history["val_mape"].append(val_mape)  # NEU: fehlte bisher

            if print_every and (epoch % print_every == 0 or epoch == 1):
                print(
                    f"Epoch {epoch:3d} | "
                    f"train: loss={train_loss:.4f} recon={train_recon:.4f} kl={train_kl:.4f} | "
                    f"val: loss={val_loss:.4f} recon={val_recon:.4f} kl={val_kl:.4f} "
                    f"val_rel_err={val_mape:.2%}"
                )

            if eval_every and epoch % eval_every == 0:
                mse, mae, nmae_std, true_vals, pred_vals = ImputationEval.evaluate_imputation_quality(model, X_val, M_val, device=device)
                history["val_imputation_mse"].append((epoch, mse))
                history["val_imputation_mae"].append((epoch, mae))
                history["val_imputation_nmae_std"].append((epoch, nmae_std))
                #print(f"           -> Imputation-Check (val, künstlich versteckt): MSE={mse:.4f} MAE={mae:.4f} NMAE_STD={nmae_std:.4f}")

            # --- NEU: Early Stopping -------------------------------------------
            if early_stopping_patience is not None and epoch >= early_stopping_start_epoch:
                current_metric = val_recon if early_stopping_metric == "val_recon" else val_loss
                if current_metric < best_metric - early_stopping_min_delta:
                    best_metric = current_metric
                    best_epoch = epoch
                    best_state_dict = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"\nEarly Stopping bei Epoche {epoch}: seit {early_stopping_patience} "
                        f"Epochen keine Verbesserung in {early_stopping_metric} "
                        f"(bestes Ergebnis: {best_metric:.4f} bei Epoche {best_epoch})."
                    )
                    break

        # NEU: bestes Modell wiederherstellen, BEVOR die finale Evaluation läuft -
        # sonst wertet ihr unten das letzte (ggf. schon wieder leicht schlechtere
        # oder frühzeitig abgebrochene) statt das beste Modell aus.
        if best_state_dict is not None and best_epoch is not None:
            model.load_state_dict(best_state_dict)
            best_idx = best_epoch - 1
            #print(f"Bestes Modell wiederhergestellt: Epoche {best_epoch}, {early_stopping_metric}={best_metric:.4f}")
        else:
            # Fallback: Early Stopping war deaktiviert oder greift nicht -> nimm die letzte Epoche
            best_idx = -1
            best_epoch = epochs

        #print("\n=== FINALE EVALUATION GEGEN BASELINES ===")
        #ReconstructionEval.compute_baselines_and_compare(model=model, test_loader=val_loader, device=device)
        #print("\n=== FINALE EVALUATION (Originalraum in Euro) ===")
        #df_euro_results = ReconstructionEval.evaluate_reconstruction_original_space(model=model, loader=val_loader, scaler=scaler, feature_names=feature_names, device=device)
        #df_kl = ReconstructionEval.analyze_kl_per_dimension(model=model, loader=val_loader, device=device)
        #ReconstructionEval.print_final_summary(model, X_val, M_val, feature_names=feature_names, device=device)


        final_metrics = {
            "best_epoch": best_epoch,
            "train_loss": history["train_loss"][best_idx],
            "train_recon": history["train_recon"][best_idx],
            "train_kl": history["train_kl"][best_idx],
            "val_loss": history["val_loss"][best_idx],
            "val_recon": history["val_recon"][best_idx],
            "val_kl": history["val_kl"][best_idx],
            "val_mape": history["val_mape"][best_idx],
            "beta": history["beta"][best_idx],
        }
        return model, final_metrics, history

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
