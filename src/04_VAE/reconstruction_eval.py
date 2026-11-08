import torch
import numpy as np
import pandas as pd

class ReconstructionEval:
    @staticmethod
    def evaluate_reconstruction_scaled(x_scaled: torch.Tensor, x_hat_scaled: torch.Tensor, mask: torch.Tensor) -> dict:
        """Berechnet MAE und MSE direkt im skalierten Modellraum (nur für vorhandene Werte mask == 1)."""
        abs_err = torch.abs(x_scaled - x_hat_scaled) * mask
        sq_err = ((x_scaled - x_hat_scaled) ** 2) * mask

        n_valid = mask.sum()

        if n_valid == 0:
            return {"mae": 0.0, "mse": 0.0}

        mae = (abs_err.sum() / n_valid).item()
        mse = (sq_err.sum() / n_valid).item()

        return {"mae": mae, "mse": mse}

    @staticmethod
    def compute_baselines_and_compare(
        model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device
    ) -> dict:
        """Evaluiert das VAE gegen Mittelwert- und Median-Baselines auf dem Test-Dataset."""

        model.eval()
        all_x, all_masks, all_x_hat = [], [], []

        # 1. Vorhersagen auf dem Testset sammeln
        with torch.no_grad():
            for x, m in test_loader:
                x, m = x.to(device), m.to(device)
                x_hat, _, _ = model(x, m)

                all_x.append(x.cpu()) # mit .cpu() werden die Daten vom VRAM der Grafikkarte in den RAM der CPU verlegt. Die Daten können auch solange sie auf device cuda sind nicht in numpy umgewandelt werden
                all_masks.append(m.cpu())
                all_x_hat.append(x_hat.cpu())

        X = torch.cat(all_x, dim=0) #all_x ist eine Liste von Tensoren. .cat() macht daraus einen einzigen Tensor mit shape (samples, features)
        M = torch.cat(all_masks, dim=0)
        X_hat = torch.cat(all_x_hat, dim=0)

        # 2. Naive Baselines (Mittelwert & Median pro Feature) berechnen
        X_np = X.numpy()
        M_np = M.numpy()
        X_np_masked = np.where(M_np == 1, X_np, np.nan)

        feature_means = torch.tensor(
            np.nanmean(X_np_masked, axis=0), dtype=torch.float32 # berechnet den Mittelwert für jede Spalte, also jedes Feature
        )
        feature_medians = torch.tensor(
            np.nanmedian(X_np_masked, axis=0), dtype=torch.float32
        )

        X_mean_baseline = feature_means.repeat(X.shape[0], 1) # Füllt die ursprüngliche feature Matrix mit den mean Werten auf, damit so die baseline vorhersage entsteht
        X_median_baseline = feature_medians.repeat(X.shape[0], 1)

        # 3. Fehler vergleichen
        vae_eval = ReconstructionEval.evaluate_reconstruction_scaled(X, X_hat, M)
        mean_eval = ReconstructionEval.evaluate_reconstruction_scaled(X, X_mean_baseline, M)
        median_eval = ReconstructionEval.evaluate_reconstruction_scaled(X, X_median_baseline, M)

        print("\n=== REKONSTRUKTIONS-EVALUATION (Skalierter Raum) ===")
        print(f"VAE Modell      -> MAE: {vae_eval['mae']:.4f} | MSE: {vae_eval['mse']:.4f}")
        print(f"Mittelwert-Base -> MAE: {mean_eval['mae']:.4f} | MSE: {mean_eval['mse']:.4f}")
        print(f"Median-Base     -> MAE: {median_eval['mae']:.4f} | MSE: {median_eval['mse']:.4f}")

        return {
            "vae": vae_eval,
            "mean_baseline": mean_eval,
            "median_baseline": median_eval,
        }
    @staticmethod
    def relative_error(x_scaled, x_hat_scaled, mask, scaler, eps=1.0):
        """
        Berechnet den relativen Fehler stabil gegen Nullwerte und Exponenten-Explosionen.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
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

    @staticmethod
    def print_final_summary(model, X_val, M_val, feature_names=None, device=None, top_n=10):
        """Druckt eine Zusammenfassung aller Kernmetriken nach Trainingsende."""
        res = ReconstructionEval.evaluate_reconstruction(model, X_val, M_val, device=device, feature_names=feature_names)
        print("\n=== Finale Evaluationsmetriken (Val-Set) ===")
        print(f"Overall Reconstruction-MSE (beobachtete Zellen): {res['overall_mse']:.4f}")

        if feature_names is not None:
            sorted_items = sorted(res["per_feature_mse"].items(), key=lambda kv: kv[1], reverse=True)
            print(f"Schlechteste {top_n} Features (höchster MSE):")
            for name, mse in sorted_items[:top_n]:
                print(f"  {name}: {mse:.4f}")

        imp_mse, imp_mae, nmae_std, true_vals, pred_vals = ImputationEval.evaluate_imputation_quality(model, X_val, M_val, device=device)
        print(f"Imputation-Check MSE (künstlich versteckte, aber bekannte Werte): {imp_mse:.4f}")
        print(f"Imputation-Check MAE (künstlich versteckte, aber bekannte Werte): {imp_mae:.4f}")
        print(f"Imputation-Check relativen MAE zur Standardabweichung: {nmae_std:.4f}")
        print("=============================================\n")
    
    def inverse_transform_tensor(
        x_scaled: torch.Tensor, scaler
    ) -> torch.Tensor:
        """
        Transformiert skalierte Tensoren zurück in den Originalraum (Euro/Einheiten).
        Rückrichtung: StandardScaler^(-1) -> sinh()
        """
        device = x_scaled.device
        center = torch.tensor(
            scaler.center_, dtype=torch.float32, device=device
        )
        scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

        # 1. StandardScaler rückgängig machen
        x_unscaled = (x_scaled * scale) + center

        # 2. Arcsinh-Transformation umkehren (sinh)
        # Numerische Absicherung gegen extrem große Werte (Überlauf verhindern)
        x_clamped = torch.clamp(x_unscaled, min=-80.0, max=80.0)
        return torch.sinh(x_clamped)

    @staticmethod
    def evaluate_reconstruction_original_space(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        scaler,
        feature_names: list,
        device: torch.device,
    ) -> pd.DataFrame:
        """Berechnet den absoluten Fehler im Originalraum (Euro) pro Feature und vergleicht ihn mit dem Median."""
        model.eval()

        all_x, all_masks, all_x_hat = [], [], []

        with torch.no_grad():
            for x, m in loader:
                x, m = x.to(device), m.to(device)
                x_hat, _, _ = model(x, m)

                all_x.append(x)
                all_masks.append(m)
                all_x_hat.append(x_hat)

        X_scaled = torch.cat(all_x, dim=0)
        M = torch.cat(all_masks, dim=0).cpu().numpy()
        X_hat_scaled = torch.cat(all_x_hat, dim=0)

        # In den Originalraum (Euro) zurücktransformieren
        X_real = (
            ReconstructionEval.inverse_transform_tensor(X_scaled, scaler)
            .cpu()
            .numpy()
        )
        X_hat_real = (
            ReconstructionEval.inverse_transform_tensor(X_hat_scaled, scaler)
            .cpu()
            .numpy()
        )

        # Naive Baseline: Feature-Median im Originalraum über vorhandene Werte (M == 1)
        X_real_masked = np.where(M == 1, X_real, np.nan)
        feature_medians_real = np.nanmedian(X_real_masked, axis=0)

        results = []

        for j, col_name in enumerate(feature_names):
            valid_mask = M[:, j] == 1
            if not np.any(valid_mask):
                continue

            real_vals = X_real[valid_mask, j]
            pred_vals = X_hat_real[valid_mask, j]
            base_vals = feature_medians_real[j]

            # Absolute Fehler im Originalraum
            abs_err_vae = np.abs(real_vals - pred_vals)
            abs_err_base = np.abs(real_vals - base_vals)

            results.append(
                {
                    "Feature": col_name,
                    "VAE MedAE (€)": np.median(abs_err_vae),
                    "Base MedAE (€)": np.median(abs_err_base),
                    "VAE MAE (€)": np.mean(abs_err_vae),
                    "Base MAE (€)": np.mean(abs_err_base),
                    "Anzahl Werte": np.sum(valid_mask),
                }
            )

        df_res = pd.DataFrame(results)

        # Formatierung/Print-Ausgabe
        print("\n=== REKONSTRUKTIONS-EVALUATION PRO FEATURE (Originalraum €) ===")
        #print(df_res[["Feature", "VAE MedAE (€)", "Base MedAE (€)", "VAE MAE (€)", "Base MAE (€)",]].to_string(index=False, float_format=lambda x: f"{x:,.2f}".replace(",", " "),))

        return df_res
    @staticmethod
    def analyze_kl_per_dimension(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> pd.DataFrame:
        """
        Berechnet die mittlere KL-Divergenz pro latenter Dimension über den gesamten DataLoader.
        
        Hilft bei der Diagnose von Posterior Collapse und der Bestimmung der 
        effektiven intrinsischen Dimensionalität des Modells.
        """
        model.eval()
        kl_per_dim_batches = []

        with torch.no_grad():
            for x, m in loader:
                x, m = x.to(device), m.to(device)
                
                # Encoder-Pass: Erwartet mu und logvar (oder log_var)
                # Falls dein Model-Forward x_hat, mu, logvar zurückgibt:
                _, mu, logvar = model(x, m)

                # Analytische KL-Divergenz pro Sample und pro Dimension:
                # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
                kl_elementwise = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                
                # Mittelwert über den Batch (Batch-Dimension = 0)
                kl_batch_mean = kl_elementwise.mean(dim=0).cpu().numpy()
                kl_per_dim_batches.append(kl_batch_mean)

        # Mittelwert über alle Batches hinweg berechnen
        mean_kl_per_dim = np.mean(kl_per_dim_batches, axis=0)
        
        # Als übersichtliches DataFrame strukturieren
        df_kl = pd.DataFrame({
            "Dimension": [f"z_{i}" for i in range(len(mean_kl_per_dim))],
            "KL_Divergence": mean_kl_per_dim,
            "Status": ["Aktiv" if kl > 0.05 else "Inaktiv (Kollabiert)" for kl in mean_kl_per_dim]
        }).sort_values(by="KL_Divergence", ascending=False).reset_index(drop=True)

        # Zusammenfassung im Terminal ausgeben
        active_dims = (df_kl["KL_Divergence"] > 0.05).sum()
        total_dims = len(mean_kl_per_dim)
        
        #print(f"\n=== LATENT SPACE KL-ANALYSE ===")
        #print(f"Aktive Dimensionen: {active_dims} / {total_dims} (Schwellenwert KL > 0.05)")
        #print(df_kl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        return df_kl

    @torch.no_grad()
    def evaluate_reconstruction(model, X, M, device=None, feature_names=None):
   
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)

        x = torch.as_tensor(X, dtype=torch.float32).to(device)
        m = torch.as_tensor(M, dtype=torch.float32).to(device)

        x_hat, mu, logvar = model(x, m)
        sq_error = (x - x_hat) ** 2 * m
        abs_err = torch.abs(x - x_hat) * m

        overall_mse = (sq_error.sum() / m.sum().clamp(min=1.0)).item()
        per_feature_mse = (sq_error.sum(dim=0) / m.sum(dim=0).clamp(min=1.0)).cpu().numpy()
        overall_mae = (abs_err.sum() / m.sum().clamp(min=1.0)).item()
        per_feature_mae = (abs_err.sum(dim=0) / m.sum(dim=0).clamp(min=1.0)).cpu().numpy()

        result = {"overall_mse": overall_mse, "per_feature_mse": per_feature_mse, "overall_mae": overall_mae, "per_feature_mae": per_feature_mae}
        if feature_names is not None:
            per_feature_mse = dict(zip(feature_names, per_feature_mse))
            per_feature_mae = dict(zip(feature_names, per_feature_mae))
        return {
            "x_hat": x_hat,
            "overall_mse": overall_mse,
            "per_feature_mse": per_feature_mse,
            "overall_mae": overall_mae,
            "per_feature_mae": per_feature_mae,
         }

class ImputationEval:
    @staticmethod
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
        X_eval = X.copy()

        row_indices = hide_idx[:, 0]
        col_indices = hide_idx[:, 1]

        M_eval[row_indices, col_indices] = 0
        X_eval[row_indices, col_indices] = 0
 
        model.eval().to(device)
        with torch.no_grad():
            x_eval = torch.as_tensor(X_eval, dtype=torch.float32).to(device)
            m_eval = torch.as_tensor(M_eval, dtype=torch.float32).to(device)
            x_hat, _, _ = model(x_eval, m_eval)

        true_vals = X[row_indices, col_indices]
        pred_vals = x_hat.cpu().numpy()[row_indices, col_indices]
        mse = float(np.mean((true_vals - pred_vals) ** 2))
        mae = float(np.mean(np.abs(true_vals - pred_vals)))
        std_per_hidden_val = np.std(X, axis=0)[col_indices]
        
        # Vermeidung von Division durch Null bei konstanten Features (z. B. std = 0)
        std_per_hidden_val = np.where(std_per_hidden_val == 0, 1e-8, std_per_hidden_val)
        
        # Mittlerer relativer Fehler im Verhältnis zur jeweiligen Feature-Streuung
        nmae_std = float(np.mean(np.abs(true_vals - pred_vals) / std_per_hidden_val))

        return mse, mae, nmae_std, true_vals, pred_vals