import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
# Gezielt genau diese KMeans-Warnung stummschalten
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*KMeans is known to have a memory leak on Windows.*"
)
import pandas as pd
import yfinance as yf
import numpy as np
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

###########################################################################################################################################
class PriceData:
    @staticmethod
    def fetch_close_prices(ticker_list, start, end):
        """Lädt wöchentliche Schlusskurse für eine Liste von Tickern herunter."""
        df = yf.download(
            ticker_list,
            interval='1d',
            start=start,     
            end=end,
            auto_adjust=True,
            keepna=True,
            prepost=False,
            progress=False,
        )

        # MultiIndex / Single-Ticker Absicherung für Spaltenzugriff
        if isinstance(df.columns, pd.MultiIndex):
            close_prices = df["Close"].copy()
        else:
            close_prices = df[["Close"]].copy()

        return close_prices
    @staticmethod
    def to_weekly_close(daily_close, week_anchor="W-FRI"):
        """Aggregiert täglich -> wöchentlich. last() nimmt den letzten gültigen Kurs der Woche,
        ist die ganze Woche NaN, bleibt NaN (kein Forward-Fill => keine künstlichen 0%-Renditen)."""
        weekly = daily_close.resample(week_anchor).last()
        # Unvollständige letzte Woche verwerfen (Toleranz 1 Tag für Feiertage am Freitag)
        gap_days = (weekly.index[-1] - daily_close.index.max()).days
        if gap_days >= 2:
            weekly = weekly.iloc[:-1]
        return weekly
    @staticmethod
    def to_monthly_close(daily_close, month_anchor="M"):
        """Aggregiert täglich -> monatlich. last() nimmt den letzten gültigen Kurs des Monats,
        ist der ganze Monat NaN, bleibt NaN (kein Forward-Fill => keine künstlichen 0%-Renditen)."""
        #Nimmt den letzten verfügbaren Kurs im Monat. Wenn ganzer Monat NaN dann auch da NaN
        monthly = daily_close.resample(month_anchor).last()
        # Unvollständigen laufenden Monat verwerfen (Toleranz 4 Tage für Feiertage/Wochenenden am Monatsende) Nur für letzten Monat in der Zeitreihe
        gap_days = (monthly.index[-1] - daily_close.index.max()).days
        if gap_days >= 5:
            monthly = monthly.iloc[:-1]
        return monthly
    @staticmethod
    def prepare_weekly_returns(weekly_close, max_nan_pct, log_returns=False):
        nan_share = weekly_close.isna().mean()
        valid = nan_share[nan_share <= max_nan_pct].index
        px = weekly_close[valid].where(weekly_close[valid] > 0)

        if log_returns:
            returns = np.log(px).diff()
        else:
            returns = px.pct_change(fill_method=None)   # kein implizites ffill

        returns = returns.replace([np.inf, -np.inf], np.nan)
        valid_dates = returns.dropna(how="all").index
        returns = returns.loc[valid_dates]
        px = px.loc[valid_dates]
        print(f"Eingangs-Ticker:          {weekly_close.shape[1]}")
        print(f"Nutzbare Ticker nach {max_nan_pct} Filterung: {returns.shape[1]}")
        return (returns, px)

    @classmethod
    def execute(cls, tickers, start, end, max_nan_pct, log_returns=False):
        if isinstance(tickers, str):
            ticker_list = [tickers]
        elif hasattr(tickers, "tolist"):
            ticker_list = tickers.tolist()
        else:
            ticker_list = list(tickers)

        daily = cls.fetch_close_prices(ticker_list, start, end)
        weekly = cls.to_weekly_close(daily)
        monthly = cls.to_monthly_close(daily)
        return cls.prepare_weekly_returns(weekly, max_nan_pct), cls.prepare_weekly_returns(monthly, max_nan_pct)
    
###########################################################################################################################################
class DistanceMetrics:
    def cosine_sim(embeddings, entitycodes):
        sim_matrix = cosine_similarity(embeddings)
        df_sim = pd.DataFrame(sim_matrix, index=entitycodes, columns=entitycodes)
        return df_sim

    def euclidean_dist(embeddings, entitycodes):
        euc_matrix = euclidean_distances(embeddings)
        df_euc = pd.DataFrame(euc_matrix, index=entitycodes, columns=entitycodes)
        return df_euc

###########################################################################################################################################
class PortfolioSelection:
    @staticmethod
    def select_farthest_portfolio_rand(
        dist_matrix: pd.DataFrame, 
        available_companies: list, 
        n_assets: int = 15,
        random_state: Optional[int] = None
    ) -> list:
        """Wählt n_assets Firmen aus, die im Latent Space maximal zueinander beabstandet sind
        (Farthest Point Sampling / Greedy Max-Min Diversity).

        Wenn random_state gesetzt ist, wird die Startfirma zufällig gewählt.
        """
        # 1. Schnittmenge bilden
        valid_companies = [
            c for c in dist_matrix.index if c in set(available_companies)
        ]

        if len(valid_companies) < n_assets:
            raise ValueError(
                f"Nur {len(valid_companies)} übereinstimmende Unternehmen verfügbar, "
                f"aber {n_assets} angefordert."
            )

        # Distanzmatrix beschränken
        sub_dist_df = dist_matrix.loc[valid_companies, valid_companies]
        matrix = sub_dist_df.values
        companies = sub_dist_df.index.tolist()

        # Random Generator initialisieren
        rng = np.random.default_rng(random_state)

        # 2. Start-Strategie festlegen:
        if random_state is not None:
            # A) Zufällige erste Firma auswählen
            first_idx = rng.choice(len(companies))
            
            # B) Die zweite Firma ist diejenige mit der größten Distanz zur ersten
            second_idx = np.argmax(matrix[first_idx])
            selected_indices = [first_idx, second_idx]
        else:
            # C) Determinastisch: Das Paar mit der absolut größten Distanz
            first_pair_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
            selected_indices = [first_pair_idx[0], first_pair_idx[1]]

        # 3. Schrittweise weitere Firmen hinzufügen (Max-Min Kriterium)
        while len(selected_indices) < n_assets:
            # Distanzen aller Firmen zu den bereits ausgewählten Firmen
            sub_matrix = matrix[:, selected_indices]

            # Für jede Firma den Mindestabstand zur nächsten bereits gewählten Firma finden
            min_distances_to_selected = np.min(sub_matrix, axis=1)

            # Bereits gewählte Firmen maskieren (Distanzen auf -1 setzen)
            #min_distances_to_selected[selected_indices] = -1.0
            min_distances_to_selected[selected_indices] = -np.inf
            # Die Firma mit dem größten Mindestabstand auswählen
            next_idx = np.argmax(min_distances_to_selected)
            selected_indices.append(next_idx)

        return [companies[i] for i in selected_indices]
    
    def random_portfolio(num_assets, returns):
        random_portfolio = random.sample(list(returns.columns), k=num_assets)
        return random_portfolio


    @staticmethod
    def select_cluster_diverse_portfolio(
        embeddings: np.ndarray,
        entities: list,
        available_companies: list,
        n_assets: int = 10,
        random_state: Optional[int] = None,
        use_cosine: bool = False,
    ) -> list:
        """
        Wählt n_assets Firmen so, dass sie aus n_assets verschiedenen Clustern im
        Embedding-Raum kommen (KMeans, ein Cluster pro Asset) - Alternative zum
        Greedy-Max-Min-Ansatz in select_farthest_portfolio_rand.

        Wichtig: Das Clustering wird AUSSCHLIESSLICH auf den Embeddings der Firmen
        berechnet, die in available_companies vorkommen - nicht auf allen
        Embeddings. Sonst würde die Cluster-Struktur von Firmen mitgeformt, die
        ohnehin nie wählbar sind (z.B. fehlende Kursdaten), und die Cluster
        würden nicht mehr zur tatsächlich handelbaren Menge passen.

        use_cosine: L2-normiert die Embeddings vor dem Clustering (Spherical
        KMeans) - approximiert Cosine-basiertes Clustering mit Standard-KMeans.
        """
        embeddings = np.asarray(embeddings, dtype=float)
        emb_df = pd.DataFrame(embeddings, index=list(entities))

        valid_companies = [c for c in emb_df.index if c in set(available_companies)]
        if len(valid_companies) < n_assets:
            raise ValueError(
                f"Nur {len(valid_companies)} übereinstimmende Unternehmen verfügbar, "
                f"aber {n_assets} angefordert."
            )

        sub_emb = emb_df.loc[valid_companies]
        if sub_emb.index.duplicated().any():
            raise ValueError("Doppelte EntityCodes in embeddings/entities gefunden.")

        X = sub_emb.values
        if use_cosine:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Nullvektoren defensiv abfangen
            X = X / norms

        kmeans = KMeans(n_clusters=n_assets, init='k-means++', random_state=random_state, n_init=100)
        cluster_labels = kmeans.fit_predict(X)

        selected = []
        for cluster_id in range(n_assets):
            members_idx = np.where(cluster_labels == cluster_id)[0]
            if len(members_idx) == 0:
                continue  # leerer Cluster - bei KMeans selten, aber möglich
            centroid = kmeans.cluster_centers_[cluster_id]
            dists_to_centroid = np.linalg.norm(X[members_idx] - centroid, axis=1)
            medoid_local_idx = members_idx[np.argmin(dists_to_centroid)]
            selected.append(sub_emb.index[medoid_local_idx])

        if len(selected) < n_assets:
            raise ValueError(
                f"Nur {len(selected)} von {n_assets} Clustern waren nicht-leer - "
                "ungewöhnlich, ggf. n_assets relativ zur verfügbaren Firmenzahl zu hoch."
            )
        return selected
    @staticmethod
    def select_agglomerative_diverse_portfolio(
        embeddings: np.ndarray,
        entities: list,
        available_companies: list,
        n_assets: int = 10,
        random_state: Optional[int] = None,  # Für Signatur-Kompatibilität (Agglomerative ist deterministisch)
        use_cosine: bool = False,
        linkage: str = "average",
    ) -> list:
        """
        Wählt n_assets Firmen mittels Hierarchischem / Agglomerative Clustering.
        
        Wählt pro Cluster das Medoid (den repräsentativsten Datenpunkt mit dem 
        minimalen kumulierten Abstand zu allen anderen Cluster-Mitgliedern).
        
        use_cosine: Verwendet Cosine Distance als Metrik für das Hierarchische Clustering.
                    Andernfalls wird Euklidische Distanz verwendet.
        linkage: Linkage-Kriterium ('average', 'complete', 'single').
        """
        embeddings = np.asarray(embeddings, dtype=float)
        emb_df = pd.DataFrame(embeddings, index=list(entities))

        valid_companies = [c for c in emb_df.index if c in set(available_companies)]
        if len(valid_companies) < n_assets:
            raise ValueError(
                f"Nur {len(valid_companies)} übereinstimmende Unternehmen verfügbar, "
                f"aber {n_assets} angefordert."
            )

        sub_emb = emb_df.loc[valid_companies]
        if sub_emb.index.duplicated().any():
            raise ValueError("Doppelte EntityCodes in embeddings/entities gefunden.")

        X = sub_emb.values

        # Distanzmatrix berechnen
        if use_cosine:
            dist_matrix = cosine_similarity(X)
        else:
            dist_matrix = euclidean_distances(X)

        clustering = AgglomerativeClustering(
            n_clusters=n_assets,
            metric="precomputed",
            linkage=linkage,
        )
        cluster_labels = clustering.fit_predict(dist_matrix)

        selected = []
        for cluster_id in range(n_assets):
            members_idx = np.where(cluster_labels == cluster_id)[0]
            if len(members_idx) == 0:
                continue

            # Medoid bestimmen: Punkt mit minimaler Distanzsumme zu allen Cluster-Mitgliedern
            sub_dists = dist_matrix[np.ix_(members_idx, members_idx)]
            medoid_local_idx = members_idx[np.argmin(sub_dists.sum(axis=1))]
            selected.append(sub_emb.index[medoid_local_idx])

        if len(selected) < n_assets:
            raise ValueError(
                f"Nur {len(selected)} von {n_assets} Clustern waren nicht-leer - "
                "ungewöhnlich, ggf. n_assets relativ zur verfügbaren Firmenzahl zu hoch."
            )
        return selected

    @staticmethod
    def select_gmm_diverse_portfolio(
        embeddings: np.ndarray,
        entities: list,
        available_companies: list,
        n_assets: int = 10,
        random_state: Optional[int] = None,
        use_cosine: bool = False,
        covariance_type: str = "full",
    ) -> list:
        """
        Wählt n_assets Firmen mittels Gaussian Mixture Model (GMM) Clustering.
        
        Wählt pro Cluster die Firma mit der höchsten Zugehörigkeitswahrscheinlichkeit 
        (Posterior Probability) für die jeweilige Gauß-Komponente.
        
        use_cosine: L2-normiert die Embeddings vor dem Fitting, um reine Winkel-
                    Unterschiede in der Gauß-Verteilung abzubilden.
        """
        embeddings = np.asarray(embeddings, dtype=float)
        emb_df = pd.DataFrame(embeddings, index=list(entities))

        valid_companies = [c for c in emb_df.index if c in set(available_companies)]
        if len(valid_companies) < n_assets:
            raise ValueError(
                f"Nur {len(valid_companies)} übereinstimmende Unternehmen verfügbar, "
                f"aber {n_assets} angefordert."
            )

        sub_emb = emb_df.loc[valid_companies]
        if sub_emb.index.duplicated().any():
            raise ValueError("Doppelte EntityCodes in embeddings/entities gefunden.")

        X = sub_emb.values
        if use_cosine:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Nullvektoren defensiv abfangen
            X = X / norms

        gmm = GaussianMixture(
            n_components=n_assets,
            random_state=random_state,
            covariance_type=covariance_type,
            n_init=5,
        )
        cluster_labels = gmm.fit_predict(X)
        probs = gmm.predict_proba(X)  # Probabilities aller Punkte für alle Komponenten

        selected = []
        for cluster_id in range(n_assets):
            members_idx = np.where(cluster_labels == cluster_id)[0]
            if len(members_idx) == 0:
                continue

            # Wähle das Firmen-Mitglied mit der höchsten Posterior-Wahrscheinlichkeit
            best_member_local_idx = members_idx[np.argmax(probs[members_idx, cluster_id])]
            selected.append(sub_emb.index[best_member_local_idx])

        if len(selected) < n_assets:
            raise ValueError(
                f"Nur {len(selected)} von {n_assets} Clustern waren nicht-leer - "
                "ungewöhnlich, ggf. n_assets relativ zur verfügbaren Firmenzahl zu hoch."
            )
        return selected

class DiversificationAnalysis:
    def diversification_ratio(
        returns: pd.DataFrame,
        tickers: list,
    ):
        selected_tickers = list(tickers)
        actual_num_assets = len(selected_tickers)

        portfolio_returns = returns[selected_tickers]

        # Kovarianzmatrix (Σ) und Einzelvolatilitäten (σ) auf Jahresbasis (52 Wochen)
        cov_matrix_annual = portfolio_returns.cov() * 52
        volatilities_annual = portfolio_returns.std() * np.sqrt(52)

        # Gleichgewichtung (1 / N)
        weights = np.ones(actual_num_assets) / actual_num_assets

        # Gewichtiges Mittel der Einzelvolatilitäten (Zähler)
        weighted_vol_sum = np.dot(weights, volatilities_annual)

        # Portfolio-Gesamtvolatilität (Nenner)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Diversification Ratio
        diversification_ratio = weighted_vol_sum / portfolio_volatility

        return {
            "diversification_ratio": diversification_ratio,
            "portfolio_volatility_annual": portfolio_volatility,
            "weighted_avg_volatility_annual": weighted_vol_sum,
            "selected_tickers": selected_tickers,
        }