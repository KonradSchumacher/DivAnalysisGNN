import pandas as pd
import numpy as np
from typing import Optional
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib

class DataPrep:
    def data_split(data: pd.DataFrame, train_size, val_size, test_size, test_cutoff_date: str, random_state):
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError(f"Die Summe aus train_size ({train_size}), val_size ({val_size}) und test_size ({test_size}) muss genau 1.0 ergeben.")
        # 1. Daten auf Stichtag begrenzen (Kein Blick in die Zukunft für den VAE!)
        df_historical = data[data['Date'] <= test_cutoff_date].copy()
        # 2. Unique Firmen (Entities) splitten
        unique_entities = df_historical['EntityCode'].unique()
        # Sicherstellen, dass die Prozentangaben genau 1.0 (100%) ergeben
        
        train_entities, temp_entities = train_test_split(unique_entities, train_size=train_size, random_state=random_state, shuffle=True)
        relative_val_size = val_size / (val_size + test_size)
        val_entities, test_entities = train_test_split(temp_entities, train_size=relative_val_size, random_state=random_state, shuffle=True)
        # 3. Train & Val (Panel-Daten: alle historischen Bilanzen <= Cutoff)
        train = df_historical[df_historical['EntityCode'].isin(train_entities)].reset_index(drop=True)
        val   = df_historical[df_historical['EntityCode'].isin(val_entities)].sort_values('Date').groupby('EntityCode').last().reset_index()
        # 4. Test (Exakt die aktuellste Bilanz pro Firma bis zum Cutoff-Datum)
        test = (df_historical[df_historical['EntityCode'].isin(test_entities)].sort_values('Date').groupby('EntityCode').last().reset_index())
        #Hier soll verhindert werden, dass ganz alte Fälle aus 2021 oderso im Test landen
        test = test[test['Date'] > '2023-01-01']
        val = val[val['Date'] > '2023-01-01']
        print(f"Train Rows: {len(train)} ({len(train_entities)} Firmen)")
        print(f"Val Rows:   {len(val)} ({len(val_entities)} Firmen)")
        print(f"Test Rows:  {len(test)} ({len(test_entities)} Firmen)")
        return train, val, test
    
    def time_split(data: pd.DataFrame, test_cutoff_date, val_months: int = 3):
        # Sicherstellen, dass Date und Cutoff echte Timestamps sind
        dates = pd.to_datetime(data['Date'])
        val_end = pd.to_datetime(test_cutoff_date)
        train_end = val_end - pd.DateOffset(months=val_months)

        train = data[dates <= train_end].copy()
        val = data[(dates > train_end) & (dates <= val_end)].copy()
        test = data[dates > val_end].copy()

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
        scaled, scaler = DataPrep.robust_scaling(arcsinh, scaler)
        scaled_array = np.nan_to_num(scaled, nan=0.0).astype(np.float32)
        return scaled_array, mask, scaler

    def full_data_prep(train_size, val_size, test_size, test_cutoff_date, random_state, gnn=True):
        data = joblib.load("../../data/03_gold/full_data.pkl")
        df = data['data']
        usable_features = data['usable_features']
        if gnn == True:
            train, val, test = DataPrep.time_split(df, test_cutoff_date)
        else:
            train, val, test = DataPrep.data_split(df, train_size, val_size, test_size, test_cutoff_date, random_state)
        test_entities = test['EntityCode']
        val_entities = val['EntityCode']
        train_entities = train['EntityCode']
        train_dates = train['Date']
        val_dates = val['Date']
        test_dates = test['Date']
        train, train_mask, scaler = DataPrep.transformation(train, usable_features, None)
        val, val_mask, _ = DataPrep.transformation(val, usable_features, scaler)
        test, test_mask, _ = DataPrep.transformation(test, usable_features, scaler)
        data_preped = {"train": train, "train_mask": train_mask, "train_entities": train_entities, "train_dates": train_dates, "val": val, "val_mask": val_mask, "val_entities": val_entities, "val_dates": val_dates,"test": test, "test_mask": test_mask, "test_entities": test_entities, "test_dates": test_dates,"usable_features": usable_features, "scaler": scaler}
        return data_preped
