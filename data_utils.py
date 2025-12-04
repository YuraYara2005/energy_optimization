import pandas as pd
import numpy as np


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        low_memory=False,
        na_values="?",
        dtype=str
    )
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(
        df['Date'] + " " + df['Time'],
        format="%d/%m/%Y %H:%M:%S",
        errors='coerce'
    )

    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

    df = df[['timestamp', 'Global_active_power']]

    df = df.dropna()

    df = df.set_index('timestamp')

    df = df.resample('H').mean()
    df = df.rename(columns={'Global_active_power': 'consumption'})

    df = df.dropna()

    df['hour'] = df.index.hour

    df['price'] = 0.15 + 0.10 * np.sin(2 * np.pi * (df['hour'] - 8) / 24)
    df['price'] = df['price'].clip(lower=0.05)

    df['price_norm'] = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())

    df['baseline_cost'] = df['price'] * df['consumption']

    return df.reset_index()


def compute_cost(price: np.ndarray, x: np.ndarray) -> np.ndarray:
    return price * x


def add_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    df['is_peak_hour'] = df['hour'].isin([17, 18, 19, 20, 21]).astype(int)
    df['weekday'] = df['timestamp'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Mean consumption (kW)": [df['consumption'].mean()],
        "Max consumption (kW)": [df['consumption'].max()],
        "Min consumption (kW)": [df['consumption'].min()],
        "Mean price ($/kWh)": [df['price'].mean()],
        "Total baseline cost ($)": [df['baseline_cost'].sum()]
    })