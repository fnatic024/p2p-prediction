"""
features.py — Motor de features (versión GitHub Actions / CSV)
===============================================================
Los crons de Actions NO son puntuales: el intervalo real entre snapshots
suele oscilar entre 5 y 12 minutos. Por eso:

  1. Los features rodantes se calculan sobre una rejilla re-muestreada
     a 5 minutos (forward-fill), para que "span=6" siempre signifique
     ~30 minutos reales.
  2. Las etiquetas se asignan POR TIEMPO (primer snapshot >= t + horizonte),
     no por número de filas.

Compartido entre train.py y predict.py para evitar training/serving skew.
"""

import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = "data"
GRID = "5min"  # rejilla de re-muestreo

FEATURE_COLS = [
    # Momentum de precio (periodos de 5 min sobre la rejilla)
    "ret_1", "ret_3", "ret_6", "ret_12",
    "ema_ratio", "rsi_14",
    # Volatilidad
    "bb_width", "bb_squeeze",
    # Microestructura del libro
    "spread_pct", "imbalance", "imbalance_d1", "imbalance_ma6",
    "vol_ratio", "vwap_skew",
    # Brecha con el BCV
    "brecha", "brecha_d1", "brecha_d6",
    # Calendario (hora Venezuela)
    "hour", "dow", "is_weekend", "is_quincena", "banking_hours",
]


def load_snapshots(data_dir: str = DATA_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No hay CSVs en {data_dir}/")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df["bcv"] = pd.to_numeric(df["bcv"], errors="coerce").ffill()
    return df


def _resample(df: pd.DataFrame) -> pd.DataFrame:
    """Rejilla regular de 5 min con forward-fill (máx. 30 min de hueco)."""
    g = (df.set_index("ts")
           .resample(GRID).last()
           .ffill(limit=6))
    g = g.dropna(subset=["mid"])
    g["ts_local"] = g.index.tz_convert("America/Caracas")
    return g


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _resample(df)
    mid = out["mid"]

    for k in (1, 3, 6, 12):
        out[f"ret_{k}"] = mid.pct_change(k) * 100

    ema_fast = mid.ewm(span=6, min_periods=6).mean()
    ema_slow = mid.ewm(span=24, min_periods=24).mean()
    out["ema_ratio"] = (ema_fast / ema_slow - 1) * 100
    out["rsi_14"] = _rsi(mid)

    ma20, sd20 = mid.rolling(20).mean(), mid.rolling(20).std()
    out["bb_width"] = 4 * sd20 / ma20 * 100
    out["bb_squeeze"] = (
        out["bb_width"].rolling(100, min_periods=30)
        .apply(lambda w: (w < w.iloc[-1]).mean(), raw=False)
    )

    out["imbalance_d1"] = out["imbalance"].diff()
    out["imbalance_ma6"] = out["imbalance"].rolling(6).mean()
    out["vol_ratio"] = np.log((out["vol_bid"] + 1) / (out["vol_ask"] + 1))
    out["vwap_skew"] = ((out["vwap_bid5"] + out["vwap_ask5"]) / 2 - mid) / mid * 100

    out["brecha"] = (mid - out["bcv"]) / out["bcv"] * 100
    out["brecha_d1"] = out["brecha"].diff()
    out["brecha_d6"] = out["brecha"].diff(6)

    loc = out["ts_local"].dt
    out["hour"] = loc.hour
    out["dow"] = loc.dayofweek
    out["is_weekend"] = (loc.dayofweek >= 5).astype(int)
    out["is_quincena"] = loc.day.isin([14, 15, 16, 29, 30, 31, 1]).astype(int)
    out["banking_hours"] = ((loc.hour.between(8, 15)) &
                            (loc.dayofweek < 5)).astype(int)
    return out


def add_labels(feat: pd.DataFrame, horizon_min: int = 15,
               threshold_pct: float = 0.25,
               tolerance_min: int = 10) -> pd.DataFrame:
    """
    Etiqueta por TIEMPO real: busca el primer snapshot en
    [t + horizonte, t + horizonte + tolerancia] y compara el mid.
    """
    out = feat.copy()
    idx = out.index  # DatetimeIndex (rejilla)
    target = idx + pd.Timedelta(minutes=horizon_min)

    future = pd.merge_asof(
        pd.DataFrame({"target": target}).set_index("target"),
        out[["mid"]].rename(columns={"mid": "future_mid"}),
        left_index=True, right_index=True,
        direction="forward",
        tolerance=pd.Timedelta(minutes=tolerance_min),
    )["future_mid"].to_numpy()

    change = (future / out["mid"].to_numpy() - 1) * 100
    out["future_change_pct"] = change
    out["label"] = np.select(
        [change > threshold_pct, change < -threshold_pct],
        [1, -1], default=0,
    ).astype(float)
    out.loc[np.isnan(change), "label"] = np.nan
    return out


def build_dataset(data_dir: str = DATA_DIR, horizon_min: int = 15,
                  threshold_pct: float = 0.25):
    feat = compute_features(load_snapshots(data_dir))
    feat = add_labels(feat, horizon_min, threshold_pct)
    ready = feat.dropna(subset=FEATURE_COLS + ["label"])
    return ready[FEATURE_COLS], ready["label"].astype(int), ready


def latest_feature_row(data_dir: str = DATA_DIR):
    feat = compute_features(load_snapshots(data_dir))
    row = feat.iloc[[-1]]
    if row[FEATURE_COLS].isna().any(axis=None):
        missing = [c for c in FEATURE_COLS if row[c].isna().any()]
        raise ValueError(f"Historial insuficiente para: {missing}")
    return row
