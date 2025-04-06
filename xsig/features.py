"""The algo-wheel signal's inputs, unchanged: eight price-and-volume features, each rank-normalised to N(0,1) across the
universe every day, and the cross-sectionally demeaned forward return at several horizons.  Features use data to the
close of day t (rolling windows tolerate a few missing bars); the target starts at that close.  The same code builds the frame for equities, ETFs and futures, so the
transfer test changes nothing but the universe."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

FEATURES = ["mom_12_1", "rev_1m", "rev_1w", "vol_20", "vol_ratio", "volume_trend", "range_20", "hi_52w"]
GROUPS = {"momentum": ["mom_12_1", "hi_52w"], "reversal": ["rev_1m", "rev_1w"], "volatility": ["vol_20", "vol_ratio"], "liquidity": ["volume_trend", "range_20"]}
HORIZONS = [1, 2, 5, 10, 21]
DEFAULT_HORIZON = 5


def raw_features(panel) -> dict[str, pd.DataFrame]:
    adj, vol, hi, lo = panel.adj, panel.volume, panel.high, panel.low
    r = np.log(adj).diff()
    v = vol.replace(0, np.nan)
    return {
        "mom_12_1": adj.shift(21) / adj.shift(252) - 1.0,
        "rev_1m": adj / adj.shift(21) - 1.0,
        "rev_1w": adj / adj.shift(5) - 1.0,
        "vol_20": r.rolling(20, min_periods=15).std(),
        "vol_ratio": r.rolling(20, min_periods=15).std() / r.rolling(120, min_periods=90).std(),
        "volume_trend": np.log(v.rolling(5, min_periods=3).mean() / v.rolling(60, min_periods=40).mean()),
        "range_20": (np.log(hi / lo)).rolling(20, min_periods=15).mean(),
        "hi_52w": adj / adj.rolling(252, min_periods=200).max() - 1.0,
    }


def targets(panel, horizons=HORIZONS) -> dict[str, pd.DataFrame]:
    """forward h-day return from the close of t, demeaned across the eligible names each day"""
    out = {}
    for h in horizons:
        fwd = panel.adj.shift(-h) / panel.adj - 1.0
        fwd = fwd.where(panel.eligible)
        out[f"target_{h}"] = fwd.sub(fwd.mean(axis=1), axis=0)
    return out


def rank_normalise(df: pd.DataFrame, cols: list[str], by: str = "date") -> pd.DataFrame:
    """Gaussian rank transform within each day: the feature keeps its ordering and loses its scale and outliers"""
    g = df.groupby(by)
    for c in cols:
        rk = g[c].rank(pct=True)
        n = g[c].transform("size")
        df[c] = norm.ppf(((rk * n - 0.5) / n).clip(0.001, 0.999))
    return df


def build_frame(panel, horizons=HORIZONS, min_names: int = 10) -> pd.DataFrame:
    """long table date, symbol, the eight features (rank-normalised), sigma, adv_usd, spread_bp, sector, target_h..."""
    f = raw_features(panel)
    t = targets(panel, horizons)
    cols = [x.where(panel.eligible).stack(future_stack=True).rename(k) for k, x in f.items()]
    cols += [x.stack(future_stack=True).rename(k) for k, x in t.items()]
    cols += [panel.vol.stack(future_stack=True).rename("sigma"), panel.adv_usd.stack(future_stack=True).rename("adv_usd"), panel.spread_bp.stack(future_stack=True).rename("spread_bp")]
    if getattr(panel, "notional", None) is not None:
        cols.append(panel.notional.stack(future_stack=True).rename("notional"))
    df = pd.concat(cols, axis=1)
    df.index.names = ["date", "symbol"]
    df = df.dropna(subset=FEATURES + ["sigma", "adv_usd"]).reset_index()
    n_day = df.groupby("date")["symbol"].transform("size")
    df = df[n_day >= min_names].copy()
    df["sector"] = df["symbol"].map(panel.sector).fillna("Other")
    if getattr(panel, "group", None) is not None:
        df["group"] = df["symbol"].map(panel.group)
    df = rank_normalise(df, FEATURES)
    df["size"] = rank_normalise(df[["date", "adv_usd"]].assign(size=np.log(df["adv_usd"])), ["size"])["size"]
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)
