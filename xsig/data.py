"""Daily bars for the three product universes (committed parquet), the wide panels a signal is built from, and the
per-name liquidity statistics the cost model needs: 20-day ADV in dollars, daily volatility, a quoted-spread rule.
A DuckDB connection with views over every parquet is available for SQL (`python -m xsig sql ...`)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DER = os.path.join(DATA, "derived")
RES = os.path.join(ROOT, "results")
UNIVERSES = ("equity", "etf", "futures")


def load_universe(universe: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, f"universe_{universe}.csv"))


def load_prices(universe: str) -> pd.DataFrame:
    """long table symbol, date, open, high, low, close, adjclose, volume"""
    p = pd.read_parquet(os.path.join(DER, f"{universe}.parquet"))
    p["date"] = pd.to_datetime(p["date"])
    p = p[(p["close"] > 0) & (p["open"] > 0) & (p["adjclose"] > 0)]
    if universe != "futures":                       # a zero-volume day on a listed product is a holiday artefact
        p = p[p["volume"] > 0]
    syms = set(load_universe(universe)["symbol"])
    p = p[p["symbol"].isin(syms)]
    return p.sort_values(["symbol", "date"]).reset_index(drop=True)


def spread_rule(adv_usd) -> pd.DataFrame | pd.Series:
    """quoted spread in bp from dollar ADV, the algo-wheel rule: 1.2 bp for a $10bn name, 5 bp at $100m, 13 bp at $10m"""
    return (0.8 + 40.0 / np.sqrt((adv_usd / 1e6).clip(lower=1.0))).clip(0.8, 30.0)


class Panel:
    """wide daily panels indexed by date, one column per symbol, plus rolling liquidity statistics known at the close"""

    def __init__(self, prices: pd.DataFrame, universe: str, lookback: int = 20, min_adv_usd: float = 0.0):
        self.universe = universe
        pv = lambda f: prices.pivot(index="date", columns="symbol", values=f).sort_index()
        self.open, self.close, self.adj, self.volume = pv("open"), pv("close"), pv("adjclose"), pv("volume")
        self.high, self.low = pv("high"), pv("low")
        self.ret = self.adj.pct_change(fill_method=None)
        self.dates = self.close.index
        self.adv = self.volume.rolling(lookback, min_periods=10).mean()
        self.vol = np.log(self.adj).diff().rolling(lookback, min_periods=10).std().clip(lower=0.002)
        u = load_universe(universe).set_index("symbol")
        if universe == "futures":
            mult = u["multiplier"].reindex(self.close.columns)
            self.notional = self.close * mult                       # dollar value of one contract
            self.adv_usd = (self.volume * self.notional).rolling(lookback, min_periods=10).mean()
            tick_bp = 1e4 * u["tick"].reindex(self.close.columns) / self.close    # one tick in bp of price
            self.spread_bp = tick_bp.clip(lower=0.05)               # quoted spread = one tick on a liquid contract
        else:
            self.adv_usd = (self.volume * self.close).rolling(lookback, min_periods=10).mean()
            self.spread_bp = spread_rule(self.adv_usd)
        self.sector = u["sector"].reindex(self.close.columns)
        self.group = u["group"].reindex(self.close.columns) if "group" in u else None
        if min_adv_usd > 0:                                         # liquidity screen, applied to the eligible universe
            self.eligible = self.adv_usd >= min_adv_usd
        else:
            self.eligible = self.adv_usd.notna()


def load_panel(universe: str, min_adv_usd: float | None = None) -> Panel:
    if min_adv_usd is None:
        min_adv_usd = {"equity": 5e7, "etf": 5e6, "futures": 0.0}[universe]
    return Panel(load_prices(universe), universe, min_adv_usd=min_adv_usd)


def store(path: str | None = None):
    """DuckDB connection with a view per parquet under data/derived and results/ (read-only use)"""
    import duckdb
    con = duckdb.connect(path or ":memory:")
    q = lambda p: p.replace(os.sep, "/").replace("'", "''")          # SQL string literal of a path
    for d in (DER, RES):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".parquet"):
                name = f[:-8].replace("-", "_")
                con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{q(os.path.join(d, f))}')")
    for u in UNIVERSES:
        con.execute(f"CREATE OR REPLACE VIEW universe_{u} AS SELECT * FROM read_csv_auto('{q(os.path.join(DATA, f'universe_{u}.csv'))}')")
    return con
