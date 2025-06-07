"""A simple risk model and the neutralisation of the signal against it.  Exposures per name-day: market beta (250-day,
against the equal-weighted universe), size (rank-normalised log dollar ADV), sector dummies, and two style factors that
are also the signal's own inputs, 12-1 momentum and 20-day volatility.  Neutralising the signal is a cross-sectional
regression each day; the residual is what is left once the risk model can price the exposures.  Factor returns come
from the same regression on the forward return (Fama and MacBeth 1973), and a portfolio is attributed to them by a
time-series regression."""
from __future__ import annotations

import numpy as np
import pandas as pd

STYLE = ["mom_12_1", "vol_20"]
BASE = ["beta", "size"]


def add_beta(df: pd.DataFrame, panel, window: int = 250) -> pd.DataFrame:
    """rolling market beta of each name to the equal-weighted return of the eligible universe, known at the close of t"""
    r = panel.ret.where(panel.eligible)
    m = r.mean(axis=1)
    cov = r.rolling(window, min_periods=120).cov(m)
    var = m.rolling(window, min_periods=120).var()
    beta = cov.div(var, axis=0).clip(-1.0, 4.0)
    b = beta.stack(future_stack=True).rename("beta")
    b.index.names = ["date", "symbol"]
    out = df.merge(b.reset_index(), on=["date", "symbol"], how="left")
    out["beta"] = out["beta"].fillna(1.0)
    return out


def _design(g: pd.DataFrame, factors: list[str], sector: bool) -> np.ndarray:
    cols = [np.ones(len(g))] + [g[f].values for f in factors]
    if sector:
        d = pd.get_dummies(g["sector"]).values.astype(float)
        cols.append(d[:, 1:] if d.shape[1] > 1 else np.zeros((len(g), 0)))
    return np.column_stack(cols)


def neutralise(df: pd.DataFrame, col: str, factors: list[str], sector: bool = True, out: str | None = None) -> pd.DataFrame:
    """residual of the daily cross-sectional regression of `col` on the factors (+ sector dummies), z-scored per day"""
    out = out or f"{col}_neutral"
    res = np.full(len(df), np.nan)
    for _, idx in df.groupby("date").indices.items():
        g = df.iloc[idx]
        X = _design(g, factors, sector); y = g[col].values
        if len(g) <= X.shape[1] + 2:
            continue
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        sd = e.std()
        res[idx] = e / sd if sd > 0 else 0.0
    df = df.copy(); df[out] = res
    return df


def factor_returns(df: pd.DataFrame, target: str, factors: list[str], sector: bool = True) -> pd.DataFrame:
    """Fama-MacBeth: per-day cross-sectional regression of the forward return on the exposures; columns = factor returns"""
    rows = {}
    names = ["const"] + factors
    for d, idx in df.groupby("date").indices.items():
        g = df.iloc[idx].dropna(subset=[target])
        if len(g) < 20:
            continue
        X = _design(g, factors, sector); y = g[target].values
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows[d] = b[: len(names)]
    return pd.DataFrame.from_dict(rows, orient="index", columns=names).sort_index()


def attribute(port: pd.Series, fr: pd.DataFrame, factors: list[str]) -> dict:
    """time-series regression of a portfolio's per-period return on the factor returns: alpha, loadings, R^2"""
    j = pd.concat([port.rename("p"), fr[factors]], axis=1, sort=True).dropna()
    X = np.column_stack([np.ones(len(j))] + [j[f].values for f in factors]); y = j["p"].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    r2 = 1.0 - e.var() / y.var() if y.var() > 0 else float("nan")
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * e.var())
    return {"alpha": float(b[0]), "alpha_t": float(b[0] / se[0]) if se[0] > 0 else float("nan"), "loadings": dict(zip(factors, map(float, b[1:]))), "r2": float(r2), "n": int(len(j))}
