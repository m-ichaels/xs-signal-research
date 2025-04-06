"""Signal statistics with the inference a serially correlated daily series needs.  The IC is the daily cross-sectional
Spearman correlation between the prediction and the h-day forward return; its mean has a Newey-West standard error with
lag h and a moving-block-bootstrap interval (block length 2h+1, Kunsch 1989).  Also here: IC decay (the signal at t
against the single-day return k days ahead), decile spreads, the signal's own autocorrelation and the turnover it
implies, per-year tables, and the deflated Sharpe ratio of Bailey and Lopez de Prado (2014) for the multiple-testing
question the ablations raise."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


def ic_by_day(pred: pd.DataFrame, col: str, target: str) -> pd.Series:
    p = pred.dropna(subset=[target, col])
    return p.groupby("date").apply(lambda g: g[col].corr(g[target], method="spearman") if len(g) > 5 else np.nan, include_groups=False).dropna()


def newey_west_t(x: np.ndarray, lag: int) -> tuple[float, float]:
    """mean and t-statistic of a series with Bartlett-kernel HAC variance at the given lag"""
    x = np.asarray(x, dtype=float); n = len(x); m = x.mean(); e = x - m
    s = e @ e / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * (e[k:] @ e[:-k]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    return float(m), float(m / se)


def block_bootstrap_mean(x: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0, q=(0.025, 0.975)) -> tuple[float, float]:
    """moving-block bootstrap interval for the mean of a stationary series"""
    x = np.asarray(x, dtype=float); n = len(x)
    if n_boot <= 0 or n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    idx = np.arange(block)
    for b in range(n_boot):
        s = rng.choice(starts, k)
        means[b] = x[(s[:, None] + idx[None, :]).ravel()[:n]].mean()
    lo, hi = np.quantile(means, q)
    return float(lo), float(hi)


def ic_summary(ic: pd.Series, horizon: int, n_boot: int = 2000, seed: int = 0) -> dict:
    m, t = newey_west_t(ic.values, horizon)
    lo, hi = block_bootstrap_mean(ic.values, 2 * horizon + 1, n_boot, seed)
    return {"ic_mean": m, "ic_sd": float(ic.std()), "t_nw": t, "ic_lo": lo, "ic_hi": hi, "n_days": int(len(ic)), "share_positive": float((ic > 0).mean()),
            "ir_annual": float(m / ic.std() * np.sqrt(252.0 / horizon)) if ic.std() > 0 else float("nan")}


def by_year(ic: pd.Series) -> pd.DataFrame:
    g = ic.groupby(ic.index.year)
    return pd.DataFrame({"ic_mean": g.mean(), "ic_sd": g.std(), "n_days": g.size()})


def decile_spread(pred: pd.DataFrame, col: str, target: str, n: int = 10) -> dict:
    """mean target by decile and the top-minus-bottom spread per day (mean and t with NW lag h)"""
    p = pred.dropna(subset=[target, col]).copy()
    p["dec"] = p.groupby("date")[col].transform(lambda s: pd.qcut(s.rank(method="first"), n, labels=False))
    by_dec = p.groupby("dec")[target].mean()
    daily = p.groupby(["date", "dec"])[target].mean().unstack()
    spread = (daily[n - 1] - daily[0]).dropna()
    h = int(target.split("_")[1])
    m, t = newey_west_t(spread.values, h)
    return {"by_decile": by_dec.tolist(), "spread_mean": m, "spread_t": t, "spread_daily": spread}


def ic_decay(pred: pd.DataFrame, panel, col: str = "z", max_lag: int = 21) -> pd.DataFrame:
    """IC of the signal at t against the demeaned single-day return on day t+k, k = 1..max_lag, and its cumulative sum"""
    z = pred.pivot(index="date", columns="symbol", values=col)
    rows = []
    for k in range(1, max_lag + 1):
        rk = panel.ret.shift(-k).reindex(index=z.index, columns=z.columns).where(z.notna())
        rk = rk.sub(rk.mean(axis=1), axis=0)
        ic = z.rank(axis=1).corrwith(rk.rank(axis=1), axis=1).dropna()
        m, t = newey_west_t(ic.values, 1)
        rows.append({"lag": k, "ic": m, "t": t})
    out = pd.DataFrame(rows).set_index("lag")
    out["cum_ic"] = out["ic"].cumsum()
    return out


def signal_autocorrelation(pred: pd.DataFrame, col: str = "z", lags=(1, 2, 5, 10, 21)) -> dict:
    """cross-sectional rank autocorrelation of the signal at each lag and the half-life implied by lag-1 decay"""
    z = pred.pivot(index="date", columns="symbol", values=col)
    out = {}
    for k in lags:
        ac = z.rank(axis=1).corrwith(z.shift(k).rank(axis=1), axis=1).dropna()
        out[f"ac_{k}"] = float(ac.mean())
    a1 = out.get("ac_1", np.nan)
    out["half_life_days"] = float(np.log(0.5) / np.log(a1)) if 0 < a1 < 1 else float("nan")
    return out


def sharpe(r: pd.Series, periods_per_year: float) -> float:
    return float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else float("nan")


def deflated_sharpe(sr: float, n_obs: int, n_trials: int, sr_var: float, r: np.ndarray) -> dict:
    """probability that an observed Sharpe (per period) exceeds the expected maximum of n_trials null Sharpes with
    variance sr_var across trials (Bailey and Lopez de Prado 2014, eq. 12); r is the per-period return series"""
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    e = np.euler_gamma
    sr0 = np.sqrt(sr_var) * ((1 - e) * norm.ppf(1 - 1.0 / max(n_trials, 2)) + e * norm.ppf(1 - 1.0 / (max(n_trials, 2) * np.e)))
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2, 1e-9))
    z = (sr - sr0) * np.sqrt(n_obs - 1) / denom
    return {"sr": sr, "sr0": float(sr0), "dsr": float(norm.cdf(z)), "n_trials": n_trials, "skew": g3, "kurtosis": g4}
