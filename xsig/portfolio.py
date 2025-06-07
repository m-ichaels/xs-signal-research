"""From signal to book: a rank-weighted, dollar-neutral long-short portfolio rebalanced every h trading days, its
turnover, and the cost of each rebalance from the algo-wheel model at a stated capital.  Returns are reported gross
and net; the capacity curve runs the same book across capitals and finds where the net return crosses zero.  A `net
IC` is also defined per name: the forward return less the cost the signal's own trade in that name incurs, signed by
the position, so that the correlation with the signal is what a follower of the signal earns after paying to hold it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel
from .metrics import sharpe, ic_summary


@dataclass
class BookConfig:
    capital: float = 1e9
    gross: float = 1.0          # sum |w|; long 0.5, short 0.5
    w_max: float = 0.05
    horizon: int = 5


def weights_from_signal(z: np.ndarray, gross: float, w_max: float) -> np.ndarray:
    """w proportional to the demeaned signal, dollar neutral, capped, with sum |w| = gross"""
    w = z - z.mean()
    if np.abs(w).sum() == 0:
        return np.zeros_like(w)
    w = w / np.abs(w).sum() * gross
    for _ in range(20):                       # cap and renormalise; converges in a few passes
        w = np.clip(w, -w_max, w_max)
        w = w - w.mean()
        s = np.abs(w).sum()
        if s == 0 or abs(s - gross) < 1e-9:
            break
        w = w / s * gross
    return w


def run_book(pred: pd.DataFrame, col: str, target: str, model: CostModel, cfg: BookConfig, offset: int = 0) -> dict:
    """rebalance at every h-th date from `offset`; per period: gross return, cost, net, turnover; plus per-name net IC rows"""
    h = int(target.split("_")[1])
    dates = np.array(sorted(pred["date"].unique()))
    rb = dates[offset::h]
    p = pred[pred["date"].isin(rb)].dropna(subset=[col, target]).sort_values(["date", "symbol"])
    w_prev = pd.Series(dtype=float)
    r_prev = None
    rows, net_rows = [], []
    for d, g in p.groupby("date", sort=True):
        z = g[col].values
        w = pd.Series(weights_from_signal(z, cfg.gross, cfg.w_max), index=g["symbol"].values)
        # drift the old book to today's prices, then trade to the new weights
        if len(w_prev) and r_prev is not None:
            r = r_prev.reindex(w_prev.index).fillna(0.0)
            drift = w_prev * (1.0 + r) / (1.0 + float((w_prev * r).sum()))
        else:
            drift = pd.Series(0.0, index=w.index)
        all_idx = w.index.union(drift.index)
        dw = w.reindex(all_idx).fillna(0.0) - drift.reindex(all_idx).fillna(0.0)
        gi = g.set_index("symbol")
        sig_bp = 1e4 * gi["sigma"].reindex(all_idx).fillna(gi["sigma"].median())
        spr = gi["spread_bp"].reindex(all_idx).fillna(gi["spread_bp"].median())
        adv = gi["adv_usd"].reindex(all_idx).fillna(gi["adv_usd"].median())
        pct_adv = (dw.abs() * cfg.capital / adv.clip(lower=1.0)).values
        notional = gi["notional"].reindex(all_idx).fillna(gi["notional"].median()).values if "notional" in gi else None
        c_bp = model.cost_bp(sig_bp.values, spr.values, pct_adv, notional)
        cost = float((c_bp * 1e-4 * dw.abs().values).sum())          # fraction of capital
        turnover = float(dw.abs().sum() / 2.0)
        gross_ret = float((w * gi[target].reindex(w.index)).sum())
        rows.append({"date": d, "gross": gross_ret, "cost": cost, "net": gross_ret - cost, "turnover": turnover, "n": len(w), "avg_cost_bp": float((c_bp * dw.abs().values).sum() / max(dw.abs().sum(), 1e-12)), "max_pct_adv": float(pct_adv.max())})
        # per-name net target: return less the signed cost of this period's trade in the name, per unit held
        held = w[w != 0]
        drag = pd.Series(c_bp, index=all_idx).reindex(held.index) * 1e-4 * (dw.reindex(held.index).abs() / held.abs()).clip(upper=2.0)
        nt = gi.loc[held.index, target] - np.sign(held) * drag
        net_rows.append(pd.DataFrame({"date": d, "symbol": held.index, col: gi.loc[held.index, col].values, "w": held.values, "net_target": nt.values, target: gi.loc[held.index, target].values}))
        w_prev = w
        # realised return of each name over the holding period, for the drift
        r_prev = gi[target].reindex(w.index)
    per = pd.DataFrame(rows).set_index("date")
    return {"periods": per, "net_rows": pd.concat(net_rows, ignore_index=True) if net_rows else pd.DataFrame()}


def book_stats(per: pd.DataFrame, h: int) -> dict:
    ppy = 252.0 / h
    nav = (1.0 + per["net"]).cumprod()
    dd = float((nav / nav.cummax() - 1.0).min())
    return {"gross_ann": float(per["gross"].mean() * ppy), "net_ann": float(per["net"].mean() * ppy), "cost_ann": float(per["cost"].mean() * ppy),
            "vol_ann": float(per["net"].std() * np.sqrt(ppy)), "sharpe_gross": sharpe(per["gross"], ppy), "sharpe_net": sharpe(per["net"], ppy),
            "turnover": float(per["turnover"].mean()), "avg_cost_bp": float(per["avg_cost_bp"].mean()), "max_dd_net": dd, "n_periods": int(len(per)), "max_pct_adv": float(per["max_pct_adv"].quantile(0.99))}


def book(pred: pd.DataFrame, col: str, target: str, model: CostModel, cfg: BookConfig, offsets: int | None = None, n_boot: int = 1000) -> dict:
    """offset-averaged statistics of the book, the net IC with its band, and the per-period series of offset 0"""
    h = int(target.split("_")[1])
    offsets = min(h, 5) if offsets is None else offsets
    stats, first = [], None
    net_ic_all = []
    for o in range(offsets):
        r = run_book(pred, col, target, model, cfg, o)
        if r["periods"].empty:
            continue
        stats.append(book_stats(r["periods"], h))
        if first is None:
            first = r["periods"]
        if not r["net_rows"].empty:
            nr = r["net_rows"]
            net_ic_all.append(nr.groupby("date").apply(lambda g: g[col].corr(g["net_target"], method="spearman"), include_groups=False))
    s = pd.DataFrame(stats).mean().to_dict()
    s["n_offsets"] = len(stats)
    if net_ic_all:
        ic = pd.concat(net_ic_all).sort_index()
        s["net_ic"] = ic_summary(ic, h, n_boot)          # one observation per rebalance and offset; NW lag h
    return {"stats": s, "periods": first}


def capacity_curve(pred: pd.DataFrame, col: str, target: str, model: CostModel, cfg: BookConfig, capitals) -> pd.DataFrame:
    rows = []
    for C in capitals:
        c = BookConfig(capital=C, gross=cfg.gross, w_max=cfg.w_max, horizon=cfg.horizon)
        s = book(pred, col, target, model, c, n_boot=0)["stats"]
        rows.append({"capital": C, **{k: s[k] for k in ("gross_ann", "net_ann", "cost_ann", "sharpe_net", "turnover", "avg_cost_bp", "max_pct_adv")}, "net_ic": s.get("net_ic", {}).get("ic_mean", float("nan"))})
    return pd.DataFrame(rows).set_index("capital")


def breakeven_capital(curve: pd.DataFrame) -> float:
    """capital at which the net annual return crosses zero, log-linear interpolation on the curve"""
    c = curve["net_ann"]
    x = np.log(curve.index.values.astype(float)); y = c.values
    for i in range(len(y) - 1):
        if y[i] > 0 >= y[i + 1]:
            t = y[i] / (y[i] - y[i + 1])
            return float(np.exp(x[i] + t * (x[i + 1] - x[i])))
    return float("inf") if y[-1] > 0 else float(curve.index[0])
