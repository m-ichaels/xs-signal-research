"""The new-products test.  The models fitted walk-forward on equities are applied, unchanged, to the ETF and futures
frames of the same years; the same features are also refitted walk-forward within each universe; and each feature's
own univariate IC is tabulated per universe, so the reader sees which inputs carry over in sign and which do not."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import CostModel
from .features import FEATURES
from .metrics import by_year, decile_spread, ic_by_day, ic_summary
from .models import LearnerConfig, apply_models, walk_forward
from .portfolio import BookConfig, book


def univariate_ic(df: pd.DataFrame, target: str, features=FEATURES, horizon: int = 5) -> pd.DataFrame:
    rows = []
    for f in features:
        ic = ic_by_day(df, f, target)
        s = ic_summary(ic, horizon, n_boot=0)
        rows.append({"feature": f, "ic": s["ic_mean"], "t_nw": s["t_nw"], "n_days": s["n_days"]})
    return pd.DataFrame(rows).set_index("feature")


def evaluate(pred: pd.DataFrame, target: str, horizon: int, n_boot: int, universe: str, capital: float, cost_name: str = "routed") -> dict:
    ic = ic_by_day(pred, "z", target)
    if ic.empty:
        return {"ic": {"ic_mean": float("nan"), "n_days": 0}}
    out = {"ic": ic_summary(ic, horizon, n_boot), "by_year": by_year(ic).round(4).to_dict(orient="index")}
    d = decile_spread(pred, "z", target, n=5)
    out["quintile_spread"] = {"mean": d["spread_mean"], "t": d["spread_t"], "by_quintile": d["by_decile"]}
    b = book(pred, "z", target, CostModel.named(cost_name, universe), BookConfig(capital=capital, horizon=horizon, w_max=0.10 if universe != "equity" else 0.02), n_boot=n_boot)
    out["book"] = b["stats"]
    return out


def transfer_test(frames: dict[str, pd.DataFrame], models: dict, target: str, first_year: int, last_year: int, cfg: LearnerConfig, n_boot: int, capitals: dict[str, float]) -> dict:
    """for each non-equity universe: the equity model applied as is, and the same learner refitted within the universe"""
    horizon = int(target.split("_")[1])
    out = {}
    for u, df in frames.items():
        res = {"n_symbols": int(df["symbol"].nunique()), "n_days": int(df["date"].nunique())}
        ev = df[(df["date"].dt.year >= first_year) & (df["date"].dt.year <= last_year)]
        if u != "equity":
            tr = apply_models(ev, models)
            res["equity_model"] = evaluate(tr, target, horizon, n_boot, u, capitals[u])
            if "group" in df:
                res["equity_model_by_group"] = {}
                for gname, gdf in tr.groupby("group"):
                    if gdf["symbol"].nunique() >= 5:
                        ic = ic_by_day(gdf, "z", target)
                        res["equity_model_by_group"][gname] = {"n_symbols": int(gdf["symbol"].nunique()), **{k: v for k, v in ic_summary(ic, horizon, 0).items() if k in ("ic_mean", "t_nw", "n_days")}} if len(ic) > 50 else {"n_symbols": int(gdf["symbol"].nunique())}
        try:
            own = walk_forward(df, first_year, last_year, target, FEATURES, cfg, min_train_days=400)
            res["own_model"] = evaluate(own, target, horizon, n_boot, u, capitals[u])
        except ValueError as e:
            res["own_model"] = {"error": str(e)}
        res["univariate"] = univariate_ic(ev, target, FEATURES, horizon).round(4).to_dict(orient="index")
        out[u] = res
    return out
