"""The learner: ridge and gradient boosting on the rank-normalised features, each prediction z-scored across the day and
the two averaged.  Walk-forward refits every January on all history to date with an embargo of horizon + 5 days, so no
training target overlaps a test date.  The fitted models of each year are kept, because the transfer test applies
exactly these models to the other product universes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .features import DEFAULT_HORIZON, FEATURES

EMBARGO = 5


@dataclass
class LearnerConfig:
    ridge_alpha: float = 10.0
    gbm_iter: int = 150
    gbm_lr: float = 0.05
    gbm_depth: int = 3
    gbm_leaf: int = 200
    blend: tuple[float, float] = (0.5, 0.5)
    seed: int = 7


class Learner:
    """ridge + gradient boosting blend on a feature list"""

    def __init__(self, features: list[str] = FEATURES, cfg: LearnerConfig = LearnerConfig(), kind: str = "blend"):
        self.features, self.cfg, self.kind = list(features), cfg, kind
        self.ridge = self.gbm = None

    def fit(self, df: pd.DataFrame, target: str):
        X, y = df[self.features].values, df[target].values
        if self.kind in ("blend", "ridge"):
            self.ridge = Ridge(alpha=self.cfg.ridge_alpha).fit(X, y)
        if self.kind in ("blend", "gbm"):
            c = self.cfg
            self.gbm = HistGradientBoostingRegressor(max_iter=c.gbm_iter, learning_rate=c.gbm_lr, max_depth=c.gbm_depth, min_samples_leaf=c.gbm_leaf, l2_regularization=1.0, random_state=c.seed).fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """z-scored across each day; the blend averages the two z-scored components"""
        X = df[self.features].values
        parts, w = [], []
        if self.ridge is not None:
            parts.append(zscore_by_day(df, self.ridge.predict(X))); w.append(self.cfg.blend[0])
        if self.gbm is not None:
            parts.append(zscore_by_day(df, self.gbm.predict(X))); w.append(self.cfg.blend[1])
        w = np.array(w) / sum(w)
        return sum(wi * p for wi, p in zip(w, parts))


def zscore_by_day(df: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(p, dtype=float), index=df.index)
    g = s.groupby(df["date"].values)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0).values


def train_cut(year: int, horizon: int) -> pd.Timestamp:
    """last training date for predictions in `year`: horizon + EMBARGO business days before 1 January"""
    return pd.Timestamp(f"{year}-01-01") - pd.tseries.offsets.BDay(EMBARGO + horizon)


def walk_forward(df: pd.DataFrame, first_year: int, last_year: int, target: str = f"target_{DEFAULT_HORIZON}", features=FEATURES, cfg=LearnerConfig(), kind="blend", min_train_days: int = 500, models: dict | None = None, train_ic: bool = False) -> pd.DataFrame:
    """predictions `z` for every date in [first_year, last_year] from a model refitted each January on all earlier data;
    with train_ic the in-sample IC of each training window is recorded (`ic_train`); fitted learners land in `models[year]`"""
    horizon = int(target.split("_")[1])
    out = []
    for year in range(first_year, last_year + 1):
        tr = df[(df["date"] <= train_cut(year, horizon)) & df[target].notna()]
        te = df[(df["date"].dt.year == year)]
        if tr["date"].nunique() < min_train_days or te.empty:
            continue
        m = Learner(features, cfg, kind).fit(tr, target)
        if models is not None:
            models[year] = m
        te = te.copy()
        te["z"] = m.predict(te)
        if train_ic:
            p_tr = m.predict(tr)
            te["ic_train"] = float(pd.DataFrame({"d": tr["date"].values, "p": p_tr, "y": tr[target].values}).groupby("d").apply(lambda g: g["p"].corr(g["y"], method="spearman"), include_groups=False).mean())
        out.append(te)
    if not out:
        raise ValueError("no prediction year has enough training history")
    return pd.concat(out, ignore_index=True)


def apply_models(df: pd.DataFrame, models: dict[int, Learner]) -> pd.DataFrame:
    """the equity-fitted model of each year applied to another universe's frame for that year (the transfer test)"""
    out = []
    for year, m in models.items():
        te = df[df["date"].dt.year == year]
        if te.empty:
            continue
        te = te.copy(); te["z"] = m.predict(te)
        out.append(te)
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0].assign(z=np.nan)
