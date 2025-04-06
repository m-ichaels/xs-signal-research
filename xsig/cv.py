"""Honest validation.  A cross-sectional signal with an h-day target leaks in two ways under ordinary K-fold: a training
row's target window overlaps the test dates (overlap leakage), and a shuffled split puts other names from the same
test day into training (same-day leakage).  Purged K-fold (Lopez de Prado 2018, ch. 7) splits by date in contiguous
blocks, drops from training every date whose target window touches the test block, and embargoes the dates just after
it.  The ladder run here reports the same learner under five schemes, from the in-sample fit down to walk-forward."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .features import FEATURES
from .models import Learner, LearnerConfig, EMBARGO, walk_forward
from .metrics import ic_by_day


def purged_kfold_splits(dates: np.ndarray, n_splits: int, horizon: int, embargo: int = EMBARGO):
    """yield (train_dates, test_dates) over the sorted unique dates; training excludes
    [test_start - horizon - embargo, test_end + embargo] in trading-day positions"""
    u = np.array(sorted(pd.unique(dates)))
    n = len(u)
    pos = {d: i for i, d in enumerate(u)}
    blocks = np.array_split(np.arange(n), n_splits)
    for b in blocks:
        lo, hi = b[0], b[-1]
        keep = np.ones(n, dtype=bool)
        keep[max(0, lo - horizon - embargo): min(n, hi + embargo + 1)] = False
        yield u[keep], u[b]


def cv_predict(df: pd.DataFrame, target: str, scheme: str, n_splits: int = 5, features=FEATURES, cfg=LearnerConfig(), kind="blend") -> pd.Series:
    """out-of-fold prediction z for every row under one scheme: in_sample | kfold_shuffled | kfold_blocked | purged"""
    horizon = int(target.split("_")[1])
    d = df[df[target].notna()].reset_index(drop=True)
    z = pd.Series(np.nan, index=d.index)
    if scheme == "in_sample":
        m = Learner(features, cfg, kind).fit(d, target)
        z[:] = m.predict(d)
    elif scheme == "kfold_shuffled":
        for tr, te in KFold(n_splits, shuffle=True, random_state=cfg.seed).split(d):
            m = Learner(features, cfg, kind).fit(d.iloc[tr], target)
            z.iloc[te] = m.predict(d.iloc[te])
    elif scheme in ("kfold_blocked", "purged"):
        dates = d["date"].values
        for tr_d, te_d in purged_kfold_splits(dates, n_splits, horizon if scheme == "purged" else 0, EMBARGO if scheme == "purged" else 0):
            tr, te = np.isin(dates, tr_d), np.isin(dates, te_d)
            m = Learner(features, cfg, kind).fit(d[tr], target)
            z[te] = m.predict(d[te])
    else:
        raise ValueError(scheme)
    return pd.Series(z.values, index=d.index), d


def validation_ladder(df: pd.DataFrame, target: str, first_year: int, last_year: int, n_splits: int = 5, cfg=LearnerConfig(), features=FEATURES, kind: str = "blend") -> dict:
    """mean daily Spearman IC of the same learner under each scheme, on the same evaluation years"""
    ev = df[(df["date"].dt.year >= first_year) & (df["date"].dt.year <= last_year)]
    out = {}
    for scheme in ("in_sample", "kfold_shuffled", "kfold_blocked", "purged"):
        z, d = cv_predict(ev, target, scheme, n_splits, features, cfg, kind)
        d = d.assign(z=z.values)
        ic = ic_by_day(d, "z", target)
        out[scheme] = {"ic_mean": float(ic.mean()), "ic_sd": float(ic.std()), "n_days": int(len(ic))}
    wf = walk_forward(df, first_year, last_year, target, features, cfg, kind)
    ic = ic_by_day(wf, "z", target)
    out["walk_forward"] = {"ic_mean": float(ic.mean()), "ic_sd": float(ic.std()), "n_days": int(len(ic))}
    return out
