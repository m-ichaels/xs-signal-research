"""Features carry no look-ahead, the rank transform does what it says, and purged K-fold removes the leak that
ordinary K-fold has on overlapping targets (a synthetic case with a known leak and no true signal)."""
import numpy as np
import pandas as pd
import pytest

from xsig.cv import cv_predict, purged_kfold_splits
from xsig.data import Panel
from xsig.features import FEATURES, build_frame, rank_normalise
from xsig.metrics import ic_by_day
from xsig.models import LearnerConfig


def synthetic_prices(n_names=30, n_days=700, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rows = []
    for i in range(n_names):
        r = rng.normal(0.0003, 0.02, n_days)
        px = 50 * np.exp(np.cumsum(r))
        vol = rng.lognormal(14, 0.3, n_days)
        for d, p, v in zip(dates, px, vol):
            rows.append((f"S{i:02d}", d, p * 0.995, p * 1.01, p * 0.99, p, p, v))
    return pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "adjclose", "volume"])


@pytest.fixture(scope="module")
def frame():
    return synthetic_prices()


def _panel(prices):
    import xsig.data as D
    # a universe file is not needed: patch the sector lookup with a constant
    class P(Panel):
        def __init__(self, prices):
            pv = lambda f: prices.pivot(index="date", columns="symbol", values=f).sort_index()
            self.universe = "equity"
            self.open, self.close, self.adj, self.volume = pv("open"), pv("close"), pv("adjclose"), pv("volume")
            self.high, self.low = pv("high"), pv("low")
            self.ret = self.adj.pct_change(fill_method=None)
            self.dates = self.close.index
            self.adv = self.volume.rolling(20, min_periods=10).mean()
            self.adv_usd = (self.volume * self.close).rolling(20, min_periods=10).mean()
            self.vol = np.log(self.adj).diff().rolling(20, min_periods=10).std().clip(lower=0.002)
            self.spread_bp = D.spread_rule(self.adv_usd)
            self.sector = pd.Series("X", index=self.close.columns)
            self.group = None
            self.eligible = self.adv_usd.notna()
    return P(prices)


def test_rank_normalise_is_gaussian_per_day():
    df = pd.DataFrame({"date": np.repeat(pd.bdate_range("2020-01-01", periods=5), 100), "x": np.random.default_rng(1).lognormal(size=500)})
    out = rank_normalise(df.copy(), ["x"])
    g = out.groupby("date")["x"]
    assert np.allclose(g.mean(), 0.0, atol=1e-9)
    assert np.all(g.std() > 0.9) and np.all(g.std() < 1.05)
    assert (out.sort_values(["date", "x"]).index == df.sort_values(["date", "x"]).index).all()     # order preserved


def test_features_have_no_lookahead(frame):
    prices = frame
    p1 = _panel(prices)
    f1 = build_frame(p1, horizons=[5])
    # perturb every price after a cut date; features on or before the cut must not change
    cut = pd.Timestamp("2016-06-01")
    p2 = prices.copy()
    m = p2["date"] > cut
    p2.loc[m, ["open", "high", "low", "close", "adjclose"]] *= 1.5
    p2.loc[m, "volume"] *= 3
    f2 = build_frame(_panel(p2), horizons=[5])
    a = f1[f1["date"] <= cut].set_index(["date", "symbol"])[FEATURES]
    b = f2[f2["date"] <= cut].set_index(["date", "symbol"])[FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_target_is_forward_and_demeaned(frame):
    p = _panel(frame)
    f = build_frame(p, horizons=[5])
    d = f["date"].iloc[len(f) // 2]
    g = f[f["date"] == d]
    assert abs(g["target_5"].mean()) < 1e-12
    sym = g["symbol"].iloc[0]
    i = p.dates.get_loc(d)
    expected = p.adj[sym].iloc[i + 5] / p.adj[sym].iloc[i] - 1.0
    raw = g.set_index("symbol")["target_5"] + (p.adj.loc[p.dates[i + 5]] / p.adj.loc[d] - 1.0).reindex(g["symbol"]).mean()
    assert abs(raw[sym] - expected) < 1e-9


def test_purged_splits_exclude_overlap():
    dates = np.array(sorted(pd.bdate_range("2018-01-01", periods=300)))
    h, e = 5, 5
    for tr, te in purged_kfold_splits(dates, 5, h, e):
        pos = {d: i for i, d in enumerate(dates)}
        lo, hi = pos[te[0]], pos[te[-1]]
        tr_pos = np.array([pos[d] for d in tr])
        assert not np.any((tr_pos >= lo - h - e) & (tr_pos <= hi + e))
        assert len(tr) + len(te) <= len(dates)


def test_purged_cv_removes_leak_that_kfold_keeps():
    """targets are overlapping 5-day sums of iid noise (no true signal); the features are a persistent per-name random
    walk and calendar time, which let a flexible learner find a test row's temporal neighbours in training.  Under
    shuffled K-fold those neighbours share four of the five target days and the IC is large; purged K-fold removes them"""
    rng = np.random.default_rng(3)
    n_names, n_days, h = 40, 400, 5
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    r = rng.normal(0, 0.01, (n_days + h, n_names))
    fwd = np.stack([r[t + 1: t + 1 + h].sum(axis=0) for t in range(n_days)])
    fwd = fwd - fwd.mean(axis=1, keepdims=True)
    rw = np.cumsum(rng.normal(0, 0.05, (n_days, n_names)), axis=0) + rng.normal(0, 3, (1, n_names))
    rows = pd.DataFrame({"date": np.repeat(dates, n_names), "symbol": np.tile([f"S{i}" for i in range(n_names)], n_days),
                         "rw": rw.ravel(), "t": np.repeat(np.arange(n_days) / n_days, n_names), "target_5": fwd.ravel()})
    cfg = LearnerConfig(gbm_iter=100, gbm_leaf=10, gbm_depth=6)
    ic = {}
    for scheme in ("kfold_shuffled", "purged"):
        z, d = cv_predict(rows, "target_5", scheme, 5, ["rw", "t"], cfg, kind="gbm")
        ic[scheme] = ic_by_day(d.assign(z=z.values), "z", "target_5").mean()
    assert ic["kfold_shuffled"] > 0.15
    assert abs(ic["purged"]) < 0.08
