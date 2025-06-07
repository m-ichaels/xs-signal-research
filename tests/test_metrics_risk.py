"""Inference and the risk model: Newey-West on iid and on autocorrelated data, the block bootstrap covers the mean,
IC of a perfect signal is one, neutralisation leaves zero exposure, factor returns recover a planted premium."""
import numpy as np
import pandas as pd

from xsig.metrics import block_bootstrap_mean, deflated_sharpe, ic_by_day, ic_summary, newey_west_t, signal_autocorrelation
from xsig.risk import attribute, factor_returns, neutralise


def test_newey_west_matches_plain_t_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.1, 1.0, 5000)
    m, t = newey_west_t(x, 0)
    assert abs(m - x.mean()) < 1e-12
    assert abs(t - x.mean() / x.std(ddof=0) * np.sqrt(len(x))) < 1e-9


def test_newey_west_widens_for_overlapping_series():
    """a 5-day moving sum of iid noise is MA(4): the HAC t with lag 5 is well below the naive t"""
    rng = np.random.default_rng(1)
    e = rng.normal(0.02, 1.0, 20000)
    x = np.convolve(e, np.ones(5), mode="valid")
    _, t0 = newey_west_t(x, 0)
    _, t5 = newey_west_t(x, 5)
    assert t5 < 0.6 * t0


def test_block_bootstrap_covers_mean():
    rng = np.random.default_rng(2)
    x = rng.normal(0.05, 1.0, 3000)
    lo, hi = block_bootstrap_mean(x, 11, 500, seed=1)
    assert lo < x.mean() < hi
    assert 0.02 < hi - lo < 0.2


def test_ic_of_perfect_and_random_signal():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=50)
    df = pd.DataFrame({"date": np.repeat(dates, 60), "symbol": np.tile([f"S{i}" for i in range(60)], 50), "target_5": rng.normal(size=3000)})
    df["z"] = df["target_5"]
    assert np.allclose(ic_by_day(df, "z", "target_5"), 1.0)
    df["z"] = rng.normal(size=3000)
    s = ic_summary(ic_by_day(df, "z", "target_5"), 5, 200)
    assert abs(s["ic_mean"]) < 0.08 and s["ic_lo"] < 0 < s["ic_hi"]


def test_signal_autocorrelation_of_persistent_signal():
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2020-01-01", periods=200)
    base = rng.normal(size=(200, 30)).cumsum(axis=0)          # random walk: high autocorrelation
    df = pd.DataFrame({"date": np.repeat(dates, 30), "symbol": np.tile([f"S{i}" for i in range(30)], 200), "z": base.ravel()})
    ac = signal_autocorrelation(df, "z", lags=(1, 5))
    assert ac["ac_1"] > 0.9 and ac["ac_5"] > 0.7 and ac["half_life_days"] > 5


def test_neutralise_removes_exposure_and_factor_returns_recover_premium():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=120)
    n = 80
    rows = []
    for d in dates:
        beta = rng.normal(1, 0.3, n); size = rng.normal(size=n); sector = rng.choice(["A", "B", "C"], n)
        z = 0.7 * size + rng.normal(size=n)                                  # signal loads on size
        target = 0.002 * size - 0.001 * beta + rng.normal(0, 0.02, n)         # planted premia
        rows.append(pd.DataFrame({"date": d, "symbol": [f"S{i}" for i in range(n)], "beta": beta, "size": size, "sector": sector, "z": z, "target_5": target}))
    df = pd.concat(rows, ignore_index=True)
    out = neutralise(df, "z", ["beta", "size"], sector=True, out="zn")
    expo = out.groupby("date").apply(lambda g: abs(np.dot(g["zn"], g["size"])) + abs(np.dot(g["zn"], g["beta"] - g["beta"].mean())), include_groups=False)
    assert expo.max() < 1e-6
    assert abs(out.groupby("date")["zn"].std().mean() - 1.0) < 0.05
    fr = factor_returns(df, "target_5", ["beta", "size"], sector=True)
    assert abs(fr["size"].mean() - 0.002) < 0.0005 and abs(fr["beta"].mean() + 0.001) < 0.0005
    port = 0.5 * fr["size"] + rng.normal(0, 1e-4, len(fr))
    port.index = fr.index
    a = attribute(port, fr, ["beta", "size"])
    assert abs(a["loadings"]["size"] - 0.5) < 0.05 and a["r2"] > 0.9


def test_deflated_sharpe_falls_with_trials():
    rng = np.random.default_rng(6)
    r = rng.normal(0.002, 0.02, 500)
    sr = r.mean() / r.std()
    d1 = deflated_sharpe(sr, len(r), 1, 0.001, r)
    d50 = deflated_sharpe(sr, len(r), 50, 0.001, r)
    assert 0 <= d50["dsr"] <= d1["dsr"] <= 1
