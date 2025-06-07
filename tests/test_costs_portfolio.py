"""The cost model reproduces the algo-wheel's numbers, the book is dollar-neutral and capped, turnover and costs behave,
and net never exceeds gross."""
import numpy as np
import pandas as pd

from xsig.costs import CostModel
from xsig.portfolio import BookConfig, book, breakeven_capital, capacity_curve, run_book, weights_from_signal


def test_cost_model_matches_algo_wheel_reduced_form():
    m = CostModel.named("routed")
    # a 1 % of ADV order in a name with 150 bp daily vol: 0.863 + 0.2007 * 150 * sqrt(0.01) = 3.87 bp
    assert abs(m.cost_bp(150.0, 3.0, 0.01) - (0.8631 + 0.2007 * 150.0 * 0.1)) < 1e-9
    o = CostModel.named("observed")
    assert abs(o.cost_bp(150.0, 3.0, 0.01) - (-1.14 + 0.257 * 15.0 + 0.79 * 3.0)) < 1e-9
    x = np.array([0.001, 0.01, 0.1, 1.0])
    c = m.cost_bp(100.0, 2.0, x)
    assert np.all(np.diff(c) > 0)                       # monotone in size
    f = CostModel.named("routed", "futures")
    # ES: one tick 0.25 on 5000 = 0.5 bp; half a tick + $2 on $250,000 = 0.25 + 0.08 bp
    assert abs(f.cost_bp(100.0, 0.5, 0.0, 250000.0) - (0.25 + 1e4 * 2.0 / 250000.0)) < 1e-9


def test_weights_are_neutral_capped_and_normalised():
    rng = np.random.default_rng(0)
    z = rng.normal(size=150); z[0] = 8.0
    w = weights_from_signal(z, 1.0, 0.02)
    assert abs(w.sum()) < 1e-9 and abs(np.abs(w).sum() - 1.0) < 1e-6
    assert np.abs(w).max() <= 0.02 + 1e-9
    assert np.corrcoef(w, z)[0, 1] > 0.9


def _pred(n_days=60, n=100, seed=1, h=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    rows = []
    z_prev = rng.normal(size=n)
    for d in dates:
        z = 0.9 * z_prev + np.sqrt(1 - 0.81) * rng.normal(size=n); z_prev = z
        t = 0.002 * z + rng.normal(0, 0.03, n); t -= t.mean()
        rows.append(pd.DataFrame({"date": d, "symbol": [f"S{i}" for i in range(n)], "z": z, f"target_{h}": t, "sigma": 0.015, "adv_usd": rng.lognormal(20, 1, n), "spread_bp": 3.0}))
    return pd.concat(rows, ignore_index=True)


def test_book_accounting():
    p = _pred()
    r = run_book(p, "z", "target_5", CostModel.named("routed"), BookConfig(capital=1e9, gross=1.0, w_max=0.02, horizon=5))
    per = r["periods"]
    assert len(per) == 12
    assert np.all(per["cost"] >= 0) and np.all(per["net"] <= per["gross"])
    assert abs(per["turnover"].iloc[0] - 0.5) < 1e-6          # from cash: half the gross is bought, half sold
    assert per["turnover"].iloc[1:].mean() < 0.5             # a persistent signal trades less than a fresh book
    # net IC rows: the net target is the target less a signed drag, so long names are lower and short names higher
    nr = r["net_rows"]
    d = nr["net_target"] - nr["target_5"]
    assert np.all(d[nr["w"] > 0] <= 1e-12) and np.all(d[nr["w"] < 0] >= -1e-12)


def test_costs_rise_with_capital_and_breakeven_is_found():
    p = _pred(n_days=100, n=80, seed=2)
    cfg = BookConfig(capital=1e9, gross=1.0, w_max=0.05, horizon=5)
    curve = capacity_curve(p, "z", "target_5", CostModel.named("routed"), cfg, [1e7, 1e8, 1e9, 1e10, 1e11])
    assert np.all(np.diff(curve["cost_ann"].values) > 0)
    assert np.allclose(curve["gross_ann"].values, curve["gross_ann"].values[0])
    be = breakeven_capital(curve)
    assert be > 0
    s = book(p, "z", "target_5", CostModel.named("routed"), cfg, n_boot=50)["stats"]
    assert "net_ic" in s and s["net_ic"]["ic_mean"] <= s["net_ic"]["ic_hi"]
