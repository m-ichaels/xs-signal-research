"""One short end-to-end pass on the committed data: frames for all three universes, a two-year walk-forward, the
equity model applied to ETFs and futures, the book, and the SQL store."""
import numpy as np

from xsig.costs import CostModel
from xsig.data import load_panel, store
from xsig.features import FEATURES, build_frame
from xsig.metrics import ic_by_day, ic_summary
from xsig.models import LearnerConfig, apply_models, walk_forward
from xsig.portfolio import BookConfig, book
from xsig.risk import BASE, add_beta, neutralise


def test_end_to_end_on_committed_data():
    cfg = LearnerConfig(gbm_iter=20)
    panel = load_panel("equity")
    df = add_beta(build_frame(panel, horizons=[5]), panel)
    assert df["symbol"].nunique() > 100 and df["date"].nunique() > 3000
    assert np.isfinite(df[FEATURES].values).all()
    models = {}
    pred = walk_forward(df, 2024, 2025, "target_5", FEATURES, cfg, models=models)
    assert set(models) == {2024, 2025}
    ic = ic_by_day(pred, "z", "target_5")
    s = ic_summary(ic, 5, 100)
    assert s["n_days"] > 450 and -0.2 < s["ic_mean"] < 0.2
    pred = neutralise(pred, "z", BASE, sector=True, out="zn")
    assert pred["zn"].notna().mean() > 0.99
    b = book(pred, "z", "target_5", CostModel.named("routed"), BookConfig(capital=1e9, gross=1.0, w_max=0.02, horizon=5), offsets=1, n_boot=0)["stats"]
    assert b["net_ann"] <= b["gross_ann"] and 0 < b["turnover"] < 1
    for u in ("etf", "futures"):
        f = build_frame(load_panel(u), horizons=[5])
        t = apply_models(f, models)
        assert t["z"].notna().all() and t["date"].dt.year.isin([2024, 2025]).all()
        assert len(ic_by_day(t, "z", "target_5")) > 400
    con = store()
    n = con.execute("SELECT COUNT(*) FROM equity").fetchone()[0]
    assert n > 500000
