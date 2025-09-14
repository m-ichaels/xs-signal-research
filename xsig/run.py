"""Pipeline.   python -m xsig signal | validate | ablate | neutralise | costs | transfer | all [--quick]
               python -m xsig sql "SELECT ..."

Each step writes results/<step>.json (and parquets for the notebook); `all` runs them in order.  --quick runs the
last years only with a smaller learner and fewer bootstraps: the CI configuration."""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from .costs import CostModel
from .cv import validation_ladder
from .data import RES, ROOT, load_panel, store
from .features import FEATURES, GROUPS, build_frame
from .metrics import by_year, decile_spread, deflated_sharpe, ic_by_day, ic_decay, ic_summary, signal_autocorrelation
from .models import LearnerConfig, walk_forward
from .portfolio import BookConfig, book, breakeven_capital, capacity_curve
from .risk import BASE, STYLE, add_beta, attribute, factor_returns, neutralise
from .transfer import transfer_test

CFG_PATH = os.path.join(ROOT, "configs", "pipeline.json")


def load_cfg(quick: bool) -> dict:
    with open(CFG_PATH) as f:
        cfg = json.load(f)
    if quick:
        q = cfg.pop("quick")
        cfg["learner"]["gbm_iter"] = q.pop("gbm_iter")
        cfg.update(q)
    else:
        cfg.pop("quick", None)
    cfg["quick"] = quick
    return cfg


def learner_cfg(cfg) -> LearnerConfig:
    l = cfg["learner"]
    return LearnerConfig(ridge_alpha=l["ridge_alpha"], gbm_iter=l["gbm_iter"], gbm_lr=l["gbm_lr"], gbm_depth=l["gbm_depth"], gbm_leaf=l["gbm_leaf"], seed=l["seed"])


def dump(name: str, obj):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"{name}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (pd.Series, pd.DataFrame)):
        return o.to_dict()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def frame_path(universe: str) -> str:
    return os.path.join(RES, f"frame_{universe}.parquet")


def get_frame(universe: str, rebuild: bool = False) -> pd.DataFrame:
    p = frame_path(universe)
    if os.path.exists(p) and not rebuild:
        return pd.read_parquet(p)
    panel = load_panel(universe)
    df = build_frame(panel)
    if universe == "equity":
        df = add_beta(df, panel)
    df.to_parquet(p, index=False)
    return df


def tgt(h: int) -> str:
    return f"target_{h}"


# ----------------------------------------------------------------------------------------------------------------- signal
def cmd_signal(cfg):
    t0 = time.monotonic()
    h, y0, y1, nb = cfg["horizon"], cfg["first_year"], cfg["last_year"], cfg["n_boot"]
    lc = learner_cfg(cfg)
    panel = load_panel("equity")
    df = get_frame("equity", rebuild=True)
    print(f"equity frame: {len(df):,} rows, {df['symbol'].nunique()} names, {df['date'].min().date()} to {df['date'].max().date()}")
    models = {}
    pred = walk_forward(df, y0, y1, tgt(h), FEATURES, lc, models=models, train_ic=True)
    with open(os.path.join(RES, "models_equity.pkl"), "wb") as f:
        pickle.dump(models, f)
    pred.to_parquet(os.path.join(RES, "pred_equity.parquet"), index=False)
    ic = ic_by_day(pred, "z", tgt(h))
    out = {"horizon": h, "years": [y0, y1], "n_rows": int(len(pred)), "n_names": int(pred["symbol"].nunique()),
           "ic": ic_summary(ic, h, nb), "by_year": by_year(ic).round(4).to_dict(orient="index"),
           "ic_train_by_year": pred.groupby(pred["date"].dt.year)["ic_train"].first().round(4).to_dict()}
    d = decile_spread(pred, "z", tgt(h))
    out["deciles"] = {"by_decile": d["by_decile"], "spread_mean": d["spread_mean"], "spread_t": d["spread_t"]}
    out["decay"] = ic_decay(pred, panel, "z", 21).round(5).to_dict(orient="index")
    out["autocorrelation"] = signal_autocorrelation(pred, "z")
    # the components on their own
    for kind in ("ridge", "gbm"):
        pk = walk_forward(df, y0, y1, tgt(h), FEATURES, lc, kind=kind)
        out[f"ic_{kind}"] = ic_summary(ic_by_day(pk, "z", tgt(h)), h, 0)
    # the horizon question: a model per horizon, each evaluated at its own horizon and at the others
    hz = {}
    for hh in cfg["horizons"]:
        ph = pred if hh == h else walk_forward(df, y0, y1, tgt(hh), FEATURES, lc)
        ph.to_parquet(os.path.join(RES, f"pred_equity_h{hh}.parquet"), index=False)
        row = {"own": ic_summary(ic_by_day(ph, "z", tgt(hh)), hh, nb)}
        row["cross"] = {str(k): float(ic_by_day(ph, "z", tgt(k)).mean()) for k in cfg["horizons"]}
        row["autocorrelation"] = signal_autocorrelation(ph, "z", lags=(1, 5, 21))
        hz[str(hh)] = row
    out["horizons"] = hz
    out["seconds"] = time.monotonic() - t0
    dump("signal", out)
    print(f"signal: IC {out['ic']['ic_mean']:.4f} [{out['ic']['ic_lo']:.4f}, {out['ic']['ic_hi']:.4f}] t={out['ic']['t_nw']:.1f} over {out['ic']['n_days']} days ({out['seconds']:.0f}s)")


# --------------------------------------------------------------------------------------------------------------- validate
def cmd_validate(cfg):
    t0 = time.monotonic()
    df = get_frame("equity")
    lc = learner_cfg(cfg)
    flex = LearnerConfig(gbm_iter=max(lc.gbm_iter * 2, 60), gbm_depth=6, gbm_leaf=20, seed=lc.seed)     # a learner flexible enough to memorise
    out = {"years": [cfg["first_year"], cfg["last_year"]], "n_splits": cfg["cv_splits"], "ladder": {}, "ladder_flexible": {}}
    for hh in sorted({cfg["horizon"], max(cfg["horizons"])}):
        out["ladder"][str(hh)] = validation_ladder(df, tgt(hh), cfg["first_year"], cfg["last_year"], cfg["cv_splits"], lc)
        print(f"  h={hh}: " + ", ".join(f"{k} {v['ic_mean']:.4f}" for k, v in out["ladder"][str(hh)].items()))
        out["ladder_flexible"][str(hh)] = validation_ladder(df, tgt(hh), cfg["first_year"], cfg["last_year"], cfg["cv_splits"], flex, kind="gbm")
        print(f"  h={hh} flexible gbm: " + ", ".join(f"{k} {v['ic_mean']:.4f}" for k, v in out["ladder_flexible"][str(hh)].items()))
    out["seconds"] = time.monotonic() - t0
    dump("validate", out)


# ----------------------------------------------------------------------------------------------------------------- ablate
def cmd_ablate(cfg):
    t0 = time.monotonic()
    df = get_frame("equity")
    h, y0, y1, nb = cfg["horizon"], cfg["first_year"], cfg["last_year"], cfg["n_boot"]
    lc = learner_cfg(cfg)
    variants = {"full": FEATURES}
    for f in FEATURES:
        variants[f"drop_{f}"] = [x for x in FEATURES if x != f]
    for g, fs in GROUPS.items():
        variants[f"only_{g}"] = fs
        variants[f"drop_{g}"] = [x for x in FEATURES if x not in fs]
    res = {}
    base_ic = None
    for name, fs in variants.items():
        p = walk_forward(df, y0, y1, tgt(h), fs, lc)
        ic = ic_by_day(p, "z", tgt(h))
        s = ic_summary(ic, h, nb if name == "full" else 0)
        if name == "full":
            base_ic = ic
        else:
            diff = (ic - base_ic).dropna()
            from .metrics import newey_west_t
            m, t = newey_west_t(diff.values, h)
            s["delta_vs_full"], s["delta_t"] = m, t
        d = decile_spread(p, "z", tgt(h))
        s["decile_spread"] = d["spread_mean"]
        res[name] = s
        print(f"  {name:22s} IC {s['ic_mean']:.4f}  spread {d['spread_mean'] * 1e4:.0f} bp")
    # single features, linear (the univariate view)
    single = {}
    for f in FEATURES:
        p = walk_forward(df, y0, y1, tgt(h), [f], lc, kind="ridge")
        single[f] = ic_summary(ic_by_day(p, "z", tgt(h)), h, 0)
    out = {"horizon": h, "variants": res, "single_feature": single, "n_trials": len(variants) + len(single) + len(cfg["horizons"]) + 2, "seconds": time.monotonic() - t0}
    dump("ablate", out)


# ------------------------------------------------------------------------------------------------------------- neutralise
def cmd_neutralise(cfg):
    t0 = time.monotonic()
    h, nb = cfg["horizon"], cfg["n_boot"]
    pred = pd.read_parquet(os.path.join(RES, "pred_equity.parquet"))
    pred = neutralise(pred, "z", BASE, sector=True, out="z_base")
    pred = neutralise(pred, "z", BASE + STYLE, sector=True, out="z_style")
    pred = neutralise(pred, "z", [], sector=True, out="z_sector")
    pred.to_parquet(os.path.join(RES, "pred_equity.parquet"), index=False)
    out = {"horizon": h, "signals": {}}
    for col, label in (("z", "raw"), ("z_sector", "sector-neutral"), ("z_base", "beta, size and sector neutral"), ("z_style", "plus momentum and volatility neutral")):
        ic = ic_by_day(pred, col, tgt(h))
        d = decile_spread(pred, col, tgt(h))
        out["signals"][col] = {"label": label, "ic": ic_summary(ic, h, nb), "decile_spread": d["spread_mean"], "decile_t": d["spread_t"], "by_year": by_year(ic)["ic_mean"].round(4).to_dict()}
        print(f"  {label:40s} IC {out['signals'][col]['ic']['ic_mean']:.4f}")
    # exposures of the raw signal: the mean cross-sectional correlation with each factor
    out["exposure_corr"] = {f: float(pred.groupby("date").apply(lambda g: g["z"].corr(g[f]), include_groups=False).mean()) for f in BASE + STYLE}
    # factor returns and the attribution of the raw rank book (gross, at the rebalance dates)
    fr = factor_returns(pred, tgt(h), BASE + STYLE, sector=True)
    fr.to_parquet(os.path.join(RES, "factor_returns.parquet"))
    ppy = 252.0 / h
    out["factor_premia"] = {f: {"mean_ann": float(fr[f].mean() * ppy), "sharpe": float(fr[f].mean() / fr[f].std() * np.sqrt(ppy / h))} for f in BASE + STYLE}
    b = book(pred, "z", tgt(h), CostModel.named("routed"), BookConfig(**cfg["book"], horizon=h), offsets=1, n_boot=0)
    per = b["periods"]
    out["attribution"] = attribute(per["gross"], fr, BASE + STYLE)
    out["attribution_style_only"] = attribute(per["gross"], fr, STYLE)
    out["seconds"] = time.monotonic() - t0
    dump("neutralise", out)


# ------------------------------------------------------------------------------------------------------------------ costs
def cmd_costs(cfg):
    t0 = time.monotonic()
    h, nb = cfg["horizon"], cfg["n_boot"]
    bc = BookConfig(**cfg["book"], horizon=h)
    pred = pd.read_parquet(os.path.join(RES, "pred_equity.parquet"))
    out = {"horizon": h, "book": cfg["book"], "models": {}}
    for name in cfg["cost_models"]:
        m = CostModel.named(name)
        b = book(pred, "z", tgt(h), m, bc, n_boot=nb)
        out["models"][name] = b["stats"]
        if name == cfg["cost_models"][0]:
            b["periods"].to_parquet(os.path.join(RES, "book_periods.parquet"))
            curve = capacity_curve(pred, "z", tgt(h), m, bc, cfg["capital_grid"])
            out["capacity"] = curve.round(6).to_dict(orient="index")
            out["breakeven_capital"] = breakeven_capital(curve)
        print(f"  {name}: gross {b['stats']['gross_ann'] * 100:.2f}%  net {b['stats']['net_ann'] * 100:.2f}%  Sharpe {b['stats']['sharpe_gross']:.2f} -> {b['stats']['sharpe_net']:.2f}  turnover {b['stats']['turnover']:.2f}")
    # neutralised signals through the same book
    out["neutral"] = {}
    for col in ("z_sector", "z_base", "z_style"):
        if col in pred:
            out["neutral"][col] = book(pred, col, tgt(h), CostModel.named("routed"), bc, n_boot=0)["stats"]
    # horizon versus turnover: each horizon's own model in its own book
    out["by_horizon"] = {}
    for hh in cfg["horizons"]:
        p = os.path.join(RES, f"pred_equity_h{hh}.parquet")
        if not os.path.exists(p):
            continue
        ph = pd.read_parquet(p)
        s = book(ph, "z", tgt(hh), CostModel.named("routed"), BookConfig(**cfg["book"], horizon=hh), n_boot=0)["stats"]
        out["by_horizon"][str(hh)] = s
        print(f"  h={hh:2d}: gross {s['gross_ann'] * 100:.2f}%  net {s['net_ann'] * 100:.2f}%  turnover {s['turnover']:.2f}")
    # the multiple-testing question: the best net Sharpe across everything tried, deflated
    per = pd.read_parquet(os.path.join(RES, "book_periods.parquet"))
    n_ablate = 0
    if os.path.exists(os.path.join(RES, "ablate.json")):
        with open(os.path.join(RES, "ablate.json")) as f:
            n_ablate = json.load(f)["n_trials"]
    trials = 1 + len(out["by_horizon"]) + len(out["neutral"]) + n_ablate
    srs = [v["sharpe_net"] for v in out["by_horizon"].values()] + [v["sharpe_net"] for v in out["neutral"].values()] + [out["models"]["routed"]["sharpe_net"]]
    ppy = 252.0 / h
    sr_period = per["net"].mean() / per["net"].std()
    out["deflated"] = deflated_sharpe(float(sr_period), len(per), trials, float(np.var(np.array(srs) / np.sqrt(ppy))) if len(srs) > 1 else 0.01, per["net"].values)
    out["seconds"] = time.monotonic() - t0
    dump("costs", out)


# --------------------------------------------------------------------------------------------------------------- transfer
def cmd_transfer(cfg):
    t0 = time.monotonic()
    h, nb = cfg["horizon"], cfg["n_boot"]
    with open(os.path.join(RES, "models_equity.pkl"), "rb") as f:
        models = pickle.load(f)
    frames = {u: get_frame(u, rebuild=(u != "equity")) for u in ("equity", "etf", "futures")}
    for u, f in frames.items():
        print(f"  {u}: {len(f):,} rows, {f['symbol'].nunique()} symbols")
    out = transfer_test(frames, models, tgt(h), cfg["first_year"], cfg["last_year"], learner_cfg(cfg), nb, cfg["transfer_capital"])
    for u, r in out.items():
        e = r.get("equity_model", r.get("own_model", {})).get("ic", {})
        o = r.get("own_model", {}).get("ic", {})
        print(f"  {u:8s} equity-model IC {e.get('ic_mean', float('nan')):.4f}   own-model IC {o.get('ic_mean', float('nan')):.4f}")
    out["seconds"] = time.monotonic() - t0
    dump("transfer", out)


def cmd_sql(q: str):
    con = store()
    print(con.execute(q).df().to_string())


STEPS = {"signal": cmd_signal, "validate": cmd_validate, "ablate": cmd_ablate, "neutralise": cmd_neutralise, "costs": cmd_costs, "transfer": cmd_transfer}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__); return
    cmd = argv[0]
    if cmd == "sql":
        cmd_sql(" ".join(argv[1:])); return
    quick = "--quick" in argv
    cfg = load_cfg(quick)
    os.makedirs(RES, exist_ok=True)
    steps = list(STEPS) if cmd == "all" else [cmd]
    for s in steps:
        print(f"== {s}" + (" (quick)" if quick else ""))
        STEPS[s](cfg)


if __name__ == "__main__":
    main()
