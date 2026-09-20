#!/usr/bin/env python3
"""results/summary.md: every table in the README, generated from results/*.json.   python scripts/summarize.py"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(n):
    p = os.path.join(RES, f"{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def md(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in cols]) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(out)


def fmt_ic(s):
    lo, hi = s.get("ic_lo", np.nan), s.get("ic_hi", np.nan)
    band = f" [{lo:.4f}, {hi:.4f}]" if np.isfinite(lo) else ""
    return f"{s['ic_mean']:.4f}{band} (t = {s['t_nw']:.1f}, n = {s['n_days']})"


def main():
    S, V, A, N, K, T = (load(n) for n in ("signal", "validate", "ablate", "neutralise", "costs", "transfer"))
    L = []
    if S:
        L.append(f"# Summary\n\nHorizon {S['horizon']} days, out-of-sample years {S['years'][0]}–{S['years'][1]}, {S['n_names']} names, {S['n_rows']:,} name-days.\n")
        L.append("## Signal\n")
        L.append(f"- Blend IC: {fmt_ic(S['ic'])}; IR {S['ic']['ir_annual']:.2f}; positive on {S['ic']['share_positive'] * 100:.0f} % of days")
        L.append(f"- Ridge alone: {fmt_ic(S['ic_ridge'])}; gradient boosting alone: {fmt_ic(S['ic_gbm'])}")
        d = S["deciles"]
        L.append(f"- Decile 10 − decile 1: {d['spread_mean'] * 1e4:.0f} bp per {S['horizon']} days (t = {d['spread_t']:.1f})")
        ac = S["autocorrelation"]
        L.append(f"- Signal rank autocorrelation: 1 day {ac['ac_1']:.2f}, 5 days {ac['ac_5']:.2f}, 21 days {ac['ac_21']:.2f}; half-life {ac['half_life_days']:.1f} days\n")
        by = pd.DataFrame(S["by_year"]).T
        by["ic_train"] = pd.Series(S["ic_train_by_year"])
        by.index.name = "year"
        L.append("### IC by year\n\n" + md(by[["ic_mean", "ic_sd", "n_days", "ic_train"]]) + "\n")
        hz = S["horizons"]
        rows = []
        for h in sorted(hz, key=int):
            r = hz[h]["own"]
            rows.append({"horizon": int(h), "ic": r["ic_mean"], "lo": r["ic_lo"], "hi": r["ic_hi"], "t_nw": r["t_nw"], "ir_annual": r["ir_annual"], "ac_5": hz[h]["autocorrelation"]["ac_5"], **{f"ic_vs_h{k}": v for k, v in hz[h]["cross"].items()}})
        L.append("### A model per horizon\n\n" + md(pd.DataFrame(rows).set_index("horizon")) + "\n")
        dec = pd.DataFrame(S["decay"]).T; dec.index.name = "days ahead"
        L.append("### IC decay\n\n" + md(dec.head(10)) + "\n")
    if V:
        L.append("## Validation ladder\n")
        for key, lab in (("ladder", "regularised blend"), ("ladder_flexible", "flexible gradient boosting")):
            for h, lad in sorted(V.get(key, {}).items(), key=lambda kv: int(kv[0])):
                df = pd.DataFrame(lad).T; df.index.name = f"{lab}, h = {h}"
                L.append(md(df) + "\n")
    if A:
        L.append("## Ablations\n")
        v = pd.DataFrame(A["variants"]).T[["ic_mean", "t_nw", "decile_spread", "delta_vs_full", "delta_t"]]
        v.index.name = "variant"
        L.append(md(v) + "\n")
        s = pd.DataFrame(A["single_feature"]).T[["ic_mean", "t_nw", "ir_annual"]]; s.index.name = "single feature (ridge)"
        L.append(md(s) + "\n")
    if N:
        L.append("## Neutralisation\n")
        rows = {v["label"]: {"ic": v["ic"]["ic_mean"], "lo": v["ic"]["ic_lo"], "hi": v["ic"]["ic_hi"], "t_nw": v["ic"]["t_nw"], "decile_spread_bp": v["decile_spread"] * 1e4} for v in N["signals"].values()}
        df = pd.DataFrame(rows).T; df.index.name = "signal"
        L.append(md(df) + "\n")
        L.append("Correlation of the raw signal with each exposure: " + ", ".join(f"{k} {v:+.2f}" for k, v in N["exposure_corr"].items()) + "\n")
        fp = pd.DataFrame(N["factor_premia"]).T; fp.index.name = "factor"
        L.append("Factor premia (Fama–MacBeth, annualised):\n\n" + md(fp) + "\n")
        att = N["attribution"]
        L.append(f"Attribution of the gross rank book: alpha {att['alpha'] * 1e4:.1f} bp per period (t = {att['alpha_t']:.1f}), R² {att['r2']:.2f}, loadings " + ", ".join(f"{k} {v:+.3f}" for k, v in att["loadings"].items()) + "\n")
    if K:
        L.append("## Costs and capacity\n")
        m = pd.DataFrame(K["models"]).T
        m["net_ic"] = [v.get("net_ic", {}).get("ic_mean", np.nan) for v in K["models"].values()]
        m["net_ic_lo"] = [v.get("net_ic", {}).get("ic_lo", np.nan) for v in K["models"].values()]
        m["net_ic_hi"] = [v.get("net_ic", {}).get("ic_hi", np.nan) for v in K["models"].values()]
        m.index.name = "cost model"
        L.append(md(m[["gross_ann", "net_ann", "cost_ann", "sharpe_gross", "sharpe_net", "turnover", "avg_cost_bp", "max_dd_net", "net_ic", "net_ic_lo", "net_ic_hi"]]) + "\n")
        cap = pd.DataFrame(K["capacity"]).T; cap.index = [f"${float(c) / 1e9:.2f}bn" for c in cap.index]; cap.index.name = "capital"
        be = K["breakeven_capital"]
        L.append(f"Breakeven capital: {'$' + format(be / 1e9, '.1f') + 'bn' if np.isfinite(be) else 'beyond the grid'}\n\n" + md(cap[["gross_ann", "net_ann", "cost_ann", "sharpe_net", "turnover", "avg_cost_bp", "net_ic"]]) + "\n")
        bh = pd.DataFrame(K["by_horizon"]).T; bh.index.name = "horizon"
        L.append("By horizon (each horizon's own model, held for the horizon):\n\n" + md(bh[["gross_ann", "net_ann", "cost_ann", "sharpe_gross", "sharpe_net", "turnover", "avg_cost_bp"]]) + "\n")
        if K.get("neutral"):
            nb = pd.DataFrame(K["neutral"]).T; nb.index.name = "neutralised signal"
            L.append(md(nb[["gross_ann", "net_ann", "sharpe_gross", "sharpe_net", "turnover"]]) + "\n")
        d = K["deflated"]
        L.append(f"Deflated Sharpe of the net book: SR {d['sr'] * np.sqrt(252 / K['horizon']):.2f} annualised against an expected maximum of {d['sr0'] * np.sqrt(252 / K['horizon']):.2f} over {d['n_trials']} trials: DSR = {d['dsr']:.2f}\n")
    if T:
        L.append("## Transfer to ETFs and futures\n")
        rows = []
        for u, r in T.items():
            if not isinstance(r, dict) or "n_symbols" not in r:
                continue
            for key in ("equity_model", "own_model"):
                if key in r and "ic" in r[key]:
                    ic = r[key]["ic"]; b = r[key].get("book", {})
                    rows.append({"universe": u, "model": key.replace("_", " "), "n_symbols": r["n_symbols"], "ic": ic["ic_mean"], "lo": ic.get("ic_lo", np.nan), "hi": ic.get("ic_hi", np.nan), "t_nw": ic["t_nw"], "quintile_spread_bp": r[key]["quintile_spread"]["mean"] * 1e4,
                                 "gross_ann": b.get("gross_ann", np.nan), "net_ann": b.get("net_ann", np.nan), "sharpe_net": b.get("sharpe_net", np.nan), "turnover": b.get("turnover", np.nan)})
        L.append(md(pd.DataFrame(rows).set_index("universe")) + "\n")
        uni = {u: {f: r["univariate"][f]["ic"] for f in r["univariate"]} for u, r in T.items() if isinstance(r, dict) and "univariate" in r}
        df = pd.DataFrame(uni); df.index.name = "feature"
        L.append("Univariate IC by universe:\n\n" + md(df) + "\n")
        if "etf" in T and "equity_model_by_group" in T["etf"]:
            g = pd.DataFrame(T["etf"]["equity_model_by_group"]).T; g.index.name = "ETF group"
            L.append("Equity model on ETFs by group:\n\n" + md(g) + "\n")
    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    print("results/summary.md")


if __name__ == "__main__":
    main()
