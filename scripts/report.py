#!/usr/bin/env python3
"""report.pdf from results/*.json and results/figures.   python scripts/report.py"""
import json
import os

import numpy as np
import pandas as pd
from fpdf import FPDF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")


def load(n):
    p = os.path.join(RES, f"{n}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def clean(s):
    return str(s).replace("−", "-").replace("–", "-").replace("—", "-").encode("latin-1", "replace").decode("latin-1")


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, "xs-signal-research: cross-sectional ML signal research on equities, ETFs and index futures", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-12); self.set_font("Helvetica", "I", 8); self.cell(0, 8, f"{self.page_no()}", align="C")

    def heading(self, t, size=13):
        self.set_font("Helvetica", "B", size); self.multi_cell(0, 7, clean(t)); self.ln(1)

    def para(self, t):
        self.set_font("Helvetica", "", 9.5); self.multi_cell(0, 4.8, clean(t)); self.ln(2)

    def fig(self, name, caption, w=185):
        p = os.path.join(FIG, name)
        if os.path.exists(p):
            if self.get_y() > 190:
                self.add_page()
            self.image(p, w=w); self.set_font("Helvetica", "I", 8.5); self.multi_cell(0, 4.2, clean(caption)); self.ln(3)

    def tab(self, df, widths=None, fmt="{:.4f}", size=7.5):
        cols = list(df.columns)
        widths = widths or [34] + [max(14, int(150 / max(len(cols), 1)))] * len(cols)
        if self.get_y() > 240:
            self.add_page()
        self.set_font("Helvetica", "B", size)
        self.cell(widths[0], 5, clean(df.index.name or "")[:22], border="B")
        for c, w in zip(cols, widths[1:]):
            self.cell(w, 5, clean(c)[:20], border="B")
        self.ln()
        self.set_font("Helvetica", "", size)
        for i, r in df.iterrows():
            self.cell(widths[0], 4.6, clean(i)[:30])
            for c, w in zip(cols, widths[1:]):
                v = r[c]
                self.cell(w, 4.6, clean(fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else v)[:18])
            self.ln()
        self.ln(3)


S, V, A, N, K, T = (load(n) for n in ("signal", "validate", "ablate", "neutralise", "costs", "transfer"))
pdf = PDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
pdf.heading("How much of a cross-sectional equity signal survives honest validation, and does it carry to ETFs and index futures?", 14)
ic = S["ic"]; h = S["horizon"]
pdf.para(f"Question.  The algo-wheel project built an ordinary machine-learning signal on {S['n_names']} US large caps (eight price-and-volume features, ridge and gradient boosting, refit each January) to study what execution costs do to it.  Here the signal itself is the object: how much of its information coefficient is real once validation is honest, which features carry it, how fast it decays, what a risk model leaves of it, what the algo-wheel's own cost model leaves of it at a stated capital, and whether the same feature set, fitted on equities, says anything about ETFs and futures.")
rk = K.get("models", {}).get("routed", {}); nic = rk.get("net_ic", {})
lad5 = V.get("ladder", {}).get(str(h), {})
tr = T.get("etf", {}); fu = T.get("futures", {})
pdf.para(f"Answer.  (1) Out of sample over {S['years'][0]}-{S['years'][1]} the {h}-day IC is {ic['ic_mean']:.4f} with a block-bootstrap band [{ic['ic_lo']:.4f}, {ic['ic_hi']:.4f}] and a Newey-West t of {ic['t_nw']:.1f} over {ic['n_days']:,} days; the in-sample IC of the training windows runs {np.mean(list(S['ic_train_by_year'].values())):.3f}.  "
         + (f"The same learner reports {lad5.get('kfold_shuffled', {}).get('ic_mean', float('nan')):.3f} under shuffled K-fold, {lad5.get('purged', {}).get('ic_mean', float('nan')):.4f} under purged and embargoed K-fold and {lad5.get('walk_forward', {}).get('ic_mean', float('nan')):.4f} walk-forward: the leak is the difference between the first and the second.  " if lad5 else "")
         + f"(2) The signal's rank autocorrelation is {S['autocorrelation']['ac_5']:.2f} at five days (half-life {S['autocorrelation']['half_life_days']:.0f} days), and its IC against a single day's return falls from {S['decay']['1']['ic']:.4f} one day ahead to {S['decay']['10']['ic']:.4f} ten days ahead.  "
         + (f"(3) Neutralising beta, size and sector leaves {N['signals']['z_base']['ic']['ic_mean']:.4f}; neutralising momentum and volatility as well leaves {N['signals']['z_style']['ic']['ic_mean']:.4f}, and the gross rank book has a factor R-squared of {N['attribution']['r2']:.2f} with an alpha t of {N['attribution']['alpha_t']:.1f}.  " if N else "")
         + (f"(4) At ${K['book']['capital'] / 1e9:.0f}bn the rank book earns {rk.get('gross_ann', 0) * 100:.1f} % a year gross (Sharpe {rk.get('sharpe_gross', 0):.2f}) and {rk.get('net_ann', 0) * 100:.1f} % net of the routed cost model (Sharpe {rk.get('sharpe_net', 0):.2f}) at {rk.get('turnover', 0) * 100:.0f} % turnover per rebalance; the net IC is {nic.get('ic_mean', float('nan')):.4f} [{nic.get('ic_lo', float('nan')):.4f}, {nic.get('ic_hi', float('nan')):.4f}]; breakeven capital {('$' + format(K['breakeven_capital'] / 1e9, '.1f') + 'bn') if np.isfinite(K.get('breakeven_capital', np.inf)) else 'beyond the grid'}; the deflated Sharpe over {K['deflated']['n_trials']} trials is {K['deflated']['dsr']:.2f}.  " if K else "")
         + (f"(5) The equity-fitted model applied unchanged gives an IC of {tr.get('equity_model', {}).get('ic', {}).get('ic_mean', float('nan')):.4f} on {tr.get('n_symbols', 0)} ETFs and {fu.get('equity_model', {}).get('ic', {}).get('ic_mean', float('nan')):.4f} on {fu.get('n_symbols', 0)} futures; refitted within each universe, {tr.get('own_model', {}).get('ic', {}).get('ic_mean', float('nan')):.4f} and {fu.get('own_model', {}).get('ic', {}).get('ic_mean', float('nan')):.4f}." if T else ""))
pdf.fig("signal.png", f"Figure 1.  The signal out of sample: cumulative and rolling IC; IC by year against the in-sample IC of each training window; IC decay by days ahead; a model per horizon with the signal's five-day autocorrelation.")
pdf.heading("Validation ladder and ablations", 11)
for key, lab in (("ladder", "blend"), ("ladder_flexible", "flexible gbm")):
    for hh, lad in sorted(V.get(key, {}).items(), key=lambda kv: int(kv[0])):
        df = pd.DataFrame(lad).T; df.index.name = f"{lab}, h = {hh}"
        pdf.tab(df[["ic_mean", "ic_sd", "n_days"]], [40, 30, 30, 30])
if A:
    v = pd.DataFrame(A["variants"]).T[["ic_mean", "t_nw", "delta_vs_full", "delta_t"]]; v.index.name = "variant"
    pdf.tab(v, [40, 28, 28, 28, 28])
pdf.fig("validation.png", "Figure 2.  Left: the same learner under five validation schemes.  Right: IC change of each ablation against the full model.")
if N:
    pdf.heading("Neutralisation", 11)
    rows = {v["label"]: {"ic": v["ic"]["ic_mean"], "lo": v["ic"]["ic_lo"], "hi": v["ic"]["ic_hi"], "t": v["ic"]["t_nw"]} for v in N["signals"].values()}
    df = pd.DataFrame(rows).T; df.index.name = "signal"
    pdf.tab(df, [60, 24, 24, 24, 24])
    pdf.fig("neutral.png", "Figure 3.  IC after each neutralisation, and the raw signal's correlation with the risk model's exposures.")
if K:
    pdf.heading("Costs, capacity and horizon", 11)
    m = pd.DataFrame(K["models"]).T[["gross_ann", "net_ann", "cost_ann", "sharpe_gross", "sharpe_net", "turnover", "avg_cost_bp"]]; m.index.name = "cost model"
    pdf.tab(m, [30, 22, 22, 22, 22, 22, 22, 22])
    bh = pd.DataFrame(K["by_horizon"]).T[["gross_ann", "net_ann", "sharpe_gross", "sharpe_net", "turnover", "avg_cost_bp"]]; bh.index.name = "horizon"
    pdf.tab(bh, [30, 24, 24, 24, 24, 24, 24])
    pdf.fig("costs.png", "Figure 4.  The rank book gross and net at $1bn; the capacity curve; each horizon's model in its own book with its turnover.")
if T:
    pdf.heading("New products", 11)
    rows = []
    for u, r in T.items():
        if not isinstance(r, dict) or "n_symbols" not in r:
            continue
        for key in ("equity_model", "own_model"):
            if key in r and "ic" in r[key]:
                icx = r[key]["ic"]; b = r[key].get("book", {})
                rows.append({"universe / model": f"{u}, {key.replace('_', ' ')}", "n": r["n_symbols"], "ic": icx["ic_mean"], "lo": icx.get("ic_lo", np.nan), "hi": icx.get("ic_hi", np.nan), "t": icx["t_nw"], "net_ann": b.get("net_ann", np.nan), "sharpe_net": b.get("sharpe_net", np.nan)})
    df = pd.DataFrame(rows).set_index("universe / model")
    pdf.tab(df, [50, 16, 20, 20, 20, 20, 22, 22])
    pdf.fig("transfer.png", "Figure 5.  The equity-fitted model applied unchanged and refitted within each universe; univariate IC of each feature by universe.")
pdf.heading("What is real and what is simulated", 11)
pdf.para("Real: daily bars from the Yahoo chart API for 156 US large caps (the algo-wheel panel), 92 ETFs and 34 CME/ICE continuous futures; the features, targets, models and every statistic.  Modelled: the cost of trading, from the algo-wheel's pre-trade model, which was itself fitted on a simulated market at the published scale of Almgren et al. (2005); the futures cost is half a tick plus a commission because the recorded contract volumes are not reliable enough for an impact term.  The equity universe is today's large caps, so the gross numbers carry survivorship bias; futures are unadjusted front-month series, so roll gaps sit in their returns.")
pdf.para("Method summary.  Features rank-normalised to N(0,1) per day; target the h-day forward return demeaned across the eligible universe; ridge (alpha 10) and gradient boosting (150 trees, depth 3) z-scored per day and averaged; refits each January on all earlier data with an embargo of h + 5 days.  IC = daily Spearman correlation; t from a Bartlett HAC variance at lag h; bands from a moving-block bootstrap with blocks of 2h + 1 days.  Purged K-fold drops training dates within h + 5 days of the test block on either side.  Neutralisation is the residual of a daily cross-sectional regression on the exposures; factor returns from the same regression on the forward return.  The book is rank-weighted, dollar-neutral, capped at 2 %, rebalanced every h days with drift, and each rebalance pays the cost model per name at the order's share of ADV.  Deflated Sharpe after Bailey and Lopez de Prado (2014).")
out = os.path.join(ROOT, "report.pdf")
pdf.output(out)
print(out)
