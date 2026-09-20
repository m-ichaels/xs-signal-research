#!/usr/bin/env python3
"""Figures from results/*.json and the run parquets.   python scripts/plots.py  -> results/figures/*.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
C = ["#1f4e79", "#c0504d", "#4f9d69", "#8064a2", "#f79646", "#4bacc6", "#7f7f7f", "#9bbb59"]


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print(" ", name)


def fig_signal(S):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    # cumulative IC and rolling mean
    pred = pd.read_parquet(os.path.join(RES, "pred_equity.parquet"))
    from xsig.metrics import ic_by_day
    ic = ic_by_day(pred, "z", f"target_{S['horizon']}")
    a = ax[0, 0]
    a.plot(ic.index, ic.cumsum(), color=C[0], lw=1.2, label="cumulative daily IC")
    a2 = a.twinx(); a2.plot(ic.index, ic.rolling(126).mean(), color=C[1], lw=1.0, label="6-month rolling mean"); a2.set_ylabel("rolling mean IC"); a2.grid(False)
    a.set_title(f"Out-of-sample IC, h = {S['horizon']}: mean {S['ic']['ic_mean']:.3f} [{S['ic']['ic_lo']:.3f}, {S['ic']['ic_hi']:.3f}], t = {S['ic']['t_nw']:.1f}")
    a.set_ylabel("cumulative IC"); a.legend(loc="upper left", fontsize=7); a2.legend(loc="lower right", fontsize=7)
    # by year, with the in-sample IC of each training window
    a = ax[0, 1]
    by = pd.DataFrame(S["by_year"]).T
    tr = pd.Series(S["ic_train_by_year"]); tr.index = tr.index.astype(int)
    a.bar(by.index.astype(int), by["ic_mean"], color=C[0], label="out of sample")
    a.plot(tr.index, tr.values, "o-", color=C[1], ms=3, lw=1, label="in sample (training window)")
    yrs = by.index.astype(int); a.set_xticks(yrs[::2]); a.set_xticklabels(yrs[::2], fontsize=7)
    a.axhline(0, color="k", lw=0.6); a.set_title("IC by year"); a.legend(fontsize=7)
    # decay
    a = ax[1, 0]
    dec = pd.DataFrame(S["decay"]).T; dec.index = dec.index.astype(int)
    a.bar(dec.index, dec["ic"], color=C[0], label="IC vs single-day return k days ahead")
    a.plot(dec.index, dec["cum_ic"], "-o", color=C[1], ms=2.5, lw=1, label="cumulative")
    a.set_xlabel("days ahead k"); a.set_title("IC decay"); a.legend(fontsize=7); a.axhline(0, color="k", lw=0.6)
    # horizons: own IC and IR, with autocorrelation
    a = ax[1, 1]
    hz = S["horizons"]
    hs = sorted(int(h) for h in hz)
    own = [hz[str(h)]["own"]["ic_mean"] for h in hs]; lo = [hz[str(h)]["own"]["ic_lo"] for h in hs]; hi = [hz[str(h)]["own"]["ic_hi"] for h in hs]
    a.errorbar(hs, own, yerr=[np.array(own) - np.array(lo), np.array(hi) - np.array(own)], fmt="o-", color=C[0], capsize=3, label="IC at the model's own horizon")
    ac = [hz[str(h)]["autocorrelation"]["ac_5"] for h in hs]
    a2 = a.twinx(); a2.plot(hs, ac, "s--", color=C[2], ms=4, lw=1, label="signal 5-day autocorrelation"); a2.set_ylim(0, 1); a2.grid(False); a2.set_ylabel("autocorrelation")
    a.set_xscale("log"); a.set_xticks(hs); a.set_xticklabels(hs); a.set_xlabel("horizon h (days)"); a.set_title("A model per horizon")
    a.legend(loc="upper left", fontsize=7); a2.legend(loc="lower right", fontsize=7)
    save(fig, "signal.png")


def fig_validation(V, A):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    schemes = ["in_sample", "kfold_shuffled", "kfold_blocked", "purged", "walk_forward"]
    labels = ["in-sample", "K-fold\nshuffled", "K-fold\ndate blocks", "purged +\nembargoed", "walk-\nforward"]
    for a, key, title in ((ax[0], "ladder", "The ladder: the regularised blend (ridge + shallow boosting)"), (ax[1], "ladder_flexible", "The ladder: a flexible learner (deep boosting, small leaves)")):
        lad = V.get(key, {})
        if not lad:
            continue
        x = np.arange(len(schemes)); w = 0.8 / len(lad)
        for i, (h, r) in enumerate(sorted(lad.items(), key=lambda kv: int(kv[0]))):
            a.bar(x + i * w, [r[s]["ic_mean"] for s in schemes], w, color=C[i], label=f"h = {h}")
        a.set_xticks(x + w * (len(lad) - 1) / 2); a.set_xticklabels(labels, fontsize=7); a.set_ylabel("mean daily IC"); a.set_title(title, fontsize=8.5); a.legend(fontsize=7); a.axhline(0, color="k", lw=0.6)
    a = ax[2]
    if A:
        v = A["variants"]
        names = [k for k in v if k != "full"]
        base = v["full"]["ic_mean"]
        d = [v[k]["delta_vs_full"] for k in names]
        col = [C[1] if k.startswith("drop_") and k[5:] in A["single_feature"] else (C[2] if k.startswith("only_") else C[3]) for k in names]
        a.barh(range(len(names)), d, color=col)
        a.set_yticks(range(len(names))); a.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)
        a.axvline(0, color="k", lw=0.6); a.set_xlabel(f"IC change against the full model ({base:.4f})"); a.set_title("Ablations (red: drop one feature; green: one group only; purple: drop a group)", fontsize=8.5)
        a.invert_yaxis()
    save(fig, "validation.png")


def fig_neutral(N):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    a = ax[0]
    sig = N["signals"]
    keys = list(sig)
    m = [sig[k]["ic"]["ic_mean"] for k in keys]; lo = [sig[k]["ic"]["ic_lo"] for k in keys]; hi = [sig[k]["ic"]["ic_hi"] for k in keys]
    a.errorbar(range(len(keys)), m, yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)], fmt="o", color=C[0], capsize=4)
    a.set_xticks(range(len(keys))); a.set_xticklabels([sig[k]["label"].replace(" and ", "\n& ") for k in keys], fontsize=7); a.axhline(0, color="k", lw=0.6)
    a.set_ylabel("mean IC"); a.set_title("IC after neutralisation against the risk model")
    a = ax[1]
    ex = N["exposure_corr"]
    a.bar(range(len(ex)), list(ex.values()), color=C[3])
    a.set_xticks(range(len(ex))); a.set_xticklabels(list(ex)); a.axhline(0, color="k", lw=0.6)
    att = N["attribution"]
    a.set_title(f"Raw signal's correlation with the factors; book alpha t = {att['alpha_t']:.1f}, factor R² = {att['r2']:.2f}")
    save(fig, "neutral.png")


def fig_costs(K):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    a = ax[0]
    per = pd.read_parquet(os.path.join(RES, "book_periods.parquet"))
    a.plot(per.index, (1 + per["gross"]).cumprod() - 1, color=C[0], label=f"gross ({K['models']['routed']['gross_ann'] * 100:.1f} %/yr, Sharpe {K['models']['routed']['sharpe_gross']:.2f})")
    a.plot(per.index, (1 + per["net"]).cumprod() - 1, color=C[1], label=f"net of the routed cost model ({K['models']['routed']['net_ann'] * 100:.1f} %/yr, Sharpe {K['models']['routed']['sharpe_net']:.2f})")
    a.set_title(f"Rank book, ${K['book']['capital'] / 1e9:.0f}bn gross, h = {K['horizon']}; turnover {K['models']['routed']['turnover'] * 100:.0f} % per rebalance"); a.legend(fontsize=7); a.set_ylabel("cumulative return")
    a = ax[1]
    cap = pd.DataFrame(K["capacity"]).T; cap.index = cap.index.astype(float)
    a.plot(cap.index, cap["gross_ann"] * 100, "o-", color=C[0], label="gross"); a.plot(cap.index, cap["net_ann"] * 100, "o-", color=C[1], label="net")
    a.set_xscale("log"); a.axhline(0, color="k", lw=0.6); a.set_xlabel("capital ($)"); a.set_ylabel("% per year")
    if "net_ic" in cap:
        a2 = a.twinx(); a2.plot(cap.index, cap["net_ic"], "s--", color=C[2], ms=4, lw=1, label="net IC"); a2.set_ylabel("net IC"); a2.grid(False); a2.axhline(0, color=C[2], lw=0.5, ls=":"); a2.legend(loc="lower left", fontsize=7)
    be = K["breakeven_capital"]
    a.set_title("Capacity: " + (f"breakeven at ${be / 1e9:.1f}bn" if np.isfinite(be) else "breakeven beyond the grid")); a.legend(fontsize=7)
    a = ax[2]
    bh = K["by_horizon"]; hs = sorted(int(h) for h in bh)
    a.plot(hs, [bh[str(h)]["gross_ann"] * 100 for h in hs], "o-", color=C[0], label="gross")
    a.plot(hs, [bh[str(h)]["net_ann"] * 100 for h in hs], "o-", color=C[1], label="net")
    a2 = a.twinx(); a2.plot(hs, [bh[str(h)]["turnover"] * 100 for h in hs], "s--", color=C[6], ms=4, lw=1, label="turnover per rebalance"); a2.set_ylabel("% of book"); a2.grid(False)
    a.set_xscale("log"); a.set_xticks(hs); a.set_xticklabels(hs); a.set_xlabel("horizon = holding period (days)"); a.axhline(0, color="k", lw=0.6)
    a.set_title("Each horizon's model in its own book"); a.legend(loc="upper left", fontsize=7); a2.legend(loc="upper right", fontsize=7); a.set_ylabel("% per year")
    save(fig, "costs.png")


def fig_transfer(T):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    a = ax[0]
    us = [u for u in ("equity", "etf", "futures") if u in T]
    x = np.arange(len(us))
    def get(u, key):
        r = T[u].get(key, {}).get("ic", {})
        return r.get("ic_mean", np.nan), r.get("ic_lo", np.nan), r.get("ic_hi", np.nan)
    for i, (key, lab, col) in enumerate((("equity_model", "equity-fitted model, applied unchanged", C[0]), ("own_model", "same features refitted within the universe", C[2]))):
        m, lo, hi = zip(*[get(u, key) for u in us])
        m, lo, hi = np.array(m), np.array(lo), np.array(hi)
        a.errorbar(x + (i - 0.5) * 0.2, m, yerr=[np.nan_to_num(m - lo), np.nan_to_num(hi - m)], fmt="o", color=col, capsize=4, label=lab)
    a.set_xticks(x); a.set_xticklabels([f"{u}\n({T[u]['n_symbols']} names)" for u in us]); a.axhline(0, color="k", lw=0.6); a.set_ylabel("mean IC, h = 5"); a.legend(fontsize=7); a.set_title("The new-products test")
    a = ax[1]
    feats = list(T[us[0]]["univariate"])
    W = 0.8 / len(us)
    for i, u in enumerate(us):
        a.bar(np.arange(len(feats)) + i * W, [T[u]["univariate"][f]["ic"] for f in feats], W, color=C[i], label=u)
    a.set_xticks(np.arange(len(feats)) + W); a.set_xticklabels(feats, rotation=25, fontsize=7); a.axhline(0, color="k", lw=0.6); a.set_title("Univariate IC of each feature by universe"); a.legend(fontsize=7)
    save(fig, "transfer.png")


if __name__ == "__main__":
    S, V, A, N, K, T = (load(n) for n in ("signal", "validate", "ablate", "neutralise", "costs", "transfer"))
    if S: fig_signal(S)
    if V: fig_validation(V, A)
    if N: fig_neutral(N)
    if K: fig_costs(K)
    if T: fig_transfer(T)
