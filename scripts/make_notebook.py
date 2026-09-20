#!/usr/bin/env python3
"""notebooks/results.ipynb: a walk through the results, reading results/*.json and the run parquets.  With --execute
the code cells run in-process (no Jupyter needed) and their outputs are stored, so the notebook renders with results.
    python scripts/make_notebook.py [--execute]"""
import base64
import contextlib
import io
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md("# xs-signal-research: results\n\nEverything here reads `results/*.json` and the run parquets written by `python -m xsig all`; re-run the pipeline and re-execute to refresh."),
    code("import json, os, sys\nimport numpy as np, pandas as pd\nsys.path.insert(0, os.path.abspath('..'))\nfrom IPython.display import Image, display\nfrom xsig import metrics as M\nR = {n: json.load(open(f'../results/{n}.json')) for n in ('signal', 'validate', 'ablate', 'neutralise', 'costs', 'transfer') if os.path.exists(f'../results/{n}.json')}\npd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)"),
    md("## 1. The signal out of sample\n\nDaily Spearman IC of the walk-forward blend against the 5-day demeaned forward return, with a Newey–West t and a moving-block-bootstrap band; by year; IC decay; a model per horizon."),
    code("S = R['signal']\nprint({k: round(v, 4) if isinstance(v, float) else v for k, v in S['ic'].items()})\nby = pd.DataFrame(S['by_year']).T; by['ic_train'] = pd.Series(S['ic_train_by_year']); display(by.round(4))\ndisplay(Image('../results/figures/signal.png'))"),
    code("hz = S['horizons']\ndisplay(pd.DataFrame({h: {'ic': v['own']['ic_mean'], 'lo': v['own']['ic_lo'], 'hi': v['own']['ic_hi'], 't': v['own']['t_nw'], 'ac_5': v['autocorrelation']['ac_5']} for h, v in hz.items()}).T.round(4))"),
    md("## 2. Honest validation\n\nThe same learner under five schemes.  Shuffled K-fold sees the test day's neighbours in training; purged and embargoed K-fold does not, and it lands where walk-forward lands."),
    code("for h, lad in R['validate']['ladder'].items():\n    print('h =', h); display(pd.DataFrame(lad).T.round(4))\ndisplay(Image('../results/figures/validation.png'))"),
    md("## 3. Ablations\n\nDrop one feature, keep one group, drop one group; the change in IC against the full model with a Newey–West t on the daily difference; each feature alone through ridge."),
    code("A = R['ablate']\ndisplay(pd.DataFrame(A['variants']).T[['ic_mean', 't_nw', 'decile_spread', 'delta_vs_full', 'delta_t']].round(4))\ndisplay(pd.DataFrame(A['single_feature']).T[['ic_mean', 't_nw']].round(4))"),
    md("## 4. Neutralisation against the risk model\n\nBeta, size, sector, then momentum and volatility (which are also the signal's inputs); Fama–MacBeth factor premia and the attribution of the gross rank book."),
    code("N = R['neutralise']\ndisplay(pd.DataFrame({v['label']: {'ic': v['ic']['ic_mean'], 'lo': v['ic']['ic_lo'], 'hi': v['ic']['ic_hi'], 't': v['ic']['t_nw'], 'decile_bp': v['decile_spread'] * 1e4} for v in N['signals'].values()}).T.round(4))\nprint('exposure correlations', {k: round(v, 3) for k, v in N['exposure_corr'].items()})\nprint('attribution', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in N['attribution'].items()})\ndisplay(Image('../results/figures/neutral.png'))"),
    md("## 5. Costs and capacity\n\nThe algo-wheel cost model on the rank book at $1bn; the capacity curve; each horizon in its own book; the net IC; the deflated Sharpe."),
    code("K = R['costs']\ndisplay(pd.DataFrame(K['models']).T[['gross_ann', 'net_ann', 'cost_ann', 'sharpe_gross', 'sharpe_net', 'turnover', 'avg_cost_bp', 'max_dd_net']].round(4))\nprint('net IC', {k: round(v, 4) for k, v in K['models']['routed']['net_ic'].items()})\ncap = pd.DataFrame(K['capacity']).T; display(cap.round(4)); print('breakeven capital', K['breakeven_capital'])\ndisplay(pd.DataFrame(K['by_horizon']).T[['gross_ann', 'net_ann', 'sharpe_net', 'turnover', 'avg_cost_bp']].round(4))\nprint('deflated Sharpe', K['deflated'])\ndisplay(Image('../results/figures/costs.png'))"),
    md("## 6. New products: ETFs and futures\n\nThe equity-fitted models of each year applied unchanged to the ETF and futures frames, the same learner refitted within each universe, and each feature's univariate IC by universe."),
    code("T = R['transfer']\nrows = []\nfor u, r in T.items():\n    if not isinstance(r, dict) or 'n_symbols' not in r: continue\n    for key in ('equity_model', 'own_model'):\n        if key in r and 'ic' in r[key]:\n            ic = r[key]['ic']; b = r[key].get('book', {})\n            rows.append({'universe': u, 'model': key, 'n': r['n_symbols'], 'ic': ic['ic_mean'], 'lo': ic.get('ic_lo'), 'hi': ic.get('ic_hi'), 't': ic['t_nw'], 'net_ann': b.get('net_ann'), 'sharpe_net': b.get('sharpe_net')})\ndisplay(pd.DataFrame(rows).round(4))\ndisplay(pd.DataFrame({u: {f: r['univariate'][f]['ic'] for f in r['univariate']} for u, r in T.items() if isinstance(r, dict) and 'univariate' in r}).round(4))\ndisplay(Image('../results/figures/transfer.png'))"),
    md("## 7. SQL over the store\n\nDuckDB views over the committed bars, the feature frames and the run parquets."),
    code("from xsig.data import store\ncon = store()\ndisplay(con.execute(\"SELECT symbol, COUNT(*) AS days, MIN(date) AS first, MAX(date) AS last, ROUND(AVG(volume * close) / 1e6) AS adv_musd FROM equity GROUP BY symbol ORDER BY adv_musd DESC LIMIT 10\").df())\ndisplay(con.execute(\"SELECT EXTRACT(year FROM date) AS year, ROUND(AVG(z * target_5) * 1e4, 2) AS mean_z_times_target_bp, COUNT(*) AS n FROM pred_equity GROUP BY 1 ORDER BY 1\").df())"),
]


def execute(nb):
    """run the code cells in one namespace; stdout, DataFrames and images become stored outputs"""
    ns = {}
    cwd = os.getcwd(); os.chdir(os.path.join(ROOT, "notebooks"))
    try:
        for n, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            outputs = []

            def display(obj):
                import pandas as pd
                if isinstance(obj, _Image):
                    with open(obj.filename, "rb") as fh:
                        outputs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(fh.read()).decode()}})
                elif isinstance(obj, (pd.DataFrame, pd.Series)):
                    df = obj.to_frame() if isinstance(obj, pd.Series) else obj
                    outputs.append({"output_type": "display_data", "metadata": {}, "data": {"text/html": df.to_html(max_rows=60, max_cols=30), "text/plain": df.to_string(max_rows=60, max_cols=30)}})
                else:
                    outputs.append({"output_type": "display_data", "metadata": {}, "data": {"text/plain": repr(obj)}})

            class _Image:
                def __init__(self, filename):
                    self.filename = filename
            ns["display"] = display; ns["Image"] = _Image
            buf = io.StringIO()
            src = c["source"].replace("from IPython.display import Image, display", "")
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(src, f"<cell {n}>", "exec"), ns)
            except Exception:
                outputs.append({"output_type": "stream", "name": "stderr", "text": traceback.format_exc()})
            if buf.getvalue():
                outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
            c["outputs"] = outputs; c["execution_count"] = n + 1
    finally:
        os.chdir(cwd)
    return nb


if __name__ == "__main__":
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    if "--execute" in sys.argv:
        nb = execute(nb)
        errs = [o for c in nb["cells"] for o in c.get("outputs", []) if o.get("name") == "stderr"]
        if errs:
            print("cell errors:", len(errs)); print(errs[0]["text"][-600:])
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    with open(os.path.join(ROOT, "notebooks", "results.ipynb"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1)
    print("wrote notebooks/results.ipynb", "(executed)" if "--execute" in sys.argv else "")
