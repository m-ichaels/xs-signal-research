#!/usr/bin/env bash
# Full pipeline: (data ->) signal -> validate -> ablate -> neutralise -> costs -> transfer -> tests -> figures -> summary -> report -> notebook
# Usage: scripts/run_all.sh [--download] [--quick]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
Q=""; [[ " $* " == *" --quick "* ]] && Q="--quick"
[[ " $* " == *" --download "* ]] && python tools/download.py etf futures
python -u -m xsig all $Q | tee results/run.log
python -m pytest -q | tee results/tests.txt
python scripts/plots.py
python scripts/summarize.py
python scripts/report.py
python scripts/make_notebook.py --execute
echo done
