#!/usr/bin/env python3
"""Free daily bars from the Yahoo chart API (no key) for the three universes in data/.

    python tools/download.py [equity|etf|futures ...]     default: etf futures

Raw JSON goes to data/raw/yahoo/<symbol>.json (git-ignored, cached) and the long tables to
data/derived/{equity,etf,futures}.parquet (committed, so the pipeline runs without this step).  The equity panel is the
one the algo-wheel project (ProjectF) downloaded; it is refreshed here only on request.  Futures are Yahoo's
continuous front-month series: unadjusted at rolls, with the contract's own volume."""
import csv
import datetime as dt
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "yahoo")
DER = os.path.join(ROOT, "data", "derived")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def get(url, retries=3, timeout=60):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                print("  failed", url[:80], e)
                return None
            time.sleep(2 * (k + 1))


def yahoo(sym):
    raw = get(f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}?range=20y&interval=1d&events=div,splits", retries=2, timeout=30)
    if raw is None:
        return None
    try:
        return json.loads(raw)["chart"]["result"][0]
    except Exception:  # noqa: BLE001
        return None


def build(universe: str):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(RAW, exist_ok=True); os.makedirs(DER, exist_ok=True)
    with open(os.path.join(ROOT, "data", f"universe_{universe}.csv"), newline="") as f:
        uni = list(csv.DictReader(f))
    bars = []
    for u in uni:
        sym = u["symbol"]; p = os.path.join(RAW, f"{sym.replace('=', '_')}.json")
        if os.path.exists(p):
            r = json.load(open(p))
        else:
            r = yahoo(sym)
            if r is None:
                print("  no data", sym); continue
            json.dump(r, open(p, "w")); time.sleep(0.3)
        ts = r.get("timestamp") or []; q = r["indicators"]["quote"][0]
        adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose", [None] * len(ts))
        n = 0
        for i, t in enumerate(ts):
            c = q["close"][i]
            if c is None or c <= 0:
                continue
            d = dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat()
            o = q["open"][i] if q["open"][i] else c; h = q["high"][i] if q["high"][i] else c; lo = q["low"][i] if q["low"][i] else c
            bars.append((sym, d, o, h, lo, c, adj[i] if adj[i] else c, q["volume"][i] or 0)); n += 1
        print(f"  {sym}: {n} bars")
    cols = list(zip(*bars))
    table = pa.table({"symbol": cols[0], "date": cols[1], "open": pa.array(cols[2], pa.float64()), "high": pa.array(cols[3], pa.float64()), "low": pa.array(cols[4], pa.float64()),
                      "close": pa.array(cols[5], pa.float64()), "adjclose": pa.array(cols[6], pa.float64()), "volume": pa.array(cols[7], pa.float64())})
    out = os.path.join(DER, f"{universe}.parquet")
    pq.write_table(table, out, compression="zstd")
    print(f"{universe}: {len(bars)} bars, {len(set(cols[0]))} symbols -> {out}")


if __name__ == "__main__":
    for u in (sys.argv[1:] or ["etf", "futures"]):
        build(u)
