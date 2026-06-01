"""Build the orb 5-min bar cache (cache5min/*.pkl) for the whole N500 universe.

The cache is gitignored + not baked into the image, so it must be rebuilt after
a deploy before the orb backtest/sweep can run. Live ORB does NOT need this (it
fetches 5-min bars from Fyers per scan). One Fyers history call per ~60-day
chunk per symbol — slow; run in the background.

Run:  python3 tools/models/orb_momentum_intraday/build_cache.py [--from YYYY-MM-DD --to YYYY-MM-DD]
"""
import sys, argparse
from pathlib import Path
from datetime import date, datetime

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.index_membership import universe_union
from tools.models.orb_momentum_intraday.data import get_5min, _fyers, CACHE_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2025-02-01")
    ap.add_argument("--to", dest="end", default=None)
    a = ap.parse_args()
    start = datetime.strptime(a.start, "%Y-%m-%d").date()
    end = datetime.strptime(a.end, "%Y-%m-%d").date() if a.end else date.today()
    syms = sorted(universe_union("n500"))
    fy = _fyers()
    print(f"Building 5-min cache for {len(syms)} symbols {start}..{end} -> {CACHE_DIR}", flush=True)
    ok = miss = 0
    for i, s in enumerate(syms, 1):
        try:
            df = get_5min(s, start, end, fy=fy)
            if df is not None and len(df) >= 300:
                ok += 1
            else:
                miss += 1
        except Exception as e:
            miss += 1
            print(f"  ERR {s}: {type(e).__name__}", flush=True)
        if i % 25 == 0:
            print(f"  {i}/{len(syms)}  cached={ok} miss={miss}", flush=True)
    print(f"DONE: cached={ok} miss={miss} -> {CACHE_DIR}", flush=True)


if __name__ == "__main__":
    main()
