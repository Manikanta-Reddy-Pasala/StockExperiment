#!/usr/bin/env python3
"""
Sweep option-selling variants for one underlying + WALK-FORWARD validation.

Walk-forward: for each test year Y, pick the best variant on all trades that
ENTERED before Y (in-sample), then report that variant's performance on Y
(out-of-sample). Stitches OOS years => the honest forward expectation.

Selection metric on train = expectancy (total_ret_margin) subject to WR>=wr_min,
because the user wants a HIGH win rate that is also profitable.

Usage:
  python opt_sweep.py --underlying NIFTY --structures ic strangle
"""
import argparse, json, itertools, sys
from collections import defaultdict
import opt_backtest as ob


def year_of(d):
    return int(str(d)[:4])


def summarize_subset(trades):
    return ob.summarize(trades)


def sweep(underlying, structures, min_vol, min_oi, wr_min=70.0,
          start=None, end=None):
    # grids
    cad_g = ["weekly", "monthly"]
    otm_g = [0.02, 0.03, 0.04]
    wing_g = [0.01, 0.015]
    dte_g = [3, 5, 7]
    pt_g = [0.5, 0.6]
    stop_g = [1.0, 1.5]
    iv_g = [0.0, 0.012, 0.02, 0.03]   # ATM-straddle/strike floor (rich-IV gate)
    variants = []
    for structure in structures:
        wings = wing_g if structure == "ic" else [0.0]
        for cad, otm, wing, dte, pt, stop, ivf in itertools.product(
                cad_g, otm_g, wings, dte_g, pt_g, stop_g, iv_g):
            variants.append(dict(structure=structure, cadence=cad, otm=otm,
                                 wing=wing, dte=dte, pt=pt, stop=stop,
                                 iv_floor=ivf))
    print(f"# {underlying}: {len(variants)} variants", file=sys.stderr, flush=True)

    # run every variant once, store full trade list (entry-dated)
    results = []
    for i, v in enumerate(variants):
        tr = ob.run(underlying, v["structure"], v["otm"], v["wing"], v["dte"],
                    v["pt"], v["stop"], min_vol, min_oi, v["iv_floor"],
                    v["cadence"], start, end)
        if not tr:
            continue
        s = ob.summarize(tr)
        results.append((v, tr, s))
        if (i + 1) % 50 == 0:
            print(f"  ..{i+1}/{len(variants)}", file=sys.stderr, flush=True)

    # overall best by total return with WR gate
    def score(s):  # robustness: positive expectancy AND risk-adjusted
        if s.get("total_ret_margin_pct", 0) <= 0:
            return -1e9
        return s.get("calmar") or 0.0
    gated = [(v, tr, s) for (v, tr, s) in results if s["win_rate"] >= wr_min]
    pool = gated or results
    pool.sort(key=lambda x: score(x[2]), reverse=True)
    top = [dict(cfg=v, summary=s) for (v, tr, s) in pool[:8]]

    # WALK-FORWARD by calendar year
    years = sorted({year_of(t["entry"]) for (_, tr, _) in results for t in tr})
    test_years = years[1:]  # need >=1 prior year to train
    wf = []
    stitched = []
    for Y in test_years:
        # pick best variant on trades entered before Y
        best, bestkey = None, -1e18
        for (v, tr, s) in results:
            train = [t for t in tr if year_of(t["entry"]) < Y]
            ts = ob.summarize(train)
            if ts.get("n", 0) < 15 or ts["win_rate"] < wr_min:
                continue
            if ts["total_ret_margin_pct"] <= 0:
                continue
            key = ts.get("calmar") or 0.0   # select most robust, not most return
            if key > bestkey:
                bestkey, best = key, (v, tr)
        if best is None:
            continue
        v, tr = best
        test = [t for t in tr if year_of(t["entry"]) == Y]
        ts = ob.summarize(test)
        wf.append(dict(year=Y, cfg=v, oos=ts))
        stitched += test
    wf_overall = ob.summarize(stitched)
    return dict(underlying=underlying, n_variants=len(variants),
                top_in_sample=top, walk_forward=wf, wf_stitched=wf_overall)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", required=True)
    ap.add_argument("--structures", nargs="+", default=["ic", "strangle"])
    ap.add_argument("--min-vol", type=int, default=100)
    ap.add_argument("--min-oi", type=int, default=500)
    ap.add_argument("--wr-min", type=float, default=70.0)
    ap.add_argument("--start"); ap.add_argument("--end")
    a = ap.parse_args()
    out = sweep(a.underlying, a.structures, a.min_vol, a.min_oi, a.wr_min,
                a.start, a.end)
    print(json.dumps(out, indent=2, default=str))
