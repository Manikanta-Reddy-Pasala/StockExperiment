#!/usr/bin/env python3
"""
DIVERSIFIED stock-options premium-selling BASKET + walk-forward.

Sells the same structure on MANY liquid F&O stocks each month (stock options
are monthly-only in India), volume/OI filtered, then aggregates equal-weight
per calendar month. Diversification across uncorrelated single names is the one
lever that can repair the fat tail that sinks single-underlying index selling.

Portfolio monthly return = mean(ret_margin) over all stock-trades entered that
month. Walk-forward: pick the config with best in-sample monthly-Calmar (trade
win-rate gated) on months before year Y, apply to year Y, stitch OOS.
"""
import argparse, json, itertools, sys, statistics as st
from collections import defaultdict
import opt_backtest as ob

STOCKS = ['RELIANCE', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'INFY', 'TCS', 'ITC', 'AXISBANK', 'KOTAKBANK', 'LT', 'BHARTIARTL', 'MARUTI', 'SUNPHARMA', 'BAJFINANCE', 'HINDUNILVR', 'ASIANPAINT', 'WIPRO', 'ADANIENT', 'ADANIPORTS', 'TATAMOTORS', 'TATASTEEL', 'JINDALSTEL', 'HINDALCO', 'VEDL', 'SAIL', 'NMDC', 'COALINDIA', 'NTPC', 'POWERGRID', 'TATAPOWER', 'ADANIPOWER', 'ADANIGREEN', 'BEL', 'HAL', 'BHEL', 'BHARATFORG', 'CGPOWER', 'SIEMENS', 'ABB', 'POLYCAB', 'DIXON', 'PERSISTENT', 'LTIM', 'IRCTC', 'PNB', 'CANBK', 'BANKBARODA', 'RECLTD', 'PFC', 'MOTHERSON', 'ASHOKLEY', 'DLF', 'TRENT', 'TATACONSUM', 'INDUSINDBK', 'GAIL', 'IOC', 'BPCL', 'LUPIN']


def ym(d):
    return str(d)[:7]


def year_of(d):
    return int(str(d)[:4])


def run_all_configs(stocks, grid, min_vol, min_oi, start=None, end=None):
    """Memory-safe: one stock chain resident at a time; run every config on it,
    accumulate per-config trade lists, then free the chain before next stock."""
    by_cfg = {i: [] for i in range(len(grid))}
    for u in stocks:
        ob._CHAIN_CACHE.clear()        # drop previous stock's chain
        for i, cfg in enumerate(grid):
            tr = ob.run(u, cfg["structure"], cfg["otm"], cfg["wing"], cfg["dte"],
                        cfg["pt"], cfg["stop"], min_vol, min_oi, cfg["iv_floor"],
                        "monthly", start, end)
            for t in tr:
                t["underlying"] = u
            by_cfg[i] += tr
        print(f"  {u} done", file=sys.stderr, flush=True)
    ob._CHAIN_CACHE.clear()
    return by_cfg


def portfolio_stats(trades):
    """Equal-weight monthly aggregation -> portfolio metrics."""
    if not trades:
        return dict(n=0)
    by_month = defaultdict(list)
    for t in trades:
        by_month[ym(t["entry"])].append(t["ret_margin"])
    months = sorted(by_month)
    monthly = [sum(by_month[m]) / len(by_month[m]) for m in months]  # equal-wt
    n_tr = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    total = sum(monthly)
    # additive equity (margin-return units) for DD; compounded equity for CAGR
    eq, peak, mdd = 0.0, 0.0, 0.0
    ceq = 1.0
    for r in monthly:
        eq += r; peak = max(peak, eq); mdd = max(mdd, peak - eq)
        ceq *= (1 + max(r, -0.99))     # cap a single month at -99% (no ruin past 0)
    calmar = round(total / mdd, 2) if mdd > 1e-9 else None
    cagr = round(100 * (ceq ** (12.0 / len(months)) - 1), 1) if months else None
    pos_months = sum(1 for r in monthly if r > 0)
    return dict(n=n_tr, n_months=len(months),
                trade_win_rate=round(100 * wins / n_tr, 1),
                month_win_rate=round(100 * pos_months / len(months), 1),
                avg_month_pct=round(100 * st.mean(monthly), 2),
                total_pct=round(100 * total, 1), cagr_pct=cagr,
                mdd_pct=round(100 * mdd, 1), calmar=calmar,
                worst_month_pct=round(100 * min(monthly), 1),
                best_month_pct=round(100 * max(monthly), 1))


def sweep(stocks, min_vol, min_oi, wr_min, start=None, end=None):
    grid = []
    # SHORT-STRANGLE basket (the proven seller winner) across the full high-beta
    # universe: more + higher-premium names = more diversification, richer credit.
    for structure in ("strangle",):
        for otm, dte, pt, stop, ivf in itertools.product(
                [0.03, 0.05, 0.07], [3, 5, 7], [0.5, 0.6], [1.0, 1.5, 99.0],
                [0.0]):
            grid.append(dict(structure=structure, otm=otm, wing=0.0, dte=dte,
                             pt=pt, stop=stop, iv_floor=ivf))
    print(f"# basket {len(stocks)} stocks x {len(grid)} configs",
          file=sys.stderr, flush=True)
    by_cfg = run_all_configs(stocks, grid, min_vol, min_oi, start, end)
    runs = [(grid[i], tr, portfolio_stats(tr)) for i, tr in by_cfg.items() if tr]

    def score(s):
        if s.get("total_pct", 0) <= 0:
            return -1e9
        return s.get("calmar") or 0.0
    gated = [(c, tr, s) for (c, tr, s) in runs if s["trade_win_rate"] >= wr_min]
    pool = gated or runs
    pool.sort(key=lambda x: score(x[2]), reverse=True)
    top = [dict(cfg=c, stats=s) for (c, tr, s) in pool[:8]]
    by_cagr = sorted(runs, key=lambda x: x[2].get("cagr_pct") or -1e9, reverse=True)
    top_cagr = [dict(cfg=c, stats=s) for (c, tr, s) in by_cagr[:8]]

    years = sorted({year_of(t["entry"]) for (_, tr, _) in runs for t in tr})
    wf, stitched = [], []
    for Y in years[1:]:
        best, bestkey = None, -1e18
        for (c, tr, s) in runs:
            train = [t for t in tr if year_of(t["entry"]) < Y]
            ts = portfolio_stats(train)
            if ts.get("n", 0) < 20 or ts["trade_win_rate"] < wr_min:
                continue
            if ts["total_pct"] <= 0:
                continue
            key = ts.get("calmar") or 0.0
            if key > bestkey:
                bestkey, best = key, (c, tr)
        if not best:
            continue
        c, tr = best
        test = [t for t in tr if year_of(t["entry"]) == Y]
        wf.append(dict(year=Y, cfg=c, oos=portfolio_stats(test)))
        stitched += test
    return dict(stocks=stocks, top_in_sample=top, top_by_cagr=top_cagr,
                walk_forward=wf, wf_stitched=portfolio_stats(stitched))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-vol", type=int, default=100)
    ap.add_argument("--min-oi", type=int, default=300)
    ap.add_argument("--wr-min", type=float, default=70.0)
    ap.add_argument("--start"); ap.add_argument("--end")
    a = ap.parse_args()
    out = sweep(STOCKS, a.min_vol, a.min_oi, a.wr_min, a.start, a.end)
    print(json.dumps(out, indent=2, default=str))
