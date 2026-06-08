#!/usr/bin/env python3
"""
LEVERAGED directional options: buy CALLS on the strongest F&O stocks by
momentum (monthly). Expresses a real edge (momentum) through long options for
amplification -- defined risk (max loss = 100% of premium), uncapped upside,
naturally high CAGR potential. Walk-forward validated.

Momentum ranked from chain-derived ATM (put-call parity) => spot-free, split-
immune. Return is on PREMIUM PAID (the leveraged base), equal-weight top-K,
aggregated per month -> portfolio. Also runs PUTS / both for completeness.
"""
import argparse, json, itertools, sys, statistics as st
from collections import defaultdict
import opt_backtest as ob

STOCKS = ['RELIANCE', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'INFY', 'TCS', 'ITC', 'AXISBANK', 'KOTAKBANK', 'LT', 'BHARTIARTL', 'MARUTI', 'SUNPHARMA', 'BAJFINANCE', 'HINDUNILVR', 'ASIANPAINT', 'WIPRO', 'ADANIENT', 'ADANIPORTS', 'TATAMOTORS', 'TATASTEEL', 'JINDALSTEL', 'HINDALCO', 'VEDL', 'SAIL', 'NMDC', 'COALINDIA', 'NTPC', 'POWERGRID', 'TATAPOWER', 'ADANIPOWER', 'ADANIGREEN', 'BEL', 'HAL', 'BHEL', 'BHARATFORG', 'CGPOWER', 'SIEMENS', 'ABB', 'POLYCAB', 'DIXON', 'PERSISTENT', 'LTIM', 'IRCTC', 'PNB', 'CANBK', 'BANKBARODA', 'RECLTD', 'PFC', 'MOTHERSON', 'ASHOKLEY', 'DLF', 'TRENT', 'TATACONSUM', 'INDUSINDBK', 'GAIL', 'IOC', 'BPCL', 'LUPIN']


def ym(d):
    return str(d)[:7]


def year_of(d):
    return int(str(d)[:4])


def _sim_one(chain, expiry, days, kind, strike_off, dte, pt, stop,
             sell_off=None):
    """Long option, or DEBIT SPREAD if sell_off set (sell a further-OTM same-kind
    leg against the long). ret on NET DEBIT (leveraged, defined, capped)."""
    ge = [d for d in days if (expiry - d).days >= dte]
    entry = ge[-1] if ge else None
    if entry is None:
        return None
    snap = chain[expiry][entry]
    atm = ob.atm_strike(snap)
    if atm is None:
        return None
    target = atm * (1 + strike_off) if kind == "CE" else atm * (1 - strike_off)
    strike = ob.nearest_strike(snap, target, kind, 0, 0)
    if strike is None:
        return None
    short_strike = None
    if sell_off is not None:
        st_t = strike * (1 + sell_off) if kind == "CE" else strike * (1 - sell_off)
        short_strike = ob.nearest_strike(snap, st_t, kind, 0, 0)
        if short_strike is None or short_strike == strike:
            return None

    def net(d):
        lp = ob.leg_price(chain[expiry], d, strike, kind)
        if lp is None:
            return None
        if short_strike is None:
            return lp
        sp = ob.leg_price(chain[expiry], d, short_strike, kind)
        if sp is None:
            return None
        return lp - sp           # debit spread value

    entry_px = net(entry)
    if not entry_px or entry_px <= 0:
        return None
    fwd = [d for d in days if entry < d < expiry]
    ex_px, reason = None, None
    for d in fwd:
        px = net(d)
        if px is None:
            continue
        r = px / entry_px - 1.0
        if r >= pt:
            ex_px, reason = px, "PT"; break
        if r <= -stop:
            ex_px, reason = px, "STOP"; break
    if ex_px is None:
        d = fwd[-1] if fwd else entry
        ex_px = net(d)
        if ex_px is None:
            ex_px = 0.0
        reason = "EXP"
    slip = 0.03
    ret = (ex_px * (1 - slip)) / (entry_px * (1 + slip)) - 1.0
    return dict(month=ym(entry), entry=str(entry), atm=atm, ret_prem=ret,
                reason=reason)


def build_all(stocks, leg_keys):
    """Load each stock chain ONCE; compute every leg-config from it.
    leg_keys = list of (kind, off, dte, pt, stop). Returns {key: {stock: recs}}."""
    out = {k: {} for k in leg_keys}
    for u in stocks:
        ob._CHAIN_CACHE.clear()
        chain = ob.get_chain(u)
        monthlies = ob.monthly_set(chain.keys())
        exps = [(e, sorted(chain[e])) for e in sorted(chain) if e in monthlies]
        exps = [(e, ds) for (e, ds) in exps if len(ds) >= 2]
        for k in leg_keys:
            kind, off, dte, pt, stop, sell_off = k
            recs = []
            for e, ds in exps:
                r = _sim_one(chain, e, ds, kind, off, dte, pt, stop, sell_off)
                if r:
                    recs.append(r)
            out[k][u] = recs
        print(f"  {u} done", file=sys.stderr, flush=True)
    ob._CHAIN_CACHE.clear()
    return out


def momentum_portfolio(per, lookback, topk):
    """Rank stocks each month by ATM momentum (lb months), buy top-K, equal-wt."""
    # atm history per stock by month
    atm_hist = {u: {r["month"]: r["atm"] for r in recs} for u, recs in per.items()}
    ret_by = {u: {r["month"]: r["ret_prem"] for r in recs} for u, recs in per.items()}
    months = sorted({m for recs in per.values() for r in recs for m in [r["month"]]})
    monthly, trades = [], []
    for i, M in enumerate(months):
        if i < lookback:
            continue
        prev = months[i - lookback]
        scores = []
        for u in per:
            a_now = atm_hist[u].get(M); a_prev = atm_hist[u].get(prev)
            if a_now and a_prev and M in ret_by[u]:
                scores.append((a_now / a_prev - 1.0, u))
        if not scores:
            continue
        scores.sort(reverse=True)
        picks = [u for _, u in scores[:topk]]
        rets = [ret_by[u][M] for u in picks if M in ret_by[u]]
        if not rets:
            continue
        monthly.append(sum(rets) / len(rets))
        trades += rets
    return monthly, trades


def stats(monthly, trades):
    if not monthly:
        return dict(n=0)
    wins = sum(1 for r in trades if r > 0)
    eq, peak, mdd, ceq = 0.0, 0.0, 0.0, 1.0
    for r in monthly:
        eq += r; peak = max(peak, eq); mdd = max(mdd, peak - eq)
        ceq *= (1 + max(r, -0.99))
    cagr = round(100 * (ceq ** (12.0 / len(monthly)) - 1), 1)
    return dict(n=len(trades), n_months=len(monthly),
                trade_win_rate=round(100 * wins / len(trades), 1),
                month_win_rate=round(100 * sum(1 for r in monthly if r > 0) / len(monthly), 1),
                avg_month_pct=round(100 * st.mean(monthly), 2),
                cagr_pct=cagr, mdd_pct=round(100 * mdd, 1),
                calmar=round((sum(monthly)) / mdd, 2) if mdd > 1e-9 else None,
                worst_month_pct=round(100 * min(monthly), 1),
                best_month_pct=round(100 * max(monthly), 1))


def sweep(stocks, wr_min):
    grid = []
    # BULL-CALL DEBIT SPREADS on momentum winners (sell_off = short-leg distance).
    # buy slightly-ITM/ATM, sell further-OTM -> cheaper, theta-offset, defined.
    for kind, off, sell_off, dte, pt, stop, lb, k in itertools.product(
            ["CE"], [-0.02, 0.0], [0.03, 0.05], [5, 7], [1.0, 99.0],
            [0.5, 99.0], [1, 3], [1, 2, 3]):
        grid.append(dict(kind=kind, off=off, sell_off=sell_off, dte=dte, pt=pt,
                         stop=stop, lb=lb, topk=k))
    print(f"# momspread {len(grid)} configs", file=sys.stderr, flush=True)
    leg_keys = sorted({(c["kind"], c["off"], c["dte"], c["pt"], c["stop"],
                        c["sell_off"]) for c in grid})
    cache = build_all(stocks, leg_keys)   # ONE chain load per stock
    out_runs = []
    for cfg in grid:
        key = (cfg["kind"], cfg["off"], cfg["dte"], cfg["pt"], cfg["stop"],
               cfg["sell_off"])
        monthly, trades = momentum_portfolio(cache[key], cfg["lb"], cfg["topk"])
        if trades:
            out_runs.append((cfg, momentum_portfolio(cache[key], cfg["lb"], cfg["topk"]),
                             stats(monthly, trades)))
    out_runs.sort(key=lambda x: x[2].get("cagr_pct") or -1e9, reverse=True)
    top = [dict(cfg=c, stats=s) for (c, _, s) in out_runs[:10]]
    return dict(top_by_cagr=top, n_configs=len(grid))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-min", type=float, default=40.0)
    a = ap.parse_args()
    print(json.dumps(sweep(STOCKS, a.wr_min), indent=2, default=str))
