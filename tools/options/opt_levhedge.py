#!/usr/bin/env python3
"""
LEVERAGE + CRASH-HEDGE on the 59-name far-OTM short-strangle seller.

The seller is 17% CAGR / ~1% DD on margin -- but that low DD is a 2023-26
artifact (no crash month). Naked far-OTM selling's true tail is a correlated
multi-name gap. This module:
  1. rebuilds the seller's REAL monthly return series,
  2. adds a REAL NIFTY far-OTM put hedge (premium cost every month, payoff in
     the down months that actually occurred),
  3. sweeps leverage L x hedge-budget c for in-sample CAGR/DD,
  4. CRASH STRESS-TESTS: injects a synthetic -18% gap month (seller ~ -90% on
     margin, all names breached at once) and shows which (L, c) survive vs ruin.

The hedge is pure cost in calm data; its value only appears under stress, so the
stress test -- not the in-sample CAGR -- is the real decision input.
"""
import json, sys, statistics as st
from collections import defaultdict
import opt_backtest as ob

STOCKS = ['RELIANCE','HDFCBANK','ICICIBANK','SBIN','INFY','TCS','ITC','AXISBANK',
'KOTAKBANK','LT','BHARTIARTL','MARUTI','SUNPHARMA','BAJFINANCE','HINDUNILVR',
'ASIANPAINT','WIPRO','ADANIENT','ADANIPORTS','TATAMOTORS','TATASTEEL','JINDALSTEL',
'HINDALCO','VEDL','SAIL','NMDC','COALINDIA','NTPC','POWERGRID','TATAPOWER',
'ADANIPOWER','ADANIGREEN','BEL','HAL','BHEL','BHARATFORG','CGPOWER','SIEMENS',
'ABB','POLYCAB','DIXON','PERSISTENT','LTIM','IRCTC','PNB','CANBK','BANKBARODA',
'RECLTD','PFC','MOTHERSON','ASHOKLEY','DLF','TRENT','TATACONSUM','INDUSINDBK',
'GAIL','IOC','BPCL','LUPIN']

# winning seller config: far-OTM 7%, dte7, 60% profit-target, 1.5x credit stop
CFG = dict(structure="strangle", otm=0.07, wing=0.0, dte=7, pt=0.6, stop=1.5,
           iv_floor=0.0)


def ym(d):
    return str(d)[:7]


def seller_monthly(min_vol=50, min_oi=150):
    trades = []
    for u in STOCKS:
        ob._CHAIN_CACHE.clear()
        tr = ob.run(u, CFG["structure"], CFG["otm"], CFG["wing"], CFG["dte"],
                    CFG["pt"], CFG["stop"], min_vol, min_oi, CFG["iv_floor"],
                    "monthly")
        trades += tr
    ob._CHAIN_CACHE.clear()
    by_m = defaultdict(list)
    for t in trades:
        by_m[ym(t["entry"])].append(t["ret_margin"])
    return {m: sum(v) / len(v) for m, v in by_m.items()}   # equal-wt monthly ret


def nifty_put_hedge():
    """Per month: payoff/cost ratio of a ~12% OTM monthly NIFTY put held to expiry.
    cost = entry premium; payoff = max(0, strike - spot_at_expiry) using chain ATM
    on the last pre-expiry day as the spot proxy (split-immune)."""
    ob._CHAIN_CACHE.clear()
    chain = ob.get_chain("NIFTY")
    monthlies = ob.monthly_set(chain.keys())
    ratio = {}
    for expiry in sorted(chain):
        if expiry not in monthlies:
            continue
        days = sorted(chain[expiry])
        if len(days) < 2:
            continue
        ge = [d for d in days if (expiry - d).days >= 7]
        entry = ge[-1] if ge else None
        if entry is None:
            continue
        snap = chain[expiry][entry]
        atm = ob.atm_strike(snap)
        if atm is None:
            continue
        strike = ob.nearest_strike(snap, atm * 0.88, "PE", 0, 0)
        if strike is None:
            continue
        cost = ob.leg_price(chain[expiry], entry, strike, "PE")
        if not cost or cost <= 0:
            continue
        last = days[-1]                       # last day with data ~ expiry
        atm_exp = ob.atm_strike(chain[expiry][last]) or atm
        payoff = max(0.0, strike - atm_exp)
        ratio[ym(entry)] = payoff / cost
    ob._CHAIN_CACHE.clear()
    return ratio


def curve_stats(monthly):
    eq, peak, mdd, ceq = 0.0, 0.0, 0.0, 1.0
    ruin = False
    for r in monthly:
        eq += r; peak = max(peak, eq); mdd = max(mdd, peak - eq)
        ceq *= (1 + r)
        if ceq <= 0:
            ruin = True; ceq = 1e-9
    n = len(monthly)
    cagr = (ceq ** (12.0 / n) - 1) if n else 0
    return dict(cagr_pct=round(100 * cagr, 1), mdd_pct=round(100 * mdd, 1),
                worst_mo=round(100 * min(monthly), 1), ruin=ruin)


def combine(seller, hedge, L, c):
    """portfolio_m = L*seller_m + c*(hedge_ratio_m - 1).  c = monthly premium budget."""
    months = sorted(seller)
    out = []
    for m in months:
        s = seller[m]
        h = c * (hedge.get(m, 0.0) - 1.0)   # -c drag if no payoff
        out.append(L * s + h)
    return out


def main():
    print("building seller monthly series...", file=sys.stderr, flush=True)
    seller = seller_monthly()
    print("building NIFTY put hedge...", file=sys.stderr, flush=True)
    hedge = nifty_put_hedge()
    months = sorted(seller)
    # crash month: seller far-OTM 7% strangle in a -18% correlated gap loses
    # ~ (0.18-0.07)/0.12 of margin ~ -0.92; a 12%-OTM NIFTY put then ~ (0.18-0.12)
    # of spot intrinsic on a ~0.5% cost => payoff/cost ~ (0.06*ATM)/(0.005*ATM)=~12x.
    CRASH_SELLER, CRASH_HEDGE_RATIO = -0.92, 12.0

    out = {"base_seller": curve_stats([seller[m] for m in months]),
           "hedge_months_paid": sum(1 for m in months if hedge.get(m, 0) > 1),
           "n_months": len(months), "frontier": [], "stress": []}
    for L in (1, 2, 4, 6, 8):
        for c in (0.0, 0.003, 0.006, 0.01):
            series = combine(seller, hedge, L, c)
            s = curve_stats(series)
            # stress: append crash month
            crash_m = L * CRASH_SELLER + c * (CRASH_HEDGE_RATIO - 1.0)
            s_stress = curve_stats(series + [crash_m])
            rec = dict(L=L, hedge_c_pct=round(100 * c, 1),
                       cagr=s["cagr_pct"], mdd=s["mdd_pct"],
                       crash_month_pct=round(100 * crash_m, 1),
                       cagr_after_crash=s_stress["cagr_pct"],
                       mdd_after_crash=s_stress["mdd_pct"],
                       ruin=s_stress["ruin"])
            out["frontier"].append(rec)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
