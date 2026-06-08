#!/usr/bin/env python3
"""
Options premium-SELLING backtest engine (StockExperiment).

Daily EOD bhavcopy data in historical_options (volume + oi populated).
High-win-rate theta strategies: short strangle and iron condor.

Design choices that kill two prior bugs:
  - ATM is derived from the chain itself (strike minimizing |CE-PE|, put-call
    parity) => no dependence on spot price => IMMUNE to stock split adjustment
    (the bug that produced garbage stock-option results last time).
  - IV is proxied by the ATM straddle / ATM strike => no India-VIX dependency
    (works uniformly for indices AND stocks).
  - LIQUIDITY FILTER on the legs we actually sell (min volume + min OI) -- the
    piece that was missed last time.

Everything is measured per-unit (premium points) so lot-size history is
irrelevant to win-rate and percent returns.

Usage:
  python opt_backtest.py --underlying NIFTY --structure ic --otm 0.025 \
      --wing 0.015 --dte 5 --pt 0.5 --stop 2.0 --min-vol 100 --min-oi 500
"""
import argparse, json, math
from datetime import date
import psycopg

DB = "postgresql://trader:trader_password@database:5432/trading_system"

# ---- per-trade transaction cost model (per UNIT premium, both legs both ways)
# STT on sell-side premium 0.1% (options), exchange+sebi+gst+brokerage rolled in,
# plus slippage as a fraction of each leg's traded premium.
SLIP_PCT = 0.02      # 2% of leg premium slippage each fill
FLAT_PER_LEG = 0.0   # set to model fixed brokerage in points if desired


def load_chain(cur, underlying, start=None, end=None):
    """Return dict: expiry -> {date -> {(strike,kind): (close,volume,oi)}}."""
    q = ("SELECT expiry, candle_time::date, strike, opt_type, close, volume, oi "
         "FROM historical_options WHERE underlying=%s")
    args = [underlying]
    if start:
        q += " AND candle_time::date>=%s"; args.append(start)
    if end:
        q += " AND candle_time::date<=%s"; args.append(end)
    cur.execute(q, args)
    chain = {}
    for expiry, d, strike, kind, close, vol, oi in cur.fetchall():
        kind = kind.strip()
        chain.setdefault(expiry, {}).setdefault(d, {})[(strike, kind)] = (
            float(close), int(vol or 0), int(oi or 0))
    return chain


def atm_strike(snap):
    """ATM via put-call parity: strike minimizing |CE_close - PE_close|."""
    strikes = sorted({s for (s, k) in snap})
    best, bestdiff = None, 1e18
    for s in strikes:
        ce = snap.get((s, "CE")); pe = snap.get((s, "PE"))
        if ce and pe:
            diff = abs(ce[0] - pe[0])
            if diff < bestdiff:
                bestdiff, best = diff, s
    return best


def nearest_strike(snap, target, kind, min_vol=0, min_oi=0):
    """Nearest strike to target having `kind`, passing liquidity filter."""
    cands = [s for (s, k) in snap if k == kind]
    cands = [s for s in cands
             if snap[(s, kind)][1] >= min_vol and snap[(s, kind)][2] >= min_oi]
    if not cands:
        return None
    return min(cands, key=lambda s: abs(s - target))


def leg_price(chain_exp, d, strike, kind):
    snap = chain_exp.get(d)
    if not snap:
        return None
    v = snap.get((strike, kind))
    return v[0] if v else None


_CHAIN_CACHE = {}


def get_chain(underlying, start=None, end=None, cur=None):
    key = (underlying, str(start), str(end))
    if key not in _CHAIN_CACHE:
        own = cur is None
        if own:
            conn = psycopg.connect(DB); cur = conn.cursor()
        _CHAIN_CACHE[key] = load_chain(cur, underlying, start, end)
        if own:
            conn.close()
    return _CHAIN_CACHE[key]


def monthly_set(expiries):
    """Monthly expiry = the latest expiry within its calendar (year, month)."""
    by_ym = {}
    for e in expiries:
        by_ym.setdefault((e.year, e.month), []).append(e)
    return {max(v) for v in by_ym.values()}


def run(underlying, structure="ic", otm=0.025, wing=0.015, dte=5,
        pt=0.5, stop=2.0, min_vol=50, min_oi=200, iv_floor=0.0,
        cadence="all", start=None, end=None, cur=None):
    chain = get_chain(underlying, start, end, cur)
    monthlies = monthly_set(chain.keys())
    trades = []
    for expiry in sorted(chain):
        if cadence == "monthly" and expiry not in monthlies:
            continue
        if cadence == "weekly" and expiry in monthlies:
            continue
        days = sorted(chain[expiry])
        if len(days) < 2:
            continue
        # entry day: DTE closest to target (prefer >= target so we don't enter too late)
        ge = [d for d in days if (expiry - d).days >= dte]
        entry = ge[-1] if ge else None      # latest day still >= target dte
        if entry is None:
            continue
        snap = chain[expiry][entry]
        atm = atm_strike(snap)
        if atm is None:
            continue
        ce_atm = snap.get((atm, "CE")); pe_atm = snap.get((atm, "PE"))
        if not ce_atm or not pe_atm:
            continue
        iv_proxy = (ce_atm[0] + pe_atm[0]) / atm
        if iv_proxy < iv_floor:
            continue
        # sell strikes (liquidity-filtered)
        sc = nearest_strike(snap, atm * (1 + otm), "CE", min_vol, min_oi)
        sp = nearest_strike(snap, atm * (1 - otm), "PE", min_vol, min_oi)
        if sc is None or sp is None:
            continue
        sc_px = snap[(sc, "CE")][0]; sp_px = snap[(sp, "PE")][0]
        legs = [(sc, "CE", -1), (sp, "PE", -1)]   # -1 = sold
        wing_w = None
        if structure == "ic":
            bc = nearest_strike(snap, sc * (1 + wing), "CE")  # buy wings (no liq req)
            bp = nearest_strike(snap, sp * (1 - wing), "PE")
            if bc is None or bp is None or bc <= sc or bp >= sp:
                continue
            legs += [(bc, "CE", +1), (bp, "PE", +1)]
            wing_w = min(bc - sc, sp - bp)
            if wing_w <= 0:
                continue
        # entry credit (per unit), net of slippage on every leg
        def fill(px, side):  # side -1 sold (receive), +1 bought (pay)
            slip = SLIP_PCT * px
            return (-side) * px - (SLIP_PCT * px) - FLAT_PER_LEG  # cashflow w/ slip drag
        credit = 0.0
        entry_ok = True
        for (s, k, side) in legs:
            px = leg_price(chain[expiry], entry, s, k)
            if px is None:
                entry_ok = False; break
            credit += fill(px, side)
        if not entry_ok or credit <= 0:
            continue
        # an IC cannot legitimately collect more credit than its wing width;
        # if it appears to, the wing fill was illiquid/mispriced -> reject.
        if structure == "ic" and credit >= 0.95 * wing_w:
            continue
        max_profit = credit
        # simulate forward to expiry-1 (no expiry-day gamma)
        fwd = [d for d in days if entry < d < expiry]
        exit_d, exit_pnl, reason = None, None, None
        for d in fwd:
            cost_to_close = 0.0
            ok = True
            for (s, k, side) in legs:
                px = leg_price(chain[expiry], d, s, k)
                if px is None:
                    ok = False; break
                # closing reverses side; pay slippage again
                cost_to_close += side * px - SLIP_PCT * px * 0  # mark at close
            if not ok:
                continue
            # current pnl per unit = credit collected - cost to buy back net position
            # net buyback cost = sum(side*px) where sold legs (side -1) cost +px to close
            buyback = 0.0
            for (s, k, side) in legs:
                px = leg_price(chain[expiry], d, s, k) or 0.0
                buyback += (-side) * px            # sold => pay px; bought => receive px
            buyback += SLIP_PCT * sum(abs(leg_price(chain[expiry], d, s, k) or 0)
                                      for (s, k, _) in legs)
            pnl = credit - buyback
            if pnl >= pt * max_profit:
                exit_d, exit_pnl, reason = d, pnl, "PT"; break
            if pnl <= -stop * max_profit:
                exit_d, exit_pnl, reason = d, pnl, "STOP"; break
        if exit_d is None:
            # exit at last pre-expiry day close
            d = fwd[-1] if fwd else entry
            buyback = 0.0
            for (s, k, side) in legs:
                px = leg_price(chain[expiry], d, s, k) or 0.0
                buyback += (-side) * px
            buyback += SLIP_PCT * sum(abs(leg_price(chain[expiry], d, s, k) or 0)
                                      for (s, k, _) in legs)
            exit_d, exit_pnl, reason = d, credit - buyback, "EXP"
        # margin proxy per unit: IC=wing_width-credit (defined); strangle=ATM*span
        if structure == "ic":
            # defined risk = wing width - credit; floor at 10% of wing so a
            # near-max-credit IC can't fake an infinite return.
            margin = max(wing_w - max_profit, 0.10 * wing_w)
        else:
            margin = 0.12 * atm   # ~SPAN+exposure approx for index/stock strangle
        trades.append(dict(expiry=str(expiry), entry=str(entry), exit=str(exit_d),
                           dte=(expiry - entry).days, atm=atm, iv=round(iv_proxy, 4),
                           credit=round(credit, 2), pnl=round(exit_pnl, 2),
                           ret_credit=round(exit_pnl / max_profit, 4),
                           ret_margin=round(exit_pnl / margin, 4),
                           reason=reason, margin=round(margin, 2)))
    return trades


def summarize(trades):
    if not trades:
        return dict(n=0)
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    rm = [t["ret_margin"] for t in trades]
    total_rm = sum(rm)
    avg_rm = total_rm / n
    # equity curve on margin-return for DD
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in rm:
        eq = eq + r          # additive equity in margin-return units (avoids
        peak = max(peak, eq)  # compounding blow-ups from a single -1.0 trade)
        mdd = max(mdd, peak - eq)
    import statistics as st
    calmar = round(total_rm / mdd, 2) if mdd > 1e-9 else None
    return dict(n=n, win_rate=round(100 * wins / n, 1),
                avg_ret_margin_pct=round(100 * avg_rm, 2),
                median_ret_margin_pct=round(100 * st.median(rm), 2),
                total_ret_margin_pct=round(100 * total_rm, 1),
                mdd_pct=round(100 * mdd, 1), calmar=calmar,
                worst_pct=round(100 * min(rm), 1), best_pct=round(100 * max(rm), 1),
                exits={r: sum(1 for t in trades if t["reason"] == r)
                       for r in ("PT", "STOP", "EXP")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", required=True)
    ap.add_argument("--structure", default="ic", choices=["ic", "strangle"])
    ap.add_argument("--otm", type=float, default=0.025)
    ap.add_argument("--wing", type=float, default=0.015)
    ap.add_argument("--dte", type=int, default=5)
    ap.add_argument("--pt", type=float, default=0.5)
    ap.add_argument("--stop", type=float, default=2.0)
    ap.add_argument("--min-vol", type=int, default=50)
    ap.add_argument("--min-oi", type=int, default=200)
    ap.add_argument("--iv-floor", type=float, default=0.0)
    ap.add_argument("--cadence", default="all", choices=["all", "weekly", "monthly"])
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args()
    tr = run(a.underlying, a.structure, a.otm, a.wing, a.dte, a.pt, a.stop,
             a.min_vol, a.min_oi, a.iv_floor, a.cadence, a.start, a.end)
    out = dict(cfg=vars(a), summary=summarize(tr))
    if a.ledger:
        out["trades"] = tr
    print(json.dumps(out, indent=2, default=str))
