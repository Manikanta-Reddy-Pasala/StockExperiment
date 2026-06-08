#!/usr/bin/env python3
"""
0DTE NIFTY expiry-day selling, from real EXPIRY-DAY OHLC bars.

Fyers serves no intraday for expired contracts and NSE has no intraday history,
so full tick paths are unavailable. BUT historical_options has each weekly
option's EXPIRY-DAY daily bar (open/high/low/close). A 0DTE seller:
  - ENTERS at the expiry-day OPEN (sell OTM strangle; ~full day of theta left),
  - the option's intraday HIGH bounds worst adverse move (stop proxy),
  - SETTLES at the CLOSE (= near-intrinsic at expiry).
This captures the expiry-day theta crush that is the real 0DTE edge, using data
we have. Approx: OHLC not full path, so stops use the day's high (conservative-ish).

Return on realistic per-lot SPAN margin. NIFTY weekly only (Tuesday post-2025).
"""
import argparse, json, sys, itertools, statistics as st
import opt_backtest as ob

# NIFTY 0DTE short-strangle SPAN+exposure margin per lot is lower than normal
# (expires same day): ~ 6-8% of notional. notional = ATM*lot. Use frac*ATM
# (lot cancels in ret). 0.07 is a realistic 0DTE strangle margin fraction.
MARGIN_FRAC = 0.07
SLIP = 0.02


# 0DTE needs open/high (opt_backtest kept only close), so load OHLC directly.
import psycopg
DB = "postgresql://trader:trader_password@database:5432/trading_system"


def load_expiry_ohlc(start, end):
    """expiry -> {(strike,kind): (open,high,low,close,vol)} for the EXPIRY-DAY bar."""
    c = psycopg.connect(DB); cur = c.cursor()
    q = ("SELECT expiry, strike, opt_type, open, high, low, close, volume "
         "FROM historical_options WHERE underlying='NIFTY' "
         "AND candle_time::date = expiry")        # expiry-day bar only
    if start:
        q += " AND expiry >= %s"
    args = [start] if start else []
    if end:
        q += " AND expiry <= %s"; args.append(end)
    cur.execute(q, args)
    out = {}
    for exp, strike, kind, o, h, l, cl, v in cur.fetchall():
        out.setdefault(exp, {})[(strike, kind.strip())] = (
            float(o), float(h), float(l), float(cl), int(v or 0))
    c.close()
    return out


def atm_from_open(snap):
    """Spot via put-call parity: S = K + CE - PE, over LIQUID strikes (both legs
    real), take the median (robust to deep-OTM 0.2 placeholders), then ATM =
    nearest available strike. The naive min|CE-PE| breaks on expiry-day chains."""
    implied = []
    strikes = sorted({s for (s, k) in snap})
    for s in strikes:
        ce = snap.get((s, "CE")); pe = snap.get((s, "PE"))
        # PIT: use only the 9:15 OPEN price (known at entry), NOT full-day volume
        if ce and pe and ce[0] > 1 and pe[0] > 1:
            implied.append(s + ce[0] - pe[0])      # CE/PE open
    if not implied:
        return None
    import statistics as _st
    spot = _st.median(implied)
    return min(strikes, key=lambda s: abs(s - spot))


def nearest(snap, target, kind, min_vol):
    # PIT: select on OPEN price availability (known at 9:15), not full-day volume.
    cs = [s for (s, k) in snap if k == kind and snap[(s, kind)][0] > 0.5]
    return min(cs, key=lambda s: abs(s - target)) if cs else None


def backtest(otm, stop, structure, wing, min_vol, start, end, maxloss_margin=None):
    """maxloss_margin: if set (e.g. 0.12), hard-stop at that fraction of margin
    instead of `stop`× credit — a true −X% of capital per-trade cap."""
    data = load_expiry_ohlc(start, end)
    trades = []
    for exp in sorted(data):
        snap = data[exp]
        atm = atm_from_open(snap)
        if atm is None:
            continue
        sc = nearest(snap, atm * (1 + otm), "CE", min_vol)
        sp = nearest(snap, atm * (1 - otm), "PE", min_vol)
        if sc is None or sp is None:
            continue
        ce, pe = snap[(sc, "CE")], snap[(sp, "PE")]
        # credit at OPEN net slippage
        credit = (ce[0] + pe[0]) * (1 - SLIP)
        if credit <= 0:
            continue
        legs_buyback_close = ce[3] + pe[3]      # settle at CLOSE (intrinsic)
        bw = None
        if structure == "ironfly":
            bc = nearest(snap, sc * (1 + wing), "CE", 0)
            bp = nearest(snap, sp * (1 - wing), "PE", 0)
            if bc is None or bp is None:
                continue
            credit -= (snap[(bc, "CE")][0] + snap[(bp, "PE")][0]) * (1 + SLIP)
            legs_buyback_close -= snap[(bc, "CE")][3] + snap[(bp, "PE")][3]
            bw = min(bc - sc, sp - bp)
            if bw <= 0 or credit <= 0:
                continue
        # intraday worst: short legs' HIGH => max cost to be forced out
        worst_buyback = ce[1] + pe[1]           # both highs (conservative upper)
        if structure == "ironfly":
            worst_buyback -= snap[(bc, "CE")][2] + snap[(bp, "PE")][2]
        # margin first (the stop can be expressed as a % of it)
        if structure == "ironfly":
            if not bw or credit >= 0.9 * bw:     # reject mispriced/illiquid wing
                continue
            margin = max(bw - credit, 0.25 * bw)  # floor at 25% of wing, no blowup
        else:
            margin = MARGIN_FRAC * atm
        # stop threshold: a hard −X% of margin if maxloss_margin set, else stop×credit
        max_loss_allowed = (maxloss_margin * margin) if maxloss_margin else (stop * credit)
        pnl_close = credit - legs_buyback_close * (1 + SLIP)
        pnl_worst = credit - worst_buyback * (1 + SLIP)
        if -pnl_worst >= max_loss_allowed:
            pnl = -max_loss_allowed             # stopped intraday at the cap
            reason = "STOP"
        else:
            pnl = pnl_close
            reason = "SETTLE"
        legvol = min(ce[4], pe[4])
        if structure == "ironfly":
            legvol = min(legvol, snap[(bc, "CE")][4], snap[(bp, "PE")][4])
        MIN_FILL = 100   # contracts traded that day to consider the leg fillable
        def leg(role, side, strike, v):
            return dict(role=role, action=side, strike=strike,
                        pct=round(100 * (strike / atm - 1), 2),
                        price=round(v[0], 2), volume=int(v[4]),
                        filled=bool(v[4] >= MIN_FILL))
        legs = [leg("short_CE", "SELL", sc, ce), leg("short_PE", "SELL", sp, pe)]
        if structure == "ironfly":
            legs += [leg("long_CE", "BUY", bc, snap[(bc, "CE")]),
                     leg("long_PE", "BUY", bp, snap[(bp, "PE")])]
        all_filled = all(l["filled"] for l in legs)
        trades.append(dict(expiry=str(exp), spot=atm, credit=round(credit, 1),
                           pnl=round(pnl, 1), ret=pnl / margin, reason=reason,
                           atm=atm, minvol=legvol, all_filled=all_filled,
                           legs=legs))
    return trades


def stats(trades):
    if not trades:
        return dict(n=0)
    rets = [t["ret"] for t in trades]
    wins = sum(1 for t in trades if t["pnl"] > 0)
    eq = peak = mdd = 0.0; ceq = 1.0; ruin = False
    for r in rets:
        eq += r; peak = max(peak, eq); mdd = max(mdd, peak - eq)
        ceq *= (1 + r)
        if ceq <= 0:
            ruin = True; ceq = 1e-9
    n = len(rets)
    # ~52 weekly expiries/yr
    cagr = (ceq ** (52.0 / n) - 1) if n else 0
    return dict(n=n, win_rate=round(100 * wins / n, 1),
                avg_ret_pct=round(100 * st.mean(rets), 2),
                total_ret_pct=round(100 * sum(rets), 1),
                cagr_pct=round(100 * cagr, 1), mdd_pct=round(100 * mdd, 1),
                worst_pct=round(100 * min(rets), 1), ruin=ruin,
                stops=sum(1 for t in trades if t["reason"] == "STOP"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-03-01"); ap.add_argument("--end")
    ap.add_argument("--min-vol", type=int, default=1000)
    a = ap.parse_args()
    grid = []
    for structure, otm, stop, wing in itertools.product(
            ["strangle", "ironfly"], [0.0, 0.003, 0.005, 0.008, 0.012],
            [1.0, 1.5, 2.0, 99.0], [0.005, 0.01, 0.015, 0.02]):
        if structure == "strangle" and wing != 0.01:
            continue   # wing irrelevant for strangle, dedupe
        grid.append((structure, otm, stop, wing))
    out = []
    for structure, otm, stop, wing in grid:
        tr = backtest(otm, stop, structure, wing, a.min_vol, a.start, a.end)
        s = stats(tr)
        if s.get("n", 0) >= 10:
            out.append(dict(structure=structure, otm=otm, stop=stop, wing=wing,
                            **s))
    # iron-fly worst loss is STRUCTURALLY CAPPED (wing-credit); strangle is not.
    stopped = [r for r in out if r["structure"] == "strangle" and r["stop"] < 99
               and not r["ruin"] and r["mdd_pct"] < 80]
    ironfly = [r for r in out if r["structure"] == "ironfly"
               and not r["ruin"] and r["mdd_pct"] < 80]
    stopped.sort(key=lambda x: x["cagr_pct"], reverse=True)
    ironfly.sort(key=lambda x: x["cagr_pct"], reverse=True)
    print(json.dumps(dict(window=f"{a.start}..{a.end or 'now'}",
                          hard_stop_strangle=stopped[:10],
                          iron_fly_defined_risk=ironfly[:10]),
                     indent=2, default=str))
