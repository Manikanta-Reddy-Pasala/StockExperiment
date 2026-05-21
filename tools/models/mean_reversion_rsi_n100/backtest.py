"""mean_reversion_rsi_n100 — candidate 5th model.

De-correlated from the 4 momentum/breakout models. Buys oversold stocks
(RSI<30) in confirmed uptrends (close > 200d SMA), exits on RSI bounce
back to neutral or fixed risk targets.

Strategy
--------
Universe : NSE Nifty 100 (read from CSV)
Lookback : RSI(14), SMA(200)
Entry rule (daily after market close):
    - close > SMA(200)         (uptrend filter — avoid falling knives)
    - RSI(14) < RSI_LOW        (oversold, default 30)
    - close > previous_low * (1 + MIN_BOUNCE) optional anti-knife
Pick     : lowest RSI across qualifying universe (most oversold)
Position : max_concurrent = 1, all-in on rank-1
Exit rule (any of):
    - RSI(14) > RSI_EXIT       (bounce confirmed, default 50)
    - close >= entry * (1 + TARGET_PCT)
    - close <= entry * (1 - STOP_PCT)
    - days_held >= MAX_HOLD
Charges   : approx Fyers CNC (₹20 brokerage + STT + GST + exchange + SEBI + DP)
            via tools/live/broker_charges.compute_charges — matches live infra
"""
from __future__ import annotations
import sys, csv, argparse
from pathlib import Path
from datetime import date, timedelta
from decimal import Decimal

sys.path.insert(0, "/app")
import pandas as pd
from sqlalchemy import text

from tools.shared.ohlcv_cache import _get_engine
from tools.live.broker_charges import compute_charges


# Strategy params (defaults — overridable via CLI)
RSI_WIN     = 14
RSI_LOW     = 30
RSI_EXIT    = 50
SMA_LONG    = 200
TARGET_PCT  = 0.08   # +8% take-profit
STOP_PCT    = 0.05   # -5% hard stop
MAX_HOLD    = 20     # trading days
N100_CSV    = "/app/src/data/symbols/nifty100.csv"

DEFAULT_START = date(2023, 5, 15)
DEFAULT_END   = date(2026, 5, 12)
DEFAULT_CAP   = 1_000_000.0


def load_n100() -> list:
    syms = []
    with open(N100_CSV) as f:
        for r in csv.DictReader(f):
            if r.get("Series", "").strip() == "EQ":
                syms.append(r["Symbol"].strip())
    return syms


def rsi(closes: pd.Series, window: int = 14) -> pd.Series:
    """Standard Wilder RSI."""
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _chg(side: str, qty: int, price: float) -> float:
    if qty < 1:
        return 0.0
    return float(compute_charges(side, qty, float(price), "CNC").get("total", 0))


def run(start: date, end: date, capital: float, out_dir: Path | None = None,
        rsi_low: float = RSI_LOW, rsi_exit: float = RSI_EXIT,
        target_pct: float = TARGET_PCT, stop_pct: float = STOP_PCT,
        max_hold: int = MAX_HOLD):
    n100_syms = load_n100()
    fyers_syms = [f"NSE:{s}-EQ" for s in n100_syms]
    print(f"Universe: NSE Nifty 100 ({len(fyers_syms)} stocks)")

    eng = _get_engine()
    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b "
            "AND data_source='fyers' ORDER BY symbol,date"
        ), c, params={"s": fyers_syms, "a": start - timedelta(days=400),
                       "b": end})
    df["date"] = pd.to_datetime(df["date"])
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    sma200 = cl.rolling(SMA_LONG).mean()
    rsi_df = cl.apply(lambda c: rsi(c, RSI_WIN))

    dates = cl.index
    cap = capital
    hold = None   # symbol
    qty = 0
    entry_px = 0.0
    entry_date = None
    trades = []
    peak_cap = capital
    max_dd = 0.0

    for d in dates:
        if d.date() < start or d.date() > end:
            continue
        # Min warmup
        di = dates.get_loc(d)
        if di < max(RSI_WIN, SMA_LONG):
            continue

        # NAV mark-to-market for DD tracking
        if hold and qty > 0:
            px_now = cl[hold].iloc[di]
            if pd.notna(px_now):
                nav = cap + qty * float(px_now)
                peak_cap = max(peak_cap, nav)
                dd = (peak_cap - nav) / peak_cap if peak_cap > 0 else 0
                max_dd = max(max_dd, dd)
        else:
            peak_cap = max(peak_cap, cap)
            dd = (peak_cap - cap) / peak_cap if peak_cap > 0 else 0
            max_dd = max(max_dd, dd)

        # Check exits first
        if hold and qty > 0:
            px = cl[hold].iloc[di]
            r = rsi_df[hold].iloc[di]
            if pd.isna(px) or pd.isna(r):
                continue
            px = float(px)
            r = float(r)
            age = (dates[di] - entry_date).days
            ret = (px - entry_px) / entry_px
            exit_reason = None
            if r > rsi_exit:
                exit_reason = "RSI_EXIT"
            elif ret >= target_pct:
                exit_reason = "TARGET"
            elif ret <= -stop_pct:
                exit_reason = "STOP"
            elif age >= max_hold:
                exit_reason = "MAX_HOLD"
            if exit_reason:
                proceeds = qty * px - _chg("SELL", qty, px)
                cost     = qty * entry_px
                pnl      = proceeds - cost
                cap += proceeds
                trades.append({
                    "entry_date": entry_date.date().isoformat(),
                    "exit_date":  d.date().isoformat(),
                    "sym":        hold.replace("NSE:", "").replace("-EQ", ""),
                    "qty":        qty,
                    "entry_px":   round(entry_px, 2),
                    "exit_px":    round(px, 2),
                    "pnl":        round(pnl, 2),
                    "ret_pct":    round(ret * 100, 2),
                    "age_days":   age,
                    "reason":     exit_reason,
                    "cap_after":  round(cap, 2),
                })
                hold = None
                qty = 0
                entry_px = 0
                entry_date = None

        # Entry scan only if flat
        if not hold:
            r_today = rsi_df.iloc[di]
            sma_today = sma200.iloc[di]
            close_today = cl.iloc[di]
            qual = []
            for sym in fyers_syms:
                rv = r_today.get(sym)
                sv = sma_today.get(sym)
                cv = close_today.get(sym)
                if pd.isna(rv) or pd.isna(sv) or pd.isna(cv):
                    continue
                if cv > sv and rv < rsi_low:
                    qual.append((sym, float(rv), float(cv)))
            if qual:
                qual.sort(key=lambda x: x[1])  # most oversold first
                sym, r_, px = qual[0]
                # Size: deploy ~99% capital reserving small buffer for charges
                est_qty = int(cap * 0.99 / px)
                while est_qty > 0:
                    cost = est_qty * px + _chg("BUY", est_qty, px)
                    if cost <= cap:
                        break
                    est_qty -= 1
                if est_qty > 0:
                    cap -= est_qty * px + _chg("BUY", est_qty, px)
                    hold = sym
                    qty = est_qty
                    entry_px = px
                    entry_date = d

    # Final mark-to-market if still holding
    if hold and qty > 0:
        last_px = float(cl[hold].iloc[-1])
        final_nav = cap + qty * last_px
        # Force-close for accounting
        proc = qty * last_px - _chg("SELL", qty, last_px)
        ret = (last_px - entry_px) / entry_px
        cap += proc
        trades.append({
            "entry_date": entry_date.date().isoformat(),
            "exit_date":  dates[-1].date().isoformat(),
            "sym":        hold.replace("NSE:", "").replace("-EQ", ""),
            "qty":        qty, "entry_px": round(entry_px, 2),
            "exit_px":    round(last_px, 2),
            "pnl":        round(proc - qty * entry_px, 2),
            "ret_pct":    round(ret * 100, 2),
            "age_days":   (dates[-1] - entry_date).days,
            "reason":     "FORCE_CLOSE_END", "cap_after": round(cap, 2),
        })

    final = cap
    yrs = (end - start).days / 365.25
    total_ret = (final / capital - 1) * 100
    cagr = ((final / capital) ** (1 / max(yrs, 0.001)) - 1) * 100
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    wr = (wins / len(trades) * 100) if trades else 0
    calmar = cagr / (max_dd * 100) if max_dd > 0 else 0

    print(f"\n## MEAN-REVERSION RSI N100 RESULTS")
    print(f"  Window: {start} → {end} ({yrs:.2f}y)")
    print(f"  Params: RSI<{rsi_low} entry, RSI>{rsi_exit} exit, "
          f"target +{target_pct*100:.0f}%, stop -{stop_pct*100:.0f}%, "
          f"max_hold {max_hold}d")
    print(f"  Final NAV:    ₹{final:,.0f}")
    print(f"  Total return: {total_ret:+.2f}%")
    print(f"  CAGR:         {cagr:+.2f}%")
    print(f"  Trades:       {len(trades)} (W={wins}, L={losses}, "
          f"WR={wr:.1f}%)")
    print(f"  Max DD:       {max_dd*100:.2f}%")
    print(f"  Calmar:       {calmar:.2f}")

    if trades:
        # Last 10 trades
        print("\n  Last 10 trades:")
        for t in trades[-10:]:
            print(f"    {t['entry_date']} → {t['exit_date']}  "
                  f"{t['sym']:<14} ret={t['ret_pct']:+6.2f}%  "
                  f"age={t['age_days']:>3}d  {t['reason']}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(out_dir / "trades.json", "w") as f:
            json.dump(trades, f, indent=2, default=str)
        print(f"\n  Wrote {out_dir / 'trades.json'}")
    return cagr, max_dd * 100, wr, len(trades)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=lambda s: date.fromisoformat(s),
                    default=DEFAULT_START)
    ap.add_argument("--end", type=lambda s: date.fromisoformat(s),
                    default=DEFAULT_END)
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    ap.add_argument("--rsi-low", type=float, default=RSI_LOW)
    ap.add_argument("--rsi-exit", type=float, default=RSI_EXIT)
    ap.add_argument("--target-pct", type=float, default=TARGET_PCT)
    ap.add_argument("--stop-pct", type=float, default=STOP_PCT)
    ap.add_argument("--max-hold", type=int, default=MAX_HOLD)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    run(args.start, args.end, args.capital, args.out_dir,
        rsi_low=args.rsi_low, rsi_exit=args.rsi_exit,
        target_pct=args.target_pct, stop_pct=args.stop_pct,
        max_hold=args.max_hold)
