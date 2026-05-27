"""midcap_narrow_60d_breakout V2 — Indian mid/small-cap breakout swing.

Strategy (V2 WINNER from cap-filter sweep)
==========================================

Entry (single concurrent position, max_conc=1):
  - Stock makes fresh 40-day high (was 60d in V1)
  - Volume on breakout day > 2.0x 20-day avg volume
  - Close > 200-day SMA (Stage 2 trend filter)

Exit (whichever fires first, in this precedence):
  - TARGET:   +100% from entry (was +60% in V1)
  - STOP:     -20% from entry — catastrophe stop (added 2026-05-26)
  - TRAIL:    -20% from PEAK PRICE, armed once trade is >=+10% in profit (was -15% in V1)
              NOTE: 20% off the peak close, not a 20% drop in the gain-number.
              Peak +40% -> exits at +12% from entry, not +30%.
  - MAX_HOLD: 120 calendar days (was 30 in V1, then 90); age = (today - entry).days
  - SMA20 exit: DISABLED (was enabled — leaked winners on dips)

Universe:
  - Pseudo-midcap pool: top-100 from N500 by 20d ADV (SKIP_TOP=0)
  - V2 cap filter: Exclude NSE Nifty 100 members (keep Mid + Small caps only) -> ~42 names
  - ANGELONE: no longer excluded — historical_data restored split-adjusted 2026-05-17

Costs: 10 bps slippage, 0.10% STT on sells, ₹20/order brokerage.

Result (V2, ₹10L start, 2023-05-15 → 2026-05-15)
================================================

| Metric    | Value           |
|-----------|----------------:|
| Final NAV | ₹65,00,421      |
| CAGR      | **+86.63%**     |
| Max DD    | **15.15%**      |
| Calmar    | **5.72**        |
| Trades    | 12 (~4/yr)      |
| WR        | 75% (9W / 3L)   |
| 2023-24   | +234.30%        |
| 2024-25   | +51.78%         |
| 2025-26   | -5.55%          |

CLI usage
---------
  docker exec trading_system_app python tools/models/midcap_narrow_60d_breakout/backtest.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from tools.shared.ohlcv_cache import _get_engine
from tools.shared.universes import nifty500_symbols
from tools.shared.breakout_strategy import is_breakout, breakout_exit_reason

# Strategy params (V2 winner)
HH_WIN     = 40
VOL_MULT   = 2.0
SMA_LONG   = 200
TRAIL_PCT  = 0.20
PROFIT_TRIG = 0.10
TARGET_PCT = 1.00
STOP_PCT   = 0.20  # Catastrophe stop (sweep-chosen). Fires rarely — clear of the -15.7%
                   # deepest winner dip — so 0 CAGR cost on 3yr while capping the tail.
MAX_HOLD   = 120  # Was 90. 120d max-hold sweep-tested as winner: +141% CAGR / 8% DD / Calmar 17.46.
# SMA20 exit was tried and DISABLED (leaked winners on dips); exit lives in the shared core.

# Universe params (V3 winner: top-100 ADV minus Large, was skip-30+take-100)
ADV_WIN    = 20
SKIP_TOP   = 0     # V3: top-100 ADV from N500 (instead of skip-top-30)
KEEP_NEXT  = 100   # Take top 100. Large filter applied below via NSE Nifty 100 CSV.

# DATA_FIXES no longer needed for ANGELONE — historical_data was restored
# from yfinance (split-adjusted) on 2026-05-17. Pattern preserved for future
# data anomalies if Fyers serves bad data again.
DATA_FIXES = {}

N100_CSV = str(ROOT / "src" / "data" / "symbols" / "nifty100.csv")

DEFAULT_START = date(2023, 5, 15)
DEFAULT_END   = date(2026, 5, 15)
DEFAULT_CAP   = 1_000_000.0


def load_n100():
    """Load the NSE Nifty 100 (Large-cap) exclusion list from the CSV.

    The V2 universe is "mid + small cap only", so the bare ticker symbols of
    Nifty 100 members are later subtracted from the ADV-ranked pool.

    Returns:
        set[str]: Bare ticker symbols (e.g. "RELIANCE") of equity-series
        Nifty 100 constituents. Only rows with Series == "EQ" are kept (skips
        non-equity series); symbols are stripped of surrounding whitespace.
    """
    out = set()
    with open(N100_CSV) as f:
        for r in csv.DictReader(f):
            # Only the cash-equity series counts as a Large-cap to exclude.
            if r.get("Series", "").strip() == "EQ":
                out.add(r["Symbol"].strip())
    return out


def run(start: date, end: date, capital: float, out_dir: Path | None = None):
    """Back-test the mid/small-cap breakout swing over a date range.

    Single concurrent position (max_conc=1). Each trading day:
      1. If holding, ask the SHARED ``breakout_exit_reason`` core whether any
         exit (TARGET/STOP/TRAIL/MAX_HOLD) fires today; if so, sell.
      2. If flat, scan every universe name for a qualifying breakout via the
         SHARED ``is_breakout`` core, pick the highest volume-ratio candidate,
         and enter at NEXT day's open (realistic — the signal forms on today's
         close, so you cannot fill until tomorrow).

    Costs applied per trade: 10 bps slippage on both legs, 0.10% STT on sells,
    ₹20/order brokerage.

    Args:
        start: First trading date to simulate (inclusive).
        end: Last trading date to simulate (inclusive).
        capital: Starting cash (₹). Position sizing is all-in on the single pick.
        out_dir: If given, the trade ledger is written to
            ``out_dir/trade_ledger.json``.

    Returns:
        tuple[float, float, list[dict]]: ``(final_nav, cagr_pct, trades)`` where
        ``final_nav`` marks any still-open position to the last close, ``cagr_pct``
        is annualised return, and ``trades`` is the per-trade ledger.
    """
    n100 = load_n100()
    print(f"NSE Nifty 100 (Large-cap exclusion list): {len(n100)} stocks")

    eng = _get_engine()
    # Start the search pool from the full Nifty 500 (Fyers symbol form).
    n500 = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()]

    with eng.connect() as c:
        # Pull 400 extra calendar days BEFORE `start` so the 200d SMA and other
        # rolling windows are already warmed up on the first simulated day.
        df = pd.read_sql(text(
            "SELECT symbol,date,open,high,low,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end})

    df["date"] = pd.to_datetime(df["date"])

    # Apply data fixes — correct known price/volume discontinuities (e.g. a
    # missed split) for specific symbol/date windows before computing anything.
    # Currently empty (DATA_FIXES = {}); the loop is preserved for future use.
    for sym, fixes in DATA_FIXES.items():
        for fx in fixes:
            mask = (df["symbol"] == sym) & \
                   (df["date"] >= pd.Timestamp(fx["start"])) & \
                   (df["date"] <= pd.Timestamp(fx["end"]))
            n_rows = mask.sum()
            if n_rows > 0:
                print(f"  Applied data fix to {sym}: {n_rows} rows / price ÷{fx['price_div']}, vol ×{fx['vol_mul']}")
                # Divide prices and multiply volume to undo the split distortion.
                df.loc[mask, ["open","high","low","close"]] /= fx["price_div"]
                df.loc[mask, "volume"] *= fx["vol_mul"]

    # Average Daily Value traded (₹) = close × volume; used to rank liquidity.
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)
    # FIX 6 — staleness filter. The close pivot below is forward-filled, which
    # carries a delisted/suspended name's last close forward as a flat line; it
    # could then rank into the pool or be held at a phantom price. From the RAW
    # df (pre-ffill), keep only names whose true last bar is within
    # STALE_SESSIONS trading sessions of the panel's latest date.
    STALE_SESSIONS = 5
    _all_dates = sorted(df["date"].unique())
    _cutoff_date = (_all_dates[-(STALE_SESSIONS + 1)]
                    if len(_all_dates) > STALE_SESSIONS else _all_dates[0])
    _last_seen = df.groupby("symbol")["date"].max()
    fresh_syms = set(_last_seen[_last_seen >= _cutoff_date].index)
    # Pivot long rows -> date × symbol matrices, one per field, for vectorised
    # rolling computations. Close is forward-filled to bridge missing bars.
    cl  = df.pivot(index="date", columns="symbol", values="close").ffill()
    hi  = df.pivot(index="date", columns="symbol", values="high")
    op_p = df.pivot(index="date", columns="symbol", values="open")
    vol = df.pivot(index="date", columns="symbol", values="volume")
    adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    dates = cl.index

    sma_long = cl.rolling(SMA_LONG).mean()       # 200d SMA — Stage-2 trend filter
    # Prior HH_WIN(=40)-day HIGH, shifted by 1 so "today" is compared against
    # the high of the PRIOR 40 days (today's own high is excluded -> fresh break).
    hh = hi.rolling(HH_WIN).max().shift(1)
    vol_avg20 = vol.rolling(20).mean()           # 20d avg volume — surge baseline
    adv20 = adv_rs.rolling(ADV_WIN).mean()       # 20d avg ₹-value traded — liquidity rank

    # Build the pseudo-midcap pool from a single end-of-data liquidity snapshot:
    # rank all N500 names by their latest 20d ADV, then take ranks
    # [SKIP_TOP, SKIP_TOP+KEEP_NEXT) = top-100 (SKIP_TOP=0). Large caps are
    # removed afterwards via the Nifty 100 list, leaving mid + small caps.
    last_di = len(dates) - 1
    last_adv = adv20.iloc[last_di].dropna().sort_values(ascending=False)
    # FIX 6 — exclude stale/delisted names before slicing the pool.
    last_adv = last_adv[last_adv.index.isin(fresh_syms)]
    midcap_pool = last_adv.iloc[SKIP_TOP:SKIP_TOP + KEEP_NEXT].index.tolist()

    # V2 filter: exclude Large (NSE Nifty 100). ANGELONE no longer needs explicit
    # exclusion since DATA_FIXES normalize the price discontinuity; with clean data
    # ANGELONE never qualifies for a breakout entry anyway.
    midcap_band = [
        s for s in midcap_pool
        if s.replace("NSE:", "").replace("-EQ", "") not in n100
    ]
    print(f"V2 universe (pseudo-midcap minus Large): {len(midcap_band)} stocks")
    print(f"First 10: {[s.replace('NSE:','').replace('-EQ','') for s in midcap_band[:10]]}")

    # Restrict the iteration to the requested window (warm-up rows fall away).
    trading = [d for d in dates if start <= d.date() <= end]
    cap = capital; pos = None; trades = []
    slip, br, stt = 0.001, 20, 0.001   # 10 bps slippage, ₹20 brokerage, 0.10% STT

    for d in trading:
        di = dates.get_loc(d)
        if pos:
            if pos["sym"] not in cl.columns:
                continue
            c_today = cl[pos["sym"]].iloc[di]
            if pd.isna(c_today):
                continue
            close = float(c_today)
            # Track the highest close since entry — this peak is what TRAIL
            # measures the drawdown against (20% off the PEAK price).
            pos["peak"] = max(pos["peak"], close)
            age = (d.date() - pos["entry_date"]).days   # calendar-day hold age
            ret_e = (close - pos["entry_px"]) / pos["entry_px"]
            # Exit rule via the SHARED breakout core (same call live_signal.py makes).
            reason = breakout_exit_reason(
                pos["entry_px"], close, pos["peak"], age,
                target_pct=TARGET_PCT, stop_pct=STOP_PCT, trail_pct=TRAIL_PCT,
                profit_trigger=PROFIT_TRIG, max_hold_days=MAX_HOLD)
            if reason:
                exit_px = close * (1 - slip)        # sell fills below close by slippage
                proc = pos["qty"] * exit_px         # gross sale proceeds
                fees = proc * stt + br              # STT on the proceeds + flat brokerage
                pnl = proc - fees - pos["qty"] * pos["entry_px"]
                cap += proc - fees                  # net proceeds back into cash
                trades.append({
                    "entry_date": pos["entry_date"].isoformat(),
                    "exit_date":  d.date().isoformat(),
                    "sym":        pos["sym"].replace("NSE:", "").replace("-EQ", ""),
                    "qty":        pos["qty"],
                    "entry_px":   round(pos["entry_px"], 2),
                    "exit_px":    round(exit_px, 2),
                    "pnl":        round(pnl, 0),
                    "ret_pct":    round(ret_e * 100, 2),
                    "reason":     reason,
                    "cap_after":  round(cap, 0),
                })
                pos = None

        if pos is None:
            # Flat: scan every universe name for a qualifying breakout TODAY.
            cands = []
            for sym in midcap_band:
                if sym not in cl.columns:
                    continue
                # Pull today's close, 200d SMA, prior-40d high, 20d avg vol, vol.
                c_v = cl[sym].iloc[di]
                sma_v = sma_long[sym].iloc[di] if sym in sma_long.columns else None
                hh_v = hh[sym].iloc[di] if sym in hh.columns else None
                va_v = vol_avg20[sym].iloc[di] if sym in vol_avg20.columns else None
                v_v = vol[sym].iloc[di] if sym in vol.columns else None
                # Skip names not yet warmed up (any indicator still NaN).
                if any(pd.isna(x) for x in [c_v, sma_v, hh_v, va_v, v_v]):
                    continue
                c_v = float(c_v); sma_v = float(sma_v); hh_v = float(hh_v)
                va_v = float(va_v); v_v = float(v_v)
                # Entry qualification via the SHARED breakout core: close > 40d
                # high AND close > 200d SMA AND volume >= 2x 20d avg. vr is the
                # volume ratio used to rank competing breakouts.
                ok, vr = is_breakout(c_v, hh_v, sma_v, v_v, va_v, vol_mult=VOL_MULT)
                if not ok:
                    continue
                cands.append({"sym": sym, "vr": vr})
            cands.sort(key=lambda c: -c["vr"])   # highest volume-surge ratio wins
            if cands:
                top = cands[0]["sym"]
                # Enter at NEXT day's open: the signal forms on today's close,
                # so the earliest realistic fill is tomorrow's opening print.
                if di + 1 < len(dates):
                    op_n = op_p[top].iloc[di + 1] if top in op_p.columns else None
                    if pd.notna(op_n):
                        entry_px = float(op_n) * (1 + slip)   # buy fills above open by slippage
                        q = int(cap / entry_px)               # all-in share count
                        # Only enter if we can afford >=1 share plus brokerage.
                        if q >= 1 and q * entry_px + br <= cap:
                            cap -= q * entry_px + br
                            # peak seeded at entry so TRAIL has a baseline from day 1.
                            pos = {"sym": top, "qty": q, "entry_px": entry_px,
                                   "entry_date": dates[di + 1].date(), "peak": entry_px}

    final = cap
    if pos:
        # Mark any still-open position to the last available close.
        last = float(cl[pos["sym"]].iloc[-1])
        final = cap + pos["qty"] * last

    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    yrs = (end - start).days / 365.25
    cagr = ((final / capital) ** (1 / yrs) - 1) * 100   # annualised return

    # Max drawdown from the post-trade equity curve (cap_after snapshots):
    # running peak minus current, as a % of the running peak.
    peak = capital; mdd = 0
    for t in trades:
        peak = max(peak, t["cap_after"])
        dd = (peak - t["cap_after"]) / peak * 100
        mdd = max(mdd, dd)

    print(f"\n## V2 RESULTS")
    print(f"  Final NAV:    ₹{final:,.0f}")
    print(f"  Total return: {(final/capital-1)*100:+.2f}%")
    print(f"  CAGR ({yrs:.2f}y): {cagr:+.2f}%")
    print(f"  Trades: {len(trades)} (W={wins}, L={losses}, WR={wins/max(1,wins+losses)*100:.1f}%)")
    print(f"  Max DD: {mdd:.2f}%")
    print(f"  Calmar: {cagr/max(0.01,mdd):.2f}")

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trade_ledger.json").write_text(json.dumps(trades, indent=2))
        # summary.json — same shape the other 3 models emit (this model used to
        # write only the trade ledger; now self-generates the summary too).
        open_pos = None
        if pos:
            _last = float(cl[pos["sym"]].iloc[-1])
            open_pos = {
                "symbol": pos["sym"].replace("NSE:", "").replace("-EQ", ""),
                "qty": pos["qty"], "entry_px": round(pos["entry_px"], 2),
                "entry_date": pos["entry_date"].isoformat(),
                "last": round(_last, 2),
                "unrealized": round((_last - pos["entry_px"]) * pos["qty"], 0),
            }
        summary = {
            "model": "midcap_narrow_60d_breakout",
            "start": start.isoformat(), "end": end.isoformat(),
            "years": round(yrs, 3), "capital": capital,
            "final_nav": round(final, 0),
            "total_return_pct": round((final / capital - 1) * 100, 2),
            "cagr_pct": round(cagr, 2), "max_dd_pct": round(mdd, 2),
            "calmar": round(cagr / max(0.01, mdd), 2),
            "trades": len(trades), "wins": wins, "losses": losses,
            "win_rate_pct": round(wins / max(1, wins + losses) * 100, 1),
            "open_position": open_pos,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    return final, cagr, trades


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=DEFAULT_START.isoformat())
    ap.add_argument("--to",   dest="end",   default=DEFAULT_END.isoformat())
    ap.add_argument("--capital", type=float, default=DEFAULT_CAP)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(date.fromisoformat(a.start), date.fromisoformat(a.end), a.capital,
        Path(a.out) if a.out else None)
