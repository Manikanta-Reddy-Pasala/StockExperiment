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

Result (full-cycle 2021-04-01 → 2026-05-29, ₹10L, authoritative PIT universe)
=============================================================================
See exports/models/midcap_narrow_60d_breakout/SUMMARY.md for live numbers.

| Metric    | Value           |
|-----------|----------------:|
| CAGR      | **+1.65%**      |
| Max DD    | **68.17%**      |
| Calmar    | **0.02**        |
| Trades    | 16              |
| WR        | 37.5%           |

⚠ Effectively DEAD on authoritative PIT data. Earlier sweep showed +141%/Calmar
17 on the old 2023-26 window with the broken N100 exclusion — that was riding
large-cap winners leaked through the buggy Wayback membership. Correct PIT N100
exclusion (2026-05-31) ⇒ no edge. Kept for research only.

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
from tools.shared.universes import nifty500_symbols  # noqa: F401
from tools.shared.breakout_strategy import is_breakout, breakout_exit_reason
from tools.shared.index_membership import eligible_at, universe_union
from tools.shared.market_cap import classify_pit

# Strategy params (V2 winner)
HH_WIN     = 40
VOL_MULT   = 2.0
SMA_LONG   = 200
TRAIL_PCT  = 0.20
PROFIT_TRIG = 0.10
TARGET_PCT = 1.00
STOP_PCT   = 0.20  # Catastrophe stop (sweep-chosen). Fires rarely — clear of the -15.7%
                   # deepest winner dip — so 0 CAGR cost on 3yr while capping the tail.
MAX_HOLD   = 120  # Was 90. 120d max-hold won an OLD sweep (pre-PIT-rebuild); on
                  # authoritative PIT data midcap is now ~flat (+1.65% full-cycle).
# SMA20 exit was tried and DISABLED (leaked winners on dips); exit lives in the shared core.

# Universe params (V3 winner: top-100 ADV minus Large, was skip-30+take-100)
ADV_WIN    = 20
SKIP_TOP   = 0     # V3: top-100 ADV from N500 (instead of skip-top-30)
KEEP_NEXT  = 100   # Take top 100. Large filter applied below via NSE Nifty 100 CSV.

N100_CSV = str(ROOT / "src" / "data" / "symbols" / "nifty100.csv")

DEFAULT_START = date(2021, 4, 1)
DEFAULT_END   = date(2026, 5, 29)
DEFAULT_CAP   = 1_000_000.0


def load_n100_pit(d: date) -> set[str]:
    """Point-in-time NSE Nifty 100 exclusion set for date `d`."""
    return set(eligible_at("n100", d))


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
    print("Large-cap exclusion source: PIT n100 (eligible_at per scan day)")

    eng = _get_engine()
    # PIT n500 union — every symbol that was ever an n500 member.
    n500 = [f"NSE:{s}-EQ" for s in sorted(universe_union("n500"))]
    print(f"N500 union pool (PIT): {len(n500)}")

    with eng.connect() as c:
        # Pull 400 extra calendar days BEFORE `start` so the 200d SMA and other
        # rolling windows are already warmed up on the first simulated day.
        df = pd.read_sql(text(
            "SELECT symbol,date,open,high,low,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b ORDER BY symbol,date"
        ), c, params={"s": n500, "a": start - timedelta(days=400), "b": end})

    df["date"] = pd.to_datetime(df["date"])

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

    # Yearly-PIT midcap pool: at each year-start, rank PIT n500 members by
    # 20d-ADV and take top-100 minus PIT n100. Replaces the previous single
    # end-of-data snapshot which silently included future winners (ZOMATO,
    # NUVAMA etc.) and excluded delisted members of the early era.
    year_pools: dict[pd.Timestamp, list[str]] = {}
    ys_cursor = start
    year_starts = []
    while ys_cursor <= end:
        year_starts.append(pd.Timestamp(ys_cursor))
        ys_cursor = ys_cursor.replace(year=ys_cursor.year + 1)
    for ys in year_starts:
        fut = dates[dates >= ys]
        if len(fut) == 0:
            continue
        di_ys = dates.get_loc(fut[0])
        ys_date = fut[0].date()
        # PIT n500 eligible on year-start.
        n500_elig = {f"NSE:{s}-EQ" for s in eligible_at("n500", ys_date)}
        adv_at_ys = adv20.iloc[di_ys].dropna().sort_values(ascending=False)
        adv_at_ys = adv_at_ys[adv_at_ys.index.isin(n500_elig)]
        adv_at_ys = adv_at_ys[adv_at_ys.index.isin(fresh_syms)]
        midcap_pool = adv_at_ys.iloc[SKIP_TOP:SKIP_TOP + KEEP_NEXT].index.tolist()
        # PIT n100 exclusion on year-start.
        n100_pit = load_n100_pit(ys_date)
        year_pools[ys] = [
            s for s in midcap_pool
            if s.replace("NSE:", "").replace("-EQ", "") not in n100_pit
        ]

    def pick_band(d) -> list[str]:
        """Return the PIT midcap_band in force on date `d` (latest year-start
        on/before `d`).
        """
        chosen = year_starts[0]
        for ys in year_starts:
            if d >= ys:
                chosen = ys
        return year_pools.get(chosen, [])

    initial = pick_band(year_starts[0])
    print(f"V2 universe (PIT pseudo-midcap minus PIT large), year-1 size = "
          f"{len(initial)} stocks")
    print(f"First 10: {[s.replace('NSE:','').replace('-EQ','') for s in initial[:10]]}")

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
                    "cap":        classify_pit(pos["sym"], pos["entry_date"]),
                })
                pos = None

        if pos is None:
            # Flat: scan every universe name for a qualifying breakout TODAY.
            # The PIT midcap_band rolls forward at each year-start.
            midcap_band = pick_band(d)
            # TRADE-TIME large-cap exclusion (2026-05-31 fix): the band freezes the
            # N100 exclusion at year-start, so a name PROMOTED to Nifty 100 mid-year
            # (JINDALSTEL, CHOLAFIN, BANKBARODA, ...) used to leak into this mid/small
            # model. Re-exclude PIT n100 as of the scan day so midcap stays mid/small.
            n100_today = load_n100_pit(dates[di].date())
            cands = []
            for sym in midcap_band:
                if sym not in cl.columns:
                    continue
                if sym.replace("NSE:", "").replace("-EQ", "") in n100_today:
                    continue  # currently large-cap (in Nifty 100) — not a midcap trade
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
