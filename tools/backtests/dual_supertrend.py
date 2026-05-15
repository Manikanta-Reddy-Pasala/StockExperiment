"""Dual Supertrend positional backtest on Nifty 100.

Strategy:
  Fast Supertrend: period=7, multiplier=3
  Slow Supertrend: period=10, multiplier=4
  Entry: BOTH fast and slow trends bullish (close > both Supertrend values).
  Exit:  BOTH fast and slow trends bearish.
  Trailing stop = slow Supertrend line (close holding above it).

Portfolio:
  Capital ₹10L, max 5 concurrent positions, equal-weight per slot.
  Round-trip cost 0.13%.

Window: 2023-05-13 -> 2026-05-12 (3 years).

Run via:
    venv/bin/python tools/backtests/dual_supertrend.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.database import get_database_manager  # noqa: E402

# ---------- config ----------
START_DATE = "2023-05-13"
END_DATE = "2026-05-12"
WARMUP_DAYS = 80  # need enough bars for ATR + Supertrend settle
CAPITAL = 1_000_000.0
MAX_POSITIONS = 5
COST_RT = 0.0013
UNIVERSE_FILE = "/app/logs/momrot/universes/n100_current.json"

FAST_PERIOD = 7
FAST_MULT = 3.0
SLOW_PERIOD = 10
SLOW_MULT = 4.0

OUT_DIR = ROOT / "logs" / "dual_supertrend"


# ---------- supertrend ----------
def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder's smoothing (RMA)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def supertrend(df: pd.DataFrame, period: int, mult: float) -> pd.DataFrame:
    """Return DataFrame with cols [st, trend] where trend=+1 bullish / -1 bearish.

    Standard Supertrend (TradingView formula):
      basic_upper = hl2 + mult*ATR;  basic_lower = hl2 - mult*ATR
      final_upper[i] = basic_upper[i] if (basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]) else final_upper[i-1]
      final_lower[i] = basic_lower[i] if (basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]) else final_lower[i-1]
      trend flips up when close > final_upper(prev), flips down when close < final_lower(prev)
    """
    hl2 = ((df["high"] + df["low"]) / 2.0).to_numpy()
    atr = _atr(df, period).to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    basic_upper = hl2 + mult * atr
    basic_lower = hl2 - mult * atr

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    st = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)

    # find first index where ATR is valid
    start = 0
    while start < n and (np.isnan(atr[start])):
        start += 1
    if start >= n:
        return pd.DataFrame({"st": st, "trend": trend}, index=df.index)

    final_upper[start] = basic_upper[start]
    final_lower[start] = basic_lower[start]
    # initialise trend: pick down if close <= basic_upper, else up
    trend[start] = 1 if close[start] > basic_upper[start] else -1
    st[start] = final_lower[start] if trend[start] == 1 else final_upper[start]

    for i in range(start + 1, n):
        # final bands
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        prev_trend = trend[i - 1]
        c = close[i]
        if prev_trend == -1 and c > final_upper[i - 1]:
            trend[i] = 1
            st[i] = final_lower[i]
        elif prev_trend == 1 and c < final_lower[i - 1]:
            trend[i] = -1
            st[i] = final_upper[i]
        else:
            trend[i] = prev_trend
            st[i] = final_lower[i] if prev_trend == 1 else final_upper[i]

    return pd.DataFrame({"st": st, "trend": trend}, index=df.index)


# ---------- data ----------
def load_universe() -> List[str]:
    with open(UNIVERSE_FILE) as f:
        d = json.load(f)
    return [s["symbol"] for s in d["stocks"]]


def fyers_sym(s: str) -> str:
    return f"NSE:{s}-EQ"


def fetch_daily(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Bulk load daily OHLCV for all symbols. Returns dict symbol -> df indexed by date."""
    eng = get_database_manager().engine
    fyers_syms = [fyers_sym(s) for s in symbols]
    # widen start by warmup for ATR settle
    start_ext = (pd.to_datetime(start) - pd.Timedelta(days=WARMUP_DAYS * 2)).date()
    q = text(
        "SELECT symbol, date, open, high, low, close, volume "
        "FROM historical_data "
        "WHERE symbol = ANY(:syms) AND date BETWEEN :a AND :b "
        "ORDER BY symbol, date"
    )
    with eng.connect() as conn:
        df = pd.read_sql(
            q, conn, params={"syms": fyers_syms, "a": start_ext, "b": pd.to_datetime(end).date()}
        )
    out: Dict[str, pd.DataFrame] = {}
    if df.empty:
        return out
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"])
    for sym, g in df.groupby("symbol"):
        plain = sym.replace("NSE:", "").replace("-EQ", "")
        g2 = g.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
        out[plain] = g2[["open", "high", "low", "close", "volume"]]
    return out


# ---------- signals ----------
@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_px: float
    exit_date: pd.Timestamp
    exit_px: float
    bars_held: int

    @property
    def pnl_pct(self) -> float:
        gross = (self.exit_px / self.entry_px) - 1.0
        return gross - COST_RT

    @property
    def gross_pnl_pct(self) -> float:
        return (self.exit_px / self.entry_px) - 1.0


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fast + slow Supertrends and entry/exit signals.
    Trades use NEXT-DAY OPEN execution to avoid lookahead.
    """
    fast = supertrend(df, FAST_PERIOD, FAST_MULT)
    slow = supertrend(df, SLOW_PERIOD, SLOW_MULT)
    sig = pd.DataFrame(index=df.index)
    sig["close"] = df["close"]
    sig["open"] = df["open"]
    sig["fast_st"] = fast["st"]
    sig["fast_tr"] = fast["trend"]
    sig["slow_st"] = slow["st"]
    sig["slow_tr"] = slow["trend"]
    sig["both_bull"] = (fast["trend"] == 1) & (slow["trend"] == 1)
    sig["both_bear"] = (fast["trend"] == -1) & (slow["trend"] == -1)
    return sig


def generate_trades(symbol: str, sig: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> List[Trade]:
    """Walk the signal series. Enter on next bar's open after both_bull becomes true.
    Exit on next bar's open after both_bear becomes true.
    Constrains entries/exits to [start, end]. Force-closes any open trade at end.
    """
    trades: List[Trade] = []
    in_pos = False
    entry_idx: Optional[int] = None
    entry_px: Optional[float] = None
    entry_dt: Optional[pd.Timestamp] = None
    idx = sig.index
    bb = sig["both_bull"].values
    be = sig["both_bear"].values
    opens = sig["open"].values
    # state: was-bull / was-bear (need to track sign-flips to fire ONCE per regime)
    prev_bull = False
    prev_bear = False
    for i in range(len(sig)):
        dt = idx[i]
        if dt < start - pd.Timedelta(days=30):
            prev_bull = bool(bb[i])
            prev_bear = bool(be[i])
            continue
        # entry: when both_bull transitions from False -> True, enter at NEXT open
        if (not in_pos) and bb[i] and (not prev_bull) and (i + 1 < len(sig)):
            entry_dt_cand = idx[i + 1]
            if start <= entry_dt_cand <= end:
                entry_idx = i + 1
                entry_dt = entry_dt_cand
                entry_px = float(opens[i + 1])
                in_pos = True
        # exit: when both_bear transitions False -> True, exit at NEXT open
        elif in_pos and be[i] and (not prev_bear) and (i + 1 < len(sig)):
            exit_dt = idx[i + 1]
            exit_px = float(opens[i + 1])
            trades.append(
                Trade(
                    symbol=symbol,
                    entry_date=entry_dt,
                    entry_px=entry_px,
                    exit_date=exit_dt,
                    exit_px=exit_px,
                    bars_held=(i + 1) - entry_idx,
                )
            )
            in_pos = False
            entry_idx = entry_px = entry_dt = None
        prev_bull = bool(bb[i])
        prev_bear = bool(be[i])
    # force-close open trade at last bar within window
    if in_pos and entry_dt is not None:
        last_i = len(sig) - 1
        while last_i >= 0 and idx[last_i] > end:
            last_i -= 1
        if last_i > (entry_idx or 0):
            trades.append(
                Trade(
                    symbol=symbol,
                    entry_date=entry_dt,
                    entry_px=entry_px,
                    exit_date=idx[last_i],
                    exit_px=float(sig["close"].iloc[last_i]),
                    bars_held=last_i - entry_idx,
                )
            )
    return trades


# ---------- portfolio sim ----------
@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_px: float
    exit_px: float
    qty: float = 0.0
    capital_used: float = 0.0

    @property
    def gross_value_at_exit(self) -> float:
        return self.qty * self.exit_px


def portfolio_sim(all_trades: List[Trade], start: pd.Timestamp, end: pd.Timestamp) -> Dict:
    """Greedy walk-forward portfolio: max 5 concurrent, equal-weight cash/open_slots at entry.
    Trades are sorted by entry_date and assigned to available slots."""
    trades_sorted = sorted(all_trades, key=lambda t: (t.entry_date, t.symbol))
    cash = CAPITAL
    open_pos: List[Position] = []
    closed: List[Position] = []
    skipped = 0
    # equity timeline (mark-to-market on close prices) — build at month-end granularity for speed
    # we use trade-event-driven cash tracking for ROI; for Sharpe/DD we approximate via daily eq curve below.

    # First: walk trade events in time order; release cash on exit events that occur first.
    events: List[Tuple[pd.Timestamp, str, Trade, Optional[Position]]] = []
    for t in trades_sorted:
        events.append((t.entry_date, "entry", t, None))
    # We'll inject exits dynamically as positions open.

    # Process trades in entry order, but also handle pending exits chronologically.
    pending_exits: List[Position] = []
    for t in trades_sorted:
        # release any positions whose exit_date <= this entry_date
        pending_exits.sort(key=lambda p: p.exit_date)
        while pending_exits and pending_exits[0].exit_date <= t.entry_date:
            p = pending_exits.pop(0)
            proceeds = p.qty * p.exit_px * (1 - COST_RT / 2)  # half cost on exit
            cash += proceeds
            open_pos.remove(p)
            closed.append(p)
        # capacity check
        if len(open_pos) >= MAX_POSITIONS:
            skipped += 1
            continue
        slots_free = MAX_POSITIONS - len(open_pos)
        alloc = cash / slots_free if slots_free > 0 else 0.0
        if alloc <= 0 or t.entry_px <= 0:
            skipped += 1
            continue
        # apply half cost on entry
        effective_alloc = alloc * (1 - COST_RT / 2)
        qty = effective_alloc / t.entry_px
        if qty <= 0:
            skipped += 1
            continue
        cash -= alloc  # full alloc removed; the half-cost is implicit in qty discount
        pos = Position(
            symbol=t.symbol,
            entry_date=t.entry_date,
            exit_date=t.exit_date,
            entry_px=t.entry_px,
            exit_px=t.exit_px,
            qty=qty,
            capital_used=alloc,
        )
        open_pos.append(pos)
        pending_exits.append(pos)
    # close any remaining
    pending_exits.sort(key=lambda p: p.exit_date)
    for p in pending_exits:
        proceeds = p.qty * p.exit_px * (1 - COST_RT / 2)
        cash += proceeds
        if p in open_pos:
            open_pos.remove(p)
        closed.append(p)

    final_equity = cash
    roi = final_equity / CAPITAL - 1.0
    return {
        "final_equity": final_equity,
        "roi": roi,
        "skipped_due_to_capacity": skipped,
        "executed_trades": closed,
    }


# ---------- analytics ----------
def daily_equity_curve(closed: List[Position], daily_data: Dict[str, pd.DataFrame],
                       start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Approximate daily equity by marking open positions to close price each day."""
    if not closed:
        return pd.Series(dtype=float)
    # build calendar from any one symbol with full history
    cal_src = None
    for sym in daily_data:
        df = daily_data[sym]
        sub = df[(df.index >= start) & (df.index <= end)]
        if len(sub) > 100:
            cal_src = sub.index
            break
    if cal_src is None:
        return pd.Series(dtype=float)
    cal = pd.DatetimeIndex(sorted(set(cal_src)))
    cal = cal[(cal >= start) & (cal <= end)]
    cash = CAPITAL
    eq = pd.Series(index=cal, dtype=float)
    # sort positions
    pos_by_entry = sorted(closed, key=lambda p: p.entry_date)
    # iterate over calendar
    open_p: List[Position] = []
    p_idx = 0
    for d in cal:
        # close any positions whose exit_date <= d (at exit_px)
        still_open: List[Position] = []
        for p in open_p:
            if p.exit_date <= d:
                cash += p.qty * p.exit_px * (1 - COST_RT / 2)
            else:
                still_open.append(p)
        open_p = still_open
        # open new positions on this date
        while p_idx < len(pos_by_entry) and pos_by_entry[p_idx].entry_date <= d:
            p = pos_by_entry[p_idx]
            # cash already deducted in main sim; mirror here:
            cash -= p.capital_used
            open_p.append(p)
            p_idx += 1
        # mark to market
        mv = 0.0
        for p in open_p:
            sd = daily_data.get(p.symbol)
            if sd is None or d not in sd.index:
                # find last available price
                if sd is not None:
                    sub = sd[sd.index <= d]
                    if len(sub) > 0:
                        mv += p.qty * float(sub["close"].iloc[-1])
                    else:
                        mv += p.qty * p.entry_px
                else:
                    mv += p.qty * p.entry_px
            else:
                mv += p.qty * float(sd.loc[d, "close"])
        eq.loc[d] = cash + mv
    return eq.dropna()


def stats_from_equity(eq: pd.Series) -> Dict:
    if len(eq) < 2:
        return {"sharpe": 0.0, "max_dd": 0.0, "cagr": 0.0}
    rets = eq.pct_change().dropna()
    mu = rets.mean()
    sd = rets.std()
    sharpe = (mu / sd * math.sqrt(252)) if sd > 0 else 0.0
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 and eq.iloc[0] > 0 else 0.0
    return {"sharpe": sharpe, "max_dd": dd, "cagr": cagr}


# ---------- run ----------
def run() -> Dict:
    print(f"Loading universe from {UNIVERSE_FILE}")
    universe = load_universe()
    print(f"  -> {len(universe)} symbols")

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    print(f"Fetching daily data {start.date()} -> {end.date()} (+{WARMUP_DAYS*2}d warmup)")
    daily = fetch_daily(universe, START_DATE, END_DATE)
    print(f"  -> {len(daily)} symbols with data")

    all_trades: List[Trade] = []
    per_stock: Dict[str, List[Trade]] = {}

    for sym in universe:
        df = daily.get(sym)
        if df is None or len(df) < FAST_PERIOD + SLOW_PERIOD + 10:
            continue
        sig = build_signals(df)
        trades = generate_trades(sym, sig, start, end)
        if trades:
            per_stock[sym] = trades
            all_trades.extend(trades)

    print(f"Generated {len(all_trades)} candidate trades across {len(per_stock)} stocks")

    port = portfolio_sim(all_trades, start, end)
    print(f"Portfolio: ROI={port['roi']*100:.2f}%, executed={len(port['executed_trades'])}, "
          f"skipped(cap)={port['skipped_due_to_capacity']}")

    # equity curve / Sharpe / DD
    eq = daily_equity_curve(port["executed_trades"], daily, start, end)
    st = stats_from_equity(eq)
    print(f"  CAGR={st['cagr']*100:.2f}%  Sharpe={st['sharpe']:.2f}  MaxDD={st['max_dd']*100:.2f}%")

    # per-stock stats
    per_stock_stats = []
    for sym, trs in per_stock.items():
        if not trs:
            continue
        wins = sum(1 for t in trs if t.pnl_pct > 0)
        total_pnl = sum(t.pnl_pct for t in trs) * 100
        avg_pnl = total_pnl / len(trs)
        avg_hold = sum(t.bars_held for t in trs) / len(trs)
        per_stock_stats.append(
            {
                "symbol": sym,
                "trades": len(trs),
                "win_rate": wins / len(trs),
                "total_pnl_pct": total_pnl,
                "avg_pnl_pct": avg_pnl,
                "avg_hold_bars": avg_hold,
            }
        )
    per_stock_stats.sort(key=lambda r: -r["total_pnl_pct"])

    # annual splits (on portfolio-EXECUTED trades, by entry_date year-bucket)
    def in_year_bucket(d: pd.Timestamp, y0: str, y1: str) -> bool:
        return pd.Timestamp(y0) <= d < pd.Timestamp(y1)

    buckets = [
        ("2023-24_bull", "2023-05-13", "2024-05-13"),
        ("2024-25_mixed", "2024-05-13", "2025-05-13"),
        ("2025-26_recent", "2025-05-13", "2026-05-13"),
    ]
    annual = []
    for name, a, b in buckets:
        # use executed trades from portfolio (those that actually got capital)
        bucket_trades = []
        for p in port["executed_trades"]:
            if in_year_bucket(p.entry_date, a, b):
                gross = (p.exit_px / p.entry_px) - 1.0
                net = gross - COST_RT
                bucket_trades.append(net)
        n = len(bucket_trades)
        wr = (sum(1 for x in bucket_trades if x > 0) / n) if n else 0.0
        avg = (sum(bucket_trades) / n * 100) if n else 0.0
        # ROI for the period: from equity curve
        if len(eq) > 0 and isinstance(eq.index, pd.DatetimeIndex):
            eq_a = eq[(eq.index >= pd.Timestamp(a)) & (eq.index < pd.Timestamp(b))]
            if len(eq_a) >= 2:
                yroi = eq_a.iloc[-1] / eq_a.iloc[0] - 1.0
            else:
                yroi = 0.0
        else:
            yroi = 0.0
        # ALSO compute strategy-level (all-signaled) win rate ignoring capacity cap
        sig_bucket = [t for t in all_trades if in_year_bucket(t.entry_date, a, b)]
        sig_pnls = [t.pnl_pct for t in sig_bucket]
        sig_wr = (sum(1 for x in sig_pnls if x > 0) / len(sig_pnls)) if sig_pnls else 0.0
        sig_avg = (sum(sig_pnls) / len(sig_pnls) * 100) if sig_pnls else 0.0
        annual.append({
            "period": name,
            "trades": n,
            "win_rate": wr,
            "avg_pnl_pct": avg,
            "period_roi_pct": yroi * 100,
            "signaled_trades": len(sig_pnls),
            "signaled_win_rate": sig_wr,
            "signaled_avg_pnl_pct": sig_avg,
        })

    # overall trade-level stats (executed trades only)
    exec_trades = port["executed_trades"]
    exec_pnls = [(p.exit_px / p.entry_px - 1.0 - COST_RT) for p in exec_trades]
    exec_holds = [(p.exit_date - p.entry_date).days for p in exec_trades]
    overall = {
        "total_trades_signaled": len(all_trades),
        "executed_trades": len(exec_trades),
        "win_rate_executed": sum(1 for x in exec_pnls if x > 0) / len(exec_pnls) if exec_pnls else 0.0,
        "avg_trade_pnl_pct": (sum(exec_pnls) / len(exec_pnls) * 100) if exec_pnls else 0.0,
        "avg_hold_days": (sum(exec_holds) / len(exec_holds)) if exec_holds else 0.0,
        "total_roi_pct": port["roi"] * 100,
        "cagr_pct": st["cagr"] * 100,
        "sharpe": st["sharpe"],
        "max_dd_pct": st["max_dd"] * 100,
        "skipped_capacity": port["skipped_due_to_capacity"],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "start": START_DATE,
            "end": END_DATE,
            "capital": CAPITAL,
            "max_positions": MAX_POSITIONS,
            "cost_rt": COST_RT,
            "fast": [FAST_PERIOD, FAST_MULT],
            "slow": [SLOW_PERIOD, SLOW_MULT],
            "universe": "n100_current",
        },
        "overall": overall,
        "annual": annual,
        "top10_by_pnl": per_stock_stats[:10],
        "bottom10_by_pnl": per_stock_stats[-10:],
        "all_per_stock": per_stock_stats,
    }
    (OUT_DIR / "dual_supertrend_report.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_DIR / 'dual_supertrend_report.json'}")

    # plain-text summary
    lines = []
    lines.append("=" * 70)
    lines.append("DUAL SUPERTREND BACKTEST — Nifty 100, 2023-05-13 -> 2026-05-12")
    lines.append("Fast(7,3) + Slow(10,4), max 5 concurrent, ₹10L, 0.13% RT cost")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OVERALL (3-yr):")
    for k, v in overall.items():
        if isinstance(v, float):
            lines.append(f"  {k:30s} {v:8.2f}")
        else:
            lines.append(f"  {k:30s} {v}")
    lines.append("")
    lines.append("ANNUAL BREAKDOWN (executed = capacity-limited portfolio):")
    lines.append(f"  {'period':20s} {'trades':>7s} {'win_rate':>10s} {'avg_pnl%':>10s} {'roi%':>10s}")
    for a in annual:
        lines.append(
            f"  {a['period']:20s} {a['trades']:>7d} {a['win_rate']*100:>9.1f}% "
            f"{a['avg_pnl_pct']:>10.2f} {a['period_roi_pct']:>10.2f}"
        )
    lines.append("")
    lines.append("ANNUAL — STRATEGY LEVEL (every signaled trade, unconstrained):")
    lines.append(f"  {'period':20s} {'trades':>7s} {'win_rate':>10s} {'avg_pnl%':>10s}")
    for a in annual:
        lines.append(
            f"  {a['period']:20s} {a['signaled_trades']:>7d} "
            f"{a['signaled_win_rate']*100:>9.1f}% {a['signaled_avg_pnl_pct']:>10.2f}"
        )
    lines.append("")
    lines.append("TOP 10 STOCKS by total P&L %:")
    lines.append(f"  {'symbol':12s} {'trades':>7s} {'win%':>7s} {'tot_pnl%':>10s} {'avg%':>8s} {'hold_d':>8s}")
    for r in per_stock_stats[:10]:
        lines.append(
            f"  {r['symbol']:12s} {r['trades']:>7d} {r['win_rate']*100:>6.1f}% "
            f"{r['total_pnl_pct']:>10.2f} {r['avg_pnl_pct']:>8.2f} {r['avg_hold_bars']:>8.1f}"
        )
    lines.append("")
    lines.append("BOTTOM 10:")
    for r in per_stock_stats[-10:]:
        lines.append(
            f"  {r['symbol']:12s} {r['trades']:>7d} {r['win_rate']*100:>6.1f}% "
            f"{r['total_pnl_pct']:>10.2f} {r['avg_pnl_pct']:>8.2f} {r['avg_hold_bars']:>8.1f}"
        )
    summary_txt = "\n".join(lines)
    (OUT_DIR / "dual_supertrend_summary.txt").write_text(summary_txt)
    print(summary_txt)

    return out


if __name__ == "__main__":
    run()
