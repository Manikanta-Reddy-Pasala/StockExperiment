"""Backtest — momentum_n100_top3_blended (v2).

Pure first-principles design, no in-sample tuning:
  - Blended momentum (z-score average of 21d + 63d + 126d returns)
  - Skip-week formation gap (lookback ends 5 trading days ago)
  - Real NSE Nifty 100 universe
  - Top-3 equal-weight, max=3 concurrent
  - Soft exit (hold while in top-10)
  - Index trend gate (NIFTY50 INDEX as N100 proxy, 200d SMA filter)
  - Realistic costs: 0.10% slippage per side + STT 0.10% sell + ₹20 brokerage
  - STCG 20% applied on yearly net P&L (each FY end)

Run from inside trading_system_app container:
  python3 tools/models/momentum_n100_top3_blended/backtest.py
"""
from __future__ import annotations

import sys
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import read_cached  # noqa: E402

# -------- knobs (none tuned on this dataset) --------
LOOKBACKS = [21, 63, 126]   # ~1m + 3m + 6m (AQR blend)
SKIP_DAYS = 5               # skip last week (Jegadeesh/Titman)
TOP_N = 3                   # max concurrent positions
SOFT_EXIT_RANK = 10         # hold while in top-10
INDEX_SMA = 200             # Faber trend filter
SLIP_PCT = 0.10 / 100       # 0.10% per side
STT_PCT = 0.10 / 100        # sell side only (delivery)
BROKERAGE = 20.0            # flat per trade
STCG_RATE = 0.20            # short-term cap gains tax (Indian eq <1y)

INDEX_SYM = "NSE:NIFTY50-INDEX"   # proxy for N100 (corr ~0.98)
N100_CSV = "/app/src/data/symbols/nifty100.csv"
CAPITAL_START = 1_000_000.0
DAYS_BACK = 365 * 3 + 200    # 3y + warmup for SMA200/LB126


def load_universe() -> List[str]:
    syms = []
    with open(N100_CSV) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            s = (r.get("Symbol") or r.get("symbol") or "").strip()
            if s:
                syms.append(s)
    return list(dict.fromkeys(syms))


def load_panel(symbols: List[str], days_back: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Returns (close_panel_DataFrame[date x symbol], index_close_Series)."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)
    s0, s1 = int(start_dt.timestamp()), int(end_dt.timestamp())

    closes: Dict[str, pd.Series] = {}
    for sym in symbols:
        fyers_sym = f"NSE:{sym}-EQ"
        df = read_cached(fyers_sym, "D", s0, s1)
        if df.empty:
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").drop_duplicates("date", keep="last")
        closes[sym] = df.set_index("date")["close"].astype(float)

    if not closes:
        raise RuntimeError("no close data — DB empty?")

    panel = pd.DataFrame(closes).sort_index()
    panel = panel.ffill(limit=3)   # tolerate single-day holiday gaps

    # index
    idx_df = read_cached(INDEX_SYM, "D", s0, s1)
    if idx_df.empty:
        raise RuntimeError(f"no {INDEX_SYM} data")
    idx_df = idx_df.copy()
    idx_df["date"] = pd.to_datetime(idx_df["date"]).dt.normalize()
    idx_df = idx_df.sort_values("date").drop_duplicates("date", keep="last")
    idx = idx_df.set_index("date")["close"].astype(float)
    idx = idx.reindex(panel.index).ffill(limit=3)

    return panel, idx


def blended_zscore_rank(panel: pd.DataFrame, di: int) -> pd.Series:
    """For each stock, z-score average of returns at lookbacks. Skip last
    SKIP_DAYS bars (formation gap)."""
    end_i = di - SKIP_DAYS
    if end_i < max(LOOKBACKS):
        return pd.Series(dtype=float)
    end_px = panel.iloc[end_i]
    parts = []
    for lb in LOOKBACKS:
        start_i = end_i - lb
        if start_i < 0:
            return pd.Series(dtype=float)
        start_px = panel.iloc[start_i]
        rets = (end_px / start_px - 1.0)
        # z-score across universe (cross-sectional standardisation)
        z = (rets - rets.mean()) / (rets.std() + 1e-9)
        parts.append(z)
    blend = pd.concat(parts, axis=1).mean(axis=1)
    return blend.dropna().sort_values(ascending=False)


def first_weekday_of_month(d: pd.Timestamp) -> pd.Timestamp:
    """First Mon-Fri of d's month."""
    x = pd.Timestamp(d.year, d.month, 1)
    while x.weekday() >= 5:
        x += pd.Timedelta(days=1)
    return x


def main():
    print("Loading universe + OHLCV…")
    universe = load_universe()
    print(f"  N100 universe: {len(universe)} symbols")
    panel, idx = load_panel(universe, DAYS_BACK)
    print(f"  Loaded {panel.shape[1]} cols x {panel.shape[0]} dates "
          f"(range {panel.index.min().date()} → {panel.index.max().date()})")
    idx_sma = idx.rolling(INDEX_SMA).mean()

    dates = panel.index
    # backtest start = first date where SMA200 + lookback126 available
    bt_start_i = max(INDEX_SMA, max(LOOKBACKS) + SKIP_DAYS) + 1
    bt_start = dates[bt_start_i]
    bt_end = dates[-1]
    print(f"  Backtest window: {bt_start.date()} → {bt_end.date()} "
          f"({(bt_end - bt_start).days} days, ~{(bt_end - bt_start).days//365}y)")

    # build month-1-weekday set
    rebal_days = set()
    cur = pd.Timestamp(bt_start.year, bt_start.month, 1)
    while cur <= bt_end:
        w = first_weekday_of_month(cur)
        rebal_days.add(w.date())
        if cur.month == 12:
            cur = pd.Timestamp(cur.year + 1, 1, 1)
        else:
            cur = pd.Timestamp(cur.year, cur.month + 1, 1)

    cash = CAPITAL_START
    holds: Dict[str, Dict] = {}   # symbol → {qty, entry_px, entry_date}
    trades = []
    nav_track = []
    fy_realized_pnl: Dict[int, float] = {}   # Indian FY: Apr-Mar
    last_in_market = True

    def _fy(d: pd.Timestamp) -> int:
        return d.year if d.month >= 4 else d.year - 1

    def _apply_stcg():
        nonlocal cash
        for fy, pnl in list(fy_realized_pnl.items()):
            if pnl > 0:
                tax = pnl * STCG_RATE
                cash -= tax
                print(f"    STCG FY{fy}-{fy+1}: net P&L ₹{pnl:,.0f} → tax ₹{tax:,.0f}")
            fy_realized_pnl[fy] = 0.0   # reset after tax taken

    last_fy_seen = _fy(bt_start)

    for i in range(bt_start_i, len(dates)):
        d = dates[i]
        # NAV
        mv = sum(h["qty"] * panel[s].iloc[i] for s, h in holds.items()
                 if not pd.isna(panel[s].iloc[i]))
        nav = cash + mv
        nav_track.append((d, nav))

        # FY rollover → STCG
        if _fy(d) != last_fy_seen:
            _apply_stcg()
            last_fy_seen = _fy(d)

        if d.date() not in rebal_days:
            continue

        # Index trend gate
        idx_val = idx.iloc[i]
        sma_val = idx_sma.iloc[i]
        in_market = bool(pd.notna(idx_val) and pd.notna(sma_val) and idx_val > sma_val)

        rank = blended_zscore_rank(panel, i)
        if rank.empty:
            continue

        target_set: List[str] = []
        if in_market:
            # filter to symbols with valid current price
            valid = [s for s in rank.index if not pd.isna(panel[s].iloc[i])]
            target_set = valid[:TOP_N]

        # Decide sells: out of top-10 (soft exit) OR market gate off
        top_k = set(rank.head(SOFT_EXIT_RANK).index) if in_market else set()
        sells = [s for s in list(holds.keys()) if s not in top_k]

        # Execute sells
        for s in sells:
            qty = holds[s]["qty"]
            px = panel[s].iloc[i]
            if pd.isna(px):
                continue
            exit_px = px * (1.0 - SLIP_PCT)
            proceeds = qty * exit_px
            fees = proceeds * STT_PCT + BROKERAGE
            cash += (proceeds - fees)
            pnl = (exit_px - holds[s]["entry_px"]) * qty - fees - (qty * holds[s]["entry_px"] * SLIP_PCT)
            fy_realized_pnl[_fy(d)] = fy_realized_pnl.get(_fy(d), 0.0) + pnl
            trades.append({
                "date": d.date().isoformat(), "side": "SELL", "sym": s, "qty": qty,
                "px": round(exit_px, 2), "pnl": round(pnl, 0),
                "reason": "INDEX_GATE_OFF" if not in_market else "SOFT_EXIT",
            })
            del holds[s]

        # Buys
        if in_market:
            need = [s for s in target_set if s not in holds]
            slots_open = TOP_N - len(holds)
            need = need[:slots_open]
            if need:
                per_slot_cash = cash / max(1, len(need))
                for s in need:
                    px = panel[s].iloc[i]
                    if pd.isna(px) or px <= 0:
                        continue
                    entry_px = px * (1.0 + SLIP_PCT)
                    qty = int((per_slot_cash - BROKERAGE) // entry_px)
                    if qty < 1:
                        continue
                    cost = qty * entry_px + BROKERAGE
                    if cost > cash:
                        qty = int((cash - BROKERAGE) // entry_px)
                        if qty < 1:
                            continue
                        cost = qty * entry_px + BROKERAGE
                    cash -= cost
                    holds[s] = {"qty": qty, "entry_px": entry_px, "entry_date": d.date().isoformat()}
                    trades.append({
                        "date": d.date().isoformat(), "side": "BUY", "sym": s, "qty": qty,
                        "px": round(entry_px, 2), "pnl": None, "reason": "ENTER_TOP3",
                    })
        last_in_market = in_market

    # Close all remaining at last bar
    last_d = dates[-1]
    for s, h in list(holds.items()):
        px = panel[s].iloc[-1]
        if pd.isna(px):
            continue
        exit_px = px * (1.0 - SLIP_PCT)
        proceeds = h["qty"] * exit_px
        fees = proceeds * STT_PCT + BROKERAGE
        cash += (proceeds - fees)
        pnl = (exit_px - h["entry_px"]) * h["qty"] - fees - (h["qty"] * h["entry_px"] * SLIP_PCT)
        fy_realized_pnl[_fy(last_d)] = fy_realized_pnl.get(_fy(last_d), 0.0) + pnl
        trades.append({
            "date": last_d.date().isoformat(), "side": "SELL", "sym": s, "qty": h["qty"],
            "px": round(exit_px, 2), "pnl": round(pnl, 0), "reason": "FINAL_LIQUIDATE",
        })
    _apply_stcg()
    holds.clear()

    # Stats
    final_nav = cash
    total_ret = (final_nav / CAPITAL_START - 1.0) * 100
    n_years = (bt_end - bt_start).days / 365.25
    cagr = ((final_nav / CAPITAL_START) ** (1.0 / n_years) - 1.0) * 100

    nav_df = pd.DataFrame(nav_track, columns=["date", "nav"]).set_index("date")
    running_max = nav_df["nav"].cummax()
    dd = (nav_df["nav"] - running_max) / running_max
    max_dd = dd.min() * 100

    sells = [t for t in trades if t["side"] == "SELL" and t.get("pnl") is not None]
    wins = [t for t in sells if t["pnl"] > 0]
    losses = [t for t in sells if t["pnl"] < 0]
    wr = (len(wins) / len(sells) * 100) if sells else 0
    total_buys = len([t for t in trades if t["side"] == "BUY"])

    print()
    print("=" * 60)
    print(f"FINAL NAV          ₹{final_nav:>14,.0f}")
    print(f"Total return       {total_ret:>+10.2f}%")
    print(f"CAGR ({n_years:.2f}y)      {cagr:>+10.2f}%")
    print(f"Max DD             {max_dd:>10.2f}%")
    print(f"Trades  BUY/SELL   {total_buys} / {len(sells)}")
    print(f"WR                 {wr:>10.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"Calmar             {(cagr / abs(max_dd)) if max_dd else 0:>10.2f}")
    print("=" * 60)

    # FY-wise NAV
    print("\nYear-by-year:")
    fy_starts = {}
    for d, n in nav_track:
        fy = _fy(d)
        if fy not in fy_starts:
            fy_starts[fy] = n
        fy_starts[fy + 1000] = n   # final overwrites
    fy_keys = sorted({_fy(d) for d, _ in nav_track})
    for fy in fy_keys:
        navs = [n for d, n in nav_track if _fy(d) == fy]
        if not navs:
            continue
        s, e = navs[0], navs[-1]
        roi = (e / s - 1) * 100
        print(f"  FY{fy}-{fy+1}: ₹{s:>12,.0f} → ₹{e:>12,.0f}  ROI {roi:>+7.2f}%")

    # Worst trades
    print("\nTop 5 losses:")
    for t in sorted(losses, key=lambda x: x["pnl"])[:5]:
        print(f"  {t['date']} SELL {t['sym']:12s} qty={t['qty']:>4}  PnL ₹{t['pnl']:>+12,.0f}  reason={t['reason']}")

    print("\nTop 5 wins:")
    for t in sorted(wins, key=lambda x: -x["pnl"])[:5]:
        print(f"  {t['date']} SELL {t['sym']:12s} qty={t['qty']:>4}  PnL ₹{t['pnl']:>+12,.0f}  reason={t['reason']}")

    out = Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_json(out / "trade_ledger.json", orient="records", indent=2)
    nav_df.to_csv(out / "nav_history.csv")
    print(f"\nWrote {out/'trade_ledger.json'} + {out/'nav_history.csv'}")


if __name__ == "__main__":
    main()
