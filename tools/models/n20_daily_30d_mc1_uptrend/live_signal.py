"""n20_daily_30d_mc1_uptrend — DAILY PIT live signal generator.

Differences from momentum_n100_top5_max1 live_signal:
  - Universe rebuilt **at signal time** from N500 by 20d ADV (top 20), not a frozen file
  - Filter: close > 200d SMA at signal time (uptrend gate)
  - Lookback for ranking: 30 days (not 60)
  - Rebalance: every trading day (not monthly)
  - Position: top-1 (max_concurrent = 1)

Logic per run:
  1. Pull last 250 daily bars for N500 from historical_data DB
  2. Compute PIT universe (top-20 ADV) + uptrend filter at most recent close
  3. Rank filtered universe by 30d return; pick top-1
  4. Load ledger -> current holding
  5. If holding != top-1 -> emit STOP_HIT for old + ENTRY1 for new
  6. Else: no signal

Usage:
  python tools/models/n20_daily_30d_mc1_uptrend/live_signal.py \
    --signals-out /app/logs/n20daily/signals/$(date +%F).json \
    --ledger /app/logs/n20daily/ledger.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.shared.ohlcv_cache import _get_engine  # noqa: E402
from tools.shared.universes import nifty500_symbols  # noqa: E402

UNIV_SIZE = 20
LOOKBACK  = 30
SMA_LONG  = 200
ADV_WIN   = 20

log = logging.getLogger("n20daily_signal")


def compute_pit_picks(asof: date):
    """Return list of (symbol, name, ret_pct, price). Top-20 PIT, filtered, ranked."""
    eng = _get_engine()
    n500 = [f"NSE:{s}-EQ" for s, _ in nifty500_symbols()]

    with eng.connect() as c:
        df = pd.read_sql(text(
            "SELECT symbol,date,close,volume FROM historical_data "
            "WHERE symbol=ANY(:s) AND date BETWEEN :a AND :b ORDER BY symbol,date"
        ), c, params={"s": n500, "a": asof - timedelta(days=400), "b": asof})

    if df.empty:
        log.warning("No historical_data rows. Run data pull first.")
        return []

    df["date"] = pd.to_datetime(df["date"])
    df["adv_rs"] = df["close"].astype(float) * df["volume"].astype(float)
    cl = df.pivot(index="date", columns="symbol", values="close").ffill()
    adv_rs = df.pivot(index="date", columns="symbol", values="adv_rs").fillna(0)
    adv20  = adv_rs.rolling(ADV_WIN).mean()
    sma200 = cl.rolling(SMA_LONG).mean()
    dates = cl.index
    if len(dates) < max(LOOKBACK, SMA_LONG):
        log.warning("Insufficient history.")
        return []

    di = len(dates) - 1
    pit_adv = adv20.iloc[di].dropna().sort_values(ascending=False)
    pit_univ = pit_adv.head(UNIV_SIZE).index.tolist()
    up = sma200.iloc[di] < cl.iloc[di]
    pit_univ = [s for s in pit_univ if bool(up.get(s, False))]
    if not pit_univ:
        log.info("Empty PIT universe after uptrend filter — market in downtrend.")
        return []

    rets = cl.iloc[di].loc[pit_univ] / cl.iloc[di - LOOKBACK].loc[pit_univ] - 1
    rk = rets.dropna().sort_values(ascending=False)
    out = []
    for sym in rk.index:
        plain = sym.replace("NSE:", "").replace("-EQ", "")
        out.append((plain, plain, float(rk[sym] * 100), float(cl[sym].iloc[di])))
    return out


def load_held(ledger_path: Path):
    if not ledger_path or not ledger_path.exists():
        return []
    try:
        return json.loads(ledger_path.read_text()).get("open", [])
    except Exception as e:
        log.warning(f"ledger read fail: {e}")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-out", required=True)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (default today)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    asof = date.fromisoformat(args.asof) if args.asof else date.today()
    log.info(f"n20daily signal asof={asof}")

    picks = compute_pit_picks(asof)
    if not picks:
        Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.signals_out).write_text(json.dumps([]))
        log.info("No picks — empty signals file.")
        return 0

    log.info(f"PIT universe (filtered): {len(picks)} stocks")
    for i, (s, _, r, p) in enumerate(picks[:5], 1):
        log.info(f"  {i}. {s:<14} {r:+7.2f}%  @ ₹{p:.2f}")

    top_sym, top_name, top_ret, top_price = picks[0]

    held = load_held(Path(args.ledger)) if args.ledger else []
    held_syms = {h["symbol"] for h in held}

    signals = []
    today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if top_sym in held_syms:
        log.info(f"Already holding {top_sym}; no signal.")
    else:
        for h in held:
            signals.append({
                "model": "n20_daily_30d_mc1_uptrend",
                "symbol": h["symbol"], "company": h["symbol"],
                "ts": today_str, "side": "BUY", "signal": "STOP_HIT",
                "price": float(top_price), "sl": 0.0, "target": 0.0,
                "note": "rotation exit (no longer rank-1 in PIT top-20 filtered)",
            })
        signals.append({
            "model": "n20_daily_30d_mc1_uptrend",
            "symbol": top_sym, "company": top_name,
            "ts": today_str, "side": "BUY", "signal": "ENTRY1",
            "price": float(top_price), "sl": 0.0, "target": 0.0,
            "note": f"30d momentum rank-1 ({top_ret:+.2f}%), PIT top-20 + uptrend",
        })

    Path(args.signals_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.signals_out).write_text(json.dumps(signals, indent=2, default=str))
    log.info(f"Wrote {len(signals)} signals to {args.signals_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
