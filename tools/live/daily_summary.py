"""Daily Telegram summary for momentum rotation.

Sends a single short message with:
  - Portfolio NAV + total P&L %
  - Day P&L (vs yesterday's close)
  - Current holding + entry-anchored % return
  - Top-1 momentum pick + 60d return
  - Rebalance status (HOLD / REBALANCED / FAILED / NO_DATA)

State is read from /app/logs/momrot/ledger/momrot_ledger.json and
universe from /app/logs/momrot/universes/n100_current.json.

Usage:
  python tools/live/daily_summary.py [--status REBALANCED|HOLD|FAILED] \
                                      [--detail "extra detail line"]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backtests.ohlcv_cache import read_cached  # noqa: E402
from tools.live.telegram_notify import send  # noqa: E402

LEDGER = Path("/app/logs/momrot/ledger/momrot_ledger.json")
UNIVERSE = Path("/app/logs/momrot/universes/n100_current.json")
HISTORY = Path("/app/logs/momrot/ledger/trade_history.jsonl")
STATE_FILE = Path("/app/logs/momrot/ledger/daily_nav.json")
STARTING_CAPITAL = 1_000_000.0


def _load(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _close(symbol: str, target_ts: int, days_back: int = 90) -> float:
    df = read_cached(symbol, "D", target_ts - days_back * 86400, target_ts)
    if df.empty:
        return 0.0
    return float(df.iloc[-1]["close"])


def _ret_60d(symbol: str, ts: int) -> float:
    cur = _close(symbol, ts)
    past = _close(symbol, ts - 60 * 86400)
    if cur > 0 and past > 0:
        return (cur / past - 1) * 100
    return 0.0


def _portfolio_nav(ledger: Dict, history: List[Dict], today_ts: int) -> Dict:
    realized = sum(h.get("pnl", 0.0) for h in history)
    pos_cost = 0.0
    mv = 0.0
    held_str = "None"
    held_pct = 0.0
    for p in ledger.get("open", []):
        qty = int(p["qty"])
        entry = float(p["entry_price"])
        live = _close(p["symbol"], today_ts)
        pos_cost += entry * qty
        mv += live * qty
        if held_str == "None":
            held_str = f"{p['symbol']} ({(live/entry-1)*100:+.1f}%)"
            held_pct = (live / entry - 1) * 100
    cash = STARTING_CAPITAL + realized - pos_cost
    total = cash + mv
    total_pct = (total / STARTING_CAPITAL - 1) * 100
    return {
        "nav": total, "total_pct": total_pct,
        "cash": cash, "market_value": mv,
        "held_str": held_str, "held_pct": held_pct,
        "realized": realized,
    }


def _day_pnl(today_nav: float) -> float:
    """Day P&L = today NAV - yesterday NAV (from STATE_FILE)."""
    s = _load(STATE_FILE, {})
    yesterday_nav = s.get("nav", today_nav)
    return today_nav - yesterday_nav


def _save_nav(nav: float):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"nav": nav, "ts": datetime.now().isoformat()}, f)


def _top1(today_ts: int) -> Optional[Dict]:
    uni = _load(UNIVERSE, {}).get("stocks", [])
    if not uni:
        return None
    rows = []
    for s in uni:
        sym = s["symbol"]
        r = _ret_60d(sym, today_ts)
        if r != 0:
            rows.append({"symbol": sym, "name": s.get("name", sym),
                         "return_60d": r, "price": _close(sym, today_ts)})
    if not rows:
        return None
    rows.sort(key=lambda r: -r["return_60d"])
    return rows[0]


def _next_rebalance(today: datetime) -> str:
    if today.day <= 7 and today.weekday() < 5:
        return today.date().isoformat()
    if today.month == 12:
        nxt = datetime(today.year + 1, 1, 1)
    else:
        nxt = datetime(today.year, today.month + 1, 1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.date().isoformat()


def build_message(status: str = "HOLD", detail: str = "") -> str:
    today = datetime.now()
    today_ts = int(today.timestamp())
    ledger = _load(LEDGER, {"open": []})
    history = []
    if HISTORY.exists():
        with open(HISTORY) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        history.append(json.loads(line))
                    except Exception:
                        pass

    nav = _portfolio_nav(ledger, history, today_ts)
    day_pnl = _day_pnl(nav["nav"])
    _save_nav(nav["nav"])
    top1 = _top1(today_ts)
    next_reb = _next_rebalance(today)

    status_emoji = {
        "REBALANCED": "🔄",
        "HOLD": "⏸️",
        "FAILED": "❌",
        "NO_DATA": "⚠️",
    }.get(status, "ℹ️")

    lines = [
        f"*Momrot {today.strftime('%Y-%m-%d')}* {status_emoji} {status}",
        f"NAV ₹{nav['nav']:,.0f}  ({nav['total_pct']:+.2f}%)",
        f"Day P&L ₹{day_pnl:+,.0f}",
        f"Hold {nav['held_str']}",
    ]
    if top1:
        held_sym = (ledger.get("open") or [{}])[0].get("symbol", "")
        marker = "✓" if top1["symbol"] == held_sym else "→"
        lines.append(f"Top1 {marker} {top1['symbol']} ({top1['return_60d']:+.1f}%)")
    lines.append(f"Next rebalance: {next_reb}")
    if detail:
        lines.append(f"_{detail}_")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="HOLD",
                    choices=["REBALANCED", "HOLD", "FAILED", "NO_DATA"])
    ap.add_argument("--detail", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print message but don't send")
    args = ap.parse_args()

    msg = build_message(args.status, args.detail)
    if args.dry_run:
        print(msg)
        return 0
    res = send(msg, "Markdown")
    if not res.get("ok"):
        print(f"FAIL: {res}", file=sys.stderr)
        return 2
    print(f"sent: id={res['result']['message_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
