"""On-demand single-stock backtest endpoint.

Wraps the offline backtest harness (`tools/backtests/run_ema_200_400_backtest.py`)
so the UI can run an EMA 200/400 1H crossover test against any symbol over a
user-chosen window without needing shell access.

Routes:
    GET  /backtest                   — UI page
    POST /api/backtest/run           — JSON: {symbol, source, days?, from?, to?}
                                       returns {ok, summary, signals, cycles, closed}

The harness module's `OfflineStrategy` is reused directly so we don't shell out.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from flask import Blueprint, jsonify, render_template, request

try:
    from flask_login import login_required
except ImportError:  # pragma: no cover — flask_login may be optional
    def login_required(f):  # type: ignore[no-redef]
        return f

# Import harness pieces. Path is set up by the running app.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.backtests.run_ema_200_400_backtest import (  # noqa: E402
    OfflineStrategy,
    StrategyConfig,
    build_cycles,
    df_to_candles,
    fetch_1h_data,
    simulate_pnl,
)

logger = logging.getLogger(__name__)

backtest_bp = Blueprint("backtest", __name__)


def _resolve_window(payload: Dict[str, Any]) -> int:
    """`days` int from days|from|to fields. Capped to Yahoo's 730d."""
    if payload.get("from"):
        from_dt = datetime.strptime(payload["from"], "%Y-%m-%d")
        days = max(1, (datetime.now() - from_dt).days)
        return min(days, 730)
    return min(int(payload.get("days", 720)), 730)


def _normalize_symbol(sym: str) -> str:
    """Accept Fyers (NSE:XYZ-EQ), index (^NSEI), or Yahoo (XYZ.NS)."""
    s = sym.strip().upper()
    if s.startswith("NSE:") and s.endswith("-EQ"):
        return s.replace("NSE:", "").replace("-EQ", "") + ".NS"
    if s.endswith("-INDEX"):
        return "^" + s.replace("NSE:", "").replace("-INDEX", "")
    return s


@backtest_bp.route("/backtest")
@login_required
def backtest_page():
    return render_template("backtest.html")


@backtest_bp.route("/api/backtest/run", methods=["POST"])
@login_required
def run_backtest():
    payload = request.get_json(silent=True) or {}
    symbol = (payload.get("symbol") or "").strip()
    if not symbol:
        return jsonify({"ok": False, "error": "symbol required"}), 400

    source = payload.get("source", "auto")
    if source not in ("auto", "fyers", "yahoo"):
        return jsonify({"ok": False, "error": "invalid source"}), 400

    user_id = int(payload.get("user_id", 1))
    days = _resolve_window(payload)
    yahoo_sym = _normalize_symbol(symbol)

    logger.info(f"Backtest request: symbol={yahoo_sym} source={source} days={days}")

    df, src_used = fetch_1h_data(yahoo_sym, days=days, source=source, user_id=user_id)
    if df.empty:
        return jsonify({
            "ok": False,
            "error": f"No 1H data for {yahoo_sym} from source={source}"
        }), 404

    candles = df_to_candles(df)
    config = StrategyConfig()
    if len(candles) < config.ema_slow_period + 5:
        return jsonify({
            "ok": False,
            "error": f"Insufficient bars: {len(candles)} (need ≥{config.ema_slow_period + 5}). "
                     "Increase days."
        }), 400

    strat = OfflineStrategy(config)
    signals = strat.evaluate(user_id=user_id, symbol=yahoo_sym, candles=candles)
    pnl = simulate_pnl(
        signals, df, yahoo_sym,
        target_points=config.target_points,
        rr_multiple=config.rr_multiple,
    )

    cycles = build_cycles(signals)
    cycles_out: List[Dict[str, Any]] = []
    for idx, cyc in enumerate(cycles, 1):
        cycles_out.append({
            "index": idx,
            "trend": cyc["trend"],
            "started": str(cyc["events"][0]["candle_time"]),
            "events": [
                {
                    "stage": e["signal_type"],
                    "trend": e["trend"],
                    "time": str(e["candle_time"]),
                    "price": round(float(e["price"]), 2),
                    "ema_200": round(float(e["ema_200"]), 2),
                    "ema_400": round(float(e["ema_400"]), 2),
                    "note": e.get("note", ""),
                }
                for e in cyc["events"]
            ],
        })

    closed_out = [
        {
            "trend": t["trend"],
            "entry_time": str(t["time"]),
            "entry_price": round(float(t["price"]), 2),
            "exit_time": str(t["exit_time"]),
            "exit_price": round(float(t["exit_price"]), 2),
            "exit_reason": t.get("exit_reason", ""),
            "pnl": round(float(t["pnl"]), 2),
        }
        for t in pnl["closed"]
    ]

    summary = {
        "symbol": yahoo_sym,
        "source": src_used,
        "bars": len(candles),
        "first_bar": str(df["candle_time"].iloc[0]),
        "last_bar": str(df["candle_time"].iloc[-1]),
        "last_close": round(float(df["close"].iloc[-1]), 2),
        "trades_closed": pnl["trades_closed"],
        "trades_open": pnl["trades_open"],
        "winners": pnl["winners"],
        "losers": pnl["losers"],
        "target_hits": pnl["target_hits"],
        "ema_exits": pnl["ema_exits"],
        "total_pnl": round(float(pnl["total_pnl"]), 2),
        "avg_pnl": round(float(pnl["avg_pnl"]), 2),
        "win_rate": (round(pnl["winners"] / pnl["trades_closed"] * 100, 1)
                     if pnl["trades_closed"] else 0.0),
    }

    signal_counts: Dict[str, int] = {}
    for s in signals:
        signal_counts[s["signal_type"]] = signal_counts.get(s["signal_type"], 0) + 1

    return jsonify({
        "ok": True,
        "summary": summary,
        "signal_counts": signal_counts,
        "cycles": cycles_out,
        "closed": closed_out,
    })
