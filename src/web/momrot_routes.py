"""Momentum Rotation admin panel routes.

Backend for the Momentum Rotation paper-trading dashboard.

Endpoints:
  GET  /admin/momrot                 → render UI page
  GET  /admin/momrot/state           → current portfolio: held, cash, P&L
  GET  /admin/momrot/ranking?top=10  → top-N current momentum ranking
  GET  /admin/momrot/universe        → current N100 universe list
  GET  /admin/momrot/next-rebalance  → next rotation trigger date
  GET  /admin/momrot/history         → all closed trades
  GET  /admin/momrot/run-logs        → recent daily-run logs
  POST /admin/momrot/run-now         → force a rebalance now
  POST /admin/momrot/rebuild-universe → rebuild N100 universe

Persistent state lives at /app/logs/momrot/ (host bind mount).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, render_template, request
from sqlalchemy import text

logger = logging.getLogger(__name__)

momrot_bp = Blueprint("momrot", __name__, url_prefix="/admin/momrot")

# Host paths (inside container they map to /app/logs/momrot/)
ROOT = Path("/app/logs/momrot")
LEDGER_PATH = ROOT / "ledger" / "momrot_ledger.json"
HISTORY_PATH = ROOT / "ledger" / "trade_history.jsonl"
UNIVERSE_PATH = ROOT / "universes" / "n100_current.json"
SIGNALS_DIR = ROOT / "signals"
RUN_LOGS_DIR = ROOT / "run_logs"

STARTING_CAPITAL = 1_000_000  # ₹10L locked
TOP_N = 5                     # top-N rank from N100 momentum
LOOKBACK_DAYS = 60


# ---- Data helpers ---------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"load_json({path}) fail: {e}")
        return default


def _read_history() -> List[Dict]:
    """Read all closed trades from JSONL history."""
    if not HISTORY_PATH.exists():
        return []
    out = []
    with open(HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def _live_price(symbol: str) -> float:
    """Latest close from historical_data table."""
    from src.models.database import get_database_manager

    db = get_database_manager()
    with db.get_session() as session:
        q = text("""
            SELECT close FROM historical_data
            WHERE symbol = :sym OR symbol = :fyers_sym
            ORDER BY date DESC LIMIT 1
        """)
        fyers_sym = f"NSE:{symbol}-EQ"
        row = session.execute(q, {"sym": symbol, "fyers_sym": fyers_sym}).fetchone()
        return float(row[0]) if row else 0.0


def _return_60d(symbol: str) -> tuple:
    """(60d_return_pct, current_close) for symbol."""
    from src.models.database import get_database_manager

    db = get_database_manager()
    with db.get_session() as session:
        q = text("""
            SELECT close, date FROM historical_data
            WHERE symbol = :sym OR symbol = :fyers_sym
            ORDER BY date DESC LIMIT 90
        """)
        fyers_sym = f"NSE:{symbol}-EQ"
        rows = session.execute(q, {"sym": symbol, "fyers_sym": fyers_sym}).fetchall()
    if not rows or len(rows) < 50:
        return 0.0, 0.0
    cur = float(rows[0][0])
    # 60d ago = row index ~60 (descending order)
    idx = min(60, len(rows) - 1)
    past = float(rows[idx][0])
    if past == 0:
        return 0.0, cur
    ret = (cur / past - 1) * 100
    return ret, cur


def _portfolio_state() -> Dict:
    """Compute full portfolio state from ledger + history + live prices."""
    ledger = _load_json(LEDGER_PATH, {"open": [], "closed_today": []})
    history = _read_history()

    realized_pnl = sum(h.get("pnl", 0.0) for h in history)
    open_positions = []
    position_cost = 0.0
    market_value = 0.0
    day_pnl = 0.0

    for p in ledger.get("open", []):
        sym = p["symbol"]
        qty = int(p["qty"])
        entry = float(p["entry_price"])
        live = _live_price(sym)
        cost = entry * qty
        mv = live * qty
        unreal = mv - cost
        day_open = live  # without intraday data, approximate with live
        position_cost += cost
        market_value += mv
        open_positions.append({
            "symbol": sym,
            "qty": qty,
            "entry_price": entry,
            "live_price": live,
            "cost": cost,
            "market_value": mv,
            "unrealized_pnl": unreal,
            "unrealized_pct": (live / entry - 1) * 100 if entry > 0 else 0,
        })

    cash = STARTING_CAPITAL + realized_pnl - position_cost
    total_value = cash + market_value
    total_pnl = total_value - STARTING_CAPITAL
    total_pct = (total_value / STARTING_CAPITAL - 1) * 100

    return {
        "starting_capital": STARTING_CAPITAL,
        "cash": cash,
        "position_cost": position_cost,
        "market_value": market_value,
        "total_value": total_value,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": market_value - position_cost,
        "total_pnl": total_pnl,
        "total_pct": total_pct,
        "day_pnl": ledger.get("day_pnl", 0.0),
        "open_positions": open_positions,
        "closed_trades_count": len(history),
        "ledger_updated_at": ledger.get("updated_at"),
    }


def _next_rebalance() -> Dict:
    """Compute next rotation date (first weekday on/after 1st of next month)."""
    today = datetime.now()
    # First trading day on/after the 1st of current month — if today already past day 7, target next month
    if today.day <= 7 and today.weekday() < 5:
        nxt = today.date()
        days_to = 0
    else:
        # Next month's 1st
        if today.month == 12:
            target = datetime(today.year + 1, 1, 1)
        else:
            target = datetime(today.year, today.month + 1, 1)
        # Skip to weekday
        while target.weekday() >= 5:
            target += timedelta(days=1)
        nxt = target.date()
        days_to = (target.date() - today.date()).days
    return {"date": str(nxt), "days_until": days_to, "weekday": nxt.strftime("%A")}


def _current_ranking(top: int = 10) -> List[Dict]:
    universe = _load_json(UNIVERSE_PATH, {"stocks": []}).get("stocks", [])
    rows = []
    for s in universe:
        sym = s["symbol"]
        ret, price = _return_60d(sym)
        if price > 0:
            rows.append({
                "symbol": sym,
                "name": s.get("name", sym),
                "return_60d": ret,
                "price": price,
            })
    rows.sort(key=lambda r: -r["return_60d"])
    return rows[:top]


# ---- Routes ---------------------------------------------------------------

@momrot_bp.route("")
@momrot_bp.route("/")
def momrot_dashboard():
    return render_template("admin/momrot_dashboard.html")


@momrot_bp.route("/state")
def api_state():
    try:
        return jsonify({"success": True, "data": _portfolio_state()})
    except Exception as e:
        logger.exception("momrot state fail")
        return jsonify({"success": False, "error": str(e)}), 500


@momrot_bp.route("/ranking")
def api_ranking():
    try:
        top = int(request.args.get("top", 10))
        ranking = _current_ranking(top)
        # Also tag held symbols
        ledger = _load_json(LEDGER_PATH, {"open": []})
        held_syms = {p["symbol"] for p in ledger.get("open", [])}
        for r in ranking:
            r["held"] = r["symbol"] in held_syms
        return jsonify({"success": True, "ranking": ranking})
    except Exception as e:
        logger.exception("momrot ranking fail")
        return jsonify({"success": False, "error": str(e)}), 500


@momrot_bp.route("/universe")
def api_universe():
    data = _load_json(UNIVERSE_PATH, {"stocks": []})
    return jsonify({"success": True, "data": data})


@momrot_bp.route("/next-rebalance")
def api_next_rebalance():
    return jsonify({"success": True, "data": _next_rebalance()})


@momrot_bp.route("/history")
def api_history():
    return jsonify({"success": True, "trades": _read_history()})


@momrot_bp.route("/run-logs")
def api_run_logs():
    try:
        limit = int(request.args.get("limit", 10))
        logs = sorted(RUN_LOGS_DIR.glob("*.log"), reverse=True)[:limit]
        out = []
        for f in logs:
            out.append({
                "name": f.name,
                "size": f.stat().st_size,
                "mtime": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "tail": f.read_text(errors="replace").splitlines()[-40:],
            })
        return jsonify({"success": True, "logs": out})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@momrot_bp.route("/run-now", methods=["POST"])
def api_run_now():
    """Force a rebalance check NOW. Bypasses first-week-of-month gate."""
    try:
        date_str = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        sig_path = SIGNALS_DIR / f"manual_{date_str}_momrot.json"

        def run():
            try:
                env = os.environ.copy()
                env.update({
                    "CAPITAL_INR": str(STARTING_CAPITAL),
                    "MAX_CONCURRENT": "1",
                    "MIN_PRICE": "10",
                    "MAX_PER_TRADE_INR": str(STARTING_CAPITAL),
                })
                # Emit signals (forced)
                subprocess.run([
                    "python", "tools/live/momentum_rotation_signal.py",
                    "--universe-file", str(UNIVERSE_PATH),
                    "--top-n", str(TOP_N), "--force",
                    "--ledger", str(LEDGER_PATH),
                    "--signals-out", str(sig_path),
                ], check=True, cwd="/app", env=env)

                # Execute via paper_executor (only if non-empty)
                if sig_path.exists() and sig_path.read_text().strip() not in ("[]", ""):
                    subprocess.run([
                        "python", "tools/live/paper_executor.py",
                        "--signals", str(sig_path),
                        "--ledger", str(LEDGER_PATH),
                    ], check=True, cwd="/app", env=env)

                    # Append closed trades to history
                    ledger = _load_json(LEDGER_PATH, {})
                    for c in ledger.get("closed_today", []):
                        with open(HISTORY_PATH, "a") as fh:
                            fh.write(json.dumps(c) + "\n")
            except Exception:
                logger.exception("run-now thread fail")

        threading.Thread(target=run, daemon=True).start()
        return jsonify({"success": True, "message": "Rebalance triggered",
                        "signals_path": str(sig_path)})
    except Exception as e:
        logger.exception("run-now fail")
        return jsonify({"success": False, "error": str(e)}), 500


@momrot_bp.route("/rebuild-universe", methods=["POST"])
def api_rebuild_universe():
    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_path = UNIVERSE_PATH.parent / f"n100_{date_str}.json"

        def run():
            try:
                subprocess.run([
                    "python", "tools/backtests/build_universe_by_adv.py",
                    "--top", "100", "--end-date", date_str,
                    "--out", str(out_path),
                ], check=True, cwd="/app")
                # Promote to current
                if out_path.exists():
                    UNIVERSE_PATH.write_text(out_path.read_text())
            except Exception:
                logger.exception("rebuild-universe thread fail")

        threading.Thread(target=run, daemon=True).start()
        return jsonify({"success": True, "message": "Universe rebuild triggered",
                        "output": str(out_path)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@momrot_bp.route("/summary")
def api_summary():
    """One-shot dashboard payload — state + ranking + next rebalance."""
    try:
        return jsonify({
            "success": True,
            "state": _portfolio_state(),
            "ranking": _current_ranking(10),
            "next_rebalance": _next_rebalance(),
            "universe_size": len(_load_json(UNIVERSE_PATH, {"stocks": []}).get("stocks", [])),
        })
    except Exception as e:
        logger.exception("summary fail")
        return jsonify({"success": False, "error": str(e)}), 500


# ---- Fyers live account ---------------------------------------------------

@momrot_bp.route("/fyers/account")
def api_fyers_account():
    """Pull live data from real Fyers account: funds + holdings + positions."""
    try:
        from src.services.brokers.fyers_service import FyersService
        user_id = int(request.args.get("user_id", 1))
        svc = FyersService()

        out = {"user_id": user_id}

        # Funds
        try:
            funds = svc.funds(user_id)
            out["funds"] = funds
        except Exception as e:
            out["funds_error"] = str(e)

        # Holdings
        try:
            holdings = svc.holdings(user_id)
            out["holdings"] = holdings
        except Exception as e:
            out["holdings_error"] = str(e)

        # Positions
        try:
            positions = svc.positions(user_id)
            out["positions"] = positions
        except Exception as e:
            out["positions_error"] = str(e)

        return jsonify({"success": True, "data": out})
    except Exception as e:
        logger.exception("fyers account fail")
        return jsonify({"success": False, "error": str(e)}), 500


@momrot_bp.route("/fyers/orderbook")
def api_fyers_orderbook():
    """Live order history from Fyers."""
    try:
        from src.services.brokers.fyers_service import FyersService
        user_id = int(request.args.get("user_id", 1))
        svc = FyersService()
        ob = svc.orderbook(user_id)
        tb = svc.tradebook(user_id)
        return jsonify({"success": True, "orderbook": ob, "tradebook": tb})
    except Exception as e:
        logger.exception("fyers orderbook fail")
        return jsonify({"success": False, "error": str(e)}), 500


# ---- Telegram notifications ----------------------------------------------

@momrot_bp.route("/telegram/test", methods=["POST"])
def api_telegram_test():
    """Send test message to Telegram chat."""
    try:
        from tools.live.telegram_notify import send
        text = request.json.get("text") if request.is_json else None
        if not text:
            text = "🤖 momrot test from UI: bot wired correctly."
        res = send(text, "Markdown")
        return jsonify({"success": bool(res.get("ok")), "result": res})
    except Exception as e:
        logger.exception("telegram test fail")
        return jsonify({"success": False, "error": str(e)}), 500


@momrot_bp.route("/telegram/status")
def api_telegram_status():
    """Check if Telegram credentials are configured."""
    return jsonify({
        "success": True,
        "configured": bool(os.environ.get("TG_BOT_TOKEN") and os.environ.get("TG_CHAT_ID")),
        "chat_id": os.environ.get("TG_CHAT_ID", ""),
    })
