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
    """Compute portfolio state from LIVE Fyers account (funds + holdings)."""
    try:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        funds_resp = svc.funds(1)
        holdings_resp = svc.holdings(1)
    except Exception as e:
        logger.warning(f"Fyers fetch failed: {e}")
        funds_resp = {"data": {}}
        holdings_resp = {"data": []}

    funds = (funds_resp or {}).get("data") or {}
    cash = float(funds.get("available_cash") or 0)
    total_margin = float(funds.get("total_margin") or 0)
    utilized_margin = float(funds.get("utilized_margin") or 0)

    holdings = (holdings_resp or {}).get("data") or []
    open_positions = []
    position_cost = 0.0
    market_value = 0.0
    unrealized_total = 0.0

    for p in holdings:
        sym = (p.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")
        qty = float(p.get("quantity") or 0)
        if qty <= 0:
            continue
        avg = float(p.get("average_price") or 0)
        ltp = float(p.get("last_price") or 0)
        pnl = float(p.get("pnl") or (ltp - avg) * qty)
        cost = avg * qty
        mv = ltp * qty
        position_cost += cost
        market_value += mv
        unrealized_total += pnl
        open_positions.append({
            "symbol": sym,
            "qty": int(qty),
            "entry_price": avg,
            "live_price": ltp,
            "cost": cost,
            "market_value": mv,
            "unrealized_pnl": pnl,
            "unrealized_pct": (ltp / avg - 1) * 100 if avg > 0 else 0,
        })

    total_value = cash + market_value
    # P&L based on ACTUAL invested capital, not hardcoded baseline
    total_pnl = unrealized_total
    total_pct = (total_pnl / position_cost * 100) if position_cost > 0 else 0.0
    has_data = bool(open_positions) or cash > 0 or total_margin > 0

    history = _read_history()
    return {
        "source": "fyers",
        "has_data": has_data,
        "starting_capital": STARTING_CAPITAL,  # reference ceiling, not P&L baseline
        "cash": cash,
        "total_margin": total_margin,
        "utilized_margin": utilized_margin,
        "position_cost": position_cost,
        "market_value": market_value,
        "total_value": total_value,
        "realized_pnl": 0.0,
        "unrealized_pnl": unrealized_total,
        "total_pnl": total_pnl,
        "total_pct": total_pct,
        "day_pnl": 0.0,
        "open_positions": open_positions,
        "closed_trades_count": len(history),
        "ledger_updated_at": datetime.now().isoformat(),
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


def _current_ranking(top: int = 10, live_prices: bool = True) -> List[Dict]:
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
                "price": price,  # cached close, may be 1 day old
            })
    rows.sort(key=lambda r: -r["return_60d"])
    top_rows = rows[:top]

    # Enrich top-N with LIVE Fyers quotes — overrides cached close
    if live_prices and top_rows:
        try:
            from src.services.brokers.fyers_service import FyersService
            svc = FyersService()
            fyers_syms = [f"NSE:{r['symbol']}-EQ" for r in top_rows]
            q = svc.quotes_multiple(1, fyers_syms) or {}
            quotes = (q.get("data") or {})
            for r in top_rows:
                key = f"NSE:{r['symbol']}-EQ"
                qd = quotes.get(key)
                if qd:
                    ltp = float(qd.get("ltp") or 0)
                    if ltp > 0:
                        # recompute 60d return using live LTP
                        if r["price"] > 0:
                            past_price = r["price"] / (1 + r["return_60d"] / 100)
                            r["return_60d"] = (ltp / past_price - 1) * 100 if past_price > 0 else r["return_60d"]
                        r["price"] = ltp
                        r["live"] = True
        except Exception as e:
            logger.warning(f"live quotes enrich fail: {e}")

    # Re-sort after live update
    top_rows.sort(key=lambda r: -r["return_60d"])
    return top_rows


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
        # Tag held symbols from LIVE Fyers (not paper ledger)
        held_syms = {
            (p.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")
            for p in _fyers_holdings()
        }
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


def _fyers_place_market(symbol: str, qty: int, side: str, user_id: int = 1) -> Dict:
    """Place LIVE Fyers market order. Returns {ok, order_id, error}."""
    from src.services.brokers.fyers_service import FyersService
    fyers_sym = symbol if symbol.startswith("NSE:") else f"NSE:{symbol}-EQ"
    payload = {
        "symbol": fyers_sym,
        "qty": int(qty),
        "type": 2,                 # 2 = Market
        "side": 1 if side == "BUY" else -1,
        "productType": "CNC",      # delivery (equity)
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
    }
    svc = FyersService()
    try:
        res = svc.place_order(user_id=user_id, order=payload)
        return {"ok": res.get("status") in ("success", "ok"), "result": res}
    except Exception as e:
        logger.exception(f"place_order fail {symbol}")
        return {"ok": False, "error": str(e)}


def _fyers_available_cash(user_id: int = 1) -> float:
    """Fetch Fyers available_cash."""
    try:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        f = svc.funds(user_id) or {}
        return float((f.get("data") or {}).get("available_cash") or 0)
    except Exception:
        return 0.0


def _fyers_holdings(user_id: int = 1) -> List[Dict]:
    """Fetch active Fyers holdings (qty > 0)."""
    try:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        h = svc.holdings(user_id) or {}
        return [
            p for p in (h.get("data") or [])
            if float(p.get("quantity") or 0) > 0
        ]
    except Exception:
        return []


def _notify_tg(text: str) -> None:
    try:
        from tools.live.telegram_notify import send
        send(text, "Markdown")
    except Exception:
        logger.exception("tg notify fail")


@momrot_bp.route("/run-now", methods=["POST"])
def api_run_now():
    """Force LIVE rebalance: rank N100, sell drop-outs from Fyers, buy new rank-1.

    Real orders placed via Fyers if LIVE_TRADING=true.
    """
    if os.environ.get("LIVE_TRADING", "false").lower() != "true":
        return jsonify({
            "success": False,
            "error": "LIVE_TRADING not enabled. Set LIVE_TRADING=true env to place real orders.",
        }), 403

    try:
        user_id = int(request.args.get("user_id", 1))
        actions = []

        # Fetch ranks
        ranks = _current_ranking(TOP_N)
        if not ranks:
            return jsonify({"success": False, "error": "No ranking available — rebuild universe first"}), 400
        top_syms = {r["symbol"] for r in ranks}
        rank1 = ranks[0]

        # Fetch current Fyers holdings
        held = _fyers_holdings(user_id)
        already_in_top = any(
            ((p.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")) in top_syms
            for p in held
        )

        # 1. Sell holdings not in top-N
        for p in held:
            sym = (p.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")
            if sym in top_syms:
                continue
            qty = int(float(p.get("quantity") or 0))
            if qty < 1:
                continue
            res = _fyers_place_market(sym, qty, "SELL", user_id)
            actions.append({"action": "SELL", "symbol": sym, "qty": qty, "result": res})

        # 2. Buy rank-1 if not already held
        if not already_in_top:
            cash = _fyers_available_cash(user_id)
            price = rank1["price"]
            qty = int(cash // price) if price > 0 else 0
            if qty < 1:
                actions.append({"action": "BUY_SKIP", "symbol": rank1["symbol"],
                                "reason": f"insufficient cash ₹{cash:,.0f} for price ₹{price:.2f}"})
            else:
                res = _fyers_place_market(rank1["symbol"], qty, "BUY", user_id)
                actions.append({"action": "BUY", "symbol": rank1["symbol"],
                                "qty": qty, "price": price, "deploy": qty * price, "result": res})

        # Telegram alert
        if actions:
            lines = ["🔴 *LIVE Rebalance*"]
            for a in actions:
                lines.append(f"- {a['action']} {a.get('symbol','')} qty={a.get('qty','')}")
            _notify_tg("\n".join(lines))
        else:
            _notify_tg("ℹ️ Rebalance: no action needed (rank-1 already held)")

        return jsonify({"success": True, "actions": actions, "rank1": rank1})
    except Exception as e:
        logger.exception("run-now fail")
        _notify_tg(f"❌ Rebalance error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@momrot_bp.route("/buy-now", methods=["POST"])
def api_buy_now():
    """Manual buy: place market order for rank-1 (or specified symbol) using available cash."""
    if os.environ.get("LIVE_TRADING", "false").lower() != "true":
        return jsonify({"success": False, "error": "LIVE_TRADING not enabled"}), 403

    try:
        user_id = int(request.args.get("user_id", 1))
        body = request.get_json(silent=True) or {}
        symbol = body.get("symbol")
        if not symbol:
            ranks = _current_ranking(1)
            if not ranks:
                return jsonify({"success": False, "error": "No ranking"}), 400
            symbol = ranks[0]["symbol"]
            price = ranks[0]["price"]
        else:
            price = _live_price(symbol)

        cash = _fyers_available_cash(user_id)
        qty = body.get("qty")
        if qty:
            qty = int(qty)
        else:
            qty = int(cash // price) if price > 0 else 0

        if qty < 1:
            return jsonify({"success": False, "error": f"qty<1 (cash ₹{cash:,.0f}, price ₹{price:.2f})"}), 400

        res = _fyers_place_market(symbol, qty, "BUY", user_id)
        _notify_tg(f"🟢 *BUY* {symbol} qty={qty} @ ₹{price:.2f} (₹{qty*price:,.0f})")
        return jsonify({"success": res["ok"], "symbol": symbol, "qty": qty,
                        "price": price, "result": res})
    except Exception as e:
        logger.exception("buy-now fail")
        _notify_tg(f"❌ Buy error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@momrot_bp.route("/rebalance-preview")
def api_rebalance_preview():
    """Show what /run-now would do WITHOUT executing. Read-only."""
    try:
        user_id = int(request.args.get("user_id", 1))
        ranks = _current_ranking(TOP_N)
        if not ranks:
            return jsonify({"success": False, "error": "No ranking — rebuild universe first"}), 400
        top_syms = {r["symbol"] for r in ranks}
        rank1 = ranks[0]
        held = _fyers_holdings(user_id)
        already = any(((p.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")) in top_syms for p in held)
        cash = _fyers_available_cash(user_id)

        sells = []
        for p in held:
            sym = (p.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")
            qty = int(float(p.get("quantity") or 0))
            if sym in top_syms or qty < 1:
                continue
            ltp = float(p.get("last_price") or 0)
            sells.append({"symbol": sym, "qty": qty, "ltp": ltp, "proceeds": qty * ltp})

        buy = None
        proj_cash = cash + sum(s["proceeds"] for s in sells)
        if not already:
            qty_buy = int(proj_cash // rank1["price"]) if rank1["price"] > 0 else 0
            buy = {
                "symbol": rank1["symbol"],
                "price": rank1["price"],
                "return_60d": rank1["return_60d"],
                "qty": qty_buy,
                "deploy": qty_buy * rank1["price"],
                "available_cash_after_sells": proj_cash,
            }

        return jsonify({
            "success": True,
            "top_n": TOP_N,
            "max_concurrent": 1,
            "rank1": rank1,
            "current_cash": cash,
            "sells": sells,
            "buy": buy,
            "already_holding_rank_member": already,
        })
    except Exception as e:
        logger.exception("rebalance-preview fail")
        return jsonify({"success": False, "error": str(e)}), 500


@momrot_bp.route("/sell-now", methods=["POST"])
def api_sell_now():
    """Manual sell: place SELL market order for symbol + qty."""
    if os.environ.get("LIVE_TRADING", "false").lower() != "true":
        return jsonify({"success": False, "error": "LIVE_TRADING not enabled"}), 403

    try:
        user_id = int(request.args.get("user_id", 1))
        body = request.get_json(silent=True) or {}
        symbol = body.get("symbol")
        qty = body.get("qty")
        if not symbol or not qty:
            return jsonify({"success": False, "error": "symbol + qty required"}), 400
        qty = int(qty)
        res = _fyers_place_market(symbol, qty, "SELL", user_id)
        _notify_tg(f"🔴 *SELL* {symbol} qty={qty}")
        return jsonify({"success": res["ok"], "symbol": symbol, "qty": qty, "result": res})
    except Exception as e:
        logger.exception("sell-now fail")
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
