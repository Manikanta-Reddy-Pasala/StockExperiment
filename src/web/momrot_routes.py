"""Momentum Rotation admin panel routes.

Backend for the Momentum Rotation live-trading dashboard.

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

from flask import Blueprint, jsonify, make_response, render_template, request
from sqlalchemy import text

logger = logging.getLogger(__name__)

momrot_bp = Blueprint("momrot", __name__, url_prefix="/admin/momrot")

# Shared Dragonfly UI cache (fail-open). The Fyers-backed read endpoints
# (/state, /ranking, /fyers/*) make live broker API calls per request — caching
# them a few seconds slashes latency + Fyers rate-limit pressure. A successful
# POST (buy/sell/run/rebuild) invalidates ui:* so account state refreshes at once.
from src.web.ui_cache import ui_cached, invalidate_on_mutation  # noqa: E402
momrot_bp.after_request(invalidate_on_mutation)

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
    """Compute portfolio state from LIVE Fyers account (funds + holdings + CNC positions).

    HFCL etc bought today via algo end up in POSITIONS (CNC, netQty>0) until
    T+1 settlement, then move to HOLDINGS. We merge both.
    """
    try:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        funds_resp = svc.funds(1)
        holdings_resp = svc.holdings(1)
        # Raw positions has more fields (buyAvg, netQty, pl) than standardized
        positions_raw = svc._get_api_instance(1)._make_request("GET", "positions") or {}
    except Exception as e:
        logger.warning(f"Fyers fetch failed: {e}")
        funds_resp = {"data": {}}
        holdings_resp = {"data": []}
        positions_raw = {"data": []}

    funds = (funds_resp or {}).get("data") or {}
    cash = float(funds.get("available_cash") or 0)
    total_margin = float(funds.get("total_margin") or 0)
    utilized_margin = float(funds.get("utilized_margin") or 0)

    holdings = (holdings_resp or {}).get("data") or []
    positions = (positions_raw or {}).get("data") or []

    # Build symbol set + batch-fetch fresh quotes. Fyers holdings.last_price is
    # often stale (yesterday's close or pre-market snapshot). We override with
    # real-time quotes; DB latest close is the final fallback.
    fresh_ltp: Dict[str, float] = {}
    fresh_pc: Dict[str, float] = {}   # prev_close → Today's P&L (day MTM)
    fyers_syms_for_quote = []
    for p in holdings:
        s = (p.get("symbol") or "").strip()
        if s:
            fyers_syms_for_quote.append(s if s.startswith("NSE:") else f"NSE:{s.replace('-EQ','')}-EQ")
    for p in positions:
        s = (p.get("symbol") or "").strip()
        if not s or p.get("productType") != "CNC":
            continue
        fyers_syms_for_quote.append(s if s.startswith("NSE:") else f"NSE:{s.replace('-EQ','')}-EQ")
    fyers_syms_for_quote = list({s for s in fyers_syms_for_quote if s})
    if fyers_syms_for_quote:
        try:
            from src.services.brokers.fyers_service import FyersService
            qsvc = FyersService()
            qr = qsvc.quotes_multiple(1, fyers_syms_for_quote) or {}
            qdata = (qr.get("data") or {})
            for fs, qd in qdata.items():
                bare = fs.replace("NSE:", "").replace("-EQ", "")
                v = float((qd or {}).get("ltp") or 0)
                if v > 0:
                    fresh_ltp[bare] = v
                pcv = float((qd or {}).get("prev_close") or 0)
                if pcv > 0:
                    fresh_pc[bare] = pcv
        except Exception as e:
            logger.warning(f"quotes_multiple fail in _portfolio_state: {e}")

    def _resolve_ltp(bare_sym: str, fyers_ltp: float) -> float:
        v = fresh_ltp.get(bare_sym)
        if v and v > 0:
            return v
        v = _live_price(bare_sym)  # DB latest close
        if v and v > 0:
            return v
        return fyers_ltp  # last-resort: stale Fyers field

    open_positions = []
    seen_syms = set()
    position_cost = 0.0
    market_value = 0.0
    unrealized_total = 0.0
    today_pnl_total = 0.0   # Σ qty × (LTP − prev_close) over all broker lots

    # 1. Settled holdings (T+2+)
    for p in holdings:
        sym = (p.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")
        qty = float(p.get("quantity") or 0)
        if qty <= 0:
            continue
        avg = float(p.get("average_price") or 0)
        stale_ltp = float(p.get("last_price") or 0)
        ltp = _resolve_ltp(sym, stale_ltp)
        pnl = (ltp - avg) * qty
        cost = avg * qty
        mv = ltp * qty
        position_cost += cost
        market_value += mv
        unrealized_total += pnl
        _pc = fresh_pc.get(sym, 0.0)
        if _pc > 0:
            today_pnl_total += (ltp - _pc) * qty
        seen_syms.add(sym)
        open_positions.append({
            "symbol": sym,
            "qty": int(qty),
            "entry_price": avg,
            "live_price": ltp,
            "cost": cost,
            "market_value": mv,
            "unrealized_pnl": pnl,
            "unrealized_pct": (ltp / avg - 1) * 100 if avg > 0 else 0,
            "source": "holding",
        })

    # 2. CNC positions (intraday-bought, pending T+1 settlement)
    for p in positions:
        sym_full = p.get("symbol") or ""
        sym = sym_full.replace("NSE:", "").replace("BSE:", "").replace("-EQ", "").replace("-B", "")
        if p.get("productType") != "CNC":
            continue
        net_qty = float(p.get("netQty") or 0)
        if net_qty <= 0:
            continue
        if sym in seen_syms:
            continue  # already counted as holding
        avg = float(p.get("buyAvg") or p.get("netAvg") or 0)
        stale_ltp = float(p.get("ltp") or 0)
        ltp = _resolve_ltp(sym, stale_ltp)
        pnl = (ltp - avg) * net_qty
        cost = avg * net_qty
        mv = ltp * net_qty
        position_cost += cost
        market_value += mv
        unrealized_total += pnl
        _pc = fresh_pc.get(sym, 0.0)
        if _pc > 0:
            today_pnl_total += (ltp - _pc) * net_qty
        seen_syms.add(sym)
        open_positions.append({
            "symbol": sym,
            "qty": int(net_qty),
            "entry_price": avg,
            "live_price": ltp,
            "cost": cost,
            "market_value": mv,
            "unrealized_pnl": pnl,
            "unrealized_pct": (ltp / avg - 1) * 100 if avg > 0 else 0,
            "source": "position",   # marker — T+1 pending
        })

    # Account TOTAL value = cash + GROSS market value of every broker lot.
    # holdings() (settled) and CNC positions() (today, pre-settlement) are
    # DISJOINT lots — a symbol in BOTH (e.g. HFCL: 203 settled + 502 today) is
    # 705 real shares, so the value sum must NOT dedup by symbol (the
    # open_positions list above dedups for display; that under-counted the
    # account total, e.g. showing 212k for a real ~308k account).
    holdings_value = 0.0
    for p in holdings:
        q = float(p.get("quantity") or 0)
        if q <= 0:
            continue
        bsym = (p.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")
        holdings_value += _resolve_ltp(bsym, float(p.get("last_price") or 0)) * q
    for p in positions:
        if p.get("productType") != "CNC":
            continue
        nq = float(p.get("netQty") or 0)
        if nq <= 0:
            continue
        bsym = (p.get("symbol") or "").replace("NSE:", "").replace("BSE:", "").replace("-EQ", "").replace("-B", "")
        holdings_value += _resolve_ltp(bsym, float(p.get("ltp") or 0)) * nq
    account_total = cash + holdings_value

    total_value = cash + market_value
    # P&L based on ACTUAL invested capital, not hardcoded baseline
    total_pnl = unrealized_total
    total_pct = (total_pnl / position_cost * 100) if position_cost > 0 else 0.0
    has_data = bool(open_positions) or cash > 0 or total_margin > 0

    # Txn charges — computed from the FYERS API + published rates (NO CSV):
    # acquire-cost of current holdings + today's per-order fills. Fyers has no
    # charges API and no historical trades, so this is the automatic estimate.
    _acct_chg = {"total": 0.0, "holdings": 0.0, "today": 0.0}
    try:
        from src.web.admin_routes import _fyers_account_txn_charges
        _acct_chg = _fyers_account_txn_charges(1)
    except Exception as _e:
        logger.debug(f"acct charges (fyers api) failed: {_e}")

    history = _read_history()
    return {
        "source": "fyers",
        "has_data": has_data,
        "starting_capital": STARTING_CAPITAL,  # reference ceiling, not P&L baseline
        # Txn charges from the Fyers API (holdings acquire-cost + today's fills).
        "account_txn_charges": _acct_chg.get("total", 0.0),
        "account_txn_charges_holdings": _acct_chg.get("holdings", 0.0),
        "account_txn_charges_today": _acct_chg.get("today", 0.0),
        "cash": cash,
        "total_margin": total_margin,
        "utilized_margin": utilized_margin,
        "position_cost": position_cost,
        "market_value": market_value,
        "total_value": total_value,
        # Plain-language funds for the portfolio card (gross, no symbol-dedup):
        "holdings_value": holdings_value,        # live ₹ in all stock lots
        "account_total": account_total,          # cash + holdings_value (true total)
        # WHOLE-ACCOUNT P&L straight from Fyers (matches the broker app's
        # Holdings tab): total = live unrealized on all lots; today = day MTM
        # (Σ qty × (LTP − prev_close)). Broker truth — the per-model figures
        # (a subset) won't sum to this when the account holds untracked lots.
        "account_total_pnl": round(unrealized_total, 2),
        "account_today_pnl": round(today_pnl_total, 2),
        "account_invested": round(position_cost, 2),
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
    """Next monthly rotation = first NSE TRADING day of the next month (or this
    month's, if it hasn't occurred yet), HOLIDAY-AWARE. `days_until` counts
    trading SESSIONS (not calendar days), mirroring the picks.html per-model
    countdown and the shared rebalance_calendar rule so the two never disagree."""
    from tools.shared.nse_calendar import first_trading_day_of_month, is_trading_day
    today = datetime.now().date()
    this_first = first_trading_day_of_month(today)
    if today < this_first:
        target = this_first
    else:
        nxt_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
        target = first_trading_day_of_month(nxt_month)
    sessions, cur = 0, today
    while cur < target:                       # trading sessions strictly after today
        cur += timedelta(days=1)
        if is_trading_day(cur):
            sessions += 1
    return {"date": str(target), "days_until": sessions, "weekday": target.strftime("%A")}


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
    # Disable browser cache so JS/template changes deploy immediately
    # without users needing to hard-refresh.
    resp = make_response(render_template("admin/momrot_dashboard.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@momrot_bp.route("/state")
@ui_cached('momrot_state', ttl=15)
def api_state():
    try:
        return jsonify({"success": True, "data": _portfolio_state()})
    except Exception as e:
        logger.exception("momrot state fail")
        return jsonify({"success": False, "error": str(e)}), 500


@momrot_bp.route("/ranking")
@ui_cached('momrot_ranking', ttl=20)
def api_ranking():
    try:
        top = int(request.args.get("top", 10))
        ranking = _current_ranking(top)
        # Tag held symbols from LIVE Fyers
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
@ui_cached('momrot_history', ttl=30)
def api_history():
    """Trade history: Fyers tradebook (real trades) + local override file."""
    trades = []
    try:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        raw = svc._get_api_instance(1)._make_request("GET", "tradebook") or {}
        for t in (raw.get("data") or []):
            side = t.get("side")
            trades.append({
                "ts": t.get("orderDateTime", ""),
                "symbol": (t.get("symbol") or "").replace("NSE:", "").replace("BSE:", "").replace("-EQ", "").replace("-B", ""),
                "side": "BUY" if side == 1 else "SELL",
                "qty_closed": int(t.get("tradedQty") or 0),
                "entry_price": 0.0,                       # tradebook = single fill, no entry/exit pair
                "exit_price": float(t.get("tradePrice") or 0),
                "value": float(t.get("tradeValue") or 0),
                "order_id": t.get("orderNumber", ""),
                "trade_id": t.get("tradeNumber", ""),
                "reason": "FYERS_FILL",
                "pnl": 0.0,                               # tradebook doesn't compute realized; needs entry/exit pairing
                "tag": t.get("orderTag", ""),
            })
    except Exception as e:
        logger.warning(f"tradebook fetch fail: {e}")
    # Merge local history file (annotated trades)
    trades.extend(_read_history())
    # Sort newest first
    trades.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return jsonify({"success": True, "trades": trades})


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


def _fyers_place_market(symbol: str, qty: int, side: str, user_id: int = 1,
                        model_name: str = "momentum_n100_top5_max1") -> Dict:
    """Place LIVE Fyers market order via standardized placeorder.

    Also writes an audit_orders row (with computed broker charges) so every
    UI-triggered trade is captured in the charges-summary aggregates.
    """
    from src.services.brokers.fyers_service import FyersService
    fyers_sym = symbol if symbol.startswith("NSE:") else f"NSE:{symbol}-EQ"
    svc = FyersService()
    req = {
        "symbol": fyers_sym, "qty": int(qty), "side": side.upper(),
        "product": "CNC", "pricetype": "MARKET",
        "price": 0.0, "tag": "momrot",
    }
    res: Dict = {}
    err_msg: Optional[str] = None
    try:
        res = svc.placeorder(
            user_id=user_id,
            symbol=fyers_sym,
            quantity=str(int(qty)),
            action=side,                # BUY or SELL
            product="CNC",              # delivery
            pricetype="MARKET",
            price="0",
            trigger_price="0",
            validity="DAY",
            tag="momrot",
        )
        ok = (res or {}).get("status") in ("success", "ok") or (res or {}).get("s") == "ok"
    except Exception as e:
        logger.exception(f"placeorder fail {symbol}")
        err_msg = str(e)
        ok = False
    # Audit hook — always write, success OR failure
    try:
        from src.services.audit_service import write_order
        oid = (res or {}).get("id") or ((res or {}).get("data") or {}).get("orderid") or ""
        # Use LTP as ordered_price for MARKET orders so charges compute on a real number
        px = _fyers_live_ltp(symbol, user_id) or 0.0
        write_order(
            model_name=model_name,
            symbol=fyers_sym, side=side.upper(), qty=int(qty),
            ordered_price=float(px), fill_price=None, fill_qty=None,
            product="CNC", pricetype="MARKET",
            status=("placed" if ok else "rejected"),
            fyers_order_id=oid,
            error_text=(err_msg or (None if ok else (res or {}).get("message"))),
            raw_request=req, raw_response=res if res else {"error": err_msg},
        )
    except Exception as _e:
        logger.debug(f"audit write_order failed: {_e}")
    if err_msg:
        return {"ok": False, "error": err_msg}
    return {"ok": ok, "result": res}


def _fyers_available_cash(user_id: int = 1) -> float:
    """Fetch Fyers available_cash."""
    try:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        f = svc.funds(user_id) or {}
        return float((f.get("data") or {}).get("available_cash") or 0)
    except Exception:
        return 0.0


def _fyers_live_ltp(symbol: str, user_id: int = 1) -> float:
    """Fetch live LTP for a symbol via Fyers quotes API (real-time)."""
    try:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        fyers_sym = symbol if symbol.startswith("NSE:") else f"NSE:{symbol}-EQ"
        r = svc.quotes_multiple(user_id, [fyers_sym]) or {}
        data = (r.get("data") or {})
        q = data.get(fyers_sym) or {}
        return float(q.get("ltp") or 0)
    except Exception:
        return 0.0


def _fyers_holdings(user_id: int = 1) -> List[Dict]:
    """Fetch active Fyers holdings (settled + CNC pending T+1).

    Merges holdings/ + positions/ (CNC, netQty>0) into a unified list with
    standardized keys: symbol, quantity, average_price, last_price, pnl.
    """
    out = []
    seen = set()
    try:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        api = svc._get_api_instance(user_id)

        # Settled holdings
        h = svc.holdings(user_id) or {}
        for p in (h.get("data") or []):
            qty = float(p.get("quantity") or 0)
            if qty <= 0:
                continue
            sym = (p.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")
            seen.add(sym)
            out.append({
                "symbol": p.get("symbol"),
                "quantity": qty,
                "average_price": p.get("average_price"),
                "last_price": p.get("last_price"),
                "pnl": p.get("pnl"),
            })

        # CNC positions pending T+1
        pos_raw = api._make_request("GET", "positions") or {}
        for p in (pos_raw.get("data") or []):
            if p.get("productType") != "CNC":
                continue
            net_qty = float(p.get("netQty") or 0)
            if net_qty <= 0:
                continue
            sym_full = p.get("symbol") or ""
            sym = sym_full.replace("NSE:", "").replace("BSE:", "").replace("-EQ", "").replace("-B", "")
            if sym in seen:
                continue
            out.append({
                "symbol": sym_full,
                "quantity": net_qty,
                "average_price": str(p.get("buyAvg") or p.get("netAvg") or 0),
                "last_price": str(p.get("ltp") or 0),
                "pnl": str(p.get("unrealized_profit") or p.get("pl") or 0),
            })
    except Exception:
        logger.exception("_fyers_holdings fail")
    return out


def _notify_tg(text: str) -> None:
    try:
        from tools.live.telegram_notify import send
        send(text, "Markdown")
    except Exception:
        logger.exception("tg notify fail")


@momrot_bp.route("/run-now", methods=["POST"])
def api_run_now():
    """Force LIVE rebalance: rank N100, sell drop-outs from Fyers, buy new rank-1.

    Always live — places real Fyers orders.
    """
    _lk = None
    try:
        # (N1) serialise live placement against the cron executor + other
        # gunicorn workers on the shared Fyers account.
        from src.services.trading.trade_lock import trading_lock
        _lk = trading_lock()
        if not _lk.__enter__():
            return jsonify({"success": False,
                            "error": "another trade/rebalance is in progress — retry shortly"}), 409
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
            # Use LIVE Fyers LTP (not cached) so qty math reflects real fill price
            live_ltp = _fyers_live_ltp(rank1["symbol"], user_id)
            price = live_ltp if live_ltp > 0 else rank1["price"]
            # 0.5% buffer for brokerage + STT + GST (delivery ~0.13% actual)
            usable = cash * 0.995
            qty = int(usable // price) if price > 0 else 0
            if qty < 1:
                actions.append({"action": "BUY_SKIP", "symbol": rank1["symbol"],
                                "reason": f"insufficient cash ₹{cash:,.0f} for price ₹{price:.2f}"})
            else:
                res = _fyers_place_market(rank1["symbol"], qty, "BUY", user_id)
                actions.append({"action": "BUY", "symbol": rank1["symbol"],
                                "qty": qty, "price": price, "deploy": qty * price, "result": res})

        # Aggregate success/fail
        order_actions = [a for a in actions if a["action"] in ("BUY", "SELL")]
        successes = [a for a in order_actions if a.get("result", {}).get("ok")]
        failures = [a for a in order_actions if not a.get("result", {}).get("ok")]
        all_ok = bool(order_actions) and not failures

        # Telegram alert + UI message
        if actions:
            lines = ["🔴 *LIVE Rebalance*"]
            for a in actions:
                res = a.get("result", {})
                if a["action"] == "BUY_SKIP":
                    lines.append(f"- ⏭️ SKIP BUY {a.get('symbol','')} — {a.get('reason','')}")
                elif res.get("ok"):
                    order_id = ((res.get("result") or {}).get("data") or {}).get("orderid") or (res.get("result") or {}).get("orderid") or (res.get("result") or {}).get("id") or ""
                    lines.append(f"- ✅ {a['action']} {a.get('symbol','')} qty={a.get('qty','')} order_id=`{order_id}`")
                else:
                    err = (res.get("result") or {}).get("message") or res.get("error") or "unknown"
                    lines.append(f"- ❌ {a['action']} {a.get('symbol','')} qty={a.get('qty','')} — {err}")
            _notify_tg("\n".join(lines))
        else:
            _notify_tg("ℹ️ Rebalance: no action needed (rank-1 already held)")

        return jsonify({
            "success": all_ok if order_actions else True,
            "all_orders_ok": all_ok,
            "successes": len(successes),
            "failures": len(failures),
            "error_summary": "; ".join(
                ((a.get("result") or {}).get("result") or {}).get("message") or "unknown"
                for a in failures
            ) if failures else None,
            "actions": actions,
            "rank1": rank1,
        })
    except Exception as e:
        logger.exception("run-now fail")
        _notify_tg(f"❌ Rebalance error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if _lk is not None:
            try:
                _lk.__exit__(None, None, None)
            except Exception:
                pass


@momrot_bp.route("/buy-now", methods=["POST"])
def api_buy_now():
    """Manual buy: place market order for rank-1 (or specified symbol) using available cash."""

    _lk = None
    try:
        # (N1) serialise placement against cron + other gunicorn workers.
        from src.services.trading.trade_lock import trading_lock
        _lk = trading_lock()
        if not _lk.__enter__():
            return jsonify({"success": False,
                            "error": "another trade/rebalance is in progress — retry shortly"}), 409
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

        # (N1 TOCTOU) re-check live Fyers holdings INSIDE the lock — refuse to
        # double-buy a symbol the account already holds (the manual path had no
        # equivalent of the executor's duplicate-buy guard).
        _sym_bare = (symbol or "").replace("NSE:", "").replace("-EQ", "")
        for _p in _fyers_holdings(user_id):
            _ph = (_p.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")
            if _ph == _sym_bare and int(float(_p.get("quantity") or 0)) > 0:
                return jsonify({"success": False, "symbol": symbol,
                                "error": f"already holding {_sym_bare} at broker — skipping duplicate buy"}), 409

        # Override `price` with live Fyers LTP for accurate qty
        live_ltp = _fyers_live_ltp(symbol, user_id)
        if live_ltp > 0:
            price = live_ltp
        cash = _fyers_available_cash(user_id)
        qty = body.get("qty")
        if qty:
            qty = int(qty)
        else:
            # 0.5% buffer for brokerage/STT/GST
            usable = cash * 0.995
            qty = int(usable // price) if price > 0 else 0

        if qty < 1:
            return jsonify({"success": False, "error": f"qty<1 (cash ₹{cash:,.0f}, price ₹{price:.2f})"}), 400

        res = _fyers_place_market(symbol, qty, "BUY", user_id)
        if res.get("ok"):
            order_id = ((res.get("result") or {}).get("data") or {}).get("orderid") or (res.get("result") or {}).get("orderid") or (res.get("result") or {}).get("id") or ""
            _notify_tg(f"✅ *BUY OK* {symbol} qty={qty} @ ₹{price:.2f} (₹{qty*price:,.0f}) order_id=`{order_id}`")
            return jsonify({"success": True, "symbol": symbol, "qty": qty,
                            "price": price, "order_id": order_id, "result": res})
        else:
            err = (res.get("result") or {}).get("message") or res.get("error") or "unknown error"
            _notify_tg(f"❌ *BUY FAIL* {symbol} qty={qty} — {err}")
            return jsonify({"success": False, "symbol": symbol, "qty": qty,
                            "error": err, "result": res}), 400
    except Exception as e:
        logger.exception("buy-now fail")
        _notify_tg(f"❌ Buy error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if _lk is not None:
            try:
                _lk.__exit__(None, None, None)
            except Exception:
                pass


@momrot_bp.route("/rebalance-preview")
@ui_cached('momrot_rebpreview', ttl=20)
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
    _lk = None
    try:
        from src.services.trading.trade_lock import trading_lock
        # Longer wait than buy: an operator sell should queue behind a cron
        # execute rather than 409 during the morning window.
        _lk = trading_lock(wait_s=120)
        if not _lk.__enter__():
            return jsonify({"success": False,
                            "error": "another trade/rebalance is in progress — retry shortly"}), 409
        user_id = int(request.args.get("user_id", 1))
        body = request.get_json(silent=True) or {}
        symbol = body.get("symbol")
        qty = body.get("qty")
        if not symbol or not qty:
            return jsonify({"success": False, "error": "symbol + qty required"}), 400
        qty = int(qty)
        res = _fyers_place_market(symbol, qty, "SELL", user_id)
        if res.get("ok"):
            order_id = ((res.get("result") or {}).get("data") or {}).get("orderid") or (res.get("result") or {}).get("orderid") or (res.get("result") or {}).get("id") or ""
            _notify_tg(f"✅ *SELL OK* {symbol} qty={qty} order_id=`{order_id}`")
            return jsonify({"success": True, "symbol": symbol, "qty": qty,
                            "order_id": order_id, "result": res})
        else:
            err = (res.get("result") or {}).get("message") or res.get("error") or "unknown error"
            _notify_tg(f"❌ *SELL FAIL* {symbol} qty={qty} — {err}")
            return jsonify({"success": False, "symbol": symbol, "qty": qty,
                            "error": err, "result": res}), 400
    except Exception as e:
        logger.exception("sell-now fail")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if _lk is not None:
            try:
                _lk.__exit__(None, None, None)
            except Exception:
                pass


@momrot_bp.route("/rebuild-universe", methods=["POST"])
def api_rebuild_universe():
    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_path = UNIVERSE_PATH.parent / f"n100_{date_str}.json"

        def run():
            try:
                subprocess.run([
                    "python", "tools/models/momentum_n100_top5_max1/build_universe.py",
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
@ui_cached('momrot_summary', ttl=30)
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
@ui_cached('momrot_fyacct', ttl=20)
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
@ui_cached('momrot_fyob', ttl=15)
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


# ---------------------------------------------------------------------------
# Per-model "Invest more": suggest deploying a model's idle sleeve cash into
# its OWN current picks (no selling). The operator may DECREASE the qty before
# Approve. Server re-derives the MAX allowed buys and CLAMPS the client request
# to it (decrease-only, own-picks-only, idle-capped) — it never trusts the
# client list to add symbols or increase qty. Order placement + ledger writes
# REUSE the existing primitives (_fyers_place_market, record_buy[_multi]); this
# module adds NO new buy/sell logic. Sizing lives in model_invest_service.py.
# ---------------------------------------------------------------------------

def _model_idle_cash(model_name: str) -> float:
    """The model's uninvested sleeve cash (ModelLedger.cash)."""
    try:
        from src.models.database import get_database_manager
        from src.models.model_ledger_models import ModelLedger
        db = get_database_manager()
        with db.get_session() as s:
            row = s.query(ModelLedger).filter_by(model_name=model_name).first()
            return float(row.cash or 0) if row else 0.0
    except Exception:
        logger.exception("idle-cash read fail")
        return 0.0


def _model_own_held(model_name: str):
    """Symbols THIS model holds (its own sleeve) — NOT account-wide. Multi-slot
    models use model_holdings; single-position uses ledger.open_symbol."""
    try:
        from src.services.trading.model_ledger_service import model_max_holdings
        if model_max_holdings(model_name):
            from src.services.trading.multi_holding_service import held_symbols
            return set(held_symbols(model_name) or set())
        from src.models.database import get_database_manager
        from src.models.model_ledger_models import ModelLedger
        db = get_database_manager()
        with db.get_session() as s:
            row = s.query(ModelLedger).filter_by(model_name=model_name).first()
            sym = ((row.open_symbol or "").replace("NSE:", "").replace("-EQ", "")
                   if row else "")
            return {sym} if sym else set()
    except Exception:
        logger.exception("own-held read fail")
        return set()


def _model_ranking_targets(model_name: str, user_id: int = 1):
    """Model's current picks [{symbol, ltp}] from its ranking file, live LTP."""
    import json as _json
    import os as _os
    from datetime import datetime as _dt
    try:
        from src.web.admin_routes import MODEL_PATHS
    except Exception:
        return []
    paths = MODEL_PATHS.get(model_name) or {}
    rdir = paths.get("ranking_dir")
    if not rdir:
        return []
    f = _os.path.join(rdir, _dt.now().strftime("%Y-%m-%d") + ".json")
    if not _os.path.exists(f):
        return []
    try:
        top = (_json.load(open(f)).get("top_n") or [])
    except Exception:
        return []
    out = []
    for r in top:
        sym = (r.get("symbol") or "").replace("NSE:", "").replace("-EQ", "")
        if not sym:
            continue
        ltp = _fyers_live_ltp(sym, user_id) or float(r.get("price") or 0)
        out.append({"symbol": sym, "ltp": float(ltp)})
    return out


def _held_targets(symbols, user_id: int = 1):
    """[{symbol, ltp}] for a set of held symbols, live LTP, priced rows only."""
    out = []
    for sym in symbols:
        ltp = _fyers_live_ltp(sym, user_id)
        if ltp and float(ltp) > 0:
            out.append({"symbol": sym, "ltp": float(ltp)})
    return out


def _derive_invest(model_name: str, user_id: int = 1):
    """Server-side source of truth: the MAX buys this model may make right now,
    deploying idle cash WITHOUT ever exceeding max_holdings.
      - single position: top up its current holding (rank-1 if flat).
      - multi-slot with FREE slots: fill the free slots with top-ranked unheld
        ranking names.
      - multi-slot FULL book: top up the existing holdings (no new name added).
    Sized to min(idle, broker)."""
    from src.services.trading.model_invest_service import compute_buys
    from src.services.trading.model_ledger_service import model_max_holdings
    idle = _model_idle_cash(model_name)
    broker = _fyers_available_cash(user_id)
    maxh = model_max_holdings(model_name) or 1
    is_multi = bool(model_max_holdings(model_name))
    own_held = _model_own_held(model_name)
    if is_multi:
        free = maxh - len(own_held)
        if free > 0:
            # fill free slots with top-ranked names not yet held
            targets = _model_ranking_targets(model_name, user_id)
            open_syms = own_held
            buys = compute_buys(idle, broker, maxh, targets, open_syms)
        else:
            # FULL book -> top up the existing holdings (stays at max_holdings;
            # record_buy_multi accumulates onto the existing rows, no 5th name).
            targets = _held_targets(own_held, user_id)
            buys = compute_buys(idle, broker, len(targets) or 1, targets, set())
    else:
        # single position: top up the symbol actually held (avoids record_buy's
        # "different symbol" raise); use rank-1 only when the model is flat.
        held_sym = next(iter(own_held), "")
        if held_sym:
            ltp = _fyers_live_ltp(held_sym, user_id)
            targets = [{"symbol": held_sym, "ltp": float(ltp or 0)}]
        else:
            targets = _model_ranking_targets(model_name, user_id)[:1]
        buys = compute_buys(idle, broker, maxh, targets, set())
    return {"idle": idle, "broker": broker,
            "deployable": min(max(0.0, idle), max(0.0, broker)),
            "buys": buys, "is_multi": is_multi}


def is_market_open_now() -> bool:
    from datetime import datetime as _dt
    from src.services.trading.model_invest_service import is_market_open
    try:
        from zoneinfo import ZoneInfo
        now = _dt.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        # tzdata missing — derive IST explicitly from UTC (never trust naive
        # local time on a UTC container, which would shift the trading window).
        from datetime import timedelta
        now = _dt.utcnow() + timedelta(hours=5, minutes=30)
    return is_market_open(now)


def _token_consume(token: str) -> bool:
    """Atomic single-use token via Redis SET NX. True only for the first caller.
    On cache outage returns True (fail-open) — the trading lock plus live
    ledger-cash re-derivation are the PRIMARY double-deploy guards."""
    if not token:
        return False
    try:
        from src.services.utils.cache_service import get_cache_service
        return bool(get_cache_service().set_if_absent(
            f"invest:token:{token}", "1", 3600))
    except Exception:
        logger.warning("token cache unavailable; relying on lock+ledger guard")
        return True


@momrot_bp.route("/models/<model_name>/invest-preview", methods=["GET"])
def api_invest_preview(model_name):
    """Read-only: suggest deploying the model's idle cash into its own picks.
    The qty shown is the MAXIMUM; the operator may decrease before approving."""
    try:
        from datetime import datetime as _dt
        from src.services.trading.model_invest_service import make_token
        user_id = int(request.args.get("user_id", 1))
        d = _derive_invest(model_name, user_id)
        day = _dt.now().strftime("%Y-%m-%d")
        return jsonify({
            "success": True, "model": model_name,
            "idle_cash": d["idle"], "broker_free": d["broker"],
            "deployable": d["deployable"], "market_open": is_market_open_now(),
            "buys": d["buys"], "token": make_token(model_name, d["buys"], day),
        })
    except Exception:
        logger.exception("invest-preview fail")
        return jsonify({"success": False, "error": "internal error"}), 500


@momrot_bp.route("/models/<model_name>/invest-execute", methods=["POST"])
def api_invest_execute(model_name):
    """Place BUYs for the model. Re-derives the MAX allowed server-side and
    CLAMPS the client request to it (operator may DECREASE qty, never increase
    or substitute symbols). Market-hours only, idempotent, ledger-recorded.
    Uses the existing _fyers_place_market + record_buy[_multi] primitives."""
    _lk = None
    try:
        from datetime import datetime as _dt
        from src.services.trading.trade_lock import trading_lock
        from src.services.trading.model_invest_service import make_token
        from src.services.trading.model_ledger_service import record_buy
        from src.services.trading.multi_holding_service import record_buy_multi
        body = request.get_json(silent=True) or {}
        req_buys = body.get("buys") or []
        if not req_buys:
            return jsonify({"success": False, "error": "no buys"}), 400
        if not is_market_open_now():
            return jsonify({"success": False,
                            "error": "market closed — deploys next session"}), 400
        user_id = int(request.args.get("user_id", 1))
        # serialise against cron rebalance / other order placement FIRST
        _lk = trading_lock()
        if not _lk.__enter__():
            return jsonify({"success": False,
                            "error": "another trade/rebalance in progress — retry shortly"}), 409
        # re-derive the MAX allowed buys live (own picks, idle+broker capped)
        d = _derive_invest(model_name, user_id)
        allowed = {b["symbol"]: b for b in d["buys"]}
        # clamp client request: decrease-only, own-picks-only
        final = []
        for b in req_buys:
            sym = (b.get("symbol") or "")
            if sym not in allowed:
                continue
            try:
                want = int(b.get("qty") or 0)
            except Exception:
                continue
            qty = min(want, int(allowed[sym]["qty"]))
            if qty >= 1:
                final.append({"symbol": sym, "qty": qty, "ltp": allowed[sym]["ltp"]})
        if not final:
            return jsonify({"success": False,
                            "error": "nothing to deploy (suggestion changed — re-open)"}), 400
        # idempotency (secondary to lock+ledger): token over the FINAL buys
        day = _dt.now().strftime("%Y-%m-%d")
        if not _token_consume(make_token(model_name, final, day)):
            return jsonify({"success": False, "error": "already submitted — retry shortly"}), 409
        broker = _fyers_available_cash(user_id)
        spent, fills = 0.0, []
        for b in final:
            sym, qty = b["symbol"], int(b["qty"])
            ltp = _fyers_live_ltp(sym, user_id) or float(b.get("ltp") or 0)
            if qty < 1 or ltp <= 0:
                continue
            if spent + qty * ltp > broker * 0.995:   # live broker-cash ceiling
                continue
            res = _fyers_place_market(sym, qty, "BUY", user_id)
            if res.get("ok"):
                oid = (((res.get("result") or {}).get("data") or {}).get("orderid")
                       or (res.get("result") or {}).get("orderid")
                       or (res.get("result") or {}).get("id") or "")
                try:
                    if d["is_multi"]:
                        record_buy_multi(model_name, sym, qty, ltp, fyers_order_id=oid)
                    else:
                        record_buy(model_name, sym, qty, ltp, fyers_order_id=oid)
                except Exception as e:
                    logger.warning(f"ledger record failed {model_name} {sym}: {e}")
                spent += qty * ltp
                fills.append({"symbol": sym, "qty": qty, "price": ltp, "order_id": oid})
                _notify_tg(f"✅ *INVEST* `{model_name}` BUY {sym} x{qty} @ ₹{ltp:.2f} (₹{qty*ltp:,.0f})")
            else:
                err = (res.get("result") or {}).get("message") or res.get("error") or "unknown"
                fills.append({"symbol": sym, "qty": qty, "error": err})
                _notify_tg(f"❌ *INVEST FAIL* `{model_name}` {sym} x{qty} — {err}")
        ok_all = bool(fills) and all(f.get("order_id") for f in fills)
        return jsonify({"success": ok_all, "model": model_name,
                        "fills": fills, "deployed": round(spent, 2)})
    except Exception:
        logger.exception("invest-execute fail")
        return jsonify({"success": False, "error": "internal error"}), 500
    finally:
        if _lk is not None:
            try:
                _lk.__exit__(None, None, None)
            except Exception:
                pass
