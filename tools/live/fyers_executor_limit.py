"""LIMIT-order executor for price_meanrev_n500 (resting dip-buy limits, K=3).

WHY A SEPARATE EXECUTOR: this model's edge lives in the LIMIT fill at the dip
level (close-fill drops 2025-03->now CAGR from 102.8% to 36.1% — see the model's
strategy.py). fyers_executor_multi fills NOW at live LTP (limit-with-market-
fallback) — the exact close-fill semantics that kill the edge. This executor
instead mirrors the backtest's mechanics with REAL broker orders:

  --place      (09:16, post-open)  For each free slot, place a resting LIMIT
               BUY *day* order at the signal's level (below market). Unfilled
               orders auto-expire at the close (day validity) — exactly the
               backtest's "order lapses if the low never touches".
  --reconcile  (15:20)  Poll the order book: traded orders are recorded into
               model_holdings via record_buy_multi at the ACTUAL fill price,
               and the frozen stop (fill − 1.5×ATR) + target (SMA50 at signal
               time) are persisted to the meta state.
  --exits      (every 5 min, 09:30–15:10)  For each holding, check live LTP
               against the frozen stop / target / 40-trading-day time exit
               (stop checked FIRST — backtest parity) and square off at market
               via place_limit_with_fallback. 5-min polling approximates the
               backtest's intraday touch.

State: /app/logs/price_meanrev_n500/live_orders.json
  {"orders": [{order_id, symbol, qty, limit, atr, target, placed}],
   "meta": {symbol: {stop, target, entry_date}}}

Reuses the shared primitives (placement, order-book polling, locks, universe
guard, margin gate, audit, Telegram) so broker behaviour matches the fleet.
"""
import sys, json, argparse, logging
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.shared.model_universe import is_in_universe

log = logging.getLogger("fyers_executor_limit")
MODEL = "price_meanrev_n500"
STATE_DIR = Path("/app/logs/price_meanrev_n500")
ORDERS_FILE = STATE_DIR / "live_orders.json"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def decide_exit(ltp: float, stop: float, target: float, held_days: int,
                maxhold: int) -> str | None:
    """Exit reason for a holding given live LTP — or None to keep holding.

    Stop is checked FIRST (backtest parity: stop fills before target on a
    both-hit day). Time exit fires once held_days >= maxhold.
    """
    if ltp is None or ltp <= 0:
        return None
    if stop and ltp <= stop:
        return "STOP"
    if target and ltp >= target:
        return "TARGET"
    if held_days >= maxhold:
        return "TIME"
    return None


def trading_days_between(start: date, end: date, is_trading_day_fn) -> int:
    """Count NSE trading days in (start, end] — held-days for the time exit."""
    n, d = 0, start
    while d < end:
        d += timedelta(days=1)
        if is_trading_day_fn(datetime.combine(d, datetime.min.time())):
            n += 1
    return n


def load_state() -> dict:
    if ORDERS_FILE.exists():
        try:
            return json.loads(ORDERS_FILE.read_text())
        except Exception as e:
            log.error(f"live_orders state corrupt ({e}) — starting fresh")
    return {"orders": [], "meta": {}}


def save_state(st: dict):
    ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ORDERS_FILE.write_text(json.dumps(st, indent=2))


# ---------------------------------------------------------------------------
# Gates (mirror fyers_executor_multi)
# ---------------------------------------------------------------------------
def _settings_row():
    from src.services.trading.model_ledger_service import get_all_settings
    return next((s for s in (get_all_settings() or [])
                 if s["model_name"] == MODEL), None)


def _preflight(svc, user_id) -> str | None:
    """Abort reason, or None when live trading is allowed. Fail-closed."""
    try:
        row = _settings_row()
        if not row or not row.get("enabled"):
            return "model disabled — refusing to execute"
        if row.get("signals_only"):
            return "signals_only=True — OBSERVE mode, placing nothing"
        from tools.shared.nse_calendar import is_trading_day
        if not is_trading_day(datetime.now()):
            return "not an NSE trading day"
        from tools.live.fyers_executor import _fetch_live_ltp
        if not (_fetch_live_ltp(svc, user_id, "NSE:SBIN-EQ") or 0) > 0:
            return "Fyers token pre-flight FAILED (SBIN quote empty)"
    except Exception as e:
        return f"pre-flight guard failed ({e}) — fail-closed"
    return None


# ---------------------------------------------------------------------------
# --place: resting LIMIT BUY day orders at the signal levels
# ---------------------------------------------------------------------------
def do_place(a, svc, st) -> int:
    from tools.live.fyers_executor import _placeorder, _extract_order_id, _is_ok
    from src.services.trading.multi_holding_service import get_holdings
    from src.services.trading import model_ledger_service as MLS
    from tools.models.price_meanrev_n500 import strategy as S

    sig_file = STATE_DIR / "signals" / "latest.json"
    if not sig_file.exists():
        log.warning("no signals file — nothing to place"); return 0
    sig = json.loads(sig_file.read_text())
    if sig.get("date") != date.today().isoformat():
        log.warning(f"signals dated {sig.get('date')} != today — refusing to "
                    f"place on stale levels"); return 0
    buys = sig.get("buys", [])
    held = {h["symbol"] for h in get_holdings(MODEL)}
    pending_syms = {o["symbol"] for o in st["orders"]}
    free = S.K - len(held) - len(pending_syms)
    if free <= 0 or not buys:
        log.info(f"no free slots ({len(held)} held, {len(pending_syms)} pending) "
                 f"or no candidates"); return 0

    row = _settings_row() or {}
    invested = float(row.get("invested_amount") or 0)
    led = MLS.get_ledger(MODEL) or {}
    cash = float(led.get("cash") or 0)
    per_slot = invested / S.K if invested else cash / max(1, free)
    from tools.live.fyers_executor import _fetch_account_available_cash
    acct = None if a.dry_run else _fetch_account_available_cash(svc, a.user_id)
    log.info(f"place: free={free} per_slot=₹{per_slot:,.0f} cash=₹{cash:,.0f} "
             f"acct={'?' if acct is None else f'₹{acct:,.0f}'}")

    placed = 0
    for b in buys:
        if placed >= free:
            break
        sym = MLS._normalize_symbol(b["symbol"])
        limit = float(b.get("limit_price") or b.get("limit") or 0)
        atr = float(b.get("atr") or 0)
        target = float(b.get("target") or 0)
        if sym in held or sym in pending_syms or limit <= 0 or atr <= 0:
            continue
        if is_in_universe(MODEL, sym, date.today()) is False:
            log.error(f"  {sym}: universe guard — REFUSING"); continue
        budget = min(per_slot, cash)
        qty = int(budget / limit)
        if qty < 1:
            log.warning(f"  {sym}: qty<1 (budget {budget:.0f} @ limit {limit})")
            continue
        cost = qty * limit
        if acct is not None and cost > acct:
            log.error(f"  {sym}: cost ₹{cost:,.0f} > account cash — margin gate")
            continue
        if a.dry_run:
            log.info(f"  [dry] LIMIT BUY {qty}x{sym} @ {limit}"); placed += 1
            continue
        res = _placeorder(svc, a.user_id, sym, qty, "BUY",
                          pricetype="LIMIT", price=limit,
                          tag=f"{MODEL}_lim", _audit_model=MODEL)
        oid = _extract_order_id(res)
        if _is_ok(res) and oid:
            st["orders"].append({"order_id": oid, "symbol": sym, "qty": qty,
                                 "limit": limit, "atr": atr, "target": target,
                                 "placed": date.today().isoformat()})
            placed += 1
            if acct is not None:
                acct -= cost
            cash = max(0.0, cash - cost)
            log.info(f"  RESTING LIMIT BUY {qty}x{sym} @ {limit} (id {oid})")
            _tg(f"📌 *LIMIT placed {MODEL}*\n`{sym}` x{qty} @ ₹{limit:.2f} "
                f"(target ₹{target:.2f}) — fills only on a dip-touch, "
                f"expires at close")
        else:
            log.error(f"  LIMIT BUY {sym} rejected: {res}")
    save_state(st)
    return 0


# ---------------------------------------------------------------------------
# --reconcile: record traded limit orders into the ledger
# ---------------------------------------------------------------------------
def do_reconcile(a, svc, st) -> int:
    from tools.live.fyers_executor import _get_order_status, _extract_traded_price
    from src.services.trading.multi_holding_service import record_buy_multi
    from tools.models.price_meanrev_n500 import strategy as S

    keep = []
    for o in st["orders"]:
        row = None if a.dry_run else _get_order_status(svc, a.user_id, o["order_id"])
        status = int((row or {}).get("status") or 0)
        if status == 2:                                   # traded
            fill = _extract_traded_price(row) or o["limit"]
            if not a.dry_run:
                record_buy_multi(MODEL, o["symbol"], o["qty"], float(fill),
                                 fyers_order_id=o["order_id"])
            stop = S.stop_price(float(fill), o["atr"])
            st["meta"][o["symbol"]] = {"stop": round(stop, 2),
                                       "target": round(o["target"], 2),
                                       "entry_date": o["placed"]}
            log.info(f"  FILLED {o['qty']}x{o['symbol']} @ {fill} "
                     f"(stop {stop:.2f}, target {o['target']:.2f})")
            _tg(f"✅ *LIMIT FILLED {MODEL}*\n`{o['symbol']}` x{o['qty']} @ "
                f"₹{float(fill):.2f}\nstop ₹{stop:.2f} · target ₹{o['target']:.2f}")
        elif status in (1, 5, 7):                         # cancelled/rejected/expired
            log.info(f"  order {o['order_id']} {o['symbol']} lapsed (status {status})")
        elif (date.today() - date.fromisoformat(o["placed"])).days > 1:
            log.warning(f"  order {o['order_id']} {o['symbol']} stale (> 1d, "
                        f"status {status}) — dropping from tracking")
        else:
            keep.append(o)                                # still pending today
    st["orders"] = keep
    save_state(st)
    return 0


# ---------------------------------------------------------------------------
# --exits: stop / target / time square-off (5-min intraday polling)
# ---------------------------------------------------------------------------
def do_exits(a, svc, st) -> int:
    from tools.live.fyers_executor import (
        place_limit_with_fallback, _fetch_live_ltp, _resolve_fill_price,
        resolve_sell_product_qty)
    from src.services.trading.multi_holding_service import get_holdings, record_sell_multi
    from src.services.trading import model_ledger_service as MLS
    from tools.live.risk_manager import RiskManager
    from tools.shared.nse_calendar import is_trading_day
    from tools.models.price_meanrev_n500 import strategy as S
    import tools.live.fyers_executor as _fe
    _fe._CURRENT_MODEL = MODEL
    rm_cfg = RiskManager.from_model(MODEL).cfg

    for h in get_holdings(MODEL):
        sym = h["symbol"]
        m = st["meta"].get(sym)
        if not m:
            # meta lost (crash/manual) — conservative fallback so the position
            # is never unprotected: stop from entry via current ATR is unknown
            # here; use entry_px ± strategy multiples of a 2% proxy and WARN.
            log.warning(f"  {sym}: no exit meta — fallback stop/target from entry")
            m = {"stop": h["entry_px"] * 0.94, "target": h["entry_px"] * 1.06,
                 "entry_date": h.get("entry_date") or date.today().isoformat()}
            st["meta"][sym] = m
        ltp = _fetch_live_ltp(svc, a.user_id, sym) if svc else None
        held_days = trading_days_between(
            date.fromisoformat(m["entry_date"]), date.today(), is_trading_day)
        why = decide_exit(ltp, m["stop"], m["target"], held_days, S.MAXHOLD)
        if not why:
            continue
        qty = h["qty"]
        if a.dry_run:
            log.info(f"  [dry] EXIT {why} {qty}x{sym} @ ~{ltp}"); continue
        sell_prod, sell_qty, src = resolve_sell_product_qty(
            svc, a.user_id, sym, qty, MLS.product_for_model(MODEL))
        if src == "flat" or sell_qty <= 0:
            log.warning(f"  {sym}: broker flat — closing ledger row only")
            record_sell_multi(MODEL, sym, float(ltp or h["entry_px"]),
                              f"{why}_BROKER_FLAT", fyers_order_id="")
            st["meta"].pop(sym, None)
            continue
        res = place_limit_with_fallback(svc, a.user_id, sym, sell_qty, "SELL",
                                        ltp, rm_cfg, tag=f"{MODEL}_exit",
                                        product=sell_prod)
        if res.get("filled"):
            px = _resolve_fill_price(res, ltp)
            fq = int(res.get("fill_qty") or 0)
            record_sell_multi(MODEL, sym, float(px), why,
                              fyers_order_id=res.get("order_id"),
                              qty=(fq if fq > 0 else None))
            st["meta"].pop(sym, None)
            log.info(f"  EXIT {why}: SOLD {fq or sell_qty}x{sym} @ {px}")
            _tg(f"💰 *EXIT {why} {MODEL}*\n`{sym}` x{fq or sell_qty} @ "
                f"₹{float(px):.2f}")
        else:
            log.error(f"  EXIT {why} {sym} NOT filled: {res.get('status')}")
            _tg(f"🛑 *EXIT FAILED {MODEL}*\n`{sym}` {why} — will retry next "
                f"5-min cycle", fail=True)
    save_state(st)
    return 0


def _tg(msg, fail=False):
    try:
        from tools.live.fyers_executor import _tg_safe
        _tg_safe(msg, is_fail=fail)
    except Exception as e:
        log.debug(f"tg failed: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["place", "reconcile", "exits"])
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    svc = None
    if not a.dry_run:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        if not svc.get_broker_config(a.user_id):
            log.error("No Fyers token — abort"); return 1
        block = _preflight(svc, a.user_id)
        if block:
            log.warning(f"{MODEL}: {block}")
            return 0 if "signals_only" in block or "disabled" in block else 2

    st = load_state()
    if a.mode in ("place", "exits") and not a.dry_run:
        from src.services.trading.trade_lock import trading_lock
        lk = trading_lock(wait_s=600)
        if not lk.__enter__():
            log.error("trading lock unavailable — skip"); lk.__exit__(None, None, None)
            return 4
        try:
            if a.mode == "place":
                return do_place(a, svc, st)
            # exits: reconcile FIRST so a limit that filled minutes ago gets its
            # frozen stop/target meta before the exit logic sees the holding
            # (otherwise the fill-day position would run on the fallback meta).
            do_reconcile(a, svc, st)
            return do_exits(a, svc, st)
        finally:
            lk.__exit__(None, None, None)
    if a.mode == "place":
        return do_place(a, svc, st)
    if a.mode == "exits":
        do_reconcile(a, svc, st)
        return do_exits(a, svc, st)
    return do_reconcile(a, svc, st)


if __name__ == "__main__":
    raise SystemExit(main())
