"""Multi-holding order executor — for K>1 models (momentum_retest_n500).

Reads a multi-holding signals file ({model, sells:[{symbol,reason}], buys:[{symbol}]})
and places the orders, REUSING the single executor's order primitives
(place_limit_with_fallback, _fetch_live_ltp, FyersService) so the broker-side
behaviour (limit→market fallback, fill wait) is identical. Only the bookkeeping
differs: it records fills via multi_holding_service (record_buy_multi /
record_sell_multi) into model_holdings, not the single-position ledger.

The single-holding executor (fyers_executor.py) is untouched.

Usage:
  python tools/live/fyers_executor_multi.py --signals <signals.json> [--dry-run] [--user-id 1]
"""
import sys, json, argparse, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.live.fyers_executor import (
    place_limit_with_fallback, _fetch_live_ltp, to_fyers_symbol,
    _resolve_fill_price,
)
from src.services.trading.multi_holding_service import (
    get_holdings, record_buy_multi, record_sell_multi,
)
from src.services.trading import model_ledger_service as MLS

log = logging.getLogger("fyers_executor_multi")


def multi_trading_blocked(enabled: bool, is_trading_day: bool):
    """Pre-flight gate for a LIVE multi-holding run. Returns an abort reason
    string, or None when trading is allowed.

    Mirrors the single executor's backstops (the multi executor previously had
    NONE): refuse to place real orders for a disabled model, or on an NSE
    non-trading day (holiday/weekend). Pure so it is unit-testable; the IO
    (settings read, calendar lookup) happens in main() and feeds this.
    """
    if not enabled:
        return "model_settings.enabled is False/missing — refusing to execute (backstop)"
    if not is_trading_day:
        return "not an NSE trading day (holiday/weekend) — refusing to execute"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sig = json.loads(Path(a.signals).read_text())
    model = sig["model"]; sells = sig.get("sells", []); buys = sig.get("buys", [])
    # signals_only: per-model observe mode. Placement + ledger writes below are
    # gated on `not dry_run`, so forcing dry-run here logs intended trades and
    # places NOTHING, while live_signal still emitted the signals.
    if not a.dry_run:
        try:
            from src.services.trading.model_ledger_service import get_all_settings
            _settings = get_all_settings()
            _row = next((s for s in _settings if s["model_name"] == model), None)
            if _row and _row.get("signals_only"):
                log.info(f"{model}: signals_only=True — OBSERVE mode (place NOTHING).")
                a.dry_run = True
        except Exception as _se:
            _row = None
            log.warning(f"signals_only read failed ({_se}) — proceeding live.")

    # Pre-flight backstops (mirror the single executor — the multi path had
    # none): enabled-model + NSE trading-day. Fail-closed on a read error so a
    # disabled/holiday run can never place real orders blind.
    if not a.dry_run:
        try:
            from tools.shared.nse_calendar import is_trading_day
            from datetime import datetime
            _enabled = bool(_row.get("enabled")) if _row else False
            _trading = is_trading_day(datetime.now())
            _block = multi_trading_blocked(_enabled, _trading)
            if _block:
                log.error(f"{model}: {_block}")
                return 2
        except Exception as _ge:
            log.error(f"{model}: pre-flight guard failed ({_ge}) — refusing "
                      f"to execute (fail-closed).")
            return 2
    log.info(f"{model}: {len(sells)} sells, {len(buys)} buys (dry_run={a.dry_run})")

    # Init broker svc even in dry-run so we can fetch LTP for realistic sizing;
    # actual order placement + recording stay gated on `not dry_run`.
    svc = None
    try:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        if not svc.get_broker_config(a.user_id):
            if not a.dry_run:
                log.error(f"No Fyers token for user {a.user_id} — abort"); return 1
            log.warning("No Fyers token — dry-run sizing will fall back to entry/signal px")
            svc = None
        elif not a.dry_run:
            # Token PRE-FLIGHT — a token can exist yet be expired. Probe a
            # reference quote before placing any real order (mirrors single
            # executor); abort rather than fire blind on a dead token.
            _probe = _fetch_live_ltp(svc, a.user_id, "NSE:SBIN-EQ")
            if not _probe or _probe <= 0:
                log.error("Fyers token pre-flight FAILED (NSE:SBIN quote empty) "
                          "— token likely expired; aborting.")
                return 2
    except Exception as e:
        if not a.dry_run:
            raise
        log.warning(f"broker init failed in dry-run: {e}")

    held = {h["symbol"]: h for h in get_holdings(model)}

    # (N1) Cross-process trading lock — serialise live placement against cron +
    # manual endpoints on the shared Fyers account. Skipped in dry-run.
    _trade_lock = None
    if not a.dry_run:
        from src.services.trading.trade_lock import trading_lock
        # Long wait so concurrent same-minute executes QUEUE rather than skip
        # (see fyers_executor.py). The executor must run its trade.
        _trade_lock = trading_lock(wait_s=600)
        if not _trade_lock.__enter__():
            log.error("Trading lock unavailable after 600s — skipping this run "
                      "(stuck holder?).")
            _trade_lock.__exit__(None, None, None)
            return 4
    try:
        return _run_orders(a, model, sells, buys, held, svc)
    finally:
        if _trade_lock is not None:
            try:
                _trade_lock.__exit__(None, None, None)
            except Exception:
                pass


def _run_orders(a, model, sells, buys, held, svc) -> int:
    # ---- SELLS first (free up cash + slots) ----
    for sl in sells:
        sym = sl["symbol"]; reason = sl.get("reason", "ROTATE")
        h = held.get(sym)
        if not h:
            log.warning(f"  SELL {sym}: not held, skip"); continue
        qty = h["qty"]
        if a.dry_run:
            log.info(f"  [dry] SELL {qty}x{sym} ({reason})"); continue
        ltp = _fetch_live_ltp(svc, a.user_id, sym) or h["entry_px"]
        res = place_limit_with_fallback(svc, a.user_id, sym, qty, "SELL", ltp,
                                        tag=f"{model}_sell")
        if res.get("filled"):
            px = _resolve_fill_price(res, ltp)
            # Forward ACTUAL filled qty so a partial fill keeps the residual
            # holding instead of deleting the whole row.
            _fq = int(res.get("fill_qty") or 0)
            record_sell_multi(model, sym, float(px), reason,
                              fyers_order_id=res.get("order_id"),
                              qty=(_fq if _fq > 0 else None))
            log.info(f"  SOLD {_fq or qty}x{sym}@{px} ({reason})")
        else:
            log.error(f"  SELL {sym} not filled: {res.get('status')}")

    # ---- BUYS (equal-weight across open slots) ----
    if buys:
        led = MLS.get_ledger(model) or {}
        cash = float(led.get("cash") or 0)
        n = len(buys)
        alloc = cash / n if n else 0
        for b in buys:
            sym = b["symbol"]
            ltp = (h["entry_px"] if (h := held.get(sym)) else None)
            if svc:
                ltp = _fetch_live_ltp(svc, a.user_id, sym) or ltp
            if not ltp:
                log.warning(f"  BUY {sym}: no price, skip"); continue
            qty = int(alloc / float(ltp))
            if qty < 1:
                log.warning(f"  BUY {sym}: alloc {alloc:.0f} < 1 share @ {ltp}"); continue
            if a.dry_run:
                log.info(f"  [dry] BUY {qty}x{sym} @~{ltp} (alloc {alloc:.0f})"); continue
            res = place_limit_with_fallback(svc, a.user_id, sym, qty, "BUY", ltp,
                                            tag=f"{model}_buy")
            if res.get("filled"):
                px = _resolve_fill_price(res, ltp)
                record_buy_multi(model, sym, res.get("fill_qty") or qty, float(px),
                                 fyers_order_id=res.get("order_id"))
                log.info(f"  BOUGHT {qty}x{sym}@{px}")
            else:
                log.error(f"  BUY {sym} not filled: {res.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
