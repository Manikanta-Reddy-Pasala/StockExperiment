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
)
from src.services.trading.multi_holding_service import (
    get_holdings, record_buy_multi, record_sell_multi,
)
from src.services.trading import model_ledger_service as MLS

log = logging.getLogger("fyers_executor_multi")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sig = json.loads(Path(a.signals).read_text())
    model = sig["model"]; sells = sig.get("sells", []); buys = sig.get("buys", [])
    log.info(f"{model}: {len(sells)} sells, {len(buys)} buys (dry_run={a.dry_run})")

    svc = None
    if not a.dry_run:
        from src.services.brokers.fyers_service import FyersService
        svc = FyersService()
        if not svc.get_broker_config(a.user_id):
            log.error(f"No Fyers token for user {a.user_id} — abort"); return 1

    held = {h["symbol"]: h for h in get_holdings(model)}

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
            px = res.get("traded_px") or ltp
            record_sell_multi(model, sym, float(px), reason,
                              fyers_order_id=res.get("order_id"))
            log.info(f"  SOLD {qty}x{sym}@{px} ({reason})")
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
            if not a.dry_run:
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
                px = res.get("traded_px") or ltp
                record_buy_multi(model, sym, res.get("fill_qty") or qty, float(px),
                                 fyers_order_id=res.get("order_id"))
                log.info(f"  BOUGHT {qty}x{sym}@{px}")
            else:
                log.error(f"  BUY {sym} not filled: {res.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
