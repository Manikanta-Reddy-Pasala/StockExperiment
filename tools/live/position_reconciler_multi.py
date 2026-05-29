"""Multi-holding reconciler — mirror Fyers truth → model_holdings (K>1 models).

The single-position reconciler (position_reconciler.py) skips multi-holding
models (their ledger.open_symbol is NULL — positions live in model_holdings).
This reconciles each model_holdings row against Fyers, on the SHARED account,
subtracting what SIBLING models (single-position ledgers + other multi-holdings)
claim of the same symbol.

ALERT-ONLY (no auto-fix) — a new multi-holding model should surface drift for a
human, not silently mutate the live ledger. Reuses position_reconciler's
_fyers_positions_by_symbol + _normalize so broker-side parsing is identical.

Usage: python tools/live/position_reconciler_multi.py [--model momentum_retest_n500] [--dry-run]
"""
import sys, argparse, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.live.position_reconciler import _fyers_positions_by_symbol, _normalize

log = logging.getLogger("reconciler_multi")
DEFAULT_MODEL = "momentum_retest_n500"


def _sibling_claims(sym, this_model, single_ledgers, all_holdings):
    """Qty of `sym` claimed by OTHER models on the shared Fyers account.

    single_ledgers: ModelLedger rows (single-position).  all_holdings: ModelHolding
    rows (multi). Excludes this_model's own holding of `sym` (that's the row under
    test).
    """
    n = 0
    for L in single_ledgers:
        if L.open_symbol and _normalize(L.open_symbol) == sym:
            n += int(L.open_qty or 0)
    for h in all_holdings:
        if h.model_name != this_model and _normalize(h.symbol) == sym:
            n += int(h.qty or 0)
    return n


def reconcile_multi(model_name=DEFAULT_MODEL, user_id=1, dry_run=False):
    from src.models.database import get_database_manager
    from src.models.model_ledger_models import ModelLedger, ModelSettings, ModelHolding
    from src.services.brokers.fyers_service import FyersService

    svc = FyersService()
    fyers, _neg = _fyers_positions_by_symbol(svc, user_id)
    db = get_database_manager()
    alerts = []
    with db.get_session() as s:
        cfg = s.query(ModelSettings).filter_by(model_name=model_name).first()
        if not cfg or not cfg.enabled:
            log.info(f"{model_name} disabled/missing — skip multi-reconcile")
            return []
        single_ledgers = s.query(ModelLedger).all()
        all_holdings = s.query(ModelHolding).all()
        my = [h for h in all_holdings if h.model_name == model_name]
        for h in my:
            sym = _normalize(h.symbol); exp = int(h.qty or 0)
            fy = fyers.get(sym, 0)
            sib = _sibling_claims(sym, model_name, single_ledgers, all_holdings)
            avail = fy - sib
            if avail < exp:                                  # Fyers short of ledger
                alerts.append({"model": model_name, "symbol": sym,
                               "ledger_qty": exp, "fyers_qty": fy, "sibling": sib,
                               "available": avail, "drift": exp - avail,
                               "action": "MANUAL: external sell / missed record_sell — verify Fyers"})
        # orphan: ledger says we hold N positions but Fyers has none of a symbol
        for h in my:
            if fyers.get(_normalize(h.symbol), 0) == 0:
                alerts.append({"model": model_name, "symbol": _normalize(h.symbol),
                               "ledger_qty": int(h.qty or 0), "fyers_qty": 0,
                               "action": "MANUAL: ORPHAN — ledger holds, Fyers flat"})
    for a in alerts:
        log.warning(f"DRIFT {a['model']} {a['symbol']}: {a}")
    if alerts and not dry_run:
        try:
            from tools.live.telegram_notify import send as _tg
            _tg(f"⚠️ {model_name} multi-holding drift: {len(alerts)} issue(s)\n"
                + "\n".join(f"• {a['symbol']}: ledger {a['ledger_qty']} vs Fyers {a['fyers_qty']}" for a in alerts[:6]))
        except Exception as e:
            log.debug(f"tg failed: {e}")
    if not alerts:
        log.info(f"{model_name}: holdings reconciled clean vs Fyers")
    return alerts


def _all_multi_models():
    """Distinct model_names that have any row in model_holdings (the multi-holding
    models). Falls back to the known set if the table is empty/unreachable."""
    # emerging_momentum was rebuilt 2026-05-30 as a SINGLE-position rotation
    # model (max-1, uses model_ledger.open_symbol via fyers_executor.py), so it
    # is no longer reconciled here — it is handled by the single-position
    # reconciler. Only momentum_retest_n500 remains multi-holding.
    known = [DEFAULT_MODEL]
    try:
        from src.models.database import get_database_manager
        from src.models.model_ledger_models import ModelHolding
        db = get_database_manager()
        with db.get_session() as s:
            rows = s.query(ModelHolding.model_name).distinct().all()
        found = [r[0] for r in rows]
        # union: anything in the table + known models (so a disabled-but-registered
        # model is still checked once it has holdings)
        return sorted(set(found) | set(known)) if found else known
    except Exception as e:
        log.debug(f"model discovery failed, using known set: {e}")
        return known


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="reconcile one model; default = all multi-holding models")
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    models = [a.model] if a.model else _all_multi_models()
    for m in models:
        reconcile_multi(m, a.user_id, a.dry_run)
