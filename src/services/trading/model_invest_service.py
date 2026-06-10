"""Per-model 'Invest more': pure policy/sizing core (no I/O).

Suggests how to deploy a model's idle sleeve cash into its OWN current picks:
single-position models top up rank-1; multi-slot (Retest) fills empty slots.
Routes in momrot_routes.py supply the live numbers (ledger cash, broker cash,
ranking targets + LTP, open symbols) and place/record the orders.
"""
from __future__ import annotations

import hashlib
import json as _json
from datetime import datetime, time as _time
from typing import Dict, List, Set

CASH_BUFFER = 0.997  # 0.3% headroom for brokerage/STT/GST (delivery CNC ~0.1-0.15%)

try:
    from tools.shared.nse_calendar import is_trading_day
except Exception:  # pragma: no cover - import shim for bare test envs
    def is_trading_day(d=None):
        return True

_OPEN = _time(9, 15)
_CLOSE = _time(15, 30)


def compute_buys(idle_cash: float, broker_cash: float, max_holdings: int,
                 targets: List[Dict], open_symbols: Set[str]) -> List[Dict]:
    """Return a list of {symbol, ltp, qty, amount} buys, total <= deployable.

    idle_cash    : model's uninvested ledger cash
    broker_cash  : Fyers available_cash
    max_holdings : model's slot count (1 = single position)
    targets      : model's current picks, ordered best-first, each {symbol, ltp}
    open_symbols : bare symbols the model already holds (skip filled slots,
                   except a single-position model may top up its held rank-1)
    """
    deployable = min(max(0.0, idle_cash), max(0.0, broker_cash))
    if deployable <= 0 or not targets:
        return []

    if max_holdings <= 1:
        # single position: deploy all into rank-1 (top up even if already held)
        slots = targets[:1]
    else:
        # multi-slot: fill ONLY the FREE slots (max_holdings - already-held), so
        # total holdings can never exceed max_holdings. On a rotation (a held
        # name dropped out of the ranking) "invest more" must NOT add a new name
        # on top of a full book — that's a SELL+BUY rotation, the rebalance
        # job's responsibility, not an idle-cash top-up.
        free = max_holdings - len(open_symbols)
        if free <= 0:
            return []
        slots = [t for t in targets if t["symbol"] not in open_symbols][:free]
    if not slots:
        return []

    per_slot = (deployable * CASH_BUFFER) / len(slots)
    buys: List[Dict] = []
    for t in slots:
        ltp = float(t["ltp"] or 0)
        qty = int(per_slot // ltp) if ltp > 0 else 0
        if qty < 1:
            continue
        buys.append({"symbol": t["symbol"], "ltp": ltp,
                     "qty": qty, "amount": round(qty * ltp, 2)})
    return buys


def is_market_open(now: datetime) -> bool:
    """True only on an NSE trading day, 09:15..15:30 IST. `now` must be IST."""
    if not is_trading_day(now.date()):
        return False
    return _OPEN <= now.time() <= _CLOSE


def make_token(model_name: str, buys: list, day: str) -> str:
    """Deterministic idempotency token = hash(model, day, symbol+qty list)."""
    payload = _json.dumps(
        {"m": model_name, "d": day,
         "b": sorted((b["symbol"], int(b["qty"])) for b in buys)},
        sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
