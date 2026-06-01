"""Tests for the multi-holding reconciler's pure helper.

`holding_status(expected_qty, fyers_qty, sibling_qty)` classifies ONE
model_holdings row against the broker net (after subtracting what sibling
models claim of the same symbol on the shared Fyers account):

  * ORPHAN — broker shows zero of this symbol (ledger holds, Fyers flat)
  * SHORT  — available (fyers - sibling) < expected (external/missed sell)
  * OK     — available >= expected

Regression it locks: reconcile_multi used to do ``fy = fyers.get(sym, 0)``
where ``fyers`` maps symbol -> {qty,...} (a DICT), then ``fy - sib`` blew up
with ``TypeError: unsupported operand type(s) for -: 'dict' and 'int'`` — so
the multi reconciler crashed on every run and retest_n500 drift was never
checked. The qty must be pulled out of the dict, and the compare lives in this
pure helper so it is unit-testable without Fyers/DB IO.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.live.position_reconciler_multi import holding_status


def test_ok_exact_match_no_siblings():
    assert holding_status(1053, 1053, 0) == ("OK", 1053)


def test_ok_broker_has_more_no_siblings():
    # Broker net exceeds this row (another unrecorded buy); still OK for THIS row.
    assert holding_status(36, 50, 0) == ("OK", 50)


def test_orphan_broker_flat():
    # Ledger holds, Fyers shows nothing -> ORPHAN regardless of siblings.
    assert holding_status(1053, 0, 0) == ("ORPHAN", 0)


def test_short_after_external_sell():
    # Broker has fewer than the row expects -> SHORT.
    assert holding_status(1053, 800, 0) == ("SHORT", 800)


def test_ok_after_sibling_subtract():
    # Shared account: 500 at broker, a sibling claims 200, this row expects 300.
    assert holding_status(300, 500, 200) == ("OK", 300)


def test_short_when_siblings_eat_the_net():
    # Broker 500, siblings claim 400, this row expects 300 -> only 100 left.
    assert holding_status(300, 500, 400) == ("SHORT", 100)
