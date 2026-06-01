"""Tests for the model-aware position reconciler.

Locks two pure helpers that decouple the drift decision from Fyers/DB IO so it
can be unit-tested:

  * `decide_drift(expected_qty, expected_px, actual_qty, actual_px, sibling_qty)`
    returns one of {NO_DRIFT, AUTO_MIRROR, QTY_REDUCED, SIBLING_OVERCLAIM} plus
    the corrected qty/px to write (or None when "no action needed" / "don't
    touch under overlap"). When `sibling_qty>0` (other model_ledger rows hold
    the same symbol on the shared Fyers account), the helper compares the
    ledger's expected_qty against the broker's net qty MINUS sibling claims
    instead of the raw net, and refuses to overwrite entry_px (the broker's
    blended avg is no longer THIS model's truth).

  * `sibling_qty_for(ledgers, model_name, symbol)` sums `open_qty` across
    OTHER ledger rows holding the same (normalized) symbol — the lookup the
    reconciler does once per ledger row before calling `decide_drift`.

Background — the bug these helpers fix: when two models (e.g. n100 + n20)
both hold ADANIPOWER on ONE shared Fyers account, the broker reports a single
merged position with qty = sum of both. The original reconciler compared each
model's ledger.qty against that merged total, saw `actual > expected`, and
AUTO_MIRRORed the merged qty into BOTH ledgers, so each thought it owned 2x
the position. Sibling-subtraction restores per-model attribution that
record_buy / record_sell already wrote correctly at fill time.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.live.position_reconciler import decide_drift, sibling_qty_for


# ---------------- decide_drift (pure) ----------------

def test_no_drift_no_siblings():
    assert decide_drift(100, 10.0, 100, 10.0, 0) == ("NO_DRIFT", None, None)


def test_auto_mirror_extra_fill_no_siblings():
    kind, fix_qty, fix_px = decide_drift(100, 10.0, 110, 10.0, 0)
    assert kind == "AUTO_MIRROR"
    assert fix_qty == 110
    assert fix_px == 10.0


def test_auto_mirror_px_drift_only_no_siblings():
    # Qty matches but Fyers avg drifted (e.g. corporate-action recalc) -> still mirror.
    kind, fix_qty, fix_px = decide_drift(100, 10.0, 100, 10.5, 0)
    assert kind == "AUTO_MIRROR"
    assert fix_qty == 100
    assert fix_px == 10.5


def test_qty_reduced_no_siblings():
    # External partial SELL -> alert, do not auto-fix.
    kind, fix_qty, fix_px = decide_drift(100, 10.0, 80, 10.0, 0)
    assert kind == "QTY_REDUCED"
    assert fix_qty is None and fix_px is None


# ---------------- decide_drift under cross-model overlap ----------------

def test_overlap_no_drift_after_sibling_subtract_n100_pov():
    # Live state captured 2026-05-28: n100 owns 49806 ADANIPOWER + n20 owns
    # 65141 -> broker reports a merged 114947. From n100's POV the sibling
    # (n20) claims 65141, so my slice is 49806 -> matches expected -> NO_DRIFT.
    kind, fix_qty, fix_px = decide_drift(49806, 157.11, 114947, 175.20, 65141)
    assert kind == "NO_DRIFT"
    assert fix_qty is None and fix_px is None


def test_overlap_no_drift_after_sibling_subtract_n20_pov():
    # Same broker state, n20's POV: sibling (n100) claims 49806, my slice 65141.
    kind, fix_qty, fix_px = decide_drift(65141, 181.35, 114947, 175.20, 49806)
    assert kind == "NO_DRIFT"


def test_overlap_extra_fill_mirrors_only_my_slice_and_keeps_px():
    # Broker now shows 120000 (some extra fill landed on me). Sibling still 65141.
    # My slice = 120000 - 65141 = 54859 > expected 49806 -> AUTO_MIRROR to 54859.
    # Entry_px MUST stay untouched: broker px is merged across both buys, so
    # writing it onto a single model's ledger would corrupt that model's P&L.
    kind, fix_qty, fix_px = decide_drift(49806, 157.11, 120000, 175.20, 65141)
    assert kind == "AUTO_MIRROR"
    assert fix_qty == 54859
    assert fix_px is None  # leave the ledger's true per-model entry_px alone


def test_overlap_partial_external_sell_alerts():
    # Broker dropped to 100000 (someone closed part on Fyers UI). Sibling 65141.
    # My slice = 34859 < expected 49806 -> QTY_REDUCED alert (manual review).
    kind, fix_qty, fix_px = decide_drift(49806, 157.11, 100000, 175.20, 65141)
    assert kind == "QTY_REDUCED"
    assert fix_qty is None


def test_sibling_overclaim_alerts_does_not_auto_fix():
    # Some ledger lies: siblings claim 65141 but broker only shows 30000 total.
    # Refuse to AUTO_MIRROR (could write a NEGATIVE my_share) -> surface as alert.
    kind, fix_qty, fix_px = decide_drift(49806, 157.11, 30000, 175.20, 65141)
    assert kind == "SIBLING_OVERCLAIM"
    assert fix_qty is None and fix_px is None


# ---------------- sibling_qty_for (pure) ----------------

class _L:
    """Minimal stand-in for a ModelLedger row (only the fields the helper reads)."""
    def __init__(self, model_name, open_symbol=None, open_qty=0):
        self.model_name = model_name
        self.open_symbol = open_symbol
        self.open_qty = open_qty


def test_sibling_qty_for_two_overlapping_models():
    ledgers = [
        _L("momentum_n100_top5_max1", "NSE:ADANIPOWER-EQ", 49806),
        _L("n20_daily_large_only",    "NSE:ADANIPOWER-EQ", 65141),
        _L("momentum_pseudo_n100_adv", "NSE:ADANIGREEN-EQ", 11743),
    ]
    assert sibling_qty_for(ledgers, "momentum_n100_top5_max1",
                           "NSE:ADANIPOWER-EQ") == 65141
    assert sibling_qty_for(ledgers, "n20_daily_large_only",
                           "NSE:ADANIPOWER-EQ") == 49806


def test_sibling_qty_for_three_way_overlap():
    ledgers = [_L("a", "NSE:X-EQ", 10), _L("b", "NSE:X-EQ", 20), _L("c", "NSE:X-EQ", 30)]
    assert sibling_qty_for(ledgers, "a", "NSE:X-EQ") == 50
    assert sibling_qty_for(ledgers, "b", "NSE:X-EQ") == 40
    assert sibling_qty_for(ledgers, "c", "NSE:X-EQ") == 30


def test_sibling_qty_for_ignores_own_row():
    ledgers = [_L("n100", "NSE:RELIANCE-EQ", 10)]
    assert sibling_qty_for(ledgers, "n100", "NSE:RELIANCE-EQ") == 0


def test_sibling_qty_for_normalizes_plain_symbol():
    # Live ledgers may store plain "ADANIPOWER" or the full "NSE:ADANIPOWER-EQ" —
    # the helper must normalize both sides before comparing.
    ledgers = [
        _L("n100", "ADANIPOWER", 49806),
        _L("n20",  "NSE:ADANIPOWER-EQ", 65141),
    ]
    assert sibling_qty_for(ledgers, "pseudo", "ADANIPOWER") == 49806 + 65141


def test_sibling_qty_for_skips_flat_and_different_symbols():
    ledgers = [
        _L("a", None, 0),                       # flat ledger
        _L("b", "NSE:OTHER-EQ", 99),            # different symbol
        _L("c", "NSE:TARGET-EQ", 50),
    ]
    assert sibling_qty_for(ledgers, "z", "NSE:TARGET-EQ") == 50


# ---------------- decide_drift affordability cap (2026-06-01) ----------------

def test_decide_drift_caps_mirror_above_affordable():
    from tools.live.position_reconciler import decide_drift
    # pseudo: owns 203, broker shows 705 (sibling 0), could only afford 413 ->
    # refuse to mirror (the HFCL double-buy dumped onto it). Alert, no write.
    assert decide_drift(203, 145.46, 705, 169.53, 0,
                        max_affordable_qty=413) == ("MIRROR_CAP_EXCEEDED", None, None)


def test_decide_drift_mirrors_within_affordable():
    from tools.live.position_reconciler import decide_drift
    # genuine missed buy within cap -> still AUTO_MIRROR
    kind, q, px = decide_drift(100, 145.0, 250, 145.0, 0, max_affordable_qty=413)
    assert kind == "AUTO_MIRROR" and q == 250


def test_decide_drift_cap_none_is_legacy():
    from tools.live.position_reconciler import decide_drift
    # no cap passed -> unchanged behaviour
    kind, q, _ = decide_drift(100, 10.0, 705, 10.0, 0)
    assert kind == "AUTO_MIRROR" and q == 705


# ---------------- multi_claimed_symbols (pure) — 2026-06-01 ----------------
# Bug: the single-position reconciler built `claimed_syms` from model_ledger
# .open_symbol only, so a MULTI-holding model (momentum_retest_n500, positions
# live in model_holdings) had its real, recorded holdings (BHEL, IDEA) flagged
# as FYERS_ORPHAN every pass. multi_claimed_symbols folds the model_holdings
# symbols into the claimed set so they stop looking like orphans.

class _H:
    """Minimal stand-in for a ModelHolding row."""
    def __init__(self, model_name, symbol, qty=0):
        self.model_name = model_name
        self.symbol = symbol
        self.qty = qty


def test_multi_claimed_symbols_normalizes_and_dedups():
    from tools.live.position_reconciler import multi_claimed_symbols
    holdings = [
        _H("momentum_retest_n500", "NSE:BHEL-EQ", 36),
        _H("momentum_retest_n500", "IDEA", 1053),      # plain form -> normalized
    ]
    assert multi_claimed_symbols(holdings) == {"NSE:BHEL-EQ", "NSE:IDEA-EQ"}


def test_multi_claimed_symbols_skips_zero_and_blank():
    from tools.live.position_reconciler import multi_claimed_symbols
    holdings = [
        _H("m", "NSE:X-EQ", 0),    # zero qty -> not a real claim
        _H("m", "", 50),           # blank symbol -> skip
        _H("m", "NSE:Y-EQ", 10),
    ]
    assert multi_claimed_symbols(holdings) == {"NSE:Y-EQ"}


def test_multi_claimed_symbols_empty():
    from tools.live.position_reconciler import multi_claimed_symbols
    assert multi_claimed_symbols([]) == set()


# ---------------- corrections_signature (pure) — alert de-dup 2026-06-01 ----
# The reconciler runs every few minutes 09:45-15:30 with --tg-on-fix. A KNOWN
# drift (e.g. the HFCL double-fill duplicate awaiting a manual sell) re-fired
# the SAME Telegram alert every cycle = spam. corrections_signature lets main()
# send only when the drift set CHANGES from the last alerted state.

def test_corrections_signature_empty_is_blank():
    from tools.live.position_reconciler import corrections_signature
    assert corrections_signature([]) == ""
    assert corrections_signature(None) == ""


def test_corrections_signature_order_independent():
    from tools.live.position_reconciler import corrections_signature
    a = [{"type": "MIRROR_CAP_EXCEEDED", "model": "emerging_momentum", "before": "HFCL x251 @ 179.75"},
         {"type": "MIRROR_CAP_EXCEEDED", "model": "momentum_pseudo_n100_adv", "before": "HFCL x203 @ 145.46"}]
    assert corrections_signature(a) == corrections_signature(list(reversed(a)))


def test_corrections_signature_changes_on_content():
    from tools.live.position_reconciler import corrections_signature
    a = [{"type": "MIRROR_CAP_EXCEEDED", "model": "emerging_momentum", "before": "HFCL x251 @ 179.75"}]
    b = [{"type": "MIRROR_CAP_EXCEEDED", "model": "emerging_momentum", "before": "HFCL x100 @ 179.75"}]
    assert corrections_signature(a) != corrections_signature(b)
    # adding a 2nd drift changes the signature too
    assert corrections_signature(a) != corrections_signature(a + b)
