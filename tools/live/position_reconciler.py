"""Mirror Fyers positions → model_ledger to catch drift.

Background: record_buy / record_sell in src/services/trading/model_ledger_service.py
keep ledger.cash + open_symbol/qty/entry_px in sync when fyers_executor
detects fills. When detection silently fails (status-mapping bug, exec crash,
external trade), ledger drifts from Fyers truth.

This reconciler:
  1. Pulls Fyers positions (single source of truth)
  2. For each enabled model with a known open_symbol, compares qty + avg_price
  3. Auto-mirrors Fyers → ledger when Fyers qty exceeds ledger qty (missed BUY)
  4. Alerts (no auto-fix) when Fyers qty under ledger qty (externally closed) —
     manual review needed because we don't know SELL price without Fyers tradebook
  5. Flags orphans (Fyers holds X, no ledger row claims X)

Cash invariant assumed after auto-fix:
  cash = invested_amount + realized_pnl - (new_open_qty * new_open_entry_px)
  (clamped >= 0; charges drift accepted as small)

Usage:
  python tools/live/position_reconciler.py [--dry-run] [--tg-on-fix]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("position_reconciler")


def _normalize(sym: str) -> str:
    """Match record_buy convention (uppercase, NSE:…-EQ form)."""
    if not sym:
        return ""
    s = sym.strip().upper()
    if not s.startswith("NSE:") and "-" not in s:
        s = f"NSE:{s}-EQ"
    return s


def sibling_qty_for(ledgers, model_name: str, symbol: str) -> int:
    """Sum open_qty across OTHER ledger rows holding the same (normalized) symbol.

    Used by the reconciler to attribute a merged Fyers holding back to each
    claiming model when the shared account has 2+ model_ledger rows on one
    symbol. record_buy / record_sell already stamp each model's own slice at
    fill time; this helper recovers the per-model expected qty by subtracting
    sibling claims from the broker net.

    Args:
        ledgers: iterable of ModelLedger-like rows (.model_name, .open_symbol,
            .open_qty). Disabled rows are still counted — what matters is
            whether the row claims open shares on the shared account.
        model_name: this row's model_name (excluded from the sum).
        symbol: target symbol; both sides are run through _normalize() before
            comparing so a ledger storing the plain "ADANIPOWER" form still
            collides with one storing the full "NSE:ADANIPOWER-EQ" form.

    Returns:
        Total sibling open_qty (0 if no siblings hold the same symbol).
    """
    target = _normalize(symbol)
    total = 0
    for L in ledgers:
        if L.model_name == model_name:
            continue
        if not L.open_symbol:
            continue
        if _normalize(L.open_symbol) == target:
            total += int(L.open_qty or 0)
    return total


def decide_drift(expected_qty: int, expected_px: float,
                 actual_qty: int, actual_px: float,
                 sibling_qty: int = 0, max_affordable_qty: int = None):
    """Pure: classify drift between ledger (expected) and broker (actual).

    When ``sibling_qty == 0`` this is the pre-overlap behaviour: qty + px both
    reconcile against the broker numbers (an extra fill / px drift triggers
    AUTO_MIRROR; an external partial sell triggers QTY_REDUCED).

    When ``sibling_qty > 0`` other ledger rows also claim the same symbol on
    the shared Fyers account, so the broker's net qty has to be RATIONED
    before compare: this model's "actual" is ``actual_qty - sibling_qty``.
    The broker's avg_price is a merge across multiple models' buys and is NOT
    this model's truth, so the helper refuses to write entry_px under overlap
    (``fix_px`` stays None — caller leaves the ledger's per-model entry_px
    alone, which keeps that model's cash math consistent).

    Args:
        expected_qty / expected_px: what THIS model_ledger row claims it owns.
        actual_qty / actual_px:     Fyers net across all models on this symbol.
        sibling_qty: qty other ledger rows claim on the same symbol; 0 if
            this model is the only claimant.

    Returns:
        (kind, fix_qty, fix_px). ``fix_qty`` / ``fix_px`` is the value to
        write into the ledger; ``None`` means "do not update that field".
        ``kind`` is one of:
          NO_DRIFT          — ledger matches broker (after sibling subtract).
          AUTO_MIRROR       — broker has more on our slice; mirror qty (and
                              px when no siblings).
          QTY_REDUCED       — broker has less on our slice; alert, no auto-fix
                              (we don't know the external sell price).
          SIBLING_OVERCLAIM — siblings claim more than the broker net; some
                              ledger lies. Don't auto-fix; surface for manual
                              cross-check.
    """
    def _cap(qty):
        """Refuse to AUTO_MIRROR a model UP to a qty it could NOT afford
        (cost > invested+realized). Such excess is another model's intraday
        position or unrecorded drift wrongly attributed here (the 2026-06-01
        HFCL double-buy that got dumped onto pseudo as 705). Alert instead."""
        if max_affordable_qty is not None and qty > max_affordable_qty:
            return ("MIRROR_CAP_EXCEEDED", None, None)
        return None

    if sibling_qty > 0:
        # Overlap path: ration broker qty between this model and its siblings.
        if sibling_qty > actual_qty:
            # Some ledger row is lying — auto-fixing here would write a
            # negative my_share. Surface and stop.
            return ("SIBLING_OVERCLAIM", None, None)
        my_share = actual_qty - sibling_qty
        if my_share == expected_qty:
            return ("NO_DRIFT", None, None)
        if my_share < expected_qty:
            return ("QTY_REDUCED", None, None)
        # AUTO_MIRROR but keep entry_px — broker avg is the cross-model blend.
        return _cap(my_share) or ("AUTO_MIRROR", my_share, None)
    # No overlap — reconcile qty AND px against the broker (legacy behaviour).
    if actual_qty == expected_qty and abs(actual_px - expected_px) < 0.01:
        return ("NO_DRIFT", None, None)
    if actual_qty < expected_qty:
        return ("QTY_REDUCED", None, None)
    return _cap(actual_qty) or ("AUTO_MIRROR", actual_qty, actual_px)


def _merge_pos(out: Dict[str, Dict], rows, source: str) -> None:
    """Merge Fyers position/holding rows into {symbol: {qty, avg_price,
    last_price, source}}. Same symbol in both: sum qty + weighted-avg price."""
    for p in rows or []:
        sym = _normalize(p.get("symbol") or "")
        if not sym:
            continue
        qty = int(float(p.get("quantity") or 0))
        if qty == 0:
            continue
        px = float(p.get("average_price") or 0)
        lp = float(p.get("last_price") or 0)
        if sym in out:
            prev_qty = out[sym]["qty"]
            prev_px = out[sym]["avg_price"]
            total_qty = prev_qty + qty
            new_px = ((prev_qty * prev_px) + (qty * px)) / total_qty if total_qty else 0
            out[sym]["qty"] = total_qty
            out[sym]["avg_price"] = new_px
            out[sym]["source"] = f"{out[sym]['source']}+{source}"
            if lp and not out[sym]["last_price"]:
                out[sym]["last_price"] = lp
        else:
            out[sym] = {"qty": qty, "avg_price": px, "last_price": lp,
                        "source": source}


def _fyers_positions_by_symbol(svc, user_id: int = 1):
    """Union of intraday positions + settled holdings. Both are real exposure
    and the model_ledger should match the sum.

    Background: CNC orders show in positions() on the day of purchase, then
    move to holdings() after T+1 settlement. Reconciler must check both or
    it will think the position vanished (false LEDGER_AHEAD alert).

    Returns (positives, negatives): positives = {sym: {qty>0,...}} real long
    holdings; negatives = {sym: net_qty<0} net-short symbols (settling sell or
    over-sell) surfaced for a distinct claimed-symbol alert.
    """
    out: Dict[str, Dict] = {}
    for fn_name, source in [("positions", "pos"), ("holdings", "hold")]:
        try:
            fn = getattr(svc, fn_name)
            res = fn(user_id=user_id)
            if not isinstance(res, dict) or res.get("status") != "success":
                log.warning(f"{fn_name}() returned non-success: {res}")
                continue
            _merge_pos(out, res.get("data", []), source)
        except Exception as e:
            log.error(f"fyers {fn_name} fetch failed: {e}")
    # Long-only CNC models never short. A net qty <= 0 is a SELL leg still
    # settling (a same-day CNC sell shows as negative in positions() until the
    # T+1 delivery nets it out) or a fluke — NOT a holding the ledger should
    # claim. Dropping these (after summing pos+hold so partial sells still net
    # correctly) prevents false FYERS_ORPHAN / LEDGER_AHEAD alerts, e.g. the
    # "-68 ADANIPOWER @ 247.90" orphan ping right after a rotation sell.
    # Split: positives = real long holdings (feed orphan/drift checks).
    # negatives (net < 0) = a settling same-day CNC sell OR a genuine over-sell;
    # surfaced separately so a LEDGER-CLAIMED symbol that nets negative gets a
    # distinct "possible external double-sell" alert instead of being silently
    # dropped (an UNCLAIMED negative stays ignored — it's a settling sell, not
    # an orphan). net == 0 = flat, ignored.
    negatives = {k: v["qty"] for k, v in out.items() if v["qty"] < 0}
    if negatives:
        log.info(f"reconciler: net-negative positions (settling sell or "
                 f"over-sell): {negatives}")
    positives = {k: v for k, v in out.items() if v["qty"] > 0}
    return positives, negatives


def reconcile_once(user_id: int = 1, dry_run: bool = False) -> List[Dict]:
    """One pass. Returns list of correction dicts."""
    from src.models.database import get_database_manager
    from src.models.model_ledger_models import ModelLedger, ModelSettings
    from src.services.brokers.fyers_service import FyersService

    svc = FyersService()
    fyers, fyers_neg = _fyers_positions_by_symbol(svc, user_id)

    db = get_database_manager()
    corrections: List[Dict] = []
    with db.get_session() as s:
        ledgers = s.query(ModelLedger).all()
        settings_map = {x.model_name: x for x in s.query(ModelSettings).all()}
        claimed_syms = set()

        for l in ledgers:
            cfg = settings_map.get(l.model_name)
            if not cfg or not cfg.enabled:
                continue

            expected_sym = _normalize(l.open_symbol or "")
            expected_qty = int(l.open_qty or 0)
            expected_px = float(l.open_entry_px or 0)

            if not expected_sym:
                continue  # ledger flat — orphan check below

            claimed_syms.add(expected_sym)
            actual = fyers.get(expected_sym)

            if expected_sym in fyers_neg:
                # Ledger holds it, but Fyers nets NEGATIVE — an over-sell beyond
                # what we hold (possible external/duplicate SELL). Distinct from
                # plain LEDGER_AHEAD (flat): the broker shows a short position.
                corrections.append({
                    "model": l.model_name,
                    "type": "FYERS_NET_NEGATIVE",
                    "before": f"{expected_sym} x{expected_qty} @ {expected_px:.2f}",
                    "after": f"Fyers NET SHORT x{fyers_neg[expected_sym]} "
                             f"(possible external/double SELL)",
                    "action": "MANUAL: check Fyers tradebook — over-sold vs ledger",
                })
                continue

            if not actual:
                # Ledger thinks holding, Fyers shows nothing. External SELL?
                # Don't auto-clear (would lose realized PnL). Just alert.
                corrections.append({
                    "model": l.model_name,
                    "type": "LEDGER_AHEAD",
                    "before": f"{expected_sym} x{expected_qty} @ {expected_px:.2f}",
                    "after": "Fyers shows no position",
                    "action": "MANUAL: check Fyers tradebook, run record_sell",
                })
                continue

            actual_qty = actual["qty"]
            actual_px = actual["avg_price"]
            # Cross-model attribution: if a sibling ledger also claims this
            # symbol on the shared Fyers account, ration the broker net before
            # comparing. Without this, two models holding the same name would
            # each AUTO_MIRROR the merged total into their own row (each
            # claiming 2x what it actually owns) on every reconcile pass.
            sibling_qty = sibling_qty_for(ledgers, l.model_name, expected_sym)

            # Affordability cap: the most shares this model could OWN is
            # (invested + realized) / its entry price. The reconciler must never
            # AUTO_MIRROR it above that — excess is another model's intraday
            # position / unrecorded drift (the HFCL double-buy dumped onto
            # pseudo as 705 vs its ~413 cap). Use the broker avg as the price
            # basis (the buy that created the excess), falling back to entry_px.
            _cap_px = actual_px if actual_px and actual_px > 0 else expected_px
            _invested = float(cfg.invested_amount or 0) + float(l.realized_pnl or 0)
            max_affordable = int(_invested / _cap_px) if _cap_px > 0 else None

            kind, fix_qty, fix_px = decide_drift(
                expected_qty, expected_px, actual_qty, actual_px, sibling_qty,
                max_affordable_qty=max_affordable,
            )

            if kind == "NO_DRIFT":
                continue

            if kind == "MIRROR_CAP_EXCEEDED":
                corrections.append({
                    "model": l.model_name,
                    "type": "MIRROR_CAP_EXCEEDED",
                    "before": f"{expected_sym} x{expected_qty} @ {expected_px:.2f}",
                    "after": (f"broker slice x{actual_qty - sibling_qty} > max "
                              f"affordable x{max_affordable} (cap ₹{_invested:,.0f} "
                              f"@ ₹{_cap_px:.2f}) — NOT mirrored (likely another "
                              f"model's position / unrecorded drift)"),
                    "action": "MANUAL: a buy on this symbol wasn't recorded to the "
                              "right model; check Fyers tradebook + record_buy/sell",
                })
                continue

            if kind == "QTY_REDUCED":
                slice_after = (f"my slice x{actual_qty - sibling_qty} of broker "
                               f"x{actual_qty} (sibling claim x{sibling_qty})"
                               if sibling_qty
                               else f"{expected_sym} x{actual_qty} @ {actual_px:.2f}")
                corrections.append({
                    "model": l.model_name,
                    "type": "QTY_REDUCED",
                    "before": f"{expected_sym} x{expected_qty} @ {expected_px:.2f}",
                    "after": slice_after,
                    "action": "MANUAL: partial external SELL, run record_sell for diff",
                })
                continue

            if kind == "SIBLING_OVERCLAIM":
                # Build the full claimant table so the alert points at the
                # offending row, not just "some ledger lies". One reconciler
                # pass on an overclaim hits every claimant model with the
                # same alert — including the breakdown lets the operator
                # spot which row's open_qty exceeds its share of the broker
                # net.
                claimants = []
                for sib in ledgers:
                    if not sib.open_symbol:
                        continue
                    if _normalize(sib.open_symbol) != expected_sym:
                        continue
                    claimants.append(
                        f"{sib.model_name}={int(sib.open_qty or 0)}"
                    )
                breakdown = ", ".join(claimants) if claimants else "(none)"
                corrections.append({
                    "model": l.model_name,
                    "type": "SIBLING_OVERCLAIM",
                    "before": (f"{expected_sym} x{expected_qty} "
                               f"(siblings claim x{sibling_qty})"),
                    "after": (f"broker net only x{actual_qty} — claimants: "
                              f"[{breakdown}], total x{sibling_qty + expected_qty}"),
                    "action": "MANUAL: a ledger row's open_qty exceeds its "
                              "share of the broker net; cross-check Fyers "
                              "tradebook + run record_sell on the stale row",
                })
                continue

            # AUTO_MIRROR path. Under overlap fix_px is None and we keep the
            # ledger's own per-model entry_px (broker avg is a cross-model
            # blend and would corrupt this model's P&L if written here).
            new_qty = fix_qty
            effective_px = fix_px if fix_px is not None else expected_px
            invested = float(cfg.invested_amount or 0)
            realized = float(l.realized_pnl or 0)
            new_cost = new_qty * effective_px
            new_cash = max(0.0, invested + realized - new_cost)

            overlap_note = (f" (overlap: kept entry_px, sibling x{sibling_qty})"
                            if sibling_qty else "")
            corrections.append({
                "model": l.model_name,
                "type": "AUTO_MIRROR",
                "before": (f"{expected_sym} x{expected_qty} @ {expected_px:.2f} "
                           f"cash=₹{float(l.cash or 0):.2f}"),
                "after": (f"{expected_sym} x{new_qty} @ {effective_px:.2f} "
                          f"cash=₹{new_cash:.2f}{overlap_note}"),
                "action": "applied" if not dry_run else "would-apply",
            })

            if not dry_run:
                l.open_symbol = expected_sym
                l.open_qty = new_qty
                if fix_px is not None:
                    l.open_entry_px = Decimal(str(round(fix_px, 4)))
                if not l.open_entry_date:
                    l.open_entry_date = date.today()
                l.cash = Decimal(str(round(new_cash, 2)))
                l.updated_at = datetime.now()

        # Orphan detection: Fyers holds symbol no enabled-ledger claims
        for sym, pos in fyers.items():
            if sym in claimed_syms:
                continue
            corrections.append({
                "model": "?",
                "type": "FYERS_ORPHAN",
                "before": "no ledger row claims it",
                "after": f"{sym} x{pos['qty']} @ {pos['avg_price']:.2f}",
                "action": "MANUAL: assign to a model or sell",
            })

        if not dry_run:
            s.commit()

    return corrections


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tg-on-fix", action="store_true",
                    help="Send Telegram alert when corrections happen")
    ap.add_argument("--user-id", type=int, default=1)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    corrections = reconcile_once(user_id=args.user_id, dry_run=args.dry_run)

    if not corrections:
        log.info("Reconcile: no drift")
        return 0

    log.warning(f"Reconcile: {len(corrections)} item(s)"
                f"{'  (dry-run)' if args.dry_run else ''}")
    for c in corrections:
        log.warning(f"  [{c['type']}] {c['model']}: {c['before']} → {c['after']}"
                    f" ({c['action']})")

    if args.tg_on_fix:
        try:
            from tools.live.telegram_notify import send
            # Per-kind glyph to make the digest scannable on mobile. New
            # cross-model SIBLING_OVERCLAIM gets a dedicated red-flag icon so
            # it stands apart from the routine AUTO_MIRROR rows.
            _icon = {
                "AUTO_MIRROR":        "🔄",
                "QTY_REDUCED":        "⚠️",
                "LEDGER_AHEAD":       "❓",
                "FYERS_NET_NEGATIVE": "🛑",
                "SIBLING_OVERCLAIM":  "🚩",
                "FYERS_ORPHAN":       "👻",
            }
            lines = [f"*Ledger reconciler* — {len(corrections)} item(s)"]
            for c in corrections[:8]:
                glyph = _icon.get(c["type"], "•")
                lines.append(f"{glyph} `{c['type']}` `{c['model']}`")
                lines.append(f"  was: {c['before']}")
                lines.append(f"  now: {c['after']}")
                lines.append(f"  → {c['action']}")
            send("\n".join(lines))
        except Exception as e:
            log.error(f"TG alert failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
