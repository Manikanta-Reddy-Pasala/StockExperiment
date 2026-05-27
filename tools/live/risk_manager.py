"""Risk manager — enforces capital lock, max concurrent, per-trade cap.

Pure Python. Used by fyers_executor before placing any
order. Reads current portfolio state, decides if entry is allowed.

Constraints (configurable via env or YAML):
  - CAPITAL_INR: locked-in pool (no add/withdraw)
  - MAX_CONCURRENT: simultaneous open positions cap
  - MAX_PER_TRADE_INR: per-position size cap
  - MIN_PRICE: penny filter
  - MIN_ADV_INR: liquidity filter (skip illiquid)

Usage:
  rm = RiskManager(capital=200000, max_concurrent=2)
  if rm.can_enter(symbol, price, side, open_positions):
      qty = rm.size_position(price, open_positions)
      place_order(...)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

log = logging.getLogger("risk_manager")


def _env_model(model_name, key, default, cast=float):
    """Per-model env override with global fallback. Looks up
    ``<MODEL>_<KEY>`` first (model upper-cased, '-' -> '_'), then the global
    ``<KEY>``, else ``default``. Lets a thin midcap use a wider slippage/LIMIT
    band than a liquid n100 without a DB schema change — set e.g.
    ``MIDCAP_NARROW_60D_BREAKOUT_MAX_SLIPPAGE_PCT=3`` in .env.
    """
    mn = (model_name or "").upper().replace("-", "_")
    for k in ((f"{mn}_{key}" if mn else None), key):
        if k and k in os.environ:
            try:
                return cast(os.environ[k])
            except (TypeError, ValueError):
                pass
    return default


@dataclass
class RiskConfig:
    capital_inr: int = 1_000_000        # ₹10L locked-in pool
    max_concurrent: int = 2
    max_per_trade_inr: int = 500_000    # capital / max_concurrent
    min_price: float = 50.0
    enable_short: bool = False
    # LIMIT-with-tolerance + MARKET-fallback knobs (Phase 2 Task 6).
    # Per-model overridable via env: <MODEL>_LIMIT_TOL_PCT etc (see _env_model);
    # falls back to the global env knob, then these defaults.
    limit_tol_pct: float = 0.1          # initial LIMIT slack vs last price (standard)
    limit_retry_pct: float = 0.2        # widened slack on first re-quote (standard)
    limit_fallback_s: int = 20          # total seconds before MARKET fallback
    max_slippage_pct: float = 2.0       # abort MARKET fallback if LTP moved > this
    # Hard ceiling: total BUY exposure must never exceed allocated capital +
    # cumulative realized P&L. Set per-model in from_model(); kept at -1 for
    # env-based callers (treated as 'no cap' for backward-compat).
    max_total_buy_inr: int = -1


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    side: str = "BUY"
    sl: float = 0.0
    target: float = 0.0


class RiskManager:
    def __init__(self, config: Optional[RiskConfig] = None,
                 capital_is_net_cash: bool = False):
        self.cfg = config or RiskConfig.from_env()
        # When True, cfg.capital_inr was sourced from live NET cash (from_model:
        # ledger.cash), which is ALREADY debited for any open position at
        # record_buy. In that case size_position must NOT subtract the open
        # position value again (that double-counts and starves a 2nd slot when
        # max_concurrent>1). The env path uses GROSS capital, so it keeps the
        # original behaviour (subtract used).
        self._capital_is_net_cash = capital_is_net_cash

    @classmethod
    def from_env(cls):
        cfg = RiskConfig(
            capital_inr=int(os.environ.get("CAPITAL_INR", 1_000_000)),
            max_concurrent=int(os.environ.get("MAX_CONCURRENT", 2)),
            max_per_trade_inr=int(os.environ.get("MAX_PER_TRADE_INR",
                int(os.environ.get("CAPITAL_INR", 1_000_000)) //
                int(os.environ.get("MAX_CONCURRENT", 2)))),
            min_price=float(os.environ.get("MIN_PRICE", 50.0)),
            enable_short=os.environ.get("ENABLE_SHORT", "false").lower() == "true",
            limit_tol_pct=float(os.environ.get("LIMIT_TOL_PCT", 0.1)),
            limit_retry_pct=float(os.environ.get("LIMIT_RETRY_PCT", 0.2)),
            limit_fallback_s=int(os.environ.get("LIMIT_FALLBACK_S", 20)),
            max_slippage_pct=float(os.environ.get("MAX_SLIPPAGE_PCT", 2.0)),
        )
        return cls(cfg)

    @classmethod
    def from_model(cls, model_name: str):
        """Build RiskManager using per-model AVAILABLE CASH as capital.

        Reads `model_ledger.cash` (idle, un-invested) — the truthful amount
        currently spendable on a new BUY. Previously used model_settings.
        current_amount, which is a stale NAV cache; if a position is
        already open, NAV includes position MTM that is *not* available.

        Concrete bug avoided: today's manual rebalance clicks on n20 all
        saw current_amount=30000 and each placed a fresh 67-share BUY.
        Sizing off live cash makes the 2nd/3rd clicks return qty=0 because
        cash dropped to ₹471 after the first BUY.

        Fallback rules:
          - Unknown model OR DB error → fall back to from_env(), log WARNING.
          - cash <= 0 → capital floor = 0; size_position returns 0.
        """
        try:
            from src.models.database import get_database_manager
            from src.models.model_ledger_models import ModelLedger, ModelSettings
            db = get_database_manager()
            with db.get_session() as s:
                ledger = s.query(ModelLedger).filter_by(model_name=model_name).first()
                settings = s.query(ModelSettings).filter_by(model_name=model_name).first()
                if not ledger:
                    log.warning(
                        "RiskManager.from_model('%s'): no model_ledger row, "
                        "falling back to from_env()", model_name,
                    )
                    return cls.from_env()
                capital = int(float(ledger.cash or 0))
                # Hard ceiling: allocated + cumulative realized P&L. Any BUY
                # whose value pushes us above this is blocked even if cash
                # somehow over-states (concurrency bug, manual DB tweak, etc.).
                allocated = float(settings.invested_amount or 0) if settings else 0.0
                realized = float(ledger.realized_pnl or 0)
                max_total_buy = int(allocated + realized)
        except Exception as e:
            log.warning(
                "RiskManager.from_model('%s') failed (%s), falling back to from_env()",
                model_name, e,
            )
            return cls.from_env()

        # Other risk knobs from env (overridable per deployment)
        max_concurrent = int(os.environ.get("MAX_CONCURRENT", 2))
        env_per_trade = os.environ.get("MAX_PER_TRADE_INR")
        max_per_trade = (
            int(env_per_trade) if env_per_trade
            else max(1, capital // max(1, max_concurrent))
        )
        cfg = RiskConfig(
            capital_inr=capital,
            max_concurrent=max_concurrent,
            max_per_trade_inr=max_per_trade,
            min_price=float(os.environ.get("MIN_PRICE", 50.0)),
            enable_short=os.environ.get("ENABLE_SHORT", "false").lower() == "true",
            limit_tol_pct=_env_model(model_name, "LIMIT_TOL_PCT", 0.1),
            limit_retry_pct=_env_model(model_name, "LIMIT_RETRY_PCT", 0.2),
            limit_fallback_s=_env_model(model_name, "LIMIT_FALLBACK_S", 20, int),
            max_slippage_pct=_env_model(model_name, "MAX_SLIPPAGE_PCT", 2.0),
            max_total_buy_inr=max_total_buy,
        )
        log.info(
            "RiskManager.from_model('%s'): capital=₹%s max_concurrent=%d "
            "max_per_trade=₹%s max_total_buy=₹%s (allocated+realized)",
            model_name, f"{capital:,}", max_concurrent,
            f"{max_per_trade:,}", f"{max_total_buy:,}",
        )
        # capital came from live NET cash → don't re-subtract open positions.
        return cls(cfg, capital_is_net_cash=True)

    # Convenience: build from explicit model when name given, else env
    @classmethod
    def for_model_or_env(cls, model_name: Optional[str]):
        if model_name:
            return cls.from_model(model_name)
        return cls.from_env()

    def can_enter(self, symbol: str, price: float, side: str,
                   open_positions: List[Position]) -> tuple:
        """Returns (allow: bool, reason: str)."""
        if price < self.cfg.min_price:
            return False, f"price {price} < min_price {self.cfg.min_price}"
        if side == "SELL" and not self.cfg.enable_short:
            return False, "shorting disabled"
        if len(open_positions) >= self.cfg.max_concurrent:
            return False, f"max_concurrent {self.cfg.max_concurrent} reached"
        # Already in position on this symbol?
        for p in open_positions:
            if p.symbol == symbol:
                return False, f"already in {symbol}"
        return True, "ok"

    def size_position(self, price: float, open_positions: List[Position]) -> int:
        """Equal-share remaining cash across remaining slots, floor to lot.

        Subtracts approximate broker charges (formula-based) so the sized qty
        leaves enough cash for fees. If qty*price + charges still exceeds
        available cash (e.g. fill-price slippage estimate), shrinks qty by 1%
        in a loop until cost fits or qty drops to 0.

        Final guardrail: total BUY exposure capped at
        allocated_capital + cumulative_realized_pnl.
        """
        used = sum(p.qty * p.entry_price for p in open_positions)
        # Net-cash capital (from_model) is already net of open positions, so
        # subtracting `used` again would double-count. Gross capital (env path)
        # still needs the subtraction to reserve cash for existing positions.
        cash = self.cfg.capital_inr if self._capital_is_net_cash \
            else self.cfg.capital_inr - used
        slots_left = self.cfg.max_concurrent - len(open_positions)
        if slots_left <= 0 or cash <= 0:
            return 0
        slot_alloc = min(cash / slots_left, self.cfg.max_per_trade_inr)
        qty = int(slot_alloc // price)

        def _approx_buy_charges(q: int, px: float) -> float:
            if q < 1:
                return 0.0
            try:
                from tools.live.broker_charges import compute_charges
                br = compute_charges("BUY", q, px, "CNC")
                return float(br.get("total", 0.0))
            except Exception:
                return 20.0  # safe fallback (Fyers flat brokerage approx)

        # Pre-deduct charges, then 1%-shrink loop until fits
        for _ in range(200):  # max ~87% shrink from start, bounded loop
            if qty < 1:
                qty = 0
                break
            cost = qty * price + _approx_buy_charges(qty, price)
            if cost <= cash:
                break
            shrunk = int(qty * 0.99)
            qty = shrunk if shrunk < qty else qty - 1

        # Hard ceiling: allocated + realized.
        if self.cfg.max_total_buy_inr is not None and self.cfg.max_total_buy_inr > 0:
            headroom = self.cfg.max_total_buy_inr - used
            max_qty_by_ceiling = int(max(0, headroom) // price)
            if qty > max_qty_by_ceiling:
                log.warning(
                    "size_position: clamping qty %d → %d for price ₹%.2f "
                    "(used=₹%s, ceiling=₹%s, headroom=₹%s)",
                    qty, max_qty_by_ceiling, price,
                    f"{used:,.0f}",
                    f"{self.cfg.max_total_buy_inr:,}",
                    f"{headroom:,.0f}",
                )
                qty = max_qty_by_ceiling
        return max(qty, 0)


if __name__ == "__main__":
    rm = RiskManager.from_env()
    print(f"RiskManager: capital=₹{rm.cfg.capital_inr:,} "
          f"max_concurrent={rm.cfg.max_concurrent} "
          f"max_per_trade=₹{rm.cfg.max_per_trade_inr:,}")
    # Smoke test
    pos = [Position("RELIANCE", 100, 1400.0)]
    ok, reason = rm.can_enter("TCS", 4000.0, "BUY", pos)
    qty = rm.size_position(4000.0, pos)
    print(f"can_enter(TCS @ 4000)={ok} ({reason}), qty={qty}")
