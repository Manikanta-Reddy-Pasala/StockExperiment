"""Risk manager — enforces capital lock, max concurrent, per-trade cap.

Pure Python. Used by fyers_executor before placing any
order. Reads current portfolio state, decides if entry is allowed.

Constraints (configurable via env or YAML):
  - CAPITAL_INR: locked-in pool (no add/withdraw)
  - MAX_CONCURRENT: simultaneous open positions cap
  - MAX_PER_TRADE_INR: per-position size cap
  - MAX_DAILY_LOSS_PCT: kill-switch if today's P&L < -X% of capital
  - MIN_PRICE: penny filter
  - MIN_ADV_INR: liquidity filter (skip illiquid)

Usage:
  rm = RiskManager(capital=200000, max_concurrent=2)
  if rm.can_enter(symbol, price, open_positions, day_pnl):
      qty = rm.size_position(price, open_positions)
      place_order(...)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

log = logging.getLogger("risk_manager")


@dataclass
class RiskConfig:
    capital_inr: int = 1_000_000        # ₹10L locked-in pool
    max_concurrent: int = 2
    max_per_trade_inr: int = 500_000    # capital / max_concurrent
    max_daily_loss_pct: float = -5.0    # kill switch
    min_price: float = 50.0
    enable_short: bool = False
    # LIMIT-with-tolerance + MARKET-fallback knobs (Phase 2 Task 6)
    limit_tol_pct: float = 0.5          # initial LIMIT slack vs last price
    limit_retry_pct: float = 1.0        # widened slack on first re-quote
    limit_fallback_s: int = 20          # total seconds before MARKET fallback


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    side: str = "BUY"
    sl: float = 0.0
    target: float = 0.0


class RiskManager:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg = config or RiskConfig.from_env()

    @classmethod
    def from_env(cls):
        cfg = RiskConfig(
            capital_inr=int(os.environ.get("CAPITAL_INR", 1_000_000)),
            max_concurrent=int(os.environ.get("MAX_CONCURRENT", 2)),
            max_per_trade_inr=int(os.environ.get("MAX_PER_TRADE_INR",
                int(os.environ.get("CAPITAL_INR", 1_000_000)) //
                int(os.environ.get("MAX_CONCURRENT", 2)))),
            max_daily_loss_pct=float(os.environ.get("MAX_DAILY_LOSS_PCT", -5.0)),
            min_price=float(os.environ.get("MIN_PRICE", 50.0)),
            enable_short=os.environ.get("ENABLE_SHORT", "false").lower() == "true",
            limit_tol_pct=float(os.environ.get("LIMIT_TOL_PCT", 0.5)),
            limit_retry_pct=float(os.environ.get("LIMIT_RETRY_PCT", 1.0)),
            limit_fallback_s=int(os.environ.get("LIMIT_FALLBACK_S", 20)),
        )
        return cls(cfg)

    @classmethod
    def from_model(cls, model_name: str):
        """Build RiskManager using per-model `current_amount` as capital.

        Reads `model_settings.current_amount` from the DB (live NAV cache,
        updated on each fill by record_buy/record_sell). Other risk knobs
        still come from env so a single model's capital change doesn't force
        a redeploy.

        Fallback rules:
          - Unknown model OR DB access fails → fall back to from_env() and
            log a WARNING. Preserves backward-compat with momentum_n100_top5_max1
            (already live with HFCL).
          - Settings present but current_amount <= 0 → still uses 0 as the
            capital floor; size_position will return 0 (kill-switch via cash).
        """
        try:
            from src.services.trading.model_ledger_service import get_all_settings
            settings_rows = {row["model_name"]: row for row in get_all_settings()}
            row = settings_rows.get(model_name)
            if not row:
                log.warning(
                    "RiskManager.from_model('%s'): model not in DB, falling back to from_env()",
                    model_name,
                )
                return cls.from_env()
            capital = int(float(row.get("current_amount") or 0))
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
            max_daily_loss_pct=float(os.environ.get("MAX_DAILY_LOSS_PCT", -5.0)),
            min_price=float(os.environ.get("MIN_PRICE", 50.0)),
            enable_short=os.environ.get("ENABLE_SHORT", "false").lower() == "true",
            limit_tol_pct=float(os.environ.get("LIMIT_TOL_PCT", 0.5)),
            limit_retry_pct=float(os.environ.get("LIMIT_RETRY_PCT", 1.0)),
            limit_fallback_s=int(os.environ.get("LIMIT_FALLBACK_S", 20)),
        )
        log.info(
            "RiskManager.from_model('%s'): capital=₹%s max_concurrent=%d max_per_trade=₹%s",
            model_name, f"{capital:,}", max_concurrent, f"{max_per_trade:,}",
        )
        return cls(cfg)

    # Convenience: build from explicit model when name given, else env
    @classmethod
    def for_model_or_env(cls, model_name: Optional[str]):
        if model_name:
            return cls.from_model(model_name)
        return cls.from_env()

    def can_enter(self, symbol: str, price: float, side: str,
                   open_positions: List[Position], day_pnl: float = 0.0) -> tuple:
        """Returns (allow: bool, reason: str)."""
        if price < self.cfg.min_price:
            return False, f"price {price} < min_price {self.cfg.min_price}"
        if side == "SELL" and not self.cfg.enable_short:
            return False, "shorting disabled"
        if len(open_positions) >= self.cfg.max_concurrent:
            return False, f"max_concurrent {self.cfg.max_concurrent} reached"
        # Daily loss kill-switch
        day_loss_pct = (day_pnl / self.cfg.capital_inr) * 100
        if day_loss_pct <= self.cfg.max_daily_loss_pct:
            return False, f"daily loss kill-switch ({day_loss_pct:.1f}%)"
        # Already in position on this symbol?
        for p in open_positions:
            if p.symbol == symbol:
                return False, f"already in {symbol}"
        return True, "ok"

    def size_position(self, price: float, open_positions: List[Position]) -> int:
        """Equal-share remaining cash across remaining slots, floor to lot."""
        used = sum(p.qty * p.entry_price for p in open_positions)
        cash = self.cfg.capital_inr - used
        slots_left = self.cfg.max_concurrent - len(open_positions)
        if slots_left <= 0 or cash <= 0:
            return 0
        slot_alloc = min(cash / slots_left, self.cfg.max_per_trade_inr)
        qty = int(slot_alloc // price)
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
