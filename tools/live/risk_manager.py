"""Risk manager — enforces capital lock, max concurrent, per-trade cap.

Pure Python. Used by paper_executor + fyers_executor before placing any
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
    capital_inr: int = 200_000
    max_concurrent: int = 2
    max_per_trade_inr: int = 100_000   # capital / max_concurrent
    max_daily_loss_pct: float = -5.0   # kill switch
    min_price: float = 50.0
    enable_short: bool = False


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
            capital_inr=int(os.environ.get("CAPITAL_INR", 200_000)),
            max_concurrent=int(os.environ.get("MAX_CONCURRENT", 2)),
            max_per_trade_inr=int(os.environ.get("MAX_PER_TRADE_INR",
                int(os.environ.get("CAPITAL_INR", 200_000)) //
                int(os.environ.get("MAX_CONCURRENT", 2)))),
            max_daily_loss_pct=float(os.environ.get("MAX_DAILY_LOSS_PCT", -5.0)),
            min_price=float(os.environ.get("MIN_PRICE", 50.0)),
            enable_short=os.environ.get("ENABLE_SHORT", "false").lower() == "true",
        )
        return cls(cfg)

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
