"""
EMA 200/400 Crossover Strategy Runner.

Drives the strategy across a universe of symbols for a given user:

  1. Pulls/refreshes the latest 1H candles via Historical1HService.
  2. Loads the rolling window of candles needed for EMA200/400.
  3. Runs the state machine and persists fresh signals.
  4. Promotes ENTRY1 / ENTRY2 signals into `daily_suggested_stocks` with
     ``strategy = 'ema_200_400'`` so the auto-trader can act on them.

The runner is purely additive — it does not interfere with login or token
flow. Auth is reused from the existing FyersService.
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional

from sqlalchemy import text

try:
    from ...models.database import get_database_manager
    from ...models.stock_models import Stock
    from ..data.historical_1h_service import get_historical_1h_service
    from ..data.nifty500_universe import load_nifty500
    from .ema_crossover_strategy import (
        EMACrossoverStrategy,
        StrategyConfig,
        get_ema_crossover_strategy,
    )
except ImportError:
    from src.models.database import get_database_manager
    from src.models.stock_models import Stock
    from src.services.data.historical_1h_service import get_historical_1h_service
    from src.services.data.nifty500_universe import load_nifty500
    from src.services.technical.ema_crossover_strategy import (
        EMACrossoverStrategy,
        StrategyConfig,
        get_ema_crossover_strategy,
    )

logger = logging.getLogger(__name__)


class EMACrossoverRunner:
    """Top-level orchestrator for the EMA 200/400 strategy."""

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.db = get_database_manager()
        self.candles = get_historical_1h_service()
        self.strategy: EMACrossoverStrategy = get_ema_crossover_strategy(config)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def run_for_user(
        self,
        user_id: int,
        symbols: Optional[List[str]] = None,
        max_symbols: int = 500,
        backfill_days: int = 5,
    ) -> Dict:
        """Refresh data + run strategy for every symbol in the universe."""
        if symbols is None:
            symbols = self._default_universe(max_symbols)

        if not symbols:
            return {"success": False, "error": "no symbols configured"}

        signals_total: List[dict] = []
        errors: List[Dict] = []

        for symbol in symbols:
            try:
                # 1. Refresh data
                self.candles.update_latest(user_id, symbol, lookback_days=backfill_days)

                # 2. Load candles
                window = self.candles.load_candles(symbol, limit=600)
                if len(window) < self.strategy.config.ema_slow_period + 5:
                    logger.debug(f"{symbol}: insufficient 1H data ({len(window)})")
                    continue

                # 3. Evaluate strategy
                signals = self.strategy.evaluate(user_id, symbol, window)
                if signals:
                    signals_total.extend((symbol, s) for s in signals)
                    self._promote_to_daily_picks(symbol, signals)
            except Exception as e:
                logger.error(f"Runner error for {symbol}: {e}", exc_info=True)
                errors.append({"symbol": symbol, "error": str(e)})

        return {
            "success": True,
            "user_id": user_id,
            "symbols_processed": len(symbols),
            "signals_emitted": len(signals_total),
            "errors": errors,
        }

    def backfill_universe(
        self,
        user_id: int,
        symbols: Optional[List[str]] = None,
        days: int = 120,
        max_symbols: int = 500,
    ) -> Dict:
        """One-shot: pull a long history before the first strategy run."""
        if symbols is None:
            symbols = self._default_universe(max_symbols)
        return self.candles.backfill_universe(user_id, symbols, days=days)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _default_universe(self, max_symbols: int) -> List[str]:
        """Universe = Nifty 500 (CSV cache). Falls back to top market-cap from
        the ``stocks`` table if the cache is missing."""
        symbols = load_nifty500()
        if symbols:
            return symbols[:max_symbols]

        logger.warning(
            "Nifty 500 cache unavailable; falling back to top market-cap stocks. "
            "Run tools/refresh_nifty500.py to restore."
        )
        with self.db.get_session() as session:
            rows = (
                session.query(Stock.symbol)
                .filter(Stock.is_active.is_(True), Stock.is_tradeable.is_(True))
                .order_by(Stock.market_cap.desc().nullslast())
                .limit(max_symbols)
                .all()
            )
        return [r[0] for r in rows]

    def _promote_to_daily_picks(self, symbol: str, signals: List[dict]) -> None:
        """Write actionable ENTRY signals to ``daily_suggested_stocks``.

        v2 BTC rules: target = entry * (1 + target_pct), SL is per-entry:
            ENTRY1 -> EMA400 (close-based exit)
            ENTRY2 -> retest2.low (BUY) / retest2.high (SELL)
        """
        actionable = [s for s in signals if s["signal_type"] in ("ENTRY1", "ENTRY2")]
        if not actionable:
            return

        today = date.today()
        target_pct = self.strategy.config.target_pct
        with self.db.get_session() as session:
            for sig in actionable:
                recommendation = "BUY" if sig["trend"] == "BUY" else "SELL"
                price = float(sig["price"])
                ema400 = float(sig["ema_400"])
                # Strategy stamps target/sl onto each ENTRY signal directly.
                target_price = float(sig.get("target") or
                                     (price * (1 + target_pct) if recommendation == "BUY"
                                      else price * (1 - target_pct)))
                stop_loss = float(sig.get("sl") or ema400)

                session.execute(
                    text(
                        """
                        INSERT INTO daily_suggested_stocks
                            (date, symbol, strategy, model_type, stock_name,
                             current_price, target_price, stop_loss, recommendation,
                             selection_score, signal_quality, reason, created_at)
                        VALUES
                            (:date, :symbol, 'ema_200_400', 'crossover',
                             :stock_name, :current_price, :target_price, :stop_loss,
                             :recommendation, :score, :quality, :reason,
                             CURRENT_TIMESTAMP)
                        ON CONFLICT (date, symbol, strategy, model_type) DO UPDATE
                          SET current_price = EXCLUDED.current_price,
                              target_price  = EXCLUDED.target_price,
                              stop_loss     = EXCLUDED.stop_loss,
                              recommendation = EXCLUDED.recommendation,
                              selection_score = EXCLUDED.selection_score,
                              signal_quality = EXCLUDED.signal_quality,
                              reason = EXCLUDED.reason
                        """
                    ),
                    {
                        "date": today,
                        "symbol": symbol,
                        "stock_name": symbol,
                        "current_price": price,
                        "target_price": target_price,
                        "stop_loss": stop_loss,
                        "recommendation": recommendation,
                        "score": 100.0 if sig["signal_type"] == "ENTRY1" else 90.0,
                        "quality": "high",
                        "reason": sig.get("note") or sig["signal_type"],
                    },
                )
            session.commit()


_runner: Optional[EMACrossoverRunner] = None


def get_ema_crossover_runner(config: Optional[StrategyConfig] = None) -> EMACrossoverRunner:
    global _runner
    if _runner is None:
        _runner = EMACrossoverRunner(config)
    return _runner
