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
    from ..data.historical_15m_service import get_historical_15m_service
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
    from src.services.data.historical_15m_service import get_historical_15m_service
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
        self.candles_15m = get_historical_15m_service()
        # Default = pure Strategy-1 v1.4 spec. Required guards stay ON via
        # StrategyConfig defaults (sma_seed_ema, sanity_flip_trend). Quality
        # filters (min_crossover_gap_pct, HTF SMA, slope50 SELL) default OFF
        # — users opt in per-stock via /settings.
        if config is None:
            config = StrategyConfig()
        self._base_config = config
        self.strategy: EMACrossoverStrategy = get_ema_crossover_strategy(config)

    # ------------------------------------------------------------------
    @staticmethod
    def min_required_bars(config: StrategyConfig) -> int:
        """Minimum 1H bar count for every active feature in `config` to be
        valid. Used to gate evaluation and to size backfills."""
        bars = config.ema_slow_period + 5
        if config.htf_filter_enabled:
            bars = max(bars, config.htf_buy_period_bars or 0,
                       config.htf_sell_period_bars or 0,
                       config.htf_period_bars or 0)
        if config.sell_slope_bars and config.sell_slope_bars > 0:
            bars = max(bars, config.ema_fast_period + config.sell_slope_bars)
        if config.buy_slope_bars and config.buy_slope_bars > 0:
            bars = max(bars, config.ema_fast_period + config.buy_slope_bars)
        if config.volume_confirm_mult > 0:
            bars = max(bars, config.ema_slow_period + (config.volume_confirm_bars or 0))
        return bars

    @staticmethod
    def required_backfill_days(config: StrategyConfig) -> int:
        """Convert min_required_bars to calendar days. NSE = ~7 1H bars/day,
        trading days are 5/7 of calendar days. Add 30-day holiday buffer."""
        bars = EMACrossoverRunner.min_required_bars(config)
        trading_days = (bars + 6) // 7
        return int(trading_days * (7.0 / 5.0)) + 30

    def _effective_config(self, user_id: int) -> StrategyConfig:
        """Merge per-user overrides from auto_trading_settings.ema_strategy_config
        onto the base config. Missing column / row / fields = defaults."""
        try:
            from ...models.models import AutoTradingSettings
        except ImportError:
            from src.models.models import AutoTradingSettings
        try:
            with self.db.get_session() as session:
                row = (
                    session.query(AutoTradingSettings)
                    .filter_by(user_id=user_id)
                    .first()
                )
                overrides = (row.ema_strategy_config or {}) if row else {}
        except Exception as e:
            logger.warning(f"ema_strategy_config load failed for user {user_id}: {e}")
            return self._base_config
        if not overrides:
            return self._base_config

        from dataclasses import asdict, fields
        valid_keys = {f.name for f in fields(StrategyConfig)}
        merged = asdict(self._base_config)
        for k, v in overrides.items():
            if k in valid_keys and v is not None:
                merged[k] = v
        return StrategyConfig(**merged)

    def _ensure_data_sufficient(
        self, user_id: int, symbol: str, config: StrategyConfig
    ) -> int:
        """Make sure local 1H history has enough bars for the active config.
        If short, auto-trigger a single backfill call sized to the requirement.

        Returns: current bar count after any auto-backfill.
        """
        window = self.candles.load_candles(symbol, limit=1)
        try:
            with self.db.get_session() as session:
                from sqlalchemy import func
                try:
                    from ...models.historical_models import HistoricalData1H
                except ImportError:
                    from src.models.historical_models import HistoricalData1H
                count = (
                    session.query(func.count(HistoricalData1H.id))
                    .filter(HistoricalData1H.symbol == symbol)
                    .scalar()
                ) or 0
        except Exception:
            count = len(window)

        need = self.min_required_bars(config)
        if count >= need:
            return count

        days = self.required_backfill_days(config)
        logger.info(
            f"{symbol}: only {count} 1H bars, need {need} for active config — "
            f"backfilling {days}d"
        )
        try:
            self.candles.backfill_symbol(user_id, symbol, days=days)
        except Exception as e:
            logger.error(f"{symbol}: auto-backfill failed: {e}")
        # Recount
        try:
            with self.db.get_session() as session:
                from sqlalchemy import func
                try:
                    from ...models.historical_models import HistoricalData1H
                except ImportError:
                    from src.models.historical_models import HistoricalData1H
                count = (
                    session.query(func.count(HistoricalData1H.id))
                    .filter(HistoricalData1H.symbol == symbol)
                    .scalar()
                ) or 0
        except Exception:
            pass
        return count

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

        # Per-user config: overlay DB overrides onto the base config and
        # swap it onto the strategy for this run.
        effective = self._effective_config(user_id)
        self.strategy.config = effective
        need_bars = self.min_required_bars(effective)
        load_limit = max(need_bars + 50, 600)

        for symbol in symbols:
            try:
                # 1. Refresh data — auto-extends backfill if local history is short
                self._ensure_data_sufficient(user_id, symbol, effective)
                self.candles.update_latest(user_id, symbol, lookback_days=backfill_days)
                # 15m refresh — only fetched when the user enabled 15m sustain.
                latest_15m = None
                if effective.use_15m_sustain:
                    try:
                        self.candles_15m.update_latest(user_id, symbol, lookback_days=2)
                        latest_15m = self.candles_15m.latest_candle(symbol)
                    except Exception as e:
                        logger.warning(f"{symbol}: 15m refresh failed: {e}")

                # 2. Load candles (window sized to active config requirements)
                window = self.candles.load_candles(symbol, limit=load_limit)
                if len(window) < need_bars:
                    logger.debug(f"{symbol}: insufficient 1H data "
                                 f"({len(window)} < {need_bars})")
                    continue

                # 3. Evaluate strategy
                signals = self.strategy.evaluate(user_id, symbol, window, latest_15m_bar=latest_15m)
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
        days: Optional[int] = None,
        max_symbols: int = 500,
    ) -> Dict:
        """One-shot: pull a long history before the first strategy run.

        ``days=None`` (default) sizes the window from the user's effective
        config so every active feature (EMA400 + HTF SMA + slope + swing +
        volume) has enough warmup history.
        """
        if symbols is None:
            symbols = self._default_universe(max_symbols)
        if days is None:
            days = self.required_backfill_days(self._effective_config(user_id))
            logger.info(f"Auto-sized backfill window: {days}d ({len(symbols)} symbols)")
        return self.candles.backfill_universe(user_id, symbols, days=days)

    def run_pending_sustains(
        self,
        user_id: int,
        backfill_days: int = 5,
    ) -> Dict:
        """Process only symbols whose state has a pending retest break.

        Cheap intra-1H pass: refresh 15m for the subset, then re-run
        ``evaluate`` so the 15m sustain check can fire ENTRY without waiting
        for the next 1H close.
        """
        try:
            from ...models.historical_models import EMACrossoverState
        except ImportError:
            from src.models.historical_models import EMACrossoverState

        with self.db.get_session() as session:
            rows = (
                session.query(EMACrossoverState.symbol)
                .filter(EMACrossoverState.user_id == user_id)
                .filter(
                    (EMACrossoverState.retest1_pending_cross_ts.isnot(None))
                    | (EMACrossoverState.retest2_pending_cross_ts.isnot(None))
                )
                .all()
            )
        symbols = [r[0] for r in rows]
        if not symbols:
            return {"success": True, "user_id": user_id, "pending_symbols": 0,
                    "signals_emitted": 0}

        signals_total: List[dict] = []
        errors: List[Dict] = []
        for symbol in symbols:
            try:
                self.candles_15m.update_latest(user_id, symbol, lookback_days=2)
                window = self.candles.load_candles(symbol, limit=600)
                if len(window) < self.strategy.config.ema_slow_period + 5:
                    continue
                latest_15m = self.candles_15m.latest_candle(symbol)
                if latest_15m is None:
                    continue
                signals = self.strategy.evaluate(
                    user_id, symbol, window, latest_15m_bar=latest_15m
                )
                if signals:
                    signals_total.extend((symbol, s) for s in signals)
                    self._promote_to_daily_picks(symbol, signals)
            except Exception as e:
                logger.error(f"Pending sustain error for {symbol}: {e}", exc_info=True)
                errors.append({"symbol": symbol, "error": str(e)})

        return {
            "success": True,
            "user_id": user_id,
            "pending_symbols": len(symbols),
            "signals_emitted": len(signals_total),
            "errors": errors,
        }

    def backfill_15m_universe(
        self,
        user_id: int,
        symbols: Optional[List[str]] = None,
        days: int = 30,
        max_symbols: int = 500,
    ) -> Dict:
        """One-shot 15m backfill — only needed once per symbol set."""
        if symbols is None:
            symbols = self._default_universe(max_symbols)
        return self.candles_15m.backfill_universe(user_id, symbols, days=days)

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
                             selection_score, reason, created_at)
                        VALUES
                            (:date, :symbol, 'ema_200_400', 'crossover',
                             :stock_name, :current_price, :target_price, :stop_loss,
                             :recommendation, :score, :reason,
                             CURRENT_TIMESTAMP)
                        ON CONFLICT (date, symbol, strategy, model_type) DO UPDATE
                          SET current_price = EXCLUDED.current_price,
                              target_price  = EXCLUDED.target_price,
                              stop_loss     = EXCLUDED.stop_loss,
                              recommendation = EXCLUDED.recommendation,
                              selection_score = EXCLUDED.selection_score,
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
