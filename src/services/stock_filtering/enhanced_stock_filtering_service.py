"""
Enhanced Stock Filtering Service
Implements comprehensive stock filtering with Stage 1 and Stage 2 filters
Supports all features from the enhanced YAML configuration
"""

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - optional dependency, fall back to custom calcs
    ta = None

try:
    from ...models.database import get_database_manager
    from ...models.historical_models import HistoricalData
except ImportError:  # pragma: no cover - package-relative import fallback
    from src.models.database import get_database_manager
    from src.models.historical_models import HistoricalData

from .enhanced_config_loader import (
    EnhancedFilteringConfig,
    FilteringThresholds,
    FundamentalRatios,
    RiskMetrics,
    ScoringWeights,
    SelectionConfig,
    Stage1Filters,
    UniverseConfig,
    get_enhanced_config_loader,
    get_enhanced_filtering_config,
)

logger = logging.getLogger(__name__)


@dataclass
class StockContext:
    """Aggregated runtime context for a single stock."""

    stock: Any
    symbol: str
    current_price: float = 0.0
    volume: float = 0.0
    avg_volume_20d: Optional[float] = None
    avg_turnover: Optional[float] = None
    latest_turnover: Optional[float] = None
    liquidity_score: Optional[float] = None
    listing_days: Optional[int] = None
    listing_date: Optional[date] = None
    trades_per_day: Optional[int] = None
    bid_ask_spread: Optional[float] = None
    atr: Optional[float] = None
    atr_percentage: Optional[float] = None
    historical: Optional[pd.DataFrame] = None
    technicals: Dict[str, Optional[float]] = field(default_factory=dict)
    fundamentals: Dict[str, Optional[float]] = field(default_factory=dict)
    risk_metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    momentum_metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    volume_signals: Dict[str, Optional[float]] = field(default_factory=dict)
    sector: Optional[str] = None
    sector_performance: Optional[float] = None
    market_cap_category: Optional[str] = None
    market_cap: Optional[float] = None
    distance_from_resistance_pct: Optional[float] = None


@dataclass
class StockScore:
    """Stock scoring results."""

    symbol: str
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    risk_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    total_score: float = 0.0
    filters_passed: List[str] = field(default_factory=list)
    reject_reasons: List[str] = field(default_factory=list)
    stock_object: Any = None
    context: Optional[StockContext] = None


@dataclass
class FilteringResult:
    """Complete filtering result."""
    selected_stocks: List[StockScore]
    rejected_stocks: List[StockScore]
    total_processed: int
    stage1_passed: int
    stage2_passed: int
    final_selected: int
    execution_time: float
    config_used: EnhancedFilteringConfig


class EnhancedStockFilteringService:
    """Enhanced stock filtering service with comprehensive filtering capabilities."""
    
    def __init__(self, config: Optional[EnhancedFilteringConfig] = None):
        """Initialize the enhanced filtering service."""
        self._config_loader = get_enhanced_config_loader()
        self.config = config or self._config_loader.get_config()
        self.raw_config = self._load_raw_config()
        self.db_manager = None
        try:
            self.db_manager = get_database_manager()
        except Exception as exc:  # pragma: no cover - database may be unavailable during tests
            logger.warning(f"Database manager unavailable for filtering service: {exc}")
            self.db_manager = None
        self.filter_stats = {
            'total_processed': 0,
            'stage1_passed': 0,
            'stage2_passed': 0,
            'final_selected': 0,
            'execution_time': 0.0
        }

    def _load_raw_config(self) -> Dict[str, Any]:
        """Load the raw YAML configuration for access to nested thresholds."""
        try:
            config_path = getattr(self._config_loader, 'config_path', None)
            if config_path is None:
                project_root = Path(__file__).parent.parent.parent.parent
                config_path = project_root / "config" / "stock_filters.yaml"
            if not Path(config_path).exists():
                return {}
            with open(config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to load raw stock filter config: {exc}")
            return {}
    
    def filter_stocks(self, stocks: List[Any], user_id: int = 1) -> FilteringResult:
        """
        Comprehensive stock filtering with Stage 1 and Stage 2 filters.
        
        Args:
            stocks: List of stock objects to filter
            user_id: User ID for personalized filtering
            
        Returns:
            FilteringResult with selected and rejected stocks
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting enhanced stock filtering for {len(stocks)} stocks")
            stock_contexts = self._build_stock_contexts(stocks)
            
            # Stage 1: Basic filtering
            stage1_stocks, stage1_rejected = self._apply_stage1_filters(stock_contexts)
            logger.info(f"Stage 1: {len(stage1_stocks)} stocks passed, {len(stage1_rejected)} rejected")
            
            # Stage 2: Advanced analysis and scoring
            stage2_stocks, stage2_rejected = self._apply_stage2_filters(stage1_stocks)
            logger.info(f"Stage 2: {len(stage2_stocks)} stocks passed, {len(stage2_rejected)} rejected")
            
            # Final selection with guardrails
            final_stocks, final_rejected = self._apply_final_selection(stage2_stocks)
            logger.info(f"Final selection: {len(final_stocks)} stocks selected, {len(final_rejected)} rejected")
            
            # Combine all rejected stocks
            all_rejected = stage1_rejected + stage2_rejected + final_rejected
            
            execution_time = time.time() - start_time
            
            # Update statistics
            self.filter_stats.update({
                'total_processed': len(stocks),
                'stage1_passed': len(stage1_stocks),
                'stage2_passed': len(stage2_stocks),
                'final_selected': len(final_stocks),
                'execution_time': execution_time
            })
            
            result = FilteringResult(
                selected_stocks=final_stocks,
                rejected_stocks=all_rejected,
                total_processed=len(stocks),
                stage1_passed=len(stage1_stocks),
                stage2_passed=len(stage2_stocks),
                final_selected=len(final_stocks),
                execution_time=execution_time,
                config_used=self.config
            )

            logging_cfg = self.raw_config.get('logging', {})
            if logging_cfg.get('include_reject_reasons', False):
                self._log_rejected_symbols(stage1_rejected, stage2_rejected, final_rejected)
            top_n = logging_cfg.get('top_stocks_per_category', 0) or 0
            if top_n > 0:
                self._log_top_stocks(stage2_stocks, top_n)
            
            logger.info(f"Enhanced filtering completed in {execution_time:.2f}s")
            logger.info(f"Final result: {len(final_stocks)} stocks selected from {len(stocks)} processed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced stock filtering: {e}")
            # Return empty result on error
            return FilteringResult(
                selected_stocks=[],
                rejected_stocks=[],
                total_processed=len(stocks),
                stage1_passed=0,
                stage2_passed=0,
                final_selected=0,
                execution_time=time.time() - start_time,
                config_used=self.config
            )
    
    def _build_stock_contexts(self, stocks: List[Any]) -> List[StockContext]:
        """Construct runtime analysis context for each stock."""
        lookback_days = max(260, getattr(self.config.universe, 'min_history_days', 220) + 40)
        start_date = datetime.utcnow().date() - timedelta(days=lookback_days)
        contexts: List[StockContext] = []

        def build_with_session(session) -> List[StockContext]:
            built_contexts: List[StockContext] = []
            for stock in stocks:
                symbol = getattr(stock, 'symbol', None)
                if not symbol:
                    continue
                context = StockContext(stock=stock, symbol=symbol)
                context.current_price = float(getattr(stock, 'current_price', 0.0) or 0.0)
                context.volume = float(getattr(stock, 'volume', 0.0) or 0.0)
                context.avg_volume_20d = self._safe_numeric(getattr(stock, 'avg_daily_volume_20d', None))
                context.avg_turnover = self._safe_numeric(getattr(stock, 'avg_daily_turnover', None))
                context.latest_turnover = context.current_price * context.volume if context.current_price and context.volume else None
                context.trades_per_day = self._safe_numeric(getattr(stock, 'trades_per_day', None))
                context.bid_ask_spread = self._safe_numeric(getattr(stock, 'bid_ask_spread', None))
                context.market_cap_category = getattr(stock, 'market_cap_category', None)
                context.market_cap = self._safe_numeric(getattr(stock, 'market_cap', None))
                context.sector = getattr(stock, 'sector', 'Unknown') or 'Unknown'

                listing_date_value = getattr(stock, 'listing_date', None)
                if isinstance(listing_date_value, datetime):
                    context.listing_date = listing_date_value.date()
                elif isinstance(listing_date_value, date):
                    context.listing_date = listing_date_value

                df = self._fetch_historical_df(session, symbol, start_date)
                context.historical = df
                if df is not None and not df.empty:
                    df = df.sort_values('date')
                    first_trade = pd.to_datetime(df['date'].iloc[0]).date()
                    history_listing_days = (datetime.utcnow().date() - first_trade).days
                    context.listing_days = history_listing_days
                    if context.listing_date is None or first_trade < context.listing_date:
                        context.listing_date = first_trade
                    # Use latest close/volume if current price missing or stale
                    if context.current_price <= 0:
                        context.current_price = float(df['close'].iloc[-1])
                    if context.volume <= 0:
                        context.volume = float(df['volume'].iloc[-1])
                    if not context.avg_volume_20d:
                        context.avg_volume_20d = float(df['volume'].tail(20).mean())
                    if not context.avg_turnover:
                        context.avg_turnover = float((df['close'] * df['volume']).tail(20).mean())
                    context.latest_turnover = float(df['close'].iloc[-1] * df['volume'].iloc[-1])

                    tech = self._compute_technical_indicators(df)
                    context.technicals = tech
                    context.atr = tech.get('atr')
                    context.atr_percentage = tech.get('atr_pct')

                    context.volume_signals = self._compute_volume_signals(df, tech, context)
                    context.momentum_metrics = self._compute_momentum_metrics(df, tech, context)
                    context.risk_metrics = self._compute_risk_metrics(df, context)
                    context.distance_from_resistance_pct = context.momentum_metrics.get('distance_from_52w_high_pct')
                else:
                    context.listing_days = None
                    context.technicals = {}
                    context.volume_signals = {}
                    context.momentum_metrics = {}
                    context.risk_metrics = {
                        'beta': self._safe_numeric(getattr(stock, 'beta', None)),
                        'historical_volatility_pct': self._safe_numeric(getattr(stock, 'historical_volatility_1y', None))
                    }
                    context.atr = self._safe_numeric(getattr(stock, 'atr_14', None))
                    context.atr_percentage = self._safe_numeric(getattr(stock, 'atr_percentage', None))

                context.liquidity_score = self._compute_liquidity_score(context)
                context.market_cap_category = context.market_cap_category or self._derive_market_cap_category(stock)
                context.fundamentals = self._extract_fundamentals(stock)
                if context.listing_days is None and context.listing_date is not None:
                    context.listing_days = (datetime.utcnow().date() - context.listing_date).days
                if context.listing_days is not None and context.listing_days < 0:
                    context.listing_days = 0

                built_contexts.append(context)
            return built_contexts

        if self.db_manager is not None:
            try:
                with self.db_manager.get_session() as session:
                    contexts = build_with_session(session)
            except Exception as exc:  # pragma: no cover - database optional
                logger.warning(f"Historical data unavailable during context build: {exc}")
                contexts = build_with_session(None)
        else:
            contexts = build_with_session(None)

        self._attach_sector_performance(contexts)
        return contexts

    def _apply_stage1_filters(self, contexts: List[StockContext]) -> Tuple[List[StockScore], List[StockScore]]:
        """Apply Stage 1 basic filtering criteria."""
        passed_stocks = []
        rejected_stocks = []
        
        stage1_config = self.config.stage_1_filters
        
        for context in contexts:
            try:
                score = StockScore(symbol=context.symbol, stock_object=context.stock, context=context)
                reject_reasons = []
                
                # Price range filter
                current_price = context.current_price or 0
                if current_price <= 0:
                    reject_reasons.append(f"Invalid price {current_price}")
                elif current_price < stage1_config.minimum_price:
                    reject_reasons.append(f"Price {current_price} below minimum {stage1_config.minimum_price}")
                elif current_price > stage1_config.maximum_price:
                    reject_reasons.append(f"Price {current_price} above maximum {stage1_config.maximum_price}")
                else:
                    score.filters_passed.append("price_range")
                
                # Volume and turnover filter
                volume = context.volume or 0
                daily_turnover = context.latest_turnover or (current_price * volume if current_price and volume else 0)
                
                if daily_turnover < stage1_config.minimum_daily_turnover_inr:
                    if volume < stage1_config.fallback_minimum_volume:
                        reject_reasons.append(f"Volume {volume} below fallback minimum {stage1_config.fallback_minimum_volume}")
                    else:
                        score.filters_passed.append("volume_fallback")
                else:
                    score.filters_passed.append("turnover")
                
                # Liquidity score filter (use volume as proxy if liquidity_score not available)
                liquidity_score = context.liquidity_score
                if liquidity_score is None:
                    liquidity_score = 0.0

                if liquidity_score < stage1_config.minimum_liquidity_score:
                    reject_reasons.append(
                        f"Liquidity score {liquidity_score:.2f} below minimum {stage1_config.minimum_liquidity_score}"
                    )
                else:
                    score.filters_passed.append("liquidity")
                
                # Trading status filter
                is_active = getattr(context.stock, 'is_active', False)
                is_tradeable = getattr(context.stock, 'is_tradeable', False)
                trading_status_cfg = self.raw_config.get('stage_1_filters', {}).get('trading_status', {})
                trading_status_passed = True

                if not is_active:
                    reject_reasons.append("Stock not active")
                    trading_status_passed = False
                elif not is_tradeable:
                    reject_reasons.append("Stock not tradeable")
                    trading_status_passed = False

                if trading_status_cfg.get('exclude_suspended', False) and getattr(context.stock, 'is_suspended', False):
                    reject_reasons.append("Stock suspended")
                    trading_status_passed = False
                if trading_status_cfg.get('exclude_delisted', False) and getattr(context.stock, 'is_delisted', False):
                    reject_reasons.append("Stock delisted")
                    trading_status_passed = False
                if trading_status_cfg.get('exclude_stage_listed', False) and getattr(context.stock, 'is_stage_listed', False):
                    reject_reasons.append("Stock in stage listing")
                    trading_status_passed = False

                if trading_status_passed:
                    score.filters_passed.append("trading_status")
                
                # ATR volatility filter
                atr_percentage = context.atr_percentage
                if atr_percentage is None:
                    # Skip ATR filter if not available
                    score.filters_passed.append("atr_volatility_skip")
                elif atr_percentage < stage1_config.min_atr_pct_of_price:
                    reject_reasons.append(f"ATR {atr_percentage}% below minimum {stage1_config.min_atr_pct_of_price}%")
                elif atr_percentage > stage1_config.max_atr_pct_of_price:
                    reject_reasons.append(f"ATR {atr_percentage}% above maximum {stage1_config.max_atr_pct_of_price}%")
                else:
                    score.filters_passed.append("atr_volatility")

                # Listing history filter
                min_listing_days = stage1_config.min_listing_days
                if min_listing_days and min_listing_days > 0:
                    if context.listing_days is None:
                        reject_reasons.append("Listing history unavailable")
                    elif context.listing_days < min_listing_days:
                        reject_reasons.append(
                            f"Listed for {context.listing_days} days < minimum {min_listing_days}"
                        )
                    else:
                        score.filters_passed.append("listing_history")
                
                # Apply rejection logic
                # Additional Stage 1 checks from configuration
                max_spread_pct = stage1_config.max_bid_ask_spread_pct
                spread_value = context.bid_ask_spread
                if spread_value is not None:
                    spread_pct = spread_value / 100 if spread_value > 1 else spread_value
                    if spread_pct * 100 > max_spread_pct:
                        reject_reasons.append(
                            f"Bid/ask spread {spread_pct * 100:.2f}% above {max_spread_pct}%"
                        )
                    else:
                        score.filters_passed.append("bid_ask_spread")

                min_trades = stage1_config.minimum_trades_per_day
                if min_trades and min_trades > 0:
                    trades_per_day = context.trades_per_day
                    if trades_per_day is not None:
                        if trades_per_day < min_trades:
                            reject_reasons.append(
                                f"Trades/day {trades_per_day} below minimum {min_trades}"
                            )
                        else:
                            score.filters_passed.append("trade_frequency")

                min_listing_days = stage1_config.min_listing_days
                if min_listing_days and context.listing_days is not None:
                    if context.listing_days < min_listing_days:
                        reject_reasons.append(
                            f"Listed for {context.listing_days} days < minimum {min_listing_days}"
                        )
                    else:
                        score.filters_passed.append("listing_history")

                if reject_reasons:
                    score.reject_reasons = reject_reasons
                    rejected_stocks.append(score)
                else:
                    passed_stocks.append(score)
                    
            except Exception as e:
                logger.warning(f"Error processing stock {context.symbol} in Stage 1: {e}")
                score = StockScore(symbol=context.symbol, stock_object=context.stock, context=context)
                score.reject_reasons = [f"Processing error: {str(e)}"]
                rejected_stocks.append(score)
        
        return passed_stocks, rejected_stocks
    
    def _apply_stage2_filters(self, stocks: List[StockScore]) -> Tuple[List[StockScore], List[StockScore]]:
        """Apply Stage 2 advanced analysis and scoring."""
        passed_stocks = []
        rejected_stocks = []
        
        for stock_score in stocks:
            try:
                context = stock_score.context
                # Calculate technical score
                technical_score = self._calculate_technical_score(context)
                stock_score.technical_score = technical_score
                
                # Calculate fundamental score
                fundamental_score = self._calculate_fundamental_score(context)
                stock_score.fundamental_score = fundamental_score
                
                # Calculate risk score
                risk_score = self._calculate_risk_score(context)
                stock_score.risk_score = risk_score
                
                # Calculate momentum score
                momentum_score = self._calculate_momentum_score(context)
                stock_score.momentum_score = momentum_score
                
                # Calculate volume score
                volume_score = self._calculate_volume_score(context)
                stock_score.volume_score = volume_score
                
                # Calculate total weighted score
                weights = self.config.scoring_weights
                total_score = (
                    technical_score * weights.technical_score +
                    fundamental_score * weights.fundamental_score +
                    risk_score * weights.risk_score +
                    momentum_score * weights.momentum_score +
                    volume_score * weights.volume_score
                ) * 100  # Convert to 0-100 scale
                
                # Convert individual scores to 0-100 scale for comparison
                technical_score = technical_score * 100
                fundamental_score = fundamental_score * 100
                risk_score = risk_score * 100
                momentum_score = momentum_score * 100
                volume_score = volume_score * 100
                
                stock_score.total_score = total_score
                
                # Apply filtering thresholds
                thresholds = self.config.filtering_thresholds
                reject_reasons = []
                
                # Debug logging for first few stocks
                if len(passed_stocks) < 3:
                    logger.info(f"Stock {stock_score.symbol} scores: total={total_score:.1f}, technical={technical_score:.1f}, fundamental={fundamental_score:.1f}, risk={risk_score:.1f}")
                
                if total_score < thresholds.minimum_total_score:
                    reject_reasons.append(f"Total score {total_score:.1f} below minimum {thresholds.minimum_total_score}")
                
                if technical_score < thresholds.minimum_technical_score:
                    reject_reasons.append(f"Technical score {technical_score:.1f} below minimum {thresholds.minimum_technical_score}")
                
                if fundamental_score < thresholds.minimum_fundamental_score:
                    reject_reasons.append(f"Fundamental score {fundamental_score:.1f} below minimum {thresholds.minimum_fundamental_score}")
                
                if risk_score < thresholds.minimum_risk_score:
                    reject_reasons.append(f"Risk score {risk_score:.1f} below minimum {thresholds.minimum_risk_score}")
                
                # Apply rejection logic
                if reject_reasons:
                    stock_score.reject_reasons.extend(reject_reasons)
                    rejected_stocks.append(stock_score)
                else:
                    stock_score.filters_passed.append("stage2_scoring")
                    passed_stocks.append(stock_score)
                    
            except Exception as e:
                logger.warning(f"Error processing stock {stock_score.symbol} in Stage 2: {e}")
                stock_score.reject_reasons.append(f"Stage 2 processing error: {str(e)}")
                rejected_stocks.append(stock_score)
        
        return passed_stocks, rejected_stocks
    
    def _apply_final_selection(self, stocks: List[StockScore]) -> Tuple[List[StockScore], List[StockScore]]:
        """Apply final selection with guardrails."""
        if not stocks:
            return [], []
        
        selection_config = self.config.selection
        max_count = selection_config.max_suggested_stocks or len(stocks)

        # Sort by total score (descending) and apply tie-breakers
        sorted_stocks = sorted(stocks, key=lambda x: x.total_score, reverse=True)
        if selection_config.tie_breaker_priority:
            for criteria in reversed(selection_config.tie_breaker_priority):
                key_func = None
                if criteria == "momentum_score":
                    key_func = lambda s: s.momentum_score
                elif criteria == "risk_score":
                    key_func = lambda s: s.risk_score
                elif criteria == "technical_score":
                    key_func = lambda s: s.technical_score
                if key_func:
                    sorted_stocks.sort(key=key_func, reverse=True)

        whitelist = set(selection_config.whitelist_symbols or [])
        blacklist = set(selection_config.blacklist_symbols or [])
        sector_cap = max(1, math.floor(max_count * selection_config.sector_concentration_limit_pct / 100))
        required_large_mid = math.ceil(max_count * selection_config.min_large_mid_pct / 100) if selection_config.min_large_mid_pct else 0

        selected: List[StockScore] = []
        sector_counts: Dict[str, int] = defaultdict(int)
        large_mid_count = 0
        deferred_small_caps: List[StockScore] = []

        def is_large_mid(stock: StockScore) -> bool:
            category = getattr(stock.context, 'market_cap_category', None)
            return category in ('large_cap', 'mid_cap', None)

        for stock in sorted_stocks:
            if len(selected) >= max_count:
                break

            if stock.symbol in blacklist:
                stock.reject_reasons.append('Blacklisted symbol')
                continue
            if whitelist and stock.symbol not in whitelist:
                stock.reject_reasons.append('Not in whitelist')
                continue

            context = stock.context
            sector = getattr(context, 'sector', 'Unknown') or 'Unknown'
            distance = context.distance_from_resistance_pct
            if distance is not None and distance < selection_config.min_distance_from_resistance_pct:
                stock.reject_reasons.append(
                    f"Too close to resistance ({distance:.2f}% < {selection_config.min_distance_from_resistance_pct}%)"
                )
                continue

            if sector_counts[sector] >= sector_cap and stock.symbol not in whitelist:
                stock.reject_reasons.append(f"Sector guardrail hit for {sector}")
                continue

            category = getattr(context, 'market_cap_category', None)
            remaining_slots = max_count - (len(selected) + 1)
            remaining_large_mid_needed = max(0, required_large_mid - large_mid_count)
            if category == 'small_cap' and remaining_large_mid_needed > remaining_slots:
                stock.reject_reasons.append('Large/mid-cap mix guardrail')
                deferred_small_caps.append(stock)
                continue

            selected.append(stock)
            sector_counts[sector] += 1
            if is_large_mid(stock):
                large_mid_count += 1

        # Fill remaining slots with deferred small caps if mix requirement satisfied
        if large_mid_count >= required_large_mid and len(selected) < max_count:
            for stock in deferred_small_caps:
                if len(selected) >= max_count:
                    break
                context = stock.context
                sector = getattr(context, 'sector', 'Unknown') or 'Unknown'
                if sector_counts[sector] >= sector_cap and stock.symbol not in whitelist:
                    continue
                selected.append(stock)
                sector_counts[sector] += 1

        selected_stocks = selected[:max_count]

        # Determine rejected stocks with guardrail reasons
        rejected_stocks: List[StockScore] = []
        for stock in stocks:
            if stock not in selected_stocks:
                if not stock.reject_reasons:
                    stock.reject_reasons.append("Final selection guardrail")
                rejected_stocks.append(stock)
        
        return selected_stocks, rejected_stocks
    
    def _calculate_technical_score(self, context: Optional[StockContext]) -> float:
        """Calculate technical score (0-1) leveraging indicator configuration."""
        if context is None:
            return 0.0

        tech = context.technicals or {}
        momentum = context.momentum_metrics or {}
        components: List[float] = []

        rsi_cfg = self.raw_config.get('stage_2_filters', {}).get('technical_indicators', {}).get('rsi', {})
        rsi_value = tech.get('rsi')
        if rsi_value is not None:
            neutral_min = rsi_cfg.get('neutral_range_min', 45)
            neutral_max = rsi_cfg.get('neutral_range_max', 60)
            oversold = rsi_cfg.get('oversold_threshold', 30)
            overbought = rsi_cfg.get('overbought_threshold', 70)
            if neutral_min <= rsi_value <= neutral_max:
                components.append(1.0)
            elif oversold < rsi_value < overbought:
                components.append(0.7)
            else:
                components.append(0.2)

        macd_hist = tech.get('macd_histogram')
        if macd_hist is not None:
            components.append(1.0 if macd_hist > 0 else 0.3)

        price_above_ma200 = momentum.get('price_above_ma200')
        if price_above_ma200 is not None:
            components.append(1.0 if price_above_ma200 else 0.3)

        adx_value = tech.get('adx')
        if adx_value is not None:
            components.append(1.0 if adx_value >= 20 else 0.4)

        bb_width = tech.get('bb_width')
        squeeze_threshold = self.raw_config.get('stage_2_filters', {}).get('technical_indicators', {}).get('bollinger_bands', {}).get('squeeze_threshold', 0.05)
        if bb_width is not None:
            if bb_width >= squeeze_threshold:
                components.append(1.0)
            else:
                components.append(0.5)

        atr_pct = context.atr_percentage
        baseline_cfg = self.raw_config.get('stage_1_filters', {}).get('baseline_volatility', {})
        atr_min = baseline_cfg.get('min_atr_pct_of_price', self.config.stage_1_filters.min_atr_pct_of_price)
        atr_max = baseline_cfg.get('max_atr_pct_of_price', self.config.stage_1_filters.max_atr_pct_of_price)
        if atr_pct is not None and atr_pct > 0:
            if atr_min <= atr_pct <= atr_max:
                components.append(1.0)
            else:
                # Penalize softly when just outside the band
                distance = min(abs(atr_pct - atr_min), abs(atr_pct - atr_max))
                components.append(max(0.1, 1 - distance / max(atr_max, 1)))

        roc_20 = momentum.get('roc_20')
        if roc_20 is not None:
            if roc_20 > 2:
                components.append(1.0)
            elif roc_20 > 0:
                components.append(0.7)
            else:
                components.append(0.2)

        return float(np.mean(components)) if components else 0.0

    def _calculate_fundamental_score(self, context: Optional[StockContext]) -> float:
        """Calculate fundamental score (0-1) using configured thresholds."""
        if context is None:
            return 0.0

        fundamentals = context.fundamentals or {}
        ratios_cfg: FundamentalRatios = self.config.fundamental_ratios
        components: List[float] = []

        pe = fundamentals.get('pe_ratio')
        if pe is not None and pe > 0:
            if ratios_cfg.pe_min <= pe <= ratios_cfg.pe_max:
                components.append(1.0)
            elif pe > ratios_cfg.pe_max:
                components.append(max(0.2, 1 - (pe - ratios_cfg.pe_max) / max(ratios_cfg.pe_max, 1)))
            else:
                components.append(max(0.2, pe / max(ratios_cfg.pe_min, 1)))

        pb = fundamentals.get('pb_ratio')
        if pb is not None and pb > 0:
            if ratios_cfg.pb_min <= pb <= ratios_cfg.pb_max:
                components.append(1.0)
            elif pb > ratios_cfg.pb_max:
                components.append(max(0.2, 1 - (pb - ratios_cfg.pb_max) / max(ratios_cfg.pb_max, 1)))
            else:
                components.append(max(0.2, pb / max(ratios_cfg.pb_min, 1)))

        roe = fundamentals.get('roe')
        if roe is not None:
            components.append(min(1.0, roe / max(ratios_cfg.roe_min, 1)))

        debt_equity = fundamentals.get('debt_to_equity')
        if debt_equity is not None and debt_equity >= 0:
            if debt_equity <= ratios_cfg.debt_to_equity_max:
                components.append(1.0)
            else:
                components.append(max(0.1, 1 - (debt_equity - ratios_cfg.debt_to_equity_max)))

        current_ratio = fundamentals.get('current_ratio')
        if current_ratio is not None and current_ratio > 0:
            if current_ratio >= ratios_cfg.current_ratio_min:
                components.append(1.0)
            else:
                components.append(max(0.2, current_ratio / ratios_cfg.current_ratio_min))

        peg = fundamentals.get('peg_ratio')
        if peg is not None and peg > 0:
            components.append(1.0 if peg <= 1.5 else max(0.2, 1 - (peg - 1.5)))

        profit_margin = fundamentals.get('profit_margin') or fundamentals.get('net_margin')
        if profit_margin is not None:
            components.append(min(1.0, profit_margin / 15))

        revenue_growth = fundamentals.get('revenue_growth')
        if revenue_growth is not None:
            components.append(1.0 if revenue_growth >= 10 else max(0.2, revenue_growth / 10))

        earnings_growth = fundamentals.get('earnings_growth')
        if earnings_growth is not None:
            components.append(1.0 if earnings_growth >= 10 else max(0.2, earnings_growth / 10))

        roa_value = fundamentals.get('roa')
        if roa_value is not None:
            components.append(min(1.0, roa_value / 12))

        operating_margin = fundamentals.get('operating_margin')
        if operating_margin is not None:
            components.append(min(1.0, operating_margin / 20))

        dividend_yield = fundamentals.get('dividend_yield')
        if dividend_yield is not None:
            components.append(min(1.0, dividend_yield / 5))

        return float(np.mean(components)) if components else 0.0

    def _calculate_risk_score(self, context: Optional[StockContext]) -> float:
        """Calculate risk score (0-1) balancing volatility, beta, and drawdown."""
        if context is None:
            return 0.0

        risk_cfg: RiskMetrics = self.config.risk_metrics
        metrics = context.risk_metrics or {}
        components: List[float] = []

        beta = metrics.get('beta')
        if beta is not None:
            if risk_cfg.beta_min <= beta <= risk_cfg.beta_max:
                components.append(1.0)
            else:
                components.append(max(0.2, 1 - abs(beta - 1.0)))

        vol_annual = metrics.get('volatility_annual_pct')
        if vol_annual is not None:
            if vol_annual <= risk_cfg.volatility_max_annual:
                components.append(1.0)
            else:
                over = vol_annual - risk_cfg.volatility_max_annual
                components.append(max(0.2, 1 - over / max(risk_cfg.volatility_max_annual, 1)))

        sharpe = metrics.get('sharpe_ratio')
        if sharpe is not None:
            if sharpe >= risk_cfg.sharpe_ratio_min:
                components.append(1.0)
            else:
                components.append(max(0.2, sharpe / max(risk_cfg.sharpe_ratio_min, 0.1)))

        max_drawdown = metrics.get('max_drawdown_pct')
        if max_drawdown is not None:
            components.append(max(0.1, 1 - (max_drawdown / 40)))

        atr_pct = metrics.get('atr_pct') or context.atr_percentage
        atr_min = self.config.stage_1_filters.min_atr_pct_of_price
        atr_max = self.config.stage_1_filters.max_atr_pct_of_price
        if atr_pct is not None and atr_pct > 0:
            if atr_min <= atr_pct <= atr_max:
                components.append(1.0)
            else:
                components.append(max(0.1, 1 - abs(atr_pct - ((atr_min + atr_max) / 2)) / max(atr_max, 1)))

        cap_category = context.market_cap_category
        if cap_category:
            components.append({'large_cap': 1.0, 'mid_cap': 0.8, 'small_cap': 0.5}.get(cap_category, 0.6))

        return float(np.mean(components)) if components else 0.0

    def _calculate_momentum_score(self, context: Optional[StockContext]) -> float:
        """Calculate momentum score (0-1) including sector-relative strength."""
        if context is None:
            return 0.0

        momentum = context.momentum_metrics or {}
        components: List[float] = []

        for period in [20, 63, 126]:
            roc_value = momentum.get(f'roc_{period}')
            if roc_value is not None:
                if roc_value > 5:
                    components.append(1.0)
                elif roc_value > 0:
                    components.append(0.7)
                else:
                    components.append(0.2)

        price_vs_ma200 = momentum.get('price_vs_ma200_pct')
        if price_vs_ma200 is not None:
            components.append(1.0 if price_vs_ma200 > 0 else max(0.2, 1 + price_vs_ma200 / 10))

        sector_strength = momentum.get('sector_relative_strength')
        if sector_strength is not None:
            components.append(1.0 if sector_strength >= 0 else max(0.2, 1 + sector_strength / 10))

        universe_strength = momentum.get('relative_strength_vs_universe')
        if universe_strength is not None:
            components.append(1.0 if universe_strength >= 0 else max(0.2, 1 + universe_strength / 10))

        distance_resistance = momentum.get('distance_from_52w_high_pct') or context.distance_from_resistance_pct
        min_distance = self.config.selection.min_distance_from_resistance_pct
        if distance_resistance is not None and distance_resistance >= 0:
            if distance_resistance >= min_distance:
                components.append(1.0)
            else:
                components.append(max(0.2, distance_resistance / max(min_distance, 1)))

        return float(np.mean(components)) if components else 0.0

    def _calculate_volume_score(self, context: Optional[StockContext]) -> float:
        """Calculate volume score (0-1) blending surge, OBV trend, and liquidity."""
        if context is None:
            return 0.0

        signals = context.volume_signals or {}
        components: List[float] = []

        volume_ratio = signals.get('volume_ratio')
        if volume_ratio is not None:
            if volume_ratio >= 1.5:
                components.append(1.0)
            elif volume_ratio >= 1.0:
                components.append(0.7)
            else:
                components.append(max(0.2, volume_ratio / 1.0))

        obv_trend = signals.get('obv_trend')
        if obv_trend is not None:
            components.append(1.0 if obv_trend > 0 else 0.4)

        mfi = signals.get('mfi')
        if mfi is not None:
            if 45 <= mfi <= 65:
                components.append(1.0)
            elif 35 <= mfi <= 70:
                components.append(0.7)
            else:
                components.append(0.3)

        liquidity_score = context.liquidity_score
        if liquidity_score is not None:
            components.append(min(1.0, max(0.0, liquidity_score)))

        trades_per_day = context.trades_per_day
        if trades_per_day is not None and trades_per_day > 0:
            components.append(min(1.0, trades_per_day / 10000))

        return float(np.mean(components)) if components else 0.0
    
    # ---------------------------------------------------------------------
    # Helper methods for data preparation and scoring
    # ---------------------------------------------------------------------

    def _safe_numeric(self, value: Any) -> Optional[float]:
        """Convert a value to float if possible."""
        if value is None:
            return None
        try:
            # bool inherits from int, guard against accidental booleans
            if isinstance(value, bool):
                return float(int(value))
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fetch_historical_df(self, session, symbol: str, start_date) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data for a symbol."""
        if session is None:
            return None
        try:
            records = session.query(HistoricalData).filter(
                HistoricalData.symbol == symbol,
                HistoricalData.date >= start_date
            ).order_by(HistoricalData.date.asc()).all()
            if not records:
                return None
            frame = pd.DataFrame([
                {
                    'date': rec.date,
                    'open': float(rec.open),
                    'high': float(rec.high),
                    'low': float(rec.low),
                    'close': float(rec.close),
                    'volume': float(rec.volume)
                }
                for rec in records
            ])
            frame['date'] = pd.to_datetime(frame['date'])
            frame = frame.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            return frame
        except Exception as exc:
            logger.debug(f"Failed to fetch historical data for {symbol}: {exc}")
            return None

    def _compute_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Compute core technical indicators using pandas_ta when available."""
        indicators: Dict[str, Optional[float]] = {}
        if df is None or df.empty:
            return indicators

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        rsi_period = self.raw_config.get('stage_2_filters', {}).get('technical_indicators', {}).get('rsi', {}).get('period', 14)
        atr_period = self.raw_config.get('stage_1_filters', {}).get('baseline_volatility', {}).get('atr_period', 14)
        macd_cfg = self.raw_config.get('stage_2_filters', {}).get('technical_indicators', {}).get('macd', {})
        macd_fast = macd_cfg.get('fast_period', 12)
        macd_slow = macd_cfg.get('slow_period', 26)
        macd_signal = macd_cfg.get('signal_period', 9)
        bb_cfg = self.raw_config.get('stage_2_filters', {}).get('technical_indicators', {}).get('bollinger_bands', {})
        bb_period = bb_cfg.get('period', 20)
        bb_std = bb_cfg.get('std_dev', 2.0)
        ma_cfg = self.raw_config.get('stage_2_filters', {}).get('technical_indicators', {}).get('moving_averages', {})
        adx_period = self.raw_config.get('stage_2_filters', {}).get('technical_indicators', {}).get('adx', {}).get('period', 14)
        mfi_period = self.raw_config.get('stage_2_filters', {}).get('technical_indicators', {}).get('mfi', {}).get('period', 14)

        try:
            if ta:
                rsi_series = ta.rsi(close, length=rsi_period)
                indicators['rsi'] = self._last_value(rsi_series)

                macd_df = ta.macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
                if macd_df is not None and not macd_df.empty:
                    indicators['macd'] = self._last_value(macd_df.iloc[:, 0])
                    indicators['macd_histogram'] = self._last_value(macd_df.iloc[:, 1])
                    indicators['macd_signal'] = self._last_value(macd_df.iloc[:, 2])

                bb_df = ta.bbands(close, length=bb_period, std=bb_std)
                if bb_df is not None and not bb_df.empty:
                    indicators['bb_lower'] = self._last_value(bb_df.iloc[:, 0])
                    indicators['bb_middle'] = self._last_value(bb_df.iloc[:, 1])
                    indicators['bb_upper'] = self._last_value(bb_df.iloc[:, 2])
                    indicators['bb_width'] = self._last_value(bb_df.iloc[:, 3])

                for period in [5, 10, 20, 50, 100, 200]:
                    indicators[f'sma_{period}'] = self._last_value(ta.sma(close, length=period))
                for period in [12, 26, 50]:
                    indicators[f'ema_{period}'] = self._last_value(ta.ema(close, length=period))

                atr_series = ta.atr(high=high, low=low, close=close, length=atr_period)
                indicators['atr'] = self._last_value(atr_series)
                if indicators['atr'] and indicators['atr'] > 0 and self._last_value(close) > 0:
                    indicators['atr_pct'] = indicators['atr'] / self._last_value(close) * 100

                adx_df = ta.adx(high=high, low=low, close=close, length=adx_period)
                if adx_df is not None and not adx_df.empty:
                    indicators['adx'] = self._last_value(adx_df.iloc[:, 0])

                obv_series = ta.obv(close=close, volume=volume)
                indicators['obv'] = self._last_value(obv_series)

                mfi_series = ta.mfi(high=high, low=low, close=close, volume=volume, length=mfi_period)
                indicators['mfi'] = self._last_value(mfi_series)

                roc_20 = ta.roc(close, length=20)
                indicators['roc_20'] = self._last_value(roc_20)
                roc_63 = ta.roc(close, length=63)
                indicators['roc_63'] = self._last_value(roc_63)
                roc_126 = ta.roc(close, length=126)
                indicators['roc_126'] = self._last_value(roc_126)
            else:  # pragma: no cover - fallback logic when pandas_ta not installed
                indicators['rsi'] = self._simple_rsi(close, rsi_period)
                indicators['atr'] = self._simple_atr(high, low, close, atr_period)
                if indicators['atr'] and self._last_value(close) > 0:
                    indicators['atr_pct'] = indicators['atr'] / self._last_value(close) * 100
                indicators['sma_200'] = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else None
                indicators['sma_50'] = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else None
                indicators['roc_20'] = close.pct_change(periods=20).iloc[-1] * 100 if len(close) > 20 else None
                indicators['roc_63'] = close.pct_change(periods=63).iloc[-1] * 100 if len(close) > 63 else None
                indicators['roc_126'] = close.pct_change(periods=126).iloc[-1] * 100 if len(close) > 126 else None

        except Exception as exc:
            logger.debug(f"Failed to compute pandas_ta indicators: {exc}")

        return indicators

    def _compute_volume_signals(self, df: pd.DataFrame, tech: Dict[str, Optional[float]], context: StockContext) -> Dict[str, Optional[float]]:
        """Derive volume-based signals."""
        signals: Dict[str, Optional[float]] = {}
        if df is None or df.empty:
            return signals

        recent_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        signals['volume_ratio'] = recent_volume / avg_volume if avg_volume else None
        signals['volume_surge_flag'] = 1 if signals['volume_ratio'] and signals['volume_ratio'] >= 1.5 else 0

        if ta and 'obv' not in tech:
            obv_series = ta.obv(close=df['close'], volume=df['volume'])
            tech['obv'] = self._last_value(obv_series)

        obv_series = None
        if ta:
            try:
                obv_series = ta.obv(close=df['close'], volume=df['volume'])
            except Exception:
                obv_series = None

        if obv_series is not None and len(obv_series.dropna()) > 5:
            signals['obv_trend'] = self._series_trend(obv_series.tail(20))
        else:
            signals['obv_trend'] = None

        signals['mfi'] = tech.get('mfi')
        signals['avg_volume_20d'] = avg_volume
        signals['recent_volume'] = recent_volume

        return signals

    def _compute_momentum_metrics(self, df: pd.DataFrame, tech: Dict[str, Optional[float]], context: StockContext) -> Dict[str, Optional[float]]:
        """Calculate momentum and trend metrics."""
        momentum: Dict[str, Optional[float]] = {}
        if df is None or df.empty:
            return momentum

        close = df['close']
        latest_close = close.iloc[-1]

        for period in [5, 20, 63, 126, 252]:
            if len(close) > period:
                pct = close.pct_change(periods=period).iloc[-1]
                momentum[f'roc_{period}'] = pct * 100 if not np.isnan(pct) else None
            else:
                momentum[f'roc_{period}'] = None

        sma_200 = tech.get('sma_200')
        if sma_200 and sma_200 > 0:
            momentum['price_vs_ma200_pct'] = (latest_close / sma_200 - 1) * 100
            momentum['price_above_ma200'] = 1 if latest_close > sma_200 else 0
        else:
            momentum['price_vs_ma200_pct'] = None
            momentum['price_above_ma200'] = None

        sma_50 = tech.get('sma_50')
        if sma_50 and sma_50 > 0:
            momentum['price_vs_ma50_pct'] = (latest_close / sma_50 - 1) * 100

        window_52w = min(len(close), 252)
        if window_52w > 0:
            rolling_high = close.tail(window_52w).max()
            if rolling_high and rolling_high > 0:
                momentum['distance_from_52w_high_pct'] = (rolling_high - latest_close) / latest_close * 100

        return momentum

    def _compute_risk_metrics(self, df: pd.DataFrame, context: StockContext) -> Dict[str, Optional[float]]:
        """Calculate risk metrics such as volatility, sharpe, and drawdown."""
        metrics: Dict[str, Optional[float]] = {}
        if df is None or df.empty:
            metrics['beta'] = self._safe_numeric(getattr(context.stock, 'beta', None))
            metrics['atr_pct'] = context.atr_percentage
            return metrics

        close = df['close']
        returns = close.pct_change().dropna()
        if returns.empty:
            metrics['beta'] = self._safe_numeric(getattr(context.stock, 'beta', None))
            metrics['atr_pct'] = context.atr_percentage
            return metrics

        daily_std = returns.std()
        annual_std = daily_std * math.sqrt(252)
        metrics['volatility_daily_pct'] = daily_std * 100 if not np.isnan(daily_std) else None
        metrics['volatility_annual_pct'] = annual_std * 100 if not np.isnan(annual_std) else None

        risk_free = getattr(self.config.risk_metrics, 'risk_free_rate', 6.5)
        mean_daily = returns.mean()
        denominator = daily_std if daily_std else None
        if denominator and denominator > 0:
            sharpe = ((mean_daily * 252) - (risk_free / 100)) / (denominator * math.sqrt(252))
            metrics['sharpe_ratio'] = sharpe
        else:
            metrics['sharpe_ratio'] = None

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        if not drawdown.empty:
            min_drawdown = drawdown.min()
            metrics['max_drawdown_pct'] = abs(min_drawdown * 100) if not np.isnan(min_drawdown) else None
        else:
            metrics['max_drawdown_pct'] = None

        metrics['beta'] = self._safe_numeric(getattr(context.stock, 'beta', None))
        metrics['atr_pct'] = context.atr_percentage
        metrics['historical_volatility_pct'] = metrics['volatility_annual_pct']

        return metrics

    def _compute_liquidity_score(self, context: StockContext) -> Optional[float]:
        """Compute liquidity score using volume and spread weighting."""
        volume_norm = self.raw_config.get('liquidity_scoring', {}).get('volume_normalization', 1_000_000)
        weights = self.raw_config.get('liquidity_scoring', {}).get('weights', {'volume': 0.7, 'spread': 0.3})
        spread_multiplier = self.raw_config.get('liquidity_scoring', {}).get('spread_multiplier', 10)

        weight_volume = weights.get('volume', 0.7)
        weight_spread = weights.get('spread', 0.3)
        weight_sum = weight_volume + weight_spread
        if weight_sum == 0:
            weight_volume = 0.7
            weight_spread = 0.3
            weight_sum = 1.0
        weight_volume /= weight_sum
        weight_spread /= weight_sum

        avg_volume = context.avg_volume_20d or context.volume
        if not avg_volume or avg_volume <= 0:
            return None
        volume_component = min(1.0, avg_volume / volume_norm) if volume_norm else 0.0

        spread = context.bid_ask_spread
        if spread is None:
            spread_component = 0.5
        else:
            spread_pct = spread / 100 if spread > 1 else spread
            spread_component = max(0.0, 1 - min(1.0, spread_pct * spread_multiplier))

        liquidity_score = weight_volume * volume_component + weight_spread * spread_component
        return round(liquidity_score, 4)

    def _derive_market_cap_category(self, stock: Any) -> Optional[str]:
        """Infer market cap category from configuration thresholds."""
        market_cap = self._safe_numeric(getattr(stock, 'market_cap', None))
        if market_cap is None:
            return None

        categories = self.raw_config.get('stage_1_filters', {}).get('market_cap_categories', {})
        large_cfg = categories.get('large_cap', {})
        mid_cfg = categories.get('mid_cap', {})
        small_cfg = categories.get('small_cap', {})

        if large_cfg and market_cap >= large_cfg.get('minimum', 20000):
            return large_cfg.get('label', 'large_cap')
        if mid_cfg:
            min_val = mid_cfg.get('minimum', 5000)
            max_val = mid_cfg.get('maximum', 20000)
            if market_cap >= min_val and market_cap <= max_val:
                return mid_cfg.get('label', 'mid_cap')
        if small_cfg and market_cap <= small_cfg.get('maximum', 5000):
            return small_cfg.get('label', 'small_cap')
        return None

    def _extract_fundamentals(self, stock: Any) -> Dict[str, Optional[float]]:
        """Collect fundamental ratios from the stock record."""
        fields = [
            'pe_ratio', 'pb_ratio', 'peg_ratio', 'roe', 'roa', 'debt_to_equity',
            'profit_margin', 'operating_margin', 'net_margin', 'current_ratio',
            'quick_ratio', 'revenue_growth', 'earnings_growth', 'dividend_yield'
        ]
        fundamentals = {field: self._safe_numeric(getattr(stock, field, None)) for field in fields}
        fundamentals['market_cap'] = self._safe_numeric(getattr(stock, 'market_cap', None))
        fundamentals['eps'] = self._safe_numeric(getattr(stock, 'eps', None))
        fundamentals['book_value'] = self._safe_numeric(getattr(stock, 'book_value', None))
        return fundamentals

    def _attach_sector_performance(self, contexts: List[StockContext]) -> None:
        """Calculate sector level performance metrics for momentum comparisons."""
        sector_returns: Dict[str, List[float]] = defaultdict(list)
        roc_key = 'roc_63'
        for context in contexts:
            value = context.momentum_metrics.get(roc_key) if context.momentum_metrics else None
            if value is not None and context.sector:
                sector_returns[context.sector].append(value)

        sector_avg = {sector: float(np.nanmean(values)) for sector, values in sector_returns.items() if values}
        universe_returns = [val for values in sector_returns.values() for val in values if val is not None]
        universe_avg = float(np.nanmean(universe_returns)) if universe_returns else None

        for context in contexts:
            if context.sector in sector_avg:
                context.sector_performance = sector_avg[context.sector]
                if context.momentum_metrics is not None:
                    context.momentum_metrics['sector_relative_strength'] = (
                        context.momentum_metrics.get(roc_key) - sector_avg[context.sector]
                    ) if context.momentum_metrics.get(roc_key) is not None else None
            if universe_avg is not None and context.momentum_metrics is not None:
                roc_value = context.momentum_metrics.get(roc_key)
                context.momentum_metrics['relative_strength_vs_universe'] = (
                    roc_value - universe_avg if roc_value is not None else None
                )

    def _log_rejected_symbols(self, stage1: List[StockScore], stage2: List[StockScore], final: List[StockScore]) -> None:
        """Log rejected symbols with reasons for debugging."""
        def summarize(label: str, items: List[StockScore]) -> None:
            if not items:
                return
            logger.info(f"Rejected at {label}: {len(items)}")
            for stock in items[:10]:  # limit noise
                reasons = ', '.join(stock.reject_reasons[:3]) if stock.reject_reasons else 'No reason recorded'
                logger.info(f"  - {stock.symbol}: {reasons}")

        summarize('Stage 1', stage1)
        summarize('Stage 2', stage2)
        summarize('Final selection', final)

    def _log_top_stocks(self, stocks: List[StockScore], top_n: int) -> None:
        """Log top scoring stocks per category for quick inspection."""
        if not stocks:
            return

        def top_by(attribute: str) -> List[StockScore]:
            return sorted(stocks, key=lambda s: getattr(s, attribute), reverse=True)[:top_n]

        categories = {
            'technical_score': 'Technical',
            'fundamental_score': 'Fundamental',
            'risk_score': 'Risk',
            'momentum_score': 'Momentum',
            'volume_score': 'Volume',
            'total_score': 'Total'
        }

        for attr, label in categories.items():
            leaders = top_by(attr)
            if not leaders:
                continue
            logger.info(f"Top {min(top_n, len(leaders))} by {label} score:")
            for stock in leaders:
                logger.info(f"  {stock.symbol}: {getattr(stock, attr):.1f}")

    def _last_value(self, series: Optional[pd.Series]) -> Optional[float]:
        """Return the last finite value from a pandas Series."""
        if series is None or series.empty:
            return None
        value = series.iloc[-1]
        if pd.isna(value):
            value = series.dropna().iloc[-1] if not series.dropna().empty else None
        return float(value) if value is not None else None

    def _series_trend(self, series: pd.Series) -> Optional[float]:
        """Calculate normalized slope of a series."""
        clean = series.dropna()
        if clean.empty:
            return None
        y = clean.values
        x = np.arange(len(y))
        if len(x) < 2:
            return None
        slope, _ = np.polyfit(x, y, 1)
        return float(slope / (abs(y).mean() + 1e-9))

    def _simple_rsi(self, prices: pd.Series, period: int) -> Optional[float]:
        """Fallback RSI calculation."""
        if len(prices) <= period:
            return None
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        value = rsi.iloc[-1]
        return float(value) if not pd.isna(value) else None

    def _simple_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> Optional[float]:
        """Fallback ATR calculation."""
        if len(close) <= period:
            return None
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        value = atr.iloc[-1]
        return float(value) if not pd.isna(value) else None
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get current filter statistics."""
        return self.filter_stats.copy()
    
    def reset_statistics(self):
        """Reset filter statistics."""
        self.filter_stats = {
            'total_processed': 0,
            'stage1_passed': 0,
            'stage2_passed': 0,
            'final_selected': 0,
            'execution_time': 0.0
        }


# Global service instance
_enhanced_filtering_service = None


def get_enhanced_filtering_service() -> EnhancedStockFilteringService:
    """Get the global enhanced filtering service instance."""
    global _enhanced_filtering_service
    if _enhanced_filtering_service is None:
        _enhanced_filtering_service = EnhancedStockFilteringService()
    return _enhanced_filtering_service
