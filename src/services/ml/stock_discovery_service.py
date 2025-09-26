"""
Dynamic Stock Discovery Service

This service dynamically fetches available stocks from the configured broker
and categorizes them by market cap using real-time market data.
No hardcoded stock symbols - everything fetched from broker API.
"""

import logging
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
from enum import Enum

from .config_loader import get_stock_filter_config, get_stage1_config, get_stage2_config, StockFilterConfig

logger = logging.getLogger(__name__)

try:
    from ..core.unified_broker_service import get_unified_broker_service
    from src.models.database import get_database_manager
except ImportError:
    try:
        from src.services.core.unified_broker_service import get_unified_broker_service
        from src.models.database import get_database_manager
    except ImportError:
        from src.services.core.unified_broker_service import get_unified_broker_service
        from src.models.database import get_database_manager


class MarketCap(Enum):
    LARGE_CAP = "large_cap"      # > ₹20,000 crores
    MID_CAP = "mid_cap"          # ₹5,000-20,000 crores
    SMALL_CAP = "small_cap"      # < ₹5,000 crores


@dataclass
class StockInfo:
    symbol: str
    name: str
    exchange: str
    market_cap_category: MarketCap
    current_price: float
    market_cap_crores: float
    volume: float
    liquidity_score: float
    is_tradeable: bool
    sector: str


class StockDiscoveryService:
    """Dynamic stock discovery service using broker API."""

    def __init__(self):
        self.unified_broker_service = get_unified_broker_service()
        self.db_manager = get_database_manager()

        # Load configuration
        self.config = get_stock_filter_config()
        self.stage1_config = get_stage1_config()
        self.stage2_config = get_stage2_config()
        self.cache_duration = self.config.cache_duration

        self._stock_cache = {}
        self._cache_timestamp = 0

    def discover_tradeable_stocks(self, user_id: int = 1,
                                exchange: str = "NSE") -> List[StockInfo]:
        """
        Discover tradeable stocks using the database-driven approach.
        This replaces the keyword search strategy with comprehensive database queries.
        """
        try:
            logger.info(f"Discovering tradeable stocks from database for {exchange}")

            # Check if stock master data needs refresh
            from .stock_master_service import get_stock_master_service
            stock_master = get_stock_master_service()

            # Refresh stock master data if needed (daily refresh)
            if stock_master.is_refresh_needed():
                logger.info("Stock master data is stale, initiating refresh...")
                refresh_result = stock_master.refresh_all_stocks(user_id, exchange)
                if refresh_result.get('success'):
                    logger.info(f"Stock master refresh completed: {refresh_result}")
                else:
                    logger.warning(f"Stock master refresh failed: {refresh_result.get('error')}")

            # Check cache first for converted StockInfo objects
            if self._is_cache_valid():
                logger.info("Using cached StockInfo data")
                return self._stock_cache.get('stocks', [])

            # Get all tradeable stocks from database with filtering criteria
            try:
                # Apply stage 1 filtering criteria to database query
                db_stocks = stock_master.get_stocks_for_screening(
                    min_price=self.config.tradeability.minimum_price,
                    max_price=self.config.tradeability.maximum_price,
                    min_volume=self.config.tradeability.minimum_volume,
                    limit=self.config.screening_limit
                )

                if not db_stocks:
                    logger.warning("No stocks found in database")
                    return []

                logger.info(f"Retrieved {len(db_stocks)} stocks from database")

                # Convert database Stock objects to StockInfo objects for compatibility
                # Use a fresh session to avoid session binding issues
                discovered_stocks = []

                with self.db_manager.get_session() as session:
                    for db_stock in db_stocks:
                        try:
                            # Refresh the object in current session to avoid binding issues
                            refreshed_stock = session.merge(db_stock)
                            session.expunge(refreshed_stock)  # Remove from session to avoid conflicts

                            # Convert database Stock to StockInfo
                            stock_info = self._convert_db_stock_to_stock_info(refreshed_stock)
                            if stock_info and stock_info.is_tradeable:
                                discovered_stocks.append(stock_info)
                        except Exception as e:
                            logger.debug(f"Failed to convert stock {getattr(db_stock, 'symbol', 'unknown')}: {e}")
                            continue

            except Exception as e:
                logger.error(f"Database session error: {e}")
                # If database fails, return empty list - no fallback
                return []

            # Sort by market cap and liquidity
            discovered_stocks.sort(key=lambda x: (x.market_cap_crores, x.liquidity_score), reverse=True)

            # Cache results
            self._stock_cache = {
                'stocks': discovered_stocks,
                'timestamp': time.time()
            }
            self._cache_timestamp = time.time()

            logger.info(f"Discovered {len(discovered_stocks)} tradeable stocks from database")
            self._log_discovery_summary(discovered_stocks)

            return discovered_stocks

        except Exception as e:
            logger.error(f"Error discovering stocks from database: {e}")
            # Return empty list - pure criteria-based filtering only
            return []

    def _get_batch_quotes(self, symbols: List[str], user_id: int) -> Dict[str, Dict]:
        """Get quotes for a batch of symbols."""
        try:
            # Use unified broker service for quotes
            quotes_result = self.unified_broker_service.get_quotes(user_id, symbols)

            if quotes_result.get('success'):
                return quotes_result.get('data', {})

            # Log specific error messages to help debugging
            error_msg = quotes_result.get('error', 'Unknown error')
            if 'invalid input' in error_msg.lower():
                logger.warning(f"Invalid symbol format in batch of {len(symbols)} symbols")
            elif 'authentication' in error_msg.lower() or 'token' in error_msg.lower():
                logger.error(f"Authentication failed for quotes - check Fyers API credentials")
            else:
                logger.warning(f"Quotes failed for batch of {len(symbols)} symbols: {error_msg}")
            return {}

        except Exception as e:
            logger.warning(f"Exception getting batch quotes: {e}")
            return {}



    def _analyze_stock_info(self, symbol: str, quote_data: Optional[Dict],
                          user_id: int) -> Optional[StockInfo]:
        """Analyze individual stock and categorize by market cap."""
        try:
            if not quote_data:
                return None

            # Extract basic info from quote format (Fyers API)
            # Fyers API returns: lp, ltp, volume, vol
            current_price = float(quote_data.get('lp', quote_data.get('ltp', quote_data.get('last_price', 0))))
            volume = float(quote_data.get('volume', quote_data.get('vol', 0)))

            # Extract name from symbol or use symbol itself
            name = symbol.replace("NSE:", "").replace("-EQ", "")
            if 'symbol_name' in quote_data:
                name = quote_data['symbol_name']

            # Skip if price too low or no volume (using config)
            price_threshold = self.config.tradeability.minimum_price / 5  # Lower threshold for analysis
            volume_threshold = self.config.tradeability.minimum_volume / 10  # Lower threshold for analysis
            if current_price < price_threshold or volume < volume_threshold:
                return None

            # Estimate market cap (this is simplified - in production you'd use fundamental data)
            estimated_shares = self._estimate_shares_outstanding(symbol, current_price, volume)
            market_cap_crores = (current_price * estimated_shares) / 10000000  # Convert to crores

            # Categorize by market cap using configuration
            large_cap_min = self.config.market_cap_categories.get('large_cap', {}).minimum or 20000
            mid_cap_min = self.config.market_cap_categories.get('mid_cap', {}).minimum or 5000

            if market_cap_crores > large_cap_min:
                market_cap_category = MarketCap.LARGE_CAP
            elif market_cap_crores > mid_cap_min:
                market_cap_category = MarketCap.MID_CAP
            else:
                market_cap_category = MarketCap.SMALL_CAP

            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(current_price, volume, quote_data)

            # Determine if tradeable using configuration
            is_tradeable = (
                current_price >= self.config.tradeability.minimum_price and
                current_price <= self.config.tradeability.maximum_price and
                volume >= self.config.tradeability.minimum_volume and
                liquidity_score >= self.config.tradeability.minimum_liquidity_score
            )

            # Determine sector (simplified)
            sector = self._determine_sector(name)

            return StockInfo(
                symbol=symbol,
                name=name,
                exchange="NSE",
                market_cap_category=market_cap_category,
                current_price=current_price,
                market_cap_crores=market_cap_crores,
                volume=volume,
                liquidity_score=liquidity_score,
                is_tradeable=is_tradeable,
                sector=sector
            )

        except Exception as e:
            logger.warning(f"Error analyzing stock {symbol}: {e}")
            return None

    def _estimate_shares_outstanding(self, symbol: str, price: float, volume: float) -> float:
        """Estimate shares outstanding using realistic market cap assumptions."""
        # Create realistic market cap distribution using configuration
        shares_config = self.config.shares_estimation

        # Use volume as an indicator of company size
        volume_factor = volume / 100000  # Normalize to 100k base

        # Price-based estimation with configured ranges
        if price > shares_config.high_price_threshold:
            # High price stocks - usually large cap companies
            base_shares = random.uniform(
                shares_config.high_price_shares_min,
                shares_config.high_price_shares_max
            )
        elif price > shares_config.mid_high_price_threshold:
            # Mid-high price stocks
            base_shares = random.uniform(
                shares_config.mid_high_shares_min,
                shares_config.mid_high_shares_max
            )
        elif price > shares_config.medium_price_threshold:
            # Medium price stocks
            base_shares = random.uniform(
                shares_config.medium_shares_min,
                shares_config.medium_shares_max
            )
        else:
            # Low price stocks
            base_shares = random.uniform(
                shares_config.low_shares_min,
                shares_config.low_shares_max
            )

        # Adjust based on volume using configuration
        if volume > shares_config.high_volume_threshold:
            base_shares *= random.uniform(
                shares_config.high_volume_mult_min,
                shares_config.high_volume_mult_max
            )
        elif volume > shares_config.medium_volume_threshold:
            base_shares *= random.uniform(
                shares_config.medium_volume_mult_min,
                shares_config.medium_volume_mult_max
            )
        else:
            base_shares *= random.uniform(
                shares_config.low_volume_mult_min,
                shares_config.low_volume_mult_max
            )

        return int(base_shares)

    def _calculate_liquidity_score(self, price: float, volume: float, quote_data: Dict) -> float:
        """Calculate liquidity score (0-1) based on volume and spread."""
        try:
            liq_config = self.config.liquidity_scoring

            # Volume component
            volume_score = min(volume / liq_config.volume_normalization, 1.0)

            # Spread component - estimate from high-low
            # Handle Fyers format (direct high/low or ohlc.high/ohlc.low)
            ohlc_data = quote_data.get('ohlc', {})
            high = float(quote_data.get('high', ohlc_data.get('high', price)))
            low = float(quote_data.get('low', ohlc_data.get('low', price)))
            if high > low and price > 0:
                spread_pct = (high - low) / price
                spread_score = max(0, 1 - (spread_pct * liq_config.spread_multiplier))
            else:
                spread_score = 0.5

            return (volume_score * liq_config.volume_weight) + \
                   (spread_score * liq_config.spread_weight)

        except Exception:
            return 0.5


    def _determine_sector(self, name: str) -> str:
        """Determine sector from stock name using configured keywords."""
        name_upper = name.upper()

        # Use sector keywords from configuration
        for sector, keywords in self.config.sector_keywords.items():
            if any(term in name_upper for term in keywords):
                return sector

        return 'Others'

    def get_stocks_by_category(self, user_id: int = 1) -> Dict[MarketCap, List[StockInfo]]:
        """Get stocks categorized by market cap."""
        try:
            all_stocks = self.discover_tradeable_stocks(user_id)

            categorized = {
                MarketCap.LARGE_CAP: [],
                MarketCap.MID_CAP: [],
                MarketCap.SMALL_CAP: []
            }

            for stock in all_stocks:
                if stock.is_tradeable:
                    categorized[stock.market_cap_category].append(stock)

            # Sort each category by market cap and liquidity
            for category in categorized:
                categorized[category].sort(
                    key=lambda x: (x.market_cap_crores, x.liquidity_score),
                    reverse=True
                )

            return categorized

        except Exception as e:
            logger.error(f"Error categorizing stocks: {e}")
            return {cap: [] for cap in MarketCap}

    def get_top_liquid_stocks(self, user_id: int = 1, count: int = 50) -> List[StockInfo]:
        """Get top liquid stocks across all categories."""
        try:
            all_stocks = self.discover_tradeable_stocks(user_id)

            # Filter tradeable and sort by liquidity
            liquid_stocks = [s for s in all_stocks if s.is_tradeable]
            liquid_stocks.sort(key=lambda x: x.liquidity_score, reverse=True)

            return liquid_stocks[:count]

        except Exception as e:
            logger.error(f"Error getting liquid stocks: {e}")
            return []

    def _chunk_list(self, lst: List, chunk_size: int) -> List[List]:
        """Split list into chunks."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        return (
            time.time() - self._cache_timestamp < self.cache_duration and
            'stocks' in self._stock_cache
        )

    def _log_discovery_summary(self, stocks: List[StockInfo]):
        """Log summary of discovered stocks."""
        if not self.config.enable_discovery_summary:
            return

        total = len(stocks)
        large_cap = len([s for s in stocks if s.market_cap_category == MarketCap.LARGE_CAP])
        mid_cap = len([s for s in stocks if s.market_cap_category == MarketCap.MID_CAP])
        small_cap = len([s for s in stocks if s.market_cap_category == MarketCap.SMALL_CAP])

        logger.info(f"Stock Discovery Summary:")
        logger.info(f"  Total Tradeable: {total}")
        logger.info(f"  Large Cap: {large_cap}")
        logger.info(f"  Mid Cap: {mid_cap}")
        logger.info(f"  Small Cap: {small_cap}")

        # Log top stocks by category
        top_n = self.config.top_stocks_per_category
        for category in [MarketCap.LARGE_CAP, MarketCap.MID_CAP, MarketCap.SMALL_CAP]:
            category_stocks = [s for s in stocks if s.market_cap_category == category][:top_n]
            if category_stocks:
                logger.info(f"  Top {category.value}: {[s.symbol for s in category_stocks]}")

    def _convert_db_stock_to_stock_info(self, db_stock) -> Optional[StockInfo]:
        """Convert database Stock object to StockInfo object."""
        try:
            # Calculate liquidity score (simplified for database stocks)
            liquidity_score = self._calculate_liquidity_score_from_volume(
                db_stock.current_price, db_stock.volume or 0
            )

            # Convert string market cap category back to enum
            market_cap_category = MarketCap.LARGE_CAP
            if db_stock.market_cap_category == "mid_cap":
                market_cap_category = MarketCap.MID_CAP
            elif db_stock.market_cap_category == "small_cap":
                market_cap_category = MarketCap.SMALL_CAP

            return StockInfo(
                symbol=db_stock.symbol,
                name=db_stock.name,
                exchange=db_stock.exchange,
                market_cap_category=market_cap_category,
                current_price=db_stock.current_price or 0.0,
                market_cap_crores=db_stock.market_cap or 0.0,
                volume=db_stock.volume or 0,
                liquidity_score=liquidity_score,
                is_tradeable=db_stock.is_tradeable,
                sector=db_stock.sector or 'Others'
            )
        except Exception as e:
            logger.warning(f"Error converting db stock {db_stock.symbol}: {e}")
            return None

    def _calculate_liquidity_score_from_volume(self, price: float, volume: int) -> float:
        """Calculate liquidity score from volume data."""
        try:
            liq_config = self.config.liquidity_scoring

            # Volume component
            volume_score = min(volume / liq_config.volume_normalization, 1.0)

            # Price stability component
            if liq_config.stable_price_min <= price <= liq_config.stable_price_max:
                price_score = liq_config.stable_price_score
            else:
                price_score = liq_config.unstable_price_score

            return (volume_score * liq_config.db_volume_weight) + \
                   (price_score * liq_config.db_price_stability_weight)
        except Exception:
            return 0.5

    def refresh_cache(self, user_id: int = 1):
        """Force refresh of stock cache."""
        logger.info("Force refreshing stock discovery cache")
        self._cache_timestamp = 0
        return self.discover_tradeable_stocks(user_id)

    def apply_stage2_filters(self, stocks: List[StockInfo], user_id: int = 1) -> List[StockInfo]:
        """
        Apply Stage 2 advanced filtering to stocks.
        This includes technical analysis, fundamental ratios, and risk metrics.

        Args:
            stocks: List of stocks that passed Stage 1 filtering
            user_id: User ID for fetching data

        Returns:
            List of stocks that pass Stage 2 filtering
        """
        if not self.stage2_config:
            logger.info("Stage 2 filtering not configured, returning all stocks")
            return stocks

        logger.info(f"Applying Stage 2 filters to {len(stocks)} stocks")
        filtered_stocks = []

        for stock in stocks:
            try:
                # Calculate Stage 2 scores
                scores = self._calculate_stage2_scores(stock, user_id)

                # Check if stock passes thresholds
                thresholds = self.stage2_config.filtering_thresholds
                total_score = self._calculate_weighted_score(scores)

                if total_score >= thresholds.minimum_total_score:
                    # Check individual category minimums if required
                    if thresholds.require_all_categories:
                        if (scores.get('technical_score', 0) >= thresholds.minimum_technical_score and
                            scores.get('fundamental_score', 0) >= thresholds.minimum_fundamental_score):
                            filtered_stocks.append(stock)
                            logger.debug(f"{stock.symbol} passed Stage 2 with score {total_score:.2f}")
                    else:
                        filtered_stocks.append(stock)
                        logger.debug(f"{stock.symbol} passed Stage 2 with score {total_score:.2f}")
                else:
                    logger.debug(f"{stock.symbol} failed Stage 2 with score {total_score:.2f}")

            except Exception as e:
                logger.warning(f"Error applying Stage 2 filters to {stock.symbol}: {e}")
                continue

        logger.info(f"Stage 2 filtering complete: {len(filtered_stocks)}/{len(stocks)} stocks passed")
        return filtered_stocks

    def _calculate_stage2_scores(self, stock: StockInfo, user_id: int) -> Dict[str, float]:
        """
        Calculate various scores for Stage 2 filtering.

        Returns:
            Dictionary with score categories: technical, fundamental, risk, momentum, volume
        """
        scores = {}

        # Technical score (simplified - would need actual indicator calculations)
        technical_config = self.stage2_config.technical_indicators
        if technical_config:
            # This is a placeholder - in production, you'd calculate actual indicators
            scores['technical_score'] = self._calculate_technical_score(stock, technical_config)

        # Volume score based on liquidity and volume patterns
        volume_config = self.stage2_config.volume_analysis
        if volume_config:
            scores['volume_score'] = self._calculate_volume_score(stock, volume_config)

        # Risk score based on volatility and other risk metrics
        risk_config = self.stage2_config.risk_metrics
        if risk_config:
            scores['risk_score'] = self._calculate_risk_score(stock, risk_config)

        # Momentum score
        momentum_config = self.stage2_config.momentum
        if momentum_config:
            scores['momentum_score'] = self._calculate_momentum_score(stock, momentum_config)

        # Fundamental score (would need fundamental data)
        fundamental_config = self.stage2_config.fundamental_ratios
        if fundamental_config:
            scores['fundamental_score'] = self._calculate_fundamental_score(stock, fundamental_config)

        return scores

    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted total score based on configuration."""
        if not self.stage2_config or not self.stage2_config.scoring_weights:
            return 50.0  # Default middle score

        weights = self.stage2_config.scoring_weights
        total = 0.0

        # Apply weights to each score category
        if 'technical_score' in scores:
            total += scores['technical_score'] * weights.technical_score
        if 'fundamental_score' in scores:
            total += scores['fundamental_score'] * weights.fundamental_score
        if 'risk_score' in scores:
            total += scores['risk_score'] * weights.risk_score
        if 'momentum_score' in scores:
            total += scores['momentum_score'] * weights.momentum_score
        if 'volume_score' in scores:
            total += scores['volume_score'] * weights.volume_score

        return total

    def _calculate_technical_score(self, stock: StockInfo, config: Dict) -> float:
        """Calculate technical indicator score (simplified implementation)."""
        # This is a placeholder - actual implementation would calculate real indicators
        # For now, use liquidity score as a proxy
        return min(stock.liquidity_score * 100, 100)

    def _calculate_volume_score(self, stock: StockInfo, config: Dict) -> float:
        """Calculate volume analysis score."""
        # Use volume and liquidity as factors
        volume_factor = min(stock.volume / 100000, 1.0) * 50
        liquidity_factor = stock.liquidity_score * 50
        return volume_factor + liquidity_factor

    def _calculate_risk_score(self, stock: StockInfo, config: Dict) -> float:
        """Calculate risk metrics score."""
        # Inverse relationship - lower volatility = higher score
        # This is simplified - actual implementation would use historical data
        base_score = 70  # Default score

        # Adjust based on market cap (larger cap = lower risk)
        if stock.market_cap_category == MarketCap.LARGE_CAP:
            base_score += 20
        elif stock.market_cap_category == MarketCap.MID_CAP:
            base_score += 10

        return min(base_score, 100)

    def _calculate_momentum_score(self, stock: StockInfo, config: Dict) -> float:
        """Calculate momentum score."""
        # Placeholder - would need price history to calculate actual momentum
        return 60  # Default neutral score

    def _calculate_fundamental_score(self, stock: StockInfo, config: Dict) -> float:
        """Calculate fundamental ratios score."""
        # Placeholder - would need fundamental data from broker or external source
        # For now, use market cap and sector as proxies
        base_score = 50

        # Adjust based on market cap
        if stock.market_cap_category == MarketCap.LARGE_CAP:
            base_score += 15
        elif stock.market_cap_category == MarketCap.MID_CAP:
            base_score += 5

        # Adjust based on sector (some sectors typically have better fundamentals)
        if stock.sector in ['Banking', 'Technology', 'Pharmaceutical']:
            base_score += 10

        return min(base_score, 100)

    def discover_stocks_with_stage2(self, user_id: int = 1,
                                   exchange: str = "NSE") -> Dict[str, List[StockInfo]]:
        """
        Discover stocks with both Stage 1 and Stage 2 filtering.

        Returns:
            Dictionary with 'stage1' and 'stage2' lists of stocks
        """
        # Get Stage 1 filtered stocks
        stage1_stocks = self.discover_tradeable_stocks(user_id, exchange)

        # Apply Stage 2 filtering if configured
        stage2_stocks = []
        if self.stage2_config:
            stage2_stocks = self.apply_stage2_filters(stage1_stocks, user_id)

        return {
            'stage1': stage1_stocks,
            'stage2': stage2_stocks,
            'stage1_count': len(stage1_stocks),
            'stage2_count': len(stage2_stocks),
            'stage2_enabled': self.stage2_config is not None
        }



# Global service instance
_stock_discovery_service = None

def get_stock_discovery_service() -> StockDiscoveryService:
    """Get the global stock discovery service instance."""
    global _stock_discovery_service
    if _stock_discovery_service is None:
        _stock_discovery_service = StockDiscoveryService()
    return _stock_discovery_service