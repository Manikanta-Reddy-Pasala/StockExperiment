"""
Dynamic Stock Discovery Service

This service dynamically fetches available stocks from the configured broker
and categorizes them by market cap using real-time market data.
No hardcoded stock symbols - everything fetched from broker API.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
from enum import Enum

from .config_loader import get_stock_filter_config, get_stage1_config, StockFilterConfig

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




# Global service instance
_stock_discovery_service = None

def get_stock_discovery_service() -> StockDiscoveryService:
    """Get the global stock discovery service instance."""
    global _stock_discovery_service
    if _stock_discovery_service is None:
        _stock_discovery_service = StockDiscoveryService()
    return _stock_discovery_service