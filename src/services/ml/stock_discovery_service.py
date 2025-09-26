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
        self.cache_duration = 3600  # 1 hour cache
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

            # Get all tradeable stocks from database with proper session handling
            try:
                db_stocks = stock_master.get_stocks_for_screening(limit=1000)  # Get top 1000 by market cap and volume

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
                # If database fails, return empty list instead of legacy fallback
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
            # Return empty list - no fallbacks
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

            # Skip if price too low or no volume
            if current_price < 1 or volume < 1000:
                return None

            # Estimate market cap (this is simplified - in production you'd use fundamental data)
            estimated_shares = self._estimate_shares_outstanding(symbol, current_price, volume)
            market_cap_crores = (current_price * estimated_shares) / 10000000  # Convert to crores

            # Categorize by market cap
            if market_cap_crores > 20000:
                market_cap_category = MarketCap.LARGE_CAP
            elif market_cap_crores > 5000:
                market_cap_category = MarketCap.MID_CAP
            else:
                market_cap_category = MarketCap.SMALL_CAP

            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(current_price, volume, quote_data)

            # Determine if tradeable (basic criteria)
            is_tradeable = (
                current_price >= 5 and  # Minimum price
                volume >= 10000 and     # Minimum volume
                liquidity_score >= 0.3  # Minimum liquidity
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
        # Create realistic market cap distribution for Indian stocks

        # Use volume as an indicator of company size (higher volume = larger company typically)
        volume_factor = volume / 100000  # Normalize to 100k base

        # Price-based estimation with realistic ranges
        if price > 1500:  # High price stocks
            # Usually large cap companies with fewer shares outstanding
            base_shares = random.uniform(50000000, 300000000)  # 5-30 crores shares
        elif price > 500:  # Mid-high price stocks
            # Mix of large and mid cap
            base_shares = random.uniform(100000000, 800000000)  # 10-80 crores shares
        elif price > 100:  # Medium price stocks
            # Mostly mid cap companies
            base_shares = random.uniform(200000000, 1500000000)  # 20-150 crores shares
        else:  # Low price stocks
            # Small cap and penny stocks with many shares
            base_shares = random.uniform(500000000, 5000000000)  # 50-500 crores shares

        # Adjust based on volume (higher volume typically indicates larger companies)
        if volume > 200000:  # High volume
            base_shares *= random.uniform(0.8, 1.2)  # Large companies
        elif volume > 50000:  # Medium volume
            base_shares *= random.uniform(0.9, 1.3)  # Mid companies
        else:  # Low volume
            base_shares *= random.uniform(1.0, 2.0)  # Smaller companies

        return int(base_shares)

    def _calculate_liquidity_score(self, price: float, volume: float, quote_data: Dict) -> float:
        """Calculate liquidity score (0-1) based on volume and spread."""
        try:
            # Volume component (70%)
            volume_score = min(volume / 1000000, 1.0)  # Normalize to 1M shares

            # Spread component (30%) - estimate from high-low
            # Handle Fyers format (direct high/low or ohlc.high/ohlc.low)
            ohlc_data = quote_data.get('ohlc', {})
            high = float(quote_data.get('high', ohlc_data.get('high', price)))
            low = float(quote_data.get('low', ohlc_data.get('low', price)))
            if high > low and price > 0:
                spread_pct = (high - low) / price
                spread_score = max(0, 1 - (spread_pct * 10))  # Lower spread = higher score
            else:
                spread_score = 0.5

            return (volume_score * 0.7) + (spread_score * 0.3)

        except Exception:
            return 0.5


    def _determine_sector(self, name: str) -> str:
        """Determine sector from stock name using only generic keywords (no company names)."""
        name_upper = name.upper()

        # Banking & Financial Services
        if any(term in name_upper for term in ['BANK', 'FINANCE', 'FINANCIAL', 'LENDING', 'CREDIT']):
            return 'Banking'
        # Technology & Software
        elif any(term in name_upper for term in ['IT', 'TECH', 'SOFTWARE', 'SYSTEM', 'COMPUTER', 'DIGITAL']):
            return 'Technology'
        # Pharmaceutical & Healthcare
        elif any(term in name_upper for term in ['PHARMA', 'DRUG', 'MEDICINE', 'HEALTHCARE', 'BIO', 'LABORATORY']):
            return 'Pharmaceutical'
        # Automobile & Transportation
        elif any(term in name_upper for term in ['AUTO', 'MOTOR', 'VEHICLE', 'TRANSPORT', 'LOGISTICS']):
            return 'Automobile'
        # Consumer Goods
        elif any(term in name_upper for term in ['FMCG', 'CONSUMER', 'FOOD', 'BEVERAGE', 'RETAIL']):
            return 'FMCG'
        # Metals & Mining
        elif any(term in name_upper for term in ['METAL', 'STEEL', 'IRON', 'COAL', 'MINING', 'ALUMINIUM']):
            return 'Metals'
        # Energy & Power
        elif any(term in name_upper for term in ['ENERGY', 'POWER', 'OIL', 'GAS', 'PETROLEUM', 'SOLAR', 'ELECTRIC']):
            return 'Energy'
        # Infrastructure & Construction
        elif any(term in name_upper for term in ['INFRA', 'CEMENT', 'CONSTRUCTION', 'BUILDING', 'ENGINEERING']):
            return 'Infrastructure'
        # Telecommunications
        elif any(term in name_upper for term in ['TELECOM', 'COMMUNICATION', 'WIRELESS', 'NETWORK']):
            return 'Telecommunications'
        # Textiles
        elif any(term in name_upper for term in ['TEXTILE', 'COTTON', 'FABRIC', 'GARMENT', 'APPAREL']):
            return 'Textiles'
        else:
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
        for category in [MarketCap.LARGE_CAP, MarketCap.MID_CAP, MarketCap.SMALL_CAP]:
            category_stocks = [s for s in stocks if s.market_cap_category == category][:5]
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
            # Volume component (80% weight for database stocks)
            volume_score = min(volume / 1000000, 1.0)  # Normalize to 1M shares

            # Price stability component (20% weight)
            price_score = 0.8 if 100 <= price <= 1000 else 0.6

            return (volume_score * 0.8) + (price_score * 0.2)
        except Exception:
            return 0.5

    def _discover_stocks_legacy_search(self, user_id: int, exchange: str) -> List[StockInfo]:
        """
        Legacy stock discovery using keyword search.
        Used as fallback when database approach fails.
        """
        try:
            logger.info("Using legacy keyword search approach")

            # Get popular/liquid stocks using search functionality
            discovered_stocks = []

            # Search for major indices and generic sector terms only (no company names)
            search_terms = [
                # Major indices components
                "NIFTY", "SENSEX", "BANKNIFTY", "NIFTYNXT50", "FINNIFTY",
                # Generic sector keywords only
                "BANK", "IT", "PHARMA", "AUTO", "FMCG", "METAL", "INFRA", "ENERGY",
                "FINANCE", "TECH", "HEALTHCARE", "CONSUMER", "COMMODITY",
                # Generic size/volume based searches
                "LTD", "LIMITED", "CORP", "INC", "INDUSTRIES"
            ]

            all_symbols = set()

            for term in search_terms:
                try:
                    # Get the fyers provider directly for more reliable search
                    try:
                        from ..interfaces.broker_feature_factory import get_broker_feature_factory
                    except ImportError:
                        from src.services.interfaces.broker_feature_factory import get_broker_feature_factory
                    factory = get_broker_feature_factory()
                    provider = factory.get_suggested_stocks_provider(user_id)

                    if provider:
                        # Call the search method directly with proper signature
                        search_result = provider.fyers_service.search(user_id, term, exchange)

                        if search_result.get('status') == 'success':
                            symbols = search_result.get('data', [])
                            logger.debug(f"Search term '{term}' returned {len(symbols)} results")

                            for symbol_info in symbols:
                                symbol = symbol_info.get('symbol', '')
                                name = symbol_info.get('name', symbol_info.get('symbol_name', ''))

                                # Filter for equity stocks only
                                if ('-EQ' in symbol and
                                    symbol.startswith(f"{exchange}:") and
                                    len(name) > 2):
                                    all_symbols.add(symbol)

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Search failed for term '{term}': {e}")
                    continue

            logger.info(f"Legacy search discovered {len(all_symbols)} potential stocks")

            # Convert to list and process in batches for quote retrieval
            symbols_list = list(all_symbols)

            # Try to get real-time quotes in batches
            batch_size = 20
            quotes_data = {}

            for i in range(0, len(symbols_list), batch_size):
                batch_symbols = symbols_list[i:i + batch_size]
                try:
                    batch_quotes = self._get_batch_quotes(batch_symbols, user_id)
                    quotes_data.update(batch_quotes)
                    time.sleep(0.5)  # Rate limiting between batches
                except Exception as e:
                    logger.warning(f"Failed to get quotes for batch {i//batch_size + 1}: {e}")
                    continue

            logger.info(f"Retrieved quotes for {len(quotes_data)} out of {len(symbols_list)} symbols")

            # Process symbols with available quotes
            for symbol in symbols_list:
                try:
                    quote_data = quotes_data.get(symbol)
                    if quote_data:
                        # Use real quote data
                        stock_info = self._analyze_stock_info(symbol, quote_data, user_id)
                    else:
                        # Skip symbols without quotes
                        logger.debug(f"No quote data available for {symbol}, skipping")
                        continue

                    if stock_info and stock_info.is_tradeable:
                        discovered_stocks.append(stock_info)
                except Exception as e:
                    logger.debug(f"Failed to analyze {symbol}: {e}")
                    continue

            # Sort by market cap and liquidity
            discovered_stocks.sort(key=lambda x: (x.market_cap_crores, x.liquidity_score), reverse=True)

            logger.info(f"Legacy search discovered {len(discovered_stocks)} tradeable stocks")
            return discovered_stocks

        except Exception as e:
            logger.error(f"Error in legacy stock discovery: {e}")
            return []

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