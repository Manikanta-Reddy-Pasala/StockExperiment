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

logger = logging.getLogger(__name__)

try:
    from ...services.brokers.fyers_service import get_fyers_service
    from ...models.database import get_database_manager
except ImportError:
    from services.brokers.fyers_service import get_fyers_service
    from models.database import get_database_manager


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
        self.fyers_service = get_fyers_service()
        self.db_manager = get_database_manager()
        self.cache_duration = 3600  # 1 hour cache
        self._stock_cache = {}
        self._cache_timestamp = 0

    def discover_tradeable_stocks(self, user_id: int = 1,
                                exchange: str = "NSE") -> List[StockInfo]:
        """
        Discover all tradeable stocks from the broker API.
        Returns categorized stocks by market cap.
        """
        try:
            logger.info(f"Discovering tradeable stocks from {exchange} via broker API")

            # Check cache first
            if self._is_cache_valid():
                logger.info("Using cached stock data")
                return self._stock_cache.get('stocks', [])

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
                    search_result = self.fyers_service.search(user_id, term, exchange)
                    if search_result.get('status') == 'success':
                        symbols = search_result.get('data', [])
                        for symbol_info in symbols:
                            symbol = symbol_info.get('symbol', '')
                            # Filter for equity stocks only
                            if ('-EQ' in symbol and
                                symbol.startswith(f"{exchange}:") and
                                len(symbol_info.get('name', '')) > 2):
                                all_symbols.add(symbol)

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Search failed for term '{term}': {e}")
                    continue

            logger.info(f"Discovered {len(all_symbols)} potential stocks")

            # Get market data for discovered stocks and categorize
            stock_batches = list(self._chunk_list(list(all_symbols), 20))  # Process in batches

            for batch in stock_batches:
                try:
                    # Get quotes for batch
                    batch_quotes = self._get_batch_quotes(batch, user_id)

                    for symbol in batch:
                        try:
                            stock_info = self._analyze_stock_info(symbol, batch_quotes.get(symbol), user_id)
                            if stock_info and stock_info.is_tradeable:
                                discovered_stocks.append(stock_info)
                        except Exception as e:
                            logger.warning(f"Failed to analyze {symbol}: {e}")
                            continue

                    # Rate limiting between batches
                    time.sleep(0.5)

                except Exception as e:
                    logger.warning(f"Failed to process batch: {e}")
                    continue

            # Sort by market cap and liquidity
            discovered_stocks.sort(key=lambda x: (x.market_cap_crores, x.liquidity_score), reverse=True)

            # Cache results
            self._stock_cache = {
                'stocks': discovered_stocks,
                'timestamp': time.time()
            }
            self._cache_timestamp = time.time()

            logger.info(f"Discovered {len(discovered_stocks)} tradeable stocks")
            self._log_discovery_summary(discovered_stocks)

            return discovered_stocks

        except Exception as e:
            logger.error(f"Error discovering stocks: {e}")
            return []

    def _get_batch_quotes(self, symbols: List[str], user_id: int) -> Dict[str, Dict]:
        """Get quotes for a batch of symbols."""
        try:
            # Try multiple quotes first
            quotes_result = self.fyers_service.quotes_multiple(user_id, symbols)
            if quotes_result.get('status') == 'success':
                return quotes_result.get('data', {})

            # Fallback to individual quotes
            batch_quotes = {}
            for symbol in symbols:
                try:
                    clean_symbol = symbol.replace("NSE:", "").replace("-EQ", "")
                    quote_result = self.fyers_service.quotes(user_id, clean_symbol, "NSE")
                    if quote_result.get('status') == 'success':
                        batch_quotes[symbol] = quote_result.get('data', {})
                    time.sleep(0.05)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Failed to get quote for {symbol}: {e}")
                    continue

            return batch_quotes

        except Exception as e:
            logger.error(f"Error getting batch quotes: {e}")
            return {}

    def _analyze_stock_info(self, symbol: str, quote_data: Optional[Dict],
                          user_id: int) -> Optional[StockInfo]:
        """Analyze individual stock and categorize by market cap."""
        try:
            if not quote_data:
                return None

            # Extract basic info
            current_price = float(quote_data.get('ltp', 0))
            volume = float(quote_data.get('volume', 0))
            name = symbol.replace("NSE:", "").replace("-EQ", "")

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
        """Estimate shares outstanding (simplified approach)."""
        # This is a rough estimation - in production use fundamental data APIs

        # Use volume and price patterns to estimate
        base_shares = volume * 100  # Rough multiplier

        # Adjust based on price level (higher price usually means fewer shares)
        if price > 1000:
            multiplier = 50000000    # Large cap with high price
        elif price > 500:
            multiplier = 100000000   # Large/mid cap
        elif price > 100:
            multiplier = 200000000   # Mid cap
        else:
            multiplier = 500000000   # Small cap or low price

        return min(base_shares * 1000, multiplier)

    def _calculate_liquidity_score(self, price: float, volume: float, quote_data: Dict) -> float:
        """Calculate liquidity score (0-1) based on volume and spread."""
        try:
            # Volume component (70%)
            volume_score = min(volume / 1000000, 1.0)  # Normalize to 1M shares

            # Spread component (30%) - estimate from high-low
            high = float(quote_data.get('high', price))
            low = float(quote_data.get('low', price))
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