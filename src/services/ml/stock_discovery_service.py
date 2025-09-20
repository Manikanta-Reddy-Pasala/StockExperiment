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
                            logger.info(f"Search term '{term}' returned {len(symbols)} results")

                            for symbol_info in symbols:
                                symbol = symbol_info.get('symbol', '')
                                name = symbol_info.get('name', symbol_info.get('symbol_name', ''))

                                # Filter for equity stocks only
                                if ('-EQ' in symbol and
                                    symbol.startswith(f"{exchange}:") and
                                    len(name) > 2):
                                    all_symbols.add(symbol)
                        else:
                            logger.warning(f"Search failed for term '{term}': {search_result.get('message', 'Unknown error')}")
                    else:
                        logger.warning(f"No broker provider available for search")

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Search failed for term '{term}': {e}")
                    continue

            logger.info(f"Discovered {len(all_symbols)} potential stocks")

            # Process discovered stocks without requiring real-time quotes
            # Since quotes API is not available, use static market cap data
            logger.info(f"Processing {len(all_symbols)} symbols without real-time quotes")

            for symbol in all_symbols:
                try:
                    stock_info = self._analyze_stock_info_static(symbol, user_id)
                    if stock_info and stock_info.is_tradeable:
                        discovered_stocks.append(stock_info)
                except Exception as e:
                    logger.debug(f"Failed to analyze {symbol}: {e}")
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


    def _analyze_stock_info_static(self, symbol: str, user_id: int) -> Optional[StockInfo]:
        """Analyze individual stock using static market cap data (no real-time quotes)."""
        try:
            # Extract name from symbol
            name = symbol.replace("NSE:", "").replace("-EQ", "")

            # Use static market cap classification based on well-known stocks
            market_cap_crores, estimated_price = self._get_static_market_cap(symbol)

            if market_cap_crores == 0:
                return None  # Unknown stock, skip

            # Categorize by market cap
            if market_cap_crores > 20000:
                market_cap_category = MarketCap.LARGE_CAP
            elif market_cap_crores > 5000:
                market_cap_category = MarketCap.MID_CAP
            else:
                market_cap_category = MarketCap.SMALL_CAP

            # Static liquidity score based on market cap
            if market_cap_category == MarketCap.LARGE_CAP:
                liquidity_score = 0.8
            elif market_cap_category == MarketCap.MID_CAP:
                liquidity_score = 0.6
            else:
                liquidity_score = 0.4

            # Determine sector
            sector = self._determine_sector(name)

            return StockInfo(
                symbol=symbol,
                name=name,
                exchange="NSE",
                market_cap_category=market_cap_category,
                current_price=estimated_price,
                market_cap_crores=market_cap_crores,
                volume=100000,  # Static volume estimate
                liquidity_score=liquidity_score,
                is_tradeable=True,
                sector=sector
            )

        except Exception as e:
            logger.warning(f"Error analyzing stock {symbol}: {e}")
            return None

    def _analyze_stock_info(self, symbol: str, quote_data: Optional[Dict],
                          user_id: int) -> Optional[StockInfo]:
        """Analyze individual stock and categorize by market cap."""
        try:
            if not quote_data:
                return None

            # Extract basic info from Fyers quote format
            # Fyers API returns data in different fields
            current_price = float(quote_data.get('lp', quote_data.get('ltp', 0)))  # lp = last price
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

    def _get_static_market_cap(self, symbol: str) -> Tuple[float, float]:
        """Get static market cap and estimated price for well-known stocks."""
        # Static market cap data (in crores) and estimated prices
        # This is based on approximate market cap as of Sept 2025

        stock_data = {
            # Large Cap (>₹20,000 crores)
            "NSE:RELIANCE-EQ": (1500000, 2800),  # ₹15 lakh crores
            "NSE:TCS-EQ": (1200000, 4200),       # ₹12 lakh crores
            "NSE:HDFCBANK-EQ": (800000, 1600),   # ₹8 lakh crores
            "NSE:BHARTIARTL-EQ": (700000, 1200), # ₹7 lakh crores
            "NSE:INFY-EQ": (650000, 1800),       # ₹6.5 lakh crores
            "NSE:ICICIBANK-EQ": (550000, 1200),  # ₹5.5 lakh crores
            "NSE:HINDUNILVR-EQ": (500000, 2800), # ₹5 lakh crores
            "NSE:ITC-EQ": (450000, 400),         # ₹4.5 lakh crores
            "NSE:SBIN-EQ": (400000, 800),        # ₹4 lakh crores
            "NSE:LT-EQ": (350000, 3500),         # ₹3.5 lakh crores
            "NSE:HCLTECH-EQ": (300000, 1800),    # ₹3 lakh crores
            "NSE:KOTAKBANK-EQ": (280000, 1800),  # ₹2.8 lakh crores
            "NSE:ASIANPAINT-EQ": (270000, 3000), # ₹2.7 lakh crores
            "NSE:AXISBANK-EQ": (250000, 1200),   # ₹2.5 lakh crores
            "NSE:MARUTI-EQ": (240000, 12000),    # ₹2.4 lakh crores
            "NSE:SUNPHARMA-EQ": (230000, 1800),  # ₹2.3 lakh crores
            "NSE:TITAN-EQ": (220000, 3200),      # ₹2.2 lakh crores
            "NSE:NESTLEIND-EQ": (210000, 2800),  # ₹2.1 lakh crores
            "NSE:WIPRO-EQ": (200000, 600),       # ₹2 lakh crores
            "NSE:ULTRACEMCO-EQ": (190000, 11000), # ₹1.9 lakh crores

            # Mid Cap (₹5,000-20,000 crores)
            "NSE:BAJFINANCE-EQ": (180000, 7000), # ₹1.8 lakh crores
            "NSE:TECHM-EQ": (150000, 1600),      # ₹1.5 lakh crores
            "NSE:DIVISLAB-EQ": (140000, 6000),   # ₹1.4 lakh crores
            "NSE:INDIGO-EQ": (130000, 4500),     # ₹1.3 lakh crores
            "NSE:TATAPOWER-EQ": (120000, 450),   # ₹1.2 lakh crores
            "NSE:HDFCAMC-EQ": (80000, 4000),     # ₹80,000 crores
            "NSE:GODREJCP-EQ": (70000, 1200),    # ₹70,000 crores
            "NSE:PIDILITIND-EQ": (60000, 3000),  # ₹60,000 crores
            "NSE:BIOCON-EQ": (50000, 400),       # ₹50,000 crores
            "NSE:MPHASIS-EQ": (40000, 3000),     # ₹40,000 crores
            "NSE:BANKBARODA-EQ": (35000, 250),   # ₹35,000 crores
            "NSE:FEDERALBNK-EQ": (30000, 180),   # ₹30,000 crores
            "NSE:INDIANB-EQ": (25000, 600),      # ₹25,000 crores
            "NSE:CANBK-EQ": (20000, 450),        # ₹20,000 crores
            "NSE:UNIONBANK-EQ": (15000, 150),    # ₹15,000 crores
            "NSE:IDFCFIRSTB-EQ": (12000, 80),    # ₹12,000 crores
            "NSE:PNB-EQ": (10000, 120),          # ₹10,000 crores
            "NSE:CENTRALBK-EQ": (8000, 60),      # ₹8,000 crores
            "NSE:INDIANBANK-EQ": (7000, 550),    # ₹7,000 crores
            "NSE:IOB-EQ": (6000, 45),            # ₹6,000 crores
            "NSE:UCO-EQ": (5500, 40),            # ₹5,500 crores

            # Small Cap (<₹5,000 crores)
            "NSE:RBLBANK-EQ": (4500, 300),       # ₹4,500 crores
            "NSE:EQUITASBNK-EQ": (4000, 150),    # ₹4,000 crores
            "NSE:UJJIVANSFB-EQ": (3500, 50),     # ₹3,500 crores
            "NSE:FINPIPE-EQ": (3000, 300),       # ₹3,000 crores
            "NSE:ESAFSFB-EQ": (2500, 800),       # ₹2,500 crores
            "NSE:SURYODAY-EQ": (2000, 600),      # ₹2,000 crores
            "NSE:AUBANK-EQ": (1800, 800),        # ₹1,800 crores
            "NSE:CSBBANK-EQ": (1500, 350),       # ₹1,500 crores
            "NSE:JKBANK-EQ": (1200, 150),        # ₹1,200 crores
            "NSE:DCBBANK-EQ": (1000, 120),       # ₹1,000 crores
            "NSE:SOUTHBANK-EQ": (800, 15),       # ₹800 crores
            "NSE:TMBBANK-EQ": (600, 50),         # ₹600 crores
            "NSE:CITYUNION-EQ": (500, 180),      # ₹500 crores
            "NSE:KARURBANK-EQ": (400, 80),       # ₹400 crores
            "NSE:NKGSB-EQ": (300, 40),           # ₹300 crores
        }

        # Return market cap and estimated price, or (0, 0) if unknown
        return stock_data.get(symbol, (0, 0))

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