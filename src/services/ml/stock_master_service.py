"""
Stock Master Data Service

This service manages the complete stock master database.
It fetches all available stocks from the broker API and maintains
comprehensive stock information for efficient filtering and screening.
"""

import logging
import time
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from sqlalchemy import and_, or_

logger = logging.getLogger(__name__)

try:
    from ..core.unified_broker_service import get_unified_broker_service
    from src.models.database import get_database_manager
    from src.models.stock_models import Stock, MarketCapCategory
except ImportError:
    from src.services.core.unified_broker_service import get_unified_broker_service
    from src.models.database import get_database_manager
    from src.models.stock_models import Stock, MarketCapCategory


class StockMasterService:
    """Service to manage comprehensive stock master data."""

    def __init__(self):
        self.unified_broker_service = get_unified_broker_service()
        self.db_manager = get_database_manager()
        self.refresh_interval_hours = 24  # Refresh daily

    def refresh_all_stocks(self, user_id: int = 1, exchange: str = "NSE") -> Dict:
        """
        Refresh the complete stock master database with all available stocks.
        This should be run daily to keep the database current.
        """
        try:
            logger.info(f"Starting complete stock master refresh for {exchange}")

            # Get all available stocks from broker APIs
            all_symbols = self._get_all_available_symbols(user_id, exchange)
            logger.info(f"Found {len(all_symbols)} total symbols from broker APIs")

            if not all_symbols:
                logger.error("No symbols found from broker APIs")
                return {'success': False, 'error': 'No symbols found'}

            # Process symbols in batches to get quotes and fundamental data
            batch_size = 50
            processed_count = 0
            updated_count = 0
            new_count = 0

            with self.db_manager.get_session() as session:
                for i in range(0, len(all_symbols), batch_size):
                    batch_symbols = all_symbols[i:i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_symbols)} symbols")

                    # Get quotes for this batch
                    batch_quotes = self._get_batch_quotes(batch_symbols, user_id)

                    # Update database with this batch
                    batch_results = self._update_stocks_batch(session, batch_symbols, batch_quotes, exchange)
                    processed_count += batch_results['processed']
                    updated_count += batch_results['updated']
                    new_count += batch_results['new']

                    # Rate limiting between batches
                    time.sleep(1)

                # Mark inactive stocks that weren't found in this refresh
                inactive_count = self._mark_inactive_stocks(session, all_symbols, exchange)

                # Commit all changes
                session.commit()

            result = {
                'success': True,
                'exchange': exchange,
                'total_symbols_found': len(all_symbols),
                'processed': processed_count,
                'new_stocks': new_count,
                'updated_stocks': updated_count,
                'marked_inactive': inactive_count,
                'refresh_timestamp': datetime.utcnow().isoformat()
            }

            logger.info(f"Stock master refresh completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Error refreshing stock master data: {e}")
            return {'success': False, 'error': str(e)}

    def _get_all_available_symbols(self, user_id: int, exchange: str) -> List[str]:
        """
        Get all available symbols from the broker API.
        Use the broker's symbol search with comprehensive search terms.
        """
        all_symbols = set()

        try:
            # Get the broker feature factory for symbol search
            try:
                from ..interfaces.broker_feature_factory import get_broker_feature_factory
            except ImportError:
                from src.services.interfaces.broker_feature_factory import get_broker_feature_factory

            factory = get_broker_feature_factory()
            provider = factory.get_suggested_stocks_provider(user_id)

            if not provider:
                logger.error("No broker provider available for symbol search")
                return []

            # Use alphabet-based search to get comprehensive coverage
            # This is more thorough than keyword-based search
            search_patterns = [
                # Alphabet prefixes - covers most stock symbols
                *[chr(i) for i in range(ord('A'), ord('Z') + 1)],  # A-Z
                # Common numeric prefixes for some stocks
                "1", "2", "3", "5", "7", "9",
                # Common stock name patterns
                "INDIA", "BHARAT", "HINDUSTAN", "TATA", "RELIANCE", "ADANI",
                "BAJAJ", "MAHINDRA", "GODREJ", "BIRLA", "JINDAL", "ESSAR",
                # Major sectors to ensure coverage
                "BANK", "PHARMA", "AUTO", "IT", "TECH", "FINANCE", "INFRA",
                "STEEL", "CEMENT", "TEXTILE", "CHEMICAL", "POWER", "OIL"
            ]

            for pattern in search_patterns:
                try:
                    search_result = provider.fyers_service.search(user_id, pattern, exchange)

                    if search_result.get('status') == 'success':
                        symbols = search_result.get('data', [])
                        logger.debug(f"Search pattern '{pattern}' returned {len(symbols)} results")

                        for symbol_info in symbols:
                            symbol = symbol_info.get('symbol', '')
                            name = symbol_info.get('name', symbol_info.get('symbol_name', ''))

                            # Filter for equity stocks only
                            if (symbol.startswith(f"{exchange}:") and
                                '-EQ' in symbol and
                                len(name) > 2 and
                                'ETF' not in name.upper() and  # Exclude ETFs for now
                                'INDEX' not in name.upper()):
                                all_symbols.add(symbol)

                    # Rate limiting
                    time.sleep(0.2)

                except Exception as e:
                    logger.warning(f"Search failed for pattern '{pattern}': {e}")
                    continue

            logger.info(f"Comprehensive symbol search found {len(all_symbols)} unique symbols")
            return list(all_symbols)

        except Exception as e:
            logger.error(f"Error getting all available symbols: {e}")
            return []

    def _get_batch_quotes(self, symbols: List[str], user_id: int) -> Dict[str, Dict]:
        """Get quotes for a batch of symbols."""
        try:
            quotes_result = self.unified_broker_service.get_quotes(user_id, symbols)
            if quotes_result.get('success'):
                return quotes_result.get('data', {})
            else:
                logger.warning(f"Batch quotes failed: {quotes_result.get('error', 'Unknown error')}")
                return {}
        except Exception as e:
            logger.warning(f"Exception getting batch quotes: {e}")
            return {}

    def _update_stocks_batch(self, session, symbols: List[str], quotes: Dict[str, Dict],
                           exchange: str) -> Dict[str, int]:
        """Update database with a batch of stocks."""
        processed = 0
        updated = 0
        new = 0

        for symbol in symbols:
            try:
                quote_data = quotes.get(symbol)
                if not quote_data:
                    logger.debug(f"No quote data for {symbol}, skipping")
                    continue

                # Check if stock already exists
                existing_stock = session.query(Stock).filter_by(symbol=symbol).first()

                if existing_stock:
                    # Update existing stock
                    updated += self._update_existing_stock(existing_stock, quote_data)
                else:
                    # Create new stock
                    new_stock = self._create_new_stock(symbol, quote_data, exchange)
                    if new_stock:
                        session.add(new_stock)
                        new += 1

                processed += 1

            except Exception as e:
                logger.warning(f"Error processing stock {symbol}: {e}")
                continue

        return {'processed': processed, 'updated': updated, 'new': new}

    def _update_existing_stock(self, stock: Stock, quote_data: Dict) -> int:
        """Update an existing stock record."""
        try:
            # Extract price and volume data
            current_price = float(quote_data.get('lp', quote_data.get('ltp', quote_data.get('last_price', 0))))
            volume = int(quote_data.get('volume', quote_data.get('vol', 0)))

            if current_price <= 0:
                return 0

            # Update current market data
            stock.current_price = current_price
            stock.volume = volume
            stock.last_updated = datetime.utcnow()
            stock.is_active = True

            # Recalculate market cap and category if we have the data
            if stock.market_cap:
                # Update market cap based on current price (simplified)
                # In production, you'd get shares outstanding from fundamental data APIs
                stock.market_cap = self._estimate_market_cap(current_price, volume)
                stock.market_cap_category = self._determine_market_cap_category(stock.market_cap)

            # Calculate liquidity and tradeability
            stock.is_tradeable = self._calculate_tradeability(current_price, volume, quote_data)

            return 1

        except Exception as e:
            logger.warning(f"Error updating stock {stock.symbol}: {e}")
            return 0

    def _create_new_stock(self, symbol: str, quote_data: Dict, exchange: str) -> Optional[Stock]:
        """Create a new stock record."""
        try:
            # Extract basic data
            current_price = float(quote_data.get('lp', quote_data.get('ltp', quote_data.get('last_price', 0))))
            volume = int(quote_data.get('volume', quote_data.get('vol', 0)))

            if current_price <= 0 or volume <= 0:
                return None

            # Extract name from symbol
            name = symbol.replace(f"{exchange}:", "").replace("-EQ", "")
            if 'symbol_name' in quote_data:
                name = quote_data['symbol_name']

            # Calculate market cap and determine category
            market_cap = self._estimate_market_cap(current_price, volume)
            market_cap_category = self._determine_market_cap_category(market_cap)

            # Determine sector
            sector = self._determine_sector(name)

            # Calculate tradeability
            is_tradeable = self._calculate_tradeability(current_price, volume, quote_data)

            # Create new stock
            new_stock = Stock(
                symbol=symbol,
                name=name,
                exchange=exchange,
                sector=sector,
                current_price=current_price,
                volume=volume,
                market_cap=market_cap,
                market_cap_category=market_cap_category,
                is_active=True,
                is_tradeable=is_tradeable,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )

            return new_stock

        except Exception as e:
            logger.warning(f"Error creating new stock {symbol}: {e}")
            return None

    def _estimate_market_cap(self, price: float, volume: float) -> float:
        """Estimate market cap based on price and volume patterns."""
        # Use volume and price to estimate shares outstanding
        # This is simplified - in production use fundamental data APIs

        if price > 1500:
            base_shares = 100000000  # 10 crores
        elif price > 500:
            base_shares = 500000000  # 50 crores
        elif price > 100:
            base_shares = 1000000000  # 100 crores
        else:
            base_shares = 2000000000  # 200 crores

        # Adjust based on volume (higher volume typically = larger company)
        if volume > 200000:
            base_shares *= 2  # Large companies
        elif volume > 50000:
            base_shares *= 1.5  # Mid companies

        market_cap_crores = (price * base_shares) / 10000000
        return market_cap_crores

    def _determine_market_cap_category(self, market_cap_crores: float) -> str:
        """Determine market cap category."""
        if market_cap_crores > 20000:
            return "large_cap"
        elif market_cap_crores > 5000:
            return "mid_cap"
        else:
            return "small_cap"

    def _determine_sector(self, name: str) -> str:
        """Determine sector from stock name using keywords."""
        name_upper = name.upper()

        if any(term in name_upper for term in ['BANK', 'FINANCE', 'FINANCIAL', 'LENDING', 'CREDIT']):
            return 'Banking'
        elif any(term in name_upper for term in ['IT', 'TECH', 'SOFTWARE', 'SYSTEM', 'COMPUTER', 'DIGITAL']):
            return 'Technology'
        elif any(term in name_upper for term in ['PHARMA', 'DRUG', 'MEDICINE', 'HEALTHCARE', 'BIO', 'LABORATORY']):
            return 'Pharmaceutical'
        elif any(term in name_upper for term in ['AUTO', 'MOTOR', 'VEHICLE', 'TRANSPORT', 'LOGISTICS']):
            return 'Automobile'
        elif any(term in name_upper for term in ['FMCG', 'CONSUMER', 'FOOD', 'BEVERAGE', 'RETAIL']):
            return 'FMCG'
        elif any(term in name_upper for term in ['METAL', 'STEEL', 'IRON', 'COAL', 'MINING', 'ALUMINIUM']):
            return 'Metals'
        elif any(term in name_upper for term in ['ENERGY', 'POWER', 'OIL', 'GAS', 'PETROLEUM', 'SOLAR', 'ELECTRIC']):
            return 'Energy'
        elif any(term in name_upper for term in ['INFRA', 'CEMENT', 'CONSTRUCTION', 'BUILDING', 'ENGINEERING']):
            return 'Infrastructure'
        elif any(term in name_upper for term in ['TELECOM', 'COMMUNICATION', 'WIRELESS', 'NETWORK']):
            return 'Telecommunications'
        elif any(term in name_upper for term in ['TEXTILE', 'COTTON', 'FABRIC', 'GARMENT', 'APPAREL']):
            return 'Textiles'
        else:
            return 'Others'

    def _calculate_tradeability(self, price: float, volume: int, quote_data: Dict) -> bool:
        """Calculate if a stock is tradeable based on basic criteria."""
        # Basic tradeability criteria
        return (
            price >= 5 and      # Minimum price
            volume >= 10000 and # Minimum volume
            price <= 10000      # Maximum price (avoid extreme prices)
        )

    def _mark_inactive_stocks(self, session, active_symbols: List[str], exchange: str) -> int:
        """Mark stocks as inactive if they weren't found in the current refresh."""
        try:
            # Get all stocks for this exchange that weren't in the active list
            inactive_stocks = session.query(Stock).filter(
                and_(
                    Stock.exchange == exchange,
                    Stock.is_active == True,
                    ~Stock.symbol.in_(active_symbols)
                )
            ).all()

            count = 0
            for stock in inactive_stocks:
                stock.is_active = False
                stock.last_updated = datetime.utcnow()
                count += 1

            logger.info(f"Marked {count} stocks as inactive")
            return count

        except Exception as e:
            logger.error(f"Error marking inactive stocks: {e}")
            return 0

    def get_stocks_for_screening(self,
                               market_caps: List[str] = None,
                               min_price: float = None,
                               max_price: float = None,
                               min_volume: int = None,
                               sectors: List[str] = None,
                               limit: int = None) -> List[Stock]:
        """
        Get stocks from database for screening with filtering criteria.
        This replaces the keyword search approach.
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(Stock).filter(
                    Stock.is_active == True,
                    Stock.is_tradeable == True
                )

                # Apply filters
                if market_caps:
                    query = query.filter(Stock.market_cap_category.in_(market_caps))

                if min_price is not None:
                    query = query.filter(Stock.current_price >= min_price)

                if max_price is not None:
                    query = query.filter(Stock.current_price <= max_price)

                if min_volume is not None:
                    query = query.filter(Stock.volume >= min_volume)

                if sectors:
                    query = query.filter(Stock.sector.in_(sectors))

                # Order by market cap and volume for better selection
                query = query.order_by(Stock.market_cap.desc(), Stock.volume.desc())

                if limit:
                    query = query.limit(limit)

                stocks = query.all()
                logger.info(f"Database query returned {len(stocks)} stocks for screening")
                return stocks

        except Exception as e:
            logger.error(f"Error querying stocks for screening: {e}")
            return []

    def get_stock_count_by_category(self) -> Dict[str, int]:
        """Get count of stocks by market cap category."""
        try:
            with self.db_manager.get_session() as session:
                results = {}

                # Count by each category using string values
                for category_value in ["large_cap", "mid_cap", "small_cap"]:
                    count = session.query(Stock).filter(
                        Stock.is_active == True,
                        Stock.is_tradeable == True,
                        Stock.market_cap_category == category_value
                    ).count()
                    results[category_value] = count

                total = session.query(Stock).filter(
                    Stock.is_active == True,
                    Stock.is_tradeable == True
                ).count()
                results['total'] = total

                return results

        except Exception as e:
            logger.error(f"Error getting stock count by category: {e}")
            return {}

    def is_refresh_needed(self) -> bool:
        """Check if stock master data needs refresh."""
        try:
            with self.db_manager.get_session() as session:
                # Get the most recent update timestamp
                latest_stock = session.query(Stock).filter(
                    Stock.is_active == True
                ).order_by(Stock.last_updated.desc()).first()

                if not latest_stock:
                    return True  # No data, need refresh

                # Check if data is older than refresh interval
                time_diff = datetime.utcnow() - latest_stock.last_updated
                return time_diff.total_seconds() > (self.refresh_interval_hours * 3600)

        except Exception as e:
            logger.error(f"Error checking refresh status: {e}")
            return True


# Global service instance
_stock_master_service = None

def get_stock_master_service() -> StockMasterService:
    """Get the global stock master service instance."""
    global _stock_master_service
    if _stock_master_service is None:
        _stock_master_service = StockMasterService()
    return _stock_master_service