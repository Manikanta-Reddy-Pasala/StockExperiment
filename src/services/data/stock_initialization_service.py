"""
Complete Stock Initialization Service

This service handles the complete flow:
1. Load symbols from Fyers API → symbol_master table (with fytoken as primary key)
2. Verify each stock with quotes → update is_fyers_verified flag
3. Create stock records with current prices → stocks table
4. Handle updates for existing stocks vs new stocks

The service ensures proper verification flow and handles the fytoken-based updates.
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import and_, or_
from decimal import Decimal

logger = logging.getLogger(__name__)

try:
    from ..core.unified_broker_service import get_unified_broker_service
    from ...models.database import get_database_manager
    from ...models.stock_models import Stock, SymbolMaster
    from ..data.fyers_symbol_service import get_fyers_symbol_service
except ImportError:
    from src.services.core.unified_broker_service import get_unified_broker_service
    from src.models.database import get_database_manager
    from src.models.stock_models import Stock, SymbolMaster
    from src.services.data.fyers_symbol_service import get_fyers_symbol_service


class StockInitializationService:
    """Complete stock initialization service with proper verification flow."""

    def __init__(self):
        self.unified_broker_service = get_unified_broker_service()
        self.db_manager = get_database_manager()
        self.fyers_service = get_fyers_symbol_service()
        self.rate_limit_delay = 0.2  # 200ms between API calls
        self.batch_size = 10  # Process in small batches

    def initialize_complete_stock_system(self, user_id: int = 1) -> Dict:
        """
        Complete initialization flow:
        1. Load symbol master from Fyers API
        2. Verify symbols with quotes and update verification flags
        3. Create/update stocks table with current prices
        """
        logger.info("🚀 Starting complete stock system initialization")

        try:
            # Step 1: Load symbol master data from Fyers
            logger.info("📊 Step 1: Loading symbol master data from Fyers API")
            symbol_result = self._load_symbol_master_from_fyers()
            if not symbol_result['success']:
                return symbol_result

            # Step 2: Verify symbols and update verification flags
            logger.info("🔍 Step 2: Verifying symbols with quotes")
            verification_result = self._verify_symbols_with_quotes(user_id)
            if not verification_result['success']:
                return verification_result

            # Step 3: Create/update stocks with current prices
            logger.info("📈 Step 3: Creating/updating stocks with current prices")
            stocks_result = self._create_update_stocks_from_verified_symbols(user_id)

            # Step 4: Get final statistics
            stats = self._get_initialization_statistics()

            result = {
                'success': True,
                'symbol_master': symbol_result,
                'verification': verification_result,
                'stocks': stocks_result,
                'statistics': stats,
                'completed_at': datetime.utcnow().isoformat()
            }

            logger.info(f"✅ Complete stock system initialization finished: {result}")
            return result

        except Exception as e:
            logger.error(f"❌ Stock system initialization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stage': 'complete_initialization'
            }

    def _load_symbol_master_from_fyers(self) -> Dict:
        """Load comprehensive symbol master data from Fyers API."""
        try:
            logger.info("🔄 Loading symbol master data from Fyers CSV APIs")

            # Use fyers symbol service to get comprehensive data
            nse_symbols = self.fyers_service.get_nse_symbols(force_refresh=True, use_database=False)
            logger.info(f"📊 Retrieved {len(nse_symbols)} NSE symbols from Fyers")

            if not nse_symbols:
                return {
                    'success': False,
                    'error': 'No NSE symbols retrieved from Fyers API'
                }

            # Store in symbol_master table with proper fytoken handling
            stored_count = 0
            updated_count = 0

            with self.db_manager.get_session() as session:
                for symbol_data in nse_symbols:
                    try:
                        fytoken = symbol_data.get('fytoken', '')
                        symbol = symbol_data.get('symbol', '')

                        if not fytoken or not symbol:
                            continue

                        # Use fytoken as primary key for upsert logic
                        existing_record = session.query(SymbolMaster).filter(
                            SymbolMaster.fytoken == fytoken
                        ).first()

                        if existing_record:
                            # Update existing record, preserve verification data
                            existing_record.symbol = symbol
                            existing_record.name = symbol_data.get('name', existing_record.name)
                            existing_record.exchange = symbol_data.get('exchange', existing_record.exchange)
                            existing_record.segment = symbol_data.get('segment', existing_record.segment)
                            existing_record.instrument_type = symbol_data.get('instrument_type', existing_record.instrument_type)
                            existing_record.lot_size = symbol_data.get('lot', existing_record.lot_size)
                            existing_record.tick_size = symbol_data.get('tick', existing_record.tick_size)
                            existing_record.isin = symbol_data.get('isin', existing_record.isin)
                            existing_record.updated_at = datetime.utcnow()
                            # Preserve: is_fyers_verified, verification_date, etc.
                            updated_count += 1

                        else:
                            # Create new record
                            symbol_master = SymbolMaster(
                                symbol=symbol,
                                fytoken=fytoken,
                                name=symbol_data.get('name', ''),
                                exchange=symbol_data.get('exchange', 'NSE'),
                                segment=symbol_data.get('segment', 'CM'),
                                instrument_type=symbol_data.get('instrument_type', 'EQ'),
                                lot_size=symbol_data.get('lot', 1),
                                tick_size=symbol_data.get('tick', 0.05),
                                isin=symbol_data.get('isin', ''),
                                data_source='fyers',
                                is_active=True,
                                is_equity=True,
                                created_at=datetime.utcnow(),
                                updated_at=datetime.utcnow()
                            )
                            session.add(symbol_master)
                            stored_count += 1

                    except Exception as e:
                        logger.warning(f"Error processing symbol {symbol_data.get('symbol')}: {e}")
                        continue

                # Commit all changes
                session.commit()

            return {
                'success': True,
                'total_symbols': len(nse_symbols),
                'new_symbols': stored_count,
                'updated_symbols': updated_count,
                'source': 'fyers_api'
            }

        except Exception as e:
            logger.error(f"Error loading symbol master from Fyers: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _verify_symbols_with_quotes(self, user_id: int, limit: int = 100) -> Dict:
        """
        Verify symbols by getting quotes and update is_fyers_verified flag.
        This step ensures only valid symbols with working quotes are marked as verified.
        """
        try:
            logger.info("🔍 Starting symbol verification with quotes")

            # First, get the symbols to verify using a separate query
            with self.db_manager.get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                unverified_symbols = session.query(SymbolMaster.fytoken, SymbolMaster.symbol).filter(
                    and_(
                        SymbolMaster.is_active == True,
                        SymbolMaster.is_equity == True,
                        or_(
                            SymbolMaster.is_fyers_verified.is_(None),
                            SymbolMaster.is_fyers_verified == False,
                            SymbolMaster.verification_date < cutoff_date
                        )
                    )
                ).limit(limit).all()

                # Convert to list of tuples
                symbols_to_verify = [(row.fytoken, row.symbol) for row in unverified_symbols]

            logger.info(f"🔍 Found {len(symbols_to_verify)} symbols to verify")

            verified_count = 0
            failed_count = 0

            # Now process each symbol with its own session
            for i, (fytoken, symbol) in enumerate(symbols_to_verify):
                try:
                    logger.info(f"🔍 Verifying {i+1}/{len(symbols_to_verify)}: {symbol}")

                    # Get quote for this symbol
                    quote_result = self._get_individual_quote(symbol, user_id)

                    # Update in a fresh session
                    with self.db_manager.get_session() as update_session:
                        symbol_record = update_session.query(SymbolMaster).filter(
                            SymbolMaster.fytoken == fytoken
                        ).first()

                        if symbol_record:
                            if quote_result['success']:
                                # Verification successful
                                symbol_record.is_fyers_verified = True
                                symbol_record.verification_date = datetime.utcnow()
                                symbol_record.verification_error = None
                                symbol_record.last_quote_check = datetime.utcnow()
                                verified_count += 1
                                logger.info(f"✅ Verified: {symbol}")
                            else:
                                # Verification failed
                                symbol_record.is_fyers_verified = False
                                symbol_record.verification_date = datetime.utcnow()
                                symbol_record.verification_error = quote_result.get('error', 'Unknown error')
                                symbol_record.last_quote_check = datetime.utcnow()
                                failed_count += 1
                                logger.warning(f"❌ Failed: {symbol} - {quote_result.get('error')}")

                            update_session.commit()

                    # Rate limiting
                    time.sleep(self.rate_limit_delay)

                    # Progress logging every 10 records
                    if (i + 1) % 10 == 0:
                        logger.info(f"📊 Progress: {i+1}/{len(symbols_to_verify)} verified")

                except Exception as e:
                    logger.warning(f"Error verifying {symbol}: {e}")
                    # Try to mark as failed
                    try:
                        with self.db_manager.get_session() as error_session:
                            symbol_record = error_session.query(SymbolMaster).filter(
                                SymbolMaster.fytoken == fytoken
                            ).first()
                            if symbol_record:
                                symbol_record.is_fyers_verified = False
                                symbol_record.verification_error = str(e)
                                symbol_record.verification_date = datetime.utcnow()
                                error_session.commit()
                    except:
                        pass  # If error update fails, continue
                    failed_count += 1
                    continue

            return {
                'success': True,
                'total_checked': len(symbols_to_verify),
                'verified': verified_count,
                'failed': failed_count,
                'verification_rate': verified_count / len(symbols_to_verify) if symbols_to_verify else 0
            }

        except Exception as e:
            logger.error(f"Error in symbol verification: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _create_update_stocks_from_verified_symbols(self, user_id: int, limit: int = 50) -> Dict:
        """
        Create/update stocks table from verified symbols with current prices.
        Only processes symbols that have is_fyers_verified = True.
        """
        try:
            logger.info("📈 Creating/updating stocks from verified symbols")

            # First, get the verified symbols data
            with self.db_manager.get_session() as session:
                verified_symbols = session.query(
                    SymbolMaster.fytoken, SymbolMaster.symbol, SymbolMaster.name, SymbolMaster.exchange
                ).filter(
                    and_(
                        SymbolMaster.is_active == True,
                        SymbolMaster.is_equity == True,
                        SymbolMaster.is_fyers_verified == True
                    )
                ).limit(limit).all()

                # Convert to list of dictionaries
                symbols_to_process = [
                    {
                        'fytoken': row.fytoken,
                        'symbol': row.symbol,
                        'name': row.name,
                        'exchange': row.exchange
                    }
                    for row in verified_symbols
                ]

            logger.info(f"📈 Found {len(symbols_to_process)} verified symbols to process")

            created_count = 0
            updated_count = 0
            error_count = 0

            for i, symbol_info in enumerate(symbols_to_process):
                try:
                    logger.info(f"📈 Processing {i+1}/{len(symbols_to_process)}: {symbol_info['symbol']}")

                    # Get current quote with price data
                    quote_result = self._get_individual_quote(symbol_info['symbol'], user_id)

                    if not quote_result['success']:
                        logger.warning(f"❌ No quote data for {symbol_info['symbol']}")
                        error_count += 1
                        continue

                    quote_data = quote_result['data']

                    # Validate and extract price data
                    price_data = self._extract_validate_price_data(quote_data, symbol_info['symbol'])
                    if not price_data['valid']:
                        logger.warning(f"❌ Invalid price data for {symbol_info['symbol']}")
                        error_count += 1
                        continue

                    # Process in a separate session
                    with self.db_manager.get_session() as stock_session:
                        # Check if stock already exists
                        existing_stock = stock_session.query(Stock).filter_by(symbol=symbol_info['symbol']).first()

                        if existing_stock:
                            # Update existing stock
                            self._update_stock_with_price_data_from_dict(existing_stock, price_data, symbol_info)
                            updated_count += 1
                            logger.info(f"📝 Updated: {symbol_info['symbol']}")
                        else:
                            # Create new stock
                            new_stock = self._create_stock_with_price_data_from_dict(price_data, symbol_info)
                            if new_stock:
                                stock_session.add(new_stock)
                                created_count += 1
                                logger.info(f"🆕 Created: {symbol_info['symbol']}")

                        stock_session.commit()

                    # Rate limiting
                    time.sleep(self.rate_limit_delay)

                    # Progress logging every 10 records
                    if (i + 1) % 10 == 0:
                        logger.info(f"📊 Progress: {i+1}/{len(symbols_to_process)} processed")

                except Exception as e:
                    logger.warning(f"Error processing stock {symbol_info['symbol']}: {e}")
                    error_count += 1
                    continue

            return {
                'success': True,
                'total_processed': len(symbols_to_process),
                'created': created_count,
                'updated': updated_count,
                'errors': error_count
            }

        except Exception as e:
            logger.error(f"Error creating/updating stocks: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _get_individual_quote(self, symbol: str, user_id: int) -> Dict:
        """Get quote for individual symbol with error handling."""
        try:
            # Use unified broker service for quotes
            result = self.unified_broker_service.get_quotes(user_id, symbols=[symbol])

            # Debug the actual response structure
            logger.debug(f"Raw quote response for {symbol}: {result}")

            # Handle different response structures from Fyers API
            if result.get('status') == 'success' and result.get('data'):
                # Direct Fyers API response format
                quote_data = result['data'].get(symbol)
                if quote_data:
                    return {
                        'success': True,
                        'data': quote_data
                    }
            elif result.get('success') and result.get('data'):
                # Unified service wrapper format
                quote_data = result['data'].get(symbol)
                if quote_data:
                    return {
                        'success': True,
                        'data': quote_data
                    }

            return {
                'success': False,
                'error': result.get('error', 'No quote data available')
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_validate_price_data(self, quote_data: Dict, symbol: str) -> Dict:
        """Extract and validate price data from quote."""
        try:
            # Extract price (try multiple fields, handle string values from Fyers)
            price = None
            for field in ['lp', 'ltp', 'last_price', 'close']:
                if field in quote_data:
                    try:
                        price_value = float(quote_data[field])
                        if price_value > 0:
                            price = price_value
                            break
                    except (ValueError, TypeError):
                        continue

            if not price or price <= 0:
                return {
                    'valid': False,
                    'error': f'No valid price found for {symbol}. Quote data: {quote_data}'
                }

            # Extract volume (handle string values from Fyers)
            volume = 0
            for field in ['volume', 'vol', 'total_volume']:
                if field in quote_data:
                    try:
                        volume_value = int(float(quote_data.get(field, 0)))
                        volume = volume_value
                        break
                    except (ValueError, TypeError):
                        continue

            # Basic validation
            if price < 1.0:
                logger.warning(f"Very low price for {symbol}: ₹{price}")
            elif price > 50000:
                logger.warning(f"Very high price for {symbol}: ₹{price}")

            return {
                'valid': True,
                'price': price,
                'volume': volume,
                'quote_data': quote_data
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f'Error extracting price data: {str(e)}'
            }

    def _update_stock_with_price_data(self, stock: Stock, price_data: Dict, symbol_master: SymbolMaster):
        """Update existing stock with current price data."""
        try:
            # Update price and volume
            stock.current_price = price_data['price']
            stock.volume = price_data['volume']
            stock.last_updated = datetime.utcnow()

            # Update basic info from symbol_master
            stock.name = symbol_master.name
            stock.exchange = symbol_master.exchange

            # Update market cap and category
            if stock.market_cap:
                # Adjust existing market cap based on price change
                if stock.current_price and stock.current_price > 0:
                    price_ratio = price_data['price'] / stock.current_price
                    stock.market_cap = stock.market_cap * price_ratio
            else:
                stock.market_cap = self._estimate_market_cap(price_data['price'], price_data['volume'])

            stock.market_cap_category = self._determine_market_cap_category(stock.market_cap)

            # Update tradeability
            stock.is_tradeable = self._calculate_tradeability(price_data['price'], price_data['volume'])
            stock.is_active = True

            # Note: Verification status is tracked in symbol_master table

        except Exception as e:
            logger.error(f"Error updating stock {stock.symbol}: {e}")

    def _update_stock_with_price_data_from_dict(self, stock: Stock, price_data: Dict, symbol_info: Dict):
        """Update existing stock with current price data using symbol dict."""
        try:
            # Update price and volume
            stock.current_price = price_data['price']
            stock.volume = price_data['volume']
            stock.last_updated = datetime.utcnow()

            # Update basic info from symbol_info
            stock.name = symbol_info['name']
            stock.exchange = symbol_info['exchange']

            # Update market cap and category
            if stock.market_cap:
                # Adjust existing market cap based on price change
                if stock.current_price and stock.current_price > 0:
                    price_ratio = price_data['price'] / stock.current_price
                    stock.market_cap = stock.market_cap * price_ratio
            else:
                stock.market_cap = self._estimate_market_cap(price_data['price'], price_data['volume'])

            stock.market_cap_category = self._determine_market_cap_category(stock.market_cap)

            # Update tradeability
            stock.is_tradeable = self._calculate_tradeability(price_data['price'], price_data['volume'])
            stock.is_active = True

            # Note: Verification status is tracked in symbol_master table

        except Exception as e:
            logger.error(f"Error updating stock {stock.symbol}: {e}")

    def _create_stock_with_price_data(self, price_data: Dict, symbol_master: SymbolMaster) -> Optional[Stock]:
        """Create new stock with price data."""
        try:
            # Calculate market metrics
            market_cap = self._estimate_market_cap(price_data['price'], price_data['volume'])
            market_cap_category = self._determine_market_cap_category(market_cap)
            sector = self._determine_sector(symbol_master.name)
            is_tradeable = self._calculate_tradeability(price_data['price'], price_data['volume'])

            # Create stock record
            new_stock = Stock(
                symbol=symbol_master.symbol,
                name=symbol_master.name,
                exchange=symbol_master.exchange,
                sector=sector,
                current_price=price_data['price'],
                volume=price_data['volume'],
                market_cap=market_cap,
                market_cap_category=market_cap_category,
                is_active=True,
                is_tradeable=is_tradeable,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )

            return new_stock

        except Exception as e:
            logger.error(f"Error creating stock for {symbol_master.symbol}: {e}")
            return None

    def _create_stock_with_price_data_from_dict(self, price_data: Dict, symbol_info: Dict) -> Optional[Stock]:
        """Create new stock with price data using symbol dict."""
        try:
            # Calculate market metrics
            market_cap = self._estimate_market_cap(price_data['price'], price_data['volume'])
            market_cap_category = self._determine_market_cap_category(market_cap)
            sector = self._determine_sector(symbol_info['name'])
            is_tradeable = self._calculate_tradeability(price_data['price'], price_data['volume'])

            # Create stock record
            new_stock = Stock(
                symbol=symbol_info['symbol'],
                name=symbol_info['name'],
                exchange=symbol_info['exchange'],
                sector=sector,
                current_price=price_data['price'],
                volume=price_data['volume'],
                market_cap=market_cap,
                market_cap_category=market_cap_category,
                is_active=True,
                is_tradeable=is_tradeable,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )

            return new_stock

        except Exception as e:
            logger.error(f"Error creating stock for {symbol_info['symbol']}: {e}")
            return None

    def _estimate_market_cap(self, price: float, volume: int) -> float:
        """Estimate market cap based on price and volume patterns."""
        try:
            # Estimate shares outstanding based on price tiers
            if price > 2000:
                estimated_shares = 50000000    # 5 crore shares
            elif price > 500:
                estimated_shares = 200000000   # 20 crore shares
            elif price > 100:
                estimated_shares = 500000000   # 50 crore shares
            else:
                estimated_shares = 1000000000  # 100 crore shares

            # Adjust based on volume (higher volume = larger company typically)
            if volume > 500000:
                estimated_shares *= 3
            elif volume > 100000:
                estimated_shares *= 2
            elif volume > 50000:
                estimated_shares *= 1.5

            # Calculate market cap in crores
            market_cap_crores = (price * estimated_shares) / 10000000
            return market_cap_crores

        except Exception:
            return 1000.0  # Default

    def _determine_market_cap_category(self, market_cap_crores: float) -> str:
        """Determine market cap category."""
        if market_cap_crores > 20000:  # ₹20,000 crores
            return "large_cap"
        elif market_cap_crores > 5000:  # ₹5,000 crores
            return "mid_cap"
        else:
            return "small_cap"

    def _determine_sector(self, company_name: str) -> str:
        """Determine sector from company name."""
        name_upper = company_name.upper()

        if any(term in name_upper for term in ['BANK', 'FINANCE', 'FINANCIAL']):
            return 'Banking'
        elif any(term in name_upper for term in ['IT', 'TECH', 'SOFTWARE']):
            return 'Technology'
        elif any(term in name_upper for term in ['PHARMA', 'DRUG', 'HEALTHCARE']):
            return 'Pharmaceutical'
        elif any(term in name_upper for term in ['AUTO', 'MOTOR', 'TYRE']):
            return 'Automobile'
        elif any(term in name_upper for term in ['ENERGY', 'POWER', 'OIL']):
            return 'Energy'
        else:
            return 'Others'

    def _calculate_tradeability(self, price: float, volume: int) -> bool:
        """Calculate if stock is tradeable."""
        return (
            price >= 5.0 and        # Minimum price ₹5
            price <= 50000 and     # Maximum price ₹50,000
            volume >= 1000         # Minimum daily volume
        )

    def _get_initialization_statistics(self) -> Dict:
        """Get statistics after initialization."""
        try:
            with self.db_manager.get_session() as session:
                # Symbol master stats
                total_symbols = session.query(SymbolMaster).count()
                verified_symbols = session.query(SymbolMaster).filter(
                    SymbolMaster.is_fyers_verified == True
                ).count()

                # Stock stats
                total_stocks = session.query(Stock).count()
                active_stocks = session.query(Stock).filter(Stock.is_active == True).count()
                tradeable_stocks = session.query(Stock).filter(
                    and_(Stock.is_active == True, Stock.is_tradeable == True)
                ).count()

                return {
                    'symbol_master': {
                        'total': total_symbols,
                        'verified': verified_symbols,
                        'verification_rate': verified_symbols / total_symbols if total_symbols > 0 else 0
                    },
                    'stocks': {
                        'total': total_stocks,
                        'active': active_stocks,
                        'tradeable': tradeable_stocks
                    },
                    'sync_rate': total_stocks / verified_symbols if verified_symbols > 0 else 0
                }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}


# Global service instance
_stock_initialization_service = None

def get_stock_initialization_service() -> StockInitializationService:
    """Get the global stock initialization service instance."""
    global _stock_initialization_service
    if _stock_initialization_service is None:
        _stock_initialization_service = StockInitializationService()
    return _stock_initialization_service