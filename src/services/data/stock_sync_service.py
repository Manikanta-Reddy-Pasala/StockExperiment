"""
Comprehensive Stock Synchronization Service

This service provides robust stock-by-stock data synchronization with:
- Individual stock processing for better error handling
- Fytoken-based matching and updates
- Real-time price and amount validation
- Startup initialization support
- Comprehensive logging and monitoring
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
except ImportError:
    from src.services.core.unified_broker_service import get_unified_broker_service
    from src.models.database import get_database_manager
    from src.models.stock_models import Stock, SymbolMaster


class StockSyncService:
    """Comprehensive stock synchronization service for real-time data updates."""

    def __init__(self):
        self.unified_broker_service = get_unified_broker_service()
        self.db_manager = get_database_manager()
        self.rate_limit_delay = 0.1  # 100ms between individual stock calls
        self.batch_size = 1  # Process stocks one by one for better error handling

    def initialize_stocks_on_startup(self, user_id: int = 1) -> Dict:
        """
        Initialize stocks from broker to database during application startup.
        This ensures the database has current stock data before the app starts.
        """
        logger.info("🚀 Starting stock initialization on application startup")

        try:
            # Step 1: Load symbol master data
            symbol_result = self._load_symbol_master_data(user_id)
            if not symbol_result['success']:
                return symbol_result

            # Step 2: Sync stock data with real-time prices
            sync_result = self._sync_all_stocks_comprehensive(user_id)

            # Step 3: Update database statistics
            stats = self._get_sync_statistics()

            result = {
                'success': True,
                'symbol_master': symbol_result,
                'stock_sync': sync_result,
                'statistics': stats,
                'initialization_time': datetime.utcnow().isoformat()
            }

            logger.info(f"✅ Stock initialization completed successfully: {result}")
            return result

        except Exception as e:
            logger.error(f"❌ Stock initialization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stage': 'initialization'
            }

    def _load_symbol_master_data(self, user_id: int) -> Dict:
        """Load comprehensive symbol master data from broker."""
        try:
            logger.info("📊 Loading symbol master data from broker...")

            # Use existing stock master service to load symbol data
            from ..ml.stock_master_service import get_stock_master_service
            stock_master_service = get_stock_master_service()

            # Load all available symbols
            result = stock_master_service.refresh_all_stocks(user_id)

            if result.get('success'):
                logger.info(f"✅ Symbol master loaded: {result.get('total_symbols_found', 0)} symbols")
                return result
            else:
                logger.error(f"❌ Symbol master loading failed: {result.get('error')}")
                return result

        except Exception as e:
            logger.error(f"❌ Error loading symbol master data: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _sync_all_stocks_comprehensive(self, user_id: int) -> Dict:
        """Comprehensive stock-by-stock synchronization with real-time data."""
        try:
            logger.info("🔄 Starting comprehensive stock-by-stock synchronization...")

            with self.db_manager.get_session() as session:
                # Get all verified symbols with fytoken
                verified_symbols = session.query(SymbolMaster).filter(
                    and_(
                        SymbolMaster.is_active == True,
                        SymbolMaster.is_equity == True,
                        SymbolMaster.is_fyers_verified == True,
                        SymbolMaster.fytoken.isnot(None)
                    )
                ).limit(50).all()  # Limit to 50 for testing

                logger.info(f"📈 Found {len(verified_symbols)} verified symbols to sync")

                # Extract data to avoid session issues
                symbol_data = []
                for sm in verified_symbols:
                    symbol_data.append({
                        'symbol': sm.symbol,
                        'fytoken': sm.fytoken,
                        'name': sm.name,
                        'exchange': sm.exchange
                    })

                # Process each stock individually
                results = {
                    'total_symbols': len(symbol_data),
                    'processed': 0,
                    'updated': 0,
                    'created': 0,
                    'errors': 0,
                    'error_details': []
                }

                for i, symbol_info in enumerate(symbol_data):
                    try:
                        logger.info(f"🔄 Processing stock {i+1}/{len(symbol_data)}: {symbol_info['symbol']}")
                        # Process individual stock
                        stock_result = self._process_individual_stock_with_data(session, symbol_info, user_id)
                        logger.info(f"📊 Stock {symbol_info['symbol']} result: {stock_result}")

                        # Update counters
                        results['processed'] += 1
                        if stock_result['action'] == 'updated':
                            results['updated'] += 1
                        elif stock_result['action'] == 'created':
                            results['created'] += 1

                        # Log progress every 10 stocks
                        if (i + 1) % 10 == 0:
                            logger.info(f"📊 Progress: {i + 1}/{len(symbol_data)} stocks processed")

                        # Rate limiting to avoid overwhelming the API
                        time.sleep(self.rate_limit_delay)

                    except Exception as e:
                        results['errors'] += 1
                        error_detail = {
                            'symbol': symbol_info['symbol'],
                            'fytoken': symbol_info['fytoken'],
                            'error': str(e)
                        }
                        results['error_details'].append(error_detail)
                        logger.warning(f"⚠️  Error processing {symbol_info['symbol']}: {e}")
                        continue

                # Commit all changes
                session.commit()

                results['success'] = True
                logger.info(f"✅ Stock synchronization completed: {results}")
                return results

        except Exception as e:
            logger.error(f"❌ Comprehensive stock sync failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _process_individual_stock_with_data(self, session, symbol_info: Dict, user_id: int) -> Dict:
        """Process a single stock with real-time data validation using extracted data."""
        try:
            symbol = symbol_info['symbol']
            fytoken = symbol_info['fytoken']

            # Get real-time quote for this specific stock
            quote_result = self._get_individual_stock_quote(symbol, user_id)

            # Check for both 'success' and 'status' fields to handle different broker response formats
            if not quote_result.get('success', False) and not quote_result.get('status') == 'success':
                logger.debug(f"❌ No quote data for {symbol}: {quote_result.get('error')}")
                return {'action': 'skipped', 'reason': 'no_quote_data'}

            quote_data = quote_result['data']

            # Find existing stock by symbol (primary lookup)
            existing_stock = session.query(Stock).filter_by(symbol=symbol).first()
            logger.info(f"🔍 Checking for existing stock {symbol}: {'Found' if existing_stock else 'Not found'}")

            if existing_stock:
                # Update existing stock with real-time data
                logger.info(f"📝 Updating existing stock {symbol}")
                update_result = self._update_stock_with_realtime_data_from_info(existing_stock, quote_data, symbol_info)
                return {'action': 'updated', 'stock_id': existing_stock.id, 'details': update_result}
            else:
                # Create new stock with real-time data
                logger.info(f"🆕 Creating new stock {symbol}")
                new_stock = self._create_stock_with_realtime_data_from_info(symbol_info, quote_data)
                if new_stock:
                    session.add(new_stock)
                    logger.info(f"✅ Successfully created new stock {symbol}")
                    return {'action': 'created', 'stock_id': 'new', 'symbol': symbol}
                else:
                    logger.warning(f"❌ Failed to create new stock {symbol}")
                    return {'action': 'skipped', 'reason': 'creation_failed'}

        except Exception as e:
            logger.warning(f"❌ Error processing individual stock {symbol_info['symbol']}: {e}")
            raise

    def _process_individual_stock(self, session, symbol_master: SymbolMaster, user_id: int) -> Dict:
        """Process a single stock with real-time data validation."""
        try:
            symbol = symbol_master.symbol
            fytoken = symbol_master.fytoken

            # Get real-time quote for this specific stock
            quote_result = self._get_individual_stock_quote(symbol, user_id)

            # Check for both 'success' and 'status' fields to handle different broker response formats
            if not quote_result.get('success', False) and not quote_result.get('status') == 'success':
                logger.debug(f"❌ No quote data for {symbol}: {quote_result.get('error')}")
                return {'action': 'skipped', 'reason': 'no_quote_data'}

            quote_data = quote_result['data']

            # Find existing stock by symbol (primary lookup)
            existing_stock = session.query(Stock).filter_by(symbol=symbol).first()

            if existing_stock:
                # Update existing stock with real-time data
                update_result = self._update_stock_with_realtime_data(existing_stock, quote_data, symbol_master)
                return {'action': 'updated', 'stock_id': existing_stock.id, 'details': update_result}
            else:
                # Create new stock with real-time data
                new_stock = self._create_stock_with_realtime_data(symbol_master, quote_data)
                if new_stock:
                    session.add(new_stock)
                    return {'action': 'created', 'stock_id': 'new', 'symbol': symbol}
                else:
                    return {'action': 'skipped', 'reason': 'creation_failed'}

        except Exception as e:
            logger.warning(f"❌ Error processing individual stock {symbol_master.symbol}: {e}")
            raise

    def _get_individual_stock_quote(self, symbol: str, user_id: int) -> Dict:
        """Get real-time quote for a single stock with validation."""
        try:
            # Use unified broker service to get quote for single symbol
            result = self.unified_broker_service.get_quotes(user_id, symbols=[symbol])

            if result.get('success') and result.get('data'):
                quote_data = result['data'].get(symbol)
                if quote_data:
                    # Validate quote data quality
                    validation_result = self._validate_quote_data(symbol, quote_data)
                    if validation_result['valid']:
                        return {
                            'success': True,
                            'data': quote_data,
                            'validation': validation_result
                        }
                    else:
                        return {
                            'success': False,
                            'error': f"Invalid quote data: {validation_result['errors']}"
                        }
                else:
                    return {
                        'success': False,
                        'error': 'No data in response'
                    }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _validate_quote_data(self, symbol: str, quote_data: Dict) -> Dict:
        """Comprehensive validation of quote data."""
        errors = []
        warnings = []

        try:
            # Extract price information
            price = float(quote_data.get('lp', quote_data.get('ltp', quote_data.get('last_price', 0))))
            volume = int(quote_data.get('volume', quote_data.get('vol', 0)))

            # Price validation
            if price <= 0:
                errors.append(f"Invalid price: {price}")
            elif price < 1.0:
                warnings.append(f"Very low price: ₹{price:.2f}")
            elif price > 100000:
                warnings.append(f"Very high price: ₹{price:.2f}")

            # Volume validation
            if volume < 0:
                errors.append(f"Negative volume: {volume}")
            elif volume == 0:
                warnings.append("Zero volume")
            elif volume > 100000000:
                warnings.append(f"Very high volume: {volume:,}")

            # Additional fields validation
            required_fields = ['lp', 'volume']
            available_fields = [field for field in required_fields if field in quote_data]

            if len(available_fields) < 1:
                errors.append("Missing required price fields")

            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'price': price,
                'volume': volume
            }

        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }

    def _update_stock_with_realtime_data_from_info(self, stock: Stock, quote_data: Dict, symbol_info: Dict) -> Dict:
        """Update existing stock with real-time market data using extracted symbol info."""
        try:
            # Validate quote data first
            validation = self._validate_quote_data(stock.symbol, quote_data)
            if not validation['valid']:
                logger.warning(f"⚠️  Invalid quote data for {stock.symbol}: {validation['errors']}")
                return {'updated': False, 'reason': 'invalid_data'}

            # Update price and volume data
            old_price = stock.current_price
            new_price = validation['price']
            old_volume = stock.volume
            new_volume = validation['volume']

            stock.current_price = new_price
            stock.volume = new_volume
            stock.last_updated = datetime.utcnow()

            # Update additional market data if available
            self._update_additional_market_data(stock, quote_data)

            # Update market cap and category if price changed significantly
            if old_price and abs(new_price - old_price) / old_price > 0.05:  # 5% change
                stock.market_cap = self._calculate_market_cap(new_price, new_volume, stock)
                stock.market_cap_category = self._determine_market_cap_category(stock.market_cap)

            # Update tradeability
            stock.is_tradeable = self._calculate_tradeability(new_price, new_volume)
            stock.is_active = True

            # Synchronize with symbol master data
            stock.name = symbol_info['name']
            stock.exchange = symbol_info['exchange']

            # Log significant changes
            if old_price and abs(new_price - old_price) / old_price > 0.10:  # 10% change
                logger.info(f"📈 Significant price change for {stock.symbol}: ₹{old_price:.2f} → ₹{new_price:.2f}")

            return {
                'updated': True,
                'price_change': new_price - (old_price or 0),
                'volume_change': new_volume - (old_volume or 0),
                'warnings': validation['warnings']
            }

        except Exception as e:
            logger.error(f"❌ Error updating stock {stock.symbol}: {e}")
            return {'updated': False, 'reason': str(e)}

    def _create_stock_with_realtime_data_from_info(self, symbol_info: Dict, quote_data: Dict) -> Optional[Stock]:
        """Create new stock with real-time market data using extracted symbol info."""
        try:
            # Validate quote data first
            validation = self._validate_quote_data(symbol_info['symbol'], quote_data)
            logger.info(f"🔍 Validation for {symbol_info['symbol']}: {validation}")
            if not validation['valid']:
                logger.warning(f"⚠️  Invalid quote data for new stock {symbol_info['symbol']}: {validation['errors']}")
                return None

            price = validation['price']
            volume = validation['volume']

            # Calculate market metrics
            market_cap = self._calculate_market_cap(price, volume)
            market_cap_category = self._determine_market_cap_category(market_cap)
            sector = self._determine_sector(symbol_info['name'])
            is_tradeable = self._calculate_tradeability(price, volume)

            # Create new stock record
            new_stock = Stock(
                symbol=symbol_info['symbol'],
                name=symbol_info['name'],
                exchange=symbol_info['exchange'],
                sector=sector,
                current_price=price,
                volume=volume,
                market_cap=market_cap,
                market_cap_category=market_cap_category,
                is_active=True,
                is_tradeable=is_tradeable,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )

            # Add additional market data if available
            self._update_additional_market_data(new_stock, quote_data)

            logger.info(f"✅ Created new stock: {symbol_info['symbol']} at ₹{price:.2f}")
            return new_stock

        except Exception as e:
            logger.error(f"❌ Error creating new stock {symbol_info['symbol']}: {e}")
            return None

    def _update_stock_with_realtime_data(self, stock: Stock, quote_data: Dict, symbol_master: SymbolMaster) -> Dict:
        """Update existing stock with real-time market data."""
        try:
            # Validate quote data first
            validation = self._validate_quote_data(stock.symbol, quote_data)
            if not validation['valid']:
                logger.warning(f"⚠️  Invalid quote data for {stock.symbol}: {validation['errors']}")
                return {'updated': False, 'reason': 'invalid_data'}

            # Update price and volume data
            old_price = stock.current_price
            new_price = validation['price']
            old_volume = stock.volume
            new_volume = validation['volume']

            stock.current_price = new_price
            stock.volume = new_volume
            stock.last_updated = datetime.utcnow()

            # Update additional market data if available
            self._update_additional_market_data(stock, quote_data)

            # Update market cap and category if price changed significantly
            if old_price and abs(new_price - old_price) / old_price > 0.05:  # 5% change
                stock.market_cap = self._calculate_market_cap(new_price, new_volume, stock)
                stock.market_cap_category = self._determine_market_cap_category(stock.market_cap)

            # Update tradeability
            stock.is_tradeable = self._calculate_tradeability(new_price, new_volume)
            stock.is_active = True

            # Synchronize with symbol master data
            stock.name = symbol_master.name
            stock.exchange = symbol_master.exchange

            # Log significant changes
            if old_price and abs(new_price - old_price) / old_price > 0.10:  # 10% change
                logger.info(f"📈 Significant price change for {stock.symbol}: ₹{old_price:.2f} → ₹{new_price:.2f}")

            return {
                'updated': True,
                'price_change': new_price - (old_price or 0),
                'volume_change': new_volume - (old_volume or 0),
                'warnings': validation['warnings']
            }

        except Exception as e:
            logger.error(f"❌ Error updating stock {stock.symbol}: {e}")
            return {'updated': False, 'reason': str(e)}

    def _create_stock_with_realtime_data(self, symbol_master: SymbolMaster, quote_data: Dict) -> Optional[Stock]:
        """Create new stock with real-time market data."""
        try:
            # Validate quote data first
            validation = self._validate_quote_data(symbol_master.symbol, quote_data)
            if not validation['valid']:
                logger.warning(f"⚠️  Invalid quote data for new stock {symbol_master.symbol}: {validation['errors']}")
                return None

            price = validation['price']
            volume = validation['volume']

            # Calculate market metrics
            market_cap = self._calculate_market_cap(price, volume)
            market_cap_category = self._determine_market_cap_category(market_cap)
            sector = self._determine_sector(symbol_master.name)
            is_tradeable = self._calculate_tradeability(price, volume)

            # Create new stock record
            new_stock = Stock(
                symbol=symbol_master.symbol,
                name=symbol_master.name,
                exchange=symbol_master.exchange,
                sector=sector,
                current_price=price,
                volume=volume,
                market_cap=market_cap,
                market_cap_category=market_cap_category,
                is_active=True,
                is_tradeable=is_tradeable,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )

            # Add additional market data if available
            self._update_additional_market_data(new_stock, quote_data)

            logger.info(f"✅ Created new stock: {symbol_master.symbol} at ₹{price:.2f}")
            return new_stock

        except Exception as e:
            logger.error(f"❌ Error creating new stock {symbol_master.symbol}: {e}")
            return None

    def _update_additional_market_data(self, stock: Stock, quote_data: Dict):
        """Update additional market data fields from quote."""
        try:
            # Update PE ratio if available
            if 'pe' in quote_data:
                stock.pe_ratio = float(quote_data['pe'])

            # Update 52-week high/low if available
            if 'high_52' in quote_data:
                # Store in a custom field or log for analysis
                pass

            # Update market cap if available directly from quote
            if 'market_cap' in quote_data:
                stock.market_cap = float(quote_data['market_cap'])
                stock.market_cap_category = self._determine_market_cap_category(stock.market_cap)

            # Update dividend yield if available
            if 'dividend_yield' in quote_data:
                stock.dividend_yield = float(quote_data['dividend_yield'])

        except Exception as e:
            logger.debug(f"Could not update additional data for {stock.symbol}: {e}")

    def _calculate_market_cap(self, price: float, volume: int, existing_stock: Stock = None) -> float:
        """Calculate estimated market cap based on price and volume patterns."""
        try:
            # Use existing market cap as base if available
            if existing_stock and existing_stock.market_cap:
                # Adjust existing market cap based on price change
                if existing_stock.current_price and existing_stock.current_price > 0:
                    price_ratio = price / existing_stock.current_price
                    return existing_stock.market_cap * price_ratio

            # Estimate shares outstanding based on price and volume patterns
            if price > 2000:  # High-price stocks typically have fewer shares
                estimated_shares = 50000000  # 5 crore shares
            elif price > 500:
                estimated_shares = 200000000  # 20 crore shares
            elif price > 100:
                estimated_shares = 500000000  # 50 crore shares
            else:
                estimated_shares = 1000000000  # 100 crore shares

            # Adjust based on volume (higher volume = larger company typically)
            if volume > 500000:
                estimated_shares *= 3  # Large companies
            elif volume > 100000:
                estimated_shares *= 2  # Mid-size companies
            elif volume > 50000:
                estimated_shares *= 1.5  # Smaller companies

            # Calculate market cap in crores
            market_cap_crores = (price * estimated_shares) / 10000000
            return market_cap_crores

        except Exception as e:
            logger.debug(f"Error calculating market cap: {e}")
            return 1000.0  # Default to 1000 crores

    def _determine_market_cap_category(self, market_cap_crores: float) -> str:
        """Determine market cap category based on Indian market standards."""
        if market_cap_crores > 20000:  # ₹20,000 crores
            return "large_cap"
        elif market_cap_crores > 5000:  # ₹5,000 crores
            return "mid_cap"
        else:
            return "small_cap"

    def _determine_sector(self, company_name: str) -> str:
        """Determine sector from company name using keyword analysis."""
        name_upper = company_name.upper()

        # Banking & Financial
        if any(term in name_upper for term in ['BANK', 'FINANCE', 'FINANCIAL', 'CAPITAL', 'CREDIT']):
            return 'Banking'

        # Technology
        elif any(term in name_upper for term in ['IT', 'TECH', 'SOFTWARE', 'COMPUTER', 'DIGITAL', 'INFO']):
            return 'Technology'

        # Pharmaceutical
        elif any(term in name_upper for term in ['PHARMA', 'DRUG', 'MEDICINE', 'HEALTHCARE', 'BIO', 'MEDICAL']):
            return 'Pharmaceutical'

        # Automobile
        elif any(term in name_upper for term in ['AUTO', 'MOTOR', 'VEHICLE', 'TYRE', 'TYRES']):
            return 'Automobile'

        # Energy
        elif any(term in name_upper for term in ['ENERGY', 'POWER', 'OIL', 'GAS', 'PETROLEUM', 'SOLAR']):
            return 'Energy'

        # FMCG
        elif any(term in name_upper for term in ['CONSUMER', 'FOOD', 'BEVERAGE', 'FMCG', 'PRODUCTS']):
            return 'FMCG'

        # Infrastructure
        elif any(term in name_upper for term in ['CEMENT', 'CONSTRUCTION', 'INFRA', 'BUILDING', 'ENGINEERING']):
            return 'Infrastructure'

        # Metals
        elif any(term in name_upper for term in ['STEEL', 'METAL', 'IRON', 'COAL', 'MINING', 'ALUMINIUM']):
            return 'Metals'

        else:
            return 'Others'

    def _calculate_tradeability(self, price: float, volume: int) -> bool:
        """Calculate if stock is tradeable based on liquidity criteria."""
        return (
            price >= 5.0 and          # Minimum price ₹5
            price <= 50000 and       # Maximum price ₹50,000
            volume >= 1000           # Minimum daily volume
        )

    def _get_sync_statistics(self) -> Dict:
        """Get comprehensive synchronization statistics."""
        try:
            with self.db_manager.get_session() as session:
                # Overall statistics
                total_stocks = session.query(Stock).count()
                active_stocks = session.query(Stock).filter(Stock.is_active == True).count()
                tradeable_stocks = session.query(Stock).filter(
                    and_(Stock.is_active == True, Stock.is_tradeable == True)
                ).count()

                # By market cap category
                large_cap = session.query(Stock).filter(
                    and_(Stock.is_active == True, Stock.market_cap_category == 'large_cap')
                ).count()
                mid_cap = session.query(Stock).filter(
                    and_(Stock.is_active == True, Stock.market_cap_category == 'mid_cap')
                ).count()
                small_cap = session.query(Stock).filter(
                    and_(Stock.is_active == True, Stock.market_cap_category == 'small_cap')
                ).count()

                # Price ranges
                under_100 = session.query(Stock).filter(
                    and_(Stock.is_active == True, Stock.current_price < 100)
                ).count()
                between_100_1000 = session.query(Stock).filter(
                    and_(Stock.is_active == True, Stock.current_price.between(100, 1000))
                ).count()
                over_1000 = session.query(Stock).filter(
                    and_(Stock.is_active == True, Stock.current_price > 1000)
                ).count()

                # Recently updated
                recent_threshold = datetime.utcnow() - timedelta(hours=24)
                recently_updated = session.query(Stock).filter(
                    Stock.last_updated > recent_threshold
                ).count()

                return {
                    'total_stocks': total_stocks,
                    'active_stocks': active_stocks,
                    'tradeable_stocks': tradeable_stocks,
                    'market_cap_distribution': {
                        'large_cap': large_cap,
                        'mid_cap': mid_cap,
                        'small_cap': small_cap
                    },
                    'price_distribution': {
                        'under_100': under_100,
                        'between_100_1000': between_100_1000,
                        'over_1000': over_1000
                    },
                    'recently_updated': recently_updated,
                    'data_freshness': recently_updated / total_stocks if total_stocks > 0 else 0
                }

        except Exception as e:
            logger.error(f"Error getting sync statistics: {e}")
            return {}

    def sync_specific_stocks(self, symbols: List[str], user_id: int = 1) -> Dict:
        """Sync specific stocks by symbol list."""
        try:
            logger.info(f"🔄 Syncing specific stocks: {symbols}")

            results = {
                'requested': len(symbols),
                'processed': 0,
                'updated': 0,
                'errors': 0,
                'details': []
            }

            with self.db_manager.get_session() as session:
                for symbol in symbols:
                    try:
                        # Find symbol master record
                        symbol_master = session.query(SymbolMaster).filter_by(symbol=symbol).first()
                        if not symbol_master:
                            results['errors'] += 1
                            results['details'].append({
                                'symbol': symbol,
                                'status': 'error',
                                'message': 'Symbol not found in master data'
                            })
                            continue

                        # Process the stock
                        stock_result = self._process_individual_stock(session, symbol_master, user_id)
                        results['processed'] += 1

                        if stock_result['action'] in ['updated', 'created']:
                            results['updated'] += 1

                        results['details'].append({
                            'symbol': symbol,
                            'status': 'success',
                            'action': stock_result['action']
                        })

                    except Exception as e:
                        results['errors'] += 1
                        results['details'].append({
                            'symbol': symbol,
                            'status': 'error',
                            'message': str(e)
                        })
                        logger.warning(f"Error syncing {symbol}: {e}")

                session.commit()

            results['success'] = results['errors'] == 0
            return results

        except Exception as e:
            logger.error(f"Error in specific stock sync: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Global service instance
_stock_sync_service = None

def get_stock_sync_service() -> StockSyncService:
    """Get the global stock sync service instance."""
    global _stock_sync_service
    if _stock_sync_service is None:
        _stock_sync_service = StockSyncService()
    return _stock_sync_service
