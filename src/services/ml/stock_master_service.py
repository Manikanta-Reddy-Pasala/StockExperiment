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
        Get all available symbols from comprehensive CSV data.
        Uses Fyers symbol service to get complete symbol list instead of limited search.
        """
        all_symbols = set()

        try:
            logger.info(f"Getting comprehensive symbol list for {exchange} using CSV data")

            # Use Fyers symbol service to get all equity symbols from CSV
            from ..data.fyers_symbol_service import get_fyers_symbol_service
            fyers_service = get_fyers_symbol_service()

            # Get all equity symbols (this uses the comprehensive CSV data)
            equity_symbols = fyers_service.get_equity_symbols(exchange)

            logger.info(f"Retrieved {len(equity_symbols)} equity symbols from CSV data")

            # Extract symbol names and store in symbol_master table
            with self.db_manager.get_session() as session:
                from ...models.stock_models import SymbolMaster

                symbols_added = 0
                for symbol_data in equity_symbols:
                    original_symbol = symbol_data.get('symbol', '')
                    if original_symbol and '-EQ' in original_symbol:  # Only equity symbols
                        # Clean the symbol before storing
                        cleaned_symbol = self._clean_symbol(original_symbol)
                        all_symbols.add(cleaned_symbol)

                        # Store in symbol_master table for future reference
                        try:
                            # Smart refresh logic: fytoken is primary key, so direct lookup
                            fytoken = symbol_data.get('fytoken', '')
                            if not fytoken:
                                logger.warning(f"No fytoken for symbol {cleaned_symbol}, skipping")
                                continue

                            # Check if this fytoken already exists (primary key lookup)
                            existing_record = session.query(SymbolMaster).filter(
                                SymbolMaster.fytoken == fytoken
                            ).first()

                            if existing_record:
                                # Update existing record with latest data while preserving verification
                                logger.info(f"Updating existing record for fytoken {fytoken}: {existing_record.symbol} -> {cleaned_symbol}")

                                # Update basic symbol data
                                existing_record.symbol = cleaned_symbol
                                existing_record.name = symbol_data.get('name', existing_record.name)
                                existing_record.exchange = exchange
                                existing_record.segment = symbol_data.get('segment', existing_record.segment)
                                existing_record.instrument_type = symbol_data.get('instrument_type', existing_record.instrument_type)
                                existing_record.lot_size = symbol_data.get('lot', existing_record.lot_size)
                                existing_record.tick_size = symbol_data.get('tick', existing_record.tick_size)
                                existing_record.isin = symbol_data.get('isin', existing_record.isin)
                                existing_record.updated_at = datetime.utcnow()

                                # Verification data is automatically preserved (not overwritten)
                                logger.info(f"Preserved verification status: {getattr(existing_record, 'is_fyers_verified', False)}")

                            else:
                                # Create new record
                                # Create symbol with preserved verification data if applicable
                                symbol_master_data = {
                                    'symbol': cleaned_symbol,
                                    'fytoken': symbol_data.get('fytoken', ''),
                                    'name': symbol_data.get('name', ''),
                                    'exchange': exchange,
                                    'segment': symbol_data.get('segment', 'CM'),
                                    'instrument_type': symbol_data.get('instrument_type', 'EQ'),
                                    'lot_size': symbol_data.get('lot', 1),
                                    'tick_size': symbol_data.get('tick', 0.05),
                                    'isin': symbol_data.get('isin', ''),
                                    'data_source': 'fyers',
                                    'is_active': True,
                                    'is_equity': True
                                }

                                # Preserve verification data if applicable
                                if should_preserve_verification and existing_verification_data:
                                    symbol_master_data.update({
                                        'is_fyers_verified': existing_verification_data['is_fyers_verified'],
                                        'verification_date': existing_verification_data['verification_date'],
                                        'verification_error': existing_verification_data['verification_error'],
                                        'last_quote_check': existing_verification_data['last_quote_check']
                                    })

                                symbol_master = SymbolMaster(**symbol_master_data)
                                session.add(symbol_master)
                                symbols_added += 1

                        except Exception as e:
                            logger.warning(f"Error storing symbol {cleaned_symbol} in symbol_master: {e}")
                            continue

                # Commit all symbol_master entries
                session.commit()
                logger.info(f"Stored {symbols_added} new symbols in symbol_master table")

            # Fall back to search method if CSV method fails
            if not all_symbols:
                logger.warning("CSV method failed, falling back to search-based discovery")
                return self._get_symbols_search_fallback(user_id, exchange)

            result = sorted(list(all_symbols))
            logger.info(f"Found {len(result)} unique symbols from {exchange} CSV data")
            return result

        except Exception as e:
            logger.error(f"Error getting comprehensive symbols from CSV: {e}")
            # Fall back to search method
            return self._get_symbols_search_fallback(user_id, exchange)

    def _get_symbols_search_fallback(self, user_id: int, exchange: str) -> List[str]:
        """
        Fallback method: Get symbols using search-based discovery.
        """
        all_symbols = set()

        try:
            logger.info(f"Using search-based fallback for {exchange}")

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

            # Use alphabet-based search as fallback
            search_patterns = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z

            for pattern in search_patterns:
                try:
                    search_result = provider.fyers_service.search(user_id, pattern, exchange)

                    if search_result.get('status') == 'success':
                        symbols = search_result.get('data', [])
                        logger.debug(f"Search pattern '{pattern}' returned {len(symbols)} results")

                        for symbol_info in symbols:
                            original_symbol = symbol_info.get('symbol', '')
                            name = symbol_info.get('name', symbol_info.get('symbol_name', ''))

                            # Filter for equity stocks only
                            if (original_symbol.startswith(f"{exchange}:") and
                                '-EQ' in original_symbol and
                                len(name) > 2):
                                # Clean the symbol before adding to collection
                                cleaned_symbol = self._clean_symbol(original_symbol)
                                all_symbols.add(cleaned_symbol)

                    # Rate limiting
                    time.sleep(0.2)

                except Exception as e:
                    logger.warning(f"Search failed for pattern '{pattern}': {e}")
                    continue

            logger.info(f"Search fallback found {len(all_symbols)} unique symbols")
            return list(all_symbols)

        except Exception as e:
            logger.error(f"Error in search fallback: {e}")
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

    def _clean_symbol(self, symbol: str) -> str:
        """
        Clean symbol to ensure Fyers API compatibility.
        Removes problematic characters that cause 'invalid input' errors.
        """
        if not symbol:
            return symbol

        # Handle common problematic patterns
        cleaned_symbol = symbol

        # Replace ampersands with appropriate text
        ampersand_mappings = {
            'ARE&M': 'AREAM',
            'GMRP&UI': 'GMRPUI',
            'GVT&D': 'GVTD',
            'J&KBANK': 'JKBANK',
            'M&M': 'MM',
            'M&MFIN': 'MMFIN',
            'S&SPOWER': 'SSPOWER',
            'SURANAT&P': 'SURANATPHARMA'
        }

        # Extract the base symbol for checking
        if ':' in cleaned_symbol and '-EQ' in cleaned_symbol:
            prefix = cleaned_symbol.split(':')[0] + ':'
            suffix = '-EQ'
            base_symbol = cleaned_symbol.replace(prefix, '').replace(suffix, '')

            # Apply ampersand mappings
            if base_symbol in ampersand_mappings:
                cleaned_symbol = f"{prefix}{ampersand_mappings[base_symbol]}{suffix}"
            else:
                # General ampersand cleaning (remove & entirely)
                if '&' in base_symbol:
                    # Replace & with empty string, but handle common cases
                    base_symbol = base_symbol.replace('&', '')
                    cleaned_symbol = f"{prefix}{base_symbol}{suffix}"

        # Handle double hyphens in company names
        double_hyphen_mappings = {
            'BAJAJ-AUTO': 'BAJAJAUTO',
            'HCL-INSYS': 'HCLINSYS',
            'NAM-INDIA': 'NAMINDIA',
            'UMIYA-MRO': 'UMIYAMRO'
        }

        if ':' in cleaned_symbol and '-EQ' in cleaned_symbol:
            prefix = cleaned_symbol.split(':')[0] + ':'
            suffix = '-EQ'
            base_symbol = cleaned_symbol.replace(prefix, '').replace(suffix, '')

            # Apply double hyphen mappings
            if base_symbol in double_hyphen_mappings:
                cleaned_symbol = f"{prefix}{double_hyphen_mappings[base_symbol]}{suffix}"
            else:
                # General double hyphen cleaning - only keep the first part before first hyphen
                # unless it's the standard -EQ suffix
                if base_symbol.count('-') > 0:
                    # Keep only alphanumeric characters for the base part
                    # This removes extra hyphens while preserving the main symbol
                    base_parts = base_symbol.split('-')
                    if len(base_parts) > 1:
                        # Take the first part and any numeric parts, remove extra text parts
                        clean_base = base_parts[0]
                        for part in base_parts[1:]:
                            if part.isdigit() or len(part) <= 2:  # Keep short suffixes and numbers
                                clean_base += part
                        cleaned_symbol = f"{prefix}{clean_base}{suffix}"

        # Remove any other special characters that might cause issues
        special_chars = ['[', ']', '(', ')', '/', '\\', '+', '=', '|', '<', '>', '?', '*']
        for char in special_chars:
            if char in cleaned_symbol:
                cleaned_symbol = cleaned_symbol.replace(char, '')

        # Log if symbol was changed
        if cleaned_symbol != symbol:
            logger.info(f"Cleaned symbol: {symbol} → {cleaned_symbol}")

        return cleaned_symbol

    def _create_new_stock(self, symbol: str, quote_data: Dict, exchange: str) -> Optional[Stock]:
        """Create a new stock record."""
        try:
            # Clean the symbol first to prevent API compatibility issues
            cleaned_symbol = self._clean_symbol(symbol)

            # Extract basic data
            current_price = float(quote_data.get('lp', quote_data.get('ltp', quote_data.get('last_price', 0))))
            volume = int(quote_data.get('volume', quote_data.get('vol', 0)))

            if current_price <= 0 or volume <= 0:
                return None

            # Extract name from symbol
            name = cleaned_symbol.replace(f"{exchange}:", "").replace("-EQ", "")
            if 'symbol_name' in quote_data:
                name = quote_data['symbol_name']

            # Calculate market cap and determine category
            market_cap = self._estimate_market_cap(current_price, volume)
            market_cap_category = self._determine_market_cap_category(market_cap)

            # Determine sector
            sector = self._determine_sector(name)

            # Calculate tradeability
            is_tradeable = self._calculate_tradeability(current_price, volume, quote_data)

            # Create new stock with cleaned symbol
            new_stock = Stock(
                symbol=cleaned_symbol,
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
        """Determine sector from stock name using comprehensive keyword mapping."""
        name_upper = name.upper()

        # Banking & Financial Services - Comprehensive list
        banking_keywords = [
            'BANK', 'FINANCE', 'FINANCIAL', 'LENDING', 'CREDIT', 'CAPITAL', 'SECURITIES',
            'MUTUAL', 'ASSET', 'HOUSING', 'HFC', 'NBFC', 'LEASING', 'INVESTMENTS',
            'EQUITY', 'WEALTH', 'INSURANCE', 'LIFE', 'GENERAL', 'BROKING', 'TRADING',
            'FUNDS', 'INVEST', 'LOANS', 'MORTGAG', 'MICROFINANCE', 'MFIN'
        ]

        # Technology & IT - Extended with more patterns
        technology_keywords = [
            'IT', 'TECH', 'SOFTWARE', 'SYSTEM', 'COMPUTER', 'DIGITAL', 'INFO', 'DATA',
            'CYBER', 'SOLUTIONS', 'SERVICES', 'AUTOMATION', 'ANALYTICS', 'AI', 'CLOUD',
            'MICRO', 'ELECTRONICS', 'SEMICONDUCTOR', 'CHIPS', 'COMMUNICATIONS', 'COMM',
            'SILICON', 'PROCESSORS', 'MEMORY', 'DEVICES', 'INFOTECH', 'DATACOM',
            'NETWORKS', 'TELESERVICES', 'ONLINE', 'WEB', 'INTERNET', 'APPS'
        ]

        # Pharmaceutical & Healthcare - More medical terms
        pharma_keywords = [
            'PHARMA', 'DRUG', 'MEDICINE', 'HEALTHCARE', 'BIO', 'LABORATORY', 'LABS',
            'MEDICAL', 'HEALTH', 'DIAGNOSTICS', 'THERAPEU', 'VACCINES', 'HOSPITAL',
            'CLINIC', 'SURGICAL', 'DIAGNOSTIC', 'LIFE', 'SCIENCES', 'BIOTECH',
            'WELLNESS', 'MEDICARE', 'MEDIC', 'APOLLO', 'CARE', 'CURE'
        ]

        # Automobile & Transportation - More auto terms
        auto_keywords = [
            'AUTO', 'MOTOR', 'VEHICLE', 'TRANSPORT', 'LOGISTICS', 'CARGO', 'SHIPPING',
            'FREIGHT', 'DELIVERY', 'MOBILITY', 'TYRE', 'TYRES', 'WHEELS', 'PARTS',
            'COMPONENT', 'BEARINGS', 'FORGING', 'CASTINGS', 'ENGINES', 'GEARS',
            'AUTOMOBILES', 'CARS', 'BIKES', 'SCOOTER', 'BAJAJ', 'MARUTI'
        ]

        # FMCG & Consumer Goods - More consumer terms
        fmcg_keywords = [
            'FMCG', 'CONSUMER', 'FOOD', 'BEVERAGE', 'RETAIL', 'PRODUCTS', 'GOODS',
            'BRANDS', 'PERSONAL', 'CARE', 'COSMETIC', 'BEAUTY', 'HYGIENE', 'SOAP',
            'DETERGENT', 'TOBACCO', 'CIGARETTE', 'DAIRY', 'MILK', 'NUTRITION',
            'CONFECTION', 'BISCUIT', 'SNACKS', 'EDIBLE', 'OIL', 'SUGAR', 'TEA', 'COFFEE',
            'FOODS', 'DRINKS', 'FROZEN', 'SPICES', 'FLOUR', 'RICE', 'WHEAT'
        ]

        # Metals & Mining - More metal terms
        metals_keywords = [
            'METAL', 'STEEL', 'IRON', 'COAL', 'MINING', 'ALUMINIUM', 'ALUMINUM',
            'COPPER', 'ZINC', 'LEAD', 'NICKEL', 'TIN', 'ALLOY', 'FOUNDRY',
            'SMELTING', 'ORE', 'MINERAL', 'METALLURGY', 'ROLLING', 'WIRE', 'TUBES',
            'PIPE', 'PIPES', 'RODS', 'BARS', 'SHEETS', 'COILS', 'STRIPS'
        ]

        # Energy & Power - More energy terms
        energy_keywords = [
            'ENERGY', 'POWER', 'OIL', 'GAS', 'PETROLEUM', 'SOLAR', 'ELECTRIC', 'ELECTRICITY',
            'RENEWABLE', 'WIND', 'HYDRO', 'THERMAL', 'NUCLEAR', 'FUEL', 'COAL',
            'REFINERY', 'PIPELINE', 'DISTRIBUTION', 'TRANSMISSION', 'GRID', 'UTILITIES'
        ]

        # Infrastructure & Construction - More construction terms
        infra_keywords = [
            'INFRA', 'CEMENT', 'CONSTRUCTION', 'BUILDING', 'ENGINEERING', 'CONCRETE',
            'ROADS', 'BRIDGE', 'INFRASTRUCTURE', 'CONTRACTORS', 'PROJECTS', 'REAL',
            'ESTATE', 'PROPERTY', 'DEVELOPMENT', 'HOUSING', 'RESIDENTIAL', 'COMMERCIAL',
            'BUILDERS', 'CONSTRUCT', 'HOMES', 'FLATS', 'TOWERS', 'TOWNSHIP'
        ]

        # Telecommunications - More telecom terms
        telecom_keywords = [
            'TELECOM', 'COMMUNICATION', 'WIRELESS', 'NETWORK', 'TELEPHONE', 'MOBILE',
            'BROADBAND', 'INTERNET', 'SATELLITE', 'CABLE', 'FIBER', 'OPTIC', 'BHARTI'
        ]

        # Textiles & Apparel - More textile terms
        textile_keywords = [
            'TEXTILE', 'COTTON', 'FABRIC', 'GARMENT', 'APPAREL', 'CLOTH', 'YARN',
            'SPINNING', 'WEAVING', 'FASHION', 'CLOTHING', 'SYNTHETIC', 'FIBER',
            'DENIM', 'SILK', 'WOOL', 'POLYESTER', 'KNITTING', 'MILLS', 'FABRICS'
        ]

        # Chemical & Petrochemicals - More chemical terms
        chemical_keywords = [
            'CHEMICAL', 'CHEMICALS', 'PETRO', 'POLYMER', 'PLASTIC', 'RESIN', 'DYES',
            'PIGMENT', 'SPECIALTY', 'INDUSTRIAL', 'AGRO', 'FERTILIZER', 'PESTICIDE',
            'CROP', 'PROTECTION', 'PAINT', 'COATING', 'ADHESIVE', 'RUBBER'
        ]

        # Media & Entertainment - More media terms
        media_keywords = [
            'MEDIA', 'ENTERTAINMENT', 'TELEVISION', 'TV', 'BROADCASTING', 'NEWS',
            'FILM', 'MOVIE', 'PRODUCTION', 'ADVERTISING', 'MARKETING', 'PUBLISHING',
            'PRINT', 'CONTENT', 'STUDIOS', 'RADIO', 'CHANNEL'
        ]

        # Agriculture & Allied - More agri terms
        agri_keywords = [
            'AGRI', 'AGRICULTURE', 'FARMING', 'CROPS', 'SEEDS', 'IRRIGATION',
            'TRACTORS', 'EQUIPMENT', 'MACHINERY', 'FERTILIZERS', 'ORGANIC',
            'PLANTATION', 'HORTICULTURE', 'AQUACULTURE', 'FISHERIES', 'SUGAR'
        ]

        # Manufacturing & Industrial - New category for better classification
        manufacturing_keywords = [
            'MANUFACTUR', 'INDUSTRIAL', 'INDUSTRIES', 'IND', 'MACHINERY', 'EQUIPMENT',
            'TOOLS', 'INSTRUMENTS', 'PRECISION', 'ENGINEERING', 'GEAR', 'VALVES',
            'PUMPS', 'COMPRESSOR', 'TURBINE', 'BOILER', 'CRANE', 'MACHINE'
        ]

        # Trading & Commerce - New category
        trading_keywords = [
            'TRADING', 'TRADERS', 'COMMERCE', 'EXPORTS', 'IMPORTS', 'GLOBAL',
            'INTERNATIONAL', 'OVERSEAS', 'FOREIGN', 'WORLDWIDE', 'UNIVERSAL'
        ]

        # Check each sector in priority order (most specific first)
        if any(term in name_upper for term in banking_keywords):
            return 'Banking'
        elif any(term in name_upper for term in technology_keywords):
            return 'Technology'
        elif any(term in name_upper for term in pharma_keywords):
            return 'Pharmaceutical'
        elif any(term in name_upper for term in auto_keywords):
            return 'Automobile'
        elif any(term in name_upper for term in fmcg_keywords):
            return 'FMCG'
        elif any(term in name_upper for term in metals_keywords):
            return 'Metals'
        elif any(term in name_upper for term in energy_keywords):
            return 'Energy'
        elif any(term in name_upper for term in infra_keywords):
            return 'Infrastructure'
        elif any(term in name_upper for term in telecom_keywords):
            return 'Telecommunications'
        elif any(term in name_upper for term in textile_keywords):
            return 'Textiles'
        elif any(term in name_upper for term in chemical_keywords):
            return 'Chemicals'
        elif any(term in name_upper for term in media_keywords):
            return 'Media'
        elif any(term in name_upper for term in agri_keywords):
            return 'Agriculture'
        elif any(term in name_upper for term in manufacturing_keywords):
            return 'Manufacturing'
        elif any(term in name_upper for term in trading_keywords):
            return 'Trading'
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