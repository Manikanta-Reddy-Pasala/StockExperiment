"""
Market Data Service
Fetches real-time market data from Fyers API for dashboard display
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketDataService:
    """Service for fetching market data from Fyers API."""
    
    def __init__(self, broker_service=None):
        self.broker_service = broker_service
        self.fyers_connector = None
        
        # Initialize broker service if not provided
        if not self.broker_service:
            try:
                from .broker_service import BrokerService
                self.broker_service = BrokerService()
                logger.info("MarketDataService: Initialized broker service")
            except Exception as e:
                logger.error(f"MarketDataService: Failed to initialize broker service: {e}")
        
        # Market indices symbols for Fyers API
        self.market_indices = {
            'NIFTY 50': 'NSE:NIFTY50-INDEX',
            'BANK NIFTY': 'NSE:NIFTYBANK-INDEX',
            'NIFTY MIDCAP 150': 'NSE:NIFTYMIDCAP150-INDEX',
            'NIFTY SMALLCAP 250': 'NSE:NIFTYSMALLCAP250-INDEX'
        }
    
    def _initialize_fyers_connector(self, user_id: int = 1):
        """Initialize FYERS connector for API calls."""
        try:
            if self.broker_service:
                config = self.broker_service.get_broker_config('fyers', user_id)
                logger.info(f"FYERS config for user {user_id}: {config}")
                
                # Check if we have the required credentials (don't require is_connected to be True)
                client_id = config.get('client_id') if config else None
                access_token = config.get('access_token') if config else None
                
                logger.info(f"FYERS credential check: config={bool(config)}, client_id={bool(client_id)}, access_token={bool(access_token)}")
                logger.info(f"FYERS client_id value: {client_id}")
                logger.info(f"FYERS access_token length: {len(access_token) if access_token else 0}")
                
                if config and client_id and access_token:
                    logger.info("FYERS credentials found, initializing connector")
                    from ..broker_service import FyersAPIConnector
                    self.fyers_connector = FyersAPIConnector(
                        client_id=client_id,
                        access_token=access_token
                    )
                    logger.info("FYERS connector initialized successfully")
                    return True
                else:
                    logger.warning(f"FYERS credentials missing: config={bool(config)}, client_id={bool(client_id)}, access_token={bool(access_token)}")
        except Exception as e:
            logger.error(f"Error initializing FYERS connector: {e}")
        return False
    
    def get_market_overview(self, user_id: int = 1) -> Dict[str, Any]:
        """Get market overview data for all major indices."""
        try:
            # Initialize FYERS connector
            if not self._initialize_fyers_connector(user_id):
                logger.warning("FYERS connector not available for market overview")
                return {
                    'success': False,
                    'error': 'FYERS connection not available. Please configure and connect your broker.',
                    'data': {},
                    'timestamp': datetime.now().isoformat(),
                    'source': 'unavailable'
                }
            
            market_data = {}
            
            # Fetch data for all indices in a single batch call
            symbols = list(self.market_indices.values())
            symbols_str = ','.join(symbols)
            
            try:
                # Get batch quotes for all indices
                quotes_data = self.fyers_connector.quotes(symbols_str)

                # Support multiple possible payload shapes from FYERS
                payload = None
                if quotes_data:
                    if isinstance(quotes_data, dict) and quotes_data.get('d'):
                        payload = quotes_data['d']
                    elif isinstance(quotes_data, dict) and quotes_data.get('data'):
                        payload = quotes_data['data']

                if payload:
                    # Process list format
                    if isinstance(payload, list):
                        for item in payload:
                            if isinstance(item, dict):
                                symbol = item.get('symbol') or item.get('n')
                                index_name = self._get_index_name_from_symbol(symbol) if symbol else None
                                if index_name:
                                    extracted = self._extract_index_data(item)
                                    if extracted:
                                        market_data[index_name] = extracted
                    # Process dict format
                    elif isinstance(payload, dict):
                        for symbol, quote in payload.items():
                            index_name = self._get_index_name_from_symbol(symbol)
                            if index_name and index_name in self.market_indices:
                                extracted = self._extract_index_data(quote)
                                if extracted:
                                    market_data[index_name] = extracted
                
                # Do not fill missing data with mock values; return only what was fetched
                
                logger.info("Market overview data fetched successfully from Fyers API")
                return {
                    'success': True,
                    'data': market_data,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'fyers_api'
                }
                
            except Exception as e:
                logger.error(f"Error fetching market data from Fyers API: {e}")
                return {
                    'success': False,
                    'error': 'Failed to fetch market data from FYERS',
                    'data': {},
                    'timestamp': datetime.now().isoformat(),
                    'source': 'error'
                }
                
        except Exception as e:
            logger.error(f"Error in get_market_overview: {e}")
            return {
                'success': False,
                'error': 'Unexpected error while fetching market overview',
                'data': {},
                'timestamp': datetime.now().isoformat(),
                'source': 'error'
            }
    
    def _get_index_name_from_symbol(self, symbol: str) -> Optional[str]:
        """Get index name from Fyers symbol."""
        for name, fyers_symbol in self.market_indices.items():
            if fyers_symbol == symbol:
                return name
        return None
    
    def _extract_index_data(self, quote_data: Dict) -> Dict[str, Any]:
        """Extract index data from Fyers quote response."""
        try:
            # FYERS may return either the quote data directly or nested under 'v'
            v_data = quote_data.get('v', quote_data)
            
            current_price = float(v_data.get('lp', 0))  # Last price
            prev_close = float(v_data.get('prev_close_price', v_data.get('pc', current_price)))  # Previous close
            
            # Calculate percentage change
            if prev_close > 0:
                change_percent = ((current_price - prev_close) / prev_close) * 100
            else:
                change_percent = 0.0
            
            # Determine if positive or negative change
            is_positive = change_percent >= 0
            
            # Handle multiple possible key names from FYERS response
            high_value = v_data.get('high_price', v_data.get('h', current_price))
            low_value = v_data.get('low_price', v_data.get('l', current_price))
            volume_value = v_data.get('tt', v_data.get('volume', v_data.get('ttv', 0)))

            return {
                'current_price': round(current_price, 2),
                'change_percent': round(change_percent, 2),
                'is_positive': is_positive,
                'prev_close': round(prev_close, 2),
                'high': round(float(high_value), 2) if high_value is not None else None,
                'low': round(float(low_value), 2) if low_value is not None else None,
                'volume': int(volume_value) if volume_value is not None else 0
            }
            
        except Exception as e:
            logger.error(f"Error extracting index data: {e}")
            return {}


def get_market_data_service(broker_service=None) -> MarketDataService:
    """Get market data service instance."""
    return MarketDataService(broker_service)
