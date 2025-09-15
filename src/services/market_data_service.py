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
        
        # Market indices symbols for Fyers API
        self.market_indices = {
            'NIFTY 50': 'NSE:NIFTY50-INDEX',
            'SENSEX': 'BSE:SENSEX-INDEX', 
            'BANK NIFTY': 'NSE:NIFTYBANK-INDEX',
            'NIFTY IT': 'NSE:NIFTYIT-INDEX'
        }
    
    def _initialize_fyers_connector(self, user_id: int = 1):
        """Initialize FYERS connector for API calls."""
        try:
            if self.broker_service:
                config = self.broker_service.get_broker_config('fyers', user_id)
                if config and config.get('is_connected') and config.get('access_token'):
                    from .broker_service import FyersAPIConnector
                    self.fyers_connector = FyersAPIConnector(
                        client_id=config.get('client_id'),
                        access_token=config.get('access_token')
                    )
                    return True
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
                quotes_data = self.fyers_connector.get_quotes(symbols_str)
                
                if quotes_data and 'd' in quotes_data and quotes_data['d']:
                    # Process the response
                    if isinstance(quotes_data['d'], list):
                        # Handle list response format
                        for item in quotes_data['d']:
                            if isinstance(item, dict):
                                symbol = item.get('symbol') or item.get('n')
                                if symbol in symbols:
                                    index_name = self._get_index_name_from_symbol(symbol)
                                    if index_name:
                                        market_data[index_name] = self._extract_index_data(item)
                    else:
                        # Handle dict response format
                        for symbol, quote in quotes_data['d'].items():
                            if symbol in symbols:
                                index_name = self._get_index_name_from_symbol(symbol)
                                if index_name:
                                    market_data[index_name] = self._extract_index_data(quote)
                
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
            v_data = quote_data.get('v', {})
            
            current_price = float(v_data.get('lp', 0))  # Last price
            prev_close = float(v_data.get('prev_close_price', current_price))  # Previous close
            
            # Calculate percentage change
            if prev_close > 0:
                change_percent = ((current_price - prev_close) / prev_close) * 100
            else:
                change_percent = 0.0
            
            # Determine if positive or negative change
            is_positive = change_percent >= 0
            
            return {
                'current_price': round(current_price, 2),
                'change_percent': round(change_percent, 2),
                'is_positive': is_positive,
                'prev_close': round(prev_close, 2),
                'high': round(float(v_data.get('high_price', current_price)), 2),
                'low': round(float(v_data.get('low_price', current_price)), 2),
                'volume': int(v_data.get('tt', 0))  # Total traded quantity
            }
            
        except Exception as e:
            logger.error(f"Error extracting index data: {e}")
            return {}


def get_market_data_service(broker_service=None) -> MarketDataService:
    """Get market data service instance."""
    return MarketDataService(broker_service)
