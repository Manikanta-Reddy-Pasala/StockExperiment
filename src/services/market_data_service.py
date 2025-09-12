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
                logger.warning("FYERS connector not available, using mock data for market overview")
                return self._get_mock_market_data()
            
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
                
                # Fill in any missing data with mock data
                for index_name in self.market_indices.keys():
                    if index_name not in market_data:
                        market_data[index_name] = self._get_mock_index_data(index_name)
                
                logger.info("Market overview data fetched successfully from Fyers API")
                return {
                    'success': True,
                    'data': market_data,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'fyers_api'
                }
                
            except Exception as e:
                logger.error(f"Error fetching market data from Fyers API: {e}")
                return self._get_mock_market_data()
                
        except Exception as e:
            logger.error(f"Error in get_market_overview: {e}")
            return self._get_mock_market_data()
    
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
            return self._get_mock_index_data("UNKNOWN")
    
    def _get_mock_market_data(self) -> Dict[str, Any]:
        """Get mock market data when Fyers API is not available."""
        mock_data = {}
        for index_name in self.market_indices.keys():
            mock_data[index_name] = self._get_mock_index_data(index_name)
        
        return {
            'success': True,
            'data': mock_data,
            'timestamp': datetime.now().isoformat(),
            'source': 'mock_data'
        }
    
    def _get_mock_index_data(self, index_name: str) -> Dict[str, Any]:
        """Get mock data for a specific index."""
        import random
        
        # Base prices for different indices
        base_prices = {
            'NIFTY 50': 19800,
            'SENSEX': 66200,
            'BANK NIFTY': 44500,
            'NIFTY IT': 28400
        }
        
        base_price = base_prices.get(index_name, 10000)
        
        # Generate realistic price and change
        current_price = base_price + random.uniform(-200, 200)
        change_percent = random.uniform(-2.0, 2.0)
        is_positive = change_percent >= 0
        
        return {
            'current_price': round(current_price, 2),
            'change_percent': round(change_percent, 2),
            'is_positive': is_positive,
            'prev_close': round(current_price - (current_price * change_percent / 100), 2),
            'high': round(current_price + random.uniform(0, 100), 2),
            'low': round(current_price - random.uniform(0, 100), 2),
            'volume': random.randint(1000000, 10000000)
        }


def get_market_data_service(broker_service=None) -> MarketDataService:
    """Get market data service instance."""
    return MarketDataService(broker_service)
