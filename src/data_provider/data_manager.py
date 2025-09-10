"""
Data Manager for the Trading System
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class DataManager:
    """Manages market data and provides data access methods."""
    
    def __init__(self):
        """Initialize the data manager."""
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.last_update = {}
    
    def get_stock_data(self, symbol: str, period: str = '1d') -> Optional[Dict[str, Any]]:
        """
        Get stock data for a symbol.
        
        Args:
            symbol (str): Stock symbol
            period (str): Data period
            
        Returns:
            Optional[Dict[str, Any]]: Stock data or None
        """
        try:
            # In a real implementation, this would fetch from data provider
            # For now, we'll return simulated data
            current_price = 100 + (hash(symbol) % 900)  # Simulate price between 100-1000
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'open': current_price * 0.99,
                'high': current_price * 1.02,
                'low': current_price * 0.98,
                'close': current_price,
                'volume': 1000000 + (hash(symbol) % 5000000),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting stock data for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days of historical data
            
        Returns:
            List[Dict[str, Any]]: Historical data
        """
        try:
            # In a real implementation, this would fetch from data provider
            # For now, we'll return simulated historical data
            historical_data = []
            base_price = 100 + (hash(symbol) % 900)
            
            for i in range(days):
                date = datetime.now() - timedelta(days=days-i)
                price = base_price * (1 + (i * 0.001))  # Simulate slight upward trend
                
                historical_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': price * 0.99,
                    'high': price * 1.02,
                    'low': price * 0.98,
                    'close': price,
                    'volume': 1000000 + (hash(symbol) % 5000000)
                })
            
            return historical_data
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
    
    def get_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get market data for multiple symbols.
        
        Args:
            symbols (List[str]): List of stock symbols
            
        Returns:
            Dict[str, Dict[str, Any]]: Market data by symbol
        """
        market_data = {}
        
        for symbol in symbols:
            data = self.get_stock_data(symbol)
            if data:
                market_data[symbol] = data
        
        return market_data
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            bool: True if market is open
        """
        now = datetime.now()
        # Simple market hours check (9:15 AM to 3:30 PM IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close and now.weekday() < 5  # Monday-Friday


# Global data manager instance
_data_manager = None


def get_data_manager() -> DataManager:
    """
    Get the global data manager instance.
    
    Returns:
        DataManager: Data manager instance
    """
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager
