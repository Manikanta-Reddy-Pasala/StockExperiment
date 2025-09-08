"""
Base Data Provider Interface
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd


class BaseDataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def get_historical_data(self, symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol (str): Trading symbol
            period (str): Time period (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (e.g., 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            pd.DataFrame: Historical market data
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Current price
        """
        pass
    
    @abstractmethod
    def get_symbols(self) -> List[str]:
        """
        Get list of available symbols.
        
        Returns:
            List[str]: List of available symbols
        """
        pass