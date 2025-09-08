"""
Data Provider Manager
"""
import pandas as pd
from typing import List, Dict, Any, Optional
from .base_provider import BaseDataProvider
from .yfinance_provider import YFinanceProvider


class DataProviderManager:
    """Manages multiple data providers with fallback capabilities."""
    
    def __init__(self):
        """Initialize the data provider manager."""
        self.providers = {}
        self.primary_provider = None
        self.fallback_provider = None
        
        # Register default providers
        self.register_provider("yfinance", YFinanceProvider())
        
        # Set default providers
        self.set_primary_provider("yfinance")
    
    def register_provider(self, name: str, provider: BaseDataProvider):
        """
        Register a data provider.
        
        Args:
            name (str): Provider name
            provider (BaseDataProvider): Provider instance
        """
        self.providers[name] = provider
    
    def set_primary_provider(self, name: str) -> bool:
        """
        Set the primary data provider.
        
        Args:
            name (str): Provider name
            
        Returns:
            bool: True if successful, False if provider not found
        """
        if name in self.providers:
            self.primary_provider = name
            return True
        return False
    
    def set_fallback_provider(self, name: str) -> bool:
        """
        Set the fallback data provider.
        
        Args:
            name (str): Provider name
            
        Returns:
            bool: True if successful, False if provider not found
        """
        if name in self.providers:
            self.fallback_provider = name
            return True
        return False
    
    def get_historical_data(self, symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol (str): Trading symbol
            period (str): Time period
            interval (str): Data interval
            
        Returns:
            pd.DataFrame: Historical market data
        """
        # Try primary provider first
        if self.primary_provider and self.primary_provider in self.providers:
            try:
                data = self.providers[self.primary_provider].get_historical_data(symbol, period, interval)
                if not data.empty:
                    return data
            except Exception as e:
                print(f"Primary provider failed: {e}")
        
        # Fall back to fallback provider
        if self.fallback_provider and self.fallback_provider in self.providers:
            try:
                data = self.providers[self.fallback_provider].get_historical_data(symbol, period, interval)
                if not data.empty:
                    return data
            except Exception as e:
                print(f"Fallback provider failed: {e}")
        
        # Return empty DataFrame if all providers fail
        return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Current price
        """
        # Try primary provider first
        if self.primary_provider and self.primary_provider in self.providers:
            try:
                price = self.providers[self.primary_provider].get_current_price(symbol)
                if price > 0:
                    return price
            except Exception as e:
                print(f"Primary provider failed: {e}")
        
        # Fall back to fallback provider
        if self.fallback_provider and self.fallback_provider in self.providers:
            try:
                price = self.providers[self.fallback_provider].get_current_price(symbol)
                if price > 0:
                    return price
            except Exception as e:
                print(f"Fallback provider failed: {e}")
        
        # Return 0 if all providers fail
        return 0.0
    
    def get_symbols(self) -> List[str]:
        """
        Get list of available symbols.
        
        Returns:
            List[str]: List of available symbols
        """
        # Try primary provider first
        if self.primary_provider and self.primary_provider in self.providers:
            try:
                return self.providers[self.primary_provider].get_symbols()
            except Exception as e:
                print(f"Primary provider failed: {e}")
        
        # Fall back to fallback provider
        if self.fallback_provider and self.fallback_provider in self.providers:
            try:
                return self.providers[self.fallback_provider].get_symbols()
            except Exception as e:
                print(f"Fallback provider failed: {e}")
        
        # Return empty list if all providers fail
        return []


# Global data provider manager instance
data_manager = None


def get_data_manager() -> DataProviderManager:
    """
    Get the global data provider manager instance.
    
    Returns:
        DataProviderManager: Data provider manager instance
    """
    global data_manager
    if data_manager is None:
        data_manager = DataProviderManager()
    return data_manager