"""
Data Provider Manager
"""
import logging
from typing import Dict, Optional, List
from .data_provider import DataProvider, YahooFinanceProvider
from .fyers_provider import FyersDataProvider

logger = logging.getLogger(__name__)

class DataProviderManager:
    """Manages multiple data providers with fallback support."""

    def __init__(self, fyers_connector=None):
        """Initialize data provider manager."""
        self.providers: List[DataProvider] = []
        self.fyers_connector = fyers_connector
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available data providers."""
        # Add Yahoo Finance (always available)
        self.providers.append(YahooFinanceProvider())

        # Add FYERS API if connector is available
        if self.fyers_connector:
            self.providers.append(FyersDataProvider(self.fyers_connector))
            logger.info("FYERS API provider initialized for Indian stocks")
        else:
            logger.info("FYERS connector not available, using Yahoo Finance only")

    def get_stock_data(self, symbol: str, period: str = "1y") -> Optional[Dict]:
        """Get stock data from available providers with fallback."""
        for provider in self.providers:
            try:
                data = provider.get_stock_data(symbol, period)
                if data:
                    logger.info(f"Got data for {symbol} from {provider.name}")
                    return data
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed for {symbol}: {e}")
                continue

        logger.error(f"All providers failed for {symbol}")
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from available providers with fallback."""
        for provider in self.providers:
            try:
                price = provider.get_current_price(symbol)
                if price:
                    return price
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed for {symbol}: {e}")
                continue

        return None

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get historical data from available providers with fallback."""
        for provider in self.providers:
            try:
                data = provider.get_historical_data(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    return data
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed for {symbol}: {e}")
                continue

        return None

    def get_multiple_stocks_data(self, symbols: List[str], period: str = "1y") -> Dict[str, Dict]:
        """Get data for multiple stocks."""
        results = {}

        for symbol in symbols:
            data = self.get_stock_data(symbol, period)
            if data:
                results[symbol] = data

        logger.info(f"Retrieved data for {len(results)}/{len(symbols)} stocks")
        return results

    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return [provider.name for provider in self.providers]


_data_provider_manager = None

def get_data_provider_manager(fyers_connector=None) -> "DataProviderManager":
    """Get global data provider manager instance."""
    global _data_provider_manager
    if _data_provider_manager is None or fyers_connector:
        _data_provider_manager = DataProviderManager(fyers_connector)
    return _data_provider_manager
