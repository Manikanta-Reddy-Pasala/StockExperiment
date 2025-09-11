"""
Multiple Data Sources for Stock Data
Provides unified interface to multiple data providers
"""
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_stock_data(self, symbol: str, period: str = "1y") -> Optional[Dict]:
        """Get stock data for a symbol."""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        pass

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol."""
        pass


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider."""

    def __init__(self):
        """Initialize Yahoo Finance provider."""
        self.name = "Yahoo Finance"
        self.rate_limit = 2000  # requests per hour

    def get_stock_data(self, symbol: str, period: str = "1y") -> Optional[Dict]:
        """Get comprehensive stock data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get historical data
            hist = ticker.history(period=period)

            if hist.empty:
                return None

            # Calculate technical indicators
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            hist['Volume_MA'] = hist['Volume'].rolling(window=20).mean()

            current_price = hist['Close'].iloc[-1]
            sma_50 = hist['SMA_50'].iloc[-1]
            sma_200 = hist['SMA_200'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume_MA'].iloc[-1]

            # Calculate momentum
            momentum_20 = ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21]) * 100 if len(hist) > 20 else 0

            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'current_price': float(current_price),
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margins': info.get('profitMargins', 0),
                'sma_50': float(sma_50) if not pd.isna(sma_50) else 0,
                'sma_200': float(sma_200) if not pd.isna(sma_200) else 0,
                'volume': int(volume),
                'avg_volume': int(avg_volume) if not pd.isna(avg_volume) else 0,
                'momentum_20': float(momentum_20),
                'high_52w': info.get('fiftyTwoWeekHigh', 0),
                'low_52w': info.get('fiftyTwoWeekLow', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'data_source': self.name,
                'last_updated': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting data for {symbol} from Yahoo Finance: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get historical data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            return hist if not hist.empty else None
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None


# Global instance
_data_provider_manager = None

def get_data_provider_manager(fyers_connector=None) -> "DataProviderManager":
    """Get global data provider manager instance."""
    from .manager import DataProviderManager
    global _data_provider_manager
    if _data_provider_manager is None or fyers_connector:
        _data_provider_manager = DataProviderManager(fyers_connector)
    return _data_provider_manager
