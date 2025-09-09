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


class FyersDataProvider(DataProvider):
    """FYERS API data provider for Indian stocks."""
    
    def __init__(self, client_id: str = None, access_token: str = None):
        """Initialize FYERS provider."""
        self.name = "FYERS API"
        self.client_id = client_id or "your_client_id"
        self.access_token = access_token or "your_access_token"
        self.base_url = "https://api-t1.fyers.in/api/v3"
        self.rate_limit = 1000  # requests per hour
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> Optional[Dict]:
        """Get stock data from FYERS API."""
        try:
            # Convert symbol format (RELIANCE.NS -> NSE:RELIANCE-EQ)
            fyers_symbol = self._convert_symbol_format(symbol)
            
            # Get market data
            headers = {
                'Authorization': f'{self.client_id}:{self.access_token}',
                'Content-Type': 'application/json'
            }
            
            # Get quote data
            quote_url = f"{self.base_url}/market_status"
            quote_response = requests.get(quote_url, headers=headers)
            
            if quote_response.status_code != 200:
                logger.error(f"FYERS API error: {quote_response.status_code}")
                return None
            
            # Get historical data for calculations
            hist_data = self.get_historical_data(symbol, 
                                               (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                                               datetime.now().strftime('%Y-%m-%d'))
            
            if hist_data is None or hist_data.empty:
                logger.warning(f"No historical data available for {symbol}")
                return None
            
            # Calculate metrics from historical data
            current_price = hist_data['Close'].iloc[-1]
            high_52w = hist_data['High'].max()
            low_52w = hist_data['Low'].min()
            volume = hist_data['Volume'].iloc[-1]
            avg_volume = hist_data['Volume'].rolling(window=20).mean().iloc[-1]
            
            # Calculate moving averages
            sma_50 = hist_data['Close'].rolling(window=50).mean().iloc[-1] if len(hist_data) >= 50 else current_price
            sma_200 = hist_data['Close'].rolling(window=200).mean().iloc[-1] if len(hist_data) >= 200 else current_price
            
            # Calculate momentum
            momentum_20 = ((current_price - hist_data['Close'].iloc[-21]) / hist_data['Close'].iloc[-21]) * 100 if len(hist_data) > 20 else 0
            
            # Estimate market cap (simplified - in production, you'd get this from company data)
            estimated_market_cap = current_price * 1000000000  # Rough estimate
            
            return {
                'symbol': symbol,
                'name': self._get_company_name(symbol),
                'current_price': float(current_price),
                'market_cap': estimated_market_cap,
                'sector': 'Unknown',  # Would need additional API call for sector info
                'industry': 'Unknown',
                'pe_ratio': 0,  # Would need additional API call for fundamental data
                'pb_ratio': 0,
                'debt_to_equity': 0,
                'roe': 0,
                'revenue_growth': 0,
                'profit_margins': 0,
                'sma_50': float(sma_50) if not pd.isna(sma_50) else 0,
                'sma_200': float(sma_200) if not pd.isna(sma_200) else 0,
                'volume': int(volume),
                'avg_volume': int(avg_volume) if not pd.isna(avg_volume) else 0,
                'momentum_20': float(momentum_20),
                'high_52w': float(high_52w),
                'low_52w': float(low_52w),
                'dividend_yield': 0,
                'beta': 1.0,
                'data_source': self.name,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol} from FYERS API: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from FYERS API."""
        try:
            # Get historical data and return latest close price
            hist_data = self.get_historical_data(symbol, 
                                               (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                                               datetime.now().strftime('%Y-%m-%d'))
            
            if hist_data is None or hist_data.empty:
                return None
            
            return float(hist_data['Close'].iloc[-1])
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get historical data from FYERS API."""
        try:
            # Convert symbol format
            fyers_symbol = self._convert_symbol_format(symbol)
            
            headers = {
                'Authorization': f'{self.client_id}:{self.access_token}',
                'Content-Type': 'application/json'
            }
            
            # Get historical data from FYERS
            hist_url = f"{self.base_url}/history"
            params = {
                'symbol': fyers_symbol,
                'resolution': 'D',  # Daily
                'date_format': '1',
                'range_from': start_date,
                'range_to': end_date
            }
            
            response = requests.get(hist_url, headers=headers, params=params)
            
            if response.status_code != 200:
                logger.error(f"FYERS historical data error: {response.status_code}")
                return None
            
            data = response.json()
            
            if data.get('s') != 'ok':
                logger.error(f"FYERS API error: {data.get('message', 'Unknown error')}")
                return None
            
            # Convert to DataFrame
            timestamps = data.get('t', [])
            opens = data.get('o', [])
            highs = data.get('h', [])
            lows = data.get('l', [])
            closes = data.get('c', [])
            volumes = data.get('v', [])
            
            if not timestamps:
                return None
            
            # Create DataFrame
            df_data = {
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': closes,
                'Volume': volumes
            }
            
            df = pd.DataFrame(df_data)
            df.index = pd.to_datetime(timestamps, unit='s')
            df = df.sort_index()
            
            return df if not df.empty else None
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol format from Yahoo Finance to FYERS format."""
        # Convert RELIANCE.NS to NSE:RELIANCE-EQ
        if symbol.endswith('.NS'):
            base_symbol = symbol[:-3]  # Remove .NS
            return f"NSE:{base_symbol}-EQ"
        elif symbol.endswith('.BO'):
            base_symbol = symbol[:-3]  # Remove .BO
            return f"BSE:{base_symbol}-EQ"
        else:
            # Default to NSE
            return f"NSE:{symbol}-EQ"
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name for symbol."""
        # This would typically come from a company database
        # For now, return a formatted version of the symbol
        base_symbol = symbol.replace('.NS', '').replace('.BO', '')
        return f"{base_symbol} Ltd"


class DataProviderManager:
    """Manages multiple data providers with fallback support."""
    
    def __init__(self):
        """Initialize data provider manager."""
        self.providers = []
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available data providers."""
        # Add Yahoo Finance (always available)
        self.providers.append(YahooFinanceProvider())
        
        # Add FYERS API if credentials are available
        fyers_credentials = self._get_fyers_credentials()
        if fyers_credentials['client_id'] and fyers_credentials['access_token']:
            self.providers.append(FyersDataProvider(
                fyers_credentials['client_id'], 
                fyers_credentials['access_token']
            ))
            logger.info("FYERS API provider initialized for Indian stocks")
        else:
            logger.info("FYERS API credentials not found, using Yahoo Finance only")
    
    def _get_fyers_credentials(self) -> Dict[str, Optional[str]]:
        """Get FYERS API credentials from environment."""
        import os
        return {
            'client_id': os.environ.get('FYERS_CLIENT_ID'),
            'access_token': os.environ.get('FYERS_ACCESS_TOKEN')
        }
    
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


# Global instance
_data_provider_manager = None

def get_data_provider_manager(fyers_connector=None) -> DataProviderManager:
    """Get global data provider manager instance."""
    global _data_provider_manager
    if _data_provider_manager is None or fyers_connector:
        _data_provider_manager = DataProviderManager()
    return _data_provider_manager


if __name__ == "__main__":
    # Test the data provider manager
    manager = get_data_provider_manager()
    
    # Test with a few stocks
    test_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}:")
        data = manager.get_stock_data(symbol)
        if data:
            print(f"  Name: {data['name']}")
            print(f"  Price: ₹{data['current_price']:.2f}")
            print(f"  Market Cap: ₹{data['market_cap']:,.0f}")
            print(f"  Sector: {data['sector']}")
            print(f"  Source: {data['data_source']}")
        else:
            print(f"  Failed to get data")
    
    print(f"\nAvailable providers: {manager.get_available_providers()}")
