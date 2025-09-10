"""
FYERS API Data Provider for Indian Stocks
Uses the existing FYERS connector for better integration
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
from data_sources.data_provider import DataProvider

logger = logging.getLogger(__name__)


class FyersDataProvider(DataProvider):
    """FYERS API data provider for Indian stocks using existing connector."""
    
    def __init__(self, fyers_connector=None):
        """Initialize FYERS provider."""
        self.name = "FYERS API"
        self.fyers_connector = fyers_connector
        self.rate_limit = 1000  # requests per hour
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> Optional[Dict]:
        """Get stock data from FYERS API."""
        try:
            if not self.fyers_connector:
                logger.warning("FYERS connector not available")
                return None
            
            # Convert symbol format (RELIANCE.NS -> NSE:RELIANCE-EQ)
            fyers_symbol = self._convert_symbol_format(symbol)
            
            # Get historical data for calculations
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            hist_data = self.get_historical_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
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
            
            # Get company profile (if available)
            company_info = self._get_company_info(fyers_symbol)
            
            return {
                'symbol': symbol,
                'name': company_info.get('name', self._get_company_name(symbol)),
                'current_price': float(current_price),
                'market_cap': company_info.get('market_cap', current_price * 1000000000),  # Rough estimate
                'sector': company_info.get('sector', 'Unknown'),
                'industry': company_info.get('industry', 'Unknown'),
                'pe_ratio': company_info.get('pe_ratio', 0),
                'pb_ratio': company_info.get('pb_ratio', 0),
                'debt_to_equity': company_info.get('debt_to_equity', 0),
                'roe': company_info.get('roe', 0),
                'revenue_growth': company_info.get('revenue_growth', 0),
                'profit_margins': company_info.get('profit_margins', 0),
                'sma_50': float(sma_50) if not pd.isna(sma_50) else 0,
                'sma_200': float(sma_200) if not pd.isna(sma_200) else 0,
                'volume': int(volume),
                'avg_volume': int(avg_volume) if not pd.isna(avg_volume) else 0,
                'momentum_20': float(momentum_20),
                'high_52w': float(high_52w),
                'low_52w': float(low_52w),
                'dividend_yield': company_info.get('dividend_yield', 0),
                'beta': company_info.get('beta', 1.0),
                'data_source': self.name,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol} from FYERS API: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from FYERS API."""
        try:
            if not self.fyers_connector:
                return None
            
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
            if not self.fyers_connector:
                return None
            
            # Convert symbol format
            fyers_symbol = self._convert_symbol_format(symbol)
            
            # Use FYERS connector to get historical data
            # This would use the existing FYERS connector methods
            # For now, we'll create a placeholder that would integrate with the actual FYERS API
            
            # In a real implementation, this would call:
            # hist_data = self.fyers_connector.get_historical_data(fyers_symbol, start_date, end_date)
            
            # For now, return None to fall back to Yahoo Finance
            logger.info(f"FYERS historical data request for {symbol} - would integrate with FYERS connector")
            return None
            
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
    
    def _get_company_info(self, fyers_symbol: str) -> Dict:
        """Get company information from FYERS API."""
        try:
            if not self.fyers_connector:
                return {}
            
            # This would use the FYERS connector to get company profile
            # For now, return empty dict
            return {}
            
        except Exception as e:
            logger.error(f"Error getting company info for {fyers_symbol}: {e}")
            return {}


# Enhanced Data Provider Manager that can use FYERS connector
class EnhancedDataProviderManager:
    """Enhanced data provider manager that can integrate with existing FYERS connector."""
    
    def __init__(self, fyers_connector=None):
        """Initialize enhanced data provider manager."""
        self.providers = []
        self.fyers_connector = fyers_connector
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available data providers."""
        # Add Yahoo Finance (always available)
        from data_sources.data_provider import YahooFinanceProvider
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


# Global instance
_enhanced_data_provider_manager = None

def get_enhanced_data_provider_manager(fyers_connector=None) -> EnhancedDataProviderManager:
    """Get global enhanced data provider manager instance."""
    global _enhanced_data_provider_manager
    if _enhanced_data_provider_manager is None or fyers_connector:
        _enhanced_data_provider_manager = EnhancedDataProviderManager(fyers_connector)
    return _enhanced_data_provider_manager


if __name__ == "__main__":
    # Test the enhanced data provider manager
    manager = get_enhanced_data_provider_manager()
    
    # Test with a few Indian stocks
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

