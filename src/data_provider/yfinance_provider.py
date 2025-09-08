"""
yFinance Data Provider
"""
import yfinance as yf
import pandas as pd
from typing import List, Dict, Any, Optional
from .base_provider import BaseDataProvider


class YFinanceProvider(BaseDataProvider):
    """Data provider using yFinance library."""
    
    def __init__(self):
        """Initialize the yFinance provider."""
        self.cache = {}
        # Expanded list of Indian stocks including mid-cap and small-cap stocks
        self.indian_stocks = [
            # Large Cap Stocks
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", 
            "BHARTIARTL", "LT", "AXISBANK", "MARUTI", "SUNPHARMA", "TITAN", "WIPRO",
            "ULTRACEMCO", "M&M", "NESTLEIND", "TECHM", "BAJFINANCE", "ASIANPAINT",
            "KOTAKBANK", "ITC", "HCLTECH", "NTPC", "POWERGRID", "ONGC", "IOC", "JSWSTEEL",
            "GRASIM", "HEROMOTOCO", "HINDALCO", "DIVISLAB", "DRREDDY", "CIPLA", "SBILIFE",
            "BPCL", "BRITANNIA", "EICHERMOT", "TATACONSUM", "TATAMOTORS", "TATAPOWER",
            "TATASTEEL", "ADANIPORTS", "COALINDIA", "UPL", "HDFCLIFE", "LTIM",
            
            # Mid Cap Stocks
            "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPOWER", "ADANITRANS", "ATGL",
            "ABCAPITAL", "ABFRL", "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE",
            "ASHOKLEY", "ASTRAL", "AUBANK", "AUROPHARMA", "BAJAJHLDNG", "BALKRISIND",
            "BANDHANBNK", "BANKBARODA", "BEL", "BERGEPAINT", "BHARATFORG", "BIOCON",
            "BOSCHLTD", "CANBK", "CANFINHOME", "CASTROLIND", "CHAMBLFERT", "CHOLAFIN",
            "COFORGE", "CONCOR", "CUMMINSIND", "DALBHARAT", "DEEPAKNTR", "DELTACORP",
            "DLF", "DABUR", "DCBBANK", "DIXON", "DLINKINDIA", "ESCORTS", "EXIDEIND",
            "GAIL", "GLENMARK", "GLAND", "GMRINFRA", "GNFC", "GODREJCP", "GODREJPROP",
            "GRAPHITE", "GRINDWELL", "GUJGASLTD", "HAL", "HAVELLS", "HDFCAMC", "HEG",
            "HINDPETRO", "HUDCO", "IDBI", "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL",
            "INDIACEM", "INDIAMART", "INDIGO", "INDUSTOWER", "INFRATEL", "IPCALAB",
            "IRB", "IRCTC", "ISEC", "JINDALSTEL", "JKCEMENT", "JSWENERGY", "JUBLFOOD",
            "JUBLPHARMA", "KAJARIACER", "KEI", "KPRMILL", "KRBL", "L&TFH", "LALPATHLAB",
            "LAURUSLABS", "LICHSGFIN", "LTTS", "LUPIN", "M&MFIN", "MANAPPURAM",
            "MARICO", "MCDOWELL-N", "MCX", "METROPOLIS", "MFSL", "MGL", "MINDTREE",
            "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN", "NAM-INDIA", "NATIONALUM",
            "NAUKRI", "NAVINFLUOR", "NEOGEN", "NHPC", "NIITLTD", "NMDC", "NTPC",
            
            # Small Cap Stocks
            "AFFLE", "AARTIIND", "AAVAS", "ABCAPITAL", "ABFRL", "ACC", "ACI",
            "ADANIGAS", "ADANIHOME", "ADANIPORTS", "ADANITRANS", "ADVENZYMES",
            "AEGISCHEM", "AETHER", "AFFLE", "AHLUCONT", "AIAENG", "AJANTPHARM",
            "AKZOINDIA", "ALBK", "ALEMBICLTD", "ALICON", "ALKEM", "ALKYLAMINE",
            "ALLCARGO", "AMARAJABAT", "AMBER", "AMBUJACEM", "AMDIND", "AMIORG",
            "AMJLAND", "APARINDS", "APLAPOLLO", "APOLLOHOSP", "APOLLOTYRE",
            "ARVIND", "ARVINDFASN", "ASAHIINDIA", "ASHOKLEY", "ASHOKLEY", "ASIANPAINT",
            "ASTERDM", "ASTRAL", "ASTRAZEN", "ATFL", "ATUL", "AUBANK", "AURIONPRO",
            "AUROPHARMA", "AVANTIFEED", "AVTNPL", "AXISBANK", "BAJAJ-AUTO",
            "BAJAJCORP", "BAJAJELEC", "BAJAJFINSV", "BAJAJHLDNG", "BAJFINANCE"
        ]
    
    def get_historical_data(self, symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        """
        Get historical market data for a symbol using yFinance.
        
        Args:
            symbol (str): Trading symbol (e.g., "AAPL", "MSFT", "RELIANCE.NS")
            period (str): Time period (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (e.g., 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            pd.DataFrame: Historical market data with columns [Open, High, Low, Close, Volume]
        """
        try:
            # For Indian stocks, yFinance typically requires ".NS" suffix
            if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
                # Assume NSE as default for Indian stocks
                ticker_symbol = f"{symbol}.NS"
            else:
                ticker_symbol = symbol
            
            # Check cache first
            cache_key = f"{ticker_symbol}_{period}_{interval}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Fetch data from yFinance
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period=period, interval=interval)
            
            # Cache the data
            self.cache[cache_key] = data
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol using yFinance.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Current price
        """
        try:
            # For Indian stocks, yFinance typically requires ".NS" suffix
            if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
                # Assume NSE as default for Indian stocks
                ticker_symbol = f"{symbol}.NS"
            else:
                ticker_symbol = symbol
            
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period="1d")
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            else:
                return 0.0
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return 0.0
    
    def get_symbols(self) -> List[str]:
        """
        Get list of available symbols.
        For yFinance, we'll return a predefined list of common Indian stocks.
        
        Returns:
            List[str]: List of available symbols
        """
        # Add NSE suffix for yFinance
        return [f"{stock}.NS" for stock in self.indian_stocks]