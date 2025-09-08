"""
Momentum Stock Selector Engine
"""
import pandas as pd
from typing import List, Dict, Any, Optional
from .base_strategy import BaseStrategy
from .momentum_strategy import MomentumStrategy, BreakoutStrategy


class SelectorEngine:
    """Engine for selecting momentum stocks using pluggable strategies."""
    
    # Indian mid-cap and small-cap stock lists (simplified for demonstration)
    # In a real implementation, this would come from a database or API
    MID_CAP_STOCKS = {
        "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS",
        "BAJAJFINSV.NS", "BHARTIARTL.NS", "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS",
        "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
        "HCLTECH.NS", "HDFC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
        "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
        "IOC.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "LTIM.NS",
        "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
        "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS",
        "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
        "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS"
    }
    
    SMALL_CAP_STOCKS = {
        "ABCAPITAL.NS", "ABFRL.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGAS.NS",
        "ADANIGREEN.NS", "ADANIPOWER.NS", "ADANITRANS.NS", "ALKEM.NS", "AMBUJACEM.NS",
        "APOLLOHOSP.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASTRAL.NS", "ATGL.NS",
        "AUBANK.NS", "AUROPHARMA.NS", "BAJAJHLDNG.NS", "BALKRISIND.NS", "BANDHANBNK.NS",
        "BANKBARODA.NS", "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS", "BIOCON.NS",
        "BOSCHLTD.NS", "BPCL.NS", "CANBK.NS", "CANFINHOME.NS", "CASTROLIND.NS",
        "CHAMBLFERT.NS", "CHOLAFIN.NS", "CIPLA.NS", "COFORGE.NS", "CONCOR.NS",
        "CUMMINSIND.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DELTACORP.NS", "DLF.NS",
        "DABUR.NS", "DCBBANK.NS", "DIVISLAB.NS", "DIXON.NS", "DLINKINDIA.NS"
    }
    
    def __init__(self):
        """Initialize the selector engine."""
        self.strategies = {}
        self.active_strategy = None
        self.data_manager = None
        self.chatgpt_validator = None
        self._register_default_strategies()
    
    def set_data_manager(self, data_manager):
        """
        Set the data manager for the selector engine.
        
        Args:
            data_manager: Data provider manager instance
        """
        self.data_manager = data_manager
    
    def set_chatgpt_validator(self, validator):
        """
        Set the ChatGPT validator for the selector engine.
        
        Args:
            validator: ChatGPT validator instance
        """
        self.chatgpt_validator = validator
    
    def _register_default_strategies(self):
        """Register default strategies."""
        self.register_strategy(MomentumStrategy())
        self.register_strategy(BreakoutStrategy())
    
    def register_strategy(self, strategy: BaseStrategy):
        """
        Register a strategy with the engine.
        
        Args:
            strategy (BaseStrategy): Strategy to register
        """
        self.strategies[strategy.name] = strategy
    
    def unregister_strategy(self, strategy_name: str):
        """
        Unregister a strategy from the engine.
        
        Args:
            strategy_name (str): Name of the strategy to unregister
        """
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            
            # If the active strategy is being unregistered, clear it
            if self.active_strategy == strategy_name:
                self.active_strategy = None
    
    def set_active_strategy(self, strategy_name: str) -> bool:
        """
        Set the active strategy.
        
        Args:
            strategy_name (str): Name of the strategy to set as active
            
        Returns:
            bool: True if strategy was set successfully, False otherwise
        """
        if strategy_name in self.strategies:
            self.active_strategy = strategy_name
            return True
        return False
    
    def get_active_strategy(self) -> Optional[BaseStrategy]:
        """
        Get the active strategy.
        
        Returns:
            Optional[BaseStrategy]: Active strategy or None if none is active
        """
        if self.active_strategy and self.active_strategy in self.strategies:
            return self.strategies[self.active_strategy]
        return None
    
    def get_available_strategies(self) -> List[str]:
        """
        Get list of available strategies.
        
        Returns:
            List[str]: List of available strategy names
        """
        return list(self.strategies.keys())
    
    def is_mid_small_cap(self, symbol: str) -> bool:
        """
        Check if a stock is a mid-cap or small-cap stock.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            bool: True if the stock is mid-cap or small-cap, False otherwise
        """
        # Normalize symbol format
        normalized_symbol = symbol.upper()
        if not normalized_symbol.endswith(".NS") and not normalized_symbol.endswith(".BO"):
            normalized_symbol = f"{normalized_symbol}.NS"
            
        return normalized_symbol in self.MID_CAP_STOCKS or normalized_symbol in self.SMALL_CAP_STOCKS
    
    def filter_mid_small_cap_stocks(self, stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter stocks to include only mid-cap and small-cap stocks.
        
        Args:
            stocks (List[Dict[str, Any]]): List of stock dictionaries with symbol information
            
        Returns:
            List[Dict[str, Any]]: Filtered list of mid-cap and small-cap stocks
        """
        return [stock for stock in stocks if self.is_mid_small_cap(stock.get('symbol', ''))]
    
    def select_stocks(self, market_data: pd.DataFrame = None, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """
        Select stocks using the active strategy.
        
        Args:
            market_data (pd.DataFrame): Market data for stock selection (optional)
            symbols (List[str]): List of symbols to analyze (optional)
            
        Returns:
            List[Dict[str, Any]]: List of selected stocks with details
            
        Raises:
            ValueError: If no active strategy is set
        """
        if not self.active_strategy:
            raise ValueError("No active strategy set. Please set an active strategy first.")
        
        # If no market data provided, fetch using data manager
        if market_data is None and self.data_manager is not None and symbols is not None:
            market_data = self._fetch_market_data(symbols)
        elif market_data is None:
            # Return empty list if no data available
            return []
        
        strategy = self.strategies[self.active_strategy]
        selected_stocks = strategy.select_stocks(market_data)
        
        # Filter for mid-cap and small-cap stocks
        selected_stocks = self.filter_mid_small_cap_stocks(selected_stocks)
        
        # Validate stocks with ChatGPT if validator is available
        if self.chatgpt_validator is not None and selected_stocks:
            selected_stocks = self.chatgpt_validator.validate_stocks(selected_stocks)
        
        return selected_stocks
    
    def _fetch_market_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch market data for symbols using the data manager.
        
        Args:
            symbols (List[str]): List of symbols to fetch data for
            
        Returns:
            pd.DataFrame: Market data for the symbols
        """
        if self.data_manager is None:
            return pd.DataFrame()
        
        # Fetch historical data for each symbol
        all_data = []
        for symbol in symbols:
            try:
                # Get 1 month of daily data
                data = self.data_manager.get_historical_data(symbol, period="1mo", interval="1d")
                if not data.empty:
                    # Add symbol column
                    data['symbol'] = symbol
                    all_data.append(data)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Rename columns to match expected format
        if 'Date' in combined_data.columns:
            combined_data = combined_data.rename(columns={
                'Date': 'timestamp',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume'
            })
        
        # Ensure required columns exist
        required_columns = ['symbol', 'close_price', 'volume', 'timestamp']
        for col in required_columns:
            if col not in combined_data.columns:
                combined_data[col] = None
        
        return combined_data
    
    def select_stocks_with_strategy(self, strategy_name: str, market_data: pd.DataFrame = None, 
                                  symbols: List[str] = None) -> List[Dict[str, Any]]:
        """
        Select stocks using a specific strategy.
        
        Args:
            strategy_name (str): Name of the strategy to use
            market_data (pd.DataFrame): Market data for stock selection (optional)
            symbols (List[str]): List of symbols to analyze (optional)
            
        Returns:
            List[Dict[str, Any]]: List of selected stocks with details
            
        Raises:
            ValueError: If strategy is not found
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found.")
        
        # If no market data provided, fetch using data manager
        if market_data is None and self.data_manager is not None and symbols is not None:
            market_data = self._fetch_market_data(symbols)
        elif market_data is None:
            # Return empty list if no data available
            return []
        
        strategy = self.strategies[strategy_name]
        selected_stocks = strategy.select_stocks(market_data)
        
        # Filter for mid-cap and small-cap stocks
        selected_stocks = self.filter_mid_small_cap_stocks(selected_stocks)
        
        # Validate stocks with ChatGPT if validator is available
        if self.chatgpt_validator is not None and selected_stocks:
            selected_stocks = self.chatgpt_validator.validate_stocks(selected_stocks)
        
        return selected_stocks