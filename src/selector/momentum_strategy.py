"""
Momentum-based Stock Selection Strategy
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Momentum-based stock selection strategy."""
    
    def __init__(self, name: str = "MomentumStrategy", description: str = ""):
        """
        Initialize the momentum strategy.
        
        Args:
            name (str): Name of the strategy
            description (str): Description of the strategy
        """
        super().__init__(name, description)
        self.set_parameters({
            'lookback_period': 20,  # days
            'min_volume_filter': 100000,  # minimum daily volume
            'top_n': 15  # number of stocks to select (increased for better selection)
        })
    
    def select_stocks(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Select stocks based on momentum strategy.
        
        Args:
            market_data (pd.DataFrame): Market data for stock selection
                Expected columns: symbol, close_price, volume, timestamp
            
        Returns:
            List[Dict[str, Any]]: List of selected stocks with details
        """
        if market_data.empty:
            return []
        
        # Apply volume filter
        filtered_data = market_data[
            market_data['volume'] >= self.parameters['min_volume_filter']
        ].copy()
        
        if filtered_data.empty:
            return []
        
        # Calculate momentum (rate of change)
        lookback = self.parameters['lookback_period']
        filtered_data['momentum'] = (
            filtered_data['close_price'] / filtered_data['close_price'].shift(lookback) - 1
        )
        
        # Remove NaN values
        filtered_data = filtered_data.dropna()
        
        # Sort by momentum and select top N
        selected_stocks = filtered_data.nlargest(
            self.parameters['top_n'], 
            'momentum'
        )
        
        # Prepare result
        result = []
        for _, row in selected_stocks.iterrows():
            result.append({
                'symbol': row['symbol'],
                'momentum': row['momentum'],
                'close_price': row['close_price'],
                'volume': row['volume'],
                'signal': 'BUY'  # Assuming long-only strategy
            })
        
        return result


class BreakoutStrategy(BaseStrategy):
    """Breakout-based stock selection strategy."""
    
    def __init__(self, name: str = "BreakoutStrategy", description: str = ""):
        """
        Initialize the breakout strategy.
        
        Args:
            name (str): Name of the strategy
            description (str): Description of the strategy
        """
        super().__init__(name, description)
        self.set_parameters({
            'lookback_period': 20,  # days
            'min_volume_filter': 100000,  # minimum daily volume
            'top_n': 15  # number of stocks to select (increased for better selection)
        })
    
    def select_stocks(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Select stocks based on breakout strategy.
        
        Args:
            market_data (pd.DataFrame): Market data for stock selection
                Expected columns: symbol, close_price, high_price, low_price, volume, timestamp
            
        Returns:
            List[Dict[str, Any]]: List of selected stocks with details
        """
        if market_data.empty:
            return []
        
        # Apply volume filter
        filtered_data = market_data[
            market_data['volume'] >= self.parameters['min_volume_filter']
        ].copy()
        
        if filtered_data.empty:
            return []
        
        # Calculate rolling high and low
        lookback = self.parameters['lookback_period']
        filtered_data['rolling_high'] = filtered_data['high_price'].rolling(window=lookback).max()
        filtered_data['rolling_low'] = filtered_data['low_price'].rolling(window=lookback).min()
        
        # Identify breakout signals
        filtered_data['breakout_signal'] = (
            (filtered_data['close_price'] > filtered_data['rolling_high'].shift(1)) |
            (filtered_data['close_price'] < filtered_data['rolling_low'].shift(1))
        )
        
        # Filter for breakout stocks
        breakout_stocks = filtered_data[filtered_data['breakout_signal']].copy()
        
        if breakout_stocks.empty:
            return []
        
        # Rank by volume surge (simplified)
        breakout_stocks['volume_rank'] = breakout_stocks['volume'].rank(ascending=False)
        
        # Select top N
        selected_stocks = breakout_stocks.nsmallest(
            self.parameters['top_n'], 
            'volume_rank'
        )
        
        # Prepare result
        result = []
        for _, row in selected_stocks.iterrows():
            signal = 'BUY' if row['close_price'] > row['rolling_high'].shift(1) else 'SELL'
            result.append({
                'symbol': row['symbol'],
                'signal': signal,
                'close_price': row['close_price'],
                'volume': row['volume'],
                'rolling_high': row['rolling_high'],
                'rolling_low': row['rolling_low']
            })
        
        return result