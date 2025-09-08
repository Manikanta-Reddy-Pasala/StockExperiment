"""
Backtesting Strategies
"""
import pandas as pd
from typing import Dict, Any, List


def simple_momentum_strategy(current_prices: Dict[str, float], 
                           positions: Dict[str, int], 
                           capital: float, 
                           lookback_period: int = 20,
                           max_positions: int = 5,
                           **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Simple momentum strategy for backtesting.
    
    Args:
        current_prices (Dict[str, float]): Current prices for symbols
        positions (Dict[str, int]): Current positions
        capital (float): Available capital
        lookback_period (int): Lookback period for momentum calculation
        max_positions (int): Maximum number of positions
        **kwargs: Additional parameters
        
    Returns:
        Dict[str, Dict[str, Any]]: Trading signals for each symbol
    """
    # For simplicity in backtesting, we'll generate signals based on price changes
    # In a real implementation, you would have historical data to calculate momentum
    
    signals = {}
    
    # Calculate simple momentum (price change)
    # This is a simplified version - in practice, you'd use actual historical data
    momentum_scores = {}
    for symbol, price in current_prices.items():
        # Simulate momentum as random for demonstration
        # In real implementation, this would be calculated from historical data
        momentum_scores[symbol] = price * 0.01  # Simple proxy
    
    # Rank symbols by momentum
    ranked_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select top symbols
    selected_symbols = [symbol for symbol, score in ranked_symbols[:max_positions]]
    
    # Generate signals
    for symbol in current_prices.keys():
        if symbol in selected_symbols and symbol not in positions:
            # Buy signal for selected symbols not currently held
            # Calculate position size (simplified)
            position_size = int((capital * 0.1) / current_prices[symbol])  # 10% of capital per position
            if position_size > 0:
                signals[symbol] = {'action': 'BUY', 'quantity': position_size}
        elif symbol in positions and symbol not in selected_symbols:
            # Sell signal for held symbols no longer selected
            signals[symbol] = {'action': 'SELL', 'quantity': positions[symbol]}
        else:
            # Hold signal
            signals[symbol] = {'action': 'HOLD', 'quantity': 0}
    
    return signals


def mean_reversion_strategy(current_prices: Dict[str, float], 
                          positions: Dict[str, int], 
                          capital: float, 
                          **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Simple mean reversion strategy for backtesting.
    
    Args:
        current_prices (Dict[str, float]): Current prices for symbols
        positions (Dict[str, int]): Current positions
        capital (float): Available capital
        **kwargs: Additional parameters
        
    Returns:
        Dict[str, Dict[str, Any]]: Trading signals for each symbol
    """
    signals = {}
    
    # Calculate average price (simplified)
    avg_price = sum(current_prices.values()) / len(current_prices) if current_prices else 0
    
    # Generate signals based on deviation from average
    for symbol, price in current_prices.items():
        deviation = (price - avg_price) / avg_price if avg_price > 0 else 0
        
        if symbol not in positions and deviation < -0.02:  # 2% below average
            # Buy undervalued symbols
            position_size = int((capital * 0.05) / price)  # 5% of capital per position
            if position_size > 0:
                signals[symbol] = {'action': 'BUY', 'quantity': position_size}
        elif symbol in positions and deviation > 0.02:  # 2% above average
            # Sell overvalued symbols
            signals[symbol] = {'action': 'SELL', 'quantity': positions[symbol]}
        else:
            # Hold
            signals[symbol] = {'action': 'HOLD', 'quantity': 0}
    
    return signals