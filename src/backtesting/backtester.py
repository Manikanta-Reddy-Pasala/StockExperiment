"""
Backtesting Engine
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np


class Backtester:
    """Backtesting engine for strategy validation."""
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize the backtester.
        
        Args:
            initial_capital (float): Initial capital for backtesting
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> quantity
        self.trades = []  # list of trades
        self.portfolio_values = []  # list of portfolio values over time
        self.asof_date = None  # For historical analysis override
    
    def set_asof_date(self, date: datetime):
        """
        Set ASOF date for historical analysis.
        
        Args:
            date (datetime): ASOF date for backtesting
        """
        self.asof_date = date
    
    def run_backtest(self, strategy_func, historical_data: pd.DataFrame, 
                     symbols: List[str], **strategy_params) -> Dict[str, Any]:
        """
        Run backtest for a strategy.
        
        Args:
            strategy_func: Function that implements the trading strategy
            historical_data (pd.DataFrame): Historical market data
            symbols (List[str]): List of symbols to trade
            **strategy_params: Additional parameters for the strategy
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        # Reset backtester state
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        
        # Sort data by date
        if 'Date' in historical_data.columns:
            historical_data = historical_data.sort_values('Date')
        elif 'Datetime' in historical_data.columns:
            historical_data = historical_data.sort_values('Datetime')
        
        # Group data by date
        if 'Date' in historical_data.columns:
            grouped_data = historical_data.groupby('Date')
        elif 'Datetime' in historical_data.columns:
            grouped_data = historical_data.groupby('Datetime')
        else:
            # Assume index is date
            grouped_data = historical_data.groupby(historical_data.index)
        
        # Run strategy for each date
        for date, group in grouped_data:
            # Apply ASOF date filter if set
            if self.asof_date and date > self.asof_date:
                break
            
            # Get current prices for all symbols
            current_prices = {}
            for symbol in symbols:
                symbol_data = group[group['Symbol'] == symbol] if 'Symbol' in group.columns else group
                if not symbol_data.empty:
                    current_prices[symbol] = symbol_data['Close'].iloc[-1] if 'Close' in symbol_data.columns else 0
            
            # Run strategy
            signals = strategy_func(current_prices, self.positions, self.current_capital, **strategy_params)
            
            # Execute trades based on signals
            for symbol, signal in signals.items():
                if symbol in current_prices and current_prices[symbol] > 0:
                    self._execute_trade(symbol, signal, current_prices[symbol], date)
            
            # Calculate and store portfolio value
            portfolio_value = self._calculate_portfolio_value(current_prices)
            self.portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'capital': self.current_capital,
                'positions_value': portfolio_value - self.current_capital
            })
        
        # Generate performance metrics
        results = self._generate_performance_metrics()
        return results
    
    def _execute_trade(self, symbol: str, signal: Dict[str, Any], price: float, date):
        """
        Execute a trade based on signal.
        
        Args:
            symbol (str): Trading symbol
            signal (Dict[str, Any]): Trade signal with keys 'action' and 'quantity'
            price (float): Execution price
            date: Trade date
        """
        action = signal.get('action', 'HOLD')
        quantity = signal.get('quantity', 0)
        
        if action == 'BUY' and quantity > 0:
            cost = price * quantity
            if self.current_capital >= cost:
                # Execute buy
                self.current_capital -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                
                # Record trade
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'cost': cost,
                    'type': 'MARKET'
                })
        
        elif action == 'SELL' and quantity > 0:
            available_quantity = self.positions.get(symbol, 0)
            if available_quantity >= quantity:
                # Execute sell
                proceeds = price * quantity
                self.current_capital += proceeds
                self.positions[symbol] = available_quantity - quantity
                
                # Record trade
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'proceeds': proceeds,
                    'type': 'MARKET'
                })
    
    def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            current_prices (Dict[str, float]): Current prices for symbols
            
        Returns:
            float: Portfolio value
        """
        positions_value = 0.0
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                positions_value += current_prices[symbol] * quantity
        
        return self.current_capital + positions_value
    
    def _generate_performance_metrics(self) -> Dict[str, Any]:
        """
        Generate performance metrics from backtest results.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        if not self.portfolio_values:
            return {}
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(self.portfolio_values)
        df['returns'] = df['value'].pct_change().fillna(0)
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        
        # Calculate metrics
        total_return = (df['value'].iloc[-1] / self.initial_capital) - 1
        volatility = df['returns'].std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (df['returns'].mean() / df['returns'].std()) * np.sqrt(252) if df['returns'].std() > 0 else 0
        
        # Calculate max drawdown
        rolling_max = df['value'].expanding().max()
        daily_drawdown = df['value'] / rolling_max - 1
        max_drawdown = daily_drawdown.min()
        
        # Calculate Sortino ratio (using downside deviation)
        negative_returns = df['returns'][df['returns'] < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (df['returns'].mean() / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Win rate
        winning_trades = len([t for t in self.trades if t.get('action') == 'SELL' and t.get('proceeds', 0) > t.get('cost', 0)])
        total_exits = len([t for t in self.trades if t.get('action') == 'SELL'])
        win_rate = winning_trades / total_exits if total_exits > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': df['value'].iloc[-1],
            'total_return': total_return,
            'total_return_percentage': total_return * 100,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_percentage': max_drawdown * 100,
            'win_rate': win_rate,
            'win_rate_percentage': win_rate * 100,
            'total_trades': len(self.trades),
            'portfolio_values': self.portfolio_values,
            'trades': self.trades
        }