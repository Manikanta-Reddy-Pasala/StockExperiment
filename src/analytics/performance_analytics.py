"""
Performance Analytics Module
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class PerformanceAnalytics:
    """Calculate performance metrics including Sharpe/Sortino ratios and drawdown metrics."""
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize performance analytics.
        
        Args:
            risk_free_rate (float): Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, portfolio_values: List[Dict[str, Any]]) -> pd.Series:
        """
        Calculate returns from portfolio values.
        
        Args:
            portfolio_values (List[Dict[str, Any]]): List of portfolio values with dates
            
        Returns:
            pd.Series: Time series of returns
        """
        if not portfolio_values:
            return pd.Series(dtype=float)
        
        # Convert to DataFrame
        df = pd.DataFrame(portfolio_values)
        
        # Ensure date column exists
        if 'date' not in df.columns and 'timestamp' in df.columns:
            df['date'] = df['timestamp']
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate returns
        df['returns'] = df['value'].pct_change().fillna(0)
        
        # Set date as index
        df = df.set_index('date')
        
        return df['returns']
    
    def calculate_sharpe_ratio(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns (pd.Series): Time series of returns
            annualize (bool): Whether to annualize the ratio
            
        Returns:
            float: Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        # Calculate excess returns
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        
        # Calculate Sharpe ratio
        mean_excess_return = excess_returns.mean()
        std_dev = returns.std()
        
        if std_dev == 0:
            return 0.0
        
        sharpe_ratio = mean_excess_return / std_dev
        
        # Annualize if requested
        if annualize:
            sharpe_ratio *= np.sqrt(252)  # Annualize using trading days
        
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns (pd.Series): Time series of returns
            annualize (bool): Whether to annualize the ratio
            
        Returns:
            float: Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        # Calculate excess returns
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        
        # Calculate downside deviation (std of negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            downside_deviation = 0.0
        else:
            downside_deviation = negative_returns.std()
        
        # Calculate Sortino ratio
        mean_excess_return = excess_returns.mean()
        
        if downside_deviation == 0:
            return 0.0
        
        sortino_ratio = mean_excess_return / downside_deviation
        
        # Annualize if requested
        if annualize:
            sortino_ratio *= np.sqrt(252)  # Annualize using trading days
        
        return sortino_ratio
    
    def calculate_max_drawdown(self, portfolio_values: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate maximum drawdown metrics.
        
        Args:
            portfolio_values (List[Dict[str, Any]]): List of portfolio values with dates
            
        Returns:
            Dict[str, float]: Drawdown metrics
        """
        if not portfolio_values:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_percentage': 0.0,
                'drawdown_duration': 0
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(portfolio_values)
        
        # Ensure date column exists
        if 'date' not in df.columns and 'timestamp' in df.columns:
            df['date'] = df['timestamp']
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate cumulative max
        df['cumulative_max'] = df['value'].expanding().max()
        
        # Calculate drawdown
        df['drawdown'] = (df['value'] / df['cumulative_max']) - 1
        
        # Find maximum drawdown
        max_drawdown = df['drawdown'].min()
        max_drawdown_percentage = max_drawdown * 100
        
        # Calculate drawdown duration
        df['is_in_drawdown'] = df['drawdown'] < 0
        df['drawdown_group'] = (df['is_in_drawdown'] != df['is_in_drawdown'].shift()).cumsum()
        df['drawdown_duration'] = df.groupby('drawdown_group').cumcount() + 1
        max_drawdown_duration = df['drawdown_duration'].max()
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_percentage': max_drawdown_percentage,
            'drawdown_duration': max_drawdown_duration
        }
    
    def calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate win rate from trades.
        
        Args:
            trades (List[Dict[str, Any]]): List of trades
            
        Returns:
            float: Win rate (0-1)
        """
        if not trades:
            return 0.0
        
        # Count winning trades (positive P&L)
        winning_trades = 0
        total_trades = 0
        
        for trade in trades:
            # For sell trades, positive proceeds - cost is a win
            if trade.get('action') == 'SELL':
                proceeds = trade.get('proceeds', 0)
                cost = trade.get('cost', 0)
                if proceeds > cost:
                    winning_trades += 1
                total_trades += 1
            # For buy trades, we need to look at the corresponding sell
            elif trade.get('action') == 'BUY':
                # This is more complex as we need to match buy-sell pairs
                # For simplicity, we'll focus on sell trades for win rate
                pass
        
        # If no sell trades, look for P&L in trade data
        if total_trades == 0:
            for trade in trades:
                pnl = trade.get('pnl', 0)
                if pnl > 0:
                    winning_trades += 1
                if pnl != 0:  # Only count trades with P&L
                    total_trades += 1
        
        if total_trades == 0:
            return 0.0
        
        return winning_trades / total_trades
    
    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            returns (pd.Series): Time series of returns
            annualize (bool): Whether to annualize the volatility
            
        Returns:
            float: Volatility
        """
        if len(returns) < 2:
            return 0.0
        
        volatility = returns.std()
        
        # Annualize if requested
        if annualize:
            volatility *= np.sqrt(252)  # Annualize using trading days
        
        return volatility
    
    def generate_performance_report(self, portfolio_values: List[Dict[str, Any]], 
                                  trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            portfolio_values (List[Dict[str, Any]]): List of portfolio values with dates
            trades (List[Dict[str, Any]]): List of trades
            
        Returns:
            Dict[str, Any]: Performance report
        """
        if not portfolio_values:
            return {}
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(portfolio_values)
        
        # Ensure date column exists
        if 'date' not in df.columns and 'timestamp' in df.columns:
            df['date'] = df['timestamp']
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate returns
        returns = self.calculate_returns(portfolio_values)
        
        # Calculate metrics
        initial_value = df['value'].iloc[0] if len(df) > 0 else 0
        final_value = df['value'].iloc[-1] if len(df) > 0 else 0
        total_return = (final_value / initial_value - 1) if initial_value > 0 else 0
        total_return_percentage = total_return * 100
        
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        volatility = self.calculate_volatility(returns)
        
        drawdown_metrics = self.calculate_max_drawdown(portfolio_values)
        win_rate = self.calculate_win_rate(trades)
        
        # Calculate additional metrics
        total_trades = len([t for t in trades if t.get('action') in ['BUY', 'SELL']])
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_percentage': total_return_percentage,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'volatility': volatility,
            'max_drawdown': drawdown_metrics['max_drawdown'],
            'max_drawdown_percentage': drawdown_metrics['max_drawdown_percentage'],
            'drawdown_duration': drawdown_metrics['drawdown_duration'],
            'win_rate': win_rate,
            'win_rate_percentage': win_rate * 100,
            'total_trades': total_trades,
            'annualized_return': sharpe_ratio * volatility if volatility > 0 else 0,  # Approximation
        }