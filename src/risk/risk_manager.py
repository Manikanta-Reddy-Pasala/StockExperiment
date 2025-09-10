"""
Risk Manager for the Automated Trading System
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class RiskManager:
    """Manages risk for individual users."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the risk manager.
        
        Args:
            config (Dict[str, Any]): Risk configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk limits
        self.max_capital_per_trade = config.get('max_capital_per_trade', 0.01)
        self.max_concurrent_trades = config.get('max_concurrent_trades', 10)
        self.daily_loss_limit = config.get('daily_loss_limit', 0.02)
        self.single_name_exposure_limit = config.get('single_name_exposure_limit', 0.05)
        
        # Track current positions and P&L
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.trade_count = 0
        
    def check_trade_risk(self, symbol: str, quantity: int, price: float, 
                        total_capital: float) -> Dict[str, Any]:
        """
        Check if a trade meets risk criteria.
        
        Args:
            symbol (str): Stock symbol
            quantity (int): Number of shares
            price (float): Price per share
            total_capital (float): Total available capital
            
        Returns:
            Dict[str, Any]: Risk assessment result
        """
        trade_value = quantity * price
        capital_ratio = trade_value / total_capital if total_capital > 0 else 0
        
        # Check capital per trade limit
        if capital_ratio > self.max_capital_per_trade:
            return {
                'approved': False,
                'reason': f'Trade value {capital_ratio:.2%} exceeds max per trade limit {self.max_capital_per_trade:.2%}'
            }
        
        # Check concurrent trades limit
        if len(self.current_positions) >= self.max_concurrent_trades:
            return {
                'approved': False,
                'reason': f'Maximum concurrent trades limit {self.max_concurrent_trades} reached'
            }
        
        # Check single name exposure
        current_exposure = self.current_positions.get(symbol, 0)
        new_exposure = current_exposure + trade_value
        exposure_ratio = new_exposure / total_capital if total_capital > 0 else 0
        
        if exposure_ratio > self.single_name_exposure_limit:
            return {
                'approved': False,
                'reason': f'Single name exposure {exposure_ratio:.2%} exceeds limit {self.single_name_exposure_limit:.2%}'
            }
        
        return {
            'approved': True,
            'trade_value': trade_value,
            'capital_ratio': capital_ratio,
            'exposure_ratio': exposure_ratio
        }
    
    def update_position(self, symbol: str, quantity: int, price: float):
        """
        Update position tracking.
        
        Args:
            symbol (str): Stock symbol
            quantity (int): Number of shares (positive for buy, negative for sell)
            price (float): Price per share
        """
        trade_value = quantity * price
        
        if symbol in self.current_positions:
            self.current_positions[symbol] += trade_value
        else:
            self.current_positions[symbol] = trade_value
        
        # Remove position if it's closed
        if abs(self.current_positions[symbol]) < 0.01:  # Small threshold for rounding
            del self.current_positions[symbol]
        
        self.trade_count += 1
    
    def update_daily_pnl(self, pnl: float):
        """
        Update daily P&L.
        
        Args:
            pnl (float): P&L amount
        """
        self.daily_pnl += pnl
    
    def check_daily_loss_limit(self, total_capital: float) -> bool:
        """
        Check if daily loss limit is exceeded.
        
        Args:
            total_capital (float): Total available capital
            
        Returns:
            bool: True if within limits, False if exceeded
        """
        if total_capital <= 0:
            return True
        
        loss_ratio = abs(self.daily_pnl) / total_capital if self.daily_pnl < 0 else 0
        return loss_ratio <= self.daily_loss_limit
    
    def get_risk_metrics(self, total_capital: float) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Args:
            total_capital (float): Total available capital
            
        Returns:
            Dict[str, Any]: Risk metrics
        """
        total_exposure = sum(abs(value) for value in self.current_positions.values())
        exposure_ratio = total_exposure / total_capital if total_capital > 0 else 0
        
        return {
            'daily_pnl': self.daily_pnl,
            'daily_pnl_ratio': self.daily_pnl / total_capital if total_capital > 0 else 0,
            'total_exposure': total_exposure,
            'exposure_ratio': exposure_ratio,
            'position_count': len(self.current_positions),
            'trade_count': self.trade_count,
            'positions': self.current_positions.copy()
        }
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of new trading day)."""
        self.daily_pnl = 0.0
        self.trade_count = 0
