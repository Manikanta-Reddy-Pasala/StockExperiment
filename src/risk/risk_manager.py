"""
Risk Manager for the Automated Trading System
"""
from typing import Dict, Any, List, Tuple
import pandas as pd


class RiskManager:
    """Manages risk controls including position sizing and exposure limits."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the risk manager.
        
        Args:
            config (Dict[str, Any]): Configuration parameters for risk management
        """
        self.config = config
        self.positions = {}  # Track current positions
        self.daily_pnl = 0.0  # Track daily P&L
        self.total_equity = 1000000.0  # Default equity, should be updated from broker
        self.stop_loss_percent = config.get('stop_loss_percent', 0.05)  # Default 5% stop loss
    
    def set_total_equity(self, equity: float):
        """
        Set the total equity for position sizing calculations.
        
        Args:
            equity (float): Total equity in the account
        """
        self.total_equity = equity
    
    def calculate_position_size(self, atr: float, risk_per_trade: float = None) -> int:
        """
        Calculate position size based on ATR and risk per trade.
        
        Args:
            atr (float): Average True Range for the instrument
            risk_per_trade (float, optional): Risk amount per trade as fraction of equity
            
        Returns:
            int: Calculated position size
        """
        if risk_per_trade is None:
            risk_per_trade = self.config.get('max_capital_per_trade', 0.01)
        
        risk_amount = self.total_equity * risk_per_trade
        if atr <= 0:
            return 0
        
        position_size = int(risk_amount / atr)
        return max(0, position_size)
    
    def check_trade_limits(self, symbol: str, quantity: int) -> Tuple[bool, str]:
        """
        Check if a trade complies with risk limits.
        
        Args:
            symbol (str): Trading symbol
            quantity (int): Quantity to trade
            
        Returns:
            Tuple[bool, str]: (approval_status, message)
        """
        # Check max capital per trade
        max_capital_fraction = self.config.get('max_capital_per_trade', 0.01)
        max_capital = self.total_equity * max_capital_fraction
        # In a real implementation, we would check the actual capital requirement
        
        # Check max concurrent trades
        max_concurrent = self.config.get('max_concurrent_trades', 10)
        if len(self.positions) >= max_concurrent:
            return False, f"Maximum concurrent trades limit ({max_concurrent}) reached"
        
        # Check daily loss limit
        daily_loss_limit = self.config.get('daily_loss_limit', 0.02)
        max_daily_loss = self.total_equity * daily_loss_limit
        if self.daily_pnl <= -max_daily_loss:
            return False, f"Daily loss limit exceeded ({daily_loss_limit*100}% of equity)"
        
        # Check single name exposure limit
        single_name_limit = self.config.get('single_name_exposure_limit', 0.05)
        max_exposure = self.total_equity * single_name_limit
        current_exposure = self.positions.get(symbol, 0)
        # In a real implementation, we would check the actual exposure
        
        return True, "Trade approved"
    
    def update_position(self, symbol: str, quantity: int, price: float):
        """
        Update position tracking.
        
        Args:
            symbol (str): Trading symbol
            quantity (int): Quantity change (+ for buy, - for sell)
            price (float): Execution price
        """
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] += quantity
    
    def update_daily_pnl(self, pnl: float):
        """
        Update daily P&L tracking.
        
        Args:
            pnl (float): P&L to add to daily total
        """
        self.daily_pnl += pnl
    
    def reset_daily_tracking(self):
        """Reset daily P&L tracking (to be called at the start of each trading day)."""
        self.daily_pnl = 0.0
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            Dict[str, Any]: Current risk metrics
        """
        return {
            'total_equity': self.total_equity,
            'daily_pnl': self.daily_pnl,
            'positions_count': len(self.positions),
            'positions': self.positions.copy()
        }
    
    def check_kill_switch(self) -> bool:
        """
        Check if kill switch should be activated.
        
        Returns:
            bool: True if kill switch should be activated
        """
        # Check daily loss limit
        daily_loss_limit = self.config.get('daily_loss_limit', 0.02)
        max_daily_loss = self.total_equity * daily_loss_limit
        if self.daily_pnl <= -max_daily_loss:
            return True
        
        return False
    
    def set_stop_loss_percent(self, percent: float):
        """
        Set the stop loss percentage.
        
        Args:
            percent (float): Stop loss percentage (e.g., 0.05 for 5%)
        """
        self.stop_loss_percent = percent
    
    def calculate_stop_loss_price(self, entry_price: float, transaction_type: str) -> float:
        """
        Calculate stop loss price based on entry price and transaction type.
        
        Args:
            entry_price (float): Entry price
            transaction_type (str): 'BUY' or 'SELL'
            
        Returns:
            float: Stop loss price
        """
        if transaction_type == 'BUY':
            # For long positions, stop loss is below entry price
            return entry_price * (1 - self.stop_loss_percent)
        elif transaction_type == 'SELL':
            # For short positions, stop loss is above entry price
            return entry_price * (1 + self.stop_loss_percent)
        else:
            return 0.0
    
    def should_exit_position(self, symbol: str, current_price: float, entry_price: float, 
                            transaction_type: str) -> bool:
        """
        Check if a position should be exited based on stop loss.
        
        Args:
            symbol (str): Trading symbol
            current_price (float): Current market price
            entry_price (float): Entry price
            transaction_type (str): 'BUY' or 'SELL'
            
        Returns:
            bool: True if position should be exited
        """
        stop_loss_price = self.calculate_stop_loss_price(entry_price, transaction_type)
        
        if transaction_type == 'BUY' and current_price <= stop_loss_price:
            # Long position - exit if price falls below stop loss
            return True
        elif transaction_type == 'SELL' and current_price >= stop_loss_price:
            # Short position - exit if price rises above stop loss
            return True
        
        return False