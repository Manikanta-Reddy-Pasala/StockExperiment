"""
Dry Run Portfolio Management System
Manages fake portfolios for strategy testing and evaluation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from datastore.database import get_database_manager
from datastore.models import User, Order, Trade, Position
import logging

logger = logging.getLogger(__name__)


class DryRunPortfolio:
    """Represents a dry run portfolio for strategy testing."""
    
    def __init__(self, portfolio_id: str, initial_capital: float = 100000):
        """
        Initialize dry run portfolio.
        
        Args:
            portfolio_id (str): Unique portfolio identifier
            initial_capital (float): Initial capital in INR
        """
        self.portfolio_id = portfolio_id
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # {symbol: {'quantity': int, 'avg_price': float, 'current_price': float}}
        self.trades = []
        self.performance_history = []
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
    
    def add_position(self, symbol: str, quantity: int, price: float, current_price: float = None):
        """
        Add or update a position in the portfolio.
        
        Args:
            symbol (str): Stock symbol
            quantity (int): Number of shares
            price (float): Purchase price
            current_price (float): Current market price
        """
        if current_price is None:
            current_price = price
        
        if symbol in self.positions:
            # Update existing position
            existing = self.positions[symbol]
            total_quantity = existing['quantity'] + quantity
            total_cost = (existing['quantity'] * existing['avg_price']) + (quantity * price)
            avg_price = total_cost / total_quantity if total_quantity > 0 else 0
            
            self.positions[symbol] = {
                'quantity': total_quantity,
                'avg_price': avg_price,
                'current_price': current_price
            }
        else:
            # Add new position
            self.positions[symbol] = {
                'quantity': quantity,
                'current_price': current_price,
                'avg_price': price
            }
        
        # Record trade
        self.trades.append({
            'timestamp': datetime.utcnow(),
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': price,
            'current_price': current_price
        })
        
        # Update capital
        self.current_capital -= quantity * price
        self.last_updated = datetime.utcnow()
    
    def remove_position(self, symbol: str, quantity: int, price: float):
        """
        Remove or reduce a position in the portfolio.
        
        Args:
            symbol (str): Stock symbol
            quantity (int): Number of shares to sell
            price (float): Selling price
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        if quantity >= position['quantity']:
            # Sell entire position
            sell_quantity = position['quantity']
            del self.positions[symbol]
        else:
            # Partial sell
            sell_quantity = quantity
            self.positions[symbol]['quantity'] -= quantity
        
        # Record trade
        self.trades.append({
            'timestamp': datetime.utcnow(),
            'symbol': symbol,
            'action': 'SELL',
            'quantity': sell_quantity,
            'price': price,
            'current_price': price
        })
        
        # Update capital
        self.current_capital += sell_quantity * price
        self.last_updated = datetime.utcnow()
    
    def update_prices(self, price_data: Dict[str, float]):
        """
        Update current prices for all positions.
        
        Args:
            price_data (Dict[str, float]): {symbol: current_price}
        """
        for symbol, position in self.positions.items():
            if symbol in price_data:
                position['current_price'] = price_data[symbol]
        
        self.last_updated = datetime.utcnow()
    
    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value.
        
        Returns:
            float: Total portfolio value
        """
        total_value = self.current_capital
        
        for position in self.positions.values():
            total_value += position['quantity'] * position['current_price']
        
        return total_value
    
    def get_portfolio_performance(self) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            Dict: Performance metrics
        """
        current_value = self.get_portfolio_value()
        total_return = current_value - self.initial_capital
        return_percentage = (total_return / self.initial_capital) * 100
        
        # Calculate individual position performance
        position_performance = {}
        for symbol, position in self.positions.items():
            cost_basis = position['quantity'] * position['avg_price']
            current_value_pos = position['quantity'] * position['current_price']
            pnl = current_value_pos - cost_basis
            pnl_percentage = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
            
            position_performance[symbol] = {
                'quantity': position['quantity'],
                'avg_price': position['avg_price'],
                'current_price': position['current_price'],
                'cost_basis': cost_basis,
                'current_value': current_value_pos,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage
            }
        
        return {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'total_return': total_return,
            'return_percentage': return_percentage,
            'positions': position_performance,
            'num_positions': len(self.positions),
            'last_updated': self.last_updated
        }
    
    def record_performance_snapshot(self):
        """Record current performance for historical tracking."""
        performance = self.get_portfolio_performance()
        performance['timestamp'] = datetime.utcnow()
        self.performance_history.append(performance)


class DryRunManager:
    """Manages multiple dry run portfolios for strategy testing."""
    
    def __init__(self):
        """Initialize dry run manager."""
        self.db_manager = get_database_manager()
        self.portfolios = {}  # {portfolio_id: DryRunPortfolio}
        self.strategy_portfolios = {}  # {strategy_name: portfolio_id}
    
    def create_portfolio(self, portfolio_id: str, strategy_name: str, initial_capital: float = 100000) -> DryRunPortfolio:
        """
        Create a new dry run portfolio.
        
        Args:
            portfolio_id (str): Unique portfolio identifier
            strategy_name (str): Name of the strategy
            initial_capital (float): Initial capital in INR
            
        Returns:
            DryRunPortfolio: Created portfolio
        """
        portfolio = DryRunPortfolio(portfolio_id, initial_capital)
        self.portfolios[portfolio_id] = portfolio
        self.strategy_portfolios[strategy_name] = portfolio_id
        
        logger.info(f"Created dry run portfolio {portfolio_id} for strategy {strategy_name}")
        return portfolio
    
    def get_portfolio(self, portfolio_id: str) -> Optional[DryRunPortfolio]:
        """
        Get a portfolio by ID.
        
        Args:
            portfolio_id (str): Portfolio identifier
            
        Returns:
            Optional[DryRunPortfolio]: Portfolio or None if not found
        """
        return self.portfolios.get(portfolio_id)
    
    def get_strategy_portfolio(self, strategy_name: str) -> Optional[DryRunPortfolio]:
        """
        Get portfolio for a specific strategy.
        
        Args:
            strategy_name (str): Strategy name
            
        Returns:
            Optional[DryRunPortfolio]: Portfolio or None if not found
        """
        portfolio_id = self.strategy_portfolios.get(strategy_name)
        if portfolio_id:
            return self.get_portfolio(portfolio_id)
        return None
    
    def execute_strategy(self, strategy_name: str, selected_stocks: List[Dict], 
                        allocation_strategy: str = 'equal_weight') -> bool:
        """
        Execute a strategy on a dry run portfolio.
        
        Args:
            strategy_name (str): Name of the strategy
            selected_stocks (List[Dict]): List of selected stocks
            allocation_strategy (str): Allocation strategy ('equal_weight', 'market_cap_weight', 'custom')
            
        Returns:
            bool: True if successful
        """
        try:
            portfolio = self.get_strategy_portfolio(strategy_name)
            if not portfolio:
                # Create new portfolio for strategy
                portfolio_id = f"{strategy_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                portfolio = self.create_portfolio(portfolio_id, strategy_name)
            
            # Calculate allocations
            allocations = self._calculate_allocations(selected_stocks, allocation_strategy, portfolio.current_capital)
            
            # Execute trades
            for stock, allocation in allocations.items():
                if allocation['amount'] > 0:
                    quantity = int(allocation['amount'] / allocation['price'])
                    if quantity > 0:
                        portfolio.add_position(stock, quantity, allocation['price'])
            
            # Record performance snapshot
            portfolio.record_performance_snapshot()
            
            logger.info(f"Executed strategy {strategy_name} with {len(allocations)} positions")
            return True
            
        except Exception as e:
            logger.error(f"Error executing strategy {strategy_name}: {e}")
            return False
    
    def _calculate_allocations(self, stocks: List[Dict], strategy: str, available_capital: float) -> Dict:
        """
        Calculate position allocations based on strategy.
        
        Args:
            stocks (List[Dict]): List of selected stocks
            strategy (str): Allocation strategy
            available_capital (float): Available capital
            
        Returns:
            Dict: {symbol: {'amount': float, 'price': float}}
        """
        allocations = {}
        
        if strategy == 'equal_weight':
            # Equal weight allocation
            amount_per_stock = available_capital / len(stocks)
            for stock in stocks:
                allocations[stock['symbol']] = {
                    'amount': amount_per_stock,
                    'price': stock.get('current_price', stock.get('price', 100))
                }
        
        elif strategy == 'market_cap_weight':
            # Market cap weighted allocation
            total_market_cap = sum(stock.get('market_cap', 1000) for stock in stocks)
            for stock in stocks:
                weight = stock.get('market_cap', 1000) / total_market_cap
                amount = available_capital * weight
                allocations[stock['symbol']] = {
                    'amount': amount,
                    'price': stock.get('current_price', stock.get('price', 100))
                }
        
        elif strategy == 'custom':
            # Custom allocation based on strategy-specific logic
            # This would be implemented based on specific strategy requirements
            amount_per_stock = available_capital / len(stocks)
            for stock in stocks:
                allocations[stock['symbol']] = {
                    'amount': amount_per_stock,
                    'price': stock.get('current_price', stock.get('price', 100))
                }
        
        return allocations
    
    def update_all_portfolios(self, price_data: Dict[str, float]):
        """
        Update prices for all portfolios.
        
        Args:
            price_data (Dict[str, float]): {symbol: current_price}
        """
        for portfolio in self.portfolios.values():
            portfolio.update_prices(price_data)
            portfolio.record_performance_snapshot()
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[Dict]:
        """
        Get performance data for a strategy.
        
        Args:
            strategy_name (str): Strategy name
            
        Returns:
            Optional[Dict]: Performance data or None if not found
        """
        portfolio = self.get_strategy_portfolio(strategy_name)
        if portfolio:
            return portfolio.get_portfolio_performance()
        return None
    
    def get_all_strategy_performance(self) -> Dict[str, Dict]:
        """
        Get performance data for all strategies.
        
        Returns:
            Dict[str, Dict]: {strategy_name: performance_data}
        """
        performance_data = {}
        for strategy_name in self.strategy_portfolios.keys():
            performance = self.get_strategy_performance(strategy_name)
            if performance:
                performance_data[strategy_name] = performance
        
        return performance_data
    
    def cleanup_portfolio(self, portfolio_id: str):
        """
        Clean up a dry run portfolio (exit dry run mode).
        
        Args:
            portfolio_id (str): Portfolio identifier
        """
        if portfolio_id in self.portfolios:
            del self.portfolios[portfolio_id]
            
            # Remove from strategy mapping
            strategy_to_remove = None
            for strategy, pid in self.strategy_portfolios.items():
                if pid == portfolio_id:
                    strategy_to_remove = strategy
                    break
            
            if strategy_to_remove:
                del self.strategy_portfolios[strategy_to_remove]
            
            logger.info(f"Cleaned up dry run portfolio {portfolio_id}")
    
    def cleanup_all_portfolios(self):
        """Clean up all dry run portfolios."""
        self.portfolios.clear()
        self.strategy_portfolios.clear()
        logger.info("Cleaned up all dry run portfolios")


if __name__ == "__main__":
    # Test the dry run manager
    manager = DryRunManager()
    
    # Create test portfolios
    portfolio1 = manager.create_portfolio("test1", "momentum_strategy")
    portfolio2 = manager.create_portfolio("test2", "value_strategy")
    
    # Test performance
    performance = manager.get_all_strategy_performance()
    print(f"Strategy performance: {performance}")
