"""
Trading Strategy Engine
Implements various trading strategies for stock selection and portfolio management
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from datastore.database import get_database_manager
import logging

logger = logging.getLogger(__name__)


class BaseStrategy:
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, parameters: Dict = None):
        """
        Initialize strategy.
        
        Args:
            name (str): Strategy name
            parameters (Dict): Strategy parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.db_manager = get_database_manager()
    
    def select_stocks(self, screened_stocks: List[Dict]) -> List[Dict]:
        """
        Select stocks based on strategy criteria.
        
        Args:
            screened_stocks (List[Dict]): List of screened stocks
            
        Returns:
            List[Dict]: Selected stocks
        """
        raise NotImplementedError("Subclasses must implement select_stocks method")
    
    def calculate_position_size(self, stock: Dict, portfolio_value: float) -> Tuple[int, float]:
        """
        Calculate position size for a stock.
        
        Args:
            stock (Dict): Stock data
            portfolio_value (float): Total portfolio value
            
        Returns:
            Tuple[int, float]: (quantity, allocation_percentage)
        """
        raise NotImplementedError("Subclasses must implement calculate_position_size method")
    
    def should_exit_position(self, stock: Dict, entry_price: float, current_price: float) -> bool:
        """
        Determine if a position should be exited.
        
        Args:
            stock (Dict): Stock data
            entry_price (float): Entry price
            current_price (float): Current price
            
        Returns:
            bool: True if position should be exited
        """
        raise NotImplementedError("Subclasses must implement should_exit_position method")


class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy."""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'lookback_period': 20,
            'momentum_threshold': 0.05,  # 5% momentum
            'volume_threshold': 1.5,  # 1.5x average volume
            'max_positions': 10,
            'position_size_percent': 0.1  # 10% per position
        }
        params = {**default_params, **(parameters or {})}
        super().__init__("Momentum Strategy", params)
    
    def select_stocks(self, screened_stocks: List[Dict]) -> List[Dict]:
        """Select stocks based on momentum criteria."""
        selected_stocks = []
        
        for stock in screened_stocks:
            try:
                screening_data = stock.get('screening_data', {})
                
                # Calculate momentum
                current_price = screening_data.get('current_price', 0)
                dma_50 = screening_data.get('dma_50', 0)
                
                if dma_50 > 0:
                    momentum = (current_price - dma_50) / dma_50
                    
                    # Check volume criteria
                    volume = screening_data.get('volume', 0)
                    avg_volume = screening_data.get('avg_volume_1week', 0)
                    volume_ratio = volume / avg_volume if avg_volume > 0 else 0
                    
                    # Apply momentum criteria
                    if (momentum >= self.parameters['momentum_threshold'] and 
                        volume_ratio >= self.parameters['volume_threshold']):
                        
                        stock['momentum_score'] = momentum
                        stock['volume_ratio'] = volume_ratio
                        selected_stocks.append(stock)
                
            except Exception as e:
                logger.error(f"Error processing stock {stock.get('symbol', 'unknown')} in momentum strategy: {e}")
                continue
        
        # Sort by momentum score and limit positions
        selected_stocks.sort(key=lambda x: x.get('momentum_score', 0), reverse=True)
        return selected_stocks[:self.parameters['max_positions']]
    
    def calculate_position_size(self, stock: Dict, portfolio_value: float) -> Tuple[int, float]:
        """Calculate position size based on momentum strength."""
        momentum_score = stock.get('momentum_score', 0)
        current_price = stock.get('screening_data', {}).get('current_price', 100)
        
        # Adjust position size based on momentum strength
        base_allocation = self.parameters['position_size_percent']
        momentum_multiplier = min(2.0, 1.0 + momentum_score)  # Cap at 2x
        allocation_percent = base_allocation * momentum_multiplier
        
        allocation_amount = portfolio_value * allocation_percent
        quantity = int(allocation_amount / current_price)
        
        return quantity, allocation_percent
    
    def should_exit_position(self, stock: Dict, entry_price: float, current_price: float) -> bool:
        """Exit if momentum turns negative."""
        screening_data = stock.get('screening_data', {})
        dma_50 = screening_data.get('dma_50', 0)
        
        if dma_50 > 0:
            current_momentum = (current_price - dma_50) / dma_50
            return current_momentum < 0  # Exit if momentum turns negative
        
        return False


class ValueStrategy(BaseStrategy):
    """Value-based trading strategy."""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'pe_ratio_max': 15,
            'pb_ratio_max': 2,
            'debt_to_equity_max': 0.3,
            'roe_min': 0.15,  # 15% ROE
            'max_positions': 8,
            'position_size_percent': 0.125  # 12.5% per position
        }
        params = {**default_params, **(parameters or {})}
        super().__init__("Value Strategy", params)
    
    def select_stocks(self, screened_stocks: List[Dict]) -> List[Dict]:
        """Select stocks based on value criteria."""
        selected_stocks = []
        
        for stock in screened_stocks:
            try:
                screening_data = stock.get('screening_data', {})
                
                # Get financial ratios (mock data for now)
                pe_ratio = screening_data.get('pe_ratio', 20)
                pb_ratio = screening_data.get('pb_ratio', 3)
                debt_to_equity = screening_data.get('debt_to_equity', 0.2)
                roe = screening_data.get('roe', 0.1)
                
                # Apply value criteria
                if (pe_ratio <= self.parameters['pe_ratio_max'] and
                    pb_ratio <= self.parameters['pb_ratio_max'] and
                    debt_to_equity <= self.parameters['debt_to_equity_max'] and
                    roe >= self.parameters['roe_min']):
                    
                    # Calculate value score (lower is better for PE/PB)
                    value_score = (pe_ratio / self.parameters['pe_ratio_max']) + \
                                 (pb_ratio / self.parameters['pb_ratio_max']) - \
                                 (roe / self.parameters['roe_min'])
                    
                    stock['value_score'] = value_score
                    selected_stocks.append(stock)
                
            except Exception as e:
                logger.error(f"Error processing stock {stock.get('symbol', 'unknown')} in value strategy: {e}")
                continue
        
        # Sort by value score (ascending - lower is better)
        selected_stocks.sort(key=lambda x: x.get('value_score', 999))
        return selected_stocks[:self.parameters['max_positions']]
    
    def calculate_position_size(self, stock: Dict, portfolio_value: float) -> Tuple[int, float]:
        """Calculate position size based on value attractiveness."""
        value_score = stock.get('value_score', 999)
        current_price = stock.get('screening_data', {}).get('current_price', 100)
        
        # Adjust position size based on value score (lower score = larger position)
        base_allocation = self.parameters['position_size_percent']
        value_multiplier = max(0.5, 2.0 - value_score)  # Higher multiplier for better value
        allocation_percent = base_allocation * value_multiplier
        
        allocation_amount = portfolio_value * allocation_percent
        quantity = int(allocation_amount / current_price)
        
        return quantity, allocation_percent
    
    def should_exit_position(self, stock: Dict, entry_price: float, current_price: float) -> bool:
        """Exit if stock becomes overvalued."""
        screening_data = stock.get('screening_data', {})
        pe_ratio = screening_data.get('pe_ratio', 20)
        
        # Exit if PE ratio becomes too high
        return pe_ratio > self.parameters['pe_ratio_max'] * 1.5


class GrowthStrategy(BaseStrategy):
    """Growth-based trading strategy."""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'revenue_growth_min': 0.2,  # 20% revenue growth
            'profit_growth_min': 0.15,  # 15% profit growth
            'roe_min': 0.2,  # 20% ROE
            'max_positions': 6,
            'position_size_percent': 0.167  # 16.7% per position
        }
        params = {**default_params, **(parameters or {})}
        super().__init__("Growth Strategy", params)
    
    def select_stocks(self, screened_stocks: List[Dict]) -> List[Dict]:
        """Select stocks based on growth criteria."""
        selected_stocks = []
        
        for stock in screened_stocks:
            try:
                screening_data = stock.get('screening_data', {})
                
                # Calculate growth metrics
                sales_current = screening_data.get('sales_current_year', 0)
                sales_previous = screening_data.get('sales_preceding_year', 0)
                revenue_growth = (sales_current - sales_previous) / sales_previous if sales_previous > 0 else 0
                
                op_profit_current = screening_data.get('op_profit_latest_quarter', 0)
                op_profit_previous = screening_data.get('op_profit_preceding_quarter', 0)
                profit_growth = (op_profit_current - op_profit_previous) / op_profit_previous if op_profit_previous > 0 else 0
                
                roe = screening_data.get('roe', 0)
                
                # Apply growth criteria
                if (revenue_growth >= self.parameters['revenue_growth_min'] and
                    profit_growth >= self.parameters['profit_growth_min'] and
                    roe >= self.parameters['roe_min']):
                    
                    # Calculate growth score
                    growth_score = revenue_growth + profit_growth + roe
                    
                    stock['growth_score'] = growth_score
                    stock['revenue_growth'] = revenue_growth
                    stock['profit_growth'] = profit_growth
                    selected_stocks.append(stock)
                
            except Exception as e:
                logger.error(f"Error processing stock {stock.get('symbol', 'unknown')} in growth strategy: {e}")
                continue
        
        # Sort by growth score (descending - higher is better)
        selected_stocks.sort(key=lambda x: x.get('growth_score', 0), reverse=True)
        return selected_stocks[:self.parameters['max_positions']]
    
    def calculate_position_size(self, stock: Dict, portfolio_value: float) -> Tuple[int, float]:
        """Calculate position size based on growth potential."""
        growth_score = stock.get('growth_score', 0)
        current_price = stock.get('screening_data', {}).get('current_price', 100)
        
        # Adjust position size based on growth score
        base_allocation = self.parameters['position_size_percent']
        growth_multiplier = min(2.0, 1.0 + growth_score)  # Cap at 2x
        allocation_percent = base_allocation * growth_multiplier
        
        allocation_amount = portfolio_value * allocation_percent
        quantity = int(allocation_amount / current_price)
        
        return quantity, allocation_percent
    
    def should_exit_position(self, stock: Dict, entry_price: float, current_price: float) -> bool:
        """Exit if growth slows down significantly."""
        screening_data = stock.get('screening_data', {})
        revenue_growth = stock.get('revenue_growth', 0)
        
        # Exit if revenue growth drops below threshold
        return revenue_growth < self.parameters['revenue_growth_min'] * 0.5


class StrategyEngine:
    """Main strategy engine that manages multiple strategies."""
    
    def __init__(self):
        """Initialize strategy engine."""
        self.strategies = {
            'momentum': MomentumStrategy(),
            'value': ValueStrategy(),
            'growth': GrowthStrategy()
        }
        self.db_manager = get_database_manager()
    
    def run_strategies(self, screened_stocks: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Run all strategies on screened stocks.
        
        Args:
            screened_stocks (List[Dict]): List of screened stocks
            
        Returns:
            Dict[str, List[Dict]]: {strategy_name: selected_stocks}
        """
        results = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                selected_stocks = strategy.select_stocks(screened_stocks)
                results[strategy_name] = selected_stocks
                logger.info(f"Strategy {strategy_name} selected {len(selected_stocks)} stocks")
                
            except Exception as e:
                logger.error(f"Error running strategy {strategy_name}: {e}")
                results[strategy_name] = []
        
        return results
    
    def get_strategy_performance_metrics(self, strategy_name: str, selected_stocks: List[Dict]) -> Dict:
        """
        Calculate performance metrics for a strategy.
        
        Args:
            strategy_name (str): Strategy name
            selected_stocks (List[Dict]): Selected stocks
            
        Returns:
            Dict: Performance metrics
        """
        if not selected_stocks:
            return {
                'strategy_name': strategy_name,
                'num_stocks': 0,
                'avg_score': 0,
                'sector_distribution': {},
                'market_cap_distribution': {}
            }
        
        # Calculate average score
        scores = []
        for stock in selected_stocks:
            if strategy_name == 'momentum':
                scores.append(stock.get('momentum_score', 0))
            elif strategy_name == 'value':
                scores.append(stock.get('value_score', 0))
            elif strategy_name == 'growth':
                scores.append(stock.get('growth_score', 0))
        
        avg_score = np.mean(scores) if scores else 0
        
        # Calculate sector distribution
        sector_dist = {}
        market_cap_dist = {'small_cap': 0, 'mid_cap': 0, 'large_cap': 0}
        
        for stock in selected_stocks:
            sector = stock.get('sector', 'Unknown')
            sector_dist[sector] = sector_dist.get(sector, 0) + 1
            
            market_cap = stock.get('market_cap', 0)
            if market_cap < 10000:
                market_cap_dist['small_cap'] += 1
            elif market_cap < 20000:
                market_cap_dist['mid_cap'] += 1
            else:
                market_cap_dist['large_cap'] += 1
        
        return {
            'strategy_name': strategy_name,
            'num_stocks': len(selected_stocks),
            'avg_score': avg_score,
            'sector_distribution': sector_dist,
            'market_cap_distribution': market_cap_dist,
            'selected_stocks': selected_stocks
        }
    
    def compare_strategies(self, strategy_results: Dict[str, List[Dict]]) -> Dict:
        """
        Compare performance of different strategies.
        
        Args:
            strategy_results (Dict[str, List[Dict]]): Results from all strategies
            
        Returns:
            Dict: Comparison metrics
        """
        comparison = {
            'total_strategies': len(strategy_results),
            'strategy_metrics': {},
            'best_strategy': None,
            'diversification_score': 0
        }
        
        best_score = -1
        all_selected_stocks = set()
        
        for strategy_name, selected_stocks in strategy_results.items():
            metrics = self.get_strategy_performance_metrics(strategy_name, selected_stocks)
            comparison['strategy_metrics'][strategy_name] = metrics
            
            # Track best strategy
            if metrics['avg_score'] > best_score:
                best_score = metrics['avg_score']
                comparison['best_strategy'] = strategy_name
            
            # Track diversification
            for stock in selected_stocks:
                all_selected_stocks.add(stock['symbol'])
        
        # Calculate diversification score
        total_unique_stocks = len(all_selected_stocks)
        total_selections = sum(len(stocks) for stocks in strategy_results.values())
        comparison['diversification_score'] = total_unique_stocks / total_selections if total_selections > 0 else 0
        
        return comparison


if __name__ == "__main__":
    # Test the strategy engine
    engine = StrategyEngine()
    
    # Mock screened stocks
    mock_stocks = [
        {
            'symbol': 'STOCK1',
            'name': 'Test Stock 1',
            'market_cap': 8000,
            'sector': 'IT',
            'screening_data': {
                'current_price': 100,
                'dma_50': 95,
                'volume': 1000000,
                'avg_volume_1week': 500000,
                'pe_ratio': 12,
                'pb_ratio': 1.5,
                'debt_to_equity': 0.1,
                'roe': 0.2,
                'sales_current_year': 1000,
                'sales_preceding_year': 800,
                'op_profit_latest_quarter': 100,
                'op_profit_preceding_quarter': 80
            }
        }
    ]
    
    results = engine.run_strategies(mock_stocks)
    comparison = engine.compare_strategies(results)
    
    print(f"Strategy results: {results}")
    print(f"Strategy comparison: {comparison}")
