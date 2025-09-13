"""
Suggested Stocks Interface Definition

Defines the contract for suggested stocks features across different brokers.
Each broker implementation must provide these methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class StrategyType(Enum):
    """Stock screening strategy types."""
    MOMENTUM = "momentum"
    VALUE = "value"
    GROWTH = "growth"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


class ISuggestedStocksProvider(ABC):
    """
    Interface for suggested stocks data providers.
    
    This interface defines all the methods that must be implemented
    by each broker to provide stock screening and suggestions functionality.
    """

    @abstractmethod
    def get_suggested_stocks(self, user_id: int, strategies: List[StrategyType] = None, 
                           limit: int = 50) -> Dict[str, Any]:
        """
        Get suggested stocks based on screening strategies.
        
        Args:
            user_id: The user ID for broker-specific authentication
            strategies: List of screening strategies to apply
            limit: Maximum number of stocks to return
            
        Returns:
            Dict containing:
            - success: bool
            - data: List of suggested stocks with details
            - strategies_applied: List of strategies used
            - last_updated: timestamp
        """
        pass

    @abstractmethod
    def get_stock_analysis(self, user_id: int, symbol: str) -> Dict[str, Any]:
        """
        Get detailed analysis for a specific stock.
        
        Args:
            user_id: The user ID for broker-specific authentication
            symbol: Stock symbol to analyze
            
        Returns:
            Dict containing:
            - success: bool
            - data: Detailed stock analysis including technicals and fundamentals
            - last_updated: timestamp
        """
        pass
    @abstractmethod
    def get_strategy_performance(self, user_id: int, strategy: StrategyType, 
                               period: str = '1M') -> Dict[str, Any]:
        """
        Get performance metrics for a specific strategy.
        
        Args:
            user_id: The user ID for broker-specific authentication
            strategy: The strategy to analyze
            period: Time period for analysis
            
        Returns:
            Dict containing:
            - success: bool
            - data: Strategy performance metrics
            - last_updated: timestamp
        """
        pass

    @abstractmethod
    def get_sector_analysis(self, user_id: int) -> Dict[str, Any]:
        """
        Get sector-wise analysis and recommendations.
        
        Args:
            user_id: The user ID for broker-specific authentication
            
        Returns:
            Dict containing:
            - success: bool
            - data: Sector analysis with top performers and recommendations
            - last_updated: timestamp
        """
        pass

    @abstractmethod
    def get_technical_screener(self, user_id: int, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Screen stocks based on technical criteria.
        
        Args:
            user_id: The user ID for broker-specific authentication
            criteria: Technical screening criteria
            
        Returns:
            Dict containing:
            - success: bool
            - data: List of stocks matching technical criteria
            - criteria_applied: Applied screening criteria
            - last_updated: timestamp
        """
        pass

    @abstractmethod
    def get_fundamental_screener(self, user_id: int, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Screen stocks based on fundamental criteria.
        
        Args:
            user_id: The user ID for broker-specific authentication
            criteria: Fundamental screening criteria
            
        Returns:
            Dict containing:
            - success: bool
            - data: List of stocks matching fundamental criteria
            - criteria_applied: Applied screening criteria
            - last_updated: timestamp
        """
        pass
class SuggestedStock:
    """Data class for suggested stock information."""
    
    def __init__(self, symbol: str, name: str, strategy: StrategyType, 
                 current_price: float, recommendation: str):
        self.symbol = symbol
        self.name = name
        self.strategy = strategy
        self.current_price = current_price
        self.recommendation = recommendation
        self.target_price: Optional[float] = None
        self.stop_loss: Optional[float] = None
        self.reason: Optional[str] = None
        self.market_cap: Optional[float] = None
        self.pe_ratio: Optional[float] = None
        self.pb_ratio: Optional[float] = None
        self.roe: Optional[float] = None
        self.sales_growth: Optional[float] = None
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'strategy': self.strategy.value,
            'current_price': self.current_price,
            'recommendation': self.recommendation,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'reason': self.reason,
            'market_cap': self.market_cap,
            'pe_ratio': self.pe_ratio,
            'pb_ratio': self.pb_ratio,
            'roe': self.roe,
            'sales_growth': self.sales_growth,
            'last_updated': self.last_updated.isoformat()
        }
