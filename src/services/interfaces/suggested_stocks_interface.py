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
    """Strategy types for stock screening."""
    DEFAULT_RISK = "default_risk"
    HIGH_RISK = "high_risk"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    UNIFIED = "unified"  # Single unified 8-21 EMA strategy


class ISuggestedStocksProvider(ABC):
    """
    Interface for suggested stocks data providers.
    
    This interface defines all the methods that must be implemented
    by each broker to provide stock screening and suggestions functionality.
    """

    @abstractmethod
    def discover_tradeable_stocks(self, user_id: int, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Discover all tradeable stocks from the broker API.

        Args:
            user_id: The user ID for broker-specific authentication
            exchange: The exchange to search (e.g., NSE, BSE)

        Returns:
            Dict containing:
            - success: bool
            - data: List of discovered stocks with categorization
            - total_discovered: int
            - filtering_statistics: Dict with counts by category
            - last_updated: timestamp
        """
        pass

    @abstractmethod
    def search_stocks(self, user_id: int, search_term: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Search for stocks using a search term.

        Args:
            user_id: The user ID for broker-specific authentication
            search_term: Term to search for (e.g., "BANK", "IT", "RELIANCE")
            exchange: The exchange to search

        Returns:
            Dict containing:
            - success: bool
            - data: List of matching stocks
            - search_term: The term that was searched
            - total_results: int
        """
        pass

    @abstractmethod
    def get_suggested_stocks(self, user_id: int, strategies: Optional[List[str]] = None,
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
    def get_strategy_performance(self, user_id: int, strategy: Optional[str] = None,
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
    
    def __init__(self, symbol: str, name: str, strategy: Optional[str] = None,
                 current_price: float = 0.0, recommendation: str = ''):
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
            'strategy': self.strategy.value if hasattr(self.strategy, 'value') else str(self.strategy),
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
