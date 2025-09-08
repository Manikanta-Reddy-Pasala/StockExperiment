"""
Base Strategy Interface for Momentum Selection
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for momentum selection strategies."""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the strategy.
        
        Args:
            name (str): Name of the strategy
            description (str): Description of the strategy
        """
        self.name = name
        self.description = description
        self.parameters = {}
    
    @abstractmethod
    def select_stocks(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Select stocks based on the strategy.
        
        Args:
            market_data (pd.DataFrame): Market data for stock selection
            
        Returns:
            List[Dict[str, Any]]: List of selected stocks with details
        """
        pass
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Set strategy parameters.
        
        Args:
            parameters (Dict[str, Any]): Strategy parameters
        """
        self.parameters.update(parameters)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters.
        
        Returns:
            Dict[str, Any]: Strategy parameters
        """
        return self.parameters.copy()