"""
Base Broker Connector Interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BrokerConnector(ABC):
    """Abstract base class for broker connectors."""
    
    @abstractmethod
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile."""
        pass
    
    @abstractmethod
    def get_margins(self) -> Dict[str, Any]:
        """Get user margins."""
        pass
    
    @abstractmethod
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders."""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def get_holdings(self) -> List[Dict[str, Any]]:
        """Get current holdings."""
        pass
    
    @abstractmethod
    def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order."""
        pass
    
    @abstractmethod
    def modify_order(self, order_id: str, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an existing order."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def get_instruments(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get instrument master data."""
        pass
    
    @abstractmethod
    def get_ltp(self, instrument_tokens: List[str]) -> Dict[str, Any]:
        """Get last traded price for instruments."""
        pass
    
    @abstractmethod
    def get_quote(self, instrument_tokens: List[str]) -> Dict[str, Any]:
        """Get quote data for instruments."""
        pass