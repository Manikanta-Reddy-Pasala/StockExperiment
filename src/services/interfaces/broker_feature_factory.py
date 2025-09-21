"""
Broker Feature Factory

Implements the Factory and Strategy patterns to provide broker-specific
feature implementations based on user settings.
"""

from typing import Dict, Type, Optional
try:
    from ..utils.user_settings_service import get_user_settings_service
except ImportError:
    from src.services.utils.user_settings_service import get_user_settings_service
from .dashboard_interface import IDashboardProvider
from .suggested_stocks_interface import ISuggestedStocksProvider
from .orders_interface import IOrdersProvider
from .portfolio_interface import IPortfolioProvider
from .reports_interface import IReportsProvider


class BrokerFeatureFactory:
    """
    Factory class for creating broker-specific feature providers.
    
    This class implements the Factory pattern to create appropriate
    broker implementations based on user settings.
    """
    
    def __init__(self):
        self.user_settings_service = get_user_settings_service()
        self._dashboard_providers: Dict[str, Type[IDashboardProvider]] = {}
        self._suggested_stocks_providers: Dict[str, Type[ISuggestedStocksProvider]] = {}
        self._orders_providers: Dict[str, Type[IOrdersProvider]] = {}
        self._portfolio_providers: Dict[str, Type[IPortfolioProvider]] = {}
        self._reports_providers: Dict[str, Type[IReportsProvider]] = {}
        
        # Initialize broker providers
        self._register_default_providers()
    
    def _register_default_providers(self):
        """Register default broker providers."""
        # Import and register FYERS providers
        try:
            from ..implementations.fyers_dashboard_provider import FyersDashboardProvider
            from ..implementations.fyers_suggested_stocks_provider import FyersSuggestedStocksProvider
            from ..implementations.fyers_orders_provider import FyersOrdersProvider
            from ..implementations.fyers_portfolio_provider import FyersPortfolioProvider
            from ..implementations.fyers_reports_provider import FyersReportsProvider
            
            self.register_dashboard_provider('fyers', FyersDashboardProvider)
            self.register_suggested_stocks_provider('fyers', FyersSuggestedStocksProvider)
            self.register_orders_provider('fyers', FyersOrdersProvider)
            self.register_portfolio_provider('fyers', FyersPortfolioProvider)
            self.register_reports_provider('fyers', FyersReportsProvider)
        except ImportError:
            pass  # FYERS providers not available
        
        # Import and register Zerodha providers
        try:
            from ..implementations.zerodha_dashboard_provider import ZerodhaDashboardProvider
            from ..implementations.zerodha_suggested_stocks_provider import ZerodhaSuggestedStocksProvider
            from ..implementations.zerodha_orders_provider import ZerodhaOrdersProvider
            from ..implementations.zerodha_portfolio_provider import ZerodhaPortfolioProvider
            from ..implementations.zerodha_reports_provider import ZerodhaReportsProvider
            
            self.register_dashboard_provider('zerodha', ZerodhaDashboardProvider)
            self.register_suggested_stocks_provider('zerodha', ZerodhaSuggestedStocksProvider)
            self.register_orders_provider('zerodha', ZerodhaOrdersProvider)
            self.register_portfolio_provider('zerodha', ZerodhaPortfolioProvider)
            self.register_reports_provider('zerodha', ZerodhaReportsProvider)
        except ImportError:
            pass  # Zerodha providers not available
        
    
    # Registration methods
    def register_dashboard_provider(self, broker: str, provider_class: Type[IDashboardProvider]):
        """Register a dashboard provider for a broker."""
        self._dashboard_providers[broker] = provider_class
    
    def register_suggested_stocks_provider(self, broker: str, provider_class: Type[ISuggestedStocksProvider]):
        """Register a suggested stocks provider for a broker."""
        self._suggested_stocks_providers[broker] = provider_class
    
    def register_orders_provider(self, broker: str, provider_class: Type[IOrdersProvider]):
        """Register an orders provider for a broker."""
        self._orders_providers[broker] = provider_class
    
    def register_portfolio_provider(self, broker: str, provider_class: Type[IPortfolioProvider]):
        """Register a portfolio provider for a broker."""
        self._portfolio_providers[broker] = provider_class
    
    def register_reports_provider(self, broker: str, provider_class: Type[IReportsProvider]):
        """Register a reports provider for a broker."""
        self._reports_providers[broker] = provider_class
    
    # Factory methods
    def get_dashboard_provider(self, user_id: int) -> Optional[IDashboardProvider]:
        """Get dashboard provider based on user's broker selection."""
        broker = self.user_settings_service.get_broker_provider(user_id)
        provider_class = self._dashboard_providers.get(broker)
        return provider_class() if provider_class else None
    
    def get_suggested_stocks_provider(self, user_id: int) -> Optional[ISuggestedStocksProvider]:
        """Get suggested stocks provider based on user's broker selection."""
        broker = self.user_settings_service.get_broker_provider(user_id)
        provider_class = self._suggested_stocks_providers.get(broker)
        return provider_class() if provider_class else None
    
    def get_orders_provider(self, user_id: int) -> Optional[IOrdersProvider]:
        """Get orders provider based on user's broker selection."""
        broker = self.user_settings_service.get_broker_provider(user_id)
        provider_class = self._orders_providers.get(broker)
        return provider_class() if provider_class else None
    
    def get_portfolio_provider(self, user_id: int) -> Optional[IPortfolioProvider]:
        """Get portfolio provider based on user's broker selection."""
        broker = self.user_settings_service.get_broker_provider(user_id)
        provider_class = self._portfolio_providers.get(broker)
        return provider_class() if provider_class else None
    
    def get_reports_provider(self, user_id: int) -> Optional[IReportsProvider]:
        """Get reports provider based on user's broker selection."""
        broker = self.user_settings_service.get_broker_provider(user_id)
        provider_class = self._reports_providers.get(broker)
        return provider_class() if provider_class else None
    
    def get_available_brokers(self) -> Dict[str, Dict[str, bool]]:
        """Get list of available brokers and their supported features."""
        brokers = {}
        all_brokers = set()
        all_brokers.update(self._dashboard_providers.keys())
        all_brokers.update(self._suggested_stocks_providers.keys())
        all_brokers.update(self._orders_providers.keys())
        all_brokers.update(self._portfolio_providers.keys())
        all_brokers.update(self._reports_providers.keys())
        
        for broker in all_brokers:
            brokers[broker] = {
                'dashboard': broker in self._dashboard_providers,
                'suggested_stocks': broker in self._suggested_stocks_providers,
                'orders': broker in self._orders_providers,
                'portfolio': broker in self._portfolio_providers,
                'reports': broker in self._reports_providers
            }
        
        return brokers


# Global factory instance
_broker_feature_factory = None

def get_broker_feature_factory() -> BrokerFeatureFactory:
    """Get the global broker feature factory instance."""
    global _broker_feature_factory
    if _broker_feature_factory is None:
        _broker_feature_factory = BrokerFeatureFactory()
    return _broker_feature_factory
