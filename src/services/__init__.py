"""
Services package - Business logic layer

This package contains all business logic services organized into logical modules:

- core/       : Core business services (users, orders, strategies, brokers)
- brokers/    : Broker integration services (Fyers, Zerodha)
- ml/         : Machine learning and AI services
- data/       : Data management and synchronization services
- portfolio/  : Portfolio management and tracking services
- market/     : Market data and stock screening services
- utils/      : Utility services (cache, alerts, settings)
- interfaces/ : Service interface definitions
- implementations/ : Broker-specific service implementations
"""

# Import commonly used services for backward compatibility
from .core import (
    get_user_service,
    get_unified_broker_service,
    get_broker_service,
    get_dashboard_service
)

from .data import (
    get_symbol_database_service,
    get_fyers_symbol_service
)

from .portfolio import (
    get_portfolio_service
)

from .market import (
    get_basic_stock_screening_service,
    get_market_data_service
)

from .utils import (
    get_cache_service
)

# ML services (imported directly due to their specific nature)
from .ml.stock_discovery_service import get_stock_discovery_service

__all__ = [
    # Core services
    'get_user_service',
    'get_unified_broker_service',
    'get_broker_service',
    'get_dashboard_service',

    # Data services
    'get_symbol_database_service',
    'get_fyers_symbol_service',

    # Portfolio services
    'get_portfolio_service',

    # Market services
    'get_basic_stock_screening_service',
    'get_market_data_service',

    # Utility services
    'get_cache_service',

    # ML services
    'get_stock_discovery_service'
]
