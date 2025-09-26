"""
Broker Implementations Package

This package contains the concrete implementations of the broker interfaces
for different brokers like FYERS and Zerodha.
"""

# FYERS implementations (conditional import)
try:
    from .fyers_dashboard_provider import FyersDashboardProvider
    from .fyers_suggested_stocks_provider import FyersSuggestedStocksProvider
    from .fyers_orders_provider import FyersOrdersProvider
    from .fyers_portfolio_provider import FyersPortfolioProvider
    from .fyers_reports_provider import FyersReportsProvider
    FYERS_AVAILABLE = True
except ImportError:
    FYERS_AVAILABLE = False

# Zerodha implementations (removed - not implemented)
ZERODHA_AVAILABLE = False

# Build __all__ list conditionally
__all__ = []

# Add FYERS if available
if FYERS_AVAILABLE:
    __all__.extend([
        'FyersDashboardProvider',
        'FyersSuggestedStocksProvider',
        'FyersOrdersProvider',
        'FyersPortfolioProvider',
        'FyersReportsProvider'
    ])

# Zerodha implementations removed - not available
