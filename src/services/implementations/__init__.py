"""
Broker Implementations Package

This package contains the concrete implementations of the broker interfaces
for different brokers like FYERS, Zerodha, and Simulator.
"""

# FYERS implementations
from .fyers_dashboard_provider import FyersDashboardProvider
from .fyers_suggested_stocks_provider import FyersSuggestedStocksProvider
from .fyers_orders_provider import FyersOrdersProvider
from .fyers_portfolio_provider import FyersPortfolioProvider
from .fyers_reports_provider import FyersReportsProvider

# Zerodha implementations
from .zerodha_dashboard_provider import ZerodhaDashboardProvider
from .zerodha_suggested_stocks_provider import ZerodhaSuggestedStocksProvider
from .zerodha_orders_provider import ZerodhaOrdersProvider
from .zerodha_portfolio_provider import ZerodhaPortfolioProvider
from .zerodha_reports_provider import ZerodhaReportsProvider

# Simulator implementations
from .simulator_dashboard_provider import SimulatorDashboardProvider
from .simulator_suggested_stocks_provider import SimulatorSuggestedStocksProvider
from .simulator_orders_provider import SimulatorOrdersProvider
from .simulator_portfolio_provider import SimulatorPortfolioProvider
from .simulator_reports_provider import SimulatorReportsProvider

__all__ = [
    # FYERS
    'FyersDashboardProvider',
    'FyersSuggestedStocksProvider',
    'FyersOrdersProvider',
    'FyersPortfolioProvider',
    'FyersReportsProvider',
    
    # Zerodha
    'ZerodhaDashboardProvider',
    'ZerodhaSuggestedStocksProvider',
    'ZerodhaOrdersProvider',
    'ZerodhaPortfolioProvider',
    'ZerodhaReportsProvider',
    
    # Simulator
    'SimulatorDashboardProvider',
    'SimulatorSuggestedStocksProvider',
    'SimulatorOrdersProvider',
    'SimulatorPortfolioProvider',
    'SimulatorReportsProvider'
]
