"""
Stock Screening Package

Comprehensive stock screening system with meaningful stages:
- Market Data Screening: API-based data validation and volatility analysis
- Business Logic Screening: Fundamental analysis and portfolio optimization

Main Components:
- ScreeningCoordinator: Central orchestrator for the entire pipeline
- MarketDataScreener: Real-time quotes and historical volatility analysis
- BusinessLogicScreener: Fundamental ratios and portfolio optimization
"""

from .screening_coordinator import ScreeningCoordinator
from .market_data_screener import MarketDataScreener
from .business_logic_screener import BusinessLogicScreener, StrategyType

__all__ = [
    'ScreeningCoordinator',
    'MarketDataScreener',
    'BusinessLogicScreener',
    'StrategyType'
]

# Package version
__version__ = '2.0.0'

# Default configuration for quick setup
DEFAULT_SCREENING_CONFIG = {
    'market_data_initial': {
        'min_price_threshold': 100.0,
        'min_daily_volume': 100000,
        'max_daily_change_percent': 20.0,
        'batch_size': 50,
        'rate_limit_delay': 0.2
    },
    'market_data_historical': {
        'max_atr_percentage': 5.0,
        'max_beta': 1.2,
        'max_historical_volatility': 60.0,
        'min_avg_volume_20d': 50000,
        'min_daily_turnover': 1.0,
        'days_lookback': 252,
        'rate_limit_delay': 0.1
    },
    'business_logic': {
        'max_sector_allocation': 30.0,
        'min_market_cap_crores': 500,
        'max_pe_ratio': 50.0,
        'min_roe': 5.0,
        'max_debt_equity': 2.0,
        'max_final_stocks': 50
    }
}


def create_screening_coordinator(fyers_service, volatility_calculator_service, config=None):
    """
    Factory function to create a ScreeningCoordinator with optional custom configuration.

    Args:
        fyers_service: FYERS API service instance
        volatility_calculator_service: Volatility calculation service instance
        config: Optional custom configuration dict

    Returns:
        Configured ScreeningCoordinator instance
    """
    coordinator = ScreeningCoordinator(fyers_service, volatility_calculator_service)

    if config:
        coordinator.update_pipeline_configuration(config)

    return coordinator


def get_default_strategies():
    """Get the default strategy types for screening."""
    return [StrategyType.DEFAULT_RISK, StrategyType.HIGH_RISK]


def get_strategy_descriptions():
    """Get descriptions of available strategies."""
    return {
        StrategyType.DEFAULT_RISK: {
            'name': 'Conservative Swing Trading',
            'description': 'Low-risk strategy with strict volatility limits and large-cap focus',
            'target_return': '5-7%',
            'holding_period': '2-4 weeks',
            'risk_level': 'Low'
        },
        StrategyType.HIGH_RISK: {
            'name': 'Aggressive Swing Trading',
            'description': 'Higher-risk strategy with broader criteria and higher return potential',
            'target_return': '8-12%',
            'holding_period': '1-3 weeks',
            'risk_level': 'High'
        },
        StrategyType.MEDIUM_RISK: {
            'name': 'Balanced Swing Trading',
            'description': 'Moderate-risk strategy balancing returns and stability',
            'target_return': '6-9%',
            'holding_period': '2-3 weeks',
            'risk_level': 'Medium'
        }
    }