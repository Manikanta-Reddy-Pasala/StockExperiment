"""
Stock Filtering Module

This module provides essential and enhanced services for stock filtering and data access.
It includes core functionality and comprehensive filtering with Stage 1 and Stage 2 analysis.
"""

from .stock_filtering_service import StockFilteringService
from .stock_data_repository import StockDataRepository
from .stock_data_transformer import StockDataTransformer
from .error_handler import (
    ErrorHandler,
    StockFilteringError,
    DatabaseError,
    FilteringError,
    TransformationError,
    ScreeningError,
    ErrorSeverity
)

# Enhanced filtering services
from .enhanced_config_loader import (
    EnhancedFilteringConfig, get_enhanced_filtering_config,
    UniverseConfig, Stage1Filters, TechnicalIndicators, 
    FundamentalRatios, RiskMetrics, ScoringWeights, 
    FilteringThresholds, SelectionConfig
)
from .enhanced_stock_filtering_service import (
    EnhancedStockFilteringService, get_enhanced_filtering_service,
    FilteringResult, StockScore
)
from .technical_indicators_calculator import (
    TechnicalIndicatorsCalculator, get_technical_calculator,
    TechnicalIndicators
)
from .enhanced_stock_discovery_service import (
    EnhancedStockDiscoveryService, get_enhanced_discovery_service,
    DiscoveryResult
)

__all__ = [
    # Original services
    'StockFilteringService',
    'StockDataRepository',
    'StockDataTransformer',
    'ErrorHandler',
    'StockFilteringError',
    'DatabaseError',
    'FilteringError',
    'TransformationError',
    'ScreeningError',
    'ErrorSeverity',
    
    # Enhanced services
    'EnhancedFilteringConfig',
    'get_enhanced_filtering_config',
    'UniverseConfig',
    'Stage1Filters',
    'TechnicalIndicators',
    'FundamentalRatios',
    'RiskMetrics',
    'ScoringWeights',
    'FilteringThresholds',
    'SelectionConfig',
    'EnhancedStockFilteringService',
    'get_enhanced_filtering_service',
    'FilteringResult',
    'StockScore',
    'TechnicalIndicatorsCalculator',
    'get_technical_calculator',
    'TechnicalIndicators',
    'EnhancedStockDiscoveryService',
    'get_enhanced_discovery_service',
    'DiscoveryResult'
]