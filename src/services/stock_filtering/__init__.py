"""
Stock Filtering Module

This module provides services for stock filtering, data access, and transformation.
It follows the principle of separation of concerns with distinct classes for:
- Filtering logic (StockFilteringService)
- Database access (StockDataRepository)
- Data transformation (StockDataTransformer)
- Error handling (ErrorHandler and custom exceptions)
"""

from .stock_filtering_service import StockFilteringService, FilterType
from .stock_data_repository import StockDataRepository
from .stock_data_transformer import StockDataTransformer
from .error_handler import (
    ErrorHandler,
    StockFilteringError,
    DatabaseError,
    FilteringError,
    TransformationError,
    ScreeningError,
    ValidationError,
    ExternalAPIError,
    ErrorSeverity,
    ErrorCategory
)

__all__ = [
    'StockFilteringService',
    'FilterType',
    'StockDataRepository',
    'StockDataTransformer',
    'ErrorHandler',
    'StockFilteringError',
    'DatabaseError',
    'FilteringError',
    'TransformationError',
    'ScreeningError',
    'ValidationError',
    'ExternalAPIError',
    'ErrorSeverity',
    'ErrorCategory'
]