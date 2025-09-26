"""
Stock Filtering Error Handler

Provides centralized error handling for the stock filtering pipeline.
Implements proper error categorization, logging, and recovery strategies.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""
    LOW = "low"  # Recoverable, non-critical
    MEDIUM = "medium"  # Partially affects functionality
    HIGH = "high"  # Significantly affects functionality
    CRITICAL = "critical"  # System-breaking errors


class ErrorCategory(Enum):
    """Categories of errors in stock filtering pipeline."""
    DATABASE = "database"
    FILTERING = "filtering"
    TRANSFORMATION = "transformation"
    SCREENING = "screening"
    VALIDATION = "validation"
    EXTERNAL_API = "external_api"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class StockFilteringError(Exception):
    """Base exception class for stock filtering errors."""

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, details: Optional[Dict] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.timestamp = datetime.now()


class DatabaseError(StockFilteringError):
    """Error related to database operations."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCategory.DATABASE, ErrorSeverity.HIGH, details)


class FilteringError(StockFilteringError):
    """Error related to filtering operations."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCategory.FILTERING, ErrorSeverity.MEDIUM, details)


class TransformationError(StockFilteringError):
    """Error related to data transformation."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCategory.TRANSFORMATION, ErrorSeverity.LOW, details)


class ScreeningError(StockFilteringError):
    """Error related to screening pipeline."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCategory.SCREENING, ErrorSeverity.MEDIUM, details)


class ValidationError(StockFilteringError):
    """Error related to data validation."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.LOW, details)


class ExternalAPIError(StockFilteringError):
    """Error related to external API calls."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorCategory.EXTERNAL_API, ErrorSeverity.HIGH, details)


class ErrorHandler:
    """
    Centralized error handler for the stock filtering pipeline.

    Provides:
    - Error categorization and severity assessment
    - Logging with appropriate levels
    - Error recovery strategies
    - Error statistics and reporting
    """

    def __init__(self):
        """Initialize the error handler."""
        self.error_history: List[StockFilteringError] = []
        self.error_stats: Dict[str, int] = {
            category.value: 0 for category in ErrorCategory
        }
        self.recovery_attempts: Dict[str, int] = {}

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error with appropriate logging and recovery.

        Args:
            error: The exception to handle
            context: Additional context about where the error occurred

        Returns:
            Dictionary with error details and recovery suggestions
        """
        # Categorize the error
        if isinstance(error, StockFilteringError):
            categorized_error = error
        else:
            categorized_error = self._categorize_error(error, context)

        # Log the error
        self._log_error(categorized_error, context)

        # Record error statistics
        self._record_error_stats(categorized_error)

        # Determine recovery strategy
        recovery_strategy = self._get_recovery_strategy(categorized_error)

        return {
            'error_type': categorized_error.__class__.__name__,
            'message': str(categorized_error),
            'category': categorized_error.category.value,
            'severity': categorized_error.severity.value,
            'timestamp': categorized_error.timestamp.isoformat(),
            'details': categorized_error.details,
            'context': context,
            'recovery_strategy': recovery_strategy
        }

    def handle_batch_errors(self, errors: List[Exception],
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle multiple errors from batch operations.

        Args:
            errors: List of exceptions to handle
            context: Additional context about the batch operation

        Returns:
            Dictionary with batch error summary and recovery suggestions
        """
        handled_errors = []
        for error in errors:
            handled_errors.append(self.handle_error(error, context))

        # Analyze batch errors for patterns
        error_summary = self._analyze_batch_errors(handled_errors)

        return {
            'total_errors': len(errors),
            'error_summary': error_summary,
            'individual_errors': handled_errors[:10],  # Limit to first 10 for brevity
            'batch_recovery_strategy': self._get_batch_recovery_strategy(error_summary)
        }

    def attempt_recovery(self, error: StockFilteringError,
                        recovery_func: callable, max_attempts: int = 3) -> Optional[Any]:
        """
        Attempt to recover from an error by retrying an operation.

        Args:
            error: The error to recover from
            recovery_func: Function to call for recovery
            max_attempts: Maximum number of recovery attempts

        Returns:
            Result of recovery function if successful, None otherwise
        """
        error_key = f"{error.category.value}_{error.__class__.__name__}"
        attempts = self.recovery_attempts.get(error_key, 0)

        if attempts >= max_attempts:
            logger.warning(f"Max recovery attempts ({max_attempts}) reached for {error_key}")
            return None

        try:
            logger.info(f"Attempting recovery for {error_key} (attempt {attempts + 1})")
            result = recovery_func()
            self.recovery_attempts[error_key] = 0  # Reset on success
            return result

        except Exception as e:
            self.recovery_attempts[error_key] = attempts + 1
            logger.error(f"Recovery attempt failed: {e}")
            return None

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error statistics.

        Returns:
            Dictionary containing error statistics
        """
        total_errors = sum(self.error_stats.values())

        severity_distribution = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1

        return {
            'total_errors': total_errors,
            'errors_by_category': self.error_stats.copy(),
            'errors_by_severity': severity_distribution,
            'recent_errors': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'category': e.category.value,
                    'severity': e.severity.value,
                    'message': str(e)
                }
                for e in self.error_history[-10:]  # Last 10 errors
            ],
            'recovery_attempts': self.recovery_attempts.copy()
        }

    def reset_statistics(self):
        """Reset error statistics for a new session."""
        self.error_history = []
        self.error_stats = {category.value: 0 for category in ErrorCategory}
        self.recovery_attempts = {}
        logger.debug("Error statistics reset")

    # Private helper methods

    def _categorize_error(self, error: Exception, context: Optional[Dict[str, Any]]) -> StockFilteringError:
        """Categorize a generic exception into a StockFilteringError."""
        error_message = str(error)
        error_type = type(error).__name__

        # Database errors
        if 'database' in error_message.lower() or 'sql' in error_message.lower():
            return DatabaseError(error_message, {'original_type': error_type})

        # API errors
        if 'api' in error_message.lower() or 'request' in error_message.lower():
            return ExternalAPIError(error_message, {'original_type': error_type})

        # Validation errors
        if 'valid' in error_message.lower() or isinstance(error, (ValueError, TypeError)):
            return ValidationError(error_message, {'original_type': error_type})

        # Default to generic filtering error
        return FilteringError(error_message, {'original_type': error_type})

    def _log_error(self, error: StockFilteringError, context: Optional[Dict[str, Any]]):
        """Log error with appropriate level based on severity."""
        log_message = f"[{error.category.value}] {error}"

        if context:
            log_message += f" | Context: {context}"

        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, exc_info=True)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message, exc_info=True)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def _record_error_stats(self, error: StockFilteringError):
        """Record error in statistics."""
        self.error_history.append(error)
        self.error_stats[error.category.value] += 1

        # Keep history limited to prevent memory issues
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]

    def _get_recovery_strategy(self, error: StockFilteringError) -> Dict[str, Any]:
        """Determine recovery strategy based on error type and severity."""
        strategies = {
            ErrorCategory.DATABASE: {
                'retry': True,
                'wait_time': 5,
                'fallback': 'use_cached_data',
                'message': 'Database error - retrying with exponential backoff'
            },
            ErrorCategory.EXTERNAL_API: {
                'retry': True,
                'wait_time': 10,
                'fallback': 'use_local_data',
                'message': 'External API error - falling back to local data'
            },
            ErrorCategory.FILTERING: {
                'retry': False,
                'fallback': 'relax_criteria',
                'message': 'Filtering error - relaxing filter criteria'
            },
            ErrorCategory.TRANSFORMATION: {
                'retry': False,
                'fallback': 'use_defaults',
                'message': 'Transformation error - using default values'
            },
            ErrorCategory.SCREENING: {
                'retry': True,
                'wait_time': 3,
                'fallback': 'simplify_screening',
                'message': 'Screening error - simplifying screening logic'
            },
            ErrorCategory.VALIDATION: {
                'retry': False,
                'fallback': 'skip_invalid',
                'message': 'Validation error - skipping invalid entries'
            }
        }

        return strategies.get(error.category, {
            'retry': False,
            'fallback': 'skip',
            'message': 'Unknown error - skipping operation'
        })

    def _analyze_batch_errors(self, handled_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze batch errors for patterns."""
        category_counts = {}
        severity_counts = {}

        for error in handled_errors:
            category = error['category']
            severity = error['severity']

            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'by_category': category_counts,
            'by_severity': severity_counts,
            'dominant_category': max(category_counts, key=category_counts.get) if category_counts else None,
            'dominant_severity': max(severity_counts, key=severity_counts.get) if severity_counts else None
        }

    def _get_batch_recovery_strategy(self, error_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Determine recovery strategy for batch errors."""
        dominant_category = error_summary.get('dominant_category')
        dominant_severity = error_summary.get('dominant_severity')

        if dominant_severity == ErrorSeverity.CRITICAL.value:
            return {
                'action': 'abort',
                'message': 'Critical errors detected - aborting batch operation'
            }

        if dominant_category == ErrorCategory.DATABASE.value:
            return {
                'action': 'retry_batch',
                'message': 'Database errors dominant - retrying entire batch'
            }

        return {
            'action': 'retry_failed',
            'message': 'Retrying only failed items in batch'
        }