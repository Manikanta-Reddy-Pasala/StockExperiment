"""
Stock Data Transformer

Handles data transformation and formatting for stock data.
Separates data transformation logic from filtering and database operations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class StockDataTransformer:
    """
    Service for transforming stock data between different formats.

    This class handles all data transformation operations, including:
    - Converting database models to dictionaries
    - Formatting data for API responses
    - Enriching data with calculated fields
    - Validating and sanitizing data
    """

    def __init__(self):
        """Initialize the stock data transformer."""
        self.transformation_stats = {
            'total_transformed': 0,
            'transformation_errors': 0,
            'null_fields_replaced': 0
        }

    def transform_stock_to_dict(self, stock: Any,
                               include_volatility: bool = True,
                               include_fundamentals: bool = True) -> Dict[str, Any]:
        """
        Transform a single stock object to dictionary format.

        Args:
            stock: Stock object to transform
            include_volatility: Whether to include volatility metrics
            include_fundamentals: Whether to include fundamental metrics

        Returns:
            Dictionary representation of the stock
        """
        try:
            # Base stock data
            stock_data = {
                'symbol': self._get_field_value(stock, 'symbol', ''),
                'name': self._get_field_value(stock, 'name', 'Unknown'),
                'current_price': self._get_field_value(stock, 'current_price', 0.0),
                'volume': self._get_field_value(stock, 'volume', 0),
                'sector': self._get_field_value(stock, 'sector', 'Unknown'),
                'market_cap': self._get_field_value(stock, 'market_cap', 0.0),
                'market_cap_category': self._get_field_value(stock, 'market_cap_category', 'mid_cap'),
                'last_updated': self._format_datetime(
                    self._get_field_value(stock, 'last_updated', None)
                )
            }

            # Add volatility metrics if requested
            if include_volatility:
                volatility_data = self._extract_volatility_metrics(stock)
                stock_data.update(volatility_data)

            # Add fundamental metrics if requested
            if include_fundamentals:
                fundamental_data = self._extract_fundamental_metrics(stock)
                stock_data.update(fundamental_data)

            self.transformation_stats['total_transformed'] += 1
            return stock_data

        except Exception as e:
            logger.error(f"Error transforming stock to dict: {e}")
            self.transformation_stats['transformation_errors'] += 1
            return self._get_empty_stock_dict()

    def transform_stocks_batch(self, stocks: List[Any],
                              include_volatility: bool = True,
                              include_fundamentals: bool = True) -> List[Dict[str, Any]]:
        """
        Transform a batch of stock objects to dictionary format.

        Args:
            stocks: List of stock objects to transform
            include_volatility: Whether to include volatility metrics
            include_fundamentals: Whether to include fundamental metrics

        Returns:
            List of dictionary representations
        """
        transformed_stocks = []

        for stock in stocks:
            stock_dict = self.transform_stock_to_dict(
                stock,
                include_volatility=include_volatility,
                include_fundamentals=include_fundamentals
            )
            transformed_stocks.append(stock_dict)

        logger.info(f"Transformed batch of {len(transformed_stocks)} stocks")
        return transformed_stocks

    def enrich_stock_data(self, stock_data: Dict[str, Any],
                         enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich stock data with additional information.

        Args:
            stock_data: Base stock data dictionary
            enrichment_data: Additional data to merge

        Returns:
            Enriched stock data dictionary
        """
        try:
            enriched = stock_data.copy()

            # Add calculated fields
            if 'current_price' in enriched and 'market_cap' in enriched:
                if enriched['current_price'] > 0:
                    enriched['shares_outstanding'] = (
                        enriched['market_cap'] / enriched['current_price']
                        if enriched['market_cap'] else 0
                    )

            # Add risk metrics
            enriched['risk_score'] = self._calculate_risk_score(enriched)
            enriched['liquidity_score'] = self._calculate_liquidity_score(enriched)

            # Merge enrichment data
            for key, value in enrichment_data.items():
                if key not in enriched or enriched[key] is None:
                    enriched[key] = value

            return enriched

        except Exception as e:
            logger.error(f"Error enriching stock data: {e}")
            return stock_data

    def format_for_api_response(self, stocks: List[Dict[str, Any]],
                               include_metadata: bool = True) -> Dict[str, Any]:
        """
        Format stock data for API response.

        Args:
            stocks: List of stock dictionaries
            include_metadata: Whether to include response metadata

        Returns:
            Formatted API response dictionary
        """
        response = {
            'success': True,
            'data': stocks,
            'count': len(stocks)
        }

        if include_metadata:
            response['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'transformation_stats': self.get_transformation_stats()
            }

        return response

    def validate_stock_data(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize stock data.

        Args:
            stock_data: Stock data dictionary to validate

        Returns:
            Validated and sanitized stock data
        """
        validated = stock_data.copy()

        # Validate required fields
        required_fields = ['symbol', 'name', 'current_price']
        for field in required_fields:
            if field not in validated or validated[field] is None:
                logger.warning(f"Missing required field '{field}' for stock")
                validated[field] = self._get_default_value(field)

        # Validate numeric fields
        numeric_fields = [
            'current_price', 'volume', 'market_cap', 'pe_ratio',
            'pb_ratio', 'roe', 'debt_to_equity', 'dividend_yield',
            'beta', 'atr_14', 'atr_percentage', 'historical_volatility_1y'
        ]

        for field in numeric_fields:
            if field in validated:
                validated[field] = self._validate_numeric(validated[field], field)

        # Validate price bounds
        if validated.get('current_price', 0) < 0:
            validated['current_price'] = 0

        return validated

    def aggregate_stock_metrics(self, stocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics for a list of stocks.

        Args:
            stocks: List of stock dictionaries

        Returns:
            Dictionary of aggregate metrics
        """
        if not stocks:
            return {}

        total_market_cap = sum(s.get('market_cap', 0) for s in stocks)
        avg_pe_ratio = self._safe_average([s.get('pe_ratio') for s in stocks])
        avg_volume = self._safe_average([s.get('volume') for s in stocks])

        sector_distribution = {}
        for stock in stocks:
            sector = stock.get('sector', 'Unknown')
            sector_distribution[sector] = sector_distribution.get(sector, 0) + 1

        return {
            'total_stocks': len(stocks),
            'total_market_cap': total_market_cap,
            'average_pe_ratio': avg_pe_ratio,
            'average_volume': avg_volume,
            'sector_distribution': sector_distribution,
            'price_range': {
                'min': min((s.get('current_price', 0) for s in stocks), default=0),
                'max': max((s.get('current_price', 0) for s in stocks), default=0)
            }
        }

    def get_transformation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about data transformations.

        Returns:
            Dictionary of transformation statistics
        """
        return self.transformation_stats.copy()

    def reset_stats(self):
        """Reset transformation statistics."""
        self.transformation_stats = {
            'total_transformed': 0,
            'transformation_errors': 0,
            'null_fields_replaced': 0
        }

    # Private helper methods

    def _get_field_value(self, obj: Any, field: str, default: Any = None) -> Any:
        """Safely get field value from object."""
        try:
            value = getattr(obj, field, default)
            if value is None:
                self.transformation_stats['null_fields_replaced'] += 1
                return default
            return value
        except Exception:
            return default

    def _format_datetime(self, dt: Any) -> Optional[str]:
        """Format datetime object to ISO string."""
        if dt is None:
            return None
        if isinstance(dt, datetime):
            return dt.isoformat()
        return str(dt)

    def _extract_volatility_metrics(self, stock: Any) -> Dict[str, Any]:
        """Extract volatility-related metrics from stock object."""
        return {
            'atr_14': self._get_field_value(stock, 'atr_14', 0.0),
            'atr_percentage': self._get_field_value(stock, 'atr_percentage', 0.0),
            'historical_volatility_1y': self._get_field_value(
                stock, 'historical_volatility_1y', 0.0
            ),
            'avg_daily_volume_20d': self._get_field_value(stock, 'avg_daily_volume_20d', 0),
            'avg_daily_turnover': self._get_field_value(stock, 'avg_daily_turnover', 0.0),
            'bid_ask_spread': self._get_field_value(stock, 'bid_ask_spread', 0.0),
            'trades_per_day': self._get_field_value(stock, 'trades_per_day', 0),
            'beta': self._get_field_value(stock, 'beta', 1.0)
        }

    def _extract_fundamental_metrics(self, stock: Any) -> Dict[str, Any]:
        """Extract fundamental metrics from stock object."""
        return {
            'pe_ratio': self._get_field_value(stock, 'pe_ratio', 0.0),
            'pb_ratio': self._get_field_value(stock, 'pb_ratio', 0.0),
            'roe': self._get_field_value(stock, 'roe', 0.0),
            'debt_to_equity': self._get_field_value(stock, 'debt_to_equity', 0.0),
            'dividend_yield': self._get_field_value(stock, 'dividend_yield', 0.0)
        }

    def _get_empty_stock_dict(self) -> Dict[str, Any]:
        """Get an empty stock dictionary with default values."""
        return {
            'symbol': '',
            'name': 'Unknown',
            'current_price': 0.0,
            'volume': 0,
            'sector': 'Unknown',
            'market_cap': 0.0,
            'market_cap_category': 'unknown',
            'last_updated': None
        }

    def _get_default_value(self, field: str) -> Any:
        """Get default value for a field."""
        defaults = {
            'symbol': '',
            'name': 'Unknown',
            'current_price': 0.0,
            'volume': 0,
            'market_cap': 0.0,
            'sector': 'Unknown',
            'market_cap_category': 'unknown'
        }
        return defaults.get(field, None)

    def _validate_numeric(self, value: Any, field: str) -> float:
        """Validate and convert numeric value."""
        try:
            if value is None:
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid numeric value for field '{field}': {value}")
            return 0.0

    def _calculate_risk_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate risk score for a stock (0-100)."""
        score = 50.0  # Base score

        # Adjust based on volatility
        volatility = stock_data.get('historical_volatility_1y', 0)
        if volatility > 0.5:
            score += 20
        elif volatility > 0.3:
            score += 10

        # Adjust based on beta
        beta = stock_data.get('beta', 1.0)
        if beta > 1.5:
            score += 15
        elif beta > 1.0:
            score += 5

        # Adjust based on debt-to-equity
        debt_to_equity = stock_data.get('debt_to_equity', 0)
        if debt_to_equity > 2:
            score += 15
        elif debt_to_equity > 1:
            score += 5

        return min(100, max(0, score))

    def _calculate_liquidity_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate liquidity score for a stock (0-100)."""
        score = 0.0

        # Volume contribution
        volume = stock_data.get('volume', 0)
        if volume > 10000000:
            score += 40
        elif volume > 1000000:
            score += 25
        elif volume > 100000:
            score += 10

        # Average volume contribution
        avg_volume = stock_data.get('avg_daily_volume_20d', 0)
        if avg_volume > 5000000:
            score += 30
        elif avg_volume > 1000000:
            score += 20
        elif avg_volume > 100000:
            score += 10

        # Bid-ask spread contribution (lower is better)
        spread = stock_data.get('bid_ask_spread', 0)
        if spread < 0.01:
            score += 30
        elif spread < 0.05:
            score += 20
        elif spread < 0.1:
            score += 10

        return min(100, score)

    def _safe_average(self, values: List[Any]) -> float:
        """Calculate average of numeric values, handling None values."""
        numeric_values = [v for v in values if v is not None and isinstance(v, (int, float))]
        if not numeric_values:
            return 0.0
        return sum(numeric_values) / len(numeric_values)