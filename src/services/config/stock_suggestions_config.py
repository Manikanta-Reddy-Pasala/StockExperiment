"""
Stock Suggestions Configuration Loader
Loads and validates stock_suggestions.yaml configuration
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class StockSuggestionsConfig:
    """Loads and provides access to stock suggestions configuration"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader

        Args:
            config_path: Path to config file (defaults to config/stock_suggestions.yaml)
        """
        if config_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / 'config' / 'stock_suggestions.yaml'

        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")

            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)

            logger.info(f"✅ Loaded stock suggestions config from {self.config_path}")
            self._validate_config()

        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            raise

    def _validate_config(self):
        """Validate configuration structure"""
        required_sections = [
            'model_selection',
            'valuation',
            'risk',
            'liquidity',
            'display',
            'strategies',
            'data_quality'
        ]

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")

        logger.info("✅ Config validation passed")

    # ==========================================
    # MODEL SELECTION
    # ==========================================

    def get_enabled_models(self) -> List[str]:
        """Get list of enabled models"""
        return self._config['model_selection']['enabled_models']

    def get_minimum_score(self, model_type: str) -> float:
        """Get minimum ML score for a specific model"""
        scores = self._config['model_selection']['minimum_scores']
        return scores.get(model_type, 0.50)  # Default 0.50 if not specified

    def get_allowed_recommendations(self) -> List[str]:
        """Get allowed recommendation types (e.g., ['BUY'])"""
        return self._config['model_selection']['allowed_recommendations']

    def get_minimum_confidence(self, strategy: str) -> float:
        """Get minimum confidence for a strategy"""
        confidence = self._config['model_selection']['minimum_confidence']
        strategy_key = strategy.lower().replace('_risk', '_risk')
        return confidence.get(strategy_key, 0.50)

    # ==========================================
    # VALUATION FILTERS
    # ==========================================

    def get_minimum_upside_pct(self) -> float:
        """Get minimum upside percentage"""
        return self._config['valuation']['minimum_upside_pct']

    def get_maximum_upside_pct(self) -> float:
        """Get maximum upside percentage"""
        return self._config['valuation']['maximum_upside_pct']

    def get_pe_ratio_range(self) -> Dict[str, Any]:
        """Get PE ratio range and settings"""
        return self._config['valuation']['pe_ratio']

    def get_pb_ratio_range(self) -> Dict[str, Any]:
        """Get PB ratio range and settings"""
        return self._config['valuation']['pb_ratio']

    def get_roe_range(self) -> Dict[str, Any]:
        """Get ROE range and settings"""
        return self._config['valuation']['roe']

    # ==========================================
    # RISK MANAGEMENT
    # ==========================================

    def get_maximum_risk_score(self) -> float:
        """Get maximum allowed risk score"""
        return self._config['risk']['maximum_risk_score']

    def require_stop_loss(self) -> bool:
        """Check if stop loss is required"""
        return self._config['risk']['require_stop_loss']

    def get_stop_loss_range(self) -> Dict[str, float]:
        """Get stop loss percentage range"""
        return {
            'minimum': self._config['risk']['minimum_stop_loss_pct'],
            'maximum': self._config['risk']['maximum_stop_loss_pct']
        }

    def get_minimum_risk_reward_ratio(self) -> float:
        """Get minimum risk-reward ratio"""
        return self._config['risk']['minimum_risk_reward_ratio']

    # ==========================================
    # LIQUIDITY FILTERS
    # ==========================================

    def get_liquidity_config(self, strategy: str) -> Dict[str, Any]:
        """Get liquidity configuration for a strategy"""
        strategy_key = strategy.lower().replace('_risk', '_risk')
        return self._config['liquidity'].get(strategy_key, self._config['liquidity']['default_risk'])

    def get_minimum_daily_volume(self) -> int:
        """Get minimum daily trading volume"""
        return self._config['liquidity']['minimum_daily_volume']

    def get_minimum_daily_turnover(self) -> float:
        """Get minimum daily turnover in Crores"""
        return self._config['liquidity']['minimum_daily_turnover_cr']

    # ==========================================
    # DISPLAY SETTINGS
    # ==========================================

    def get_default_limit(self) -> int:
        """Get default number of stocks to display"""
        return self._config['display']['default_limit']

    def get_maximum_limit(self) -> int:
        """Get maximum number of stocks allowed"""
        return self._config['display']['maximum_limit']

    def get_sort_settings(self) -> Dict[str, Any]:
        """Get sorting configuration"""
        return {
            'sort_by': self._config['display']['sort_by'],
            'sort_order': self._config['display']['sort_order'],
            'secondary_sort': self._config['display']['secondary_sort']
        }

    # ==========================================
    # STRATEGY-SPECIFIC RULES
    # ==========================================

    def get_strategy_config(self, strategy: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy"""
        strategy_key = strategy.upper().replace('DEFAULT_RISK', 'DEFAULT_RISK').replace('HIGH_RISK', 'HIGH_RISK')
        return self._config['strategies'].get(strategy_key, {})

    # ==========================================
    # DATA QUALITY
    # ==========================================

    def get_maximum_data_age_days(self) -> int:
        """Get maximum age of data in days"""
        return self._config['data_quality']['maximum_data_age_days']

    def get_required_fields(self) -> List[str]:
        """Get list of required fields"""
        return self._config['data_quality']['required_fields']

    def should_remove_outliers(self) -> bool:
        """Check if outlier removal is enabled"""
        return self._config['data_quality']['remove_outliers']

    def get_outlier_z_score_threshold(self) -> float:
        """Get Z-score threshold for outlier detection"""
        return self._config['data_quality']['outlier_z_score_threshold']

    # ==========================================
    # SQL FILTER BUILDER
    # ==========================================

    def build_sql_filters(self, model_type: str, strategy: str) -> Dict[str, Any]:
        """
        Build SQL filter parameters for a specific model and strategy

        Returns:
            Dictionary of filter parameters for SQL query
        """
        filters = {
            # Model-specific score
            'min_ml_score': self.get_minimum_score(model_type),

            # Recommendations
            'allowed_recommendations': self.get_allowed_recommendations(),

            # Confidence
            'min_confidence': self.get_minimum_confidence(strategy),

            # Upside
            'min_upside_pct': self.get_minimum_upside_pct(),
            'max_upside_pct': self.get_maximum_upside_pct(),

            # Risk
            'max_risk_score': self.get_maximum_risk_score(),
            'require_stop_loss': self.require_stop_loss(),
            'min_risk_reward_ratio': self.get_minimum_risk_reward_ratio(),

            # PE Ratio
            'pe_min': self.get_pe_ratio_range()['minimum'],
            'pe_max': self.get_pe_ratio_range()['maximum'],
            'pe_allow_null': self.get_pe_ratio_range()['allow_null'],

            # PB Ratio
            'pb_min': self.get_pb_ratio_range()['minimum'],
            'pb_max': self.get_pb_ratio_range()['maximum'],
            'pb_allow_null': self.get_pb_ratio_range()['allow_null'],

            # ROE
            'roe_min': self.get_roe_range()['minimum'],
            'roe_max': self.get_roe_range()['maximum'],
            'roe_allow_null': self.get_roe_range()['allow_null'],

            # Liquidity
            'min_daily_volume': self.get_minimum_daily_volume(),
            'min_daily_turnover': self.get_minimum_daily_turnover(),

            # Strategy-specific
            'strategy_config': self.get_strategy_config(strategy)
        }

        return filters

    def build_where_clause(self, model_type: str, strategy: str) -> tuple[str, Dict[str, Any]]:
        """
        Build SQL WHERE clause with parameters

        Returns:
            Tuple of (WHERE clause string, parameters dict)
        """
        filters = self.build_sql_filters(model_type, strategy)

        where_clauses = []
        params = {}

        # Recommendation filter
        if filters['allowed_recommendations']:
            where_clauses.append("d.recommendation IN :allowed_recs")
            params['allowed_recs'] = tuple(filters['allowed_recommendations'])

        # ML Score (per-model threshold)
        where_clauses.append("d.ml_prediction_score >= :min_score")
        params['min_score'] = filters['min_ml_score']

        # Confidence
        where_clauses.append("d.ml_confidence >= :min_confidence")
        params['min_confidence'] = filters['min_confidence']

        # Upside validation
        where_clauses.append("d.ml_price_target > d.current_price")
        where_clauses.append("((d.ml_price_target - d.current_price) / d.current_price * 100) >= :min_upside")
        where_clauses.append("((d.ml_price_target - d.current_price) / d.current_price * 100) <= :max_upside")
        params['min_upside'] = filters['min_upside_pct']
        params['max_upside'] = filters['max_upside_pct']

        # Risk score
        where_clauses.append("d.ml_risk_score <= :max_risk")
        params['max_risk'] = filters['max_risk_score']

        # PE Ratio
        if not filters['pe_allow_null']:
            where_clauses.append("d.pe_ratio IS NOT NULL")
        where_clauses.append("(d.pe_ratio IS NULL OR (d.pe_ratio >= :pe_min AND d.pe_ratio <= :pe_max))")
        params['pe_min'] = filters['pe_min']
        params['pe_max'] = filters['pe_max']

        # PB Ratio
        if not filters['pb_allow_null']:
            where_clauses.append("d.pb_ratio IS NOT NULL")
        where_clauses.append("(d.pb_ratio IS NULL OR (d.pb_ratio >= :pb_min AND d.pb_ratio <= :pb_max))")
        params['pb_min'] = filters['pb_min']
        params['pb_max'] = filters['pb_max']

        # ROE
        if not filters['roe_allow_null']:
            where_clauses.append("d.roe IS NOT NULL")
        where_clauses.append("(d.roe IS NULL OR (d.roe >= :roe_min AND d.roe <= :roe_max))")
        params['roe_min'] = filters['roe_min']
        params['roe_max'] = filters['roe_max']

        where_clause = " AND ".join(where_clauses)

        return where_clause, params


# Singleton instance
_config_instance: Optional[StockSuggestionsConfig] = None


@lru_cache(maxsize=1)
def get_stock_suggestions_config() -> StockSuggestionsConfig:
    """Get singleton instance of stock suggestions config"""
    global _config_instance
    if _config_instance is None:
        _config_instance = StockSuggestionsConfig()
    return _config_instance
