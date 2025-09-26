"""
Enhanced Stock Discovery Service
Integrates comprehensive stock filtering with Stage 1 and Stage 2 analysis
Supports all features from the enhanced YAML configuration
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
from dataclasses import dataclass

from .enhanced_config_loader import (
    EnhancedFilteringConfig, get_enhanced_filtering_config
)
from .enhanced_stock_filtering_service import (
    EnhancedStockFilteringService, get_enhanced_filtering_service,
    FilteringResult, StockScore
)
from .technical_indicators_calculator import (
    TechnicalIndicatorsCalculator, get_technical_calculator,
    TechnicalIndicators
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Stock discovery result."""
    selected_stocks: List[Dict[str, Any]]
    rejected_stocks: List[Dict[str, Any]]
    total_processed: int
    stage1_passed: int
    stage2_passed: int
    final_selected: int
    execution_time: float
    config_used: Dict[str, Any]
    summary: Dict[str, Any]


class EnhancedStockDiscoveryService:
    """Enhanced stock discovery service with comprehensive filtering."""
    
    def __init__(self, config: Optional[EnhancedFilteringConfig] = None):
        """Initialize the enhanced discovery service."""
        self.config = config or get_enhanced_filtering_config()
        self.filtering_service = get_enhanced_filtering_service()
        self.technical_calculator = get_technical_calculator()
        
        # Statistics tracking
        self.discovery_stats = {
            'total_discoveries': 0,
            'successful_discoveries': 0,
            'average_execution_time': 0.0,
            'total_stocks_processed': 0,
            'total_stocks_selected': 0
        }
    
    def discover_stocks(self, user_id: int = 1, 
                       limit: Optional[int] = None) -> DiscoveryResult:
        """
        Discover stocks using comprehensive filtering.
        
        Args:
            user_id: User ID for personalized discovery
            limit: Maximum number of stocks to process
            
        Returns:
            DiscoveryResult with selected and rejected stocks
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting enhanced stock discovery for user {user_id}")
            
            # Get stocks from database
            stocks = self._get_stocks_from_database(limit)
            if not stocks:
                logger.warning("No stocks found in database")
                return self._create_empty_result()
            
            logger.info(f"Retrieved {len(stocks)} stocks from database")
            
            # Apply comprehensive filtering
            filtering_result = self.filtering_service.filter_stocks(stocks, user_id)
            
            # Convert results to discovery format
            selected_stocks = self._convert_to_discovery_format(filtering_result.selected_stocks)
            rejected_stocks = self._convert_to_discovery_format(filtering_result.rejected_stocks)
            
            # Generate summary
            summary = self._generate_summary(filtering_result)
            
            # Create discovery result
            result = DiscoveryResult(
                selected_stocks=selected_stocks,
                rejected_stocks=rejected_stocks,
                total_processed=filtering_result.total_processed,
                stage1_passed=filtering_result.stage1_passed,
                stage2_passed=filtering_result.stage2_passed,
                final_selected=filtering_result.final_selected,
                execution_time=filtering_result.execution_time,
                config_used=self._config_to_dict(),
                summary=summary
            )
            
            # Update statistics
            self._update_statistics(result)
            
            logger.info(f"Enhanced stock discovery completed in {result.execution_time:.2f}s")
            logger.info(f"Result: {result.final_selected} stocks selected from {result.total_processed} processed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced stock discovery: {e}")
            return self._create_empty_result()
    
    def _get_stocks_from_database(self, limit: Optional[int] = None) -> List[Any]:
        """Get stocks from database."""
        try:
            # This is a simplified implementation
            # In a real implementation, you would query the database
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            logger.error(f"Error getting stocks from database: {e}")
            return []
    
    def _convert_to_discovery_format(self, stock_scores: List[StockScore]) -> List[Dict[str, Any]]:
        """Convert stock scores to discovery format."""
        result = []
        
        for score in stock_scores:
            stock_data = {
                'symbol': score.symbol,
                'scores': {
                    'technical': score.technical_score,
                    'fundamental': score.fundamental_score,
                    'risk': score.risk_score,
                    'momentum': score.momentum_score,
                    'volume': score.volume_score,
                    'total': score.total_score
                },
                'filters_passed': score.filters_passed,
                'reject_reasons': score.reject_reasons,
                'discovery_timestamp': datetime.now().isoformat()
            }
            result.append(stock_data)
        
        return result
    
    def _generate_summary(self, filtering_result: FilteringResult) -> Dict[str, Any]:
        """Generate discovery summary."""
        try:
            summary = {
                'total_processed': filtering_result.total_processed,
                'stage1_passed': filtering_result.stage1_passed,
                'stage2_passed': filtering_result.stage2_passed,
                'final_selected': filtering_result.final_selected,
                'execution_time': filtering_result.execution_time,
                'success_rate': (
                    filtering_result.final_selected / filtering_result.total_processed * 100
                    if filtering_result.total_processed > 0 else 0
                ),
                'stage1_pass_rate': (
                    filtering_result.stage1_passed / filtering_result.total_processed * 100
                    if filtering_result.total_processed > 0 else 0
                ),
                'stage2_pass_rate': (
                    filtering_result.stage2_passed / filtering_result.stage1_passed * 100
                    if filtering_result.stage1_passed > 0 else 0
                )
            }
            
            # Add top performing stocks summary
            if filtering_result.selected_stocks:
                top_stocks = sorted(
                    filtering_result.selected_stocks, 
                    key=lambda x: x.total_score, 
                    reverse=True
                )[:5]
                
                summary['top_stocks'] = [
                    {
                        'symbol': stock.symbol,
                        'total_score': stock.total_score,
                        'technical_score': stock.technical_score,
                        'fundamental_score': stock.fundamental_score,
                        'risk_score': stock.risk_score,
                        'momentum_score': stock.momentum_score,
                        'volume_score': stock.volume_score
                    }
                    for stock in top_stocks
                ]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {}
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        try:
            return {
                'universe': {
                    'exchanges': self.config.universe.exchanges,
                    'instrument_types': self.config.universe.instrument_types,
                    'min_history_days': self.config.universe.min_history_days
                },
                'stage_1_filters': {
                    'minimum_price': self.config.stage_1_filters.minimum_price,
                    'maximum_price': self.config.stage_1_filters.maximum_price,
                    'minimum_daily_turnover_inr': self.config.stage_1_filters.minimum_daily_turnover_inr,
                    'minimum_liquidity_score': self.config.stage_1_filters.minimum_liquidity_score
                },
                'scoring_weights': {
                    'technical_score': self.config.scoring_weights.technical_score,
                    'fundamental_score': self.config.scoring_weights.fundamental_score,
                    'risk_score': self.config.scoring_weights.risk_score,
                    'momentum_score': self.config.scoring_weights.momentum_score,
                    'volume_score': self.config.scoring_weights.volume_score
                },
                'selection': {
                    'max_suggested_stocks': self.config.selection.max_suggested_stocks,
                    'sector_concentration_limit_pct': self.config.selection.sector_concentration_limit_pct
                }
            }
            
        except Exception as e:
            logger.error(f"Error converting config to dict: {e}")
            return {}
    
    def _create_empty_result(self) -> DiscoveryResult:
        """Create empty discovery result."""
        return DiscoveryResult(
            selected_stocks=[],
            rejected_stocks=[],
            total_processed=0,
            stage1_passed=0,
            stage2_passed=0,
            final_selected=0,
            execution_time=0.0,
            config_used={},
            summary={}
        )
    
    def _update_statistics(self, result: DiscoveryResult):
        """Update discovery statistics."""
        try:
            self.discovery_stats['total_discoveries'] += 1
            self.discovery_stats['successful_discoveries'] += 1
            self.discovery_stats['total_stocks_processed'] += result.total_processed
            self.discovery_stats['total_stocks_selected'] += result.final_selected
            
            # Update average execution time
            total_time = self.discovery_stats['average_execution_time'] * (self.discovery_stats['total_discoveries'] - 1)
            self.discovery_stats['average_execution_time'] = (total_time + result.execution_time) / self.discovery_stats['total_discoveries']
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        return self.discovery_stats.copy()
    
    def reset_statistics(self):
        """Reset discovery statistics."""
        self.discovery_stats = {
            'total_discoveries': 0,
            'successful_discoveries': 0,
            'average_execution_time': 0.0,
            'total_stocks_processed': 0,
            'total_stocks_selected': 0
        }
    
    def get_config(self) -> EnhancedFilteringConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, new_config: EnhancedFilteringConfig):
        """Update configuration."""
        self.config = new_config
        self.filtering_service.config = new_config
        logger.info("Configuration updated successfully")


# Global service instance
_enhanced_discovery_service = None


def get_enhanced_discovery_service() -> EnhancedStockDiscoveryService:
    """Get the global enhanced discovery service instance."""
    global _enhanced_discovery_service
    if _enhanced_discovery_service is None:
        _enhanced_discovery_service = EnhancedStockDiscoveryService()
    return _enhanced_discovery_service
