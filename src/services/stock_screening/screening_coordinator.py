"""
Stock Screening Coordinator

Orchestrates the complete stock screening pipeline using modular helper classes:
- Market Data Screening: Real-time quotes and historical volatility analysis
- Business Logic Screening: Fundamental analysis and portfolio optimization

Provides centralized coordination, configuration management, and comprehensive reporting.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .simple_market_data_screener import SimpleMarketDataScreener
from .business_logic_screener import BusinessLogicScreener, StrategyType

logger = logging.getLogger(__name__)


class ScreeningCoordinator:
    """
    Central coordinator for the complete stock screening pipeline.

    Orchestrates multiple screening stages:
    1. Market Data Screening: Real-time quotes and historical volatility analysis
    2. Business Logic Screening: Fundamental analysis and portfolio optimization

    Provides:
    - Centralized configuration management
    - Comprehensive logging and reporting
    - Performance monitoring
    - Error handling and recovery
    """

    def __init__(self, fyers_service, volatility_calculator_service):
        self.fyers_service = fyers_service
        self.volatility_service = volatility_calculator_service

        # Initialize screening stages
        self.market_data_screener = SimpleMarketDataScreener(fyers_service)
        self.business_logic_screener = BusinessLogicScreener()

        # Pipeline execution tracking
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'total_execution_time': None,
            'stage_timings': {},
            'total_api_calls': 0,
            'total_stocks_processed': 0,
            'pipeline_success': False
        }

    def execute_screening_pipeline(self, user_id: int, tradeable_stocks: List,
                                 strategies: Optional[List[StrategyType]] = None) -> Dict[str, Any]:
        """
        Execute the complete stock screening pipeline.

        Args:
            user_id: User ID for FYERS API calls
            tradeable_stocks: Initial list of tradeable stocks
            strategies: List of strategy types to apply (default: [DEFAULT_RISK])

        Returns:
            Dict containing:
            - success: Boolean indicating pipeline success
            - data: List of final screened stocks
            - stats: Comprehensive pipeline statistics
            - screening_results: Individual screening stage results
        """
        self.pipeline_stats['start_time'] = datetime.now()
        self.pipeline_stats['total_stocks_processed'] = len(tradeable_stocks)

        logger.info(f"🚀 Starting complete stock screening pipeline for {len(tradeable_stocks)} stocks")

        try:
            # Print pipeline header
            self._print_pipeline_header(len(tradeable_stocks))

            # Market Data Screening
            market_data_start = datetime.now()
            market_data_candidates = self._execute_market_data_screening(user_id, tradeable_stocks)
            market_data_time = (datetime.now() - market_data_start).total_seconds()
            self.pipeline_stats['stage_timings']['market_data_screening'] = market_data_time

            if not market_data_candidates:
                logger.warning("No candidates passed market data screening - pipeline terminated")
                return self._create_empty_result("No stocks passed market data screening")

            # Business Logic Screening
            business_logic_start = datetime.now()
            final_stocks = self._execute_business_logic_screening(market_data_candidates, strategies)
            business_logic_time = (datetime.now() - business_logic_start).total_seconds()
            self.pipeline_stats['stage_timings']['business_logic_screening'] = business_logic_time

            # Pipeline completion
            self.pipeline_stats['end_time'] = datetime.now()
            self.pipeline_stats['total_execution_time'] = (
                self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            ).total_seconds()
            self.pipeline_stats['pipeline_success'] = True

            # Generate comprehensive results
            result = self._create_success_result(final_stocks)

            # Print pipeline summary
            self._print_pipeline_summary(result)

            logger.info(f"✅ Stock screening pipeline completed successfully: {len(final_stocks)} final stocks")
            return result

        except Exception as e:
            logger.error(f"❌ Stock screening pipeline failed: {e}")
            self.pipeline_stats['pipeline_success'] = False
            return self._create_error_result(str(e))

    def _execute_market_data_screening(self, user_id: int, tradeable_stocks: List) -> List:
        """Execute market data screening."""
        candidates = self.market_data_screener.screen_stocks(user_id, tradeable_stocks)
        return candidates

    def _execute_business_logic_screening(self, market_data_candidates: List, strategies: Optional[List[StrategyType]]) -> List:
        """Execute business logic screening."""
        if not strategies:
            strategies = [StrategyType.DEFAULT_RISK]

        final_stocks = self.business_logic_screener.screen_stocks(market_data_candidates, strategies)
        return final_stocks

    def _print_pipeline_header(self, total_stocks: int):
        """Print comprehensive pipeline header."""
        print(f"")
        print(f"🚀 STOCK SCREENING PIPELINE EXECUTION")
        print(f"=" * 80)
        print(f"📊 Input: {total_stocks} tradeable stocks")
        print(f"⏰ Started: {self.pipeline_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Pipeline: MarketDataScreening → BusinessLogicScreening")
        print(f"=" * 80)
        print()

    def _print_pipeline_summary(self, result: Dict[str, Any]):
        """Print comprehensive pipeline summary."""
        stats = result['stats']

        print(f"🎯 STOCK SCREENING PIPELINE EXECUTION SUMMARY")
        print(f"=" * 80)
        print(f"⏰ Total Execution Time: {self.pipeline_stats['total_execution_time']:.2f} seconds")
        print(f"📊 Total Stocks Processed: {self.pipeline_stats['total_stocks_processed']}")
        print(f"✅ Final Portfolio Size: {len(result['data'])}")
        print(f"📈 Overall Success Rate: {(len(result['data']) / self.pipeline_stats['total_stocks_processed'] * 100):.2f}%")
        print()

        print(f"⏱️  STAGE EXECUTION TIMES:")
        for stage, timing in self.pipeline_stats['stage_timings'].items():
            print(f"   {stage}: {timing:.2f} seconds")
        print()

        print(f"📊 SCREENING STAGE EFFICIENCY:")
        if 'market_data_screening' in stats:
            mds = stats['market_data_screening']
            if 'volatility_results' in mds and 'final_candidates' in mds['volatility_results']:
                candidates_mds = len(mds['volatility_results']['final_candidates'])
                total_input_mds = mds['quotes_results']['total_input'] if 'quotes_results' in mds else 0
                if total_input_mds > 0:
                    print(f"   Market Data Screening: {candidates_mds}/{total_input_mds} passed ({(candidates_mds/total_input_mds*100):.1f}%)")

        if 'business_logic_screening' in stats:
            bls = stats['business_logic_screening']
            if 'results' in bls and 'final_stocks' in bls['results']:
                final_bls = len(bls['results']['final_stocks'])
                total_input_bls = bls['results']['total_input']
                print(f"   Business Logic Screening: {final_bls}/{total_input_bls} passed ({(final_bls/total_input_bls*100):.1f}%)")

        print(f"=" * 80)

    def _create_success_result(self, final_stocks: List) -> Dict[str, Any]:
        """Create success result with comprehensive statistics."""
        return {
            'success': True,
            'data': final_stocks,
            'total': len(final_stocks),
            'pipeline_stats': self.pipeline_stats,
            'stats': {
                'market_data_screening': self.market_data_screener.get_screening_stats(),
                'business_logic_screening': self.business_logic_screener.get_filter_stats(),
                'portfolio_metrics': self.business_logic_screener.get_portfolio_metrics()
            },
            'execution_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_execution_time': self.pipeline_stats['total_execution_time'],
                'stages_executed': list(self.pipeline_stats['stage_timings'].keys()),
                'pipeline_version': '3.0'
            }
        }

    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result for early pipeline termination."""
        return {
            'success': True,
            'data': [],
            'total': 0,
            'message': reason,
            'pipeline_stats': self.pipeline_stats,
            'stats': {},
            'execution_metadata': {
                'timestamp': datetime.now().isoformat(),
                'early_termination': True,
                'termination_reason': reason
            }
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result for pipeline failure."""
        return {
            'success': False,
            'error': error_message,
            'data': [],
            'total': 0,
            'pipeline_stats': self.pipeline_stats,
            'execution_metadata': {
                'timestamp': datetime.now().isoformat(),
                'pipeline_failed': True,
                'error_stage': 'Unknown'
            }
        }

    def get_pipeline_configuration(self) -> Dict[str, Any]:
        """Get current pipeline configuration across all screening stages."""
        return {
            'market_data_screening_config': self.market_data_screener.config,
            'business_logic_screening_config': self.business_logic_screener.config,
            'strategy_criteria': {k.value: v for k, v in self.business_logic_screener.strategy_criteria.items()}
        }

    def update_pipeline_configuration(self, config_updates: Dict[str, Dict[str, Any]]):
        """Update pipeline configuration across screening stages."""
        if 'market_data_screening' in config_updates:
            self.market_data_screener.update_config(config_updates['market_data_screening'])

        if 'business_logic_screening' in config_updates:
            self.business_logic_screener.update_config(config_updates['business_logic_screening'])

        if 'strategy_criteria' in config_updates:
            for strategy_name, criteria in config_updates['strategy_criteria'].items():
                try:
                    strategy = StrategyType(strategy_name)
                    self.business_logic_screener.update_strategy_criteria(strategy, criteria)
                except ValueError:
                    logger.warning(f"Unknown strategy type: {strategy_name}")

        logger.info(f"Updated screening pipeline configuration: {list(config_updates.keys())}")

    def get_pipeline_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance metrics."""
        return {
            'execution_stats': self.pipeline_stats,
            'screening_performance': {
                'market_data_screening': self.market_data_screener.get_screening_stats() if hasattr(self.market_data_screener, 'quotes_results') else {},
                'business_logic_screening': self.business_logic_screener.get_filter_stats() if hasattr(self.business_logic_screener, 'results') else {}
            },
            'api_efficiency': {
                'quotes_api_batches': getattr(self.market_data_screener, 'quotes_results', {}).get('total_input', 0) // self.market_data_screener.config.get('batch_size', 50),
                'history_api_calls': getattr(self.market_data_screener, 'volatility_results', {}).get('total_input', 0),
                'total_api_efficiency': 'High' if self.pipeline_stats.get('pipeline_success') else 'Low'
            },
            'configuration_snapshot': self.get_pipeline_configuration()
        }