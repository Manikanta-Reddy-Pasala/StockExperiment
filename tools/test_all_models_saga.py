#!/usr/bin/env python3
"""
Unified Test Script for All 3 ML Models × 2 Strategies
Tests the complete saga pipeline for Traditional ML, Raw LSTM, and Kronos
with both DEFAULT_RISK and HIGH_RISK strategies.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.data.suggested_stocks_saga import get_suggested_stocks_saga_orchestrator
from src.models.database import get_database_manager
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedModelTester:
    """Test all 3 models with both strategies through saga pipeline."""

    def __init__(self):
        self.orchestrator = get_suggested_stocks_saga_orchestrator()
        self.db_manager = get_database_manager()
        self.test_results = []

        # Model configurations
        self.models = {
            'traditional': {
                'name': 'Traditional ML',
                'description': 'Random Forest + XGBoost ensemble',
                'icon': '🎯'
            },
            'raw_lstm': {
                'name': 'Raw LSTM',
                'description': 'Deep learning with OHLCV sequences',
                'icon': '🧠'
            },
            'kronos': {
                'name': 'Kronos',
                'description': 'K-line tokenization patterns',
                'icon': '🔮'
            }
        }

        # Strategy configurations
        self.strategies = {
            'default_risk': {
                'name': 'DEFAULT_RISK',
                'description': 'Conservative (Large-cap >20K Cr)',
                'icon': '🛡️'
            },
            'high_risk': {
                'name': 'HIGH_RISK',
                'description': 'Aggressive (Small/Mid-cap 1K-20K Cr)',
                'icon': '⚡'
            }
        }

    def print_header(self):
        """Print test header."""
        print("\n" + "=" * 100)
        print("UNIFIED MODEL TESTING - ALL 3 MODELS × 2 STRATEGIES")
        print("=" * 100)
        print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Combinations: {len(self.models)} models × {len(self.strategies)} strategies = 6 tests")
        print("=" * 100)
        print("\nModels:")
        for model_type, config in self.models.items():
            print(f"  {config['icon']} {config['name']}: {config['description']}")
        print("\nStrategies:")
        for strategy_key, config in self.strategies.items():
            print(f"  {config['icon']} {config['name']}: {config['description']}")
        print("=" * 100 + "\n")

    def test_single_combination(self, model_type: str, strategy: str, limit: int = 10) -> Dict[str, Any]:
        """Test a single model + strategy combination."""
        model_config = self.models[model_type]
        strategy_config = self.strategies[strategy]

        print(f"\n{'─' * 100}")
        print(f"{model_config['icon']} Testing: {model_config['name']} × {strategy_config['icon']} {strategy_config['name']}")
        print(f"{'─' * 100}")

        test_result = {
            'model_type': model_type,
            'model_name': model_config['name'],
            'strategy': strategy,
            'strategy_name': strategy_config['name'],
            'status': 'pending',
            'start_time': datetime.now(),
            'end_time': None,
            'duration_seconds': 0,
            'stocks_count': 0,
            'error': None,
            'saga_summary': None
        }

        try:
            # Execute saga for this model + strategy
            logger.info(f"Executing saga: model={model_type}, strategy={strategy}")

            result = self.orchestrator.execute_suggested_stocks_saga(
                user_id=1,
                strategies=[strategy],
                limit=limit,
                model_type=model_type
            )

            test_result['end_time'] = datetime.now()
            test_result['duration_seconds'] = (test_result['end_time'] - test_result['start_time']).total_seconds()
            test_result['status'] = result.get('status', 'unknown')
            test_result['saga_summary'] = result.get('summary', {})

            if result['status'] == 'completed':
                stocks_count = result['summary'].get('final_result_count', 0)
                test_result['stocks_count'] = stocks_count

                # Print saga steps
                print(f"\n✅ Saga completed successfully in {test_result['duration_seconds']:.1f}s")
                print(f"   Stocks selected: {stocks_count}")

                # Print step details
                print("\n   Saga Steps:")
                for step in result['summary']['step_summary']:
                    status_icon = '✅' if step['status'] == 'completed' else '❌'
                    print(f"     {status_icon} {step['name']}: "
                          f"{step['output_count']} out, "
                          f"{step['duration_seconds']:.2f}s")

                # Print ML prediction details
                ml_step = next((s for s in result['summary']['step_summary']
                              if s['step_id'] == 'step6_ml_prediction'), None)
                if ml_step and ml_step['status'] == 'completed':
                    metadata = ml_step.get('metadata', {})
                    print(f"\n   ML Predictions:")
                    print(f"     Avg Score: {metadata.get('avg_prediction_score', 0):.3f}")
                    print(f"     Avg Confidence: {metadata.get('avg_confidence', 0):.3f}")
                    print(f"     Avg Risk: {metadata.get('avg_risk_score', 0):.3f}")

                # Query database to verify save
                with self.db_manager.get_session() as session:
                    count_query = text("""
                        SELECT COUNT(*)
                        FROM daily_suggested_stocks
                        WHERE model_type = :model_type
                        AND strategy = :strategy
                        AND date = CURRENT_DATE
                    """)
                    db_count = session.execute(count_query, {
                        'model_type': model_type,
                        'strategy': strategy
                    }).scalar()

                    print(f"\n   Database Verification:")
                    print(f"     Records in DB: {db_count} (model_type='{model_type}', strategy='{strategy}')")

                    if db_count > 0:
                        # Get sample stock
                        sample_query = text("""
                            SELECT symbol, stock_name, current_price, ml_prediction_score,
                                   recommendation, model_type, strategy
                            FROM daily_suggested_stocks
                            WHERE model_type = :model_type
                            AND strategy = :strategy
                            AND date = CURRENT_DATE
                            ORDER BY ml_prediction_score DESC
                            LIMIT 3
                        """)
                        samples = session.execute(sample_query, {
                            'model_type': model_type,
                            'strategy': strategy
                        }).fetchall()

                        print(f"\n   Top 3 Stocks:")
                        for idx, row in enumerate(samples, 1):
                            print(f"     {idx}. {row.symbol} - {row.stock_name}")
                            print(f"        Price: ₹{row.current_price:.2f}, "
                                  f"ML Score: {row.ml_prediction_score:.3f}, "
                                  f"Signal: {row.recommendation}")

                logger.info(f"✅ Test passed: {model_type} × {strategy}")

            else:
                test_result['error'] = f"Saga failed with status: {result['status']}"
                errors = result.get('errors', [])
                if errors:
                    test_result['error'] += f" - Errors: {', '.join(errors)}"
                print(f"\n❌ Saga failed: {test_result['error']}")
                logger.error(f"❌ Test failed: {model_type} × {strategy} - {test_result['error']}")

        except Exception as e:
            test_result['end_time'] = datetime.now()
            test_result['duration_seconds'] = (test_result['end_time'] - test_result['start_time']).total_seconds()
            test_result['status'] = 'error'
            test_result['error'] = str(e)
            print(f"\n❌ Exception occurred: {e}")
            logger.error(f"❌ Test error: {model_type} × {strategy} - {e}", exc_info=True)

        return test_result

    def test_all_combinations(self, limit: int = 10):
        """Test all model + strategy combinations."""
        self.print_header()

        total_tests = len(self.models) * len(self.strategies)
        current_test = 0

        # Test all combinations
        for model_type in self.models.keys():
            for strategy_key in self.strategies.keys():
                current_test += 1
                print(f"\n[Test {current_test}/{total_tests}]")

                result = self.test_single_combination(model_type, strategy_key, limit)
                self.test_results.append(result)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        print("\n\n" + "=" * 100)
        print("TEST SUMMARY")
        print("=" * 100)

        passed = len([r for r in self.test_results if r['status'] == 'completed'])
        failed = len([r for r in self.test_results if r['status'] != 'completed'])
        total_stocks = sum(r['stocks_count'] for r in self.test_results)
        total_duration = sum(r['duration_seconds'] for r in self.test_results)

        print(f"\nOverall Results:")
        print(f"  ✅ Passed: {passed}/{len(self.test_results)}")
        print(f"  ❌ Failed: {failed}/{len(self.test_results)}")
        print(f"  📊 Total Stocks: {total_stocks}")
        print(f"  ⏱️  Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")

        print(f"\nDetailed Results:")
        print(f"{'Model':<20} {'Strategy':<20} {'Status':<12} {'Stocks':<10} {'Duration':<12}")
        print("─" * 100)

        for result in self.test_results:
            status_icon = '✅' if result['status'] == 'completed' else '❌'
            print(f"{result['model_name']:<20} "
                  f"{result['strategy_name']:<20} "
                  f"{status_icon} {result['status']:<10} "
                  f"{result['stocks_count']:<10} "
                  f"{result['duration_seconds']:.1f}s")

            if result['error']:
                print(f"  Error: {result['error']}")

        print("=" * 100)

        # Database summary
        print(f"\nDatabase Summary (CURRENT_DATE):")
        with self.db_manager.get_session() as session:
            summary_query = text("""
                SELECT
                    model_type,
                    strategy,
                    COUNT(*) as count,
                    AVG(ml_prediction_score) as avg_score,
                    AVG(ml_confidence) as avg_confidence
                FROM daily_suggested_stocks
                WHERE date = CURRENT_DATE
                GROUP BY model_type, strategy
                ORDER BY model_type, strategy
            """)

            results = session.execute(summary_query).fetchall()

            if results:
                print(f"{'Model':<20} {'Strategy':<20} {'Count':<10} {'Avg Score':<12} {'Avg Confidence':<12}")
                print("─" * 100)
                for row in results:
                    print(f"{row.model_type:<20} "
                          f"{row.strategy:<20} "
                          f"{row.count:<10} "
                          f"{row.avg_score:.3f}{' ' * 8} "
                          f"{row.avg_confidence:.3f}")
            else:
                print("  (No records found for today)")

        print("=" * 100 + "\n")

        # Final verdict
        if failed == 0:
            print("🎉 ALL TESTS PASSED! All 3 models × 2 strategies working correctly via saga!")
        else:
            print(f"⚠️  {failed} TEST(S) FAILED. Please check the errors above.")


def main():
    """Main test execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Test all ML models with saga pipeline')
    parser.add_argument('--limit', type=int, default=10,
                       help='Maximum stocks per combination (default: 10)')
    parser.add_argument('--model', type=str, choices=['traditional', 'raw_lstm', 'kronos'],
                       help='Test only specific model (default: all)')
    parser.add_argument('--strategy', type=str, choices=['default_risk', 'high_risk'],
                       help='Test only specific strategy (default: both)')

    args = parser.parse_args()

    tester = UnifiedModelTester()

    # Filter models and strategies if specified
    if args.model:
        tester.models = {k: v for k, v in tester.models.items() if k == args.model}

    if args.strategy:
        tester.strategies = {k: v for k, v in tester.strategies.items() if k == args.strategy}

    # Run tests
    tester.test_all_combinations(limit=args.limit)


if __name__ == '__main__':
    main()
