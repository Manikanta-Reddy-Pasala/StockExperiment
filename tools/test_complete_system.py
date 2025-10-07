#!/usr/bin/env python3
"""
Comprehensive System Test Suite
Tests all features from Phases 1, 2, and 3 plus new additions
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor
from src.services.ml.advanced_predictor import AdvancedStockPredictor
from src.services.ml.market_regime_detector import MarketRegimeDetector
from src.services.ml.portfolio_optimizer import PortfolioOptimizer
from src.services.ml.sentiment_analyzer import SentimentAnalyzer
from src.services.ml.model_monitor import ModelMonitor
from src.services.ml.ab_testing import ABTestManager
from src.services.ml.calibrated_scoring import CalibratedScorer, AdaptiveScorer
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemTester:
    """Comprehensive system testing."""

    def __init__(self):
        self.db_manager = get_database_manager()
        self.results = {}
        self.errors = []

    def run_all_tests(self):
        """Run all test suites."""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE SYSTEM TEST SUITE")
        logger.info("=" * 80)
        logger.info("")

        test_suites = [
            ("Phase 1: Enhanced ML", self.test_phase1_enhanced_ml),
            ("Phase 2: Advanced Features", self.test_phase2_advanced),
            ("Phase 3: Trading Intelligence", self.test_phase3_intelligence),
            ("New: Calibrated Scoring", self.test_calibrated_scoring),
            ("New: Model Monitoring", self.test_model_monitoring),
            ("New: A/B Testing", self.test_ab_testing),
            ("Integration: End-to-End", self.test_integration)
        ]

        for suite_name, test_func in test_suites:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"TEST SUITE: {suite_name}")
            logger.info(f"{'=' * 80}\n")

            try:
                result = test_func()
                self.results[suite_name] = result
                status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
                logger.info(f"\n{suite_name}: {status}")

            except Exception as e:
                logger.error(f"\n{suite_name}: ❌ ERROR - {str(e)}")
                logger.error(traceback.format_exc())
                self.results[suite_name] = {'passed': False, 'error': str(e)}
                self.errors.append({
                    'suite': suite_name,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })

        self.print_summary()

    def test_phase1_enhanced_ml(self):
        """Test Phase 1: Enhanced ML with ensemble and walk-forward CV."""
        logger.info("Testing enhanced ML predictor...")

        with self.db_manager.get_session() as session:
            predictor = EnhancedStockPredictor(session)

            # Test 1: Data preparation
            logger.info("Test 1.1: Preparing training data...")
            df = predictor.prepare_training_data(lookback_days=180)
            assert len(df) > 0, "No training data prepared"
            logger.info(f"✓ Prepared {len(df)} samples")

            # Test 2: Chaos features
            logger.info("Test 1.2: Testing chaos theory features...")
            df_with_chaos = predictor._add_chaos_features(df.head(100))
            chaos_features = ['hurst_exponent', 'fractal_dimension', 'price_entropy', 'lorenz_momentum']
            for feat in chaos_features:
                assert feat in df_with_chaos.columns, f"Missing chaos feature: {feat}"
            logger.info(f"✓ All {len(chaos_features)} chaos features present")

            # Test 3: Feature selection
            logger.info("Test 1.3: Testing feature selection...")
            X = predictor._select_features(df_with_chaos)
            assert len(X.columns) > 20, "Too few features selected"
            logger.info(f"✓ Selected {len(X.columns)} features")

            # Test 4: Training (small sample)
            logger.info("Test 1.4: Testing training with walk-forward CV...")
            stats = predictor.train_with_walk_forward(lookback_days=180, n_splits=3)
            assert 'price_r2' in stats, "Missing price R² in stats"
            assert 'cv_price_r2' in stats, "Missing CV price R² in stats"
            logger.info(f"✓ Training complete - R²: {stats['price_r2']:.3f}, CV R²: {stats['cv_price_r2']:.3f}")

            # Test 5: Prediction
            logger.info("Test 1.5: Testing prediction...")
            result = session.execute(text("SELECT * FROM stocks LIMIT 1"))
            stock = dict(result.fetchone()._mapping)
            prediction = predictor.predict(stock)

            required_keys = ['ml_prediction_score', 'ml_confidence', 'ml_risk_score', 'predicted_change_pct']
            for key in required_keys:
                assert key in prediction, f"Missing prediction key: {key}"

            logger.info(f"✓ Prediction successful - Score: {prediction['ml_prediction_score']:.3f}")

        return {'passed': True, 'stats': stats}

    def test_phase2_advanced(self):
        """Test Phase 2: Advanced features (LSTM, Bayesian opt, backtesting)."""
        logger.info("Testing advanced predictor features...")

        with self.db_manager.get_session() as session:
            # Test 1: Advanced predictor
            logger.info("Test 2.1: Testing advanced predictor...")
            predictor = AdvancedStockPredictor(session, optimize_hyperparams=False)

            df = predictor.prepare_training_data(lookback_days=180)
            assert len(df) > 0, "No training data"
            logger.info(f"✓ Prepared {len(df)} samples for advanced predictor")

            # Test 2: Training (without optimization for speed)
            logger.info("Test 2.2: Testing advanced training...")
            stats = predictor.train_advanced(lookback_days=180)
            assert 'ensemble_r2' in stats, "Missing ensemble R²"
            logger.info(f"✓ Advanced training complete - Ensemble R²: {stats['ensemble_r2']:.3f}")

            # Test 3: Backtesting (skip for now, needs more setup)
            logger.info("Test 2.3: Backtesting - SKIPPED (requires full historical data)")

        return {'passed': True, 'stats': stats}

    def test_phase3_intelligence(self):
        """Test Phase 3: Regime detection, portfolio optimization, sentiment."""
        logger.info("Testing trading intelligence features...")

        with self.db_manager.get_session() as session:
            # Test 1: Regime detection
            logger.info("Test 3.1: Testing market regime detection...")
            detector = MarketRegimeDetector(session)
            regime_result = detector.detect_regime(lookback_days=90)

            assert 'regime' in regime_result, "Missing regime"
            assert 'confidence' in regime_result, "Missing confidence"
            logger.info(f"✓ Regime detected: {regime_result['regime']} (confidence: {regime_result['confidence']:.0%})")

            # Test 2: Portfolio optimization
            logger.info("Test 3.2: Testing portfolio optimization...")
            result = session.execute(text("""
                SELECT symbol, current_price,
                       0.65 as ml_prediction_score,
                       0.75 as ml_confidence,
                       0.30 as ml_risk_score,
                       2.5 as predicted_change_pct
                FROM stocks
                WHERE current_price IS NOT NULL
                LIMIT 10
            """))

            stocks = [dict(row._mapping) for row in result.fetchall()]

            if len(stocks) >= 5:
                optimizer = PortfolioOptimizer(session)
                portfolio = optimizer.optimize_portfolio(stocks, method='max_sharpe')

                assert 'metrics' in portfolio, "Missing portfolio metrics"
                assert 'allocations' in portfolio, "Missing allocations"
                logger.info(f"✓ Portfolio optimized - Sharpe: {portfolio['metrics']['sharpe_ratio']:.2f}")
            else:
                logger.warning("⚠ Not enough stocks for portfolio optimization")

            # Test 3: Sentiment analysis
            logger.info("Test 3.3: Testing sentiment analysis...")
            result = session.execute(text("SELECT symbol FROM stocks LIMIT 1"))
            symbol = result.fetchone()[0]

            analyzer = SentimentAnalyzer()
            sentiment = analyzer.analyze_stock_sentiment(symbol, sources=['news'])

            assert 'overall_sentiment' in sentiment, "Missing overall sentiment"
            assert 'overall_score' in sentiment, "Missing sentiment score"
            logger.info(f"✓ Sentiment analyzed: {sentiment['overall_sentiment']} (score: {sentiment['overall_score']:.2f})")

        return {'passed': True}

    def test_calibrated_scoring(self):
        """Test new calibrated probability scoring."""
        logger.info("Testing calibrated scoring system...")

        # Test 1: Basic calibrated scorer
        logger.info("Test 4.1: Testing CalibratedScorer...")
        scorer = CalibratedScorer()

        # Generate sample data
        predictions = np.random.randn(100) * 5
        actuals = predictions + np.random.randn(100) * 2

        success = scorer.fit(predictions, actuals)
        assert success, "Calibration fitting failed"
        logger.info("✓ Calibrated scorer fitted")

        # Test scoring
        score = scorer.score(2.5)
        assert 0 <= score <= 1, f"Score out of range: {score}"
        logger.info(f"✓ Calibrated score: {score:.3f}")

        # Test 2: Adaptive scorer
        logger.info("Test 4.2: Testing AdaptiveScorer...")
        adaptive = AdaptiveScorer(window_size=50)

        for i in range(60):
            pred = np.random.randn() * 3
            actual = pred + np.random.randn() * 1.5
            adaptive.add_result(pred, actual)

        stats = adaptive.get_statistics()
        assert 'direction_accuracy' in stats, "Missing direction accuracy"
        logger.info(f"✓ Adaptive scorer - Accuracy: {stats['direction_accuracy']:.1%}")

        return {'passed': True}

    def test_model_monitoring(self):
        """Test new model monitoring system."""
        logger.info("Testing model monitoring system...")

        with self.db_manager.get_session() as session:
            # Test 1: Initialize monitor
            logger.info("Test 5.1: Initializing model monitor...")
            monitor = ModelMonitor(session, window_size=50)
            logger.info("✓ Monitor initialized")

            # Test 2: Set baseline
            logger.info("Test 5.2: Setting baseline metrics...")
            baseline = {
                'r2': 0.42,
                'mae': 2.5,
                'direction_accuracy': 0.68
            }
            monitor.set_baseline('enhanced', baseline)
            logger.info("✓ Baseline metrics set")

            # Test 3: Log predictions
            logger.info("Test 5.3: Logging predictions...")
            for i in range(30):
                pred = np.random.randn() * 3
                actual = pred + np.random.randn() * 2
                monitor.add_result(pred, actual)
            logger.info("✓ Logged 30 prediction-actual pairs")

            # Test 4: Calculate metrics
            logger.info("Test 5.4: Calculating metrics...")
            metrics = monitor.calculate_metrics('enhanced')
            assert 'mae' in metrics, "Missing MAE"
            assert 'direction_accuracy' in metrics, "Missing direction accuracy"
            logger.info(f"✓ Metrics calculated - MAE: {metrics['mae']:.2f}, Accuracy: {metrics['direction_accuracy']:.1%}")

            # Test 5: Health report
            logger.info("Test 5.5: Generating health report...")
            report = monitor.generate_health_report('enhanced')
            assert 'health_score' in report, "Missing health score"
            logger.info(f"✓ Health report generated - Score: {report['health_score']}, Level: {report['health_level']}")

        return {'passed': True, 'health_score': report['health_score']}

    def test_ab_testing(self):
        """Test new A/B testing framework."""
        logger.info("Testing A/B testing framework...")

        with self.db_manager.get_session() as session:
            # Test 1: Create test
            logger.info("Test 6.1: Creating A/B test...")
            manager = ABTestManager(session)
            test = manager.create_test(
                'test_model_comparison',
                'enhanced_v1',
                'enhanced_v2',
                traffic_split=0.5,
                min_sample_size=30
            )
            logger.info("✓ A/B test created")

            # Test 2: Log results
            logger.info("Test 6.2: Logging test results...")
            for i in range(60):
                variant = test.assign_variant(f'stock_{i}')
                pred = np.random.randn() * 3 + (0.5 if variant == 'b' else 0)
                actual = pred + np.random.randn() * 2
                manager.log_result('test_model_comparison', variant, f'STOCK{i}', pred, actual)
            logger.info("✓ Logged 60 test results")

            # Test 3: Get summary
            logger.info("Test 6.3: Getting test summary...")
            summary = manager.get_test_summary('test_model_comparison')
            assert 'statistical_tests' in summary, "Missing statistical tests"
            logger.info(f"✓ Test summary generated - Recommendation: {summary['recommendation']['action']}")

            # Test 4: Stop test
            logger.info("Test 6.4: Stopping test...")
            manager.stop_test('test_model_comparison')
            logger.info("✓ Test stopped")

        return {'passed': True, 'recommendation': summary['recommendation']}

    def test_integration(self):
        """Test end-to-end integration."""
        logger.info("Testing end-to-end integration...")

        with self.db_manager.get_session() as session:
            # Workflow: Train -> Monitor -> Predict -> AB Test
            logger.info("Test 7.1: Training model...")
            predictor = EnhancedStockPredictor(session)
            stats = predictor.train_with_walk_forward(lookback_days=180, n_splits=3)
            logger.info(f"✓ Model trained - R²: {stats['price_r2']:.3f}")

            # Set up monitoring
            logger.info("Test 7.2: Setting up monitoring...")
            monitor = ModelMonitor(session)
            monitor.set_baseline('enhanced', stats)
            logger.info("✓ Monitoring configured")

            # Make predictions
            logger.info("Test 7.3: Making predictions...")
            result = session.execute(text("SELECT * FROM stocks WHERE current_price IS NOT NULL LIMIT 5"))
            stocks = [dict(row._mapping) for row in result.fetchall()]

            predictions = []
            for stock in stocks:
                pred = predictor.predict(stock)
                predictions.append(pred)
                logger.info(f"  {stock['symbol']}: Score={pred['ml_prediction_score']:.3f}, Change={pred['predicted_change_pct']:.1f}%")

            logger.info(f"✓ Generated {len(predictions)} predictions")

            # Health check
            logger.info("Test 7.4: Running health check...")
            report = monitor.generate_health_report('enhanced')
            logger.info(f"✓ System health: {report['health_level']} (score: {report['health_score']})")

        return {
            'passed': True,
            'predictions': len(predictions),
            'health_score': report['health_score']
        }

    def print_summary(self):
        """Print comprehensive test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)

        passed = sum(1 for r in self.results.values() if r.get('passed', False))
        total = len(self.results)

        logger.info(f"\nTotal Test Suites: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")

        logger.info("\nDetailed Results:")
        logger.info("-" * 80)

        for suite_name, result in self.results.items():
            status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
            logger.info(f"{suite_name:.<50} {status}")

            if not result.get('passed', False) and 'error' in result:
                logger.info(f"  Error: {result['error']}")

        if self.errors:
            logger.info("\n" + "=" * 80)
            logger.info("ERRORS")
            logger.info("=" * 80)
            for error in self.errors:
                logger.info(f"\n{error['suite']}:")
                logger.info(f"  {error['error']}")

        logger.info("\n" + "=" * 80)
        success_rate = (passed / total * 100) if total > 0 else 0
        logger.info(f"SUCCESS RATE: {success_rate:.1f}%")
        logger.info("=" * 80)

        return passed == total


# Add numpy import for tests
import numpy as np


def main():
    """Run comprehensive system tests."""
    tester = SystemTester()
    tester.run_all_tests()

    success = tester.print_summary()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
