#!/usr/bin/env python3
"""
Quick System Validation Test
Tests basic functionality without full training
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor
from src.services.ml.market_regime_detector import MarketRegimeDetector
from src.services.ml.portfolio_optimizer import PortfolioOptimizer
from src.services.ml.sentiment_analyzer import SentimentAnalyzer
from src.services.ml.model_monitor import ModelMonitor
from src.services.ml.ab_testing import ABTestManager
from src.services.ml.calibrated_scoring import CalibratedScorer, AdaptiveScorer
import numpy as np
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_calibrated_scoring():
    """Test calibrated scoring."""
    logger.info("Testing Calibrated Scoring...")

    # Test CalibratedScorer
    scorer = CalibratedScorer()
    predictions = np.random.randn(100) * 5
    actuals = predictions + np.random.randn(100) * 2

    success = scorer.fit(predictions, actuals)
    assert success, "Calibration failed"

    score = scorer.score(2.5)
    assert 0 <= score <= 1, f"Score out of range: {score}"
    logger.info(f"✓ CalibratedScorer works - Score: {score:.3f}")

    # Test AdaptiveScorer
    adaptive = AdaptiveScorer()
    for i in range(60):
        adaptive.add_result(np.random.randn() * 3, np.random.randn() * 3)

    stats = adaptive.get_statistics()
    assert 'direction_accuracy' in stats
    logger.info(f"✓ AdaptiveScorer works - Accuracy: {stats['direction_accuracy']:.1%}")

def test_model_monitoring():
    """Test model monitoring."""
    logger.info("\nTesting Model Monitoring...")

    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        monitor = ModelMonitor(session)

        # Set baseline
        monitor.set_baseline('test_model', {
            'r2': 0.42,
            'mae': 2.5,
            'direction_accuracy': 0.68
        })
        logger.info("✓ Baseline set")

        # Log predictions
        for i in range(30):
            monitor.add_result(np.random.randn() * 3, np.random.randn() * 3)
        logger.info("✓ Logged 30 predictions")

        # Calculate metrics
        metrics = monitor.calculate_metrics('test_model')
        assert 'mae' in metrics
        logger.info(f"✓ Metrics calculated - MAE: {metrics['mae']:.2f}")

        # Health report
        report = monitor.generate_health_report('test_model')
        logger.info(f"✓ Health report - Score: {report['health_score']}, Level: {report['health_level']}")

def test_ab_testing():
    """Test A/B testing."""
    logger.info("\nTesting A/B Testing...")

    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        manager = ABTestManager(session)

        # Create test
        test = manager.create_test('quick_test', 'model_a', 'model_b')
        logger.info("✓ A/B test created")

        # Log results
        for i in range(60):
            variant = test.assign_variant(f'item_{i}')
            pred = np.random.randn() * 3
            actual = pred + np.random.randn() * 2
            manager.log_result('quick_test', variant, f'STOCK{i}', pred, actual)
        logger.info("✓ Logged 60 test results")

        # Get summary
        summary = manager.get_test_summary('quick_test')
        logger.info(f"✓ Test summary - Recommendation: {summary['recommendation']['action']}")

def test_regime_detection():
    """Test regime detection."""
    logger.info("\nTesting Regime Detection...")

    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        detector = MarketRegimeDetector(session)
        result = detector.detect_regime(lookback_days=90)

        assert 'regime' in result
        logger.info(f"✓ Regime detected: {result['regime']} (confidence: {result['confidence']:.0%})")

def test_portfolio_optimization():
    """Test portfolio optimization."""
    logger.info("\nTesting Portfolio Optimization...")

    db_manager = get_database_manager()
    with db_manager.get_session() as session:
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
            logger.info(f"✓ Portfolio optimized - Sharpe: {portfolio['metrics']['sharpe_ratio']:.2f}")
        else:
            logger.warning("⚠ Not enough stocks for optimization")

def test_sentiment():
    """Test sentiment analysis."""
    logger.info("\nTesting Sentiment Analysis...")

    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        result = session.execute(text("SELECT symbol FROM stocks LIMIT 1"))
        symbol = result.fetchone()[0]

        analyzer = SentimentAnalyzer()
        sentiment = analyzer.analyze_stock_sentiment(symbol, sources=['news'])

        assert 'overall_sentiment' in sentiment
        logger.info(f"✓ Sentiment: {sentiment['overall_sentiment']} (score: {sentiment['overall_score']:.2f})")

def test_data_preparation():
    """Test data preparation."""
    logger.info("\nTesting Data Preparation...")

    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        predictor = EnhancedStockPredictor(session)
        df = predictor.prepare_training_data(lookback_days=90)

        assert len(df) > 0, "No training data"
        logger.info(f"✓ Prepared {len(df)} training samples")

        # Test chaos features
        df_chaos = predictor._add_chaos_features(df.head(50))
        assert 'hurst_exponent' in df_chaos.columns
        logger.info("✓ Chaos features added")

def main():
    """Run quick tests."""
    logger.info("=" * 80)
    logger.info("QUICK SYSTEM VALIDATION TEST")
    logger.info("=" * 80)

    tests = [
        ("Calibrated Scoring", test_calibrated_scoring),
        ("Model Monitoring", test_model_monitoring),
        ("A/B Testing", test_ab_testing),
        ("Regime Detection", test_regime_detection),
        ("Portfolio Optimization", test_portfolio_optimization),
        ("Sentiment Analysis", test_sentiment),
        ("Data Preparation", test_data_preparation)
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"\n❌ {name} FAILED: {e}")
            failed += 1

    logger.info("\n" + "=" * 80)
    logger.info(f"RESULTS: {passed}/{len(tests)} passed")
    logger.info("=" * 80)

    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
