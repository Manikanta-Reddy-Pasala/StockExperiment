#!/usr/bin/env python3
"""
Advanced ML Training Script - Phase 2
Trains models with LSTM, Bayesian optimization, and advanced features
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.ml.advanced_predictor import AdvancedStockPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train advanced ML models with all Phase 2 features."""
    parser = argparse.ArgumentParser(description='Train advanced ML models (Phase 2)')
    parser.add_argument('--optimize', action='store_true',
                       help='Enable Bayesian hyperparameter optimization')
    parser.add_argument('--lstm-lookback', type=int, default=20,
                       help='LSTM lookback window (default: 20)')
    parser.add_argument('--lookback-days', type=int, default=365,
                       help='Historical data lookback in days (default: 365)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PHASE 2: ADVANCED ML MODEL TRAINING")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Features Enabled:")
    logger.info("✓ Multi-model ensemble (RF + XGBoost + LSTM)")
    logger.info("✓ Chaos theory features (Hurst, Entropy)")
    logger.info("✓ Enhanced feature engineering (40+ features)")
    if args.optimize:
        logger.info("✓ Bayesian hyperparameter optimization (Optuna)")
    else:
        logger.info("  (Bayesian optimization disabled - use --optimize to enable)")
    logger.info(f"✓ LSTM lookback window: {args.lstm_lookback}")
    logger.info("")
    logger.info("=" * 80)

    db_manager = get_database_manager()

    try:
        with db_manager.get_session() as session:
            # Initialize advanced predictor
            predictor = AdvancedStockPredictor(
                session,
                lstm_lookback=args.lstm_lookback,
                optimize_hyperparams=args.optimize
            )

            # Train models
            logger.info(f"Training with {args.lookback_days} days of historical data...")
            stats = predictor.train_advanced(lookback_days=args.lookback_days)

            # Display results
            logger.info("")
            logger.info("=" * 80)
            logger.info("TRAINING COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"Training Samples: {stats['samples']:,}")
            logger.info(f"Features Used: {stats['features']}")
            logger.info(f"Models Trained: {', '.join(stats['models']).upper()}")
            logger.info(f"Hyperparameter Optimization: {'Enabled' if stats['optimized'] else 'Disabled'}")
            logger.info("")
            logger.info("Top 10 Most Important Features:")
            for i, feat in enumerate(stats['top_features'][:10], 1):
                logger.info(f"  {i:2d}. {feat}")
            logger.info("=" * 80)

            # Test prediction
            logger.info("")
            logger.info("Testing prediction on sample stock...")

            from sqlalchemy import text
            result = session.execute(text("""
                SELECT
                    s.symbol, s.current_price, s.market_cap,
                    s.pe_ratio, s.pb_ratio, s.roe, s.eps, s.beta,
                    s.debt_to_equity, s.revenue_growth, s.earnings_growth,
                    s.operating_margin, s.net_margin, s.historical_volatility_1y, s.atr_14,
                    ti.rsi_14, ti.macd, ti.macd_signal, ti.macd_histogram,
                    ti.sma_50, ti.sma_200, ti.ema_12, ti.ema_26, ti.atr_percentage,
                    ti.bollinger_upper, ti.bollinger_lower
                FROM stocks s
                LEFT JOIN (
                    SELECT DISTINCT ON (symbol) *
                    FROM technical_indicators
                    ORDER BY symbol, date DESC
                ) ti ON s.symbol = ti.symbol
                WHERE s.current_price IS NOT NULL
                AND s.market_cap IS NOT NULL
                ORDER BY s.market_cap DESC
                LIMIT 1
            """))

            row = result.fetchone()
            if row:
                stock_data = dict(row._mapping)
                prediction = predictor.predict(stock_data)

                logger.info("")
                logger.info(f"Sample Prediction for {stock_data['symbol']}:")
                logger.info(f"  Current Price: ₹{stock_data['current_price']:,.2f}")
                logger.info(f"  ML Prediction Score: {prediction['ml_prediction_score']:.4f}")
                logger.info(f"  ML Price Target: ₹{prediction['ml_price_target']:,.2f}")
                logger.info(f"  ML Confidence: {prediction['ml_confidence']:.4f}")
                logger.info(f"  ML Risk Score: {prediction['ml_risk_score']:.4f}")
                logger.info(f"  Predicted Change: {prediction['predicted_change_pct']:+.2f}%")
                logger.info(f"  Models Used: {prediction['models_used']}")

            logger.info("")
            logger.info("=" * 80)
            logger.info("✓ Advanced ML models trained successfully!")
            logger.info("✓ Models include LSTM for sequential patterns")
            logger.info("✓ Ready for production use")
            logger.info("=" * 80)

            return 0

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("TRAINING FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
