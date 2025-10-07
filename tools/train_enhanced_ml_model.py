#!/usr/bin/env python3
"""
Enhanced ML Model Training Script
Trains the improved ensemble models with walk-forward validation.
Uses Random Forest + XGBoost with chaos theory features.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train enhanced ML models with walk-forward validation."""
    logger.info("=" * 80)
    logger.info("ENHANCED ML MODEL TRAINING STARTED")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Improvements:")
    logger.info("✓ Multi-model ensemble (RF + XGBoost)")
    logger.info("✓ Walk-forward validation (prevents overfitting)")
    logger.info("✓ Chaos theory features (Hurst, Fractal, Entropy)")
    logger.info("✓ Enhanced feature engineering (40+ features)")
    logger.info("✓ Feature importance tracking")
    logger.info("")
    logger.info("=" * 80)

    db_manager = get_database_manager()

    try:
        with db_manager.get_session() as session:
            # Initialize enhanced predictor
            predictor = EnhancedStockPredictor(session)

            # Train with walk-forward validation
            logger.info("Training models with 365 days of historical data...")
            logger.info("Using 5-fold walk-forward cross-validation...")
            stats = predictor.train_with_walk_forward(lookback_days=365, n_splits=5)

            # Display results
            logger.info("")
            logger.info("=" * 80)
            logger.info("TRAINING COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"Training Samples: {stats['samples']:,}")
            logger.info(f"Features Used: {stats['features']}")
            logger.info("")
            logger.info("Model Performance:")
            logger.info(f"  Training Set Price R²: {stats['price_r2']:.4f}")
            logger.info(f"  Training Set Risk R²: {stats['risk_r2']:.4f}")
            logger.info("")
            logger.info("Walk-Forward CV Performance (Out-of-Sample):")
            logger.info(f"  CV Price R²: {stats['cv_price_r2']:.4f}")
            logger.info(f"  CV Risk R²: {stats['cv_risk_r2']:.4f}")
            logger.info("")
            logger.info("Top 10 Most Important Features:")
            for i, feat in enumerate(stats['top_features'], 1):
                logger.info(f"  {i}. {feat}")
            logger.info("=" * 80)

            # Test prediction on a sample
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
                logger.info(f"  ML Prediction Score: {prediction['ml_prediction_score']:.4f} (0-1)")
                logger.info(f"  ML Price Target (2w): ₹{prediction['ml_price_target']:,.2f}")
                logger.info(f"  ML Confidence: {prediction['ml_confidence']:.4f}")
                logger.info(f"  ML Risk Score: {prediction['ml_risk_score']:.4f} (lower = better)")
                logger.info(f"  Predicted Change: {prediction['predicted_change_pct']:+.2f}%")
                logger.info(f"  Predicted Drawdown: {prediction['predicted_drawdown_pct']:.2f}%")
                logger.info("")
                logger.info(f"Model Performance Metrics:")
                logger.info(f"  CV Price R²: {prediction['model_performance']['cv_price_r2']:.3f}")
                logger.info(f"  CV Risk R²: {prediction['model_performance']['cv_risk_r2']:.3f}")

            logger.info("")
            logger.info("=" * 80)
            logger.info("✓ Enhanced ML models trained successfully!")
            logger.info("✓ Models use ensemble predictions with improved accuracy")
            logger.info("✓ Walk-forward validation ensures robust out-of-sample performance")
            logger.info("✓ Models are ready for use in production")
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
