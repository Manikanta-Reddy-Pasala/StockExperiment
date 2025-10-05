#!/usr/bin/env python3
"""
ML Model Training Script
Trains the Random Forest models for stock prediction.
Run this periodically to update ML models with latest data.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.database import get_database_manager
from src.services.ml.stock_predictor import StockMLPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train ML models with historical data."""
    logger.info("=" * 80)
    logger.info("ML Model Training Started")
    logger.info("=" * 80)

    # Initialize database and train models
    db_manager = get_database_manager()

    try:
        with db_manager.get_session() as session:
            # Initialize predictor
            predictor = StockMLPredictor(session)

            # Train models (using 1 year of historical data)
            logger.info("Training models with 365 days of historical data...")
            stats = predictor.train(lookback_days=365)

            # Display results
            logger.info("=" * 80)
            logger.info("Training Complete!")
            logger.info("=" * 80)
            logger.info(f"Training Samples: {stats['samples']:,}")
            logger.info(f"Features Used: {stats['features']}")
            logger.info(f"Price Model R² Score: {stats['price_r2']:.4f}")
            logger.info(f"Risk Model R² Score: {stats['risk_r2']:.4f}")
            logger.info("=" * 80)

            # Test prediction on a sample
            logger.info("\nTesting prediction on sample stock...")

            from sqlalchemy import text
            result = session.execute(text("""
            SELECT 
                s.symbol, s.current_price, s.market_cap,
                s.pe_ratio, s.pb_ratio, s.roe, s.eps, s.beta,
                s.debt_to_equity, s.revenue_growth, s.earnings_growth,
                s.operating_margin, s.net_margin, s.historical_volatility_1y, s.atr_14,
                ti.rsi_14, ti.macd, ti.macd_signal, ti.macd_histogram,
                ti.sma_50, ti.sma_200, ti.ema_12, ti.ema_26, ti.atr_percentage
            FROM stocks s
            LEFT JOIN (
                SELECT DISTINCT ON (symbol) *
                FROM technical_indicators
                ORDER BY symbol, date DESC
            ) ti ON s.symbol = ti.symbol
            WHERE s.current_price IS NOT NULL
            AND s.market_cap IS NOT NULL
            LIMIT 1
        """))
        
            row = result.fetchone()
            if row:
                stock_data = dict(row._mapping)
                prediction = predictor.predict(stock_data)

                logger.info(f"\nSample Prediction for {stock_data['symbol']}:")
                logger.info(f"  Current Price: ₹{stock_data['current_price']:,.2f}")
                logger.info(f"  ML Prediction Score: {prediction['ml_prediction_score']:.4f} (0-1)")
                logger.info(f"  ML Price Target (2w): ₹{prediction['ml_price_target']:,.2f}")
                logger.info(f"  ML Confidence: {prediction['ml_confidence']:.4f}")
                logger.info(f"  ML Risk Score: {prediction['ml_risk_score']:.4f} (lower = better)")
                logger.info(f"  Predicted Change: {prediction['predicted_change_pct']:+.2f}%")
                logger.info(f"  Predicted Drawdown: {prediction['predicted_drawdown_pct']:.2f}%")

            logger.info("\n✓ ML models trained successfully!")
            logger.info("✓ Models are ready for use in suggested stocks endpoint")

            return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
