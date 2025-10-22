#!/usr/bin/env python3
"""
Generate ML predictions for ALL stocks (not just filtered subset)
This runs all 3 models on all ~2259 stocks and stores predictions in a separate table.
"""
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from sqlalchemy import text
from src.models.database import get_database_manager
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor
from src.services.ml.raw_lstm_prediction_service import RawLSTMPredictionService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_traditional_ml_all_stocks():
    """Generate Traditional ML predictions for ALL stocks."""
    logger.info("\n" + "="*80)
    logger.info("MODEL 1: TRADITIONAL ML (RF + XGBoost) - ALL STOCKS")
    logger.info("="*80)

    try:
        db_manager = get_database_manager()

        with db_manager.get_session() as session:
            # Get ALL stocks with basic data requirements
            query = text("""
                SELECT DISTINCT s.symbol, s.name, s.current_price
                FROM stocks s
                WHERE s.current_price IS NOT NULL
                  AND s.current_price > 0
                ORDER BY s.symbol
            """)

            result = session.execute(query)
            all_stocks = [dict(row._mapping) for row in result]

            logger.info(f"Found {len(all_stocks)} stocks to predict")

            # Load ML predictor
            predictor = EnhancedStockPredictor(session, auto_load=True)

            predictions = []
            successful = 0
            failed = 0

            for idx, stock in enumerate(all_stocks, 1):
                try:
                    symbol = stock['symbol']

                    if idx % 100 == 0:
                        logger.info(f"  Processing {idx}/{len(all_stocks)}...")

                    # Get prediction
                    prediction = predictor.predict_stock(symbol)

                    if prediction and prediction.get('ml_prediction_score'):
                        predictions.append({
                            'symbol': symbol,
                            'name': stock['name'],
                            'current_price': stock['current_price'],
                            'model_type': 'traditional',
                            **prediction
                        })
                        successful += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.debug(f"  Skipped {symbol}: {e}")
                    failed += 1

            logger.info(f"\n✅ Traditional ML predictions complete")
            logger.info(f"  Successful: {successful}")
            logger.info(f"  Failed/Skipped: {failed}")
            logger.info(f"  Total: {len(all_stocks)}")

            return predictions

    except Exception as e:
        logger.error(f"❌ Traditional ML prediction failed: {e}", exc_info=True)
        return []


def generate_lstm_all_stocks():
    """Generate Raw LSTM predictions for ALL stocks (where models exist)."""
    logger.info("\n" + "="*80)
    logger.info("MODEL 2: RAW LSTM - ALL TRAINED STOCKS")
    logger.info("="*80)

    try:
        from pathlib import Path

        db_manager = get_database_manager()
        lstm_service = RawLSTMPredictionService()

        # Get all symbols with trained LSTM models
        model_dir = Path('ml_models/raw_ohlcv_lstm')
        if not model_dir.exists():
            logger.warning("  No LSTM models directory found")
            return []

        trained_symbols = []
        for symbol_dir in model_dir.iterdir():
            if symbol_dir.is_dir():
                model_file = symbol_dir / 'lstm_model.h5'
                if model_file.exists():
                    trained_symbols.append(symbol_dir.name)

        logger.info(f"Found {len(trained_symbols)} symbols with trained LSTM models")

        predictions = []
        successful = 0
        failed = 0

        with db_manager.get_session() as session:
            for idx, symbol in enumerate(trained_symbols, 1):
                try:
                    if idx % 50 == 0:
                        logger.info(f"  Processing {idx}/{len(trained_symbols)}...")

                    # Get prediction
                    prediction = lstm_service.predict_symbol(symbol, session)

                    if prediction and prediction.get('ml_prediction_score'):
                        # Get stock info
                        stock_query = text("SELECT name, current_price FROM stocks WHERE symbol = :symbol")
                        stock_result = session.execute(stock_query, {'symbol': symbol}).fetchone()

                        if stock_result:
                            predictions.append({
                                'symbol': symbol,
                                'name': stock_result[0],
                                'current_price': stock_result[1],
                                'model_type': 'raw_lstm',
                                **prediction
                            })
                            successful += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.debug(f"  Skipped {symbol}: {e}")
                    failed += 1

            logger.info(f"\n✅ LSTM predictions complete")
            logger.info(f"  Successful: {successful}")
            logger.info(f"  Failed/Skipped: {failed}")
            logger.info(f"  Total: {len(trained_symbols)}")

            return predictions

    except Exception as e:
        logger.error(f"❌ LSTM prediction failed: {e}", exc_info=True)
        return []


def generate_kronos_all_stocks():
    """Generate Kronos predictions for ALL stocks."""
    logger.info("\n" + "="*80)
    logger.info("MODEL 3: KRONOS (K-line Tokenization) - ALL STOCKS")
    logger.info("="*80)

    try:
        from src.services.ml.kronos_prediction_service import KronosPredictionService

        db_manager = get_database_manager()
        kronos_service = KronosPredictionService()

        with db_manager.get_session() as session:
            # Get ALL stocks with sufficient historical data (200+ days for Kronos)
            query = text("""
                SELECT DISTINCT s.symbol, s.name, s.current_price
                FROM stocks s
                WHERE s.current_price IS NOT NULL
                  AND s.current_price > 0
                  AND EXISTS (
                      SELECT 1 FROM historical_data h
                      WHERE h.symbol = s.symbol
                      GROUP BY h.symbol
                      HAVING COUNT(*) >= 200
                  )
                ORDER BY s.symbol
            """)

            result = session.execute(query)
            all_stocks = [dict(row._mapping) for row in result]

            logger.info(f"Found {len(all_stocks)} stocks with sufficient history")

            predictions = []
            successful = 0
            failed = 0

            for idx, stock in enumerate(all_stocks, 1):
                try:
                    symbol = stock['symbol']

                    if idx % 100 == 0:
                        logger.info(f"  Processing {idx}/{len(all_stocks)}...")

                    # Get prediction
                    prediction = kronos_service.predict_stock(symbol, session)

                    if prediction and prediction.get('ml_prediction_score'):
                        predictions.append({
                            'symbol': symbol,
                            'name': stock['name'],
                            'current_price': stock['current_price'],
                            'model_type': 'kronos',
                            **prediction
                        })
                        successful += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.debug(f"  Skipped {symbol}: {e}")
                    failed += 1

            logger.info(f"\n✅ Kronos predictions complete")
            logger.info(f"  Successful: {successful}")
            logger.info(f"  Failed/Skipped: {failed}")
            logger.info(f"  Total: {len(all_stocks)}")

            return predictions

    except Exception as e:
        logger.error(f"❌ Kronos prediction failed: {e}", exc_info=True)
        return []


def save_predictions_to_db(predictions, model_type):
    """Save predictions to ml_predictions table."""
    if not predictions:
        logger.warning(f"No predictions to save for {model_type}")
        return 0

    try:
        db_manager = get_database_manager()

        with db_manager.get_session() as session:
            # Create table if not exists
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(50) NOT NULL,
                    stock_name VARCHAR(200),
                    current_price DECIMAL(10,2),
                    model_type VARCHAR(20) NOT NULL,
                    ml_prediction_score DECIMAL(5,4),
                    ml_price_target DECIMAL(10,2),
                    ml_confidence DECIMAL(5,4),
                    ml_risk_score DECIMAL(5,4),
                    prediction_date DATE NOT NULL DEFAULT CURRENT_DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, model_type, prediction_date)
                )
            """))

            # Delete old predictions for this model/date
            session.execute(text("""
                DELETE FROM ml_predictions
                WHERE model_type = :model_type
                  AND prediction_date = CURRENT_DATE
            """), {'model_type': model_type})

            # Insert new predictions
            for pred in predictions:
                session.execute(text("""
                    INSERT INTO ml_predictions (
                        symbol, stock_name, current_price, model_type,
                        ml_prediction_score, ml_price_target, ml_confidence, ml_risk_score
                    ) VALUES (
                        :symbol, :name, :current_price, :model_type,
                        :ml_prediction_score, :ml_price_target, :ml_confidence, :ml_risk_score
                    )
                    ON CONFLICT (symbol, model_type, prediction_date) DO UPDATE SET
                        ml_prediction_score = EXCLUDED.ml_prediction_score,
                        ml_price_target = EXCLUDED.ml_price_target,
                        ml_confidence = EXCLUDED.ml_confidence,
                        ml_risk_score = EXCLUDED.ml_risk_score
                """), {
                    'symbol': pred['symbol'],
                    'name': pred['name'],
                    'current_price': pred['current_price'],
                    'model_type': pred['model_type'],
                    'ml_prediction_score': pred.get('ml_prediction_score', 0),
                    'ml_price_target': pred.get('ml_price_target', pred['current_price']),
                    'ml_confidence': pred.get('ml_confidence', 0),
                    'ml_risk_score': pred.get('ml_risk_score', 0)
                })

            session.commit()
            logger.info(f"  ✅ Saved {len(predictions)} {model_type} predictions to database")
            return len(predictions)

    except Exception as e:
        logger.error(f"  ❌ Failed to save predictions: {e}", exc_info=True)
        return 0


def main():
    """Generate ML predictions for all stocks (all 3 models)."""
    logger.info("\n" + "█"*80)
    logger.info("GENERATING ML PREDICTIONS FOR ALL STOCKS - ALL 3 MODELS")
    logger.info("█"*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = datetime.now()
    total_saved = 0

    # Model 1: Traditional ML (RF + XGBoost)
    traditional_predictions = generate_traditional_ml_all_stocks()
    total_saved += save_predictions_to_db(traditional_predictions, 'traditional')

    # Model 2: Raw LSTM
    lstm_predictions = generate_lstm_all_stocks()
    total_saved += save_predictions_to_db(lstm_predictions, 'raw_lstm')

    # Model 3: Kronos
    kronos_predictions = generate_kronos_all_stocks()
    total_saved += save_predictions_to_db(kronos_predictions, 'kronos')

    # Final summary
    duration = (datetime.now() - start_time).total_seconds()

    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Total predictions saved: {total_saved}")
    logger.info(f"  Traditional ML: {len(traditional_predictions)}")
    logger.info(f"  Raw LSTM: {len(lstm_predictions)}")
    logger.info(f"  Kronos: {len(kronos_predictions)}")

    # Show database stats
    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            result = session.execute(text("""
                SELECT model_type, COUNT(*) as count
                FROM ml_predictions
                WHERE prediction_date = CURRENT_DATE
                GROUP BY model_type
                ORDER BY model_type
            """))

            logger.info("\nDatabase Verification:")
            for row in result:
                logger.info(f"  {row[0]:15} {row[1]:,} predictions")

    except Exception as e:
        logger.warning(f"Could not verify database: {e}")

    logger.info("="*80)
    logger.info("✅ ML PREDICTION GENERATION COMPLETE!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
