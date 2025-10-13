#!/usr/bin/env python3
"""
Generate Traditional ML Predictions for ALL stocks (similar to Kronos approach)

This script generates predictions using Random Forest + XGBoost for all active stocks,
filtered by strategy market cap requirements.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy import text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_all_active_symbols():
    """Get all active stock symbols from database."""
    db = get_database_manager()
    with db.get_session() as session:
        query = text("""
            SELECT symbol
            FROM stocks
            WHERE is_active = true
            ORDER BY symbol
        """)
        result = session.execute(query)
        symbols = [row[0] for row in result]

    logger.info(f"Found {len(symbols)} active stocks")
    return symbols


def filter_symbols_by_strategy(symbols, strategy):
    """
    Filter symbols by market cap based on risk strategy.

    DEFAULT_RISK: Large cap only (> 20,000 Cr)
    HIGH_RISK: Small/Mid cap only (1,000 - 20,000 Cr)
    """
    db = get_database_manager()
    with db.get_session() as session:
        query = text("""
            SELECT symbol, market_cap
            FROM stocks
            WHERE symbol = ANY(:symbols)
            AND is_active = true
        """)

        result = session.execute(query, {'symbols': symbols})
        rows = result.fetchall()

        filtered = []
        for row in rows:
            symbol = row[0]
            market_cap = float(row[1]) if row[1] else 0

            if strategy.upper() == 'DEFAULT_RISK':
                # Large cap only: > 20,000 Cr
                if market_cap > 20000:
                    filtered.append(symbol)
            elif strategy.upper() == 'HIGH_RISK':
                # Small/Mid cap: 1,000 - 20,000 Cr
                if 1000 <= market_cap <= 20000:
                    filtered.append(symbol)

        return filtered


def generate_traditional_ml_predictions(symbols, strategy='DEFAULT_RISK'):
    """Generate Traditional ML predictions using Enhanced Stock Predictor."""
    logger.info(f"\n{'='*100}")
    logger.info(f"GENERATING TRADITIONAL ML PREDICTIONS - {strategy}")
    logger.info(f"{'='*100}")

    filtered_symbols = filter_symbols_by_strategy(symbols, strategy)
    logger.info(f"Filtered to {len(filtered_symbols)} symbols for {strategy} strategy")

    if not filtered_symbols:
        logger.warning(f"No symbols found for {strategy} strategy!")
        return []

    db = get_database_manager()

    with db.get_session() as session:
        # Initialize predictor
        predictor = EnhancedStockPredictor(session, auto_load=True)

        predictions = []
        success_count = 0
        fail_count = 0

        logger.info(f"\nGenerating predictions for {len(filtered_symbols)} stocks...")
        logger.info("-" * 100)

        for i, symbol in enumerate(filtered_symbols, 1):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(filtered_symbols)} stocks processed...")

            try:
                # Generate prediction
                result = predictor.predict_stock(symbol)

                if result and result.get('success'):
                    pred = result['prediction']

                    # Add strategy-specific adjustments
                    if strategy.upper() == 'DEFAULT_RISK':
                        # Conservative: 7% target, 5% stop loss
                        pred['target_price'] = pred['current_price'] * 1.07
                        pred['stop_loss'] = pred['current_price'] * 0.95
                    else:  # HIGH_RISK
                        # Aggressive: 12% target, 10% stop loss
                        pred['target_price'] = pred['current_price'] * 1.12
                        pred['stop_loss'] = pred['current_price'] * 0.90

                    pred['strategy'] = strategy
                    pred['model_type'] = 'traditional'
                    predictions.append(pred)
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                logger.debug(f"Failed to predict {symbol}: {e}")
                fail_count += 1

        logger.info("-" * 100)
        logger.info(f"✅ Successfully generated: {success_count} predictions")
        logger.info(f"❌ Failed: {fail_count} stocks")

        # Sort by ML prediction score
        predictions.sort(key=lambda x: x.get('ml_prediction_score', 0), reverse=True)

        return predictions


def save_to_database(predictions, strategy, prediction_date=None):
    """Save predictions to daily_suggested_stocks table."""
    if not predictions:
        logger.warning("No predictions to save")
        return

    if prediction_date is None:
        prediction_date = datetime.now().date()

    logger.info(f"\nSaving {len(predictions)} predictions to database...")

    db = get_database_manager()
    with db.get_session() as session:
        for rank, pred in enumerate(predictions, 1):
            insert_query = text("""
                INSERT INTO daily_suggested_stocks (
                    date, symbol, stock_name, current_price, market_cap,
                    strategy, selection_score, rank,
                    ml_prediction_score, ml_price_target, ml_confidence, ml_risk_score,
                    pe_ratio, pb_ratio, roe, eps, beta,
                    revenue_growth, earnings_growth, operating_margin,
                    target_price, stop_loss, recommendation,
                    sector, market_cap_category, model_type, created_at
                ) VALUES (
                    :date, :symbol, :stock_name, :current_price, :market_cap,
                    :strategy, :selection_score, :rank,
                    :ml_prediction_score, :ml_price_target, :ml_confidence, :ml_risk_score,
                    :pe_ratio, :pb_ratio, :roe, :eps, :beta,
                    :revenue_growth, :earnings_growth, :operating_margin,
                    :target_price, :stop_loss, :recommendation,
                    :sector, :market_cap_category, :model_type, NOW()
                )
                ON CONFLICT (date, symbol, strategy) DO UPDATE SET
                    stock_name = EXCLUDED.stock_name,
                    current_price = EXCLUDED.current_price,
                    market_cap = EXCLUDED.market_cap,
                    selection_score = EXCLUDED.selection_score,
                    rank = EXCLUDED.rank,
                    ml_prediction_score = EXCLUDED.ml_prediction_score,
                    ml_price_target = EXCLUDED.ml_price_target,
                    ml_confidence = EXCLUDED.ml_confidence,
                    ml_risk_score = EXCLUDED.ml_risk_score,
                    target_price = EXCLUDED.target_price,
                    stop_loss = EXCLUDED.stop_loss,
                    recommendation = EXCLUDED.recommendation,
                    model_type = EXCLUDED.model_type
            """)

            session.execute(insert_query, {
                'date': prediction_date,
                'symbol': pred['symbol'],
                'stock_name': pred.get('stock_name'),
                'current_price': pred.get('current_price'),
                'market_cap': pred.get('market_cap'),
                'strategy': strategy,
                'selection_score': pred.get('ml_prediction_score'),
                'rank': rank,
                'ml_prediction_score': pred.get('ml_prediction_score'),
                'ml_price_target': pred.get('ml_price_target'),
                'ml_confidence': pred.get('ml_confidence'),
                'ml_risk_score': pred.get('ml_risk_score'),
                'pe_ratio': pred.get('pe_ratio'),
                'pb_ratio': pred.get('pb_ratio'),
                'roe': pred.get('roe'),
                'eps': pred.get('eps'),
                'beta': pred.get('beta'),
                'revenue_growth': pred.get('revenue_growth'),
                'earnings_growth': pred.get('earnings_growth'),
                'operating_margin': pred.get('operating_margin'),
                'target_price': pred.get('target_price'),
                'stop_loss': pred.get('stop_loss'),
                'recommendation': pred.get('recommendation', 'HOLD'),
                'sector': pred.get('sector'),
                'market_cap_category': pred.get('market_cap_category'),
                'model_type': 'traditional'
            })

        session.commit()
        logger.info(f"✅ Saved {len(predictions)} predictions to database")


def main():
    """Main execution."""
    logger.info("\n\n")
    logger.info("█" * 100)
    logger.info("█" + " "*98 + "█")
    logger.info("█" + " "*15 + "TRADITIONAL ML PREDICTION GENERATOR - All Stocks" + " "*40 + "█")
    logger.info("█" + " "*98 + "█")
    logger.info("█" * 100)

    # Get all active symbols
    all_symbols = get_all_active_symbols()

    if not all_symbols:
        logger.error("No active symbols found!")
        return 1

    today = datetime.now().date()

    # Strategy configurations
    strategies = ['DEFAULT_RISK', 'HIGH_RISK']

    for strategy in strategies:
        logger.info(f"\n📊 Processing {strategy} strategy...")

        # Generate predictions
        predictions = generate_traditional_ml_predictions(all_symbols, strategy)

        if predictions:
            # Show top 10
            logger.info(f"\nTop 10 stocks for {strategy}:")
            logger.info("-" * 100)
            logger.info(f"{'Rank':<6} {'Symbol':<20} {'Score':<8} {'Target':<10} {'Rec':<6}")
            logger.info("-" * 100)

            for i, pred in enumerate(predictions[:10], 1):
                score_pct = pred.get('ml_prediction_score', 0) * 100
                logger.info(
                    f"{i:<6} {pred['symbol']:<20} {score_pct:>6.2f}%  "
                    f"₹{pred.get('ml_price_target', 0):>8.2f}  {pred.get('recommendation', 'HOLD'):<6}"
                )

            # Save to database
            save_to_database(predictions, strategy, today)
        else:
            logger.warning(f"No predictions generated for {strategy}")

    # Final verification
    logger.info(f"\n{'='*100}")
    logger.info("FINAL VERIFICATION")
    logger.info(f"{'='*100}")

    db = get_database_manager()
    with db.get_session() as session:
        query = text("""
            SELECT strategy, COUNT(*) as count,
                   ROUND(AVG(ml_prediction_score)::numeric, 3) as avg_score
            FROM daily_suggested_stocks
            WHERE date = :today AND model_type = 'traditional'
            GROUP BY strategy
            ORDER BY strategy
        """)

        result = session.execute(query, {'today': today})
        rows = result.fetchall()

        logger.info(f"\nTraditional ML predictions for {today}:")
        logger.info("-" * 80)
        logger.info(f"{'Strategy':<20} {'Count':<10} {'Avg Score'}")
        logger.info("-" * 80)

        total_count = 0
        for row in rows:
            logger.info(f"{row[0]:<20} {row[1]:<10} {row[2]}")
            total_count += row[1]

        logger.info("-" * 80)
        logger.info(f"{'TOTAL':<20} {total_count:<10}")

    logger.info(f"\n{'='*100}")
    logger.info("✅ TRADITIONAL ML GENERATION COMPLETE!")
    logger.info(f"{'='*100}")

    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
