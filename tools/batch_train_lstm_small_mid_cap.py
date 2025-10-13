#!/usr/bin/env python3
"""
Batch Train Raw LSTM Models for Small/Mid-Cap Stocks (HIGH_RISK Strategy)

Trains LSTM models for top liquid small/mid-cap stocks (1,000-20,000 Cr market cap)
to populate Raw LSTM predictions for HIGH_RISK strategy.
Uses triple_barrier_config.yaml for model parameters.
"""

import sys
import logging
from pathlib import Path
from sqlalchemy import text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.ml.data_service import get_raw_ohlcv_data
from src.services.ml.raw_ohlcv_lstm import RawOHLCVLSTM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_small_mid_cap_stocks(limit=20):
    """Get top liquid small/mid-cap stocks by volume and liquidity."""
    db = get_database_manager()
    with db.get_session() as session:
        query = text("""
            SELECT symbol, name, market_cap, current_price, volume
            FROM stocks
            WHERE is_active = true
              AND market_cap >= 1000        -- Small/Mid cap: 1,000-20,000 Cr
              AND market_cap <= 20000
              AND current_price > 50        -- Avoid penny stocks
              AND current_price < 5000      -- Avoid extremely high-priced stocks
              AND volume > 50000            -- Good liquidity
            ORDER BY volume DESC, market_cap DESC
            LIMIT :limit
        """)

        result = session.execute(query, {'limit': limit})
        stocks = [dict(row._mapping) for row in result]

    return stocks


def train_lstm_for_stock(symbol, hidden_size=8, window_length=100):
    """Train LSTM model for a single stock."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Training Raw LSTM for {symbol}")
    logger.info(f"{'='*80}")

    try:
        # Fetch data
        logger.info("Fetching OHLCV data...")
        df = get_raw_ohlcv_data(symbol=symbol, period='3y', user_id=1)

        if len(df) < 300:
            logger.warning(f"❌ Insufficient data for {symbol}: {len(df)} days (need 300+)")
            return {'success': False, 'error': 'Insufficient data'}

        logger.info(f"✅ Data: {len(df)} days from {df.index.min()} to {df.index.max()}")

        # Create model (uses config from triple_barrier_config.yaml)
        logger.info("Initializing LSTM model...")
        model = RawOHLCVLSTM(
            hidden_size=hidden_size,
            window_length=window_length,
            use_full_ohlcv=True,
            dropout=0.2
        )

        # Train
        logger.info("Training...")
        metrics = model.train(
            df,
            test_size=0.2,
            epochs=50,  # Reduced for faster training
            batch_size=32,
            validation_split=0.1,
            early_stopping_patience=10,
            verbose=0  # Silent training
        )

        # Save
        model_dir = f'ml_models/raw_ohlcv_lstm/{symbol}'
        model.save(model_dir)

        logger.info(f"✅ Training complete!")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"   Model saved to: {model_dir}")

        return {
            'success': True,
            'metrics': metrics,
            'model_dir': model_dir
        }

    except Exception as e:
        logger.error(f"❌ Failed to train {symbol}: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Main execution."""
    logger.info("\n\n")
    logger.info("█" * 100)
    logger.info("█" + " "*98 + "█")
    logger.info("█" + " "*15 + "BATCH LSTM TRAINING - Small/Mid-Cap Stocks (HIGH_RISK)" + " "*30 + "█")
    logger.info("█" + " "*98 + "█")
    logger.info("█" * 100)

    # Get small/mid-cap stocks
    logger.info("\nFetching top liquid small/mid-cap stocks...")
    stocks = get_small_mid_cap_stocks(limit=20)

    if not stocks:
        logger.error("❌ No small/mid-cap stocks found matching criteria!")
        return 1

    logger.info(f"Selected {len(stocks)} stocks for training:")
    for i, stock in enumerate(stocks, 1):
        logger.info(f"  {i}. {stock['symbol']}: {stock['name']} (MCap: {stock['market_cap']:.0f} Cr, Vol: {stock['volume']:,})")

    # Train models
    results = []
    success_count = 0
    fail_count = 0

    for stock in stocks:
        symbol = stock['symbol']
        result = train_lstm_for_stock(symbol)
        results.append({
            'symbol': symbol,
            'name': stock['name'],
            **result
        })

        if result['success']:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    logger.info("\n\n")
    logger.info("=" * 100)
    logger.info("TRAINING SUMMARY - SMALL/MID-CAP STOCKS")
    logger.info("=" * 100)
    logger.info(f"\nTotal stocks: {len(stocks)}")
    logger.info(f"✅ Successfully trained: {success_count}")
    logger.info(f"❌ Failed: {fail_count}")

    if success_count > 0:
        logger.info(f"\n✅ Successfully trained {success_count} Raw LSTM models for small/mid-cap stocks!")
        logger.info("   Models saved to: ml_models/raw_ohlcv_lstm/")
        logger.info("\nNext steps:")
        logger.info("  1. Run: python3 tools/generate_raw_lstm_all_strategies.py")
        logger.info("  2. This will generate predictions for HIGH_RISK strategy")
        logger.info("  3. Check UI to see Raw LSTM HIGH_RISK predictions")

    logger.info("\n" + "=" * 100)

    return 0 if success_count > 0 else 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
