#!/usr/bin/env python3
"""
Scheduled Tasks Orchestrator
Runs daily ML training and pipeline updates at scheduled times.
"""

import sys
import logging
import schedule
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.database import get_database_manager
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor
from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator
from src.services.data.daily_snapshot_service import DailySnapshotService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_traditional_ml_models():
    """Train Traditional ML models (RF + XGBoost)."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING MODEL 1: TRADITIONAL ML (RF + XGBoost)")
    logger.info("=" * 80)

    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            logger.info("Using ENHANCED ML Predictor (RF + XGBoost + Chaos Features)")
            predictor = EnhancedStockPredictor(session, auto_load=True)

            # Train with walk-forward validation
            logger.info("Training enhanced models with 365 days + walk-forward CV...")
            stats = predictor.train_with_walk_forward(lookback_days=365, n_splits=5)

            logger.info("Traditional ML Training Complete!")
            logger.info(f"  Training Samples: {stats['samples']:,}")
            logger.info(f"  Features Used: {stats['features']}")
            logger.info(f"  Price Model R²: {stats['price_r2']:.4f}")
            logger.info(f"  Risk Model R²: {stats['risk_r2']:.4f}")
            logger.info(f"  CV Price R²: {stats['cv_price_r2']:.4f} (walk-forward)")
            logger.info(f"  CV Risk R²: {stats['cv_risk_r2']:.4f} (walk-forward)")
            logger.info(f"  Top Features: {', '.join(stats['top_features'][:5])}")
            logger.info("✅ Traditional ML models trained and saved successfully")

    except Exception as e:
        logger.error(f"❌ Traditional ML training failed: {e}", exc_info=True)


def train_lstm_models():
    """Train Raw LSTM models for top liquid stocks."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING MODEL 2: RAW LSTM (Deep Learning)")
    logger.info("=" * 80)

    try:
        import subprocess
        from sqlalchemy import text

        # Get existing trained models count
        model_dir = Path('ml_models/raw_ohlcv_lstm')
        existing_models = len([d for d in model_dir.iterdir() if d.is_dir()]) if model_dir.exists() else 0

        logger.info(f"Existing LSTM models: {existing_models}")

        # Train large-cap stocks (DEFAULT_RISK)
        logger.info("\n🎯 Training LSTM models for large-cap stocks (DEFAULT_RISK)...")
        result1 = subprocess.run(['python3', 'tools/batch_train_lstm_top_stocks.py'],
                                capture_output=True, text=True)
        if result1.returncode == 0:
            logger.info("✅ Large-cap LSTM training completed")
        else:
            logger.error(f"❌ Large-cap LSTM training failed: {result1.stderr}")

        # Train small/mid-cap stocks (HIGH_RISK)
        logger.info("\n🎯 Training LSTM models for small/mid-cap stocks (HIGH_RISK)...")
        result2 = subprocess.run(['python3', 'tools/batch_train_lstm_small_mid_cap.py'],
                                capture_output=True, text=True)
        if result2.returncode == 0:
            logger.info("✅ Small/mid-cap LSTM training completed")
        else:
            logger.error(f"❌ Small/mid-cap LSTM training failed: {result2.stderr}")

        # Count new models
        new_models = len([d for d in model_dir.iterdir() if d.is_dir()]) if model_dir.exists() else 0
        logger.info(f"\n✅ LSTM Training Complete: {new_models} total models ({new_models - existing_models} new)")

    except Exception as e:
        logger.error(f"❌ LSTM training failed: {e}", exc_info=True)


def train_kronos_models():
    """Train/Update Kronos models (K-line tokenization)."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING MODEL 3: KRONOS (K-line Tokenization)")
    logger.info("=" * 80)

    try:
        import subprocess

        logger.info("🎯 Generating Kronos predictions for all strategies...")
        result = subprocess.run(['python3', 'tools/generate_kronos_predictions.py'],
                               capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ Kronos predictions generated successfully")
        else:
            logger.error(f"❌ Kronos generation failed: {result.stderr}")

    except Exception as e:
        logger.error(f"❌ Kronos training failed: {e}", exc_info=True)


def train_all_ml_models():
    """Train all 3 ML models: Traditional, LSTM, Kronos."""
    logger.info("\n\n" + "█" * 80)
    logger.info("DAILY ML TRAINING - ALL 3 MODELS")
    logger.info("█" * 80)

    start_time = datetime.now()

    # Train all 3 models
    train_traditional_ml_models()
    train_lstm_models()
    train_kronos_models()

    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL ML MODELS TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"  Models Trained: Traditional ML, Raw LSTM, Kronos")
    logger.info("=" * 80)


def update_daily_snapshot():
    """Update daily suggested stocks snapshot (runs at 10:15 PM, after ML training)."""
    logger.info("=" * 80)
    logger.info("Starting Daily Suggested Stocks Update - DUAL MODEL + DUAL STRATEGY MODE")
    logger.info("=" * 80)

    # Run BOTH strategies to populate all 4 model/risk combinations
    strategies_to_run = ['default_risk', 'high_risk']
    total_stocks_stored = 0

    try:
        # ============================================================
        # PART 1: Traditional Model (via Saga Orchestrator)
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("TRADITIONAL MODEL PREDICTIONS")
        logger.info("="*80)

        orchestrator = SuggestedStocksSagaOrchestrator()

        for strategy in strategies_to_run:
            logger.info("-" * 80)
            logger.info(f"Running Traditional Model: {strategy.upper()}")
            logger.info("-" * 80)

            try:
                # Run suggested stocks saga for this strategy
                result = orchestrator.execute_suggested_stocks_saga(
                    user_id=1,
                    strategies=[strategy],
                    limit=50  # Store top 50 daily picks per strategy
                )

                if result['status'] == 'completed':
                    stocks_count = result['summary'].get('final_result_count', 0)
                    total_stocks_stored += stocks_count
                    logger.info(f"✅ Traditional {strategy} snapshot updated successfully")
                    logger.info(f"  Stocks stored: {stocks_count}")

                    # Check if ML predictions were applied
                    ml_step = next((s for s in result['summary']['step_summary']
                                  if s['step_id'] == 'step6_ml_prediction'), None)
                    if ml_step and ml_step['status'] == 'completed':
                        logger.info(f"  ML predictions: ✅ Applied (Avg Score: {ml_step['metadata'].get('avg_prediction_score', 0):.3f})")

                    # Check snapshot save
                    snapshot_step = next((s for s in result['summary']['step_summary']
                                        if s['step_id'] == 'step7_daily_snapshot'), None)
                    if snapshot_step and snapshot_step['status'] == 'completed':
                        metadata = snapshot_step.get('metadata', {})
                        logger.info(f"  Snapshot: ✅ Saved ({metadata.get('inserted', 0)} inserted, {metadata.get('updated', 0)} updated)")
                else:
                    logger.error(f"❌ Traditional {strategy} snapshot update failed: {result.get('errors', [])}")

            except Exception as e:
                logger.error(f"❌ Traditional {strategy} strategy failed: {e}", exc_info=True)
                continue

        # ============================================================
        # PART 2: Raw LSTM Model (Direct Service Call)
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("RAW LSTM MODEL PREDICTIONS")
        logger.info("="*80)

        try:
            from pathlib import Path
            from src.services.ml.raw_lstm_prediction_service import get_raw_lstm_prediction_service

            # Get trained Raw LSTM models
            model_dir = Path('ml_models/raw_ohlcv_lstm')
            if model_dir.exists():
                symbols = [d.name for d in model_dir.iterdir()
                          if d.is_dir() and ((d / 'lstm_model.h5').exists() or (d / 'model.h5').exists())]

                if symbols:
                    logger.info(f"Found {len(symbols)} trained Raw LSTM models")

                    service = get_raw_lstm_prediction_service()

                    for strategy in strategies_to_run:
                        logger.info("-" * 80)
                        logger.info(f"Running Raw LSTM Model: {strategy.upper()}")
                        logger.info("-" * 80)

                        try:
                            # Generate predictions with strategy
                            predictions = service.batch_predict(symbols, user_id=1, strategy=strategy)

                            if predictions:
                                # Sort by prediction score
                                predictions.sort(key=lambda x: x['ml_prediction_score'], reverse=True)

                                # Save to database
                                service.save_to_suggested_stocks(predictions, strategy=strategy)

                                total_stocks_stored += len(predictions)
                                logger.info(f"✅ Raw LSTM {strategy} predictions saved")
                                logger.info(f"  Stocks stored: {len(predictions)}")
                            else:
                                logger.warning(f"⚠️  No Raw LSTM predictions generated for {strategy}")

                        except Exception as e:
                            logger.error(f"❌ Raw LSTM {strategy} failed: {e}", exc_info=True)
                            continue
                else:
                    logger.warning("⚠️  No trained Raw LSTM models found - skipping")
            else:
                logger.warning("⚠️  Raw LSTM model directory not found - skipping")

        except Exception as e:
            logger.error(f"❌ Raw LSTM predictions failed: {e}", exc_info=True)

        # ============================================================
        # PART 3: Kronos Model (Direct Service Call)
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("KRONOS MODEL PREDICTIONS")
        logger.info("="*80)

        try:
            import subprocess

            logger.info("🎯 Generating Kronos predictions for all strategies...")
            result = subprocess.run(['python3', 'tools/generate_kronos_predictions.py'],
                                   capture_output=True, text=True)

            if result.returncode == 0:
                # Count Kronos predictions from stdout
                logger.info("✅ Kronos predictions generated successfully")

                # Parse output to get count
                from sqlalchemy import text
                with db_manager.get_session() as session:
                    count_query = text("""
                        SELECT COUNT(*)
                        FROM daily_suggested_stocks
                        WHERE model_type = 'kronos'
                        AND date = CURRENT_DATE
                    """)
                    kronos_count = session.execute(count_query).scalar()
                    total_stocks_stored += kronos_count
                    logger.info(f"  Stocks stored: {kronos_count}")
            else:
                logger.error(f"❌ Kronos generation failed: {result.stderr}")

        except Exception as e:
            logger.error(f"❌ Kronos predictions failed: {e}", exc_info=True)

        # ============================================================
        # PART 4: Ollama Enhancement (Optional - If Enabled)
        # ============================================================
        try:
            from src.config.ollama_config import get_ollama_config
            ollama_config = get_ollama_config()

            # Check if daily predictions enhancement is enabled
            daily_pred_config = ollama_config._config.get('daily_predictions', {})
            if daily_pred_config.get('enabled', False):
                logger.info("\n" + "="*80)
                logger.info("OLLAMA AI ENHANCEMENT")
                logger.info("="*80)

                from src.services.data.strategy_ollama_enhancement_service import get_strategy_ollama_enhancement_service
                from sqlalchemy import text

                ollama_service = get_strategy_ollama_enhancement_service()
                enhancement_level = daily_pred_config.get('enhancement_level', 'light')
                max_stocks = daily_pred_config.get('max_stocks_to_enhance', 50)

                logger.info(f"🔍 Enhancing top {max_stocks} stocks with Ollama AI")
                logger.info(f"   Enhancement Level: {enhancement_level}")

                total_enhanced = 0

                for strategy in strategies_to_run:
                    try:
                        # Get top stocks for this strategy from database
                        db_manager = get_database_manager()
                        with db_manager.get_session() as session:
                            query = text("""
                                SELECT
                                    symbol, stock_name as name, current_price,
                                    ml_prediction_score, recommendation
                                FROM daily_suggested_stocks
                                WHERE date = CURRENT_DATE
                                AND strategy = :strategy
                                ORDER BY ml_prediction_score DESC
                                LIMIT :limit
                            """)

                            result = session.execute(query, {
                                'strategy': strategy,
                                'limit': max_stocks
                            })
                            stocks = [dict(row._mapping) for row in result]

                        if stocks:
                            logger.info(f"\n🎯 Enhancing {len(stocks)} stocks for {strategy}...")

                            # Enhance stocks (this adds market intelligence)
                            enhanced_stocks = ollama_service.enhance_strategy_recommendations(
                                stocks, strategy, enhancement_level
                            )

                            total_enhanced += len(enhanced_stocks)
                            logger.info(f"   ✅ Enhanced {len(enhanced_stocks)} stocks")

                            # Note: Ollama adds metadata to predictions but doesn't
                            # update database - it's available in real-time API calls

                    except Exception as e:
                        if daily_pred_config.get('skip_on_failure', True):
                            logger.warning(f"⚠️  Ollama enhancement failed for {strategy}, skipping: {e}")
                            continue
                        else:
                            raise

                logger.info(f"\n✅ Ollama enhancement completed: {total_enhanced} stocks enhanced")

            else:
                logger.info("\n⏭️  Ollama enhancement disabled in config - skipping")

        except Exception as e:
            logger.warning(f"⚠️  Ollama enhancement service unavailable: {e}")
            logger.info("   Continuing without Ollama enhancement...")

        # ============================================================
        # Summary
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info(f"✅ Daily Snapshot Update Complete (Triple Model + Dual Strategy + AI)!")
        logger.info("="*80)
        logger.info(f"  Total stocks stored across ALL models and strategies: {total_stocks_stored}")
        logger.info(f"  Traditional Model strategies: {len(strategies_to_run)}")
        logger.info(f"  Raw LSTM Model strategies: {len(strategies_to_run)}")
        logger.info(f"  Kronos Model strategies: {len(strategies_to_run)}")
        logger.info(f"  Ollama AI Enhancement: {'✅ Enabled' if daily_pred_config.get('enabled', False) else '❌ Disabled'}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"❌ Daily snapshot update failed: {e}", exc_info=True)


def cleanup_old_snapshots():
    """Clean up old snapshots (runs weekly on Sunday at 3 AM)."""
    logger.info("=" * 80)
    logger.info("Starting Old Snapshots Cleanup")
    logger.info("=" * 80)
    
    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            snapshot_service = DailySnapshotService(session)
            
            # Delete snapshots older than 90 days
            deleted = snapshot_service.delete_old_snapshots(keep_days=90)
            logger.info(f"✅ Cleaned up {deleted} old snapshot records (>90 days)")
            
    except Exception as e:
        logger.error(f"❌ Snapshot cleanup failed: {e}", exc_info=True)


def check_training_needed():
    """Check if training is needed today, train if yes."""
    logger.info("=" * 80)
    logger.info("Startup Training Check")
    logger.info("=" * 80)

    try:
        from sqlalchemy import text

        # Check if training happened today
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            today = datetime.now().date()

            # Check Traditional ML last training
            traditional_query = text("""
                SELECT created_at::date as training_date
                FROM daily_suggested_stocks
                WHERE model_type = 'traditional'
                ORDER BY created_at DESC
                LIMIT 1
            """)
            traditional_result = session.execute(traditional_query).fetchone()
            traditional_trained_today = traditional_result and traditional_result[0] == today

            # Check LSTM last training
            lstm_query = text("""
                SELECT created_at::date as training_date
                FROM daily_suggested_stocks
                WHERE model_type = 'raw_lstm'
                ORDER BY created_at DESC
                LIMIT 1
            """)
            lstm_result = session.execute(lstm_query).fetchone()
            lstm_trained_today = lstm_result and lstm_result[0] == today

            # Check Kronos last training
            kronos_query = text("""
                SELECT created_at::date as training_date
                FROM daily_suggested_stocks
                WHERE model_type = 'kronos'
                ORDER BY created_at DESC
                LIMIT 1
            """)
            kronos_result = session.execute(kronos_query).fetchone()
            kronos_trained_today = kronos_result and kronos_result[0] == today

            logger.info("\nTraining Status for Today:")
            logger.info(f"  Traditional ML: {'✅ Done' if traditional_trained_today else '❌ Needed'}")
            logger.info(f"  Raw LSTM:       {'✅ Done' if lstm_trained_today else '❌ Needed'}")
            logger.info(f"  Kronos:         {'✅ Done' if kronos_trained_today else '❌ Needed'}")

            # If any model needs training, train all
            if not (traditional_trained_today and lstm_trained_today and kronos_trained_today):
                logger.info("\n🔄 Training needed - starting full training cycle...")
                train_all_ml_models()
                update_daily_snapshot()
            else:
                logger.info("\n✅ All models already trained today - skipping")

    except Exception as e:
        logger.error(f"❌ Training check failed: {e}", exc_info=True)
        logger.info("🔄 Attempting training anyway as a safety measure...")
        train_all_ml_models()


def run_scheduler():
    """Main scheduler loop."""
    logger.info("=" * 80)
    logger.info("Trading System Scheduler Started")
    logger.info("=" * 80)
    logger.info("Scheduled Tasks:")
    logger.info("  - ML Training (ALL 3 MODELS): Daily at 10:00 PM (after data pipeline)")
    logger.info("    → Model 1: Traditional ML (RF + XGBoost)")
    logger.info("    → Model 2: Raw LSTM (Deep Learning)")
    logger.info("    → Model 3: Kronos (K-line Tokenization)")
    logger.info("  - Daily Snapshot Update:     Daily at 10:15 PM (after ML training)")
    logger.info("    → Models: TRADITIONAL + RAW_LSTM + KRONOS (all 3)")
    logger.info("    → Strategies: DEFAULT_RISK + HIGH_RISK (both)")
    logger.info("    → Total: 6 combinations (3 models × 2 strategies)")
    logger.info("  - Cleanup Old Snapshots:     Weekly (Sunday) at 03:00 AM")
    logger.info("=" * 80)

    # Check if training is needed today and train if necessary
    check_training_needed()

    # Schedule daily ML training at 10:00 PM (all 3 models)
    schedule.every().day.at("22:00").do(train_all_ml_models)

    # Schedule daily snapshot update at 10:15 PM (after ML training)
    schedule.every().day.at("22:15").do(update_daily_snapshot)

    # Schedule weekly cleanup on Sunday at 3:00 AM
    schedule.every().sunday.at("03:00").do(cleanup_old_snapshots)

    # Keep scheduler running
    logger.info("Scheduler is now running. Press Ctrl+C to stop.")
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Scheduler error: {e}", exc_info=True)
            time.sleep(60)


if __name__ == '__main__':
    run_scheduler()
