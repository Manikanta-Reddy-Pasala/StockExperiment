#!/usr/bin/env python3
"""
Scheduled Tasks Orchestrator
Runs daily ML training and pipeline updates at scheduled times.
"""

import sys
import logging
import schedule
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.database import get_database_manager
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor
from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator
from src.services.data.daily_snapshot_service import DailySnapshotService
from src.services.trading.auto_trading_service import get_auto_trading_service
from src.services.trading.order_performance_tracking_service import get_performance_tracking_service

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


def _get_last_trading_day() -> datetime.date:
    """
    Get the last expected trading day (accounting for weekends).

    Returns:
        date: The last trading day (Friday if today is Saturday/Sunday)
    """
    today = datetime.now().date()

    # If Saturday, last trading day is Friday
    if today.weekday() == 5:  # Saturday
        return today - timedelta(days=1)
    # If Sunday, last trading day is Friday
    elif today.weekday() == 6:  # Sunday
        return today - timedelta(days=2)
    else:
        # Weekday - check if market has closed (3:30 PM IST = 10:00 AM UTC)
        now = datetime.utcnow()
        market_close_utc = now.replace(hour=10, minute=0, second=0, microsecond=0)

        if now >= market_close_utc:
            # Market closed, today is the last trading day
            return today
        else:
            # Market not yet closed, yesterday is the last complete day
            yesterday = today - timedelta(days=1)
            # If yesterday was weekend, go back to Friday
            if yesterday.weekday() == 5:  # Saturday
                return yesterday - timedelta(days=1)
            elif yesterday.weekday() == 6:  # Sunday
                return yesterday - timedelta(days=2)
            return yesterday


def check_data_freshness(max_age_days: int = 3) -> dict:
    """
    Check if historical data is fresh enough for ML training.
    Considers weekends and holidays.

    Args:
        max_age_days: Maximum acceptable age of data (default 3 days to handle long weekends)

    Returns:
        dict with keys:
            - fresh (bool): True if data is fresh enough
            - last_data_date (date): Latest data date in database
            - expected_date (date): Expected last trading day
            - age_days (int): Age of data in days
            - message (str): Description of data status
    """
    try:
        from sqlalchemy import text

        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # Get the latest data date from historical_data
            query = text("""
                SELECT MAX(date) as latest_date, COUNT(DISTINCT symbol) as symbols_count
                FROM historical_data
            """)
            result = session.execute(query).fetchone()

            if not result or not result[0]:
                return {
                    'fresh': False,
                    'last_data_date': None,
                    'expected_date': _get_last_trading_day(),
                    'age_days': 999,
                    'message': 'No historical data found in database'
                }

            last_data_date = result[0]
            symbols_count = result[1]
            expected_date = _get_last_trading_day()

            # Calculate age in days
            age_days = (expected_date - last_data_date).days

            # Data is fresh if it's from the expected last trading day or within max_age_days
            # We allow some tolerance for holidays and long weekends
            is_fresh = age_days <= max_age_days

            if age_days == 0:
                message = f'✅ Data is current ({last_data_date}, {symbols_count:,} symbols)'
            elif age_days <= max_age_days:
                message = f'✅ Data is acceptable ({last_data_date}, {age_days} days old, {symbols_count:,} symbols)'
            else:
                message = f'❌ Data is stale ({last_data_date}, {age_days} days old, expected {expected_date})'

            return {
                'fresh': is_fresh,
                'last_data_date': last_data_date,
                'expected_date': expected_date,
                'age_days': age_days,
                'symbols_count': symbols_count,
                'message': message
            }

    except Exception as e:
        logger.error(f"Failed to check data freshness: {e}")
        return {
            'fresh': False,
            'last_data_date': None,
            'expected_date': _get_last_trading_day(),
            'age_days': 999,
            'message': f'Error checking data: {str(e)}'
        }


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
        # Kronos doesn't require separate training - it uses K-line tokenization on-the-fly
        # The actual predictions happen during the saga execution in update_daily_snapshot()
        logger.info("ℹ️  Kronos uses on-the-fly K-line tokenization")
        logger.info("   No pre-training required - predictions generated during saga execution")
        logger.info("✅ Kronos ready for prediction (will run during snapshot update)")

    except Exception as e:
        logger.error(f"❌ Kronos training check failed: {e}", exc_info=True)


def generate_all_ml_predictions():
    """Generate ML predictions for ALL stocks (not just filtered subset)."""
    logger.info("\n\n" + "█" * 80)
    logger.info("GENERATING ML PREDICTIONS FOR ALL STOCKS")
    logger.info("█" * 80)

    try:
        result = subprocess.run(
            ['python3', 'tools/generate_ml_all_stocks.py'],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes
        )

        if result.returncode == 0:
            logger.info("✅ ML predictions generated successfully for all stocks")
            # Log summary (last 30 lines which contain the summary)
            output_lines = result.stdout.split('\n')
            summary_lines = output_lines[-30:]
            for line in summary_lines:
                if line.strip():
                    logger.info(f"  {line}")
        else:
            logger.error(f"❌ ML prediction generation failed with return code {result.returncode}")
            logger.error(f"Error:\n{result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("❌ ML prediction generation timeout after 30 minutes")
    except Exception as e:
        logger.error(f"❌ ML prediction generation error: {e}", exc_info=True)


def train_all_ml_models():
    """Train all 3 ML models: Traditional, LSTM, Kronos."""
    logger.info("\n\n" + "█" * 80)
    logger.info("DAILY ML TRAINING - ALL 3 MODELS")
    logger.info("█" * 80)

    # Check data freshness before training
    logger.info("\n" + "=" * 80)
    logger.info("DATA FRESHNESS CHECK")
    logger.info("=" * 80)

    freshness = check_data_freshness(max_age_days=3)
    logger.info(freshness['message'])

    if not freshness['fresh']:
        logger.warning("⚠️  Data is too old for reliable ML training!")
        logger.warning(f"   Last data: {freshness['last_data_date']}")
        logger.warning(f"   Expected: {freshness['expected_date']}")
        logger.warning(f"   Age: {freshness['age_days']} days")
        logger.warning("")
        logger.warning("Possible reasons:")
        logger.warning("  1. Market holiday (NSE closed)")
        logger.warning("  2. Data pipeline didn't run")
        logger.warning("  3. Weekend - data should be from Friday")
        logger.warning("")
        logger.warning("⏭️  SKIPPING ML TRAINING - Run data pipeline first!")
        logger.warning("   Manual command: docker compose restart data_scheduler")
        logger.warning("   Or run: python3 run_pipeline.py")
        return

    logger.info(f"✅ Data is fresh - proceeding with ML training")
    logger.info(f"   Latest data: {freshness['last_data_date']}")
    logger.info(f"   Symbols: {freshness['symbols_count']:,}")

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
        # PART 2: Raw LSTM Model (via Saga Orchestrator)
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("RAW LSTM MODEL PREDICTIONS")
        logger.info("="*80)

        try:
            from pathlib import Path

            # Check if any trained Raw LSTM models exist
            model_dir = Path('ml_models/raw_ohlcv_lstm')
            if model_dir.exists():
                symbols = [d.name for d in model_dir.iterdir()
                          if d.is_dir() and ((d / 'lstm_model.h5').exists() or (d / 'model.h5').exists())]

                if symbols:
                    logger.info(f"Found {len(symbols)} trained Raw LSTM models")

                    # Use saga orchestrator for Raw LSTM (same pipeline as Traditional/Kronos)
                    for strategy in strategies_to_run:
                        logger.info("-" * 80)
                        logger.info(f"Running Raw LSTM Model: {strategy.upper()}")
                        logger.info("-" * 80)

                        try:
                            # Run suggested stocks saga for Raw LSTM model
                            result = orchestrator.execute_suggested_stocks_saga(
                                user_id=1,
                                strategies=[strategy],
                                limit=50,  # Store top 50 daily picks per strategy
                                model_type='raw_lstm'  # Use Raw LSTM model
                            )

                            if result['status'] == 'completed':
                                stocks_count = result['summary'].get('final_result_count', 0)
                                total_stocks_stored += stocks_count
                                logger.info(f"✅ Raw LSTM {strategy} snapshot updated successfully")
                                logger.info(f"  Stocks stored: {stocks_count}")

                                # Check if ML predictions were applied
                                ml_step = next((s for s in result['summary']['step_summary']
                                              if s['step_id'] == 'step6_ml_prediction'), None)
                                if ml_step and ml_step['status'] == 'completed':
                                    logger.info(f"  LSTM predictions: ✅ Applied (Avg Score: {ml_step['metadata'].get('avg_prediction_score', 0):.3f})")

                                # Check snapshot save
                                snapshot_step = next((s for s in result['summary']['step_summary']
                                                    if s['step_id'] == 'step7_daily_snapshot'), None)
                                if snapshot_step and snapshot_step['status'] == 'completed':
                                    metadata = snapshot_step.get('metadata', {})
                                    logger.info(f"  Snapshot: ✅ Saved ({metadata.get('inserted', 0)} inserted, {metadata.get('updated', 0)} updated)")
                            else:
                                logger.error(f"❌ Raw LSTM {strategy} snapshot update failed: {result.get('errors', [])}")

                        except Exception as e:
                            logger.error(f"❌ Raw LSTM {strategy} strategy failed: {e}", exc_info=True)
                            continue
                else:
                    logger.warning("⚠️  No trained Raw LSTM models found - skipping")
            else:
                logger.warning("⚠️  Raw LSTM model directory not found - skipping")

        except Exception as e:
            logger.error(f"❌ Raw LSTM predictions failed: {e}", exc_info=True)

        # ============================================================
        # PART 3: Kronos Model (via Saga Orchestrator)
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("KRONOS MODEL PREDICTIONS")
        logger.info("="*80)

        try:
            # Reuse the same orchestrator instance
            for strategy in strategies_to_run:
                logger.info("-" * 80)
                logger.info(f"Running Kronos Model: {strategy.upper()}")
                logger.info("-" * 80)

                try:
                    # Run suggested stocks saga for Kronos model
                    result = orchestrator.execute_suggested_stocks_saga(
                        user_id=1,
                        strategies=[strategy],
                        limit=50,  # Store top 50 daily picks per strategy
                        model_type='kronos'  # Use Kronos model
                    )

                    if result['status'] == 'completed':
                        stocks_count = result['summary'].get('final_result_count', 0)
                        total_stocks_stored += stocks_count
                        logger.info(f"✅ Kronos {strategy} snapshot updated successfully")
                        logger.info(f"  Stocks stored: {stocks_count}")

                        # Check if ML predictions were applied
                        ml_step = next((s for s in result['summary']['step_summary']
                                      if s['step_id'] == 'step6_ml_prediction'), None)
                        if ml_step and ml_step['status'] == 'completed':
                            logger.info(f"  Kronos predictions: ✅ Applied (Avg Score: {ml_step['metadata'].get('avg_prediction_score', 0):.3f})")

                        # Check snapshot save
                        snapshot_step = next((s for s in result['summary']['step_summary']
                                            if s['step_id'] == 'step7_daily_snapshot'), None)
                        if snapshot_step and snapshot_step['status'] == 'completed':
                            metadata = snapshot_step.get('metadata', {})
                            logger.info(f"  Snapshot: ✅ Saved ({metadata.get('inserted', 0)} inserted, {metadata.get('updated', 0)} updated)")
                    else:
                        logger.error(f"❌ Kronos {strategy} snapshot update failed: {result.get('errors', [])}")

                except Exception as e:
                    logger.error(f"❌ Kronos {strategy} strategy failed: {e}", exc_info=True)
                    continue

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
        logger.info(f"✅ Daily Snapshot Update Complete (Triple Model + Dual Strategy + UNIFIED Saga Pipeline)!")
        logger.info("="*80)
        logger.info(f"  Total stocks stored across ALL models and strategies: {total_stocks_stored}")
        logger.info(f"  Traditional Model: {len(strategies_to_run)} strategies (via Saga)")
        logger.info(f"  Raw LSTM Model: {len(strategies_to_run)} strategies (via Saga)")
        logger.info(f"  Kronos Model: {len(strategies_to_run)} strategies (via Saga)")
        logger.info(f"  Pipeline: ALL 3 MODELS use full 7-step saga (Stage 1 + Stage 2 filtering)")
        logger.info(f"  Quality: Only liquid, tradeable stocks with proper fundamentals")
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


def execute_auto_trading():
    """Execute auto-trading for all enabled users (runs daily at 9:00 AM)."""
    logger.info("\n" + "=" * 80)
    logger.info("AUTOMATED TRADING EXECUTION")
    logger.info("=" * 80)

    try:
        auto_trading_service = get_auto_trading_service()

        logger.info("🤖 Starting auto-trading for all enabled users...")

        result = auto_trading_service.execute_auto_trading_for_all_users()

        if result.get('success'):
            total_users = result.get('total_users', 0)
            logger.info(f"✅ Auto-trading completed for {total_users} users")

            # Log results per user
            for user_result in result.get('results', []):
                user_id = user_result['user_id']
                user_res = user_result['result']

                if user_res.get('success'):
                    status = user_res.get('status')
                    if status == 'success':
                        logger.info(f"  User {user_id}: ✅ {user_res.get('orders_created', 0)} orders, "
                                   f"₹{user_res.get('total_invested', 0):.2f} invested")
                    elif status == 'skipped':
                        logger.info(f"  User {user_id}: ⏭️  {user_res.get('message', 'Skipped')}")
                else:
                    logger.error(f"  User {user_id}: ❌ {user_res.get('error', 'Failed')}")
        else:
            logger.error(f"❌ Auto-trading failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"❌ Auto-trading execution failed: {e}", exc_info=True)


def update_order_performance():
    """Update performance tracking for all active orders (runs daily at 6:00 PM)."""
    logger.info("\n" + "=" * 80)
    logger.info("ORDER PERFORMANCE UPDATE")
    logger.info("=" * 80)

    try:
        performance_service = get_performance_tracking_service()

        logger.info("📊 Updating performance for all active orders...")

        result = performance_service.update_all_active_orders()

        if result.get('success'):
            logger.info(f"✅ Performance update completed")
            logger.info(f"  Orders updated: {result.get('orders_updated', 0)}")
            logger.info(f"  Snapshots created: {result.get('snapshots_created', 0)}")
            logger.info(f"  Orders closed: {result.get('orders_closed', 0)}")
        else:
            logger.error(f"❌ Performance update failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"❌ Performance tracking failed: {e}", exc_info=True)


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


def check_broker_token_status():
    """
    Check Fyers broker token status and warn if expiring soon.
    Runs every 6 hours to monitor token health.
    """
    logger.info("=" * 80)
    logger.info("Checking Broker Token Status")
    logger.info("=" * 80)

    try:
        from src.services.utils.token_manager_service import get_token_manager
        from src.models.models import BrokerConfiguration

        db_manager = get_database_manager()
        token_manager = get_token_manager()

        with db_manager.get_session() as session:
            # Get all Fyers configurations
            fyers_configs = session.query(BrokerConfiguration).filter_by(
                broker_name='fyers'
            ).all()

            if not fyers_configs:
                logger.info("  ℹ️  No Fyers broker configurations found")
                return

            for config in fyers_configs:
                user_id = config.user_id or 1  # Default to user 1 if None

                try:
                    # Get token status
                    status = token_manager.get_token_status(user_id, 'fyers')

                    if not status['has_token']:
                        logger.warning(f"  ⚠️  User {user_id}: No token found - re-authentication required")
                        continue

                    if status['is_expired']:
                        logger.error(f"  ❌ User {user_id}: Token EXPIRED - re-authentication required!")
                        logger.error(f"     Please login to Fyers at: http://localhost:5001/brokers/fyers")

                        # Mark as disconnected
                        config.is_connected = False
                        config.connection_status = 'reauth_required'
                        session.commit()
                        continue

                    # Check if expiring soon (within 12 hours)
                    if status['expires_at']:
                        expiry_time = datetime.fromisoformat(status['expires_at'])
                        time_until_expiry = expiry_time - datetime.now()
                        hours_until_expiry = time_until_expiry.total_seconds() / 3600

                        if hours_until_expiry < 12:
                            logger.warning(f"  ⚠️  User {user_id}: Token expires in {hours_until_expiry:.1f} hours!")
                            logger.warning(f"     Expiry: {status['expires_at']}")
                            logger.warning(f"     Please re-authenticate before expiry")
                        else:
                            logger.info(f"  ✅ User {user_id}: Token valid for {hours_until_expiry:.1f} hours")

                        # Start auto-refresh if not already running
                        if not status['auto_refresh_active']:
                            logger.info(f"  🔄 User {user_id}: Starting auto-refresh monitoring...")
                            token_manager.start_auto_refresh(user_id, 'fyers', check_interval_minutes=30)
                    else:
                        logger.info(f"  ✅ User {user_id}: Token valid (no expiry info)")

                except Exception as e:
                    logger.error(f"  ❌ User {user_id}: Error checking token - {e}")

        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Token status check failed: {e}", exc_info=True)


def initialize_token_monitoring():
    """
    Initialize token monitoring for all Fyers users.
    Runs once at startup to enable auto-refresh.
    """
    logger.info("=" * 80)
    logger.info("Initializing Token Monitoring")
    logger.info("=" * 80)

    try:
        from src.services.utils.token_manager_service import get_token_manager
        from src.models.models import BrokerConfiguration

        db_manager = get_database_manager()
        token_manager = get_token_manager()

        with db_manager.get_session() as session:
            # Get all Fyers configurations
            fyers_configs = session.query(BrokerConfiguration).filter_by(
                broker_name='fyers'
            ).all()

            if not fyers_configs:
                logger.info("  ℹ️  No Fyers broker configurations found")
                return

            for config in fyers_configs:
                user_id = config.user_id or 1

                if config.access_token and config.is_connected:
                    try:
                        # Start auto-refresh for this user
                        logger.info(f"  🔄 Starting auto-refresh for user {user_id}...")
                        token_manager.start_auto_refresh(user_id, 'fyers', check_interval_minutes=30)
                        logger.info(f"  ✅ Auto-refresh started for user {user_id}")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Could not start auto-refresh for user {user_id}: {e}")
                else:
                    logger.info(f"  ⏭️  User {user_id}: No active token, skipping auto-refresh")

        logger.info("✅ Token monitoring initialization complete")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Token monitoring initialization failed: {e}", exc_info=True)


def run_scheduler():
    """Main scheduler loop."""
    logger.info("=" * 80)
    logger.info("Trading System Scheduler Started")
    logger.info("=" * 80)
    logger.info("Scheduled Tasks:")
    logger.info("  - Auto-Trading Execution:    Daily at 09:20 AM (5 min after market opens)")
    logger.info("    → Check AI market sentiment, weekly limits, account balance")
    logger.info("    → Auto-place orders with stop-loss and target price")
    logger.info("  - Performance Tracking:      Daily at 06:00 PM (after market close)")
    logger.info("    → Update order performance, create daily snapshots")
    logger.info("    → Check stop-loss/target, close orders if needed")
    logger.info("  - ML Training (ALL 3 MODELS): Daily at 06:00 AM IST")
    logger.info("    → Model 1: Traditional ML (RF + XGBoost)")
    logger.info("    → Model 2: Raw LSTM (Deep Learning)")
    logger.info("    → Model 3: Kronos (K-line Tokenization - on-the-fly)")
    logger.info("  - ML Predictions (ALL STOCKS): Daily at 06:30 AM IST")
    logger.info("    → Predict ALL ~2,259 stocks with all 3 models")
    logger.info("    → Saves to ml_predictions table")
    logger.info("  - Daily Snapshot Update:     Daily at 07:00 AM IST")
    logger.info("    → Ready 2+ hours before market open (9:15 AM IST)")
    logger.info("    → Models: TRADITIONAL + RAW_LSTM + KRONOS (all 3)")
    logger.info("    → Strategies: DEFAULT_RISK + HIGH_RISK (both)")
    logger.info("    → Total: 6 combinations (3 models × 2 strategies)")
    logger.info("    → Pipeline: ALL 3 MODELS use unified 7-step saga")
    logger.info("    → Filtering: Stage 1 (tradeability) + Stage 2 (scoring) for all")
    logger.info("    → Data Check: Kronos pre-filters stocks with <200 days history")
    logger.info("  - Cleanup Old Snapshots:     Weekly (Sunday) at 03:00 AM")
    logger.info("  - Token Status Check:        Every 6 hours (monitors Fyers token)")
    logger.info("=" * 80)

    # Initialize token monitoring on startup
    initialize_token_monitoring()

    # Check if training is needed today and train if necessary
    check_training_needed()

    # Schedule auto-trading at 9:20 AM (5 minutes after market opens at 9:15 AM)
    schedule.every().day.at("09:20").do(execute_auto_trading)

    # Schedule performance tracking at 6:00 PM (after market closes)
    schedule.every().day.at("18:00").do(update_order_performance)

    # Schedule daily ML training at 6:00 AM IST - all 3 models
    # Container timezone is set to Asia/Kolkata, so times are in IST
    schedule.every().day.at("06:00").do(train_all_ml_models)

    # Schedule ML predictions for ALL stocks at 6:30 AM IST - after training
    # Generates predictions for all ~2,259 stocks with all 3 models
    schedule.every().day.at("06:30").do(generate_all_ml_predictions)

    # Schedule daily snapshot update at 7:00 AM IST - after ML predictions
    # This ensures predictions are ready 2+ hours before market open (9:15 AM IST)
    schedule.every().day.at("07:00").do(update_daily_snapshot)

    # Schedule weekly cleanup on Sunday at 3:00 AM
    schedule.every().sunday.at("03:00").do(cleanup_old_snapshots)

    # Schedule token status check every 6 hours (monitoring Fyers token expiry)
    schedule.every(6).hours.do(check_broker_token_status)

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
