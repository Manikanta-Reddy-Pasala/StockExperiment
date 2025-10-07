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
from src.services.ml.stock_predictor import StockMLPredictor
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor
from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator
from src.services.data.daily_snapshot_service import DailySnapshotService

# Configuration: Choose which ML predictor to use
USE_ENHANCED_MODEL = True  # Set to False to use original model

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


def train_ml_models():
    """Daily ML model training task (runs at 2 AM)."""
    logger.info("=" * 80)
    logger.info("Starting Scheduled ML Model Training")
    logger.info("=" * 80)

    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            if USE_ENHANCED_MODEL:
                logger.info("Using ENHANCED ML Predictor (RF + XGBoost + Chaos Features)")
                predictor = EnhancedStockPredictor(session)

                # Train with walk-forward validation
                logger.info("Training enhanced models with 365 days + walk-forward CV...")
                stats = predictor.train_with_walk_forward(lookback_days=365, n_splits=5)

                logger.info("ML Training Complete!")
                logger.info(f"  Training Samples: {stats['samples']:,}")
                logger.info(f"  Features Used: {stats['features']}")
                logger.info(f"  Price Model R²: {stats['price_r2']:.4f}")
                logger.info(f"  Risk Model R²: {stats['risk_r2']:.4f}")
                logger.info(f"  CV Price R²: {stats['cv_price_r2']:.4f} (walk-forward)")
                logger.info(f"  CV Risk R²: {stats['cv_risk_r2']:.4f} (walk-forward)")
                logger.info(f"  Top Features: {', '.join(stats['top_features'][:5])}")
                logger.info("✅ Enhanced ML models trained successfully")
            else:
                logger.info("Using ORIGINAL ML Predictor (RF only)")
                predictor = StockMLPredictor(session)

                # Train models with 1 year of historical data
                logger.info("Training ML models with 365 days of historical data...")
                stats = predictor.train(lookback_days=365)

                logger.info("ML Training Complete!")
                logger.info(f"  Training Samples: {stats['samples']:,}")
                logger.info(f"  Features Used: {stats['features']}")
                logger.info(f"  Price Model R²: {stats['price_r2']:.4f}")
                logger.info(f"  Risk Model R²: {stats['risk_r2']:.4f}")
                logger.info("✅ ML models trained successfully")

    except Exception as e:
        logger.error(f"❌ ML training failed: {e}", exc_info=True)


def update_daily_snapshot():
    """Update daily suggested stocks snapshot (runs at 2:15 AM, after ML training)."""
    logger.info("=" * 80)
    logger.info("Starting Daily Suggested Stocks Update")
    logger.info("=" * 80)
    
    try:
        # Run suggested stocks saga to get fresh recommendations
        orchestrator = SuggestedStocksSagaOrchestrator()
        result = orchestrator.execute_suggested_stocks_saga(
            user_id=1,
            strategies=['default_risk'],
            limit=50  # Store top 50 daily picks
        )
        
        if result['status'] == 'completed':
            logger.info("✅ Daily snapshot updated successfully at 2:15 AM")
            logger.info(f"  Total stocks stored: {result['summary'].get('final_result_count', 0)}")
            
            # Check if ML predictions were applied
            ml_step = next((s for s in result['summary']['step_summary'] 
                          if s['step_id'] == 'step6_ml_prediction'), None)
            if ml_step and ml_step['status'] == 'completed':
                logger.info(f"  ML predictions applied to all stocks")
            
            # Check snapshot save
            snapshot_step = next((s for s in result['summary']['step_summary'] 
                                if s['step_id'] == 'step7_daily_snapshot'), None)
            if snapshot_step and snapshot_step['status'] == 'completed':
                logger.info(f"  Snapshot saved: {snapshot_step.get('metadata', {})}")
        else:
            logger.error(f"❌ Daily snapshot update failed: {result.get('errors', [])}")
            
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


def run_scheduler():
    """Main scheduler loop."""
    logger.info("=" * 80)
    logger.info("Trading System Scheduler Started")
    logger.info("=" * 80)
    logger.info("Scheduled Tasks:")
    logger.info("  - ML Training:           Daily at 10:00 PM (after data pipeline)")
    logger.info("  - Daily Snapshot Update: Daily at 10:15 PM (after ML training)")
    logger.info("  - Cleanup Old Snapshots: Weekly (Sunday) at 03:00 AM")
    logger.info("=" * 80)

    # Schedule daily ML training at 10:00 PM (after data pipeline completes)
    schedule.every().day.at("22:00").do(train_ml_models)

    # Schedule daily snapshot update at 10:15 PM (after ML training)
    schedule.every().day.at("22:15").do(update_daily_snapshot)

    # Schedule weekly cleanup on Sunday at 3:00 AM
    schedule.every().sunday.at("03:00").do(cleanup_old_snapshots)
    
    # Run immediately on startup (optional - for testing)
    # Uncomment these lines to run tasks immediately when scheduler starts
    # logger.info("Running initial ML training...")
    # train_ml_models()
    # time.sleep(5)
    # logger.info("Running initial snapshot update...")
    # update_daily_snapshot()
    
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
