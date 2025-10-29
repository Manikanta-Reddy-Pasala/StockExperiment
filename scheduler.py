#!/usr/bin/env python3
"""
Simplified Scheduled Tasks Orchestrator
Runs daily data pipeline and technical indicator calculations at scheduled times.
NO ML TRAINING - Pure technical analysis approach.
"""

import sys
import logging
import schedule
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.database import get_database_manager
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
    """Get the last expected trading day (accounting for weekends)."""
    today = datetime.now().date()

    if today.weekday() == 5:  # Saturday
        return today - timedelta(days=1)
    elif today.weekday() == 6:  # Sunday
        return today - timedelta(days=2)
    else:
        # Weekday - check if market has closed (3:30 PM IST)
        now = datetime.now()
        market_close = now.replace(hour=15, minute=30)

        if now >= market_close:
            return today
        else:
            yesterday = today - timedelta(days=1)
            if yesterday.weekday() == 5:  # Saturday
                return yesterday - timedelta(days=1)
            elif yesterday.weekday() == 6:  # Sunday
                return yesterday - timedelta(days=2)
            return yesterday


def check_data_freshness(max_age_days: int = 3) -> dict:
    """
    Check if historical data is fresh enough for technical indicator calculations.

    Args:
        max_age_days: Maximum acceptable age of data

    Returns:
        dict with freshness status
    """
    try:
        from sqlalchemy import text

        db_manager = get_database_manager()
        with db_manager.get_session() as session:
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

            age_days = (expected_date - last_data_date).days
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


def calculate_technical_indicators():
    """Calculate technical indicators for all stocks (RS Rating, Waves, Signals)."""
    logger.info("\n" + "=" * 80)
    logger.info("CALCULATING TECHNICAL INDICATORS")
    logger.info("=" * 80)

    try:
        from src.services.technical.indicators_calculator import get_indicators_calculator
        from sqlalchemy import text

        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # Get all active tradeable stocks
            query = text("""
                SELECT symbol
                FROM stocks
                WHERE is_active = TRUE
                AND is_tradeable = TRUE
                AND is_suspended = FALSE
                ORDER BY symbol
            """)
            result = session.execute(query)
            symbols = [row[0] for row in result]

            logger.info(f"Found {len(symbols)} tradeable stocks")

            # Calculate indicators
            calculator = get_indicators_calculator(session)
            indicators_results = calculator.calculate_indicators_bulk(symbols, lookback_days=252)

            # Update stocks table with calculated indicators
            update_count = 0
            for symbol, indicators in indicators_results.items():
                update_query = text("""
                    UPDATE stocks
                    SET
                        rs_rating = :rs_rating,
                        fast_wave = :fast_wave,
                        slow_wave = :slow_wave,
                        delta = :delta,
                        buy_signal = :buy_signal,
                        sell_signal = :sell_signal,
                        indicators_last_updated = CURRENT_TIMESTAMP
                    WHERE symbol = :symbol
                """)

                session.execute(update_query, {
                    'symbol': symbol,
                    **indicators
                })
                update_count += 1

            session.commit()

            logger.info(f"✅ Technical indicators calculated and saved for {update_count} stocks")
            logger.info(f"  RS Rating: Relative strength vs NIFTY 50")
            logger.info(f"  Wave Indicators: Fast Wave, Slow Wave, Delta")
            logger.info(f"  Signals: Buy/Sell signals based on wave crossovers")

    except Exception as e:
        logger.error(f"❌ Technical indicator calculation failed: {e}", exc_info=True)


def update_daily_snapshot():
    """Update daily suggested stocks snapshot using technical indicators."""
    logger.info("\n" + "=" * 80)
    logger.info("UPDATING DAILY STOCK PICKS (Technical Indicators)")
    logger.info("=" * 80)

    try:
        from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator

        orchestrator = SuggestedStocksSagaOrchestrator()

        # Run for both strategies
        strategies = ['default_risk', 'high_risk']
        total_stocks_stored = 0

        for strategy in strategies:
            logger.info(f"\n📊 Running {strategy.upper()} strategy...")

            try:
                result = orchestrator.execute_suggested_stocks_saga(
                    user_id=1,
                    strategies=[strategy],
                    limit=50  # Top 50 stocks per strategy
                )

                if result['status'] == 'completed':
                    stocks_count = result['summary'].get('final_result_count', 0)
                    total_stocks_stored += stocks_count
                    logger.info(f"✅ {strategy} strategy completed: {stocks_count} stocks selected")
                else:
                    logger.error(f"❌ {strategy} strategy failed: {result.get('errors', [])}")

            except Exception as e:
                logger.error(f"❌ {strategy} strategy error: {e}", exc_info=True)

        logger.info("\n" + "=" * 80)
        logger.info(f"✅ Daily snapshot update complete!")
        logger.info(f"  Total stocks stored: {total_stocks_stored}")
        logger.info(f"  Strategies: DEFAULT_RISK + HIGH_RISK")
        logger.info(f"  Selection method: Hybrid Strategy (RS Rating + Wave + 8-21 EMA + DeMarker + Fibonacci)")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Daily snapshot update failed: {e}", exc_info=True)


def cleanup_old_snapshots():
    """Clean up old snapshots (runs weekly on Sunday at 3 AM)."""
    logger.info("=" * 80)
    logger.info("CLEANING UP OLD SNAPSHOTS")
    logger.info("=" * 80)

    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            snapshot_service = DailySnapshotService(session)
            deleted = snapshot_service.delete_old_snapshots(keep_days=90)
            logger.info(f"✅ Cleaned up {deleted} old snapshot records (>90 days)")

    except Exception as e:
        logger.error(f"❌ Snapshot cleanup failed: {e}", exc_info=True)


def execute_auto_trading():
    """Execute auto-trading for all enabled users (runs daily at 9:20 AM)."""
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


def check_broker_token_status():
    """Check Fyers broker token status and warn if expiring soon."""
    logger.info("=" * 80)
    logger.info("Checking Broker Token Status")
    logger.info("=" * 80)

    try:
        from src.services.utils.token_manager_service import get_token_manager
        from src.models.models import BrokerConfiguration

        db_manager = get_database_manager()
        token_manager = get_token_manager()

        with db_manager.get_session() as session:
            fyers_configs = session.query(BrokerConfiguration).filter_by(
                broker_name='fyers'
            ).all()

            if not fyers_configs:
                logger.info("  ℹ️  No Fyers broker configurations found")
                return

            for config in fyers_configs:
                user_id = config.user_id or 1

                try:
                    status = token_manager.get_token_status(user_id, 'fyers')

                    if not status['has_token']:
                        logger.warning(f"  ⚠️  User {user_id}: No token found - re-authentication required")
                        continue

                    if status['is_expired']:
                        logger.error(f"  ❌ User {user_id}: Token EXPIRED - re-authentication required!")
                        logger.error(f"     Please login to Fyers at: http://localhost:5001/brokers/fyers")
                        config.is_connected = False
                        config.connection_status = 'reauth_required'
                        session.commit()
                        continue

                    if status['expires_at']:
                        expiry_time = datetime.fromisoformat(status['expires_at'])
                        time_until_expiry = expiry_time - datetime.now()
                        hours_until_expiry = time_until_expiry.total_seconds() / 3600

                        if hours_until_expiry < 12:
                            logger.warning(f"  ⚠️  User {user_id}: Token expires in {hours_until_expiry:.1f} hours!")
                        else:
                            logger.info(f"  ✅ User {user_id}: Token valid for {hours_until_expiry:.1f} hours")

                        if not status['auto_refresh_active']:
                            logger.info(f"  🔄 User {user_id}: Starting auto-refresh monitoring...")
                            token_manager.start_auto_refresh(user_id, 'fyers', check_interval_minutes=30)

                except Exception as e:
                    logger.error(f"  ❌ User {user_id}: Error checking token - {e}")

        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Token status check failed: {e}", exc_info=True)


def initialize_token_monitoring():
    """Initialize token monitoring for all Fyers users."""
    logger.info("=" * 80)
    logger.info("Initializing Token Monitoring")
    logger.info("=" * 80)

    try:
        from src.services.utils.token_manager_service import get_token_manager
        from src.models.models import BrokerConfiguration

        db_manager = get_database_manager()
        token_manager = get_token_manager()

        with db_manager.get_session() as session:
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
    logger.info("📊 SIMPLIFIED TRADING SYSTEM SCHEDULER STARTED")
    logger.info("=" * 80)
    logger.info("Scheduled Tasks:")
    logger.info("  - Auto-Trading Execution:      Daily at 09:20 AM (5 min after market opens)")
    logger.info("  - Performance Tracking:        Daily at 06:00 PM (after market close)")
    logger.info("  - Technical Indicators:        Daily at 10:00 PM (after data pipeline)")
    logger.info("  - Daily Stock Picks:           Daily at 10:15 PM (after indicators)")
    logger.info("  - Cleanup Old Snapshots:       Weekly (Sunday) at 03:00 AM")
    logger.info("  - Token Status Check:          Every 6 hours")
    logger.info("")
    logger.info("📈 SELECTION METHOD: Technical Indicators Only")
    logger.info("  - RS Rating: Relative strength vs NIFTY 50")
    logger.info("  - Wave Indicators: Fast/Slow wave momentum")
    logger.info("  - Buy/Sell Signals: Wave crossovers")
    logger.info("  - NO ML MODELS - Pure technical analysis")
    logger.info("=" * 80)

    # Initialize token monitoring on startup
    initialize_token_monitoring()

    # Check data freshness
    freshness = check_data_freshness(max_age_days=3)
    logger.info(f"\n{freshness['message']}\n")

    # Schedule auto-trading at 9:20 AM
    schedule.every().day.at("09:20").do(execute_auto_trading)

    # Schedule performance tracking at 6:00 PM
    schedule.every().day.at("18:00").do(update_order_performance)

    # Schedule technical indicators calculation at 10:00 PM (after data pipeline runs at 9:00 PM)
    schedule.every().day.at("22:00").do(calculate_technical_indicators)

    # Schedule daily snapshot update at 10:15 PM (after indicators calculated)
    schedule.every().day.at("22:15").do(update_daily_snapshot)

    # Schedule weekly cleanup on Sunday at 3:00 AM
    schedule.every().sunday.at("03:00").do(cleanup_old_snapshots)

    # Schedule token status check every 6 hours
    schedule.every(6).hours.do(check_broker_token_status)

    # Keep scheduler running
    logger.info("✅ Scheduler is now running. Press Ctrl+C to stop.\n")
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
