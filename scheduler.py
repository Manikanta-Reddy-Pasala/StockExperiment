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
from src.services.brokers.fyers_token_refresh import FyersTokenRefreshService
# EMA crossover runner removed (rejected model). Model 3 momentum rotation
# is invoked as subprocess via tools/models/momentum_n100_top5_max1/live_signal.py.
import subprocess

# Configure logging with rotation (max 50MB per file, keep 5 backups)
import os
from logging.handlers import RotatingFileHandler

_log_handlers = [logging.StreamHandler()]
try:
    os.makedirs('logs', exist_ok=True)
    _log_handlers.append(
        RotatingFileHandler('logs/scheduler.log', maxBytes=50*1024*1024, backupCount=5)
    )
except (PermissionError, OSError) as _log_err:
    # Log directory not writable (e.g. mounted volume owned by another user).
    # Fall back to stdout-only logging so the scheduler still starts.
    print(f"WARNING: Cannot write to logs/scheduler.log ({_log_err}). Logging to stdout only.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=_log_handlers
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


# Model 3 trading-side jobs are defined in
# tools/models/momentum_n100_top5_max1/cron.py and registered below in
# run_scheduler() via register_trading_jobs(schedule).


def cleanup_old_snapshots():
    """Delete suggested-stocks rows older than 90 days (runs Sunday at 03:00)."""
    logger.info("=" * 80)
    logger.info("CLEANING UP OLD SNAPSHOTS")
    logger.info("=" * 80)

    try:
        from sqlalchemy import text

        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            res = session.execute(
                text(
                    """
                    DELETE FROM daily_suggested_stocks
                    WHERE date < CURRENT_DATE - INTERVAL '90 days'
                    """
                )
            )
            deleted = res.rowcount or 0
            session.commit()
            logger.info(f"✅ Cleaned up {deleted} old snapshot rows (>90 days)")
    except Exception as e:
        logger.error(f"❌ Snapshot cleanup failed: {e}", exc_info=True)


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


def refresh_all_fyers_tokens():
    """Refresh FYERS tokens for all users using the v3 API (no browser needed)."""
    logger.info("=" * 80)
    logger.info("Starting API-based FYERS Token Refresh")
    logger.info("=" * 80)

    try:
        from src.models.models import BrokerConfiguration
        from src.services.utils.token_manager_service import get_token_manager

        db_manager = get_database_manager()
        refresh_service = FyersTokenRefreshService()
        token_manager = get_token_manager()

        with db_manager.get_session() as session:
            fyers_configs = session.query(BrokerConfiguration).filter_by(
                broker_name='fyers'
            ).all()

            if not fyers_configs:
                logger.info("  No Fyers configurations found")
                return

            for config in fyers_configs:
                user_id = config.user_id or 1

                try:
                    # Check if token needs refresh
                    status = token_manager.get_token_status(user_id, 'fyers')

                    if not status.get('has_token'):
                        logger.warning(f"  User {user_id}: No token - needs initial OAuth login")
                        continue

                    needs_refresh = False
                    if status.get('is_expired'):
                        logger.info(f"  User {user_id}: Token expired, refreshing...")
                        needs_refresh = True
                    elif status.get('expires_at'):
                        expiry_time = datetime.fromisoformat(status['expires_at'])
                        hours_until_expiry = (expiry_time - datetime.now()).total_seconds() / 3600
                        if hours_until_expiry < 6:
                            logger.info(f"  User {user_id}: Token expiring in {hours_until_expiry:.1f}h, refreshing...")
                            needs_refresh = True
                        else:
                            logger.info(f"  User {user_id}: Token valid for {hours_until_expiry:.1f}h, skipping")

                    if needs_refresh:
                        result = refresh_service.refresh_fyers_token(user_id, config.refresh_token)
                        if result:
                            logger.info(f"  User {user_id}: Token refreshed successfully via API")
                        else:
                            logger.error(f"  User {user_id}: API refresh failed - may need manual re-auth")

                except Exception as e:
                    logger.error(f"  User {user_id}: Error during refresh - {e}")

    except Exception as e:
        logger.error(f"Token refresh failed: {e}", exc_info=True)

    logger.info("=" * 80)


def run_scheduler():
    """Main scheduler loop."""
    logger.info("=" * 80)
    logger.info("📊 EMA 200/400 CROSSOVER (1H) TRADING SCHEDULER")
    logger.info("=" * 80)
    logger.info("Scheduled Tasks:")
    logger.info("  - Auto-Trading Execution:      Daily at 09:20 AM")
    logger.info("  - Strategy Run (1H close):     10:16, 11:16, 12:16, 13:16, 14:16, 15:31")
    logger.info("  - 15m Sustain Check:           every 15m during market hours")
    logger.info("  - Position Monitoring:         10:00, 11:00, 12:00, 13:00, 14:00, 15:15")
    logger.info("  - Day Trade Close:             Daily at 03:20 PM")
    logger.info("  - Performance Reconciliation:  Daily at 06:00 PM")
    logger.info("  - Cleanup Old Snapshots:       Weekly (Sunday) at 03:00 AM")
    logger.info("  - Token Status Check:          Every 6 hours")
    logger.info("")
    logger.info("📈 EMA 200/400 1H CROSSOVER STRATEGY")
    logger.info("  - Trend: EMA200 vs EMA400 on 1H closes")
    logger.info("  - Alerts: 1) Crossover-candle break  2) EMA200 retest  3) EMA400 touch")
    logger.info("  - Entries: Break of retest candle high (long) / low (short)")
    logger.info("  - Stop:    1H close on the wrong side of EMA400")
    logger.info("  - Target:  +/-5000 pts for indices, 1:3 RR for equities")
    logger.info("  - NO ML MODELS - Pure technical analysis")
    logger.info("=" * 80)

    # Initialize token monitoring on startup
    initialize_token_monitoring()

    # Check data freshness
    freshness = check_data_freshness(max_age_days=3)
    logger.info(f"\n{freshness['message']}\n")

    # Per-model trading-side jobs (signal + execute). Add new models by
    # creating tools/models/<name>/cron.py with a register_trading_jobs()
    # function, then add an import + register call below.
    from tools.models.momentum_n100_top5_max1.cron import (
        register_trading_jobs as register_momentum_n100_jobs,
    )
    from tools.models.midcap_narrow_60d_breakout.cron import (
        register_trading_jobs as register_midcap_narrow_jobs,
    )
    register_momentum_n100_jobs(schedule)
    register_midcap_narrow_jobs(schedule)
    # FinNifty IC is unwired — no trading jobs registered.

    # Schedule weekly cleanup on Sunday at 3:00 AM
    schedule.every().sunday.at("03:00").do(cleanup_old_snapshots)

    # Schedule token status check every 6 hours
    schedule.every(6).hours.do(check_broker_token_status)

    # Schedule API-based token refresh every 5 hours
    schedule.every(5).hours.do(refresh_all_fyers_tokens)

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
