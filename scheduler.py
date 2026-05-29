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

    # Also prune the cross-model overlap analysis artefacts the operator may
    # have generated from analyze_model_overlap.py / combined_portfolio_sim.py
    # — they're regenerable JSON, no need to keep more than two weeks of runs.
    try:
        import os, time
        cutoff = time.time() - 14 * 86400
        overlap_dir = "/app/exports/overlap"
        if os.path.isdir(overlap_dir):
            n = 0
            for root, _, files in os.walk(overlap_dir):
                for fn in files:
                    p = os.path.join(root, fn)
                    try:
                        if os.path.getmtime(p) < cutoff:
                            os.remove(p); n += 1
                    except OSError:
                        pass
            if n:
                logger.info(f"✅ Pruned {n} stale overlap-analysis file(s) (>14 days)")
    except Exception as e:
        logger.warning(f"overlap-artefact prune skipped: {e}")


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


# Canonical morning signal files written by each model's emit_signal (cron.py).
# Used ONLY for boot-time miss DETECTION (M3) — keep in sync with the per-model
# cron.py SIGNALS_DIR + filename. {today} is substituted at check time.
_MODEL_SIGNAL_FILES = {
    "momentum_n100_top5_max1":   "/app/logs/momrot/signals/{today}_momrot_n100.json",
    "momentum_pseudo_n100_adv":  "/app/logs/momrot_pseudo/signals/{today}_pseudo_n100.json",
    "midcap_narrow_60d_breakout": "/app/logs/midcap_narrow/signals/{today}_midcap_narrow.json",
    "n20_daily_large_only":      "/app/logs/n20_daily/signals/{today}_n20.json",
}


def _detect_missed_trade_window():
    """Boot-time SAFE catch-up DETECTION (M3) — alert only, never auto-trade.

    The `schedule` library does NOT run missed jobs. A container restart at,
    say, 09:31 IST silently drops that day's 09:25 emit + 09:30 execute with no
    catch-up. Auto-PLACING orders on restart is unsafe (we can't know what
    already executed before the crash), so this only DETECTS a likely miss and
    sends ONE Telegram alert for a human to check.

    Conditions to alert (all must hold):
      - today is an NSE trading day, AND
      - now > 09:35 IST (the morning emit 09:25 + execute 09:30/09:32 window
        has passed), AND
      - for at least one model: its canonical signal file for today is MISSING
        (emit never ran), OR a signal file exists but NO audit_orders row was
        placed today (execute never ran).

    We deliberately do NOT re-run emit here: before 09:35 the registered 09:25
    job still fires normally, and after the window a fresh emit could mislead
    (it would scan against an intra-day state). execute_orders is NEVER invoked
    from catch-up.
    """
    try:
        from tools.shared.nse_calendar import is_trading_day
    except Exception as e:
        logger.warning(f"M3 catch-up: nse_calendar import failed ({e}); skipping")
        return
    now = datetime.now()
    if not is_trading_day(now):
        return
    # 09:35 IST = end of the morning emit(09:25)+execute(09:30/09:32) window.
    window_end_min = 9 * 60 + 35
    if (now.hour * 60 + now.minute) <= window_end_min:
        return  # still before/within the window — registered jobs will run.

    today = now.strftime("%Y-%m-%d")
    # Best-effort: which models placed an order today (DB). Empty set on any
    # failure — detection then relies on the signal-file presence alone.
    ordered_models = set()
    try:
        from src.models.database import get_database_manager
        from src.models.audit_models import AuditOrder
        from sqlalchemy import func
        db = get_database_manager()
        with db.get_session() as s:
            rows = (s.query(AuditOrder.model_name)
                    .filter(func.date(AuditOrder.placed_at) == now.date())
                    .distinct().all())
            ordered_models = {r[0] for r in rows if r[0]}
    except Exception as e:
        logger.warning(f"M3 catch-up: audit_orders lookup failed ({e}); "
                       "using signal-file check only")

    suspects = []
    for model, tmpl in _MODEL_SIGNAL_FILES.items():
        sig_path = Path(tmpl.format(today=today))
        if not sig_path.exists():
            suspects.append(f"{model}: morning signal file MISSING")
        elif model not in ordered_models:
            # Signal emitted but no order row today — execute may have been
            # skipped. NOTE: legitimately no-trade days (no rotation) also have
            # a signal file + no order; this is a soft heuristic, hence "check".
            suspects.append(f"{model}: signal present but no order placed today")

    if not suspects:
        logger.info("M3 catch-up: morning window passed, all models look intact")
        return

    msg = (
        "⚠️ *Scheduler restarted AFTER the trade window*\n"
        f"Time now: {now.strftime('%H:%M')} IST (> 09:35). The `schedule` lib "
        "does not run missed jobs, so today's signals/orders MAY have been "
        "missed. *No orders were auto-placed* (unsafe on restart). Manual check "
        "needed:\n" + "\n".join(f"- {s}" for s in suspects)
    )
    logger.warning(msg.replace("*", ""))
    try:
        from tools.live.telegram_notify import send as _tg
        _tg(msg)
    except Exception as e:
        logger.error(f"M3 catch-up: telegram alert failed: {e}")


def _assert_ist_or_die():
    """Refuse to start unless the process timezone is IST (UTC+05:30).

    Every schedule.at("HH:MM") job and every date gate in this system assumes
    the container clock is Asia/Kolkata. If the TZ is wrong, jobs fire at the
    wrong wall-clock time (e.g. a 09:30 execute would run hours off) and date
    gates misfire — silently trading at the wrong moment. NOT trading is far
    safer than trading at the wrong time, so on a mismatch we alert and exit(1)
    rather than continue.
    """
    offset = datetime.now().astimezone().utcoffset()
    if offset != timedelta(hours=5, minutes=30):
        msg = (
            "🛑 *technical_scheduler REFUSING TO START* — process timezone is "
            f"NOT IST. utcoffset={offset} (expected +05:30, %z="
            f"{time.strftime('%z')}). Every job time + date gate assumes "
            "Asia/Kolkata; running with the wrong TZ would fire orders at the "
            "wrong time. Fix TZ=Asia/Kolkata and restart."
        )
        logger.critical(msg)
        try:
            from tools.live.telegram_notify import send as _tg
            _tg(msg)
        except Exception as _e:
            logger.error(f"tg alert on TZ-exit failed: {_e}")
        sys.exit(1)
    logger.info(f"TZ check OK: utcoffset=+05:30 (%z={time.strftime('%z')})")


def run_scheduler():
    """Main scheduler loop."""
    # Fail-safe: verify IST before registering any time-gated jobs.
    _assert_ist_or_die()
    logger.info("=" * 80)
    logger.info("📊 TECHNICAL SCHEDULER — per-model trading jobs")
    logger.info("=" * 80)
    logger.info("Registered models (trading-side):")
    logger.info("  - momentum_n100_top5_max1:    signal 09:25 + execute 09:30 (always live)")
    logger.info("  - momentum_pseudo_n100_adv:   signal 09:25 + execute 09:30 (monthly rebal)")
    logger.info("  - midcap_narrow_60d_breakout: signal 09:25 + execute 09:32 + EOD signal 15:25")
    logger.info("  - n20_daily_large_only:       signal 09:25 + execute 09:30 (daily rotation)")
    logger.info("")
    logger.info("Maintenance:")
    logger.info("  - Cleanup Old Snapshots: Weekly (Sunday) at 03:00 AM")
    logger.info("  - Token Status Check:    Every 6 hours")
    logger.info("  - Fyers Token Refresh:   Every 5 hours")
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
    from tools.models.momentum_pseudo_n100_adv.cron import (
        register_trading_jobs as register_pseudo_n100_jobs,
    )
    from tools.models.midcap_narrow_60d_breakout.cron import (
        register_trading_jobs as register_midcap_narrow_jobs,
    )
    from tools.models.n20_daily_large_only.cron import (
        register_trading_jobs as register_n20_daily_jobs,
    )
    # Multi-holding model (K=3). Trading jobs run daily but live_signal.py no-ops
    # while model_settings.enabled is False — keep it disabled until the position
    # reconciler is made multi-holding-aware (it currently mirrors a single
    # position per model and would mishandle this model's 3 holdings).
    from tools.models.momentum_retest_n500.cron import (
        register_trading_jobs as register_mr500_jobs,
        register_data_jobs as register_mr500_data_jobs,
    )
    register_momentum_n100_jobs(schedule)
    register_pseudo_n100_jobs(schedule)
    register_midcap_narrow_jobs(schedule)
    register_n20_daily_jobs(schedule)
    register_mr500_jobs(schedule)
    register_mr500_data_jobs(schedule)

    # M3 — SAFE catch-up: if we restarted after the morning trade window on a
    # trading day, ALERT (never auto-execute) so a human can check for misses.
    _detect_missed_trade_window()

    # Position reconciler — mirrors Fyers truth into model_ledger every 5 min
    # during market hours (09:30–15:30 IST). Catches drift when record_buy /
    # record_sell silently miss (status-mapping bugs, executor crashes,
    # external trades). Auto-fixes safe drift, alerts on unsafe.
    def _reconcile_market_hours():
        from datetime import datetime as _dt
        now = _dt.now()
        # Weekday + NSE-trading-day + 09:30-15:30 IST window (container TZ
        # assumed IST/UTC+5:30). The NSE-holiday guard catches Bakri Id /
        # Muharram / etc. where the weekday-only check would still fire a
        # 5-min loop into a closed market — Fyers quotes() would 200-OK with
        # SYMBOL_NOT_FOUND and the reconciler would spam noise.
        if now.weekday() >= 5:
            return
        try:
            from tools.shared.nse_calendar import is_trading_day
            if not is_trading_day(now.date()):
                return
        except Exception:
            # Fail-open: if the calendar module can't be reached, fall back to
            # the weekday-only gate so the reconciler still runs on normal days.
            pass
        hm = now.hour * 60 + now.minute
        # Start at 09:45, NOT 09:30 — executes run 09:30-09:35 (+ n100 mid-month
        # 09:35). Reconciling during placement races record_buy/record_sell and
        # can clobber ledger cash/qty mid-cycle. Reconcile only after they settle.
        if hm < (9 * 60 + 45) or hm > (15 * 60 + 30):
            return
        try:
            import subprocess
            r = subprocess.run(
                ["python3", "tools/live/position_reconciler.py", "--tg-on-fix"],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode != 0:
                logger.error(f"reconciler exit={r.returncode}: {r.stderr[-400:]}")
            elif r.stdout:
                logger.info(r.stdout[-400:])
        except Exception as e:
            logger.error(f"reconciler call failed: {e}")

    schedule.every(5).minutes.do(_reconcile_market_hours)

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
