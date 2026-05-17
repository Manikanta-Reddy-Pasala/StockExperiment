#!/usr/bin/env python3
"""
Data Pipeline Scheduler
Runs data updates, CSV pulls, stock history fetches, and calculations at scheduled times.
"""

import sys
import logging
import schedule
import time
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging with rotation (max 50MB per file, keep 5 backups)
import os
from logging.handlers import RotatingFileHandler

_log_handlers = [logging.StreamHandler()]
try:
    os.makedirs('logs', exist_ok=True)
    _log_handlers.append(
        RotatingFileHandler('logs/data_scheduler.log', maxBytes=50*1024*1024, backupCount=5)
    )
except (PermissionError, OSError) as _log_err:
    print(f"WARNING: Cannot write to logs/data_scheduler.log ({_log_err}). Logging to stdout only.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=_log_handlers
)
logger = logging.getLogger(__name__)


def _run_subprocess_with_retry(cmd: list, label: str, timeout: int = 3600, max_retries: int = 2):
    """Run a subprocess with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Running {label} (attempt {attempt}/{max_retries})...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                logger.info(f"  {label} completed successfully")
                if result.stdout:
                    # Log last 20 lines to avoid flooding
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 20:
                        logger.info(f"  Output (last 20 lines):\n{''.join(lines[-20:])}")
                    else:
                        logger.info(f"  Output:\n{result.stdout}")
                return True
            else:
                logger.error(f"  {label} failed (return code {result.returncode})")
                if result.stderr:
                    logger.error(f"  Error:\n{result.stderr[-500:]}")
                if attempt < max_retries:
                    import time as _time
                    wait = 30 * attempt
                    logger.info(f"  Retrying in {wait}s...")
                    _time.sleep(wait)

        except subprocess.TimeoutExpired:
            logger.error(f"  {label} timeout after {timeout}s")
        except Exception as e:
            logger.error(f"  {label} error: {e}", exc_info=True)

    logger.error(f"  {label} failed after {max_retries} attempts")
    return False


def run_data_pipeline():
    """Run complete data pipeline (Daily at 9:00 PM after market close)."""
    logger.info("=" * 80)
    logger.info("Starting Data Pipeline (4-Step Saga)")
    logger.info("=" * 80)
    _run_subprocess_with_retry(['python3', 'run_pipeline.py'], 'Data Pipeline', timeout=3600, max_retries=2)


def backfill_full_history():
    """Ensure ALL NSE-EQ stocks have 4 years (1500d) of daily OHLCV.

    Uses existing tools/shared/prefetch_ohlcv.py --universe all (every
    NSE:...-EQ symbol from the stocks master table) with --skip-frac=0.85.
    Idempotent — re-running on a complete cache is cheap (per-symbol
    coverage check is a single SELECT COUNT(*)).

    Daily-only by design. No live trading model uses hourly (1h) bars.
    Backfilling 1h for 2400+ symbols would 25x the Fyers API calls for
    zero trading benefit.

    Runs weekly (Sunday 03:00 IST — before any market activity) and
    once on scheduler startup if env BACKFILL_ON_BOOT=true.

    Daily incremental pulls (per-model data_pull at 20:45) keep the
    latest 2 days fresh. This job only fills HISTORICAL gaps.
    """
    logger.info("=" * 80)
    logger.info("Full History Backfill — 4 years (1500d) Daily for ALL NSE-EQ")
    logger.info("=" * 80)
    _run_subprocess_with_retry(
        ['python3', 'tools/shared/prefetch_ohlcv.py',
         '--universe', 'all',
         '--days', '1500',
         '--intervals', 'D',
         '--sleep', '0.15',
         '--retry-passes', '2'],
        'backfill_4y_history',
        timeout=21600,  # 6 hours — ~2400 syms, mostly cached after first run
        max_retries=1,
    )



def export_daily_csv():
    """
    Export daily data to CSV (Daily at 10:00 PM).
    Creates CSV files with latest stock data for backup/analysis.
    """
    logger.info("=" * 80)
    logger.info("Exporting Daily CSV Files")
    logger.info("=" * 80)
    
    try:
        from src.models.database import get_database_manager
        import pandas as pd
        import os
        
        # Create exports directory
        export_dir = Path('exports')
        export_dir.mkdir(exist_ok=True)
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # Export stocks data
            logger.info("Exporting stocks data...")
            stocks_query = """
                SELECT symbol, name, current_price, market_cap, pe_ratio, pb_ratio, roe, 
                       eps, book_value, beta, peg_ratio, roa, debt_to_equity,
                       current_ratio, quick_ratio, revenue_growth, earnings_growth,
                       operating_margin, net_margin, profit_margin, dividend_yield,
                       volume, sector, market_cap_category, last_updated
                FROM stocks
                WHERE current_price IS NOT NULL
                ORDER BY market_cap DESC NULLS LAST
            """
            stocks_df = pd.read_sql(stocks_query, session.connection())
            stocks_file = export_dir / f'stocks_{today}.csv'
            stocks_df.to_csv(stocks_file, index=False)
            logger.info(f"  ✅ Exported {len(stocks_df)} stocks to {stocks_file}")
            
            # Export historical data (last 30 days)
            logger.info("Exporting recent historical data (30 days)...")
            history_query = """
                SELECT symbol, date, open, high, low, close, adj_close, volume,
                       data_source, price_change_pct
                FROM historical_data
                WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY symbol, date DESC
            """
            history_df = pd.read_sql(history_query, session.connection())
            history_file = export_dir / f'historical_30d_{today}.csv'
            history_df.to_csv(history_file, index=False)
            logger.info(f"  ✅ Exported {len(history_df)} historical records to {history_file}")
            
            # Export technical indicators (latest)
            logger.info("Exporting latest technical indicators...")
            tech_query = """
                SELECT DISTINCT ON (symbol)
                    symbol, date, sma_50, sma_200
                FROM technical_indicators
                ORDER BY symbol, date DESC
            """
            tech_df = pd.read_sql(tech_query, session.connection())
            tech_file = export_dir / f'technical_indicators_{today}.csv'
            tech_df.to_csv(tech_file, index=False)
            logger.info(f"  ✅ Exported {len(tech_df)} technical indicators to {tech_file}")
            
            # Export suggested stocks (today)
            logger.info("Exporting today's suggested stocks...")
            suggested_query = """
                SELECT date, symbol, stock_name, current_price, market_cap,
                       strategy, selection_score, rank,
                       rsi_14, macd, sma_50, sma_200,
                       pe_ratio, pb_ratio, roe, eps, beta,
                       revenue_growth, earnings_growth, operating_margin,
                       target_price, stop_loss, recommendation, reason,
                       sector, market_cap_category
                FROM daily_suggested_stocks
                WHERE date = CURRENT_DATE
                ORDER BY rank
            """
            suggested_df = pd.read_sql(suggested_query, session.connection())
            if len(suggested_df) > 0:
                suggested_file = export_dir / f'suggested_stocks_{today}.csv'
                suggested_df.to_csv(suggested_file, index=False)
                logger.info(f"  ✅ Exported {len(suggested_df)} suggested stocks to {suggested_file}")
            else:
                logger.warning("  ⚠️  No suggested stocks found for today")
        
        logger.info("✅ CSV export completed successfully at 10:00 PM")
        
        # Cleanup old CSV files (keep last 90 days for extended testing)
        logger.info("Cleaning up old CSV files (>90 days)...")
        cleanup_old_csv_files(export_dir, keep_days=90)
        
    except Exception as e:
        logger.error(f"❌ CSV export failed: {e}", exc_info=True)


def cleanup_old_csv_files(export_dir: Path, keep_days: int = 30):
    """Delete CSV files older than keep_days."""
    try:
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        deleted_count = 0
        for csv_file in export_dir.glob('*.csv'):
            # Get file modification time
            mtime = datetime.fromtimestamp(csv_file.stat().st_mtime)
            if mtime < cutoff_date:
                csv_file.unlink()
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"  ✅ Deleted {deleted_count} old CSV files (>{keep_days} days)")
        else:
            logger.info(f"  ℹ️  No old CSV files to delete")
            
    except Exception as e:
        logger.error(f"  ❌ CSV cleanup failed: {e}")


def update_symbol_master():
    """
    Update symbol master CSV from NSE (Weekly on Monday at 6:00 AM).
    Refreshes the complete list of tradeable NSE symbols.
    """
    logger.info("=" * 80)
    logger.info("Updating Symbol Master from NSE")
    logger.info("=" * 80)
    
    try:
        from src.services.data.fyers_symbol_service import FyersSymbolService
        from src.models.database import get_database_manager
        
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            symbol_service = FyersSymbolService()
            
            # Refresh symbol master
            logger.info("Fetching latest NSE symbols from Fyers...")
            result = symbol_service.refresh_all_symbols(sync_to_database=True)

            logger.info(f"Symbol master updated successfully")
            logger.info(f"  Result: {result}")
            
    except Exception as e:
        logger.error(f"❌ Symbol master update failed: {e}", exc_info=True)


def validate_data_quality():
    """
    Validate data quality and generate report (Daily at 10:30 PM).
    Checks for missing data, anomalies, and data consistency.
    """
    logger.info("=" * 80)
    logger.info("Validating Data Quality")
    logger.info("=" * 80)

    try:
        from src.models.database import get_database_manager
        from sqlalchemy import text

        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # Check stocks table
            stocks_stats = session.execute(text("""
                SELECT
                    COUNT(*) as total_stocks,
                    COUNT(current_price) as with_price,
                    COUNT(market_cap) as with_market_cap,
                    COUNT(pe_ratio) as with_pe,
                    COUNT(eps) as with_eps,
                    COUNT(sector) as with_sector
                FROM stocks
            """)).fetchone()

            total_stocks = stocks_stats.total_stocks if stocks_stats else 0

            logger.info("Stocks Table:")
            logger.info(f"  Total stocks: {total_stocks}")
            if total_stocks > 0:
                logger.info(f"  With price: {stocks_stats.with_price} ({stocks_stats.with_price/total_stocks*100:.1f}%)")
                logger.info(f"  With market cap: {stocks_stats.with_market_cap} ({stocks_stats.with_market_cap/total_stocks*100:.1f}%)")
                logger.info(f"  With PE ratio: {stocks_stats.with_pe} ({stocks_stats.with_pe/total_stocks*100:.1f}%)")
                logger.info(f"  With EPS: {stocks_stats.with_eps} ({stocks_stats.with_eps/total_stocks*100:.1f}%)")
                logger.info(f"  With sector: {stocks_stats.with_sector} ({stocks_stats.with_sector/total_stocks*100:.1f}%)")
            else:
                logger.warning("  ⚠️ No stocks data available")

            # Check historical data
            history_stats = session.execute(text("""
                SELECT
                    COUNT(DISTINCT symbol) as symbols_with_history,
                    COUNT(*) as total_records,
                    MAX(date) as latest_date,
                    MIN(date) as earliest_date
                FROM historical_data
            """)).fetchone()

            logger.info("\nHistorical Data:")
            logger.info(f"  Symbols with history: {history_stats.symbols_with_history}")
            logger.info(f"  Total records: {history_stats.total_records:,}")
            logger.info(f"  Date range: {history_stats.earliest_date} to {history_stats.latest_date}")

            # Check technical indicators
            tech_stats = session.execute(text("""
                SELECT
                    COUNT(DISTINCT symbol) as symbols_with_tech,
                    COUNT(*) as total_records,
                    COUNT(sma_50) as with_sma50,
                    COUNT(sma_200) as with_sma200
                FROM technical_indicators
            """)).fetchone()

            logger.info("\nTechnical Indicators:")
            logger.info(f"  Symbols with indicators: {tech_stats.symbols_with_tech}")
            logger.info(f"  Total records: {tech_stats.total_records:,}")
            logger.info(f"  With SMA-50: {tech_stats.with_sma50:,}")
            logger.info(f"  With SMA-200: {tech_stats.with_sma200:,}")

            logger.info("\n✅ Data quality validation completed at 10:30 PM")

    except Exception as e:
        logger.error(f"❌ Data quality validation failed: {e}", exc_info=True)


def run_scheduler():
    """Main scheduler loop."""
    logger.info("=" * 80)
    logger.info("Data Pipeline Scheduler Started")
    logger.info("=" * 80)
    logger.info("Scheduled Tasks:")
    logger.info("  - Symbol Master Update:    Weekly (Monday) at 06:00 AM")
    logger.info("  - Per-model data jobs registered below")
    logger.info("  - Data Pipeline (4 steps): Daily at 09:00 PM (after market close)")
    logger.info("  - CSV Export:              Daily at 10:00 PM")
    logger.info("  - Data Quality Check:      Daily at 10:00 PM (parallel with CSV)")
    logger.info("=" * 80)

    # Weekly symbol master update (Monday 6 AM)
    schedule.every().monday.at("06:00").do(update_symbol_master)

    # Per-model data jobs. Add new models by creating
    # tools/models/<name>/cron.py with register_data_jobs(schedule).
    from tools.models.momentum_n100_top5_max1.cron import (
        register_data_jobs as register_momentum_n100_data,
    )
    from tools.models.momentum_pseudo_n100_adv.cron import (
        register_data_jobs as register_pseudo_n100_data,
    )
    from tools.models.midcap_narrow_60d_breakout.cron import (
        register_data_jobs as register_midcap_narrow_data,
    )
    from tools.models.n20_daily_large_only.cron import (
        register_data_jobs as register_n20_daily_data,
    )
    from tools.models.finnifty_ic_otm4_w300_lots5.cron import (
        register_data_jobs as register_finnifty_ic_data,
    )
    register_momentum_n100_data(schedule)
    register_pseudo_n100_data(schedule)
    register_midcap_narrow_data(schedule)
    register_n20_daily_data(schedule)
    register_finnifty_ic_data(schedule)

    # Legacy 4-step saga (kept for admin UI compat — populates technical_indicators,
    # stocks.market_cap/PE/PB/ROE used by /admin and /suggested-stocks dashboards).
    # Model 3 needs only step 3 (HISTORICAL_DATA), which is also covered by the
    # per-model data_pull above as a fallback.
    schedule.every().day.at("21:00").do(run_data_pipeline)

    # Export CSV & validate quality (10 PM - after pipeline completes)
    schedule.every().day.at("22:00").do(export_daily_csv)
    schedule.every().day.at("22:00").do(validate_data_quality)

    # Weekly full-history backfill — fills gaps for stocks added after the
    # initial seed (newly-listed midcaps, universe changes, etc.). Daily
    # incremental pulls only fetch last 2 days, so any stock with <4y of
    # history would stay short forever without this job.
    schedule.every().sunday.at("03:00").do(backfill_full_history)
    logger.info("  - Full History Backfill (4y): Weekly Sunday at 03:00 AM")

    # Run backfill immediately on startup if env flag set. Useful for
    # first-time deployment + manual recovery after data loss.
    if os.environ.get("BACKFILL_ON_BOOT", "false").lower() == "true":
        logger.info("BACKFILL_ON_BOOT=true → running backfill_full_history now")
        try:
            backfill_full_history()
        except Exception as e:
            logger.error(f"Boot backfill failed: {e}", exc_info=True)
    
    # Keep scheduler running
    logger.info("Data scheduler is now running. Press Ctrl+C to stop.")
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Data scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Scheduler error: {e}", exc_info=True)
            time.sleep(60)


if __name__ == '__main__':
    run_scheduler()
