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
from datetime import datetime
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging with rotation (max 50MB per file, keep 5 backups)
from logging.handlers import RotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/data_scheduler.log', maxBytes=50*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
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
    logger.info("Starting Data Pipeline (6-Step Saga)")
    logger.info("=" * 80)
    _run_subprocess_with_retry(['python3', 'run_pipeline.py'], 'Data Pipeline', timeout=3600, max_retries=2)


def fill_missing_data():
    """Fill missing data fields (Daily at 9:30 PM after pipeline)."""
    logger.info("=" * 80)
    logger.info("Filling Missing Data Fields")
    logger.info("=" * 80)
    _run_subprocess_with_retry(['python3', 'fill_data_sql.py'], 'Fill Missing Data', timeout=600)


def calculate_business_logic():
    """Calculate derived financial metrics (Daily at 9:45 PM)."""
    logger.info("=" * 80)
    logger.info("Calculating Business Logic & Derived Metrics")
    logger.info("=" * 80)
    _run_subprocess_with_retry(['python3', 'fix_business_logic.py'], 'Business Logic', timeout=600)


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
                    symbol, date, ema_8, ema_21, demarker, sma_50, sma_200
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
                       ema_8, ema_21, ema_trend_score, demarker,
                       fib_target_1, fib_target_2, fib_target_3,
                       buy_signal, sell_signal, signal_quality,
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

            # Check technical indicators (updated columns)
            tech_stats = session.execute(text("""
                SELECT
                    COUNT(DISTINCT symbol) as symbols_with_tech,
                    COUNT(*) as total_records,
                    COUNT(ema_8) as with_ema8,
                    COUNT(ema_21) as with_ema21,
                    COUNT(demarker) as with_demarker
                FROM technical_indicators
            """)).fetchone()

            logger.info("\nTechnical Indicators:")
            logger.info(f"  Symbols with indicators: {tech_stats.symbols_with_tech}")
            logger.info(f"  Total records: {tech_stats.total_records:,}")
            logger.info(f"  With EMA-8: {tech_stats.with_ema8:,}")
            logger.info(f"  With EMA-21: {tech_stats.with_ema21:,}")
            logger.info(f"  With DeMarker: {tech_stats.with_demarker:,}")

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
    logger.info("  - Data Pipeline (6 steps): Daily at 09:00 PM (after market close)")
    logger.info("  - Fill Missing Data:       Daily at 09:30 PM")
    logger.info("  - Business Logic Calc:     Daily at 09:30 PM (parallel with fill)")
    logger.info("  - CSV Export:              Daily at 10:00 PM")
    logger.info("  - Data Quality Check:      Daily at 10:00 PM (parallel with CSV)")
    logger.info("=" * 80)

    # Weekly symbol master update (Monday 6 AM)
    schedule.every().monday.at("06:00").do(update_symbol_master)

    # Daily data pipeline (9 PM - after market close at 3:30 PM + buffer)
    schedule.every().day.at("21:00").do(run_data_pipeline)

    # Fill missing data & business logic in parallel (9:30 PM - after pipeline completes)
    schedule.every().day.at("21:30").do(fill_missing_data)
    schedule.every().day.at("21:30").do(calculate_business_logic)

    # Export CSV & validate quality in parallel (10 PM - after calculations)
    schedule.every().day.at("22:00").do(export_daily_csv)
    schedule.every().day.at("22:00").do(validate_data_quality)
    
    # Optional: Run immediately on startup for testing
    # Uncomment to run tasks on scheduler start
    # logger.info("Running initial data pipeline...")
    # run_data_pipeline()
    # fill_missing_data()
    # calculate_business_logic()
    
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
