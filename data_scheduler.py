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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_data_pipeline():
    """
    Run complete data pipeline (Daily at 9:00 PM after market close).
    Steps:
    1. SYMBOL_MASTER - Fetch NSE symbols
    2. STOCKS - Update stock prices and market cap
    3. HISTORICAL_DATA - Fetch 1-year OHLCV data
    4. TECHNICAL_INDICATORS - Calculate RSI, MACD, SMA, EMA, ATR
    5. COMPREHENSIVE_METRICS - Calculate volatility metrics
    6. PIPELINE_VALIDATION - Validate data quality
    """
    logger.info("=" * 80)
    logger.info("Starting Data Pipeline (6-Step Saga)")
    logger.info("=" * 80)
    
    try:
        # Run the main pipeline
        result = subprocess.run(
            ['python3', 'run_pipeline.py'],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("✅ Data pipeline completed successfully at 9:00 PM")
            logger.info(f"Output:\n{result.stdout}")
        else:
            logger.error(f"❌ Data pipeline failed with return code {result.returncode}")
            logger.error(f"Error:\n{result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Data pipeline timeout after 1 hour")
    except Exception as e:
        logger.error(f"❌ Data pipeline error: {e}", exc_info=True)


def fill_missing_data():
    """
    Fill missing data fields (Daily at 9:30 PM after pipeline).
    - adj_close
    - liquidity_score
    - atr_14, atr_percentage
    - avg_daily_volume_20d
    - historical_volatility_1y
    """
    logger.info("=" * 80)
    logger.info("Filling Missing Data Fields")
    logger.info("=" * 80)
    
    try:
        result = subprocess.run(
            ['python3', 'fill_data_sql.py'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )
        
        if result.returncode == 0:
            logger.info("✅ Missing data filled successfully at 9:30 PM")
            logger.info(f"Output:\n{result.stdout}")
        else:
            logger.error(f"❌ Fill data failed with return code {result.returncode}")
            logger.error(f"Error:\n{result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Fill data timeout after 10 minutes")
    except Exception as e:
        logger.error(f"❌ Fill data error: {e}", exc_info=True)


def calculate_business_logic():
    """
    Calculate derived financial metrics (Daily at 9:45 PM).
    - EPS, Book Value, PEG Ratio, ROA
    - Operating Margin, Net Margin, Profit Margin
    - Current Ratio, Quick Ratio
    - Revenue Growth, Earnings Growth
    - Bid-Ask Spread, Trades Per Day
    - Beta
    """
    logger.info("=" * 80)
    logger.info("Calculating Business Logic & Derived Metrics")
    logger.info("=" * 80)
    
    try:
        result = subprocess.run(
            ['python3', 'fix_business_logic.py'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )
        
        if result.returncode == 0:
            logger.info("✅ Business logic calculated successfully at 9:45 PM")
            logger.info(f"Output:\n{result.stdout}")
        else:
            logger.error(f"❌ Business logic calculation failed with return code {result.returncode}")
            logger.error(f"Error:\n{result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Business logic calculation timeout after 10 minutes")
    except Exception as e:
        logger.error(f"❌ Business logic calculation error: {e}", exc_info=True)


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
                    symbol, date, rsi_14, macd, signal_line, macd_histogram,
                    sma_50, sma_200, ema_12, ema_26, atr_14, atr_percentage
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
                       ml_prediction_score, ml_price_target, ml_confidence, ml_risk_score,
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
        
        # Cleanup old CSV files (keep last 30 days)
        logger.info("Cleaning up old CSV files (>30 days)...")
        cleanup_old_csv_files(export_dir, keep_days=30)
        
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
            symbols = symbol_service.refresh_symbol_database()
            
            logger.info(f"✅ Symbol master updated successfully at 6:00 AM (Monday)")
            logger.info(f"  Total symbols: {len(symbols)}")
            
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
        
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # Check stocks table
            stocks_stats = session.execute("""
                SELECT 
                    COUNT(*) as total_stocks,
                    COUNT(current_price) as with_price,
                    COUNT(market_cap) as with_market_cap,
                    COUNT(pe_ratio) as with_pe,
                    COUNT(eps) as with_eps,
                    COUNT(sector) as with_sector
                FROM stocks
            """).fetchone()
            
            logger.info("Stocks Table:")
            logger.info(f"  Total stocks: {stocks_stats.total_stocks}")
            logger.info(f"  With price: {stocks_stats.with_price} ({stocks_stats.with_price/stocks_stats.total_stocks*100:.1f}%)")
            logger.info(f"  With market cap: {stocks_stats.with_market_cap} ({stocks_stats.with_market_cap/stocks_stats.total_stocks*100:.1f}%)")
            logger.info(f"  With PE ratio: {stocks_stats.with_pe} ({stocks_stats.with_pe/stocks_stats.total_stocks*100:.1f}%)")
            logger.info(f"  With EPS: {stocks_stats.with_eps} ({stocks_stats.with_eps/stocks_stats.total_stocks*100:.1f}%)")
            logger.info(f"  With sector: {stocks_stats.with_sector} ({stocks_stats.with_sector/stocks_stats.total_stocks*100:.1f}%)")
            
            # Check historical data
            history_stats = session.execute("""
                SELECT 
                    COUNT(DISTINCT symbol) as symbols_with_history,
                    COUNT(*) as total_records,
                    MAX(date) as latest_date,
                    MIN(date) as earliest_date
                FROM historical_data
            """).fetchone()
            
            logger.info("\nHistorical Data:")
            logger.info(f"  Symbols with history: {history_stats.symbols_with_history}")
            logger.info(f"  Total records: {history_stats.total_records:,}")
            logger.info(f"  Date range: {history_stats.earliest_date} to {history_stats.latest_date}")
            
            # Check technical indicators
            tech_stats = session.execute("""
                SELECT 
                    COUNT(DISTINCT symbol) as symbols_with_tech,
                    COUNT(*) as total_records,
                    COUNT(rsi_14) as with_rsi,
                    COUNT(macd) as with_macd
                FROM technical_indicators
            """).fetchone()
            
            logger.info("\nTechnical Indicators:")
            logger.info(f"  Symbols with indicators: {tech_stats.symbols_with_tech}")
            logger.info(f"  Total records: {tech_stats.total_records:,}")
            logger.info(f"  With RSI: {tech_stats.with_rsi:,}")
            logger.info(f"  With MACD: {tech_stats.with_macd:,}")
            
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
