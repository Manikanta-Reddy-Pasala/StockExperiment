#!/usr/bin/env python3
"""
Populate Stock Data using YFinance

This tool populates the database with NSE stock data using Yahoo Finance,
enabling paper trading without requiring broker credentials.

Usage:
    python tools/populate_data_yfinance.py --stocks      # Populate stocks table
    python tools/populate_data_yfinance.py --history     # Fetch historical data
    python tools/populate_data_yfinance.py --quotes      # Update current prices
    python tools/populate_data_yfinance.py --all         # Do everything
    python tools/populate_data_yfinance.py --test        # Test yfinance connection

Environment:
    Set DATA_PROVIDER_MODE=yfinance in your environment for paper trading.
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.database import get_database_manager
from src.models.stock_models import Stock
from src.models.historical_models import HistoricalData
from src.services.data.yfinance_data_service import get_yfinance_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# NSE Stock Lists
NIFTY_50 = [
    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
    'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL',
    'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
    'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE',
    'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'ITC',
    'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LT',
    'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC',
    'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SUNPHARMA',
    'TCS', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TECHM',
    'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO', 'LTIM'
]

NIFTY_NEXT_50 = [
    'ABB', 'ADANIGREEN', 'ADANIPOWER', 'AMBUJACEM', 'ATGL',
    'AUROPHARMA', 'BAJAJHLDNG', 'BANDHANBNK', 'BANKBARODA', 'BEL',
    'BERGEPAINT', 'BOSCHLTD', 'CANBK', 'CHOLAFIN', 'COLPAL',
    'DLF', 'DABUR', 'GODREJCP', 'GAIL', 'HAVELLS',
    'HINDPETRO', 'ICICIGI', 'ICICIPRULI', 'INDHOTEL', 'INDUSTOWER',
    'IOC', 'IRCTC', 'JINDALSTEL', 'LICI', 'LUPIN',
    'MARICO', 'MUTHOOTFIN', 'NAUKRI', 'PFC', 'PIDILITIND',
    'PNB', 'RECLTD', 'SRF', 'SHREECEM', 'SIEMENS',
    'SBICARD', 'SHRIRAMFIN', 'TATAPOWER', 'TORNTPHARM', 'TRENT',
    'UNIONBANK', 'VBL', 'VEDL', 'YESBANK', 'ZOMATO'
]

# Popular mid-cap stocks for broader coverage
MIDCAP_POPULAR = [
    'ASTRAL', 'BALKRISIND', 'BHARATFORG', 'BIOCON', 'CUMMINSIND',
    'DEEPAKNTR', 'ESCORTS', 'FEDERALBNK', 'GMRAIRPORT', 'IDFCFIRSTB',
    'JUBLFOOD', 'LICHSGFIN', 'MRF', 'MPHASIS', 'NATIONALUM',
    'OFSS', 'PAGEIND', 'PERSISTENT', 'PETRONET', 'POLYCAB',
    'RAMCOCEM', 'SAIL', 'SONACOMS', 'TVSMOTOR', 'VOLTAS'
]


def get_all_stock_symbols():
    """Get all stock symbols to track."""
    all_stocks = NIFTY_50 + NIFTY_NEXT_50 + MIDCAP_POPULAR
    # Remove duplicates while preserving order
    seen = set()
    unique_stocks = []
    for stock in all_stocks:
        if stock not in seen:
            seen.add(stock)
            unique_stocks.append(stock)
    return [f"NSE:{symbol}-EQ" for symbol in unique_stocks]


def test_connection():
    """Test yfinance connection."""
    logger.info("Testing yfinance connection...")
    yf_service = get_yfinance_service()
    result = yf_service.test_connection()

    if result['success']:
        logger.info(f"Connection successful!")
        logger.info(f"  Provider: {result['provider']}")
        logger.info(f"  Test Symbol: {result['test_symbol']}")
        logger.info(f"  Last Price: {result['last_price']}")
        return True
    else:
        logger.error(f"Connection failed: {result['message']}")
        return False


def populate_stocks():
    """Populate stocks table with NSE stocks."""
    logger.info("Populating stocks table...")

    db_manager = get_database_manager()
    yf_service = get_yfinance_service()

    symbols = get_all_stock_symbols()
    logger.info(f"Processing {len(symbols)} stocks...")

    # Get initial quotes to populate stock data
    quotes = yf_service.get_quotes_bulk(symbols)

    success_count = 0
    with db_manager.get_session() as session:
        for symbol in symbols:
            try:
                # Check if stock already exists
                existing = session.query(Stock).filter(Stock.symbol == symbol).first()

                # Get quote data if available
                quote = quotes.get(symbol, {})
                current_price = quote.get('last_price')
                volume = quote.get('volume', 0)

                # Extract clean symbol name
                clean_symbol = symbol.replace('NSE:', '').replace('-EQ', '')

                if existing:
                    # Update existing stock
                    if current_price:
                        existing.current_price = current_price
                        existing.volume = volume
                        existing.updated_at = datetime.utcnow()
                else:
                    # Create new stock
                    stock = Stock(
                        symbol=symbol,
                        name=clean_symbol,
                        exchange='NSE',
                        current_price=current_price,
                        volume=volume,
                        is_active=True,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    session.add(stock)

                success_count += 1

            except Exception as e:
                logger.warning(f"Error processing {symbol}: {e}")
                continue

        session.commit()

    logger.info(f"Successfully processed {success_count}/{len(symbols)} stocks")
    return success_count


def populate_historical_data(days: int = 365):
    """Fetch and store historical data for all stocks."""
    logger.info(f"Fetching {days} days of historical data...")

    db_manager = get_database_manager()
    yf_service = get_yfinance_service()

    # Get all active stocks
    with db_manager.get_session() as session:
        stocks = session.query(Stock).filter(Stock.is_active == True).all()
        symbols = [stock.symbol for stock in stocks]

    if not symbols:
        symbols = get_all_stock_symbols()

    logger.info(f"Processing historical data for {len(symbols)} stocks...")

    success_count = 0
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        try:
            logger.info(f"[{i+1}/{total}] Fetching {symbol}...")

            df = yf_service.get_historical_data(symbol, days)

            if df is None or df.empty:
                logger.warning(f"No data for {symbol}")
                continue

            # Store in database
            records_added = store_historical_data(symbol, df)

            if records_added > 0:
                success_count += 1
                logger.info(f"  Added {records_added} records")

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            continue

    logger.info(f"Successfully fetched historical data for {success_count}/{total} stocks")
    return success_count


def store_historical_data(symbol: str, df) -> int:
    """Store historical data in database."""
    db_manager = get_database_manager()
    records_added = 0

    with db_manager.get_session() as session:
        # Get existing dates for this symbol
        existing_dates = set(
            date for (date,) in session.query(HistoricalData.date).filter(
                HistoricalData.symbol == symbol
            ).all()
        )

        for _, row in df.iterrows():
            try:
                if row['date'] in existing_dates:
                    continue

                record = HistoricalData(
                    symbol=symbol,
                    date=row['date'],
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume']),
                    data_source='yfinance',
                    api_resolution='1D',
                    data_quality_score=1.0
                )
                session.add(record)
                records_added += 1

            except Exception as e:
                logger.debug(f"Error storing record: {e}")
                continue

        session.commit()

    return records_added


def update_quotes():
    """Update current prices for all stocks."""
    logger.info("Updating current stock prices...")

    db_manager = get_database_manager()
    yf_service = get_yfinance_service()

    # Get all active stocks
    with db_manager.get_session() as session:
        stocks = session.query(Stock).filter(Stock.is_active == True).all()
        symbols = [stock.symbol for stock in stocks]

    if not symbols:
        logger.warning("No stocks found. Run --stocks first.")
        return 0

    logger.info(f"Fetching quotes for {len(symbols)} stocks...")

    # Fetch quotes in bulk
    quotes = yf_service.get_quotes_bulk(symbols)

    success_count = 0
    with db_manager.get_session() as session:
        for symbol, quote in quotes.items():
            try:
                stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                if stock and quote.get('last_price'):
                    stock.current_price = quote['last_price']
                    stock.open_price = quote.get('open')
                    stock.high_price = quote.get('high')
                    stock.low_price = quote.get('low')
                    stock.volume = quote.get('volume', 0)
                    stock.updated_at = datetime.utcnow()
                    success_count += 1
            except Exception as e:
                logger.warning(f"Error updating {symbol}: {e}")
                continue

        session.commit()

    logger.info(f"Updated prices for {success_count}/{len(symbols)} stocks")
    return success_count


def run_all():
    """Run all data population steps."""
    logger.info("=" * 60)
    logger.info("YFinance Data Population - Full Run")
    logger.info("=" * 60)

    # Test connection first
    if not test_connection():
        logger.error("Connection test failed. Aborting.")
        return False

    # Populate stocks
    logger.info("\n" + "=" * 40)
    logger.info("Step 1: Populating stocks table")
    logger.info("=" * 40)
    populate_stocks()

    # Fetch historical data
    logger.info("\n" + "=" * 40)
    logger.info("Step 2: Fetching historical data")
    logger.info("=" * 40)
    populate_historical_data(365)

    # Update quotes
    logger.info("\n" + "=" * 40)
    logger.info("Step 3: Updating current prices")
    logger.info("=" * 40)
    update_quotes()

    logger.info("\n" + "=" * 60)
    logger.info("Data population complete!")
    logger.info("You can now run paper trading without broker credentials.")
    logger.info("Set DATA_PROVIDER_MODE=yfinance in your environment.")
    logger.info("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Populate stock data using Yahoo Finance for paper trading'
    )
    parser.add_argument('--test', action='store_true', help='Test yfinance connection')
    parser.add_argument('--stocks', action='store_true', help='Populate stocks table')
    parser.add_argument('--history', action='store_true', help='Fetch historical data')
    parser.add_argument('--quotes', action='store_true', help='Update current prices')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data (default: 365)')

    args = parser.parse_args()

    if not any([args.test, args.stocks, args.history, args.quotes, args.all]):
        parser.print_help()
        return

    if args.test:
        test_connection()

    if args.all:
        run_all()
    else:
        if args.stocks:
            populate_stocks()
        if args.history:
            populate_historical_data(args.days)
        if args.quotes:
            update_quotes()


if __name__ == '__main__':
    main()
