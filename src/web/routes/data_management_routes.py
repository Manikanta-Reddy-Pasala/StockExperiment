"""
Data Management Routes

Handles data population, yfinance integration, and data provider status.
Essential for paper trading setup without broker credentials.
"""

import logging
import os
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprint
data_management_bp = Blueprint('data_management', __name__, url_prefix='/api/data')


@data_management_bp.route('/provider/status', methods=['GET'])
@login_required
def get_data_provider_status():
    """
    Get current data provider status and configuration.

    Returns:
        JSON with data provider mode, yfinance status, and broker status
    """
    try:
        from src.services.core.unified_broker_service import DATA_PROVIDER_MODE

        result = {
            'success': True,
            'data_provider_mode': DATA_PROVIDER_MODE,
            'yfinance': {
                'available': False,
                'connection_tested': False,
                'last_price': None
            },
            'broker': {
                'configured': False,
                'connected': False
            }
        }

        # Test yfinance connection
        try:
            from src.services.data.yfinance_data_service import get_yfinance_service
            yf_service = get_yfinance_service()
            yf_result = yf_service.test_connection()

            result['yfinance'] = {
                'available': True,
                'connection_tested': yf_result.get('success', False),
                'last_price': yf_result.get('last_price'),
                'message': yf_result.get('message')
            }
        except ImportError:
            result['yfinance']['message'] = 'yfinance not installed'
        except Exception as e:
            result['yfinance']['message'] = str(e)

        # Check broker configuration
        try:
            from src.models.database import get_database_manager
            from src.models.models import BrokerConfiguration

            with get_database_manager().get_session() as session:
                broker_config = session.query(BrokerConfiguration).filter(
                    BrokerConfiguration.user_id == current_user.id,
                    BrokerConfiguration.is_active == True
                ).first()

                if broker_config:
                    result['broker'] = {
                        'configured': True,
                        'connected': broker_config.is_connected or False,
                        'broker_name': broker_config.broker_name,
                        'last_connection_test': broker_config.last_connection_test.isoformat() if broker_config.last_connection_test else None
                    }
        except Exception as e:
            logger.warning(f"Error checking broker config: {e}")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error getting data provider status: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_management_bp.route('/yfinance/test', methods=['POST'])
@login_required
def test_yfinance_connection():
    """
    Test yfinance connection and return status.
    """
    try:
        from src.services.data.yfinance_data_service import get_yfinance_service

        yf_service = get_yfinance_service()
        result = yf_service.test_connection()

        return jsonify({
            'success': result.get('success', False),
            'provider': 'yfinance',
            'test_symbol': result.get('test_symbol'),
            'last_price': result.get('last_price'),
            'message': result.get('message')
        }), 200 if result.get('success') else 400

    except ImportError:
        return jsonify({
            'success': False,
            'error': 'yfinance not installed. Install with: pip install yfinance'
        }), 400
    except Exception as e:
        logger.error(f"Error testing yfinance: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_management_bp.route('/yfinance/populate/stocks', methods=['POST'])
@login_required
def populate_stocks_yfinance():
    """
    Populate stocks table using yfinance data.
    """
    try:
        from src.services.data.yfinance_data_service import get_yfinance_service
        from src.models.database import get_database_manager
        from src.models.stock_models import Stock

        yf_service = get_yfinance_service()
        db_manager = get_database_manager()

        # Get NIFTY 50 + NIFTY Next 50 symbols
        nifty_symbols = yf_service.get_nifty50_stocks()

        # Add more popular stocks
        additional_symbols = [
            'NSE:ADANIGREEN-EQ', 'NSE:ADANIPOWER-EQ', 'NSE:ZOMATO-EQ',
            'NSE:IRCTC-EQ', 'NSE:DLF-EQ', 'NSE:TRENT-EQ'
        ]

        all_symbols = list(set(nifty_symbols + additional_symbols))

        logger.info(f"Populating {len(all_symbols)} stocks using yfinance...")

        # Get quotes for all symbols
        quotes = yf_service.get_quotes_bulk(all_symbols)

        success_count = 0
        with db_manager.get_session() as session:
            for symbol in all_symbols:
                try:
                    quote = quotes.get(symbol, {})
                    current_price = quote.get('last_price')

                    # Extract clean symbol name
                    clean_symbol = symbol.replace('NSE:', '').replace('-EQ', '')

                    existing = session.query(Stock).filter(Stock.symbol == symbol).first()

                    if existing:
                        if current_price:
                            existing.current_price = current_price
                            existing.volume = quote.get('volume', 0)
                            existing.updated_at = datetime.utcnow()
                    else:
                        stock = Stock(
                            symbol=symbol,
                            name=clean_symbol,
                            exchange='NSE',
                            current_price=current_price,
                            volume=quote.get('volume', 0),
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

        return jsonify({
            'success': True,
            'message': f'Successfully populated {success_count}/{len(all_symbols)} stocks',
            'total_symbols': len(all_symbols),
            'successful': success_count
        }), 200

    except Exception as e:
        logger.error(f"Error populating stocks: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_management_bp.route('/yfinance/populate/history', methods=['POST'])
@login_required
def populate_history_yfinance():
    """
    Fetch historical data using yfinance.

    Request body:
        {
            "days": 365,          # Number of days of history (default: 365)
            "max_stocks": 100     # Maximum stocks to process (default: 100)
        }
    """
    try:
        from src.services.data.yfinance_data_service import get_yfinance_service
        from src.models.database import get_database_manager
        from src.models.stock_models import Stock
        from src.models.historical_models import HistoricalData

        data = request.get_json() or {}
        days = data.get('days', 365)
        max_stocks = data.get('max_stocks', 100)

        yf_service = get_yfinance_service()
        db_manager = get_database_manager()

        # Get active stocks
        with db_manager.get_session() as session:
            stocks = session.query(Stock).filter(
                Stock.is_active == True
            ).limit(max_stocks).all()
            symbols = [stock.symbol for stock in stocks]

        if not symbols:
            return jsonify({
                'success': False,
                'error': 'No stocks found. Run /api/data/yfinance/populate/stocks first.'
            }), 400

        logger.info(f"Fetching {days} days of history for {len(symbols)} stocks...")

        success_count = 0
        total_records = 0

        for symbol in symbols:
            try:
                df = yf_service.get_historical_data(symbol, days)

                if df is None or df.empty:
                    continue

                # Store historical data
                with db_manager.get_session() as session:
                    existing_dates = set(
                        date for (date,) in session.query(HistoricalData.date).filter(
                            HistoricalData.symbol == symbol
                        ).all()
                    )

                    records_added = 0
                    for _, row in df.iterrows():
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

                    session.commit()

                    if records_added > 0:
                        success_count += 1
                        total_records += records_added

            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
                continue

        return jsonify({
            'success': True,
            'message': f'Successfully fetched history for {success_count}/{len(symbols)} stocks',
            'total_symbols': len(symbols),
            'successful': success_count,
            'total_records': total_records
        }), 200

    except Exception as e:
        logger.error(f"Error populating history: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_management_bp.route('/yfinance/update/quotes', methods=['POST'])
@login_required
def update_quotes_yfinance():
    """
    Update current prices for all stocks using yfinance.
    """
    try:
        from src.services.data.yfinance_data_service import get_yfinance_service
        from src.models.database import get_database_manager
        from src.models.stock_models import Stock

        yf_service = get_yfinance_service()
        db_manager = get_database_manager()

        # Get all active stocks
        with db_manager.get_session() as session:
            stocks = session.query(Stock).filter(Stock.is_active == True).all()
            symbols = [stock.symbol for stock in stocks]

        if not symbols:
            return jsonify({
                'success': False,
                'error': 'No stocks found. Run /api/data/yfinance/populate/stocks first.'
            }), 400

        logger.info(f"Updating quotes for {len(symbols)} stocks...")

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

        return jsonify({
            'success': True,
            'message': f'Updated prices for {success_count}/{len(symbols)} stocks',
            'total_symbols': len(symbols),
            'updated': success_count
        }), 200

    except Exception as e:
        logger.error(f"Error updating quotes: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_management_bp.route('/stats', methods=['GET'])
@login_required
def get_data_stats():
    """
    Get statistics about the data in the database.
    """
    try:
        from src.models.database import get_database_manager
        from src.models.stock_models import Stock
        from src.models.historical_models import HistoricalData
        from sqlalchemy import func

        db_manager = get_database_manager()

        with db_manager.get_session() as session:
            # Count stocks
            total_stocks = session.query(func.count(Stock.id)).scalar()
            active_stocks = session.query(func.count(Stock.id)).filter(
                Stock.is_active == True
            ).scalar()
            stocks_with_price = session.query(func.count(Stock.id)).filter(
                Stock.current_price.isnot(None)
            ).scalar()

            # Historical data stats
            total_historical = session.query(func.count(HistoricalData.id)).scalar()
            historical_symbols = session.query(
                func.count(func.distinct(HistoricalData.symbol))
            ).scalar()

            # Date range
            min_date = session.query(func.min(HistoricalData.date)).scalar()
            max_date = session.query(func.max(HistoricalData.date)).scalar()

        return jsonify({
            'success': True,
            'stocks': {
                'total': total_stocks,
                'active': active_stocks,
                'with_price': stocks_with_price
            },
            'historical_data': {
                'total_records': total_historical,
                'symbols_covered': historical_symbols,
                'date_range': {
                    'start': min_date.isoformat() if min_date else None,
                    'end': max_date.isoformat() if max_date else None
                }
            }
        }), 200

    except Exception as e:
        logger.error(f"Error getting data stats: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
