"""
Suggested Stocks Routes
Handles stock suggestions based on swing trading strategies with proper filtering pipeline.
"""

import logging
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

logger = logging.getLogger(__name__)

# Create blueprint
suggested_stocks_bp = Blueprint('suggested_stocks', __name__, url_prefix='/api/suggested-stocks')


@suggested_stocks_bp.route('/', methods=['GET'])
@login_required
def get_suggested_stocks():
    """
    Get suggested stocks based on swing trading strategies.

    Query parameters:
    - strategy: 'default_risk' or 'high_risk' (default: 'default_risk')
    - limit: Number of suggestions to return (default: 50)
    - search: Optional search term
    - sort_by: Sort field (default: 'current_price')
    - sort_order: 'asc' or 'desc' (default: 'desc')
    - sector: Optional sector filter
    """
    try:
        # Get parameters
        strategy = request.args.get('strategy', 'default_risk')
        limit = int(request.args.get('limit', 50))
        search = request.args.get('search')
        sort_by = request.args.get('sort_by', 'current_price')
        sort_order = request.args.get('sort_order', 'desc')
        sector = request.args.get('sector')

        user_id = current_user.id

        logger.info(f"🎯 Suggested stocks request: strategy={strategy}, limit={limit}, user={user_id}")

        # Use unified broker service to get the appropriate provider
        from ...services.core.unified_broker_service import get_unified_broker_service
        from ...services.interfaces.suggested_stocks_interface import StrategyType

        # Map strategy type to enum
        strategy_map = {
            'default_risk': StrategyType.DEFAULT_RISK,
            'high_risk': StrategyType.HIGH_RISK
        }

        if strategy not in strategy_map:
            return jsonify({
                'success': False,
                'error': f'Invalid strategy. Must be one of: {list(strategy_map.keys())}'
            }), 400

        strategy_enum = strategy_map[strategy]

        # Get the unified broker service
        unified_service = get_unified_broker_service()

        # Get the provider through the factory for advanced features
        provider = unified_service.factory.get_suggested_stocks_provider(user_id)
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No suggested stocks provider available for your configured broker'
            }), 503

        # Use the provider directly for full feature support
        result = provider.get_suggested_stocks(
            user_id=user_id,
            strategies=[strategy_enum],
            limit=limit,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            sector=sector
        )

        if result.get('success'):
            logger.info(f"✅ Returned {len(result.get('data', []))} suggested stocks for {strategy}")
            return jsonify(result), 200
        else:
            logger.warning(f"❌ Suggested stocks failed: {result.get('error')}")
            return jsonify(result), 500

    except ValueError as e:
        logger.error(f"Invalid parameter: {e}")
        return jsonify({
            'success': False,
            'error': f'Invalid parameter: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"Error getting suggested stocks: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@suggested_stocks_bp.route('/discover', methods=['GET'])
@login_required
def discover_tradeable_stocks():
    """
    Discover all tradeable stocks from the exchange.

    Query parameters:
    - exchange: Exchange name (default: 'NSE')
    """
    try:
        exchange = request.args.get('exchange', 'NSE')
        user_id = current_user.id

        logger.info(f"📊 Discovering tradeable stocks from {exchange} for user {user_id}")

        # Use unified broker service
        from ...services.core.unified_broker_service import get_unified_broker_service

        unified_service = get_unified_broker_service()
        result = unified_service.discover_tradeable_stocks(
            user_id=user_id,
            exchange=exchange
        )

        if result.get('success'):
            logger.info(f"✅ Discovered {result.get('total_discovered', 0)} tradeable stocks")
            return jsonify(result), 200
        else:
            logger.warning(f"❌ Discovery failed: {result.get('error')}")
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Error discovering tradeable stocks: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@suggested_stocks_bp.route('/search', methods=['GET'])
@login_required
def search_stocks():
    """
    Search for stocks using a search term.

    Query parameters:
    - q: Search term (required)
    - exchange: Exchange name (default: 'NSE')
    """
    try:
        search_term = request.args.get('q')
        exchange = request.args.get('exchange', 'NSE')
        user_id = current_user.id

        if not search_term:
            return jsonify({
                'success': False,
                'error': 'Search term (q) is required'
            }), 400

        logger.info(f"🔍 Searching stocks for '{search_term}' on {exchange}")

        # Use unified broker service
        from ...services.core.unified_broker_service import get_unified_broker_service

        unified_service = get_unified_broker_service()
        result = unified_service.search_stocks(
            user_id=user_id,
            search_term=search_term,
            exchange=exchange
        )

        if result.get('success'):
            logger.info(f"✅ Found {result.get('total_results', 0)} stocks matching '{search_term}'")
            return jsonify(result), 200
        else:
            logger.warning(f"❌ Search failed: {result.get('error')}")
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Error searching stocks: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@suggested_stocks_bp.route('/analysis/<symbol>', methods=['GET'])
@login_required
def get_stock_analysis(symbol):
    """
    Get detailed analysis for a specific stock.

    Path parameters:
    - symbol: Stock symbol (e.g., 'RELIANCE')
    """
    try:
        user_id = current_user.id

        logger.info(f"📈 Getting analysis for stock {symbol}")

        # Use unified broker service
        from ...services.core.unified_broker_service import get_unified_broker_service

        unified_service = get_unified_broker_service()
        result = unified_service.get_stock_analysis(
            user_id=user_id,
            symbol=symbol
        )

        if result.get('success'):
            logger.info(f"✅ Retrieved analysis for {symbol}")
            return jsonify(result), 200
        else:
            logger.warning(f"❌ Analysis failed for {symbol}: {result.get('error')}")
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Error getting stock analysis for {symbol}: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@suggested_stocks_bp.route('/strategies', methods=['GET'])
def get_available_strategies():
    """Get list of available swing trading strategies."""
    try:
        strategies = [
            {
                'key': 'default_risk',
                'name': 'Default Risk Strategy',
                'description': 'Conservative swing trading with 5-7% profit targets and 3% stop loss',
                'profit_target': '5-7%',
                'stop_loss': '3%',
                'timeframe': '2 weeks',
                'risk_level': 'Low to Medium'
            },
            {
                'key': 'high_risk',
                'name': 'High Risk Strategy',
                'description': 'Aggressive swing trading with 8-10% profit targets and 3% stop loss',
                'profit_target': '8-10%',
                'stop_loss': '3%',
                'timeframe': '2 weeks',
                'risk_level': 'High'
            }
        ]

        return jsonify({
            'success': True,
            'data': strategies,
            'total': len(strategies)
        }), 200

    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500