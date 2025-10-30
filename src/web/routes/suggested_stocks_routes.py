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


@suggested_stocks_bp.route('/triple-model-view', methods=['GET'])
@login_required
def get_triple_model_view():
    """
    Get suggested stocks using 8-21 EMA Swing Trading Strategy for BOTH risk levels.

    Returns a view showing:
    - Default Risk Strategy (conservative, large-cap)
    - High Risk Strategy (aggressive, small/mid-cap)

    Query parameters:
    - limit: Number of stocks per strategy (default: 10)
    - date: Date to fetch (default: most recent data)
    """
    try:
        from datetime import datetime, date as dt_date
        from sqlalchemy import text
        from ...models.database import get_database_manager

        # Get parameters
        limit = int(request.args.get('limit', 10))
        max_limit = 50
        if limit > max_limit:
            limit = max_limit

        date_str = request.args.get('date')

        if date_str:
            try:
                query_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid date format. Use YYYY-MM-DD'
                }), 400
        else:
            # Get the most recent date with data
            with get_database_manager().get_session() as session:
                recent_date_result = session.execute(text("""
                    SELECT MAX(date) as max_date FROM daily_suggested_stocks
                """)).first()

                if recent_date_result and recent_date_result[0]:
                    query_date = recent_date_result[0]
                else:
                    query_date = dt_date.today()

        user_id = current_user.id

        logger.info(f"🎯 8-21 EMA Strategy view request: limit={limit}, date={query_date}, user={user_id}")

        db = get_database_manager()

        with db.get_session() as session:
            # Query 8-21 EMA strategy stocks
            query = text("""
                SELECT
                    d.symbol,
                    COALESCE(d.stock_name, s.name) as stock_name,
                    d.current_price,
                    d.strategy,
                    d.selection_score,
                    d.ema_8,
                    d.ema_21,
                    d.ema_trend_score,
                    d.demarker,
                    d.fib_target_1,
                    d.fib_target_2,
                    d.fib_target_3,
                    d.buy_signal,
                    d.sell_signal,
                    d.signal_quality,
                    d.recommendation,
                    d.target_price,
                    d.stop_loss,
                    d.rank,
                    d.reason,
                    d.pe_ratio,
                    d.pb_ratio,
                    d.roe,
                    d.market_cap,
                    d.sector,
                    d.market_cap_category
                FROM daily_suggested_stocks d
                LEFT JOIN stocks s ON d.symbol = s.symbol
                WHERE d.date = :date
                  AND d.buy_signal = TRUE
                  AND d.target_price > d.current_price
                ORDER BY d.strategy, d.selection_score DESC, d.ema_trend_score DESC
            """)

            result = session.execute(query, {'date': query_date})
            all_stocks = [dict(row._mapping) for row in result]

        # Group results by strategy only
        grouped_results = {
            'ema_strategy': {
                'default_risk': [],
                'high_risk': []
            }
        }

        # Filter stocks by strategy
        for stock in all_stocks:
            strategy = stock['strategy']

            # Map strategy names (handle case variations)
            if 'default' in strategy.lower() or strategy.upper() == 'DEFAULT_RISK':
                risk_level = 'default_risk'
            elif 'high' in strategy.lower() or strategy.upper() == 'HIGH_RISK':
                risk_level = 'high_risk'
            else:
                # Skip unknown strategies
                continue

            # Add to appropriate group
            grouped_results['ema_strategy'][risk_level].append(stock)

        # Apply limit to each group
        for risk_level in grouped_results['ema_strategy']:
            grouped_results['ema_strategy'][risk_level] = grouped_results['ema_strategy'][risk_level][:limit]

        # Calculate statistics
        stats = {
            'ema_strategy': {}
        }
        for risk_level in grouped_results['ema_strategy']:
            stocks_in_group = grouped_results['ema_strategy'][risk_level]
            stats['ema_strategy'][risk_level] = {
                'count': len(stocks_in_group),
                'avg_selection_score': sum(s.get('selection_score', 0) or 0 for s in stocks_in_group) / len(stocks_in_group) if stocks_in_group else 0,
                'avg_ema_trend_score': sum(s.get('ema_trend_score', 0) or 0 for s in stocks_in_group) / len(stocks_in_group) if stocks_in_group else 0,
                'high_quality_signals': sum(1 for s in stocks_in_group if s.get('signal_quality') == 'high')
            }

        response = {
            'success': True,
            'date': str(query_date),
            'limit_per_group': limit,
            'data': grouped_results,
            'statistics': stats,
            'total_stocks': sum(len(grouped_results['ema_strategy'][rl]) for rl in grouped_results['ema_strategy']),
            'strategy_type': '8-21 EMA Swing Trading'
        }

        logger.info(f"✅ 8-21 EMA Strategy view: {response['total_stocks']} total stocks across both risk levels")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error getting EMA strategy view: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


# Keep old endpoint for backward compatibility
@suggested_stocks_bp.route('/dual-model-view', methods=['GET'])
@login_required
def get_dual_model_view():
    """
    DEPRECATED: Use /triple-model-view instead.
    Redirects to triple-model-view endpoint.
    """
    return get_triple_model_view()


@suggested_stocks_bp.route('/market-sentiment', methods=['GET'])
@login_required
def get_market_sentiment():
    """
    Get market sentiment (simplified - AI disabled).

    Returns neutral market sentiment without AI analysis.
    """
    try:
        logger.info("📊 Market sentiment request received (AI disabled)")

        # Simple neutral response (AI/Ollama removed)
        response = {
            'success': True,
            'sentiment_type': 'neutral',
            'recommendation': 'Markets are stable. Consider balanced approach.',
            'analysis': 'AI-based market sentiment analysis is disabled. Using neutral sentiment.',
            'emoji': '📊',
            'color': 'info',
            'sources': []
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Error getting market sentiment: {e}", exc_info=True)
        # Return neutral sentiment instead of error
        return jsonify({
            'success': True,
            'sentiment_type': 'neutral',
            'recommendation': 'Markets are stable. Consider balanced approach.',
            'analysis': 'Market sentiment service temporarily unavailable.',
            'emoji': '📊',
            'color': 'info',
            'sources': []
        }), 200