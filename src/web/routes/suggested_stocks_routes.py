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
    Get suggested stocks from pre-calculated daily recommendations.

    Query parameters:
    - limit: Number of suggestions to return (default: 50)
    - search: Optional search term
    - sort_by: Sort field (default: 'selection_score')
    - sort_order: 'asc' or 'desc' (default: 'desc')
    - sector: Optional sector filter
    """
    try:
        from datetime import datetime, date as dt_date
        from sqlalchemy import text
        from ...models.database import get_database_manager

        # Get parameters
        limit = int(request.args.get('limit', 50))
        search = request.args.get('search')
        sort_by = request.args.get('sort_by', 'selection_score')
        sort_order = request.args.get('sort_order', 'desc')
        sector = request.args.get('sector')

        user_id = current_user.id

        logger.info(f"🎯 Suggested stocks request: limit={limit}, user={user_id}")

        db = get_database_manager()

        with db.get_session() as session:
            # Get the most recent date with data
            recent_date_result = session.execute(text("""
                SELECT MAX(date) as max_date FROM daily_suggested_stocks
            """)).first()

            if recent_date_result and recent_date_result[0]:
                query_date = recent_date_result[0]
            else:
                return jsonify({
                    'success': True,
                    'data': [],
                    'total': 0,
                    'message': 'No recommendations available yet'
                }), 200

            # Build query with filters - use subquery with DISTINCT ON to avoid duplicates
            # Then wrap in outer query to apply proper sorting
            # Also show both buy AND sell signals (not just buy_signal = TRUE)
            valid_sort_fields = ['selection_score', 'current_price', 'ema_trend_score', 'demarker', 'market_cap']
            if sort_by not in valid_sort_fields:
                sort_by = 'selection_score'
            order_dir = 'ASC' if sort_order.lower() == 'asc' else 'DESC'

            # Build where clause for inner query
            # Show ALL suggested stocks - they're in the snapshot for a reason
            # Fresh signals come from stocks table via COALESCE
            where_clauses = ["d.date = :date"]
            params = {'date': query_date, 'limit': limit}

            if search:
                where_clauses.append("(d.symbol ILIKE :search OR d.stock_name ILIKE :search)")
                params['search'] = f'%{search}%'

            if sector:
                where_clauses.append("d.sector = :sector")
                params['sector'] = sector

            where_clause = " AND ".join(where_clauses)

            # Use subquery to deduplicate, then sort in outer query
            query_sql = f"""
                SELECT * FROM (
                    SELECT DISTINCT ON (d.symbol)
                        d.symbol,
                        COALESCE(d.stock_name, s.name) as stock_name,
                        COALESCE(s.current_price, d.current_price) as current_price,
                        d.strategy,
                        d.selection_score,
                        COALESCE(s.ema_8, d.ema_8) as ema_8,
                        COALESCE(s.ema_21, d.ema_21) as ema_21,
                        d.ema_trend_score,
                        COALESCE(s.demarker, d.demarker) as demarker,
                        d.fib_target_1,
                        d.fib_target_2,
                        d.fib_target_3,
                        COALESCE(s.buy_signal, d.buy_signal) as buy_signal,
                        COALESCE(s.sell_signal, d.sell_signal) as sell_signal,
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
                    WHERE {where_clause}
                    ORDER BY d.symbol, d.selection_score DESC
                ) AS unique_stocks
                ORDER BY {sort_by} {order_dir}
                LIMIT :limit
            """

            query = text(query_sql)
            result = session.execute(query, params)
            stocks = [dict(row._mapping) for row in result]

        logger.info(f"✅ Returned {len(stocks)} suggested stocks from {query_date}")

        return jsonify({
            'success': True,
            'data': stocks,
            'total': len(stocks),
            'date': str(query_date),
            'last_updated': datetime.now().isoformat()
        }), 200

    except ValueError as e:
        logger.error(f"Invalid parameter: {e}")
        return jsonify({
            'success': False,
            'error': f'Invalid parameter: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"Error getting suggested stocks: {e}", exc_info=True)
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
            # Query 8-21 EMA strategy stocks with fresh data from stocks table
            # Use DISTINCT ON to avoid duplicates from multiple strategies
            query = text("""
                SELECT * FROM (
                    SELECT DISTINCT ON (d.symbol)
                        d.symbol,
                        COALESCE(d.stock_name, s.name) as stock_name,
                        COALESCE(s.current_price, d.current_price) as current_price,
                        d.strategy,
                        d.selection_score,
                        COALESCE(s.ema_8, d.ema_8) as ema_8,
                        COALESCE(s.ema_21, d.ema_21) as ema_21,
                        d.ema_trend_score,
                        COALESCE(s.demarker, d.demarker) as demarker,
                        d.fib_target_1,
                        d.fib_target_2,
                        d.fib_target_3,
                        COALESCE(s.buy_signal, d.buy_signal) as buy_signal,
                        COALESCE(s.sell_signal, d.sell_signal) as sell_signal,
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
                    ORDER BY d.symbol, d.selection_score DESC
                ) AS unique_stocks
                ORDER BY selection_score DESC, ema_trend_score DESC
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
            strategy = stock.get('strategy', '')
            market_cap = stock.get('market_cap_category', '')

            # Map strategy names (handle case variations)
            if 'default' in strategy.lower() or strategy.upper() == 'DEFAULT_RISK':
                risk_level = 'default_risk'
            elif 'high' in strategy.lower() or strategy.upper() == 'HIGH_RISK':
                risk_level = 'high_risk'
            elif strategy.lower() == 'unified':
                # For unified strategy, categorize by market cap:
                # large_cap -> default_risk (conservative)
                # mid_cap, small_cap -> high_risk (aggressive)
                if market_cap == 'large_cap':
                    risk_level = 'default_risk'
                else:
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


@suggested_stocks_bp.route('/recalculate', methods=['POST'])
@login_required
def recalculate_suggestions():
    """
    Trigger on-demand recalculation of suggested stocks.

    This runs the full 8-21 EMA strategy pipeline and updates
    the daily_suggested_stocks table with fresh recommendations.
    """
    try:
        import threading
        from datetime import datetime

        user_id = current_user.id
        logger.info(f"🔄 Manual recalculation requested by user {user_id}")

        # Check if user is admin (optional - you can remove this check)
        # if not current_user.is_admin:
        #     return jsonify({
        #         'success': False,
        #         'error': 'Admin privileges required for recalculation'
        #     }), 403

        def run_recalculation():
            """Background task to run recalculation."""
            try:
                from ...services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator

                logger.info("📊 Starting suggested stocks recalculation...")

                orchestrator = SuggestedStocksSagaOrchestrator()
                result = orchestrator.execute_suggested_stocks_saga(
                    user_id=1,  # System user
                    strategies=['unified'],
                    limit=5  # 13mo backtest-optimized: top 5 picks for PF 3.84
                )

                if result.get('success'):
                    logger.info(f"✅ Recalculation completed: {result.get('total_stocks', 0)} stocks")
                else:
                    logger.error(f"❌ Recalculation failed: {result.get('error')}")

            except Exception as e:
                logger.error(f"❌ Recalculation error: {e}", exc_info=True)

        # Run in background thread to avoid timeout
        thread = threading.Thread(target=run_recalculation, daemon=True)
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Recalculation started in background. Refresh the page in a few moments to see updated results.',
            'started_at': datetime.now().isoformat()
        }), 202  # 202 Accepted

    except Exception as e:
        logger.error(f"Error starting recalculation: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Failed to start recalculation: {str(e)}'
        }), 500


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