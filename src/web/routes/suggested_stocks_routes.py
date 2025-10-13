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
    Get suggested stocks from ALL THREE models (traditional + raw_lstm + kronos) and BOTH risk levels.

    Returns a comprehensive view showing:
    - Traditional Model + Default Risk / High Risk
    - Raw LSTM Model + Default Risk / High Risk
    - Kronos Model + Default Risk / High Risk

    Query parameters:
    - limit: Number of stocks per model/strategy combination (default: from config)
    - date: Date to fetch (default: most recent data)

    Filtering is configuration-driven via config/stock_suggestions.yaml
    """
    try:
        from datetime import datetime, date as dt_date
        from sqlalchemy import text
        from ...models.database import get_database_manager
        from ...services.config.stock_suggestions_config import get_stock_suggestions_config

        # Load configuration
        config = get_stock_suggestions_config()

        # Get parameters (with config-driven defaults)
        limit = int(request.args.get('limit', config.get_default_limit()))
        max_limit = config.get_maximum_limit()
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

        logger.info(f"🎯 Triple model view request: limit={limit}, date={query_date}, user={user_id}")
        logger.info(f"📊 Config-driven filters: "
                   f"recommendations={config.get_allowed_recommendations()}, "
                   f"upside={config.get_minimum_upside_pct()}-{config.get_maximum_upside_pct()}%, "
                   f"max_risk={config.get_maximum_risk_score()}, "
                   f"model_scores=(trad:{config.get_minimum_score('traditional')}, "
                   f"lstm:{config.get_minimum_score('raw_lstm')}, "
                   f"kronos:{config.get_minimum_score('kronos')})")

        db = get_database_manager()

        with db.get_session() as session:
            # Build dynamic SQL query based on configuration
            # Get allowed recommendations from config
            allowed_recs = config.get_allowed_recommendations()
            allowed_recs_str = "', '".join(allowed_recs)

            # Get upside thresholds
            min_upside = config.get_minimum_upside_pct()
            max_upside = config.get_maximum_upside_pct()

            # Get risk threshold
            max_risk = config.get_maximum_risk_score()

            # Get PE ratio range
            pe_range = config.get_pe_ratio_range()

            # Get PB ratio range
            pb_range = config.get_pb_ratio_range()

            # Get ROE range
            roe_range = config.get_roe_range()

            # Build WHERE clause with config-driven filters
            query = text(f"""
                SELECT
                    d.symbol,
                    COALESCE(d.stock_name, s.name) as stock_name,
                    d.current_price,
                    d.model_type,
                    d.strategy,
                    d.ml_prediction_score,
                    d.ml_price_target,
                    d.ml_confidence,
                    d.ml_risk_score,
                    d.recommendation,
                    d.target_price,
                    d.stop_loss,
                    d.rank,
                    d.pe_ratio,
                    d.pb_ratio,
                    d.roe,
                    d.market_cap,
                    d.sector,
                    d.market_cap_category
                FROM daily_suggested_stocks d
                LEFT JOIN stocks s ON d.symbol = s.symbol
                WHERE d.date = :date
                  AND d.recommendation IN ('{allowed_recs_str}')
                  AND d.ml_price_target > d.current_price
                  AND ((d.ml_price_target - d.current_price) / d.current_price * 100) >= :min_upside
                  AND ((d.ml_price_target - d.current_price) / d.current_price * 100) <= :max_upside
                  AND (d.ml_risk_score IS NULL OR d.ml_risk_score <= :max_risk)
                  AND (d.pe_ratio IS NULL OR (d.pe_ratio >= :pe_min AND d.pe_ratio <= :pe_max))
                  AND (d.pb_ratio IS NULL OR (d.pb_ratio >= :pb_min AND d.pb_ratio <= :pb_max))
                  AND (d.roe IS NULL OR (d.roe >= :roe_min AND d.roe <= :roe_max))
                ORDER BY d.model_type, d.strategy, d.ml_prediction_score DESC
            """)

            result = session.execute(query, {
                'date': query_date,
                'min_upside': min_upside,
                'max_upside': max_upside,
                'max_risk': max_risk,
                'pe_min': pe_range['minimum'],
                'pe_max': pe_range['maximum'],
                'pb_min': pb_range['minimum'],
                'pb_max': pb_range['maximum'],
                'roe_min': roe_range['minimum'],
                'roe_max': roe_range['maximum']
            })
            all_stocks = [dict(row._mapping) for row in result]

        # Group results by model_type and strategy
        grouped_results = {
            'traditional': {
                'default_risk': [],
                'high_risk': []
            },
            'raw_lstm': {
                'default_risk': [],
                'high_risk': []
            },
            'kronos': {
                'default_risk': [],
                'high_risk': []
            }
        }

        # Filter stocks by model-specific and strategy-specific thresholds
        for stock in all_stocks:
            model_type = stock['model_type']
            strategy = stock['strategy']

            # Map strategy names (handle case variations)
            if 'default' in strategy.lower() or strategy.upper() == 'DEFAULT_RISK':
                risk_level = 'default_risk'
            elif 'high' in strategy.lower() or strategy.upper() == 'HIGH_RISK':
                risk_level = 'high_risk'
            else:
                # Fallback
                risk_level = 'default_risk'

            # Apply model-specific score threshold from config
            min_score = config.get_minimum_score(model_type)
            if stock['ml_prediction_score'] < min_score:
                logger.debug(f"🚫 Filtered {stock['symbol']} ({model_type}): score {stock['ml_prediction_score']} < {min_score}")
                continue

            # Apply strategy-specific confidence threshold
            min_confidence = config.get_minimum_confidence(risk_level)
            if stock['ml_confidence'] and stock['ml_confidence'] < min_confidence:
                logger.debug(f"🚫 Filtered {stock['symbol']} ({model_type}/{risk_level}): confidence {stock['ml_confidence']} < {min_confidence}")
                continue

            # Initialize if model_type doesn't exist
            if model_type not in grouped_results:
                grouped_results[model_type] = {'default_risk': [], 'high_risk': []}

            # Add to appropriate group
            if risk_level in grouped_results[model_type]:
                grouped_results[model_type][risk_level].append(stock)

        # Apply limit to each group
        for model_type in grouped_results:
            for risk_level in grouped_results[model_type]:
                grouped_results[model_type][risk_level] = grouped_results[model_type][risk_level][:limit]

        # ============================================================
        # Apply Ollama Enhancement (Real-time)
        # ============================================================
        try:
            from src.config.ollama_config import get_ollama_config
            from src.services.data.strategy_ollama_enhancement_service import get_strategy_ollama_enhancement_service

            ollama_config = get_ollama_config()
            daily_pred_config = ollama_config._config.get('daily_predictions', {})

            if daily_pred_config.get('enabled', False):
                logger.info("🔍 Applying Ollama enhancement to UI results...")
                ollama_service = get_strategy_ollama_enhancement_service()
                enhancement_level = 'fast'  # Use fast mode for real-time UI

                enhanced_count = 0
                for model_type in grouped_results:
                    for risk_level in grouped_results[model_type]:
                        stocks = grouped_results[model_type][risk_level]
                        if stocks:
                            try:
                                # Enhance stocks with Ollama
                                enhanced_stocks = ollama_service.enhance_strategy_recommendations(
                                    stocks, risk_level, enhancement_level
                                )
                                grouped_results[model_type][risk_level] = enhanced_stocks
                                enhanced_count += len(enhanced_stocks)
                            except Exception as e:
                                logger.warning(f"⚠️  Ollama enhancement failed for {model_type}/{risk_level}: {e}")

                if enhanced_count > 0:
                    logger.info(f"✅ Ollama enhanced {enhanced_count} stocks for UI")

        except Exception as e:
            logger.warning(f"⚠️  Ollama enhancement unavailable: {e}")
            logger.info("   Continuing without Ollama enhancement...")

        # Calculate statistics
        stats = {}
        for model_type in grouped_results:
            stats[model_type] = {}
            for risk_level in grouped_results[model_type]:
                stocks_in_group = grouped_results[model_type][risk_level]
                stats[model_type][risk_level] = {
                    'count': len(stocks_in_group),
                    'avg_score': sum(s['ml_prediction_score'] or 0 for s in stocks_in_group) / len(stocks_in_group) if stocks_in_group else 0,
                    'avg_confidence': sum(s['ml_confidence'] or 0 for s in stocks_in_group) / len(stocks_in_group) if stocks_in_group else 0,
                }

        response = {
            'success': True,
            'date': str(query_date),
            'limit_per_group': limit,
            'data': grouped_results,
            'statistics': stats,
            'total_stocks': sum(len(grouped_results[mt][rl]) for mt in grouped_results for rl in grouped_results[mt])
        }

        logger.info(f"✅ Triple model view: {response['total_stocks']} total stocks across all groups")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error getting triple model view: {e}", exc_info=True)
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
    Get overall Indian market sentiment using Ollama AI.

    Returns market sentiment analysis including:
    - sentiment_type: 'greedy', 'fear', or 'neutral'
    - recommendation: investment advice
    - analysis: detailed market analysis
    - sources: news sources used
    """
    try:
        logger.info("📊 Market sentiment request received")

        # Get Ollama service
        from ...services.data.strategy_ollama_enhancement_service import get_strategy_ollama_enhancement_service

        ollama_service = get_strategy_ollama_enhancement_service()

        # Get market sentiment
        sentiment_result = ollama_service.get_market_sentiment()

        if sentiment_result.get('success'):
            logger.info(f"✅ Market sentiment: {sentiment_result['sentiment_type'].upper()}")
            return jsonify(sentiment_result), 200
        else:
            logger.warning(f"⚠️ Market sentiment fetch failed: {sentiment_result.get('error')}")
            return jsonify(sentiment_result), 500

    except Exception as e:
        logger.error(f"❌ Error getting market sentiment: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'sentiment_type': 'unknown',
            'recommendation': 'Unable to fetch market sentiment at this time.',
            'emoji': '❌',
            'color': 'secondary'
        }), 500