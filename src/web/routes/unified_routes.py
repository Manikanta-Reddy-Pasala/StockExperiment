"""
Unified Multi-Broker API Routes

This module provides unified API routes that automatically use the appropriate
broker implementation based on user settings. It follows SOLID principles by
using the broker feature factory to get the correct provider for each feature.
"""

from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

try:
    from ...services.interfaces.broker_feature_factory import get_broker_feature_factory
    from ...services.interfaces.suggested_stocks_interface import StrategyType
    from ...services.interfaces.orders_interface import OrderType, OrderSide
    from ...services.interfaces.reports_interface import ReportType, ReportFormat
except ImportError:
    from services.interfaces.broker_feature_factory import get_broker_feature_factory
    from services.interfaces.suggested_stocks_interface import StrategyType
    from services.interfaces.orders_interface import OrderType, OrderSide
    from services.interfaces.reports_interface import ReportType, ReportFormat

# Create Blueprint for unified routes
unified_bp = Blueprint('unified', __name__, url_prefix='/api/unified')


# ============================================================================
# DASHBOARD ROUTES
# ============================================================================

@unified_bp.route('/dashboard/market-overview', methods=['GET'])
@login_required
def api_get_market_overview():
    """Get market overview using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_dashboard_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No dashboard provider available for your selected broker'
            }), 400
        
        result = provider.get_market_overview(current_user.id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching market overview for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/dashboard/portfolio-summary', methods=['GET'])
@login_required
def api_get_portfolio_summary():
    """Get portfolio summary using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_dashboard_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No dashboard provider available for your selected broker'
            }), 400
        
        result = provider.get_portfolio_summary(current_user.id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching portfolio summary for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/dashboard/top-holdings', methods=['GET'])
@login_required
def api_get_top_holdings():
    """Get top holdings using user's selected broker."""
    try:
        limit = request.args.get('limit', 5, type=int)
        factory = get_broker_feature_factory()
        provider = factory.get_dashboard_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No dashboard provider available for your selected broker'
            }), 400
        
        result = provider.get_top_holdings(current_user.id, limit)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching top holdings for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/dashboard/recent-activity', methods=['GET'])
@login_required
def api_get_recent_activity():
    """Get recent activity using user's selected broker."""
    try:
        limit = request.args.get('limit', 10, type=int)
        factory = get_broker_feature_factory()
        provider = factory.get_dashboard_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No dashboard provider available for your selected broker'
            }), 400
        
        result = provider.get_recent_activity(current_user.id, limit)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching recent activity for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/dashboard/account-balance', methods=['GET'])
@login_required
def api_get_account_balance():
    """Get account balance using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_dashboard_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No dashboard provider available for your selected broker'
            }), 400
        
        result = provider.get_account_balance(current_user.id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching account balance for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/dashboard/daily-pnl-chart', methods=['GET'])
@login_required
def api_get_daily_pnl_chart():
    """Get daily P&L chart data using user's selected broker."""
    try:
        days = request.args.get('days', 30, type=int)
        factory = get_broker_feature_factory()
        provider = factory.get_dashboard_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No dashboard provider available for your selected broker'
            }), 400
        
        result = provider.get_daily_pnl_chart_data(current_user.id, days)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching daily P&L chart for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/dashboard/performance-metrics', methods=['GET'])
@login_required
def api_get_performance_metrics():
    """Get performance metrics using user's selected broker."""
    try:
        period = request.args.get('period', '1M')
        factory = get_broker_feature_factory()
        provider = factory.get_dashboard_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No dashboard provider available for your selected broker'
            }), 400
        
        result = provider.get_performance_metrics(current_user.id, period)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching performance metrics for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/dashboard/watchlist-quotes', methods=['GET'])
@login_required
def api_get_watchlist_quotes():
    """Get watchlist quotes using user's selected broker."""
    try:
        symbols_param = request.args.get('symbols', '')
        symbols = symbols_param.split(',') if symbols_param else None
        
        factory = get_broker_feature_factory()
        provider = factory.get_dashboard_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No dashboard provider available for your selected broker'
            }), 400
        
        result = provider.get_watchlist_quotes(current_user.id, symbols)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching watchlist quotes for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# SUGGESTED STOCKS ROUTES
# ============================================================================

@unified_bp.route('/suggested-stocks', methods=['GET'])
@login_required
def api_get_suggested_stocks():
    """Get suggested stocks using user's selected broker."""
    try:
        strategies_list = request.args.getlist('strategies')
        strategies = []
        if strategies_list:
            for s in strategies_list:
                try:
                    strategies.append(StrategyType(s))
                except ValueError:
                    logger.warning(f"Invalid strategy type: {s}")
                    continue
        
        if not strategies:
            strategies = None
            
        limit = request.args.get('limit', 50, type=int)
        time_filter = request.args.get('time_filter', 'week')  # For future use
        
        factory = get_broker_feature_factory()
        provider = factory.get_suggested_stocks_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No suggested stocks provider available for your selected broker'
            }), 400
        
        result = provider.get_suggested_stocks(current_user.id, strategies, limit)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching suggested stocks for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/suggested-stocks/analysis/<symbol>', methods=['GET'])
@login_required
def api_get_stock_analysis(symbol):
    """Get stock analysis using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_suggested_stocks_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No suggested stocks provider available for your selected broker'
            }), 400
        
        result = provider.get_stock_analysis(current_user.id, symbol)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching stock analysis for user {current_user.id}, symbol {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/suggested-stocks/strategy-performance', methods=['GET'])
@login_required
def api_get_strategy_performance():
    """Get strategy performance using user's selected broker."""
    try:
        strategy_param = request.args.get('strategy')
        period = request.args.get('period', '1M')
        
        if not strategy_param:
            return jsonify({
                'success': False,
                'error': 'Strategy parameter is required'
            }), 400
        
        strategy = StrategyType(strategy_param)
        factory = get_broker_feature_factory()
        provider = factory.get_suggested_stocks_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No suggested stocks provider available for your selected broker'
            }), 400
        
        result = provider.get_strategy_performance(current_user.id, strategy, period)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching strategy performance for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/suggested-stocks/sector-analysis', methods=['GET'])
@login_required
def api_get_sector_analysis():
    """Get sector analysis using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_suggested_stocks_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No suggested stocks provider available for your selected broker'
            }), 400
        
        result = provider.get_sector_analysis(current_user.id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching sector analysis for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/suggested-stocks/technical-screener', methods=['POST'])
@login_required
def api_technical_screener():
    """Screen stocks using technical criteria and user's selected broker."""
    try:
        criteria = request.get_json()
        if not criteria:
            return jsonify({
                'success': False,
                'error': 'Screening criteria is required'
            }), 400
        
        factory = get_broker_feature_factory()
        provider = factory.get_suggested_stocks_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No suggested stocks provider available for your selected broker'
            }), 400
        
        result = provider.get_technical_screener(current_user.id, criteria)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error running technical screener for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/suggested-stocks/fundamental-screener', methods=['POST'])
@login_required
def api_fundamental_screener():
    """Screen stocks using fundamental criteria and user's selected broker."""
    try:
        criteria = request.get_json()
        if not criteria:
            return jsonify({
                'success': False,
                'error': 'Screening criteria is required'
            }), 400
        
        factory = get_broker_feature_factory()
        provider = factory.get_suggested_stocks_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No suggested stocks provider available for your selected broker'
            }), 400
        
        result = provider.get_fundamental_screener(current_user.id, criteria)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error running fundamental screener for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# ORDERS ROUTES
# ============================================================================

@unified_bp.route('/orders/history', methods=['GET'])
@login_required
def api_get_orders_history():
    """Get orders history using user's selected broker."""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', 100, type=int)
        
        # Parse dates if provided
        start_date_obj = datetime.fromisoformat(start_date) if start_date else None
        end_date_obj = datetime.fromisoformat(end_date) if end_date else None
        
        factory = get_broker_feature_factory()
        provider = factory.get_orders_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No orders provider available for your selected broker'
            }), 400
        
        result = provider.get_orders_history(current_user.id, start_date_obj, end_date_obj, limit)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching orders history for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/orders/pending', methods=['GET'])
@login_required
def api_get_pending_orders():
    """Get pending orders using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_orders_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No orders provider available for your selected broker'
            }), 400
        
        result = provider.get_pending_orders(current_user.id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching pending orders for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/orders/trades', methods=['GET'])
@login_required
def api_get_trades_history():
    """Get trades history using user's selected broker."""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', 100, type=int)
        
        # Parse dates if provided
        start_date_obj = datetime.fromisoformat(start_date) if start_date else None
        end_date_obj = datetime.fromisoformat(end_date) if end_date else None
        
        factory = get_broker_feature_factory()
        provider = factory.get_orders_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No orders provider available for your selected broker'
            }), 400
        
        result = provider.get_trades_history(current_user.id, start_date_obj, end_date_obj, limit)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching trades history for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/orders/place', methods=['POST'])
@login_required
def api_place_order():
    """Place an order using user's selected broker."""
    try:
        order_data = request.get_json()
        if not order_data:
            return jsonify({
                'success': False,
                'error': 'Order data is required'
            }), 400
        
        factory = get_broker_feature_factory()
        provider = factory.get_orders_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No orders provider available for your selected broker'
            }), 400
        
        result = provider.place_order(current_user.id, order_data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error placing order for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/orders/modify/<order_id>', methods=['PUT'])
@login_required
def api_modify_order(order_id):
    """Modify an order using user's selected broker."""
    try:
        order_data = request.get_json()
        if not order_data:
            return jsonify({
                'success': False,
                'error': 'Order data is required'
            }), 400
        
        factory = get_broker_feature_factory()
        provider = factory.get_orders_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No orders provider available for your selected broker'
            }), 400
        
        result = provider.modify_order(current_user.id, order_id, order_data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error modifying order {order_id} for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/orders/cancel/<order_id>', methods=['DELETE'])
@login_required
def api_cancel_order(order_id):
    """Cancel an order using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_orders_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No orders provider available for your selected broker'
            }), 400
        
        result = provider.cancel_order(current_user.id, order_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error cancelling order {order_id} for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/orders/details/<order_id>', methods=['GET'])
@login_required
def api_get_order_details(order_id):
    """Get order details using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_orders_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No orders provider available for your selected broker'
            }), 400
        
        result = provider.get_order_details(current_user.id, order_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching order details {order_id} for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# PORTFOLIO ROUTES
# ============================================================================

@unified_bp.route('/portfolio/holdings', methods=['GET'])
@login_required
def api_get_holdings():
    """Get holdings using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_portfolio_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No portfolio provider available for your selected broker'
            }), 400
        
        result = provider.get_holdings(current_user.id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching holdings for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/portfolio/positions', methods=['GET'])
@login_required
def api_get_positions():
    """Get positions using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_portfolio_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No portfolio provider available for your selected broker'
            }), 400
        
        result = provider.get_positions(current_user.id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching positions for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/portfolio/summary', methods=['GET'])
@login_required
def api_get_portfolio_summary_detailed():
    """Get portfolio summary using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_portfolio_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No portfolio provider available for your selected broker'
            }), 400
        
        result = provider.get_portfolio_summary(current_user.id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching portfolio summary for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/portfolio/allocation', methods=['GET'])
@login_required
def api_get_portfolio_allocation():
    """Get portfolio allocation using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_portfolio_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No portfolio provider available for your selected broker'
            }), 400
        
        result = provider.get_portfolio_allocation(current_user.id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching portfolio allocation for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/portfolio/performance', methods=['GET'])
@login_required
def api_get_portfolio_performance():
    """Get portfolio performance using user's selected broker."""
    try:
        period = request.args.get('period', '1M')
        factory = get_broker_feature_factory()
        provider = factory.get_portfolio_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No portfolio provider available for your selected broker'
            }), 400
        
        result = provider.get_portfolio_performance(current_user.id, period)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching portfolio performance for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/portfolio/dividends', methods=['GET'])
@login_required
def api_get_dividend_history():
    """Get dividend history using user's selected broker."""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Parse dates if provided
        start_date_obj = datetime.fromisoformat(start_date) if start_date else None
        end_date_obj = datetime.fromisoformat(end_date) if end_date else None
        
        factory = get_broker_feature_factory()
        provider = factory.get_portfolio_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No portfolio provider available for your selected broker'
            }), 400
        
        result = provider.get_dividend_history(current_user.id, start_date_obj, end_date_obj)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching dividend history for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/portfolio/risk-metrics', methods=['GET'])
@login_required
def api_get_portfolio_risk_metrics():
    """Get portfolio risk metrics using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_portfolio_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No portfolio provider available for your selected broker'
            }), 400
        
        result = provider.get_portfolio_risk_metrics(current_user.id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching portfolio risk metrics for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# REPORTS ROUTES
# ============================================================================

@unified_bp.route('/reports/pnl', methods=['POST'])
@login_required
def api_generate_pnl_report():
    """Generate P&L report using user's selected broker."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Report parameters are required'
            }), 400
        
        start_date = datetime.fromisoformat(data.get('start_date'))
        end_date = datetime.fromisoformat(data.get('end_date'))
        report_format = ReportFormat(data.get('format', 'json'))
        
        factory = get_broker_feature_factory()
        provider = factory.get_reports_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No reports provider available for your selected broker'
            }), 400
        
        result = provider.generate_pnl_report(current_user.id, start_date, end_date, report_format)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating P&L report for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/reports/tax', methods=['POST'])
@login_required
def api_generate_tax_report():
    """Generate tax report using user's selected broker."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Report parameters are required'
            }), 400
        
        financial_year = data.get('financial_year')
        report_format = ReportFormat(data.get('format', 'json'))
        
        if not financial_year:
            return jsonify({
                'success': False,
                'error': 'Financial year is required'
            }), 400
        
        factory = get_broker_feature_factory()
        provider = factory.get_reports_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No reports provider available for your selected broker'
            }), 400
        
        result = provider.generate_tax_report(current_user.id, financial_year, report_format)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating tax report for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/reports/portfolio', methods=['POST'])
@login_required
def api_generate_portfolio_report():
    """Generate portfolio report using user's selected broker."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Report parameters are required'
            }), 400
        
        report_type = ReportType(data.get('type', 'monthly'))
        report_format = ReportFormat(data.get('format', 'json'))
        
        factory = get_broker_feature_factory()
        provider = factory.get_reports_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No reports provider available for your selected broker'
            }), 400
        
        result = provider.generate_portfolio_report(current_user.id, report_type, report_format)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating portfolio report for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/reports/trading-summary', methods=['POST'])
@login_required
def api_generate_trading_summary():
    """Generate trading summary report using user's selected broker."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Report parameters are required'
            }), 400
        
        start_date = datetime.fromisoformat(data.get('start_date'))
        end_date = datetime.fromisoformat(data.get('end_date'))
        report_format = ReportFormat(data.get('format', 'json'))
        
        factory = get_broker_feature_factory()
        provider = factory.get_reports_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No reports provider available for your selected broker'
            }), 400
        
        result = provider.generate_trading_summary(current_user.id, start_date, end_date, report_format)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating trading summary for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/reports/history', methods=['GET'])
@login_required
def api_get_report_history():
    """Get report history using user's selected broker."""
    try:
        limit = request.args.get('limit', 50, type=int)
        factory = get_broker_feature_factory()
        provider = factory.get_reports_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No reports provider available for your selected broker'
            }), 400
        
        result = provider.get_report_history(current_user.id, limit)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching report history for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@unified_bp.route('/reports/download/<report_id>', methods=['GET'])
@login_required
def api_download_report(report_id):
    """Download a report using user's selected broker."""
    try:
        factory = get_broker_feature_factory()
        provider = factory.get_reports_provider(current_user.id)
        
        if not provider:
            return jsonify({
                'success': False,
                'error': 'No reports provider available for your selected broker'
            }), 400
        
        result = provider.download_report(current_user.id, report_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error downloading report {report_id} for user {current_user.id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# BROKER INFO ROUTES
# ============================================================================

@unified_bp.route('/broker/info', methods=['GET'])
@login_required
def api_get_broker_info():
    """Get information about available brokers and their features."""
    try:
        factory = get_broker_feature_factory()
        available_brokers = factory.get_available_brokers()
        
        return jsonify({
            'success': True,
            'data': available_brokers,
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching broker info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
