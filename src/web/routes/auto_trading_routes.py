"""
Auto-Trading Routes
Handles auto-trading settings, execution history, and performance tracking.
"""

import logging
import json
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Create blueprint
auto_trading_bp = Blueprint('auto_trading', __name__, url_prefix='/api/auto-trading')


@auto_trading_bp.route('/settings', methods=['GET'])
@login_required
def get_settings():
    """
    Get auto-trading settings for current user.

    Returns:
        JSON with settings or default values
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import AutoTradingSettings

        user_id = current_user.id

        with get_database_manager().get_session() as session:
            settings = session.query(AutoTradingSettings).filter_by(user_id=user_id).first()

            if not settings:
                # Return default settings
                return jsonify({
                    'success': True,
                    'settings': {
                        'is_enabled': False,
                        'max_amount_per_week': 10000.0,
                        'max_buys_per_week': 5,
                        'preferred_strategies': ['unified'],
                        'minimum_confidence_score': 0.7,
                        'minimum_market_sentiment': 0.0,
                        'auto_stop_loss_enabled': True,
                        'auto_target_price_enabled': True,
                        'execution_time': '09:20'
                    }
                }), 200

            return jsonify({
                'success': True,
                'settings': {
                    'is_enabled': settings.is_enabled,
                    'max_amount_per_week': settings.max_amount_per_week,
                    'max_buys_per_week': settings.max_buys_per_week,
                    'preferred_strategies': json.loads(settings.preferred_strategies or '["unified"]'),
                    'minimum_confidence_score': settings.minimum_confidence_score,
                    'minimum_market_sentiment': settings.minimum_market_sentiment,
                    'auto_stop_loss_enabled': settings.auto_stop_loss_enabled,
                    'auto_target_price_enabled': settings.auto_target_price_enabled,
                    'execution_time': settings.execution_time,
                    'created_at': settings.created_at.isoformat() if settings.created_at else None,
                    'updated_at': settings.updated_at.isoformat() if settings.updated_at else None
                }
            }), 200

    except Exception as e:
        logger.error(f"Error getting auto-trading settings: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@auto_trading_bp.route('/settings', methods=['PUT'])
@login_required
def update_settings():
    """
    Update auto-trading settings for current user.

    Request body:
        {
            "is_enabled": bool,
            "max_amount_per_week": float,
            "max_buys_per_week": int,
            "preferred_strategies": list,
            "minimum_confidence_score": float,
            "minimum_market_sentiment": float,
            "auto_stop_loss_enabled": bool,
            "auto_target_price_enabled": bool,
            "execution_time": string
        }
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import AutoTradingSettings

        user_id = current_user.id
        data = request.get_json()

        # Validate inputs
        if 'max_amount_per_week' in data and data['max_amount_per_week'] < 1000:
            return jsonify({
                'success': False,
                'error': 'Minimum weekly amount is ₹1,000'
            }), 400

        if 'max_buys_per_week' in data and data['max_buys_per_week'] < 1:
            return jsonify({
                'success': False,
                'error': 'Minimum 1 buy per week'
            }), 400

        if 'minimum_confidence_score' in data:
            score = data['minimum_confidence_score']
            if score < 0 or score > 1:
                return jsonify({
                    'success': False,
                    'error': 'Confidence score must be between 0 and 1'
                }), 400

        with get_database_manager().get_session() as session:
            settings = session.query(AutoTradingSettings).filter_by(user_id=user_id).first()

            if not settings:
                # Create new settings
                settings = AutoTradingSettings(user_id=user_id)
                session.add(settings)

            # Update fields
            if 'is_enabled' in data:
                settings.is_enabled = data['is_enabled']
            if 'max_amount_per_week' in data:
                settings.max_amount_per_week = data['max_amount_per_week']
            if 'max_buys_per_week' in data:
                settings.max_buys_per_week = data['max_buys_per_week']
            if 'preferred_strategies' in data:
                settings.preferred_strategies = json.dumps(data['preferred_strategies'])
            if 'minimum_confidence_score' in data:
                settings.minimum_confidence_score = data['minimum_confidence_score']
            if 'minimum_market_sentiment' in data:
                settings.minimum_market_sentiment = data['minimum_market_sentiment']
            if 'auto_stop_loss_enabled' in data:
                settings.auto_stop_loss_enabled = data['auto_stop_loss_enabled']
            if 'auto_target_price_enabled' in data:
                settings.auto_target_price_enabled = data['auto_target_price_enabled']
            if 'execution_time' in data:
                settings.execution_time = data['execution_time']

            session.commit()

            logger.info(f"✅ Auto-trading settings updated for user {user_id}")

            return jsonify({
                'success': True,
                'message': 'Settings updated successfully'
            }), 200

    except Exception as e:
        logger.error(f"Error updating auto-trading settings: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@auto_trading_bp.route('/executions', methods=['GET'])
@login_required
def get_executions():
    """
    Get auto-trading execution history for current user.

    Query parameters:
        - limit: Number of executions to return (default: 30)
        - days: Number of days to look back (default: 30)
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import AutoTradingExecution
        from sqlalchemy import and_

        user_id = current_user.id
        limit = int(request.args.get('limit', 30))
        days = int(request.args.get('days', 30))

        start_date = datetime.now() - timedelta(days=days)

        with get_database_manager().get_session() as session:
            executions = session.query(AutoTradingExecution).filter(
                and_(
                    AutoTradingExecution.user_id == user_id,
                    AutoTradingExecution.execution_date >= start_date
                )
            ).order_by(AutoTradingExecution.execution_date.desc()).limit(limit).all()

            result = []
            for exe in executions:
                result.append({
                    'id': exe.id,
                    'execution_date': exe.execution_date.isoformat(),
                    'status': exe.status,
                    'market_sentiment_type': exe.market_sentiment_type,
                    'market_sentiment_score': exe.market_sentiment_score,
                    'ai_confidence': exe.ai_confidence,
                    'weekly_amount_spent': exe.weekly_amount_spent,
                    'weekly_buys_count': exe.weekly_buys_count,
                    'remaining_weekly_amount': exe.remaining_weekly_amount,
                    'remaining_weekly_buys': exe.remaining_weekly_buys,
                    'orders_created': exe.orders_created,
                    'total_amount_invested': exe.total_amount_invested,
                    'selected_strategies': json.loads(exe.selected_strategies or '[]'),
                    'error_message': exe.error_message
                })

            return jsonify({
                'success': True,
                'total': len(result),
                'executions': result
            }), 200

    except Exception as e:
        logger.error(f"Error getting executions: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@auto_trading_bp.route('/performance', methods=['GET'])
@login_required
def get_performance():
    """
    Get auto-trading performance summary for current user.

    Query parameters:
        - days: Number of days to look back (default: 30)
    """
    try:
        from src.services.trading.order_performance_tracking_service import get_performance_tracking_service

        user_id = current_user.id
        days = int(request.args.get('days', 30))

        performance_service = get_performance_tracking_service()
        result = performance_service.get_performance_summary(user_id, days)

        return jsonify(result), 200 if result.get('success') else 500

    except Exception as e:
        logger.error(f"Error getting performance: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@auto_trading_bp.route('/orders', methods=['GET'])
@login_required
def get_orders():
    """
    Get active and closed orders with performance data.

    Query parameters:
        - status: 'active' or 'closed' (default: 'active')
        - limit: Number of orders to return (default: 50)
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import OrderPerformance

        user_id = current_user.id
        status = request.args.get('status', 'active')
        limit = int(request.args.get('limit', 50))

        with get_database_manager().get_session() as session:
            query = session.query(OrderPerformance).filter_by(user_id=user_id)

            if status == 'active':
                query = query.filter_by(is_active=True)
            elif status == 'closed':
                query = query.filter_by(is_active=False)

            orders = query.order_by(OrderPerformance.created_at.desc()).limit(limit).all()

            result = []
            for order in orders:
                result.append({
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'entry_price': order.entry_price,
                    'quantity': order.quantity,
                    'stop_loss': order.stop_loss,
                    'target_price': order.target_price,
                    'strategy': order.strategy,
                    'current_price': order.current_price,
                    'unrealized_pnl': order.unrealized_pnl,
                    'unrealized_pnl_pct': order.unrealized_pnl_pct,
                    'realized_pnl': order.realized_pnl,
                    'realized_pnl_pct': order.realized_pnl_pct,
                    'exit_price': order.exit_price,
                    'exit_date': order.exit_date.isoformat() if order.exit_date else None,
                    'exit_reason': order.exit_reason,
                    'days_held': order.days_held,
                    'is_active': order.is_active,
                    'is_profitable': order.is_profitable,
                    'performance_rating': order.performance_rating,
                    'created_at': order.created_at.isoformat(),
                    'last_checked_at': order.last_checked_at.isoformat() if order.last_checked_at else None
                })

            return jsonify({
                'success': True,
                'total': len(result),
                'orders': result
            }), 200

    except Exception as e:
        logger.error(f"Error getting orders: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@auto_trading_bp.route('/test', methods=['POST'])
@login_required
def test_execution():
    """
    Manually trigger auto-trading execution for testing.

    Note: This will respect all limits and checks, just like the scheduled execution.
    """
    try:
        from src.services.trading.auto_trading_service import get_auto_trading_service

        user_id = current_user.id

        logger.info(f"🧪 Manual auto-trading test triggered by user {user_id}")

        auto_trading_service = get_auto_trading_service()
        result = auto_trading_service.execute_auto_trading_for_user(user_id)

        return jsonify(result), 200 if result.get('success') else 500

    except Exception as e:
        logger.error(f"Error in test execution: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@auto_trading_bp.route('/weekly-status', methods=['GET'])
@login_required
def get_weekly_status():
    """
    Get current week's trading status (amount spent, trades made).
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import AutoTradingSettings, Order
        from sqlalchemy import and_, func

        user_id = current_user.id

        # Get start of current week (Monday)
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)

        with get_database_manager().get_session() as session:
            # Get settings
            settings = session.query(AutoTradingSettings).filter_by(user_id=user_id).first()

            if not settings:
                return jsonify({
                    'success': False,
                    'error': 'Auto-trading not configured'
                }), 404

            # Query weekly orders
            weekly_orders = session.query(
                func.count(Order.id).label('count'),
                func.coalesce(func.sum(Order.price * Order.quantity), 0.0).label('total')
            ).filter(
                and_(
                    Order.user_id == user_id,
                    Order.created_at >= start_of_week,
                    Order.transaction_type == 'BUY',
                    Order.order_status.in_(['COMPLETE', 'EXECUTED'])
                )
            ).first()

            weekly_buys = weekly_orders.count if weekly_orders else 0
            weekly_spent = float(weekly_orders.total) if weekly_orders else 0.0

            # Calculate remaining
            remaining_amount = settings.max_amount_per_week - weekly_spent
            remaining_buys = settings.max_buys_per_week - weekly_buys

            # Calculate percentages
            amount_used_pct = (weekly_spent / settings.max_amount_per_week) * 100 if settings.max_amount_per_week > 0 else 0
            buys_used_pct = (weekly_buys / settings.max_buys_per_week) * 100 if settings.max_buys_per_week > 0 else 0

            return jsonify({
                'success': True,
                'week_start': start_of_week.isoformat(),
                'week_end': (start_of_week + timedelta(days=7)).isoformat(),
                'limits': {
                    'max_amount': settings.max_amount_per_week,
                    'max_buys': settings.max_buys_per_week
                },
                'spent': {
                    'amount': weekly_spent,
                    'buys': weekly_buys
                },
                'remaining': {
                    'amount': max(0, remaining_amount),
                    'buys': max(0, remaining_buys)
                },
                'percentage_used': {
                    'amount': round(amount_used_pct, 1),
                    'buys': round(buys_used_pct, 1)
                }
            }), 200

    except Exception as e:
        logger.error(f"Error getting weekly status: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
