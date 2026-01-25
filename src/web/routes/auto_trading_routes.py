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


# ==========================================
# PAPER TRADING ENDPOINTS
# ==========================================

@auto_trading_bp.route('/paper-trading/status', methods=['GET'])
@login_required
def get_paper_trading_status():
    """
    Get paper trading status and mode for current user.
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import User

        user_id = current_user.id

        with get_database_manager().get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()

            if not user:
                return jsonify({
                    'success': False,
                    'error': 'User not found'
                }), 404

            return jsonify({
                'success': True,
                'paper_trading_enabled': user.is_mock_trading_mode,
                'mode': 'paper' if user.is_mock_trading_mode else 'live'
            }), 200

    except Exception as e:
        logger.error(f"Error getting paper trading status: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@auto_trading_bp.route('/paper-trading/toggle', methods=['POST'])
@login_required
def toggle_paper_trading():
    """
    Toggle paper trading mode on/off.
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import User

        user_id = current_user.id
        data = request.get_json() or {}
        enable = data.get('enable', True)

        with get_database_manager().get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()

            if not user:
                return jsonify({
                    'success': False,
                    'error': 'User not found'
                }), 404

            user.is_mock_trading_mode = enable
            session.commit()

            return jsonify({
                'success': True,
                'paper_trading_enabled': user.is_mock_trading_mode,
                'mode': 'paper' if user.is_mock_trading_mode else 'live',
                'message': f"Paper trading {'enabled' if enable else 'disabled'}"
            }), 200

    except Exception as e:
        logger.error(f"Error toggling paper trading: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@auto_trading_bp.route('/paper-trading/portfolio', methods=['GET'])
@login_required
def get_paper_portfolio():
    """
    Get paper trading portfolio summary with P&L.

    Returns:
        - Initial capital (configurable, default 100,000)
        - Current capital (cash + positions value)
        - Cash available
        - Positions value
        - Total P&L (realized + unrealized)
        - Total P&L percentage
        - Active positions count
        - Win rate
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import OrderPerformance, Order, User
        from sqlalchemy import func, and_

        user_id = current_user.id

        # Default initial capital for paper trading
        INITIAL_CAPITAL = 100000.0

        with get_database_manager().get_session() as session:
            # Verify user is in paper trading mode
            user = session.query(User).filter_by(id=user_id).first()
            if not user or not user.is_mock_trading_mode:
                return jsonify({
                    'success': False,
                    'error': 'Paper trading not enabled for this user'
                }), 400

            # Get all paper trading orders (is_mock_order = True)
            mock_orders = session.query(Order).filter(
                and_(
                    Order.user_id == user_id,
                    Order.is_mock_order == True,
                    Order.transaction_type == 'BUY',
                    Order.order_status.in_(['COMPLETE', 'EXECUTED'])
                )
            ).all()

            order_ids = [o.id for o in mock_orders]

            if not order_ids:
                # No paper trades yet
                return jsonify({
                    'success': True,
                    'initial_capital': INITIAL_CAPITAL,
                    'current_capital': INITIAL_CAPITAL,
                    'cash_available': INITIAL_CAPITAL,
                    'positions_value': 0.0,
                    'total_invested': 0.0,
                    'total_pnl': 0.0,
                    'total_pnl_pct': 0.0,
                    'realized_pnl': 0.0,
                    'unrealized_pnl': 0.0,
                    'active_positions': 0,
                    'closed_positions': 0,
                    'profitable_trades': 0,
                    'loss_trades': 0,
                    'win_rate': 0.0
                }), 200

            # Get performance data for paper orders
            performances = session.query(OrderPerformance).filter(
                OrderPerformance.order_id.in_(order_ids)
            ).all()

            # Calculate portfolio metrics
            active_positions = [p for p in performances if p.is_active]
            closed_positions = [p for p in performances if not p.is_active]

            # Calculate values
            total_invested = sum((p.entry_price or 0) * (p.quantity or 0) for p in performances)
            positions_value = sum((p.current_price or p.entry_price or 0) * (p.quantity or 0) for p in active_positions)

            realized_pnl = sum(p.realized_pnl or 0 for p in closed_positions)
            unrealized_pnl = sum(p.unrealized_pnl or 0 for p in active_positions)
            total_pnl = realized_pnl + unrealized_pnl

            # Calculate cash: initial capital - invested in active positions + realized gains/losses
            active_invested = sum((p.entry_price or 0) * (p.quantity or 0) for p in active_positions)
            cash_available = INITIAL_CAPITAL - active_invested + realized_pnl

            current_capital = cash_available + positions_value

            # Calculate win rate
            profitable_trades = len([p for p in closed_positions if (p.realized_pnl or 0) > 0])
            loss_trades = len([p for p in closed_positions if (p.realized_pnl or 0) < 0])
            total_closed = len(closed_positions)
            win_rate = (profitable_trades / total_closed * 100) if total_closed > 0 else 0.0

            # Calculate P&L percentage
            total_pnl_pct = (total_pnl / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL > 0 else 0.0

            return jsonify({
                'success': True,
                'initial_capital': INITIAL_CAPITAL,
                'current_capital': round(current_capital, 2),
                'cash_available': round(cash_available, 2),
                'positions_value': round(positions_value, 2),
                'total_invested': round(total_invested, 2),
                'total_pnl': round(total_pnl, 2),
                'total_pnl_pct': round(total_pnl_pct, 2),
                'realized_pnl': round(realized_pnl, 2),
                'unrealized_pnl': round(unrealized_pnl, 2),
                'active_positions': len(active_positions),
                'closed_positions': len(closed_positions),
                'profitable_trades': profitable_trades,
                'loss_trades': loss_trades,
                'win_rate': round(win_rate, 2)
            }), 200

    except Exception as e:
        logger.error(f"Error getting paper portfolio: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@auto_trading_bp.route('/paper-trading/daily-pnl', methods=['GET'])
@login_required
def get_paper_daily_pnl():
    """
    Get daily P&L history for paper trading portfolio.

    Query parameters:
        - days: Number of days to retrieve (default: 30)

    Returns:
        Array of daily P&L data points for charting.
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import OrderPerformance, OrderPerformanceSnapshot, Order, User
        from sqlalchemy import func, and_
        from datetime import date

        user_id = current_user.id
        days = int(request.args.get('days', 30))

        INITIAL_CAPITAL = 100000.0
        start_date = date.today() - timedelta(days=days)

        with get_database_manager().get_session() as session:
            # Verify user is in paper trading mode
            user = session.query(User).filter_by(id=user_id).first()
            if not user or not user.is_mock_trading_mode:
                return jsonify({
                    'success': False,
                    'error': 'Paper trading not enabled for this user'
                }), 400

            # Get paper trading order IDs
            mock_orders = session.query(Order).filter(
                and_(
                    Order.user_id == user_id,
                    Order.is_mock_order == True,
                    Order.transaction_type == 'BUY',
                    Order.order_status.in_(['COMPLETE', 'EXECUTED'])
                )
            ).all()

            order_ids = [o.id for o in mock_orders]

            if not order_ids:
                # Return empty chart data
                return jsonify({
                    'success': True,
                    'daily_pnl': [],
                    'total_days': 0
                }), 200

            # Get performance IDs for these orders
            perf_ids = session.query(OrderPerformance.id).filter(
                OrderPerformance.order_id.in_(order_ids)
            ).all()
            perf_ids = [p[0] for p in perf_ids]

            if not perf_ids:
                return jsonify({
                    'success': True,
                    'daily_pnl': [],
                    'total_days': 0
                }), 200

            # Query daily snapshots grouped by date
            daily_data = session.query(
                OrderPerformanceSnapshot.snapshot_date,
                func.sum(OrderPerformanceSnapshot.unrealized_pnl).label('total_pnl'),
                func.sum(OrderPerformanceSnapshot.value).label('total_value'),
                func.count(OrderPerformanceSnapshot.id).label('positions_count')
            ).filter(
                and_(
                    OrderPerformanceSnapshot.order_performance_id.in_(perf_ids),
                    OrderPerformanceSnapshot.snapshot_date >= start_date
                )
            ).group_by(OrderPerformanceSnapshot.snapshot_date).order_by(
                OrderPerformanceSnapshot.snapshot_date
            ).all()

            # Format response
            daily_pnl = []
            for row in daily_data:
                pnl = float(row.total_pnl or 0)
                pnl_pct = (pnl / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL > 0 else 0

                daily_pnl.append({
                    'date': row.snapshot_date.isoformat() if row.snapshot_date else None,
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl_pct, 2),
                    'portfolio_value': round(float(row.total_value or 0), 2),
                    'positions_count': row.positions_count
                })

            return jsonify({
                'success': True,
                'daily_pnl': daily_pnl,
                'total_days': len(daily_pnl),
                'initial_capital': INITIAL_CAPITAL
            }), 200

    except Exception as e:
        logger.error(f"Error getting daily P&L: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@auto_trading_bp.route('/paper-trading/positions', methods=['GET'])
@login_required
def get_paper_positions():
    """
    Get paper trading positions (active and/or closed).

    Query parameters:
        - status: 'active', 'closed', or 'all' (default: 'all')

    Returns:
        List of positions with entry price, current price, P&L, etc.
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import OrderPerformance, Order, User
        from sqlalchemy import and_

        user_id = current_user.id
        status = request.args.get('status', 'all')

        with get_database_manager().get_session() as session:
            # Verify user is in paper trading mode
            user = session.query(User).filter_by(id=user_id).first()
            if not user or not user.is_mock_trading_mode:
                return jsonify({
                    'success': False,
                    'error': 'Paper trading not enabled for this user'
                }), 400

            # Get paper trading order IDs
            mock_orders = session.query(Order).filter(
                and_(
                    Order.user_id == user_id,
                    Order.is_mock_order == True,
                    Order.transaction_type == 'BUY',
                    Order.order_status.in_(['COMPLETE', 'EXECUTED'])
                )
            ).all()

            order_ids = [o.id for o in mock_orders]

            if not order_ids:
                return jsonify({
                    'success': True,
                    'positions': [],
                    'total': 0
                }), 200

            # Query positions
            query = session.query(OrderPerformance).filter(
                OrderPerformance.order_id.in_(order_ids)
            )

            if status == 'active':
                query = query.filter(OrderPerformance.is_active == True)
            elif status == 'closed':
                query = query.filter(OrderPerformance.is_active == False)

            performances = query.order_by(OrderPerformance.created_at.desc()).all()

            # Format positions
            positions = []
            for p in performances:
                position = {
                    'id': p.id,
                    'symbol': p.symbol,
                    'quantity': p.quantity,
                    'entry_price': float(p.entry_price) if p.entry_price else None,
                    'current_price': float(p.current_price) if p.current_price else None,
                    'entry_date': p.created_at.isoformat() if p.created_at else None,
                    'strategy': p.strategy,
                    'is_active': p.is_active,
                    'days_held': p.days_held,

                    # P&L
                    'unrealized_pnl': float(p.unrealized_pnl) if p.unrealized_pnl else 0,
                    'unrealized_pnl_pct': float(p.unrealized_pnl_pct) if p.unrealized_pnl_pct else 0,
                    'realized_pnl': float(p.realized_pnl) if p.realized_pnl else 0,
                    'realized_pnl_pct': float(p.realized_pnl_pct) if p.realized_pnl_pct else 0,

                    # Targets
                    'stop_loss': float(p.stop_loss) if p.stop_loss else None,
                    'target_price': float(p.target_price) if p.target_price else None,

                    # Performance
                    'is_profitable': p.is_profitable,
                    'performance_rating': p.performance_rating,
                    'max_profit_reached': float(p.max_profit_reached) if p.max_profit_reached else None,
                    'max_loss_reached': float(p.max_loss_reached) if p.max_loss_reached else None,

                    # Exit info (for closed positions)
                    'exit_price': float(p.exit_price) if p.exit_price else None,
                    'exit_date': p.exit_date.isoformat() if p.exit_date else None,
                    'exit_reason': p.exit_reason
                }
                positions.append(position)

            return jsonify({
                'success': True,
                'positions': positions,
                'total': len(positions),
                'active_count': len([p for p in positions if p['is_active']]),
                'closed_count': len([p for p in positions if not p['is_active']])
            }), 200

    except Exception as e:
        logger.error(f"Error getting paper positions: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@auto_trading_bp.route('/paper-trading/reset', methods=['POST'])
@login_required
def reset_paper_portfolio():
    """
    Reset paper trading portfolio - closes all positions and resets capital.

    This will:
    - Close all active paper positions
    - Delete all paper trading orders and performance data
    - Reset to initial capital
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import OrderPerformance, OrderPerformanceSnapshot, Order, User
        from sqlalchemy import and_

        user_id = current_user.id

        with get_database_manager().get_session() as session:
            # Verify user is in paper trading mode
            user = session.query(User).filter_by(id=user_id).first()
            if not user or not user.is_mock_trading_mode:
                return jsonify({
                    'success': False,
                    'error': 'Paper trading not enabled for this user'
                }), 400

            # Get paper trading orders
            mock_orders = session.query(Order).filter(
                and_(
                    Order.user_id == user_id,
                    Order.is_mock_order == True
                )
            ).all()

            order_ids = [o.id for o in mock_orders]
            deleted_orders = len(order_ids)

            if order_ids:
                # Get performance IDs
                perf_ids = session.query(OrderPerformance.id).filter(
                    OrderPerformance.order_id.in_(order_ids)
                ).all()
                perf_ids = [p[0] for p in perf_ids]

                # Delete snapshots
                if perf_ids:
                    session.query(OrderPerformanceSnapshot).filter(
                        OrderPerformanceSnapshot.order_performance_id.in_(perf_ids)
                    ).delete(synchronize_session=False)

                # Delete performance records
                session.query(OrderPerformance).filter(
                    OrderPerformance.order_id.in_(order_ids)
                ).delete(synchronize_session=False)

                # Delete orders
                session.query(Order).filter(
                    Order.id.in_(order_ids)
                ).delete(synchronize_session=False)

            session.commit()

            return jsonify({
                'success': True,
                'message': 'Paper trading portfolio reset successfully',
                'deleted_orders': deleted_orders,
                'initial_capital': 100000.0
            }), 200

    except Exception as e:
        logger.error(f"Error resetting paper portfolio: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
