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
                        'execution_time': '09:20',
                        'trading_mode': 'swing',
                        'virtual_capital': 100000.0
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
                    'trading_mode': settings.trading_mode or 'swing',
                    'virtual_capital': settings.virtual_capital or 100000.0,
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
            if 'trading_mode' in data:
                if data['trading_mode'] not in ('swing', 'day', 'both'):
                    return jsonify({
                        'success': False,
                        'error': "trading_mode must be 'swing', 'day', or 'both'"
                    }), 400
                settings.trading_mode = data['trading_mode']
            if 'virtual_capital' in data:
                if data['virtual_capital'] < 10000:
                    return jsonify({
                        'success': False,
                        'error': 'Minimum virtual capital is 10,000'
                    }), 400
                settings.virtual_capital = data['virtual_capital']

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
                    'original_quantity': order.original_quantity,
                    'remaining_quantity': order.remaining_quantity,
                    'stop_loss': order.stop_loss,
                    'target_price': order.target_price,
                    'target_price_1': order.target_price_1,
                    'target_price_2': order.target_price_2,
                    'target_price_3': order.target_price_3,
                    'strategy': order.strategy,
                    'trading_type': order.trading_type or 'swing',
                    'current_price': order.current_price,
                    'unrealized_pnl': order.unrealized_pnl,
                    'unrealized_pnl_pct': order.unrealized_pnl_pct,
                    'realized_pnl': order.realized_pnl,
                    'realized_pnl_pct': order.realized_pnl_pct,
                    'partial_pnl_realized': order.partial_pnl_realized,
                    'partial_exit_1_done': order.partial_exit_1_done,
                    'partial_exit_2_done': order.partial_exit_2_done,
                    'partial_exit_3_done': order.partial_exit_3_done,
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
                    Order.order_status.in_(['COMPLETE', 'COMPLETED', 'EXECUTED'])
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
        from src.models.models import OrderPerformance, Order, User, AutoTradingSettings
        from sqlalchemy import func, and_

        user_id = current_user.id

        with get_database_manager().get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                return jsonify({'success': False, 'error': 'User not found'}), 404

            # Get configurable virtual capital from settings
            at_settings = session.query(AutoTradingSettings).filter_by(user_id=user_id).first()
            INITIAL_CAPITAL = (at_settings.virtual_capital if at_settings and at_settings.virtual_capital else 100000.0)

            # Get all paper trading orders (is_mock_order = True)
            mock_orders = session.query(Order).filter(
                and_(
                    Order.user_id == user_id,
                    Order.is_mock_order == True,
                    Order.transaction_type == 'BUY',
                    Order.order_status.in_(['COMPLETE', 'COMPLETED', 'EXECUTED'])
                )
            ).all()

            order_ids = [o.order_id for o in mock_orders]

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

            # Fetch live prices from stocks table for active positions
            from src.models.stock_models import Stock
            active_symbols = list(set(p.symbol for p in active_positions if p.symbol))
            live_prices = {}
            if active_symbols:
                stocks = session.query(Stock).filter(Stock.symbol.in_(active_symbols)).all()
                live_prices = {s.symbol: float(s.current_price) for s in stocks if s.current_price and s.current_price > 0}

            # Calculate values using live prices
            total_invested = sum((p.entry_price or 0) * (p.quantity or 0) for p in performances)
            positions_value = 0.0
            unrealized_pnl = 0.0
            for p in active_positions:
                live_price = live_prices.get(p.symbol, float(p.current_price or p.entry_price or 0))
                remaining_qty = p.remaining_quantity or p.quantity or 0
                positions_value += live_price * remaining_qty
                entry_value = (p.entry_price or 0) * remaining_qty
                unrealized_pnl += (live_price * remaining_qty - entry_value) + (p.partial_pnl_realized or 0)

            realized_pnl = sum(p.realized_pnl or 0 for p in closed_positions)
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
        from src.models.models import OrderPerformance, OrderPerformanceSnapshot, Order, User, AutoTradingSettings
        from sqlalchemy import func, and_
        from datetime import date

        user_id = current_user.id
        days = int(request.args.get('days', 30))

        start_date = date.today() - timedelta(days=days)

        with get_database_manager().get_session() as session:
            # Get configurable virtual capital
            at_settings = session.query(AutoTradingSettings).filter_by(user_id=user_id).first()
            INITIAL_CAPITAL = (at_settings.virtual_capital if at_settings and at_settings.virtual_capital else 100000.0)

            # Get paper trading order IDs
            mock_orders = session.query(Order).filter(
                and_(
                    Order.user_id == user_id,
                    Order.is_mock_order == True,
                    Order.transaction_type == 'BUY',
                    Order.order_status.in_(['COMPLETE', 'COMPLETED', 'EXECUTED'])
                )
            ).all()

            order_ids = [o.order_id for o in mock_orders]

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
            # Get paper trading order IDs
            mock_orders = session.query(Order).filter(
                and_(
                    Order.user_id == user_id,
                    Order.is_mock_order == True,
                    Order.transaction_type == 'BUY',
                    Order.order_status.in_(['COMPLETE', 'COMPLETED', 'EXECUTED'])
                )
            ).all()

            order_ids = [o.order_id for o in mock_orders]

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

            # Fetch live prices from stocks table for active positions
            from src.models.stock_models import Stock
            active_symbols = list(set(p.symbol for p in performances if p.symbol and p.is_active))
            live_prices = {}
            if active_symbols:
                stocks = session.query(Stock).filter(Stock.symbol.in_(active_symbols)).all()
                live_prices = {s.symbol: float(s.current_price) for s in stocks if s.current_price and s.current_price > 0}

            # Format positions
            positions = []
            for p in performances:
                # Use live price for active positions, stored price for closed
                if p.is_active and p.symbol in live_prices:
                    current_price = live_prices[p.symbol]
                    remaining_qty = p.remaining_quantity or p.quantity or 0
                    entry_price = float(p.entry_price) if p.entry_price else 0
                    entry_value = entry_price * (p.original_quantity or p.quantity or 0)
                    unrealized = (current_price - entry_price) * remaining_qty + (p.partial_pnl_realized or 0)
                    unrealized_pct = (unrealized / entry_value * 100) if entry_value > 0 else 0
                else:
                    current_price = float(p.current_price) if p.current_price else None
                    unrealized = float(p.unrealized_pnl) if p.unrealized_pnl else 0
                    unrealized_pct = float(p.unrealized_pnl_pct) if p.unrealized_pnl_pct else 0

                position = {
                    'id': p.id,
                    'symbol': p.symbol,
                    'quantity': p.quantity,
                    'original_quantity': p.original_quantity,
                    'remaining_quantity': p.remaining_quantity,
                    'entry_price': float(p.entry_price) if p.entry_price else None,
                    'current_price': current_price,
                    'entry_date': p.created_at.isoformat() if p.created_at else None,
                    'strategy': p.strategy,
                    'trading_type': p.trading_type or 'swing',
                    'is_active': p.is_active,
                    'days_held': p.days_held,

                    # P&L
                    'unrealized_pnl': round(unrealized, 2),
                    'unrealized_pnl_pct': round(unrealized_pct, 2),
                    'realized_pnl': float(p.realized_pnl) if p.realized_pnl else 0,
                    'realized_pnl_pct': float(p.realized_pnl_pct) if p.realized_pnl_pct else 0,
                    'partial_pnl_realized': float(p.partial_pnl_realized) if p.partial_pnl_realized else 0,

                    # Targets
                    'stop_loss': float(p.stop_loss) if p.stop_loss else None,
                    'target_price': float(p.target_price) if p.target_price else None,
                    'target_price_1': float(p.target_price_1) if p.target_price_1 else None,
                    'target_price_2': float(p.target_price_2) if p.target_price_2 else None,
                    'target_price_3': float(p.target_price_3) if p.target_price_3 else None,

                    # Partial exit status
                    'partial_exit_1_done': p.partial_exit_1_done,
                    'partial_exit_2_done': p.partial_exit_2_done,
                    'partial_exit_3_done': p.partial_exit_3_done,

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


@auto_trading_bp.route('/paper-trading/positions/<int:position_id>/close', methods=['POST'])
@login_required
def close_paper_position(position_id):
    """
    Close a paper trading position manually.

    Sets exit_price to current market price, marks position inactive,
    and calculates realized P&L.
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import OrderPerformance, Order, User
        from src.models.stock_models import Stock
        from sqlalchemy import and_

        user_id = current_user.id

        with get_database_manager().get_session() as session:
            # Get the position
            perf = session.query(OrderPerformance).filter_by(id=position_id).first()
            if not perf:
                return jsonify({
                    'success': False,
                    'error': 'Position not found'
                }), 404

            if not perf.is_active:
                return jsonify({
                    'success': False,
                    'error': 'Position already closed'
                }), 400

            # Verify ownership via order
            order = session.query(Order).filter_by(order_id=perf.order_id).first()
            if not order or order.user_id != user_id or not order.is_mock_order:
                return jsonify({
                    'success': False,
                    'error': 'Position not found'
                }), 404

            # Get current price from stocks table
            symbol_clean = perf.symbol.replace('NSE:', '').replace('-EQ', '')
            stock = session.query(Stock).filter(
                Stock.symbol.ilike(f'%{symbol_clean}%')
            ).first()

            exit_price = float(stock.current_price) if stock and stock.current_price else float(perf.current_price or perf.entry_price)

            # Calculate realized P&L
            qty = perf.remaining_quantity or perf.quantity or 0
            realized_pnl = (exit_price - float(perf.entry_price)) * qty + float(perf.partial_pnl_realized or 0)

            # Update position
            perf.exit_price = exit_price
            perf.exit_date = datetime.now()
            perf.exit_reason = 'manual'
            perf.is_active = False
            perf.remaining_quantity = 0
            perf.realized_pnl = realized_pnl
            perf.realized_pnl_pct = (realized_pnl / (float(perf.entry_price) * (perf.quantity or 1))) * 100 if perf.entry_price and perf.quantity else 0
            perf.is_profitable = realized_pnl > 0
            perf.unrealized_pnl = 0
            perf.unrealized_pnl_pct = 0

            session.commit()

            logger.info(f"Paper position {position_id} closed for user {user_id}: P&L={realized_pnl:.2f}")

            return jsonify({
                'success': True,
                'realized_pnl': round(realized_pnl, 2),
                'exit_price': round(exit_price, 2),
                'message': f'Position closed at {exit_price:.2f}'
            }), 200

    except Exception as e:
        logger.error(f"Error closing paper position: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@auto_trading_bp.route('/paper-trading/alerts', methods=['GET'])
@login_required
def get_paper_alerts():
    """
    Get alerts for paper trading positions nearing SL/target or with sell signals.
    """
    try:
        from src.models.database import get_database_manager
        from src.models.models import OrderPerformance, Order, User
        from src.models.stock_models import Stock
        from sqlalchemy import and_

        user_id = current_user.id

        with get_database_manager().get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user or not user.is_mock_trading_mode:
                return jsonify({'success': True, 'alerts': []}), 200

            # Get active paper positions
            mock_orders = session.query(Order).filter(
                and_(
                    Order.user_id == user_id,
                    Order.is_mock_order == True,
                    Order.transaction_type == 'BUY',
                    Order.order_status.in_(['COMPLETE', 'COMPLETED', 'EXECUTED'])
                )
            ).all()

            order_ids = [o.order_id for o in mock_orders]
            if not order_ids:
                return jsonify({'success': True, 'alerts': []}), 200

            active = session.query(OrderPerformance).filter(
                and_(
                    OrderPerformance.order_id.in_(order_ids),
                    OrderPerformance.is_active == True
                )
            ).all()

            # Fetch live prices from stocks table
            from src.models.stock_models import Stock as AlertStock
            alert_symbols = list(set(p.symbol for p in active if p.symbol))
            alert_live_prices = {}
            if alert_symbols:
                alert_stocks = session.query(AlertStock).filter(AlertStock.symbol.in_(alert_symbols)).all()
                alert_live_prices = {s.symbol: float(s.current_price) for s in alert_stocks if s.current_price and s.current_price > 0}

            alerts = []
            for p in active:
                current = alert_live_prices.get(p.symbol, float(p.current_price or p.entry_price or 0))
                entry = float(p.entry_price or 0)
                if not current or not entry:
                    continue

                sl = float(p.stop_loss) if p.stop_loss else None
                t1 = float(p.target_price_1) if p.target_price_1 else None
                t2 = float(p.target_price_2) if p.target_price_2 else None
                t3 = float(p.target_price_3) if p.target_price_3 else None

                # Check stop loss proximity (< 3%)
                if sl and sl > 0:
                    dist_sl = (current - sl) / current * 100
                    if dist_sl < 3:
                        alerts.append({
                            'type': 'stop_loss',
                            'severity': 'danger' if dist_sl < 1 else 'warning',
                            'symbol': p.symbol,
                            'position_id': p.id,
                            'current_price': current,
                            'threshold': sl,
                            'message': f'{p.symbol} is {dist_sl:.1f}% from stop loss ({Trading_formatINR(sl)})'
                        })

                # Check next target proximity (< 2%)
                next_target = None
                if t1 and not p.partial_exit_1_done and current < t1:
                    next_target = t1
                elif t2 and not p.partial_exit_2_done and current < t2:
                    next_target = t2
                elif t3 and not p.partial_exit_3_done and current < t3:
                    next_target = t3

                if next_target:
                    dist_t = (next_target - current) / current * 100
                    if dist_t < 2:
                        alerts.append({
                            'type': 'target',
                            'severity': 'success',
                            'symbol': p.symbol,
                            'position_id': p.id,
                            'current_price': current,
                            'threshold': next_target,
                            'message': f'{p.symbol} is {dist_t:.1f}% from target {Trading_formatINR(next_target)}'
                        })

                # Check sell signal from stocks table
                symbol_clean = (p.symbol or '').replace('NSE:', '').replace('-EQ', '')
                stock = session.query(Stock).filter(
                    Stock.symbol.ilike(f'%{symbol_clean}%')
                ).first()
                if stock and stock.sell_signal:
                    alerts.append({
                        'type': 'sell_signal',
                        'severity': 'warning',
                        'symbol': p.symbol,
                        'position_id': p.id,
                        'current_price': current,
                        'threshold': None,
                        'message': f'{p.symbol} has an active SELL signal'
                    })

            # Sort by severity (danger first)
            severity_order = {'danger': 0, 'warning': 1, 'success': 2}
            alerts.sort(key=lambda a: severity_order.get(a['severity'], 3))

            return jsonify({
                'success': True,
                'alerts': alerts,
                'total': len(alerts)
            }), 200

    except Exception as e:
        logger.error(f"Error getting paper alerts: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


def Trading_formatINR(amount):
    """Helper to format INR amounts for alert messages."""
    if amount is None:
        return '--'
    return f'\u20B9{amount:,.2f}'


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
        from src.models.models import OrderPerformance, OrderPerformanceSnapshot, Order, User, AutoTradingSettings
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

            # Get configurable virtual capital
            at_settings = session.query(AutoTradingSettings).filter_by(user_id=user_id).first()
            initial_capital = (at_settings.virtual_capital if at_settings and at_settings.virtual_capital else 100000.0)

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
                'initial_capital': initial_capital
            }), 200

    except Exception as e:
        logger.error(f"Error resetting paper portfolio: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
