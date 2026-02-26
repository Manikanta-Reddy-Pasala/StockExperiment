"""
Order Performance Tracking Service
Updates order performance metrics daily and creates snapshots.
Supports partial exits at Fibonacci levels and day trading close.
"""

import logging
import math
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_

from src.models.models import OrderPerformance, OrderPerformanceSnapshot
from src.models.database import get_database_manager

logger = logging.getLogger(__name__)


class OrderPerformanceTrackingService:
    """Service for tracking order performance and creating daily snapshots."""

    def update_all_active_orders(self) -> Dict[str, Any]:
        """
        Update performance metrics for all active orders.
        Uses a fresh session per invocation to avoid stale connections.
        """
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            try:
                logger.info("Starting performance update for all active orders")

                active_orders = session.query(OrderPerformance).filter_by(is_active=True).all()

                if not active_orders:
                    logger.info("No active orders to track")
                    return {
                        'success': True,
                        'message': 'No active orders',
                        'orders_updated': 0
                    }

                logger.info(f"Found {len(active_orders)} active orders to update")

                updated_count = 0
                snapshot_count = 0
                closed_count = 0
                partial_exit_count = 0

                for order_perf in active_orders:
                    try:
                        current_price = self._get_current_price(session, order_perf.symbol, order_perf.user_id)
                        if not current_price:
                            logger.warning(f"Could not fetch price for {order_perf.symbol}")
                            continue

                        self._update_order_performance(order_perf, current_price)
                        self._create_daily_snapshot(session, order_perf)
                        snapshot_count += 1

                        exit_result = self._process_exit_logic(order_perf, current_price)
                        if exit_result.get('fully_closed'):
                            closed_count += 1
                        if exit_result.get('partial_exit'):
                            partial_exit_count += 1

                        updated_count += 1

                    except Exception as e:
                        logger.error(f"Error updating order {order_perf.order_id}: {e}")
                        continue

                session.commit()

                logger.info(f"Performance update completed: {updated_count} updated, "
                           f"{snapshot_count} snapshots, {closed_count} closed, "
                           f"{partial_exit_count} partial exits")

                return {
                    'success': True,
                    'orders_updated': updated_count,
                    'snapshots_created': snapshot_count,
                    'orders_closed': closed_count,
                    'partial_exits': partial_exit_count
                }

            except Exception as e:
                logger.error(f"Performance update failed: {e}", exc_info=True)
                return {
                    'success': False,
                    'error': str(e)
                }

    def _get_current_price(self, session: Session, symbol: str, user_id: int = 1) -> Optional[float]:
        """Get current price for a symbol. Tries stocks table first, then broker API."""
        try:
            # Try stocks table first (always available, updated by data pipeline)
            from src.models.stock_models import Stock
            stock = session.query(Stock).filter_by(symbol=symbol).first()
            if stock and stock.current_price and stock.current_price > 0:
                stocks_price = float(stock.current_price)
            else:
                stocks_price = None

            # Try broker API for fresher intraday data (may fail for paper trading)
            broker_price = None
            try:
                from src.services.brokers.fyers_service import get_fyers_service
                fyers_service = get_fyers_service()
                result = fyers_service.quotes(user_id, symbol)
                if result.get('status') == 'success' and result.get('data'):
                    data = result['data']
                    if isinstance(data, dict):
                        ltp = data.get('ltp') or data.get('v', {}).get('lp')
                        if ltp and ltp > 0:
                            broker_price = float(ltp)
                    elif isinstance(data, list) and len(data) > 0:
                        ltp = data[0].get('v', {}).get('lp')
                        if ltp and ltp > 0:
                            broker_price = float(ltp)
            except Exception as e:
                logger.debug(f"Broker quote failed for {symbol}: {e}")

            # Prefer broker price (live) if available, otherwise use stocks table
            return broker_price or stocks_price

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def _update_order_performance(self, order_perf: OrderPerformance, current_price: float) -> None:
        """Update order performance metrics."""
        try:
            order_perf.current_price = current_price
            remaining_qty = order_perf.remaining_quantity or order_perf.quantity
            order_perf.current_value = current_price * remaining_qty

            entry_value = order_perf.entry_price * remaining_qty
            if entry_value <= 0:
                return

            unrealized = order_perf.current_value - entry_value
            partial_realized = order_perf.partial_pnl_realized or 0.0
            order_perf.unrealized_pnl = unrealized + partial_realized
            total_entry_value = order_perf.entry_price * (order_perf.original_quantity or order_perf.quantity)
            order_perf.unrealized_pnl_pct = (order_perf.unrealized_pnl / total_entry_value) * 100 if total_entry_value > 0 else 0

            order_perf.is_profitable = order_perf.unrealized_pnl > 0

            if order_perf.max_profit_reached is None or order_perf.unrealized_pnl > order_perf.max_profit_reached:
                order_perf.max_profit_reached = order_perf.unrealized_pnl

            if order_perf.max_loss_reached is None or order_perf.unrealized_pnl < order_perf.max_loss_reached:
                order_perf.max_loss_reached = order_perf.unrealized_pnl

            order_perf.days_held = (datetime.now() - order_perf.created_at).days

            # Calculate prediction accuracy using target_price
            target = order_perf.target_price
            if target and target > 0:
                target_move_pct = ((target - order_perf.entry_price) / order_perf.entry_price) * 100
                actual_move_pct = order_perf.unrealized_pnl_pct

                if target_move_pct != 0:
                    order_perf.prediction_accuracy = min(100.0, (actual_move_pct / target_move_pct) * 100)

            order_perf.performance_rating = self._calculate_performance_rating(order_perf)
            order_perf.last_checked_at = datetime.now()

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def _process_exit_logic(self, order_perf: OrderPerformance, current_price: float) -> Dict[str, Any]:
        """
        Process exit logic with partial exits at Fibonacci levels.
        Returns dict with 'fully_closed' and 'partial_exit' booleans.
        """
        result = {'fully_closed': False, 'partial_exit': False}

        try:
            remaining_qty = order_perf.remaining_quantity or order_perf.quantity
            original_qty = order_perf.original_quantity or order_perf.quantity

            if remaining_qty <= 0:
                order_perf.is_active = False
                result['fully_closed'] = True
                return result

            # 1. Stop-loss check → full exit of remaining quantity
            if order_perf.stop_loss and current_price <= order_perf.stop_loss:
                logger.info(f"Stop-loss hit for {order_perf.symbol}: "
                           f"{current_price} <= {order_perf.stop_loss}")
                pnl = (current_price - order_perf.entry_price) * remaining_qty
                self._execute_full_exit(order_perf, current_price, 'stop_loss', pnl)
                result['fully_closed'] = True
                return result

            # 2. Swing time limit: 14 days → force close all remaining
            if order_perf.trading_type == 'swing' and order_perf.days_held and order_perf.days_held >= 14:
                logger.info(f"Swing time limit reached for {order_perf.symbol}: {order_perf.days_held} days")
                pnl = (current_price - order_perf.entry_price) * remaining_qty
                self._execute_full_exit(order_perf, current_price, 'time_based', pnl)
                result['fully_closed'] = True
                return result

            # 3. Partial exits at Fibonacci levels (only for swing trades with targets set)
            if order_perf.target_price_1 or order_perf.target_price_2 or order_perf.target_price_3:
                # Fib 127.2% → sell 25% of original_quantity
                if (order_perf.target_price_1
                        and not order_perf.partial_exit_1_done
                        and current_price >= order_perf.target_price_1):
                    sell_qty = max(1, math.floor(original_qty * 0.25))
                    sell_qty = min(sell_qty, remaining_qty)
                    if sell_qty > 0:
                        pnl = (current_price - order_perf.entry_price) * sell_qty
                        order_perf.partial_exit_1_done = True
                        order_perf.remaining_quantity = remaining_qty - sell_qty
                        order_perf.partial_pnl_realized = (order_perf.partial_pnl_realized or 0.0) + pnl
                        remaining_qty = order_perf.remaining_quantity
                        result['partial_exit'] = True
                        logger.info(f"Partial exit 1 (25%) for {order_perf.symbol}: "
                                   f"sold {sell_qty} @ {current_price}, P&L: {pnl:.2f}")

                # Fib 161.8% → sell 50% of original_quantity
                if (order_perf.target_price_2
                        and not order_perf.partial_exit_2_done
                        and current_price >= order_perf.target_price_2):
                    sell_qty = max(1, math.floor(original_qty * 0.50))
                    sell_qty = min(sell_qty, remaining_qty)
                    if sell_qty > 0:
                        pnl = (current_price - order_perf.entry_price) * sell_qty
                        order_perf.partial_exit_2_done = True
                        order_perf.remaining_quantity = remaining_qty - sell_qty
                        order_perf.partial_pnl_realized = (order_perf.partial_pnl_realized or 0.0) + pnl
                        remaining_qty = order_perf.remaining_quantity
                        result['partial_exit'] = True
                        logger.info(f"Partial exit 2 (50%) for {order_perf.symbol}: "
                                   f"sold {sell_qty} @ {current_price}, P&L: {pnl:.2f}")

                # Fib 200% → sell remaining 25%
                if (order_perf.target_price_3
                        and not order_perf.partial_exit_3_done
                        and current_price >= order_perf.target_price_3):
                    sell_qty = remaining_qty  # Sell all remaining
                    if sell_qty > 0:
                        pnl = (current_price - order_perf.entry_price) * sell_qty
                        order_perf.partial_exit_3_done = True
                        order_perf.remaining_quantity = 0
                        order_perf.partial_pnl_realized = (order_perf.partial_pnl_realized or 0.0) + pnl
                        result['partial_exit'] = True
                        logger.info(f"Partial exit 3 (remaining) for {order_perf.symbol}: "
                                   f"sold {sell_qty} @ {current_price}, P&L: {pnl:.2f}")

                # If all quantity sold, fully close
                if order_perf.remaining_quantity is not None and order_perf.remaining_quantity <= 0:
                    total_pnl = order_perf.partial_pnl_realized or 0.0
                    self._execute_full_exit(order_perf, current_price, 'target_reached', total_pnl, skip_pnl_calc=True)
                    result['fully_closed'] = True
                    return result

            else:
                # Legacy single-target exit (no partial exit targets set)
                if order_perf.target_price and current_price >= order_perf.target_price:
                    logger.info(f"Target reached for {order_perf.symbol}: "
                               f"{current_price} >= {order_perf.target_price}")
                    pnl = (current_price - order_perf.entry_price) * remaining_qty
                    self._execute_full_exit(order_perf, current_price, 'target_reached', pnl)
                    result['fully_closed'] = True
                    return result

                # Legacy 30-day time limit
                if order_perf.days_held and order_perf.days_held >= 30:
                    logger.info(f"Max holding period reached for {order_perf.symbol}: "
                               f"{order_perf.days_held} days")
                    pnl = (current_price - order_perf.entry_price) * remaining_qty
                    self._execute_full_exit(order_perf, current_price, 'time_based', pnl)
                    result['fully_closed'] = True
                    return result

        except Exception as e:
            logger.error(f"Error in exit logic for {order_perf.order_id}: {e}")

        return result

    def _execute_full_exit(self, order_perf: OrderPerformance, exit_price: float,
                           exit_reason: str, pnl: float, skip_pnl_calc: bool = False) -> None:
        """Close order and record exit details."""
        try:
            order_perf.is_active = False
            order_perf.exit_price = exit_price
            order_perf.exit_date = datetime.now()
            order_perf.exit_reason = exit_reason
            order_perf.remaining_quantity = 0

            if skip_pnl_calc:
                order_perf.realized_pnl = pnl
            else:
                order_perf.realized_pnl = pnl + (order_perf.partial_pnl_realized or 0.0)

            total_entry = order_perf.entry_price * (order_perf.original_quantity or order_perf.quantity)
            order_perf.realized_pnl_pct = (order_perf.realized_pnl / total_entry) * 100 if total_entry > 0 else 0

            logger.info(f"Order closed: {order_perf.order_id} ({exit_reason}) "
                       f"P&L: {order_perf.realized_pnl:.2f} ({order_perf.realized_pnl_pct:.2f}%)")

        except Exception as e:
            logger.error(f"Error closing order: {e}")

    def close_day_trading_positions(self) -> Dict[str, Any]:
        """
        Close all active day trading positions at current market price.
        Called at 3:20 PM to force-close all day trades before market close.
        """
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            try:
                logger.info("Closing all active day trading positions (end of day)")

                day_orders = session.query(OrderPerformance).filter(
                    and_(
                        OrderPerformance.is_active == True,
                        OrderPerformance.trading_type == 'day'
                    )
                ).all()

                if not day_orders:
                    logger.info("No active day trading positions to close")
                    return {
                        'success': True,
                        'message': 'No day trading positions to close',
                        'positions_closed': 0
                    }

                closed_count = 0
                total_pnl = 0.0

                for order_perf in day_orders:
                    try:
                        current_price = self._get_current_price(session, order_perf.symbol, order_perf.user_id)
                        if not current_price:
                            logger.warning(f"Could not fetch price for {order_perf.symbol}, using last known")
                            current_price = order_perf.current_price or order_perf.entry_price

                        remaining_qty = order_perf.remaining_quantity or order_perf.quantity
                        pnl = (current_price - order_perf.entry_price) * remaining_qty
                        self._execute_full_exit(order_perf, current_price, 'end_of_day', pnl)
                        closed_count += 1
                        total_pnl += order_perf.realized_pnl or 0

                    except Exception as e:
                        logger.error(f"Error closing day trade {order_perf.order_id}: {e}")

                session.commit()

                logger.info(f"Day trading close completed: {closed_count} positions, "
                           f"total P&L: {total_pnl:.2f}")

                return {
                    'success': True,
                    'positions_closed': closed_count,
                    'total_pnl': round(total_pnl, 2)
                }

            except Exception as e:
                logger.error(f"Day trading close failed: {e}", exc_info=True)
                return {
                    'success': False,
                    'error': str(e)
                }

    def _create_daily_snapshot(self, session: Session, order_perf: OrderPerformance) -> None:
        """Create daily snapshot of order performance."""
        try:
            today = datetime.now().date()
            existing_snapshot = session.query(OrderPerformanceSnapshot).filter(
                and_(
                    OrderPerformanceSnapshot.order_performance_id == order_perf.id,
                    OrderPerformanceSnapshot.snapshot_date >= datetime.combine(today, datetime.min.time())
                )
            ).first()

            if existing_snapshot:
                existing_snapshot.price = order_perf.current_price
                existing_snapshot.value = order_perf.current_value
                existing_snapshot.unrealized_pnl = order_perf.unrealized_pnl
                existing_snapshot.unrealized_pnl_pct = order_perf.unrealized_pnl_pct
                existing_snapshot.days_since_entry = order_perf.days_held
                existing_snapshot.price_change_from_entry_pct = order_perf.unrealized_pnl_pct
            else:
                snapshot = OrderPerformanceSnapshot(
                    order_performance_id=order_perf.id,
                    price=order_perf.current_price,
                    value=order_perf.current_value,
                    unrealized_pnl=order_perf.unrealized_pnl,
                    unrealized_pnl_pct=order_perf.unrealized_pnl_pct,
                    days_since_entry=order_perf.days_held,
                    price_change_from_entry_pct=order_perf.unrealized_pnl_pct
                )

                if order_perf.target_price and order_perf.current_price:
                    snapshot.distance_to_target_pct = (
                        (order_perf.target_price - order_perf.current_price) / order_perf.current_price
                    ) * 100

                if order_perf.stop_loss and order_perf.current_price:
                    snapshot.distance_to_stoploss_pct = (
                        (order_perf.stop_loss - order_perf.current_price) / order_perf.current_price
                    ) * 100

                session.add(snapshot)

        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")

    def _calculate_performance_rating(self, order_perf: OrderPerformance) -> str:
        """Calculate performance rating based on P&L."""
        pnl_pct = order_perf.unrealized_pnl_pct or 0.0

        if pnl_pct >= 10.0:
            return 'excellent'
        elif pnl_pct >= 5.0:
            return 'good'
        elif pnl_pct >= 0.0:
            return 'neutral'
        elif pnl_pct >= -3.0:
            return 'poor'
        else:
            return 'loss'

    def get_performance_summary(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for a user."""
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            try:
                start_date = datetime.now() - timedelta(days=days)

                orders = session.query(OrderPerformance).filter(
                    and_(
                        OrderPerformance.user_id == user_id,
                        OrderPerformance.created_at >= start_date
                    )
                ).all()

                if not orders:
                    return {
                        'success': True,
                        'total_orders': 0,
                        'message': 'No orders in this period'
                    }

                total_orders = len(orders)
                active_orders = len([o for o in orders if o.is_active])
                closed_orders = total_orders - active_orders

                profitable_orders = len([o for o in orders if o.is_profitable])
                win_rate = (profitable_orders / total_orders) * 100 if total_orders > 0 else 0

                total_realized_pnl = sum([o.realized_pnl or 0 for o in orders if not o.is_active])
                total_unrealized_pnl = sum([o.unrealized_pnl or 0 for o in orders if o.is_active])
                total_pnl = total_realized_pnl + total_unrealized_pnl

                avg_pnl_pct = sum([o.unrealized_pnl_pct or 0 for o in orders]) / total_orders if total_orders > 0 else 0

                rating_counts = {}
                for order in orders:
                    rating = order.performance_rating or 'unknown'
                    rating_counts[rating] = rating_counts.get(rating, 0) + 1

                return {
                    'success': True,
                    'period_days': days,
                    'total_orders': total_orders,
                    'active_orders': active_orders,
                    'closed_orders': closed_orders,
                    'profitable_orders': profitable_orders,
                    'win_rate': round(win_rate, 2),
                    'total_pnl': round(total_pnl, 2),
                    'realized_pnl': round(total_realized_pnl, 2),
                    'unrealized_pnl': round(total_unrealized_pnl, 2),
                    'avg_pnl_pct': round(avg_pnl_pct, 2),
                    'rating_distribution': rating_counts
                }

            except Exception as e:
                logger.error(f"Error getting performance summary: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }


def get_performance_tracking_service() -> OrderPerformanceTrackingService:
    """Get performance tracking service instance (stateless, no global needed)."""
    return OrderPerformanceTrackingService()
