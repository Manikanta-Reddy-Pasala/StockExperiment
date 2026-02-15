"""
Order Performance Tracking Service
Updates order performance metrics daily and creates snapshots.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_

from src.models.models import OrderPerformance, OrderPerformanceSnapshot
from src.models.database import get_database_manager

logger = logging.getLogger(__name__)


class OrderPerformanceTrackingService:
    """Service for tracking order performance and creating daily snapshots."""

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize performance tracking service.

        Args:
            session: Database session (optional, will create if not provided)
        """
        self.session = session
        self._own_session = False
        if not session:
            self.db_manager = get_database_manager()
            self.session = self.db_manager.get_session().__enter__()
            self._own_session = True

    def __del__(self):
        """Cleanup session if we created it."""
        if self._own_session and self.session:
            try:
                self.session.__exit__(None, None, None)
            except:
                pass

    def update_all_active_orders(self) -> Dict[str, Any]:
        """
        Update performance metrics for all active orders.

        Returns:
            Dict with update results
        """
        try:
            logger.info("📊 Starting performance update for all active orders")

            # Get all active order performances
            active_orders = self.session.query(OrderPerformance).filter_by(is_active=True).all()

            if not active_orders:
                logger.info("ℹ️  No active orders to track")
                return {
                    'success': True,
                    'message': 'No active orders',
                    'orders_updated': 0
                }

            logger.info(f"Found {len(active_orders)} active orders to update")

            updated_count = 0
            snapshot_count = 0
            closed_count = 0

            for order_perf in active_orders:
                try:
                    # Get current price
                    current_price = self._get_current_price(order_perf.symbol)
                    if not current_price:
                        logger.warning(f"⚠️  Could not fetch price for {order_perf.symbol}")
                        continue

                    # Update performance metrics
                    self._update_order_performance(order_perf, current_price)

                    # Create daily snapshot
                    self._create_daily_snapshot(order_perf)
                    snapshot_count += 1

                    # Check if order should be closed
                    if self._should_close_order(order_perf):
                        self._close_order(order_perf)
                        closed_count += 1

                    updated_count += 1

                except Exception as e:
                    logger.error(f"❌ Error updating order {order_perf.order_id}: {e}")
                    continue

            self.session.commit()

            logger.info(f"✅ Performance update completed: {updated_count} updated, "
                       f"{snapshot_count} snapshots, {closed_count} closed")

            return {
                'success': True,
                'orders_updated': updated_count,
                'snapshots_created': snapshot_count,
                'orders_closed': closed_count
            }

        except Exception as e:
            logger.error(f"❌ Performance update failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol from stocks table or broker API."""
        try:
            # Primary: Use stocks table (always has latest data from pipeline)
            from src.models.stock_models import Stock
            stock = self.session.query(Stock).filter_by(symbol=symbol).first()
            if stock and stock.current_price and stock.current_price > 0:
                return float(stock.current_price)

            # Fallback: Try broker API
            try:
                from src.services.brokers.fyers_service import get_fyers_service
                fyers_service = get_fyers_service()
                result = fyers_service.quotes(1, symbol)
                if result.get('status') == 'success' and result.get('data'):
                    data = result['data']
                    if isinstance(data, dict):
                        return data.get('ltp') or data.get('v', {}).get('lp')
                    elif isinstance(data, list) and len(data) > 0:
                        return data[0].get('v', {}).get('lp')
            except Exception as e:
                logger.debug(f"Broker quote fallback failed for {symbol}: {e}")

            return None

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def _update_order_performance(self, order_perf: OrderPerformance, current_price: float) -> None:
        """Update order performance metrics."""
        try:
            # Update current price and value
            order_perf.current_price = current_price
            order_perf.current_value = current_price * order_perf.quantity

            # Calculate unrealized P&L
            entry_value = order_perf.entry_price * order_perf.quantity
            order_perf.unrealized_pnl = order_perf.current_value - entry_value
            order_perf.unrealized_pnl_pct = (order_perf.unrealized_pnl / entry_value) * 100

            # Update profitability status
            order_perf.is_profitable = order_perf.unrealized_pnl > 0

            # Track max profit/loss
            if order_perf.max_profit_reached is None or order_perf.unrealized_pnl > order_perf.max_profit_reached:
                order_perf.max_profit_reached = order_perf.unrealized_pnl

            if order_perf.max_loss_reached is None or order_perf.unrealized_pnl < order_perf.max_loss_reached:
                order_perf.max_loss_reached = order_perf.unrealized_pnl

            # Calculate days held
            order_perf.days_held = (datetime.now() - order_perf.created_at).days

            # Calculate prediction accuracy
            if order_perf.ml_price_target:
                target_move_pct = ((order_perf.ml_price_target - order_perf.entry_price) / order_perf.entry_price) * 100
                actual_move_pct = order_perf.unrealized_pnl_pct

                if target_move_pct != 0:
                    order_perf.prediction_accuracy = min(100.0, (actual_move_pct / target_move_pct) * 100)

            # Update performance rating
            order_perf.performance_rating = self._calculate_performance_rating(order_perf)

            # Update timestamp
            order_perf.last_checked_at = datetime.now()

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def _create_daily_snapshot(self, order_perf: OrderPerformance) -> None:
        """Create daily snapshot of order performance."""
        try:
            # Check if snapshot already exists for today
            today = datetime.now().date()
            existing_snapshot = self.session.query(OrderPerformanceSnapshot).filter(
                and_(
                    OrderPerformanceSnapshot.order_performance_id == order_perf.id,
                    OrderPerformanceSnapshot.snapshot_date >= datetime.combine(today, datetime.min.time())
                )
            ).first()

            if existing_snapshot:
                # Update existing snapshot
                existing_snapshot.price = order_perf.current_price
                existing_snapshot.value = order_perf.current_value
                existing_snapshot.unrealized_pnl = order_perf.unrealized_pnl
                existing_snapshot.unrealized_pnl_pct = order_perf.unrealized_pnl_pct
                existing_snapshot.days_since_entry = order_perf.days_held
                existing_snapshot.price_change_from_entry_pct = order_perf.unrealized_pnl_pct
            else:
                # Create new snapshot
                snapshot = OrderPerformanceSnapshot(
                    order_performance_id=order_perf.id,
                    price=order_perf.current_price,
                    value=order_perf.current_value,
                    unrealized_pnl=order_perf.unrealized_pnl,
                    unrealized_pnl_pct=order_perf.unrealized_pnl_pct,
                    days_since_entry=order_perf.days_held,
                    price_change_from_entry_pct=order_perf.unrealized_pnl_pct
                )

                # Calculate distance to target and stop-loss
                if order_perf.target_price:
                    snapshot.distance_to_target_pct = (
                        (order_perf.target_price - order_perf.current_price) / order_perf.current_price
                    ) * 100

                if order_perf.stop_loss:
                    snapshot.distance_to_stoploss_pct = (
                        (order_perf.stop_loss - order_perf.current_price) / order_perf.current_price
                    ) * 100

                self.session.add(snapshot)

        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")

    def _should_close_order(self, order_perf: OrderPerformance) -> bool:
        """Check if order should be closed (stop-loss or target reached)."""
        try:
            # Check stop-loss
            if order_perf.stop_loss and order_perf.current_price <= order_perf.stop_loss:
                logger.info(f"🛑 Stop-loss hit for {order_perf.symbol}: "
                           f"{order_perf.current_price} <= {order_perf.stop_loss}")
                return True

            # Check target price
            if order_perf.target_price and order_perf.current_price >= order_perf.target_price:
                logger.info(f"🎯 Target reached for {order_perf.symbol}: "
                           f"{order_perf.current_price} >= {order_perf.target_price}")
                return True

            # Check max holding period (e.g., 30 days)
            if order_perf.days_held >= 30:
                logger.info(f"⏰ Max holding period reached for {order_perf.symbol}: "
                           f"{order_perf.days_held} days")
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking close condition: {e}")
            return False

    def _close_order(self, order_perf: OrderPerformance) -> None:
        """Close order and record exit details."""
        try:
            order_perf.is_active = False
            order_perf.exit_price = order_perf.current_price
            order_perf.exit_date = datetime.now()

            # Determine exit reason
            if order_perf.stop_loss and order_perf.current_price <= order_perf.stop_loss:
                order_perf.exit_reason = 'stop_loss'
            elif order_perf.target_price and order_perf.current_price >= order_perf.target_price:
                order_perf.exit_reason = 'target_reached'
            elif order_perf.days_held >= 30:
                order_perf.exit_reason = 'time_based'
            else:
                order_perf.exit_reason = 'manual'

            # Record realized P&L
            order_perf.realized_pnl = order_perf.unrealized_pnl
            order_perf.realized_pnl_pct = order_perf.unrealized_pnl_pct

            logger.info(f"✅ Order closed: {order_perf.order_id} ({order_perf.exit_reason}) "
                       f"P&L: ₹{order_perf.realized_pnl:.2f} ({order_perf.realized_pnl_pct:.2f}%)")

        except Exception as e:
            logger.error(f"Error closing order: {e}")

    def _calculate_performance_rating(self, order_perf: OrderPerformance) -> str:
        """Calculate performance rating based on P&L."""
        try:
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

        except Exception as e:
            logger.error(f"Error calculating rating: {e}")
            return 'unknown'

    def get_performance_summary(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Get performance summary for a user.

        Args:
            user_id: User ID
            days: Number of days to look back

        Returns:
            Dict with performance summary
        """
        try:
            start_date = datetime.now() - timedelta(days=days)

            # Get all orders for user in timeframe
            orders = self.session.query(OrderPerformance).filter(
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

            # Calculate metrics
            total_orders = len(orders)
            active_orders = len([o for o in orders if o.is_active])
            closed_orders = total_orders - active_orders

            profitable_orders = len([o for o in orders if o.is_profitable])
            win_rate = (profitable_orders / total_orders) * 100 if total_orders > 0 else 0

            total_realized_pnl = sum([o.realized_pnl or 0 for o in orders if not o.is_active])
            total_unrealized_pnl = sum([o.unrealized_pnl or 0 for o in orders if o.is_active])
            total_pnl = total_realized_pnl + total_unrealized_pnl

            avg_pnl_pct = sum([o.unrealized_pnl_pct or 0 for o in orders]) / total_orders if total_orders > 0 else 0

            # Rating distribution
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


# Global service instance
_performance_tracking_service = None


def get_performance_tracking_service() -> OrderPerformanceTrackingService:
    """Get global performance tracking service instance."""
    global _performance_tracking_service
    if _performance_tracking_service is None:
        _performance_tracking_service = OrderPerformanceTrackingService()
    return _performance_tracking_service
