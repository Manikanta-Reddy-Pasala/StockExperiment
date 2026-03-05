"""
Mock Trading Service
Handles mock order placement and tracking for model evaluation without real money.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from src.models.models import Order, User, OrderPerformance
from src.models.stock_models import Stock, DailySuggestedStock
from src.models.database import get_database_manager

logger = logging.getLogger(__name__)


class MockTradingService:
    """Service for managing mock trading orders."""

    def __init__(self, session: Session):
        self.session = session

    def place_mock_order(
        self,
        user_id: int,
        symbol: str,
        quantity: int
    ) -> Dict:
        """
        Place a mock order for evaluation purposes.

        Args:
            user_id: User ID
            symbol: Stock symbol (e.g., 'RELIANCE-EQ')
            quantity: Number of shares

        Returns:
            Dict with order details and status
        """
        try:
            if not quantity or quantity < 1:
                return {
                    'success': False,
                    'error': 'Quantity must be at least 1'
                }

            # Get user to check trading mode
            user = self.session.query(User).filter(User.id == user_id).first()
            if not user:
                return {
                    'success': False,
                    'error': 'User not found'
                }

            # Note: In development mode, all orders are mock orders by default
            # The is_mock_trading_mode field can be used for additional control if needed

            # Get stock details
            stock = self.session.query(Stock).filter(Stock.symbol == symbol).first()
            if not stock:
                return {
                    'success': False,
                    'error': f'Stock {symbol} not found'
                }

            # Get current price (use close price from latest data)
            current_price = stock.current_price
            if not current_price or current_price <= 0:
                return {
                    'success': False,
                    'error': f'Invalid price for {symbol}: {current_price}'
                }

            # Calculate order value
            total_value = current_price * quantity

            # Check virtual capital
            from src.models.models import AutoTradingSettings, OrderPerformance
            settings = self.session.query(AutoTradingSettings).filter_by(user_id=user_id).first()
            virtual_capital = float(settings.virtual_capital) if settings and settings.virtual_capital else 100000.0

            # Calculate invested amount in active positions
            active_positions = self.session.query(OrderPerformance).filter(
                OrderPerformance.user_id == user_id,
                OrderPerformance.is_active == True
            ).all()
            invested = sum(
                float(p.entry_price or 0) * (p.remaining_quantity or p.quantity or 0)
                for p in active_positions
            )
            # Add realized P&L from closed trades
            from sqlalchemy import func
            realized_pnl = self.session.query(
                func.coalesce(func.sum(OrderPerformance.realized_pnl), 0)
            ).filter(
                OrderPerformance.user_id == user_id,
                OrderPerformance.is_active == False
            ).scalar() or 0

            available_capital = virtual_capital - invested + float(realized_pnl)

            if total_value > available_capital:
                return {
                    'success': False,
                    'error': f'Insufficient capital. Available: \u20B9{available_capital:,.2f}, Required: \u20B9{total_value:,.2f}'
                }

            # Get suggested stock details for targets/stop loss
            suggested = self.session.query(DailySuggestedStock).filter(
                DailySuggestedStock.symbol == symbol
            ).order_by(desc(DailySuggestedStock.date)).first()

            # Create mock order
            now = datetime.now()
            mock_order_id = f"MOCK-{uuid.uuid4().hex[:12].upper()}"
            order = Order(
                user_id=user_id,
                order_id=mock_order_id,
                tradingsymbol=symbol,
                transaction_type='BUY',
                order_type='MARKET',
                quantity=quantity,
                price=current_price,
                average_price=current_price,
                filled_quantity=quantity,
                pending_quantity=0,
                order_status='COMPLETE',
                created_at=now,
                is_mock_order=True,
                strategy=suggested.strategy if suggested else 'ema_8_21'
            )

            self.session.add(order)
            self.session.flush()  # Get order.id before creating performance record

            # Compute stop_loss from suggested or default 5% below entry
            stop_loss = suggested.stop_loss if suggested else round(current_price * 0.95, 2)
            target_price = suggested.target_price if suggested else round(current_price * 1.10, 2)

            # Create OrderPerformance record for portfolio tracking
            perf = OrderPerformance(
                order_id=mock_order_id,
                user_id=user_id,
                symbol=symbol,
                entry_price=current_price,
                quantity=quantity,
                original_quantity=quantity,
                remaining_quantity=quantity,
                current_price=current_price,
                current_value=total_value,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                stop_loss=stop_loss,
                target_price=target_price,
                target_price_1=suggested.fib_target_1 if suggested else None,
                target_price_2=suggested.fib_target_2 if suggested else None,
                target_price_3=suggested.fib_target_3 if suggested else None,
                strategy=suggested.strategy if suggested else 'ema_8_21',
                trading_type='swing',
                is_active=True,
                days_held=0,
                created_at=now
            )

            self.session.add(perf)
            self.session.commit()

            logger.info(f"Mock order placed: {symbol} x{quantity} @ ₹{current_price:.2f}")

            return {
                'success': True,
                'order_id': order.id,
                'symbol': symbol,
                'quantity': quantity,
                'price': current_price,
                'total_value': total_value,
                'created_at': order.created_at.isoformat()
            }

        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to place mock order: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def get_mock_orders(
        self,
        user_id: int,
        limit: int = 50
    ) -> List[Dict]:
        """
        Retrieve mock orders for a user.

        Args:
            user_id: User ID
            limit: Maximum number of orders to return

        Returns:
            List of order dictionaries
        """
        try:
            query = self.session.query(Order).filter(
                and_(
                    Order.user_id == user_id,
                    Order.is_mock_order == True
                )
            )

            orders = query.order_by(desc(Order.created_at)).limit(limit).all()

            result = []
            for order in orders:
                # Get current stock price for P&L calculation
                stock = self.session.query(Stock).filter(Stock.symbol == order.tradingsymbol).first()
                current_price = stock.current_price if stock else None

                # Calculate P&L
                entry_price = order.average_price or order.price
                pnl = None
                pnl_percent = None
                if current_price and entry_price:
                    pnl = (current_price - entry_price) * order.quantity
                    pnl_percent = ((current_price - entry_price) / entry_price) * 100

                result.append({
                    'order_id': order.id,
                    'symbol': order.tradingsymbol,
                    'quantity': order.quantity,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'created_at': order.created_at.isoformat() if order.created_at else None
                })

            return result

        except Exception as e:
            logger.error(f"Failed to retrieve mock orders: {e}", exc_info=True)
            return []

    def calculate_performance(self, user_id: int) -> Dict:
        """
        Calculate overall performance metrics.

        Args:
            user_id: User ID

        Returns:
            Dict with performance metrics
        """
        try:
            orders = self.get_mock_orders(user_id, limit=1000)

            if not orders:
                return {
                    'success': False,
                    'error': 'No mock orders found'
                }

            # Calculate overall metrics
            total_orders = len(orders)
            total_invested = sum(o['entry_price'] * o['quantity'] for o in orders)

            valid_pnl_orders = [o for o in orders if o['pnl'] is not None]
            total_pnl = sum(o['pnl'] for o in valid_pnl_orders)

            valid_pnl_percent_orders = [o for o in orders if o['pnl_percent'] is not None]
            avg_pnl_percent = (
                sum(o['pnl_percent'] for o in valid_pnl_percent_orders) / len(valid_pnl_percent_orders)
                if valid_pnl_percent_orders else 0
            )

            winning_orders = sum(1 for o in valid_pnl_orders if o['pnl'] > 0)
            losing_orders = sum(1 for o in valid_pnl_orders if o['pnl'] < 0)
            win_rate = (winning_orders / len(valid_pnl_orders) * 100) if valid_pnl_orders else 0

            performance = {
                'total_orders': total_orders,
                'total_invested': round(total_invested, 2),
                'total_pnl': round(total_pnl, 2),
                'avg_pnl_percent': round(avg_pnl_percent, 2),
                'winning_orders': winning_orders,
                'losing_orders': losing_orders,
                'win_rate': round(win_rate, 2)
            }

            return {
                'success': True,
                'performance': performance
            }

        except Exception as e:
            logger.error(f"Failed to calculate performance: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def get_suggested_stock_details(self, symbol: str) -> Optional[Dict]:
        """
        Get EMA strategy details for a suggested stock.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with EMA strategy details or None
        """
        try:
            suggested = self.session.query(DailySuggestedStock).filter(
                DailySuggestedStock.symbol == symbol
            ).order_by(desc(DailySuggestedStock.date)).first()

            if not suggested:
                return None

            return {
                'symbol': suggested.symbol,
                'strategy': suggested.strategy,
                'rank': suggested.rank,
                'selection_score': suggested.selection_score,
                'target_price': suggested.target_price,
                'stop_loss': suggested.stop_loss,
                'date': suggested.date.isoformat() if suggested.date else None
            }

        except Exception as e:
            logger.error(f"Failed to get suggested stock details: {e}", exc_info=True)
            return None


def get_mock_trading_service(session: Session = None) -> MockTradingService:
    """Get a MockTradingService instance."""
    if session:
        return MockTradingService(session)
    else:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            return MockTradingService(session)
