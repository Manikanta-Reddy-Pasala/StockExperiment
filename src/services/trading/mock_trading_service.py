"""
Mock Trading Service
Handles mock order placement and tracking for model evaluation without real money.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from src.models.models import Order, User, Stock, DailySuggestedStock
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
        quantity: int,
        strategy: str
    ) -> Dict:
        """
        Place a mock order for evaluation purposes.

        Args:
            user_id: User ID
            symbol: Stock symbol (e.g., 'RELIANCE-EQ')
            quantity: Number of shares
            strategy: '8-21 EMA strategy: 'default_risk' or 'high_risk'

        Returns:
            Dict with order details and status
        """
        try:
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
            current_price = stock.close_price
            if not current_price or current_price <= 0:
                return {
                    'success': False,
                    'error': f'Invalid price for {symbol}: {current_price}'
                }

            # Calculate order value
            total_value = current_price * quantity

            # Create mock order
            order = Order(
                user_id=user_id,
                symbol=symbol,
                order_type='BUY',
                quantity=quantity,
                price=current_price,
                status='COMPLETED',  # Mock orders are instantly "completed"
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                is_mock_order=True,
                strategy=strategy
            )

            self.session.add(order)
            self.session.commit()

            logger.info(f"Mock order placed: {symbol} x{quantity} @ ₹{current_price:.2f} "
                       f"(Strategy: {strategy})")

            return {
                'success': True,
                'order_id': order.id,
                'symbol': symbol,
                'quantity': quantity,
                'price': current_price,
                'total_value': total_value,
                'strategy': strategy,
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
        strategy: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Retrieve mock orders for a user.

        Args:
            user_id: User ID
            strategy: Filter by strategy ('default_risk' or 'high_risk')
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

            if strategy:
                query = query.filter(Order.strategy == strategy)

            orders = query.order_by(desc(Order.created_at)).limit(limit).all()

            result = []
            for order in orders:
                # Get current stock price for P&L calculation
                stock = self.session.query(Stock).filter(Stock.symbol == order.symbol).first()
                current_price = stock.close_price if stock else None

                # Calculate P&L
                pnl = None
                pnl_percent = None
                if current_price and order.price:
                    pnl = (current_price - order.price) * order.quantity
                    pnl_percent = ((current_price - order.price) / order.price) * 100

                result.append({
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'quantity': order.quantity,
                    'entry_price': order.price,
                    'current_price': current_price,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'strategy': order.strategy,
                    'created_at': order.created_at.isoformat() if order.created_at else None
                })

            return result

        except Exception as e:
            logger.error(f"Failed to retrieve mock orders: {e}", exc_info=True)
            return []

    def calculate_strategy_performance(self, user_id: int) -> Dict:
        """
        Calculate performance metrics for each strategy.

        Args:
            user_id: User ID

        Returns:
            Dict with performance metrics for each strategy
        """
        try:
            orders = self.get_mock_orders(user_id, limit=1000)

            if not orders:
                return {
                    'success': False,
                    'error': 'No mock orders found'
                }

            # Group by strategy
            performance = {}

            for strategy in ['default_risk', 'high_risk']:
                # Filter orders for this strategy
                strategy_orders = [
                    o for o in orders
                    if o['strategy'] == strategy
                ]

                if not strategy_orders:
                    performance[strategy] = {
                        'strategy': strategy,
                        'total_orders': 0,
                        'total_invested': 0,
                        'total_pnl': 0,
                        'avg_pnl_percent': 0,
                        'winning_orders': 0,
                        'losing_orders': 0,
                        'win_rate': 0
                    }
                    continue

                # Calculate metrics
                total_orders = len(strategy_orders)
                total_invested = sum(o['entry_price'] * o['quantity'] for o in strategy_orders)

                valid_pnl_orders = [o for o in strategy_orders if o['pnl'] is not None]
                total_pnl = sum(o['pnl'] for o in valid_pnl_orders)

                valid_pnl_percent_orders = [o for o in strategy_orders if o['pnl_percent'] is not None]
                avg_pnl_percent = (
                    sum(o['pnl_percent'] for o in valid_pnl_percent_orders) / len(valid_pnl_percent_orders)
                    if valid_pnl_percent_orders else 0
                )

                winning_orders = sum(1 for o in valid_pnl_orders if o['pnl'] > 0)
                losing_orders = sum(1 for o in valid_pnl_orders if o['pnl'] < 0)
                win_rate = (winning_orders / len(valid_pnl_orders) * 100) if valid_pnl_orders else 0

                performance[strategy] = {
                    'strategy': strategy,
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
            logger.error(f"Failed to calculate strategy performance: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def get_suggested_stock_details(self, symbol: str, strategy: str) -> Optional[Dict]:
        """
        Get EMA strategy details for a suggested stock.

        Args:
            symbol: Stock symbol
            strategy: Strategy name

        Returns:
            Dict with EMA strategy details or None
        """
        try:
            suggested = self.session.query(DailySuggestedStock).filter(
                and_(
                    DailySuggestedStock.symbol == symbol,
                    DailySuggestedStock.strategy == strategy
                )
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
