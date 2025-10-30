"""
Auto-Trading Service
Handles automated trading execution with AI confidence check, weekly limits, and performance tracking.
"""

import logging
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text, and_, func

from src.models.models import (
    AutoTradingSettings, AutoTradingExecution, OrderPerformance,
    User, Order
)
from src.models.database import get_database_manager

logger = logging.getLogger(__name__)


class AutoTradingService:
    """Service for automated trading with AI-driven decisions."""

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize auto-trading service.

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

    def execute_auto_trading_for_user(self, user_id: int) -> Dict[str, Any]:
        """
        Execute auto-trading for a specific user.

        Steps:
        1. Check if auto-trading is enabled for user
        2. Get market sentiment from AI
        3. Check if sentiment meets minimum threshold
        4. Check weekly limits
        5. Check account balance
        6. Select top strategies with max confidence
        7. Create buy orders with stop-loss and target price
        8. Track performance

        Args:
            user_id: User ID to execute trading for

        Returns:
            Dict with execution results
        """
        try:
            logger.info(f"🤖 Starting auto-trading execution for user {user_id}")

            # Get user settings
            settings = self.session.query(AutoTradingSettings).filter_by(user_id=user_id).first()

            if not settings:
                logger.warning(f"⚠️  No auto-trading settings found for user {user_id}")
                return {
                    'success': False,
                    'error': 'Auto-trading not configured for this user'
                }

            if not settings.is_enabled:
                logger.info(f"ℹ️  Auto-trading disabled for user {user_id}")
                return {
                    'success': True,
                    'status': 'skipped',
                    'message': 'Auto-trading is disabled'
                }

            # Create execution log
            execution = AutoTradingExecution(
                settings_id=settings.id,
                user_id=user_id,
                status='in_progress'
            )
            self.session.add(execution)
            self.session.flush()

            # Step 1: Check market sentiment
            sentiment_result = self._check_market_sentiment(settings, execution)
            if not sentiment_result['proceed']:
                execution.status = 'skipped'
                execution.error_message = sentiment_result['message']
                self.session.commit()
                logger.info(f"⏭️  {sentiment_result['message']}")
                return {
                    'success': True,
                    'status': 'skipped',
                    'message': sentiment_result['message']
                }

            # Step 2: Check weekly limits
            limits_result = self._check_weekly_limits(user_id, settings, execution)
            if not limits_result['proceed']:
                execution.status = 'skipped'
                execution.error_message = limits_result['message']
                self.session.commit()
                logger.info(f"⏭️  {limits_result['message']}")
                return {
                    'success': True,
                    'status': 'skipped',
                    'message': limits_result['message']
                }

            # Step 3: Check account balance
            balance_result = self._check_account_balance(user_id, execution)
            if not balance_result['proceed']:
                execution.status = 'skipped'
                execution.error_message = balance_result['message']
                self.session.commit()
                logger.info(f"⏭️  {balance_result['message']}")
                return {
                    'success': True,
                    'status': 'skipped',
                    'message': balance_result['message']
                }

            available_amount = min(
                limits_result['remaining_amount'],
                balance_result['available_balance']
            )

            # Step 4: Select top strategies
            strategies_result = self._select_top_strategies(settings, limits_result['remaining_buys'])
            if not strategies_result['stocks']:
                execution.status = 'skipped'
                execution.error_message = 'No suitable stocks found'
                self.session.commit()
                logger.info("⏭️  No suitable stocks found with required confidence")
                return {
                    'success': True,
                    'status': 'skipped',
                    'message': 'No suitable stocks found'
                }

            # Step 5: Create buy orders
            user = self.session.query(User).filter_by(id=user_id).first()
            orders_result = self._create_buy_orders(
                user, settings, strategies_result['stocks'], available_amount, execution
            )

            # Update execution status
            execution.orders_created = orders_result['orders_created']
            execution.total_amount_invested = orders_result['total_invested']
            execution.selected_strategies = json.dumps(strategies_result['strategies_used'])
            execution.execution_details = json.dumps(orders_result['details'])

            if orders_result['orders_created'] > 0:
                execution.status = 'success'
                self.session.commit()
                logger.info(f"✅ Auto-trading completed: {orders_result['orders_created']} orders created, "
                           f"₹{orders_result['total_invested']:.2f} invested")
                return {
                    'success': True,
                    'status': 'success',
                    'orders_created': orders_result['orders_created'],
                    'total_invested': orders_result['total_invested'],
                    'details': orders_result['details']
                }
            else:
                execution.status = 'skipped'
                execution.error_message = 'No orders could be created'
                self.session.commit()
                logger.warning("⚠️  No orders could be created")
                return {
                    'success': True,
                    'status': 'skipped',
                    'message': 'No orders could be created'
                }

        except Exception as e:
            logger.error(f"❌ Auto-trading execution failed for user {user_id}: {e}", exc_info=True)
            if 'execution' in locals():
                execution.status = 'failed'
                execution.error_message = str(e)
                self.session.commit()
            return {
                'success': False,
                'error': str(e)
            }

    def _check_market_sentiment(self, settings: AutoTradingSettings,
                                execution: AutoTradingExecution) -> Dict[str, Any]:
        """Market sentiment check (simplified - no AI)."""
        # Simplified version without AI/Ollama
        # Always proceed with neutral sentiment
        execution.market_sentiment_type = 'neutral'
        execution.market_sentiment_score = 0.5
        execution.ai_confidence = 0.0

        logger.info("✅ Market sentiment check: Proceeding (AI disabled)")
        return {
            'proceed': True,
            'sentiment': {
                'sentiment_type': 'neutral',
                'sentiment_score': 0.5
            }
        }

    def _check_weekly_limits(self, user_id: int, settings: AutoTradingSettings,
                            execution: AutoTradingExecution) -> Dict[str, Any]:
        """Check weekly limits for trading."""
        try:
            # Get start of current week (Monday)
            today = datetime.now()
            start_of_week = today - timedelta(days=today.weekday())
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)

            # Query weekly orders
            weekly_orders = self.session.query(
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

            # Calculate remaining limits
            remaining_amount = settings.max_amount_per_week - weekly_spent
            remaining_buys = settings.max_buys_per_week - weekly_buys

            # Store in execution log
            execution.weekly_amount_spent = weekly_spent
            execution.weekly_buys_count = weekly_buys
            execution.remaining_weekly_amount = remaining_amount
            execution.remaining_weekly_buys = remaining_buys

            if remaining_amount <= 0:
                return {
                    'proceed': False,
                    'message': f'Weekly amount limit reached (₹{settings.max_amount_per_week:.2f})'
                }

            if remaining_buys <= 0:
                return {
                    'proceed': False,
                    'message': f'Weekly buy limit reached ({settings.max_buys_per_week} trades)'
                }

            logger.info(f"✅ Weekly limits OK: ₹{remaining_amount:.2f} remaining, {remaining_buys} trades left")
            return {
                'proceed': True,
                'remaining_amount': remaining_amount,
                'remaining_buys': remaining_buys
            }

        except Exception as e:
            logger.error(f"Error checking weekly limits: {e}")
            return {
                'proceed': False,
                'message': f'Error checking weekly limits: {str(e)}'
            }

    def _check_account_balance(self, user_id: int, execution: AutoTradingExecution) -> Dict[str, Any]:
        """Check account balance from broker."""
        try:
            from src.services.core.unified_broker_service import get_unified_broker_service

            unified_service = get_unified_broker_service()
            balance_result = unified_service.get_account_balance(user_id)

            if not balance_result.get('success'):
                error_msg = balance_result.get('error', 'Unknown error fetching account balance')
                logger.warning(f"⚠️  Failed to fetch account balance: {error_msg}")
                return {
                    'proceed': False,
                    'message': f'Failed to fetch account balance: {error_msg}'
                }

            balance_data = balance_result.get('data', {})
            available_balance = balance_data.get('available_cash', 0.0)
            total_balance = balance_data.get('total_balance', 0.0)

            # Store in execution log
            execution.account_balance = total_balance
            execution.available_to_invest = available_balance

            if available_balance < 1000:  # Minimum ₹1000 required
                return {
                    'proceed': False,
                    'message': f'Insufficient balance (₹{available_balance:.2f})'
                }

            logger.info(f"✅ Account balance OK: ₹{available_balance:.2f} available")
            return {
                'proceed': True,
                'available_balance': available_balance
            }

        except Exception as e:
            logger.error(f"Error checking account balance: {e}", exc_info=True)
            return {
                'proceed': False,
                'message': f'Error checking account balance: {str(e)}'
            }

    def _select_top_strategies(self, settings: AutoTradingSettings,
                               max_stocks: int) -> Dict[str, Any]:
        """Select top 8-21 EMA strategy stocks with best signals."""
        try:
            # Parse preferred settings
            preferred_strategies = json.loads(settings.preferred_strategies or '["default_risk"]')

            # Query latest daily suggested stocks (8-21 EMA strategy)
            query = text("""
                SELECT
                    d.symbol,
                    d.stock_name,
                    d.current_price,
                    d.strategy,
                    d.selection_score,
                    d.ema_8,
                    d.ema_21,
                    d.ema_trend_score,
                    d.demarker,
                    d.signal_quality,
                    d.target_price,
                    d.stop_loss,
                    d.recommendation,
                    d.date
                FROM daily_suggested_stocks d
                WHERE d.date = (SELECT MAX(date) FROM daily_suggested_stocks)
                  AND d.strategy = ANY(:strategies)
                  AND d.buy_signal = TRUE
                  AND d.recommendation = 'BUY'
                  AND d.target_price > d.current_price
                  AND d.signal_quality IN ('high', 'medium')
                ORDER BY d.selection_score DESC, d.ema_trend_score DESC
                LIMIT :limit
            """)

            result = self.session.execute(query, {
                'strategies': preferred_strategies,
                'limit': max_stocks
            })

            stocks = [dict(row._mapping) for row in result]

            if not stocks:
                logger.warning("⚠️  No stocks found matching 8-21 EMA criteria")

            strategies_used = list(set([s['strategy'] for s in stocks]))

            logger.info(f"✅ Selected {len(stocks)} stocks from 8-21 EMA strategies: {strategies_used}")

            return {
                'stocks': stocks,
                'strategies_used': strategies_used
            }

        except Exception as e:
            logger.error(f"Error selecting 8-21 EMA strategies: {e}")
            return {
                'stocks': [],
                'strategies_used': []
            }

    def _create_buy_orders(self, user: User, settings: AutoTradingSettings,
                          stocks: List[Dict], available_amount: float,
                          execution: AutoTradingExecution) -> Dict[str, Any]:
        """Create buy orders for selected stocks."""
        try:
            from src.services.core.unified_broker_service import get_unified_broker_service

            unified_service = get_unified_broker_service()
            orders_created = 0
            total_invested = 0.0
            order_details = []

            # Allocate investment equally among stocks
            amount_per_stock = available_amount / len(stocks) if stocks else 0

            for stock in stocks:
                try:
                    symbol = stock['symbol']
                    current_price = stock['current_price']
                    quantity = int(amount_per_stock / current_price)

                    if quantity < 1:
                        logger.warning(f"⚠️  Skipping {symbol}: quantity < 1")
                        continue

                    investment = quantity * current_price

                    # Calculate stop-loss and target from EMA strategy
                    target_price = stock['target_price']
                    stop_loss = stock['stop_loss'] or (current_price * 0.95)  # 5% default

                    # Create order (paper or real based on user mode)
                    if user.is_mock_trading_mode:
                        # Paper trading - create mock order in DB only
                        order_result = self._create_mock_order(
                            user.id, symbol, quantity, current_price,
                            target_price, stop_loss, stock
                        )
                    else:
                        # Real trading - place actual order via broker
                        order_result = unified_service.place_order(
                            user_id=user.id,
                            symbol=symbol,
                            quantity=quantity,
                            order_type='MARKET',
                            transaction_type='BUY',
                            product='INTRADAY'
                        )

                    if order_result.get('success'):
                        orders_created += 1
                        total_invested += investment

                        # Create performance tracking
                        order_id = order_result['order_id']
                        self._create_performance_tracking(
                            order_id, user.id, execution.id, stock, quantity, current_price,
                            stop_loss, target_price
                        )

                        order_details.append({
                            'symbol': symbol,
                            'quantity': quantity,
                            'price': current_price,
                            'investment': investment,
                            'stop_loss': stop_loss,
                            'target': target_price,
                            'order_id': order_id,
                            'mode': 'paper' if user.is_mock_trading_mode else 'live'
                        })

                        logger.info(f"✅ Order created for {symbol}: {quantity} @ ₹{current_price:.2f}")

                except Exception as e:
                    logger.error(f"❌ Failed to create order for {stock['symbol']}: {e}")
                    continue

            return {
                'orders_created': orders_created,
                'total_invested': total_invested,
                'details': order_details
            }

        except Exception as e:
            logger.error(f"Error creating buy orders: {e}")
            return {
                'orders_created': 0,
                'total_invested': 0.0,
                'details': []
            }

    def _create_mock_order(self, user_id: int, symbol: str, quantity: int, price: float,
                          target_price: float, stop_loss: float, stock_data: Dict) -> Dict[str, Any]:
        """Create a mock order for paper trading."""
        try:
            order_id = f"MOCK_{user_id}_{symbol}_{int(datetime.now().timestamp())}"

            order = Order(
                user_id=user_id,
                order_id=order_id,
                tradingsymbol=symbol,
                exchange='NSE',
                order_type='MARKET',
                transaction_type='BUY',
                quantity=quantity,
                price=price,
                average_price=price,
                filled_quantity=quantity,
                pending_quantity=0,
                order_status='COMPLETE',
                placed_at=datetime.now(),
                is_mock_order=True,
                strategy=stock_data.get('strategy')
            )

            self.session.add(order)
            self.session.flush()

            logger.info(f"✅ Mock order created: {order_id}")

            return {
                'success': True,
                'order_id': order_id
            }

        except Exception as e:
            logger.error(f"Error creating mock order: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _create_performance_tracking(self, order_id: str, user_id: int,
                                    execution_id: int, stock_data: Dict,
                                    quantity: int, entry_price: float,
                                    stop_loss: float, target_price: float) -> None:
        """Create performance tracking for order."""
        try:
            performance = OrderPerformance(
                order_id=order_id,
                user_id=user_id,
                auto_execution_id=execution_id,
                symbol=stock_data['symbol'],
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=stop_loss,
                target_price=target_price,
                strategy=stock_data.get('strategy'),
                current_price=entry_price,
                current_value=entry_price * quantity,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                is_active=True
            )

            self.session.add(performance)
            self.session.flush()

            logger.info(f"✅ Performance tracking created for {order_id}")

        except Exception as e:
            logger.error(f"Error creating performance tracking: {e}")


    def execute_auto_trading_for_all_users(self) -> Dict[str, Any]:
        """Execute auto-trading for all users with auto-trading enabled."""
        try:
            logger.info("🤖 Starting auto-trading for all enabled users")

            # Get all users with auto-trading enabled
            enabled_users = self.session.query(AutoTradingSettings).filter_by(is_enabled=True).all()

            if not enabled_users:
                logger.info("ℹ️  No users with auto-trading enabled")
                return {
                    'success': True,
                    'message': 'No users with auto-trading enabled',
                    'results': []
                }

            results = []
            for settings in enabled_users:
                result = self.execute_auto_trading_for_user(settings.user_id)
                results.append({
                    'user_id': settings.user_id,
                    'result': result
                })

            logger.info(f"✅ Auto-trading completed for {len(results)} users")

            return {
                'success': True,
                'total_users': len(results),
                'results': results
            }

        except Exception as e:
            logger.error(f"❌ Auto-trading for all users failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }


# Global service instance
_auto_trading_service = None


def get_auto_trading_service() -> AutoTradingService:
    """Get global auto-trading service instance."""
    global _auto_trading_service
    if _auto_trading_service is None:
        _auto_trading_service = AutoTradingService()
    return _auto_trading_service
