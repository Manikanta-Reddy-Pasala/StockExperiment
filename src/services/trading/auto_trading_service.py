"""
Auto-Trading Service
Handles automated trading execution with weekly limits and performance tracking.
Supports swing trading (EMA strategy) and day trading (gap-up/momentum).
"""

import logging
import json
from typing import Dict, List, Any, Optional
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
    """Service for automated trading with EMA strategy decisions."""

    def __init__(self):
        """Initialize auto-trading service (stateless - sessions created per operation)."""
        self.db_manager = get_database_manager()

    def execute_auto_trading_for_all_users(self) -> Dict[str, Any]:
        """Execute auto-trading for all users with auto-trading enabled."""
        try:
            logger.info("Starting auto-trading for all enabled users")

            with self.db_manager.get_session() as session:
                enabled_users = session.query(AutoTradingSettings).filter_by(is_enabled=True).all()
                user_ids = [s.user_id for s in enabled_users]

            if not user_ids:
                logger.info("No users with auto-trading enabled")
                return {
                    'success': True,
                    'message': 'No users with auto-trading enabled',
                    'results': []
                }

            results = []
            for user_id in user_ids:
                result = self.execute_auto_trading_for_user(user_id)
                results.append({
                    'user_id': user_id,
                    'result': result
                })

            logger.info(f"Auto-trading completed for {len(results)} users")

            return {
                'success': True,
                'total_users': len(results),
                'results': results
            }

        except Exception as e:
            logger.error(f"Auto-trading for all users failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def execute_auto_trading_for_user(self, user_id: int) -> Dict[str, Any]:
        """
        Execute auto-trading for a specific user.
        Supports swing, day, or both trading modes.
        """
        with self.db_manager.get_session() as session:
            try:
                logger.info(f"Starting auto-trading execution for user {user_id}")

                settings = session.query(AutoTradingSettings).filter_by(user_id=user_id).first()

                if not settings:
                    logger.warning(f"No auto-trading settings found for user {user_id}")
                    return {
                        'success': False,
                        'error': 'Auto-trading not configured for this user'
                    }

                if not settings.is_enabled:
                    logger.info(f"Auto-trading disabled for user {user_id}")
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
                session.add(execution)
                session.flush()

                # Step 1: Check market sentiment (always passes - AI disabled)
                execution.market_sentiment_type = 'neutral'
                execution.market_sentiment_score = 0.5
                execution.ai_confidence = 0.0

                # Step 2: Check weekly limits
                limits_result = self._check_weekly_limits(session, user_id, settings, execution)
                if not limits_result['proceed']:
                    execution.status = 'skipped'
                    execution.error_message = limits_result['message']
                    session.commit()
                    logger.info(f"Skipped: {limits_result['message']}")
                    return {
                        'success': True,
                        'status': 'skipped',
                        'message': limits_result['message']
                    }

                # Step 3: Check account balance
                balance_result = self._check_account_balance(session, user_id, settings, execution)
                if not balance_result['proceed']:
                    execution.status = 'skipped'
                    execution.error_message = balance_result['message']
                    session.commit()
                    logger.info(f"Skipped: {balance_result['message']}")
                    return {
                        'success': True,
                        'status': 'skipped',
                        'message': balance_result['message']
                    }

                available_amount = min(
                    limits_result['remaining_amount'],
                    balance_result['available_balance']
                )

                trading_mode = settings.trading_mode or 'swing'
                all_stocks = []
                all_strategies = []

                # Step 4a: Swing trading stock selection
                if trading_mode in ('swing', 'both'):
                    strategies_result = self._select_top_strategies(session, settings, limits_result['remaining_buys'])
                    if strategies_result['stocks']:
                        for stock in strategies_result['stocks']:
                            stock['_trading_type'] = 'swing'
                        all_stocks.extend(strategies_result['stocks'])
                        all_strategies.extend(strategies_result['strategies_used'])

                # Step 4b: Day trading stock selection
                if trading_mode in ('day', 'both'):
                    from src.services.trading.day_trading_service import get_day_trading_service
                    day_service = get_day_trading_service()
                    remaining_slots = limits_result['remaining_buys'] - len(all_stocks)
                    if remaining_slots > 0:
                        day_result = day_service.select_day_trading_stocks(
                            session, user_id, max_stocks=min(remaining_slots, 5)
                        )
                        if day_result['stocks']:
                            for stock in day_result['stocks']:
                                stock['_trading_type'] = 'day'
                            all_stocks.extend(day_result['stocks'])
                            all_strategies.extend(day_result['strategies_used'])

                if not all_stocks:
                    execution.status = 'skipped'
                    execution.error_message = 'No suitable stocks found'
                    session.commit()
                    logger.info("Skipped: No suitable stocks found")
                    return {
                        'success': True,
                        'status': 'skipped',
                        'message': 'No suitable stocks found'
                    }

                # Step 5: Create buy orders
                user = session.query(User).filter_by(id=user_id).first()
                execution_id = execution.id
                orders_result = self._create_buy_orders(
                    session, user, settings, all_stocks, available_amount, execution_id
                )

                # Re-fetch execution after external API calls may have detached it
                execution = session.get(AutoTradingExecution, execution_id)
                execution.orders_created = orders_result['orders_created']
                execution.total_amount_invested = orders_result['total_invested']
                execution.selected_strategies = json.dumps(list(set(all_strategies)))
                execution.execution_details = json.dumps(orders_result['details'])

                if orders_result['orders_created'] > 0:
                    execution.status = 'success'
                    session.commit()
                    logger.info(f"Auto-trading completed: {orders_result['orders_created']} orders, "
                               f"{orders_result['total_invested']:.2f} invested")
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
                    session.commit()
                    logger.warning("No orders could be created")
                    return {
                        'success': True,
                        'status': 'skipped',
                        'message': 'No orders could be created'
                    }

            except Exception as e:
                logger.error(f"Auto-trading execution failed for user {user_id}: {e}", exc_info=True)
                try:
                    if 'execution' in locals():
                        execution.status = 'failed'
                        execution.error_message = str(e)
                        session.commit()
                except Exception:
                    pass
                return {
                    'success': False,
                    'error': str(e)
                }

    def _check_weekly_limits(self, session: Session, user_id: int,
                            settings: AutoTradingSettings,
                            execution: AutoTradingExecution) -> Dict[str, Any]:
        """Check weekly limits for trading."""
        try:
            today = datetime.now()
            start_of_week = today - timedelta(days=today.weekday())
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)

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

            remaining_amount = settings.max_amount_per_week - weekly_spent
            remaining_buys = settings.max_buys_per_week - weekly_buys

            execution.weekly_amount_spent = weekly_spent
            execution.weekly_buys_count = weekly_buys
            execution.remaining_weekly_amount = remaining_amount
            execution.remaining_weekly_buys = remaining_buys

            if remaining_amount <= 0:
                return {
                    'proceed': False,
                    'message': f'Weekly amount limit reached ({settings.max_amount_per_week:.2f})'
                }

            if remaining_buys <= 0:
                return {
                    'proceed': False,
                    'message': f'Weekly buy limit reached ({settings.max_buys_per_week} trades)'
                }

            logger.info(f"Weekly limits OK: {remaining_amount:.2f} remaining, {remaining_buys} trades left")
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

    def _check_account_balance(self, session: Session, user_id: int,
                              settings: AutoTradingSettings,
                              execution: AutoTradingExecution) -> Dict[str, Any]:
        """Check account balance from broker or use virtual capital for paper trading."""
        try:
            user = session.query(User).filter(User.id == user_id).first()
            is_paper_trading = user.is_mock_trading_mode if user else False

            if is_paper_trading:
                virtual_capital = settings.virtual_capital or 100000.0

                # Query active positions from order_performance
                used_capital = session.execute(text("""
                    SELECT COALESCE(SUM(entry_price * COALESCE(remaining_quantity, quantity)), 0)
                    FROM order_performance
                    WHERE user_id = :user_id AND is_active = true
                """), {'user_id': user_id}).scalar() or 0.0
                used_capital = float(used_capital)

                # Include realized P&L from closed positions
                realized_pnl = session.execute(text("""
                    SELECT COALESCE(SUM(realized_pnl), 0)
                    FROM order_performance
                    WHERE user_id = :user_id AND is_active = false AND realized_pnl IS NOT NULL
                """), {'user_id': user_id}).scalar() or 0.0
                realized_pnl = float(realized_pnl)

                available_balance = virtual_capital - used_capital + realized_pnl
                total_balance = virtual_capital

                execution.account_balance = total_balance
                execution.available_to_invest = available_balance

                logger.info(f"Paper trading: Virtual balance {available_balance:.2f} available ({used_capital:.2f} in use)")
                return {
                    'proceed': True,
                    'available_balance': available_balance,
                    'is_paper_trading': True
                }

            # Live trading: check actual broker balance
            from src.services.core.unified_broker_service import get_unified_broker_service

            unified_service = get_unified_broker_service()
            balance_result = unified_service.get_account_balance(user_id)

            if not balance_result.get('success'):
                error_msg = balance_result.get('error', 'Unknown error fetching account balance')
                logger.warning(f"Failed to fetch account balance: {error_msg}")
                return {
                    'proceed': False,
                    'message': f'Failed to fetch account balance: {error_msg}'
                }

            balance_data = balance_result.get('data', {})
            available_balance = float(balance_data.get('available_cash', 0.0) or 0.0)
            total_balance = float(balance_data.get('total_balance', 0.0) or 0.0)

            execution.account_balance = total_balance
            execution.available_to_invest = available_balance

            if available_balance < 1000:
                return {
                    'proceed': False,
                    'message': f'Insufficient balance ({available_balance:.2f})'
                }

            logger.info(f"Account balance OK: {available_balance:.2f} available")
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

    def _select_top_strategies(self, session: Session, settings: AutoTradingSettings,
                               max_stocks: int) -> Dict[str, Any]:
        """Select top 8-21 EMA strategy stocks with best signals."""
        try:
            preferred_strategies = json.loads(settings.preferred_strategies or '["unified"]')

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
                    d.date,
                    d.fib_target_1,
                    d.fib_target_2,
                    d.fib_target_3,
                    d.sector
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

            result = session.execute(query, {
                'strategies': preferred_strategies,
                'limit': max_stocks
            })

            stocks = [dict(row._mapping) for row in result]

            # Sector cap: max 2 stocks per sector to avoid concentration risk
            sector_counts = {}
            capped = []
            for s in stocks:
                sec = s.get('sector') or 'Unknown'
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
                if sector_counts[sec] <= 2:
                    capped.append(s)
                else:
                    logger.info(f"Sector cap: skipping {s['symbol']} (sector '{sec}' already has 2 picks)")
            stocks = capped

            if not stocks:
                logger.warning("No stocks found matching 8-21 EMA criteria")

            strategies_used = list(set([s['strategy'] for s in stocks]))

            logger.info(f"Selected {len(stocks)} stocks from 8-21 EMA strategies: {strategies_used}")

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

    def _create_buy_orders(self, session: Session, user: User, settings: AutoTradingSettings,
                          stocks: List[Dict], available_amount: float,
                          execution_id: int) -> Dict[str, Any]:
        """Create buy orders for selected stocks."""
        try:
            from src.services.core.unified_broker_service import get_unified_broker_service

            unified_service = get_unified_broker_service()
            orders_created = 0
            total_invested = 0.0
            order_details = []

            # Extract user fields upfront to avoid detached session issues
            user_id = user.id
            is_paper_trading = user.is_mock_trading_mode

            amount_per_stock = available_amount / len(stocks) if stocks else 0
            remaining_capital = available_amount

            for stock in stocks:
                try:
                    symbol = stock['symbol']
                    current_price = stock['current_price']

                    # Pre-market gap check: skip if price gapped >3% from signal price
                    # If live price unavailable, proceed with signal price (paper trading safe)
                    live_price = self._get_live_price(session, symbol, user_id)
                    if live_price and current_price and current_price > 0:
                        gap_pct = ((live_price - current_price) / current_price) * 100
                        if abs(gap_pct) > 3.0:
                            logger.info(f"Skipping {symbol}: gap {gap_pct:.1f}% exceeds 3% "
                                       f"(signal: {current_price:.2f}, live: {live_price:.2f})")
                            continue
                        current_price = live_price  # Use live price for order

                    # Cap per-stock allocation to remaining capital
                    allocation = min(amount_per_stock, remaining_capital)
                    quantity = int(allocation / current_price)

                    if quantity < 1:
                        logger.warning(f"Skipping {symbol}: quantity < 1 (remaining capital: {remaining_capital:.2f})")
                        continue

                    investment = quantity * current_price

                    # Check if this order would exceed remaining capital
                    if investment > remaining_capital:
                        quantity = int(remaining_capital / current_price)
                        if quantity < 1:
                            logger.warning(f"Skipping {symbol}: insufficient remaining capital ({remaining_capital:.2f})")
                            continue
                        investment = quantity * current_price

                    target_price = stock['target_price']
                    stop_loss = stock['stop_loss'] or (current_price * 0.95)
                    trading_type = stock.get('_trading_type', 'swing')

                    if is_paper_trading:
                        order_result = self._create_mock_order(
                            session, user_id, symbol, quantity, current_price,
                            target_price, stop_loss, stock
                        )
                    else:
                        order_result = unified_service.place_order(
                            user_id=user_id,
                            order_data={
                                'symbol': symbol,
                                'quantity': quantity,
                                'order_type': 'MARKET',
                                'transaction_type': 'BUY',
                                'product': 'INTRADAY'
                            }
                        )

                    if order_result.get('success'):
                        orders_created += 1
                        total_invested += investment
                        remaining_capital -= investment

                        order_id = order_result['order_id']
                        self._create_performance_tracking(
                            session, order_id, user_id, execution_id, stock, quantity, current_price,
                            stop_loss, target_price, trading_type
                        )

                        order_details.append({
                            'symbol': symbol,
                            'quantity': quantity,
                            'price': current_price,
                            'investment': investment,
                            'stop_loss': stop_loss,
                            'target': target_price,
                            'order_id': order_id,
                            'mode': 'paper' if is_paper_trading else 'live',
                            'trading_type': trading_type
                        })

                        logger.info(f"Order created for {symbol}: {quantity} @ {current_price:.2f} ({trading_type})")

                except Exception as e:
                    logger.error(f"Failed to create order for {stock['symbol']}: {e}")
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

    def _get_live_price(self, session: Session, symbol: str, user_id: int) -> Optional[float]:
        """Get live price for a symbol. Tries stocks table first, then broker API."""
        try:
            # Try stocks table first (always available for paper trading)
            from src.models.stock_models import Stock
            stocks_price = None
            stock = session.query(Stock).filter_by(symbol=symbol).first()
            if stock and stock.current_price and stock.current_price > 0:
                stocks_price = float(stock.current_price)

            # Try broker API for fresher intraday data
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
            logger.error(f"Error fetching live price for {symbol}: {e}")
            return None

    def _create_mock_order(self, session: Session, user_id: int, symbol: str,
                          quantity: int, price: float, target_price: float,
                          stop_loss: float, stock_data: Dict) -> Dict[str, Any]:
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

            session.add(order)
            session.flush()

            logger.info(f"Mock order created: {order_id}")

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

    def _create_performance_tracking(self, session: Session, order_id: str,
                                    user_id: int, execution_id: int,
                                    stock_data: Dict, quantity: int,
                                    entry_price: float, stop_loss: float,
                                    target_price: float,
                                    trading_type: str = 'swing') -> None:
        """Create performance tracking for order with partial exit targets."""
        try:
            # Determine partial exit targets
            if trading_type == 'swing':
                # Use Fib levels from daily_suggested_stocks
                target_price_1 = stock_data.get('fib_target_1') or target_price
                target_price_2 = stock_data.get('fib_target_2')
                target_price_3 = stock_data.get('fib_target_3')

                # If fib targets not available, calculate from target_price (Fib 127.2%)
                if not target_price_2 and target_price and entry_price:
                    swing_range = target_price - entry_price
                    if swing_range > 0:
                        # target_price is ~Fib 127.2%, scale up for 161.8% and 200%
                        fib_unit = swing_range / 0.272  # reverse engineer the fib range
                        target_price_2 = entry_price + (fib_unit * 0.618)
                        target_price_3 = entry_price + (fib_unit * 1.000)
            else:
                # Day trading: single target, no partial exits
                target_price_1 = None
                target_price_2 = None
                target_price_3 = None

            performance = OrderPerformance(
                order_id=order_id,
                user_id=user_id,
                auto_execution_id=execution_id,
                symbol=stock_data['symbol'],
                entry_price=entry_price,
                quantity=quantity,
                original_quantity=quantity,
                remaining_quantity=quantity,
                stop_loss=stop_loss,
                target_price=target_price,
                target_price_1=target_price_1,
                target_price_2=target_price_2,
                target_price_3=target_price_3,
                strategy=stock_data.get('strategy'),
                trading_type=trading_type,
                current_price=entry_price,
                current_value=entry_price * quantity,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                partial_pnl_realized=0.0,
                is_active=True
            )

            session.add(performance)
            session.flush()

            logger.info(f"Performance tracking created for {order_id} ({trading_type})"
                       f"{f' targets: {target_price_1}/{target_price_2}/{target_price_3}' if target_price_1 else ''}")

        except Exception as e:
            logger.error(f"Error creating performance tracking: {e}")


def get_auto_trading_service() -> AutoTradingService:
    """Get auto-trading service instance (stateless, no global needed)."""
    return AutoTradingService()
