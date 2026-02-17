"""
Paper Trading Dashboard Provider

Implements IDashboardProvider using local database tables (OrderPerformance, Orders,
daily_suggested_stocks, stocks) instead of broker APIs. Used when user is in mock
trading mode (User.is_mock_trading_mode = True).
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy import text, and_, desc, func

from ..interfaces.dashboard_interface import IDashboardProvider, DashboardMetrics, MarketIndex

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000.0


class PaperTradingDashboardProvider(IDashboardProvider):
    """Dashboard provider that reads from local DB for paper/mock trading users."""

    def _get_db_manager(self):
        from src.models.database import get_database_manager
        return get_database_manager()

    def get_market_overview(self, user_id: int) -> Dict[str, Any]:
        """Top suggested stocks from daily_suggested_stocks as market overview."""
        try:
            db_manager = self._get_db_manager()
            market_indices = []

            with db_manager.get_session() as session:
                query = text("""
                    SELECT symbol, stock_name, current_price, selection_score
                    FROM daily_suggested_stocks
                    WHERE date = (SELECT MAX(date) FROM daily_suggested_stocks)
                    AND current_price IS NOT NULL
                    ORDER BY selection_score DESC
                    LIMIT 10
                """)
                result = session.execute(query)
                rows = result.fetchall()

                for row in rows:
                    symbol, stock_name, current_price, selection_score = row
                    market_indices.append({
                        'symbol': symbol or '',
                        'name': stock_name or (symbol or '').replace('NSE:', '').replace('-EQ', ''),
                        'price': float(current_price or 0),
                        'change': 0.0,
                        'change_percent': 0.0,
                        'score': float(selection_score or 0),
                    })

            return {
                'success': True,
                'data': market_indices,
                'message': 'Top suggested stocks (paper trading mode)',
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Paper trading market overview error: {e}")
            return {
                'success': True,
                'data': [],
                'message': 'No market data available',
                'last_updated': datetime.now().isoformat()
            }

    def get_portfolio_summary(self, user_id: int) -> Dict[str, Any]:
        """Aggregate portfolio from order_performance (active mock orders)."""
        try:
            db_manager = self._get_db_manager()

            with db_manager.get_session() as session:
                query = text("""
                    SELECT
                        COUNT(*) as positions_count,
                        COALESCE(SUM(entry_price * quantity), 0) as total_invested,
                        COALESCE(SUM(current_price * quantity), 0) as total_current_value,
                        COALESCE(SUM(unrealized_pnl), 0) as total_unrealized_pnl
                    FROM order_performance
                    WHERE user_id = :user_id AND is_active = true
                """)
                row = session.execute(query, {'user_id': user_id}).fetchone()

                positions_count = int(row[0] or 0)
                total_invested = float(row[1] or 0)
                total_current_value = float(row[2] or 0)
                total_unrealized_pnl = float(row[3] or 0)

                # Get realized P&L from closed positions
                closed_query = text("""
                    SELECT COALESCE(SUM(realized_pnl), 0) as total_realized_pnl
                    FROM order_performance
                    WHERE user_id = :user_id AND is_active = false AND realized_pnl IS NOT NULL
                """)
                closed_row = session.execute(closed_query, {'user_id': user_id}).fetchone()
                total_realized_pnl = float(closed_row[0] or 0)

                total_pnl = total_unrealized_pnl + total_realized_pnl
                available_cash = INITIAL_CAPITAL - total_invested + total_realized_pnl
                total_portfolio_value = available_cash + total_current_value

                total_pnl_percent = (total_pnl / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL > 0 else 0
                daily_pnl_percent = (total_unrealized_pnl / total_invested * 100) if total_invested > 0 else 0

                metrics = DashboardMetrics(
                    total_pnl=round(total_pnl, 2),
                    total_portfolio_value=round(total_portfolio_value, 2),
                    available_cash=round(available_cash, 2),
                    holdings_count=0,
                    positions_count=positions_count,
                    daily_pnl=round(total_unrealized_pnl, 2),
                    daily_pnl_percent=round(daily_pnl_percent, 2),
                    total_pnl_percent=round(total_pnl_percent, 2)
                )

            return {
                'success': True,
                'data': metrics.to_dict(),
                'message': f'Paper trading portfolio (Initial: ₹{INITIAL_CAPITAL:,.0f})',
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Paper trading portfolio summary error: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': DashboardMetrics().to_dict(),
                'last_updated': datetime.now().isoformat()
            }

    def get_top_holdings(self, user_id: int, limit: int = 5) -> Dict[str, Any]:
        """Active positions from order_performance where is_active=True."""
        try:
            db_manager = self._get_db_manager()

            with db_manager.get_session() as session:
                query = text("""
                    SELECT symbol, quantity, entry_price, current_price,
                           unrealized_pnl, unrealized_pnl_pct, stop_loss,
                           target_price, strategy, created_at
                    FROM order_performance
                    WHERE user_id = :user_id AND is_active = true
                    ORDER BY (current_price * quantity) DESC
                    LIMIT :limit
                """)
                rows = session.execute(query, {'user_id': user_id, 'limit': limit}).fetchall()

                holdings = []
                for row in rows:
                    symbol, quantity, entry_price, current_price, unrealized_pnl, \
                        unrealized_pnl_pct, stop_loss, target_price, strategy, created_at = row
                    current_value = (float(current_price or 0)) * int(quantity or 0)
                    holdings.append({
                        'symbol': symbol or '',
                        'symbol_name': (symbol or '').replace('NSE:', '').replace('-EQ', ''),
                        'quantity': int(quantity or 0),
                        'entry_price': float(entry_price or 0),
                        'current_price': float(current_price or 0),
                        'current_value': round(current_value, 2),
                        'pnl': float(unrealized_pnl or 0),
                        'pnl_percent': float(unrealized_pnl_pct or 0),
                        'stop_loss': float(stop_loss or 0),
                        'target_price': float(target_price or 0),
                        'strategy': strategy or '',
                        'entry_date': created_at.isoformat() if created_at else '',
                    })

            return {
                'success': True,
                'data': holdings,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Paper trading top holdings error: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }

    def get_recent_activity(self, user_id: int, limit: int = 10) -> Dict[str, Any]:
        """Recent mock orders from orders table."""
        try:
            db_manager = self._get_db_manager()

            with db_manager.get_session() as session:
                query = text("""
                    SELECT order_id, tradingsymbol, transaction_type, quantity,
                           price, order_status, strategy, created_at
                    FROM orders
                    WHERE user_id = :user_id AND is_mock_order = true
                    ORDER BY created_at DESC
                    LIMIT :limit
                """)
                rows = session.execute(query, {'user_id': user_id, 'limit': limit}).fetchall()

                activities = []
                for row in rows:
                    order_id, symbol, side, quantity, price, status, strategy, created_at = row
                    activities.append({
                        'type': 'order',
                        'id': order_id or '',
                        'symbol': symbol or '',
                        'symbol_name': (symbol or '').replace('NSE:', '').replace('-EQ', ''),
                        'side': side or 'BUY',
                        'status': status or '',
                        'quantity': int(quantity or 0),
                        'price': float(price or 0),
                        'order_type': 'MARKET',
                        'strategy': strategy or '',
                        'timestamp': created_at.isoformat() if created_at else '',
                    })

            return {
                'success': True,
                'data': activities,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Paper trading recent activity error: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }

    def get_account_balance(self, user_id: int) -> Dict[str, Any]:
        """Calculate balance from INITIAL_CAPITAL minus invested + realized P&L."""
        try:
            db_manager = self._get_db_manager()

            with db_manager.get_session() as session:
                # Active positions: invested capital
                active_query = text("""
                    SELECT COALESCE(SUM(entry_price * quantity), 0) as invested
                    FROM order_performance
                    WHERE user_id = :user_id AND is_active = true
                """)
                invested = float(session.execute(active_query, {'user_id': user_id}).fetchone()[0] or 0)

                # Closed positions: realized P&L
                closed_query = text("""
                    SELECT COALESCE(SUM(realized_pnl), 0) as realized
                    FROM order_performance
                    WHERE user_id = :user_id AND is_active = false AND realized_pnl IS NOT NULL
                """)
                realized_pnl = float(session.execute(closed_query, {'user_id': user_id}).fetchone()[0] or 0)

                available_cash = INITIAL_CAPITAL - invested + realized_pnl
                total_balance = INITIAL_CAPITAL + realized_pnl

            return {
                'success': True,
                'data': {
                    'available_cash': round(available_cash, 2),
                    'total_balance': round(total_balance, 2),
                    'margin_used': round(invested, 2),
                    'initial_capital': INITIAL_CAPITAL,
                    'realized_pnl': round(realized_pnl, 2),
                },
                'message': 'Paper trading account',
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Paper trading account balance error: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': {'available_cash': 0, 'total_balance': 0, 'margin_used': 0},
                'last_updated': datetime.now().isoformat()
            }

    def get_daily_pnl_chart_data(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """P&L chart data from order_performance_snapshots."""
        try:
            db_manager = self._get_db_manager()

            with db_manager.get_session() as session:
                query = text("""
                    SELECT
                        DATE(ops.snapshot_date) as snap_date,
                        SUM(ops.unrealized_pnl) as daily_pnl
                    FROM order_performance_snapshots ops
                    JOIN order_performance op ON ops.order_performance_id = op.id
                    WHERE op.user_id = :user_id
                      AND ops.snapshot_date >= :start_date
                    GROUP BY DATE(ops.snapshot_date)
                    ORDER BY snap_date ASC
                """)
                start_date = datetime.now() - timedelta(days=days)
                rows = session.execute(query, {
                    'user_id': user_id,
                    'start_date': start_date
                }).fetchall()

                # Also get realized P&L from closed positions to include in chart
                realized_query = text("""
                    SELECT
                        DATE(exit_date) as exit_day,
                        SUM(realized_pnl) as day_realized
                    FROM order_performance
                    WHERE user_id = :user_id
                      AND is_active = false
                      AND exit_date >= :start_date
                      AND realized_pnl IS NOT NULL
                    GROUP BY DATE(exit_date)
                """)
                realized_rows = session.execute(realized_query, {
                    'user_id': user_id,
                    'start_date': start_date
                }).fetchall()
                realized_by_date = {str(r[0]): float(r[1] or 0) for r in realized_rows}

                chart_data = []
                cumulative_pnl = 0
                for row in rows:
                    snap_date, daily_pnl = row
                    date_str = snap_date.isoformat() if hasattr(snap_date, 'isoformat') else str(snap_date)
                    day_total = float(daily_pnl or 0) + realized_by_date.get(date_str, 0)
                    cumulative_pnl += day_total
                    chart_data.append({
                        'date': date_str,
                        'pnl': round(day_total, 2),
                        'cumulative_pnl': round(cumulative_pnl, 2),
                    })

            return {
                'success': True,
                'data': chart_data,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Paper trading P&L chart error: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }

    def get_performance_metrics(self, user_id: int, period: str = '1M') -> Dict[str, Any]:
        """Aggregate win rate, total P&L from order_performance."""
        try:
            period_days_map = {
                '1D': 1, '1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365
            }
            period_days = period_days_map.get(period, 30)
            start_date = datetime.now() - timedelta(days=period_days)

            db_manager = self._get_db_manager()

            with db_manager.get_session() as session:
                # All orders in period - win rate only from closed positions
                query = text("""
                    SELECT
                        COUNT(*) as total_orders,
                        COUNT(*) FILTER (WHERE is_active = false) as closed_orders,
                        COUNT(*) FILTER (WHERE is_active = false AND realized_pnl > 0) as winning_orders,
                        COALESCE(SUM(CASE WHEN is_active THEN unrealized_pnl ELSE 0 END), 0) as unrealized,
                        COALESCE(SUM(CASE WHEN NOT is_active THEN realized_pnl ELSE 0 END), 0) as realized,
                        COALESCE(MAX(CASE WHEN NOT is_active THEN realized_pnl END), 0) as best_trade,
                        COALESCE(MIN(CASE WHEN NOT is_active THEN realized_pnl END), 0) as worst_trade,
                        COUNT(*) FILTER (WHERE is_active = true) as active_count
                    FROM order_performance
                    WHERE user_id = :user_id AND created_at >= :start_date
                """)
                row = session.execute(query, {
                    'user_id': user_id,
                    'start_date': start_date
                }).fetchone()

                total_orders = int(row[0] or 0)
                closed_orders = int(row[1] or 0)
                winning_orders = int(row[2] or 0)
                unrealized = float(row[3] or 0)
                realized = float(row[4] or 0)
                best_trade = float(row[5] or 0)
                worst_trade = float(row[6] or 0)
                active_count = int(row[7] or 0)

                total_pnl = unrealized + realized
                win_rate = (winning_orders / closed_orders * 100) if closed_orders > 0 else 0
                return_percent = (total_pnl / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL > 0 else 0

            performance_data = {
                'return_percent': round(return_percent, 2),
                'annualized_return': 0.0,
                'total_pnl': round(total_pnl, 2),
                'portfolio_value': round(INITIAL_CAPITAL + total_pnl, 2),
                'period': period,
                'period_days': period_days,
                'win_rate': round(win_rate, 1),
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'best_day': round(best_trade, 2),
                'worst_day': round(worst_trade, 2),
                'total_trading_days': total_orders,
                'active_positions': active_count,
                'chart_data': [],
            }

            return {
                'success': True,
                'data': performance_data,
                'message': f'Paper trading performance ({period})',
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Paper trading performance metrics error: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': {
                    'return_percent': 0, 'win_rate': 0, 'total_pnl': 0,
                    'portfolio_value': INITIAL_CAPITAL, 'period': period,
                    'period_days': 30, 'sharpe_ratio': 0, 'max_drawdown': 0,
                    'volatility': 0, 'best_day': 0, 'worst_day': 0,
                    'total_trading_days': 0, 'chart_data': []
                },
                'last_updated': datetime.now().isoformat()
            }

    def get_watchlist_quotes(self, user_id: int, symbols: List[str] = None) -> Dict[str, Any]:
        """Get quotes from stocks table current prices."""
        try:
            db_manager = self._get_db_manager()

            with db_manager.get_session() as session:
                if symbols:
                    # Get specific symbols from stocks table
                    placeholders = ', '.join([f':sym_{i}' for i in range(len(symbols))])
                    query = text(f"""
                        SELECT symbol, name, current_price, volume
                        FROM stocks
                        WHERE symbol IN ({placeholders})
                        AND current_price IS NOT NULL
                    """)
                    params = {f'sym_{i}': s for i, s in enumerate(symbols)}
                else:
                    # Get active positions' symbols + top suggested stocks
                    query = text("""
                        (SELECT s.symbol, s.name, s.current_price, s.volume
                         FROM stocks s
                         JOIN order_performance op ON s.symbol = op.symbol
                         WHERE op.user_id = :user_id AND op.is_active = true
                         AND s.current_price IS NOT NULL)
                        UNION
                        (SELECT ds.symbol, ds.stock_name as name, ds.current_price, 0 as volume
                         FROM daily_suggested_stocks ds
                         WHERE ds.date = (SELECT MAX(date) FROM daily_suggested_stocks)
                         AND ds.current_price IS NOT NULL
                         ORDER BY ds.selection_score DESC
                         LIMIT 5)
                        LIMIT 10
                    """)
                    params = {'user_id': user_id}

                rows = session.execute(query, params).fetchall()

                quotes = []
                for row in rows:
                    symbol, name, price, volume = row
                    quotes.append({
                        'symbol': symbol or '',
                        'symbol_name': name or (symbol or '').replace('NSE:', '').replace('-EQ', ''),
                        'price': float(price or 0),
                        'change': 0.0,
                        'change_percent': 0.0,
                        'volume': int(volume or 0),
                        'high': 0.0,
                        'low': 0.0,
                        'open': 0.0,
                        'prev_close': 0.0,
                    })

            return {
                'success': True,
                'data': quotes,
                'message': 'Prices from last market data update',
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Paper trading watchlist quotes error: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }
