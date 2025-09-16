"""
Reports Sync Service
Handles fetching, caching, and synchronizing trade data for reports functionality.
Implements database + Redis dual-layer caching similar to portfolio sync service.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from ..models.database import DatabaseManager
from ..models.models import Trade, User
from ..services.cache_service import get_cache_service
from ..services.unified_broker_service import get_unified_broker_service
import re

logger = logging.getLogger(__name__)

class ReportsSyncService:
    """Service for synchronizing and managing trade reports data with dual-layer caching."""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.cache_service = get_cache_service()
        self.unified_broker_service = get_unified_broker_service()
        self.cache_duration = 300  # 5 minutes cache

    def get_reports_data(self, user_id: int, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive reports data with intelligent caching.
        Returns reports data including P&L, trades, performance metrics, and top/worst performers.
        """
        try:
            # Check cache first unless force refresh
            if not force_refresh and self._should_use_cache(user_id):
                cached_reports = self._get_cached_reports(user_id)
                if cached_reports is not None:
                    logger.info(f"Returning cached reports data for user {user_id}")
                    return cached_reports

            logger.info(f"Fetching fresh reports data for user {user_id}")

            # Get trade data from database first
            db_trades = self._get_db_trades(user_id)

            # Sync with broker data and calculate reports
            reports_data = self._sync_reports_from_broker(user_id, db_trades)

            # Cache the results
            self._cache_reports(user_id, reports_data)

            return reports_data

        except Exception as e:
            logger.error(f"Error getting reports data: {e}")
            return self._get_empty_reports_data()

    def _should_use_cache(self, user_id: int) -> bool:
        """Check if cached data should be used."""
        cache_key = f"reports_last_update_{user_id}"
        last_update = self.cache_service.get(cache_key)

        if last_update:
            last_update_time = datetime.fromisoformat(last_update)
            return (datetime.now() - last_update_time).seconds < self.cache_duration

        return False

    def _get_cached_reports(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get cached reports data."""
        try:
            cache_key = f"reports_data_{user_id}"
            cached_data = self.cache_service.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Error getting cached reports: {e}")
        return None

    def _cache_reports(self, user_id: int, reports_data: Dict[str, Any]):
        """Cache reports data."""
        try:
            cache_key = f"reports_data_{user_id}"
            last_update_key = f"reports_last_update_{user_id}"

            self.cache_service.set(cache_key, json.dumps(reports_data), self.cache_duration)
            self.cache_service.set(last_update_key, datetime.now().isoformat(), self.cache_duration)

            logger.info(f"Cached reports data for user {user_id}")
        except Exception as e:
            logger.error(f"Error caching reports: {e}")

    def _get_db_trades(self, user_id: int) -> List[Dict[str, Any]]:
        """Get existing trade data from database."""
        try:
            with self.db_manager.get_session() as db_session:
                trades = db_session.query(Trade).filter(Trade.user_id == user_id).all()

                trade_data = []
                for trade in trades:
                    trade_data.append({
                        'trade_id': trade.trade_id,
                        'order_id': trade.order_id,
                        'symbol': trade.tradingsymbol,
                        'exchange': trade.exchange,
                        'transaction_type': trade.transaction_type,
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'filled_quantity': trade.filled_quantity,
                        'trade_time': trade.trade_time,
                        'created_at': trade.created_at
                    })

                logger.info(f"Retrieved {len(trade_data)} trades from database for user {user_id}")
                return trade_data

        except Exception as e:
            logger.error(f"Error getting portfolio from database: {e}")
            return []

    def _sync_reports_from_broker(self, user_id: int, db_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sync reports data from broker and calculate metrics."""
        try:
            # Get fresh trade data from broker/extended file
            fresh_trades = self._get_fresh_trade_data(user_id)

            # Update database with new trades
            self._update_database_trades(user_id, fresh_trades)

            # Calculate comprehensive reports metrics
            reports_data = self._calculate_reports_metrics(fresh_trades)

            logger.info(f"Synced reports data for user {user_id}: {len(fresh_trades)} trades")
            return reports_data

        except Exception as e:
            logger.error(f"Error syncing reports from broker: {e}")
            return self._get_empty_reports_data()

    def _get_fresh_trade_data(self, user_id: int) -> List[Dict[str, Any]]:
        """Get fresh trade data from extended tradebook file."""
        try:
            # First try to get from actual broker API
            try:
                broker_trades = self.unified_broker_service.get_tradebook(user_id)
                if broker_trades and 'tradeBook' in broker_trades:
                    return self._process_broker_trades(broker_trades['tradeBook'])
            except Exception as e:
                logger.warning(f"Could not get trades from broker API: {e}")

            # Fallback to extended tradebook file
            with open('/Users/manip/Documents/codeRepo/poc/StockExperiment/raw_extended_tradebook.txt', 'r') as f:
                content = f.read()

            # Extract JSON from the file
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                trade_data = json.loads(json_match.group())
                if 'tradeBook' in trade_data:
                    return self._process_broker_trades(trade_data['tradeBook'])

            logger.warning("No trade data found in extended tradebook file")
            return []

        except Exception as e:
            logger.error(f"Error getting fresh trade data: {e}")
            return []

    def _process_broker_trades(self, broker_trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and normalize broker trade data."""
        processed_trades = []

        for trade in broker_trades:
            # Extract symbol name from NSE:SYMBOL-EQ format
            symbol = trade.get('symbol', '').split(':')[-1].replace('-EQ', '')

            # Determine transaction type from side (1 = BUY, -1 = SELL)
            side = trade.get('side', 1)
            transaction_type = 'BUY' if side == 1 else 'SELL'

            # Parse trade date
            trade_time = None
            if trade.get('orderDateTime'):
                try:
                    trade_time = datetime.strptime(trade['orderDateTime'], '%d-%b-%Y %H:%M:%S')
                except:
                    try:
                        trade_time = datetime.strptime(trade['orderDateTime'], '%d-%m-%Y %H:%M:%S')
                    except:
                        trade_time = datetime.now()

            processed_trade = {
                'trade_id': trade.get('tradeNumber', ''),
                'order_id': trade.get('orderNumber', ''),
                'exchange_order_id': trade.get('exchangeOrderNo', ''),
                'symbol': symbol,
                'exchange': 'NSE',
                'transaction_type': transaction_type,
                'quantity': trade.get('tradedQty', 0),
                'price': trade.get('tradePrice', 0.0),
                'filled_quantity': trade.get('tradedQty', 0),
                'trade_value': trade.get('tradeValue', 0.0),
                'trade_time': trade_time,
                'side': side,
                'product_type': trade.get('productType', 'CNC')
            }

            processed_trades.append(processed_trade)

        logger.info(f"Processed {len(processed_trades)} broker trades")
        return processed_trades

    def _update_database_trades(self, user_id: int, fresh_trades: List[Dict[str, Any]]):
        """Update database with fresh trade data."""
        try:
            with self.db_manager.get_session() as db_session:
                # Clear existing trades for this user
                db_session.query(Trade).filter(Trade.user_id == user_id).delete()

                # Add fresh trade data
                for trade_data in fresh_trades:
                    trade = Trade(
                        user_id=user_id,
                        trade_id=trade_data.get('trade_id', ''),
                        order_id=trade_data.get('order_id', ''),
                        exchange_order_id=trade_data.get('exchange_order_id', ''),
                        tradingsymbol=trade_data.get('symbol', ''),
                        exchange=trade_data.get('exchange', ''),
                        transaction_type=trade_data.get('transaction_type', ''),
                        quantity=trade_data.get('quantity', 0),
                        price=trade_data.get('price', 0.0),
                        filled_quantity=trade_data.get('filled_quantity', 0),
                        trade_time=trade_data.get('trade_time'),
                        created_at=datetime.now()
                    )
                    db_session.add(trade)

                db_session.commit()
                logger.info(f"Updated database with {len(fresh_trades)} trades for user {user_id}")

        except Exception as e:
            logger.error(f"Error updating database: {e}")

    def _calculate_reports_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive reports metrics from trade data."""
        try:
            if not trades:
                return self._get_empty_reports_data()

            # Calculate basic metrics
            total_trades = len(trades)

            # Group trades by symbol to calculate P&L
            symbol_trades = {}
            for trade in trades:
                symbol = trade['symbol']
                if symbol not in symbol_trades:
                    symbol_trades[symbol] = []
                symbol_trades[symbol].append(trade)

            # Calculate P&L for each symbol
            symbol_pnl = {}
            total_pnl = 0.0
            winning_trades = 0
            losing_trades = 0

            for symbol, symbol_trade_list in symbol_trades.items():
                # Calculate P&L for this symbol
                buy_qty = 0
                sell_qty = 0
                buy_value = 0.0
                sell_value = 0.0

                for trade in symbol_trade_list:
                    qty = trade['quantity']
                    value = trade['trade_value']

                    if trade['transaction_type'] == 'BUY':
                        buy_qty += qty
                        buy_value += value
                    else:
                        sell_qty += qty
                        sell_value += value

                # Calculate net P&L for this symbol
                if buy_qty > 0 and sell_qty > 0:
                    # Completed trades - calculate actual P&L
                    min_qty = min(buy_qty, sell_qty)
                    avg_buy_price = buy_value / buy_qty if buy_qty > 0 else 0
                    avg_sell_price = sell_value / sell_qty if sell_qty > 0 else 0

                    pnl = (avg_sell_price - avg_buy_price) * min_qty
                    symbol_pnl[symbol] = {
                        'pnl': pnl,
                        'quantity_traded': min_qty,
                        'avg_buy_price': avg_buy_price,
                        'avg_sell_price': avg_sell_price,
                        'return_pct': (pnl / (avg_buy_price * min_qty)) * 100 if avg_buy_price > 0 else 0
                    }

                    total_pnl += pnl
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1

            # Calculate win rate
            completed_trades = winning_trades + losing_trades
            win_rate = (winning_trades / completed_trades * 100) if completed_trades > 0 else 0

            # Get top and worst performers
            sorted_symbols = sorted(symbol_pnl.items(), key=lambda x: x[1]['pnl'], reverse=True)
            top_performers = sorted_symbols[:5]  # Top 5
            worst_performers = sorted_symbols[-3:] if len(sorted_symbols) > 3 else sorted_symbols  # Worst 3

            # Calculate performance summary by period
            performance_summary = self._calculate_performance_summary(trades, symbol_pnl)

            return {
                'summary': {
                    'total_pnl': total_pnl,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'completed_trades': completed_trades
                },
                'top_performers': [
                    {
                        'symbol': symbol,
                        'pnl': data['pnl'],
                        'return_pct': data['return_pct'],
                        'quantity': data['quantity_traded']
                    }
                    for symbol, data in top_performers
                ],
                'worst_performers': [
                    {
                        'symbol': symbol,
                        'pnl': data['pnl'],
                        'return_pct': data['return_pct'],
                        'quantity': data['quantity_traded']
                    }
                    for symbol, data in worst_performers
                ],
                'performance_summary': performance_summary,
                'symbol_pnl': symbol_pnl
            }

        except Exception as e:
            logger.error(f"Error calculating reports metrics: {e}")
            return self._get_empty_reports_data()

    def _calculate_performance_summary(self, trades: List[Dict[str, Any]], symbol_pnl: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate performance summary by time periods."""
        try:
            now = datetime.now()
            periods = [
                ('Today', now - timedelta(days=1)),
                ('This Week', now - timedelta(days=7)),
                ('This Month', now - timedelta(days=30)),
                ('All Time', datetime.min)
            ]

            performance = []

            for period_name, period_start in periods:
                # Filter trades for this period
                period_trades = [
                    trade for trade in trades
                    if trade.get('trade_time') and trade['trade_time'] >= period_start
                ]

                if not period_trades:
                    performance.append({
                        'period': period_name,
                        'pnl': 0.0,
                        'trades': 0,
                        'win_rate': 0.0,
                        'avg_win': 0.0,
                        'avg_loss': 0.0,
                        'max_drawdown': 0.0
                    })
                    continue

                # Calculate metrics for this period
                period_pnl = sum(symbol_pnl.get(trade['symbol'], {}).get('pnl', 0) for trade in period_trades)
                period_trade_count = len(period_trades)

                # Calculate win/loss metrics
                wins = [pnl for pnl in [symbol_pnl.get(trade['symbol'], {}).get('pnl', 0) for trade in period_trades] if pnl > 0]
                losses = [pnl for pnl in [symbol_pnl.get(trade['symbol'], {}).get('pnl', 0) for trade in period_trades] if pnl < 0]

                win_rate = (len(wins) / (len(wins) + len(losses)) * 100) if (len(wins) + len(losses)) > 0 else 0
                avg_win = sum(wins) / len(wins) if wins else 0
                avg_loss = sum(losses) / len(losses) if losses else 0

                # Simple max drawdown calculation
                max_drawdown = min(losses) if losses else 0

                performance.append({
                    'period': period_name,
                    'pnl': period_pnl,
                    'trades': period_trade_count,
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'max_drawdown': abs(max_drawdown)
                })

            return performance

        except Exception as e:
            logger.error(f"Error calculating performance summary: {e}")
            return []

    def _get_empty_reports_data(self) -> Dict[str, Any]:
        """Return empty reports data structure."""
        return {
            'summary': {
                'total_pnl': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'winning_trades': 0,
                'losing_trades': 0,
                'completed_trades': 0
            },
            'top_performers': [],
            'worst_performers': [],
            'performance_summary': [],
            'symbol_pnl': {}
        }

# Singleton pattern
_reports_sync_service = None

def get_reports_sync_service() -> ReportsSyncService:
    """Get the singleton reports sync service instance."""
    global _reports_sync_service
    if _reports_sync_service is None:
        _reports_sync_service = ReportsSyncService()
    return _reports_sync_service