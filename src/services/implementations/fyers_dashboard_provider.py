"""
FYERS Dashboard Provider Implementation

Implements the IDashboardProvider interface for FYERS broker.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ..interfaces.dashboard_interface import IDashboardProvider, DashboardMetrics, MarketIndex
from ..broker_service import get_broker_service

logger = logging.getLogger(__name__)


class FyersDashboardProvider(IDashboardProvider):
    """FYERS implementation of dashboard provider."""
    
    def __init__(self):
        self.broker_service = get_broker_service()
    
    def get_market_overview(self, user_id: int) -> Dict[str, Any]:
        """Get market overview data for major indices using FYERS API."""
        try:
            # Default symbols for major Indian indices
            symbols = 'NSE:NIFTY50-INDEX,NSE:SENSEX-INDEX,NSE:NIFTYBANK-INDEX,NSE:NIFTYIT-INDEX'
            
            quotes_data = self.broker_service.get_fyers_quotes(user_id, symbols)
            
            if not quotes_data.get('success'):
                return {
                    'success': False,
                    'error': quotes_data.get('error', 'Failed to fetch market data'),
                    'data': [],
                    'last_updated': datetime.now().isoformat()
                }
            
            market_indices = []
            symbol_mapping = {
                'NSE:NIFTY50-INDEX': 'NIFTY 50',
                'NSE:SENSEX-INDEX': 'SENSEX',
                'NSE:NIFTYBANK-INDEX': 'BANK NIFTY',
                'NSE:NIFTYIT-INDEX': 'NIFTY IT'
            }
            
            for symbol, quote in quotes_data['data'].items():
                if quote.get('v'):
                    quote_data = quote['v']
                    index = MarketIndex(
                        symbol=symbol,
                        name=symbol_mapping.get(symbol, symbol),
                        price=quote_data.get('lp', 0),
                        change=quote_data.get('ch', 0),
                        change_percent=quote_data.get('chp', 0)
                    )
                    market_indices.append(index.to_dict())
            
            return {
                'success': True,
                'data': market_indices,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching market overview for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }
    
    def get_portfolio_summary(self, user_id: int) -> Dict[str, Any]:
        """Get portfolio summary metrics using FYERS API."""
        try:
            # Get holdings and positions
            holdings_data = self.broker_service.get_fyers_holdings(user_id)
            positions_data = self.broker_service.get_fyers_positions(user_id)
            funds_data = self.broker_service.get_fyers_funds(user_id)
            
            metrics = DashboardMetrics()
            
            # Process holdings
            if holdings_data.get('success') and holdings_data.get('data'):
                holdings = holdings_data['data'].get('holdings', [])
                metrics.holdings_count = len(holdings)
                
                for holding in holdings:
                    current_value = holding.get('marketVal', 0)
                    metrics.total_portfolio_value += current_value
                    
                    pnl = holding.get('pl', 0)
                    metrics.total_pnl += pnl
            
            # Process positions
            if positions_data.get('success') and positions_data.get('data'):
                positions = positions_data['data'].get('netPositions', [])
                metrics.positions_count = len(positions)
                
                for position in positions:
                    pnl = position.get('unrealizedProfit', 0)
                    metrics.total_pnl += pnl
            
            # Process funds
            if funds_data.get('success') and funds_data.get('data'):
                funds = funds_data['data'].get('fund_limit', [])
                for fund in funds:
                    if fund.get('title') == 'Available Cash':
                        metrics.available_cash = fund.get('equityAmount', 0)
                        break
            
            return {
                'success': True,
                'data': metrics.to_dict(),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching portfolio summary for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': DashboardMetrics().to_dict(),
                'last_updated': datetime.now().isoformat()
            }
    
    def get_top_holdings(self, user_id: int, limit: int = 5) -> Dict[str, Any]:
        """Get top holdings by value using FYERS API."""
        try:
            holdings_data = self.broker_service.get_fyers_holdings(user_id)
            
            if not holdings_data.get('success'):
                return {
                    'success': False,
                    'error': holdings_data.get('error', 'Failed to fetch holdings'),
                    'data': [],
                    'last_updated': datetime.now().isoformat()
                }
            
            holdings = holdings_data['data'].get('holdings', [])
            
            # Sort by market value and take top holdings
            sorted_holdings = sorted(holdings, key=lambda x: x.get('marketVal', 0), reverse=True)
            top_holdings = sorted_holdings[:limit]
            
            processed_holdings = []
            for holding in top_holdings:
                processed_holdings.append({
                    'symbol': holding.get('symbol', ''),
                    'quantity': holding.get('qty', 0),
                    'current_value': holding.get('marketVal', 0),
                    'pnl': holding.get('pl', 0),
                    'pnl_percent': holding.get('plPercent', 0),
                    'current_price': holding.get('ltp', 0)
                })
            
            return {
                'success': True,
                'data': processed_holdings,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching top holdings for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }
    
    def get_recent_activity(self, user_id: int, limit: int = 10) -> Dict[str, Any]:
        """Get recent trading activity using FYERS API."""
        try:
            orderbook_data = self.broker_service.get_fyers_orderbook(user_id)
            
            if not orderbook_data.get('success'):
                return {
                    'success': False,
                    'error': orderbook_data.get('error', 'Failed to fetch orders'),
                    'data': [],
                    'last_updated': datetime.now().isoformat()
                }
            
            orders = orderbook_data['data'].get('orderBook', [])
            
            # Sort by order time and take recent orders
            sorted_orders = sorted(orders, key=lambda x: x.get('orderDateTime', ''), reverse=True)
            recent_orders = sorted_orders[:limit]
            
            processed_activities = []
            for order in recent_orders:
                processed_activities.append({
                    'type': 'order',
                    'symbol': order.get('symbol', ''),
                    'side': order.get('side', ''),
                    'status': order.get('status', ''),
                    'quantity': order.get('qty', 0),
                    'price': order.get('limitPrice', 0),
                    'timestamp': order.get('orderDateTime', '')
                })
            
            return {
                'success': True,
                'data': processed_activities,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching recent activity for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }
    
    def get_account_balance(self, user_id: int) -> Dict[str, Any]:
        """Get account balance and available funds using FYERS API."""
        try:
            funds_data = self.broker_service.get_fyers_funds(user_id)
            
            if not funds_data.get('success'):
                return {
                    'success': False,
                    'error': funds_data.get('error', 'Failed to fetch funds'),
                    'data': {'available_cash': 0, 'total_balance': 0, 'margin_used': 0},
                    'last_updated': datetime.now().isoformat()
                }
            
            funds = funds_data['data'].get('fund_limit', [])
            balance_data = {
                'available_cash': 0,
                'total_balance': 0,
                'margin_used': 0
            }
            
            for fund in funds:
                if fund.get('title') == 'Available Cash':
                    balance_data['available_cash'] = fund.get('equityAmount', 0)
                elif fund.get('title') == 'Total Balance':
                    balance_data['total_balance'] = fund.get('equityAmount', 0)
                elif fund.get('title') == 'Margin Used':
                    balance_data['margin_used'] = fund.get('equityAmount', 0)
            
            return {
                'success': True,
                'data': balance_data,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching account balance for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': {'available_cash': 0, 'total_balance': 0, 'margin_used': 0},
                'last_updated': datetime.now().isoformat()
            }
    
    def get_daily_pnl_chart_data(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get daily P&L data for charting (simulated for FYERS)."""
        try:
            # FYERS doesn't provide historical P&L data directly
            # This is a simplified implementation that could be enhanced
            # by storing daily P&L snapshots in the database
            
            chart_data = []
            base_date = datetime.now() - timedelta(days=days)
            
            for i in range(days):
                date = base_date + timedelta(days=i)
                # This would need to be replaced with actual historical P&L calculation
                # For now, returning empty data structure
                chart_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'pnl': 0  # Would be calculated from historical data
                })
            
            return {
                'success': True,
                'data': chart_data,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching daily P&L chart data for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }
    
    def get_performance_metrics(self, user_id: int, period: str = '1M') -> Dict[str, Any]:
        """Get performance metrics for a given period (simulated for FYERS)."""
        try:
            # This would need historical data tracking for accurate calculation
            # For now, providing basic structure
            
            performance_data = {
                'return_percent': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'period': period
            }
            
            return {
                'success': True,
                'data': performance_data,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching performance metrics for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': {},
                'last_updated': datetime.now().isoformat()
            }
    
    def get_watchlist_quotes(self, user_id: int, symbols: List[str] = None) -> Dict[str, Any]:
        """Get real-time quotes for watchlist symbols using FYERS API."""
        try:
            if not symbols:
                # Default watchlist symbols
                symbols = ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:HDFCBANK-EQ', 'NSE:INFY-EQ', 'NSE:ICICIBANK-EQ']
            
            symbols_str = ','.join(symbols)
            quotes_data = self.broker_service.get_fyers_quotes(user_id, symbols_str)
            
            if not quotes_data.get('success'):
                return {
                    'success': False,
                    'error': quotes_data.get('error', 'Failed to fetch quotes'),
                    'data': [],
                    'last_updated': datetime.now().isoformat()
                }
            
            watchlist_quotes = []
            for symbol, quote in quotes_data['data'].items():
                if quote.get('v'):
                    quote_data = quote['v']
                    watchlist_quotes.append({
                        'symbol': symbol,
                        'price': quote_data.get('lp', 0),
                        'change': quote_data.get('ch', 0),
                        'change_percent': quote_data.get('chp', 0),
                        'volume': quote_data.get('volume', 0),
                        'high': quote_data.get('h', 0),
                        'low': quote_data.get('l', 0)
                    })
            
            return {
                'success': True,
                'data': watchlist_quotes,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching watchlist quotes for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }
