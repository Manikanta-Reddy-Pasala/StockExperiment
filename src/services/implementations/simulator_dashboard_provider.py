"""
Simulator Dashboard Provider - Paper Trading Implementation

This provider simulates trading operations for testing and demo purposes.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random
from ..interfaces.dashboard_interface import IDashboardProvider, DashboardMetrics, MarketIndex


class SimulatorDashboardProvider(IDashboardProvider):
    """Simulator implementation with mock data for testing."""
    
    def get_market_overview(self, user_id: int) -> Dict[str, Any]:
        """Get simulated market overview data."""
        market_indices = [
            MarketIndex('NSE:NIFTY50-INDEX', 'NIFTY 50', 21500.0 + random.uniform(-100, 100), 
                       random.uniform(-50, 50), random.uniform(-0.5, 0.5)),
            MarketIndex('NSE:SENSEX-INDEX', 'SENSEX', 71000.0 + random.uniform(-200, 200), 
                       random.uniform(-100, 100), random.uniform(-0.5, 0.5)),
            MarketIndex('NSE:NIFTYBANK-INDEX', 'BANK NIFTY', 46000.0 + random.uniform(-300, 300), 
                       random.uniform(-150, 150), random.uniform(-0.8, 0.8)),
            MarketIndex('NSE:NIFTYIT-INDEX', 'NIFTY IT', 31000.0 + random.uniform(-500, 500), 
                       random.uniform(-200, 200), random.uniform(-1.0, 1.0))
        ]
        
        return {
            'success': True,
            'data': [index.to_dict() for index in market_indices],
            'last_updated': datetime.now().isoformat()
        }
    
    def get_portfolio_summary(self, user_id: int) -> Dict[str, Any]:
        """Get simulated portfolio summary."""
        metrics = DashboardMetrics()
        metrics.total_pnl = random.uniform(-5000, 15000)
        metrics.total_portfolio_value = random.uniform(50000, 200000)
        metrics.available_cash = random.uniform(10000, 50000)
        metrics.holdings_count = random.randint(5, 15)
        metrics.positions_count = random.randint(0, 5)
        metrics.daily_pnl = random.uniform(-2000, 3000)
        metrics.daily_pnl_percent = (metrics.daily_pnl / metrics.total_portfolio_value) * 100
        
        return {
            'success': True,
            'data': metrics.to_dict(),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_top_holdings(self, user_id: int, limit: int = 5) -> Dict[str, Any]:
        """Get simulated top holdings."""
        sample_holdings = [
            {'symbol': 'NSE:RELIANCE-EQ', 'quantity': 10, 'current_value': 25000, 'pnl': 1200},
            {'symbol': 'NSE:TCS-EQ', 'quantity': 5, 'current_value': 18000, 'pnl': -800},
            {'symbol': 'NSE:HDFCBANK-EQ', 'quantity': 15, 'current_value': 24750, 'pnl': 2100},
            {'symbol': 'NSE:INFY-EQ', 'quantity': 20, 'current_value': 29600, 'pnl': 1800},
            {'symbol': 'NSE:ICICIBANK-EQ', 'quantity': 25, 'current_value': 23750, 'pnl': -500}
        ]
        
        return {
            'success': True,
            'data': sample_holdings[:limit],
            'last_updated': datetime.now().isoformat()
        }
    
    def get_recent_activity(self, user_id: int, limit: int = 10) -> Dict[str, Any]:
        """Get simulated recent activity."""
        activities = []
        for i in range(limit):
            activities.append({
                'type': 'order',
                'symbol': f'NSE:STOCK{i+1}-EQ',
                'side': random.choice(['BUY', 'SELL']),
                'status': random.choice(['COMPLETE', 'PENDING', 'CANCELLED']),
                'quantity': random.randint(1, 50),
                'price': round(random.uniform(100, 3000), 2),
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat()
            })
        
        return {
            'success': True,
            'data': activities,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_account_balance(self, user_id: int) -> Dict[str, Any]:
        """Get simulated account balance."""
        return {
            'success': True,
            'data': {
                'available_cash': round(random.uniform(10000, 50000), 2),
                'total_balance': round(random.uniform(100000, 300000), 2),
                'margin_used': round(random.uniform(0, 20000), 2)
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def get_daily_pnl_chart_data(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get simulated daily P&L chart data."""
        chart_data = []
        base_pnl = 0
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i-1)
            daily_change = random.uniform(-1000, 1500)
            base_pnl += daily_change
            chart_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'pnl': round(base_pnl, 2)
            })
        
        return {
            'success': True,
            'data': chart_data,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_performance_metrics(self, user_id: int, period: str = '1M') -> Dict[str, Any]:
        """Get simulated performance metrics."""
        return {
            'success': True,
            'data': {
                'return_percent': round(random.uniform(-5, 15), 2),
                'win_rate': round(random.uniform(45, 75), 2),
                'sharpe_ratio': round(random.uniform(0.5, 2.5), 2),
                'max_drawdown': round(random.uniform(-15, -2), 2),
                'period': period
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def get_watchlist_quotes(self, user_id: int, symbols: List[str] = None) -> Dict[str, Any]:
        """Get simulated watchlist quotes."""
        if not symbols:
            symbols = ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:HDFCBANK-EQ', 'NSE:INFY-EQ', 'NSE:ICICIBANK-EQ']
        
        quotes = []
        for symbol in symbols:
            base_price = random.uniform(500, 3000)
            change = random.uniform(-50, 50)
            quotes.append({
                'symbol': symbol,
                'price': round(base_price, 2),
                'change': round(change, 2),
                'change_percent': round((change / base_price) * 100, 2),
                'volume': random.randint(10000, 1000000),
                'high': round(base_price + random.uniform(0, 30), 2),
                'low': round(base_price - random.uniform(0, 30), 2)
            })
        
        return {
            'success': True,
            'data': quotes,
            'last_updated': datetime.now().isoformat()
        }
