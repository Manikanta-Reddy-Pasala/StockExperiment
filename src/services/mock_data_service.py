"""
Mock Data Service

Provides realistic mock data for the trading dashboard when real broker APIs are not available.
This ensures the dashboard always shows meaningful data for demonstration purposes.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json


class MockDataService:
    """Service to provide realistic mock data for trading dashboard."""
    
    def __init__(self):
        self.market_symbols = [
            'NSE:NIFTY50-INDEX',
            'NSE:SENSEX-INDEX', 
            'NSE:NIFTYBANK-INDEX',
            'NSE:NIFTYIT-INDEX'
        ]
        
        self.stock_symbols = [
            'NSE:RELIANCE-EQ',
            'NSE:TCS-EQ',
            'NSE:INFY-EQ',
            'NSE:HDFCBANK-EQ',
            'NSE:ICICIBANK-EQ',
            'NSE:KOTAKBANK-EQ',
            'NSE:LT-EQ',
            'NSE:SBIN-EQ',
            'NSE:BHARTIARTL-EQ',
            'NSE:ASIANPAINT-EQ'
        ]
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Generate mock market overview data."""
        indices = []
        base_prices = {
            'NSE:NIFTY50-INDEX': 19650,
            'NSE:SENSEX-INDEX': 65800,
            'NSE:NIFTYBANK-INDEX': 44500,
            'NSE:NIFTYIT-INDEX': 28900
        }
        
        names = {
            'NSE:NIFTY50-INDEX': 'NIFTY 50',
            'NSE:SENSEX-INDEX': 'SENSEX',
            'NSE:NIFTYBANK-INDEX': 'BANK NIFTY',
            'NSE:NIFTYIT-INDEX': 'NIFTY IT'
        }
        
        for symbol in self.market_symbols:
            base = base_prices[symbol]
            change_percent = random.uniform(-2.5, 2.5)
            change = (base * change_percent) / 100
            current_price = base + change
            
            indices.append({
                'symbol': symbol,
                'name': names[symbol],
                'price': round(current_price, 2),
                'change': round(change, 2),
                'change_percent': round(change_percent, 2)
            })
        
        return {
            'success': True,
            'data': indices,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Generate mock portfolio summary."""
        total_value = random.uniform(450000, 550000)
        invested_value = random.uniform(400000, 480000)
        pnl = total_value - invested_value
        pnl_percent = (pnl / invested_value) * 100
        
        return {
            'success': True,
            'data': {
                'total_value': round(total_value, 2),
                'invested_value': round(invested_value, 2),
                'pnl': round(pnl, 2),
                'pnl_percent': round(pnl_percent, 2),
                'day_change': round(random.uniform(-5000, 8000), 2),
                'day_change_percent': round(random.uniform(-1.2, 1.8), 2)
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def get_portfolio_holdings(self) -> Dict[str, Any]:
        """Generate mock portfolio holdings."""
        holdings = []
        
        for i, symbol in enumerate(self.stock_symbols[:6]):  # Show 6 holdings
            quantity = random.randint(10, 100)
            avg_price = random.uniform(500, 2500)
            current_price = avg_price * random.uniform(0.85, 1.25)
            invested = quantity * avg_price
            current_value = quantity * current_price
            pnl = current_value - invested
            pnl_percent = (pnl / invested) * 100
            
            holdings.append({
                'symbol': symbol,
                'name': symbol.split(':')[1].replace('-EQ', ''),
                'quantity': quantity,
                'avg_price': round(avg_price, 2),
                'current_price': round(current_price, 2),
                'invested_value': round(invested, 2),
                'current_value': round(current_value, 2),
                'pnl': round(pnl, 2),
                'pnl_percent': round(pnl_percent, 2)
            })
        
        return {
            'success': True,
            'data': holdings,
            'total_holdings': len(holdings),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_recent_orders(self, limit: int = 5) -> Dict[str, Any]:
        """Generate mock recent orders."""
        orders = []
        
        for i in range(limit):
            symbol = random.choice(self.stock_symbols)
            side = random.choice(['BUY', 'SELL'])
            quantity = random.randint(5, 50)
            price = random.uniform(500, 2500)
            status = random.choice(['COMPLETED', 'PENDING', 'CANCELLED'])
            
            order_time = datetime.now() - timedelta(
                hours=random.randint(1, 48),
                minutes=random.randint(0, 59)
            )
            
            orders.append({
                'id': f'ORD{random.randint(100000, 999999)}',
                'symbol': symbol,
                'name': symbol.split(':')[1].replace('-EQ', ''),
                'side': side,
                'quantity': quantity,
                'price': round(price, 2),
                'status': status,
                'order_time': order_time.strftime('%Y-%m-%d %H:%M:%S'),
                'order_type': random.choice(['MARKET', 'LIMIT'])
            })
        
        return {
            'success': True,
            'data': orders,
            'total_orders': len(orders),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_pending_orders(self) -> Dict[str, Any]:
        """Generate mock pending orders."""
        orders = []
        
        for i in range(random.randint(1, 4)):
            symbol = random.choice(self.stock_symbols)
            side = random.choice(['BUY', 'SELL'])
            quantity = random.randint(5, 50)
            price = random.uniform(500, 2500)
            
            order_time = datetime.now() - timedelta(
                hours=random.randint(1, 24),
                minutes=random.randint(0, 59)
            )
            
            orders.append({
                'id': f'ORD{random.randint(100000, 999999)}',
                'symbol': symbol,
                'name': symbol.split(':')[1].replace('-EQ', ''),
                'side': side,
                'quantity': quantity,
                'price': round(price, 2),
                'status': 'PENDING',
                'order_time': order_time.strftime('%Y-%m-%d %H:%M:%S'),
                'order_type': 'LIMIT'
            })
        
        return {
            'success': True,
            'data': orders,
            'count': len(orders),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_portfolio_performance(self, period: str = '1W') -> Dict[str, Any]:
        """Generate mock portfolio performance data."""
        days = {'1D': 1, '1W': 7, '1M': 30, '3M': 90, '1Y': 365}.get(period, 7)
        
        performance_data = []
        base_value = 500000
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i-1)
            daily_change = random.uniform(-0.02, 0.02)  # -2% to +2% daily
            base_value *= (1 + daily_change)
            
            performance_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': round(base_value, 2),
                'change': round(base_value * daily_change, 2),
                'change_percent': round(daily_change * 100, 2)
            })
        
        return {
            'success': True,
            'data': performance_data,
            'period': period,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_suggested_stocks(self) -> Dict[str, Any]:
        """Generate mock suggested stocks."""
        suggestions = []
        
        for symbol in random.sample(self.stock_symbols, 5):
            current_price = random.uniform(500, 2500)
            target_price = current_price * random.uniform(1.05, 1.25)
            stop_loss = current_price * random.uniform(0.85, 0.95)
            
            suggestions.append({
                'symbol': symbol,
                'name': symbol.split(':')[1].replace('-EQ', ''),
                'current_price': round(current_price, 2),
                'target_price': round(target_price, 2),
                'stop_loss': round(stop_loss, 2),
                'recommendation': random.choice(['BUY', 'STRONG BUY', 'HOLD']),
                'strategy': random.choice(['Growth', 'Value', 'Momentum']),
                'reason': 'Strong fundamentals and technical indicators',
                'confidence': random.randint(70, 95)
            })
        
        return {
            'success': True,
            'data': suggestions,
            'total': len(suggestions),
            'last_updated': datetime.now().isoformat()
        }


# Global instance
_mock_data_service = None

def get_mock_data_service():
    """Get the global mock data service instance."""
    global _mock_data_service
    if _mock_data_service is None:
        _mock_data_service = MockDataService()
    return _mock_data_service
