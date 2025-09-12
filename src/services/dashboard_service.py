"""
Service for dashboard-related logic.
"""
from datetime import datetime
from .broker_service import get_broker_service

class DashboardService:
    def __init__(self, broker_service):
        self.broker_service = broker_service

    def get_dashboard_metrics(self, user_id: int):
        """Get dashboard metrics using FYERS API."""
        config = self.broker_service.get_broker_config('fyers', user_id)
        if not config or not config.get('is_connected'):
            raise ValueError('FYERS not connected. Please configure your broker connection.')

        # Get user profile, funds, and holdings
        profile_data = self.broker_service.get_fyers_profile(user_id)
        funds_data = self.broker_service.get_fyers_funds(user_id)
        holdings_data = self.broker_service.get_fyers_holdings(user_id)
        positions_data = self.broker_service.get_fyers_positions(user_id)

        # Calculate total P&L from positions
        total_pnl = 0
        if positions_data.get('success') and positions_data.get('data'):
            for position in positions_data['data']:
                pnl = position.get('pl', 0)
                total_pnl += float(pnl) if pnl else 0

        # Get available funds
        available_funds = 0
        if funds_data.get('success') and funds_data.get('data'):
            fund_limits = funds_data['data'].get('fund_limit', [])
            for fund in fund_limits:
                if fund.get('equity_amount'):
                    available_funds += float(fund['equity_amount'])

        # Get total portfolio value from holdings
        total_portfolio_value = 0
        if holdings_data.get('success') and holdings_data.get('data'):
            holdings = holdings_data['data'].get('holdings', [])
            for holding in holdings:
                if holding.get('market_value'):
                    total_portfolio_value += float(holding['market_value'])

        # Get market quotes for major indices
        market_quotes = self.broker_service.get_fyers_quotes(user_id, "NSE:NIFTY50-INDEX,NSE:SENSEX-INDEX,NSE:NIFTYBANK-INDEX,NSE:NIFTYIT-INDEX")

        # Process market data
        market_data = {}
        if market_quotes.get('success') and market_quotes.get('data'):
            for symbol, quote in market_quotes['data'].items():
                if quote.get('v'):
                    market_data[symbol] = {
                        'price': quote['v'].get('lp', 0),
                        'change': quote['v'].get('ch', 0),
                        'change_percent': quote['v'].get('chp', 0)
                    }

        return {
            'total_pnl': total_pnl,
            'available_funds': available_funds,
            'total_portfolio_value': total_portfolio_value,
            'market_data': market_data,
            'profile': profile_data.get('data', {}),
            'last_updated': datetime.now().isoformat()
        }

_dashboard_service = None

def get_dashboard_service():
    """Singleton factory for DashboardService."""
    global _dashboard_service
    if _dashboard_service is None:
        broker_service = get_broker_service()
        _dashboard_service = DashboardService(broker_service)
    return _dashboard_service
