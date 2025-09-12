"""
Service for portfolio-related logic.
"""
from datetime import datetime
from .broker_service import get_broker_service

class PortfolioService:
    def __init__(self, broker_service):
        self.broker_service = broker_service

    def get_portfolio_holdings(self, user_id: int):
        """Get portfolio holdings using FYERS API."""
        holdings_data = self.broker_service.get_fyers_holdings(user_id)

        if holdings_data.get('success') and holdings_data.get('data'):
            holdings = holdings_data['data'].get('holdings', [])

            processed_holdings = []
            for holding in holdings:
                processed_holdings.append({
                    'symbol': holding.get('symbol', ''),
                    'quantity': holding.get('quantity', 0),
                    'average_price': holding.get('average_price', 0),
                    'market_value': holding.get('market_value', 0),
                    'pnl': holding.get('pnl', 0),
                    'pnl_percent': holding.get('pnl_percent', 0),
                    'ltp': holding.get('ltp', 0)
                })

            return {
                'success': True,
                'data': processed_holdings,
                'last_updated': datetime.now().isoformat()
            }
        else:
            return {
                'success': False,
                'error': 'Failed to fetch holdings data from FYERS'
            }

    def get_portfolio_positions(self, user_id: int):
        """Get portfolio positions using FYERS API."""
        positions_data = self.broker_service.get_fyers_positions(user_id)

        if positions_data.get('success') and positions_data.get('data'):
            positions = positions_data['data']

            processed_positions = []
            for position in positions:
                processed_positions.append({
                    'symbol': position.get('symbol', ''),
                    'quantity': position.get('netQty', 0),
                    'average_price': position.get('avgPrice', 0),
                    'ltp': position.get('ltp', 0),
                    'pnl': position.get('pl', 0),
                    'pnl_percent': position.get('plPercent', 0),
                    'side': position.get('side', ''),
                    'product': position.get('product', '')
                })

            return {
                'success': True,
                'data': processed_positions,
                'last_updated': datetime.now().isoformat()
            }
        else:
            return {
                'success': False,
                'error': 'Failed to fetch positions data from FYERS'
            }

_portfolio_service = None

def get_portfolio_service():
    """Singleton factory for PortfolioService."""
    global _portfolio_service
    if _portfolio_service is None:
        broker_service = get_broker_service()
        _portfolio_service = PortfolioService(broker_service)
    return _portfolio_service
