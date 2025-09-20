"""
Service for portfolio-related logic.
"""
import logging
from datetime import datetime
from ..core.broker_service import get_broker_service

logger = logging.getLogger(__name__)

class PortfolioService:
    def __init__(self, broker_service):
        self.broker_service = broker_service

    def get_portfolio_holdings(self, user_id: int):
        """Get portfolio holdings using FYERS API."""
        print(f"DEBUG: get_portfolio_holdings called for user {user_id}")
        try:
            holdings_data = self.broker_service.get_fyers_holdings(user_id)
            print(f"DEBUG: holdings_data response: {holdings_data}")

            # Check if the response is successful (FYERS format: 's': 'ok')
            if holdings_data.get('s') == 'ok':
                holdings = holdings_data.get('holdings', [])

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
                error_msg = holdings_data.get('message', 'Unknown error')
                print(f"DEBUG: Error in holdings_data: {error_msg}")
                return {
                    'success': False,
                    'error': f'Failed to fetch holdings data from FYERS: {error_msg}'
                }
        except Exception as e:
            print(f"DEBUG: Exception in get_portfolio_holdings: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to fetch holdings data from FYERS: {str(e)}'
            }

    def get_portfolio_positions(self, user_id: int):
        """Get portfolio positions using FYERS API."""
        print(f"DEBUG: get_portfolio_positions called for user {user_id}")
        try:
            print("DEBUG: Calling broker_service.get_fyers_positions")
            positions_data = self.broker_service.get_fyers_positions(user_id)
            print(f"DEBUG: positions_data response: {positions_data}")
            logger.info(f"Portfolio positions response: {positions_data}")

            # Check if the response is successful (FYERS format: 's': 'ok')
            if positions_data.get('s') == 'ok':
                positions = positions_data.get('netPositions', [])
                print(f"DEBUG: Processing {len(positions)} positions")

                processed_positions = []
                for position in positions:
                    # Normalize FYERS fields and add defensive defaults
                    avg_price = position.get('average_price')
                    if avg_price in (None, 0):
                        avg_price = position.get('buyAvg', position.get('netAvg', 0))
                    quantity = position.get('quantity', position.get('netQty', position.get('qty', 0)))
                    pnl_val = position.get('pnl', position.get('pl', 0))
                    ltp_val = position.get('ltp', position.get('last_price', 0))
                    side_raw = position.get('side', '')
                    side = 'long' if side_raw == 1 or str(side_raw).lower() == 'long' else ('short' if side_raw == -1 else side_raw)
                    product = position.get('product', position.get('productType', ''))

                    # Debug each mapped position clearly for docker logs
                    print(f"DEBUG: Mapped position -> symbol={position.get('symbol','')}, qty={quantity}, avg={avg_price}, ltp={ltp_val}, pnl={pnl_val}, side={side}, product={product}")

                    processed_positions.append({
                        'symbol': position.get('symbol', ''),
                        'quantity': quantity,
                        'average_price': avg_price,
                        'ltp': ltp_val,
                        'pnl': pnl_val,
                        'pnl_percent': position.get('plPercent', position.get('pnl_percent', 0)),
                        'side': side,
                        'product': product
                    })

                return {
                    'success': True,
                    'data': processed_positions,
                    'last_updated': datetime.now().isoformat()
                }
            else:
                error_msg = positions_data.get('message', 'Unknown error')
                print(f"DEBUG: Error in positions_data: {error_msg}")
                logger.error(f"Failed to fetch positions data: {error_msg}")
                return {
                    'success': False,
                    'error': f'Failed to fetch positions data from FYERS: {error_msg}'
                }
        except Exception as e:
            print(f"DEBUG: Exception in get_portfolio_positions: {str(e)}")
            logger.error(f"Exception in get_portfolio_positions: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to fetch positions data from FYERS: {str(e)}'
            }

_portfolio_service = None

def get_portfolio_service():
    """Singleton factory for PortfolioService."""
    global _portfolio_service
    if _portfolio_service is None:
        broker_service = get_broker_service()
        _portfolio_service = PortfolioService(broker_service)
    return _portfolio_service
