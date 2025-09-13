"""
FYERS Orders Provider Implementation

Implements the IOrdersProvider interface for FYERS broker.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..interfaces.orders_interface import IOrdersProvider, Order, OrderType, OrderSide, OrderStatus
from ..broker_service import get_broker_service

logger = logging.getLogger(__name__)


class FyersOrdersProvider(IOrdersProvider):
    """FYERS implementation of orders provider."""
    
    def __init__(self):
        self.broker_service = get_broker_service()
    
    def get_orders_history(self, user_id: int, start_date: datetime = None, 
                          end_date: datetime = None, limit: int = 100) -> Dict[str, Any]:
        """Get orders history using FYERS API."""
        try:
            orderbook_data = self.broker_service.get_fyers_orderbook(user_id)
            
            if not orderbook_data.get('success'):
                return {
                    'success': False,
                    'error': orderbook_data.get('error', 'Failed to fetch orders'),
                    'data': [],
                    'total': 0,
                    'last_updated': datetime.now().isoformat()
                }
            
            orders = orderbook_data['data'].get('orderBook', [])
            
            # Process and format orders
            processed_orders = []
            for order_data in orders[:limit]:
                order = Order(
                    order_id=order_data.get('id', ''),
                    symbol=order_data.get('symbol', ''),
                    side=OrderSide.BUY if order_data.get('side') == '1' else OrderSide.SELL,
                    order_type=OrderType.LIMIT if order_data.get('type') == '1' else OrderType.MARKET,
                    quantity=order_data.get('qty', 0),
                    price=order_data.get('limitPrice', 0)
                )
                
                order.status = self._map_order_status(order_data.get('status', ''))
                order.filled_quantity = order_data.get('filledQty', 0)
                order.remaining_quantity = order_data.get('remainingQty', 0)
                order.product = order_data.get('product', '')
                
                # Parse order time
                if order_data.get('orderDateTime'):
                    try:
                        order.order_time = datetime.fromisoformat(order_data['orderDateTime'])
                    except:
                        order.order_time = datetime.now()
                
                processed_orders.append(order.to_dict())
            
            return {
                'success': True,
                'data': processed_orders,
                'total': len(processed_orders),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching orders history for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0,
                'last_updated': datetime.now().isoformat()
            }
