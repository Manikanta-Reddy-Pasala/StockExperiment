"""
Order Router for the Automated Trading System
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


class OrderRouter:
    """Routes orders to appropriate broker connectors."""
    
    def __init__(self, broker_connector):
        """
        Initialize order router.
        
        Args:
            broker_connector: Broker connector instance
        """
        self.broker_connector = broker_connector
        self.logger = logging.getLogger(__name__)
        self.order_history = []
    
    def place_order(self, symbol: str, quantity: int, order_type: str = 'BUY',
                   price: Optional[float] = None, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Place an order through the broker connector.
        
        Args:
            symbol (str): Stock symbol
            quantity (int): Number of shares
            order_type (str): Order type (BUY/SELL)
            price (Optional[float]): Price for limit orders
            user_id (Optional[int]): User ID for tracking
            
        Returns:
            Dict[str, Any]: Order result
        """
        try:
            # Route order to broker connector
            result = self.broker_connector.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type=order_type,
                price=price
            )
            
            # Add metadata
            result['user_id'] = user_id
            result['routed_at'] = datetime.now().isoformat()
            
            # Store in order history
            self.order_history.append(result)
            
            self.logger.info(f"Order routed: {order_type} {quantity} {symbol} - Result: {result.get('success', False)}")
            
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'quantity': quantity,
                'order_type': order_type,
                'price': price,
                'user_id': user_.id,
                'routed_at': datetime.now().isoformat()
            }
            
            self.order_history.append(error_result)
            self.logger.error(f"Error routing order: {e}")
            
            return error_result
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status from broker.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            Dict[str, Any]: Order status
        """
        try:
            return self.broker_connector.get_order_status(order_id)
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions from broker.
        
        Returns:
            List[Dict[str, Any]]: List of positions
        """
        try:
            return self.broker_connector.get_positions()
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        """
        Get current holdings from broker.
        
        Returns:
            List[Dict[str, Any]]: List of holdings
        """
        try:
            return self.broker_connector.get_holdings()
        except Exception as e:
            self.logger.error(f"Error getting holdings: {e}")
            return []
    
    def get_order_history(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get order history.
        
        Args:
            user_id (Optional[int]): Filter by user ID
            
        Returns:
            List[Dict[str, Any]]: Order history
        """
        if user_id is None:
            return self.order_history.copy()
        
        return [order for order in self.order_history if order.get('user_id') == user_id]
