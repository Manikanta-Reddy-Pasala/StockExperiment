"""
Order Router for the Automated Trading System
"""
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

# Add import for trade logger
from trade_logging.trade_logger import TradeLogger


class OrderStatus(Enum):
    """Order status enumeration."""
    NEW = "NEW"
    PLACED = "PLACED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL_MARKET = "SL-M"
    SL_LIMIT = "SL-LIMIT"
    MARKET_ON_OPEN = "MARKET_ON_OPEN"  # Added Market-on-Open order type


class ProductType(Enum):
    """Product type enumeration."""
    MIS = "MIS"  # Intraday
    NRML = "NRML"  # Delivery
    CNC = "CNC"  # Cash and carry


class Order:
    """Represents a single order."""
    
    def __init__(self, 
                 symbol: str,
                 quantity: int,
                 order_type: OrderType,
                 transaction_type: str,  # BUY or SELL
                 product_type: ProductType = ProductType.MIS,
                 price: float = 0.0,
                 trigger_price: float = 0.0,
                 tag: str = ""):
        """
        Initialize an order.
        
        Args:
            symbol (str): Trading symbol
            quantity (int): Order quantity
            order_type (OrderType): Type of order
            transaction_type (str): BUY or SELL
            product_type (ProductType): Product type
            price (float): Order price (for LIMIT orders)
            trigger_price (float): Trigger price (for SL orders)
            tag (str): Order tag for tracking
        """
        self.order_id = str(uuid.uuid4())
        self.symbol = symbol
        self.quantity = quantity
        self.filled_quantity = 0
        self.order_type = order_type
        self.transaction_type = transaction_type
        self.product_type = product_type
        self.price = price
        self.trigger_price = trigger_price
        self.status = OrderStatus.NEW
        self.average_price = 0.0
        self.tag = tag
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.broker_order_id = None
        self.status_message = ""
        self.is_moo_order = (order_type == OrderType.MARKET_ON_OPEN)  # Flag for MOO orders
    
    def update_status(self, status: OrderStatus, message: str = ""):
        """
        Update order status.
        
        Args:
            status (OrderStatus): New status
            message (str): Status message
        """
        self.status = status
        self.status_message = message
        self.updated_at = datetime.now()
    
    def fill(self, quantity: int, price: float):
        """
        Fill order partially or completely.
        
        Args:
            quantity (int): Filled quantity
            price (float): Fill price
        """
        self.filled_quantity += quantity
        total_value = self.average_price * (self.filled_quantity - quantity) + price * quantity
        self.average_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0
        
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL
        
        self.updated_at = datetime.now()


class OrderRouter:
    """Routes orders to broker and manages order state."""
    
    def __init__(self, broker_connector):
        """
        Initialize the order router.
        
        Args:
            broker_connector: Broker connector instance
        """
        self.broker = broker_connector
        self.orders = {}  # Order ID to Order object mapping
        self.symbol_orders = {}  # Symbol to list of order IDs mapping
        self.trade_logger = TradeLogger()  # Initialize trade logger
    
    def _validate_order(self, order: Order) -> bool:
        """
        Validate an order.
        
        Args:
            order (Order): Order to validate
            
        Returns:
            bool: True if order is valid, False otherwise
        """
        # Check if all required fields are present
        if not order.symbol or order.quantity <= 0:
            return False
        
        # Check if order type is valid
        if not isinstance(order.order_type, OrderType):
            return False
        
        # Check if transaction type is valid
        if order.transaction_type not in ['BUY', 'SELL']:
            return False
        
        # Check if product type is valid
        if not isinstance(order.product_type, ProductType):
            return False
        
        # For LIMIT and SL orders, price must be positive
        if order.order_type in [OrderType.LIMIT, OrderType.SL_LIMIT] and order.price <= 0:
            return False
        
        # For SL orders, trigger price must be positive
        if order.order_type in [OrderType.SL_MARKET, OrderType.SL_LIMIT] and order.trigger_price <= 0:
            return False
        
        return True
    
    def place_order(self, order: Order) -> str:
        """
        Place an order.
        
        Args:
            order (Order): Order to place
            
        Returns:
            str: Order ID
        """
        # Validate order
        if not self._validate_order(order):
            order.update_status(OrderStatus.ERROR, "Invalid order parameters")
            return order.order_id
        
        # Add to tracking
        self.orders[order.order_id] = order
        if order.symbol not in self.symbol_orders:
            self.symbol_orders[order.symbol] = []
        self.symbol_orders[order.symbol].append(order.order_id)
        
        # Log order placement
        self.trade_logger.log_order_placement({
            'order_id': order.order_id,
            'symbol': order.symbol,
            'quantity': order.quantity,
            'order_type': order.order_type.value,
            'transaction_type': order.transaction_type,
            'price': order.price,
            'trigger_price': order.trigger_price
        })
        
        # For Market-on-Open orders, we queue them for next market open
        if order.is_moo_order:
            order.update_status(OrderStatus.PLACED, "MOO order queued for market open")
            return order.order_id
        
        # Prepare broker order parameters
        order_params = {
            'tradingsymbol': order.symbol,
            'quantity': order.quantity,
            'order_type': order.order_type.value,
            'transaction_type': order.transaction_type,
            'product': order.product_type.value,
            'tag': order.tag
        }
        
        # Add price parameters based on order type
        if order.order_type in [OrderType.LIMIT, OrderType.SL_LIMIT]:
            order_params['price'] = order.price
        
        if order.order_type in [OrderType.SL_MARKET, OrderType.SL_LIMIT]:
            order_params['trigger_price'] = order.trigger_price
        
        try:
            # Place order with broker
            response = self.broker.place_order(order_params)
            
            if response.get('status') == 'success':
                order.broker_order_id = response.get('data', {}).get('order_id')
                order.update_status(OrderStatus.PLACED)
                
                # Log successful order placement
                self.trade_logger.log_order_placement({
                    'order_id': order.order_id,
                    'broker_order_id': order.broker_order_id,
                    'symbol': order.symbol,
                    'quantity': order.quantity,
                    'order_type': order.order_type.value,
                    'transaction_type': order.transaction_type,
                    'status': 'SUCCESS',
                    'message': 'Order placed successfully'
                })
            else:
                order.update_status(OrderStatus.ERROR, response.get('message', 'Unknown error'))
                
                # Log order error
                self.trade_logger.log_order_placement({
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'quantity': order.quantity,
                    'order_type': order.order_type.value,
                    'transaction_type': order.transaction_type,
                    'status': 'ERROR',
                    'message': response.get('message', 'Unknown error')
                })
        except Exception as e:
            order.update_status(OrderStatus.ERROR, str(e))
            
            # Log exception
            self.trade_logger.log_order_placement({
                'order_id': order.order_id,
                'symbol': order.symbol,
                'quantity': order.quantity,
                'order_type': order.order_type.value,
                'transaction_type': order.transaction_type,
                'status': 'EXCEPTION',
                'message': str(e)
            })
        
        return order.order_id
    
    def modify_order(self, order_id: str, quantity: int = None, price: float = None) -> bool:
        """
        Modify an existing order.
        
        Args:
            order_id (str): Order ID to modify
            quantity (int, optional): New quantity
            price (float, optional): New price
            
        Returns:
            bool: True if modification was successful, False otherwise
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        # Prepare modification parameters
        modify_params = {}
        if quantity is not None:
            modify_params['quantity'] = quantity
        if price is not None:
            modify_params['price'] = price
        
        try:
            # Modify order with broker
            response = self.broker.modify_order(order.broker_order_id, modify_params)
            
            if response.get('status') == 'success':
                # Update local order
                if quantity is not None:
                    order.quantity = quantity
                if price is not None:
                    order.price = price
                order.updated_at = datetime.now()
                
                # Log order modification
                self.trade_logger.log_order_modification({
                    'order_id': order_id,
                    'broker_order_id': order.broker_order_id,
                    'symbol': order.symbol,
                    'old_quantity': order.quantity,
                    'new_quantity': quantity,
                    'old_price': order.price,
                    'new_price': price,
                    'status': 'SUCCESS',
                    'message': 'Order modified successfully'
                })
                
                return True
            else:
                # Log modification error
                self.trade_logger.log_order_modification({
                    'order_id': order_id,
                    'broker_order_id': order.broker_order_id,
                    'symbol': order.symbol,
                    'new_quantity': quantity,
                    'new_price': price,
                    'status': 'ERROR',
                    'message': response.get('message', 'Unknown error')
                })
                
                return False
        except Exception as e:
            # Log exception
            self.trade_logger.log_order_modification({
                'order_id': order_id,
                'broker_order_id': order.broker_order_id,
                'symbol': order.symbol,
                'new_quantity': quantity,
                'new_price': price,
                'status': 'EXCEPTION',
                'message': str(e)
            })
            
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id (str): Order ID to cancel
            
        Returns:
            bool: True if cancellation was successful, False otherwise
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        try:
            # Cancel order with broker
            response = self.broker.cancel_order(order.broker_order_id)
            
            if response.get('status') == 'success':
                order.update_status(OrderStatus.CANCELLED)
                
                # Log order cancellation
                self.trade_logger.log_order_cancellation({
                    'order_id': order_id,
                    'broker_order_id': order.broker_order_id,
                    'symbol': order.symbol,
                    'quantity': order.quantity,
                    'status': 'SUCCESS',
                    'message': 'Order cancelled successfully'
                })
                
                return True
            else:
                # Log cancellation error
                self.trade_logger.log_order_cancellation({
                    'order_id': order_id,
                    'broker_order_id': order.broker_order_id,
                    'symbol': order.symbol,
                    'quantity': order.quantity,
                    'status': 'ERROR',
                    'message': response.get('message', 'Unknown error')
                })
                
                return False
        except Exception as e:
            # Log exception
            self.trade_logger.log_order_cancellation({
                'order_id': order_id,
                'broker_order_id': order.broker_order_id,
                'symbol': order.symbol,
                'quantity': order.quantity,
                'status': 'EXCEPTION',
                'message': str(e)
            })
            
            return False
    
    def update_order_from_broker(self, order_id: str):
        """
        Update order status from broker information.
        
        Args:
            order_id (str): Order ID to update
        """
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        
        try:
            # Get orders from broker
            broker_orders = self.broker.get_orders()
            
            # Find our order
            broker_order = None
            for bo in broker_orders:
                if bo.get('order_id') == order.broker_order_id:
                    broker_order = bo
                    break
            
            if broker_order:
                # Update order status
                status_map = {
                    'OPEN': OrderStatus.PLACED,
                    'COMPLETE': OrderStatus.FILLED,
                    'CANCELLED': OrderStatus.CANCELLED,
                    'REJECTED': OrderStatus.REJECTED,
                    'TRIGGER PENDING': OrderStatus.PLACED,
                    'OPEN PENDING': OrderStatus.PLACED
                }
                
                broker_status = broker_order.get('status')
                if broker_status in status_map:
                    old_status = order.status
                    order.update_status(status_map[broker_status], broker_order.get('status_message', ''))
                    
                    # Log status change if it's a significant change
                    if old_status != order.status and order.status in [OrderStatus.FILLED, OrderStatus.REJECTED]:
                        self.trade_logger.log_order_placement({
                            'order_id': order_id,
                            'broker_order_id': order.broker_order_id,
                            'symbol': order.symbol,
                            'quantity': order.quantity,
                            'status': order.status.name,
                            'message': broker_order.get('status_message', '')
                        })
                
                # Update fill information
                filled_qty = broker_order.get('filled_quantity', 0)
                avg_price = broker_order.get('average_price', 0.0)
                
                if filled_qty > order.filled_quantity:
                    # Calculate newly filled quantity
                    new_filled = filled_qty - order.filled_quantity
                    order.fill(new_filled, avg_price)
                    
                    # Log trade execution
                    self.trade_logger.log_trade_execution({
                        'order_id': order_id,
                        'broker_order_id': order.broker_order_id,
                        'symbol': order.symbol,
                        'quantity': new_filled,
                        'price': avg_price,
                        'transaction_type': order.transaction_type,
                        'message': 'Trade executed'
                    })
        except Exception as e:
            # Log exception
            self.trade_logger.log_system_event(
                event_type="ORDER_UPDATE_ERROR",
                message=f"Error updating order {order_id} from broker",
                details={'error': str(e), 'order_id': order_id}
            )
