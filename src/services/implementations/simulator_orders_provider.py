"""
Simulator Orders Provider - Paper Trading Implementation
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random
from ..interfaces.orders_interface import IOrdersProvider, Order, OrderType, OrderSide, OrderStatus


class SimulatorOrdersProvider(IOrdersProvider):
    """Simulator implementation for orders management."""
    
    def __init__(self):
        # In-memory storage for simulator orders and trades
        self._orders = {}
        self._trades = {}
        self._order_counter = 10000
    
    def get_orders_history(self, user_id: int, start_date: datetime = None, 
                          end_date: datetime = None, limit: int = 100) -> Dict[str, Any]:
        """Get simulated orders history."""
        try:
            # Generate sample orders if none exist
            if user_id not in self._orders:
                self._generate_sample_orders(user_id)
            
            orders = self._orders.get(user_id, [])
            
            # Apply date filtering if provided
            if start_date or end_date:
                filtered_orders = []
                for order in orders:
                    order_time = datetime.fromisoformat(order.get('order_time', ''))
                    if start_date and order_time < start_date:
                        continue
                    if end_date and order_time > end_date:
                        continue
                    filtered_orders.append(order)
                orders = filtered_orders
            
            # Apply limit
            orders = orders[:limit]
            
            return {
                'success': True,
                'data': orders,
                'total': len(orders),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    def get_pending_orders(self, user_id: int) -> Dict[str, Any]:
        """Get simulated pending orders."""
        try:
            if user_id not in self._orders:
                self._generate_sample_orders(user_id)
            
            orders = self._orders.get(user_id, [])
            pending_orders = [order for order in orders if order.get('status') in ['open', 'pending']]
            
            return {
                'success': True,
                'data': pending_orders,
                'total': len(pending_orders),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    def get_trades_history(self, user_id: int, start_date: datetime = None, 
                          end_date: datetime = None, limit: int = 100) -> Dict[str, Any]:
        """Get simulated trades history."""
        try:
            # Generate sample trades if none exist
            if user_id not in self._trades:
                self._generate_sample_trades(user_id)
            
            trades = self._trades.get(user_id, [])
            
            # Apply date filtering if provided
            if start_date or end_date:
                filtered_trades = []
                for trade in trades:
                    trade_time = datetime.fromisoformat(trade.get('trade_time', ''))
                    if start_date and trade_time < start_date:
                        continue
                    if end_date and trade_time > end_date:
                        continue
                    filtered_trades.append(trade)
                trades = filtered_trades
            
            # Apply limit
            trades = trades[:limit]
            
            return {
                'success': True,
                'data': trades,
                'total': len(trades),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    def place_order(self, user_id: int, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Place a simulated order."""
        try:
            order_id = f'SIM{self._order_counter}'
            self._order_counter += 1
            
            # Create order object
            order = Order(
                order_id=order_id,
                symbol=order_data.get('symbol', 'NSE:RELIANCE-EQ'),
                side=OrderSide.BUY if order_data.get('side') == 'buy' else OrderSide.SELL,
                order_type=OrderType.LIMIT if order_data.get('type') == 'limit' else OrderType.MARKET,
                quantity=order_data.get('quantity', 1),
                price=order_data.get('price', 100.0)
            )
            
            order.status = OrderStatus.PENDING
            order.product = order_data.get('product', 'INTRADAY')
            
            # Store order
            if user_id not in self._orders:
                self._orders[user_id] = []
            
            self._orders[user_id].append(order.to_dict())
            
            # Simulate order execution after a delay
            self._simulate_order_execution(user_id, order_id)
            
            return {
                'success': True,
                'order_id': order_id,
                'message': 'Simulated order placed successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def modify_order(self, user_id: int, order_id: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Modify a simulated order."""
        try:
            if user_id not in self._orders:
                return {
                    'success': False,
                    'error': 'No orders found for user'
                }
            
            orders = self._orders[user_id]
            for i, order in enumerate(orders):
                if order.get('order_id') == order_id:
                    # Update order data
                    if 'quantity' in order_data:
                        orders[i]['quantity'] = order_data['quantity']
                    if 'price' in order_data:
                        orders[i]['price'] = order_data['price']
                    if 'type' in order_data:
                        orders[i]['type'] = order_data['type']
                    
                    orders[i]['last_modified'] = datetime.now().isoformat()
                    
                    return {
                        'success': True,
                        'message': 'Simulated order modified successfully'
                    }
            
            return {
                'success': False,
                'error': 'Order not found'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def cancel_order(self, user_id: int, order_id: str) -> Dict[str, Any]:
        """Cancel a simulated order."""
        try:
            if user_id not in self._orders:
                return {
                    'success': False,
                    'error': 'No orders found for user'
                }
            
            orders = self._orders[user_id]
            for i, order in enumerate(orders):
                if order.get('order_id') == order_id:
                    orders[i]['status'] = 'cancelled'
                    orders[i]['cancelled_at'] = datetime.now().isoformat()
                    
                    return {
                        'success': True,
                        'message': 'Simulated order cancelled successfully'
                    }
            
            return {
                'success': False,
                'error': 'Order not found'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_order_details(self, user_id: int, order_id: str) -> Dict[str, Any]:
        """Get simulated order details."""
        try:
            if user_id not in self._orders:
                return {
                    'success': False,
                    'error': 'No orders found for user',
                    'data': {},
                    'last_updated': datetime.now().isoformat()
                }
            
            orders = self._orders[user_id]
            for order in orders:
                if order.get('order_id') == order_id:
                    return {
                        'success': True,
                        'data': order,
                        'last_updated': datetime.now().isoformat()
                    }
            
            return {
                'success': False,
                'error': 'Order not found',
                'data': {},
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': {},
                'last_updated': datetime.now().isoformat()
            }
    
    def _generate_sample_orders(self, user_id: int):
        """Generate sample orders for simulation."""
        sample_symbols = ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:HDFCBANK-EQ', 'NSE:INFY-EQ', 'NSE:ICICIBANK-EQ']
        orders = []
        
        for i in range(10):
            order_id = f'SIM{10000 + i}'
            symbol = random.choice(sample_symbols)
            side = random.choice(['buy', 'sell'])
            order_type = random.choice(['limit', 'market'])
            quantity = random.randint(1, 50)
            price = random.uniform(100, 3000)
            status = random.choice(['filled', 'cancelled', 'open', 'pending'])
            
            order_time = datetime.now() - timedelta(days=random.randint(0, 30))
            
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
                'price': round(price, 2),
                'status': status,
                'filled_quantity': quantity if status == 'filled' else random.randint(0, quantity),
                'remaining_quantity': 0 if status == 'filled' else quantity - random.randint(0, quantity),
                'order_time': order_time.isoformat(),
                'product': 'INTRADAY'
            }
            
            if status == 'filled':
                order['fill_time'] = (order_time + timedelta(minutes=random.randint(1, 60))).isoformat()
            
            orders.append(order)
        
        self._orders[user_id] = orders
    
    def _generate_sample_trades(self, user_id: int):
        """Generate sample trades for simulation."""
        sample_symbols = ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:HDFCBANK-EQ', 'NSE:INFY-EQ', 'NSE:ICICIBANK-EQ']
        trades = []
        
        for i in range(15):
            trade_id = f'TRADE{20000 + i}'
            symbol = random.choice(sample_symbols)
            side = random.choice(['buy', 'sell'])
            quantity = random.randint(1, 50)
            price = random.uniform(100, 3000)
            order_id = f'SIM{10000 + random.randint(0, 9)}'
            
            trade_time = datetime.now() - timedelta(days=random.randint(0, 30))
            
            trade = {
                'id': trade_id,
                'symbol': symbol,
                'symbol_name': symbol.split(':')[1].replace('-EQ', ''),
                'side': side.upper(),
                'quantity': quantity,
                'price': round(price, 2),
                'trade_time': trade_time.isoformat(),
                'order_id': order_id,
                'product': 'INTRADAY',
                'pnl': round(random.uniform(-1000, 2000), 2),
                'brokerage': round(price * quantity * 0.0003, 2),  # 0.03% brokerage
                'exchange': 'NSE'
            }
            
            trades.append(trade)
        
        self._trades[user_id] = trades
    
    def _simulate_order_execution(self, user_id: int, order_id: str):
        """Simulate order execution after a delay."""
        # In a real implementation, this would be handled asynchronously
        # For simulation, we'll just mark some orders as filled immediately
        if random.random() > 0.3:  # 70% chance of execution
            if user_id in self._orders:
                for order in self._orders[user_id]:
                    if order.get('order_id') == order_id:
                        order['status'] = 'filled'
                        order['fill_time'] = datetime.now().isoformat()
                        order['filled_quantity'] = order['quantity']
                        order['remaining_quantity'] = 0
                        break
