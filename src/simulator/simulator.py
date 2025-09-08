"""
Paper Trading Simulator for the Automated Trading System
"""
import random
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from broker.base_connector import BrokerConnector
from order.order_router import OrderStatus


class Simulator(BrokerConnector):
    """Simulates broker functionality for paper trading."""
    
    def __init__(self, initial_balance: float = 1000000.0):
        """
        Initialize the simulator.
        
        Args:
            initial_balance (float): Initial account balance
        """
        self.balance = initial_balance
        self.positions = {}  # symbol -> quantity
        self.orders = {}  # order_id -> order details
        self.trades = []  # list of executed trades
        self.market_data = {}  # symbol -> current price
        self.timestamp = datetime.now()
        self.order_counter = 1000  # Simulated order ID counter
    
    def set_market_data(self, market_data: Dict[str, float]):
        """
        Set current market data for simulation.
        
        Args:
            market_data (Dict[str, float]): Symbol to price mapping
        """
        self.market_data.update(market_data)
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile."""
        return {
            "status": "success",
            "data": {
                "user_id": "SIMULATED_USER",
                "user_name": "Simulated Trader",
                "user_shortname": "ST",
                "email": "simulated@example.com",
                "user_type": "investor",
                "broker": "SIMULATOR",
                "exchanges": ["NSE", "BSE"],
                "products": ["CNC", "MIS"],
                "order_types": ["MARKET", "LIMIT", "SL", "SL-M"]
            }
        }
    
    def get_margins(self) -> Dict[str, Any]:
        """Get user margins."""
        return {
            "status": "success",
            "data": {
                "equity": {
                    "enabled": True,
                    "net": self.balance,
                    "available": {
                        "live_balance": self.balance,
                        "opening_balance": self.balance,
                        "payin": 0,
                        "payout": 0
                    }
                }
            }
        }
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders."""
        return list(self.orders.values())
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        positions_data = []
        for symbol, quantity in self.positions.items():
            if quantity != 0:
                positions_data.append({
                    "tradingsymbol": symbol,
                    "exchange": "NSE",
                    "instrument_token": f"TOKEN_{symbol}",
                    "product": "MIS",
                    "quantity": quantity,
                    "overnight_quantity": 0,
                    "multiplier": 1,
                    "average_price": 0,  # Simplified
                    "close_price": 0,
                    "last_price": self.market_data.get(symbol, 0),
                    "value": quantity * self.market_data.get(symbol, 0),
                    "pnl": 0,  # Simplified
                    "m2m": 0,
                    "unrealised": 0,
                    "realised": 0,
                    "buy_quantity": max(0, quantity),
                    "buy_price": 0,  # Simplified
                    "buy_value": max(0, quantity) * self.market_data.get(symbol, 0),
                    "buy_m2m": 0,
                    "sell_quantity": max(0, -quantity),
                    "sell_price": 0,  # Simplified
                    "sell_value": max(0, -quantity) * self.market_data.get(symbol, 0),
                    "sell_m2m": 0
                })
        
        return {
            "status": "success",
            "data": {
                "net": positions_data
            }
        }
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        """Get current holdings."""
        return []
    
    def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order."""
        # Generate simulated order ID
        self.order_counter += 1
        order_id = f"SIM_ORDER_{self.order_counter}"
        
        # Extract order parameters
        symbol = order_params.get('tradingsymbol')
        quantity = int(order_params.get('quantity', 0))
        order_type = order_params.get('order_type', 'MARKET')
        transaction_type = order_params.get('transaction_type', 'BUY')
        product = order_params.get('product', 'MIS')
        price = float(order_params.get('price', 0))
        trigger_price = float(order_params.get('trigger_price', 0))
        
        # Store order
        order = {
            "order_id": order_id,
            "parent_order_id": None,
            "exchange_order_id": f"EXCH_{order_id}",
            "tradingsymbol": symbol,
            "exchange": "NSE",
            "instrument_token": f"TOKEN_{symbol}",
            "product": product,
            "order_type": order_type,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "disclosed_quantity": 0,
            "price": price,
            "trigger_price": trigger_price,
            "average_price": 0,
            "filled_quantity": 0,
            "pending_quantity": quantity,
            "order_status": "OPEN",
            "status_message": "Order submitted",
            "tag": order_params.get('tag', ''),
            "placed_at": self.timestamp.isoformat(),
            "placed_by": "SIMULATOR",
            "variety": "regular"
        }
        
        self.orders[order_id] = order
        
        # Simulate order execution
        self._simulate_order_execution(order_id)
        
        return {
            "status": "success",
            "data": {
                "order_id": order_id
            }
        }
    
    def modify_order(self, order_id: str, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an existing order."""
        if order_id not in self.orders:
            return {
                "status": "error",
                "message": "Order not found",
                "data": {}
            }
        
        order = self.orders[order_id]
        
        # Update order parameters
        if 'quantity' in order_params:
            order['quantity'] = int(order_params['quantity'])
            order['pending_quantity'] = order['quantity'] - order['filled_quantity']
        
        if 'price' in order_params:
            order['price'] = float(order_params['price'])
        
        if 'trigger_price' in order_params:
            order['trigger_price'] = float(order_params['trigger_price'])
        
        order['status_message'] = "Order modified"
        
        return {
            "status": "success",
            "data": {
                "order_id": order_id
            }
        }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        if order_id not in self.orders:
            return {
                "status": "error",
                "message": "Order not found",
                "data": {}
            }
        
        order = self.orders[order_id]
        if order['order_status'] in ['OPEN', 'TRIGGER PENDING']:
            order['order_status'] = 'CANCELLED'
            order['status_message'] = "Order cancelled"
            
            return {
                "status": "success",
                "data": {
                    "order_id": order_id
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Cannot cancel order with status {order['order_status']}",
                "data": {}
            }
    
    def get_instruments(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get instrument master data."""
        # Return empty list in simulator
        return []
    
    def get_ltp(self, instrument_tokens: List[str]) -> Dict[str, Any]:
        """Get last traded price for instruments."""
        ltp_data = {}
        for token in instrument_tokens:
            # Extract symbol from token (simplified)
            symbol = token.replace("TOKEN_", "")
            if symbol in self.market_data:
                ltp_data[token] = {
                    "instrument_token": token,
                    "last_price": self.market_data[symbol]
                }
        
        return {
            "status": "success",
            "data": ltp_data
        }
    
    def get_quote(self, instrument_tokens: List[str]) -> Dict[str, Any]:
        """Get quote data for instruments."""
        quote_data = {}
        for token in instrument_tokens:
            # Extract symbol from token (simplified)
            symbol = token.replace("TOKEN_", "")
            if symbol in self.market_data:
                last_price = self.market_data[symbol]
                # Simulate bid/ask spread
                bid_price = last_price * (1 - random.uniform(0.001, 0.005))
                ask_price = last_price * (1 + random.uniform(0.001, 0.005))
                
                quote_data[token] = {
                    "instrument_token": token,
                    "timestamp": self.timestamp.isoformat(),
                    "last_price": last_price,
                    "last_quantity": random.randint(1, 100),
                    "average_price": last_price,
                    "volume": random.randint(1000, 100000),
                    "buy_quantity": random.randint(100, 1000),
                    "sell_quantity": random.randint(100, 1000),
                    "ohlc": {
                        "open": last_price * random.uniform(0.995, 1.005),
                        "high": last_price * random.uniform(1.001, 1.01),
                        "low": last_price * random.uniform(0.99, 0.999),
                        "close": last_price * random.uniform(0.995, 1.005)
                    },
                    "depth": {
                        "buy": [{"price": bid_price, "quantity": random.randint(100, 1000), "orders": random.randint(1, 10)}],
                        "sell": [{"price": ask_price, "quantity": random.randint(100, 1000), "orders": random.randint(1, 10)}]
                    }
                }
        
        return {
            "status": "success",
            "data": quote_data
        }
    
    def _simulate_order_execution(self, order_id: str):
        """
        Simulate order execution.
        
        Args:
            order_id (str): Order ID to simulate execution for
        """
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        symbol = order['tradingsymbol']
        
        # Skip if order is already completed
        if order['order_status'] in ['COMPLETE', 'CANCELLED', 'REJECTED']:
            return
        
        # Get current market price
        if symbol not in self.market_data:
            order['order_status'] = 'REJECTED'
            order['status_message'] = 'Symbol not found'
            return
        
        current_price = self.market_data[symbol]
        
        # Simulate execution based on order type
        execution_price = current_price
        fill_quantity = order['quantity']
        
        # For LIMIT orders, check if price matches
        if order['order_type'] == 'LIMIT':
            if order['transaction_type'] == 'BUY' and order['price'] < current_price:
                # Buy limit order not filled
                return
            elif order['transaction_type'] == 'SELL' and order['price'] > current_price:
                # Sell limit order not filled
                return
            execution_price = order['price']
        
        # For SL orders, check trigger condition
        elif order['order_type'] in ['SL', 'SL-M']:
            if order['transaction_type'] == 'BUY' and order['trigger_price'] > current_price:
                # Buy stoploss not triggered
                return
            elif order['transaction_type'] == 'SELL' and order['trigger_price'] < current_price:
                # Sell stoploss not triggered
                return
            # When triggered, SL-M becomes market order, SL becomes limit order
            if order['order_type'] == 'SL-M':
                execution_price = current_price * (1 + random.uniform(-0.001, 0.001))  # Small slippage
            else:
                execution_price = order['price']
        
        # Update order status
        order['order_status'] = 'COMPLETE'
        order['filled_quantity'] = fill_quantity
        order['pending_quantity'] = 0
        order['average_price'] = execution_price
        order['status_message'] = 'Order executed'
        
        # Update balance and positions
        trade_value = execution_price * fill_quantity
        if order['transaction_type'] == 'BUY':
            self.balance -= trade_value
            self.positions[symbol] = self.positions.get(symbol, 0) + fill_quantity
        else:
            self.balance += trade_value
            self.positions[symbol] = self.positions.get(symbol, 0) - fill_quantity
        
        # Record trade
        trade = {
            "trade_id": f"SIM_TRADE_{len(self.trades) + 1}",
            "order_id": order_id,
            "exchange_order_id": order['exchange_order_id'],
            "tradingsymbol": symbol,
            "exchange": "NSE",
            "instrument_token": f"TOKEN_{symbol}",
            "transaction_type": order['transaction_type'],
            "quantity": fill_quantity,
            "price": execution_price,
            "filled_quantity": fill_quantity,
            "order_price": execution_price,
            "trade_time": self.timestamp.isoformat()
        }
        self.trades.append(trade)
    
    def update_market_data(self, symbol: str, new_price: float):
        """
        Update market data for a symbol.
        
        Args:
            symbol (str): Trading symbol
            new_price (float): New price
        """
        self.market_data[symbol] = new_price
        
        # Update timestamp
        self.timestamp = datetime.now()
        
        # Re-evaluate open orders
        for order_id in list(self.orders.keys()):
            self._simulate_order_execution(order_id)