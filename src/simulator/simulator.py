"""
Trading Simulator for Development and Testing
"""
import logging
import random
from typing import Dict, Any, Optional, List
from datetime import datetime


class Simulator:
    """Simulates broker operations for development and testing."""
    
    def __init__(self):
        """Initialize the trading simulator."""
        self.logger = logging.getLogger(__name__)
        self.positions = {}
        self.holdings = {}
        self.order_counter = 0
        self.is_connected = True
        
        self.logger.info("Trading simulator initialized")
    
    def place_order(self, symbol: str, quantity: int, order_type: str = 'BUY',
                   price: Optional[float] = None) -> Dict[str, Any]:
        """
        Simulate placing an order.
        
        Args:
            symbol (str): Stock symbol
            quantity (int): Number of shares
            order_type (str): Order type (BUY/SELL)
            price (Optional[float]): Price for limit orders
            
        Returns:
            Dict[str, Any]: Simulated order result
        """
        self.order_counter += 1
        order_id = f"SIM_{self.order_counter:06d}"
        
        # Simulate order execution with some randomness
        success = random.random() > 0.1  # 90% success rate
        
        if success:
            # Simulate execution price (slight variation from market price)
            if price is None:
                # Simulate market price with some variation
                base_price = random.uniform(100, 1000)
                execution_price = base_price * (1 + random.uniform(-0.02, 0.02))
            else:
                execution_price = price * (1 + random.uniform(-0.01, 0.01))
            
            # Update positions
            if symbol in self.positions:
                if order_type == 'BUY':
                    self.positions[symbol] += quantity
                else:
                    self.positions[symbol] -= quantity
            else:
                self.positions[symbol] = quantity if order_type == 'BUY' else -quantity
            
            # Remove position if it's closed
            if abs(self.positions[symbol]) < 1:
                del self.positions[symbol]
            
            result = {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'order_type': order_type,
                'price': execution_price,
                'status': 'COMPLETED',
                'timestamp': datetime.now().isoformat(),
                'execution_price': execution_price
            }
        else:
            result = {
                'success': False,
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'order_type': order_type,
                'price': price,
                'status': 'REJECTED',
                'timestamp': datetime.now().isoformat(),
                'error': 'Simulated order rejection'
            }
        
        self.logger.info(f"Simulated order: {order_type} {quantity} {symbol} - Success: {success}")
        return result
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Simulate getting order status.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            Dict[str, Any]: Order status
        """
        # In a real implementation, this would track order states
        # For simulation, we'll return a completed status
        return {
            'success': True,
            'order_id': order_id,
            'status': 'COMPLETED',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get simulated positions.
        
        Returns:
            List[Dict[str, Any]]: List of positions
        """
        positions = []
        for symbol, quantity in self.positions.items():
            if abs(quantity) > 0:
                # Simulate current price
                current_price = random.uniform(100, 1000)
                positions.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'current_price': current_price,
                    'market_value': quantity * current_price,
                    'unrealized_pnl': 0  # Simplified for simulation
                })
        
        return positions
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        """
        Get simulated holdings.
        
        Returns:
            List[Dict[str, Any]]: List of holdings
        """
        holdings = []
        for symbol, quantity in self.holdings.items():
            if quantity > 0:
                # Simulate current price and cost basis
                current_price = random.uniform(100, 1000)
                cost_basis = current_price * random.uniform(0.8, 1.2)
                
                holdings.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'current_price': current_price,
                    'cost_basis': cost_basis,
                    'market_value': quantity * current_price,
                    'unrealized_pnl': quantity * (current_price - cost_basis)
                })
        
        return holdings
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get simulated account summary.
        
        Returns:
            Dict[str, Any]: Account summary
        """
        positions = self.get_positions()
        holdings = self.get_holdings()
        
        total_market_value = sum(pos['market_value'] for pos in positions + holdings)
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in positions + holdings)
        
        return {
            'total_market_value': total_market_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'positions_count': len(positions),
            'holdings_count': len(holdings),
            'available_cash': random.uniform(10000, 100000),  # Simulated cash
            'timestamp': datetime.now().isoformat()
        }
    
    def reset(self):
        """Reset simulator state."""
        self.positions = {}
        self.holdings = {}
        self.order_counter = 0
        self.logger.info("Simulator reset")
