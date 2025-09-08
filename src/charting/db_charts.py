"""
Database-based Charting System
"""
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from datastore.database import get_database_manager
from datastore.models import MarketData, Trade, Position, Order


class DatabaseCharts:
    """Generate simple charts from database data."""
    
    def __init__(self, db_manager=None):
        """
        Initialize the database charts system.
        
        Args:
            db_manager: Database manager instance (optional)
        """
        self.db_manager = db_manager or get_database_manager()
    
    def get_portfolio_value_chart_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get portfolio value chart data.
        
        Args:
            days (int): Number of days of data to retrieve
            
        Returns:
            List[Dict[str, Any]]: Portfolio value data for charting
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # For this example, we'll simulate portfolio values
            # In a real implementation, this would query actual portfolio data
            chart_data = []
            current_value = 100000.0  # Starting value
            
            current_date = start_date
            while current_date <= end_date:
                # Add some randomness to simulate market movements
                daily_return = (current_date.weekday() % 3 - 1) * 0.001 + (current_date.day % 5) * 0.0001
                current_value *= (1 + daily_return)
                
                chart_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'value': round(current_value, 2),
                    'pnl': round(current_value - 100000.0, 2)
                })
                
                current_date += timedelta(days=1)
            
            return chart_data
        except Exception as e:
            print(f"Error getting portfolio value chart data: {e}")
            return []
    
    def get_pnl_chart_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get P&L chart data.
        
        Args:
            days (int): Number of days of data to retrieve
            
        Returns:
            List[Dict[str, Any]]: P&L data for charting
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # For this example, we'll simulate P&L data
            chart_data = []
            cumulative_pnl = 0.0
            
            current_date = start_date
            while current_date <= end_date:
                # Add some randomness to simulate daily P&L
                daily_pnl = (current_date.weekday() % 4 - 2) * 100 + (current_date.day % 7) * 50
                cumulative_pnl += daily_pnl
                
                chart_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'daily_pnl': round(daily_pnl, 2),
                    'cumulative_pnl': round(cumulative_pnl, 2)
                })
                
                current_date += timedelta(days=1)
            
            return chart_data
        except Exception as e:
            print(f"Error getting P&L chart data: {e}")
            return []
    
    def get_trades_chart_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get trades chart data.
        
        Args:
            days (int): Number of days of data to retrieve
            
        Returns:
            List[Dict[str, Any]]: Trades data for charting
        """
        try:
            with self.db_manager.get_session() as session:
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Query trades
                trades = session.query(Trade).filter(
                    Trade.trade_time >= start_date,
                    Trade.trade_time <= end_date
                ).order_by(Trade.trade_time).all()
                
                # Convert to chart data
                chart_data = []
                for trade in trades:
                    chart_data.append({
                        'date': trade.trade_time.strftime('%Y-%m-%d %H:%M'),
                        'symbol': trade.tradingsymbol,
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'transaction_type': trade.transaction_type,
                        'value': trade.quantity * trade.price
                    })
                
                return chart_data
        except Exception as e:
            print(f"Error getting trades chart data: {e}")
            return []
    
    def get_positions_chart_data(self) -> List[Dict[str, Any]]:
        """
        Get current positions chart data.
        
        Returns:
            List[Dict[str, Any]]: Positions data for charting
        """
        try:
            with self.db_manager.get_session() as session:
                # Query current positions
                positions = session.query(Position).filter(
                    Position.quantity != 0
                ).all()
                
                # Convert to chart data
                chart_data = []
                for position in positions:
                    chart_data.append({
                        'symbol': position.tradingsymbol,
                        'quantity': position.quantity,
                        'average_price': position.average_price,
                        'market_value': position.value,
                        'pnl': position.pnl
                    })
                
                return chart_data
        except Exception as e:
            print(f"Error getting positions chart data: {e}")
            return []
    
    def get_orders_chart_data(self, status: str = None) -> List[Dict[str, Any]]:
        """
        Get orders chart data.
        
        Args:
            status (str): Filter by order status (optional)
            
        Returns:
            List[Dict[str, Any]]: Orders data for charting
        """
        try:
            with self.db_manager.get_session() as session:
                # Query orders
                query = session.query(Order)
                if status:
                    query = query.filter(Order.order_status == status)
                
                orders = query.order_by(Order.created_at.desc()).limit(100).all()
                
                # Convert to chart data
                chart_data = []
                for order in orders:
                    chart_data.append({
                        'order_id': order.order_id,
                        'symbol': order.tradingsymbol,
                        'quantity': order.quantity,
                        'order_type': order.order_type,
                        'transaction_type': order.transaction_type,
                        'status': order.order_status,
                        'price': order.price,
                        'created_at': order.created_at.strftime('%Y-%m-%d %H:%M')
                    })
                
                return chart_data
        except Exception as e:
            print(f"Error getting orders chart data: {e}")
            return []
    
    def get_market_data_chart(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get market data chart for a symbol.
        
        Args:
            symbol (str): Trading symbol
            days (int): Number of days of data to retrieve
            
        Returns:
            List[Dict[str, Any]]: Market data for charting
        """
        try:
            with self.db_manager.get_session() as session:
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Query market data
                market_data = session.query(MarketData).join(MarketData.instrument).filter(
                    MarketData.timestamp >= start_date,
                    MarketData.timestamp <= end_date,
                    MarketData.instrument.has(tradingsymbol=symbol)
                ).order_by(MarketData.timestamp).all()
                
                # Convert to chart data
                chart_data = []
                for data in market_data:
                    chart_data.append({
                        'timestamp': data.timestamp.strftime('%Y-%m-%d %H:%M'),
                        'open': data.open_price,
                        'high': data.high_price,
                        'low': data.low_price,
                        'close': data.close_price,
                        'volume': data.volume
                    })
                
                return chart_data
        except Exception as e:
            print(f"Error getting market data chart: {e}")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary data.
        
        Returns:
            Dict[str, Any]: Performance summary
        """
        try:
            with self.db_manager.get_session() as session:
                # Get total trades
                total_trades = session.query(Trade).count()
                
                # Get total orders
                total_orders = session.query(Order).count()
                
                # Get open positions count
                open_positions = session.query(Position).filter(Position.quantity != 0).count()
                
                # For P&L, we'll simulate some data
                total_pnl = round((total_trades * 0.6 - total_trades * 0.4) * 500, 2)  # Simulated P&L
                win_rate = 60.0  # Simulated win rate
                
                return {
                    'total_trades': total_trades,
                    'total_orders': total_orders,
                    'open_positions': open_positions,
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'sharpe_ratio': 1.5,  # Simulated
                    'max_drawdown': -2.5  # Simulated
                }
        except Exception as e:
            print(f"Error getting performance summary: {e}")
            return {}