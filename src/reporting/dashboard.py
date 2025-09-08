"""
Dashboard and Reporting Module for the Automated Trading System
"""
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json


class DashboardReporter:
    """Generates dashboard metrics and reports."""
    
    def __init__(self, database_manager):
        """
        Initialize the dashboard reporter.
        
        Args:
            database_manager: Database manager instance
        """
        self.db_manager = database_manager
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """
        Get real-time dashboard metrics.
        
        Returns:
            Dict[str, Any]: Real-time metrics
        """
        metrics = {}
        
        with self.db_manager.get_session() as session:
            # Get current positions
            from datastore.models import Position
            positions = session.query(Position).all()
            
            metrics['positions'] = []
            total_pnl = 0
            for pos in positions:
                pos_data = {
                    'symbol': pos.tradingsymbol,
                    'quantity': pos.quantity,
                    'avg_price': pos.average_price,
                    'last_price': pos.last_price,
                    'pnl': pos.pnl,
                    'pnl_percentage': (pos.pnl / (pos.average_price * abs(pos.quantity)) * 100) if pos.average_price and pos.quantity else 0
                }
                metrics['positions'].append(pos_data)
                total_pnl += pos.pnl
            
            metrics['total_pnl'] = total_pnl
            
            # Get recent orders
            from datastore.models import Order
            recent_orders = session.query(Order).order_by(Order.created_at.desc()).limit(10).all()
            
            metrics['recent_orders'] = []
            for order in recent_orders:
                order_data = {
                    'order_id': order.order_id,
                    'symbol': order.tradingsymbol,
                    'type': order.order_type,
                    'transaction': order.transaction_type,
                    'quantity': order.quantity,
                    'filled': order.filled_quantity,
                    'status': order.order_status,
                    'price': order.price,
                    'created_at': order.created_at.isoformat() if order.created_at else None
                }
                metrics['recent_orders'].append(order_data)
        
        return metrics
    
    def get_performance_report(self, period: str = 'daily') -> Dict[str, Any]:
        """
        Get performance report for a specified period.
        
        Args:
            period (str): Report period ('daily', 'weekly', 'monthly', 'yearly')
            
        Returns:
            Dict[str, Any]: Performance report
        """
        report = {}
        
        # Determine date range based on period
        end_date = datetime.now()
        if period == 'daily':
            start_date = end_date - timedelta(days=1)
        elif period == 'weekly':
            start_date = end_date - timedelta(weeks=1)
        elif period == 'monthly':
            start_date = end_date - timedelta(days=30)
        elif period == 'yearly':
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=1)
        
        with self.db_manager.get_session() as session:
            # Get trades for the period
            from datastore.models import Trade
            trades = session.query(Trade).filter(
                Trade.trade_time >= start_date,
                Trade.trade_time <= end_date
            ).all()
            
            # Calculate P&L
            total_pnl = 0
            winning_trades = 0
            losing_trades = 0
            total_trades = len(trades)
            
            for trade in trades:
                # Simplified P&L calculation
                # In a real implementation, this would be more complex
                if trade.transaction_type == 'BUY':
                    pnl = (trade.price - trade.order_price) * trade.quantity
                else:
                    pnl = (trade.order_price - trade.price) * trade.quantity
                
                total_pnl += pnl
                if pnl > 0:
                    winning_trades += 1
                elif pnl < 0:
                    losing_trades += 1
            
            report['period'] = period
            report['start_date'] = start_date.isoformat()
            report['end_date'] = end_date.isoformat()
            report['total_pnl'] = total_pnl
            report['total_trades'] = total_trades
            report['winning_trades'] = winning_trades
            report['losing_trades'] = losing_trades
            report['win_rate'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            report['average_win'] = 0  # Simplified
            report['average_loss'] = 0  # Simplified
            report['profit_factor'] = 0  # Simplified
        
        return report
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics.
        
        Returns:
            Dict[str, Any]: Strategy performance metrics
        """
        metrics = {}
        
        with self.db_manager.get_session() as session:
            # Get strategies
            from datastore.models import Strategy
            strategies = session.query(Strategy).all()
            
            metrics['strategies'] = []
            for strategy in strategies:
                strat_data = {
                    'name': strategy.name,
                    'description': strategy.description,
                    'is_active': strategy.is_active,
                    'created_at': strategy.created_at.isoformat() if strategy.created_at else None
                }
                metrics['strategies'].append(strat_data)
        
        return metrics
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get risk management metrics.
        
        Returns:
            Dict[str, Any]: Risk metrics
        """
        metrics = {}
        
        with self.db_manager.get_session() as session:
            # Get positions for exposure analysis
            from datastore.models import Position
            positions = session.query(Position).all()
            
            total_exposure = 0
            sector_exposure = {}
            
            for pos in positions:
                exposure = abs(pos.average_price * pos.quantity)
                total_exposure += exposure
                
                # Simplified sector exposure (in a real implementation, 
                # you would have sector information for each symbol)
                sector = "Unknown"
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0
                sector_exposure[sector] += exposure
            
            metrics['total_exposure'] = total_exposure
            metrics['sector_exposure'] = sector_exposure
            metrics['positions_count'] = len(positions)
        
        return metrics
    
    def generate_eod_report(self) -> Dict[str, Any]:
        """
        Generate end-of-day report.
        
        Returns:
            Dict[str, Any]: EOD report
        """
        report = {}
        
        # Include performance report
        report['performance'] = self.get_performance_report('daily')
        
        # Include risk metrics
        report['risk'] = self.get_risk_metrics()
        
        # Include strategy performance
        report['strategy'] = self.get_strategy_performance()
        
        # Add timestamp
        report['generated_at'] = datetime.now().isoformat()
        
        return report
    
    def get_momentum_insights(self) -> Dict[str, Any]:
        """
        Get momentum stock selection insights.
        
        Returns:
            Dict[str, Any]: Momentum insights
        """
        insights = {}
        
        with self.db_manager.get_session() as session:
            # Get recent market data for momentum analysis
            from datastore.models import MarketData
            recent_data = session.query(MarketData).order_by(
                MarketData.timestamp.desc()
            ).limit(100).all()
            
            # Convert to DataFrame for analysis
            data = []
            for md in recent_data:
                data.append({
                    'symbol': md.instrument.tradingsymbol if md.instrument else 'Unknown',
                    'timestamp': md.timestamp,
                    'close_price': md.close_price,
                    'volume': md.volume
                })
            
            df = pd.DataFrame(data)
            
            # Calculate momentum metrics (simplified)
            if not df.empty:
                df = df.sort_values('timestamp')
                df['price_change'] = df.groupby('symbol')['close_price'].pct_change()
                df['volume_change'] = df.groupby('symbol')['volume'].pct_change()
                
                # Get top momentum stocks
                latest_data = df.groupby('symbol').last().sort_values('price_change', ascending=False)
                top_momentum = latest_data.head(10)[['price_change', 'volume_change']].to_dict('index')
                insights['top_momentum'] = top_momentum
        
        return insights


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the alert manager.
        
        Args:
            config (Dict[str, Any]): Configuration parameters
        """
        self.config = config
        self.alerts = []
    
    def add_alert(self, alert_type: str, message: str, severity: str = "info", 
                  recipient: str = None, metadata: Dict[str, Any] = None):
        """
        Add an alert.
        
        Args:
            alert_type (str): Type of alert
            message (str): Alert message
            severity (str): Severity level ('info', 'warning', 'error', 'critical')
            recipient (str): Recipient of the alert
            metadata (Dict[str, Any]): Additional metadata
        """
        alert = {
            'id': len(self.alerts) + 1,
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity,
            'recipient': recipient,
            'metadata': metadata or {},
            'status': 'new'
        }
        
        self.alerts.append(alert)
        return alert['id']
    
    def get_alerts(self, severity: str = None, status: str = None) -> List[Dict[str, Any]]:
        """
        Get alerts with optional filtering.
        
        Args:
            severity (str, optional): Filter by severity
            status (str, optional): Filter by status
            
        Returns:
            List[Dict[str, Any]]: List of alerts
        """
        filtered_alerts = self.alerts
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a['severity'] == severity]
        
        if status:
            filtered_alerts = [a for a in filtered_alerts if a['status'] == status]
        
        return filtered_alerts
    
    def mark_alert_as_read(self, alert_id: int) -> bool:
        """
        Mark an alert as read.
        
        Args:
            alert_id (int): Alert ID
            
        Returns:
            bool: True if alert was found and updated, False otherwise
        """
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['status'] = 'read'
                return True
        return False
    
    def get_unread_alerts_count(self) -> int:
        """
        Get count of unread alerts.
        
        Returns:
            int: Count of unread alerts
        """
        return len([a for a in self.alerts if a['status'] == 'new'])
    
    def clear_alerts(self, severity: str = None):
        """
        Clear alerts with optional severity filtering.
        
        Args:
            severity (str, optional): Filter by severity
        """
        if severity:
            self.alerts = [a for a in self.alerts if a['severity'] != severity]
        else:
            self.alerts = []