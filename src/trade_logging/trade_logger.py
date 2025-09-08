"""
Trade Logger Module
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json
from datastore.database import get_database_manager
from datastore.models import Log


class TradeLogger:
    """Logs detailed trade execution information."""
    
    def __init__(self, db_manager=None):
        """
        Initialize the trade logger.
        
        Args:
            db_manager: Database manager instance (optional)
        """
        self.db_manager = db_manager or get_database_manager()
    
    def log_trade_execution(self, trade_data: Dict[str, Any]) -> Optional[str]:
        """
        Log a trade execution event.
        
        Args:
            trade_data (Dict[str, Any]): Trade execution data
            
        Returns:
            Optional[str]: Log entry ID or None if failed
        """
        return self._log_event(
            module="TRADE_EXECUTION",
            event_type="TRADE_FILLED",
            message=f"Trade executed for {trade_data.get('symbol', 'UNKNOWN')}",
            details=trade_data
        )
    
    def log_order_placement(self, order_data: Dict[str, Any]) -> Optional[str]:
        """
        Log an order placement event.
        
        Args:
            order_data (Dict[str, Any]): Order placement data
            
        Returns:
            Optional[str]: Log entry ID or None if failed
        """
        return self._log_event(
            module="ORDER_MANAGEMENT",
            event_type="ORDER_PLACED",
            message=f"Order placed for {order_data.get('symbol', 'UNKNOWN')}",
            details=order_data
        )
    
    def log_order_modification(self, order_data: Dict[str, Any]) -> Optional[str]:
        """
        Log an order modification event.
        
        Args:
            order_data (Dict[str, Any]): Order modification data
            
        Returns:
            Optional[str]: Log entry ID or None if failed
        """
        return self._log_event(
            module="ORDER_MANAGEMENT",
            event_type="ORDER_MODIFIED",
            message=f"Order modified for {order_data.get('symbol', 'UNKNOWN')}",
            details=order_data
        )
    
    def log_order_cancellation(self, order_data: Dict[str, Any]) -> Optional[str]:
        """
        Log an order cancellation event.
        
        Args:
            order_data (Dict[str, Any]): Order cancellation data
            
        Returns:
            Optional[str]: Log entry ID or None if failed
        """
        return self._log_event(
            module="ORDER_MANAGEMENT",
            event_type="ORDER_CANCELLED",
            message=f"Order cancelled for {order_data.get('symbol', 'UNKNOWN')}",
            details=order_data
        )
    
    def log_position_update(self, position_data: Dict[str, Any]) -> Optional[str]:
        """
        Log a position update event.
        
        Args:
            position_data (Dict[str, Any]): Position update data
            
        Returns:
            Optional[str]: Log entry ID or None if failed
        """
        return self._log_event(
            module="POSITION_MANAGEMENT",
            event_type="POSITION_UPDATED",
            message=f"Position updated for {position_data.get('symbol', 'UNKNOWN')}",
            details=position_data
        )
    
    def log_risk_violation(self, violation_data: Dict[str, Any]) -> Optional[str]:
        """
        Log a risk violation event.
        
        Args:
            violation_data (Dict[str, Any]): Risk violation data
            
        Returns:
            Optional[str]: Log entry ID or None if failed
        """
        return self._log_event(
            module="RISK_MANAGEMENT",
            event_type="RISK_VIOLATION",
            message=f"Risk violation: {violation_data.get('violation_type', 'UNKNOWN')}",
            details=violation_data
        )
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None) -> Optional[str]:
        """
        Log a general system event.
        
        Args:
            event_type (str): Type of event
            message (str): Event message
            details (Dict[str, Any]): Additional details (optional)
            
        Returns:
            Optional[str]: Log entry ID or None if failed
        """
        return self._log_event(
            module="SYSTEM",
            event_type=event_type,
            message=message,
            details=details or {}
        )
    
    def _log_event(self, module: str, event_type: str, message: str, details: Dict[str, Any]) -> Optional[str]:
        """
        Log an event to the database.
        
        Args:
            module (str): Module name
            event_type (str): Event type
            message (str): Event message
            details (Dict[str, Any]): Event details
            
        Returns:
            Optional[str]: Log entry ID or None if failed
        """
        try:
            # Create log entry
            log_entry = Log(
                timestamp=datetime.now(),
                level="INFO",
                module=module,
                message=message,
                details=json.dumps(details) if details else "{}"
            )
            
            # Save to database
            with self.db_manager.get_session() as session:
                session.add(log_entry)
                session.flush()
                log_id = log_entry.id
            
            return str(log_id)
        except Exception as e:
            print(f"Error logging event: {e}")
            return None
    
    def get_trade_logs(self, limit: int = 100) -> list:
        """
        Get recent trade logs.
        
        Args:
            limit (int): Maximum number of logs to retrieve
            
        Returns:
            list: List of trade log entries
        """
        try:
            with self.db_manager.get_session() as session:
                logs = session.query(Log).filter(
                    Log.module.in_(['TRADE_EXECUTION', 'ORDER_MANAGEMENT', 'POSITION_MANAGEMENT'])
                ).order_by(Log.timestamp.desc()).limit(limit).all()
                
                return [{
                    'id': log.id,
                    'timestamp': log.timestamp,
                    'module': log.module,
                    'event_type': log.event_type,
                    'message': log.message,
                    'details': json.loads(log.details) if log.details else {}
                } for log in logs]
        except Exception as e:
            print(f"Error retrieving trade logs: {e}")
            return []
    
    def get_risk_logs(self, limit: int = 100) -> list:
        """
        Get recent risk management logs.
        
        Args:
            limit (int): Maximum number of logs to retrieve
            
        Returns:
            list: List of risk log entries
        """
        try:
            with self.db_manager.get_session() as session:
                logs = session.query(Log).filter(
                    Log.module == 'RISK_MANAGEMENT'
                ).order_by(Log.timestamp.desc()).limit(limit).all()
                
                return [{
                    'id': log.id,
                    'timestamp': log.timestamp,
                    'module': log.module,
                    'event_type': log.event_type,
                    'message': log.message,
                    'details': json.loads(log.details) if log.details else {}
                } for log in logs]
        except Exception as e:
            print(f"Error retrieving risk logs: {e}")
            return []