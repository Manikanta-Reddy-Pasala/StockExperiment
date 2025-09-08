"""
Compliance Logger for the Automated Trading System
"""
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional
from datastore.database import get_database_manager


class ComplianceLogger:
    """Manages immutable audit trail for regulatory compliance."""
    
    def __init__(self, database_manager=None):
        """
        Initialize the compliance logger.
        
        Args:
            database_manager: Database manager instance (optional)
        """
        self.db_manager = database_manager or get_database_manager()
    
    def log_event(self, module: str, event_type: str, message: str, 
                  details: Dict[str, Any] = None, user: str = None) -> str:
        """
        Log a compliance event.
        
        Args:
            module (str): Module where event occurred
            event_type (str): Type of event
            message (str): Event message
            details (Dict[str, Any], optional): Additional details
            user (str, optional): User associated with event
            
        Returns:
            str: Event ID (hash of the event)
        """
        timestamp = datetime.now()
        
        # Create event data
        event_data = {
            'timestamp': timestamp.isoformat(),
            'module': module,
            'event_type': event_type,
            'message': message,
            'details': details or {},
            'user': user or 'system'
        }
        
        # Create a hash of the event for immutable identification
        event_json = json.dumps(event_data, sort_keys=True)
        event_id = hashlib.sha256(event_json.encode('utf-8')).hexdigest()
        
        # Store in database
        with self.db_manager.get_session() as session:
            from datastore.models import Log
            log_entry = Log(
                timestamp=timestamp,
                level='COMPLIANCE',
                module=module,
                message=f"{event_type}: {message}",
                details=json.dumps(details) if details else None
            )
            session.add(log_entry)
        
        return event_id
    
    def log_order_event(self, order_id: str, event_type: str, status: str, 
                        message: str, details: Dict[str, Any] = None) -> str:
        """
        Log an order-related compliance event.
        
        Args:
            order_id (str): Order ID
            event_type (str): Type of event
            status (str): Order status
            message (str): Event message
            details (Dict[str, Any], optional): Additional details
            
        Returns:
            str: Event ID
        """
        event_details = {
            'order_id': order_id,
            'status': status
        }
        
        if details:
            event_details.update(details)
        
        return self.log_event(
            module='ORDER',
            event_type=event_type,
            message=message,
            details=event_details
        )
    
    def log_strategy_event(self, strategy_name: str, event_type: str, 
                           message: str, details: Dict[str, Any] = None) -> str:
        """
        Log a strategy-related compliance event.
        
        Args:
            strategy_name (str): Strategy name
            event_type (str): Type of event
            message (str): Event message
            details (Dict[str, Any], optional): Additional details
            
        Returns:
            str: Event ID
        """
        event_details = {
            'strategy_name': strategy_name
        }
        
        if details:
            event_details.update(details)
        
        return self.log_event(
            module='STRATEGY',
            event_type=event_type,
            message=message,
            details=event_details
        )
    
    def log_risk_event(self, event_type: str, message: str, 
                       details: Dict[str, Any] = None) -> str:
        """
        Log a risk-related compliance event.
        
        Args:
            event_type (str): Type of event
            message (str): Event message
            details (Dict[str, Any], optional): Additional details
            
        Returns:
            str: Event ID
        """
        return self.log_event(
            module='RISK',
            event_type=event_type,
            message=message,
            details=details
        )
    
    def log_system_event(self, event_type: str, message: str, 
                         details: Dict[str, Any] = None) -> str:
        """
        Log a system-related compliance event.
        
        Args:
            event_type (str): Type of event
            message (str): Event message
            details (Dict[str, Any], optional): Additional details
            
        Returns:
            str: Event ID
        """
        return self.log_event(
            module='SYSTEM',
            event_type=event_type,
            message=message,
            details=details
        )
    
    def get_audit_trail(self, start_time: datetime = None, 
                        end_time: datetime = None, module: str = None) -> list:
        """
        Get audit trail of compliance events.
        
        Args:
            start_time (datetime, optional): Start time filter
            end_time (datetime, optional): End time filter
            module (str, optional): Module filter
            
        Returns:
            list: List of compliance events
        """
        with self.db_manager.get_session() as session:
            from datastore.models import Log
            
            query = session.query(Log).filter(Log.level == 'COMPLIANCE')
            
            if start_time:
                query = query.filter(Log.timestamp >= start_time)
            
            if end_time:
                query = query.filter(Log.timestamp <= end_time)
            
            if module:
                query = query.filter(Log.module == module)
            
            logs = query.order_by(Log.timestamp.desc()).all()
            
            audit_trail = []
            for log in logs:
                audit_trail.append({
                    'timestamp': log.timestamp.isoformat() if log.timestamp else None,
                    'module': log.module,
                    'message': log.message,
                    'details': json.loads(log.details) if log.details else None
                })
            
            return audit_trail
    
    def get_order_audit_trail(self, order_id: str) -> list:
        """
        Get audit trail for a specific order.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            list: List of order-related compliance events
        """
        with self.db_manager.get_session() as session:
            from datastore.models import Log
            
            logs = session.query(Log).filter(
                Log.level == 'COMPLIANCE',
                Log.module == 'ORDER',
                Log.details.like(f'%{order_id}%')
            ).order_by(Log.timestamp.desc()).all()
            
            audit_trail = []
            for log in logs:
                audit_trail.append({
                    'timestamp': log.timestamp.isoformat() if log.timestamp else None,
                    'message': log.message,
                    'details': json.loads(log.details) if log.details else None
                })
            
            return audit_trail
    
    def generate_compliance_report(self, start_time: datetime, 
                                   end_time: datetime) -> Dict[str, Any]:
        """
        Generate a compliance report for a time period.
        
        Args:
            start_time (datetime): Start time
            end_time (datetime): End time
            
        Returns:
            Dict[str, Any]: Compliance report
        """
        audit_trail = self.get_audit_trail(start_time, end_time)
        
        # Group events by module
        module_events = {}
        for event in audit_trail:
            module = event['module']
            if module not in module_events:
                module_events[module] = []
            module_events[module].append(event)
        
        report = {
            'report_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'total_events': len(audit_trail),
            'events_by_module': module_events,
            'generated_at': datetime.now().isoformat()
        }
        
        return report


# Example usage
if __name__ == "__main__":
    # Create compliance logger
    compliance_logger = ComplianceLogger()
    
    # Log some events
    event_id1 = compliance_logger.log_order_event(
        order_id="ORDER123",
        event_type="ORDER_PLACED",
        status="PLACED",
        message="Order placed successfully",
        details={"symbol": "INFY", "quantity": 10, "price": 1500.0}
    )
    
    event_id2 = compliance_logger.log_strategy_event(
        strategy_name="MomentumStrategy",
        event_type="STRATEGY_EXECUTED",
        message="Strategy executed for stock selection",
        details={"selected_stocks": ["INFY", "RELIANCE", "TCS"]}
    )
    
    event_id3 = compliance_logger.log_risk_event(
        event_type="RISK_CHECK_PASSED",
        message="Trade approved within risk limits",
        details={"symbol": "INFY", "quantity": 10, "risk_amount": 1500.0}
    )
    
    print(f"Logged events with IDs: {event_id1}, {event_id2}, {event_id3}")
    
    # Get audit trail
    audit_trail = compliance_logger.get_audit_trail()
    print(f"Audit trail has {len(audit_trail)} events")
    
    # Get order audit trail
    order_trail = compliance_logger.get_order_audit_trail("ORDER123")
    print(f"Order audit trail has {len(order_trail)} events")