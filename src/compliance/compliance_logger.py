"""
Compliance Logger for Trading System
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime


class ComplianceLogger:
    """Logs compliance-related events and activities."""
    
    def __init__(self, db_manager):
        """
        Initialize compliance logger.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def log_system_event(self, event_type: str, message: str, 
                        details: Optional[Dict[str, Any]] = None):
        """
        Log a system event.
        
        Args:
            event_type (str): Type of event
            message (str): Event message
            details (Optional[Dict[str, Any]]): Additional details
        """
        try:
            # In a real implementation, this would store in database
            # For now, we'll just log to the logger
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'message': message,
                'details': details or {}
            }
            
            self.logger.info(f"COMPLIANCE: {event_type} - {message}")
            if details:
                self.logger.debug(f"Details: {details}")
                
        except Exception as e:
            self.logger.error(f"Error logging compliance event: {e}")
    
    def log_trade_event(self, user_id: int, symbol: str, action: str,
                       quantity: int, price: float, details: Optional[Dict[str, Any]] = None):
        """
        Log a trade event.
        
        Args:
            user_id (int): User ID
            symbol (str): Stock symbol
            action (str): Trade action (BUY/SELL)
            quantity (int): Number of shares
            price (float): Price per share
            details (Optional[Dict[str, Any]]): Additional details
        """
        try:
            event_details = {
                'user_id': user_id,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'trade_value': quantity * price
            }
            
            if details:
                event_details.update(details)
            
            self.log_system_event(
                event_type="TRADE_EXECUTED",
                message=f"Trade executed: {action} {quantity} {symbol} @ {price}",
                details=event_details
            )
            
        except Exception as e:
            self.logger.error(f"Error logging trade event: {e}")
    
    def log_risk_event(self, user_id: int, risk_type: str, message: str,
                      details: Optional[Dict[str, Any]] = None):
        """
        Log a risk-related event.
        
        Args:
            user_id (int): User ID
            risk_type (str): Type of risk event
            message (str): Risk message
            details (Optional[Dict[str, Any]]): Additional details
        """
        try:
            event_details = {
                'user_id': user_id,
                'risk_type': risk_type
            }
            
            if details:
                event_details.update(details)
            
            self.log_system_event(
                event_type="RISK_EVENT",
                message=f"Risk event for user {user_id}: {message}",
                details=event_details
            )
            
        except Exception as e:
            self.logger.error(f"Error logging risk event: {e}")
    
    def log_user_activity(self, user_id: int, activity: str,
                         details: Optional[Dict[str, Any]] = None):
        """
        Log user activity.
        
        Args:
            user_id (int): User ID
            activity (str): Activity description
            details (Optional[Dict[str, Any]]): Additional details
        """
        try:
            event_details = {
                'user_id': user_id,
                'activity': activity
            }
            
            if details:
                event_details.update(details)
            
            self.log_system_event(
                event_type="USER_ACTIVITY",
                message=f"User {user_id} activity: {activity}",
                details=event_details
            )
            
        except Exception as e:
            self.logger.error(f"Error logging user activity: {e}")
