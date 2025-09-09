"""
Email Alerting System for the Automated Trading System
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta


class EmailAlertingSystem:
    """Manages email notifications for critical events and reports."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the email alerting system.
        
        Args:
            config (Dict[str, Any]): Email configuration parameters
        """
        self.config = config.get('email', {})
        self.smtp_host = self.config.get('smtp_host', 'smtp.gmail.com')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.use_tls = self.config.get('use_tls', True)
        self.sender_email = self.config.get('sender_email', '')
        self.sender_password = self.config.get('sender_password', '')
        self.recipients = self.config.get('recipients', [])
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Alert deduplication
        self.sent_alerts = {}  # alert_key -> timestamp
        self.alert_cooldown = timedelta(minutes=5)  # Deduplicate alerts within 5 minutes
    
    def send_email(self, subject: str, body: str, recipients: List[str] = None, 
                   is_html: bool = False) -> bool:
        """
        Send an email.
        
        Args:
            subject (str): Email subject
            body (str): Email body
            recipients (List[str], optional): Recipients (overrides default)
            is_html (bool): Whether the body is HTML
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        if not recipients:
            recipients = self.recipients
        
        if not recipients or not self.sender_email:
            self.logger.warning("Cannot send email: No recipients or sender email configured")
            return False
        
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = ", ".join(recipients)
            
            # Create body
            if is_html:
                body_part = MIMEText(body, "html")
            else:
                body_part = MIMEText(body, "plain")
            
            message.attach(body_part)
            
            # Create SMTP session
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                if self.sender_password:
                    server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipients, message.as_string())
            
            self.logger.info(f"Email sent successfully to {', '.join(recipients)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def send_alert(self, alert_type: str, message: str, severity: str = "info",
                   recipients: List[str] = None, metadata: Dict[str, Any] = None,
                   deduplicate: bool = True) -> bool:
        """
        Send an alert email.
        
        Args:
            alert_type (str): Type of alert
            message (str): Alert message
            severity (str): Severity level ('info', 'warning', 'error', 'critical')
            recipients (List[str], optional): Recipients (overrides default)
            metadata (Dict[str, Any], optional): Additional metadata
            deduplicate (bool): Whether to deduplicate alerts
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        # Check if we should deduplicate this alert
        if deduplicate:
            alert_key = f"{alert_type}:{message}"
            now = datetime.now()
            
            if alert_key in self.sent_alerts:
                last_sent = self.sent_alerts[alert_key]
                if now - last_sent < self.alert_cooldown:
                    self.logger.info(f"Alert deduplicated: {alert_type}")
                    return True  # Consider deduplicated alerts as "sent"
            
            # Update last sent time
            self.sent_alerts[alert_key] = now
        
        # Format subject based on severity
        severity_prefix = {
            'info': '[INFO]',
            'warning': '[WARNING]',
            'error': '[ERROR]',
            'critical': '[CRITICAL]'
        }.get(severity, '[ALERT]')
        
        subject = f"{severity_prefix} {alert_type}: {message}"
        
        # Format body
        body = f"""
Alert Type: {alert_type}
Severity: {severity}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Message: {message}
"""
        
        if metadata:
            body += "\nMetadata:\n"
            for key, value in metadata.items():
                body += f"  {key}: {value}\n"
        
        return self.send_email(subject, body, recipients)
    
    def send_order_execution_alert(self, order_id: str, symbol: str, 
                                   transaction_type: str, quantity: int, 
                                   price: float, status: str) -> bool:
        """
        Send an order execution alert.
        
        Args:
            order_id (str): Order ID
            symbol (str): Trading symbol
            transaction_type (str): BUY or SELL
            quantity (int): Order quantity
            price (float): Execution price
            status (str): Order status
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        subject = f"Order {status}: {transaction_type} {quantity} {symbol}"
        
        body = f"""
Order Execution Alert
====================

Order ID: {order_id}
Symbol: {symbol}
Transaction: {transaction_type}
Quantity: {quantity}
Price: ₹{price:.2f}
Status: {status}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_email(subject, body)
    
    def send_risk_limit_alert(self, limit_type: str, current_value: float, 
                              limit_value: float, message: str) -> bool:
        """
        Send a risk limit breach alert.
        
        Args:
            limit_type (str): Type of limit breached
            current_value (float): Current value
            limit_value (float): Limit value
            message (str): Alert message
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        subject = f"Risk Limit Breach: {limit_type}"
        
        body = f"""
Risk Limit Breach Alert
======================

Limit Type: {limit_type}
Current Value: {current_value}
Limit Value: {limit_value}
Message: {message}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_email(subject, body)
    
    def send_daily_summary(self, date: str, pnl: float, num_trades: int, 
                          win_rate: float) -> bool:
        """
        Send a daily summary report.
        
        Args:
            date (str): Date of report
            pnl (float): Profit and loss
            num_trades (int): Number of trades
            win_rate (float): Win rate percentage
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        subject = f"Daily Summary Report - {date}"
        
        body = f"""
Daily Trading Summary
====================

Date: {date}
P&L: ₹{pnl:.2f}
Number of Trades: {num_trades}
Win Rate: {win_rate:.2f}%

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_email(subject, body)
    
    def send_system_health_alert(self, component: str, status: str, 
                                 details: str = "") -> bool:
        """
        Send a system health alert.
        
        Args:
            component (str): Component name
            status (str): Component status
            details (str): Additional details
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        subject = f"System Health Alert: {component} is {status}"
        
        body = f"""
System Health Alert
==================

Component: {component}
Status: {status}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if details:
            body += f"\nDetails: {details}\n"
        
        return self.send_email(subject, body)
    
    def send_test_email(self, recipient: str = None) -> bool:
        """
        Send a test email to verify configuration.
        
        Args:
            recipient (str, optional): Test recipient (overrides default)
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        subject = "Trading System Email Configuration Test"
        
        body = f"""
Trading System Email Test
========================

This is a test email to verify that the email alerting system is configured correctly.

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

If you received this email, the email configuration is working properly.
"""
        
        recipients = [recipient] if recipient else None
        return self.send_email(subject, body, recipients)
    
    def cleanup_old_alerts(self, max_age: timedelta = timedelta(hours=1)):
        """
        Clean up old alert records to prevent memory bloat.
        
        Args:
            max_age (timedelta): Maximum age of alert records to keep
        """
        now = datetime.now()
        old_alerts = [
            key for key, timestamp in self.sent_alerts.items()
            if now - timestamp > max_age
        ]
        
        for key in old_alerts:
            del self.sent_alerts[key]
        
        self.logger.info(f"Cleaned up {len(old_alerts)} old alert records")


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'email': {
            'smtp_host': 'smtp.gmail.com',
            'smtp_port': 587,
            'use_tls': True,
            'sender_email': 'trading.system@example.com',
            'sender_password': 'your-app-password',
            'recipients': ['trader@example.com', 'risk.manager@example.com']
        }
    }
    
    # Create email alerting system
    email_system = EmailAlertingSystem(config)
    
    # Send a test alert
    email_system.send_alert(
        alert_type="TEST_ALERT",
        message="This is a test alert",
        severity="info",
        metadata={"test_key": "test_value"}
    )
    
    # Send an order execution alert
    email_system.send_order_execution_alert(
        order_id="ORDER123",
        symbol="INFY",
        transaction_type="BUY",
        quantity=10,
        price=1500.0,
        status="FILLED"
    )
    
    # Send a risk limit alert
    email_system.send_risk_limit_alert(
        limit_type="DAILY_LOSS_LIMIT",
        current_value=15000.0,
        limit_value=20000.0,
        message="Daily loss limit approaching"
    )
    
    # Send a daily summary
    email_system.send_daily_summary(
        date="2023-01-01",
        pnl=25000.0,
        num_trades=25,
        win_rate=72.0
    )
    
    # Send a system health alert
    email_system.send_system_health_alert(
        component="FyersConnector",
        status="DISCONNECTED",
        details="Connection to Fyers API lost"
    )
    
    # Send a test email
    email_system.send_test_email("admin@example.com")