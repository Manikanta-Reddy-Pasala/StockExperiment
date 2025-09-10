"""
Email Alerting System for Trading System
"""
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime


class EmailAlertingSystem:
    """Handles email alerts for the trading system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize email alerting system.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Email configuration
        self.smtp_host = config.get('email', {}).get('smtp_host', 'smtp.gmail.com')
        self.smtp_port = config.get('email', {}).get('smtp_port', 587)
        self.use_tls = config.get('email', {}).get('use_tls', True)
        self.username = config.get('email', {}).get('username', '')
        self.password = config.get('email', {}).get('password', '')
        
        self.is_configured = bool(self.username and self.password)
        
        if not self.is_configured:
            self.logger.warning("Email alerts not configured - missing username/password")
    
    def send_alert(self, to_email: str, subject: str, message: str,
                   html_message: Optional[str] = None) -> bool:
        """
        Send an email alert.
        
        Args:
            to_email (str): Recipient email address
            subject (str): Email subject
            message (str): Plain text message
            html_message (Optional[str]): HTML message
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.is_configured:
            self.logger.warning("Cannot send email - not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add plain text part
            text_part = MIMEText(message, 'plain')
            msg.attach(text_part)
            
            # Add HTML part if provided
            if html_message:
                html_part = MIMEText(html_message, 'html')
                msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent to {to_email}: {subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_trade_alert(self, to_email: str, user_id: int, symbol: str,
                        action: str, quantity: int, price: float) -> bool:
        """
        Send a trade execution alert.
        
        Args:
            to_email (str): Recipient email address
            user_id (int): User ID
            symbol (str): Stock symbol
            action (str): Trade action (BUY/SELL)
            quantity (int): Number of shares
            price (float): Price per share
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        subject = f"Trade Alert: {action} {quantity} {symbol}"
        
        message = f"""
Trade Execution Alert

User ID: {user_id}
Symbol: {symbol}
Action: {action}
Quantity: {quantity}
Price: ${price:.2f}
Total Value: ${quantity * price:.2f}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is an automated alert from your trading system.
        """.strip()
        
        html_message = f"""
        <html>
        <body>
            <h2>Trade Execution Alert</h2>
            <table border="1" cellpadding="5">
                <tr><td><strong>User ID</strong></td><td>{user_id}</td></tr>
                <tr><td><strong>Symbol</strong></td><td>{symbol}</td></tr>
                <tr><td><strong>Action</strong></td><td>{action}</td></tr>
                <tr><td><strong>Quantity</strong></td><td>{quantity}</td></tr>
                <tr><td><strong>Price</strong></td><td>${price:.2f}</td></tr>
                <tr><td><strong>Total Value</strong></td><td>${quantity * price:.2f}</td></tr>
                <tr><td><strong>Timestamp</strong></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            </table>
            <p><em>This is an automated alert from your trading system.</em></p>
        </body>
        </html>
        """
        
        return self.send_alert(to_email, subject, message, html_message)
    
    def send_risk_alert(self, to_email: str, user_id: int, risk_type: str,
                       message: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a risk alert.
        
        Args:
            to_email (str): Recipient email address
            user_id (int): User ID
            risk_type (str): Type of risk event
            message (str): Risk message
            details (Optional[Dict[str, Any]]): Additional details
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        subject = f"Risk Alert: {risk_type}"
        
        alert_message = f"""
Risk Alert

User ID: {user_id}
Risk Type: {risk_type}
Message: {message}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        if details:
            alert_message += "\n\nDetails:\n"
            for key, value in details.items():
                alert_message += f"{key}: {value}\n"
        
        alert_message += "\n\nThis is an automated risk alert from your trading system."
        
        return self.send_alert(to_email, subject, alert_message)
    
    def send_system_alert(self, to_email: str, alert_type: str, message: str,
                         details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a system alert.
        
        Args:
            to_email (str): Recipient email address
            alert_type (str): Type of system alert
            message (str): Alert message
            details (Optional[Dict[str, Any]]): Additional details
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        subject = f"System Alert: {alert_type}"
        
        alert_message = f"""
System Alert

Alert Type: {alert_type}
Message: {message}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        if details:
            alert_message += "\n\nDetails:\n"
            for key, value in details.items():
                alert_message += f"{key}: {value}\n"
        
        alert_message += "\n\nThis is an automated system alert."
        
        return self.send_alert(to_email, subject, alert_message)
