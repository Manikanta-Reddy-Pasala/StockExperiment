"""
Email Alert System for Stock Picks and Notifications
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
import logging
from datetime import datetime
import os
from datastore.database import get_database_manager
from datastore.models import User

logger = logging.getLogger(__name__)


class EmailService:
    """Manages email alerts for stock picks and notifications."""

    def __init__(self, config: Dict = None):
        """Initialize email alert manager."""
        if config:
            self.smtp_server = config.get('email', {}).get('smtp_host', 'smtp.gmail.com')
            self.smtp_port = int(config.get('email', {}).get('smtp_port', 587))
            self.sender_email = config.get('email', {}).get('username', '')
            self.sender_password = config.get('email', {}).get('password', '')
        else:
            self.smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
            self.smtp_port = int(os.environ.get('SMTP_PORT', '587'))
            self.sender_email = os.environ.get('SENDER_EMAIL', '')
            self.sender_password = os.environ.get('SENDER_PASSWORD', '')
        
        self.db_manager = get_database_manager()
        self._test_email_config()

    def _test_email_config(self):
        """Test email configuration."""
        if not self.sender_email or not self.sender_password:
            logger.warning("Email configuration incomplete. Alerts will be logged only.")
            self.email_enabled = False
        else:
            self.email_enabled = True
            logger.info("Email alerts enabled")

    def send_alert(self, recipient_email: str, subject: str, message: str, html_message: Optional[str] = None) -> bool:
        """Sends a generic email alert."""
        if not self.email_enabled:
            logger.info(f"Email disabled - would send to {recipient_email}: {subject}")
            return True

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender_email
            msg["To"] = recipient_email

            # Add text and HTML parts
            text_part = MIMEText(message, "plain")
            msg.attach(text_part)

            if html_message:
                html_part = MIMEText(html_message, 'html')
                msg.attach(html_part)

            # Create secure connection and send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient_email, msg.as_string())

            return True

        except Exception as e:
            logger.error(f"Error sending email to {recipient_email}: {e}")
            return False

    def send_stock_pick_alert(self, user_id: int, stock_data: Dict, strategy_name: str,
                             recommendation: str = "BUY") -> bool:
        """
        Send stock pick alert to user.
        """
        try:
            user = self._get_user(user_id)
            if not user or not user.email:
                logger.error(f"User {user_id} not found or has no email.")
                return False

            if not self._user_wants_email_alerts(user_id):
                logger.info(f"User {user_id} has disabled email alerts")
                return True

            subject = f"🎯 Stock Pick Alert: {stock_data['symbol']} - {recommendation}"
            html_content = self._create_stock_pick_email_html(stock_data, strategy_name, recommendation)
            text_content = self._create_stock_pick_email_text(stock_data, strategy_name, recommendation)

            success = self.send_alert(user.email, subject, text_content, html_content)

            if success:
                logger.info(f"Stock pick alert sent to {user.email} for {stock_data['symbol']}")
            else:
                logger.error(f"Failed to send stock pick alert to {user.email}")

            return success

        except Exception as e:
            logger.error(f"Error sending stock pick alert: {e}")
            return False

    def _get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        try:
            with self.db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if user:
                    session.expunge(user)
                return user
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            return None

    def _user_wants_email_alerts(self, user_id: int) -> bool:
        """Check if user wants email alerts."""
        # This can be expanded to check user preferences in the database
        return True

    def _create_stock_pick_email_html(self, stock_data: Dict, strategy_name: str,
                                    recommendation: str) -> str:
        """Create HTML content for stock pick email."""
        color = "#28a745" if recommendation == "BUY" else "#dc3545" if recommendation == "SELL" else "#ffc107"

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px; }}
                .stock-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .metric {{ display: flex; justify-content: space-between; margin-bottom: 8px; }}
                .metric-label {{ font-weight: bold; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>🎯 Stock Pick Alert</h2>
                    <h3>{stock_data['symbol']} - {recommendation}</h3>
                </div>
                
                <div class="stock-info">
                    <h3>{stock_data['name']}</h3>
                    <div class="metric">
                        <span class="metric-label">Current Price:</span>
                        <span>₹{stock_data['current_price']:.2f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Market Cap:</span>
                        <span>₹{stock_data['market_cap']:,.0f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Sector:</span>
                        <span>{stock_data['sector']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">PE Ratio:</span>
                        <span>{stock_data.get('pe_ratio', 'N/A')}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Strategy:</span>
                        <span>{strategy_name}</span>
                    </div>
                </div>
                
                <div class="footer">
                    <p>This alert was generated by your automated trading system.</p>
                    <p>Please do your own research before making investment decisions.</p>
                </div>
            </div>
        </body>
        </html>
        """

    def _create_stock_pick_email_text(self, stock_data: Dict, strategy_name: str,
                                    recommendation: str) -> str:
        """Create text content for stock pick email."""
        return f"""
Stock Pick Alert: {stock_data['symbol']} - {recommendation}

Stock: {stock_data['name']}
Current Price: ₹{stock_data['current_price']:.2f}
Market Cap: ₹{stock_data['market_cap']:,.0f}
Sector: {stock_data['sector']}
PE Ratio: {stock_data.get('pe_ratio', 'N/A')}
Strategy: {strategy_name}

This alert was generated by your automated trading system.
Please do your own research before making investment decisions.
        """

_email_service = None

def get_email_service(config: Dict = None) -> "EmailService":
    """Get a global instance of the email service."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService(config)
    return _email_service
