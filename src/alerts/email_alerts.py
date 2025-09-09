"""
Email Alert System for Stock Picks and Notifications
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Optional
import logging
from datetime import datetime
import os
from datastore.database import get_database_manager
from datastore.models import User

logger = logging.getLogger(__name__)


class EmailAlertManager:
    """Manages email alerts for stock picks and notifications."""
    
    def __init__(self):
        """Initialize email alert manager."""
        self.smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.environ.get('SMTP_PORT', '587'))
        self.sender_email = os.environ.get('SENDER_EMAIL', '')
        self.sender_password = os.environ.get('SENDER_PASSWORD', '')
        self.db_manager = get_database_manager()
        
        # Test email configuration
        self._test_email_config()
    
    def _test_email_config(self):
        """Test email configuration."""
        if not self.sender_email or not self.sender_password:
            logger.warning("Email configuration incomplete. Alerts will be logged only.")
            self.email_enabled = False
        else:
            self.email_enabled = True
            logger.info("Email alerts enabled")
    
    def send_stock_pick_alert(self, user_id: int, stock_data: Dict, strategy_name: str, 
                             recommendation: str = "BUY") -> bool:
        """
        Send stock pick alert to user.
        
        Args:
            user_id (int): User ID
            stock_data (Dict): Stock data
            strategy_name (str): Strategy that selected the stock
            recommendation (str): BUY/SELL/HOLD recommendation
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            # Get user details
            user = self._get_user(user_id)
            if not user:
                logger.error(f"User {user_id} not found")
                return False
            
            # Check if user wants email alerts
            if not self._user_wants_email_alerts(user_id):
                logger.info(f"User {user_id} has disabled email alerts")
                return True
            
            # Create email content
            subject = f"🎯 Stock Pick Alert: {stock_data['symbol']} - {recommendation}"
            html_content = self._create_stock_pick_email_html(stock_data, strategy_name, recommendation)
            text_content = self._create_stock_pick_email_text(stock_data, strategy_name, recommendation)
            
            # Send email
            success = self._send_email(user.email, subject, html_content, text_content)
            
            if success:
                logger.info(f"Stock pick alert sent to {user.email} for {stock_data['symbol']}")
            else:
                logger.error(f"Failed to send stock pick alert to {user.email}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending stock pick alert: {e}")
            return False
    
    def send_portfolio_alert(self, user_id: int, portfolio_data: Dict, alert_type: str) -> bool:
        """
        Send portfolio alert to user.
        
        Args:
            user_id (int): User ID
            portfolio_data (Dict): Portfolio data
            alert_type (str): Type of alert (performance, risk, etc.)
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            user = self._get_user(user_id)
            if not user:
                return False
            
            if not self._user_wants_email_alerts(user_id):
                return True
            
            subject = f"📊 Portfolio Alert: {alert_type.title()}"
            html_content = self._create_portfolio_alert_email_html(portfolio_data, alert_type)
            text_content = self._create_portfolio_alert_email_text(portfolio_data, alert_type)
            
            success = self._send_email(user.email, subject, html_content, text_content)
            
            if success:
                logger.info(f"Portfolio alert sent to {user.email}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending portfolio alert: {e}")
            return False
    
    def send_strategy_performance_alert(self, user_id: int, strategy_results: Dict) -> bool:
        """
        Send strategy performance alert to user.
        
        Args:
            user_id (int): User ID
            strategy_results (Dict): Strategy performance results
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            user = self._get_user(user_id)
            if not user:
                return False
            
            if not self._user_wants_email_alerts(user_id):
                return True
            
            subject = "📈 Strategy Performance Report"
            html_content = self._create_strategy_performance_email_html(strategy_results)
            text_content = self._create_strategy_performance_email_text(strategy_results)
            
            success = self._send_email(user.email, subject, html_content, text_content)
            
            if success:
                logger.info(f"Strategy performance alert sent to {user.email}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending strategy performance alert: {e}")
            return False
    
    def send_daily_summary(self, user_id: int, daily_data: Dict) -> bool:
        """
        Send daily summary email to user.
        
        Args:
            user_id (int): User ID
            daily_data (Dict): Daily trading data
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            user = self._get_user(user_id)
            if not user:
                return False
            
            if not self._user_wants_email_alerts(user_id):
                return True
            
            subject = f"📋 Daily Trading Summary - {datetime.now().strftime('%Y-%m-%d')}"
            html_content = self._create_daily_summary_email_html(daily_data)
            text_content = self._create_daily_summary_email_text(daily_data)
            
            success = self._send_email(user.email, subject, html_content, text_content)
            
            if success:
                logger.info(f"Daily summary sent to {user.email}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
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
        try:
            with self.db_manager.get_session() as session:
                # Check user's email alert preference
                # This would be stored in user settings or configuration
                # For now, assume all users want alerts
                return True
        except Exception as e:
            logger.error(f"Error checking user email preferences: {e}")
            return False
    
    def _send_email(self, recipient_email: str, subject: str, html_content: str, 
                   text_content: str) -> bool:
        """Send email to recipient."""
        if not self.email_enabled:
            logger.info(f"Email disabled - would send to {recipient_email}: {subject}")
            return True
        
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = recipient_email
            
            # Add text and HTML parts
            text_part = MIMEText(text_content, "plain")
            html_part = MIMEText(html_content, "html")
            
            message.attach(text_part)
            message.attach(html_part)
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient_email, message.as_string())
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email to {recipient_email}: {e}")
            return False
    
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
    
    def _create_portfolio_alert_email_html(self, portfolio_data: Dict, alert_type: str) -> str:
        """Create HTML content for portfolio alert email."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .header {{ background-color: #007bff; color: white; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px; }}
                .alert-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>📊 Portfolio Alert</h2>
                    <h3>{alert_type.title()}</h3>
                </div>
                
                <div class="alert-info">
                    <p>Your portfolio has triggered a {alert_type} alert.</p>
                    <p>Please review your positions and consider taking appropriate action.</p>
                </div>
                
                <div class="footer">
                    <p>This alert was generated by your automated trading system.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _create_portfolio_alert_email_text(self, portfolio_data: Dict, alert_type: str) -> str:
        """Create text content for portfolio alert email."""
        return f"""
Portfolio Alert: {alert_type.title()}

Your portfolio has triggered a {alert_type} alert.
Please review your positions and consider taking appropriate action.

This alert was generated by your automated trading system.
        """
    
    def _create_strategy_performance_email_html(self, strategy_results: Dict) -> str:
        """Create HTML content for strategy performance email."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .header {{ background-color: #28a745; color: white; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px; }}
                .strategy-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>📈 Strategy Performance Report</h2>
                </div>
                
                <div class="strategy-info">
                    <p>Your trading strategies have been evaluated and performance results are available.</p>
                    <p>Please check your dashboard for detailed performance metrics.</p>
                </div>
                
                <div class="footer">
                    <p>This report was generated by your automated trading system.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _create_strategy_performance_email_text(self, strategy_results: Dict) -> str:
        """Create text content for strategy performance email."""
        return f"""
Strategy Performance Report

Your trading strategies have been evaluated and performance results are available.
Please check your dashboard for detailed performance metrics.

This report was generated by your automated trading system.
        """
    
    def _create_daily_summary_email_html(self, daily_data: Dict) -> str:
        """Create HTML content for daily summary email."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .header {{ background-color: #6c757d; color: white; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px; }}
                .summary-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>📋 Daily Trading Summary</h2>
                    <h3>{datetime.now().strftime('%Y-%m-%d')}</h3>
                </div>
                
                <div class="summary-info">
                    <p>Here's your daily trading summary:</p>
                    <ul>
                        <li>Stocks screened: {daily_data.get('stocks_screened', 0)}</li>
                        <li>Stocks selected: {daily_data.get('stocks_selected', 0)}</li>
                        <li>Strategies executed: {daily_data.get('strategies_executed', 0)}</li>
                        <li>Portfolio value: ₹{daily_data.get('portfolio_value', 0):,.2f}</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>This summary was generated by your automated trading system.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _create_daily_summary_email_text(self, daily_data: Dict) -> str:
        """Create text content for daily summary email."""
        return f"""
Daily Trading Summary - {datetime.now().strftime('%Y-%m-%d')}

Here's your daily trading summary:
- Stocks screened: {daily_data.get('stocks_screened', 0)}
- Stocks selected: {daily_data.get('stocks_selected', 0)}
- Strategies executed: {daily_data.get('strategies_executed', 0)}
- Portfolio value: ₹{daily_data.get('portfolio_value', 0):,.2f}

This summary was generated by your automated trading system.
        """


# Global instance
_email_alert_manager = None

def get_email_alert_manager() -> EmailAlertManager:
    """Get global email alert manager instance."""
    global _email_alert_manager
    if _email_alert_manager is None:
        _email_alert_manager = EmailAlertManager()
    return _email_alert_manager


if __name__ == "__main__":
    # Test the email alert manager
    manager = get_email_alert_manager()
    
    # Test stock pick alert
    test_stock_data = {
        'symbol': 'RELIANCE.NS',
        'name': 'Reliance Industries Ltd',
        'current_price': 2500.0,
        'market_cap': 15000000000000,
        'sector': 'Oil & Gas',
        'pe_ratio': 15.5
    }
    
    success = manager.send_stock_pick_alert(1, test_stock_data, 'Momentum Strategy', 'BUY')
    print(f"Stock pick alert sent: {success}")
