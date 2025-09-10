"""
Alerts API endpoints
"""
from flask_restx import Namespace, Resource
from flask import request
from flask_login import login_required, current_user
from datetime import datetime

# Create namespace for alerts API
ns_alerts = Namespace('alerts', description='Alerts operations')

@ns_alerts.route('/')
class Alerts(Resource):
    @login_required
    def get(self):
        """Get alerts."""
        try:
            # For now, we'll simulate the data
            # In a real implementation, this would query the alerts system
            alerts_data = [
                {
                    'id': 1,
                    'timestamp': '2025-09-09T10:30:00',
                    'type': 'Order Execution',
                    'message': 'Order executed successfully for RELIANCE.NS',
                    'severity': 'info',
                    'status': 'new'
                },
                {
                    'id': 2,
                    'timestamp': '2025-09-09T09:45:00',
                    'type': 'Price Alert',
                    'message': 'TCS.NS has broken above resistance level',
                    'severity': 'warning',
                    'status': 'new'
                },
                {
                    'id': 3,
                    'timestamp': '2025-09-08T11:15:00',
                    'type': 'Risk Management',
                    'message': 'Position size limit exceeded for INFY.NS',
                    'severity': 'critical',
                    'status': 'read'
                }
            ]
            
            return alerts_data
        except Exception as e:
            return {'error': str(e)}, 500

@ns_alerts.route('/send-stock-pick')
class SendStockPickAlert(Resource):
    @login_required
    def post(self):
        """Send stock pick alert via email."""
        try:
            from alerts.email_alerts import get_email_alert_manager
            
            data = request.get_json()
            stock_data = data.get('stock_data')
            strategy_name = data.get('strategy_name', 'Unknown')
            recommendation = data.get('recommendation', 'BUY')
            
            if not stock_data:
                return {'error': 'Stock data required'}, 400
            
            alert_manager = get_email_alert_manager()
            success = alert_manager.send_stock_pick_alert(
                current_user.id, stock_data, strategy_name, recommendation
            )
            
            return {
                'success': success,
                'message': 'Stock pick alert sent' if success else 'Failed to send alert',
                'timestamp': datetime.utcnow().isoformat()
            }, 200 if success else 500
            
        except Exception as e:
            return {'error': f'Failed to send stock pick alert: {str(e)}'}, 500

@ns_alerts.route('/send-portfolio-alert')
class SendPortfolioAlert(Resource):
    @login_required
    def post(self):
        """Send portfolio alert via email."""
        try:
            from alerts.email_alerts import get_email_alert_manager
            
            data = request.get_json()
            portfolio_data = data.get('portfolio_data', {})
            alert_type = data.get('alert_type', 'general')
            
            alert_manager = get_email_alert_manager()
            success = alert_manager.send_portfolio_alert(
                current_user.id, portfolio_data, alert_type
            )
            
            return {
                'success': success,
                'message': 'Portfolio alert sent' if success else 'Failed to send alert',
                'timestamp': datetime.utcnow().isoformat()
            }, 200 if success else 500
            
        except Exception as e:
            return {'error': f'Failed to send portfolio alert: {str(e)}'}, 500