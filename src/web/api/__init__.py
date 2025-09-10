"""
API Module for the Automated Trading System
"""
from flask_restx import Api

def create_api(app):
    """Create and configure the API with all namespaces."""
    # Initialize Flask-RESTX for Swagger documentation
    api = Api(
        app,
        version='1.0',
        title='Automated Trading System API',
        description='API documentation for the Automated Trading System',
        doc='/docs/',
        prefix='/api'
    )
    
    # Import all namespaces
    from .dashboard_api import ns_dashboard
    from .trading_api import ns_trading
    from .orders_api import ns_orders
    from .analytics_api import ns_analytics
    from .data_api import ns_data
    from .alerts_api import ns_alerts
    from .admin_api import ns_admin
    
    # Add namespaces to API
    api.add_namespace(ns_dashboard, path='/dashboard')
    api.add_namespace(ns_trading, path='/trading')
    api.add_namespace(ns_orders, path='/orders')
    api.add_namespace(ns_analytics, path='/analytics')
    api.add_namespace(ns_data, path='/data')
    api.add_namespace(ns_alerts, path='/alerts')
    api.add_namespace(ns_admin, path='/admin')
    
    return api