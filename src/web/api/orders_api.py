"""
Orders API endpoints
"""
from flask_restx import Namespace, Resource
from flask import request
from flask_login import login_required, current_user
from datetime import datetime

# Create namespace for orders API
ns_orders = Namespace('orders', description='Order management operations')

@ns_orders.route('/')
class Orders(Resource):
    @login_required
    def get(self):
        """Get orders."""
        try:
            from datastore.database import get_database_manager
            from datastore.models import Order
            
            db_manager = get_database_manager()
            
            with db_manager.get_session() as session:
                orders = session.query(Order).filter(
                    Order.user_id == current_user.id
                ).order_by(Order.created_at.desc()).limit(100).all()
                return [{
                    'id': order.id,
                    'order_id': order.order_id,
                    'tradingsymbol': order.tradingsymbol,
                    'transaction_type': order.transaction_type,
                    'quantity': order.quantity,
                    'order_type': order.order_type,
                    'price': order.price,
                    'order_status': order.order_status,
                    'created_at': order.created_at.isoformat()
                } for order in orders]
        except Exception as e:
            return {'error': str(e)}, 500

@ns_orders.route('/create-buy-order')
class CreateBuyOrder(Resource):
    @login_required
    def post(self):
        """Create a buy order."""
        try:
            from order.order_router import OrderRouter
            
            data = request.get_json()
            symbol = data.get('symbol')
            quantity = data.get('quantity')
            price = data.get('price')
            order_type = data.get('order_type', 'LIMIT')
            
            if not all([symbol, quantity]):
                return {'error': 'Symbol and quantity are required'}, 400
            
            # Get FYERS connector from app context if available
            from flask import current_app
            fyers_connector = getattr(current_app, 'fyers_connector', None)
            router = OrderRouter(fyers_connector)
            
            order_result = router.create_buy_order(
                current_user.id, symbol, quantity, price, order_type
            )
            
            return {
                'success': True,
                'order_result': order_result,
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to create buy order: {str(e)}'}, 500