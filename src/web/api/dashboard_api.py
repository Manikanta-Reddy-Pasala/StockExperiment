"""
Dashboard API endpoints
"""
from flask_restx import Namespace, Resource
from flask import request
from flask_login import login_required, current_user

# Create namespace for dashboard API
ns_dashboard = Namespace('dashboard', description='Dashboard operations')

@ns_dashboard.route('/metrics')
class DashboardMetrics(Resource):
    @login_required
    def get(self):
        """Get dashboard metrics."""
        try:
            from datastore.database import get_database_manager
            from datastore.models import Position, Order
            from charting.db_charts import DatabaseCharts
            
            db_manager = get_database_manager()
            charts = DatabaseCharts(db_manager)
            
            # Get performance summary
            summary = charts.get_performance_summary()
            
            # Get positions
            with db_manager.get_session() as session:
                positions = session.query(Position).filter(
                    Position.user_id == current_user.id,
                    Position.quantity != 0
                ).all()
                positions_data = [{
                    'symbol': position.tradingsymbol,
                    'quantity': position.quantity,
                    'avg_price': position.average_price,
                    'last_price': position.last_price,
                    'pnl': position.pnl
                } for position in positions]
            
            # Get recent orders
            with db_manager.get_session() as session:
                orders = session.query(Order).filter(
                    Order.user_id == current_user.id
                ).order_by(Order.created_at.desc()).limit(10).all()
                orders_data = [{
                    'order_id': order.order_id,
                    'symbol': order.tradingsymbol,
                    'type': order.order_type,
                    'status': order.order_status
                } for order in orders]
            
            return {
                'total_pnl': summary.get('total_pnl', 0),
                'positions': positions_data,
                'recent_orders': orders_data
            }
        except Exception as e:
            return {'error': str(e)}, 500

@ns_dashboard.route('/trades_chart')
class TradesChart(Resource):
    @login_required
    def get(self):
        """Get trades chart data."""
        try:
            from datastore.database import get_database_manager
            from charting.db_charts import DatabaseCharts
            
            db_manager = get_database_manager()
            charts = DatabaseCharts(db_manager)
            
            days = int(request.args.get('days', 30))
            chart_data = charts.get_trades_chart_data(days)
            return chart_data
        except Exception as e:
            return {'error': str(e)}, 500

@ns_dashboard.route('/positions_chart')
class PositionsChart(Resource):
    @login_required
    def get(self):
        """Get positions chart data."""
        try:
            from datastore.database import get_database_manager
            from charting.db_charts import DatabaseCharts
            
            db_manager = get_database_manager()
            charts = DatabaseCharts(db_manager)
            
            chart_data = charts.get_positions_chart_data()
            return chart_data
        except Exception as e:
            return {'error': str(e)}, 500

@ns_dashboard.route('/orders_chart')
class OrdersChart(Resource):
    @login_required
    def get(self):
        """Get orders chart data."""
        try:
            from datastore.database import get_database_manager
            from charting.db_charts import DatabaseCharts
            
            db_manager = get_database_manager()
            charts = DatabaseCharts(db_manager)
            
            status = request.args.get('status')
            chart_data = charts.get_orders_chart_data(status)
            return chart_data
        except Exception as e:
            return {'error': str(e)}, 500

@ns_dashboard.route('/performance_summary')
class PerformanceSummary(Resource):
    @login_required
    def get(self):
        """Get performance summary."""
        try:
            from datastore.database import get_database_manager
            from charting.db_charts import DatabaseCharts
            
            db_manager = get_database_manager()
            charts = DatabaseCharts(db_manager)
            
            summary = charts.get_performance_summary()
            return summary
        except Exception as e:
            return {'error': str(e)}, 500

@ns_dashboard.route('/market_chart/<string:symbol>')
class MarketChart(Resource):
    @login_required
    def get(self, symbol):
        """Get market data chart."""
        try:
            from datastore.database import get_database_manager
            from charting.db_charts import DatabaseCharts
            
            db_manager = get_database_manager()
            charts = DatabaseCharts(db_manager)
            
            days = int(request.args.get('days', 30))
            chart_data = charts.get_market_data_chart(symbol, days)
            return chart_data
        except Exception as e:
            return {'error': str(e)}, 500

@ns_dashboard.route('/logs')
class Logs(Resource):
    @login_required
    def get(self):
        """Get logs."""
        try:
            from datastore.database import get_database_manager
            from datastore.models import Log
            
            db_manager = get_database_manager()
            
            with db_manager.get_session() as session:
                logs = session.query(Log).filter(
                    Log.user_id == current_user.id
                ).order_by(Log.timestamp.desc()).limit(100).all()
                return [{
                    'id': log.id,
                    'timestamp': log.timestamp.isoformat(),
                    'level': log.level,
                    'module': log.module,
                    'message': log.message
                } for log in logs]
        except Exception as e:
            return {'error': str(e)}, 500