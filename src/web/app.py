"""
Flask Web Application for the Automated Trading System
"""
from flask import Flask, render_template, request, jsonify
from datastore.database import get_database_manager
from datastore.models import Log, Order, Trade, Position
# Add import for charting
from charting.db_charts import DatabaseCharts
from datetime import datetime


def create_app():
    """Create Flask application."""
    app = Flask(__name__)
    
    # Initialize database
    db_manager = get_database_manager()
    
    # Initialize charting
    charts = DatabaseCharts(db_manager)
    
    @app.route('/')
    def dashboard():
        """Dashboard page."""
        return render_template('dashboard.html')
    
    @app.route('/api/portfolio_chart')
    def portfolio_chart_data():
        """API endpoint for portfolio value chart data."""
        try:
            days = int(request.args.get('days', 30))
            chart_data = charts.get_portfolio_value_chart_data(days)
            return jsonify(chart_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard/metrics')
    def dashboard_metrics():
        """API endpoint for dashboard metrics."""
        try:
            # Get performance summary
            summary = charts.get_performance_summary()
            
            # Get positions
            with db_manager.get_session() as session:
                positions = session.query(Position).filter(Position.quantity != 0).all()
                positions_data = [{
                    'symbol': position.tradingsymbol,
                    'quantity': position.quantity,
                    'avg_price': position.average_price,
                    'last_price': position.last_price,
                    'pnl': position.pnl
                } for position in positions]
            
            # Get recent orders
            with db_manager.get_session() as session:
                orders = session.query(Order).order_by(Order.created_at.desc()).limit(10).all()
                orders_data = [{
                    'order_id': order.order_id,
                    'symbol': order.tradingsymbol,
                    'type': order.order_type,
                    'status': order.order_status
                } for order in orders]
            
            return jsonify({
                'total_pnl': summary.get('total_pnl', 0),
                'positions': positions_data,
                'recent_orders': orders_data
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/selected_stocks')
    def api_selected_stocks():
        """API endpoint for selected stocks."""
        try:
            period = request.args.get('period', 'week')
            
            # For now, we'll simulate the data
            # In a real implementation, this would query the SelectedStock table
            stocks_data = [
                {
                    'symbol': 'RELIANCE.NS',
                    'selection_date': '2025-09-05T10:30:00',
                    'selection_price': 2750.50,
                    'current_price': 2780.25,
                    'quantity': 10,
                    'status': 'Active'
                },
                {
                    'symbol': 'TCS.NS',
                    'selection_date': '2025-09-04T11:15:00',
                    'selection_price': 3850.75,
                    'current_price': 3825.50,
                    'quantity': 5,
                    'status': 'Active'
                },
                {
                    'symbol': 'INFY.NS',
                    'selection_date': '2025-09-03T09:45:00',
                    'selection_price': 1620.00,
                    'current_price': 1645.30,
                    'quantity': 15,
                    'status': 'Active'
                },
                {
                    'symbol': 'HDFCBANK.NS',
                    'selection_date': '2025-09-02T10:00:00',
                    'selection_price': 1520.25,
                    'current_price': 1510.75,
                    'quantity': 20,
                    'status': 'Active'
                },
                {
                    'symbol': 'ICICIBANK.NS',
                    'selection_date': '2025-09-01T10:30:00',
                    'selection_price': 1025.50,
                    'current_price': 1040.25,
                    'quantity': 25,
                    'status': 'Active'
                },
                {
                    'symbol': 'AXISBANK.NS',
                    'selection_date': '2025-08-28T11:00:00',
                    'selection_price': 1120.75,
                    'current_price': 1095.50,
                    'quantity': 18,
                    'status': 'Sold'
                }
            ]
            
            return jsonify({
                'stocks': stocks_data
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/pnl_chart')
    def pnl_chart_data():
        """API endpoint for P&L chart data."""
        try:
            days = int(request.args.get('days', 30))
            chart_data = charts.get_pnl_chart_data(days)
            return jsonify(chart_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/trades_chart')
    def trades_chart_data():
        """API endpoint for trades chart data."""
        try:
            days = int(request.args.get('days', 30))
            chart_data = charts.get_trades_chart_data(days)
            return jsonify(chart_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/positions_chart')
    def positions_chart_data():
        """API endpoint for positions chart data."""
        try:
            chart_data = charts.get_positions_chart_data()
            return jsonify(chart_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/orders_chart')
    def orders_chart_data():
        """API endpoint for orders chart data."""
        try:
            status = request.args.get('status')
            chart_data = charts.get_orders_chart_data(status)
            return jsonify(chart_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/performance_summary')
    def performance_summary():
        """API endpoint for performance summary."""
        try:
            summary = charts.get_performance_summary()
            return jsonify(summary)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/market_chart/<symbol>')
    def market_chart_data(symbol):
        """API endpoint for market data chart."""
        try:
            days = int(request.args.get('days', 30))
            chart_data = charts.get_market_data_chart(symbol, days)
            return jsonify(chart_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/logs')
    def logs():
        """Logs page."""
        return render_template('logs.html')
    
    @app.route('/api/logs')
    def api_logs():
        """API endpoint for logs."""
        try:
            with db_manager.get_session() as session:
                logs = session.query(Log).order_by(Log.timestamp.desc()).limit(100).all()
                return jsonify([{
                    'id': log.id,
                    'timestamp': log.timestamp.isoformat(),
                    'level': log.level,
                    'module': log.module,
                    'message': log.message
                } for log in logs])
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/orders')
    def orders():
        """Orders page."""
        return render_template('orders.html')
    
    @app.route('/api/orders')
    def api_orders():
        """API endpoint for orders."""
        try:
            with db_manager.get_session() as session:
                orders = session.query(Order).order_by(Order.created_at.desc()).limit(100).all()
                return jsonify([{
                    'id': order.id,
                    'order_id': order.order_id,
                    'tradingsymbol': order.tradingsymbol,
                    'transaction_type': order.transaction_type,
                    'quantity': order.quantity,
                    'order_type': order.order_type,
                    'price': order.price,
                    'order_status': order.order_status,
                    'created_at': order.created_at.isoformat()
                } for order in orders])
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/trades')
    def trades():
        """Trades page."""
        return render_template('trades.html')
    
    @app.route('/api/trades')
    def api_trades():
        """API endpoint for trades."""
        try:
            with db_manager.get_session() as session:
                trades = session.query(Trade).order_by(Trade.trade_time.desc()).limit(100).all()
                return jsonify([{
                    'id': trade.id,
                    'trade_id': trade.trade_id,
                    'tradingsymbol': trade.tradingsymbol,
                    'transaction_type': trade.transaction_type,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'trade_time': trade.trade_time.isoformat()
                } for trade in trades])
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/positions')
    def positions():
        """Positions page."""
        return render_template('positions.html')
    
    @app.route('/api/positions')
    def api_positions():
        """API endpoint for positions."""
        try:
            with db_manager.get_session() as session:
                positions = session.query(Position).filter(Position.quantity != 0).all()
                return jsonify([{
                    'id': position.id,
                    'tradingsymbol': position.tradingsymbol,
                    'quantity': position.quantity,
                    'average_price': position.average_price,
                    'last_price': position.last_price,
                    'pnl': position.pnl,
                    'value': position.value
                } for position in positions])
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/strategies')
    def strategies():
        """Strategies page."""
        return render_template('strategies.html')
    
    @app.route('/api/strategies')
    def api_strategies():
        """API endpoint for strategies."""
        try:
            # For now, we'll simulate the data
            # In a real implementation, this would query the Strategy table
            strategies_data = [
                {
                    'id': 1,
                    'name': 'Momentum Strategy',
                    'description': 'Selects stocks based on price momentum',
                    'is_active': True
                },
                {
                    'id': 2,
                    'name': 'Breakout Strategy',
                    'description': 'Selects stocks breaking out of resistance levels',
                    'is_active': True
                }
            ]
            
            return jsonify(strategies_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/selected_stocks')
    def selected_stocks():
        """Selected stocks page."""
        return render_template('selected_stocks.html')
    
    @app.route('/reports')
    def reports():
        """Reports page."""
        return render_template('reports.html')
    
    @app.route('/alerts')
    def alerts():
        """Alerts page."""
        return render_template('alerts.html')
    
    @app.route('/api/alerts')
    def api_alerts():
        """API endpoint for alerts."""
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
            
            return jsonify(alerts_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/settings')
    def settings():
        """Settings page."""
        return render_template('settings.html')
    
    @app.route('/api/settings')
    def api_settings():
        """API endpoint for settings."""
        try:
            # For now, we'll simulate the data
            # In a real implementation, this would query the Configuration table
            settings_data = {
                'trading_mode': 'development',
                'market_open': '09:15',
                'market_close': '15:30',
                'max_capital_per_trade': 1.0,
                'max_concurrent_trades': 10,
                'daily_loss_limit': 2.0,
                'single_name_exposure': 5.0,
                'stop_loss_percent': 5.0,
                'take_profit_percent': 10.0
            }
            
            return jsonify(settings_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)
