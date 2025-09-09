"""
Flask Web Application for the Automated Trading System
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from datastore.database import get_database_manager
from datastore.models import Log, Order, Trade, Position, User, Strategy, SelectedStock, Configuration
# Add import for charting
from charting.db_charts import DatabaseCharts
from trading_engine.multi_user_trading_engine import MultiUserTradingEngine
from datetime import datetime
import secrets


def create_app():
    """Create Flask application."""
    app = Flask(__name__)
    
    # Generate a secret key for sessions
    app.secret_key = secrets.token_hex(16)
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    # Initialize Flask-Bcrypt
    bcrypt = Bcrypt(app)
    
    # Initialize database
    db_manager = get_database_manager()
    
    # Initialize charting
    charts = DatabaseCharts(db_manager)
    
    @login_manager.user_loader
    def load_user(user_id):
        """Load user by ID for Flask-Login."""
        with db_manager.get_session() as session:
            user = session.query(User).get(int(user_id))
            if user:
                # Detach the user from the session to avoid issues
                session.expunge(user)
            return user
    
    # Authentication routes
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Login page."""
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            remember = bool(request.form.get('remember'))
            
            if not username or not password:
                flash('Please fill in all fields.', 'error')
                return render_template('login.html')
            
            with db_manager.get_session() as db_session:
                user = db_session.query(User).filter_by(username=username).first()
                
                if user and bcrypt.check_password_hash(user.password_hash, password):
                    if user.is_active:
                        login_user(user, remember=remember)
                        user.last_login = datetime.utcnow()
                        db_session.commit()
                        
                        # Redirect to next page or dashboard
                        next_page = request.args.get('next')
                        return redirect(next_page) if next_page else redirect(url_for('dashboard'))
                    else:
                        flash('Your account has been deactivated.', 'error')
                else:
                    flash('Invalid username or password.', 'error')
        
        return render_template('login.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """Registration page."""
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            first_name = request.form.get('first_name')
            last_name = request.form.get('last_name')
            
            # Validation
            if not all([username, email, password, confirm_password]):
                flash('Please fill in all required fields.', 'error')
                return render_template('register.html')
            
            if password != confirm_password:
                flash('Passwords do not match.', 'error')
                return render_template('register.html')
            
            if len(password) < 6:
                flash('Password must be at least 6 characters long.', 'error')
                return render_template('register.html')
            
            with db_manager.get_session() as db_session:
                # Check if username or email already exists
                existing_user = db_session.query(User).filter(
                    (User.username == username) | (User.email == email)
                ).first()
                
                if existing_user:
                    if existing_user.username == username:
                        flash('Username already exists.', 'error')
                    else:
                        flash('Email already registered.', 'error')
                    return render_template('register.html')
                
                # Create new user
                password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
                new_user = User(
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    first_name=first_name,
                    last_name=last_name
                )
                
                db_session.add(new_user)
                db_session.commit()
                
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
        
        return render_template('register.html')
    
    @app.route('/logout')
    @login_required
    def logout():
        """Logout user."""
        logout_user()
        flash('You have been logged out successfully.', 'info')
        return redirect(url_for('login'))
    
    @app.route('/')
    @login_required
    def dashboard():
        """Dashboard page."""
        return render_template('dashboard.html')
    
    @app.route('/api/portfolio_chart')
    @login_required
    def portfolio_chart_data():
        """API endpoint for portfolio value chart data."""
        try:
            days = int(request.args.get('days', 30))
            chart_data = charts.get_portfolio_value_chart_data(days)
            return jsonify(chart_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard/metrics')
    @login_required
    def dashboard_metrics():
        """API endpoint for dashboard metrics."""
        try:
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
            
            return jsonify({
                'total_pnl': summary.get('total_pnl', 0),
                'positions': positions_data,
                'recent_orders': orders_data
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/selected_stocks')
    @login_required
    def api_selected_stocks():
        """API endpoint for selected stocks."""
        try:
            period = request.args.get('period', 'week')
            
            with db_manager.get_session() as session:
                selected_stocks = session.query(SelectedStock).filter(
                    SelectedStock.user_id == current_user.id
                ).order_by(SelectedStock.selection_date.desc()).all()
                
                stocks_data = [{
                    'id': stock.id,
                    'symbol': stock.symbol,
                    'selection_date': stock.selection_date.isoformat(),
                    'selection_price': stock.selection_price,
                    'current_price': stock.current_price,
                    'quantity': stock.quantity,
                    'strategy_name': stock.strategy_name,
                    'status': stock.status,
                    'created_at': stock.created_at.isoformat()
                } for stock in selected_stocks]
            
            return jsonify({
                'stocks': stocks_data
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/pnl_chart')
    @login_required
    def pnl_chart_data():
        """API endpoint for P&L chart data."""
        try:
            days = int(request.args.get('days', 30))
            chart_data = charts.get_pnl_chart_data(days)
            return jsonify(chart_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/trades_chart')
    @login_required
    def trades_chart_data():
        """API endpoint for trades chart data."""
        try:
            days = int(request.args.get('days', 30))
            chart_data = charts.get_trades_chart_data(days)
            return jsonify(chart_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/positions_chart')
    @login_required
    def positions_chart_data():
        """API endpoint for positions chart data."""
        try:
            chart_data = charts.get_positions_chart_data()
            return jsonify(chart_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/orders_chart')
    @login_required
    def orders_chart_data():
        """API endpoint for orders chart data."""
        try:
            status = request.args.get('status')
            chart_data = charts.get_orders_chart_data(status)
            return jsonify(chart_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/performance_summary')
    @login_required
    def performance_summary():
        """API endpoint for performance summary."""
        try:
            summary = charts.get_performance_summary()
            return jsonify(summary)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/market_chart/<symbol>')
    @login_required
    def market_chart_data(symbol):
        """API endpoint for market data chart."""
        try:
            days = int(request.args.get('days', 30))
            chart_data = charts.get_market_data_chart(symbol, days)
            return jsonify(chart_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/logs')
    @login_required
    def logs():
        """Logs page."""
        return render_template('logs.html')
    
    @app.route('/api/logs')
    @login_required
    def api_logs():
        """API endpoint for logs."""
        try:
            with db_manager.get_session() as session:
                logs = session.query(Log).filter(
                    Log.user_id == current_user.id
                ).order_by(Log.timestamp.desc()).limit(100).all()
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
    @login_required
    def orders():
        """Orders page."""
        return render_template('orders.html')
    
    @app.route('/api/orders')
    @login_required
    def api_orders():
        """API endpoint for orders."""
        try:
            with db_manager.get_session() as session:
                orders = session.query(Order).filter(
                    Order.user_id == current_user.id
                ).order_by(Order.created_at.desc()).limit(100).all()
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
    @login_required
    def trades():
        """Trades page."""
        return render_template('trades.html')
    
    @app.route('/api/trades')
    @login_required
    def api_trades():
        """API endpoint for trades."""
        try:
            with db_manager.get_session() as session:
                trades = session.query(Trade).filter(
                    Trade.user_id == current_user.id
                ).order_by(Trade.trade_time.desc()).limit(100).all()
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
    @login_required
    def positions():
        """Positions page."""
        return render_template('positions.html')
    
    @app.route('/api/positions')
    @login_required
    def api_positions():
        """API endpoint for positions."""
        try:
            with db_manager.get_session() as session:
                positions = session.query(Position).filter(
                    Position.user_id == current_user.id,
                    Position.quantity != 0
                ).all()
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
    @login_required
    def strategies():
        """Strategies page."""
        return render_template('strategies.html')
    
    @app.route('/api/strategies')
    @login_required
    def api_strategies():
        """API endpoint for strategies."""
        try:
            with db_manager.get_session() as session:
                strategies = session.query(Strategy).filter(
                    Strategy.user_id == current_user.id
                ).all()
                
                strategies_data = [{
                    'id': strategy.id,
                    'name': strategy.name,
                    'description': strategy.description,
                    'is_active': strategy.is_active,
                    'created_at': strategy.created_at.isoformat()
                } for strategy in strategies]
            
            return jsonify(strategies_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/selected_stocks')
    @login_required
    def selected_stocks():
        """Selected stocks page."""
        return render_template('selected_stocks.html')
    
    @app.route('/reports')
    @login_required
    def reports():
        """Reports page."""
        return render_template('reports.html')
    
    @app.route('/alerts')
    @login_required
    def alerts():
        """Alerts page."""
        return render_template('alerts.html')
    
    @app.route('/api/alerts')
    @login_required
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
    @login_required
    def settings():
        """Settings page."""
        return render_template('settings.html')
    
    @app.route('/api/settings')
    @login_required
    def api_settings():
        """API endpoint for settings."""
        try:
            with db_manager.get_session() as session:
                # Get user-specific configurations
                user_configs = session.query(Configuration).filter(
                    Configuration.user_id == current_user.id
                ).all()
                
                # Get global configurations (where user_id is NULL)
                global_configs = session.query(Configuration).filter(
                    Configuration.user_id.is_(None)
                ).all()
                
                # Combine user and global configs (user configs override global ones)
                settings_data = {}
                
                # First add global configs
                for config in global_configs:
                    settings_data[config.key] = config.value
                
                # Then override with user-specific configs
                for config in user_configs:
                    settings_data[config.key] = config.value
                
                # If no configs exist, return defaults
                if not settings_data:
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
    
    # Multi-user trading engine API endpoints
    @app.route('/api/trading_engine/status')
    @login_required
    def trading_engine_status():
        """API endpoint for trading engine status."""
        try:
            # This would need to be passed from the main application
            # For now, we'll return a placeholder response
            return jsonify({
                'status': 'running',
                'message': 'Multi-user trading engine is running',
                'active_users': 0,
                'mode': 'development'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/trading_engine/user_session')
    @login_required
    def user_trading_session():
        """API endpoint for current user's trading session status."""
        try:
            # This would need to be passed from the main application
            # For now, we'll return a placeholder response
            return jsonify({
                'user_id': current_user.id,
                'username': current_user.username,
                'is_active': True,
                'last_activity': datetime.utcnow().isoformat(),
                'trading_state': {
                    'last_scan_time': None,
                    'selected_stocks': [],
                    'active_orders': [],
                    'positions': {},
                    'daily_pnl': 0.0,
                    'risk_metrics': {}
                }
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/trading_engine/start_session', methods=['POST'])
    @login_required
    def start_trading_session():
        """API endpoint to start a trading session for the current user."""
        try:
            # This would need to be passed from the main application
            # For now, we'll return a placeholder response
            return jsonify({
                'status': 'success',
                'message': f'Trading session started for user {current_user.username}',
                'user_id': current_user.id
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/trading_engine/stop_session', methods=['POST'])
    @login_required
    def stop_trading_session():
        """API endpoint to stop a trading session for the current user."""
        try:
            # This would need to be passed from the main application
            # For now, we'll return a placeholder response
            return jsonify({
                'status': 'success',
                'message': f'Trading session stopped for user {current_user.username}',
                'user_id': current_user.id
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Admin routes
    @app.route('/admin/users')
    @login_required
    def admin_users():
        """Admin page for managing users."""
        if not current_user.is_admin:
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('dashboard'))
        return render_template('admin/users.html')
    
    @app.route('/api/admin/users')
    @login_required
    def api_admin_users():
        """API endpoint for getting all users (admin only)."""
        if not current_user.is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        try:
            with db_manager.get_session() as session:
                users = session.query(User).all()
                users_data = [{
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'is_active': user.is_active,
                    'is_admin': user.is_admin,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                    'last_login': user.last_login.isoformat() if user.last_login else None
                } for user in users]
            
            return jsonify(users_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/users', methods=['POST'])
    @login_required
    def api_create_user():
        """API endpoint for creating a new user (admin only)."""
        if not current_user.is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        try:
            data = request.get_json()
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            first_name = data.get('first_name', '')
            last_name = data.get('last_name', '')
            is_admin = data.get('is_admin', False)
            
            if not all([username, email, password]):
                return jsonify({'error': 'Username, email, and password are required'}), 400
            
            with db_manager.get_session() as session:
                # Check if username or email already exists
                existing_user = session.query(User).filter(
                    (User.username == username) | (User.email == email)
                ).first()
                
                if existing_user:
                    return jsonify({'error': 'Username or email already exists'}), 400
                
                # Create new user
                password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
                new_user = User(
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    first_name=first_name,
                    last_name=last_name,
                    is_admin=is_admin,
                    is_active=True
                )
                
                session.add(new_user)
                session.commit()
                
                return jsonify({
                    'message': 'User created successfully',
                    'user_id': new_user.id
                }), 201
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
    @login_required
    def api_update_user(user_id):
        """API endpoint for updating a user (admin only)."""
        if not current_user.is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        try:
            data = request.get_json()
            
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return jsonify({'error': 'User not found'}), 404
                
                # Update user fields
                if 'username' in data:
                    user.username = data['username']
                if 'email' in data:
                    user.email = data['email']
                if 'first_name' in data:
                    user.first_name = data['first_name']
                if 'last_name' in data:
                    user.last_name = data['last_name']
                if 'is_active' in data:
                    user.is_active = data['is_active']
                if 'is_admin' in data:
                    user.is_admin = data['is_admin']
                if 'password' in data and data['password']:
                    user.password_hash = bcrypt.generate_password_hash(data['password']).decode('utf-8')
                
                session.commit()
                
                return jsonify({'message': 'User updated successfully'})
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
    @login_required
    def api_delete_user(user_id):
        """API endpoint for deleting a user (admin only)."""
        if not current_user.is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return jsonify({'error': 'User not found'}), 404
                
                # Don't allow deleting the current admin user
                if user.id == current_user.id:
                    return jsonify({'error': 'Cannot delete your own account'}), 400
                
                session.delete(user)
                session.commit()
                
                return jsonify({'message': 'User deleted successfully'})
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/users/<int:user_id>/reset-password', methods=['POST'])
    @login_required
    def api_reset_user_password(user_id):
        """API endpoint for resetting a user's password (admin only)."""
        if not current_user.is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        try:
            data = request.get_json()
            new_password = data.get('password')
            
            if not new_password:
                return jsonify({'error': 'New password is required'}), 400
            
            if len(new_password) < 6:
                return jsonify({'error': 'Password must be at least 6 characters long'}), 400
            
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return jsonify({'error': 'User not found'}), 404
                
                # Update password
                user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
                session.commit()
                
                return jsonify({'message': 'Password reset successfully'})
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint for Docker."""
        try:
            # Check database connection
            with db_manager.get_session() as session:
                session.execute("SELECT 1")
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'database': 'connected'
            }), 200
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)
