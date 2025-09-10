"""
Flask Web Application for the Automated Trading System
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from datastore.database import get_database_manager
from datastore.models import Log, Order, Trade, Position, User, Strategy, SuggestedStock, Configuration
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
    
    @app.route('/portfolio')
    @login_required
    def portfolio():
        """Portfolio page."""
        return render_template('portfolio.html')
    
    @app.route('/api/portfolio')
    @login_required
    def api_portfolio():
        """API endpoint for portfolio."""
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
    
    @app.route('/suggested_stocks')
    @login_required
    def suggested_stocks():
        """Suggested stocks page."""
        return render_template('suggested_stocks.html')
    
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
    
    # Removed old trading engine API endpoints - using new trading executor system instead
    
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

    # Trading System API Endpoints
    @app.route('/api/trading/run-screening', methods=['POST'])
    @login_required
    def run_stock_screening():
        """Run stock screening process."""
        try:
            from screening.stock_screener import StockScreener
            
            screener = StockScreener()
            screened_stocks = screener.run_daily_screening()
            
            return jsonify({
                'success': True,
                'screened_stocks': screened_stocks,
                'count': len(screened_stocks),
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to run screening: {str(e)}'}), 500

    @app.route('/api/trading/run-strategies', methods=['POST'])
    @login_required
    def run_trading_strategies():
        """Run trading strategies on screened stocks."""
        try:
            from strategies.strategy_engine import StrategyEngine
            
            data = request.get_json()
            screened_stocks = data.get('screened_stocks', [])
            
            if not screened_stocks:
                return jsonify({'error': 'No screened stocks provided'}), 400
            
            strategy_engine = StrategyEngine()
            strategy_results = strategy_engine.run_strategies(screened_stocks)
            
            return jsonify({
                'success': True,
                'strategy_results': strategy_results,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to run strategies: {str(e)}'}), 500

    @app.route('/api/trading/run-dry-run', methods=['POST'])
    @login_required
    def run_dry_run():
        """Run dry run mode for strategy testing."""
        try:
            from execution.trading_executor import TradingExecutor
            
            data = request.get_json()
            strategy_name = data.get('strategy_name')  # Optional: specific strategy
            
            executor = TradingExecutor(user_id=current_user.id)
            result = executor.run_dry_run_only(strategy_name)
            
            return jsonify({
                'success': True,
                'dry_run_result': result,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to run dry run: {str(e)}'}), 500

    @app.route('/api/trading/run-complete-workflow', methods=['POST'])
    @login_required
    def run_complete_workflow():
        """Run complete trading workflow."""
        try:
            from execution.trading_executor import TradingExecutor
            
            executor = TradingExecutor(user_id=current_user.id)
            result = executor.run_complete_workflow()
            
            return jsonify({
                'success': True,
                'execution_result': result,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to run complete workflow: {str(e)}'}), 500

    @app.route('/api/trading/start-scheduled-execution', methods=['POST'])
    @login_required
    def start_scheduled_execution():
        """Start scheduled execution of trading workflow."""
        try:
            from execution.trading_executor import TradingExecutor
            
            data = request.get_json()
            interval_hours = data.get('interval_hours', 1)
            
            executor = TradingExecutor(user_id=current_user.id)
            executor.start_scheduled_execution(interval_hours)
            
            return jsonify({
                'success': True,
                'message': f'Scheduled execution started with {interval_hours} hour interval',
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to start scheduled execution: {str(e)}'}), 500

    @app.route('/api/trading/stop-scheduled-execution', methods=['POST'])
    @login_required
    def stop_scheduled_execution():
        """Stop scheduled execution."""
        try:
            from execution.trading_executor import TradingExecutor
            
            executor = TradingExecutor(user_id=current_user.id)
            executor.stop_scheduled_execution()
            
            return jsonify({
                'success': True,
                'message': 'Scheduled execution stopped',
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to stop scheduled execution: {str(e)}'}), 500

    @app.route('/api/trading/execution-status', methods=['GET'])
    @login_required
    def get_execution_status():
        """Get current execution status."""
        try:
            from execution.trading_executor import TradingExecutor
            
            executor = TradingExecutor(user_id=current_user.id)
            status = executor.get_execution_status()
            
            return jsonify({
                'success': True,
                'status': status,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to get execution status: {str(e)}'}), 500

    @app.route('/api/trading/cleanup-dry-run', methods=['POST'])
    @login_required
    def cleanup_dry_run():
        """Clean up dry run portfolios."""
        try:
            from execution.trading_executor import TradingExecutor
            
            executor = TradingExecutor(user_id=current_user.id)
            executor.cleanup_dry_run_portfolios()
            
            return jsonify({
                'success': True,
                'message': 'Dry run portfolios cleaned up',
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to cleanup dry run: {str(e)}'}), 500

    @app.route('/api/analytics/performance-report', methods=['GET'])
    @login_required
    def get_performance_report():
        """Get performance report for strategies."""
        try:
            from analytics.performance_tracker import PerformanceTracker
            
            strategy_names = request.args.getlist('strategies')
            lookback_days = int(request.args.get('lookback_days', 90))
            
            tracker = PerformanceTracker()
            report = tracker.generate_performance_report(strategy_names, lookback_days)
            
            return jsonify({
                'success': True,
                'performance_report': report,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to get performance report: {str(e)}'}), 500

    @app.route('/api/analytics/strategy-comparison', methods=['GET'])
    @login_required
    def get_strategy_comparison():
        """Get strategy comparison analysis."""
        try:
            from analytics.performance_tracker import PerformanceTracker
            
            strategy_names = request.args.getlist('strategies')
            
            tracker = PerformanceTracker()
            comparison = tracker.compare_strategies(strategy_names)
            
            return jsonify({
                'success': True,
                'strategy_comparison': comparison,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to get strategy comparison: {str(e)}'}), 500

    @app.route('/api/ai/analyze-stock', methods=['POST'])
    @login_required
    def analyze_stock_with_ai():
        """Analyze a stock using ChatGPT."""
        try:
            from analysis.chatgpt_analyzer import ChatGPTAnalyzer
            
            data = request.get_json()
            stock_data = data.get('stock_data')
            
            if not stock_data:
                return jsonify({'error': 'Stock data required'}), 400
            
            analyzer = ChatGPTAnalyzer()
            analysis = analyzer.analyze_stock(stock_data)
            
            return jsonify({
                'success': True,
                'ai_analysis': analysis,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to analyze stock: {str(e)}'}), 500

    @app.route('/api/ai/analyze-portfolio', methods=['POST'])
    @login_required
    def analyze_portfolio_with_ai():
        """Analyze a portfolio using ChatGPT."""
        try:
            from analysis.chatgpt_analyzer import ChatGPTAnalyzer
            
            data = request.get_json()
            suggested_stocks = data.get('suggested_stocks', [])
            strategy_name = data.get('strategy_name', 'Unknown')
            
            if not suggested_stocks:
                return jsonify({'error': 'Suggested stocks required'}), 400
            
            analyzer = ChatGPTAnalyzer()
            analysis = analyzer.analyze_portfolio(suggested_stocks, strategy_name)
            
            return jsonify({
                'success': True,
                'ai_analysis': analysis,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to analyze portfolio: {str(e)}'}), 500

    @app.route('/api/ai/compare-strategies', methods=['POST'])
    @login_required
    def compare_strategies_with_ai():
        """Compare strategies using ChatGPT."""
        try:
            from analysis.chatgpt_analyzer import ChatGPTAnalyzer
            
            data = request.get_json()
            strategy_results = data.get('strategy_results', {})
            
            if not strategy_results:
                return jsonify({'error': 'Strategy results required'}), 400
            
            analyzer = ChatGPTAnalyzer()
            comparison = analyzer.compare_strategies(strategy_results)
            
            return jsonify({
                'success': True,
                'ai_comparison': comparison,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to compare strategies: {str(e)}'}), 500

    # Data Sources API Endpoints
    @app.route('/api/data/stock-data', methods=['POST'])
    @login_required
    def get_stock_data():
        """Get stock data from multiple sources including FYERS API."""
        try:
            from data_sources.fyers_provider import get_enhanced_data_provider_manager
            
            data = request.get_json()
            symbol = data.get('symbol')
            
            if not symbol:
                return jsonify({'error': 'Symbol required'}), 400
            
            # Get FYERS connector from app context if available
            fyers_connector = getattr(app, 'fyers_connector', None)
            provider_manager = get_enhanced_data_provider_manager(fyers_connector)
            stock_data = provider_manager.get_stock_data(symbol)
            
            if not stock_data:
                return jsonify({'error': f'No data available for {symbol}'}), 404
            
            return jsonify({
                'success': True,
                'stock_data': stock_data,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to get stock data: {str(e)}'}), 500

    @app.route('/api/data/current-price', methods=['POST'])
    @login_required
    def get_current_price():
        """Get current price for a stock."""
        try:
            from data_sources.fyers_provider import get_enhanced_data_provider_manager
            
            data = request.get_json()
            symbol = data.get('symbol')
            
            if not symbol:
                return jsonify({'error': 'Symbol required'}), 400
            
            # Get FYERS connector from app context if available
            fyers_connector = getattr(app, 'fyers_connector', None)
            provider_manager = get_enhanced_data_provider_manager(fyers_connector)
            price = provider_manager.get_current_price(symbol)
            
            if price is None:
                return jsonify({'error': f'No price data available for {symbol}'}), 404
            
            return jsonify({
                'success': True,
                'symbol': symbol,
                'current_price': price,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to get current price: {str(e)}'}), 500

    @app.route('/api/data/providers', methods=['GET'])
    @login_required
    def get_data_providers():
        """Get list of available data providers."""
        try:
            from data_sources.fyers_provider import get_enhanced_data_provider_manager
            
            # Get FYERS connector from app context if available
            fyers_connector = getattr(app, 'fyers_connector', None)
            provider_manager = get_enhanced_data_provider_manager(fyers_connector)
            providers = provider_manager.get_available_providers()
            
            return jsonify({
                'success': True,
                'providers': providers,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to get data providers: {str(e)}'}), 500

    # Email Alerts API Endpoints
    @app.route('/api/alerts/send-stock-pick', methods=['POST'])
    @login_required
    def send_stock_pick_alert():
        """Send stock pick alert via email."""
        try:
            from alerts.email_alerts import get_email_alert_manager
            
            data = request.get_json()
            stock_data = data.get('stock_data')
            strategy_name = data.get('strategy_name', 'Unknown')
            recommendation = data.get('recommendation', 'BUY')
            
            if not stock_data:
                return jsonify({'error': 'Stock data required'}), 400
            
            alert_manager = get_email_alert_manager()
            success = alert_manager.send_stock_pick_alert(
                current_user.id, stock_data, strategy_name, recommendation
            )
            
            return jsonify({
                'success': success,
                'message': 'Stock pick alert sent' if success else 'Failed to send alert',
                'timestamp': datetime.utcnow().isoformat()
            }), 200 if success else 500
            
        except Exception as e:
            return jsonify({'error': f'Failed to send stock pick alert: {str(e)}'}), 500

    @app.route('/api/alerts/send-portfolio-alert', methods=['POST'])
    @login_required
    def send_portfolio_alert():
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
            
            return jsonify({
                'success': success,
                'message': 'Portfolio alert sent' if success else 'Failed to send alert',
                'timestamp': datetime.utcnow().isoformat()
            }), 200 if success else 500
            
        except Exception as e:
            return jsonify({'error': f'Failed to send portfolio alert: {str(e)}'}), 500

    # Order Management API Endpoints
    @app.route('/api/orders/create-buy-order', methods=['POST'])
    @login_required
    def create_buy_order():
        """Create a buy order."""
        try:
            from orders.order_manager import get_order_manager, OrderType
            
            data = request.get_json()
            symbol = data.get('symbol')
            quantity = data.get('quantity')
            order_type = data.get('order_type', 'MARKET')
            price = data.get('price')
            stop_loss_price = data.get('stop_loss_price')
            take_profit_price = data.get('take_profit_price')
            
            if not symbol or not quantity:
                return jsonify({'error': 'Symbol and quantity required'}), 400
            
            order_manager = get_order_manager()
            order_id = order_manager.create_buy_order(
                current_user.id, symbol, quantity, 
                OrderType(order_type), price, stop_loss_price, take_profit_price
            )
            
            if not order_id:
                return jsonify({'error': 'Failed to create buy order'}), 500
            
            return jsonify({
                'success': True,
                'order_id': order_id,
                'message': 'Buy order created successfully',
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to create buy order: {str(e)}'}), 500

    @app.route('/api/orders/create-sell-order', methods=['POST'])
    @login_required
    def create_sell_order():
        """Create a sell order."""
        try:
            from orders.order_manager import get_order_manager, OrderType
            
            data = request.get_json()
            symbol = data.get('symbol')
            quantity = data.get('quantity')
            order_type = data.get('order_type', 'MARKET')
            price = data.get('price')
            
            if not symbol or not quantity:
                return jsonify({'error': 'Symbol and quantity required'}), 400
            
            order_manager = get_order_manager()
            order_id = order_manager.create_sell_order(
                current_user.id, symbol, quantity, 
                OrderType(order_type), price
            )
            
            if not order_id:
                return jsonify({'error': 'Failed to create sell order'}), 500
            
            return jsonify({
                'success': True,
                'order_id': order_id,
                'message': 'Sell order created successfully',
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to create sell order: {str(e)}'}), 500

    @app.route('/api/orders/cancel-order', methods=['POST'])
    @login_required
    def cancel_order():
        """Cancel an order."""
        try:
            from orders.order_manager import get_order_manager
            
            data = request.get_json()
            order_id = data.get('order_id')
            
            if not order_id:
                return jsonify({'error': 'Order ID required'}), 400
            
            order_manager = get_order_manager()
            success = order_manager.cancel_order(order_id)
            
            if not success:
                return jsonify({'error': 'Failed to cancel order'}), 500
            
            return jsonify({
                'success': True,
                'message': 'Order cancelled successfully',
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to cancel order: {str(e)}'}), 500

    @app.route('/api/orders/user-orders', methods=['GET'])
    @login_required
    def get_user_orders():
        """Get user's orders."""
        try:
            from orders.order_manager import get_order_manager, OrderStatus
            
            status = request.args.get('status')
            order_status = OrderStatus(status) if status else None
            
            order_manager = get_order_manager()
            orders = order_manager.get_user_orders(current_user.id, order_status)
            
            return jsonify({
                'success': True,
                'orders': orders,
                'count': len(orders),
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to get user orders: {str(e)}'}), 500

    @app.route('/api/orders/user-positions', methods=['GET'])
    @login_required
    def get_user_positions():
        """Get user's positions."""
        try:
            from orders.order_manager import get_order_manager
            
            order_manager = get_order_manager()
            positions = order_manager.get_user_positions(current_user.id)
            
            return jsonify({
                'success': True,
                'positions': positions,
                'count': len(positions),
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to get user positions: {str(e)}'}), 500

    @app.route('/api/orders/user-trades', methods=['GET'])
    @login_required
    def get_user_trades():
        """Get user's trades."""
        try:
            from orders.order_manager import get_order_manager
            
            days = int(request.args.get('days', 30))
            
            order_manager = get_order_manager()
            trades = order_manager.get_user_trades(current_user.id, days)
            
            return jsonify({
                'success': True,
                'trades': trades,
                'count': len(trades),
                'days': days,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to get user trades: {str(e)}'}), 500

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
