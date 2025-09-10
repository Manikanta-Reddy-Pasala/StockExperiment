"""
Flask Web Application for the Automated Trading System
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from datastore.database import get_database_manager
from datastore.models import Log, Order, Trade, Position, User, Strategy, SuggestedStock, Configuration
from charting.db_charts import DatabaseCharts
from trading_engine.multi_user_trading_engine import MultiUserTradingEngine
from datetime import datetime
import secrets
import logging

# Import the new API blueprint
from api.blueprint import api_bp

logger = logging.getLogger(__name__)

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
    
    # Register the new API blueprint
    app.register_blueprint(api_bp)
    
    # Authentication routes
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Login page."""
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not username or not password:
                flash('Please enter both username and password.', 'error')
                return render_template('login.html')
            
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.username == username).first()
                
                if user and bcrypt.check_password_hash(user.password_hash, password):
                    if not user.is_active:
                        flash('Your account is deactivated. Please contact an administrator.', 'error')
                        return render_template('login.html')
                    
                    login_user(user)
                    user.last_login = datetime.utcnow()
                    session.commit()
                    
                    next_page = request.args.get('next')
                    return redirect(next_page) if next_page else redirect(url_for('dashboard'))
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
    
    # Web routes (non-API)
    @app.route('/')
    @login_required
    def dashboard():
        """Dashboard page."""
        return render_template('dashboard.html')
    
    @app.route('/portfolio')
    @login_required
    def portfolio():
        """Portfolio page."""
        return render_template('portfolio.html')
    
    @app.route('/orders')
    @login_required
    def orders():
        """Orders page."""
        return render_template('orders.html')
    
    @app.route('/strategies')
    @login_required
    def strategies():
        """Strategies page."""
        return render_template('strategies.html')
    
    @app.route('/alerts')
    @login_required
    def alerts():
        """Alerts page."""
        return render_template('alerts.html')
    
    @app.route('/reports')
    @login_required
    def reports():
        """Reports page."""
        return render_template('reports.html')
    
    @app.route('/settings')
    @login_required
    def settings():
        """Settings page."""
        return render_template('settings.html')
    
    @app.route('/admin/users')
    @login_required
    def admin_users():
        """Admin users page."""
        if not current_user.is_admin:
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('dashboard'))
        return render_template('admin/users.html')
    
    # Legacy API endpoints (for backward compatibility)
    # These will be gradually migrated to the new API structure
    
    @app.route('/api/dashboard/metrics')
    @login_required
    def legacy_dashboard_metrics():
        """Legacy dashboard metrics endpoint."""
        try:
            with db_manager.get_session() as session:
                # Get basic metrics
                total_trades = session.query(Trade).count()
                total_orders = session.query(Order).count()
                total_positions = session.query(Position).count()
                
                # Get recent trades
                recent_trades = session.query(Trade).order_by(Trade.timestamp.desc()).limit(5).all()
                trades_data = []
                for trade in recent_trades:
                    trades_data.append({
                        'id': trade.id,
                        'symbol': trade.symbol,
                        'quantity': trade.quantity,
                        'price': float(trade.price),
                        'pnl': float(trade.pnl),
                        'timestamp': trade.timestamp.isoformat()
                    })
                
                return jsonify({
                    'total_trades': total_trades,
                    'total_orders': total_orders,
                    'total_positions': total_positions,
                    'recent_trades': trades_data
                })
        except Exception as e:
            logger.error(f"Error getting dashboard metrics: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/trades_chart')
    @login_required
    def legacy_trades_chart():
        """Legacy trades chart endpoint."""
        try:
            chart_data = charts.get_trades_chart_data()
            return jsonify(chart_data)
        except Exception as e:
            logger.error(f"Error getting trades chart: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/positions_chart')
    @login_required
    def legacy_positions_chart():
        """Legacy positions chart endpoint."""
        try:
            chart_data = charts.get_positions_chart_data()
            return jsonify(chart_data)
        except Exception as e:
            logger.error(f"Error getting positions chart: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/orders_chart')
    @login_required
    def legacy_orders_chart():
        """Legacy orders chart endpoint."""
        try:
            chart_data = charts.get_orders_chart_data()
            return jsonify(chart_data)
        except Exception as e:
            logger.error(f"Error getting orders chart: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/performance_summary')
    @login_required
    def legacy_performance_summary():
        """Legacy performance summary endpoint."""
        try:
            with db_manager.get_session() as session:
                # Get performance metrics
                trades = session.query(Trade).all()
                
                if trades:
                    total_pnl = sum(trade.pnl for trade in trades)
                    winning_trades = len([t for t in trades if t.pnl > 0])
                    total_trades = len(trades)
                    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                    
                    return jsonify({
                        'total_pnl': float(total_pnl),
                        'win_rate': round(win_rate, 2),
                        'total_trades': total_trades,
                        'winning_trades': winning_trades
                    })
                else:
                    return jsonify({
                        'total_pnl': 0,
                        'win_rate': 0,
                        'total_trades': 0,
                        'winning_trades': 0
                    })
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/market_chart/<symbol>')
    @login_required
    def legacy_market_chart(symbol):
        """Legacy market chart endpoint."""
        try:
            chart_data = charts.get_market_chart_data(symbol)
            return jsonify(chart_data)
        except Exception as e:
            logger.error(f"Error getting market chart for {symbol}: {str(e)}")
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
                'database': 'connected',
                'api_version': 'v1'
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
    app.run(debug=True)
