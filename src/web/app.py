"""
Flask Web Application for the Automated Trading System with Swagger Documentation
"""
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
import uuid
from datetime import datetime
try:
    # Try relative imports first (for normal usage)
    from ..models.database import get_database_manager
    from ..models.models import Log, Order, Trade, Position, User, Strategy, SuggestedStock, Configuration, BrokerConfiguration
    from ..integrations.db_charts import DatabaseCharts
    from ..integrations.multi_user_trading_engine import get_trading_engine
    from ..services.user_service import get_user_service
    from ..services.broker_service import get_broker_service
    from ..services.dashboard_service import get_dashboard_service
    from ..services.portfolio_service import get_portfolio_service
    from ..services.stock_screening_service import get_stock_screening_service, StrategyType
    from ..utils.api_logger import APILogger, log_flask_route
except ImportError:
    # Fall back to absolute imports (for testing)
    from models.database import get_database_manager
    from models.models import Log, Order, Trade, Position, User, Strategy, SuggestedStock, Configuration, BrokerConfiguration
    from integrations.db_charts import DatabaseCharts
    from integrations.multi_user_trading_engine import get_trading_engine
    from services.user_service import get_user_service
    from services.broker_service import get_broker_service
    from services.dashboard_service import get_dashboard_service
    from services.portfolio_service import get_portfolio_service
    from services.stock_screening_service import get_stock_screening_service, StrategyType
    from utils.api_logger import APILogger, log_flask_route
from datetime import datetime
import secrets
import sys
import os

# Configure logging
from ..config.logging_config import setup_logging
setup_logging()


def create_app():
    """Create Flask application."""
    app = Flask(__name__)
    
    # Generate a secret key for sessions
    app.secret_key = secrets.token_hex(16)
    
    # Add comprehensive request/response logging middleware
    try:
        from ..utils.request_logger_middleware import RequestLoggerMiddleware
        RequestLoggerMiddleware(app)
        print("🔍 Request logging middleware enabled - All API calls will be logged to console")
    except ImportError:
        from utils.request_logger_middleware import RequestLoggerMiddleware
        RequestLoggerMiddleware(app)
        print("🔍 Request logging middleware enabled - All API calls will be logged to console")
    
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    @login_manager.unauthorized_handler
    def handle_unauthorized():
        # Return JSON for API requests instead of redirecting to HTML
        try:
            path = request.path or ''
            accepts_json = 'application/json' in (request.headers.get('Accept') or '')
            is_api_request = path.startswith('/api') or '/api/' in path or path.startswith('/brokers/') and '/api/' in path
            if is_api_request or accepts_json:
                return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        except Exception:
            pass
        return redirect(url_for('login'))
    
    # Initialize Flask-Bcrypt
    bcrypt = Bcrypt(app)
    
    # Initialize database
    db_manager = get_database_manager()

    # Initialize services
    user_service = get_user_service(db_manager, bcrypt)
    broker_service = get_broker_service()
    dashboard_service = get_dashboard_service()
    portfolio_service = get_portfolio_service()
    stock_screening_service = get_stock_screening_service(broker_service)
    
    # Initialize new services
    from ..services.cache_service import get_cache_service
    from ..services.token_manager_service import get_token_manager
    from ..services.scheduler_service import get_scheduler
    
    cache_service = get_cache_service()
    token_manager = get_token_manager()
    scheduler = get_scheduler()
    
    # Start scheduler service
    scheduler.start()
    
    # Register FYERS token refresh callback
    try:
        from ..services.brokers.fyers_token_refresh import register_fyers_refresh_callback
        register_fyers_refresh_callback()
    except Exception as e:
        app.logger.warning(f"Could not register FYERS refresh callback: {e}")
    
    # Schedule default tasks
    scheduler.schedule_data_cleanup(interval_hours=24)
    
    # Initialize charting
    charts = DatabaseCharts(db_manager)
    
    # Initialize API with all namespaces
    # from .api import create_api
    # api = create_api(app)
    
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

            try:
                user = user_service.login_user(username, password)
                login_user(user, remember=remember)
                next_page = request.args.get('next')
                return redirect(next_page) if next_page else redirect(url_for('dashboard'))
            except ValueError as e:
                flash(str(e), 'error')
        
        return render_template('login.html')
    
    
    @app.route('/logout')
    @login_required
    def logout():
        """Logout user."""
        logout_user()
        flash('You have been logged out successfully.', 'info')
        return redirect(url_for('login'))
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """Register new user."""
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            if not all([username, email, password, confirm_password]):
                flash('Please fill in all fields.', 'error')
                return render_template('login.html')
            
            if password != confirm_password:
                flash('Passwords do not match.', 'error')
                return render_template('login.html')
            
            if len(password) < 6:
                flash('Password must be at least 6 characters long.', 'error')
                return render_template('login.html')

            try:
                user_service.register_user(username, email, password)
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
            except ValueError as e:
                flash(str(e), 'error')
                return render_template('login.html')
        
        return render_template('login.html')
    
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})
    
    @app.route('/')
    @login_required
    def dashboard():
        """Dashboard page."""
        return render_template('dashboard.html')
    
    @app.route('/demo')
    def demo_dashboard():
        """Demo dashboard page (no login required)."""
        return render_template('dashboard.html')
    
    # Auto-login route for demo purposes
    @app.route('/auto-login')
    def auto_login():
        """Auto-login for demo purposes."""
        try:
            # Try to find or create a demo user
            demo_user = user_service.get_or_create_demo_user()
            if demo_user:
                login_user(demo_user)
                flash('Demo login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Demo login failed. Using demo mode.', 'warning')
                return redirect(url_for('demo_dashboard'))
        except Exception as e:
            app.logger.error(f"Auto-login error: {e}")
            flash('Demo login unavailable. Using demo mode.', 'warning')
            return redirect(url_for('demo_dashboard'))
    
    @app.route('/logs')
    @login_required
    def logs():
        """Logs page."""
        return render_template('logs.html')
    
    @app.route('/orders')
    @login_required
    def orders():
        """Orders page."""
        return render_template('orders.html')
    
    @app.route('/trades')
    @login_required
    def trades():
        """Trades page."""
        return render_template('trades.html')
    
    @app.route('/portfolio')
    @login_required
    def portfolio():
        """Portfolio page."""
        return render_template('portfolio.html')
    
    @app.route('/strategies')
    @login_required
    def strategies():
        """Strategies page."""
        return render_template('strategies.html')
    
    @app.route('/ml-prediction')
    @login_required
    def ml_prediction():
        """ML Prediction page."""
        return render_template('ml_prediction.html')
    
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
    
    @app.route('/brokers')
    @login_required
    def brokers():
        """Brokers page."""
        return render_template('brokers.html')
    
    @app.route('/settings')
    @login_required
    def settings():
        """Settings page."""
        return render_template('settings.html')
    
    # Admin routes
    @app.route('/admin/users')
    @login_required
    def admin_users():
        """Admin page for managing users."""
        if not current_user.is_admin:
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('dashboard'))
        return render_template('admin/users.html')
    
    # API Routes for User Management
    @app.route('/api/admin/users', methods=['GET'])
    @login_required
    def api_get_users():
        """Get all users for admin management."""
        if not current_user.is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        try:
            users_data = user_service.get_all_users()
            return jsonify({'users': users_data})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/users', methods=['POST'])
    @login_required
    def api_create_user():
        """Create a new user."""
        if not current_user.is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        try:
            data = request.get_json()
            new_user = user_service.create_user(data)
            return jsonify({
                'message': 'User created successfully',
                'user': new_user
            }), 201
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
    @login_required
    def api_update_user(user_id):
        """Update a user."""
        if not current_user.is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        try:
            data = request.get_json()
            updated_user = user_service.update_user(user_id, data)
            return jsonify({
                'message': 'User updated successfully',
                'user': updated_user
            })
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
    @login_required
    def api_delete_user(user_id):
        """Delete a user."""
        if not current_user.is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        try:
            user_service.delete_user(user_id, current_user.id)
            return jsonify({'message': 'User deleted successfully'})
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Broker API Routes
    @app.route('/api/brokers/fyers', methods=['GET'])
    @login_required
    def api_get_fyers_info():
        """Get FYERS broker information."""
        try:
            app.logger.info(f"Fetching FYERS broker info for user {current_user.id}")
            config = broker_service.get_broker_config('fyers', current_user.id)
            
            if not config:
                app.logger.info("No FYERS configuration found for user")
                return jsonify({
                    'success': True, 'client_id': '', 'access_token': False, 'connected': False, 'last_updated': '-',
                    'stats': {'total_orders': 0, 'successful_orders': 0, 'pending_orders': 0, 'failed_orders': 0, 'last_order_time': '-', 'api_response_time': '-'}
                })

            stats = broker_service.get_broker_stats('fyers', current_user.id)
            config['access_token'] = bool(config.get('access_token'))
            
            return jsonify({'success': True, **config, 'stats': stats})
        except Exception as e:
            app.logger.error(f"Error getting FYERS broker info for user {current_user.id}: {str(e)}", exc_info=True)
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/brokers/fyers/test', methods=['POST'])
    @login_required
    def api_test_fyers_connection():
        """Test FYERS broker connection."""
        try:
            app.logger.info(f"Testing FYERS broker connection for user {current_user.id}")
            result = broker_service.test_fyers_connection(current_user.id)
            return jsonify(result)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error testing FYERS connection for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/brokers/fyers/config', methods=['POST'])
    @login_required
    def api_save_fyers_config():
        """Save FYERS broker configuration."""
        try:
            app.logger.info(f"Saving FYERS configuration for user {current_user.id}")
            data = request.get_json()
            if not data.get('client_id'):
                return jsonify({'success': False, 'error': 'Client ID is required'}), 400

            config = broker_service.save_broker_config('fyers', data, current_user.id)
            
            response_data = {'success': True, 'message': 'FYERS configuration saved successfully', 'config': config}

            if data.get('secret_key'):
                try:
                    auth_url = broker_service.generate_fyers_auth_url(current_user.id)
                    response_data['auth_url'] = auth_url
                    response_data['message'] = 'FYERS configuration saved successfully. OAuth2 authorization URL generated automatically.'
                except Exception as e:
                    app.logger.error(f"Error auto-generating OAuth2 auth URL for user {current_user.id}: {str(e)}")

            return jsonify(response_data)
        except Exception as e:
            app.logger.error(f"Error saving FYERS configuration for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/brokers/fyers/config', methods=['PUT'])
    @login_required
    def api_update_fyers_config():
        """Update FYERS broker configuration."""
        try:
            app.logger.info(f"Updating FYERS configuration for user {current_user.id}")
            data = request.get_json()
            config = broker_service.save_broker_config('fyers', data, current_user.id)
            return jsonify({'success': True, 'message': 'FYERS configuration updated successfully', 'config': config})
        except Exception as e:
            app.logger.error(f"Error updating FYERS configuration for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/brokers/fyers/refresh-token', methods=['POST'])
    @login_required
    def api_refresh_fyers_token():
        """Refresh FYERS access token."""
        try:
            app.logger.info(f"Refreshing FYERS token for user {current_user.id}")
            auth_url = broker_service.generate_fyers_auth_url(current_user.id)
            return jsonify({
                'success': True,
                'message': 'Re-authentication required. Please complete the authorization process.',
                'auth_url': auth_url
            })
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error refreshing FYERS token for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/brokers/fyers/auth-url', methods=['POST'])
    @login_required
    def api_generate_fyers_auth_url():
        """Generate FYERS OAuth2 authorization URL using database configuration."""
        try:
            app.logger.info(f"Generating FYERS auth URL for user {current_user.id}")
            auth_url = broker_service.generate_fyers_auth_url(current_user.id)
            return jsonify({
                'success': True,
                'auth_url': auth_url,
                'message': 'Authorization URL generated successfully.'
            })
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error generating FYERS auth URL for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/brokers/fyers/exchange-token', methods=['POST'])
    @login_required
    def api_exchange_fyers_auth_code():
        """Exchange FYERS authorization code for access token."""
        try:
            app.logger.info(f"Exchanging FYERS auth code for user {current_user.id}")
            data = request.get_json()
            auth_code = data.get('auth_code')
            if not auth_code:
                return jsonify({'success': False, 'error': 'Auth Code is required'}), 400
            
            result = broker_service.exchange_fyers_auth_code(current_user.id, auth_code)
            
            return jsonify({
                'success': True,
                'message': 'Access token obtained and saved successfully',
                'access_token': result['access_token'][:20] + '...'
            })
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error exchanging FYERS auth code for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/brokers/fyers/funds', methods=['GET'])
    @login_required
    def api_get_fyers_funds():
        """Get FYERS user funds."""
        try:
            app.logger.info(f"Fetching FYERS funds for user {current_user.id}")
            result = broker_service.get_fyers_funds(current_user.id)
            if 'error' in result:
                return jsonify({'success': False, 'error': result['error']}), 400
            return jsonify({'success': True, 'data': result})
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error getting FYERS funds for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/brokers/fyers/holdings', methods=['GET'])
    @login_required
    def api_get_fyers_holdings():
        """Get FYERS user holdings."""
        try:
            app.logger.info(f"Fetching FYERS holdings for user {current_user.id}")
            result = broker_service.get_fyers_holdings(current_user.id)
            if 'error' in result:
                return jsonify({'success': False, 'error': result['error']}), 400
            return jsonify({'success': True, 'data': result})
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error getting FYERS holdings for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/brokers/fyers/positions', methods=['GET'])
    @login_required
    def api_get_fyers_positions():
        """Get FYERS user positions."""
        try:
            app.logger.info(f"Fetching FYERS positions for user {current_user.id}")
            result = broker_service.get_fyers_positions(current_user.id)
            if 'error' in result:
                return jsonify({'success': False, 'error': result['error']}), 400
            return jsonify({'success': True, 'data': result})
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error getting FYERS positions for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/brokers/fyers/orderbook', methods=['GET'])
    @login_required
    def api_get_fyers_orderbook():
        """Get FYERS user orderbook."""
        try:
            app.logger.info(f"Fetching FYERS orderbook for user {current_user.id}")
            result = broker_service.get_fyers_orderbook(current_user.id)
            if 'error' in result:
                return jsonify({'success': False, 'error': result['error']}), 400
            return jsonify({'success': True, 'data': result})
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error getting FYERS orderbook for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/brokers/fyers/tradebook', methods=['GET'])
    @login_required
    def api_get_fyers_tradebook():
        """Get FYERS user tradebook."""
        try:
            app.logger.info(f"Fetching FYERS tradebook for user {current_user.id}")
            result = broker_service.get_fyers_tradebook(current_user.id)
            if 'error' in result:
                return jsonify({'success': False, 'error': result['error']}), 400
            return jsonify({'success': True, 'data': result})
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error getting FYERS tradebook for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/brokers/fyers/quotes', methods=['GET'])
    @login_required
    @log_flask_route("get_fyers_quotes")
    def api_get_fyers_quotes():
        """Get FYERS market quotes."""
        try:
            symbols = request.args.get('symbols', '')
            app.logger.info(f"Fetching FYERS quotes for symbols: {symbols} for user {current_user.id}")
            
            # Log API call to broker service
            APILogger.log_request(
                service_name="BrokerService",
                method_name="get_fyers_quotes",
                request_data={'symbols': symbols},
                user_id=current_user.id
            )
            
            result = broker_service.get_fyers_quotes(current_user.id, symbols)
            
            # Log response from broker service
            APILogger.log_response(
                service_name="BrokerService",
                method_name="get_fyers_quotes",
                response_data=result,
                user_id=current_user.id
            )
            
            if 'error' in result:
                return jsonify({'success': False, 'error': result['error']}), 400
            return jsonify({'success': True, 'data': result})
        except ValueError as e:
            APILogger.log_error(
                service_name="FlaskAPI",
                method_name="get_fyers_quotes",
                error=e,
                user_id=current_user.id if current_user else None
            )
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error getting FYERS quotes for user {current_user.id}: {str(e)}")
            APILogger.log_error(
                service_name="FlaskAPI",
                method_name="get_fyers_quotes",
                error=e,
                user_id=current_user.id if current_user else None
            )
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/brokers/fyers/history', methods=['GET'])
    @login_required
    def api_get_fyers_history():
        """Get FYERS historical data."""
        try:
            symbol = request.args.get('symbol', '')
            resolution = request.args.get('resolution', 'D')
            range_from = request.args.get('range_from')
            range_to = request.args.get('range_to')
            app.logger.info(f"Fetching FYERS historical data for symbol: {symbol} for user {current_user.id}")
            result = broker_service.get_fyers_history(current_user.id, symbol, resolution, range_from, range_to)
            if 'error' in result:
                return jsonify({'success': False, 'error': result['error']}), 400
            return jsonify({'success': True, 'data': result})
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error getting FYERS historical data for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    # Dashboard API Routes - Updated to use unified multi-broker system
    @app.route('/api/dashboard/metrics', methods=['GET'])
    @login_required
    def api_get_dashboard_metrics():
        """Get dashboard metrics using unified multi-broker system."""
        try:
            from .routes.unified_routes import api_get_portfolio_summary
            return api_get_portfolio_summary()
        except Exception as e:
            app.logger.error(f"Error fetching dashboard metrics for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    # Portfolio API Routes (Legacy - keeping for backward compatibility)
    @app.route('/api/portfolio/holdings', methods=['GET'])
    @login_required
    def api_get_portfolio_holdings_legacy():
        """Get portfolio holdings using FYERS API (Legacy endpoint)."""
        try:
            app.logger.info(f"Fetching portfolio holdings for user {current_user.id}")
            result = portfolio_service.get_portfolio_holdings(current_user.id)
            return jsonify(result)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error fetching portfolio holdings for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/portfolio/positions', methods=['GET'])
    @login_required
    def api_get_portfolio_positions():
        """Get portfolio positions using FYERS API."""
        try:
            app.logger.info(f"Fetching portfolio positions for user {current_user.id}")
            result = portfolio_service.get_portfolio_positions(current_user.id)
            return jsonify(result)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error fetching portfolio positions for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    # Orders API Routes
    @app.route('/api/orders/history', methods=['GET'])
    @login_required
    def api_get_orders_history():
        """Get orders history using FYERS API."""
        try:
            app.logger.info(f"Fetching orders history for user {current_user.id}")
            orderbook_data = broker_service.get_fyers_orderbook(current_user.id)

            if orderbook_data.get('success') and orderbook_data.get('data'):
                orders = orderbook_data['data'].get('orderBook', [])
                processed_orders = [
                    {
                        'id': o.get('id', ''), 'symbol': o.get('symbol', ''), 'side': o.get('side', ''),
                        'type': o.get('type', ''), 'quantity': o.get('qty', 0), 'price': o.get('limitPrice', 0),
                        'status': o.get('status', ''), 'order_time': o.get('orderDateTime', ''),
                        'filled_quantity': o.get('filledQty', 0), 'remaining_quantity': o.get('remainingQty', 0),
                        'product': o.get('product', '')
                    } for o in orders
                ]
                return jsonify({'success': True, 'data': processed_orders, 'last_updated': datetime.now().isoformat()})
            else:
                return jsonify({'success': False, 'error': 'Failed to fetch orders data from FYERS'}), 400
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error fetching orders history for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/orders/trades', methods=['GET'])
    @login_required
    def api_get_trades_history():
        """Get trades history using FYERS API."""
        try:
            app.logger.info(f"Fetching trades history for user {current_user.id}")
            tradebook_data = broker_service.get_fyers_tradebook(current_user.id)

            if tradebook_data.get('success') and tradebook_data.get('data'):
                trades = tradebook_data['data'].get('tradeBook', [])
                processed_trades = [
                    {
                        'id': t.get('id', ''), 'symbol': t.get('symbol', ''), 'side': t.get('side', ''),
                        'quantity': t.get('qty', 0), 'price': t.get('tradedPrice', 0),
                        'trade_time': t.get('tradeDateTime', ''), 'order_id': t.get('orderNumber', ''),
                        'product': t.get('product', ''), 'pnl': t.get('pnl', 0)
                    } for t in trades
                ]
                return jsonify({'success': True, 'data': processed_trades, 'last_updated': datetime.now().isoformat()})
            else:
                return jsonify({'success': False, 'error': 'Failed to fetch trades data from FYERS'}), 400
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error fetching trades history for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    # Market Data API Routes
    @app.route('/api/market/quotes', methods=['GET'])
    @login_required
    def api_get_market_quotes():
        """Get market quotes using FYERS API."""
        try:
            app.logger.info(f"Fetching market quotes for user {current_user.id}")
            symbols = request.args.get('symbols', 'NSE:NIFTY50-INDEX,NSE:SENSEX-INDEX,NSE:NIFTYBANK-INDEX,NSE:NIFTYIT-INDEX')
            quotes_data = broker_service.get_fyers_quotes(current_user.id, symbols)
            
            if quotes_data.get('success') and quotes_data.get('data'):
                # Processing can be moved to a service if it becomes more complex
                processed_quotes = []
                for symbol, quote in quotes_data['data'].items():
                    if quote.get('v'):
                        processed_quotes.append({
                            'symbol': symbol,
                            'price': quote['v'].get('lp', 0),
                            'change': quote['v'].get('ch', 0),
                            'change_percent': quote['v'].get('chp', 0),
                            'volume': quote['v'].get('volume', 0),
                            'high': quote['v'].get('h', 0),
                            'low': quote['v'].get('l', 0),
                            'open': quote['v'].get('open_price', 0),
                            'close': quote['v'].get('prev_close_price', 0)
                        })
                return jsonify({'success': True, 'data': processed_quotes, 'last_updated': datetime.now().isoformat()})
            else:
                return jsonify({'success': False, 'error': 'Failed to fetch quotes data from FYERS'}), 400
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error fetching market quotes for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/market/historical', methods=['GET'])
    @login_required
    def api_get_historical_data():
        """Get historical data using FYERS API."""
        try:
            app.logger.info(f"Fetching historical data for user {current_user.id}")
            symbol = request.args.get('symbol', 'NSE:NIFTY50-INDEX')
            resolution = request.args.get('resolution', 'D')
            range_from = request.args.get('from')
            range_to = request.args.get('to')

            historical_data = broker_service.get_fyers_history(current_user.id, symbol, resolution, range_from, range_to)

            if historical_data.get('success') and historical_data.get('data'):
                # This processing can also be moved to a service
                processed_data = []
                for candle in historical_data['data'].get('candles', []):
                    processed_data.append({
                        'timestamp': candle[0], 'open': candle[1], 'high': candle[2],
                        'low': candle[3], 'close': candle[4], 'volume': candle[5] if len(candle) > 5 else 0
                    })
                return jsonify({'success': True, 'data': processed_data, 'symbol': symbol, 'resolution': resolution, 'last_updated': datetime.now().isoformat()})
            else:
                return jsonify({'success': False, 'error': 'Failed to fetch historical data from FYERS'}), 400
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error fetching historical data for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/market/overview', methods=['GET'])
    @login_required
    def api_get_market_overview():
        """Get market overview for NIFTY indices using broker_service quotes directly."""
        try:
            app.logger.info(f"Fetching market overview for user {current_user.id}")
            
            # Use existing broker_service which already handles auth/config
            symbols_map = {
                'NIFTY 50': 'NSE:NIFTY50-INDEX',
                'BANK NIFTY': 'NSE:NIFTYBANK-INDEX',
                'NIFTY MIDCAP 150': 'NSE:NIFTYMIDCAP150-INDEX',
                'NIFTY SMALLCAP 250': 'NSE:NIFTYSMALLCAP250-INDEX'
            }
            symbols = ','.join(symbols_map.values())
            quotes_data = broker_service.get_fyers_quotes(current_user.id, symbols)
            
            if not quotes_data or not quotes_data.get('success'):
                error_msg = (quotes_data or {}).get('error', 'Failed to fetch quotes data from FYERS')
                app.logger.warning(f"Market overview quotes fetch failed: {error_msg}")
                return jsonify({'success': False, 'error': error_msg, 'data': {}, 'source': 'quotes'}), 400
            
            payload = quotes_data.get('data') or {}
            market = {}
            
            # Handle payload either as dict keyed by symbol or list under 'd'
            if isinstance(payload, dict):
                for symbol, quote in payload.items():
                    for name, fy_symbol in symbols_map.items():
                        if symbol == fy_symbol:
                            v = quote.get('v', quote)
                            lp = float(v.get('lp', 0))
                            pc = float(v.get('prev_close_price', v.get('pc', lp)))
                            chp = float(v.get('chp', ((lp - pc) / pc * 100) if pc > 0 else 0))
                            market[name] = {
                                'current_price': round(lp, 2),
                                'change_percent': round(chp, 2),
                                'is_positive': chp >= 0
                            }
            elif isinstance(payload, list):
                for item in payload:
                    symbol = item.get('symbol') or item.get('n')
                    for name, fy_symbol in symbols_map.items():
                        if symbol == fy_symbol:
                            v = item.get('v', item)
                            lp = float(v.get('lp', 0))
                            pc = float(v.get('prev_close_price', v.get('pc', lp)))
                            chp = float(v.get('chp', ((lp - pc) / pc * 100) if pc > 0 else 0))
                            market[name] = {
                                'current_price': round(lp, 2),
                                'change_percent': round(chp, 2),
                                'is_positive': chp >= 0
                            }
            
            return jsonify({'success': True, 'data': market, 'last_updated': datetime.now().isoformat(), 'source': 'broker_service.quotes'})
        except Exception as e:
            app.logger.error(f"Error fetching market overview: {e}")
            return jsonify({'success': False, 'error': str(e), 'data': {}, 'source': 'exception'}), 500

    @app.route('/api/dashboard/portfolio-holdings', methods=['GET'])
    @login_required
    def api_get_portfolio_holdings():
        """Get portfolio holdings using unified multi-broker system."""
        try:
            from .routes.unified_routes import api_get_holdings
            return api_get_holdings()
        except Exception as e:
            app.logger.error(f"Error fetching portfolio holdings: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/dashboard/pending-orders', methods=['GET'])
    @login_required
    def api_get_pending_orders():
        """Get pending orders using unified multi-broker system."""
        try:
            from .routes.unified_routes import api_get_pending_orders as unified_pending_orders
            return unified_pending_orders()
        except Exception as e:
            app.logger.error(f"Error fetching pending orders: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/dashboard/recent-orders', methods=['GET'])
    @login_required
    def api_get_recent_orders():
        """Get recent orders using unified multi-broker system."""
        try:
            from .routes.unified_routes import api_get_recent_activity as unified_recent_activity
            return unified_recent_activity()
        except Exception as e:
            app.logger.error(f"Error fetching recent orders: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/dashboard/portfolio-performance', methods=['GET'])
    @login_required
    def api_get_portfolio_performance():
        """Get portfolio performance data using unified multi-broker system."""
        try:
            from .routes.unified_routes import api_get_portfolio_performance as unified_portfolio_performance
            return unified_portfolio_performance()
        except Exception as e:
            app.logger.error(f"Error fetching portfolio performance: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # Suggested Stocks API Routes - Updated to use unified multi-broker system
    @app.route('/api/suggested-stocks', methods=['GET'])
    @login_required
    def api_get_suggested_stocks():
        """Get suggested stocks using unified multi-broker system."""
        try:
            from .routes.unified_routes import api_get_suggested_stocks as unified_suggested_stocks
            return unified_suggested_stocks()
        except Exception as e:
            app.logger.error(f"Error getting suggested stocks for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/suggested-stocks/refresh', methods=['POST'])
    @login_required
    def api_refresh_suggested_stocks():
        """Refresh suggested stocks by running screening again."""
        try:
            app.logger.info(f"Refreshing suggested stocks for user {current_user.id}")
            data = request.get_json() or {}
            strategies = data.get('strategies', [])
            
            strategy_types = [StrategyType(s) for s in strategies if s in StrategyType._value2member_map_]
            if not strategy_types:
                strategy_types = [StrategyType.DEFAULT_RISK, StrategyType.HIGH_RISK]

            suggested_stocks = stock_screening_service.screen_stocks(strategy_types, current_user.id)
            
            stocks_data = [
                {
                    'symbol': stock.symbol, 'name': stock.name, 'selection_date': datetime.now().strftime('%Y-%m-%d'),
                    'selection_price': round(stock.current_price, 2), 'current_price': round(stock.current_price, 2),
                    'quantity': 10, 'investment': round(stock.current_price * 10, 2),
                    'current_value': round(stock.current_price * 10, 2), 'strategy': stock.strategy, 'status': 'Active',
                    'recommendation': stock.recommendation, 'target_price': round(stock.target_price, 2) if stock.target_price else None,
                    'stop_loss': round(stock.stop_loss, 2) if stock.stop_loss else None, 'reason': stock.reason,
                    'market_cap': round(stock.market_cap, 2), 'pe_ratio': round(stock.pe_ratio, 2) if stock.pe_ratio else None,
                    'pb_ratio': round(stock.pb_ratio, 2) if stock.pb_ratio else None, 'roe': round(stock.roe * 100, 2) if stock.roe else None,
                    'sales_growth': round(stock.sales_growth, 2) if stock.sales_growth else None
                } for stock in suggested_stocks
            ]
            
            return jsonify({'success': True, 'data': stocks_data, 'total': len(stocks_data), 'strategies': [s.value for s in strategy_types], 'refreshed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        except Exception as e:
            app.logger.error(f"Error refreshing suggested stocks for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    # Settings API Routes
    @app.route('/api/settings', methods=['GET'])
    @login_required
    def api_get_settings():
        """Get user settings."""
        try:
            from ..services.user_settings_service import get_user_settings_service
            user_settings_service = get_user_settings_service()
            settings = user_settings_service.get_user_settings(current_user.id)
            return jsonify({'success': True, 'settings': settings})
        except Exception as e:
            app.logger.error(f"Error getting settings for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/settings', methods=['POST'])
    @login_required
    def api_save_settings():
        """Save user settings."""
        try:
            data = request.get_json()
            from ..services.user_settings_service import get_user_settings_service
            user_settings_service = get_user_settings_service()
            
            # Save settings to database
            saved_settings = user_settings_service.save_user_settings(current_user.id, data)
            
            app.logger.info(f"Settings saved for user {current_user.id}: {data}")
            return jsonify({'success': True, 'message': 'Settings saved successfully', 'settings': saved_settings})
        except Exception as e:
            app.logger.error(f"Error saving settings for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    # Broker Selection API
    @app.route('/api/brokers/current', methods=['GET'])
    @login_required
    def api_get_current_broker():
        """Get the currently selected broker."""
        try:
            from ..services.user_settings_service import get_user_settings_service
            user_settings_service = get_user_settings_service()
            broker_provider = user_settings_service.get_broker_provider(current_user.id)
            return jsonify({'success': True, 'broker': broker_provider})
        except Exception as e:
            app.logger.error(f"Error getting current broker for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    @app.route('/api/brokers/current', methods=['POST'])
    @login_required
    def api_set_current_broker():
        """Set the currently selected broker."""
        try:
            data = request.get_json()
            broker = data.get('broker', 'fyers')
            from ..services.user_settings_service import get_user_settings_service
            user_settings_service = get_user_settings_service()
            
            # Save broker provider to user settings
            success = user_settings_service.set_broker_provider(current_user.id, broker)
            
            if success:
                app.logger.info(f"Setting current broker to {broker} for user {current_user.id}")
                return jsonify({'success': True, 'message': f'Broker set to {broker}'})
            else:
                return jsonify({'success': False, 'error': 'Failed to save broker setting'}), 500
        except Exception as e:
            app.logger.error(f"Error setting current broker for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500

    # Register unified multi-broker routes
    try:
        from .routes.unified_routes import unified_bp
        app.register_blueprint(unified_bp)
        app.logger.info("Unified multi-broker routes registered successfully")
    except ImportError as e:
        app.logger.warning(f"Unified routes not available: {e}")

    # Register broker-specific blueprints
    from .routes.brokers import fyers_bp, zerodha_bp, simulator_bp
    app.register_blueprint(fyers_bp)
    app.register_blueprint(zerodha_bp)
    app.register_blueprint(simulator_bp)

    # Add fallback functionality to existing routes with mock data
    try:
        from ..services.mock_data_service import get_mock_data_service
        mock_service = get_mock_data_service()
        
        # Override the existing market overview route to include fallback
        @app.route('/api/market/overview', methods=['GET'])
        def api_market_overview_fallback():
            """Market overview with mock data fallback."""
            try:
                if current_user and current_user.is_authenticated:
                    # Try the original function
                    return api_get_market_overview()
                else:
                    # Return mock data for demo
                    return jsonify(mock_service.get_market_overview())
            except Exception as e:
                app.logger.error(f"Market overview error, using mock data: {e}")
                return jsonify(mock_service.get_market_overview())
        
        # Override dashboard metrics route
        @app.route('/api/dashboard/metrics', methods=['GET'])
        def api_dashboard_metrics_fallback():
            """Dashboard metrics with mock data fallback."""
            try:
                if current_user and current_user.is_authenticated:
                    return api_get_dashboard_metrics()
                else:
                    return jsonify(mock_service.get_portfolio_summary())
            except Exception as e:
                app.logger.error(f"Dashboard metrics error, using mock data: {e}")
                return jsonify(mock_service.get_portfolio_summary())
        
        # Override portfolio holdings route
        @app.route('/api/dashboard/portfolio-holdings', methods=['GET'])
        def api_portfolio_holdings_fallback():
            """Portfolio holdings with mock data fallback."""
            try:
                if current_user and current_user.is_authenticated:
                    return api_get_portfolio_holdings()
                else:
                    return jsonify(mock_service.get_portfolio_holdings())
            except Exception as e:
                app.logger.error(f"Portfolio holdings error, using mock data: {e}")
                return jsonify(mock_service.get_portfolio_holdings())
        
        # Override recent orders route
        @app.route('/api/dashboard/recent-orders', methods=['GET'])
        def api_recent_orders_fallback():
            """Recent orders with mock data fallback."""
            try:
                if current_user and current_user.is_authenticated:
                    return api_get_recent_orders()
                else:
                    limit = request.args.get('limit', 5, type=int)
                    return jsonify(mock_service.get_recent_orders(limit))
            except Exception as e:
                app.logger.error(f"Recent orders error, using mock data: {e}")
                limit = request.args.get('limit', 5, type=int)
                return jsonify(mock_service.get_recent_orders(limit))
        
        # Override pending orders route
        @app.route('/api/dashboard/pending-orders', methods=['GET'])
        def api_pending_orders_fallback():
            """Pending orders with mock data fallback."""
            try:
                if current_user and current_user.is_authenticated:
                    return api_get_pending_orders()
                else:
                    return jsonify(mock_service.get_pending_orders())
            except Exception as e:
                app.logger.error(f"Pending orders error, using mock data: {e}")
                return jsonify(mock_service.get_pending_orders())
        
        # Override portfolio performance route
        @app.route('/api/dashboard/portfolio-performance', methods=['GET'])
        def api_portfolio_performance_fallback():
            """Portfolio performance with mock data fallback."""
            try:
                if current_user and current_user.is_authenticated:
                    return api_get_portfolio_performance()
                else:
                    period = request.args.get('period', '1W')
                    return jsonify(mock_service.get_portfolio_performance(period))
            except Exception as e:
                app.logger.error(f"Portfolio performance error, using mock data: {e}")
                period = request.args.get('period', '1W')
                return jsonify(mock_service.get_portfolio_performance(period))
        
        
    except ImportError as e:
        app.logger.error(f"Could not enable mock data fallback: {e}")
    except Exception as e:
        app.logger.error(f"Error setting up fallback routes: {e}")

    # Register ML prediction blueprints
    try:
        from .routes.ml import ml_bp
        app.register_blueprint(ml_bp)
        app.logger.info("ML prediction routes registered successfully")
    except ImportError as e:
        app.logger.warning(f"ML prediction routes not available: {e}")
        app.logger.warning("ML functionality will be disabled")

    # Register strategy blueprints
    try:
        from .routes.strategy_routes import strategy_bp
        app.register_blueprint(strategy_bp)
        app.logger.info("Strategy routes registered successfully")
    except ImportError as e:
        app.logger.warning(f"Strategy routes not available: {e}")
    
    # Register strategy settings blueprints
    try:
        from .routes.strategy_settings_routes import strategy_settings_bp
        app.register_blueprint(strategy_settings_bp)
        app.logger.info("Strategy settings routes registered successfully")
    except ImportError as e:
        app.logger.warning(f"Strategy settings routes not available: {e}")
        app.logger.warning("Strategy functionality will be disabled")

    # Individual broker page routes
    @app.route('/brokers/fyers')
    @login_required
    def brokers_fyers():
        """FYERS broker page."""
        return render_template('brokers/fyers.html')

    @app.route('/brokers/zerodha')
    @login_required
    def brokers_zerodha():
        """Zerodha broker page."""
        return render_template('brokers/zerodha.html')

    @app.route('/brokers/simulator')
    @login_required
    def brokers_simulator():
        """Simulator broker page."""
        return render_template('brokers/simulator.html')


    # Add missing API endpoints that frontend expects
    @app.route('/api/portfolio', methods=['GET'])
    @login_required
    def api_get_portfolio():
        """Get portfolio summary - redirect to unified endpoint."""
        try:
            from .routes.unified_routes import api_get_portfolio_summary_detailed
            return api_get_portfolio_summary_detailed()
        except Exception as e:
            app.logger.error(f"Error fetching portfolio: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/orders/', methods=['GET'])
    @login_required
    def api_get_orders():
        """Get orders history - redirect to unified endpoint."""
        try:
            from .routes.unified_routes import api_get_orders_history
            return api_get_orders_history()
        except Exception as e:
            app.logger.error(f"Error fetching orders: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/orders', methods=['GET'])
    @login_required
    def api_get_orders_no_slash():
        """Get comprehensive orders data for Orders page including stats and history."""
        try:
            app.logger.info(f"Fetching comprehensive orders data for user {current_user.id}")

            # Fetch raw orderbook data
            orderbook_result = broker_service.get_fyers_orderbook(current_user.id)

            if not orderbook_result.get('success'):
                error_msg = orderbook_result.get('error', 'Failed to fetch orderbook')
                app.logger.error(f"Orderbook fetch failed: {error_msg}")
                return jsonify([]), 200  # Return empty array if no data

            # Extract orders from response
            orders_data = orderbook_result.get('data', {})
            if isinstance(orders_data, dict):
                raw_orders = orders_data.get('orderBook', [])
            elif isinstance(orders_data, list):
                raw_orders = orders_data
            else:
                raw_orders = []

            # Fyers status code mapping
            status_mapping = {
                0: 'PLACED',       # Order placed
                1: 'PARTIAL',      # Partially filled
                2: 'COMPLETE',     # Complete/Executed
                3: 'CANCELLED',    # Cancelled
                4: 'REJECTED',     # Rejected
                5: 'CANCELLED',    # Modified (treat as cancelled for old version)
                6: 'PENDING'       # Pending
            }

            # Fyers side mapping
            side_mapping = {
                1: 'BUY',
                -1: 'SELL'
            }

            # Fyers type mapping
            type_mapping = {
                1: 'LIMIT',
                2: 'MARKET',
                3: 'STOP_LOSS',
                4: 'STOP_LOSS_MARKET'
            }

            # Transform orders to expected format
            formatted_orders = []
            for order in raw_orders:
                status_code = order.get('status', 0)
                side_code = order.get('side', 1)
                type_code = order.get('type', 2)

                formatted_order = {
                    'order_id': str(order.get('id', '')),
                    'symbol': order.get('symbol', '').replace('NSE:', '').replace('-EQ', ''),
                    'type': type_mapping.get(type_code, 'MARKET'),
                    'transaction': side_mapping.get(side_code, 'BUY'),
                    'quantity': int(order.get('qty', 0)),
                    'filled': int(order.get('filledQty', 0)),
                    'status': status_mapping.get(status_code, 'UNKNOWN'),
                    'price': float(order.get('limitPrice', order.get('tradedPrice', 0))),
                    'created_at': order.get('orderDateTime', ''),
                    'product_type': order.get('productType', ''),
                    'remaining_quantity': int(order.get('remainingQuantity', 0))
                }
                formatted_orders.append(formatted_order)

            app.logger.info(f"Formatted {len(formatted_orders)} orders for user {current_user.id}")
            return jsonify(formatted_orders)

        except Exception as e:
            app.logger.error(f"Error fetching orders: {e}", exc_info=True)
            return jsonify([]), 200  # Return empty array on error to prevent frontend issues

    @app.route('/api/portfolio/positions', methods=['GET'])
    @login_required
    def api_get_portfolio_positions_unified():
        """Get portfolio positions - redirect to unified endpoint."""
        try:
            from .routes.unified_routes import api_get_positions
            return api_get_positions()
        except Exception as e:
            app.logger.error(f"Error fetching portfolio positions: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/orders/history', methods=['GET'])
    @login_required
    def api_get_orders_history_unified():
        """Get orders history - redirect to unified endpoint."""
        try:
            from .routes.unified_routes import api_get_orders_history as unified_orders_history
            return unified_orders_history()
        except Exception as e:
            app.logger.error(f"Error fetching orders history: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)
