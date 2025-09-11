"""
Flask Web Application for the Automated Trading System with Swagger Documentation
"""
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
try:
    # Try relative imports first (for normal usage)
    from ..models.database import get_database_manager
    from ..models.models import Log, Order, Trade, Position, User, Strategy, SuggestedStock, Configuration, BrokerConfiguration
    from ..integrations.db_charts import DatabaseCharts
    from ..integrations.multi_user_trading_engine import get_trading_engine
    from ..services.broker_service import get_broker_service, FyersAPIConnector, FyersOAuth2Flow
except ImportError:
    # Fall back to absolute imports (for testing)
    from models.database import get_database_manager
    from models.models import Log, Order, Trade, Position, User, Strategy, SuggestedStock, Configuration, BrokerConfiguration
    from integrations.db_charts import DatabaseCharts
    from integrations.multi_user_trading_engine import get_trading_engine
    from services.broker_service import get_broker_service, FyersAPIConnector, FyersOAuth2Flow
from datetime import datetime
import secrets
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


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
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """Register new user."""
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            # Validation
            if not all([username, email, password, confirm_password]):
                flash('Please fill in all fields.', 'error')
                return render_template('login.html')
            
            if password != confirm_password:
                flash('Passwords do not match.', 'error')
                return render_template('login.html')
            
            if len(password) < 6:
                flash('Password must be at least 6 characters long.', 'error')
                return render_template('login.html')
            
            with db_manager.get_session() as db_session:
                # Check if user already exists
                existing_user = db_session.query(User).filter(
                    (User.username == username) | (User.email == email)
                ).first()
                
                if existing_user:
                    if existing_user.username == username:
                        flash('Username already exists.', 'error')
                    else:
                        flash('Email already registered.', 'error')
                    return render_template('login.html')
                
                # Create new user
                password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
                new_user = User(
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    is_active=True,
                    is_admin=False,
                    created_at=datetime.now()
                )
                
                db_session.add(new_user)
                db_session.commit()
                
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
        
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
            with db_manager.get_session() as session:
                users = session.query(User).all()
                users_data = []
                for user in users:
                    users_data.append({
                        'id': user.id,
                        'username': user.username,
                        'email': user.email,
                        'first_name': user.first_name,
                        'last_name': user.last_name,
                        'is_active': user.is_active,
                        'is_admin': user.is_admin,
                        'created_at': user.created_at.isoformat() if user.created_at else None,
                        'last_login': user.last_login.isoformat() if user.last_login else None
                    })
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
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            first_name = data.get('first_name', '')
            last_name = data.get('last_name', '')
            is_admin = data.get('is_admin', False)
            
            if not all([username, email, password]):
                return jsonify({'error': 'Username, email, and password are required'}), 400
            
            with db_manager.get_session() as session:
                # Check if user already exists
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
                    is_active=True,
                    is_admin=is_admin,
                    created_at=datetime.now()
                )
                
                session.add(new_user)
                session.commit()
                
                return jsonify({
                    'message': 'User created successfully',
                    'user': {
                        'id': new_user.id,
                        'username': new_user.username,
                        'email': new_user.email,
                        'first_name': new_user.first_name,
                        'last_name': new_user.last_name,
                        'is_active': new_user.is_active,
                        'is_admin': new_user.is_admin
                    }
                }), 201
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
            with db_manager.get_session() as session:
                user = session.query(User).get(user_id)
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
                
                return jsonify({
                    'message': 'User updated successfully',
                    'user': {
                        'id': user.id,
                        'username': user.username,
                        'email': user.email,
                        'first_name': user.first_name,
                        'last_name': user.last_name,
                        'is_active': user.is_active,
                        'is_admin': user.is_admin
                    }
                })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
    @login_required
    def api_delete_user(user_id):
        """Delete a user."""
        if not current_user.is_admin:
            return jsonify({'error': 'Access denied'}), 403
        
        try:
            with db_manager.get_session() as session:
                user = session.query(User).get(user_id)
                if not user:
                    return jsonify({'error': 'User not found'}), 404
                
                # Prevent deleting the current user
                if user.id == current_user.id:
                    return jsonify({'error': 'Cannot delete your own account'}), 400
                
                session.delete(user)
                session.commit()
                
                return jsonify({'message': 'User deleted successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Broker API Routes
    @app.route('/api/brokers/fyers', methods=['GET'])
    @login_required
    def api_get_fyers_info():
        """Get FYERS broker information."""
        try:
            app.logger.info(f"Fetching FYERS broker info for user {current_user.id}")
            broker_service = get_broker_service()
            
            # Get FYERS configuration from database
            config = broker_service.get_broker_config('fyers', current_user.id)
            
            if not config:
                app.logger.info("No FYERS configuration found for user")
                return jsonify({
                    'success': True,
                    'client_id': '',
                    'access_token': False,
                    'connected': False,
                    'last_updated': '-',
                    'stats': {
                        'total_orders': 0,
                        'successful_orders': 0,
                        'pending_orders': 0,
                        'failed_orders': 0,
                        'last_order_time': '-',
                        'api_response_time': '-'
                    }
                })
            
            # Get broker statistics
            stats = broker_service.get_broker_stats('fyers', current_user.id)
            
            # Format the config data
            config_data = {
                'client_id': config.get('client_id', ''),
                'api_secret': config.get('api_secret', ''),
                'access_token': bool(config.get('access_token')),
                'connected': config.get('is_connected', False),
                'connection_status': config.get('connection_status', 'unknown'),
                'last_updated': config.get('updated_at').strftime('%Y-%m-%d %H:%M:%S') if config.get('updated_at') else '-',
                'last_connection_test': config.get('last_connection_test').strftime('%Y-%m-%d %H:%M:%S') if config.get('last_connection_test') else '-',
                'error_message': config.get('error_message', ''),
                'redirect_url': config.get('redirect_url', ''),
                'app_type': config.get('app_type', '100'),
                'is_active': config.get('is_active', True)
            }
            
            app.logger.info(f"FYERS broker info retrieved successfully for user {current_user.id}")
            return jsonify({
                'success': True,
                **config_data,
                'stats': stats
            })
        except Exception as e:
            app.logger.error(f"Error getting FYERS broker info for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    @app.route('/api/brokers/fyers/test', methods=['POST'])
    @login_required
    def api_test_fyers_connection():
        """Test FYERS broker connection."""
        try:
            app.logger.info(f"Testing FYERS broker connection for user {current_user.id}")
            broker_service = get_broker_service()
            
            # Get FYERS configuration from database
            config = broker_service.get_broker_config('fyers', current_user.id)
            
            if not config or not config.get('client_id') or not config.get('access_token'):
                app.logger.warning(f"FYERS credentials not configured for user {current_user.id}")
                return jsonify({
                    'success': False,
                    'error': 'FYERS credentials not configured'
                }), 400
            
            # Create FYERS API connector and test connection
            app.logger.info(f"Creating FYERS API connector for user {current_user.id}")
            connector = FyersAPIConnector(config.get('client_id'), config.get('access_token'))
            result = connector.test_connection()
            
            app.logger.info(f"FYERS connection test result for user {current_user.id}: {result['success']} - {result['message']}")
            
            # Update connection status in database
            with broker_service.db_manager.get_session() as session:
                # Get the config within this session
                query = session.query(BrokerConfiguration).filter_by(broker_name='fyers')
                if current_user.id:
                    query = query.filter_by(user_id=current_user.id)
                else:
                    query = query.filter_by(user_id=None)
                
                config = query.first()
                if config:
                    config.is_connected = result['success']
                    config.connection_status = 'connected' if result['success'] else 'disconnected'
                    config.last_connection_test = datetime.utcnow()
                    config.error_message = result.get('message', '') if not result['success'] else None
                    session.commit()
                    app.logger.info(f"Updated FYERS connection status in database for user {current_user.id}")
            
            return jsonify({
                'success': result['success'],
                'message': result['message'],
                'response_time': result.get('response_time', '-'),
                'status_code': result.get('status_code', 0)
            })
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
            
            # Validate required fields - for OAuth2, we need client_id and secret_key
            if not data.get('client_id'):
                app.logger.warning(f"Missing required FYERS client_id for user {current_user.id}")
                return jsonify({
                    'success': False,
                    'error': 'Client ID is required'
                }), 400
            
            broker_service = get_broker_service()
            
            # Save configuration to database - handle both OAuth2 (secret_key) and manual (access_token) configs
            config_data = {
                'client_id': data.get('client_id'),
                'redirect_url': data.get('redirect_uri') or data.get('redirect_url', 'https://trade.fyers.in/api-login/redirect-uri/index.html'),
                'app_type': data.get('app_type', '100'),
                'is_active': True
            }
            
            # For OAuth2 flow, save secret_key as api_secret
            if data.get('secret_key'):
                config_data['api_secret'] = data.get('secret_key')
            # For manual config, save access_token
            elif data.get('access_token'):
                config_data['access_token'] = data.get('access_token')
                config_data['refresh_token'] = data.get('refresh_token', '')
            
            config = broker_service.save_broker_config('fyers', config_data, current_user.id)
            
            app.logger.info(f"FYERS configuration saved successfully for user {current_user.id}")
            
            # Format the config data
            config_data = {
                'id': config.get('id'),
                'client_id': config.get('client_id'),
                'redirect_url': config.get('redirect_url'),
                'app_type': config.get('app_type'),
                'is_active': config.get('is_active'),
                'created_at': config.get('created_at').isoformat() if config.get('created_at') else None,
                'updated_at': config.get('updated_at').isoformat() if config.get('updated_at') else None
            }
            
            return jsonify({
                'success': True,
                'message': 'FYERS configuration saved successfully',
                'config': config_data
            })
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
            broker_service = get_broker_service()
            
            # Get existing config
            config = broker_service.get_broker_config('fyers', current_user.id)
            if not config:
                app.logger.warning(f"FYERS configuration not found for user {current_user.id}")
                return jsonify({
                    'success': False,
                    'error': 'FYERS configuration not found'
                }), 404
            
            # Update configuration
            update_data = {}
            if 'client_id' in data:
                update_data['client_id'] = data['client_id']
            if 'access_token' in data:
                update_data['access_token'] = data['access_token']
            if 'refresh_token' in data:
                update_data['refresh_token'] = data['refresh_token']
            if 'redirect_url' in data:
                update_data['redirect_url'] = data['redirect_url']
            if 'app_type' in data:
                update_data['app_type'] = data['app_type']
            if 'is_active' in data:
                update_data['is_active'] = data['is_active']
            
            config = broker_service.save_broker_config('fyers', update_data, current_user.id)
            
            app.logger.info(f"FYERS configuration updated successfully for user {current_user.id}")
            
            return jsonify({
                'success': True,
                'message': 'FYERS configuration updated successfully',
                'config': {
                    'id': config['id'],
                    'client_id': config['client_id'],
                    'redirect_url': config['redirect_url'],
                    'app_type': config['app_type'],
                    'is_active': config['is_active'],
                    'updated_at': config['updated_at'].isoformat() if config['updated_at'] else None
                }
            })
        except Exception as e:
            app.logger.error(f"Error updating FYERS configuration for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/brokers/fyers/refresh-token', methods=['POST'])
    @login_required
    def api_refresh_fyers_token():
        """Refresh FYERS access token."""
        try:
            app.logger.info(f"Refreshing FYERS token for user {current_user.id}")
            # In a real implementation, you would:
            # 1. Use the refresh token to get a new access token
            # 2. Update the database configuration
            # 3. Restart the connection
            
            # For now, we'll simulate a successful refresh
            app.logger.info(f"FYERS token refresh initiated successfully for user {current_user.id}")
            return jsonify({
                'success': True,
                'message': 'Token refresh initiated successfully'
            })
        except Exception as e:
            app.logger.error(f"Error refreshing FYERS token for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/brokers/fyers/auth-url', methods=['POST'])
    @login_required
    def api_generate_fyers_auth_url():
        """Generate FYERS OAuth2 authorization URL."""
        try:
            app.logger.info(f"Generating FYERS auth URL for user {current_user.id}")
            data = request.get_json()
            
            # Validate required fields
            if not data.get('client_id') or not data.get('secret_key') or not data.get('redirect_uri'):
                app.logger.warning(f"Missing required FYERS OAuth2 credentials for user {current_user.id}")
                return jsonify({
                    'success': False,
                    'error': 'Client ID, Secret Key, and Redirect URI are required'
                }), 400
            
            # Create OAuth2 flow handler
            oauth_flow = FyersOAuth2Flow(
                client_id=data.get('client_id'),
                secret_key=data.get('secret_key'),
                redirect_uri=data.get('redirect_uri')
            )
            
            # Generate authorization URL
            auth_url = oauth_flow.generate_auth_url()
            
            app.logger.info(f"FYERS auth URL generated successfully for user {current_user.id}")
            return jsonify({
                'success': True,
                'auth_url': auth_url,
                'message': 'Authorization URL generated successfully. Please visit this URL to authorize the application.'
            })
            
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
            
            # Validate required fields
            if not data.get('client_id') or not data.get('secret_key') or not data.get('redirect_uri') or not data.get('auth_code'):
                app.logger.warning(f"Missing required FYERS OAuth2 parameters for user {current_user.id}")
                return jsonify({
                    'success': False,
                    'error': 'Client ID, Secret Key, Redirect URI, and Auth Code are required'
                }), 400
            
            # Create OAuth2 flow handler
            oauth_flow = FyersOAuth2Flow(
                client_id=data.get('client_id'),
                secret_key=data.get('secret_key'),
                redirect_uri=data.get('redirect_uri')
            )
            
            # Exchange auth code for access token
            token_response = oauth_flow.exchange_auth_code_for_token(data.get('auth_code'))
            
            # Extract access token
            if 'access_token' in token_response:
                access_token = token_response['access_token']
                
                # Save the configuration to database
                broker_service = get_broker_service()
                config = broker_service.save_broker_config('fyers', {
                    'client_id': data.get('client_id'),
                    'access_token': access_token,
                    'refresh_token': token_response.get('refresh_token', ''),
                    'redirect_url': data.get('redirect_uri'),
                    'app_type': '100',
                    'is_active': True
                }, current_user.id)
                
                app.logger.info(f"FYERS access token obtained and saved for user {current_user.id}")
                return jsonify({
                    'success': True,
                    'message': 'Access token obtained and saved successfully',
                    'access_token': access_token[:20] + '...' if len(access_token) > 20 else access_token,  # Partial token for display
                    'config': {
                        'id': config.get('id'),
                        'client_id': config.get('client_id'),
                        'redirect_url': config.get('redirect_url'),
                        'app_type': config.get('app_type'),
                        'is_active': config.get('is_active'),
                        'created_at': config.get('created_at').isoformat() if config.get('created_at') else None,
                        'updated_at': config.get('updated_at').isoformat() if config.get('updated_at') else None
                    }
                })
            else:
                error_msg = token_response.get('message', 'Failed to obtain access token')
                app.logger.error(f"Failed to get access token for user {current_user.id}: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': error_msg
                }), 400
                
        except Exception as e:
            app.logger.error(f"Error exchanging FYERS auth code for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    @app.route('/api/brokers/fyers/funds', methods=['GET'])
    @login_required
    def api_get_fyers_funds():
        """Get FYERS user funds."""
        try:
            app.logger.info(f"Fetching FYERS funds for user {current_user.id}")
            broker_service = get_broker_service()
            
            # Get FYERS configuration from database
            config = broker_service.get_broker_config('fyers', current_user.id)
            
            if not config or not config.get('client_id') or not config.get('access_token'):
                return jsonify({
                    'success': False,
                    'error': 'FYERS credentials not configured'
                }), 400
            
            # Create FYERS API connector and get funds
            connector = FyersAPIConnector(config.get('client_id'), config.get('access_token'))
            result = connector.get_funds()
            
            if 'error' in result:
                return jsonify({
                    'success': False,
                    'error': result['error']
                }), 400
            
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            app.logger.error(f"Error getting FYERS funds for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    @app.route('/api/brokers/fyers/holdings', methods=['GET'])
    @login_required
    def api_get_fyers_holdings():
        """Get FYERS user holdings."""
        try:
            app.logger.info(f"Fetching FYERS holdings for user {current_user.id}")
            broker_service = get_broker_service()
            
            # Get FYERS configuration from database
            config = broker_service.get_broker_config('fyers', current_user.id)
            
            if not config or not config.get('client_id') or not config.get('access_token'):
                return jsonify({
                    'success': False,
                    'error': 'FYERS credentials not configured'
                }), 400
            
            # Create FYERS API connector and get holdings
            connector = FyersAPIConnector(config.get('client_id'), config.get('access_token'))
            result = connector.get_holdings()
            
            if 'error' in result:
                return jsonify({
                    'success': False,
                    'error': result['error']
                }), 400
            
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            app.logger.error(f"Error getting FYERS holdings for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    @app.route('/api/brokers/fyers/positions', methods=['GET'])
    @login_required
    def api_get_fyers_positions():
        """Get FYERS user positions."""
        try:
            app.logger.info(f"Fetching FYERS positions for user {current_user.id}")
            broker_service = get_broker_service()
            
            # Get FYERS configuration from database
            config = broker_service.get_broker_config('fyers', current_user.id)
            
            if not config or not config.get('client_id') or not config.get('access_token'):
                return jsonify({
                    'success': False,
                    'error': 'FYERS credentials not configured'
                }), 400
            
            # Create FYERS API connector and get positions
            connector = FyersAPIConnector(config.get('client_id'), config.get('access_token'))
            result = connector.get_positions()
            
            if 'error' in result:
                return jsonify({
                    'success': False,
                    'error': result['error']
                }), 400
            
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            app.logger.error(f"Error getting FYERS positions for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    @app.route('/api/brokers/fyers/orderbook', methods=['GET'])
    @login_required
    def api_get_fyers_orderbook():
        """Get FYERS user orderbook."""
        try:
            app.logger.info(f"Fetching FYERS orderbook for user {current_user.id}")
            broker_service = get_broker_service()
            
            # Get FYERS configuration from database
            config = broker_service.get_broker_config('fyers', current_user.id)
            
            if not config or not config.get('client_id') or not config.get('access_token'):
                return jsonify({
                    'success': False,
                    'error': 'FYERS credentials not configured'
                }), 400
            
            # Create FYERS API connector and get orderbook
            connector = FyersAPIConnector(config.get('client_id'), config.get('access_token'))
            result = connector.get_orderbook()
            
            if 'error' in result:
                return jsonify({
                    'success': False,
                    'error': result['error']
                }), 400
            
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            app.logger.error(f"Error getting FYERS orderbook for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    @app.route('/api/brokers/fyers/tradebook', methods=['GET'])
    @login_required
    def api_get_fyers_tradebook():
        """Get FYERS user tradebook."""
        try:
            app.logger.info(f"Fetching FYERS tradebook for user {current_user.id}")
            broker_service = get_broker_service()
            
            # Get FYERS configuration from database
            config = broker_service.get_broker_config('fyers', current_user.id)
            
            if not config or not config.get('client_id') or not config.get('access_token'):
                return jsonify({
                    'success': False,
                    'error': 'FYERS credentials not configured'
                }), 400
            
            # Create FYERS API connector and get tradebook
            connector = FyersAPIConnector(config.get('client_id'), config.get('access_token'))
            result = connector.get_tradebook()
            
            if 'error' in result:
                return jsonify({
                    'success': False,
                    'error': result['error']
                }), 400
            
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            app.logger.error(f"Error getting FYERS tradebook for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    @app.route('/api/brokers/fyers/quotes', methods=['GET'])
    @login_required
    def api_get_fyers_quotes():
        """Get FYERS market quotes."""
        try:
            symbols = request.args.get('symbols', 'NSE:SBIN-EQ')
            app.logger.info(f"Fetching FYERS quotes for symbols: {symbols} for user {current_user.id}")
            broker_service = get_broker_service()
            
            # Get FYERS configuration from database
            config = broker_service.get_broker_config('fyers', current_user.id)
            
            if not config or not config.get('client_id') or not config.get('access_token'):
                return jsonify({
                    'success': False,
                    'error': 'FYERS credentials not configured'
                }), 400
            
            # Create FYERS API connector and get quotes
            connector = FyersAPIConnector(config.get('client_id'), config.get('access_token'))
            result = connector.get_quotes(symbols)
            
            if 'error' in result:
                return jsonify({
                    'success': False,
                    'error': result['error']
                }), 400
            
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            app.logger.error(f"Error getting FYERS quotes for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    @app.route('/api/brokers/fyers/history', methods=['GET'])
    @login_required
    def api_get_fyers_history():
        """Get FYERS historical data."""
        try:
            symbol = request.args.get('symbol', 'NSE:SBIN-EQ')
            resolution = request.args.get('resolution', 'D')
            range_from = request.args.get('range_from')
            range_to = request.args.get('range_to')
            
            app.logger.info(f"Fetching FYERS historical data for symbol: {symbol} for user {current_user.id}")
            broker_service = get_broker_service()
            
            # Get FYERS configuration from database
            config = broker_service.get_broker_config('fyers', current_user.id)
            
            if not config or not config.get('client_id') or not config.get('access_token'):
                return jsonify({
                    'success': False,
                    'error': 'FYERS credentials not configured'
                }), 400
            
            # Create FYERS API connector and get historical data
            connector = FyersAPIConnector(config.get('client_id'), config.get('access_token'))
            result = connector.get_history(symbol, resolution, range_from, range_to)
            
            if 'error' in result:
                return jsonify({
                    'success': False,
                    'error': result['error']
                }), 400
            
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            app.logger.error(f"Error getting FYERS historical data for user {current_user.id}: {str(e)}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)