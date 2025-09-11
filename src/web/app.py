"""
Flask Web Application for the Automated Trading System with Swagger Documentation
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
try:
    # Try relative imports first (for normal usage)
    from ..models.database import get_database_manager
    from ..models.models import Log, Order, Trade, Position, User, Strategy, SuggestedStock, Configuration
    from ..integrations.db_charts import DatabaseCharts
    from ..integrations.multi_user_trading_engine import get_trading_engine
except ImportError:
    # Fall back to absolute imports (for testing)
    from models.database import get_database_manager
    from models.models import Log, Order, Trade, Position, User, Strategy, SuggestedStock, Configuration
    from integrations.db_charts import DatabaseCharts
    from integrations.multi_user_trading_engine import get_trading_engine
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
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)