"""
Admin API endpoints
"""
from flask_restx import Namespace, Resource
from flask import request
from flask_login import login_required, current_user
from datetime import datetime

# Create namespace for admin API
ns_admin = Namespace('admin', description='Admin operations')

@ns_admin.route('/settings')
class Settings(Resource):
    @login_required
    def get(self):
        """Get settings."""
        try:
            from datastore.database import get_database_manager
            from datastore.models import Configuration
            
            db_manager = get_database_manager()
            
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
            
            return settings_data
        except Exception as e:
            return {'error': str(e)}, 500

@ns_admin.route('/users')
class AdminUsers(Resource):
    @login_required
    def get(self):
        """Get all users (admin only)."""
        if not current_user.is_admin:
            return {'error': 'Access denied'}, 403
        
        try:
            from datastore.database import get_database_manager
            from datastore.models import User
            
            db_manager = get_database_manager()
            
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
            
            return users_data
        except Exception as e:
            return {'error': str(e)}, 500
    
    @login_required
    def post(self):
        """Create a new user (admin only)."""
        if not current_user.is_admin:
            return {'error': 'Access denied'}, 403
        
        try:
            from datastore.database import get_database_manager
            from datastore.models import User
            from flask_bcrypt import Bcrypt
            
            from flask import current_app
            bcrypt = Bcrypt(current_app)
            
            data = request.get_json()
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            first_name = data.get('first_name', '')
            last_name = data.get('last_name', '')
            is_admin = data.get('is_admin', False)
            
            db_manager = get_database_manager()
            
            if not all([username, email, password]):
                return {'error': 'Username, email, and password are required'}, 400
            
            with db_manager.get_session() as session:
                # Check if username or email already exists
                existing_user = session.query(User).filter(
                    (User.username == username) | (User.email == email)
                ).first()
                
                if existing_user:
                    return {'error': 'Username or email already exists'}, 400
                
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
                
                return {
                    'message': 'User created successfully',
                    'user_id': new_user.id
                }, 201
                
        except Exception as e:
            return {'error': str(e)}, 500

@ns_admin.route('/users/<int:user_id>')
class AdminUser(Resource):
    @login_required
    def put(self, user_id):
        """Update a user (admin only)."""
        if not current_user.is_admin:
            return {'error': 'Access denied'}, 403
        
        try:
            from datastore.database import get_database_manager
            from datastore.models import User
            from flask_bcrypt import Bcrypt
            
            from flask import current_app
            bcrypt = Bcrypt(current_app)
            
            data = request.get_json()
            
            db_manager = get_database_manager()
            
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return {'error': 'User not found'}, 404
                
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
                
                return {'message': 'User updated successfully'}
                
        except Exception as e:
            return {'error': str(e)}, 500
    
    @login_required
    def delete(self, user_id):
        """Delete a user (admin only)."""
        if not current_user.is_admin:
            return {'error': 'Access denied'}, 403
        
        try:
            from datastore.database import get_database_manager
            from datastore.models import User
            
            db_manager = get_database_manager()
            
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return {'error': 'User not found'}, 404
                
                # Don't allow deleting the current admin user
                if user.id == current_user.id:
                    return {'error': 'Cannot delete your own account'}, 400
                
                session.delete(user)
                session.commit()
                
                return {'message': 'User deleted successfully'}
                
        except Exception as e:
            return {'error': str(e)}, 500

@ns_admin.route('/users/<int:user_id>/reset-password')
class AdminResetPassword(Resource):
    @login_required
    def post(self, user_id):
        """Reset a user's password (admin only)."""
        if not current_user.is_admin:
            return {'error': 'Access denied'}, 403
        
        try:
            from datastore.database import get_database_manager
            from datastore.models import User
            from flask_bcrypt import Bcrypt
            
            from flask import current_app
            bcrypt = Bcrypt(current_app)
            
            data = request.get_json()
            new_password = data.get('password')
            
            db_manager = get_database_manager()
            
            if not new_password:
                return {'error': 'New password is required'}, 400
            
            if len(new_password) < 6:
                return {'error': 'Password must be at least 6 characters long'}, 400
            
            with db_manager.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return {'error': 'User not found'}, 404
                
                # Update password
                user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
                session.commit()
                
                return {'message': 'Password reset successfully'}
                
        except Exception as e:
            return {'error': str(e)}, 500