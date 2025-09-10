"""
User Management API
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from datastore.database import get_database_manager
from datastore.models import User
from api.common.decorators import api_response, admin_required, validate_json
from api.common.errors import ValidationError, NotFoundError, ConflictError
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

users_bp = Blueprint('users', __name__)

@users_bp.route('', methods=['GET'])
@login_required
@admin_required
@api_response
def get_users():
    """Get all users."""
    db_manager = get_database_manager()
    
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
                'is_admin': user.is_admin,
                'is_active': user.is_active,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'last_login': user.last_login.isoformat() if user.last_login else None
            })
        
        return users_data

@users_bp.route('', methods=['POST'])
@login_required
@admin_required
@validate_json('username', 'email', 'password')
@api_response
def create_user():
    """Create a new user."""
    db_manager = get_database_manager()
    data = request.get_json()
    
    # Validate required fields
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    first_name = data.get('first_name', '')
    last_name = data.get('last_name', '')
    is_admin = data.get('is_admin', False)
    
    with db_manager.get_session() as session:
        # Check if username already exists
        existing_user = session.query(User).filter(User.username == username).first()
        if existing_user:
            raise ConflictError(f"Username '{username}' already exists")
        
        # Check if email already exists
        existing_email = session.query(User).filter(User.email == email).first()
        if existing_email:
            raise ConflictError(f"Email '{email}' already exists")
        
        # Create new user
        from flask_bcrypt import Bcrypt
        bcrypt = Bcrypt()
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        new_user = User(
            username=username,
            email=email,
            password_hash=hashed_password,
            first_name=first_name,
            last_name=last_name,
            is_admin=is_admin,
            is_active=True
        )
        
        session.add(new_user)
        session.commit()
        
        return {
            'id': new_user.id,
            'username': new_user.username,
            'email': new_user.email,
            'first_name': new_user.first_name,
            'last_name': new_user.last_name,
            'is_admin': new_user.is_admin,
            'is_active': new_user.is_active,
            'created_at': new_user.created_at.isoformat()
        }, 201

@users_bp.route('/<int:user_id>', methods=['PUT'])
@login_required
@admin_required
@api_response
def update_user(user_id):
    """Update a user."""
    db_manager = get_database_manager()
    data = request.get_json()
    
    with db_manager.get_session() as session:
        user = session.query(User).get(user_id)
        if not user:
            raise NotFoundError(f"User with ID {user_id} not found")
        
        # Update fields
        if 'username' in data:
            # Check if username already exists (excluding current user)
            existing_user = session.query(User).filter(
                User.username == data['username'],
                User.id != user_id
            ).first()
            if existing_user:
                raise ConflictError(f"Username '{data['username']}' already exists")
            user.username = data['username']
        
        if 'email' in data:
            # Check if email already exists (excluding current user)
            existing_email = session.query(User).filter(
                User.email == data['email'],
                User.id != user_id
            ).first()
            if existing_email:
                raise ConflictError(f"Email '{data['email']}' already exists")
            user.email = data['email']
        
        if 'password' in data and data['password']:
            from flask_bcrypt import Bcrypt
            bcrypt = Bcrypt()
            user.password_hash = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        
        if 'first_name' in data:
            user.first_name = data['first_name']
        
        if 'last_name' in data:
            user.last_name = data['last_name']
        
        if 'is_active' in data:
            user.is_active = data['is_active']
        
        if 'is_admin' in data:
            user.is_admin = data['is_admin']
        
        session.commit()
        
        return {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'is_admin': user.is_admin,
            'is_active': user.is_active,
            'updated_at': datetime.utcnow().isoformat()
        }

@users_bp.route('/<int:user_id>', methods=['DELETE'])
@login_required
@admin_required
@api_response
def delete_user(user_id):
    """Delete a user."""
    db_manager = get_database_manager()
    
    with db_manager.get_session() as session:
        user = session.query(User).get(user_id)
        if not user:
            raise NotFoundError(f"User with ID {user_id} not found")
        
        # Prevent deleting current user
        if user.id == current_user.id:
            raise ValidationError("Cannot delete your own account")
        
        session.delete(user)
        session.commit()
        
        return {'message': f'User {user.username} deleted successfully'}

@users_bp.route('/<int:user_id>/reset-password', methods=['POST'])
@login_required
@admin_required
@validate_json('new_password')
@api_response
def reset_user_password(user_id):
    """Reset a user's password."""
    db_manager = get_database_manager()
    data = request.get_json()
    
    new_password = data.get('new_password')
    
    if len(new_password) < 6:
        raise ValidationError("Password must be at least 6 characters long")
    
    with db_manager.get_session() as session:
        user = session.query(User).get(user_id)
        if not user:
            raise NotFoundError(f"User with ID {user_id} not found")
        
        from flask_bcrypt import Bcrypt
        bcrypt = Bcrypt()
        user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
        
        session.commit()
        
        return {'message': f'Password reset successfully for user {user.username}'}
