"""
API Decorators and Utilities
"""
from functools import wraps
from flask import jsonify, request
from flask_login import current_user
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def api_response(func):
    """Decorator to standardize API responses."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, tuple):
                data, status_code = result
                return jsonify({
                    'success': True,
                    'data': data,
                    'timestamp': datetime.utcnow().isoformat()
                }), status_code
            else:
                return jsonify({
                    'success': True,
                    'data': result,
                    'timestamp': datetime.utcnow().isoformat()
                }), 200
        except Exception as e:
            logger.error(f"API Error in {func.__name__}: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 500
    return wrapper

def validate_json(*required_fields):
    """Decorator to validate required JSON fields."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Request must be JSON'
                }), 400
            
            data = request.get_json()
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return jsonify({
                    'success': False,
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }), 400
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def admin_required(func):
    """Decorator to require admin privileges."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({
                'success': False,
                'error': 'Authentication required'
            }), 401
        
        if not current_user.is_admin:
            return jsonify({
                'success': False,
                'error': 'Admin privileges required'
            }), 403
        
        return func(*args, **kwargs)
    return wrapper

def rate_limit(max_requests=100, window=3600):
    """Simple rate limiting decorator."""
    # This is a basic implementation - in production, use Redis or similar
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # TODO: Implement proper rate limiting with Redis
            return func(*args, **kwargs)
        return wrapper
    return decorator
