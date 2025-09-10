"""
API Error Handling
"""
from flask import jsonify
from datetime import datetime

class APIError(Exception):
    """Base API exception."""
    def __init__(self, message, status_code=400, error_code=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code

class ValidationError(APIError):
    """Validation error."""
    def __init__(self, message, field=None):
        super().__init__(message, 400, 'VALIDATION_ERROR')
        self.field = field

class AuthenticationError(APIError):
    """Authentication error."""
    def __init__(self, message="Authentication required"):
        super().__init__(message, 401, 'AUTHENTICATION_ERROR')

class AuthorizationError(APIError):
    """Authorization error."""
    def __init__(self, message="Insufficient permissions"):
        super().__init__(message, 403, 'AUTHORIZATION_ERROR')

class NotFoundError(APIError):
    """Resource not found error."""
    def __init__(self, message="Resource not found"):
        super().__init__(message, 404, 'NOT_FOUND')

class ConflictError(APIError):
    """Conflict error."""
    def __init__(self, message="Resource conflict"):
        super().__init__(message, 409, 'CONFLICT')

class InternalServerError(APIError):
    """Internal server error."""
    def __init__(self, message="Internal server error"):
        super().__init__(message, 500, 'INTERNAL_ERROR')

def handle_api_error(error):
    """Handle API errors and return standardized response."""
    if isinstance(error, APIError):
        return jsonify({
            'success': False,
            'error': error.message,
            'error_code': error.error_code,
            'timestamp': datetime.utcnow().isoformat()
        }), error.status_code
    
    # Handle unexpected errors
    return jsonify({
        'success': False,
        'error': 'An unexpected error occurred',
        'error_code': 'INTERNAL_ERROR',
        'timestamp': datetime.utcnow().isoformat()
    }), 500
