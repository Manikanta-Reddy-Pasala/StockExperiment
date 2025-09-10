"""
Admin API Module
"""
from flask import Blueprint
from .users import users_bp
from .system import system_bp

admin_bp = Blueprint('admin', __name__, url_prefix='/api/v1/admin')

# Register sub-blueprints
admin_bp.register_blueprint(users_bp, url_prefix='/users')
admin_bp.register_blueprint(system_bp, url_prefix='/system')
