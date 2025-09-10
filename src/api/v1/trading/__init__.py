"""
Trading API Module
"""
from flask import Blueprint
from .screening import screening_bp
from .strategies import strategies_bp
from .execution import execution_bp

trading_bp = Blueprint('trading', __name__, url_prefix='/api/v1/trading')

# Register sub-blueprints
trading_bp.register_blueprint(screening_bp, url_prefix='/screening')
trading_bp.register_blueprint(strategies_bp, url_prefix='/strategies')
trading_bp.register_blueprint(execution_bp, url_prefix='/execution')
