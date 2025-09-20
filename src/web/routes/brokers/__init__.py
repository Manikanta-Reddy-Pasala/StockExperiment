"""
Broker Routes Package - Individual route modules for each broker
"""

from .fyers_routes import fyers_bp
from .zerodha_routes import zerodha_bp

__all__ = [
    'fyers_bp',
    'zerodha_bp'
]
