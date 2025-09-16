"""
ML routes module - Clean __init__.py following Python best practices
This file only handles imports and exports, no route definitions
"""

# Import the blueprints from the routes module
from .ml_routes import ml_bp, ml_web_bp

# Export the blueprints for registration in the main app
__all__ = ['ml_bp', 'ml_web_bp']