"""
ML routes for stock prediction
"""
from flask import Blueprint

# Import the ML blueprints
from . import ml_bp, ml_web_bp

# This module exports the ML blueprints for registration in the main app
__all__ = ['ml_bp', 'ml_web_bp']
