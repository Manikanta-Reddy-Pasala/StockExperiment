"""
ML routes for stock prediction
"""
from flask import Blueprint

# Import the ML blueprint
from . import ml_bp

# This module exports the ML blueprint for registration in the main app
__all__ = ['ml_bp']
