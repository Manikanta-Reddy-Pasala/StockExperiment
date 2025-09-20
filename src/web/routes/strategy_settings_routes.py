"""
API Routes for Strategy Settings

Provides endpoints for managing user strategy settings and preferences.
"""

import logging
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime

from ...services.utils.user_strategy_settings_service import get_user_strategy_settings_service
from ...utils.api_logger import APILogger, log_flask_route

logger = logging.getLogger(__name__)

# Create blueprint
strategy_settings_bp = Blueprint('strategy_settings', __name__, url_prefix='/api/strategy-settings')


@strategy_settings_bp.route('/', methods=['GET'])
@login_required
@log_flask_route("get_strategy_settings")
def api_get_strategy_settings():
    """Get all strategy settings for the current user."""
    try:
        service = get_user_strategy_settings_service()
        result = service.get_user_strategy_settings(current_user.id)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error getting strategy settings: {e}")
        APILogger.log_error(
            service_name="FlaskAPI",
            method_name="get_strategy_settings",
            error=e,
            user_id=current_user.id if current_user else None
        )
        return jsonify({
            'success': False,
            'error': 'Failed to get strategy settings'
        }), 500


@strategy_settings_bp.route('/active', methods=['GET'])
@login_required
@log_flask_route("get_active_strategies")
def api_get_active_strategies():
    """Get list of active strategies for the current user."""
    try:
        service = get_user_strategy_settings_service()
        active_strategies = service.get_active_strategies(current_user.id)
        
        return jsonify({
            'success': True,
            'data': {
                'active_strategies': active_strategies,
                'count': len(active_strategies)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting active strategies: {e}")
        APILogger.log_error(
            service_name="FlaskAPI",
            method_name="get_active_strategies",
            error=e,
            user_id=current_user.id if current_user else None
        )
        return jsonify({
            'success': False,
            'error': 'Failed to get active strategies'
        }), 500


@strategy_settings_bp.route('/available', methods=['GET'])
@login_required
@log_flask_route("get_available_strategies")
def api_get_available_strategies():
    """Get all available strategies with metadata."""
    try:
        service = get_user_strategy_settings_service()
        result = service.get_available_strategies()
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error getting available strategies: {e}")
        APILogger.log_error(
            service_name="FlaskAPI",
            method_name="get_available_strategies",
            error=e,
            user_id=current_user.id if current_user else None
        )
        return jsonify({
            'success': False,
            'error': 'Failed to get available strategies'
        }), 500


@strategy_settings_bp.route('/<strategy_name>', methods=['PUT'])
@login_required
@log_flask_route("update_strategy_setting")
def api_update_strategy_setting(strategy_name):
    """Update a specific strategy setting."""
    try:
        data = request.get_json()
        
        service = get_user_strategy_settings_service()
        result = service.update_strategy_setting(
            user_id=current_user.id,
            strategy_name=strategy_name,
            is_active=data.get('is_active'),
            is_enabled=data.get('is_enabled'),
            priority=data.get('priority'),
            custom_parameters=data.get('custom_parameters')
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error updating strategy setting: {e}")
        APILogger.log_error(
            service_name="FlaskAPI",
            method_name="update_strategy_setting",
            error=e,
            user_id=current_user.id if current_user else None
        )
        return jsonify({
            'success': False,
            'error': 'Failed to update strategy setting'
        }), 500


@strategy_settings_bp.route('/bulk-update', methods=['POST'])
@login_required
@log_flask_route("bulk_update_strategy_settings")
def api_bulk_update_strategy_settings():
    """Update multiple strategy settings at once."""
    try:
        data = request.get_json()
        
        if not data or 'settings' not in data:
            return jsonify({
                'success': False,
                'error': 'Settings data is required'
            }), 400
        
        service = get_user_strategy_settings_service()
        result = service.bulk_update_strategy_settings(
            user_id=current_user.id,
            settings=data['settings']
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error bulk updating strategy settings: {e}")
        APILogger.log_error(
            service_name="FlaskAPI",
            method_name="bulk_update_strategy_settings",
            error=e,
            user_id=current_user.id if current_user else None
        )
        return jsonify({
            'success': False,
            'error': 'Failed to update strategy settings'
        }), 500


@strategy_settings_bp.route('/toggle/<strategy_name>', methods=['POST'])
@login_required
@log_flask_route("toggle_strategy")
def api_toggle_strategy(strategy_name):
    """Toggle a strategy's active status."""
    try:
        data = request.get_json()
        is_active = data.get('is_active', True)
        
        service = get_user_strategy_settings_service()
        result = service.update_strategy_setting(
            user_id=current_user.id,
            strategy_name=strategy_name,
            is_active=is_active
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error toggling strategy: {e}")
        APILogger.log_error(
            service_name="FlaskAPI",
            method_name="toggle_strategy",
            error=e,
            user_id=current_user.id if current_user else None
        )
        return jsonify({
            'success': False,
            'error': 'Failed to toggle strategy'
        }), 500
