"""
Zerodha Broker API Routes - Dedicated routes for Zerodha broker operations
"""
from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from flask_login import login_required, current_user
import logging

# Configure logging
logger = logging.getLogger(__name__)

try:
    from ...services.brokers.zerodha_service import get_zerodha_service
except ImportError:
    try:
        from services.brokers.zerodha_service import get_zerodha_service
    except ImportError:
        from src.services.brokers.zerodha_service import get_zerodha_service

# Create Blueprint for Zerodha routes
zerodha_bp = Blueprint('zerodha', __name__, url_prefix='/brokers/zerodha')

@zerodha_bp.route('/')
@login_required
def zerodha_page():
    """Zerodha broker configuration page."""
    return render_template('brokers/zerodha.html')

@zerodha_bp.route('/api/info', methods=['GET'])
@login_required
def api_get_zerodha_info():
    """Get Zerodha broker information."""
    try:
        zerodha_service = get_zerodha_service()
        config = zerodha_service.get_broker_config(current_user.id)
        
        if not config:
            return jsonify({
                'success': True, 
                'broker': 'zerodha', 
                'api_key': '', 
                'access_token': False, 
                'connected': False, 
                'last_updated': '-',
                'stats': {'total_orders': 0, 'successful_orders': 0, 'pending_orders': 0, 
                        'failed_orders': 0, 'last_order_time': '-', 'api_response_time': '-'}
            })

        stats = zerodha_service.get_broker_stats(current_user.id)
        config['access_token'] = bool(config.get('access_token'))
        
        return jsonify({'success': True, 'broker': 'zerodha', **config, 'stats': stats})
        
    except Exception as e:
        logger.error(f"Error getting Zerodha broker info for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@zerodha_bp.route('/api/test', methods=['POST'])
@login_required
def api_test_zerodha_connection():
    """Test Zerodha broker connection."""
    try:
        # For now, return a placeholder response since Zerodha integration is not ready
        return jsonify({
            'success': False, 
            'message': 'Zerodha integration is coming soon. Please use FYERS for now.',
            'status_code': 501
        })
        
    except Exception as e:
        logger.error(f"Error testing Zerodha connection for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@zerodha_bp.route('/api/config', methods=['POST'])
@login_required
def api_save_zerodha_config():
    """Save Zerodha broker configuration."""
    try:
        # For now, return a placeholder response since Zerodha integration is not ready
        return jsonify({
            'success': False, 
            'error': 'Zerodha integration is coming soon. Please use FYERS for now.'
        })
        
    except Exception as e:
        logger.error(f"Error saving Zerodha config for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@zerodha_bp.route('/api/login', methods=['POST'])
@login_required
def api_zerodha_login():
    """Generate Zerodha login URL."""
    try:
        # For now, return a placeholder response since Zerodha integration is not ready
        return jsonify({
            'success': False, 
            'error': 'Zerodha integration is coming soon. Please use FYERS for now.'
        })
        
    except Exception as e:
        logger.error(f"Error generating Zerodha login URL for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@zerodha_bp.route('/api/funds', methods=['GET'])
@login_required
def api_get_zerodha_funds():
    """Get Zerodha user funds."""
    try:
        # For now, return a placeholder response since Zerodha integration is not ready
        return jsonify({
            'success': False, 
            'error': 'Zerodha integration is coming soon. Please use FYERS for now.'
        })
        
    except Exception as e:
        logger.error(f"Error getting Zerodha funds for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@zerodha_bp.route('/api/holdings', methods=['GET'])
@login_required
def api_get_zerodha_holdings():
    """Get Zerodha user holdings."""
    try:
        # For now, return a placeholder response since Zerodha integration is not ready
        return jsonify({
            'success': False, 
            'error': 'Zerodha integration is coming soon. Please use FYERS for now.'
        })
        
    except Exception as e:
        logger.error(f"Error getting Zerodha holdings for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@zerodha_bp.route('/api/positions', methods=['GET'])
@login_required
def api_get_zerodha_positions():
    """Get Zerodha user positions."""
    try:
        # For now, return a placeholder response since Zerodha integration is not ready
        return jsonify({
            'success': False, 
            'error': 'Zerodha integration is coming soon. Please use FYERS for now.'
        })
        
    except Exception as e:
        logger.error(f"Error getting Zerodha positions for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@zerodha_bp.route('/api/orders', methods=['GET'])
@login_required
def api_get_zerodha_orders():
    """Get Zerodha user orders."""
    try:
        # For now, return a placeholder response since Zerodha integration is not ready
        return jsonify({
            'success': False, 
            'error': 'Zerodha integration is coming soon. Please use FYERS for now.'
        })
        
    except Exception as e:
        logger.error(f"Error getting Zerodha orders for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@zerodha_bp.route('/api/trades', methods=['GET'])
@login_required
def api_get_zerodha_trades():
    """Get Zerodha user trades."""
    try:
        # For now, return a placeholder response since Zerodha integration is not ready
        return jsonify({
            'success': False, 
            'error': 'Zerodha integration is coming soon. Please use FYERS for now.'
        })
        
    except Exception as e:
        logger.error(f"Error getting Zerodha trades for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@zerodha_bp.route('/api/quotes', methods=['GET'])
@login_required
def api_get_zerodha_quotes():
    """Get Zerodha market quotes."""
    try:
        # For now, return a placeholder response since Zerodha integration is not ready
        return jsonify({
            'success': False, 
            'error': 'Zerodha integration is coming soon. Please use FYERS for now.'
        })
        
    except Exception as e:
        logger.error(f"Error getting Zerodha quotes for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@zerodha_bp.route('/api/profile', methods=['GET'])
@login_required
def api_get_zerodha_profile():
    """Get Zerodha user profile."""
    try:
        # For now, return a placeholder response since Zerodha integration is not ready
        return jsonify({
            'success': False, 
            'error': 'Zerodha integration is coming soon. Please use FYERS for now.'
        })
        
    except Exception as e:
        logger.error(f"Error getting Zerodha profile for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
