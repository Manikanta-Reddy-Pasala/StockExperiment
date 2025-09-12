"""
FYERS Broker API Routes - Dedicated routes for FYERS broker operations
"""
from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from flask_login import login_required, current_user
import logging

# Configure logging
logger = logging.getLogger(__name__)

try:
    from ...services.brokers.fyers_service import get_fyers_service
except ImportError:
    from services.brokers.fyers_service import get_fyers_service

# Create Blueprint for FYERS routes
fyers_bp = Blueprint('fyers', __name__, url_prefix='/brokers/fyers')

@fyers_bp.route('/')
@login_required
def fyers_page():
    """FYERS broker configuration page."""
    return render_template('brokers/fyers.html')

@fyers_bp.route('/api/info', methods=['GET'])
@login_required
def api_get_fyers_info():
    """Get FYERS broker information."""
    try:
        fyers_service = get_fyers_service()
        config = fyers_service.get_broker_config(current_user.id)
        
        if not config:
            return jsonify({
                'success': True, 
                'broker': 'fyers', 
                'client_id': '', 
                'access_token': False, 
                'connected': False, 
                'last_updated': '-',
                'stats': {'total_orders': 0, 'successful_orders': 0, 'pending_orders': 0, 
                        'failed_orders': 0, 'last_order_time': '-', 'api_response_time': '-'}
            })

        stats = fyers_service.get_broker_stats(current_user.id)
        config['access_token'] = bool(config.get('access_token'))
        
        return jsonify({'success': True, 'broker': 'fyers', **config, 'stats': stats})
        
    except Exception as e:
        logger.error(f"Error getting FYERS broker info for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@fyers_bp.route('/api/test', methods=['POST'])
@login_required
def api_test_fyers_connection():
    """Test FYERS broker connection."""
    try:
        fyers_service = get_fyers_service()
        result = fyers_service.test_connection(current_user.id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error testing FYERS connection for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@fyers_bp.route('/api/config', methods=['POST'])
@login_required
def api_save_fyers_config():
    """Save FYERS broker configuration."""
    try:
        data = request.get_json()
        fyers_service = get_fyers_service()
        
        config_data = {
            'client_id': data.get('client_id'),
            'api_secret': data.get('secret_key'),
            'redirect_url': data.get('redirect_uri')
        }
        
        saved_config = fyers_service.save_broker_config(config_data, current_user.id)
        
        # Generate auth URL if credentials are provided
        auth_url = None
        if config_data['client_id'] and config_data['api_secret']:
            try:
                auth_url = fyers_service.generate_auth_url(current_user.id)
            except Exception as e:
                logger.warning(f"Could not generate auth URL: {str(e)}")
        
        return jsonify({
            'success': True, 
            'message': 'Configuration saved successfully',
            'auth_url': auth_url
        })
        
    except Exception as e:
        logger.error(f"Error saving FYERS config for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@fyers_bp.route('/api/refresh-token', methods=['POST'])
@login_required
def api_refresh_fyers_token():
    """Refresh FYERS access token."""
    try:
        fyers_service = get_fyers_service()
        auth_url = fyers_service.generate_auth_url(current_user.id)
        
        return jsonify({
            'success': True, 
            'message': 'Auth URL generated',
            'auth_url': auth_url
        })
        
    except Exception as e:
        logger.error(f"Error refreshing FYERS token for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@fyers_bp.route('/oauth/callback', methods=['GET'])
def api_fyers_oauth_callback():
    """FYERS OAuth2 callback handler."""
    try:
        auth_code = request.args.get('auth_code')
        state = request.args.get('state')
        
        if not auth_code:
            return jsonify({'success': False, 'error': 'Authorization code not provided'}), 400
        
        # Extract user_id from state
        user_id = int(state) if state and state.isdigit() else 1
        
        fyers_service = get_fyers_service()
        result = fyers_service.exchange_auth_code(user_id, auth_code)
        
        if result.get('success'):
            return jsonify({'success': True, 'message': 'Authorization successful'})
        else:
            return jsonify({'success': False, 'error': 'Authorization failed'}), 400
            
    except Exception as e:
        logger.error(f"Error in FYERS OAuth callback: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': 'Authorization failed'}), 500

@fyers_bp.route('/api/funds', methods=['GET'])
@login_required
def api_get_fyers_funds():
    """Get FYERS user funds."""
    try:
        fyers_service = get_fyers_service()
        funds = fyers_service.get_funds(current_user.id)
        return jsonify({'success': True, 'data': funds})
        
    except Exception as e:
        logger.error(f"Error getting FYERS funds for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@fyers_bp.route('/api/holdings', methods=['GET'])
@login_required
def api_get_fyers_holdings():
    """Get FYERS user holdings."""
    try:
        fyers_service = get_fyers_service()
        holdings = fyers_service.get_holdings(current_user.id)
        return jsonify({'success': True, 'data': holdings})
        
    except Exception as e:
        logger.error(f"Error getting FYERS holdings for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@fyers_bp.route('/api/positions', methods=['GET'])
@login_required
def api_get_fyers_positions():
    """Get FYERS user positions."""
    try:
        fyers_service = get_fyers_service()
        positions = fyers_service.get_positions(current_user.id)
        return jsonify({'success': True, 'data': positions})
        
    except Exception as e:
        logger.error(f"Error getting FYERS positions for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@fyers_bp.route('/api/orders', methods=['GET'])
@login_required
def api_get_fyers_orders():
    """Get FYERS user orders."""
    try:
        fyers_service = get_fyers_service()
        orders = fyers_service.get_orderbook(current_user.id)
        return jsonify({'success': True, 'data': orders})
        
    except Exception as e:
        logger.error(f"Error getting FYERS orders for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@fyers_bp.route('/api/trades', methods=['GET'])
@login_required
def api_get_fyers_trades():
    """Get FYERS user trades."""
    try:
        fyers_service = get_fyers_service()
        trades = fyers_service.get_tradebook(current_user.id)
        return jsonify({'success': True, 'data': trades})
        
    except Exception as e:
        logger.error(f"Error getting FYERS trades for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@fyers_bp.route('/api/quotes', methods=['GET'])
@login_required
def api_get_fyers_quotes():
    """Get FYERS market quotes."""
    try:
        symbols = request.args.get('symbols', '')
        if not symbols:
            return jsonify({'success': False, 'error': 'Symbols parameter required'}), 400
            
        fyers_service = get_fyers_service()
        quotes = fyers_service.get_quotes(current_user.id, symbols)
        return jsonify({'success': True, 'data': quotes})
        
    except Exception as e:
        logger.error(f"Error getting FYERS quotes for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@fyers_bp.route('/api/profile', methods=['GET'])
@login_required
def api_get_fyers_profile():
    """Get FYERS user profile."""
    try:
        fyers_service = get_fyers_service()
        profile = fyers_service.get_profile(current_user.id)
        return jsonify({'success': True, 'data': profile})
        
    except Exception as e:
        logger.error(f"Error getting FYERS profile for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
