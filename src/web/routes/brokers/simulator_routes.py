"""
Simulator Broker API Routes - Dedicated routes for Simulator broker operations
"""
from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from flask_login import login_required, current_user
import logging

# Configure logging
logger = logging.getLogger(__name__)

try:
    from ...services.brokers.simulator_service import get_simulator_service
except ImportError:
    try:
        from services.brokers.simulator_service import get_simulator_service
    except ImportError:
        from src.services.brokers.simulator_service import get_simulator_service

# Create Blueprint for Simulator routes
simulator_bp = Blueprint('simulator', __name__, url_prefix='/brokers/simulator')

@simulator_bp.route('/')
@login_required
def simulator_page():
    """Simulator broker configuration page."""
    return render_template('brokers/simulator.html')

@simulator_bp.route('/api/info', methods=['GET'])
@login_required
def api_get_simulator_info():
    """Get Simulator broker information."""
    try:
        simulator_service = get_simulator_service()
        config = simulator_service.get_broker_config(current_user.id)
        
        if not config:
            # Return default simulator config
            config = {
                'broker': 'simulator',
                'client_id': 'simulator',
                'access_token': True,
                'connected': True,
                'last_updated': '-',
                'stats': {'total_orders': 0, 'successful_orders': 0, 'pending_orders': 0, 
                        'failed_orders': 0, 'last_order_time': '-', 'api_response_time': '-'}
            }
        else:
            stats = simulator_service.get_broker_stats(current_user.id)
            config['access_token'] = bool(config.get('access_token'))
            config['stats'] = stats
        
        return jsonify({'success': True, **config})
        
    except Exception as e:
        logger.error(f"Error getting Simulator broker info for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@simulator_bp.route('/api/test', methods=['POST'])
@login_required
def api_test_simulator_connection():
    """Test Simulator broker connection."""
    try:
        simulator_service = get_simulator_service()
        result = simulator_service.test_connection(current_user.id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error testing Simulator connection for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@simulator_bp.route('/api/config', methods=['POST'])
@login_required
def api_save_simulator_config():
    """Save Simulator broker configuration."""
    try:
        data = request.get_json()
        simulator_service = get_simulator_service()
        
        config_data = {
            'initial_balance': data.get('initial_balance', 100000),
            'market_delay': data.get('market_delay', 100),
            'success_rate': data.get('success_rate', 95)
        }
        
        saved_config = simulator_service.save_broker_config(config_data, current_user.id)
        
        return jsonify({
            'success': True, 
            'message': 'Simulator configuration saved successfully'
        })
        
    except Exception as e:
        logger.error(f"Error saving Simulator config for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@simulator_bp.route('/api/reset', methods=['POST'])
@login_required
def api_reset_simulator():
    """Reset Simulator data."""
    try:
        simulator_service = get_simulator_service()
        
        # Reset simulator configuration to defaults
        config_data = {
            'initial_balance': 100000,
            'market_delay': 100,
            'success_rate': 95
        }
        
        simulator_service.save_broker_config(config_data, current_user.id)
        
        return jsonify({
            'success': True, 
            'message': 'Simulator has been reset successfully'
        })
        
    except Exception as e:
        logger.error(f"Error resetting Simulator for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@simulator_bp.route('/api/funds', methods=['GET'])
@login_required
def api_get_simulator_funds():
    """Get Simulator user funds."""
    try:
        simulator_service = get_simulator_service()
        funds = simulator_service.funds(current_user.id)
        return jsonify({'success': True, 'data': funds})
        
    except Exception as e:
        logger.error(f"Error getting Simulator funds for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@simulator_bp.route('/api/holdings', methods=['GET'])
@login_required
def api_get_simulator_holdings():
    """Get Simulator user holdings."""
    try:
        simulator_service = get_simulator_service()
        holdings = simulator_service.holdings(current_user.id)
        return jsonify({'success': True, 'data': holdings})
        
    except Exception as e:
        logger.error(f"Error getting Simulator holdings for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@simulator_bp.route('/api/positions', methods=['GET'])
@login_required
def api_get_simulator_positions():
    """Get Simulator user positions."""
    try:
        simulator_service = get_simulator_service()
        positions = simulator_service.positions(current_user.id)
        return jsonify({'success': True, 'data': positions})
        
    except Exception as e:
        logger.error(f"Error getting Simulator positions for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@simulator_bp.route('/api/orders', methods=['GET'])
@login_required
def api_get_simulator_orders():
    """Get Simulator user orders."""
    try:
        simulator_service = get_simulator_service()
        orders = simulator_service.orderbook(current_user.id)
        return jsonify({'success': True, 'data': orders})
        
    except Exception as e:
        logger.error(f"Error getting Simulator orders for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@simulator_bp.route('/api/trades', methods=['GET'])
@login_required
def api_get_simulator_trades():
    """Get Simulator user trades."""
    try:
        simulator_service = get_simulator_service()
        trades = simulator_service.tradebook(current_user.id)
        return jsonify({'success': True, 'data': trades})
        
    except Exception as e:
        logger.error(f"Error getting Simulator trades for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@simulator_bp.route('/api/quotes', methods=['GET'])
@login_required
def api_get_simulator_quotes():
    """Get Simulator market quotes."""
    try:
        symbols = request.args.get('symbols', '')
        if not symbols:
            return jsonify({'success': False, 'error': 'Symbols parameter required'}), 400
            
        simulator_service = get_simulator_service()
        quotes = simulator_service.quotes(current_user.id, symbols)
        return jsonify({'success': True, 'data': quotes})
        
    except Exception as e:
        logger.error(f"Error getting Simulator quotes for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@simulator_bp.route('/api/profile', methods=['GET'])
@login_required
def api_get_simulator_profile():
    """Get Simulator user profile."""
    try:
        simulator_service = get_simulator_service()
        profile = simulator_service.login(current_user.id)
        return jsonify({'success': True, 'data': profile})
        
    except Exception as e:
        logger.error(f"Error getting Simulator profile for user {current_user.id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
