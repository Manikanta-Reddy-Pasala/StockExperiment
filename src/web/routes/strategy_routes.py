"""
API Routes for Strategy System
Provides endpoints for Default Risk and High Risk strategies
"""
import logging
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from datetime import datetime
from ...utils.api_logger import APILogger, log_flask_route

logger = logging.getLogger(__name__)

# Create blueprint
strategy_bp = Blueprint('strategy', __name__, url_prefix='/api/strategy')


@strategy_bp.route('/initialize-stock-universe', methods=['POST'])
@login_required
def api_initialize_stock_universe():
    """Initialize the complete stock universe with market cap categories."""
    try:
        from ...services.stock_data_service import get_stock_data_service
        
        current_app.logger.info(f"Initializing stock universe for user {current_user.id}")
        
        stock_data_service = get_stock_data_service()
        results = stock_data_service.initialize_stock_universe(current_user.id)
        
        return jsonify({
            'success': True,
            'message': 'Stock universe initialized successfully',
            'results': results
        })
        
    except Exception as e:
        current_app.logger.error(f"Error initializing stock universe: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@strategy_bp.route('/create-portfolio', methods=['POST'])
@login_required
def api_create_portfolio_strategy():
    """Create a new portfolio strategy (Default Risk or High Risk)."""
    try:
        from ...services.strategy_service import get_advanced_strategy_service
        
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'Request data is required'}), 400
        
        strategy_type = data.get('strategy_type')
        total_capital = data.get('total_capital')
        strategy_name = data.get('strategy_name')
        
        if not strategy_type or strategy_type not in ['default_risk', 'high_risk']:
            return jsonify({
                'success': False, 
                'error': 'Invalid strategy_type. Must be "default_risk" or "high_risk"'
            }), 400
        
        if not total_capital or total_capital <= 0:
            return jsonify({
                'success': False, 
                'error': 'Valid total_capital is required'
            }), 400
        
        current_app.logger.info(f"Creating {strategy_type} portfolio strategy for user {current_user.id}")
        
        strategy_service = get_advanced_strategy_service()
        result = strategy_service.create_portfolio_strategy(
            current_user.id, strategy_type, total_capital, strategy_name
        )
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400
        
    except Exception as e:
        current_app.logger.error(f"Error creating portfolio strategy: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@strategy_bp.route('/generate-recommendations', methods=['POST'])
@login_required
def api_generate_recommendations():
    """Generate stock recommendations based on strategy type."""
    try:
        from ...services.strategy_service import get_advanced_strategy_service
        
        data = request.get_json()
        current_app.logger.info(f"Received request data: {data}")
        
        if not data:
            current_app.logger.error("No request data provided")
            return jsonify({'success': False, 'error': 'Request data is required'}), 400
        
        strategy_type = data.get('strategy_type')
        capital = data.get('capital', 100000)  # Default 1 lakh
        
        current_app.logger.info(f"Strategy type: {strategy_type}, Capital: {capital}")
        
        if not strategy_type or strategy_type not in ['default_risk', 'high_risk']:
            current_app.logger.error(f"Invalid strategy_type: {strategy_type}")
            return jsonify({
                'success': False, 
                'error': f'Invalid strategy_type: "{strategy_type}". Must be "default_risk" or "high_risk"'
            }), 400
        
        current_app.logger.info(f"Generating {strategy_type} recommendations for user {current_user.id}")
        
        strategy_service = get_advanced_strategy_service()
        result = strategy_service.generate_stock_recommendations(
            current_user.id, strategy_type, capital
        )
        
        current_app.logger.info(f"Strategy service result: {result}")
        
        if result.get('success'):
            return jsonify(result)
        else:
            current_app.logger.error(f"Strategy service returned error: {result.get('error', 'Unknown error')}")
            return jsonify(result), 400
        
    except Exception as e:
        current_app.logger.error(f"Error generating recommendations: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Health check endpoint
@strategy_bp.route('/health', methods=['GET'])
def api_strategy_health():
    """Strategy service health check."""
    try:
        return jsonify({
            'success': True,
            'service': 'Strategy Service',
            'version': '1.0',
            'timestamp': datetime.utcnow().isoformat(),
            'available_strategies': ['default_risk', 'high_risk']
        })
        
    except Exception as e:
        logger.error(f"Error in strategy health check: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
