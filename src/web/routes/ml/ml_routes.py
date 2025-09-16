"""
ML routes - Following SOA principles
Routes handle HTTP logic only, business logic in services
"""
from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
import logging

# Import the ML API service
from ....services.ml_api_service import get_ml_api_service

# Create blueprints
ml_bp = Blueprint('ml', __name__, url_prefix='/api/v1/ml')  # API routes
ml_web_bp = Blueprint('ml_web', __name__)  # Web routes

logger = logging.getLogger(__name__)
ml_api_service = get_ml_api_service()

# ===== API ROUTES =====

@ml_bp.route('/predict', methods=['POST'])
@login_required
def predict_stock_price():
    """Generate ML prediction for a stock"""
    try:
        data = request.get_json()
        if not data.get('symbol'):
            return jsonify({'error': 'Symbol is required'}), 400

        symbol = data['symbol'].upper()
        horizon = data.get('horizon', 1)

        # Delegate to service layer
        result = ml_api_service.predict_stock_price(current_user.id, symbol, horizon)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': 'Internal server error', 'success': False}), 500


@ml_bp.route('/train', methods=['POST'])
@login_required
def train_new_model():
    """Start training a new ML model"""
    try:
        data = request.get_json()

        required_fields = ['symbol', 'start_date', 'end_date']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400

        symbol = data['symbol'].upper()
        start_date = data['start_date']
        end_date = data['end_date']

        # Delegate to service layer
        result = ml_api_service.start_training_job(current_user.id, symbol, start_date, end_date)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Training endpoint error: {e}")
        return jsonify({'error': 'Internal server error', 'success': False}), 500


@ml_bp.route('/training_progress/<int:job_id>', methods=['GET'])
@login_required
def get_training_progress(job_id):
    """Get training job progress"""
    try:
        result = ml_api_service.get_training_progress(job_id)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 404

    except Exception as e:
        logger.error(f"Training progress endpoint error: {e}")
        return jsonify({'error': 'Internal server error', 'success': False}), 500


@ml_bp.route('/models', methods=['GET'])
@login_required
def get_trained_models():
    """Get list of trained models"""
    try:
        result = ml_api_service.get_trained_models(current_user.id)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Models endpoint error: {e}")
        return jsonify({'error': 'Internal server error', 'success': False}), 500


@ml_bp.route('/health', methods=['GET'])
def ml_health():
    """Health check for ML service"""
    from datetime import datetime
    return jsonify({
        'status': 'healthy',
        'service': 'ml_api',
        'timestamp': datetime.now().isoformat()
    })

# ===== WEB ROUTES =====

@ml_web_bp.route('/ml')
@login_required
def ml_dashboard():
    """ML dashboard page"""
    return render_template('ml.html', title='ML Predictions')