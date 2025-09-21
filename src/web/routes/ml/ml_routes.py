"""
ML routes - Following SOA principles
Routes handle HTTP logic only, business logic in services
"""
from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
import logging
from datetime import datetime

# Import the ML API service
from ....services.ml.ml_api_service import get_ml_api_service
from ....services.ml.backtesting_service import backtest_model

# Import screening routes
from .screening_routes import screening_bp

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
            # Add training_id for frontend compatibility
            if 'job_id' in result:
                result['training_id'] = result['job_id']
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
            return jsonify({
                'success': True,
                'data': result
            }), 200
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


@ml_bp.route('/overview', methods=['GET'])
@login_required
def get_ml_overview():
    """Get ML dashboard overview data"""
    try:
        # Get overview data from ML API service
        result = ml_api_service.get_ml_overview(current_user.id)

        if result['success']:
            return jsonify({
                'success': True,
                'data': result['data']
            }), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"ML overview endpoint error: {e}")
        return jsonify({'error': 'Internal server error', 'success': False}), 500


@ml_bp.route('/active-trainings', methods=['GET'])
@login_required
def get_active_trainings():
    """Get active training jobs"""
    try:
        result = ml_api_service.get_active_trainings(current_user.id)

        if result['success']:
            return jsonify({
                'success': True,
                'data': result['data']
            }), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Active trainings endpoint error: {e}")
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



@ml_bp.route('/trained', methods=['GET'])
def list_trained_models():
    """List all trained models in the /api/v1/ml/trained folder"""
    try:
        from ....utils.ml_helpers import get_model_dir
        import os
        from datetime import datetime

        model_dir = get_model_dir()

        if not os.path.exists(model_dir):
            return jsonify({
                'success': False,
                'error': 'Trained models directory not found',
                'path': model_dir
            }), 404

        # Get all files in the trained models directory
        files = os.listdir(model_dir)

        # Group models by symbol
        models_by_symbol = {}

        for file in files:
            if file.startswith('.') or file == '.gitkeep':
                continue

            file_path = os.path.join(model_dir, file)
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            modified_time = datetime.fromtimestamp(file_stats.st_mtime)

            # Extract symbol from filename
            if file.endswith('.pkl') or file.endswith('.h5'):
                # Handle different filename patterns
                if 'scaler' in file:
                    # For scaler files like "NSE:GESHIP-EQ_lstm_feature_scaler.pkl"
                    parts = file.replace('.pkl', '').split('_')
                    if 'feature' in file or 'target' in file:
                        # Get everything before '_lstm_feature' or '_lstm_target'
                        if '_lstm_feature_scaler' in file:
                            symbol = file.replace('_lstm_feature_scaler.pkl', '')
                            model_type = 'lstm_feature_scaler'
                        elif '_lstm_target_scaler' in file:
                            symbol = file.replace('_lstm_target_scaler.pkl', '')
                            model_type = 'lstm_target_scaler'
                        else:
                            symbol = '_'.join(parts[:-2])  # Fallback
                            model_type = '_'.join(parts[-2:])
                    else:
                        symbol = '_'.join(parts[:-1])  # Everything except scaler
                        model_type = 'scaler'
                else:
                    # For regular model files like "NSE:GESHIP-EQ_rf.pkl"
                    parts = file.split('_')
                    if len(parts) >= 2:
                        symbol = '_'.join(parts[:-1])  # Everything except the last part
                        model_type = parts[-1].replace('.pkl', '').replace('.h5', '')
                    else:
                        symbol = parts[0]
                        model_type = 'model'

                if symbol not in models_by_symbol:
                    models_by_symbol[symbol] = {
                        'symbol': symbol,
                        'models': {},
                        'last_trained': modified_time,
                        'total_size': 0
                    }

                models_by_symbol[symbol]['models'][model_type] = {
                    'file': file,
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'modified': modified_time.isoformat(),
                    'type': 'neural_network' if file.endswith('.h5') else 'traditional'
                }

                models_by_symbol[symbol]['total_size'] += file_size

                # Update last trained time to the most recent
                if modified_time > models_by_symbol[symbol]['last_trained']:
                    models_by_symbol[symbol]['last_trained'] = modified_time

        # Convert to list and add summary info
        models_list = []
        for symbol, data in models_by_symbol.items():
            data['total_size_mb'] = round(data['total_size'] / (1024 * 1024), 2)
            data['last_trained'] = data['last_trained'].isoformat()
            data['model_count'] = len(data['models'])

            # Check if it's a complete ensemble (rf, xgb, lstm + scalers)
            has_rf = 'rf' in data['models']
            has_xgb = 'xgb' in data['models']
            has_lstm = 'lstm' in data['models']
            has_feature_scaler = 'lstm_feature_scaler' in data['models']
            has_target_scaler = 'lstm_target_scaler' in data['models']

            data['is_complete'] = has_rf and has_xgb and has_lstm and has_feature_scaler and has_target_scaler
            data['missing_components'] = []

            if not has_rf:
                data['missing_components'].append('Random Forest')
            if not has_xgb:
                data['missing_components'].append('XGBoost')
            if not has_lstm:
                data['missing_components'].append('LSTM')
            if not has_feature_scaler:
                data['missing_components'].append('Feature Scaler')
            if not has_target_scaler:
                data['missing_components'].append('Target Scaler')

            models_list.append(data)

        # Sort by last trained time (most recent first)
        models_list.sort(key=lambda x: x['last_trained'], reverse=True)

        return jsonify({
            'success': True,
            'path': model_dir,
            'total_symbols': len(models_list),
            'total_files': len([f for f in files if not f.startswith('.') and f != '.gitkeep']),
            'models': models_list
        }), 200

    except Exception as e:
        logger.error(f"Error listing trained models: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to list trained models: {str(e)}'
        }), 500

# ===== WEB ROUTES =====

@ml_bp.route('/backtest', methods=['POST'])
@login_required
def backtest_ml_model():
    """Backtest ML model performance on historical data"""
    try:
        data = request.get_json()
        
        if not data.get('symbol'):
            return jsonify({'error': 'Symbol is required'}), 400
        
        symbol = data['symbol'].upper()
        test_period_days = data.get('test_period_days', 30)  # Default 30 days
        
        # Perform backtesting
        result = backtest_model(symbol, current_user.id, test_period_days)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error in backtesting endpoint: {e}")
        return jsonify({
            'success': False,
            'error': f'Backtesting failed: {str(e)}'
        }), 500

@ml_bp.route('/suggested-stocks', methods=['GET'])
@login_required
def get_suggested_stocks():
    """Get suggested stocks for swing trading with risk-based strategies."""
    try:
        # Get query parameters
        strategy_type = request.args.get('strategy', 'default_risk')
        limit = int(request.args.get('limit', 20))
        use_ml = request.args.get('use_ml', 'false').lower() == 'true'

        # Use a default user ID if current_user is not available (for testing)
        user_id = current_user.id if current_user and hasattr(current_user, 'id') else 1

        if use_ml:
            # Try ML-based suggestions first
            try:
                suggestions = _get_ml_based_suggestions(strategy_type, limit, user_id)
                return jsonify({
                    'success': True,
                    'data': suggestions,
                    'strategy_applied': strategy_type,
                    'total': len(suggestions),
                    'method': 'ml_predictions',
                    'last_updated': datetime.now().isoformat()
                }), 200
            except Exception as ml_error:
                logger.warning(f"ML predictions failed, falling back to basic screening: {ml_error}")
                # Fall through to basic screening

        # Basic screening without ML (current working approach)
        suggestions = _get_basic_screening_suggestions(strategy_type, limit, user_id)

        return jsonify({
            'success': True,
            'data': suggestions,
            'strategy_applied': strategy_type,
            'total': len(suggestions),
            'method': 'basic_screening',
            'last_updated': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting suggested stocks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def _get_basic_screening_suggestions(strategy_type: str, limit: int, user_id: int):
    """Get stock suggestions using swing trading filtering and business logic."""
    try:
        # Use Fyers suggested stocks provider with proper filtering pipeline
        from ....services.implementations.fyers_suggested_stocks_provider import FyersSuggestedStocksProvider
        from ....services.interfaces.suggested_stocks_interface import StrategyType

        # Map strategy type to enum
        strategy_map = {
            'default_risk': StrategyType.DEFAULT_RISK,
            'high_risk': StrategyType.HIGH_RISK,
            'medium_risk': StrategyType.DEFAULT_RISK  # Map medium_risk to default_risk
        }

        strategy_enum = strategy_map.get(strategy_type, StrategyType.DEFAULT_RISK)

        provider = FyersSuggestedStocksProvider()
        screening_result = provider.get_suggested_stocks(
            user_id=user_id,
            strategies=[strategy_enum],
            limit=limit
        )

        if not screening_result.get('success'):
            logger.warning(f"Swing trading screening failed: {screening_result.get('error', 'Unknown error')}")
            return []

        # Return the properly filtered suggestions (already processed by FyersSuggestedStocksProvider)
        suggestions = screening_result.get('data', [])

        logger.info(f"✅ Swing trading screening returned {len(suggestions)} suggestions")
        return suggestions

    except Exception as e:
        logger.error(f"Error in basic screening: {e}")
        return []


def _get_ml_based_suggestions(strategy_type: str, limit: int, user_id: int):
    """Get stock suggestions using ML predictions (requires trained models)."""
    # Import ML prediction service
    from ....services.ml.prediction_service import get_prediction_service
    from ....services.ml.stock_discovery_service import get_stock_discovery_service

    # Get stocks for prediction
    discovery_service = get_stock_discovery_service()
    discovered_stocks = discovery_service.get_top_liquid_stocks(user_id, count=50)

    prediction_service = get_prediction_service()
    suggestions = []

    for stock_info in discovered_stocks:
        try:
            # Get ML prediction for this stock
            prediction_result = prediction_service.predict(
                symbol=stock_info.symbol,
                user_id=user_id,
                horizon_days=30
            )

            if prediction_result.get('success') and prediction_result.get('data'):
                pred_data = prediction_result['data']

                # Only suggest stocks with positive ML predictions
                if pred_data.get('signal') == 'BUY' and pred_data.get('expected_return', 0) > 5:
                    suggestion = {
                        'symbol': stock_info.symbol,
                        'name': stock_info.name,
                        'current_price': stock_info.current_price,
                        'target_price': pred_data.get('target_price'),
                        'stop_loss': pred_data.get('stop_loss'),
                        'recommendation': pred_data.get('signal'),
                        'strategy': strategy_type,
                        'risk_level': 'ML-Based',
                        'holding_period': '4-6 weeks',
                        'market_cap': stock_info.market_cap_crores * 10000000,  # Convert to actual value
                        'pe_ratio': None,  # Would need fundamental data
                        'pb_ratio': None,
                        'roe': None,
                        'sales_growth': None,
                        'rsi': pred_data.get('rsi'),
                        'sma_20': pred_data.get('sma_20'),
                        'volume': stock_info.volume,
                        'avg_volume_20d': None,
                        'expected_return': pred_data.get('expected_return'),
                        'risk_reward_ratio': pred_data.get('risk_reward_ratio'),
                        'ml_confidence': pred_data.get('confidence'),
                        'reason': f"ML prediction with {pred_data.get('confidence', 0):.1%} confidence"
                    }
                    suggestions.append(suggestion)

        except Exception as e:
            logger.warning(f"ML prediction failed for {stock_info.symbol}: {e}")
            continue

    # Sort by ML confidence and expected return
    suggestions.sort(key=lambda x: (x.get('ml_confidence', 0) * x.get('expected_return', 0)), reverse=True)
    return suggestions[:limit]


@ml_bp.route('/suggested-stocks-test', methods=['GET'])
def get_suggested_stocks_test():
    """Test endpoint for suggested stocks without authentication."""
    try:
        # Get query parameters
        strategy_type = request.args.get('strategy', 'default_risk')
        limit = int(request.args.get('limit', 20))
        use_ml = request.args.get('use_ml', 'false').lower() == 'true'

        # Use a default user ID for testing
        user_id = 1

        if use_ml:
            # Try ML-based suggestions first
            try:
                suggestions = _get_ml_based_suggestions(strategy_type, limit, user_id)
                return jsonify({
                    'success': True,
                    'data': suggestions,
                    'strategy_applied': strategy_type,
                    'total': len(suggestions),
                    'method': 'ml_predictions',
                    'last_updated': datetime.now().isoformat()
                }), 200
            except Exception as ml_error:
                logger.warning(f"ML predictions failed, falling back to basic screening: {ml_error}")
                # Fall through to basic screening

        # Basic screening without ML (current working approach)
        suggestions = _get_basic_screening_suggestions(strategy_type, limit, user_id)

        return jsonify({
            'success': True,
            'data': suggestions,
            'strategy_applied': strategy_type,
            'total': len(suggestions),
            'method': 'basic_screening',
            'last_updated': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting suggested stocks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ml_web_bp.route('/ml')
@login_required
def ml_dashboard():
    """ML dashboard page"""
    return render_template('ml.html', title='ML Predictions')