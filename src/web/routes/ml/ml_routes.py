"""
ML routes - Following SOA principles
Routes handle HTTP logic only, business logic in services
"""
from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
import logging
from datetime import datetime

# Import the ML API service
from ....services.ml_api_service import get_ml_api_service
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
        
        # Import the stock screening service
        from ....services.stock_screening_service import get_stock_screening_service, StrategyType
        
        # Convert string to enum
        if strategy_type == 'high_risk':
            strategies = [StrategyType.HIGH_RISK]
        else:
            strategies = [StrategyType.DEFAULT_RISK]
        
        # Get stock screening service
        screening_service = get_stock_screening_service()
        
        # Use a default user ID if current_user is not available (for testing)
        user_id = current_user.id if current_user and hasattr(current_user, 'id') else 1
        
        # Screen stocks based on strategy
        screened_stocks = screening_service.screen_stocks(strategies, user_id)
        
        # Convert to API response format
        suggestions = []
        for stock in screened_stocks:
            suggestion = {
                'symbol': stock.symbol,
                'name': stock.name,
                'current_price': stock.current_price,
                'target_price': stock.target_price,
                'stop_loss': stock.stop_loss,
                'recommendation': stock.recommendation,
                'strategy': stock.strategy,
                'risk_level': stock.risk_level,
                'holding_period': stock.holding_period,
                'market_cap': stock.market_cap,
                'pe_ratio': stock.pe_ratio,
                'pb_ratio': stock.pb_ratio,
                'roe': stock.roe,
                'sales_growth': stock.sales_growth,
                'rsi': stock.rsi,
                'sma_20': stock.sma_20,
                'volume': stock.volume,
                'avg_volume_20d': stock.avg_volume_20d,
                'expected_return': ((stock.target_price - stock.current_price) / stock.current_price * 100) if stock.target_price else 0,
                'risk_reward_ratio': ((stock.target_price - stock.current_price) / (stock.current_price - stock.stop_loss)) if stock.target_price and stock.stop_loss else 0
            }
            suggestions.append(suggestion)
        
        return jsonify({
            'success': True,
            'data': suggestions,
            'strategy_applied': strategy_type,
            'total': len(suggestions),
            'last_updated': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting suggested stocks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ml_bp.route('/suggested-stocks-test', methods=['GET'])
def get_suggested_stocks_test():
    """Test endpoint for suggested stocks without authentication."""
    try:
        # Get query parameters
        strategy_type = request.args.get('strategy', 'default_risk')
        limit = int(request.args.get('limit', 20))
        
        # Import the stock screening service
        from ....services.stock_screening_service import get_stock_screening_service, StrategyType
        
        # Convert string to enum
        if strategy_type == 'high_risk':
            strategies = [StrategyType.HIGH_RISK]
        else:
            strategies = [StrategyType.DEFAULT_RISK]
        
        # Get stock screening service
        screening_service = get_stock_screening_service()
        
        # Use a default user ID for testing
        user_id = 1
        
        # Screen stocks based on strategy
        screened_stocks = screening_service.screen_stocks(strategies, user_id)
        
        # Convert to API response format
        suggestions = []
        for stock in screened_stocks:
            suggestion = {
                'symbol': stock.symbol,
                'name': stock.name,
                'current_price': stock.current_price,
                'target_price': stock.target_price,
                'stop_loss': stock.stop_loss,
                'recommendation': stock.recommendation,
                'strategy': stock.strategy,
                'risk_level': stock.risk_level,
                'holding_period': stock.holding_period,
                'market_cap': stock.market_cap,
                'pe_ratio': stock.pe_ratio,
                'pb_ratio': stock.pb_ratio,
                'roe': stock.roe,
                'sales_growth': stock.sales_growth,
                'rsi': stock.rsi,
                'sma_20': stock.sma_20,
                'volume': stock.volume,
                'avg_volume_20d': stock.avg_volume_20d,
                'expected_return': ((stock.target_price - stock.current_price) / stock.current_price * 100) if stock.target_price else 0,
                'risk_reward_ratio': ((stock.target_price - stock.current_price) / (stock.current_price - stock.stop_loss)) if stock.target_price and stock.stop_loss else 0
            }
            suggestions.append(suggestion)
        
        return jsonify({
            'success': True,
            'data': suggestions,
            'strategy_applied': strategy_type,
            'total': len(suggestions),
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