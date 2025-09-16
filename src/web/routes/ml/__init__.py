"""
Flask routes for ML stock prediction functionality
"""
from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
import logging
from datetime import datetime
import os

# Import ML services with robust error handling
ml_available = False
try:
    # Try relative imports first (for normal usage)
    from ...services.ml.training_service import train_and_tune_models
    from ...services.ml.prediction_service import get_prediction
    from ...services.ml.backtest_service import run_backtest, run_backtest_for_all_stocks
    ml_available = True
except ImportError:
    try:
        # Fall back to absolute imports (for testing)
        import sys
        import os
        # Add paths to make imports work
        current_dir = os.path.dirname(__file__)
        src_dir = os.path.join(current_dir, '..', '..', '..')
        ml_services_dir = os.path.join(src_dir, 'services', 'ml')
        sys.path.insert(0, ml_services_dir)
        
        from training_service import train_and_tune_models
        from prediction_service import get_prediction
        from backtest_service import run_backtest, run_backtest_for_all_stocks
        ml_available = True
    except ImportError as e:
        print(f"Warning: ML services not available: {e}")
        ml_available = False

# Create separate blueprints for API and web routes
ml_bp = Blueprint('ml', __name__, url_prefix='/api/v1/ml')  # API routes
ml_web_bp = Blueprint('ml_web', __name__)  # Web routes
logger = logging.getLogger(__name__)

# Only create ML functionality if ML services are available
if ml_available:

    @ml_bp.route('/train', methods=['POST'])
    @login_required
    def train_model():
        """
        Train ML models for a specific stock symbol.
        """
        try:
            data = request.get_json()
            
            # Validate input
            if not data or 'symbol' not in data:
                return jsonify({'error': 'Symbol is required'}), 400
            
            symbol = data['symbol'].upper()
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            
            logger.info(f"Starting ML training for symbol {symbol} by user {current_user.id}")
            
            # Convert string dates to date objects if provided
            if start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            
            # Train the models
            result = train_and_tune_models(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                user_id=current_user.id
            )
            
            logger.info(f"ML training completed for symbol {symbol}")
            return jsonify({
                'success': True,
                'message': result['message'],
                'symbol': symbol,
                'trained_at': datetime.now().isoformat()
            })
            
        except ValueError as e:
            logger.error(f"Validation error in ML training: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error in ML training: {str(e)}", exc_info=True)
            return jsonify({'success': False, 'error': 'Internal server error during training'}), 500

    @ml_bp.route('/predict/<symbol>', methods=['GET'])
    @login_required
    def predict_stock(symbol):
        """
        Get ML prediction for a stock symbol.
        """
        try:
            symbol = symbol.upper()
            logger.info(f"Getting ML prediction for symbol {symbol} by user {current_user.id}")
            
            # Get prediction
            prediction = get_prediction(symbol, user_id=current_user.id)
            
            return jsonify({
                'success': True,
                'data': prediction,
                'predicted_at': datetime.now().isoformat()
            })
            
        except FileNotFoundError as e:
            logger.warning(f"Models not found for symbol {symbol}: {str(e)}")
            return jsonify({
                'success': False, 
                'error': f'Models for {symbol} not found. Please train the models first.'
            }), 404
        except Exception as e:
            logger.error(f"Error in ML prediction for {symbol}: {str(e)}", exc_info=True)
            return jsonify({'success': False, 'error': 'Internal server error during prediction'}), 500

    @ml_bp.route('/backtest', methods=['POST'])
    @login_required
    def backtest_stocks():
        """
        Run backtest for specified stock symbols.
        """
        try:
            data = request.get_json()
            
            # Validate input
            if not data or 'symbols' not in data:
                return jsonify({'error': 'Symbols list is required'}), 400
            
            symbols = [s.upper() for s in data['symbols']]
            logger.info(f"Starting backtest for symbols {symbols} by user {current_user.id}")
            
            results = []
            for symbol in symbols:
                try:
                    result = run_backtest(symbol, user_id=current_user.id)
                    results.append(result)
                except FileNotFoundError:
                    logger.warning(f"Models not found for symbol {symbol}, skipping backtest")
                    continue
                except Exception as e:
                    logger.error(f"Error in backtest for {symbol}: {str(e)}")
                    continue
            
            return jsonify({
                'success': True,
                'results': results,
                'backtested_at': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}", exc_info=True)
            return jsonify({'success': False, 'error': 'Internal server error during backtest'}), 500

    @ml_bp.route('/backtest/all', methods=['POST'])
    @login_required
    def backtest_all_stocks():
        """
        Run backtest for all stocks with trained models.
        """
        try:
            logger.info(f"Starting backtest for all trained models by user {current_user.id}")
            
            results = run_backtest_for_all_stocks(user_id=current_user.id)
            
            return jsonify({
                'success': True,
                'results': results,
                'total_symbols': len(results),
                'backtested_at': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in backtest all: {str(e)}", exc_info=True)
            return jsonify({'success': False, 'error': 'Internal server error during backtest'}), 500

    # New ML Dashboard API endpoints
    @ml_bp.route('/overview', methods=['GET'])
    @login_required
    def ml_overview():
        """
        Get ML dashboard overview data.
        """
        try:
            # TODO: Implement actual data retrieval from database
            overview_data = {
                'trained_models': 12,  # Count of trained models for user
                'success_rate': 75.5,  # Average training success rate
                'active_trainings': 2,  # Number of currently running training jobs
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
            }

            return jsonify({
                'success': True,
                'data': overview_data
            })

        except Exception as e:
            logger.error(f"Error getting ML overview: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to get ML overview'}), 500

    @ml_bp.route('/models', methods=['GET'])
    @login_required
    def get_trained_models():
        """
        Get list of trained models for the user.
        """
        try:
            # TODO: Implement actual model retrieval from database/filesystem
            models = [
                {
                    'id': 'model_1',
                    'symbol': 'NSE:RELIANCE-EQ',
                    'type': 'ensemble',
                    'accuracy': 82.5,
                    'trained_date': '2024-01-15',
                },
                {
                    'id': 'model_2',
                    'symbol': 'NSE:TCS-EQ',
                    'type': 'random_forest',
                    'accuracy': 78.3,
                    'trained_date': '2024-01-14',
                },
                {
                    'id': 'model_3',
                    'symbol': 'NSE:HDFCBANK-EQ',
                    'type': 'lstm',
                    'accuracy': 71.2,
                    'trained_date': '2024-01-13',
                }
            ]

            return jsonify({
                'success': True,
                'data': models
            })

        except Exception as e:
            logger.error(f"Error getting trained models: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to get trained models'}), 500

    @ml_bp.route('/train', methods=['POST'])
    @login_required
    def train_new_model():
        """
        Start training a new ML model with dashboard parameters.
        """
        try:
            data = request.get_json()

            # Validate input
            required_fields = ['symbol', 'duration', 'model_type']
            for field in required_fields:
                if not data.get(field):
                    return jsonify({'error': f'{field} is required'}), 400

            symbol = data['symbol'].upper()
            duration = data['duration']
            model_type = data['model_type']
            use_technical_indicators = data.get('use_technical_indicators', True)

            logger.info(f"Starting ML training for {symbol} with {model_type} model by user {current_user.id}")

            # TODO: Start actual training job
            training_id = f"training_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # For now, return success with training ID
            return jsonify({
                'success': True,
                'training_id': training_id,
                'message': f'Training started for {symbol}',
                'symbol': symbol,
                'model_type': model_type,
                'duration': duration
            })

        except Exception as e:
            logger.error(f"Error starting training: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to start training'}), 500

    @ml_bp.route('/predict', methods=['POST'])
    @login_required
    def predict_stock_price():
        """
        Get ML prediction for a stock with dashboard parameters.
        """
        try:
            data = request.get_json()

            if not data.get('symbol'):
                return jsonify({'error': 'Symbol is required'}), 400

            symbol = data['symbol'].upper()
            horizon = data.get('horizon', 1)  # days

            logger.info(f"Getting prediction for {symbol} with {horizon} day horizon by user {current_user.id}")

            # TODO: Use actual trained models for prediction
            # For now, return mock prediction data
            current_price = 2500.0  # Mock current price
            predicted_price = current_price * (1 + 0.023)  # Mock 2.3% increase

            prediction_data = {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_percent': ((predicted_price - current_price) / current_price) * 100,
                'confidence': 0.78,
                'signal': 'BUY' if predicted_price > current_price else 'SELL',
                'horizon_days': horizon,
                'model_breakdown': [
                    {'name': 'Random Forest', 'prediction': predicted_price * 0.98, 'weight': 0.4, 'confidence': 0.82},
                    {'name': 'XGBoost', 'prediction': predicted_price * 1.01, 'weight': 0.35, 'confidence': 0.75},
                    {'name': 'LSTM', 'prediction': predicted_price * 1.02, 'weight': 0.25, 'confidence': 0.71},
                ]
            }

            return jsonify({
                'success': True,
                'data': prediction_data
            })

        except Exception as e:
            logger.error(f"Error getting prediction: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to get prediction'}), 500

    # Health check for ML service
    @ml_bp.route('/health', methods=['GET'])
    def ml_health():
        """
        Health check for ML service.
        """
        return jsonify({
            'status': 'healthy',
            'ml_available': ml_available,
            'service': 'ml_prediction',
            'timestamp': datetime.now().isoformat()
        })

else:
    # Create dummy API endpoints if ML services are not available

    @ml_bp.route('/health', methods=['GET'])
    def ml_health():
        """
        Health check for ML service when unavailable.
        """
        return jsonify({
            'status': 'unavailable',
            'ml_available': False,
            'service': 'ml_prediction',
            'error': 'ML dependencies not installed',
            'timestamp': datetime.now().isoformat()
        })

# Web routes (always available)
@ml_web_bp.route('/ml')
@login_required
def ml_dashboard():
    """
    Machine Learning dashboard page.
    """
    return render_template('ml.html', user=current_user)