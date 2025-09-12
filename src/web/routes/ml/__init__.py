"""
Flask routes for ML stock prediction functionality
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
import logging
from datetime import datetime

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

# Only create blueprint if ML services are available
if ml_available:

# Create blueprint
ml_bp = Blueprint('ml', __name__, url_prefix='/api/v1/ml')

logger = logging.getLogger(__name__)

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
            end_date=end_date
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
        prediction = get_prediction(symbol)
        
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
                result = run_backtest(symbol)
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
        
        results = run_backtest_for_all_stocks()
        
        return jsonify({
            'success': True,
            'results': results,
            'total_symbols': len(results),
            'backtested_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in backtest all: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error during backtest'}), 500

# Health check for ML service
@ml_bp.route('/health', methods=['GET'])
def ml_health():
    """
    Health check for ML service.
    """
    return jsonify({
        'status': 'healthy',
        'service': 'ml_prediction',
        'timestamp': datetime.now().isoformat()
    })
    # Create blueprint
    ml_bp = Blueprint('ml', __name__, url_prefix='/api/v1/ml')

    logger = logging.getLogger(__name__)

    @ml_bp.route('/train', methods=['POST'])
    @login_required
    def train_model():
        """
        Train ML models for a specific stock symbol.
        """
        if not ml_available:
            return jsonify({'success': False, 'error': 'ML services not available'}), 503
            
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
                end_date=end_date
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
        if not ml_available:
            return jsonify({'success': False, 'error': 'ML services not available'}), 503
            
        try:
            symbol = symbol.upper()
            logger.info(f"Getting ML prediction for symbol {symbol} by user {current_user.id}")
            
            # Get prediction
            prediction = get_prediction(symbol)
            
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
        if not ml_available:
            return jsonify({'success': False, 'error': 'ML services not available'}), 503
            
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
                    result = run_backtest(symbol)
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
        if not ml_available:
            return jsonify({'success': False, 'error': 'ML services not available'}), 503
            
        try:
            logger.info(f"Starting backtest for all trained models by user {current_user.id}")
            
            results = run_backtest_for_all_stocks()
            
            return jsonify({
                'success': True,
                'results': results,
                'total_symbols': len(results),
                'backtested_at': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in backtest all: {str(e)}", exc_info=True)
            return jsonify({'success': False, 'error': 'Internal server error during backtest'}), 500

    # Health check for ML service
    @ml_bp.route('/health', methods=['GET'])
    def ml_health():
        """
        Health check for ML service.
        """
        return jsonify({
            'status': 'healthy' if ml_available else 'unavailable',
            'ml_available': ml_available,
            'service': 'ml_prediction',
            'timestamp': datetime.now().isoformat()
        })

else:
    # Create a dummy blueprint if ML services are not available
    ml_bp = Blueprint('ml', __name__, url_prefix='/api/v1/ml')
    
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
