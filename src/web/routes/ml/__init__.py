"""
Flask routes for ML stock prediction functionality
"""
from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
import logging
from datetime import datetime, timedelta
import os
import threading
from sqlalchemy import func, and_

# Import database and models
try:
    from ...models.database import get_database_manager
    from ...models.models import MLTrainingJob, MLTrainedModel
except ImportError:
    from models.database import get_database_manager
    from models.models import MLTrainingJob, MLTrainedModel

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

def parse_duration_to_dates(duration, end_date=None):
    """Convert duration string to start_date and end_date."""
    if end_date is None:
        end_date = datetime.now().date()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    duration_map = {
        '1M': timedelta(days=30),
        '3M': timedelta(days=90),
        '6M': timedelta(days=180),
        '1Y': timedelta(days=365),
        '2Y': timedelta(days=730),
        '3Y': timedelta(days=1095),
        '5Y': timedelta(days=1825)
    }

    if duration in duration_map:
        start_date = end_date - duration_map[duration]
        return start_date, end_date
    else:
        # Default to 1 year if unknown duration
        start_date = end_date - timedelta(days=365)
        return start_date, end_date

def run_training_job(job_id, symbol, model_type, start_date, end_date, use_technical_indicators, user_id):
    """Run the actual ML training job in background."""
    db = get_database_manager()
    session = db.Session()

    try:
        # Get the training job from database
        job = session.query(MLTrainingJob).filter(MLTrainingJob.id == job_id).first()
        if not job:
            logger.error(f"Training job {job_id} not found")
            return

        # Update job status to running
        job.status = 'running'
        job.started_at = datetime.now()
        job.progress = 5.0
        session.commit()
        logger.info(f"Started training job {job_id} for {symbol}")

        # Progress update: Data fetching
        job.progress = 15.0
        session.commit()

        # Run the actual training
        if ml_available:
            result = train_and_tune_models(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                user_id=user_id
            )

            # Progress update: Training complete
            job.progress = 90.0
            session.commit()

            # Create trained model record
            if result.get('success', False):
                trained_model = MLTrainedModel(
                    training_job_id=job_id,
                    user_id=user_id,
                    symbol=symbol,
                    model_type=model_type,
                    model_file_path=result.get('model_path'),
                    scaler_file_path=result.get('scaler_path'),
                    accuracy=result.get('accuracy', 0),
                    mse=result.get('mse'),
                    mae=result.get('mae'),
                    start_date=start_date,
                    end_date=end_date
                )
                session.add(trained_model)

                # Update job status to completed
                job.status = 'completed'
                job.progress = 100.0
                job.accuracy = result.get('accuracy', 0)
                job.completed_at = datetime.now()
                logger.info(f"Training job {job_id} completed successfully")
            else:
                job.status = 'failed'
                job.error_message = result.get('error', 'Training failed')
                logger.error(f"Training job {job_id} failed: {job.error_message}")
        else:
            # ML not available - simulate training for demo
            import time
            for progress in [25, 50, 75, 90]:
                time.sleep(2)  # Simulate training time
                job.progress = progress
                session.commit()

            # Create mock trained model
            trained_model = MLTrainedModel(
                training_job_id=job_id,
                user_id=user_id,
                symbol=symbol,
                model_type=model_type,
                accuracy=75.0 + (hash(symbol) % 20),  # Mock accuracy
                start_date=start_date,
                end_date=end_date
            )
            session.add(trained_model)

            job.status = 'completed'
            job.progress = 100.0
            job.accuracy = trained_model.accuracy
            job.completed_at = datetime.now()
            logger.info(f"Mock training job {job_id} completed")

        session.commit()

    except Exception as e:
        logger.error(f"Error in training job {job_id}: {str(e)}", exc_info=True)
        job.status = 'failed'
        job.error_message = str(e)
        session.commit()
    finally:
        session.close()

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
            db = get_database_manager()
            session = db.Session()

            try:
                # Get trained stocks count
                trained_stocks = session.query(MLTrainedModel).filter(
                    and_(MLTrainedModel.user_id == current_user.id,
                         MLTrainedModel.is_active == True)
                ).count()

                # Get active trainings count
                active_trainings = session.query(MLTrainingJob).filter(
                    and_(MLTrainingJob.user_id == current_user.id,
                         MLTrainingJob.status.in_(['pending', 'running']))
                ).count()

                # Calculate success rate from completed training jobs
                total_completed = session.query(MLTrainingJob).filter(
                    and_(MLTrainingJob.user_id == current_user.id,
                         MLTrainingJob.status.in_(['completed', 'failed']))
                ).count()

                successful = session.query(MLTrainingJob).filter(
                    and_(MLTrainingJob.user_id == current_user.id,
                         MLTrainingJob.status == 'completed')
                ).count()

                success_rate = (successful / total_completed * 100) if total_completed > 0 else 0

                # Get last updated time from most recent training job
                last_job = session.query(MLTrainingJob).filter(
                    MLTrainingJob.user_id == current_user.id
                ).order_by(MLTrainingJob.created_at.desc()).first()

                last_updated = last_job.created_at.strftime('%Y-%m-%d %H:%M') if last_job else 'Never'

                overview_data = {
                    'trained_stocks': trained_stocks,
                    'success_rate': round(success_rate, 1),
                    'active_trainings': active_trainings,
                    'last_updated': last_updated
                }

                return jsonify({
                    'success': True,
                    'data': overview_data
                })

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error getting ML overview: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to get ML overview'}), 500

    @ml_bp.route('/models', methods=['GET'])
    @login_required
    def get_trained_stocks():
        """
        Get list of trained stocks for the user.
        """
        try:
            db = get_database_manager()
            session = db.Session()

            try:
                # Get trained models from database
                trained_models = session.query(MLTrainedModel).filter(
                    and_(MLTrainedModel.user_id == current_user.id,
                         MLTrainedModel.is_active == True)
                ).order_by(MLTrainedModel.created_at.desc()).all()

                models = []
                for model in trained_models:
                    models.append({
                        'id': f'model_{model.id}',
                        'symbol': model.symbol,
                        'type': model.model_type,
                        'accuracy': round(model.accuracy, 1) if model.accuracy else 0,
                        'trained_date': model.created_at.strftime('%Y-%m-%d'),
                        'start_date': model.start_date.strftime('%Y-%m-%d'),
                        'end_date': model.end_date.strftime('%Y-%m-%d'),
                        'mse': round(model.mse, 4) if model.mse else None,
                        'mae': round(model.mae, 4) if model.mae else None
                    })

                return jsonify({
                    'success': True,
                    'data': models
                })

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error getting trained stocks: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to get trained stocks'}), 500

    @ml_bp.route('/active-trainings', methods=['GET'])
    @login_required
    def get_active_trainings():
        """
        Get list of active training jobs for the user.
        """
        try:
            db = get_database_manager()
            session = db.Session()

            try:
                # Get active training jobs from database
                active_jobs = session.query(MLTrainingJob).filter(
                    and_(MLTrainingJob.user_id == current_user.id,
                         MLTrainingJob.status.in_(['pending', 'running']))
                ).order_by(MLTrainingJob.created_at.desc()).all()

                trainings = []
                for job in active_jobs:
                    trainings.append({
                        'id': job.id,
                        'symbol': job.symbol,
                        'model_type': job.model_type,
                        'status': job.status,
                        'progress': job.progress or 0,
                        'start_date': job.start_date.strftime('%Y-%m-%d'),
                        'end_date': job.end_date.strftime('%Y-%m-%d'),
                        'duration': job.duration,
                        'created_at': job.created_at.strftime('%Y-%m-%d %H:%M'),
                        'started_at': job.started_at.strftime('%Y-%m-%d %H:%M') if job.started_at else None
                    })

                return jsonify({
                    'success': True,
                    'data': trainings
                })

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error getting active trainings: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to get active trainings'}), 500

    @ml_bp.route('/train', methods=['POST'])
    @login_required
    def train_new_model():
        """
        Start training a new ML model with dashboard parameters.
        """
        db = None
        session = None
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

            # Parse start and end dates from duration
            try:
                start_date, end_date = parse_duration_to_dates(duration)

                # If custom dates provided, use them instead
                if data.get('start_date'):
                    start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
                if data.get('end_date'):
                    end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')

            except ValueError as e:
                logger.error(f"Invalid duration format: {duration}")
                return jsonify({'error': f'Invalid duration format: {str(e)}'}), 400

            logger.info(f"Starting ML training for {symbol} with {model_type} model by user {current_user.id}")

            # Create training job record in database
            db = get_database_manager()
            session = db.Session()

            try:
                training_job = MLTrainingJob(
                    user_id=current_user.id,
                    symbol=symbol,
                    model_type=model_type,
                    start_date=start_date,
                    end_date=end_date,
                    status='pending',
                    progress=0.0,
                    use_technical_indicators=use_technical_indicators,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )

                session.add(training_job)
                session.commit()

                job_id = training_job.id
                logger.info(f"Created training job {job_id} in database")

                # Start background training job
                training_thread = threading.Thread(
                    target=run_training_job,
                    args=(job_id, symbol, model_type, start_date, end_date, use_technical_indicators, current_user.id),
                    daemon=True
                )
                training_thread.start()

                logger.info(f"Started background training thread for job {job_id}")

                return jsonify({
                    'success': True,
                    'training_id': job_id,
                    'message': f'Training started for {symbol}',
                    'symbol': symbol,
                    'model_type': model_type,
                    'duration': duration,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                })

            except Exception as db_error:
                session.rollback()
                logger.error(f"Database error creating training job: {str(db_error)}")
                return jsonify({'success': False, 'error': 'Failed to create training job'}), 500
            finally:
                if session:
                    session.close()

        except Exception as e:
            logger.error(f"Error starting training: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to start training'}), 500

    @ml_bp.route('/validate_symbol', methods=['POST'])
    @login_required
    def validate_stock_symbol():
        """
        Validate stock symbol using Fyers exchange data.
        """
        try:
            data = request.get_json()

            if not data.get('symbol'):
                return jsonify({'error': 'Symbol is required'}), 400

            symbol = data['symbol'].upper().strip()

            # Try to import FyersService
            try:
                from ...services.brokers.fyers_service import FyersService
                fyers_service = FyersService()

                # Check if user has Fyers configuration
                fyers_config = fyers_service.get_broker_config(current_user.id)

                if not fyers_config:
                    # Return mock validation if no Fyers config available
                    logger.info(f"No Fyers config for user {current_user.id}, using basic validation")

                    # Basic validation - check format
                    is_valid = len(symbol) > 0 and symbol.isalpha()
                    return jsonify({
                        'success': True,
                        'valid': is_valid,
                        'symbol': symbol,
                        'message': 'Basic validation (Fyers not configured)',
                        'exchange': 'NSE' if is_valid else None
                    })

                # Use Fyers search API to validate symbol
                search_result = fyers_service.search(current_user.id, symbol)

                if search_result and search_result.get('status') == 'success':
                    # Symbol found in Fyers
                    return jsonify({
                        'success': True,
                        'valid': True,
                        'symbol': symbol,
                        'message': 'Symbol validated with Fyers',
                        'exchange': 'NSE',  # Default to NSE
                        'fyers_data': search_result.get('data', [])[:5]  # Return first 5 matches
                    })
                else:
                    return jsonify({
                        'success': True,
                        'valid': False,
                        'symbol': symbol,
                        'message': 'Symbol not found in Fyers exchange',
                        'exchange': None
                    })

            except ImportError:
                logger.warning("FyersService not available, using basic validation")
                is_valid = len(symbol) > 0 and symbol.isalpha()
                return jsonify({
                    'success': True,
                    'valid': is_valid,
                    'symbol': symbol,
                    'message': 'Basic validation (Fyers service not available)',
                    'exchange': 'NSE' if is_valid else None
                })

        except Exception as e:
            logger.error(f"Error validating symbol: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to validate symbol'}), 500

    @ml_bp.route('/training_progress/<int:job_id>', methods=['GET'])
    @login_required
    def get_training_progress(job_id):
        """
        Get training job progress by job ID.
        """
        try:
            db = get_database_manager()
            session = db.Session()

            try:
                # Get training job
                training_job = session.query(MLTrainingJob).filter(
                    and_(MLTrainingJob.id == job_id,
                         MLTrainingJob.user_id == current_user.id)
                ).first()

                if not training_job:
                    return jsonify({'success': False, 'error': 'Training job not found'}), 404

                return jsonify({
                    'success': True,
                    'data': {
                        'id': training_job.id,
                        'symbol': training_job.symbol,
                        'model_type': training_job.model_type,
                        'status': training_job.status,
                        'progress': training_job.progress,
                        'accuracy': training_job.accuracy,
                        'created_at': training_job.created_at.isoformat() if training_job.created_at else None,
                        'updated_at': training_job.updated_at.isoformat() if training_job.updated_at else None,
                        'completed_at': training_job.completed_at.isoformat() if training_job.completed_at else None,
                        'error_message': training_job.error_message
                    }
                })

            finally:
                if session:
                    session.close()

        except Exception as e:
            logger.error(f"Error getting training progress: {str(e)}")
            return jsonify({'success': False, 'error': 'Failed to get training progress'}), 500

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