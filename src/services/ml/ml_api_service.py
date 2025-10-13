"""
ML API Service - Handles ML operations business logic
Separates business logic from web routes following SOA principles
"""
import logging
from datetime import datetime, timedelta
import threading
from sqlalchemy import func, and_

from src.models.database import get_database_manager
from src.models.models import MLTrainingJob, MLTrainedModel
from ..core.unified_broker_service import get_unified_broker_service

logger = logging.getLogger(__name__)

class MLAPIService:
    """Service class for ML API operations"""

    def __init__(self):
        self.db_manager = get_database_manager()
        self.unified_broker_service = get_unified_broker_service()

    def get_current_price(self, user_id, symbol):
        """
        Get current price for a symbol using the unified broker service.
        Falls back to historical data if real-time quotes fail.
        """
        try:
            logger.info(f"Fetching current price for {symbol} for user {user_id}")

            # Try to get real-time quotes first
            quotes_result = self.unified_broker_service.get_quotes(user_id, [symbol])

            if quotes_result and quotes_result.get('status') == 'success':
                quotes_data = quotes_result.get('data', {})
                if symbol in quotes_data:
                    current_price = float(quotes_data[symbol].get('ltp', 0))
                    if current_price > 0:
                        logger.info(f"Successfully fetched real-time current price for {symbol}: ₹{current_price}")
                        return current_price
                    else:
                        logger.warning(f"Received invalid price for {symbol}: {current_price}")
                else:
                    logger.warning(f"Symbol {symbol} not found in quotes data")
            else:
                logger.warning(f"Failed to get real-time quotes for {symbol}: {quotes_result}")

            # Fallback: Get the most recent price from historical data
            logger.info(f"Falling back to historical data for current price of {symbol}")
            try:
                # Try to get historical data directly from the broker service
                historical_result = self.unified_broker_service.get_historical_data(user_id, symbol, "1D", "1d")
                logger.info(f"Historical data result: {historical_result}")
                
                if historical_result and historical_result.get('status') == 'success':
                    data = historical_result.get('data', {})
                    candles = data.get('candles', [])
                    logger.info(f"Found {len(candles)} candles in historical data")
                    
                    if candles and len(candles) > 0:
                        # Get the most recent close price from the last candle
                        latest_candle = candles[-1]
                        logger.info(f"Latest candle: {latest_candle}")
                        
                        latest_price = float(latest_candle.get('close', 0))
                        if latest_price > 0:
                            logger.info(f"Successfully fetched historical current price for {symbol}: ₹{latest_price}")
                            return latest_price
                        else:
                            logger.warning(f"Invalid close price in historical data for {symbol}: {latest_price}")
                    else:
                        logger.warning(f"No candles data available for {symbol}")
                else:
                    logger.warning(f"Failed to get historical data for {symbol}: {historical_result}")
                
                # If historical data fails, try the ML data service as last resort
                try:
                    from .ml.data_service import get_stock_data
                    df = get_stock_data(symbol, period="1d", user_id=user_id)
                    if df is not None and len(df) > 0:
                        # Get the most recent close price
                        latest_price = float(df['close'].iloc[-1])
                        logger.info(f"Successfully fetched ML data current price for {symbol}: ₹{latest_price}")
                        return latest_price
                    else:
                        logger.warning(f"No ML data available for {symbol}")
                        return None
                except Exception as ml_error:
                    logger.error(f"Error fetching ML data for {symbol}: {ml_error}")
                    return None
            except Exception as hist_error:
                logger.error(f"Error fetching historical data for {symbol}: {hist_error}")
                return None

        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def predict_stock_price(self, user_id, symbol, horizon=1):
        """
        Generate stock price prediction with real current price.
        Returns prediction result or error.
        """
        try:
            logger.info(f"Generating prediction for {symbol} with {horizon} day horizon for user {user_id}")

            # Import ML services
            from .ml.prediction_service import get_prediction

            # Get ML prediction
            prediction_result = get_prediction(symbol, user_id, horizon)

            if not prediction_result:
                return {
                    'success': False,
                    'error': 'Failed to generate prediction - no trained model available'
                }

            # Get real current price
            current_price = self.get_current_price(user_id, symbol)

            if current_price is None:
                logger.warning(f"Could not fetch real current price for {symbol}, prediction may be incomplete")
                # Return prediction without current price rather than using hardcoded value
                return {
                    'success': True,
                    'symbol': symbol,
                    'prediction': prediction_result,
                    'current_price_available': False,
                    'message': 'Prediction generated but current price unavailable'
                }

            # Add current price to prediction result
            prediction_result['current_price'] = current_price
            
            # Recalculate percentage change using the real current price
            final_predicted_price = prediction_result.get('final_predicted_price', 0)
            if final_predicted_price > 0 and current_price > 0:
                corrected_change_percent = ((final_predicted_price - current_price) / current_price) * 100
                prediction_result['predicted_change_percent'] = corrected_change_percent
                
                # Update signal based on corrected percentage
                if corrected_change_percent > 2:
                    prediction_result['signal'] = "BUY"
                elif corrected_change_percent < -2:
                    prediction_result['signal'] = "SELL"
                else:
                    prediction_result['signal'] = "HOLD"

            return {
                'success': True,
                'symbol': symbol,
                'prediction': prediction_result,
                'current_price': current_price,
                'current_price_available': True
            }

        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            }

    def start_training_job(self, user_id, symbol, start_date, end_date):
        """
        Start a new ML training job.
        Returns job information or error.
        """
        try:
            logger.info(f"Starting training job for {symbol} from {start_date} to {end_date} for user {user_id}")

            # Validate dates
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')

                if start_dt >= end_dt:
                    return {
                        'success': False,
                        'error': 'Start date must be before end date'
                    }

                if end_dt > datetime.now():
                    return {
                        'success': False,
                        'error': 'End date cannot be in the future'
                    }

            except ValueError as e:
                return {
                    'success': False,
                    'error': f'Invalid date format: {str(e)}'
                }

            # Check for existing running job for this symbol
            with self.db_manager.get_session() as session:
                existing_job = session.query(MLTrainingJob).filter(
                    and_(
                        MLTrainingJob.symbol == symbol,
                        MLTrainingJob.status == 'running'
                    )
                ).first()

                if existing_job:
                    return {
                        'success': False,
                        'error': f'Training job already running for {symbol}',
                        'existing_job_id': existing_job.id
                    }

            # Calculate duration based on date range
            duration_days = (end_dt - start_dt).days
            if duration_days <= 90:
                duration = '3M'
            elif duration_days <= 180:
                duration = '6M'
            elif duration_days <= 365:
                duration = '1Y'
            else:
                duration = '2Y'

            # Create new training job
            with self.db_manager.get_session() as session:
                training_job = MLTrainingJob(
                    symbol=symbol,
                    model_type='ensemble',  # Default to ensemble model
                    start_date=start_dt,
                    end_date=end_dt,
                    duration=duration,
                    use_technical_indicators=True,
                    status='running',
                    progress=0.0,
                    created_at=datetime.utcnow(),
                    user_id=user_id
                )
                session.add(training_job)
                session.commit()
                job_id = training_job.id

            # Start training in background thread
            def run_training():
                training_results = None
                try:
                    from .ml.training_service import train_and_tune_models
                    result = train_and_tune_models(symbol, start_dt.date(), end_dt.date(), job_id)

                    # Update job status
                    with self.db_manager.get_session() as session:
                        job = session.query(MLTrainingJob).filter(MLTrainingJob.id == job_id).first()
                        if job:
                            if result:
                                job.status = 'completed'
                                job.progress = 100.0
                                job.completed_at = datetime.utcnow()

                                # Store training results including backtesting data
                                import json
                                import numpy as np

                                # Convert numpy types to native Python types for JSON serialization
                                def convert_numpy_types(obj):
                                    if isinstance(obj, np.floating):
                                        return float(obj)
                                    elif isinstance(obj, np.integer):
                                        return int(obj)
                                    elif isinstance(obj, np.ndarray):
                                        return obj.tolist()
                                    elif isinstance(obj, dict):
                                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                                    elif isinstance(obj, list):
                                        return [convert_numpy_types(item) for item in obj]
                                    return obj

                                cleaned_result = convert_numpy_types(result) if result else None
                                job.training_results = json.dumps(cleaned_result) if cleaned_result else None
                                training_results = result

                                logger.info(f"Training completed successfully for {symbol}, job {job_id}")
                            else:
                                job.status = 'failed'
                                job.error_message = 'Training failed'
                            session.commit()

                except Exception as e:
                    logger.error(f"Training job {job_id} failed: {e}")
                    with self.db_manager.get_session() as session:
                        job = session.query(MLTrainingJob).filter(MLTrainingJob.id == job_id).first()
                        if job:
                            job.status = 'failed'
                            job.error_message = str(e)
                            session.commit()

            training_thread = threading.Thread(target=run_training)
            training_thread.daemon = True
            training_thread.start()

            return {
                'success': True,
                'job_id': job_id,
                'message': f'Training started for {symbol}',
                'symbol': symbol
            }

        except Exception as e:
            logger.error(f"Error starting training job for {symbol}: {e}")
            return {
                'success': False,
                'error': f'Failed to start training: {str(e)}'
            }

    def get_training_progress(self, job_id):
        """
        Get training job progress.
        Returns progress information or error.
        """
        try:
            with self.db_manager.get_session() as session:
                job = session.query(MLTrainingJob).filter(MLTrainingJob.id == job_id).first()

                if not job:
                    return {
                        'success': False,
                        'error': 'Training job not found'
                    }

                response_data = {
                    'success': True,
                    'job_id': job_id,
                    'symbol': job.symbol,
                    'status': job.status,
                    'progress': job.progress,
                    'created_at': job.created_at.isoformat() if job.created_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'error_message': job.error_message
                }

                # Include training results if the job is completed and has results
                if job.status == 'completed' and hasattr(job, 'training_results') and job.training_results:
                    try:
                        import json
                        training_results = json.loads(job.training_results)
                        response_data['training_results'] = training_results
                        logger.info(f"Including training results for completed job {job_id}")
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Failed to parse training results for job {job_id}: {e}")

                return response_data

        except Exception as e:
            logger.error(f"Error getting training progress for job {job_id}: {e}")
            return {
                'success': False,
                'error': f'Failed to get progress: {str(e)}'
            }

    def get_trained_models(self, user_id=None):
        """
        Get list of trained models.
        Returns models list or error.
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(MLTrainedModel)
                if user_id:
                    query = query.filter(MLTrainedModel.user_id == user_id)

                models = query.all()

                models_data = []
                for model in models:
                    models_data.append({
                        'id': model.id,
                        'symbol': model.symbol,
                        'type': model.model_type,
                        'model_type': model.model_type,
                        'accuracy': model.accuracy * 100 if model.accuracy else 0,
                        'accuracy_score': model.accuracy,
                        'trained_date': model.created_at.isoformat() if model.created_at else None,
                        'created_at': model.created_at.isoformat() if model.created_at else None,
                        'last_prediction': None  # This field doesn't exist in the model
                    })

                return {
                    'success': True,
                    'data': models_data,
                    'models': models_data,
                    'count': len(models_data)
                }

        except Exception as e:
            logger.error(f"Error getting trained models: {e}")
            return {
                'success': False,
                'error': f'Failed to get models: {str(e)}'
            }

    def get_ml_overview(self, user_id):
        """
        Get ML dashboard overview statistics for all 3 models.
        Returns overview data for Traditional ML, Raw LSTM, and Kronos models.
        """
        try:
            from pathlib import Path
            import os
            from sqlalchemy import text

            models_data = []

            # MODEL 1: Traditional ML (RF + XGBoost)
            traditional_model = {
                'name': 'Traditional ML',
                'description': 'Random Forest + XGBoost',
                'type': 'traditional',
                'stocks_trained': 0,
                'last_trained': 'Never',
                'status': 'not_trained',
                'accuracy': 0
            }

            # Check if traditional models exist
            model_dir = Path('ml_models')
            rf_model = model_dir / 'rf_price_model.pkl'
            xgb_model = model_dir / 'xgb_price_model.pkl'
            metadata_file = model_dir / 'metadata.pkl'

            if rf_model.exists() and xgb_model.exists():
                traditional_model['status'] = 'trained'

                # Get metadata if available
                if metadata_file.exists():
                    try:
                        import pickle
                        with open(metadata_file, 'rb') as f:
                            metadata = pickle.load(f)
                            traditional_model['stocks_trained'] = metadata.get('samples', 0)
                            traditional_model['accuracy'] = round(metadata.get('price_r2', 0) * 100, 1)

                            # Get last modified time
                            last_modified = rf_model.stat().st_mtime
                            traditional_model['last_trained'] = datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M')
                    except Exception as e:
                        logger.warning(f"Error loading traditional ML metadata: {e}")
                else:
                    # Fallback to file modification time
                    last_modified = rf_model.stat().st_mtime
                    traditional_model['last_trained'] = datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M')

            models_data.append(traditional_model)

            # MODEL 2: Raw LSTM (Deep Learning)
            lstm_model = {
                'name': 'Raw LSTM',
                'description': 'Deep Learning OHLCV',
                'type': 'raw_lstm',
                'stocks_trained': 0,
                'last_trained': 'Never',
                'status': 'not_trained',
                'accuracy': 0
            }

            # Count LSTM models (each stock has its own model)
            lstm_dir = Path('ml_models/raw_ohlcv_lstm')
            if lstm_dir.exists():
                lstm_models = [d for d in lstm_dir.iterdir() if d.is_dir() and (d / 'lstm_model.h5').exists()]
                lstm_model['stocks_trained'] = len(lstm_models)

                if lstm_models:
                    lstm_model['status'] = 'trained'

                    # Get the most recent model
                    latest_lstm = max(lstm_models, key=lambda d: (d / 'lstm_model.h5').stat().st_mtime)
                    last_modified = (latest_lstm / 'lstm_model.h5').stat().st_mtime
                    lstm_model['last_trained'] = datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M')

                    # Try to get accuracy from metadata
                    try:
                        metadata_file = latest_lstm / 'metadata.pkl'
                        if metadata_file.exists():
                            import pickle
                            with open(metadata_file, 'rb') as f:
                                metadata = pickle.load(f)
                                lstm_model['accuracy'] = round(metadata.get('val_loss', 0) * 100, 1)
                    except Exception as e:
                        logger.warning(f"Error loading LSTM metadata: {e}")

            models_data.append(lstm_model)

            # MODEL 3: Kronos (K-line Tokenization)
            kronos_model = {
                'name': 'Kronos',
                'description': 'K-line Tokenization',
                'type': 'kronos',
                'stocks_trained': 0,
                'last_trained': 'Never',
                'status': 'not_trained',
                'accuracy': 0
            }

            # Check if Kronos predictions exist in daily_suggested_stocks
            with self.db_manager.get_session() as session:
                # Count distinct stocks with Kronos predictions
                result = session.execute(text("""
                    SELECT COUNT(DISTINCT symbol), MAX(date), AVG(ml_confidence)
                    FROM daily_suggested_stocks
                    WHERE model_type = 'kronos'
                    AND date >= CURRENT_DATE - INTERVAL '7 days'
                """)).fetchone()

                if result and result[0] > 0:
                    kronos_model['stocks_trained'] = result[0]
                    kronos_model['status'] = 'trained'

                    if result[1]:
                        kronos_model['last_trained'] = result[1].strftime('%Y-%m-%d %H:%M')

                    if result[2]:
                        kronos_model['accuracy'] = round(result[2] * 100, 1)

            models_data.append(kronos_model)

            # Calculate overall stats
            total_stocks = sum(m['stocks_trained'] for m in models_data)
            trained_models = sum(1 for m in models_data if m['status'] == 'trained')

            return {
                'success': True,
                'data': {
                    'models': models_data,
                    'total_stocks_trained': total_stocks,
                    'trained_models_count': trained_models,
                    'total_models_count': 3
                }
            }

        except Exception as e:
            logger.error(f"Error getting ML overview: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Failed to get overview: {str(e)}'
            }

    def get_active_trainings(self, user_id):
        """
        Get active training jobs.
        Returns active trainings list or error.
        """
        try:
            with self.db_manager.get_session() as session:
                active_jobs = session.query(MLTrainingJob).filter(
                    and_(
                        MLTrainingJob.user_id == user_id,
                        MLTrainingJob.status.in_(['running', 'pending'])
                    )
                ).order_by(MLTrainingJob.created_at.desc()).all()

                trainings_data = []
                for job in active_jobs:
                    trainings_data.append({
                        'id': job.id,
                        'symbol': job.symbol,
                        'model_type': job.model_type,
                        'status': job.status,
                        'progress': job.progress,
                        'created_at': job.created_at.isoformat() if job.created_at else None,
                        'started_at': job.created_at.isoformat() if job.created_at else None,
                        'error_message': job.error_message
                    })

                return {
                    'success': True,
                    'data': trainings_data,
                    'count': len(trainings_data)
                }

        except Exception as e:
            logger.error(f"Error getting active trainings: {e}")
            return {
                'success': False,
                'error': f'Failed to get active trainings: {str(e)}'
            }


# Singleton instance
_ml_api_service = None

def get_ml_api_service():
    """Get the singleton ML API service instance"""
    global _ml_api_service
    if _ml_api_service is None:
        _ml_api_service = MLAPIService()
    return _ml_api_service