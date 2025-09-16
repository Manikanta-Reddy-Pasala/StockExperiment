"""
ML API Service - Handles ML operations business logic
Separates business logic from web routes following SOA principles
"""
import logging
from datetime import datetime, timedelta
import threading
from sqlalchemy import func, and_

from ..models.database import get_database_manager
from ..models.models import MLTrainingJob, MLTrainedModel
from .unified_broker_service import get_unified_broker_service

logger = logging.getLogger(__name__)

class MLAPIService:
    """Service class for ML API operations"""

    def __init__(self):
        self.db_manager = get_database_manager()
        self.unified_broker_service = get_unified_broker_service()

    def get_current_price(self, user_id, symbol):
        """
        Get current price for a symbol using the unified broker service.
        Returns None if price cannot be fetched.
        No hardcoded fallback values.
        """
        try:
            logger.info(f"Fetching current price for {symbol} for user {user_id}")

            # Use unified broker service to get current price
            quotes_result = self.unified_broker_service.get_quotes(user_id, [symbol])

            if quotes_result and quotes_result.get('status') == 'success':
                quotes_data = quotes_result.get('data', {})
                if symbol in quotes_data:
                    current_price = float(quotes_data[symbol].get('ltp', 0))
                    if current_price > 0:
                        logger.info(f"Successfully fetched current price for {symbol}: ₹{current_price}")
                        return current_price
                    else:
                        logger.warning(f"Received invalid price for {symbol}: {current_price}")
                        return None
                else:
                    logger.warning(f"Symbol {symbol} not found in quotes data")
                    return None
            else:
                logger.warning(f"Failed to get quotes for {symbol}: {quotes_result}")
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
            prediction_result = get_prediction(symbol, days_ahead=horizon)

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
                try:
                    from .ml.training_service import train_and_tune_models
                    result = train_and_tune_models(symbol, start_date, end_date, job_id)

                    # Update job status
                    with self.db_manager.get_session() as session:
                        job = session.query(MLTrainingJob).filter(MLTrainingJob.id == job_id).first()
                        if job:
                            if result:
                                job.status = 'completed'
                                job.progress = 100.0
                                job.completed_at = datetime.utcnow()
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

                return {
                    'success': True,
                    'job_id': job_id,
                    'symbol': job.symbol,
                    'status': job.status,
                    'progress': job.progress,
                    'created_at': job.created_at.isoformat() if job.created_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'error_message': job.error_message
                }

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
        Get ML dashboard overview statistics.
        Returns overview data or error.
        """
        try:
            with self.db_manager.get_session() as session:
                # Count trained models
                trained_count = session.query(MLTrainedModel).filter(
                    MLTrainedModel.user_id == user_id
                ).count()

                # Get training success rate
                total_jobs = session.query(MLTrainingJob).filter(
                    MLTrainingJob.user_id == user_id
                ).count()

                successful_jobs = session.query(MLTrainingJob).filter(
                    and_(
                        MLTrainingJob.user_id == user_id,
                        MLTrainingJob.status == 'completed'
                    )
                ).count()

                success_rate = (successful_jobs / total_jobs * 100) if total_jobs > 0 else 0

                # Get last updated date
                latest_model = session.query(MLTrainedModel).filter(
                    MLTrainedModel.user_id == user_id
                ).order_by(MLTrainedModel.created_at.desc()).first()

                last_updated = latest_model.created_at.strftime('%Y-%m-%d') if latest_model else 'Never'

                return {
                    'success': True,
                    'data': {
                        'trained_stocks': trained_count,
                        'success_rate': round(success_rate, 1),
                        'last_updated': last_updated
                    }
                }

        except Exception as e:
            logger.error(f"Error getting ML overview: {e}")
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