"""
Model Monitoring System
Tracks model performance, detects drift, and generates alerts
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
from sqlalchemy import text, Table, MetaData, Column, Integer, String, Float, DateTime, JSON
import logging
import json

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Monitors ML model performance and data drift over time.

    Features:
    - Performance tracking (accuracy, MAE, direction accuracy)
    - Feature drift detection (distribution changes)
    - Prediction distribution monitoring
    - Alert generation on anomalies
    - Historical performance comparison
    """

    def __init__(self, db_session, window_size=100):
        """
        Initialize model monitor.

        Args:
            db_session: Database session
            window_size: Number of recent predictions to track
        """
        self.db = db_session
        self.window_size = window_size

        # Recent predictions buffer
        self.recent_predictions = deque(maxlen=window_size)
        self.recent_actuals = deque(maxlen=window_size)
        self.recent_features = deque(maxlen=window_size)

        # Performance baselines (set during training)
        self.baseline_metrics = {}

        # Alerts
        self.alert_thresholds = {
            'accuracy_drop': 0.1,  # Alert if accuracy drops by 10%
            'mae_increase': 0.5,   # Alert if MAE increases by 50%
            'drift_score': 0.3     # Alert if drift score > 0.3
        }

        # Initialize monitoring table
        self._init_monitoring_table()

    def _init_monitoring_table(self):
        """Create monitoring table if it doesn't exist."""
        try:
            self.db.execute(text("""
                CREATE TABLE IF NOT EXISTS ml_model_monitoring (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))

            self.db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_monitoring_timestamp
                ON ml_model_monitoring(timestamp DESC)
            """))

            self.db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_monitoring_model_type
                ON ml_model_monitoring(model_type, metric_name)
            """))

            self.db.commit()
            logger.info("Model monitoring table initialized")

        except Exception as e:
            logger.error(f"Failed to initialize monitoring table: {e}")
            self.db.rollback()

    def set_baseline(self, model_type: str, metrics: Dict):
        """
        Set baseline metrics from training.

        Args:
            model_type: 'enhanced', 'advanced', etc.
            metrics: Dict of metric_name -> value
        """
        self.baseline_metrics[model_type] = {
            'metrics': metrics,
            'set_at': datetime.now()
        }

        # Store in database
        try:
            for metric_name, metric_value in metrics.items():
                self.db.execute(text("""
                    INSERT INTO ml_model_monitoring
                    (timestamp, model_type, metric_name, metric_value, metadata)
                    VALUES (:timestamp, :model_type, :metric_name, :metric_value, :metadata)
                """), {
                    'timestamp': datetime.now(),
                    'model_type': model_type,
                    'metric_name': f'baseline_{metric_name}',
                    'metric_value': float(metric_value),
                    'metadata': json.dumps({'type': 'baseline'})
                })

            self.db.commit()
            logger.info(f"Baseline metrics set for {model_type}: {metrics}")

        except Exception as e:
            logger.error(f"Failed to store baseline metrics: {e}")
            self.db.rollback()

    def log_prediction(self, model_type: str, prediction: float,
                      features: Dict, metadata: Dict = None):
        """
        Log a prediction for monitoring.

        Args:
            model_type: Model identifier
            prediction: Predicted value
            features: Input features dict
            metadata: Additional metadata (symbol, timestamp, etc.)
        """
        self.recent_predictions.append({
            'model_type': model_type,
            'prediction': prediction,
            'features': features,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        })

    def log_actual(self, prediction_id: int, actual: float):
        """
        Log actual outcome for a prediction.

        Args:
            prediction_id: ID of the prediction
            actual: Actual value
        """
        self.recent_actuals.append({
            'prediction_id': prediction_id,
            'actual': actual,
            'timestamp': datetime.now()
        })

    def add_result(self, prediction: float, actual: float, features: Dict = None):
        """
        Add a prediction-actual pair for monitoring.

        Args:
            prediction: Predicted value
            actual: Actual value
            features: Feature dict (optional)
        """
        self.recent_predictions.append(prediction)
        self.recent_actuals.append(actual)

        if features is not None:
            self.recent_features.append(features)

    def calculate_metrics(self, model_type: str = None) -> Dict:
        """
        Calculate current performance metrics.

        Returns:
            Dict with current metrics
        """
        if len(self.recent_predictions) < 10:
            return {
                'error': 'Insufficient data',
                'sample_size': len(self.recent_predictions)
            }

        predictions = np.array([p if isinstance(p, (int, float)) else p['prediction']
                               for p in self.recent_predictions])
        actuals = np.array([a if isinstance(a, (int, float)) else a['actual']
                           for a in self.recent_actuals[-len(predictions):]])

        # Basic metrics
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)

        # Direction accuracy
        pred_directions = np.sign(predictions)
        actual_directions = np.sign(actuals)
        direction_accuracy = np.mean(pred_directions == actual_directions)

        # R² score
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        metrics = {
            'sample_size': len(predictions),
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy),
            'mean_prediction': float(np.mean(predictions)),
            'mean_actual': float(np.mean(actuals)),
            'prediction_bias': float(np.mean(predictions - actuals)),
            'calculated_at': datetime.now().isoformat()
        }

        # Log to database
        if model_type:
            self._log_metrics_to_db(model_type, metrics)

        return metrics

    def _log_metrics_to_db(self, model_type: str, metrics: Dict):
        """Log metrics to database."""
        try:
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.db.execute(text("""
                        INSERT INTO ml_model_monitoring
                        (timestamp, model_type, metric_name, metric_value)
                        VALUES (:timestamp, :model_type, :metric_name, :metric_value)
                    """), {
                        'timestamp': datetime.now(),
                        'model_type': model_type,
                        'metric_name': metric_name,
                        'metric_value': float(metric_value)
                    })

            self.db.commit()

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            self.db.rollback()

    def detect_performance_degradation(self, model_type: str) -> Dict:
        """
        Detect if model performance has degraded.

        Returns:
            Dict with degradation status and details
        """
        if model_type not in self.baseline_metrics:
            return {'status': 'no_baseline', 'alerts': []}

        current_metrics = self.calculate_metrics(model_type)

        if 'error' in current_metrics:
            return {'status': 'insufficient_data', 'alerts': []}

        baseline = self.baseline_metrics[model_type]['metrics']
        alerts = []

        # Check accuracy drop
        if 'direction_accuracy' in baseline and 'direction_accuracy' in current_metrics:
            baseline_acc = baseline['direction_accuracy']
            current_acc = current_metrics['direction_accuracy']
            accuracy_drop = baseline_acc - current_acc

            if accuracy_drop > self.alert_thresholds['accuracy_drop']:
                alerts.append({
                    'type': 'ACCURACY_DROP',
                    'severity': 'HIGH',
                    'message': f"Accuracy dropped by {accuracy_drop:.1%} (baseline: {baseline_acc:.1%}, current: {current_acc:.1%})",
                    'baseline_value': baseline_acc,
                    'current_value': current_acc
                })

        # Check MAE increase
        if 'mae' in baseline and 'mae' in current_metrics:
            baseline_mae = baseline['mae']
            current_mae = current_metrics['mae']
            mae_increase = (current_mae - baseline_mae) / baseline_mae if baseline_mae != 0 else 0

            if mae_increase > self.alert_thresholds['mae_increase']:
                alerts.append({
                    'type': 'MAE_INCREASE',
                    'severity': 'MEDIUM',
                    'message': f"MAE increased by {mae_increase:.1%} (baseline: {baseline_mae:.3f}, current: {current_mae:.3f})",
                    'baseline_value': baseline_mae,
                    'current_value': current_mae
                })

        # Check R² drop
        if 'r2' in baseline and 'r2' in current_metrics:
            baseline_r2 = baseline['r2']
            current_r2 = current_metrics['r2']
            r2_drop = baseline_r2 - current_r2

            if r2_drop > 0.1:  # R² dropped by more than 0.1
                alerts.append({
                    'type': 'R2_DROP',
                    'severity': 'HIGH',
                    'message': f"R² dropped by {r2_drop:.3f} (baseline: {baseline_r2:.3f}, current: {current_r2:.3f})",
                    'baseline_value': baseline_r2,
                    'current_value': current_r2
                })

        return {
            'status': 'degraded' if alerts else 'healthy',
            'alerts': alerts,
            'current_metrics': current_metrics,
            'baseline_metrics': baseline,
            'checked_at': datetime.now().isoformat()
        }

    def detect_feature_drift(self, current_features: Dict, baseline_features: Dict = None) -> Dict:
        """
        Detect distribution drift in features.

        Args:
            current_features: Current feature values
            baseline_features: Baseline feature distribution (optional)

        Returns:
            Drift detection results
        """
        if not self.recent_features or len(self.recent_features) < 30:
            return {
                'status': 'insufficient_data',
                'drift_score': 0,
                'drifted_features': []
            }

        try:
            # Convert recent features to DataFrame
            recent_df = pd.DataFrame(list(self.recent_features))

            drifted_features = []

            for feature in current_features.keys():
                if feature not in recent_df.columns:
                    continue

                current_value = current_features[feature]
                historical_values = recent_df[feature].dropna()

                if len(historical_values) < 10:
                    continue

                # Calculate z-score
                mean = historical_values.mean()
                std = historical_values.std()

                if std == 0:
                    continue

                z_score = abs((current_value - mean) / std)

                # Flag if z-score > 3 (3 standard deviations)
                if z_score > 3:
                    drifted_features.append({
                        'feature': feature,
                        'current_value': float(current_value),
                        'historical_mean': float(mean),
                        'historical_std': float(std),
                        'z_score': float(z_score)
                    })

            # Calculate overall drift score
            drift_score = len(drifted_features) / len(current_features) if current_features else 0

            status = 'high_drift' if drift_score > self.alert_thresholds['drift_score'] else 'normal'

            return {
                'status': status,
                'drift_score': drift_score,
                'drifted_features': drifted_features,
                'total_features': len(current_features),
                'checked_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Feature drift detection failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def get_performance_history(self, model_type: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical performance metrics.

        Args:
            model_type: Model identifier
            days: Number of days to retrieve

        Returns:
            DataFrame with historical metrics
        """
        try:
            query = text("""
                SELECT timestamp, metric_name, metric_value, metadata
                FROM ml_model_monitoring
                WHERE model_type = :model_type
                AND timestamp >= :start_date
                ORDER BY timestamp DESC
            """)

            result = self.db.execute(query, {
                'model_type': model_type,
                'start_date': datetime.now() - timedelta(days=days)
            })

            data = []
            for row in result.fetchall():
                data.append({
                    'timestamp': row[0],
                    'metric_name': row[1],
                    'metric_value': row[2],
                    'metadata': row[3]
                })

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Failed to retrieve performance history: {e}")
            return pd.DataFrame()

    def generate_health_report(self, model_type: str) -> Dict:
        """
        Generate comprehensive health report.

        Returns:
            Health report with all monitoring data
        """
        current_metrics = self.calculate_metrics(model_type)
        degradation = self.detect_performance_degradation(model_type)

        # Recent feature drift
        drift_results = {'status': 'no_recent_features'}
        if self.recent_features and len(self.recent_features) > 0:
            drift_results = self.detect_feature_drift(self.recent_features[-1])

        # Overall health score
        health_score = 100

        # Deduct for degradation
        if degradation['status'] == 'degraded':
            health_score -= len(degradation['alerts']) * 15

        # Deduct for drift
        if drift_results['status'] == 'high_drift':
            health_score -= 20

        health_score = max(0, health_score)

        # Determine health level
        if health_score >= 80:
            health_level = 'EXCELLENT'
        elif health_score >= 60:
            health_level = 'GOOD'
        elif health_score >= 40:
            health_level = 'FAIR'
        else:
            health_level = 'POOR'

        return {
            'model_type': model_type,
            'health_score': health_score,
            'health_level': health_level,
            'current_metrics': current_metrics,
            'performance_degradation': degradation,
            'feature_drift': drift_results,
            'baseline_set': model_type in self.baseline_metrics,
            'monitored_predictions': len(self.recent_predictions),
            'generated_at': datetime.now().isoformat()
        }

    def should_retrain(self, model_type: str) -> Tuple[bool, List[str]]:
        """
        Determine if model should be retrained.

        Returns:
            (should_retrain: bool, reasons: List[str])
        """
        report = self.generate_health_report(model_type)

        reasons = []

        # Check health score
        if report['health_score'] < 60:
            reasons.append(f"Low health score: {report['health_score']}")

        # Check for critical alerts
        if report['performance_degradation']['status'] == 'degraded':
            for alert in report['performance_degradation']['alerts']:
                if alert['severity'] == 'HIGH':
                    reasons.append(f"Critical: {alert['message']}")

        # Check drift
        if report['feature_drift']['status'] == 'high_drift':
            reasons.append(f"High feature drift detected: {report['feature_drift']['drift_score']:.1%}")

        should_retrain = len(reasons) > 0

        return should_retrain, reasons
