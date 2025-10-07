"""
Calibrated Probability Scoring System
Replaces simple sigmoid with properly calibrated probabilities
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class CalibratedScorer:
    """
    Converts regression predictions to calibrated probabilities.

    Instead of using sigmoid: 1 / (1 + exp(-x/10))
    We train a calibration model on historical prediction performance.
    """

    def __init__(self):
        self.calibration_model = None
        self.price_threshold = 0.02  # 2% move threshold
        self.is_fitted = False

    def fit(self, predictions, actuals):
        """
        Fit calibration model on historical predictions vs actuals.

        Args:
            predictions: Array of predicted price changes (%)
            actuals: Array of actual price changes (%)
        """
        if len(predictions) < 50:
            logger.warning("Not enough data for calibration, need at least 50 samples")
            return False

        try:
            # Convert to binary classification problem
            # 1 = price increased by threshold, 0 = did not
            y_binary = (actuals >= self.price_threshold).astype(int)

            # Reshape predictions for sklearn
            X = np.array(predictions).reshape(-1, 1)

            # Split for calibration
            X_train, X_cal, y_train, y_cal = train_test_split(
                X, y_binary, test_size=0.3, random_state=42
            )

            # Train base classifier
            base_clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            base_clf.fit(X_train, y_train)

            # Calibrate probabilities using isotonic regression
            self.calibration_model = CalibratedClassifierCV(
                base_clf,
                method='isotonic',
                cv='prefit'
            )
            self.calibration_model.fit(X_cal, y_cal)

            self.is_fitted = True
            logger.info(f"Calibration model fitted on {len(predictions)} samples")
            return True

        except Exception as e:
            logger.error(f"Calibration fitting failed: {e}")
            return False

    def score(self, prediction):
        """
        Get calibrated probability score for a prediction.

        Args:
            prediction: Predicted price change (%)

        Returns:
            Calibrated probability between 0 and 1
        """
        if not self.is_fitted:
            # Fallback to sigmoid if not calibrated
            return self._sigmoid_score(prediction)

        try:
            X = np.array([[prediction]])
            prob = self.calibration_model.predict_proba(X)[0][1]
            return float(prob)
        except Exception as e:
            logger.error(f"Calibrated scoring failed: {e}")
            return self._sigmoid_score(prediction)

    def batch_score(self, predictions):
        """
        Score multiple predictions at once.

        Args:
            predictions: Array of predicted price changes

        Returns:
            Array of calibrated probabilities
        """
        if not self.is_fitted:
            return np.array([self._sigmoid_score(p) for p in predictions])

        try:
            X = np.array(predictions).reshape(-1, 1)
            probs = self.calibration_model.predict_proba(X)[:, 1]
            return probs
        except Exception as e:
            logger.error(f"Batch calibrated scoring failed: {e}")
            return np.array([self._sigmoid_score(p) for p in predictions])

    def _sigmoid_score(self, prediction):
        """Fallback sigmoid scoring."""
        return 1 / (1 + np.exp(-prediction / 10))

    def get_calibration_curve(self, predictions, actuals, n_bins=10):
        """
        Generate calibration curve data for visualization.

        Returns:
            dict with fraction_of_positives, mean_predicted_value
        """
        from sklearn.calibration import calibration_curve

        y_binary = (actuals >= self.price_threshold).astype(int)
        scores = self.batch_score(predictions)

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, scores, n_bins=n_bins, strategy='quantile'
        )

        return {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist(),
            'bins': n_bins
        }


class AdaptiveScorer:
    """
    Adaptive scoring that adjusts based on recent model performance.

    Maintains a sliding window of recent predictions and adjusts
    confidence scores based on recent accuracy.
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.recent_predictions = []
        self.recent_actuals = []
        self.accuracy_score = 0.5  # Start at 50%

    def add_result(self, prediction, actual):
        """Add a prediction-actual pair to the window."""
        self.recent_predictions.append(prediction)
        self.recent_actuals.append(actual)

        # Keep only recent window
        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions.pop(0)
            self.recent_actuals.pop(0)

        # Update accuracy
        self._update_accuracy()

    def _update_accuracy(self):
        """Update accuracy score based on recent performance."""
        if len(self.recent_predictions) < 10:
            return

        # Calculate how often prediction direction was correct
        pred_directions = np.sign(self.recent_predictions)
        actual_directions = np.sign(self.recent_actuals)

        correct = np.sum(pred_directions == actual_directions)
        self.accuracy_score = correct / len(self.recent_predictions)

    def adjust_score(self, base_score):
        """
        Adjust a base score based on recent accuracy.

        If model has been performing well, boost confidence.
        If model has been performing poorly, reduce confidence.
        """
        # Map accuracy (0-1) to adjustment factor (0.7-1.3)
        adjustment = 0.7 + (self.accuracy_score * 0.6)

        # Adjust and clamp to [0, 1]
        adjusted = base_score * adjustment
        return np.clip(adjusted, 0, 1)

    def get_confidence_level(self):
        """Get current confidence level."""
        if self.accuracy_score > 0.65:
            return "HIGH"
        elif self.accuracy_score > 0.55:
            return "MEDIUM"
        else:
            return "LOW"

    def get_statistics(self):
        """Get recent performance statistics."""
        if len(self.recent_predictions) < 10:
            return {
                'sample_size': len(self.recent_predictions),
                'accuracy': None,
                'confidence_level': 'INSUFFICIENT_DATA'
            }

        pred_array = np.array(self.recent_predictions)
        actual_array = np.array(self.recent_actuals)

        # Direction accuracy
        pred_directions = np.sign(pred_array)
        actual_directions = np.sign(actual_array)
        direction_accuracy = np.mean(pred_directions == actual_directions)

        # Mean absolute error
        mae = np.mean(np.abs(pred_array - actual_array))

        # Prediction bias (are we over/under predicting?)
        bias = np.mean(pred_array - actual_array)

        return {
            'sample_size': len(self.recent_predictions),
            'direction_accuracy': float(direction_accuracy),
            'mean_absolute_error': float(mae),
            'prediction_bias': float(bias),
            'confidence_level': self.get_confidence_level(),
            'window_size': self.window_size
        }


class MultiModelScorer:
    """
    Combines scores from multiple models with dynamic weighting.

    Weights are adjusted based on each model's recent performance.
    """

    def __init__(self, model_names):
        self.model_names = model_names
        self.scorers = {name: AdaptiveScorer() for name in model_names}
        self.base_weights = {name: 1.0 / len(model_names) for name in model_names}

    def add_results(self, predictions_dict, actual):
        """
        Add results from all models.

        Args:
            predictions_dict: {model_name: prediction}
            actual: Actual outcome
        """
        for model_name, prediction in predictions_dict.items():
            if model_name in self.scorers:
                self.scorers[model_name].add_result(prediction, actual)

    def get_weighted_score(self, predictions_dict, base_scores_dict):
        """
        Get weighted score across all models.

        Args:
            predictions_dict: {model_name: prediction}
            base_scores_dict: {model_name: base_score}

        Returns:
            Combined weighted score
        """
        adjusted_scores = {}
        weights = {}

        for model_name in self.model_names:
            if model_name not in predictions_dict:
                continue

            # Get adaptive adjustment
            scorer = self.scorers[model_name]
            base_score = base_scores_dict.get(model_name, 0.5)
            adjusted_score = scorer.adjust_score(base_score)

            # Weight by recent accuracy
            weight = self.base_weights[model_name] * scorer.accuracy_score

            adjusted_scores[model_name] = adjusted_score
            weights[model_name] = weight

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.5

        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Weighted average
        final_score = sum(
            adjusted_scores[k] * normalized_weights[k]
            for k in adjusted_scores
        )

        return final_score

    def get_model_statistics(self):
        """Get statistics for all models."""
        return {
            model_name: scorer.get_statistics()
            for model_name, scorer in self.scorers.items()
        }
