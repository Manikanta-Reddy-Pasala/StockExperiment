"""
A/B Testing Framework for ML Models
Compare different model versions and strategies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sqlalchemy import text
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


class ABTest:
    """
    A/B test for comparing two model variants.

    Features:
    - Random traffic splitting
    - Statistical significance testing
    - Performance metric comparison
    - Confidence intervals
    - Winner determination
    """

    def __init__(self, test_name: str, variant_a: str, variant_b: str,
                 traffic_split: float = 0.5, min_sample_size: int = 100):
        """
        Initialize A/B test.

        Args:
            test_name: Name of the test
            variant_a: Name of variant A (control)
            variant_b: Name of variant B (treatment)
            traffic_split: Fraction of traffic to variant B (0-1)
            min_sample_size: Minimum samples before statistical testing
        """
        self.test_name = test_name
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.traffic_split = traffic_split
        self.min_sample_size = min_sample_size

        # Results storage
        self.results_a = []
        self.results_b = []

        # Metadata
        self.start_time = datetime.now()
        self.is_active = True

    def assign_variant(self, identifier: str) -> str:
        """
        Assign a variant based on identifier hash.

        Args:
            identifier: Unique identifier (e.g., symbol + timestamp)

        Returns:
            'a' or 'b'
        """
        # Hash identifier for consistent assignment
        hash_value = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
        assignment = (hash_value % 100) / 100.0

        return 'b' if assignment < self.traffic_split else 'a'

    def log_result(self, variant: str, prediction: float, actual: float, metadata: Dict = None):
        """
        Log a result for the test.

        Args:
            variant: 'a' or 'b'
            prediction: Predicted value
            actual: Actual value
            metadata: Additional metadata
        """
        result = {
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction - actual),
            'direction_correct': bool(np.sign(prediction) == np.sign(actual)),
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }

        if variant == 'a':
            self.results_a.append(result)
        else:
            self.results_b.append(result)

    def calculate_metrics(self, variant: str) -> Dict:
        """Calculate metrics for a variant."""
        results = self.results_a if variant == 'a' else self.results_b

        if len(results) < 10:
            return {'error': 'insufficient_data', 'sample_size': len(results)}

        predictions = np.array([r['prediction'] for r in results])
        actuals = np.array([r['actual'] for r in results])

        # Metrics
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)

        direction_accuracy = np.mean([r['direction_correct'] for r in results])

        # R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'sample_size': len(results),
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy),
            'mean_error': float(np.mean([r['error'] for r in results])),
            'median_error': float(np.median([r['error'] for r in results]))
        }

    def statistical_test(self, metric: str = 'mae') -> Dict:
        """
        Perform statistical significance test.

        Args:
            metric: Metric to test ('mae', 'direction_accuracy', etc.)

        Returns:
            Statistical test results
        """
        if len(self.results_a) < self.min_sample_size or len(self.results_b) < self.min_sample_size:
            return {
                'status': 'insufficient_data',
                'sample_size_a': len(self.results_a),
                'sample_size_b': len(self.results_b),
                'min_required': self.min_sample_size
            }

        # Get metric values
        if metric == 'mae':
            values_a = np.array([r['error'] for r in self.results_a])
            values_b = np.array([r['error'] for r in self.results_b])
            better_is = 'lower'
        elif metric == 'direction_accuracy':
            values_a = np.array([float(r['direction_correct']) for r in self.results_a])
            values_b = np.array([float(r['direction_correct']) for r in self.results_b])
            better_is = 'higher'
        else:
            return {'error': f'Unknown metric: {metric}'}

        # Two-sample t-test
        t_statistic, p_value = stats.ttest_ind(values_a, values_b)

        # Mean and confidence intervals
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        std_a = np.std(values_a, ddof=1)
        std_b = np.std(values_b, ddof=1)

        # 95% confidence intervals
        ci_a = stats.t.interval(0.95, len(values_a) - 1,
                               loc=mean_a, scale=std_a / np.sqrt(len(values_a)))
        ci_b = stats.t.interval(0.95, len(values_b) - 1,
                               loc=mean_b, scale=std_b / np.sqrt(len(values_b)))

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2) /
                            (len(values_a) + len(values_b) - 2))
        cohens_d = (mean_b - mean_a) / pooled_std if pooled_std != 0 else 0

        # Determine winner
        is_significant = p_value < 0.05

        if better_is == 'lower':
            winner = 'b' if mean_b < mean_a and is_significant else \
                    'a' if mean_a < mean_b and is_significant else 'tie'
        else:
            winner = 'b' if mean_b > mean_a and is_significant else \
                    'a' if mean_a > mean_b and is_significant else 'tie'

        return {
            'metric': metric,
            'variant_a': {
                'mean': float(mean_a),
                'std': float(std_a),
                'ci_lower': float(ci_a[0]),
                'ci_upper': float(ci_a[1]),
                'sample_size': len(values_a)
            },
            'variant_b': {
                'mean': float(mean_b),
                'std': float(std_b),
                'ci_lower': float(ci_b[0]),
                'ci_upper': float(ci_b[1]),
                'sample_size': len(values_b)
            },
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'is_significant': is_significant,
            'significance_level': 0.05,
            'cohens_d': float(cohens_d),
            'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large',
            'winner': winner,
            'confidence': 95
        }

    def get_summary(self) -> Dict:
        """Get comprehensive test summary."""
        metrics_a = self.calculate_metrics('a')
        metrics_b = self.calculate_metrics('b')

        # Run statistical tests
        mae_test = self.statistical_test('mae')
        accuracy_test = self.statistical_test('direction_accuracy')

        return {
            'test_name': self.test_name,
            'variant_a': self.variant_a,
            'variant_b': self.variant_b,
            'status': 'active' if self.is_active else 'complete',
            'start_time': self.start_time.isoformat(),
            'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'traffic_split': self.traffic_split,
            'metrics': {
                'variant_a': metrics_a,
                'variant_b': metrics_b
            },
            'statistical_tests': {
                'mae': mae_test,
                'direction_accuracy': accuracy_test
            },
            'recommendation': self._get_recommendation(mae_test, accuracy_test)
        }

    def _get_recommendation(self, mae_test: Dict, accuracy_test: Dict) -> Dict:
        """Generate deployment recommendation."""
        if 'status' in mae_test and mae_test['status'] == 'insufficient_data':
            return {
                'action': 'CONTINUE_TEST',
                'reason': 'Insufficient data for statistical significance',
                'confidence': 'LOW'
            }

        mae_winner = mae_test.get('winner', 'tie')
        acc_winner = accuracy_test.get('winner', 'tie')

        # Both tests agree on winner
        if mae_winner == acc_winner and mae_winner != 'tie':
            return {
                'action': 'DEPLOY_VARIANT_' + mae_winner.upper(),
                'reason': f'Variant {mae_winner} wins on both MAE and accuracy with statistical significance',
                'confidence': 'HIGH',
                'winner': mae_winner,
                'mae_improvement': abs(mae_test['variant_b']['mean'] - mae_test['variant_a']['mean']),
                'accuracy_improvement': abs(accuracy_test['variant_b']['mean'] - accuracy_test['variant_a']['mean'])
            }

        # Mixed results
        elif mae_winner != 'tie' or acc_winner != 'tie':
            return {
                'action': 'CONTINUE_TEST',
                'reason': 'Mixed results - MAE and accuracy favor different variants',
                'confidence': 'MEDIUM',
                'mae_winner': mae_winner,
                'accuracy_winner': acc_winner
            }

        # No clear winner
        else:
            return {
                'action': 'KEEP_CONTROL',
                'reason': 'No significant difference detected',
                'confidence': 'MEDIUM'
            }

    def stop_test(self):
        """Mark test as complete."""
        self.is_active = False


class ABTestManager:
    """
    Manages multiple A/B tests.

    Features:
    - Create and run multiple tests
    - Store results in database
    - Retrieve test history
    - Multi-variant testing
    """

    def __init__(self, db_session):
        self.db = db_session
        self.active_tests = {}
        self._init_tables()

    def _init_tables(self):
        """Create A/B testing tables."""
        try:
            self.db.execute(text("""
                CREATE TABLE IF NOT EXISTS ab_tests (
                    id SERIAL PRIMARY KEY,
                    test_name VARCHAR(255) UNIQUE NOT NULL,
                    variant_a VARCHAR(100) NOT NULL,
                    variant_b VARCHAR(100) NOT NULL,
                    traffic_split FLOAT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    status VARCHAR(50) NOT NULL,
                    results JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))

            self.db.execute(text("""
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    id SERIAL PRIMARY KEY,
                    test_name VARCHAR(255) NOT NULL,
                    variant VARCHAR(10) NOT NULL,
                    symbol VARCHAR(20),
                    prediction FLOAT NOT NULL,
                    actual FLOAT,
                    error FLOAT,
                    direction_correct BOOLEAN,
                    metadata JSONB,
                    timestamp TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))

            self.db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_ab_test_results_test
                ON ab_test_results(test_name, timestamp DESC)
            """))

            self.db.commit()
            logger.info("A/B testing tables initialized")

        except Exception as e:
            logger.error(f"Failed to initialize A/B testing tables: {e}")
            self.db.rollback()

    def create_test(self, test_name: str, variant_a: str, variant_b: str,
                   traffic_split: float = 0.5, min_sample_size: int = 100) -> ABTest:
        """
        Create a new A/B test.

        Args:
            test_name: Unique test name
            variant_a: Control variant name
            variant_b: Treatment variant name
            traffic_split: Traffic to variant B (0-1)
            min_sample_size: Min samples before testing

        Returns:
            ABTest instance
        """
        if test_name in self.active_tests:
            logger.warning(f"Test {test_name} already exists")
            return self.active_tests[test_name]

        test = ABTest(test_name, variant_a, variant_b, traffic_split, min_sample_size)
        self.active_tests[test_name] = test

        # Store in database
        try:
            self.db.execute(text("""
                INSERT INTO ab_tests (test_name, variant_a, variant_b, traffic_split, start_time, status)
                VALUES (:test_name, :variant_a, :variant_b, :traffic_split, :start_time, :status)
                ON CONFLICT (test_name) DO NOTHING
            """), {
                'test_name': test_name,
                'variant_a': variant_a,
                'variant_b': variant_b,
                'traffic_split': traffic_split,
                'start_time': test.start_time,
                'status': 'active'
            })
            self.db.commit()
            logger.info(f"Created A/B test: {test_name} ({variant_a} vs {variant_b})")

        except Exception as e:
            logger.error(f"Failed to create test in database: {e}")
            self.db.rollback()

        return test

    def log_result(self, test_name: str, variant: str, symbol: str,
                  prediction: float, actual: float = None, metadata: Dict = None):
        """
        Log a result for a test.

        Args:
            test_name: Name of the test
            variant: 'a' or 'b'
            symbol: Stock symbol
            prediction: Predicted value
            actual: Actual value (if known)
            metadata: Additional metadata
        """
        if test_name not in self.active_tests:
            logger.warning(f"Test {test_name} not found")
            return

        test = self.active_tests[test_name]

        if actual is not None:
            test.log_result(variant, prediction, actual, metadata)

        # Store in database
        try:
            self.db.execute(text("""
                INSERT INTO ab_test_results
                (test_name, variant, symbol, prediction, actual, error, direction_correct, metadata, timestamp)
                VALUES (:test_name, :variant, :symbol, :prediction, :actual, :error, :direction_correct, :metadata, :timestamp)
            """), {
                'test_name': test_name,
                'variant': variant,
                'symbol': symbol,
                'prediction': prediction,
                'actual': actual,
                'error': abs(prediction - actual) if actual is not None else None,
                'direction_correct': bool(np.sign(prediction) == np.sign(actual)) if actual is not None else None,
                'metadata': json.dumps(metadata or {}),
                'timestamp': datetime.now()
            })
            self.db.commit()

        except Exception as e:
            logger.error(f"Failed to log test result: {e}")
            self.db.rollback()

    def get_test_summary(self, test_name: str) -> Dict:
        """Get summary for a test."""
        if test_name not in self.active_tests:
            return {'error': 'Test not found'}

        return self.active_tests[test_name].get_summary()

    def stop_test(self, test_name: str, winner: str = None):
        """
        Stop a test and optionally declare a winner.

        Args:
            test_name: Name of the test
            winner: Optional winner ('a' or 'b')
        """
        if test_name not in self.active_tests:
            logger.warning(f"Test {test_name} not found")
            return

        test = self.active_tests[test_name]
        test.stop_test()

        summary = test.get_summary()

        # Update database
        try:
            self.db.execute(text("""
                UPDATE ab_tests
                SET status = 'complete',
                    end_time = :end_time,
                    results = :results
                WHERE test_name = :test_name
            """), {
                'test_name': test_name,
                'end_time': datetime.now(),
                'results': json.dumps(summary)
            })
            self.db.commit()
            logger.info(f"Stopped test: {test_name}")

        except Exception as e:
            logger.error(f"Failed to stop test in database: {e}")
            self.db.rollback()

    def get_active_tests(self) -> List[str]:
        """Get list of active test names."""
        return [name for name, test in self.active_tests.items() if test.is_active]

    def get_test_history(self, days: int = 30) -> pd.DataFrame:
        """Get historical test data."""
        try:
            query = text("""
                SELECT test_name, variant_a, variant_b, start_time, end_time, status, results
                FROM ab_tests
                WHERE start_time >= :start_date
                ORDER BY start_time DESC
            """)

            result = self.db.execute(query, {
                'start_date': datetime.now() - timedelta(days=days)
            })

            data = []
            for row in result.fetchall():
                data.append({
                    'test_name': row[0],
                    'variant_a': row[1],
                    'variant_b': row[2],
                    'start_time': row[3],
                    'end_time': row[4],
                    'status': row[5],
                    'results': row[6]
                })

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Failed to retrieve test history: {e}")
            return pd.DataFrame()
