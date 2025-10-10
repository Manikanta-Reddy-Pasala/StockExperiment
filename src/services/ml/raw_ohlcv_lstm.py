"""
Simplified Raw OHLCV LSTM Model
Based on Korean Stock Prediction Research (2024)

Key Findings from Research:
- Simple LSTM with raw OHLCV data MATCHES performance of complex XGBoost with 100+ technical indicators
- LSTM F1 Score: 0.4312 vs XGBoost F1: 0.4316
- Optimal hyperparameters: hidden_size=8, window_length=100, just close price or full OHLCV

Philosophy:
- "Don't Overengineer" - Start with raw data, let the model learn representations
- "Information Preservation" - No filtering/transforming means no signal loss
- "Model Simplicity Wins" - Smaller models with proper tuning outperform larger ones

This implementation follows the paper's methodology exactly for reproducibility.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
import logging
from typing import Tuple, Dict, Optional
import pickle
from pathlib import Path

from .triple_barrier_labeling import TripleBarrierLabeler

logger = logging.getLogger(__name__)


class RawOHLCVLSTM:
    """
    Simplified LSTM model trained on raw OHLCV data using Triple Barrier Labeling.

    Based on research showing that simple LSTMs on raw data can match or exceed
    performance of complex feature-engineered models.

    Architecture:
    - Single LSTM layer with small hidden size (8 units by default)
    - Minimal dropout for regularization
    - Uses rolling windows of raw OHLCV data
    - Triple barrier labeling for realistic trading targets

    Parameters:
    -----------
    hidden_size : int
        Number of LSTM units (paper optimal: 8)
    window_length : int
        Number of past days to use as input (paper optimal: 100)
    use_full_ohlcv : bool
        If True, uses all 5 OHLCV features. If False, only uses close price
    dropout : float
        Dropout rate for regularization (default: 0.2)
    """

    def __init__(self,
                 hidden_size: int = 8,
                 window_length: int = 100,
                 use_full_ohlcv: bool = True,
                 dropout: float = 0.2):

        self.hidden_size = hidden_size
        self.window_length = window_length
        self.use_full_ohlcv = use_full_ohlcv
        self.dropout = dropout

        # Feature columns based on configuration
        if use_full_ohlcv:
            self.feature_cols = ['open', 'high', 'low', 'close', 'volume']
            self.n_features = 5
        else:
            self.feature_cols = ['close']
            self.n_features = 1

        # Model and preprocessing components
        self.model = None
        self.scaler = MinMaxScaler()
        self.label_encoder = None  # For converting labels if needed

        # Triple Barrier labeler
        self.labeler = TripleBarrierLabeler(
            upper_barrier=9.0,   # 9% profit target (from paper)
            lower_barrier=9.0,   # 9% stop loss (from paper)
            time_horizon=29      # 29 day horizon (from paper)
        )

        # Training history
        self.history = None
        self.evaluation_metrics = {}

        logger.info(f"RawOHLCVLSTM initialized: hidden_size={hidden_size}, "
                   f"window={window_length}, features={self.feature_cols}")

    def _build_model(self) -> Sequential:
        """
        Build the simplified LSTM architecture from the research paper.

        Architecture matches paper's optimal configuration:
        - Single LSTM layer with small hidden size
        - Dropout for regularization
        - Dense output layer for 3-class classification
        """
        model = Sequential([
            # Single LSTM layer (paper finding: simple is better)
            LSTM(
                units=self.hidden_size,
                input_shape=(self.window_length, self.n_features),
                return_sequences=False  # Only return final output
            ),

            # Dropout for regularization
            Dropout(self.dropout),

            # Output layer for 3-class classification
            # 0 = Loss, 1 = Neutral, 2 = Profit
            Dense(3, activation='softmax')
        ])

        # Compile with categorical crossentropy for multi-class
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"Model built with {model.count_params():,} parameters")
        return model

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare raw OHLCV data for LSTM training.

        Steps:
        1. Apply triple barrier labeling
        2. Extract OHLCV features
        3. Scale features to [0, 1]
        4. Create rolling windows
        5. Align labels with windows

        Parameters:
        -----------
        df : pd.DataFrame
            Raw OHLCV data with columns: open, high, low, close, volume

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (X, y) where X is windowed features and y is labels
        """
        logger.info(f"Preparing data from {len(df)} samples...")

        # Ensure column names are lowercase
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Step 1: Apply triple barrier labeling
        df_labeled, labels = self.labeler.apply_for_ml(df, price_col='close', return_multiclass=True)

        if len(df_labeled) < self.window_length + 100:
            raise ValueError(f"Insufficient data: need at least {self.window_length + 100} samples, "
                           f"got {len(df_labeled)}")

        # Step 2: Extract OHLCV features
        features = df_labeled[self.feature_cols].values

        # Step 3: Scale features to [0, 1]
        features_scaled = self.scaler.fit_transform(features)

        # Step 4: Create rolling windows
        X, y = self._create_windows(features_scaled, labels)

        logger.info(f"Created {len(X)} windowed samples with shape {X.shape}")
        logger.info(f"Label distribution: {np.bincount(y.astype(int))}")

        return X, y

    def _create_windows(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create rolling windows from time series data.

        For each window, we use the past window_length days as input (X)
        and the label at the end of the window as target (y).

        Parameters:
        -----------
        features : np.ndarray
            Scaled OHLCV features (n_samples, n_features)
        labels : np.ndarray
            Triple barrier labels (n_samples,)

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (X, y) where X has shape (n_windows, window_length, n_features)
        """
        X_windows = []
        y_windows = []

        # Create sliding windows
        for i in range(len(features) - self.window_length):
            # Input: past window_length days
            X_windows.append(features[i:i + self.window_length])

            # Target: label at the end of window
            y_windows.append(labels[i + self.window_length])

        return np.array(X_windows), np.array(y_windows)

    def train(self, df: pd.DataFrame,
              test_size: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.1,
              early_stopping_patience: int = 15,
              verbose: int = 1) -> Dict:
        """
        Train the LSTM model on raw OHLCV data.

        Parameters:
        -----------
        df : pd.DataFrame
            Raw OHLCV data
        test_size : float
            Proportion of data to use for testing
        epochs : int
            Maximum training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Proportion of training data for validation
        early_stopping_patience : int
            Patience for early stopping
        verbose : int
            Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)

        Returns:
        --------
        Dict
            Training results and evaluation metrics
        """
        logger.info("Starting training on raw OHLCV data...")

        # Prepare data
        X, y = self.prepare_data(df)

        # Time-series split (no shuffling!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Build model
        self.model = self._build_model()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Train model
        logger.info("Training LSTM model...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )

        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1_macro = f1_score(y_test, y_pred_classes, average='macro')
        f1_weighted = f1_score(y_test, y_pred_classes, average='weighted')

        self.evaluation_metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs_trained': len(self.history.history['loss'])
        }

        logger.info(f"✓ Training complete!")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Macro: {f1_macro:.4f}")
        logger.info(f"  F1 Weighted: {f1_weighted:.4f}")

        # Detailed classification report
        logger.info("\nClassification Report:")
        # Get unique classes in y_test
        unique_classes = np.unique(y_test)
        target_names = ['Loss', 'Neutral', 'Profit']
        # Only use target names for classes that exist
        valid_target_names = [target_names[int(c)] for c in unique_classes if c < len(target_names)]

        report = classification_report(
            y_test, y_pred_classes,
            labels=unique_classes,
            target_names=valid_target_names if len(valid_target_names) == len(unique_classes) else None,
            digits=4,
            zero_division=0
        )
        logger.info(f"\n{report}")

        return self.evaluation_metrics

    def predict(self, df: pd.DataFrame, return_probabilities: bool = False) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters:
        -----------
        df : pd.DataFrame
            Raw OHLCV data (must have at least window_length samples)
        return_probabilities : bool
            If True, return class probabilities. If False, return class predictions.

        Returns:
        --------
        np.ndarray
            Predictions or probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Prepare data (without labels)
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Extract features
        features = df[self.feature_cols].values

        # Scale using fitted scaler
        features_scaled = self.scaler.transform(features)

        # Create windows
        X_windows = []
        for i in range(len(features_scaled) - self.window_length):
            X_windows.append(features_scaled[i:i + self.window_length])

        X = np.array(X_windows)

        # Predict
        probabilities = self.model.predict(X, verbose=0)

        if return_probabilities:
            return probabilities
        else:
            return np.argmax(probabilities, axis=1)

    def save(self, model_dir: str = 'ml_models/raw_ohlcv_lstm'):
        """
        Save the trained model and preprocessing components.

        Parameters:
        -----------
        model_dir : str
            Directory to save model files
        """
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {model_path}...")

        # Save Keras model
        self.model.save(model_path / 'lstm_model.h5')

        # Save scaler
        with open(model_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save configuration and metadata
        metadata = {
            'hidden_size': self.hidden_size,
            'window_length': self.window_length,
            'use_full_ohlcv': self.use_full_ohlcv,
            'dropout': self.dropout,
            'feature_cols': self.feature_cols,
            'n_features': self.n_features,
            'evaluation_metrics': self.evaluation_metrics,
            'barrier_params': {
                'upper': self.labeler.upper_barrier * 100,
                'lower': self.labeler.lower_barrier * 100,
                'horizon': self.labeler.time_horizon
            }
        }

        with open(model_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"✓ Model saved to {model_path}")

    def load(self, model_dir: str = 'ml_models/raw_ohlcv_lstm'):
        """
        Load a trained model from disk.

        Parameters:
        -----------
        model_dir : str
            Directory containing saved model files
        """
        model_path = Path(model_dir)

        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        logger.info(f"Loading model from {model_path}...")

        # Load Keras model
        self.model = keras.models.load_model(model_path / 'lstm_model.h5')

        # Load scaler
        with open(model_path / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        # Load metadata
        with open(model_path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        self.hidden_size = metadata['hidden_size']
        self.window_length = metadata['window_length']
        self.use_full_ohlcv = metadata['use_full_ohlcv']
        self.dropout = metadata['dropout']
        self.feature_cols = metadata['feature_cols']
        self.n_features = metadata['n_features']
        self.evaluation_metrics = metadata.get('evaluation_metrics', {})

        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Configuration: hidden_size={self.hidden_size}, window={self.window_length}")

        if self.evaluation_metrics:
            logger.info(f"  Performance: F1={self.evaluation_metrics.get('f1_macro', 0):.4f}, "
                       f"Accuracy={self.evaluation_metrics.get('accuracy', 0):.4f}")


# Grid search for optimal hyperparameters
def grid_search_lstm(df: pd.DataFrame,
                     hidden_sizes: list = [4, 8, 16, 32],
                     window_lengths: list = [50, 100, 150],
                     use_full_ohlcv_options: list = [True, False],
                     n_trials: int = None) -> Dict:
    """
    Grid search to find optimal hyperparameters for raw OHLCV LSTM.

    Based on the research finding that simple models (hidden_size=8, window=100)
    often outperform larger configurations.

    Parameters:
    -----------
    df : pd.DataFrame
        Training data
    hidden_sizes : list
        Hidden sizes to try
    window_lengths : list
        Window lengths to try
    use_full_ohlcv_options : list
        Whether to use full OHLCV or just close price
    n_trials : int
        Max number of trials (None = try all combinations)

    Returns:
    --------
    Dict
        Results with best configuration
    """
    logger.info("Starting grid search for optimal LSTM hyperparameters...")

    results = []
    trial_count = 0

    for hidden_size in hidden_sizes:
        for window_length in window_lengths:
            for use_full_ohlcv in use_full_ohlcv_options:

                if n_trials is not None and trial_count >= n_trials:
                    break

                logger.info(f"\n--- Trial {trial_count + 1} ---")
                logger.info(f"Config: hidden_size={hidden_size}, window={window_length}, "
                           f"full_ohlcv={use_full_ohlcv}")

                try:
                    # Create and train model
                    model = RawOHLCVLSTM(
                        hidden_size=hidden_size,
                        window_length=window_length,
                        use_full_ohlcv=use_full_ohlcv
                    )

                    metrics = model.train(df, verbose=0)

                    result = {
                        'hidden_size': hidden_size,
                        'window_length': window_length,
                        'use_full_ohlcv': use_full_ohlcv,
                        'f1_macro': metrics['f1_macro'],
                        'accuracy': metrics['accuracy'],
                        'epochs': metrics['epochs_trained']
                    }

                    results.append(result)
                    logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

                except Exception as e:
                    logger.error(f"Trial failed: {e}")

                trial_count += 1

    # Find best configuration
    results_df = pd.DataFrame(results)
    best_idx = results_df['f1_macro'].idxmax()
    best_config = results_df.loc[best_idx].to_dict()

    logger.info("\n=== Grid Search Complete ===")
    logger.info(f"Best configuration:")
    logger.info(f"  Hidden size: {best_config['hidden_size']}")
    logger.info(f"  Window length: {best_config['window_length']}")
    logger.info(f"  Full OHLCV: {best_config['use_full_ohlcv']}")
    logger.info(f"  F1 Macro: {best_config['f1_macro']:.4f}")
    logger.info(f"  Accuracy: {best_config['accuracy']:.4f}")

    return {
        'best_config': best_config,
        'all_results': results_df
    }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', periods=1000, freq='D')

    # Simulate realistic stock price
    close = 100 + np.cumsum(np.random.randn(1000) * 2 + 0.05)
    df = pd.DataFrame({
        'open': close + np.random.randn(1000) * 0.5,
        'high': close + np.abs(np.random.randn(1000)) * 1.5,
        'low': close - np.abs(np.random.randn(1000)) * 1.5,
        'close': close,
        'volume': np.random.randint(1000000, 10000000, 1000)
    }, index=dates)

    # Test 1: Train with default parameters (matching paper)
    logger.info("\n=== Training with Paper's Optimal Parameters ===")
    model = RawOHLCVLSTM(hidden_size=8, window_length=100, use_full_ohlcv=True)
    metrics = model.train(df, epochs=50)

    # Test 2: Make predictions
    logger.info("\n=== Making Predictions ===")
    predictions = model.predict(df.tail(150))
    probabilities = model.predict(df.tail(150), return_probabilities=True)
    print(f"Predictions: {predictions[:10]}")
    print(f"Probabilities shape: {probabilities.shape}")

    # Test 3: Save and load
    logger.info("\n=== Testing Save/Load ===")
    model.save('test_model')
    model_loaded = RawOHLCVLSTM()
    model_loaded.load('test_model')
