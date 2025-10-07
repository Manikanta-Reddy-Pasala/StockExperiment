"""
Enhanced Machine Learning Stock Predictor with Ensemble Models
Combines Random Forest, XGBoost, and LSTM with walk-forward validation
Inspired by AlphaSuite's advanced techniques
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from sqlalchemy import text
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

# Import calibrated scoring
try:
    from src.services.ml.calibrated_scoring import CalibratedScorer, AdaptiveScorer
    HAS_CALIBRATION = True
except ImportError:
    HAS_CALIBRATION = False
    logger.warning("Calibrated scoring not available")

logger = logging.getLogger(__name__)


class EnhancedStockPredictor:
    """
    Advanced ML stock predictor with ensemble models and walk-forward validation.

    Features:
    - Multi-model ensemble (Random Forest + XGBoost + LSTM)
    - Walk-forward validation for robust performance
    - Enhanced feature engineering with chaos theory
    - Time-series cross-validation
    - Feature importance tracking
    """

    def __init__(self, db_session):
        self.db = db_session

        # Models
        self.rf_price_model = None
        self.rf_risk_model = None
        self.xgb_price_model = None if not HAS_XGB else None
        self.xgb_risk_model = None if not HAS_XGB else None
        self.lstm_price_model = None if not HAS_KERAS else None

        # Scalers
        self.feature_scaler = StandardScaler()
        self.lstm_feature_scaler = StandardScaler() if HAS_KERAS else None
        self.lstm_target_scaler = StandardScaler() if HAS_KERAS else None

        # Feature tracking
        self.feature_columns = []
        self.feature_importance = {}

        # Performance tracking
        self.cv_scores = {}
        self.ensemble_weights = {'rf': 0.4, 'xgb': 0.35, 'lstm': 0.25}

        # Calibrated scoring
        self.calibrated_scorer = CalibratedScorer() if HAS_CALIBRATION else None
        self.adaptive_scorer = AdaptiveScorer() if HAS_CALIBRATION else None

    def prepare_training_data(self, lookback_days: int = 365) -> pd.DataFrame:
        """
        Prepare training data from historical data and technical indicators.
        Returns a DataFrame with features and targets.
        """
        logger.info(f"Preparing training data with {lookback_days} days lookback")

        query = text(f"""
            SELECT
                s.symbol,
                s.current_price,
                s.market_cap,
                s.pe_ratio,
                s.pb_ratio,
                s.roe,
                s.eps,
                s.beta,
                s.debt_to_equity,
                s.revenue_growth,
                s.earnings_growth,
                s.operating_margin,
                s.net_margin,
                s.historical_volatility_1y,
                s.atr_14,
                ti.rsi_14,
                ti.macd,
                ti.macd_signal,
                ti.macd_histogram,
                ti.sma_50,
                ti.sma_200,
                ti.ema_12,
                ti.ema_26,
                ti.atr_percentage,
                ti.bb_upper as bollinger_upper,
                ti.bb_lower as bollinger_lower,
                hd.close as price_at_date,
                hd.high,
                hd.low,
                hd.volume,
                hd.date as observation_date,
                -- Target: price 14 days later
                LEAD(hd.close, 14) OVER (PARTITION BY s.symbol ORDER BY hd.date) as future_price,
                -- Risk target: max drawdown in next 14 days
                (SELECT MIN(close) FROM historical_data
                 WHERE symbol = s.symbol
                 AND date > hd.date
                 AND date <= hd.date + INTERVAL '14 days') as min_price_14d
            FROM stocks s
            JOIN historical_data hd ON s.symbol = hd.symbol
            LEFT JOIN technical_indicators ti ON s.symbol = ti.symbol AND hd.date = ti.date
            WHERE hd.date >= NOW() - INTERVAL '{lookback_days} days'
            AND s.current_price IS NOT NULL
            AND s.market_cap IS NOT NULL
            ORDER BY s.symbol, hd.date
        """)

        result = self.db.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

        # Drop rows without future prices
        df = df.dropna(subset=['future_price'])

        # Calculate targets
        df['price_change_pct'] = ((df['future_price'] - df['price_at_date']) / df['price_at_date']) * 100
        df['max_drawdown_pct'] = ((df['min_price_14d'] - df['price_at_date']) / df['price_at_date']) * 100

        logger.info(f"Prepared {len(df)} training samples")
        return df

    def _add_chaos_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add chaos theory-inspired features (AlphaSuite approach).

        Features:
        - Hurst Exponent: Trend persistence (0-1, >0.5 = trending, <0.5 = mean-reverting)
        - Fractal Dimension: Market complexity
        - Price Entropy: Uncertainty measure
        - Lorenz-inspired momentum features
        """
        df = df.copy()

        # Group by symbol for time-series features
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()

            if len(symbol_data) < 20:
                continue

            prices = symbol_data['price_at_date'].values

            # 1. Hurst Exponent (simplified)
            hurst = self._calculate_hurst_exponent(prices)
            df.loc[mask, 'hurst_exponent'] = hurst

            # 2. Fractal Dimension
            fractal_dim = self._calculate_fractal_dimension(prices)
            df.loc[mask, 'fractal_dimension'] = fractal_dim

            # 3. Price Entropy (uncertainty)
            entropy = self._calculate_entropy(prices)
            df.loc[mask, 'price_entropy'] = entropy

            # 4. Lorenz-inspired momentum (rate of change of momentum)
            if len(prices) >= 3:
                returns = np.diff(prices) / prices[:-1]
                momentum = np.diff(returns) if len(returns) > 1 else [0]
                lorenz_momentum = np.std(momentum) if len(momentum) > 0 else 0
                df.loc[mask, 'lorenz_momentum'] = lorenz_momentum

        return df

    def _calculate_hurst_exponent(self, prices: np.ndarray, max_lag: int = 20) -> float:
        """Calculate Hurst exponent using simplified R/S analysis."""
        try:
            if len(prices) < max_lag:
                return 0.5  # Neutral

            lags = range(2, min(max_lag, len(prices) // 2))
            tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]

            # Filter out zeros to avoid log(0)
            valid_idx = [i for i, t in enumerate(tau) if t > 0]
            if len(valid_idx) < 2:
                return 0.5

            lags_valid = [lags[i] for i in valid_idx]
            tau_valid = [tau[i] for i in valid_idx]

            poly = np.polyfit(np.log(lags_valid), np.log(tau_valid), 1)
            return poly[0]
        except:
            return 0.5

    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension (Higuchi method simplified)."""
        try:
            if len(prices) < 10:
                return 1.5

            n = len(prices)
            max_k = min(10, n // 2)

            lk = []
            for k in range(1, max_k):
                lm = []
                for m in range(k):
                    ll = 0
                    for i in range(1, (n - m) // k):
                        ll += abs(prices[m + i * k] - prices[m + (i - 1) * k])
                    ll = ll * (n - 1) / ((n - m) // k * k) / k
                    lm.append(ll)
                lk.append(np.mean(lm))

            if len(lk) < 2:
                return 1.5

            x = np.log(range(1, len(lk) + 1))
            y = np.log(lk)
            poly = np.polyfit(x, y, 1)
            return -poly[0]
        except:
            return 1.5

    def _calculate_entropy(self, prices: np.ndarray, bins: int = 10) -> float:
        """Calculate Shannon entropy of price distribution."""
        try:
            hist, _ = np.histogram(prices, bins=bins)
            hist = hist[hist > 0]  # Remove zeros
            probs = hist / np.sum(hist)
            entropy = -np.sum(probs * np.log2(probs))
            return entropy
        except:
            return 0.0

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and engineer features for ML models."""

        # Base feature columns
        feature_cols = [
            # Price & Market
            'current_price', 'market_cap', 'volume',
            # Fundamental ratios
            'pe_ratio', 'pb_ratio', 'roe', 'eps', 'beta', 'debt_to_equity',
            # Growth & Profitability
            'revenue_growth', 'earnings_growth', 'operating_margin', 'net_margin',
            # Volatility
            'historical_volatility_1y', 'atr_14', 'atr_percentage',
            # Technical indicators
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'sma_50', 'sma_200', 'ema_12', 'ema_26',
            'bollinger_upper', 'bollinger_lower',
            # Chaos features (if available)
            'hurst_exponent', 'fractal_dimension', 'price_entropy', 'lorenz_momentum'
        ]

        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].copy()

        # Fill NaN values with median
        X = X.fillna(X.median())

        # Engineered features
        if 'sma_50' in X.columns and 'sma_200' in X.columns:
            X['sma_ratio'] = X['sma_50'] / X['sma_200'].replace(0, np.nan)
            X['golden_cross'] = (X['sma_50'] > X['sma_200']).astype(float)

        if 'ema_12' in X.columns and 'ema_26' in X.columns:
            X['ema_diff'] = X['ema_12'] - X['ema_26']
            X['ema_cross'] = (X['ema_12'] > X['ema_26']).astype(float)

        if 'current_price' in X.columns and 'sma_50' in X.columns:
            X['price_vs_sma50'] = (X['current_price'] - X['sma_50']) / X['sma_50'].replace(0, np.nan)

        if 'current_price' in X.columns and 'sma_200' in X.columns:
            X['price_vs_sma200'] = (X['current_price'] - X['sma_200']) / X['sma_200'].replace(0, np.nan)

        if 'bollinger_upper' in X.columns and 'bollinger_lower' in X.columns:
            X['bb_width'] = (X['bollinger_upper'] - X['bollinger_lower']) / X['current_price'].replace(0, np.nan)
            X['bb_position'] = (X['current_price'] - X['bollinger_lower']) / (X['bollinger_upper'] - X['bollinger_lower']).replace(0, np.nan)

        if 'rsi_14' in X.columns:
            X['rsi_overbought'] = (X['rsi_14'] > 70).astype(float)
            X['rsi_oversold'] = (X['rsi_14'] < 30).astype(float)

        # Replace inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        self.feature_columns = X.columns.tolist()
        logger.info(f"Selected {len(self.feature_columns)} features: {self.feature_columns}")

        return X

    def train_with_walk_forward(self, lookback_days: int = 365, n_splits: int = 5):
        """
        Train models using walk-forward validation (time-series cross-validation).
        This prevents overfitting and provides realistic performance estimates.
        """
        logger.info("Starting walk-forward training with time-series CV...")

        # Prepare data
        df = self.prepare_training_data(lookback_days)

        if len(df) < 100:
            raise ValueError(f"Insufficient training data: {len(df)} samples. Need at least 100.")

        # Add chaos features
        logger.info("Adding chaos theory features...")
        df = self._add_chaos_features(df)

        # Prepare features
        X = self._select_features(df)

        # Filter valid targets
        valid_mask = df['price_change_pct'].notna() & df['max_drawdown_pct'].notna()
        X = X[valid_mask]
        df = df[valid_mask]

        logger.info(f"Training on {len(df)} samples with {len(self.feature_columns)} features")

        # Targets
        y_price = df['price_change_pct'].values
        y_risk = df['max_drawdown_pct'].values

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores_price = []
        cv_scores_risk = []

        logger.info(f"Performing {n_splits}-fold walk-forward validation...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train_price, y_val_price = y_price[train_idx], y_price[val_idx]
            y_train_risk, y_val_risk = y_risk[train_idx], y_risk[val_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train RF models
            rf_price = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            rf_price.fit(X_train_scaled, y_train_price)
            price_score = rf_price.score(X_val_scaled, y_val_price)
            cv_scores_price.append(price_score)

            rf_risk = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            rf_risk.fit(X_train_scaled, y_train_risk)
            risk_score = rf_risk.score(X_val_scaled, y_val_risk)
            cv_scores_risk.append(risk_score)

            logger.info(f"Fold {fold}/{n_splits} - Price R²: {price_score:.3f}, Risk R²: {risk_score:.3f}")

        # Store CV performance
        self.cv_scores = {
            'price_cv_mean': np.mean(cv_scores_price),
            'price_cv_std': np.std(cv_scores_price),
            'risk_cv_mean': np.mean(cv_scores_risk),
            'risk_cv_std': np.std(cv_scores_risk)
        }

        logger.info(f"Walk-forward CV complete:")
        logger.info(f"  Price R²: {self.cv_scores['price_cv_mean']:.3f} ± {self.cv_scores['price_cv_std']:.3f}")
        logger.info(f"  Risk R²: {self.cv_scores['risk_cv_mean']:.3f} ± {self.cv_scores['risk_cv_std']:.3f}")

        # Train final models on all data
        logger.info("Training final models on full dataset...")
        X_scaled = self.feature_scaler.fit_transform(X)

        # Random Forest models
        self.rf_price_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_price_model.fit(X_scaled, y_price)

        self.rf_risk_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_risk_model.fit(X_scaled, y_risk)

        # XGBoost models (if available)
        if HAS_XGB:
            logger.info("Training XGBoost models...")
            self.xgb_price_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.xgb_price_model.fit(X_scaled, y_price)

            self.xgb_risk_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.xgb_risk_model.fit(X_scaled, y_risk)

        # Track feature importance (from RF)
        self.feature_importance = dict(zip(
            self.feature_columns,
            self.rf_price_model.feature_importances_
        ))

        # Get top 10 features
        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 most important features:")
        for feat, imp in top_features:
            logger.info(f"  {feat}: {imp:.4f}")

        # Fit calibration model if available
        if self.calibrated_scorer is not None:
            logger.info("Fitting calibration model...")
            predictions = self.rf_price_model.predict(X_scaled)
            if HAS_XGB and self.xgb_price_model is not None:
                xgb_predictions = self.xgb_price_model.predict(X_scaled)
                # Average predictions for calibration
                predictions = (predictions * 0.5 + xgb_predictions * 0.5)

            self.calibrated_scorer.fit(predictions, y_price)
            logger.info("Calibration model fitted successfully")

        return {
            'samples': len(df),
            'features': len(self.feature_columns),
            'price_r2': self.rf_price_model.score(X_scaled, y_price),
            'risk_r2': self.rf_risk_model.score(X_scaled, y_risk),
            'cv_price_r2': self.cv_scores['price_cv_mean'],
            'cv_risk_r2': self.cv_scores['risk_cv_mean'],
            'top_features': [f[0] for f in top_features]
        }

    def predict(self, stock_data: Dict) -> Dict:
        """
        Predict ML scores using ensemble of all available models.
        """
        if self.rf_price_model is None:
            raise ValueError("Models not trained. Call train_with_walk_forward() first.")

        # Prepare features
        X = self._prepare_prediction_features(stock_data)
        X_scaled = self.feature_scaler.transform(X)

        # Get predictions from all models
        rf_price = self.rf_price_model.predict(X_scaled)[0]
        rf_risk = self.rf_risk_model.predict(X_scaled)[0]

        # Ensemble prediction
        price_predictions = [rf_price]
        risk_predictions = [rf_risk]
        weights = [self.ensemble_weights['rf']]

        if HAS_XGB and self.xgb_price_model is not None:
            xgb_price = self.xgb_price_model.predict(X_scaled)[0]
            xgb_risk = self.xgb_risk_model.predict(X_scaled)[0]
            price_predictions.append(xgb_price)
            risk_predictions.append(xgb_risk)
            weights.append(self.ensemble_weights['xgb'])

        # Weighted ensemble
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        ensemble_price_change = sum(p * w for p, w in zip(price_predictions, normalized_weights))
        ensemble_risk = sum(r * w for r, w in zip(risk_predictions, normalized_weights))

        # Calculate scores using calibrated scoring if available
        if self.calibrated_scorer is not None and self.calibrated_scorer.is_fitted:
            ml_prediction_score = self.calibrated_scorer.score(ensemble_price_change)
        else:
            # Fallback to sigmoid
            ml_prediction_score = 1 / (1 + np.exp(-ensemble_price_change / 10))

        current_price = stock_data.get('current_price', 0)
        ml_price_target = current_price * (1 + ensemble_price_change / 100)

        # Confidence based on feature availability and CV performance
        key_features = ['rsi_14', 'macd', 'sma_50', 'sma_200', 'pe_ratio', 'roe']
        present_features = sum(1 for f in key_features if stock_data.get(f) is not None and stock_data.get(f) != 0)
        feature_confidence = present_features / len(key_features)

        # Adjust confidence based on CV performance
        cv_confidence = self.cv_scores.get('price_cv_mean', 0.3)
        ml_confidence = (feature_confidence * 0.6 + cv_confidence * 0.4)

        # Risk score
        ml_risk_score = min(1.0, abs(ensemble_risk) / 20.0)

        return {
            'ml_prediction_score': round(ml_prediction_score, 4),
            'ml_price_target': round(ml_price_target, 2),
            'ml_confidence': round(ml_confidence, 4),
            'ml_risk_score': round(ml_risk_score, 4),
            'predicted_change_pct': round(ensemble_price_change, 2),
            'predicted_drawdown_pct': round(ensemble_risk, 2),
            'model_performance': {
                'cv_price_r2': round(self.cv_scores.get('price_cv_mean', 0), 3),
                'cv_risk_r2': round(self.cv_scores.get('risk_cv_mean', 0), 3)
            }
        }

    def _prepare_prediction_features(self, stock_data: Dict) -> pd.DataFrame:
        """Prepare feature vector for prediction."""
        feature_dict = {}

        for col in self.feature_columns:
            if col in ['sma_ratio', 'golden_cross', 'ema_diff', 'ema_cross',
                      'price_vs_sma50', 'price_vs_sma200', 'bb_width', 'bb_position',
                      'rsi_overbought', 'rsi_oversold']:
                continue  # Will be calculated below
            feature_dict[col] = stock_data.get(col, 0)

        X = pd.DataFrame([feature_dict])

        # Add engineered features (same logic as training)
        if 'sma_50' in X.columns and 'sma_200' in X.columns:
            X['sma_ratio'] = X['sma_50'] / X['sma_200'].replace(0, np.nan)
            X['golden_cross'] = (X['sma_50'] > X['sma_200']).astype(float)

        if 'ema_12' in X.columns and 'ema_26' in X.columns:
            X['ema_diff'] = X['ema_12'] - X['ema_26']
            X['ema_cross'] = (X['ema_12'] > X['ema_26']).astype(float)

        if 'current_price' in X.columns and 'sma_50' in X.columns:
            X['price_vs_sma50'] = (X['current_price'] - X['sma_50']) / X['sma_50'].replace(0, np.nan)

        if 'current_price' in X.columns and 'sma_200' in X.columns:
            X['price_vs_sma200'] = (X['current_price'] - X['sma_200']) / X['sma_200'].replace(0, np.nan)

        if 'bollinger_upper' in X.columns and 'bollinger_lower' in X.columns:
            X['bb_width'] = (X['bollinger_upper'] - X['bollinger_lower']) / X['current_price'].replace(0, np.nan)
            X['bb_position'] = (X['current_price'] - X['bollinger_lower']) / (X['bollinger_upper'] - X['bollinger_lower']).replace(0, np.nan)

        if 'rsi_14' in X.columns:
            X['rsi_overbought'] = (X['rsi_14'] > 70).astype(float)
            X['rsi_oversold'] = (X['rsi_14'] < 30).astype(float)

        # Ensure all feature columns present
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        # Reorder and clean
        X = X[self.feature_columns]
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        return X
