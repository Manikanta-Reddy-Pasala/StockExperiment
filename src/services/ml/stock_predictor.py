"""
Machine Learning Stock Predictor
Uses Random Forest Regression to predict stock price movements and risk scores.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from sqlalchemy import text

logger = logging.getLogger(__name__)


class StockMLPredictor:
    """
    ML-based stock prediction model using Random Forest.
    Predicts 2-week price targets and risk scores based on technical + fundamental features.
    """
    
    def __init__(self, db_session):
        self.db = db_session
        self.price_model = None
        self.risk_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def prepare_training_data(self, lookback_days: int = 365) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data from historical data and technical indicators.
        Returns features (X) and targets (y) DataFrames.
        """
        logger.info(f"Preparing training data with {lookback_days} days lookback")
        
        # Get training data - join stocks, historical_data, and technical_indicators
        query = text("""
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
                ti.signal_line,
                ti.macd_histogram,
                ti.sma_50,
                ti.sma_200,
                ti.ema_12,
                ti.ema_26,
                ti.atr_percentage,
                hd.close as price_at_date,
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
            WHERE hd.date >= NOW() - INTERVAL ':lookback_days days'
            AND s.current_price IS NOT NULL
            AND s.market_cap IS NOT NULL
            ORDER BY s.symbol, hd.date
        """)
        
        result = self.db.execute(query, {"lookback_days": f"{lookback_days} days"})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        # Drop rows without future prices (last 14 days of each stock)
        df = df.dropna(subset=['future_price'])
        
        # Calculate targets
        df['price_change_pct'] = ((df['future_price'] - df['price_at_date']) / df['price_at_date']) * 100
        df['max_drawdown_pct'] = ((df['min_price_14d'] - df['price_at_date']) / df['price_at_date']) * 100
        
        logger.info(f"Prepared {len(df)} training samples")
        return df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and engineer features for ML model."""
        
        # Define feature columns (exclude target and metadata columns)
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
            'rsi_14', 'macd', 'signal_line', 'macd_histogram',
            'sma_50', 'sma_200', 'ema_12', 'ema_26'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Create feature DataFrame
        X = df[available_features].copy()
        
        # Fill NaN values with median (for robust handling)
        X = X.fillna(X.median())
        
        # Add engineered features
        if 'sma_50' in X.columns and 'sma_200' in X.columns:
            X['sma_ratio'] = X['sma_50'] / X['sma_200']
        if 'ema_12' in X.columns and 'ema_26' in X.columns:
            X['ema_diff'] = X['ema_12'] - X['ema_26']
        if 'current_price' in X.columns and 'sma_50' in X.columns:
            X['price_vs_sma50'] = (X['current_price'] - X['sma_50']) / X['sma_50']
        if 'current_price' in X.columns and 'sma_200' in X.columns:
            X['price_vs_sma200'] = (X['current_price'] - X['sma_200']) / X['sma_200']
        
        self.feature_columns = X.columns.tolist()
        logger.info(f"Selected {len(self.feature_columns)} features: {self.feature_columns}")
        
        return X
    
    def train(self, lookback_days: int = 365):
        """Train both price prediction and risk assessment models."""
        logger.info("Starting ML model training...")
        
        # Prepare data
        df = self.prepare_training_data(lookback_days)
        
        if len(df) < 100:
            raise ValueError(f"Insufficient training data: {len(df)} samples. Need at least 100.")
        
        # Prepare features
        X = self._select_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train price prediction model
        logger.info("Training price prediction model...")
        y_price = df['price_change_pct'].values
        self.price_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.price_model.fit(X_scaled, y_price)
        
        # Train risk assessment model (predicting max drawdown)
        logger.info("Training risk assessment model...")
        y_risk = df['max_drawdown_pct'].values
        self.risk_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.risk_model.fit(X_scaled, y_risk)
        
        # Calculate training scores
        price_score = self.price_model.score(X_scaled, y_price)
        risk_score = self.risk_model.score(X_scaled, y_risk)
        
        logger.info(f"Training complete. Price R² = {price_score:.3f}, Risk R² = {risk_score:.3f}")
        logger.info(f"Trained on {len(df)} samples with {len(self.feature_columns)} features")
        
        return {
            'samples': len(df),
            'features': len(self.feature_columns),
            'price_r2': price_score,
            'risk_r2': risk_score
        }
    
    def predict(self, stock_data: Dict) -> Dict:
        """
        Predict ML scores for a single stock.
        
        Args:
            stock_data: Dictionary with stock features
            
        Returns:
            Dictionary with ML predictions:
            - ml_prediction_score: 0-1 score (higher = better predicted performance)
            - ml_price_target: Predicted price after 2 weeks
            - ml_confidence: Model confidence (0-1)
            - ml_risk_score: Risk score (0-1, lower = less risky)
        """
        if self.price_model is None or self.risk_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Prepare feature vector
        feature_dict = {}
        for col in self.feature_columns:
            # Use engineered features if base features available
            if col in ['sma_ratio', 'ema_diff', 'price_vs_sma50', 'price_vs_sma200']:
                continue  # Will be calculated below
            feature_dict[col] = stock_data.get(col, 0)
        
        # Create DataFrame for consistent feature engineering
        X = pd.DataFrame([feature_dict])
        
        # Add engineered features
        if 'sma_50' in X.columns and 'sma_200' in X.columns:
            X['sma_ratio'] = X['sma_50'] / X['sma_200']
        if 'ema_12' in X.columns and 'ema_26' in X.columns:
            X['ema_diff'] = X['ema_12'] - X['ema_26']
        if 'current_price' in X.columns and 'sma_50' in X.columns:
            X['price_vs_sma50'] = (X['current_price'] - X['sma_50']) / X['sma_50']
        if 'current_price' in X.columns and 'sma_200' in X.columns:
            X['price_vs_sma200'] = (X['current_price'] - X['sma_200']) / X['sma_200']
        
        # Ensure all feature columns present
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Reorder columns to match training
        X = X[self.feature_columns]
        
        # Fill NaN
        X = X.fillna(0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        price_change_pct = self.price_model.predict(X_scaled)[0]
        max_drawdown_pct = self.risk_model.predict(X_scaled)[0]
        
        # Calculate ML prediction score (0-1, based on predicted price change)
        # Positive change = higher score, normalize to 0-1 range
        ml_prediction_score = 1 / (1 + np.exp(-price_change_pct / 10))  # Sigmoid
        
        # Calculate price target
        current_price = stock_data.get('current_price', 0)
        ml_price_target = current_price * (1 + price_change_pct / 100)
        
        # Calculate confidence based on feature importance and data quality
        # Higher confidence if key features are present
        key_features = ['rsi_14', 'macd', 'sma_50', 'sma_200', 'pe_ratio', 'roe']
        present_features = sum(1 for f in key_features if stock_data.get(f) is not None and stock_data.get(f) != 0)
        ml_confidence = present_features / len(key_features)
        
        # Calculate risk score (0-1, based on predicted drawdown)
        # Lower drawdown = lower risk score (better)
        # Normalize: 0% drawdown = 0 risk, -20% drawdown = 1 risk
        ml_risk_score = min(1.0, abs(max_drawdown_pct) / 20.0)
        
        return {
            'ml_prediction_score': round(ml_prediction_score, 4),
            'ml_price_target': round(ml_price_target, 2),
            'ml_confidence': round(ml_confidence, 4),
            'ml_risk_score': round(ml_risk_score, 4),
            'predicted_change_pct': round(price_change_pct, 2),
            'predicted_drawdown_pct': round(max_drawdown_pct, 2)
        }
    
    def predict_batch(self, stocks_data: List[Dict]) -> List[Dict]:
        """Predict ML scores for multiple stocks efficiently."""
        predictions = []
        for stock_data in stocks_data:
            try:
                pred = self.predict(stock_data)
                pred['symbol'] = stock_data.get('symbol')
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Prediction failed for {stock_data.get('symbol')}: {e}")
                predictions.append({
                    'symbol': stock_data.get('symbol'),
                    'ml_prediction_score': 0.5,
                    'ml_price_target': stock_data.get('current_price', 0),
                    'ml_confidence': 0.0,
                    'ml_risk_score': 0.5,
                    'error': str(e)
                })
        return predictions
