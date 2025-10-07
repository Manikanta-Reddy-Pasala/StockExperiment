"""
Market Regime Detection System
Detects bull, bear, and sideways market conditions using multiple approaches
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Advanced market regime detection using multiple methodologies.

    Regimes:
    - BULL: Strong uptrend (high returns, low volatility)
    - BEAR: Strong downtrend (negative returns, high volatility)
    - SIDEWAYS: Ranging market (low returns, moderate volatility)
    - VOLATILE_BULL: Uptrend with high volatility
    - VOLATILE_BEAR: Downtrend with high volatility

    Methods:
    1. Gaussian Mixture Models (GMM)
    2. Technical indicators (SMA, RSI, ADX)
    3. Volatility analysis
    4. Trend strength
    """

    def __init__(self, db_session):
        self.db = db_session
        self.regime_model = None
        self.current_regime = None
        self.regime_probabilities = {}
        self.regime_history = []

        # Regime definitions
        self.regimes = {
            0: 'BULL',
            1: 'BEAR',
            2: 'SIDEWAYS',
            3: 'VOLATILE_BULL',
            4: 'VOLATILE_BEAR'
        }

    def detect_regime(self, index_symbol: str = '^NSEI', lookback_days: int = 90) -> Dict:
        """
        Detect current market regime.

        Args:
            index_symbol: Index to analyze (default: Nifty 50)
            lookback_days: Historical period to analyze

        Returns:
            Dictionary with regime analysis
        """
        logger.info(f"Detecting market regime for {index_symbol} (lookback: {lookback_days} days)")

        try:
            # Get market data
            market_data = self._get_market_data(index_symbol, lookback_days)

            if len(market_data) < 30:
                logger.warning("Insufficient data for regime detection")
                return self._default_regime()

            # Calculate features
            features = self._calculate_regime_features(market_data)

            # Multiple detection methods
            gmm_regime = self._detect_gmm_regime(features)
            technical_regime = self._detect_technical_regime(market_data)
            volatility_regime = self._detect_volatility_regime(features)
            trend_regime = self._detect_trend_regime(market_data)

            # Ensemble decision
            final_regime = self._ensemble_regime_decision(
                gmm_regime, technical_regime, volatility_regime, trend_regime
            )

            # Calculate regime strength
            regime_strength = self._calculate_regime_strength(features, final_regime)

            # Get regime characteristics
            characteristics = self._get_regime_characteristics(final_regime, features)

            result = {
                'regime': final_regime,
                'confidence': regime_strength,
                'characteristics': characteristics,
                'methods': {
                    'gmm': gmm_regime,
                    'technical': technical_regime,
                    'volatility': volatility_regime,
                    'trend': trend_regime
                },
                'detected_at': datetime.now().isoformat()
            }

            self.current_regime = final_regime
            self.regime_history.append(result)

            logger.info(f"Regime detected: {final_regime} (confidence: {regime_strength:.2%})")
            return result

        except Exception as e:
            logger.error(f"Regime detection failed: {e}", exc_info=True)
            return self._default_regime()

    def _get_market_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Get historical market data."""
        # For demo, use aggregate stock data as proxy for index
        # In production, use actual index data
        query = text("""
            SELECT
                date,
                AVG(close) as close,
                AVG(high) as high,
                AVG(low) as low,
                AVG(volume) as volume
            FROM historical_data
            WHERE date >= NOW() - INTERVAL ':days days'
            GROUP BY date
            ORDER BY date DESC
            LIMIT :days
        """)

        result = self.db.execute(query, {'days': days})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

        if len(df) > 0:
            df = df.sort_values('date')
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        return df

    def _calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for regime detection."""
        features = pd.DataFrame()

        # Returns-based features
        features['returns'] = df['returns']
        features['log_returns'] = df['log_returns']
        features['returns_ma5'] = df['returns'].rolling(5).mean()
        features['returns_ma20'] = df['returns'].rolling(20).mean()

        # Volatility features
        features['volatility_5d'] = df['returns'].rolling(5).std()
        features['volatility_20d'] = df['returns'].rolling(20).std()
        features['volatility_ratio'] = features['volatility_5d'] / features['volatility_20d']

        # Price momentum
        features['momentum_5d'] = df['close'].pct_change(5)
        features['momentum_20d'] = df['close'].pct_change(20)

        # Trend strength (ADX-like)
        features['trend_strength'] = self._calculate_trend_strength(df)

        # Higher moments
        features['skewness'] = df['returns'].rolling(20).skew()
        features['kurtosis'] = df['returns'].rolling(20).kurt()

        return features.dropna()

    def _calculate_trend_strength(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate trend strength (ADX-like indicator)."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed indicators
        atr = pd.Series(tr).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx

    def _detect_gmm_regime(self, features: pd.DataFrame) -> str:
        """Detect regime using Gaussian Mixture Model."""
        try:
            # Select key features for GMM
            X = features[['returns_ma20', 'volatility_20d']].dropna().values

            if len(X) < 10:
                return 'UNKNOWN'

            # Fit GMM with 3 components (Bull, Bear, Sideways)
            gmm = GaussianMixture(n_components=3, random_state=42)
            gmm.fit(X)

            # Predict current regime
            latest_features = X[-1].reshape(1, -1)
            regime_idx = gmm.predict(latest_features)[0]

            # Map to regime based on means
            means = gmm.means_
            return_means = means[:, 0]
            vol_means = means[:, 1]

            # Classify based on return and volatility
            if return_means[regime_idx] > 0.001 and vol_means[regime_idx] < 0.015:
                return 'BULL'
            elif return_means[regime_idx] < -0.001 and vol_means[regime_idx] > 0.015:
                return 'BEAR'
            elif return_means[regime_idx] > 0.001 and vol_means[regime_idx] > 0.02:
                return 'VOLATILE_BULL'
            elif return_means[regime_idx] < -0.001 and vol_means[regime_idx] > 0.02:
                return 'VOLATILE_BEAR'
            else:
                return 'SIDEWAYS'

        except Exception as e:
            logger.warning(f"GMM regime detection failed: {e}")
            return 'UNKNOWN'

    def _detect_technical_regime(self, df: pd.DataFrame) -> str:
        """Detect regime using technical indicators."""
        try:
            # Calculate SMAs
            sma_20 = df['close'].rolling(20).mean()
            sma_50 = df['close'].rolling(50).mean()

            current_price = df['close'].iloc[-1]
            sma_20_val = sma_20.iloc[-1]
            sma_50_val = sma_50.iloc[-1]

            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # Regime rules
            if current_price > sma_20_val > sma_50_val and current_rsi > 50:
                return 'BULL'
            elif current_price < sma_20_val < sma_50_val and current_rsi < 50:
                return 'BEAR'
            elif abs(current_price - sma_20_val) / current_price < 0.02:
                return 'SIDEWAYS'
            else:
                return 'SIDEWAYS'

        except Exception as e:
            logger.warning(f"Technical regime detection failed: {e}")
            return 'UNKNOWN'

    def _detect_volatility_regime(self, features: pd.DataFrame) -> str:
        """Detect regime based on volatility analysis."""
        try:
            vol_20d = features['volatility_20d'].iloc[-1]
            vol_ratio = features['volatility_ratio'].iloc[-1]

            # High volatility threshold
            if vol_20d > 0.025:
                returns = features['returns_ma20'].iloc[-1]
                if returns > 0:
                    return 'VOLATILE_BULL'
                else:
                    return 'VOLATILE_BEAR'
            elif vol_20d < 0.01:
                return 'SIDEWAYS'
            else:
                return 'UNKNOWN'

        except Exception as e:
            logger.warning(f"Volatility regime detection failed: {e}")
            return 'UNKNOWN'

    def _detect_trend_regime(self, df: pd.DataFrame) -> str:
        """Detect regime based on trend analysis."""
        try:
            # Calculate trend metrics
            returns_20d = df['close'].pct_change(20).iloc[-1]
            returns_50d = df['close'].pct_change(50).iloc[-1] if len(df) >= 50 else 0

            # Strong trends
            if returns_20d > 0.05 and returns_50d > 0.05:
                return 'BULL'
            elif returns_20d < -0.05 and returns_50d < -0.05:
                return 'BEAR'
            else:
                return 'SIDEWAYS'

        except Exception as e:
            logger.warning(f"Trend regime detection failed: {e}")
            return 'UNKNOWN'

    def _ensemble_regime_decision(self, gmm: str, technical: str,
                                  volatility: str, trend: str) -> str:
        """Combine multiple regime detections into final decision."""
        # Count votes for each regime
        votes = [gmm, technical, volatility, trend]
        valid_votes = [v for v in votes if v != 'UNKNOWN']

        if not valid_votes:
            return 'SIDEWAYS'

        # Majority voting
        from collections import Counter
        vote_counts = Counter(valid_votes)
        final_regime = vote_counts.most_common(1)[0][0]

        return final_regime

    def _calculate_regime_strength(self, features: pd.DataFrame, regime: str) -> float:
        """Calculate confidence in regime detection."""
        try:
            # Metrics for strength calculation
            trend_strength = features['trend_strength'].iloc[-1]
            volatility = features['volatility_20d'].iloc[-1]
            momentum = abs(features['momentum_20d'].iloc[-1])

            if regime == 'BULL' or regime == 'VOLATILE_BULL':
                strength = min(1.0, (trend_strength / 25.0) * (momentum / 0.1))
            elif regime == 'BEAR' or regime == 'VOLATILE_BEAR':
                strength = min(1.0, (trend_strength / 25.0) * (momentum / 0.1))
            else:  # SIDEWAYS
                strength = max(0.3, 1.0 - (trend_strength / 25.0))

            return max(0.3, min(1.0, strength))

        except:
            return 0.5

    def _get_regime_characteristics(self, regime: str, features: pd.DataFrame) -> Dict:
        """Get characteristics of detected regime."""
        try:
            return {
                'avg_return': float(features['returns_ma20'].iloc[-1]),
                'volatility': float(features['volatility_20d'].iloc[-1]),
                'trend_strength': float(features['trend_strength'].iloc[-1]),
                'momentum_20d': float(features['momentum_20d'].iloc[-1]),
                'description': self._get_regime_description(regime)
            }
        except:
            return {}

    def _get_regime_description(self, regime: str) -> str:
        """Get human-readable regime description."""
        descriptions = {
            'BULL': 'Strong uptrend with healthy momentum and low volatility',
            'BEAR': 'Strong downtrend with negative momentum',
            'SIDEWAYS': 'Range-bound market with no clear trend',
            'VOLATILE_BULL': 'Uptrend with high volatility and increased risk',
            'VOLATILE_BEAR': 'Downtrend with high volatility and panic selling'
        }
        return descriptions.get(regime, 'Unknown market condition')

    def _default_regime(self) -> Dict:
        """Return default regime when detection fails."""
        return {
            'regime': 'SIDEWAYS',
            'confidence': 0.3,
            'characteristics': {
                'description': 'Unable to determine regime - assuming neutral'
            },
            'methods': {},
            'detected_at': datetime.now().isoformat()
        }

    def get_regime_specific_strategy(self, regime: str) -> Dict:
        """
        Get trading strategy recommendations based on regime.

        Args:
            regime: Current market regime

        Returns:
            Strategy parameters optimized for regime
        """
        strategies = {
            'BULL': {
                'position_size': 0.15,  # Larger positions in bull market
                'stop_loss_pct': 8.0,
                'take_profit_pct': 15.0,
                'preferred_strategies': ['momentum', 'growth'],
                'risk_tolerance': 'moderate',
                'rebalance_frequency': 14,
                'ml_score_threshold': 0.55
            },
            'BEAR': {
                'position_size': 0.05,  # Smaller positions in bear market
                'stop_loss_pct': 5.0,
                'take_profit_pct': 8.0,
                'preferred_strategies': ['defensive', 'value'],
                'risk_tolerance': 'conservative',
                'rebalance_frequency': 7,
                'ml_score_threshold': 0.70
            },
            'SIDEWAYS': {
                'position_size': 0.10,
                'stop_loss_pct': 6.0,
                'take_profit_pct': 10.0,
                'preferred_strategies': ['mean_reversion', 'balanced'],
                'risk_tolerance': 'moderate',
                'rebalance_frequency': 14,
                'ml_score_threshold': 0.60
            },
            'VOLATILE_BULL': {
                'position_size': 0.08,
                'stop_loss_pct': 10.0,
                'take_profit_pct': 20.0,
                'preferred_strategies': ['breakout', 'volatility'],
                'risk_tolerance': 'aggressive',
                'rebalance_frequency': 7,
                'ml_score_threshold': 0.65
            },
            'VOLATILE_BEAR': {
                'position_size': 0.03,
                'stop_loss_pct': 4.0,
                'take_profit_pct': 6.0,
                'preferred_strategies': ['cash', 'defensive'],
                'risk_tolerance': 'very_conservative',
                'rebalance_frequency': 3,
                'ml_score_threshold': 0.80
            }
        }

        return strategies.get(regime, strategies['SIDEWAYS'])

    def get_regime_history(self, days: int = 30) -> List[Dict]:
        """Get regime history for analysis."""
        return self.regime_history[-days:]
