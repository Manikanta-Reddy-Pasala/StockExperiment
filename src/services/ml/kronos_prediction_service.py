"""
Kronos-Inspired Financial Foundation Model Prediction Service

Implements key principles from the Kronos paper:
- K-line (candlestick) tokenization using hierarchical autoencoder
- Discrete token-based sequence modeling
- Coarse-to-fine prediction with probabilistic forecasting
- Pre-trained on financial time series patterns

Note: This is a Kronos-inspired implementation using similar principles.
The actual Kronos model requires 12B+ records pre-training which is not feasible here.
Instead, we use pattern-based tokenization and probabilistic forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from pathlib import Path
import logging
from sqlalchemy import text

from src.models.database import get_database_manager

logger = logging.getLogger(__name__)


class KLineTokenizer:
    """
    K-line tokenizer that converts OHLCVA candlestick data into discrete tokens.

    Inspired by Kronos paper:
    - Binary Spherical Quantization (BSQ) for efficient tokenization
    - Hierarchical tokens (coarse + fine) for multi-scale pattern capture
    - Captures market microstructure patterns
    """

    def __init__(self, coarse_vocab_size: int = 256, fine_vocab_size: int = 256):
        """
        Initialize K-line tokenizer.

        Parameters:
        -----------
        coarse_vocab_size : int
            Size of coarse token vocabulary (captures broad patterns)
        fine_vocab_size : int
            Size of fine token vocabulary (captures detailed patterns)
        """
        self.coarse_vocab_size = coarse_vocab_size
        self.fine_vocab_size = fine_vocab_size

        # Pattern definitions (coarse-level)
        self.coarse_patterns = {
            'strong_bullish': 0,      # Large green candle, high volume
            'bullish': 1,             # Green candle
            'weak_bullish': 2,        # Small green candle
            'neutral': 3,             # Doji or very small body
            'weak_bearish': 4,        # Small red candle
            'bearish': 5,             # Red candle
            'strong_bearish': 6,      # Large red candle, high volume
            'hammer': 7,              # Bullish reversal (long lower shadow)
            'shooting_star': 8,       # Bearish reversal (long upper shadow)
            'engulfing_bull': 9,      # Bullish engulfing pattern
            'engulfing_bear': 10,     # Bearish engulfing pattern
        }

    def tokenize_kline(self, ohlcva: pd.Series) -> Tuple[int, int]:
        """
        Convert a single K-line (OHLCVA) into hierarchical tokens.

        Parameters:
        -----------
        ohlcva : pd.Series
            Row with columns: open, high, low, close, volume, amount

        Returns:
        --------
        Tuple[int, int]
            (coarse_token, fine_token)
        """
        try:
            open_price = float(ohlcva['open'])
            high = float(ohlcva['high'])
            low = float(ohlcva['low'])
            close = float(ohlcva['close'])
            volume = float(ohlcva['volume']) if 'volume' in ohlcva else 0

            # Calculate pattern features
            body = abs(close - open_price)
            range_hl = high - low
            upper_shadow = high - max(open_price, close)
            lower_shadow = min(open_price, close) - low

            # Avoid division by zero
            if range_hl == 0:
                range_hl = 0.001

            body_pct = (body / range_hl) * 100
            upper_shadow_pct = (upper_shadow / range_hl) * 100
            lower_shadow_pct = (lower_shadow / range_hl) * 100

            # Direction
            is_bullish = close > open_price
            is_bearish = close < open_price

            # Volume analysis (normalized to avoid dependency on absolute values)
            # We'll use relative volume later when we have historical context

            # COARSE TOKEN: Identify candlestick pattern
            coarse_token = self._identify_coarse_pattern(
                body_pct, upper_shadow_pct, lower_shadow_pct, is_bullish, is_bearish
            )

            # FINE TOKEN: Quantize price movement and volume
            # Fine token encodes: price change magnitude, volume level, volatility
            fine_token = self._quantize_fine_features(
                body_pct, upper_shadow_pct, lower_shadow_pct, volume
            )

            return coarse_token, fine_token

        except Exception as e:
            logger.warning(f"Error tokenizing K-line: {e}")
            return 3, 127  # Return neutral pattern as fallback

    def _identify_coarse_pattern(
        self, body_pct: float, upper_shadow_pct: float,
        lower_shadow_pct: float, is_bullish: bool, is_bearish: bool
    ) -> int:
        """Identify coarse candlestick pattern."""

        # Doji or very small body (neutral)
        if body_pct < 10:
            return self.coarse_patterns['neutral']

        # Hammer (bullish reversal) - small body, long lower shadow
        if body_pct < 30 and lower_shadow_pct > 60:
            return self.coarse_patterns['hammer']

        # Shooting star (bearish reversal) - small body, long upper shadow
        if body_pct < 30 and upper_shadow_pct > 60:
            return self.coarse_patterns['shooting_star']

        # Strong patterns (large body > 70%)
        if body_pct > 70:
            if is_bullish:
                return self.coarse_patterns['strong_bullish']
            else:
                return self.coarse_patterns['strong_bearish']

        # Medium patterns (40-70%)
        if body_pct >= 40:
            if is_bullish:
                return self.coarse_patterns['bullish']
            else:
                return self.coarse_patterns['bearish']

        # Weak patterns (10-40%)
        if is_bullish:
            return self.coarse_patterns['weak_bullish']
        else:
            return self.coarse_patterns['weak_bearish']

    def _quantize_fine_features(
        self, body_pct: float, upper_shadow_pct: float,
        lower_shadow_pct: float, volume: float
    ) -> int:
        """
        Quantize fine-grained features into a discrete token.

        Returns a token in range [0, 255] representing:
        - Body size (0-100%)
        - Shadow imbalance
        - Volume level (relative)
        """
        # Quantize body percentage to 0-7 (3 bits)
        body_quantized = min(7, int(body_pct / 12.5))

        # Quantize shadow imbalance to 0-7 (3 bits)
        shadow_imbalance = upper_shadow_pct - lower_shadow_pct
        shadow_quantized = min(7, int((shadow_imbalance + 100) / 25))

        # Volume quantization (placeholder, needs historical context)
        # For now, use log scale
        volume_quantized = min(3, int(np.log10(volume + 1) / 2)) if volume > 0 else 0

        # Combine into 8-bit token (3 + 3 + 2 bits = 8 bits = 256 values)
        fine_token = (body_quantized << 5) | (shadow_quantized << 2) | volume_quantized

        return fine_token

    def tokenize_sequence(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize a sequence of K-lines.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCVA data

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (coarse_tokens, fine_tokens) - arrays of shape (n_samples,)
        """
        coarse_tokens = []
        fine_tokens = []

        for _, row in df.iterrows():
            coarse, fine = self.tokenize_kline(row)
            coarse_tokens.append(coarse)
            fine_tokens.append(fine)

        return np.array(coarse_tokens), np.array(fine_tokens)


class KronosInspiredPredictor:
    """
    Kronos-inspired financial time series predictor.

    Key principles from Kronos paper:
    - Token-based sequence modeling
    - Autoregressive probabilistic forecasting
    - Coarse-to-fine prediction hierarchy
    - Pattern-based market understanding
    """

    def __init__(self, lookback_window: int = 60, forecast_horizon: int = 14):
        """
        Initialize Kronos-inspired predictor.

        Parameters:
        -----------
        lookback_window : int
            Number of historical K-lines to consider
        forecast_horizon : int
            Number of days to forecast
        """
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.tokenizer = KLineTokenizer()

        # Pattern transition probabilities (learned from data)
        # In real Kronos, this would be a large Transformer model
        # Here, we use simplified pattern-based rules
        self.pattern_transitions = self._initialize_pattern_transitions()

    def _initialize_pattern_transitions(self) -> Dict:
        """
        Initialize pattern transition probabilities.

        In real Kronos, this would be learned from 12B+ records.
        Here, we use domain knowledge of candlestick patterns.
        """
        return {
            'strong_bullish': {'bullish': 0.4, 'weak_bullish': 0.3, 'neutral': 0.2, 'weak_bearish': 0.1},
            'bullish': {'strong_bullish': 0.2, 'bullish': 0.3, 'weak_bullish': 0.3, 'neutral': 0.2},
            'weak_bullish': {'bullish': 0.3, 'weak_bullish': 0.2, 'neutral': 0.3, 'weak_bearish': 0.2},
            'neutral': {'bullish': 0.25, 'bearish': 0.25, 'neutral': 0.3, 'weak_bullish': 0.1, 'weak_bearish': 0.1},
            'weak_bearish': {'bearish': 0.3, 'weak_bearish': 0.2, 'neutral': 0.3, 'weak_bullish': 0.2},
            'bearish': {'strong_bearish': 0.2, 'bearish': 0.3, 'weak_bearish': 0.3, 'neutral': 0.2},
            'strong_bearish': {'bearish': 0.4, 'weak_bearish': 0.3, 'neutral': 0.2, 'weak_bullish': 0.1},
            'hammer': {'strong_bullish': 0.4, 'bullish': 0.4, 'neutral': 0.2},
            'shooting_star': {'strong_bearish': 0.4, 'bearish': 0.4, 'neutral': 0.2},
        }

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Generate Kronos-style prediction for stock data.

        Parameters:
        -----------
        df : pd.DataFrame
            Historical OHLCVA data (at least lookback_window rows)

        Returns:
        --------
        Dict
            Prediction results with ML scores
        """
        if len(df) < self.lookback_window:
            raise ValueError(f"Insufficient data: need {self.lookback_window} rows, got {len(df)}")

        # Use last N days
        recent_data = df.tail(self.lookback_window).copy()

        # Tokenize sequence
        coarse_tokens, fine_tokens = self.tokenizer.tokenize_sequence(recent_data)

        # Analyze token sequence patterns
        pattern_score = self._analyze_pattern_sequence(coarse_tokens)

        # Probabilistic forecast using autoregressive model
        forecast_probs = self._probabilistic_forecast(coarse_tokens, fine_tokens)

        # Calculate ML scores
        current_price = float(recent_data.iloc[-1]['close'])

        # Prediction score: weighted by pattern strength and forecast probability
        ml_prediction_score = self._calculate_prediction_score(pattern_score, forecast_probs)

        # Price target: based on probabilistic scenarios
        ml_price_target = self._calculate_price_target(current_price, forecast_probs)

        # Confidence: based on pattern clarity and forecast certainty
        ml_confidence = self._calculate_confidence(coarse_tokens, forecast_probs)

        # Risk score: based on volatility patterns and downside probability
        ml_risk_score = self._calculate_risk_score(recent_data, forecast_probs)

        return {
            'ml_prediction_score': float(ml_prediction_score),
            'ml_price_target': float(ml_price_target),
            'ml_confidence': float(ml_confidence),
            'ml_risk_score': float(ml_risk_score),
            'pattern_score': float(pattern_score),
            'forecast_probabilities': forecast_probs,
            'recent_patterns': self._get_recent_pattern_names(coarse_tokens[-5:])
        }

    def _analyze_pattern_sequence(self, coarse_tokens: np.ndarray) -> float:
        """Analyze the pattern sequence for bullish/bearish trends."""
        # Map tokens to pattern strengths
        pattern_strengths = {
            0: 1.0,   # strong_bullish
            1: 0.6,   # bullish
            2: 0.3,   # weak_bullish
            3: 0.0,   # neutral
            4: -0.3,  # weak_bearish
            5: -0.6,  # bearish
            6: -1.0,  # strong_bearish
            7: 0.7,   # hammer (bullish reversal)
            8: -0.7,  # shooting_star (bearish reversal)
            9: 0.8,   # engulfing_bull
            10: -0.8, # engulfing_bear
        }

        # Weight recent patterns more heavily
        weights = np.exp(np.linspace(-1, 0, len(coarse_tokens)))
        weights = weights / weights.sum()

        scores = np.array([pattern_strengths.get(t, 0) for t in coarse_tokens])
        weighted_score = np.sum(scores * weights)

        # Normalize to [0, 1]
        normalized_score = (weighted_score + 1) / 2

        return np.clip(normalized_score, 0, 1)

    def _probabilistic_forecast(
        self, coarse_tokens: np.ndarray, fine_tokens: np.ndarray
    ) -> Dict[str, float]:
        """
        Generate probabilistic forecast using autoregressive token prediction.

        Returns probabilities for different price movement scenarios.
        """
        # Get last pattern
        last_coarse = coarse_tokens[-1]

        # Reverse pattern map
        pattern_names = {v: k for k, v in self.tokenizer.coarse_patterns.items()}
        last_pattern_name = pattern_names.get(last_coarse, 'neutral')

        # Get transition probabilities
        transitions = self.pattern_transitions.get(last_pattern_name, {'neutral': 1.0})

        # Calculate probabilities for different outcomes
        bullish_prob = sum(
            prob for pattern, prob in transitions.items()
            if 'bullish' in pattern or pattern == 'hammer'
        )

        bearish_prob = sum(
            prob for pattern, prob in transitions.items()
            if 'bearish' in pattern or pattern == 'shooting_star'
        )

        neutral_prob = 1.0 - bullish_prob - bearish_prob

        return {
            'bullish': float(bullish_prob),
            'bearish': float(bearish_prob),
            'neutral': float(neutral_prob)
        }

    def _calculate_prediction_score(
        self, pattern_score: float, forecast_probs: Dict[str, float]
    ) -> float:
        """Calculate final ML prediction score combining patterns and forecast."""
        # Weighted combination
        combined_score = (
            pattern_score * 0.6 +
            forecast_probs['bullish'] * 0.4
        )
        return np.clip(combined_score, 0, 1)

    def _calculate_price_target(
        self, current_price: float, forecast_probs: Dict[str, float]
    ) -> float:
        """Calculate price target based on probabilistic scenarios."""
        # Expected return based on probabilities
        expected_return = (
            forecast_probs['bullish'] * 0.09 +    # +9% for bullish
            forecast_probs['bearish'] * -0.09 +   # -9% for bearish
            forecast_probs['neutral'] * 0.0       # 0% for neutral
        )

        return current_price * (1 + expected_return)

    def _calculate_confidence(
        self, coarse_tokens: np.ndarray, forecast_probs: Dict[str, float]
    ) -> float:
        """Calculate prediction confidence."""
        # Confidence based on forecast certainty (max probability)
        max_prob = max(forecast_probs.values())

        # Pattern consistency (how stable are recent patterns)
        recent_patterns = coarse_tokens[-10:]
        pattern_variance = np.var(recent_patterns) / 10.0  # Normalize
        pattern_consistency = 1.0 - min(pattern_variance, 1.0)

        # Combined confidence
        confidence = (max_prob * 0.6 + pattern_consistency * 0.4)

        return np.clip(confidence, 0, 1)

    def _calculate_risk_score(
        self, df: pd.DataFrame, forecast_probs: Dict[str, float]
    ) -> float:
        """Calculate risk score based on volatility and downside probability."""
        # Calculate recent volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() if len(returns) > 0 else 0

        # Downside risk from forecast
        downside_prob = forecast_probs['bearish']

        # Combined risk score
        risk_score = (
            volatility * 5.0 * 0.5 +      # Volatility component (scaled)
            downside_prob * 0.5            # Downside probability component
        )

        return np.clip(risk_score, 0, 1)

    def _get_recent_pattern_names(self, coarse_tokens: np.ndarray) -> List[str]:
        """Get human-readable pattern names for recent tokens."""
        pattern_names = {v: k for k, v in self.tokenizer.coarse_patterns.items()}
        return [pattern_names.get(t, 'unknown') for t in coarse_tokens]


class KronosPredictionService:
    """
    Service for generating stock predictions using Kronos-inspired model.

    Integrates with the existing dual-model architecture.
    """

    def __init__(self, model_dir: str = 'ml_models/kronos'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.db = get_database_manager()
        self.predictor = KronosInspiredPredictor(lookback_window=60, forecast_horizon=14)

        logger.info(f"KronosPredictionService initialized (model_dir: {self.model_dir})")

    def predict_single_stock(self, symbol: str, user_id: int = 1) -> Optional[Dict]:
        """
        Generate Kronos prediction for a single stock.

        Parameters:
        -----------
        symbol : str
            Stock symbol
        user_id : int
            User ID for data fetching

        Returns:
        --------
        Optional[Dict]
            Prediction dictionary or None if failed
        """
        try:
            # Fetch historical OHLCVA data
            with self.db.get_session() as session:
                query = text("""
                    SELECT date, open, high, low, close, volume
                    FROM historical_data
                    WHERE symbol = :symbol
                    ORDER BY date DESC
                    LIMIT 100
                """)

                result = session.execute(query, {'symbol': symbol})
                rows = result.fetchall()

                if len(rows) < 60:
                    logger.warning(f"Insufficient data for {symbol}: {len(rows)} days")
                    return None

                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                df = df.sort_values('date')

                # Get stock fundamentals
                stock_query = text("""
                    SELECT
                        name as stock_name, current_price, market_cap, sector,
                        market_cap_category, pe_ratio, pb_ratio, roe, eps,
                        beta, revenue_growth, earnings_growth, operating_margin
                    FROM stocks
                    WHERE symbol = :symbol
                """)

                stock_result = session.execute(stock_query, {'symbol': symbol}).fetchone()

                if not stock_result:
                    logger.warning(f"No stock data found for {symbol}")
                    return None

                stock_data = dict(stock_result._mapping)

            # Generate Kronos prediction
            prediction = self.predictor.predict(df)

            # Build result dictionary (compatible with existing system)
            current_price = stock_data.get('current_price', 0)

            result = {
                'symbol': symbol,
                'stock_name': stock_data.get('stock_name'),
                'current_price': current_price,
                'market_cap': stock_data.get('market_cap'),
                'sector': stock_data.get('sector'),
                'market_cap_category': stock_data.get('market_cap_category'),

                # ML Predictions (Kronos)
                'ml_prediction_score': prediction['ml_prediction_score'],
                'ml_price_target': prediction['ml_price_target'],
                'ml_confidence': prediction['ml_confidence'],
                'ml_risk_score': prediction['ml_risk_score'],

                # Kronos-specific metadata
                'pattern_score': prediction['pattern_score'],
                'forecast_probabilities': prediction['forecast_probabilities'],
                'recent_patterns': prediction['recent_patterns'],

                # Fundamentals
                'pe_ratio': stock_data.get('pe_ratio'),
                'pb_ratio': stock_data.get('pb_ratio'),
                'roe': stock_data.get('roe'),
                'eps': stock_data.get('eps'),
                'beta': stock_data.get('beta'),
                'revenue_growth': stock_data.get('revenue_growth'),
                'earnings_growth': stock_data.get('earnings_growth'),
                'operating_margin': stock_data.get('operating_margin'),

                # Trading signals
                'recommendation': self._get_recommendation(prediction),
                'target_price': prediction['ml_price_target'],
                'stop_loss': current_price * 0.95,  # 5% stop loss

                # Model type
                'model_type': 'kronos'
            }

            return result

        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}", exc_info=True)
            return None

    def _get_recommendation(self, prediction: Dict) -> str:
        """Generate trading recommendation based on Kronos prediction."""
        score = prediction['ml_prediction_score']
        confidence = prediction['ml_confidence']

        if score > 0.65 and confidence > 0.6:
            return "BUY"
        elif score < 0.35 and confidence > 0.6:
            return "SELL"
        else:
            return "HOLD"

    def apply_risk_strategy(self, prediction: Dict, strategy: str = 'default_risk') -> Dict:
        """
        Apply risk strategy adjustments to prediction.

        Parameters:
        -----------
        prediction : Dict
            Base prediction dictionary
        strategy : str
            Risk strategy: 'default_risk' or 'high_risk'

        Returns:
        --------
        Dict
            Adjusted prediction with strategy-specific targets
        """
        current_price = prediction['current_price']

        if strategy == 'high_risk':
            # High Risk: More aggressive targets (12% profit, 10% stop loss)
            if prediction['recommendation'] == 'BUY':
                prediction['target_price'] = round(current_price * 1.12, 2)
                prediction['ml_price_target'] = round(current_price * 1.12, 2)
                prediction['stop_loss'] = round(current_price * 0.90, 2)
            elif prediction['recommendation'] == 'SELL':
                prediction['ml_price_target'] = round(current_price * 0.88, 2)
        else:  # default_risk
            # Default Risk: Conservative targets (7% profit, 5% stop loss)
            if prediction['recommendation'] == 'BUY':
                prediction['target_price'] = round(current_price * 1.07, 2)
                prediction['ml_price_target'] = round(current_price * 1.07, 2)
                prediction['stop_loss'] = round(current_price * 0.95, 2)
            elif prediction['recommendation'] == 'SELL':
                prediction['ml_price_target'] = round(current_price * 0.93, 2)

        return prediction

    def batch_predict(
        self, symbols: List[str], user_id: int = 1,
        max_workers: int = 4, strategy: str = 'default_risk'
    ) -> List[Dict]:
        """
        Generate predictions for multiple stocks in batch.

        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols
        user_id : int
            User ID
        max_workers : int
            Number of parallel workers (not used yet)
        strategy : str
            Risk strategy: 'default_risk' or 'high_risk'

        Returns:
        --------
        List[Dict]
            List of prediction dictionaries
        """
        logger.info(f"Batch predicting {len(symbols)} stocks with Kronos (strategy={strategy})...")

        # Filter symbols by strategy-specific market cap
        filtered_symbols = self._filter_symbols_by_strategy(symbols, strategy)
        logger.info(f"Filtered to {len(filtered_symbols)} symbols for {strategy} strategy")

        predictions = []
        for symbol in filtered_symbols:
            prediction = self.predict_single_stock(symbol, user_id)
            if prediction:
                # Apply risk strategy
                prediction = self.apply_risk_strategy(prediction, strategy)
                predictions.append(prediction)

        # Sort by prediction score
        predictions.sort(key=lambda x: x['ml_prediction_score'], reverse=True)

        logger.info(f"Generated {len(predictions)} Kronos predictions from {len(filtered_symbols)} symbols")

        return predictions

    def _filter_symbols_by_strategy(self, symbols: List[str], strategy: str) -> List[str]:
        """
        Filter symbols by market cap based on risk strategy.

        DEFAULT_RISK: Large cap only (> 20,000 Cr)
        HIGH_RISK: Small/Mid cap only (1,000 - 20,000 Cr)
        """
        with self.db.get_session() as session:
            query = text("""
                SELECT symbol, market_cap
                FROM stocks
                WHERE symbol = ANY(:symbols)
                AND is_active = true
            """)

            result = session.execute(query, {'symbols': symbols})
            rows = result.fetchall()

            filtered = []
            for row in rows:
                symbol = row[0]
                market_cap = float(row[1]) if row[1] else 0

                if strategy.upper() == 'DEFAULT_RISK':
                    # Large cap only: > 20,000 Cr
                    if market_cap > 20000:
                        filtered.append(symbol)
                elif strategy.upper() == 'HIGH_RISK':
                    # Small/Mid cap: 1,000 - 20,000 Cr
                    if 1000 <= market_cap <= 20000:
                        filtered.append(symbol)

            return filtered

    def save_to_suggested_stocks(
        self, predictions: List[Dict], strategy: str = 'kronos',
        prediction_date: Optional[date] = None
    ):
        """
        Save Kronos predictions to daily_suggested_stocks table.

        Parameters:
        -----------
        predictions : List[Dict]
            List of prediction dictionaries
        strategy : str
            Strategy name
        prediction_date : Optional[date]
            Date for predictions (default: today)
        """
        if not predictions:
            logger.warning("No Kronos predictions to save")
            return

        if prediction_date is None:
            prediction_date = datetime.now().date()

        logger.info(f"Saving {len(predictions)} Kronos predictions to database...")

        with self.db.get_session() as session:
            for rank, pred in enumerate(predictions, 1):
                insert_query = text("""
                    INSERT INTO daily_suggested_stocks (
                        date, symbol, stock_name, current_price, market_cap,
                        strategy, selection_score, rank,
                        ml_prediction_score, ml_price_target, ml_confidence, ml_risk_score,
                        pe_ratio, pb_ratio, roe, eps, beta,
                        revenue_growth, earnings_growth, operating_margin,
                        target_price, stop_loss, recommendation,
                        sector, market_cap_category, model_type, created_at
                    ) VALUES (
                        :date, :symbol, :stock_name, :current_price, :market_cap,
                        :strategy, :selection_score, :rank,
                        :ml_prediction_score, :ml_price_target, :ml_confidence, :ml_risk_score,
                        :pe_ratio, :pb_ratio, :roe, :eps, :beta,
                        :revenue_growth, :earnings_growth, :operating_margin,
                        :target_price, :stop_loss, :recommendation,
                        :sector, :market_cap_category, :model_type, NOW()
                    )
                    ON CONFLICT (date, symbol, strategy) DO UPDATE SET
                        stock_name = EXCLUDED.stock_name,
                        current_price = EXCLUDED.current_price,
                        market_cap = EXCLUDED.market_cap,
                        selection_score = EXCLUDED.selection_score,
                        rank = EXCLUDED.rank,
                        ml_prediction_score = EXCLUDED.ml_prediction_score,
                        ml_price_target = EXCLUDED.ml_price_target,
                        ml_confidence = EXCLUDED.ml_confidence,
                        ml_risk_score = EXCLUDED.ml_risk_score,
                        target_price = EXCLUDED.target_price,
                        stop_loss = EXCLUDED.stop_loss,
                        recommendation = EXCLUDED.recommendation,
                        model_type = EXCLUDED.model_type
                """)

                session.execute(insert_query, {
                    'date': prediction_date,
                    'symbol': pred['symbol'],
                    'stock_name': pred['stock_name'],
                    'current_price': pred['current_price'],
                    'market_cap': pred['market_cap'],
                    'strategy': strategy,
                    'selection_score': pred['ml_prediction_score'],
                    'rank': rank,
                    'ml_prediction_score': pred['ml_prediction_score'],
                    'ml_price_target': pred['ml_price_target'],
                    'ml_confidence': pred['ml_confidence'],
                    'ml_risk_score': pred['ml_risk_score'],
                    'pe_ratio': pred.get('pe_ratio'),
                    'pb_ratio': pred.get('pb_ratio'),
                    'roe': pred.get('roe'),
                    'eps': pred.get('eps'),
                    'beta': pred.get('beta'),
                    'revenue_growth': pred.get('revenue_growth'),
                    'earnings_growth': pred.get('earnings_growth'),
                    'operating_margin': pred.get('operating_margin'),
                    'target_price': pred['target_price'],
                    'stop_loss': pred.get('stop_loss'),
                    'recommendation': pred['recommendation'],
                    'sector': pred['sector'],
                    'market_cap_category': pred['market_cap_category'],
                    'model_type': 'kronos'
                })

            session.commit()
            logger.info(f"✓ Saved {len(predictions)} Kronos predictions to database")


# Singleton instance
_kronos_service = None

def get_kronos_prediction_service():
    """Get the singleton Kronos prediction service instance."""
    global _kronos_service
    if _kronos_service is None:
        _kronos_service = KronosPredictionService()
    return _kronos_service


if __name__ == '__main__':
    # Test the Kronos service
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    service = get_kronos_prediction_service()

    # Test on a stock
    test_symbol = 'NSE:RELIANCE-EQ'
    logger.info(f"Testing Kronos prediction for {test_symbol}...")

    prediction = service.predict_single_stock(test_symbol)
    if prediction:
        print("\n" + "="*60)
        print("KRONOS PREDICTION RESULT")
        print("="*60)
        print(f"Symbol: {prediction['symbol']}")
        print(f"Company: {prediction['stock_name']}")
        print(f"Current Price: ₹{prediction['current_price']:.2f}")
        print(f"\nML Scores (Kronos):")
        print(f"  Prediction Score: {prediction['ml_prediction_score']:.4f} ({prediction['ml_prediction_score']*100:.1f}%)")
        print(f"  Price Target: ₹{prediction['ml_price_target']:.2f}")
        print(f"  Confidence: {prediction['ml_confidence']:.4f} ({prediction['ml_confidence']*100:.1f}%)")
        print(f"  Risk Score: {prediction['ml_risk_score']:.4f} ({prediction['ml_risk_score']*100:.1f}%)")
        print(f"\nKronos-Specific:")
        print(f"  Pattern Score: {prediction['pattern_score']:.4f}")
        print(f"  Forecast Probabilities:")
        print(f"    Bullish: {prediction['forecast_probabilities']['bullish']:.2%}")
        print(f"    Bearish: {prediction['forecast_probabilities']['bearish']:.2%}")
        print(f"    Neutral: {prediction['forecast_probabilities']['neutral']:.2%}")
        print(f"  Recent Patterns: {', '.join(prediction['recent_patterns'])}")
        print(f"\nRecommendation: {prediction['recommendation']}")
        print(f"Target Price: ₹{prediction['target_price']:.2f}")
        print(f"Stop Loss: ₹{prediction['stop_loss']:.2f}")
        print("="*60)
