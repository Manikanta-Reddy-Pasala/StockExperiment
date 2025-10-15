"""
Kronos Financial Foundation Model Prediction Service

This service uses the REAL Kronos foundation model from:
https://github.com/shiyu-coder/Kronos

Kronos is a family of decoder-only foundation models, pre-trained on 12B+
financial K-line records from 45 global exchanges. It uses:
- Specialized tokenizer for OHLCV data
- Pre-trained Transformer models (mini, small, base)
- Autoregressive probabilistic forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from pathlib import Path
import logging
from sqlalchemy import text
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from src.models.database import get_database_manager

# Import the REAL Kronos model
import sys
sys.path.insert(0, str(Path(__file__).parent / 'kronos_model'))

try:
    from src.services.ml.kronos_model import Kronos, KronosTokenizer, KronosPredictor
    KRONOS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Failed to import Kronos model: {e}")
    logging.warning("Kronos prediction service will not be available.")
    KRONOS_AVAILABLE = False

logger = logging.getLogger(__name__)


class KronosPredictionService:
    """
    Service for generating stock predictions using the real Kronos foundation model.

    Integrates with the existing dual-model architecture.
    """

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-small",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "cpu",
        max_context: int = 512,
        lookback_window: int = 200,  # Reduced from 400 (database has max 255 days)
        forecast_horizon: int = 14
    ):
        """
        Initialize Kronos prediction service with pre-trained models.

        Parameters:
        -----------
        model_name : str
            Hugging Face model name (Kronos-mini, Kronos-small, or Kronos-base)
        tokenizer_name : str
            Hugging Face tokenizer name
        device : str
            Device to run model on ('cpu' or 'cuda:0')
        max_context : int
            Maximum context length (512 for small/base, 2048 for mini)
        lookback_window : int
            Number of historical days to use for prediction
        forecast_horizon : int
            Number of days to forecast
        """
        if not KRONOS_AVAILABLE:
            raise RuntimeError("Kronos model not available. Check imports.")

        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.max_context = max_context
        self.lookback_window = min(lookback_window, max_context)  # Don't exceed max_context
        self.forecast_horizon = forecast_horizon

        self.db = get_database_manager()

        # Lazy loading of model and predictor
        self._tokenizer = None
        self._model = None
        self._predictor = None

        logger.info(
            f"KronosPredictionService initialized (model={model_name}, "
            f"device={device}, lookback={self.lookback_window}, forecast={forecast_horizon})"
        )

    @property
    def tokenizer(self):
        """Lazy load tokenizer from Hugging Face."""
        if self._tokenizer is None:
            logger.info(f"Loading Kronos tokenizer: {self.tokenizer_name}")
            try:
                self._tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
                logger.info("✓ Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                raise
        return self._tokenizer

    @property
    def model(self):
        """Lazy load Kronos model from Hugging Face."""
        if self._model is None:
            logger.info(f"Loading Kronos model: {self.model_name}")
            try:
                self._model = Kronos.from_pretrained(self.model_name)
                logger.info("✓ Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        return self._model

    @property
    def predictor(self):
        """Lazy load Kronos predictor."""
        if self._predictor is None:
            logger.info("Initializing Kronos predictor...")
            try:
                self._predictor = KronosPredictor(
                    self.model,
                    self.tokenizer,
                    device=self.device,
                    max_context=self.max_context
                )
                logger.info("✓ Predictor initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize predictor: {e}")
                raise
        return self._predictor

    def predict_single_stock(
        self,
        symbol: str,
        user_id: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1
    ) -> Optional[Dict]:
        """
        Generate Kronos prediction for a single stock.

        Parameters:
        -----------
        symbol : str
            Stock symbol (e.g., 'NSE:RELIANCE-EQ')
        user_id : int
            User ID for data fetching
        temperature : float
            Sampling temperature (higher = more random)
        top_p : float
            Nucleus sampling probability
        sample_count : int
            Number of forecast paths to generate and average

        Returns:
        --------
        Optional[Dict]
            Prediction dictionary or None if failed
        """
        try:
            # Fetch historical OHLCV data
            with self.db.get_session() as session:
                query = text("""
                    SELECT date, open, high, low, close, volume
                    FROM historical_data
                    WHERE symbol = :symbol
                    ORDER BY date DESC
                    LIMIT :limit
                """)

                result = session.execute(query, {
                    'symbol': symbol,
                    'limit': self.lookback_window + 50  # Extra buffer
                })
                rows = result.fetchall()

                if len(rows) < self.lookback_window:
                    logger.warning(
                        f"Insufficient data for {symbol}: {len(rows)} days "
                        f"(need {self.lookback_window})"
                    )
                    return None

                # Convert to DataFrame
                df = pd.DataFrame(
                    rows,
                    columns=['date', 'open', 'high', 'low', 'close', 'volume']
                )
                df = df.sort_values('date').reset_index(drop=True)

                # Prepare data for Kronos
                # Use last N days as context
                x_df = df.tail(self.lookback_window)[['open', 'high', 'low', 'close', 'volume']].copy()

                # Don't add 'amount' column - Kronos will calculate it automatically:
                # amount = volume × mean(open, high, low, close)
                # See kronos.py line 496

                # Generate timestamps (must be Series, not DatetimeIndex)
                x_timestamp = pd.Series(pd.to_datetime(df.tail(self.lookback_window)['date']).values)

                # Generate future timestamps for forecast
                last_date = x_timestamp.iloc[-1]
                y_timestamp = pd.Series(pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=self.forecast_horizon,
                    freq='B'  # Business days
                ))

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
            logger.debug(f"Predicting {symbol} with Kronos (lookback={len(x_df)}, forecast={self.forecast_horizon})")

            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=self.forecast_horizon,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False
            )

            # Calculate ML scores from predictions
            current_price = float(stock_data.get('current_price', x_df['close'].iloc[-1]))
            predicted_prices = pred_df['close'].values

            # ML Prediction Score: Based on upward trend strength
            price_changes = np.diff(predicted_prices)
            positive_changes = price_changes[price_changes > 0].sum()
            negative_changes = abs(price_changes[price_changes < 0].sum())
            total_changes = positive_changes + negative_changes

            if total_changes > 0:
                ml_prediction_score = positive_changes / total_changes
            else:
                ml_prediction_score = 0.5  # Neutral if no change

            # Price Target: Average of last 5 predicted prices
            ml_price_target = float(predicted_prices[-5:].mean())

            # Confidence: Based on prediction consistency (lower std = higher confidence)
            price_std = predicted_prices.std()
            price_mean = predicted_prices.mean()
            coefficient_of_variation = price_std / price_mean if price_mean > 0 else 1.0
            ml_confidence = float(max(0.0, min(1.0, 1.0 - coefficient_of_variation)))

            # Risk Score: Based on volatility of predictions
            returns = np.diff(predicted_prices) / predicted_prices[:-1]
            ml_risk_score = float(min(1.0, abs(returns).std() * 10))

            # Build result dictionary (compatible with existing system)
            result = {
                'symbol': symbol,
                'stock_name': stock_data.get('stock_name'),
                'current_price': current_price,
                'market_cap': stock_data.get('market_cap'),
                'sector': stock_data.get('sector'),
                'market_cap_category': stock_data.get('market_cap_category'),

                # ML Predictions (Kronos)
                'ml_prediction_score': float(ml_prediction_score),
                'ml_price_target': float(ml_price_target),
                'ml_confidence': float(ml_confidence),
                'ml_risk_score': float(ml_risk_score),

                # Kronos-specific metadata
                'forecast_prices': predicted_prices.tolist(),
                'forecast_dates': y_timestamp.dt.strftime('%Y-%m-%d').tolist(),
                'model_params': {
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'lookback': self.lookback_window
                },

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
                'recommendation': self._get_recommendation(ml_prediction_score, ml_confidence),
                'target_price': ml_price_target,
                'stop_loss': current_price * 0.95,  # 5% stop loss

                # Model type
                'model_type': 'kronos'
            }

            return result

        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}", exc_info=True)
            return None

    def _get_recommendation(self, prediction_score: float, confidence: float) -> str:
        """Generate trading recommendation based on Kronos prediction."""
        if prediction_score > 0.65 and confidence > 0.6:
            return "BUY"
        elif prediction_score < 0.35 and confidence > 0.6:
            return "SELL"
        else:
            return "HOLD"

    def apply_risk_strategy(
        self,
        prediction: Dict,
        strategy: str = 'default_risk'
    ) -> Dict:
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
        self,
        symbols: List[str],
        user_id: int = 1,
        max_workers: int = 1,  # Sequential for now (can parallelize later)
        strategy: str = 'default_risk',
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1
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
            Number of parallel workers (not implemented yet)
        strategy : str
            Risk strategy: 'default_risk' or 'high_risk'
        temperature : float
            Sampling temperature
        top_p : float
            Nucleus sampling probability
        sample_count : int
            Number of forecast paths to average

        Returns:
        --------
        List[Dict]
            List of prediction dictionaries
        """
        logger.info(
            f"Batch predicting {len(symbols)} stocks with Kronos "
            f"(strategy={strategy})..."
        )

        # Skip additional filtering - the saga orchestrator already applied
        # comprehensive model-specific + strategy-specific filters before calling us
        # No need to filter again here (would cause double filtering and reject everything)
        filtered_symbols = symbols
        logger.info(f"Processing {len(filtered_symbols)} pre-filtered symbols from saga")

        predictions = []
        success_count = 0
        fail_count = 0

        for i, symbol in enumerate(filtered_symbols, 1):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(filtered_symbols)} symbols processed ({success_count} successful)")

            try:
                prediction = self.predict_single_stock(
                    symbol,
                    user_id,
                    temperature=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )

                if prediction:
                    # Apply risk strategy
                    prediction = self.apply_risk_strategy(prediction, strategy)
                    predictions.append(prediction)
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                logger.warning(f"Failed to predict {symbol}: {e}")
                fail_count += 1

        # Sort by prediction score
        predictions.sort(key=lambda x: x['ml_prediction_score'], reverse=True)

        logger.info(
            f"Generated {len(predictions)} Kronos predictions from "
            f"{len(filtered_symbols)} symbols ({success_count} success, {fail_count} failed)"
        )

        return predictions

    def _filter_symbols_by_strategy(
        self,
        symbols: List[str],
        strategy: str
    ) -> List[str]:
        """
        Filter symbols by comprehensive quality criteria based on risk strategy.

        DEFAULT_RISK: Large cap only (> 20,000 Cr) with strict quality filters
        HIGH_RISK: Small/Mid cap (1,000 - 20,000 Cr) with moderate quality filters
        """
        with self.db.get_session() as session:
            query = text("""
                SELECT
                    symbol, market_cap, current_price, volume,
                    pe_ratio, roe, debt_to_equity,
                    operating_margin, avg_volume_30d
                FROM stocks
                WHERE symbol = ANY(:symbols)
                AND is_active = true
                AND current_price IS NOT NULL
                AND volume IS NOT NULL
            """)

            result = session.execute(query, {'symbols': symbols})
            rows = result.fetchall()

            filtered = []
            for row in rows:
                symbol = row[0]
                market_cap = float(row[1]) if row[1] else 0
                current_price = float(row[2]) if row[2] else 0
                volume = int(row[3]) if row[3] else 0
                pe_ratio = float(row[4]) if row[4] else None
                roe = float(row[5]) if row[5] else None
                debt_to_equity = float(row[6]) if row[6] else None
                operating_margin = float(row[7]) if row[7] else None
                avg_volume_30d = int(row[8]) if row[8] else 0

                # Calculate daily turnover
                daily_turnover = current_price * volume if current_price and volume else 0

                if strategy.upper() == 'DEFAULT_RISK':
                    # STRICT FILTERS for conservative strategy
                    # 1. Market Cap: Large cap only (> 20,000 Cr)
                    if market_cap <= 20000:
                        continue

                    # 2. Price Range: Avoid penny stocks and very expensive stocks
                    if not (100 <= current_price <= 10000):
                        continue

                    # 3. Liquidity: High volume and turnover required
                    if volume < 50000 or daily_turnover < 50000000:  # 5 Cr minimum turnover
                        continue

                    # 4. PE Ratio: Reasonable valuation (avoid overvalued)
                    if pe_ratio is not None and (pe_ratio < 5 or pe_ratio > 40):
                        continue

                    # 5. ROE: Minimum profitability (if available)
                    if roe is not None and roe < 5:
                        continue

                    # 6. Debt: Avoid highly leveraged companies
                    if debt_to_equity is not None and debt_to_equity > 2.0:
                        continue

                    # 7. Consistent Volume: Avoid sudden spikes (manipulation)
                    if avg_volume_30d > 0:
                        volume_ratio = volume / avg_volume_30d
                        if volume_ratio > 5.0 or volume_ratio < 0.2:  # Too much deviation
                            continue

                    filtered.append(symbol)

                elif strategy.upper() == 'HIGH_RISK':
                    # MODERATE FILTERS for aggressive strategy
                    # 1. Market Cap: Small/Mid cap (1,000 - 20,000 Cr)
                    if not (1000 <= market_cap <= 20000):
                        continue

                    # 2. Price Range: Flexible but avoid extreme penny stocks
                    if current_price < 20 or current_price > 5000:
                        continue

                    # 3. Liquidity: Minimum volume for tradeability
                    if volume < 10000 or daily_turnover < 10000000:  # 1 Cr minimum turnover
                        continue

                    # 4. PE Ratio: More lenient (growth stocks can have high PE)
                    if pe_ratio is not None and pe_ratio > 100:  # Only exclude extreme cases
                        continue

                    # 5. Operating Margin: Minimum business viability
                    if operating_margin is not None and operating_margin < -20:  # Allow losses but not huge
                        continue

                    # 6. Debt: More lenient for growth stocks
                    if debt_to_equity is not None and debt_to_equity > 5.0:
                        continue

                    filtered.append(symbol)

            return filtered

    def save_to_suggested_stocks(
        self,
        predictions: List[Dict],
        strategy: str = 'kronos',
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
                    ON CONFLICT (date, symbol, strategy, model_type) DO UPDATE SET
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


def get_kronos_prediction_service(
    model_name: str = "NeoQuasar/Kronos-small",
    device: str = "cpu"
):
    """
    Get the singleton Kronos prediction service instance.

    Parameters:
    -----------
    model_name : str
        Hugging Face model name (default: Kronos-small)
    device : str
        Device to run on ('cpu' or 'cuda:0')
    """
    global _kronos_service
    if _kronos_service is None:
        _kronos_service = KronosPredictionService(
            model_name=model_name,
            device=device
        )
    return _kronos_service


if __name__ == '__main__':
    # Test the Kronos service
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if not KRONOS_AVAILABLE:
        print("ERROR: Kronos model not available. Cannot run test.")
        exit(1)

    service = get_kronos_prediction_service(device="cpu")

    # Test on a stock
    test_symbol = 'NSE:RELIANCE-EQ'
    logger.info(f"Testing Kronos prediction for {test_symbol}...")

    prediction = service.predict_single_stock(test_symbol)
    if prediction:
        print("\n" + "="*60)
        print("KRONOS PREDICTION RESULT (Real Foundation Model)")
        print("="*60)
        print(f"Symbol: {prediction['symbol']}")
        print(f"Company: {prediction['stock_name']}")
        print(f"Current Price: ₹{prediction['current_price']:.2f}")
        print(f"\nML Scores (Kronos):")
        print(f"  Prediction Score: {prediction['ml_prediction_score']:.4f} ({prediction['ml_prediction_score']*100:.1f}%)")
        print(f"  Price Target: ₹{prediction['ml_price_target']:.2f}")
        print(f"  Confidence: {prediction['ml_confidence']:.4f} ({prediction['ml_confidence']*100:.1f}%)")
        print(f"  Risk Score: {prediction['ml_risk_score']:.4f} ({prediction['ml_risk_score']*100:.1f}%)")
        print(f"\nForecast (next {len(prediction['forecast_prices'])} days):")
        for date, price in zip(prediction['forecast_dates'][:5], prediction['forecast_prices'][:5]):
            print(f"  {date}: ₹{price:.2f}")
        print(f"  ...")
        print(f"\nRecommendation: {prediction['recommendation']}")
        print(f"Target Price: ₹{prediction['target_price']:.2f}")
        print(f"Stop Loss: ₹{prediction['stop_loss']:.2f}")
        print("="*60)
