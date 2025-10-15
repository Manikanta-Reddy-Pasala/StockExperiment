"""
Raw OHLCV LSTM Prediction Service

This service generates stock suggestions using the raw OHLCV LSTM model
with triple barrier labeling, as an alternative to the traditional
feature-engineered ensemble approach.

Features:
- Batch prediction on multiple stocks
- Integration with daily_suggested_stocks table
- Consistent API with traditional predictor
- Performance tracking and comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, date, timedelta
import logging
from pathlib import Path

from src.models.database import get_database_manager
from sqlalchemy import text
from .raw_ohlcv_lstm import RawOHLCVLSTM
from .data_service import get_raw_ohlcv_data
from .triple_barrier_labeling import TripleBarrierLabeler

logger = logging.getLogger(__name__)


class RawLSTMPredictionService:
    """
    Service for generating stock predictions using Raw OHLCV LSTM model.

    This provides an alternative to the traditional feature-engineered approach,
    allowing side-by-side comparison of both methodologies.
    """

    def __init__(self, model_dir: str = 'ml_models/raw_ohlcv_lstm'):
        self.model_dir = Path(model_dir)
        self.db = get_database_manager()
        self.model_cache = {}  # Cache loaded models by symbol

        # Triple barrier configuration (from research)
        self.labeler = TripleBarrierLabeler(
            upper_barrier=9.0,
            lower_barrier=9.0,
            time_horizon=29
        )

        logger.info(f"RawLSTMPredictionService initialized (model_dir: {self.model_dir})")

    def get_or_load_model(self, symbol: str) -> Optional[RawOHLCVLSTM]:
        """
        Load a trained model for the symbol, using cache if available.

        Parameters:
        -----------
        symbol : str
            Stock symbol

        Returns:
        --------
        Optional[RawOHLCVLSTM]
            Loaded model or None if not found
        """
        # Check cache first
        if symbol in self.model_cache:
            return self.model_cache[symbol]

        # Try to load from disk
        model_path = self.model_dir / symbol
        if not model_path.exists():
            logger.warning(f"No trained model found for {symbol} at {model_path}")
            return None

        try:
            model = RawOHLCVLSTM()
            model.load(str(model_path))
            self.model_cache[symbol] = model
            logger.info(f"Loaded model for {symbol}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")
            return None

    def predict_single_stock(self, symbol: str, user_id: int = 1) -> Optional[Dict]:
        """
        Generate prediction for a single stock.

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
            # Load model
            model = self.get_or_load_model(symbol)
            if model is None:
                logger.warning(f"No model available for {symbol}, skipping")
                return None

            # Get recent OHLCV data
            df = get_raw_ohlcv_data(symbol, period='3M', user_id=user_id)

            if len(df) < 150:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} days")
                return None

            # Make prediction
            probabilities = model.predict(df.tail(150), return_probabilities=True)

            if len(probabilities) == 0:
                logger.warning(f"No predictions generated for {symbol}")
                return None

            # Get most recent prediction
            latest_pred_probs = probabilities[-1]

            # Predicted class: 0=Loss, 1=Neutral, 2=Profit
            predicted_class = np.argmax(latest_pred_probs)

            # Get stock fundamentals from database
            with self.db.get_session() as session:
                query = text("""
                    SELECT
                        name as stock_name, current_price, market_cap, sector,
                        market_cap_category, pe_ratio, pb_ratio, roe, eps,
                        beta, revenue_growth, earnings_growth, operating_margin
                    FROM stocks
                    WHERE symbol = :symbol
                """)

                result = session.execute(query, {'symbol': symbol}).fetchone()

                if not result:
                    logger.warning(f"No stock data found for {symbol}")
                    return None

                stock_data = dict(result._mapping)

            # Calculate ML scores based on probabilities
            # Profit probability as prediction score
            ml_prediction_score = float(latest_pred_probs[2])  # Profit class probability

            # Confidence = max probability
            ml_confidence = float(np.max(latest_pred_probs))

            # Risk score = Loss probability
            ml_risk_score = float(latest_pred_probs[0])

            # Price target based on triple barrier
            current_price = stock_data.get('current_price', 0)
            if predicted_class == 2:  # Profit
                ml_price_target = current_price * 1.09  # +9% from barrier
            elif predicted_class == 0:  # Loss
                ml_price_target = current_price * 0.91  # -9% from barrier
            else:  # Neutral
                ml_price_target = current_price

            # Recommendation
            if predicted_class == 2 and ml_confidence > 0.6:
                recommendation = "BUY"
            elif predicted_class == 0 and ml_confidence > 0.6:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"

            # Build prediction result
            prediction = {
                'symbol': symbol,
                'stock_name': stock_data.get('stock_name'),
                'current_price': current_price,
                'market_cap': stock_data.get('market_cap'),
                'sector': stock_data.get('sector'),
                'market_cap_category': stock_data.get('market_cap_category'),

                # ML Predictions
                'ml_prediction_score': round(ml_prediction_score, 4),
                'ml_price_target': round(ml_price_target, 2),
                'ml_confidence': round(ml_confidence, 4),
                'ml_risk_score': round(ml_risk_score, 4),

                # Predicted class info
                'predicted_class': int(predicted_class),
                'predicted_class_name': ['Loss', 'Neutral', 'Profit'][predicted_class],
                'class_probabilities': {
                    'loss': round(float(latest_pred_probs[0]), 4),
                    'neutral': round(float(latest_pred_probs[1]), 4),
                    'profit': round(float(latest_pred_probs[2]), 4)
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
                'recommendation': recommendation,
                'target_price': round(ml_price_target, 2),
                'stop_loss': round(current_price * 0.91, 2) if predicted_class == 2 else None,

                # Model type
                'model_type': 'raw_lstm'
            }

            return prediction

        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}")
            return None

    def apply_risk_strategy(self, prediction: Dict, strategy: str = 'default_risk') -> Dict:
        """
        Apply risk strategy adjustments to prediction (default_risk or high_risk).

        Parameters:
        -----------
        prediction : Dict
            Base prediction dictionary
        strategy : str
            Risk strategy: 'default_risk' or 'high_risk'

        Returns:
        --------
        Dict
            Adjusted prediction with strategy-specific targets and stop loss
        """
        current_price = prediction['current_price']
        predicted_class = prediction['predicted_class']

        if strategy == 'high_risk':
            # High Risk: More aggressive targets (12% profit target, 5% stop loss)
            if predicted_class == 2:  # Profit prediction
                prediction['target_price'] = round(current_price * 1.12, 2)  # 12% target
                prediction['ml_price_target'] = round(current_price * 1.12, 2)
                prediction['stop_loss'] = round(current_price * 0.95, 2)  # 5% stop loss
            elif predicted_class == 0:  # Loss prediction
                prediction['ml_price_target'] = round(current_price * 0.88, 2)  # -12% target
                prediction['target_price'] = round(current_price * 0.88, 2)
                prediction['stop_loss'] = None
            else:  # Neutral
                prediction['ml_price_target'] = current_price
                prediction['target_price'] = current_price
                prediction['stop_loss'] = None

        else:  # default_risk (conservative)
            # Default Risk: Conservative targets (7% profit target, 3% stop loss)
            if predicted_class == 2:  # Profit prediction
                prediction['target_price'] = round(current_price * 1.07, 2)  # 7% target
                prediction['ml_price_target'] = round(current_price * 1.07, 2)
                prediction['stop_loss'] = round(current_price * 0.97, 2)  # 3% stop loss
            elif predicted_class == 0:  # Loss prediction
                prediction['ml_price_target'] = round(current_price * 0.93, 2)  # -7% target
                prediction['target_price'] = round(current_price * 0.93, 2)
                prediction['stop_loss'] = None
            else:  # Neutral
                prediction['ml_price_target'] = current_price
                prediction['target_price'] = current_price
                prediction['stop_loss'] = None

        return prediction

    def batch_predict(self, symbols: List[str], user_id: int = 1, max_workers: int = 4, strategy: str = 'default_risk') -> List[Dict]:
        """
        Generate predictions for multiple stocks in batch.

        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols
        user_id : int
            User ID for data fetching
        max_workers : int
            Number of parallel workers
        strategy : str
            Risk strategy: 'default_risk' or 'high_risk'

        Returns:
        --------
        List[Dict]
            List of prediction dictionaries
        """
        logger.info(f"Batch predicting {len(symbols)} stocks with strategy={strategy}...")

        # FILTER SYMBOLS BY MARKET CAP BEFORE PREDICTION (strategy-specific)
        filtered_symbols = self._filter_symbols_by_strategy(symbols, strategy)
        logger.info(f"Filtered to {len(filtered_symbols)} symbols for {strategy} strategy")

        predictions = []
        for symbol in filtered_symbols:
            prediction = self.predict_single_stock(symbol, user_id)
            if prediction:
                # Apply risk strategy adjustments
                prediction = self.apply_risk_strategy(prediction, strategy)
                predictions.append(prediction)

        logger.info(f"Generated {len(predictions)} predictions from {len(filtered_symbols)} symbols")

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
                    # Small/Mid cap only: 1,000 - 20,000 Cr (exclude large cap)
                    if 1000 <= market_cap <= 20000:
                        filtered.append(symbol)

            return filtered

    def save_to_suggested_stocks(self, predictions: List[Dict], strategy: str = 'raw_lstm', prediction_date: Optional[date] = None):
        """
        Save predictions to daily_suggested_stocks table.

        Parameters:
        -----------
        predictions : List[Dict]
            List of prediction dictionaries
        strategy : str
            Strategy name (default: 'raw_lstm')
        prediction_date : Optional[date]
            Date for predictions (default: today)
        """
        if not predictions:
            logger.warning("No predictions to save")
            return

        if prediction_date is None:
            prediction_date = datetime.now().date()

        logger.info(f"Saving {len(predictions)} raw LSTM predictions to database...")

        with self.db.get_session() as session:
            for rank, pred in enumerate(predictions, 1):
                # Insert or update
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
                    'model_type': 'raw_lstm'
                })

            session.commit()
            logger.info(f"✓ Saved {len(predictions)} predictions to database")


# Singleton instance
_prediction_service = None

def get_raw_lstm_prediction_service():
    """Get the singleton raw LSTM prediction service instance."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = RawLSTMPredictionService()
    return _prediction_service


if __name__ == '__main__':
    # Test the service
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    service = get_raw_lstm_prediction_service()

    # Test on a stock
    test_symbol = 'NSE:ADANIPOWER-EQ'
    logger.info(f"Testing prediction for {test_symbol}...")

    prediction = service.predict_single_stock(test_symbol)
    if prediction:
        print("\nPrediction Result:")
        print(f"  Symbol: {prediction['symbol']}")
        print(f"  Prediction Score: {prediction['ml_prediction_score']:.4f}")
        print(f"  Price Target: ₹{prediction['ml_price_target']:.2f}")
        print(f"  Confidence: {prediction['ml_confidence']:.4f}")
        print(f"  Risk Score: {prediction['ml_risk_score']:.4f}")
        print(f"  Recommendation: {prediction['recommendation']}")
        print(f"  Class Probabilities:")
        for class_name, prob in prediction['class_probabilities'].items():
            print(f"    {class_name.capitalize()}: {prob:.2%}")
