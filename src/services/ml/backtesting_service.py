"""
Backtesting Service for ML Models
Evaluates model performance on historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

logger = logging.getLogger(__name__)

class BacktestingService:
    """Service for backtesting ML models on historical data"""
    
    def __init__(self):
        self.results = {}
    
    def backtest_model(self, symbol: str, user_id: int = 1, test_period_days: int = 30) -> Dict[str, Any]:
        """
        Backtest a trained model on historical data
        
        Args:
            symbol: Stock symbol to backtest
            user_id: User ID for data access
            test_period_days: Number of days to test on (default 30)
            
        Returns:
            Dictionary containing backtesting results
        """
        try:
            logger.info(f"Starting backtest for {symbol} over {test_period_days} days")

            # Import ml_helpers for loading models
            try:
                from ...utils.ml_helpers import load_model, load_lstm_model, load_scaler
            except ImportError:
                # Fallback for direct execution
                import sys
                import os
                utils_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utils')
                sys.path.insert(0, utils_dir)
                from ml_helpers import load_model, load_lstm_model, load_scaler

            # Load the trained models using the helper functions
            symbol_clean = symbol.replace(':', '_')
            rf_model = load_model(f"{symbol}_rf")
            xgb_model = load_model(f"{symbol}_xgb")
            lstm_model = load_lstm_model(f"{symbol}_lstm")
            feature_scaler = load_scaler(f"{symbol}_lstm_feature")
            target_scaler = load_scaler(f"{symbol}_lstm_target")

            if not rf_model:
                return {
                    'success': False,
                    'error': f'No Random Forest model found for {symbol}'
                }

            if not xgb_model:
                return {
                    'success': False,
                    'error': f'No XGBoost model found for {symbol}'
                }

            if not lstm_model:
                return {
                    'success': False,
                    'error': f'No LSTM model found for {symbol}'
                }

            if not feature_scaler or not target_scaler:
                return {
                    'success': False,
                    'error': f'Model scalers not found for {symbol}'
                }
            
            # Get historical data for backtesting
            try:
                from .data_service import get_stock_data
            except ImportError:
                # Fallback for direct execution
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, current_dir)
                from data_service import get_stock_data

            # Get recent historical data for backtesting (should be AFTER training period)
            # Use recent data to test on out-of-sample period
            from datetime import datetime, timedelta

            # Get data from last 3 months (most recent available)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=90)  # 3 months back

            df = get_stock_data(symbol, start_date=start_date, end_date=end_date, user_id=user_id)
            if df is None or len(df) < test_period_days + 20:
                return {
                    'success': False,
                    'error': f'Insufficient historical data for {symbol}'
                }
            
            # Prepare features
            try:
                from .data_service import create_features
            except ImportError:
                # Fallback for direct execution
                from data_service import create_features
            df_featured, features = create_features(df)
            
            if len(df_featured) < test_period_days + 10:
                return {
                    'success': False,
                    'error': 'Insufficient featured data for backtesting'
                }
            
            # Perform backtesting
            backtest_results = self._perform_backtest(
                df_featured, features, rf_model, xgb_model, lstm_model, 
                feature_scaler, target_scaler, test_period_days
            )
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(backtest_results)
            
            # Generate summary
            summary = self._generate_summary(metrics, symbol, test_period_days)
            
            return {
                'success': True,
                'symbol': symbol,
                'test_period_days': test_period_days,
                'metrics': metrics,
                'summary': summary,
                'detailed_results': backtest_results,
                'model_components': {
                    'rf_model': 'Random Forest',
                    'xgb_model': 'XGBoost', 
                    'lstm_model': 'LSTM'
                }
            }
            
        except Exception as e:
            logger.error(f"Error in backtesting for {symbol}: {e}")
            return {
                'success': False,
                'error': f'Backtesting failed: {str(e)}'
            }
    
    def _perform_backtest(self, df_featured: pd.DataFrame, features: List[str], 
                         rf_model, xgb_model, lstm_model, feature_scaler, target_scaler,
                         test_period_days: int) -> List[Dict[str, Any]]:
        """Perform the actual backtesting on historical data"""
        
        results = []
        test_start_idx = len(df_featured) - test_period_days
        
        for i in range(test_start_idx, len(df_featured)):
            try:
                # Get data up to current point (simulating real-time prediction)
                current_data = df_featured.iloc[:i+1].copy()
                
                if len(current_data) < 20:  # Need minimum data for prediction
                    continue
                
                # Prepare features for prediction
                X = current_data[features].iloc[-1:].values

                # Make predictions for tabular models (use raw features)
                rf_pred = rf_model.predict(X)[0]
                xgb_pred = xgb_model.predict(X)[0]

                # For LSTM, we need sequence data
                # Get the last 10 data points for LSTM prediction
                window_size = 10
                if len(current_data) >= window_size:
                    lstm_features = current_data[features].iloc[-window_size:].values
                    lstm_features_scaled = feature_scaler.transform(lstm_features)
                    lstm_X = lstm_features_scaled.reshape(1, window_size, len(features))
                    lstm_pred_scaled = lstm_model.predict(lstm_X)[0][0]
                    # Inverse transform the LSTM prediction
                    lstm_pred = target_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
                else:
                    # If not enough data for LSTM, use average of other predictions
                    lstm_pred = (rf_pred + xgb_pred) / 2
                
                # Adaptive ensemble prediction based on recent performance
                # Calculate confidence scores based on recent volatility and trend
                recent_volatility = current_data['Close'].pct_change().iloc[-5:].std()
                trend_strength = abs(current_data['Close'].iloc[-1] / current_data['Close'].iloc[-5] - 1)

                # Adaptive weights based on market conditions
                if recent_volatility < 0.02:  # Low volatility - favor tree models
                    rf_weight, xgb_weight, lstm_weight = 0.45, 0.45, 0.1
                elif trend_strength > 0.05:  # Strong trend - favor LSTM
                    rf_weight, xgb_weight, lstm_weight = 0.25, 0.25, 0.5
                else:  # Normal conditions - balanced approach
                    rf_weight, xgb_weight, lstm_weight = 0.35, 0.35, 0.3

                ensemble_pred = (rf_pred * rf_weight + xgb_pred * xgb_weight + lstm_pred * lstm_weight)
                
                # Get actual next day price
                if i + 1 < len(df_featured):
                    actual_price = float(df_featured['Close'].iloc[i + 1])
                    current_price = float(current_data['Close'].iloc[-1])
                    
                    # Calculate prediction accuracy
                    rf_error = abs(rf_pred - actual_price)
                    xgb_error = abs(xgb_pred - actual_price)
                    lstm_error = abs(lstm_pred - actual_price)
                    ensemble_error = abs(ensemble_pred - actual_price)
                    
                    # Calculate percentage errors
                    rf_pct_error = (rf_error / actual_price) * 100
                    xgb_pct_error = (xgb_error / actual_price) * 100
                    lstm_pct_error = (lstm_error / actual_price) * 100
                    ensemble_pct_error = (ensemble_error / actual_price) * 100
                    
                    # Determine signal accuracy
                    actual_change = ((actual_price - current_price) / current_price) * 100
                    rf_change = ((rf_pred - current_price) / current_price) * 100
                    xgb_change = ((xgb_pred - current_price) / current_price) * 100
                    lstm_change = ((lstm_pred - current_price) / current_price) * 100
                    ensemble_change = ((ensemble_pred - current_price) / current_price) * 100
                    
                    # Adaptive signal generation based on volatility
                    # Calculate recent volatility for dynamic thresholds
                    recent_returns = current_data['Close'].pct_change().iloc[-10:].dropna()
                    volatility = recent_returns.std() * 100  # Daily volatility as percentage

                    # Dynamic threshold: use 1.5x daily volatility, with min 0.5% and max 3%
                    threshold = max(0.5, min(3.0, volatility * 1.5))

                    def get_signal(change, threshold):
                        if change > threshold:
                            return "BUY"
                        elif change < -threshold:
                            return "SELL"
                        else:
                            return "HOLD"
                    
                    actual_signal = get_signal(actual_change, threshold)
                    rf_signal = get_signal(rf_change, threshold)
                    xgb_signal = get_signal(xgb_change, threshold)
                    lstm_signal = get_signal(lstm_change, threshold)
                    ensemble_signal = get_signal(ensemble_change, threshold)
                    
                    results.append({
                        'date': df_featured.index[i].strftime('%Y-%m-%d'),
                        'current_price': current_price,
                        'actual_price': actual_price,
                        'actual_change_pct': actual_change,
                        'actual_signal': actual_signal,
                        'rf_prediction': rf_pred,
                        'rf_error': rf_error,
                        'rf_pct_error': rf_pct_error,
                        'rf_signal': rf_signal,
                        'rf_signal_correct': rf_signal == actual_signal,
                        'xgb_prediction': xgb_pred,
                        'xgb_error': xgb_error,
                        'xgb_pct_error': xgb_pct_error,
                        'xgb_signal': xgb_signal,
                        'xgb_signal_correct': xgb_signal == actual_signal,
                        'lstm_prediction': lstm_pred,
                        'lstm_error': lstm_error,
                        'lstm_pct_error': lstm_pct_error,
                        'lstm_signal': lstm_signal,
                        'lstm_signal_correct': lstm_signal == actual_signal,
                        'ensemble_prediction': ensemble_pred,
                        'ensemble_error': ensemble_error,
                        'ensemble_pct_error': ensemble_pct_error,
                        'ensemble_signal': ensemble_signal,
                        'ensemble_signal_correct': ensemble_signal == actual_signal
                    })
                    
            except Exception as e:
                logger.warning(f"Error in backtest iteration {i}: {e}")
                continue
        
        return results
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        if not results:
            return {'error': 'No backtest results to analyze'}
        
        # Extract arrays for calculations
        actual_prices = [r['actual_price'] for r in results]
        rf_predictions = [r['rf_prediction'] for r in results]
        xgb_predictions = [r['xgb_prediction'] for r in results]
        lstm_predictions = [r['lstm_prediction'] for r in results]
        ensemble_predictions = [r['ensemble_prediction'] for r in results]
        
        # Price prediction metrics
        def calculate_price_metrics(actual, predicted, model_name):
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100
            r2 = r2_score(actual, predicted)
            
            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2_score': float(r2)
            }
        
        # Signal accuracy metrics
        def calculate_signal_metrics(results, model_name):
            signal_key = f'{model_name}_signal_correct'
            correct_signals = sum(1 for r in results if r[signal_key])
            total_signals = len(results)
            accuracy = (correct_signals / total_signals) * 100 if total_signals > 0 else 0
            
            # Breakdown by signal type
            buy_correct = sum(1 for r in results if r[signal_key] and r['actual_signal'] == 'BUY')
            sell_correct = sum(1 for r in results if r[signal_key] and r['actual_signal'] == 'SELL')
            hold_correct = sum(1 for r in results if r[signal_key] and r['actual_signal'] == 'HOLD')
            
            total_buy = sum(1 for r in results if r['actual_signal'] == 'BUY')
            total_sell = sum(1 for r in results if r['actual_signal'] == 'SELL')
            total_hold = sum(1 for r in results if r['actual_signal'] == 'HOLD')
            
            return {
                'accuracy': accuracy,
                'correct_signals': correct_signals,
                'total_signals': total_signals,
                'buy_accuracy': (buy_correct / total_buy * 100) if total_buy > 0 else 0,
                'sell_accuracy': (sell_correct / total_sell * 100) if total_sell > 0 else 0,
                'hold_accuracy': (hold_correct / total_hold * 100) if total_hold > 0 else 0
            }
        
        metrics = {
            'total_predictions': len(results),
            'rf_model': {
                'price_metrics': calculate_price_metrics(actual_prices, rf_predictions, 'rf'),
                'signal_metrics': calculate_signal_metrics(results, 'rf')
            },
            'xgb_model': {
                'price_metrics': calculate_price_metrics(actual_prices, xgb_predictions, 'xgb'),
                'signal_metrics': calculate_signal_metrics(results, 'xgb')
            },
            'lstm_model': {
                'price_metrics': calculate_price_metrics(actual_prices, lstm_predictions, 'lstm'),
                'signal_metrics': calculate_signal_metrics(results, 'lstm')
            },
            'ensemble_model': {
                'price_metrics': calculate_price_metrics(actual_prices, ensemble_predictions, 'ensemble'),
                'signal_metrics': calculate_signal_metrics(results, 'ensemble')
            }
        }
        
        return metrics
    
    def _generate_summary(self, metrics: Dict[str, Any], symbol: str, test_period_days: int) -> Dict[str, Any]:
        """Generate a human-readable summary of backtesting results"""
        
        # Find best performing model
        best_model = 'ensemble'
        best_accuracy = metrics['ensemble_model']['signal_metrics']['accuracy']
        
        for model_name in ['rf_model', 'xgb_model', 'lstm_model']:
            accuracy = metrics[model_name]['signal_metrics']['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        # Calculate overall trust score (0-100)
        ensemble_accuracy = metrics['ensemble_model']['signal_metrics']['accuracy']
        ensemble_mape = metrics['ensemble_model']['price_metrics']['mape']
        ensemble_r2 = metrics['ensemble_model']['price_metrics']['r2_score']
        
        # Trust score calculation (weighted combination)
        trust_score = (
            ensemble_accuracy * 0.5 +  # 50% weight on signal accuracy
            max(0, 100 - ensemble_mape) * 0.3 +  # 30% weight on price accuracy (inverse of MAPE)
            max(0, ensemble_r2 * 100) * 0.2  # 20% weight on R² score
        )
        
        # Determine trust level
        if trust_score >= 80:
            trust_level = "HIGH"
            trust_color = "success"
        elif trust_score >= 60:
            trust_level = "MEDIUM"
            trust_color = "warning"
        else:
            trust_level = "LOW"
            trust_color = "danger"
        
        return {
            'trust_score': round(trust_score, 1),
            'trust_level': trust_level,
            'trust_color': trust_color,
            'best_model': best_model,
            'best_accuracy': round(best_accuracy, 1),
            'ensemble_accuracy': round(ensemble_accuracy, 1),
            'ensemble_mape': round(ensemble_mape, 1),
            'ensemble_r2': round(ensemble_r2, 3),
            'recommendation': self._get_recommendation(trust_score, ensemble_accuracy, ensemble_mape),
            'test_period': f"{test_period_days} days",
            'symbol': symbol
        }
    
    def _get_recommendation(self, trust_score: float, accuracy: float, mape: float) -> str:
        """Generate recommendation based on performance metrics"""
        
        if trust_score >= 80 and accuracy >= 70 and mape <= 5:
            return "Model shows excellent performance. High confidence in predictions."
        elif trust_score >= 60 and accuracy >= 60 and mape <= 10:
            return "Model shows good performance. Moderate confidence in predictions."
        elif trust_score >= 40 and accuracy >= 50 and mape <= 15:
            return "Model shows fair performance. Use with caution and consider as one input among others."
        else:
            return "Model shows poor performance. Not recommended for trading decisions. Consider retraining with more data."


def backtest_model(symbol: str, user_id: int = 1, test_period_days: int = 30) -> Dict[str, Any]:
    """Convenience function for backtesting a model"""
    service = BacktestingService()
    return service.backtest_model(symbol, user_id, test_period_days)

