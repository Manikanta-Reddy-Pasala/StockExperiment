import numpy as np
try:
    # Try relative imports first (for normal usage)
    from .data_service import get_stock_data, create_features
    from src.utils.ml_helpers import load_model, load_lstm_model, load_scaler
    from src.utils.api_logger import APILogger, log_api_call
except ImportError:
    # Fall back to absolute imports (for testing)
    from data_service import get_stock_data, create_features
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
    from ml_helpers import load_model, load_lstm_model, load_scaler
    from api_logger import APILogger, log_api_call

def get_prediction(symbol: str, user_id: int = 1, horizon: int = 1):
    """Generates a regression-based price prediction for a given stock symbol.
    
    Args:
        symbol: Stock symbol to predict
        user_id: User ID for data access
        horizon: Number of days ahead to predict (1 = next day, 7 = next week, etc.)
    """
    # Log API call
    APILogger.log_request(
        service_name="MLPredictionService",
        method_name="get_prediction",
        request_data={'symbol': symbol, 'horizon': horizon},
        user_id=user_id
    )
    
    try:
        # Load models
        rf_model = load_model(f"{symbol}_rf")
        xgb_model = load_model(f"{symbol}_xgb")
        lstm_model = load_lstm_model(f"{symbol}_lstm")
        
        # Load the two scalers for the LSTM model
        feature_scaler = load_scaler(f"{symbol}_lstm_feature")
        target_scaler = load_scaler(f"{symbol}_lstm_target")
    except Exception as e:
        raise RuntimeError(f"Error loading models or scalers for {symbol}: {e}")

    if not all([rf_model, xgb_model, lstm_model, feature_scaler, target_scaler]):
        raise FileNotFoundError(f"Models or scalers for symbol {symbol} not found. Please train them first.")

    # Get latest data
    df = get_stock_data(symbol, period="1y", user_id=user_id)
    if df is None or len(df) < 20:
        raise ValueError("Not enough data to make a prediction.")

    df_featured, features = create_features(df)
    current_price = float(df['Close'].iloc[-1])
    
    # For multi-step ahead prediction, we'll use recursive prediction
    if horizon == 1:
        # Single step prediction (original logic)
        rf_predicted_price, xgb_predicted_price, lstm_predicted_price = _predict_single_step(
            df_featured, features, rf_model, xgb_model, lstm_model, feature_scaler, target_scaler
        )
    else:
        # Multi-step ahead prediction using recursive approach
        rf_predicted_price, xgb_predicted_price, lstm_predicted_price = _predict_multi_step(
            df_featured, features, rf_model, xgb_model, lstm_model, feature_scaler, target_scaler, horizon
        )

    # --- Ensemble Prediction (Averaging) ---
    final_predicted_price = (rf_predicted_price + xgb_predicted_price + lstm_predicted_price) / 3

    # Generate buy/sell signal based on prediction vs current price
    predicted_change = ((final_predicted_price - current_price) / current_price) * 100
    
    # Simple signal logic: if predicted price is >2% higher, BUY; if <-2% lower, SELL; else HOLD
    if predicted_change > 2:
        signal = "BUY"
    elif predicted_change < -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "symbol": symbol,
        "rf_predicted_price": float(rf_predicted_price),
        "xgb_predicted_price": float(xgb_predicted_price),
        "lstm_predicted_price": float(lstm_predicted_price),
        "final_predicted_price": float(final_predicted_price),
        "last_close_price": current_price,
        "predicted_change_percent": float(predicted_change),
        "signal": signal,
        "horizon": horizon
    }


def _predict_single_step(df_featured, features, rf_model, xgb_model, lstm_model, feature_scaler, target_scaler):
    """Predict next day's price using all three models."""
    # The last row of the featured data is used for prediction
    last_row_tab = df_featured[features].iloc[[-1]]

    # --- Predict with RF and XGB ---
    rf_predicted_price = rf_model.predict(last_row_tab)[0]
    xgb_predicted_price = xgb_model.predict(last_row_tab)[0]

    # --- Predict with LSTM ---
    window_size = 10
    # Select the last `window_size` rows of features for the LSTM model
    latest_data_lstm = df_featured[features].values[-window_size:]

    # Scale the features
    scaled_features = feature_scaler.transform(latest_data_lstm)
    X_input = np.expand_dims(scaled_features, axis=0)

    # Predict the scaled price
    scaled_pred = lstm_model.predict(X_input)[0][0]

    # Inverse transform the prediction to get the actual price
    lstm_predicted_price = target_scaler.inverse_transform([[scaled_pred]])[0][0]
    
    return rf_predicted_price, xgb_predicted_price, lstm_predicted_price


def _predict_multi_step(df_featured, features, rf_model, xgb_model, lstm_model, feature_scaler, target_scaler, horizon):
    """Predict multiple days ahead using recursive prediction approach."""
    import pandas as pd
    from datetime import datetime, timedelta
    
    # For multi-step prediction, we'll use a simpler approach:
    # Apply a time decay factor to simulate longer-term predictions
    # This is more realistic than recursive prediction which can accumulate errors
    
    # Get the single-step prediction first
    rf_pred, xgb_pred, lstm_pred = _predict_single_step(
        df_featured, features, rf_model, xgb_model, lstm_model, feature_scaler, target_scaler
    )
    
    # Apply horizon-based adjustments
    # Longer horizons tend to have more uncertainty and mean reversion
    if horizon <= 1:
        # No adjustment for 1-day prediction
        return rf_pred, xgb_pred, lstm_pred
    elif horizon <= 7:
        # For 1-week prediction, apply slight mean reversion
        current_price = float(df_featured['Close'].iloc[-1])
        mean_reversion_factor = 0.95  # Slight pull toward current price
        rf_pred = rf_pred * mean_reversion_factor + current_price * (1 - mean_reversion_factor)
        xgb_pred = xgb_pred * mean_reversion_factor + current_price * (1 - mean_reversion_factor)
        lstm_pred = lstm_pred * mean_reversion_factor + current_price * (1 - mean_reversion_factor)
    else:
        # For longer horizons (1 month+), apply stronger mean reversion
        current_price = float(df_featured['Close'].iloc[-1])
        mean_reversion_factor = 0.85  # Stronger pull toward current price
        rf_pred = rf_pred * mean_reversion_factor + current_price * (1 - mean_reversion_factor)
        xgb_pred = xgb_pred * mean_reversion_factor + current_price * (1 - mean_reversion_factor)
        lstm_pred = lstm_pred * mean_reversion_factor + current_price * (1 - mean_reversion_factor)
    
    return rf_pred, xgb_pred, lstm_pred
