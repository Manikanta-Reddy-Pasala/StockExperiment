import numpy as np
try:
    # Try relative imports first (for normal usage)
    from .data_service import get_stock_data, create_features
    from ...utils.ml_helpers import load_model, load_lstm_model, load_scaler
    from ...utils.api_logger import APILogger, log_api_call
except ImportError:
    # Fall back to absolute imports (for testing)
    from data_service import get_stock_data, create_features
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
    from ml_helpers import load_model, load_lstm_model, load_scaler
    from api_logger import APILogger, log_api_call

def get_prediction(symbol: str, user_id: int = 1):
    """Generates a regression-based price prediction for a given stock symbol."""
    # Log API call
    APILogger.log_request(
        service_name="MLPredictionService",
        method_name="get_prediction",
        request_data={'symbol': symbol},
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

    # --- Ensemble Prediction (Averaging) ---
    final_predicted_price = (rf_predicted_price + xgb_predicted_price + lstm_predicted_price) / 3

    # Generate buy/sell signal based on prediction vs current price
    current_price = float(df['Close'].iloc[-1])
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
        "signal": signal
    }
