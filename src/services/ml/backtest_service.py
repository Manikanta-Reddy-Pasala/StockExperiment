import numpy as np
import os
try:
    # Try relative imports first (for normal usage)
    from .data_service import get_stock_data, create_features
    from src.utils.ml_helpers import load_model, load_lstm_model, load_scaler, get_model_dir
except ImportError:
    # Fall back to absolute imports (for testing)
    from data_service import get_stock_data, create_features
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
    from ml_helpers import load_model, load_lstm_model, load_scaler, get_model_dir

def run_backtest(symbol: str, initial_cash=100000, user_id: int = 1):
    """Runs a backtest for a given stock symbol using regression models."""
    # Load models and scalers
    rf_model = load_model(f"{symbol}_rf")
    xgb_model = load_model(f"{symbol}_xgb")
    lstm_model = load_lstm_model(f"{symbol}_lstm")
    feature_scaler = load_scaler(f"{symbol}_lstm_feature")
    target_scaler = load_scaler(f"{symbol}_lstm_target")

    if not all([rf_model, xgb_model, lstm_model, feature_scaler, target_scaler]):
        raise FileNotFoundError(f"Models or scalers for {symbol} not found. Please train them first.")

    df = get_stock_data(symbol, period="3y", user_id=user_id)
    if df is None:
        raise ValueError("Could not download data for backtest.")

    df_featured, features = create_features(df)
    # Align the featured data with the original DataFrame's index
    df = df.loc[df_featured.index]

    cash = initial_cash
    holdings = 0
    equity_curve = []
    window_size = 10

    for i in range(window_size, len(df_featured) - 1): # -1 to avoid predicting on the last day with no known future
        # --- Prepare data for prediction ---
        X_tab = df_featured[features].iloc[[i]]

        X_lstm_seq = feature_scaler.transform(df_featured[features].iloc[i - window_size + 1:i + 1])
        X_lstm_seq = np.expand_dims(X_lstm_seq, axis=0)

        # --- Get price predictions ---
        rf_pred = rf_model.predict(X_tab)[0]
        xgb_pred = xgb_model.predict(X_tab)[0]

        lstm_scaled_pred = lstm_model.predict(X_lstm_seq)[0][0]
        lstm_pred = target_scaler.inverse_transform([[lstm_scaled_pred]])[0][0]

        # Ensemble prediction (average)
        predicted_price_tomorrow = (rf_pred + xgb_pred + lstm_pred) / 3

        price_today = df.iloc[i]['Close']

        # --- Trading Logic ---
        # If predicted price for tomorrow is higher than today's price, buy.
        # If predicted price is lower, sell.
        if predicted_price_tomorrow > price_today:
            # Buy signal
            if cash > 0:
                holdings = cash / price_today
                cash = 0
        else:
            # Sell signal
            if holdings > 0:
                cash = holdings * price_today
                holdings = 0

        current_value = cash + holdings * price_today
        equity_curve.append(current_value)

    if not equity_curve: # Handle cases with very short data
        return {
            "symbol": symbol,
            "initial_cash": initial_cash,
            "final_value": initial_cash,
            "total_return_pct": 0,
            "equity_curve": []
        }

    final_value = equity_curve[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    return {
        "symbol": symbol,
        "initial_cash": initial_cash,
        "final_value": final_value,
        "total_return_pct": total_return,
        "equity_curve": equity_curve
    }

def run_backtest_for_all_stocks(user_id: int = 1):
    """Runs backtest for all stocks with available models."""
    results = []
    stock_symbols = set()
    model_dir = get_model_dir()
    
    for f in os.listdir(model_dir):
        if f.endswith(".pkl") or f.endswith(".h5"):
            symbol = f.split('_')[0]
            if symbol != "DUMMY":
                stock_symbols.add(symbol)

    for symbol in stock_symbols:
        try:
            result = run_backtest(symbol, user_id=user_id)
            results.append(result)
        except FileNotFoundError as e:
            print(f"Skipping backtest for {symbol}: {e}")
        except Exception as e:
            print(f"An error occurred during backtest for {symbol}: {e}")

    return results
