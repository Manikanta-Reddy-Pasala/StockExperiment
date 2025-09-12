import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import warnings
from typing import Optional
from datetime import date

try:
    # Try relative imports first (for normal usage)
    from .data_service import get_stock_data, create_features
    from ...utils.ml_helpers import save_model, save_lstm_model, save_scaler
except ImportError:
    # Fall back to absolute imports (for testing)
    from data_service import get_stock_data, create_features
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
    from ml_helpers import save_model, save_lstm_model, save_scaler

warnings.filterwarnings("ignore")

def prepare_lstm_data(df, features, target_scaler, window_size=10):
    """
    Prepares data for the LSTM model. Scales features and target separately.
    """
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(df[features])

    # Fit the target scaler only on the target data
    scaled_target = target_scaler.fit_transform(df[['Target']])

    X, y = [], []
    for i in range(window_size, len(scaled_features)):
        X.append(scaled_features[i - window_size:i])
        y.append(scaled_target[i]) # Use the scaled target

    return np.array(X), np.array(y), feature_scaler

# --- Optuna Objective Functions ---
def _objective_rf(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
    }
    model = RandomForestRegressor(**params, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

def _objective_xgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    model = XGBRegressor(**params, objective='reg:squarederror')
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

def _objective_lstm(trial, X, y):
    units_1 = trial.suggest_int("units_1", 32, 128)
    units_2 = trial.suggest_int("units_2", 16, 64)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_int("epochs", 15, 40)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = Sequential([
        LSTM(units_1, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(dropout),
        LSTM(units_2),
        Dropout(dropout),
        Dense(1, activation='linear') # Output layer for regression
    ])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test, y_test), verbose=0,
              callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

# --- Training Orchestrator ---
def train_and_tune_models(symbol: str, start_date: Optional[date] = None, end_date: Optional[date] = None):
    """Orchestrates the model training and tuning process for regression."""
    df = get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    if df is None or len(df) < 150:
        raise ValueError(f"Not enough data for {symbol} in the given date range to train models.")

    df, features = create_features(df)
    X_tab = df[features]
    y_tab = df['Target']

    X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(X_tab, y_tab, shuffle=False, test_size=0.2)

    # --- Random Forest ---
    study_rf = optuna.create_study(direction='minimize')
    study_rf.optimize(lambda trial: _objective_rf(trial, X_tab, y_tab), n_trials=12, show_progress_bar=False)
    rf_model = RandomForestRegressor(**study_rf.best_params, random_state=42)
    rf_model.fit(X_train_tab, y_train_tab)
    save_model(rf_model, f"{symbol}_rf")

    # --- XGBoost ---
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(lambda trial: _objective_xgb(trial, X_tab, y_tab), n_trials=12, show_progress_bar=False)
    xgb_model = XGBRegressor(**study_xgb.best_params, objective='reg:squarederror')
    xgb_model.fit(X_train_tab, y_train_tab)
    save_model(xgb_model, f"{symbol}_xgb")

    # --- LSTM ---
    # A separate scaler for the target variable is crucial for regression
    target_scaler = MinMaxScaler()
    X_lstm, y_lstm, feature_scaler = prepare_lstm_data(df, features, target_scaler)

    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, shuffle=False, test_size=0.2)

    study_lstm = optuna.create_study(direction='minimize')
    study_lstm.optimize(lambda trial: _objective_lstm(trial, X_lstm, y_lstm), n_trials=8, show_progress_bar=False)

    best_lstm_params = study_lstm.best_params
    lstm_model = Sequential([
        LSTM(best_lstm_params["units_1"], return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
        Dropout(best_lstm_params["dropout"]),
        LSTM(best_lstm_params["units_2"]),
        Dropout(best_lstm_params["dropout"]),
        Dense(1, activation='linear')
    ])
    lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    lstm_model.fit(X_train_lstm, y_train_lstm,
                   epochs=best_lstm_params["epochs"],
                   batch_size=best_lstm_params["batch_size"],
                   validation_data=(X_test_lstm, y_test_lstm), verbose=0,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

    save_lstm_model(lstm_model, f"{symbol}_lstm")
    # Save both scalers: one for features, one for the target
    save_scaler(feature_scaler, f"{symbol}_lstm_feature")
    save_scaler(target_scaler, f"{symbol}_lstm_target")

    return {"message": f"Successfully trained and saved models for {symbol}"}
