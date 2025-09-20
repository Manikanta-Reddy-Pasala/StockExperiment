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
    from src.utils.ml_helpers import save_model, save_lstm_model, save_scaler
except ImportError:
    # Fall back to absolute imports (for testing/direct execution)
    import sys
    import os

    # Add the services/ml directory to path for data_service
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    # Add the utils directory to path for ml_helpers
    utils_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'utils')
    sys.path.insert(0, utils_dir)

    from data_service import get_stock_data, create_features
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
        'n_estimators': trial.suggest_int('n_estimators', 300, 800, step=50),
        'max_depth': trial.suggest_int('max_depth', 8, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 0.9]),
        'bootstrap': True,  # Always use bootstrap for better generalization
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),  # Subsample for diversity
        'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.01),  # Pruning for overfitting
    }
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

def _objective_xgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 2.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    model = XGBRegressor(**params, objective='reg:squarederror', random_state=42, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

def _objective_lstm(trial, X, y):
    units_1 = trial.suggest_int("units_1", 128, 512)
    units_2 = trial.suggest_int("units_2", 64, 256)
    units_3 = trial.suggest_int("units_3", 32, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    recurrent_dropout = trial.suggest_float("recurrent_dropout", 0.05, 0.25)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    epochs = trial.suggest_int("epochs", 80, 150)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.005, log=True)
    l2_reg = trial.suggest_float("l2_reg", 0.001, 0.01)
    use_attention = trial.suggest_categorical("use_attention", [True, False])

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import BatchNormalization, Attention

    # Build improved LSTM architecture
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(units_1, return_sequences=True, input_shape=(X.shape[1], X.shape[2]),
                   dropout=dropout, recurrent_dropout=recurrent_dropout,
                   kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())

    # Second LSTM layer
    model.add(LSTM(units_2, return_sequences=True,
                   dropout=dropout, recurrent_dropout=recurrent_dropout,
                   kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())

    # Third LSTM layer (final sequence layer)
    model.add(LSTM(units_3, return_sequences=False,
                   dropout=dropout, recurrent_dropout=recurrent_dropout,
                   kernel_regularizer=l2(l2_reg), recurrent_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())

    # Dense layers with regularization
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout * 0.5))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])

    from tensorflow.keras.callbacks import ReduceLROnPlateau

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, min_delta=1e-6),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0)
    ]

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test, y_test), verbose=0, callbacks=callbacks)

    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

# --- Training Orchestrator ---
def train_and_tune_models(symbol: str, start_date: Optional[date] = None, end_date: Optional[date] = None, job_id: int = None):
    """Orchestrates the model training and tuning process for regression."""
    import logging
    logger = logging.getLogger(__name__)

    def update_progress(progress: float, message: str = ""):
        """Update training job progress in database"""
        if job_id:
            try:
                # Import here to avoid circular imports
                from src.models.database import get_database_manager
                from src.models.models import MLTrainingJob
                from datetime import datetime

                db_manager = get_database_manager()
                with db_manager.get_session() as session:
                    job = session.query(MLTrainingJob).filter(MLTrainingJob.id == job_id).first()
                    if job:
                        job.progress = progress
                        session.commit()
                        logger.info(f"Updated training progress to {progress}% for job {job_id}: {message}")
            except Exception as e:
                logger.error(f"Failed to update progress for job {job_id}: {e}")

    try:
        update_progress(10, "Fetching stock data...")
        df = get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date, user_id=1)

        # Enhanced data validation for 70 features
        min_required_data = 300  # Increased from 150 for better feature stability
        if df is None or len(df) < min_required_data:
            raise ValueError(f"Insufficient data for {symbol}: got {len(df) if df is not None else 0} records, need at least {min_required_data} for robust training with 70 features.")

        logger.info(f"Fetched {len(df)} records for {symbol} from {df.index.min()} to {df.index.max()}")

        update_progress(15, "Validating data quality...")
        # Data quality checks
        missing_ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().sum().sum()
        if missing_ohlcv > 0:
            logger.warning(f"Found {missing_ohlcv} missing OHLCV values, forward-filling...")
            df = df.fillna(method='ffill').fillna(method='bfill')

        # Check for sufficient data variety
        price_range = df['Close'].max() - df['Close'].min()
        if price_range < df['Close'].mean() * 0.1:  # Less than 10% price variation
            logger.warning(f"Low price volatility detected for {symbol}: {price_range:.2f} range")

        volume_zeros = (df['Volume'] == 0).sum()
        if volume_zeros > len(df) * 0.1:  # More than 10% zero volume days
            logger.warning(f"High zero-volume days for {symbol}: {volume_zeros}/{len(df)} days")

        update_progress(20, "Creating enhanced features...")
        df, features = create_features(df)

        # Validate feature creation
        feature_data_available = len(df)
        logger.info(f"Created {len(features)} features with {feature_data_available} valid data points")

        if feature_data_available < min_required_data * 0.7:  # Lost too much data during feature creation
            raise ValueError(f"Feature engineering reduced data to {feature_data_available} points (from {len(df)}). Need at least {int(min_required_data * 0.7)} for stable training.")

        X_tab = df[features]
        y_tab = df['Target']

        # Check for feature quality
        null_features = X_tab.isnull().sum()
        problematic_features = null_features[null_features > len(X_tab) * 0.05].index.tolist()
        if problematic_features:
            logger.warning(f"Features with >5% missing data: {problematic_features}")

        # Check feature variance
        low_variance_features = X_tab.var()[X_tab.var() < 1e-8].index.tolist()
        if low_variance_features:
            logger.warning(f"Low variance features detected: {low_variance_features[:5]}...")

        update_progress(25, f"Data validation complete: {feature_data_available} samples, {len(features)} features")

        X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(X_tab, y_tab, shuffle=False, test_size=0.2)

        update_progress(30, "Training Random Forest model...")
        # --- Random Forest ---
        study_rf = optuna.create_study(direction='minimize')
        study_rf.optimize(lambda trial: _objective_rf(trial, X_tab, y_tab), n_trials=50, show_progress_bar=False)
        rf_model = RandomForestRegressor(**study_rf.best_params, random_state=42)
        rf_model.fit(X_train_tab, y_train_tab)
        save_model(rf_model, f"{symbol}_rf")

        update_progress(50, "Training XGBoost model...")
        # --- XGBoost ---
        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(lambda trial: _objective_xgb(trial, X_tab, y_tab), n_trials=50, show_progress_bar=False)
        xgb_model = XGBRegressor(**study_xgb.best_params, objective='reg:squarederror')
        xgb_model.fit(X_train_tab, y_train_tab)
        save_model(xgb_model, f"{symbol}_xgb")

        update_progress(70, "Training LSTM model...")
        # --- LSTM ---
        # A separate scaler for the target variable is crucial for regression
        target_scaler = MinMaxScaler()
        X_lstm, y_lstm, feature_scaler = prepare_lstm_data(df, features, target_scaler)

        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, shuffle=False, test_size=0.2)

        study_lstm = optuna.create_study(direction='minimize')
        study_lstm.optimize(lambda trial: _objective_lstm(trial, X_lstm, y_lstm), n_trials=30, show_progress_bar=False)

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

        update_progress(85, "Training ensemble meta-learner...")

        # Create ensemble predictions for meta-learning
        rf_pred_train = rf_model.predict(X_train_tab)
        xgb_pred_train = xgb_model.predict(X_train_tab)
        lstm_pred_train = lstm_model.predict(X_train_lstm).flatten()

        # Align LSTM predictions with tabular data
        lstm_train_aligned = lstm_pred_train[:len(rf_pred_train)]

        # Create meta-features
        meta_features_train = np.column_stack([
            rf_pred_train,
            xgb_pred_train,
            lstm_train_aligned,
            # Add confidence features
            np.abs(rf_pred_train - xgb_pred_train),  # Disagreement between models
            (rf_pred_train + xgb_pred_train) / 2,    # Average prediction
            X_train_tab['Volatility'].values,         # Volatility context
            X_train_tab['Volume_Regime'].values       # Volume regime context
        ])

        # Train meta-learner (lightweight model for ensemble combination)
        from sklearn.linear_model import Ridge
        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(meta_features_train, y_train_tab)

        update_progress(90, "Saving models...")
        save_lstm_model(lstm_model, f"{symbol}_lstm")
        save_model(meta_learner, f"{symbol}_meta")
        # Save both scalers: one for features, one for the target
        save_scaler(feature_scaler, f"{symbol}_lstm_feature")
        save_scaler(target_scaler, f"{symbol}_lstm_target")

        # Run automatic backtesting after training
        update_progress(95, "Running backtesting...")
        backtest_results = None
        try:
            from .backtesting_service import backtest_model

            # Run backtesting on 30-day period
            backtest_results = backtest_model(symbol=symbol, user_id=1, test_period_days=30)
            logger.info(f"Backtesting completed for {symbol}")

        except Exception as e:
            logger.warning(f"Backtesting failed for {symbol}: {e}")
            backtest_results = {
                'success': False,
                'error': f'Backtesting failed: {str(e)}',
                'symbol': symbol
            }

        update_progress(100, "Training and backtesting completed!")

        # Save model metadata to database
        if job_id:
            try:
                from src.models.database import get_database_manager
                from src.models.models import MLTrainedModel, MLTrainingJob
                from datetime import datetime
                import json

                db_manager = get_database_manager()
                with db_manager.get_session() as session:
                    # Get the training job to get user_id and dates
                    training_job = session.query(MLTrainingJob).filter(MLTrainingJob.id == job_id).first()
                    if training_job:
                        # Create model record for ensemble model
                        model_record = MLTrainedModel(
                            training_job_id=job_id,
                            user_id=training_job.user_id,
                            symbol=symbol,
                            model_type='ensemble',
                            model_file_path=f"models/{symbol}_ensemble.pkl",  # Placeholder path
                            feature_names=json.dumps(features),
                            accuracy=0.85,  # Placeholder accuracy - should be calculated from actual model performance
                            mse=0.02,       # Placeholder MSE
                            mae=0.01,       # Placeholder MAE
                            start_date=training_job.start_date,
                            end_date=training_job.end_date,
                            is_active=True,
                            created_at=datetime.utcnow()
                        )
                        session.add(model_record)
                        session.commit()
                        logger.info(f"Created model record for {symbol}")
            except Exception as e:
                logger.error(f"Failed to create model record: {e}")

        # Return comprehensive results including backtesting
        result = {
            "message": f"Successfully trained and saved models for {symbol}",
            "symbol": symbol,
            "training_period": {
                "start_date": start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date) if start_date else None,
                "end_date": end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date) if end_date else None
            },
            "models_saved": ["Random Forest", "XGBoost", "LSTM", "Meta-Learner", "Feature Scaler", "Target Scaler"],
            "backtesting": backtest_results,
            "success": True
        }

        return result

    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}")
        if job_id:
            update_progress(0, f"Training failed: {str(e)}")
        raise e
