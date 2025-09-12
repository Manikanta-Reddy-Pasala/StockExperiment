import os
import pickle
import tensorflow as tf

# Adjust paths for the StockExperiment project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model(model, name):
    """Saves a scikit-learn or XGBoost model."""
    with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)

def load_model(name):
    """Loads a scikit-learn or XGBoost model."""
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_lstm_model(model, name):
    """Saves a Keras LSTM model."""
    model.save(os.path.join(MODEL_DIR, f"{name}.h5"))

def load_lstm_model(name):
    """Loads a Keras LSTM model."""
    path = os.path.join(MODEL_DIR, f"{name}.h5")
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

def save_scaler(scaler, name):
    """Saves a scikit-learn scaler."""
    with open(os.path.join(MODEL_DIR, f"{name}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

def load_scaler(name):
    """Loads a scikit-learn scaler."""
    path = os.path.join(MODEL_DIR, f"{name}_scaler.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
