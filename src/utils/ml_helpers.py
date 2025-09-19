import os
import pickle
import tensorflow as tf

def get_model_dir():
    """Get the models directory path."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base_dir, "api", "v1", "ml", "trained")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def save_model(model, name):
    """Saves a scikit-learn or XGBoost model."""
    model_dir = get_model_dir()
    with open(os.path.join(model_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)

def load_model(name):
    """Loads a scikit-learn or XGBoost model."""
    model_dir = get_model_dir()
    path = os.path.join(model_dir, f"{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_lstm_model(model, name):
    """Saves a Keras LSTM model."""
    model_dir = get_model_dir()
    model.save(os.path.join(model_dir, f"{name}.h5"))

def load_lstm_model(name):
    """Loads a Keras LSTM model."""
    model_dir = get_model_dir()
    path = os.path.join(model_dir, f"{name}.h5")
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

def save_scaler(scaler, name):
    """Saves a scikit-learn scaler."""
    model_dir = get_model_dir()
    with open(os.path.join(model_dir, f"{name}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

def load_scaler(name):
    """Loads a scikit-learn scaler."""
    model_dir = get_model_dir()
    path = os.path.join(model_dir, f"{name}_scaler.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
