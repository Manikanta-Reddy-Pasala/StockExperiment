import pandas as pd
import numpy as np
from typing import Optional
from datetime import date, timedelta, datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

try:
    from src.services.brokers.fyers_service import get_fyers_service
    from src.models.database import get_database_manager
except ImportError:
    from services.brokers.fyers_service import get_fyers_service
    from models.database import get_database_manager

def get_stock_data(
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    period: Optional[str] = "3y",
    interval: str = "1d",
    user_id: int = 1
):
    """
    Fetches stock data from Fyers API.
    If start_date and end_date are provided, they are used.
    Otherwise, the 'period' is used.
    """
    try:
        # Get Fyers service
        fyers_service = get_fyers_service()
        
        # Get user's Fyers configuration
        config = fyers_service.get_broker_config(user_id)
        if not config:
            logger.error(f"Fyers configuration not found for user {user_id}")
            return None
        if not config.get('is_connected'):
            logger.error(f"Fyers not connected for user {user_id}. Connection status: {config.get('connection_status', 'unknown')}")
            return None

        logger.info(f"Using Fyers configuration for user {user_id}: client_id={config.get('client_id', 'NOT SET')[:8]}...")
        
        # Use Fyers service for historical data
        fyers_service = get_fyers_service()
        
        # Convert period to date range if needed
        if not start_date or not end_date:
            if period == "1y":
                end_date = date.today()
                start_date = end_date - timedelta(days=365)
            elif period == "2y":
                end_date = date.today()
                start_date = end_date - timedelta(days=730)
            elif period == "3y":
                end_date = date.today()
                start_date = end_date - timedelta(days=1095)
            elif period == "5y":
                end_date = date.today()
                start_date = end_date - timedelta(days=1825)
            else:
                # Default to 3 years
                end_date = date.today()
                start_date = end_date - timedelta(days=1095)
        
        # Convert dates to timestamp format for Fyers API
        range_from = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        range_to = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        
        # Map interval to Fyers resolution
        resolution_map = {
            "1d": "D",
            "1h": "60",
            "5m": "5",
            "15m": "15",
            "30m": "30"
        }
        resolution = resolution_map.get(interval, "D")
        
        # Get historical data from Fyers
        logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date} (timestamps: {range_from} - {range_to})")
        logger.info(f"Request parameters: symbol={symbol}, exchange=NSE, interval={resolution}")

        response = fyers_service.history(
            user_id=user_id,
            symbol=symbol,
            exchange="NSE",  # Default exchange
            interval=resolution,
            start_date=str(range_from),
            end_date=str(range_to)
        )

        logger.info(f"Fyers API response status: {response.get('status', 'unknown')}")
        logger.info(f"Fyers API response keys: {list(response.keys()) if isinstance(response, dict) else 'not a dict'}")
        
        if 'error' in response or response.get('status') != 'success':
            logger.error(f"Error fetching data from Fyers: {response.get('message', 'Unknown error')}")
            return None
        
        # Convert Fyers response to pandas DataFrame
        df = _convert_fyers_to_dataframe(response.get('data', {}), symbol)
        
        if df.empty:
            logger.warning(f"No data returned for symbol {symbol}")
            return None
            
        logger.info(f"Successfully fetched {len(df)} records for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
        return None

def _convert_fyers_to_dataframe(response: dict, symbol: str) -> pd.DataFrame:
    """
    Convert Fyers API response to pandas DataFrame compatible with standard OHLCV format.
    """
    try:
        # Handle the standardized response format from FyersService
        candles = response.get('candles', [])
        if not candles:
            logger.warning(f"No candles data found for symbol {symbol}")
            return pd.DataFrame()

        # Convert formatted candles to DataFrame
        df_data = []
        for candle in candles:
            if isinstance(candle, dict):
                # Handle formatted response with string values
                timestamp = datetime.fromtimestamp(int(candle['timestamp']))
                df_data.append({
                    'Date': timestamp,
                    'Open': float(candle['open']),
                    'High': float(candle['high']),
                    'Low': float(candle['low']),
                    'Close': float(candle['close']),
                    'Volume': float(candle['volume'])
                })
            elif isinstance(candle, list) and len(candle) >= 6:
                # Handle raw candle format: [timestamp, open, high, low, close, volume]
                timestamp = datetime.fromtimestamp(candle[0])
                df_data.append({
                    'Date': timestamp,
                    'Open': candle[1],
                    'High': candle[2],
                    'Low': candle[3],
                    'Close': candle[4],
                    'Volume': candle[5]
                })

        if not df_data:
            logger.warning(f"No valid candle data found for symbol {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # Ensure all required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0

        logger.info(f"Successfully converted {len(df)} data points for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error converting Fyers data to DataFrame: {str(e)}")
        return pd.DataFrame()

def create_features(df):
    """Engineers features for the stock data."""
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD5'] = df['Close'].rolling(window=5).std()
    df['Volume_Change'] = df['Volume'].pct_change()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- Target Variable for Regression ---
    # The target is the next day's closing price.
    df['Target'] = df['Close'].shift(-1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    features = ['Return', 'MA5', 'MA10', 'MA20', 'STD5', 'Volume_Change', 'RSI']
    return df, features
