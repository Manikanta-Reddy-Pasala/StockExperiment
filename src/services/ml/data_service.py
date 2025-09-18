import pandas as pd
import numpy as np
from typing import Optional
from datetime import date, timedelta, datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

try:
    from ...services.brokers.fyers_service import get_fyers_service
    from ...models.database import get_database_manager
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
    Only uses real data - no fallback to mock data.
    """
    # Convert string dates to date objects if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Try to fetch data for the requested symbol only
    fyers_data = _try_fyers_data(symbol, start_date, end_date, period, interval, user_id)
    if fyers_data is not None and len(fyers_data) > 0:
        logger.info(f"Successfully fetched {len(fyers_data)} records from Fyers for {symbol}")
        return fyers_data

    # If the requested symbol fails, raise an error with the specific symbol
    raise Exception(f"Failed to fetch data from Fyers API for symbol {symbol}. Please check if the symbol is valid and API connection is working.")

def _try_fyers_data(symbol: str, start_date: Optional[date], end_date: Optional[date], period: str, interval: str, user_id: int):
    """
    Try to fetch data from Fyers API
    """
    try:
        # Get Fyers service
        fyers_service = get_fyers_service()

        # Get user's Fyers configuration
        config = fyers_service.get_broker_config(user_id)
        if not config:
            logger.warning(f"Fyers configuration not found for user {user_id}")
            return None
        if not config.get('is_connected'):
            logger.warning(f"Fyers not connected for user {user_id}. Connection status: {config.get('connection_status', 'unknown')}")
            return None

        logger.info(f"Attempting Fyers data fetch for {symbol}")

        # Convert period to date range if needed
        # Use a fixed reference date to ensure we're getting historical data
        # Stock markets have data up to around 2024, so use that as end date
        if not start_date or not end_date:
            reference_end_date = date(2024, 12, 31)  # Use end of 2024 as reference
            if period == "1y":
                end_date = reference_end_date
                start_date = end_date - timedelta(days=365)
            elif period == "2y":
                end_date = reference_end_date
                start_date = end_date - timedelta(days=730)
            elif period == "3y":
                end_date = reference_end_date
                start_date = end_date - timedelta(days=1095)
            elif period == "5y":
                end_date = reference_end_date
                start_date = end_date - timedelta(days=1825)
            else:
                # Default to 1 year
                end_date = reference_end_date
                start_date = end_date - timedelta(days=365)

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

        # Get historical data from Fyers with chunking support
        # Fyers API limitation: range_to cannot be 366 days greater than range_from for daily data
        logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date} (timestamps: {range_from} - {range_to})")
        logger.info(f"Request parameters: symbol={symbol}, exchange=NSE, interval={resolution}")

        # Calculate the date range in days
        total_days = (end_date - start_date).days
        max_days_per_request = 365  # Fyers limit

        all_candles = []

        if total_days <= max_days_per_request:
            # Single request
            response = fyers_service.history(
                user_id=user_id,
                symbol=symbol,
                exchange="NSE",
                interval=resolution,
                start_date=str(range_from),
                end_date=str(range_to)
            )

            logger.info(f"Fyers API response status: {response.get('status', 'unknown')}")

            if 'error' in response or response.get('status') != 'success':
                logger.warning(f"Fyers API error for {symbol}: {response.get('message', 'Unknown error')}")
                return None

            if response.get('data', {}).get('candles'):
                all_candles.extend(response['data']['candles'])
        else:
            # Multiple requests needed
            logger.info(f"Data range ({total_days} days) exceeds Fyers limit. Using chunked requests.")

            current_start = start_date
            chunk_num = 1

            while current_start < end_date:
                current_end = min(current_start + timedelta(days=max_days_per_request), end_date)

                chunk_range_from = int(datetime.combine(current_start, datetime.min.time()).timestamp())
                chunk_range_to = int(datetime.combine(current_end, datetime.max.time()).timestamp())

                logger.info(f"Chunk {chunk_num}: {current_start} to {current_end}")

                response = fyers_service.history(
                    user_id=user_id,
                    symbol=symbol,
                    exchange="NSE",
                    interval=resolution,
                    start_date=str(chunk_range_from),
                    end_date=str(chunk_range_to)
                )

                if 'error' in response or response.get('status') != 'success':
                    logger.warning(f"Fyers API error for chunk {chunk_num}: {response.get('message', 'Unknown error')}")
                    current_start = current_end + timedelta(days=1)
                    chunk_num += 1
                    continue

                if response.get('data', {}).get('candles'):
                    all_candles.extend(response['data']['candles'])
                    logger.info(f"Chunk {chunk_num}: Retrieved {len(response['data']['candles'])} candles")

                current_start = current_end + timedelta(days=1)
                chunk_num += 1

        if not all_candles:
            logger.warning(f"No candles data retrieved for {symbol}")
            return None

        logger.info(f"Total candles retrieved: {len(all_candles)}")

        # Convert all candles to DataFrame
        combined_data = {'candles': all_candles}
        df = _convert_fyers_to_dataframe(combined_data, symbol)

        if df.empty:
            logger.warning(f"No data returned from Fyers for symbol {symbol}")
            return None

        logger.info(f"Successfully fetched {len(df)} records from Fyers for {symbol}")
        return df

    except Exception as e:
        logger.warning(f"Fyers API error for {symbol}: {str(e)}")
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