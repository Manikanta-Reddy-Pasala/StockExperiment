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
        # Use current date as reference for getting most recent data
        if not start_date or not end_date:
            reference_end_date = datetime.now().date()  # Use current date as reference
            if period == "1y":
                end_date = reference_end_date
                start_date = end_date - timedelta(days=365)
            elif period == "2y":
                end_date = reference_end_date
                start_date = end_date - timedelta(days=730)
            elif period == "3y":
                end_date = reference_end_date
                start_date = end_date - timedelta(days=1095)
            elif period == "4y":
                end_date = reference_end_date
                start_date = end_date - timedelta(days=1460)
            elif period == "5y":
                end_date = reference_end_date
                start_date = end_date - timedelta(days=1825)
            else:
                # Default to 5 years for better training with 70 features
                end_date = reference_end_date
                start_date = end_date - timedelta(days=1825)

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

        # Remove duplicate dates (keep the last occurrence)
        df = df[~df.index.duplicated(keep='last')]

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
    """Engineers features for the stock data with improved feature engineering."""
    # Data sufficiency check for rolling window features
    if len(df) < 60:
        raise ValueError(f"Insufficient data for feature engineering: {len(df)} records. Need at least 60 for rolling windows.")

    # Price-based features (normalized)
    df['Return'] = df['Close'].pct_change()
    df['Return_2d'] = df['Close'].pct_change(periods=2)
    df['Return_5d'] = df['Close'].pct_change(periods=5)

    # Moving averages (as ratios to current price for scale normalization)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # Price relative to moving averages (normalized features)
    df['Price_MA5_Ratio'] = df['Close'] / df['MA5']
    df['Price_MA10_Ratio'] = df['Close'] / df['MA10']
    df['Price_MA20_Ratio'] = df['Close'] / df['MA20']
    df['Price_MA50_Ratio'] = df['Close'] / df['MA50']

    # Moving average crossovers (trend indicators)
    df['MA5_MA20_Ratio'] = df['MA5'] / df['MA20']
    df['MA10_MA50_Ratio'] = df['MA10'] / df['MA50']

    # Volatility features
    df['STD5'] = df['Close'].rolling(window=5).std()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Volatility'] = df['STD20'] / df['MA20']  # Normalized volatility

    # Volume features
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']

    # Price position within recent range
    df['High_20'] = df['High'].rolling(window=20).max()
    df['Low_20'] = df['Low'].rolling(window=20).min()
    df['Price_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['STD20'] * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # Momentum indicators
    df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
    df['ROC'] = df['Close'].pct_change(periods=10)

    # Advanced price action features
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Open']  # Candle body size
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']

    # Volume-price relationship
    df['Volume_Price_Trend'] = df['Volume'] * df['Return']  # Volume-weighted returns
    df['Price_Volume_Ratio'] = df['Close'] * df['Volume']

    # Short-term momentum features (better for next-day prediction)
    df['Return_1d'] = df['Close'].pct_change(periods=1)
    df['Return_3d'] = df['Close'].pct_change(periods=3)
    df['Volatility_3d'] = df['Return_1d'].rolling(window=3).std()
    df['Volatility_7d'] = df['Return_1d'].rolling(window=7).std()

    # Trend strength indicators
    df['MA5_Slope'] = df['MA5'].pct_change(periods=2)
    df['MA20_Slope'] = df['MA20'].pct_change(periods=5)

    # Support/Resistance levels
    df['Distance_to_High_5d'] = (df['High'].rolling(5).max() - df['Close']) / df['Close']
    df['Distance_to_Low_5d'] = (df['Close'] - df['Low'].rolling(5).min()) / df['Close']

    # Advanced momentum and trend features
    df['Williams_R'] = (df['High'].rolling(14).max() - df['Close']) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min()) * -100

    # Stochastic Oscillator
    lowest_low = df['Low'].rolling(14).min()
    highest_high = df['High'].rolling(14).max()
    df['Stoch_K'] = ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # Average True Range (volatility)
    df['TR1'] = df['High'] - df['Low']
    df['TR2'] = abs(df['High'] - df['Close'].shift(1))
    df['TR3'] = abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    df['ATR'] = df['True_Range'].rolling(14).mean()
    df['ATR_Ratio'] = df['ATR'] / df['Close']

    # Price strength relative to volume
    df['Price_Volume_Strength'] = df['Return'] * np.log1p(df['Volume'])

    # Commodity Channel Index
    tp = (df['High'] + df['Low'] + df['Close']) / 3  # Typical Price
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

    # Money Flow Index
    raw_money_flow = tp * df['Volume']
    positive_flow = pd.Series(0.0, index=df.index)
    negative_flow = pd.Series(0.0, index=df.index)

    positive_mask = tp > tp.shift(1)
    negative_mask = tp < tp.shift(1)

    positive_flow[positive_mask] = raw_money_flow[positive_mask]
    negative_flow[negative_mask] = raw_money_flow[negative_mask]

    positive_flow_14 = positive_flow.rolling(14).sum()
    negative_flow_14 = negative_flow.rolling(14).sum()

    df['MFI'] = 100 - (100 / (1 + (positive_flow_14 / negative_flow_14)))

    # Fibonacci retracement levels (approximate)
    period_high = df['High'].rolling(21).max()
    period_low = df['Low'].rolling(21).min()
    fib_range = period_high - period_low
    df['Fib_38_2'] = period_high - (0.382 * fib_range)
    df['Fib_61_8'] = period_high - (0.618 * fib_range)
    df['Price_to_Fib_38_2'] = (df['Close'] - df['Fib_38_2']) / df['Close']
    df['Price_to_Fib_61_8'] = (df['Close'] - df['Fib_61_8']) / df['Close']

    # Parabolic SAR approximation
    df['SAR_approx'] = df['Close'].rolling(10).mean() * 0.98  # Simplified version
    df['Price_SAR_Ratio'] = df['Close'] / df['SAR_approx']

    # On Balance Volume
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(10).mean()
    df['OBV_Ratio'] = df['OBV'] / df['OBV_MA']

    # Advanced Market Microstructure Features
    # Intraday price efficiency measures
    df['Intraday_Efficiency'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
    df['Price_Range_Efficiency'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])

    # Volume-weighted metrics
    df['VWAP_approx'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    df['Price_VWAP_Ratio'] = df['Close'] / df['VWAP_approx']

    # Market regime detection features
    df['Volatility_Regime'] = (df['Return'].rolling(20).std() > df['Return'].rolling(60).std()).astype(int)
    df['Trend_Regime'] = (df['MA5'] > df['MA20']).astype(int)
    df['Volume_Regime'] = (df['Volume'] > df['Volume'].rolling(20).mean()).astype(int)

    # Advanced momentum features
    df['Momentum_Acceleration'] = df['Momentum'].diff()
    df['RSI_Momentum'] = df['RSI'].diff()
    df['Volume_Momentum'] = df['Volume'].pct_change().rolling(5).mean()

    # Liquidity and market stress indicators
    df['Bid_Ask_Spread_Proxy'] = (df['High'] - df['Low']) / df['Close']  # Proxy for bid-ask spread
    df['Market_Stress'] = df['Volume'] * df['Volatility']  # High volume + high volatility = stress
    df['Liquidity_Index'] = df['Volume'] / (df['High'] - df['Low'])  # Volume per price range

    # Pattern recognition features
    df['Doji_Pattern'] = (abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1).astype(int)
    df['Hammer_Pattern'] = ((df['Low'] < df[['Open', 'Close']].min(axis=1)) &
                           (df['High'] - df[['Open', 'Close']].max(axis=1) < df[['Open', 'Close']].min(axis=1) - df['Low'])).astype(int)

    # Multi-timeframe features (using different rolling windows)
    df['Short_Long_MA_Diff'] = (df['MA5'] - df['MA50']) / df['MA50']
    df['Price_Momentum_Divergence'] = np.sign(df['Return_5d']) != np.sign(df['Volume_Change'].rolling(5).mean())

    # Advanced volatility features
    df['Volatility_Skew'] = df['Return'].rolling(20).skew()
    df['Volatility_Kurtosis'] = df['Return'].rolling(20).apply(lambda x: x.kurtosis())
    df['Realized_Volatility'] = df['Return'].rolling(20).std() * np.sqrt(252)

    # Time-based features (to capture calendar effects)
    if hasattr(df.index, 'dayofweek'):
        df['Day_of_Week'] = df.index.dayofweek / 6.0  # Normalize 0-1
        df['Month_of_Year'] = df.index.month / 12.0   # Normalize 0-1
    else:
        df['Day_of_Week'] = 0.5  # Default neutral value
        df['Month_of_Year'] = 0.5  # Default neutral value

    # Risk-adjusted returns
    df['Sharpe_Ratio_Short'] = df['Return'].rolling(20).mean() / (df['Return'].rolling(20).std() + 1e-8)
    df['Sortino_Ratio_Short'] = df['Return'].rolling(20).mean() / (df['Return'][df['Return'] < 0].rolling(20).std().fillna(df['Return'].rolling(20).std()) + 1e-8)

    # Mean reversion indicators
    df['Mean_Reversion_5d'] = (df['Close'] - df['Close'].rolling(5).mean()) / df['Close'].rolling(5).std()
    df['Mean_Reversion_20d'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()

    # Drop intermediate calculation columns
    df.drop(['TR1', 'TR2', 'TR3', 'True_Range', 'Fib_38_2', 'Fib_61_8', 'SAR_approx', 'OBV_MA', 'VWAP_approx'], axis=1, inplace=True, errors='ignore')

    # --- Target Variable for Regression ---
    # The target is the next day's closing price.
    df['Target'] = df['Close'].shift(-1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Use normalized features that don't depend on absolute price levels
    features = [
        # Basic price features
        'Return', 'Return_1d', 'Return_2d', 'Return_3d', 'Return_5d',

        # Moving average features
        'Price_MA5_Ratio', 'Price_MA10_Ratio', 'Price_MA20_Ratio', 'Price_MA50_Ratio',
        'MA5_MA20_Ratio', 'MA10_MA50_Ratio', 'MA5_Slope', 'MA20_Slope',

        # Volatility features
        'Volatility', 'Volatility_3d', 'Volatility_7d', 'ATR_Ratio',

        # Volume features
        'Volume_Change', 'Volume_Ratio', 'Volume_Price_Trend', 'Price_Volume_Strength', 'OBV_Ratio',

        # Price action features
        'Price_Position', 'High_Low_Ratio', 'Close_Open_Ratio', 'Body_Size',
        'Upper_Shadow', 'Lower_Shadow',
        'Distance_to_High_5d', 'Distance_to_Low_5d',

        # Advanced technical indicators
        'RSI', 'BB_Position', 'Williams_R', 'Stoch_K', 'Stoch_D',
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        'CCI', 'MFI',
        'Momentum', 'ROC',

        # Fibonacci and trend analysis
        'Price_to_Fib_38_2', 'Price_to_Fib_61_8', 'Price_SAR_Ratio',

        # Advanced Market Microstructure Features
        'Intraday_Efficiency', 'Price_Range_Efficiency', 'Price_VWAP_Ratio',

        # Market regime detection
        'Volatility_Regime', 'Trend_Regime', 'Volume_Regime',

        # Advanced momentum and acceleration
        'Momentum_Acceleration', 'RSI_Momentum', 'Volume_Momentum',

        # Liquidity and market stress
        'Bid_Ask_Spread_Proxy', 'Market_Stress', 'Liquidity_Index',

        # Pattern recognition
        'Doji_Pattern', 'Hammer_Pattern',

        # Multi-timeframe analysis
        'Short_Long_MA_Diff', 'Price_Momentum_Divergence',

        # Advanced volatility measures
        'Volatility_Skew', 'Volatility_Kurtosis', 'Realized_Volatility',

        # Time-based features
        'Day_of_Week', 'Month_of_Year',

        # Risk-adjusted metrics
        'Sharpe_Ratio_Short', 'Sortino_Ratio_Short',

        # Mean reversion indicators
        'Mean_Reversion_5d', 'Mean_Reversion_20d'
    ]
    return df, features