import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
from datetime import date, timedelta

def get_stock_data(
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    period: Optional[str] = "3y",
    interval: str = "1d"
):
    """
    Fetches stock data from Yahoo Finance.
    If start_date and end_date are provided, they are used.
    Otherwise, the 'period' is used.
    """
    if start_date and end_date:
        # yfinance downloads data up to (but not including) the end_date.
        # Add one day to include the end_date in the results.
        end_date_adj = end_date + timedelta(days=1)
        df = yf.download(symbol, start=start_date, end=end_date_adj, interval=interval, progress=False)
    else:
        df = yf.download(symbol, period=period, interval=interval, progress=False)

    if df.empty:
        return None
    return df

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
