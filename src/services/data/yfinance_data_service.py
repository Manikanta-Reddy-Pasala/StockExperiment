"""
YFinance Data Service

Provides market data from Yahoo Finance as a free alternative to broker APIs.
Perfect for paper trading without requiring broker authentication.

Features:
- Historical OHLCV data for NSE stocks
- Current stock quotes
- No authentication required
- Rate limit friendly
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, date
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy import yfinance to avoid import errors if not installed
_yfinance = None

def _get_yfinance():
    """Lazy load yfinance module."""
    global _yfinance
    if _yfinance is None:
        try:
            import yfinance as yf
            _yfinance = yf
        except ImportError:
            raise ImportError(
                "yfinance is not installed. Install it with: pip install yfinance"
            )
    return _yfinance


class YFinanceDataService:
    """
    Yahoo Finance data service for NSE stocks.

    Symbol Format Conversion:
    - Fyers format: NSE:RELIANCE-EQ
    - YFinance format: RELIANCE.NS
    """

    def __init__(self):
        self.rate_limit_delay = 0.5  # 500ms between requests to avoid rate limiting
        self.batch_size = 20  # Process stocks in batches
        self.max_retries = 3

    def _convert_to_yfinance_symbol(self, symbol: str) -> str:
        """
        Convert Fyers symbol format to Yahoo Finance format.

        Examples:
            NSE:RELIANCE-EQ -> RELIANCE.NS
            NSE:TCS-EQ -> TCS.NS
            NSE:INFY-EQ -> INFY.NS
        """
        if not symbol:
            return ""

        # Remove exchange prefix and suffix
        clean_symbol = symbol

        # Handle NSE:SYMBOL-EQ format
        if ':' in symbol:
            clean_symbol = symbol.split(':')[1]

        # Remove -EQ suffix
        if clean_symbol.endswith('-EQ'):
            clean_symbol = clean_symbol[:-3]

        # Add .NS suffix for NSE stocks
        return f"{clean_symbol}.NS"

    def _convert_from_yfinance_symbol(self, yf_symbol: str) -> str:
        """
        Convert Yahoo Finance symbol back to Fyers format.

        Examples:
            RELIANCE.NS -> NSE:RELIANCE-EQ
        """
        if not yf_symbol:
            return ""

        # Remove .NS suffix
        clean_symbol = yf_symbol.replace('.NS', '').replace('.BO', '')

        # Add NSE prefix and EQ suffix
        return f"NSE:{clean_symbol}-EQ"

    def get_historical_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a single stock.

        Args:
            symbol: Stock symbol in Fyers format (e.g., 'NSE:RELIANCE-EQ')
            days: Number of days of historical data to fetch

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            Returns None if fetch fails
        """
        yf = _get_yfinance()

        try:
            yf_symbol = self._convert_to_yfinance_symbol(symbol)

            if not yf_symbol:
                logger.warning(f"Invalid symbol format: {symbol}")
                return None

            logger.info(f"Fetching historical data for {symbol} ({yf_symbol}) - {days} days")

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No historical data for {symbol}")
                return None

            # Standardize column names and format
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Convert date to date only (not datetime)
            df['date'] = pd.to_datetime(df['date']).dt.date

            # Select only required columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

            # Remove any rows with NaN values
            df = df.dropna()

            logger.info(f"Fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    def get_historical_data_bulk(self, symbols: List[str], days: int = 365,
                                  progress_callback=None) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stocks.

        Args:
            symbols: List of stock symbols in Fyers format
            days: Number of days of historical data
            progress_callback: Optional callback function(processed, total, symbol)

        Returns:
            Dict mapping symbol to DataFrame
        """
        results = {}
        total = len(symbols)

        logger.info(f"Fetching historical data for {total} stocks")

        for i, symbol in enumerate(symbols):
            try:
                df = self.get_historical_data(symbol, days)
                if df is not None and not df.empty:
                    results[symbol] = df

                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, total, symbol)

                # Rate limiting
                if i < total - 1:
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error in bulk fetch for {symbol}: {e}")
                continue

        logger.info(f"Successfully fetched data for {len(results)}/{total} stocks")
        return results

    def get_current_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current quote for a single stock.

        Args:
            symbol: Stock symbol in Fyers format

        Returns:
            Dict with quote data or None if fetch fails
        """
        yf = _get_yfinance()

        try:
            yf_symbol = self._convert_to_yfinance_symbol(symbol)

            if not yf_symbol:
                return None

            ticker = yf.Ticker(yf_symbol)
            info = ticker.fast_info

            # Get latest price data
            return {
                'symbol': symbol,
                'last_price': getattr(info, 'last_price', None),
                'previous_close': getattr(info, 'previous_close', None),
                'open': getattr(info, 'open', None),
                'day_high': getattr(info, 'day_high', None),
                'day_low': getattr(info, 'day_low', None),
                'volume': getattr(info, 'last_volume', None),
                'market_cap': getattr(info, 'market_cap', None),
                'change_pct': None,  # Calculate if needed
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None

    def get_quotes_bulk(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get current quotes for multiple stocks.

        Args:
            symbols: List of stock symbols in Fyers format

        Returns:
            Dict mapping symbol to quote data
        """
        yf = _get_yfinance()
        results = {}

        try:
            # Convert all symbols
            yf_symbols = [self._convert_to_yfinance_symbol(s) for s in symbols if s]
            valid_symbols = [s for s in yf_symbols if s]

            if not valid_symbols:
                return results

            logger.info(f"Fetching quotes for {len(valid_symbols)} stocks via yfinance")

            # Use yfinance download for bulk quotes (more efficient)
            # Get just today's data for current prices
            df = yf.download(
                tickers=valid_symbols,
                period='1d',
                interval='1d',
                progress=False,
                threads=True
            )

            if df.empty:
                logger.warning("No quote data received from yfinance")
                return results

            # Handle both single and multiple ticker responses
            if len(valid_symbols) == 1:
                # Single ticker - different DataFrame structure
                yf_sym = valid_symbols[0]
                fyers_sym = self._convert_from_yfinance_symbol(yf_sym)

                if not df.empty:
                    latest = df.iloc[-1]
                    results[fyers_sym] = {
                        'symbol': fyers_sym,
                        'last_price': float(latest.get('Close', 0)),
                        'open': float(latest.get('Open', 0)),
                        'high': float(latest.get('High', 0)),
                        'low': float(latest.get('Low', 0)),
                        'volume': int(latest.get('Volume', 0)),
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                # Multiple tickers - multi-level column structure
                for yf_sym in valid_symbols:
                    try:
                        fyers_sym = self._convert_from_yfinance_symbol(yf_sym)

                        # Extract data for this symbol
                        close = df['Close'][yf_sym].iloc[-1] if 'Close' in df.columns.get_level_values(0) else None
                        open_price = df['Open'][yf_sym].iloc[-1] if 'Open' in df.columns.get_level_values(0) else None
                        high = df['High'][yf_sym].iloc[-1] if 'High' in df.columns.get_level_values(0) else None
                        low = df['Low'][yf_sym].iloc[-1] if 'Low' in df.columns.get_level_values(0) else None
                        volume = df['Volume'][yf_sym].iloc[-1] if 'Volume' in df.columns.get_level_values(0) else None

                        if close and not pd.isna(close):
                            results[fyers_sym] = {
                                'symbol': fyers_sym,
                                'last_price': float(close),
                                'open': float(open_price) if open_price and not pd.isna(open_price) else None,
                                'high': float(high) if high and not pd.isna(high) else None,
                                'low': float(low) if low and not pd.isna(low) else None,
                                'volume': int(volume) if volume and not pd.isna(volume) else None,
                                'timestamp': datetime.now().isoformat()
                            }
                    except Exception as e:
                        logger.debug(f"Error extracting quote for {yf_sym}: {e}")
                        continue

            logger.info(f"Successfully fetched {len(results)} quotes")
            return results

        except Exception as e:
            logger.error(f"Error in bulk quote fetch: {e}")
            return results

    def is_market_open(self) -> bool:
        """
        Check if NSE market is currently open.

        Returns:
            True if market is open, False otherwise
        """
        now = datetime.now()

        # Check if weekend
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check market hours (9:15 AM to 3:30 PM IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        return market_open <= now <= market_close

    def get_nifty50_stocks(self) -> List[str]:
        """
        Get list of NIFTY 50 stocks in Fyers format.

        Returns:
            List of NIFTY 50 stock symbols
        """
        # NIFTY 50 constituents (as of 2024)
        nifty50 = [
            'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
            'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL',
            'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
            'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE',
            'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'ITC',
            'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LT',
            'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC',
            'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SUNPHARMA',
            'TCS', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TECHM',
            'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO', 'LTIM'
        ]

        return [f"NSE:{symbol}-EQ" for symbol in nifty50]

    def test_connection(self) -> Dict[str, Any]:
        """
        Test yfinance connectivity by fetching NIFTY 50 index data.

        Returns:
            Dict with connection test results
        """
        try:
            yf = _get_yfinance()

            # Test with NIFTY 50 index
            ticker = yf.Ticker("^NSEI")
            info = ticker.fast_info

            last_price = getattr(info, 'last_price', None)

            if last_price:
                return {
                    'success': True,
                    'provider': 'yfinance',
                    'test_symbol': '^NSEI (NIFTY 50)',
                    'last_price': last_price,
                    'message': 'YFinance connection successful'
                }
            else:
                return {
                    'success': False,
                    'provider': 'yfinance',
                    'message': 'Could not fetch NIFTY 50 data'
                }

        except ImportError:
            return {
                'success': False,
                'provider': 'yfinance',
                'message': 'yfinance library not installed. Run: pip install yfinance'
            }
        except Exception as e:
            return {
                'success': False,
                'provider': 'yfinance',
                'message': f'Connection test failed: {str(e)}'
            }


# Singleton instance
_yfinance_service = None

def get_yfinance_service() -> YFinanceDataService:
    """Get singleton instance of YFinanceDataService."""
    global _yfinance_service
    if _yfinance_service is None:
        _yfinance_service = YFinanceDataService()
    return _yfinance_service
