"""
Technical Indicators Calculator Service
Calculates RS Rating, Wave Indicators, and Buy/Sell Signals
Based on the approach from the stock screener article
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


class TechnicalIndicatorsCalculator:
    """
    Calculates technical indicators for stock screening:
    - RS Rating (Relative Strength vs NIFTY 50)
    - Wave Indicators (Fast Wave, Slow Wave, Delta)
    - Buy/Sell Signals (based on wave crossovers)
    """

    def __init__(self, session: Session):
        self.session = session
        self.nifty_data = None  # Cache for NIFTY 50 data

    def calculate_all_indicators(self, symbol: str, lookback_days: int = 252) -> Optional[Dict]:
        """
        Calculate all technical indicators for a stock.

        Args:
            symbol: Stock symbol (e.g., 'NSE:RELIANCE-EQ')
            lookback_days: Number of days of historical data to use (default 252 = 1 trading year)

        Returns:
            Dict with indicators or None if insufficient data
        """
        try:
            # Get historical data for the stock
            stock_data = self._get_historical_data(symbol, lookback_days)

            if stock_data is None or len(stock_data) < 60:  # Need at least 60 days for calculations
                logger.warning(f"{symbol}: Insufficient data ({len(stock_data) if stock_data is not None else 0} days)")
                return None

            # Calculate each indicator
            rs_rating = self._calculate_rs_rating(symbol, stock_data)
            waves = self._calculate_wave_indicators(stock_data)

            # Generate buy/sell signals
            buy_signal, sell_signal = self._detect_signals(waves)

            result = {
                'rs_rating': rs_rating,
                'fast_wave': waves['fast_wave'],
                'slow_wave': waves['slow_wave'],
                'delta': waves['delta'],
                'buy_signal': buy_signal,
                'sell_signal': sell_signal
            }

            logger.debug(f"{symbol}: RS={rs_rating:.1f}, FastWave={waves['fast_wave']:.4f}, "
                        f"SlowWave={waves['slow_wave']:.4f}, Delta={waves['delta']:.4f}, "
                        f"Buy={buy_signal}, Sell={sell_signal}")

            return result

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}", exc_info=True)
            return None

    def _get_historical_data(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data from database."""
        try:
            query = text("""
                SELECT date, open, high, low, close, volume
                FROM historical_data
                WHERE symbol = :symbol
                AND date >= CURRENT_DATE - INTERVAL ':lookback_days days'
                ORDER BY date ASC
            """)

            result = self.session.execute(query, {'symbol': symbol, 'lookback_days': lookback_days})
            data = result.fetchall()

            if not data:
                return None

            df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    def _calculate_rs_rating(self, symbol: str, stock_data: pd.DataFrame) -> float:
        """
        Calculate Relative Strength Rating (1-99) compared to NIFTY 50.

        Uses quarterly performance with weighted scoring:
        - Most recent quarter: 40%
        - Second quarter: 20%
        - Third quarter: 20%
        - Fourth quarter: 20%

        Returns:
            RS Rating from 1 to 99 (higher = stronger relative performance)
        """
        try:
            # Get NIFTY 50 data (using NIFTY 50 index or a proxy like NSE:NIFTY50-INDEX)
            if self.nifty_data is None:
                self.nifty_data = self._get_nifty_data()

            if self.nifty_data is None or len(stock_data) < 252:
                # Fallback: calculate absolute performance if NIFTY data unavailable
                return self._calculate_absolute_rating(stock_data)

            # Calculate quarterly returns for stock
            stock_returns = self._calculate_quarterly_returns(stock_data)

            # Calculate quarterly returns for NIFTY
            nifty_returns = self._calculate_quarterly_returns(self.nifty_data)

            # Calculate relative performance vs NIFTY
            relative_performance = []
            for i in range(len(stock_returns)):
                rel_perf = stock_returns[i] - nifty_returns[i] if i < len(nifty_returns) else stock_returns[i]
                relative_performance.append(rel_perf)

            # Apply weights (Q1=40%, Q2=20%, Q3=20%, Q4=20%)
            weights = [0.4, 0.2, 0.2, 0.2]
            weighted_score = sum(rel_perf * weight
                                for rel_perf, weight in zip(relative_performance, weights))

            # Convert to 1-99 scale (normalize to percentile)
            # For now, use simple normalization: 0% = 50, positive = higher, negative = lower
            rs_rating = 50 + (weighted_score * 50)  # Scale to 1-99 range
            rs_rating = max(1, min(99, rs_rating))  # Clamp to 1-99

            return round(rs_rating, 2)

        except Exception as e:
            logger.error(f"Error calculating RS rating for {symbol}: {e}")
            return 50.0  # Return neutral rating on error

    def _get_nifty_data(self) -> Optional[pd.DataFrame]:
        """Get NIFTY 50 index data (or use a proxy like NSE:NIFTY-INDEX)."""
        try:
            # Try to get NIFTY 50 index data
            # If not available, use a large-cap stock as proxy (e.g., RELIANCE)
            nifty_symbols = ['NSE:NIFTY-INDEX', 'NSE:NIFTY50-INDEX', 'NSE:RELIANCE-EQ']

            for symbol in nifty_symbols:
                data = self._get_historical_data(symbol, 252)
                if data is not None and len(data) >= 200:
                    logger.info(f"Using {symbol} as market index proxy")
                    return data

            logger.warning("Could not find NIFTY 50 data, RS ratings will use absolute performance")
            return None

        except Exception as e:
            logger.error(f"Error fetching NIFTY data: {e}")
            return None

    def _calculate_quarterly_returns(self, df: pd.DataFrame) -> List[float]:
        """Calculate returns for last 4 quarters (63 trading days each)."""
        if len(df) < 252:
            return [0.0, 0.0, 0.0, 0.0]

        returns = []
        for i in range(4):
            start_idx = len(df) - ((i + 1) * 63)
            end_idx = len(df) - (i * 63) if i > 0 else len(df)

            if start_idx < 0:
                returns.append(0.0)
                continue

            start_price = df.iloc[start_idx]['close']
            end_price = df.iloc[end_idx - 1]['close']

            quarterly_return = ((end_price - start_price) / start_price) * 100
            returns.append(quarterly_return)

        return returns

    def _calculate_absolute_rating(self, df: pd.DataFrame) -> float:
        """Fallback: Calculate rating based on absolute performance (no NIFTY comparison)."""
        if len(df) < 20:
            return 50.0

        # Calculate simple momentum score based on recent performance
        current_price = df.iloc[-1]['close']
        price_20d_ago = df.iloc[-20]['close'] if len(df) >= 20 else df.iloc[0]['close']
        price_60d_ago = df.iloc[-60]['close'] if len(df) >= 60 else df.iloc[0]['close']

        momentum_20d = ((current_price - price_20d_ago) / price_20d_ago) * 100
        momentum_60d = ((current_price - price_60d_ago) / price_60d_ago) * 100

        # Convert to 1-99 scale (positive momentum = higher rating)
        avg_momentum = (momentum_20d * 0.6 + momentum_60d * 0.4)
        rating = 50 + avg_momentum  # Center at 50
        rating = max(1, min(99, rating))

        return round(rating, 2)

    def _calculate_wave_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Wave Indicators based on EMA.

        Steps:
        1. Calculate HLC3 (average of high, low, close)
        2. Apply 9-period EMA to HLC3
        3. Calculate deviation from EMA
        4. Create Fast Wave using 12-period EMA of deviation
        5. Create Slow Wave as 3-period MA of Fast Wave
        6. Calculate Delta (Fast - Slow)

        Returns:
            Dict with fast_wave, slow_wave, delta
        """
        try:
            # Step 1: Calculate HLC3
            df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3

            # Step 2: Apply 9-period EMA to HLC3
            df['ema_9'] = df['hlc3'].ewm(span=9, adjust=False).mean()

            # Step 3: Calculate deviation
            df['deviation'] = df['hlc3'] - df['ema_9']

            # Step 4: Fast Wave (12-period EMA of deviation)
            df['fast_wave'] = df['deviation'].ewm(span=12, adjust=False).mean()

            # Step 5: Slow Wave (3-period MA of Fast Wave)
            df['slow_wave'] = df['fast_wave'].rolling(window=3).mean()

            # Step 6: Delta
            df['delta'] = df['fast_wave'] - df['slow_wave']

            # Get latest values
            fast_wave = float(df['fast_wave'].iloc[-1])
            slow_wave = float(df['slow_wave'].iloc[-1])
            delta = float(df['delta'].iloc[-1])

            return {
                'fast_wave': fast_wave,
                'slow_wave': slow_wave,
                'delta': delta
            }

        except Exception as e:
            logger.error(f"Error calculating wave indicators: {e}")
            return {
                'fast_wave': 0.0,
                'slow_wave': 0.0,
                'delta': 0.0
            }

    def _detect_signals(self, waves: Dict[str, float]) -> Tuple[bool, bool]:
        """
        Detect buy/sell signals based on wave crossovers.

        Buy Signal: Fast Wave > Slow Wave and Delta > 0
        Sell Signal: Fast Wave < Slow Wave and Delta < 0

        Returns:
            Tuple of (buy_signal, sell_signal)
        """
        fast_wave = waves['fast_wave']
        slow_wave = waves['slow_wave']
        delta = waves['delta']

        buy_signal = fast_wave > slow_wave and delta > 0
        sell_signal = fast_wave < slow_wave and delta < 0

        return buy_signal, sell_signal

    def calculate_indicators_bulk(self, symbols: List[str], lookback_days: int = 252) -> Dict[str, Dict]:
        """
        Calculate indicators for multiple stocks in bulk.

        Args:
            symbols: List of stock symbols
            lookback_days: Number of days of historical data

        Returns:
            Dict mapping symbol to indicators dict
        """
        results = {}

        logger.info(f"Calculating indicators for {len(symbols)} stocks...")

        for i, symbol in enumerate(symbols):
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(symbols)} stocks processed")

            indicators = self.calculate_all_indicators(symbol, lookback_days)
            if indicators:
                results[symbol] = indicators

        logger.info(f"Successfully calculated indicators for {len(results)}/{len(symbols)} stocks")

        return results


def get_indicators_calculator(session: Session) -> TechnicalIndicatorsCalculator:
    """Factory function to get indicators calculator instance."""
    return TechnicalIndicatorsCalculator(session)
