"""
Improved Technical Indicators Calculator
Fixes RS Rating to use true percentile ranking and normalizes Delta properly
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


class ImprovedTechnicalIndicatorsCalculator:
    """
    Improved technical indicators calculator with:
    1. True percentile-based RS Rating (1-99)
    2. Properly normalized Delta scores
    3. Statistical distribution analysis
    """

    def __init__(self, session: Session):
        self.session = session
        self.nifty_data = None
        self.rs_performance_cache = None  # Cache for all stocks' performance
        self.delta_distribution_cache = None  # Cache for delta statistics

    def calculate_all_indicators_bulk(
        self,
        symbols: List[str],
        lookback_days: int = 252
    ) -> Dict[str, Dict]:
        """
        Calculate indicators for multiple stocks with proper percentile ranking.

        This is the CORRECT way to calculate RS Rating:
        1. Calculate performance for ALL stocks
        2. Rank them (percentile)
        3. Assign 1-99 based on position

        Args:
            symbols: List of stock symbols
            lookback_days: Historical data period

        Returns:
            Dict mapping symbol to indicators
        """
        try:
            logger.info(f"Calculating indicators for {len(symbols)} stocks...")

            # Step 1: Get all stock performance data
            logger.info("Step 1: Calculating performance for all stocks...")
            all_performance = self._calculate_all_stock_performance(symbols, lookback_days)

            if not all_performance:
                logger.error("No performance data calculated")
                return {}

            # Step 2: Calculate RS Ratings using percentile ranking
            logger.info("Step 2: Calculating true percentile-based RS Ratings...")
            rs_ratings = self._calculate_percentile_rs_ratings(all_performance)

            # Step 3: Calculate wave indicators for all stocks
            logger.info("Step 3: Calculating wave indicators...")
            all_deltas = {}
            results = {}

            for i, symbol in enumerate(symbols):
                if (i + 1) % 100 == 0:
                    logger.info(f"Progress: {i + 1}/{len(symbols)} stocks")

                # Get historical data
                stock_data = self._get_historical_data(symbol, lookback_days)
                if stock_data is None or len(stock_data) < 60:
                    continue

                # Calculate wave indicators
                waves = self._calculate_wave_indicators(stock_data)
                all_deltas[symbol] = waves['delta']

                # Store preliminary results
                results[symbol] = {
                    'rs_rating': rs_ratings.get(symbol, 50.0),
                    'fast_wave': waves['fast_wave'],
                    'slow_wave': waves['slow_wave'],
                    'delta': waves['delta'],
                    'delta_raw': waves['delta']  # Store raw for normalization
                }

            # Step 4: Normalize delta scores using actual distribution
            logger.info("Step 4: Normalizing delta scores...")
            normalized_deltas = self._normalize_delta_scores(all_deltas)

            # Step 5: Generate buy/sell signals and finalize
            logger.info("Step 5: Generating signals...")
            for symbol, indicators in results.items():
                # Update with normalized delta
                indicators['delta_normalized'] = normalized_deltas.get(symbol, 0.0)

                # Generate signals
                buy_signal, sell_signal = self._detect_signals(indicators)
                indicators['buy_signal'] = buy_signal
                indicators['sell_signal'] = sell_signal

                logger.debug(
                    f"{symbol}: RS={indicators['rs_rating']:.1f}, "
                    f"Delta_norm={indicators['delta_normalized']:.2f}, "
                    f"Buy={buy_signal}, Sell={sell_signal}"
                )

            logger.info(f"Successfully calculated indicators for {len(results)}/{len(symbols)} stocks")
            return results

        except Exception as e:
            logger.error(f"Error calculating bulk indicators: {e}", exc_info=True)
            return {}

    def _calculate_all_stock_performance(
        self,
        symbols: List[str],
        lookback_days: int
    ) -> Dict[str, float]:
        """
        Calculate weighted quarterly performance for all stocks.

        Returns:
            Dict mapping symbol to weighted performance score
        """
        try:
            # Get NIFTY data for comparison
            if self.nifty_data is None:
                self.nifty_data = self._get_nifty_data()

            nifty_returns = [0.0, 0.0, 0.0, 0.0]
            if self.nifty_data is not None:
                nifty_returns = self._calculate_quarterly_returns(self.nifty_data)

            all_performance = {}

            for symbol in symbols:
                stock_data = self._get_historical_data(symbol, lookback_days)

                if stock_data is None or len(stock_data) < 252:
                    continue

                # Calculate quarterly returns
                stock_returns = self._calculate_quarterly_returns(stock_data)

                # Calculate relative performance vs NIFTY
                relative_performance = []
                for i in range(len(stock_returns)):
                    rel_perf = stock_returns[i] - nifty_returns[i]
                    relative_performance.append(rel_perf)

                # Apply weights (Q1=40%, Q2=20%, Q3=20%, Q4=20%)
                weights = [0.4, 0.2, 0.2, 0.2]
                weighted_score = sum(
                    rel_perf * weight
                    for rel_perf, weight in zip(relative_performance, weights)
                )

                all_performance[symbol] = weighted_score

            return all_performance

        except Exception as e:
            logger.error(f"Error calculating all stock performance: {e}")
            return {}

    def _calculate_percentile_rs_ratings(
        self,
        performance_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate TRUE percentile-based RS Ratings (1-99).

        This is the CORRECT implementation:
        - Bottom 10% of stocks get RS 1-10
        - Top 10% of stocks get RS 90-99
        - Median stock gets RS ~50

        Args:
            performance_scores: Dict of symbol -> performance score

        Returns:
            Dict of symbol -> RS Rating (1-99)
        """
        try:
            if not performance_scores:
                return {}

            # Convert to sorted list
            sorted_stocks = sorted(
                performance_scores.items(),
                key=lambda x: x[1]
            )

            total_stocks = len(sorted_stocks)
            rs_ratings = {}

            for rank, (symbol, score) in enumerate(sorted_stocks):
                # Calculate percentile (0 to 100)
                percentile = (rank / (total_stocks - 1)) * 100 if total_stocks > 1 else 50

                # Convert to 1-99 scale (avoid 0 and 100)
                rs_rating = max(1, min(99, int(percentile)))

                rs_ratings[symbol] = float(rs_rating)

            logger.info(
                f"RS Ratings calculated: "
                f"Min={min(rs_ratings.values()):.0f}, "
                f"Max={max(rs_ratings.values()):.0f}, "
                f"Median={np.median(list(rs_ratings.values())):.0f}"
            )

            return rs_ratings

        except Exception as e:
            logger.error(f"Error calculating percentile RS ratings: {e}")
            return {}

    def _normalize_delta_scores(
        self,
        delta_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize delta scores to -40 to +40 range using actual distribution.

        Instead of arbitrary scaling (delta * 100), this uses statistical normalization:
        - Calculate mean and std of all deltas
        - Normalize to z-scores
        - Scale to -40 to +40 range

        Args:
            delta_scores: Dict of symbol -> raw delta value

        Returns:
            Dict of symbol -> normalized delta (-40 to +40)
        """
        try:
            if not delta_scores:
                return {}

            # Get all delta values
            deltas = list(delta_scores.values())

            # Calculate statistics
            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas)

            if std_delta == 0:
                logger.warning("Delta std is zero, all deltas are same")
                return {symbol: 0.0 for symbol in delta_scores}

            # Normalize to z-scores and scale to -40 to +40
            normalized = {}
            for symbol, delta in delta_scores.items():
                # Z-score
                z_score = (delta - mean_delta) / std_delta

                # Clamp to ±3 std (covers 99.7% of normal distribution)
                z_score = max(-3, min(3, z_score))

                # Scale to -40 to +40
                normalized_delta = (z_score / 3) * 40

                normalized[symbol] = normalized_delta

            logger.info(
                f"Delta normalization: "
                f"Mean={mean_delta:.6f}, "
                f"Std={std_delta:.6f}, "
                f"Normalized range=[{min(normalized.values()):.2f}, {max(normalized.values()):.2f}]"
            )

            return normalized

        except Exception as e:
            logger.error(f"Error normalizing delta scores: {e}")
            return {symbol: 0.0 for symbol in delta_scores}

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

    def _get_nifty_data(self) -> Optional[pd.DataFrame]:
        """Get NIFTY 50 index data."""
        try:
            nifty_symbols = ['NSE:NIFTY-INDEX', 'NSE:NIFTY50-INDEX', 'NSE:RELIANCE-EQ']

            for symbol in nifty_symbols:
                data = self._get_historical_data(symbol, 252)
                if data is not None and len(data) >= 200:
                    logger.info(f"Using {symbol} as market index proxy")
                    return data

            logger.warning("Could not find NIFTY 50 data")
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

    def _calculate_wave_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Wave Indicators based on EMA."""
        try:
            # Calculate HLC3
            df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3

            # Apply 9-period EMA to HLC3
            df['ema_9'] = df['hlc3'].ewm(span=9, adjust=False).mean()

            # Calculate deviation
            df['deviation'] = df['hlc3'] - df['ema_9']

            # Fast Wave (12-period EMA of deviation)
            df['fast_wave'] = df['deviation'].ewm(span=12, adjust=False).mean()

            # Slow Wave (3-period MA of Fast Wave)
            df['slow_wave'] = df['fast_wave'].rolling(window=3).mean()

            # Delta
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

    def _detect_signals(self, indicators: Dict[str, float]) -> Tuple[bool, bool]:
        """
        Detect buy/sell signals based on wave crossovers.

        Buy Signal: Fast Wave > Slow Wave and Delta > 0
        Sell Signal: Fast Wave < Slow Wave and Delta < 0
        """
        fast_wave = indicators['fast_wave']
        slow_wave = indicators['slow_wave']
        delta = indicators['delta']

        buy_signal = fast_wave > slow_wave and delta > 0
        sell_signal = fast_wave < slow_wave and delta < 0

        return buy_signal, sell_signal


def get_improved_indicators_calculator(session: Session) -> ImprovedTechnicalIndicatorsCalculator:
    """Factory function to get improved indicators calculator."""
    return ImprovedTechnicalIndicatorsCalculator(session)
