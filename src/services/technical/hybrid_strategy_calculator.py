"""
Hybrid Strategy Calculator Service
Combines RS Rating + Wave Indicators + 8-21 EMA + DeMarker + Fibonacci
Simple, clean implementation for swing trading stock selection
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


class HybridStrategyCalculator:
    """
    Hybrid strategy combining best of both approaches:
    1. RS Rating (1-99) for stock ranking
    2. Wave Delta for momentum confirmation
    3. 8-21 EMA for trend identification
    4. DeMarker for pullback timing
    5. Fibonacci for dynamic targets
    """

    def __init__(self, session: Session):
        self.session = session
        self.nifty_data = None  # Cache for NIFTY 50 data

    def calculate_all_indicators(
        self,
        symbols: List[str],
        lookback_days: int = 252
    ) -> Dict[str, Dict]:
        """
        Calculate complete hybrid strategy indicators for all stocks.

        Args:
            symbols: List of stock symbols
            lookback_days: Historical data period (default 252 = 1 year)

        Returns:
            Dict mapping symbol to all indicators
        """
        try:
            logger.info(f"Calculating hybrid indicators for {len(symbols)} stocks...")

            # Step 1: Calculate RS Ratings for all stocks (percentile-based)
            logger.info("Step 1/5: Calculating RS Ratings...")
            rs_ratings = self._calculate_rs_ratings_bulk(symbols, lookback_days)

            # Step 2: Calculate technical indicators for each stock
            logger.info("Step 2/5: Calculating technical indicators...")
            results = {}

            for i, symbol in enumerate(symbols):
                if (i + 1) % 100 == 0:
                    logger.info(f"  Progress: {i + 1}/{len(symbols)} stocks")

                try:
                    # Get historical data
                    stock_data = self._get_historical_data(symbol, lookback_days)
                    if stock_data is None or len(stock_data) < 60:
                        continue

                    # Calculate all indicators
                    indicators = self._calculate_stock_indicators(symbol, stock_data)

                    # Add RS Rating
                    indicators['rs_rating'] = rs_ratings.get(symbol, 50.0)

                    results[symbol] = indicators

                except Exception as e:
                    logger.error(f"Error calculating indicators for {symbol}: {e}")
                    continue

            # Step 3: Calculate EMA trend scores (PRIMARY RANKING METRIC)
            logger.info("Step 3/5: Calculating EMA trend scores...")
            self._calculate_ema_trend_scores(results)

            # Step 4: Set ranking scores and generate signals
            logger.info("Step 4/5: Setting ranking scores and generating signals...")
            self._calculate_composite_scores(results)

            logger.info(f"Successfully calculated indicators for {len(results)}/{len(symbols)} stocks")
            return results

        except Exception as e:
            logger.error(f"Error in calculate_all_indicators: {e}", exc_info=True)
            return {}

    def _calculate_stock_indicators(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators for a single stock."""

        # Calculate 8 & 21 EMA
        df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()

        # Calculate DeMarker oscillator
        demarker = self._calculate_demarker(df)

        # Calculate Wave indicators
        wave_data = self._calculate_wave_indicators(df)

        # Calculate Fibonacci targets
        fib_targets = self._calculate_fibonacci_targets(df)

        # Get latest values
        current_price = float(df['close'].iloc[-1])
        ema_8 = float(df['ema_8'].iloc[-1])
        ema_21 = float(df['ema_21'].iloc[-1])

        return {
            # Price & EMAs
            'current_price': current_price,
            'ema_8': ema_8,
            'ema_21': ema_21,

            # DeMarker
            'demarker': demarker,

            # Wave indicators
            'fast_wave': wave_data['fast_wave'],
            'slow_wave': wave_data['slow_wave'],
            'delta': wave_data['delta'],

            # Fibonacci targets
            'fib_target_1': fib_targets['target_1'],
            'fib_target_2': fib_targets['target_2'],
            'fib_target_3': fib_targets['target_3'],

            # Raw data for further calculations
            '_df': df  # Keep for later use
        }

    def _calculate_demarker(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate DeMarker oscillator (0-1 range).

        DeMarker measures buying and selling pressure.
        - < 0.30: Oversold (good buy opportunity)
        - > 0.70: Overbought (potential sell)
        """
        try:
            # Calculate DeMax and DeMin
            high_diff = df['high'].diff()
            low_diff = df['low'].diff()

            demax = high_diff.where(high_diff > 0, 0)
            demin = (-low_diff).where(low_diff < 0, 0)

            # Calculate SMA of DeMax and DeMin
            demax_sma = demax.rolling(window=period).mean()
            demin_sma = demin.rolling(window=period).mean()

            # Calculate DeMarker
            demarker = demax_sma / (demax_sma + demin_sma)

            # Get latest value
            latest = float(demarker.iloc[-1])

            # Handle NaN
            if pd.isna(latest):
                return 0.5  # Neutral if calculation fails

            return max(0.0, min(1.0, latest))  # Clamp to 0-1

        except Exception as e:
            logger.error(f"Error calculating DeMarker: {e}")
            return 0.5  # Neutral value

    def _calculate_wave_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Wave indicators (same as existing system)."""
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
            return {
                'fast_wave': float(df['fast_wave'].iloc[-1]),
                'slow_wave': float(df['slow_wave'].iloc[-1]),
                'delta': float(df['delta'].iloc[-1])
            }

        except Exception as e:
            logger.error(f"Error calculating wave indicators: {e}")
            return {'fast_wave': 0.0, 'slow_wave': 0.0, 'delta': 0.0}

    def _calculate_fibonacci_targets(self, df: pd.DataFrame, lookback: int = 60) -> Dict:
        """
        Calculate Fibonacci extension targets based on recent swing high/low.

        Returns targets at 127.2%, 161.8%, and 200% extensions.
        """
        try:
            # Get recent data for swing analysis
            recent_df = df.tail(lookback).copy()

            # Find swing high and swing low in recent period
            swing_high_idx = recent_df['high'].idxmax()
            swing_low_idx = recent_df['low'].idxmin()

            swing_high = recent_df.loc[swing_high_idx, 'high']
            swing_low = recent_df.loc[swing_low_idx, 'low']

            # Current price (entry point)
            current_price = df['close'].iloc[-1]

            # Calculate Fibonacci range
            fib_range = swing_high - swing_low

            # Calculate extension levels from current price
            # Using swing low as base, extending from current price
            target_1 = current_price + (fib_range * 0.272)  # 127.2% from current
            target_2 = current_price + (fib_range * 0.618)  # 161.8% (golden ratio)
            target_3 = current_price + (fib_range * 1.000)  # 200%

            return {
                'target_1': round(float(target_1), 2),
                'target_2': round(float(target_2), 2),
                'target_3': round(float(target_3), 2)
            }

        except Exception as e:
            logger.error(f"Error calculating Fibonacci targets: {e}")
            current_price = df['close'].iloc[-1]
            return {
                'target_1': round(float(current_price * 1.05), 2),
                'target_2': round(float(current_price * 1.10), 2),
                'target_3': round(float(current_price * 1.15), 2)
            }

    def _calculate_rs_ratings_bulk(
        self,
        symbols: List[str],
        lookback_days: int
    ) -> Dict[str, float]:
        """
        Calculate RS Ratings for all stocks using percentile ranking.
        (Reuse existing logic from improved_indicators_calculator.py)
        """
        try:
            # Get NIFTY data
            if self.nifty_data is None:
                self.nifty_data = self._get_nifty_data()

            nifty_returns = [0.0, 0.0, 0.0, 0.0]
            if self.nifty_data is not None:
                nifty_returns = self._calculate_quarterly_returns(self.nifty_data)

            # Calculate performance for all stocks
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

            # Convert to percentile-based RS Ratings (1-99)
            return self._calculate_percentile_ratings(all_performance)

        except Exception as e:
            logger.error(f"Error calculating RS ratings: {e}")
            return {}

    def _calculate_percentile_ratings(self, performance_scores: Dict[str, float]) -> Dict[str, float]:
        """Convert performance scores to percentile-based ratings (1-99)."""
        if not performance_scores:
            return {}

        # Sort by performance
        sorted_stocks = sorted(performance_scores.items(), key=lambda x: x[1])
        total_stocks = len(sorted_stocks)

        rs_ratings = {}
        for rank, (symbol, score) in enumerate(sorted_stocks):
            # Calculate percentile (0 to 100)
            percentile = (rank / (total_stocks - 1)) * 100 if total_stocks > 1 else 50

            # Convert to 1-99 scale
            rs_rating = max(1, min(99, int(percentile)))
            rs_ratings[symbol] = float(rs_rating)

        return rs_ratings

    # REMOVED: _normalize_wave_scores method
    # Wave momentum score is NO LONGER used for ranking.
    # Ranking is based on EMA Trend Score ONLY.
    # Wave indicators (Delta) are used for BUY/SELL signals, not scoring.

    def _calculate_ema_trend_scores(self, results: Dict[str, Dict]) -> None:
        """Calculate EMA trend score (0-100) based on 8-21 EMA configuration."""
        for symbol, result in results.items():
            try:
                price = result.get('current_price', 0)
                ema_8 = result.get('ema_8', 0)
                ema_21 = result.get('ema_21', 0)

                if price == 0 or ema_8 == 0 or ema_21 == 0:
                    result['ema_trend_score'] = 50.0
                    continue

                # Score based on EMA configuration
                score = 50.0  # Neutral base

                # Check bullish configuration: Price > EMA_8 > EMA_21
                if price > ema_8 > ema_21:
                    # Strong bull trend
                    # Calculate distance from EMAs as percentage
                    price_above_8 = ((price - ema_8) / price) * 100
                    ema8_above_21 = ((ema_8 - ema_21) / ema_8) * 100

                    # Score increases with separation
                    score = 70 + min(30, (price_above_8 + ema8_above_21) * 5)

                elif price > ema_8 and ema_8 <= ema_21:
                    # Early uptrend (8 EMA crossing up)
                    score = 60.0

                elif price < ema_8 < ema_21:
                    # Strong bear trend
                    score = 30.0

                elif price < ema_8 and ema_8 >= ema_21:
                    # Early downtrend
                    score = 40.0

                result['ema_trend_score'] = round(score, 2)

            except Exception as e:
                logger.error(f"Error calculating EMA trend score for {symbol}: {e}")
                result['ema_trend_score'] = 50.0

    def _calculate_composite_scores(self, results: Dict[str, Dict]) -> None:
        """
        Calculate ranking score and generate buy/sell signals.

        CORRECTED LOGIC:
        - Ranking Score = EMA Trend Score ONLY (not weighted composite)
        - Signals based on Wave crossovers + confirmations
        - RS Rating and other indicators are FILTERS, not scores
        """
        for symbol, result in results.items():
            try:
                # Use EMA trend score as the PRIMARY ranking metric
                ema_score = result.get('ema_trend_score', 50.0)
                result['hybrid_composite_score'] = round(ema_score, 2)

                # Generate buy/sell signals based on Wave + EMA + DeMarker
                signals = self._generate_signals(result)
                result.update(signals)

            except Exception as e:
                logger.error(f"Error calculating scores for {symbol}: {e}")
                result['hybrid_composite_score'] = 50.0
                result['buy_signal'] = False
                result['sell_signal'] = False
                result['signal_quality'] = 'none'

    def _generate_signals(self, indicators: Dict) -> Dict:
        """
        Generate buy/sell signals based on CORRECTED hybrid strategy rules.

        BUY Signal Logic:
        1. Wave Signal (REQUIRED): Delta > 0 (fast wave crossed above slow wave)
        2. EMA Trend Confirmation (REQUIRED): Price > EMA_8 > EMA_21
        3. DeMarker Timing (OPTIONAL): < 0.30 for high quality entry

        Quality Rating:
        - HIGH: Wave + EMA + DeMarker (all 3)
        - MEDIUM: Wave + EMA (2 of 3)
        - LOW: Wave only (1 of 3)
        - NONE: No wave signal

        SELL Signal:
        - Wave: Delta < 0 (fast wave crossed below slow wave)
        - EMA: Price < EMA_8 < EMA_21 (downtrend confirmation)

        NOTE: RS Rating should be filtered in saga (>70), not checked here!
        """
        try:
            price = indicators.get('current_price', 0)
            ema_8 = indicators.get('ema_8', 0)
            ema_21 = indicators.get('ema_21', 0)
            demarker = indicators.get('demarker', 0.5)
            delta = indicators.get('delta', 0)

            # BUY SIGNAL LOGIC
            # Condition 1: Wave signal (PRIMARY)
            wave_buy = delta > 0

            # Condition 2: EMA uptrend (CONFIRMATION)
            ema_uptrend = price > ema_8 > ema_21

            # Condition 3: DeMarker oversold (TIMING)
            demarker_oversold = demarker < 0.30

            # Determine buy signal and quality
            if wave_buy and ema_uptrend and demarker_oversold:
                # Perfect entry: Wave + Trend + Timing
                buy_signal = True
                quality = 'high'
                conditions_met = 3
            elif wave_buy and ema_uptrend:
                # Good entry: Wave + Trend confirmed
                buy_signal = True
                quality = 'medium'
                conditions_met = 2
            elif wave_buy:
                # Weak entry: Wave only (no trend confirmation)
                buy_signal = True
                quality = 'low'
                conditions_met = 1
            else:
                # No wave signal = no buy
                buy_signal = False
                quality = 'none'
                conditions_met = 0

            # SELL SIGNAL LOGIC
            # Wave turns negative AND downtrend confirmed
            sell_signal = (delta < 0) and (price < ema_8 < ema_21)

            return {
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'signal_quality': quality,
                'conditions_met': conditions_met
            }

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {
                'buy_signal': False,
                'sell_signal': False,
                'signal_quality': 'none',
                'conditions_met': 0
            }

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
        """Get NIFTY 50 index data for RS Rating calculations."""
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


def get_hybrid_strategy_calculator(session: Session) -> HybridStrategyCalculator:
    """Factory function to get hybrid strategy calculator instance."""
    return HybridStrategyCalculator(session)
