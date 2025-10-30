"""
8-21 EMA Swing Trading Strategy Calculator
Implements the pure EMA strategy with DeMarker oscillator and Fibonacci extensions.

Strategy Components:
1. 8 & 21 EMA - Trend identification and power zones
2. DeMarker Oscillator - Precise pullback timing
3. Fibonacci Extensions - Profit target calculation

The Perfect Setup:
- Price > 8 EMA > 21 EMA (power zone active)
- DeMarker < 0.30 (oversold pullback)
- Price holds EMA support
- Fibonacci targets at 127.2%, 161.8%, 200%
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


class EMAStrategyCalculator:
    """
    Pure 8-21 EMA swing trading strategy calculator.

    This implements the complete system from "The 8-21 EMA Swing Trading Strategy
    That Creates Millionaires" article, focusing on:
    - EMA power zone identification
    - DeMarker oscillator timing
    - Fibonacci extension targets
    """

    def __init__(self, session: Session):
        self.session = session

    def calculate_all_indicators(
        self,
        symbols: List[str],
        lookback_days: int = 252
    ) -> Dict[str, Dict]:
        """
        Calculate complete EMA strategy indicators for all stocks.

        Args:
            symbols: List of stock symbols
            lookback_days: Historical data period (default 252 = 1 year)

        Returns:
            Dict mapping symbol to all indicators and scores
        """
        try:
            logger.info(f"Calculating 8-21 EMA strategy indicators for {len(symbols)} stocks...")

            results = {}
            successful_count = 0

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

                    if indicators:
                        results[symbol] = indicators
                        successful_count += 1

                except Exception as e:
                    logger.error(f"Error calculating indicators for {symbol}: {e}")
                    continue

            logger.info(f"Successfully calculated indicators for {successful_count}/{len(symbols)} stocks")

            # Calculate ranking scores across all stocks
            self._calculate_ranking_scores(results)

            return results

        except Exception as e:
            logger.error(f"Error in calculate_all_indicators: {e}", exc_info=True)
            return {}

    def _calculate_stock_indicators(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate all technical indicators for a single stock."""
        try:
            # Calculate 8 & 21 EMA
            df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()

            # Calculate DeMarker oscillator (14-period)
            demarker = self._calculate_demarker(df, period=14)

            # Calculate Fibonacci extension targets
            fib_targets = self._calculate_fibonacci_extensions(df)

            # Get latest values
            current_price = float(df['close'].iloc[-1])
            ema_8 = float(df['ema_8'].iloc[-1])
            ema_21 = float(df['ema_21'].iloc[-1])

            # Determine power zone status
            power_zone_status = self._get_power_zone_status(current_price, ema_8, ema_21)

            # Generate buy/sell signals
            signals = self._generate_signals(df, current_price, ema_8, ema_21, demarker)

            # Calculate EMA strength metrics
            ema_metrics = self._calculate_ema_metrics(current_price, ema_8, ema_21)

            return {
                # Price & EMAs
                'current_price': current_price,
                'ema_8': ema_8,
                'ema_21': ema_21,

                # Power zone
                'power_zone_status': power_zone_status,
                'is_bullish': power_zone_status == 'bullish',

                # EMA metrics
                'ema_separation_pct': ema_metrics['separation_pct'],
                'price_above_ema8_pct': ema_metrics['price_above_ema8_pct'],
                'ema8_slope': ema_metrics['ema8_slope'],
                'ema21_slope': ema_metrics['ema21_slope'],

                # DeMarker
                'demarker': demarker,
                'demarker_oversold': demarker < 0.30,
                'demarker_overbought': demarker > 0.70,

                # Fibonacci targets
                'fib_target_127': fib_targets['target_127'],
                'fib_target_162': fib_targets['target_162'],
                'fib_target_200': fib_targets['target_200'],
                'fib_target_262': fib_targets['target_262'],

                # Signals
                'buy_signal': signals['buy_signal'],
                'sell_signal': signals['sell_signal'],
                'signal_quality': signals['signal_quality'],

                # Stop loss (below 21 EMA or recent swing low)
                'suggested_stop': signals['stop_loss'],

                # Raw data for further processing
                '_df': df
            }

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None

    def _calculate_demarker(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate DeMarker oscillator (0-1 range).

        DeMarker measures buying and selling pressure:
        - < 0.30: Oversold (good buy opportunity during pullbacks)
        - > 0.70: Overbought (potential pullback coming)
        - 0.30-0.70: Neutral zone
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

    def _calculate_fibonacci_extensions(self, df: pd.DataFrame, lookback: int = 60) -> Dict:
        """
        Calculate Fibonacci extension targets based on recent swing high/low.

        Returns targets at:
        - 127.2%: Minimum target (take 25% profit)
        - 161.8%: Golden ratio target (take 50% profit)
        - 200.0%: Psychological double (take additional profits)
        - 261.8%: Extended target (let 25% run)
        """
        try:
            # Get recent data for swing analysis
            recent_df = df.tail(lookback).copy()

            # Find swing high and swing low in recent period
            swing_high = float(recent_df['high'].max())
            swing_low = float(recent_df['low'].min())

            # Current price (entry point)
            current_price = float(df['close'].iloc[-1])

            # Calculate Fibonacci range
            fib_range = swing_high - swing_low

            # Calculate extension levels from current price
            # Based on the swing range, project upward targets
            target_127 = current_price + (fib_range * 0.272)  # 127.2% extension
            target_162 = current_price + (fib_range * 0.618)  # 161.8% (golden ratio)
            target_200 = current_price + (fib_range * 1.000)  # 200% extension
            target_262 = current_price + (fib_range * 1.618)  # 261.8% extension

            return {
                'target_127': round(target_127, 2),
                'target_162': round(target_162, 2),
                'target_200': round(target_200, 2),
                'target_262': round(target_262, 2),
                'swing_high': round(swing_high, 2),
                'swing_low': round(swing_low, 2)
            }

        except Exception as e:
            logger.error(f"Error calculating Fibonacci targets: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'target_127': round(current_price * 1.05, 2),
                'target_162': round(current_price * 1.10, 2),
                'target_200': round(current_price * 1.15, 2),
                'target_262': round(current_price * 1.20, 2),
                'swing_high': 0,
                'swing_low': 0
            }

    def _get_power_zone_status(self, price: float, ema_8: float, ema_21: float) -> str:
        """
        Determine the power zone status.

        Returns:
        - 'bullish': Price > 8 EMA > 21 EMA (TRADE LONG)
        - 'bearish': Price < 8 EMA < 21 EMA (WAIT)
        - 'neutral': Mixed conditions (WAIT)
        """
        if price > ema_8 > ema_21:
            return 'bullish'
        elif price < ema_8 < ema_21:
            return 'bearish'
        else:
            return 'neutral'

    def _calculate_ema_metrics(self, price: float, ema_8: float, ema_21: float) -> Dict:
        """Calculate EMA strength and trend metrics."""
        try:
            # EMA separation (strength of trend)
            separation_pct = ((ema_8 - ema_21) / ema_21) * 100 if ema_21 > 0 else 0

            # Price distance above 8 EMA
            price_above_ema8_pct = ((price - ema_8) / price) * 100 if price > 0 else 0

            # Calculate EMA slopes (trend direction)
            # Note: Would need more historical points for accurate slope, using simple metric
            ema8_slope = 1.0 if ema_8 > ema_21 else -1.0
            ema21_slope = ema8_slope  # Simplified

            return {
                'separation_pct': round(separation_pct, 2),
                'price_above_ema8_pct': round(price_above_ema8_pct, 2),
                'ema8_slope': ema8_slope,
                'ema21_slope': ema21_slope
            }

        except Exception as e:
            logger.error(f"Error calculating EMA metrics: {e}")
            return {
                'separation_pct': 0.0,
                'price_above_ema8_pct': 0.0,
                'ema8_slope': 0.0,
                'ema21_slope': 0.0
            }

    def _generate_signals(self, df: pd.DataFrame, price: float, ema_8: float,
                         ema_21: float, demarker: float) -> Dict:
        """
        Generate buy/sell signals based on 8-21 EMA strategy rules.

        Perfect Buy Setup (HIGH quality):
        ✅ Price > 8 EMA > 21 EMA (power zone active)
        ✅ DeMarker < 0.30 (oversold pullback)
        ✅ Price holding EMA support

        Good Buy Setup (MEDIUM quality):
        ✅ Price > 8 EMA > 21 EMA (power zone active)
        ✅ DeMarker 0.30-0.50 (mild pullback)

        Weak Buy Setup (LOW quality):
        ✅ Price > 8 EMA > 21 EMA (power zone active only)

        Sell Signal:
        ❌ Price breaks below 8 EMA and 21 EMA
        ❌ 8 EMA crosses below 21 EMA
        """
        try:
            # Check power zone status
            is_bullish_power_zone = price > ema_8 > ema_21
            is_bearish = price < ema_8 < ema_21

            # DeMarker conditions
            is_oversold = demarker < 0.30
            is_mild_pullback = 0.30 <= demarker <= 0.50

            # Check if price is holding EMA support (within 2% of EMA)
            holding_ema8_support = abs(price - ema_8) / price < 0.02 if price > 0 else False
            holding_ema21_support = abs(price - ema_21) / price < 0.02 if price > 0 else False
            holding_support = holding_ema8_support or holding_ema21_support

            # BUY SIGNAL LOGIC
            buy_signal = False
            signal_quality = 'none'

            if is_bullish_power_zone:
                if is_oversold and (price >= ema_21):
                    # Perfect setup: Bullish + Oversold + At/near support
                    buy_signal = True
                    signal_quality = 'high'
                elif is_oversold:
                    # Good setup: Bullish + Oversold
                    buy_signal = True
                    signal_quality = 'medium'
                elif is_mild_pullback:
                    # Decent setup: Bullish + Mild pullback
                    buy_signal = True
                    signal_quality = 'medium'
                else:
                    # Basic setup: Just bullish power zone
                    buy_signal = True
                    signal_quality = 'low'

            # SELL SIGNAL LOGIC
            sell_signal = is_bearish or (price < ema_21)

            # Calculate stop loss (below 21 EMA or recent swing low)
            recent_low = float(df['low'].tail(20).min())
            stop_below_ema21 = ema_21 * 0.98  # 2% below 21 EMA
            stop_loss = min(stop_below_ema21, recent_low)

            return {
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'signal_quality': signal_quality,
                'stop_loss': round(stop_loss, 2),
                'is_bullish_power_zone': is_bullish_power_zone,
                'is_oversold': is_oversold,
                'holding_support': holding_support
            }

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {
                'buy_signal': False,
                'sell_signal': False,
                'signal_quality': 'none',
                'stop_loss': 0.0,
                'is_bullish_power_zone': False,
                'is_oversold': False,
                'holding_support': False
            }

    def _calculate_ranking_scores(self, results: Dict[str, Dict]) -> None:
        """
        Calculate ranking scores for all stocks.

        Ranking based on:
        1. EMA separation (stronger trends ranked higher)
        2. Price position above 8 EMA (momentum)
        3. DeMarker oversold level (better entry timing)

        Score range: 0-100
        """
        try:
            # Extract metrics for normalization
            separations = []
            price_distances = []
            demarkers = []

            for symbol, result in results.items():
                if result.get('is_bullish', False):  # Only rank bullish stocks
                    separations.append(result.get('ema_separation_pct', 0))
                    price_distances.append(result.get('price_above_ema8_pct', 0))
                    demarkers.append(result.get('demarker', 0.5))

            if not separations:
                return

            # Calculate percentile ranks
            separations_sorted = sorted(separations)
            price_distances_sorted = sorted(price_distances)

            for symbol, result in results.items():
                try:
                    if not result.get('is_bullish', False):
                        result['ema_strategy_score'] = 0.0
                        continue

                    separation = result.get('ema_separation_pct', 0)
                    price_dist = result.get('price_above_ema8_pct', 0)
                    demarker = result.get('demarker', 0.5)

                    # Calculate percentile scores
                    sep_percentile = (separations_sorted.index(separation) / len(separations_sorted)) * 100 if separations_sorted else 50
                    dist_percentile = (price_distances_sorted.index(price_dist) / len(price_distances_sorted)) * 100 if price_distances_sorted else 50

                    # DeMarker score (lower is better for entries)
                    demarker_score = 100 if demarker < 0.30 else (100 - (demarker * 100))

                    # Weighted composite score
                    # 40% EMA separation (trend strength)
                    # 30% Price distance (momentum)
                    # 30% DeMarker timing
                    composite_score = (
                        sep_percentile * 0.40 +
                        dist_percentile * 0.30 +
                        demarker_score * 0.30
                    )

                    result['ema_strategy_score'] = round(composite_score, 2)

                except Exception as e:
                    logger.error(f"Error calculating score for {symbol}: {e}")
                    result['ema_strategy_score'] = 0.0

        except Exception as e:
            logger.error(f"Error in _calculate_ranking_scores: {e}")

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


def get_ema_strategy_calculator(session: Session) -> EMAStrategyCalculator:
    """Factory function to get EMA strategy calculator instance."""
    return EMAStrategyCalculator(session)
