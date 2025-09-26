"""
Stage 2 Advanced Filtering Service

This service implements comprehensive Stage 2 filtering using technical indicators,
fundamental ratios, and risk metrics based on the configuration in stock_filters.yaml
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .config_loader import get_stage2_config
from .data_service import get_stock_data, create_features
from .stock_discovery_service import StockInfo, MarketCap

logger = logging.getLogger(__name__)


@dataclass
class Stage2Score:
    """Detailed Stage 2 analysis score for a stock."""
    symbol: str
    technical_score: float
    fundamental_score: float
    risk_score: float
    momentum_score: float
    volume_score: float
    total_score: float
    passed: bool
    technical_signals: Dict[str, any]
    risk_metrics: Dict[str, float]
    recommendation: str


class Stage2FilterService:
    """Advanced Stage 2 filtering service with real technical analysis."""

    def __init__(self):
        self.stage2_config = get_stage2_config()
        if not self.stage2_config:
            logger.warning("Stage 2 configuration not found. Service will operate in minimal mode.")

    def analyze_stock(self, stock: StockInfo, user_id: int = 1,
                     lookback_days: int = 100) -> Optional[Stage2Score]:
        """
        Perform comprehensive Stage 2 analysis on a single stock.

        Args:
            stock: StockInfo object from Stage 1 filtering
            user_id: User ID for data fetching
            lookback_days: Number of days of historical data to analyze

        Returns:
            Stage2Score object with detailed analysis results
        """
        if not self.stage2_config:
            return None

        try:
            # Fetch historical data for technical analysis
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)

            # Get stock data
            df = get_stock_data(
                symbol=stock.symbol,
                start_date=start_date,
                end_date=end_date,
                user_id=user_id
            )

            if df.empty or len(df) < 30:  # Need minimum data for indicators
                logger.warning(f"Insufficient data for {stock.symbol}")
                return None

            # Calculate all indicators and scores
            technical_score, technical_signals = self._calculate_technical_score(df, stock)
            volume_score = self._calculate_volume_score(df, stock)
            momentum_score = self._calculate_momentum_score(df, stock)
            risk_score, risk_metrics = self._calculate_risk_score(df, stock)
            fundamental_score = self._calculate_fundamental_score(stock)

            # Calculate weighted total score
            weights = self.stage2_config.scoring_weights
            total_score = (
                technical_score * weights.technical_score +
                fundamental_score * weights.fundamental_score +
                risk_score * weights.risk_score +
                momentum_score * weights.momentum_score +
                volume_score * weights.volume_score
            )

            # Check if passes thresholds
            thresholds = self.stage2_config.filtering_thresholds
            passed = total_score >= thresholds.minimum_total_score

            if thresholds.require_all_categories:
                passed = passed and (
                    technical_score >= thresholds.minimum_technical_score and
                    fundamental_score >= thresholds.minimum_fundamental_score
                )

            # Generate recommendation
            recommendation = self._generate_recommendation(
                total_score, technical_signals, risk_metrics
            )

            return Stage2Score(
                symbol=stock.symbol,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                risk_score=risk_score,
                momentum_score=momentum_score,
                volume_score=volume_score,
                total_score=total_score,
                passed=passed,
                technical_signals=technical_signals,
                risk_metrics=risk_metrics,
                recommendation=recommendation
            )

        except Exception as e:
            logger.error(f"Error analyzing {stock.symbol}: {e}")
            return None

    def _calculate_technical_score(self, df: pd.DataFrame, stock: StockInfo) -> Tuple[float, Dict]:
        """Calculate technical indicator score with actual indicators."""
        if not self.stage2_config.technical_indicators:
            return 50.0, {}

        config = self.stage2_config.technical_indicators
        signals = {}
        scores = []

        # RSI Analysis
        if config.rsi and config.rsi.get('enabled', True):
            rsi_config = config.rsi
            rsi = self._calculate_rsi(df['Close'], rsi_config.get('period', 14))
            current_rsi = rsi.iloc[-1]

            signals['rsi'] = {
                'value': current_rsi,
                'signal': 'oversold' if current_rsi < rsi_config.get('oversold_threshold', 30)
                         else 'overbought' if current_rsi > rsi_config.get('overbought_threshold', 70)
                         else 'neutral'
            }

            # Score based on RSI
            if current_rsi < rsi_config.get('oversold_threshold', 30):
                scores.append(80)  # Potential buy opportunity
            elif current_rsi > rsi_config.get('overbought_threshold', 70):
                scores.append(20)  # Potential sell signal
            else:
                scores.append(50 + (50 - current_rsi) * 0.5)  # Neutral zone scoring

        # MACD Analysis
        if config.macd and config.macd.get('enabled', True):
            macd_config = config.macd
            macd, signal, histogram = self._calculate_macd(
                df['Close'],
                macd_config.get('fast_period', 12),
                macd_config.get('slow_period', 26),
                macd_config.get('signal_period', 9)
            )

            current_histogram = histogram.iloc[-1]
            prev_histogram = histogram.iloc[-2]

            signals['macd'] = {
                'macd': macd.iloc[-1],
                'signal': signal.iloc[-1],
                'histogram': current_histogram,
                'crossover': 'bullish' if current_histogram > 0 and prev_histogram <= 0
                           else 'bearish' if current_histogram < 0 and prev_histogram >= 0
                           else 'none'
            }

            # Score based on MACD
            if current_histogram > 0 and current_histogram > prev_histogram:
                scores.append(70)  # Bullish momentum
            elif current_histogram < 0 and current_histogram < prev_histogram:
                scores.append(30)  # Bearish momentum
            else:
                scores.append(50)

        # Bollinger Bands Analysis
        if config.bollinger_bands and config.bollinger_bands.get('enabled', True):
            bb_config = config.bollinger_bands
            upper, middle, lower = self._calculate_bollinger_bands(
                df['Close'],
                bb_config.get('period', 20),
                bb_config.get('std_dev', 2)
            )

            current_price = df['Close'].iloc[-1]
            bb_position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])

            signals['bollinger_bands'] = {
                'upper': upper.iloc[-1],
                'middle': middle.iloc[-1],
                'lower': lower.iloc[-1],
                'position': bb_position,
                'signal': 'oversold' if bb_position < 0.2
                        else 'overbought' if bb_position > 0.8
                        else 'neutral'
            }

            # Score based on Bollinger Bands
            if bb_position < 0.2:
                scores.append(75)  # Near lower band - potential bounce
            elif bb_position > 0.8:
                scores.append(25)  # Near upper band - potential resistance
            else:
                scores.append(50)

        # Moving Averages Analysis
        if config.moving_averages and config.moving_averages.get('enabled', True):
            ma_config = config.moving_averages
            ma_signals = {}

            # Calculate SMAs
            for period in ma_config.get('sma_periods', [20, 50, 200]):
                if len(df) >= period:
                    sma = df['Close'].rolling(window=period).mean()
                    ma_signals[f'sma_{period}'] = sma.iloc[-1]

            current_price = df['Close'].iloc[-1]

            # Check golden/death cross
            if 'sma_50' in ma_signals and 'sma_200' in ma_signals:
                if ma_signals['sma_50'] > ma_signals['sma_200']:
                    ma_signals['cross'] = 'golden'
                    scores.append(70)
                else:
                    ma_signals['cross'] = 'death'
                    scores.append(30)

            # Price above/below MA200
            if 'sma_200' in ma_signals:
                if current_price > ma_signals['sma_200']:
                    scores.append(60)  # Bullish long-term trend
                else:
                    scores.append(40)  # Bearish long-term trend

            signals['moving_averages'] = ma_signals

        # Calculate final technical score
        technical_score = np.mean(scores) if scores else 50.0
        return technical_score, signals

    def _calculate_volume_score(self, df: pd.DataFrame, stock: StockInfo) -> float:
        """Calculate volume analysis score."""
        if not self.stage2_config.volume_analysis:
            return 50.0

        config = self.stage2_config.volume_analysis
        scores = []

        # Volume surge analysis
        if config.volume_surge and config.volume_surge.get('enabled', True):
            surge_config = config.volume_surge
            lookback = surge_config.get('lookback_period', 20)
            avg_volume = df['Volume'].rolling(window=lookback).mean()
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume.iloc[-1]

            if volume_ratio > surge_config.get('surge_multiplier', 2.0):
                scores.append(80)  # High volume indicates interest
            elif volume_ratio > 1.0:
                scores.append(60)
            else:
                scores.append(40)

        # Money Flow Index
        if config.mfi and config.mfi.get('enabled', True):
            mfi_config = config.mfi
            mfi = self._calculate_mfi(df, mfi_config.get('period', 14))
            current_mfi = mfi.iloc[-1]

            if current_mfi < mfi_config.get('oversold_threshold', 20):
                scores.append(75)  # Oversold
            elif current_mfi > mfi_config.get('overbought_threshold', 80):
                scores.append(25)  # Overbought
            else:
                scores.append(50)

        # On-Balance Volume trend
        if config.obv and config.obv.get('enabled', True):
            obv = self._calculate_obv(df)
            obv_trend = self._calculate_trend(obv, config.obv.get('trend_period', 10))
            scores.append(50 + obv_trend * 30)  # Trend ranges from -1 to 1

        return np.mean(scores) if scores else 50.0

    def _calculate_momentum_score(self, df: pd.DataFrame, stock: StockInfo) -> float:
        """Calculate momentum indicators score."""
        if not self.stage2_config.momentum:
            return 50.0

        config = self.stage2_config.momentum
        scores = []

        # Price momentum
        if config.price_momentum and config.price_momentum.get('enabled', True):
            pm_config = config.price_momentum
            for period in pm_config.get('periods', [5, 10, 20]):
                if len(df) >= period:
                    momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-period] - 1) * 100
                    if momentum > 0:
                        scores.append(min(50 + momentum * 2, 80))
                    else:
                        scores.append(max(50 + momentum * 2, 20))

        # Rate of Change
        if config.roc and config.roc.get('enabled', True):
            roc_config = config.roc
            roc = self._calculate_roc(df['Close'], roc_config.get('period', 10))
            current_roc = roc.iloc[-1]

            if roc_config.get('minimum', -20) <= current_roc <= roc_config.get('maximum', 30):
                scores.append(50 + current_roc)  # Within acceptable range
            else:
                scores.append(30)  # Outside acceptable range

        return np.mean(scores) if scores else 50.0

    def _calculate_risk_score(self, df: pd.DataFrame, stock: StockInfo) -> Tuple[float, Dict]:
        """Calculate risk metrics score."""
        if not self.stage2_config.risk_metrics:
            return 50.0, {}

        config = self.stage2_config.risk_metrics
        risk_metrics = {}
        scores = []

        # Volatility
        if config.volatility and config.volatility.get('enabled', True):
            vol_config = config.volatility
            returns = df['Close'].pct_change()
            daily_vol = returns.std() * 100
            annual_vol = daily_vol * np.sqrt(252)

            risk_metrics['daily_volatility'] = daily_vol
            risk_metrics['annual_volatility'] = annual_vol

            if daily_vol <= vol_config.get('maximum_daily', 5.0):
                scores.append(80 - daily_vol * 10)  # Lower volatility = higher score
            else:
                scores.append(20)

        # Sharpe Ratio (simplified)
        if config.sharpe_ratio and config.sharpe_ratio.get('enabled', True):
            sharpe_config = config.sharpe_ratio
            returns = df['Close'].pct_change()
            avg_return = returns.mean() * 252  # Annualized
            risk_free_rate = sharpe_config.get('risk_free_rate', 6.5) / 100
            std_return = returns.std() * np.sqrt(252)

            if std_return > 0:
                sharpe = (avg_return - risk_free_rate) / std_return
                risk_metrics['sharpe_ratio'] = sharpe

                if sharpe >= sharpe_config.get('minimum', 0.5):
                    scores.append(min(50 + sharpe * 20, 90))
                else:
                    scores.append(30)

        # Maximum Drawdown
        if config.max_drawdown and config.max_drawdown.get('enabled', True):
            dd_config = config.max_drawdown
            period = min(dd_config.get('period', 252), len(df))
            recent_data = df['Close'].iloc[-period:]
            cummax = recent_data.expanding().max()
            drawdown = ((recent_data - cummax) / cummax * 100).min()

            risk_metrics['max_drawdown'] = abs(drawdown)

            if abs(drawdown) <= dd_config.get('maximum', 30.0):
                scores.append(80 - abs(drawdown) * 1.5)
            else:
                scores.append(20)

        # Adjust for market cap (lower risk for larger caps)
        if stock.market_cap_category == MarketCap.LARGE_CAP:
            scores = [s * 1.2 for s in scores]  # 20% bonus
        elif stock.market_cap_category == MarketCap.SMALL_CAP:
            scores = [s * 0.8 for s in scores]  # 20% penalty

        risk_score = min(np.mean(scores) if scores else 50.0, 100)
        return risk_score, risk_metrics

    def _calculate_fundamental_score(self, stock: StockInfo) -> float:
        """
        Calculate fundamental score (simplified - would need actual fundamental data).
        In production, this would fetch real fundamental data from broker or external API.
        """
        # This is a placeholder implementation
        base_score = 50

        # Market cap consideration
        if stock.market_cap_category == MarketCap.LARGE_CAP:
            base_score += 15
        elif stock.market_cap_category == MarketCap.MID_CAP:
            base_score += 5

        # Sector consideration
        strong_sectors = ['Banking', 'Technology', 'Pharmaceutical', 'FMCG']
        if stock.sector in strong_sectors:
            base_score += 10

        # Liquidity consideration
        if stock.liquidity_score > 0.7:
            base_score += 10

        return min(base_score, 85)

    def _generate_recommendation(self, total_score: float,
                                technical_signals: Dict,
                                risk_metrics: Dict) -> str:
        """Generate trading recommendation based on analysis."""
        if total_score >= 75:
            recommendation = "STRONG BUY"
        elif total_score >= 65:
            recommendation = "BUY"
        elif total_score >= 55:
            recommendation = "HOLD"
        elif total_score >= 45:
            recommendation = "WEAK HOLD"
        else:
            recommendation = "AVOID"

        # Adjust based on specific signals
        if technical_signals.get('rsi', {}).get('signal') == 'oversold':
            if recommendation in ["HOLD", "WEAK HOLD"]:
                recommendation = "BUY"
        elif technical_signals.get('rsi', {}).get('signal') == 'overbought':
            if recommendation in ["BUY", "STRONG BUY"]:
                recommendation = "HOLD"

        # Risk adjustment
        if risk_metrics.get('max_drawdown', 0) > 25:
            if recommendation == "STRONG BUY":
                recommendation = "BUY"
            elif recommendation == "BUY":
                recommendation = "HOLD"

        return recommendation

    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12,
                       slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20,
                                  std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)

        # Determine positive and negative money flow
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return mfi

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(0.0, index=df.index)
        obv.iloc[0] = df['Volume'].iloc[0]

        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def _calculate_roc(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Rate of Change."""
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100

    def _calculate_trend(self, series: pd.Series, period: int = 10) -> float:
        """Calculate trend direction (-1 to 1)."""
        if len(series) < period:
            return 0

        recent = series.iloc[-period:]
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent.values, 1)

        # Normalize slope to -1 to 1 range
        normalized_slope = np.tanh(slope / recent.std() if recent.std() > 0 else 0)
        return normalized_slope


# Global service instance
_stage2_filter_service = None


def get_stage2_filter_service() -> Stage2FilterService:
    """Get the global Stage 2 filter service instance."""
    global _stage2_filter_service
    if _stage2_filter_service is None:
        _stage2_filter_service = Stage2FilterService()
    return _stage2_filter_service