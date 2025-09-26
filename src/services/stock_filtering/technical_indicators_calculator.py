"""
Technical Indicators Calculator
Implements comprehensive technical analysis indicators for stock filtering
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """Technical indicators result."""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_squeeze: Optional[bool] = None
    sma_5: Optional[float] = None
    sma_10: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_100: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    ema_50: Optional[float] = None
    atr: Optional[float] = None
    adx: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    williams_r: Optional[float] = None
    obv: Optional[float] = None
    vpt: Optional[float] = None
    mfi: Optional[float] = None


class TechnicalIndicatorsCalculator:
    """Calculator for technical analysis indicators."""
    
    def __init__(self):
        """Initialize the calculator."""
        pass
    
    def calculate_all_indicators(self, df: pd.DataFrame, 
                               config: Dict[str, Any]) -> TechnicalIndicators:
        """
        Calculate all technical indicators for a stock.
        
        Args:
            df: DataFrame with OHLCV data
            config: Configuration for indicators
            
        Returns:
            TechnicalIndicators object with all calculated values
        """
        try:
            if df.empty or len(df) < 50:  # Need minimum data for reliable indicators
                return TechnicalIndicators()
            
            indicators = TechnicalIndicators()
            
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning("Missing required OHLCV columns")
                return indicators
            
            # Calculate RSI
            if config.get('rsi', {}).get('enabled', True):
                indicators.rsi = self._calculate_rsi(df['close'], config.get('rsi', {}))
            
            # Calculate MACD
            if config.get('macd', {}).get('enabled', True):
                macd_data = self._calculate_macd(df['close'], config.get('macd', {}))
                if macd_data:
                    indicators.macd = macd_data['macd']
                    indicators.macd_signal = macd_data['signal']
                    indicators.macd_histogram = macd_data['histogram']
            
            # Calculate Bollinger Bands
            if config.get('bollinger_bands', {}).get('enabled', True):
                bb_data = self._calculate_bollinger_bands(df['close'], config.get('bollinger_bands', {}))
                if bb_data:
                    indicators.bb_upper = bb_data['upper']
                    indicators.bb_middle = bb_data['middle']
                    indicators.bb_lower = bb_data['lower']
                    indicators.bb_squeeze = bb_data['squeeze']
            
            # Calculate Moving Averages
            if config.get('moving_averages', {}).get('enabled', True):
                ma_data = self._calculate_moving_averages(df['close'], config.get('moving_averages', {}))
                if ma_data:
                    indicators.sma_5 = ma_data.get('sma_5')
                    indicators.sma_10 = ma_data.get('sma_10')
                    indicators.sma_20 = ma_data.get('sma_20')
                    indicators.sma_50 = ma_data.get('sma_50')
                    indicators.sma_100 = ma_data.get('sma_100')
                    indicators.sma_200 = ma_data.get('sma_200')
                    indicators.ema_12 = ma_data.get('ema_12')
                    indicators.ema_26 = ma_data.get('ema_26')
                    indicators.ema_50 = ma_data.get('ema_50')
            
            # Calculate ATR
            if config.get('atr', {}).get('enabled', True):
                indicators.atr = self._calculate_atr(df, config.get('atr', {}))
            
            # Calculate ADX
            if config.get('adx', {}).get('enabled', True):
                indicators.adx = self._calculate_adx(df, config.get('adx', {}))
            
            # Calculate Stochastic
            if config.get('stochastic', {}).get('enabled', False):
                stoch_data = self._calculate_stochastic(df, config.get('stochastic', {}))
                if stoch_data:
                    indicators.stochastic_k = stoch_data['k']
                    indicators.stochastic_d = stoch_data['d']
            
            # Calculate Williams %R
            if config.get('williams_r', {}).get('enabled', False):
                indicators.williams_r = self._calculate_williams_r(df, config.get('williams_r', {}))
            
            # Calculate OBV
            if config.get('obv', {}).get('enabled', True):
                indicators.obv = self._calculate_obv(df)
            
            # Calculate VPT
            if config.get('vpt', {}).get('enabled', False):
                indicators.vpt = self._calculate_vpt(df)
            
            # Calculate MFI
            if config.get('mfi', {}).get('enabled', True):
                indicators.mfi = self._calculate_mfi(df, config.get('mfi', {}))
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return TechnicalIndicators()
    
    def _calculate_rsi(self, prices: pd.Series, config: Dict[str, Any]) -> Optional[float]:
        """Calculate RSI (Relative Strength Index)."""
        try:
            period = config.get('period', 14)
            if len(prices) < period + 1:
                return None
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
            
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            return None
    
    def _calculate_macd(self, prices: pd.Series, config: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            fast_period = config.get('fast_period', 12)
            slow_period = config.get('slow_period', 26)
            signal_period = config.get('signal_period', 9)
            
            if len(prices) < slow_period + signal_period:
                return None
            
            ema_fast = prices.ewm(span=fast_period).mean()
            ema_slow = prices.ewm(span=slow_period).mean()
            
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=signal_period).mean()
            histogram = macd - signal
            
            return {
                'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
                'signal': float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else None,
                'histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
            }
            
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
            return None
    
    def _calculate_bollinger_bands(self, prices: pd.Series, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate Bollinger Bands."""
        try:
            period = config.get('period', 20)
            std_dev = config.get('std_dev', 2.0)
            squeeze_threshold = config.get('squeeze_threshold', 0.05)
            
            if len(prices) < period:
                return None
            
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            # Calculate squeeze (when bands are close together)
            band_width = (upper - lower) / sma
            squeeze = band_width < squeeze_threshold
            
            return {
                'upper': float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None,
                'middle': float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None,
                'lower': float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None,
                'squeeze': bool(squeeze.iloc[-1]) if not pd.isna(squeeze.iloc[-1]) else False
            }
            
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
            return None
    
    def _calculate_moving_averages(self, prices: pd.Series, config: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Calculate various moving averages."""
        try:
            result = {}
            
            # SMA periods
            sma_periods = config.get('sma_periods', [5, 10, 20, 50, 100, 200])
            for period in sma_periods:
                if len(prices) >= period:
                    sma = prices.rolling(window=period).mean()
                    result[f'sma_{period}'] = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None
                else:
                    result[f'sma_{period}'] = None
            
            # EMA periods
            ema_periods = config.get('ema_periods', [12, 26, 50])
            for period in ema_periods:
                if len(prices) >= period:
                    ema = prices.ewm(span=period).mean()
                    result[f'ema_{period}'] = float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else None
                else:
                    result[f'ema_{period}'] = None
            
            return result
            
        except Exception as e:
            logger.warning(f"Error calculating moving averages: {e}")
            return {}
    
    def _calculate_atr(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[float]:
        """Calculate ATR (Average True Range)."""
        try:
            period = config.get('period', 14)
            
            if len(df) < period + 1:
                return None
            
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
            
        except Exception as e:
            logger.warning(f"Error calculating ATR: {e}")
            return None
    
    def _calculate_adx(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[float]:
        """Calculate ADX (Average Directional Index)."""
        try:
            period = config.get('period', 14)
            
            if len(df) < period * 2:
                return None
            
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate DM+ and DM-
            dm_plus = high.diff()
            dm_minus = -low.diff()
            
            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate smoothed values
            dm_plus_smooth = dm_plus.rolling(window=period).mean()
            dm_minus_smooth = dm_minus.rolling(window=period).mean()
            tr_smooth = tr.rolling(window=period).mean()
            
            # Calculate DI+ and DI-
            di_plus = 100 * (dm_plus_smooth / tr_smooth)
            di_minus = 100 * (dm_minus_smooth / tr_smooth)
            
            # Calculate DX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            
            # Calculate ADX
            adx = dx.rolling(window=period).mean()
            
            return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None
            
        except Exception as e:
            logger.warning(f"Error calculating ADX: {e}")
            return None
    
    def _calculate_stochastic(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Calculate Stochastic Oscillator."""
        try:
            k_period = config.get('k_period', 14)
            d_period = config.get('d_period', 3)
            
            if len(df) < k_period + d_period:
                return None
            
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate %K
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            
            # Calculate %D (smoothed %K)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                'k': float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else None,
                'd': float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else None
            }
            
        except Exception as e:
            logger.warning(f"Error calculating Stochastic: {e}")
            return None
    
    def _calculate_williams_r(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[float]:
        """Calculate Williams %R."""
        try:
            period = config.get('period', 14)
            
            if len(df) < period:
                return None
            
            high = df['high']
            low = df['low']
            close = df['close']
            
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            
            return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else None
            
        except Exception as e:
            logger.warning(f"Error calculating Williams %R: {e}")
            return None
    
    def _calculate_obv(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate OBV (On-Balance Volume)."""
        try:
            if len(df) < 2:
                return None
            
            close = df['close']
            volume = df['volume']
            
            # Calculate price change direction
            price_change = close.diff()
            
            # Calculate OBV
            obv = np.where(price_change > 0, volume,
                          np.where(price_change < 0, -volume, 0)).cumsum()
            
            return float(obv[-1]) if not pd.isna(obv[-1]) else None
            
        except Exception as e:
            logger.warning(f"Error calculating OBV: {e}")
            return None
    
    def _calculate_vpt(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate VPT (Volume Price Trend)."""
        try:
            if len(df) < 2:
                return None
            
            close = df['close']
            volume = df['volume']
            
            # Calculate price change percentage
            price_change_pct = close.pct_change()
            
            # Calculate VPT
            vpt = (volume * price_change_pct).cumsum()
            
            return float(vpt.iloc[-1]) if not pd.isna(vpt.iloc[-1]) else None
            
        except Exception as e:
            logger.warning(f"Error calculating VPT: {e}")
            return None
    
    def _calculate_mfi(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[float]:
        """Calculate MFI (Money Flow Index)."""
        try:
            period = config.get('period', 14)
            
            if len(df) < period + 1:
                return None
            
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate money flow
            money_flow = typical_price * volume
            
            # Calculate positive and negative money flow
            price_change = typical_price.diff()
            
            positive_mf = money_flow.where(price_change > 0, 0)
            negative_mf = money_flow.where(price_change < 0, 0)
            
            # Calculate smoothed values
            positive_mf_smooth = positive_mf.rolling(window=period).sum()
            negative_mf_smooth = negative_mf.rolling(window=period).sum()
            
            # Calculate MFI
            mfi = 100 - (100 / (1 + (positive_mf_smooth / negative_mf_smooth)))
            
            return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else None
            
        except Exception as e:
            logger.warning(f"Error calculating MFI: {e}")
            return None


# Global calculator instance
_technical_calculator = None


def get_technical_calculator() -> TechnicalIndicatorsCalculator:
    """Get the global technical indicators calculator instance."""
    global _technical_calculator
    if _technical_calculator is None:
        _technical_calculator = TechnicalIndicatorsCalculator()
    return _technical_calculator
