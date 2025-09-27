"""
Technical Indicators Service
Calculates comprehensive technical indicators using historical OHLCV data
Supports 20+ indicators for enhanced stock filtering and analysis
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, date
import time
from sqlalchemy import and_, desc, func

logger = logging.getLogger(__name__)

try:
    from ...models.database import get_database_manager
    from ...models.historical_models import HistoricalData, TechnicalIndicators
    from ...models.stock_models import Stock
except ImportError:
    from src.models.database import get_database_manager
    from src.models.historical_models import HistoricalData, TechnicalIndicators
    from src.models.stock_models import Stock


class TechnicalIndicatorsService:
    """Service to calculate and manage technical indicators from historical data."""

    def __init__(self):
        self.db_manager = get_database_manager()
        self.min_data_points = 200  # Minimum historical points for reliable indicators

    def calculate_indicators_bulk(self, symbols: List[str] = None, max_symbols: int = 100) -> Dict[str, Any]:
        """
        Calculate technical indicators for multiple symbols.

        Args:
            symbols: List of symbols to process (None = auto-select)
            max_symbols: Maximum number of symbols to process

        Returns:
            Dict with calculation results and statistics
        """
        start_time = time.time()

        try:
            if not symbols:
                symbols = self._get_symbols_needing_indicators(max_symbols)

            if not symbols:
                return {
                    'success': True,
                    'processed': 0,
                    'successful': 0,
                    'message': 'No symbols need indicator calculation'
                }

            logger.info(f"📊 Calculating indicators for {len(symbols)} symbols")

            results = {'processed': 0, 'successful': 0, 'failed': 0, 'errors': []}

            for symbol in symbols:
                try:
                    result = self.calculate_indicators_single(symbol)
                    results['processed'] += 1

                    if result.get('success'):
                        results['successful'] += 1
                        logger.info(f"✅ {symbol}: {result.get('indicators_calculated', 0)} indicators")
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"{symbol}: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    logger.error(f"Error calculating indicators for {symbol}: {e}")
                    results['processed'] += 1
                    results['failed'] += 1
                    results['errors'].append(f"{symbol}: {str(e)}")

            duration = time.time() - start_time

            return {
                'success': True,
                'processed': results['processed'],
                'successful': results['successful'],
                'failed': results['failed'],
                'duration_seconds': duration,
                'message': f"Processed {results['processed']} symbols with {results['successful']} successful calculations",
                'errors': results['errors'][:10]  # Limit error list
            }

        except Exception as e:
            logger.error(f"Error in bulk indicator calculation: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def calculate_indicators_single(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate technical indicators for a single symbol.

        Args:
            symbol: Stock symbol to process

        Returns:
            Dict with calculation results
        """
        try:
            logger.info(f"📈 Calculating indicators for {symbol}")

            # Get historical data
            historical_data = self._get_historical_data(symbol)

            if historical_data is None or len(historical_data) < self.min_data_points:
                return {
                    'success': False,
                    'symbol': symbol,
                    'error': f'Insufficient data: {len(historical_data) if historical_data is not None else 0} points'
                }

            # Calculate all indicators
            indicators = self._calculate_all_indicators(historical_data)

            # Store in database
            records_stored = self._store_indicators(symbol, indicators)

            return {
                'success': True,
                'symbol': symbol,
                'data_points': len(historical_data),
                'indicators_calculated': len(indicators),
                'records_stored': records_stored
            }

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e)
            }

    def _get_symbols_needing_indicators(self, max_symbols: int) -> List[str]:
        """Get symbols that need indicator calculation."""
        try:
            with self.db_manager.get_session() as session:
                # Get symbols with sufficient historical data but missing indicators
                subquery = session.query(HistoricalData.symbol).filter(
                    HistoricalData.date >= datetime.now().date() - timedelta(days=365)
                ).group_by(HistoricalData.symbol).having(
                    func.count(HistoricalData.id) >= self.min_data_points
                ).subquery()

                # Exclude symbols that already have recent indicators
                existing_indicators = session.query(TechnicalIndicators.symbol).filter(
                    TechnicalIndicators.date >= datetime.now().date() - timedelta(days=7)
                ).distinct().subquery()

                symbols = session.query(subquery.c.symbol).filter(
                    ~subquery.c.symbol.in_(existing_indicators)
                ).limit(max_symbols).all()

                return [symbol[0] for symbol in symbols]

        except Exception as e:
            logger.error(f"Error getting symbols needing indicators: {e}")
            return []

    def _get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol."""
        try:
            with self.db_manager.get_session() as session:
                # Get last 500 trading days (about 2 years)
                cutoff_date = datetime.now().date() - timedelta(days=700)

                data = session.query(HistoricalData).filter(
                    HistoricalData.symbol == symbol,
                    HistoricalData.date >= cutoff_date
                ).order_by(HistoricalData.date.asc()).all()

                if not data:
                    return None

                # Convert to DataFrame
                df = pd.DataFrame([{
                    'date': record.date,
                    'open': float(record.open),
                    'high': float(record.high),
                    'low': float(record.low),
                    'close': float(record.close),
                    'volume': int(record.volume)
                } for record in data])

                return df

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""

        # Create a copy to avoid modifying original data
        indicators = df.copy()

        # Moving Averages
        indicators['sma_5'] = df['close'].rolling(window=5).mean()
        indicators['sma_10'] = df['close'].rolling(window=10).mean()
        indicators['sma_20'] = df['close'].rolling(window=20).mean()
        indicators['sma_50'] = df['close'].rolling(window=50).mean()
        indicators['sma_100'] = df['close'].rolling(window=100).mean()
        indicators['sma_200'] = df['close'].rolling(window=200).mean()

        # Exponential Moving Averages
        indicators['ema_12'] = df['close'].ewm(span=12).mean()
        indicators['ema_26'] = df['close'].ewm(span=26).mean()
        indicators['ema_50'] = df['close'].ewm(span=50).mean()

        # RSI (Relative Strength Index)
        indicators['rsi_14'] = self._calculate_rsi(df['close'], 14)

        # MACD
        macd_data = self._calculate_macd(df['close'])
        indicators['macd'] = macd_data['macd']
        indicators['macd_signal'] = macd_data['signal']
        indicators['macd_histogram'] = macd_data['histogram']

        # ATR (Average True Range)
        atr_data = self._calculate_atr(df, 14)
        indicators['atr_14'] = atr_data['atr']
        indicators['atr_percentage'] = (atr_data['atr'] / df['close']) * 100

        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'], 20, 2)
        indicators['bb_upper'] = bb_data['upper']
        indicators['bb_middle'] = bb_data['middle']
        indicators['bb_lower'] = bb_data['lower']
        indicators['bb_width'] = ((bb_data['upper'] - bb_data['lower']) / bb_data['middle']) * 100

        # ADX (Average Directional Index)
        indicators['adx_14'] = self._calculate_adx(df, 14)

        # Volume Indicators
        indicators['obv'] = self._calculate_obv(df['close'], df['volume'])
        indicators['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        indicators['volume_ratio'] = df['volume'] / indicators['volume_sma_20']

        # Custom Momentum Indicators
        indicators['price_momentum_5d'] = ((df['close'] / df['close'].shift(5)) - 1) * 100
        indicators['price_momentum_20d'] = ((df['close'] / df['close'].shift(20)) - 1) * 100

        # Volatility Rank (percentile of ATR over last 252 days)
        indicators['volatility_rank'] = indicators['atr_14'].rolling(window=252).rank(pct=True) * 100

        # Calculate data points used for each indicator
        indicators['data_points_used'] = len(df)

        return indicators

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal

        return {
            'macd': macd,
            'signal': macd_signal,
            'histogram': macd_histogram
        }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = tr.rolling(window=period).mean()

        return {'atr': atr}

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)

        return {
            'upper': upper,
            'middle': sma,
            'lower': lower
        }

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        # Calculate directional movement
        dm_plus = np.maximum(df['high'].diff(), 0)
        dm_minus = np.maximum(-df['low'].diff(), 0)

        # Calculate true range
        tr = np.maximum(df['high'] - df['low'],
                       np.maximum(np.abs(df['high'] - df['close'].shift()),
                                 np.abs(df['low'] - df['close'].shift())))

        # Smooth the values
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()
        tr_smooth = tr.rolling(window=period).mean()

        # Calculate DI+ and DI-
        di_plus = (dm_plus_smooth / tr_smooth) * 100
        di_minus = (dm_minus_smooth / tr_smooth) * 100

        # Calculate DX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100

        # Calculate ADX
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_obv(self, prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = pd.Series(index=prices.index, dtype=float)
        obv.iloc[0] = volumes.iloc[0]

        for i in range(1, len(prices)):
            if prices.iloc[i] > prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volumes.iloc[i]
            elif prices.iloc[i] < prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volumes.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def _store_indicators(self, symbol: str, indicators_df: pd.DataFrame) -> int:
        """Store calculated indicators in database."""
        try:
            records_stored = 0

            with self.db_manager.get_session() as session:
                # Only store the last 90 days of indicators
                recent_df = indicators_df.tail(90)

                for _, row in recent_df.iterrows():
                    try:
                        # Check if record already exists
                        existing = session.query(TechnicalIndicators).filter(
                            TechnicalIndicators.symbol == symbol,
                            TechnicalIndicators.date == row['date']
                        ).first()

                        if existing:
                            # Update existing record
                            for column in ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
                                         'ema_12', 'ema_26', 'ema_50', 'rsi_14', 'macd', 'macd_signal',
                                         'macd_histogram', 'atr_14', 'atr_percentage', 'bb_upper', 'bb_middle',
                                         'bb_lower', 'bb_width', 'adx_14', 'obv', 'volume_sma_20', 'volume_ratio',
                                         'price_momentum_5d', 'price_momentum_20d', 'volatility_rank']:
                                if column in row and pd.notna(row[column]):
                                    setattr(existing, column, float(row[column]))

                            existing.data_points_used = int(row['data_points_used'])
                            existing.calculation_date = datetime.utcnow()
                        else:
                            # Create new record
                            indicator_record = TechnicalIndicators(
                                symbol=symbol,
                                date=row['date'],

                                # Moving Averages
                                sma_5=float(row['sma_5']) if pd.notna(row['sma_5']) else None,
                                sma_10=float(row['sma_10']) if pd.notna(row['sma_10']) else None,
                                sma_20=float(row['sma_20']) if pd.notna(row['sma_20']) else None,
                                sma_50=float(row['sma_50']) if pd.notna(row['sma_50']) else None,
                                sma_100=float(row['sma_100']) if pd.notna(row['sma_100']) else None,
                                sma_200=float(row['sma_200']) if pd.notna(row['sma_200']) else None,

                                # Exponential Moving Averages
                                ema_12=float(row['ema_12']) if pd.notna(row['ema_12']) else None,
                                ema_26=float(row['ema_26']) if pd.notna(row['ema_26']) else None,
                                ema_50=float(row['ema_50']) if pd.notna(row['ema_50']) else None,

                                # Momentum Indicators
                                rsi_14=float(row['rsi_14']) if pd.notna(row['rsi_14']) else None,
                                macd=float(row['macd']) if pd.notna(row['macd']) else None,
                                macd_signal=float(row['macd_signal']) if pd.notna(row['macd_signal']) else None,
                                macd_histogram=float(row['macd_histogram']) if pd.notna(row['macd_histogram']) else None,

                                # Volatility Indicators
                                atr_14=float(row['atr_14']) if pd.notna(row['atr_14']) else None,
                                atr_percentage=float(row['atr_percentage']) if pd.notna(row['atr_percentage']) else None,
                                bb_upper=float(row['bb_upper']) if pd.notna(row['bb_upper']) else None,
                                bb_middle=float(row['bb_middle']) if pd.notna(row['bb_middle']) else None,
                                bb_lower=float(row['bb_lower']) if pd.notna(row['bb_lower']) else None,
                                bb_width=float(row['bb_width']) if pd.notna(row['bb_width']) else None,

                                # Trend Indicators
                                adx_14=float(row['adx_14']) if pd.notna(row['adx_14']) else None,

                                # Volume Indicators
                                obv=float(row['obv']) if pd.notna(row['obv']) else None,
                                volume_sma_20=float(row['volume_sma_20']) if pd.notna(row['volume_sma_20']) else None,
                                volume_ratio=float(row['volume_ratio']) if pd.notna(row['volume_ratio']) else None,

                                # Custom Indicators
                                price_momentum_5d=float(row['price_momentum_5d']) if pd.notna(row['price_momentum_5d']) else None,
                                price_momentum_20d=float(row['price_momentum_20d']) if pd.notna(row['price_momentum_20d']) else None,
                                volatility_rank=float(row['volatility_rank']) if pd.notna(row['volatility_rank']) else None,

                                # Metadata
                                data_points_used=int(row['data_points_used']),
                                calculation_date=datetime.utcnow()
                            )
                            session.add(indicator_record)

                        records_stored += 1

                    except Exception as e:
                        logger.warning(f"Error storing indicator record for {symbol} on {row['date']}: {e}")
                        continue

                session.commit()

            return records_stored

        except Exception as e:
            logger.error(f"Error storing indicators for {symbol}: {e}")
            return 0

    def get_latest_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest calculated indicators for a symbol."""
        try:
            with self.db_manager.get_session() as session:
                latest = session.query(TechnicalIndicators).filter(
                    TechnicalIndicators.symbol == symbol
                ).order_by(TechnicalIndicators.date.desc()).first()

                if not latest:
                    return None

                return {
                    'symbol': latest.symbol,
                    'date': latest.date,
                    'sma_20': latest.sma_20,
                    'sma_50': latest.sma_50,
                    'sma_200': latest.sma_200,
                    'rsi_14': latest.rsi_14,
                    'macd': latest.macd,
                    'atr_percentage': latest.atr_percentage,
                    'bb_width': latest.bb_width,
                    'adx_14': latest.adx_14,
                    'volume_ratio': latest.volume_ratio,
                    'price_momentum_20d': latest.price_momentum_20d,
                    'volatility_rank': latest.volatility_rank,
                    'calculation_date': latest.calculation_date
                }

        except Exception as e:
            logger.error(f"Error getting latest indicators for {symbol}: {e}")
            return None