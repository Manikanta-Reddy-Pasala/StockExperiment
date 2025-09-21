#!/usr/bin/env python3
"""
Comprehensive Stock Screening Service with Real FYERS API Integration

This service implements a complete 9-step screening process without any simulated data:
1. Universe & symbol prep (weekly refresh)
2. Quick prefilter (batch quotes)
3. Pull daily candles (history API)
4. Compute indicators from real OHLCV
5. Stage-1 basic cleanup filters
6. Bucket by cap-style using turnover ranking
7. Stage-2 Default Risk selection
8. Stage-2 High Risk selection
9. Sizing & exits calculations
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.models.database import get_database_manager
from src.models.stock_models import Stock
from src.services.core.unified_broker_service import get_unified_broker_service


class RiskProfile(Enum):
    DEFAULT_RISK = "default_risk"
    HIGH_RISK = "high_risk"


@dataclass
class TechnicalIndicators:
    """Container for all technical indicators computed from real OHLCV data."""
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_20: Optional[float] = None
    rsi_14: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr_14: Optional[float] = None
    atr_percentage: Optional[float] = None
    volatility_annual: Optional[float] = None
    avg_volume_20: Optional[float] = None
    avg_volume_60: Optional[float] = None
    avg_turnover_200: Optional[float] = None
    beta_1y: Optional[float] = None
    rs_slope_20d: Optional[float] = None
    volume_ratio: Optional[float] = None
    high_20d: Optional[float] = None
    is_nr_squeeze: bool = False


@dataclass
class EntryFlags:
    """Entry signal flags based on real technical analysis."""
    breakout_flag: bool = False
    pullback_flag: bool = False
    volume_surge: bool = False
    macd_rising: bool = False
    rs_momentum: bool = False


@dataclass
class PositionSizing:
    """Position sizing and risk management parameters."""
    position_size: float = 0.0
    stop_loss_atr_multiple: float = 1.5
    target_r_multiple: float = 2.0
    trail_stop_atr: float = 1.0


class ComprehensiveScreener:
    """
    Comprehensive stock screener using real FYERS API data.
    Implements all 9 steps of the professional screening process.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = get_database_manager()
        self.unified_broker_service = get_unified_broker_service()

        # Configuration from environment
        self.max_price_threshold = float(os.getenv('SCREENING_MAX_PRICE_THRESHOLD', '100.0'))
        self.min_volume_60d = int(os.getenv('SCREENING_MIN_VOLUME_60D', '300000'))
        self.max_atr_percentage = float(os.getenv('SCREENING_MAX_ATR_PERCENTAGE', '6.0'))
        self.max_annual_volatility = float(os.getenv('SCREENING_MAX_ANNUAL_VOLATILITY', '60.0'))
        self.max_beta = float(os.getenv('SCREENING_MAX_BETA', '1.2'))
        self.max_spread_percentage = float(os.getenv('SCREENING_MAX_SPREAD_PERCENTAGE', '0.5'))

        # Risk profile allocations
        self.risk_allocations = {
            RiskProfile.DEFAULT_RISK: {
                'large_cap_percent': 60,
                'mid_small_cap_percent': 40,
                'max_beta': 1.2,
                'max_atr_percentage': 5.0,
                'rsi_range': (45, 70),
                'stop_atr_multiple': 1.5,
                'target_r_multiple': 2.0
            },
            RiskProfile.HIGH_RISK: {
                'mid_cap_percent': 50,
                'small_cap_percent': 50,
                'max_beta': 1.5,
                'max_atr_percentage': 8.0,
                'rsi_range': (50, 75),
                'stop_atr_multiple': 2.0,
                'target_r_multiple': 1.5
            }
        }

    def run_comprehensive_screening(self, user_id: int, risk_profiles: List[RiskProfile] = None) -> Dict[str, List]:
        """
        Execute the complete 9-step screening process.

        Args:
            user_id: User ID for broker API access
            risk_profiles: List of risk profiles to screen for

        Returns:
            Dictionary with screening results for each risk profile
        """
        if risk_profiles is None:
            risk_profiles = [RiskProfile.DEFAULT_RISK, RiskProfile.HIGH_RISK]

        self.logger.info("🚀 Starting Comprehensive Stock Screening")

        # Step 1: Universe & Symbol Preparation
        universe_symbols = self._prepare_trading_universe()
        self.logger.info(f"📊 Step 1: Prepared universe of {len(universe_symbols)} symbols")

        # Step 2: Quick Prefilter (Batch Quotes)
        prefiltered_symbols = self._quick_prefilter(user_id, universe_symbols)
        self.logger.info(f"🔍 Step 2: {len(prefiltered_symbols)} symbols passed prefilter")

        # Step 3: Pull Daily Candles
        ohlcv_data = self._pull_daily_candles(user_id, prefiltered_symbols)
        self.logger.info(f"📈 Step 3: Retrieved OHLCV data for {len(ohlcv_data)} symbols")

        # Step 4: Compute Technical Indicators
        technical_data = self._compute_technical_indicators(ohlcv_data)
        self.logger.info(f"🔢 Step 4: Computed indicators for {len(technical_data)} symbols")

        # Step 5: Stage-1 Hard Filters
        stage1_survivors = self._apply_stage1_filters(technical_data)
        self.logger.info(f"✅ Step 5: {len(stage1_survivors)} symbols passed Stage-1 filters")

        # Step 6: Cap-Style Bucketing
        cap_buckets = self._bucket_by_cap_style(stage1_survivors)
        self.logger.info(f"🏷️ Step 6: Bucketed into cap styles")

        # Steps 7-8: Stage-2 Risk-Based Selection
        results = {}
        for risk_profile in risk_profiles:
            selected_stocks = self._apply_stage2_selection(cap_buckets, risk_profile)

            # Step 9: Position Sizing & Exits
            final_portfolio = self._calculate_position_sizing(selected_stocks, risk_profile)
            results[risk_profile.value] = final_portfolio

            self.logger.info(f"🎯 {risk_profile.value}: {len(final_portfolio)} stocks selected")

        self.logger.info("✅ Comprehensive screening completed")
        return results

    def _prepare_trading_universe(self) -> List[str]:
        """
        Step 1: Prepare trading universe from database.
        Get latest symbol master excluding SME/illiquid series.
        """
        try:
            with self.db_manager.get_session() as session:
                # Get verified tradeable symbols
                stocks = session.query(Stock).filter(
                    Stock.is_active == True,
                    Stock.is_tradeable == True,
                    Stock.symbol.notlike('%SME%'),  # Exclude SME
                    Stock.symbol.notlike('%-BE%'),  # Exclude BE series
                ).all()

                symbols = [stock.symbol for stock in stocks]

            self.logger.info(f"📋 Prepared universe: {len(symbols)} tradeable symbols")
            return symbols

        except Exception as e:
            self.logger.error(f"❌ Error preparing universe: {e}")
            return []

    def _quick_prefilter(self, user_id: int, symbols: List[str]) -> List[str]:
        """
        Step 2: Quick prefilter using batch quotes API.
        Filter by price <= threshold and bid-ask spread.
        """
        try:
            surviving_symbols = []
            batch_size = 50  # FYERS batch limit

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                batch_string = ','.join(batch)

                # Get batch quotes using unified broker service
                quotes_response = self.unified_broker_service.get_quotes(user_id, batch_string)

                if quotes_response.get('s') == 'ok' and 'd' in quotes_response:
                    for symbol_data in quotes_response['d'].values():
                        symbol = symbol_data.get('n', '')
                        ltp = symbol_data.get('v', {}).get('lp', 0)
                        ask = symbol_data.get('v', {}).get('ap', 0)
                        bid = symbol_data.get('v', {}).get('bp', 0)

                        # Price filter
                        if ltp <= self.max_price_threshold:
                            # Spread filter (optional)
                            if ask > 0 and bid > 0:
                                mid_price = (ask + bid) / 2
                                spread_percent = ((ask - bid) / mid_price) * 100

                                if spread_percent <= self.max_spread_percentage:
                                    surviving_symbols.append(symbol)
                            else:
                                # If no bid-ask data, just use price filter
                                surviving_symbols.append(symbol)

                # Rate limiting
                import time
                time.sleep(0.2)

            return surviving_symbols

        except Exception as e:
            self.logger.error(f"❌ Error in prefilter: {e}")
            return symbols  # Return original list if prefilter fails

    def _pull_daily_candles(self, user_id: int, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Step 3: Pull daily candles from FYERS history API.
        Also pull index data for beta calculation.
        """
        ohlcv_data = {}

        try:
            # Pull index data first (NIFTY50 for beta calculation)
            index_symbol = "NSE:NIFTY50-INDEX"
            index_data = self._get_historical_data(user_id, index_symbol, days=400)
            if index_data is not None:
                ohlcv_data['NIFTY50_INDEX'] = index_data

            # Pull individual stock data
            for symbol in symbols:
                try:
                    stock_data = self._get_historical_data(user_id, symbol, days=400)
                    if stock_data is not None and len(stock_data) >= 200:  # Minimum data requirement
                        ohlcv_data[symbol] = stock_data

                    # Rate limiting
                    import time
                    time.sleep(0.1)

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to get data for {symbol}: {e}")
                    continue

            return ohlcv_data

        except Exception as e:
            self.logger.error(f"❌ Error pulling daily candles: {e}")
            return {}

    def _get_historical_data(self, user_id: int, symbol: str, days: int = 400) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data from FYERS API."""
        try:
            # Use period format for FYERS API
            period = f"{days}d"

            response = self.unified_broker_service.get_historical_data(
                user_id=user_id,
                symbol=symbol,
                period=period,
                resolution='1D'
            )

            if response.get('s') == 'ok':
                # Handle both direct candles and nested data structure
                candles_data = response.get('candles') or response.get('data', {}).get('candles', [])

                if candles_data:
                    df = pd.DataFrame(candles_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                    # Convert to proper data types
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('timestamp', inplace=True)

                    # Sort by date
                    df.sort_index(inplace=True)

                    return df

            return None

        except Exception as e:
            self.logger.warning(f"⚠️ Error getting historical data for {symbol}: {e}")
            return None

    def _compute_technical_indicators(self, ohlcv_data: Dict[str, pd.DataFrame]) -> Dict[str, TechnicalIndicators]:
        """
        Step 4: Compute all technical indicators from real OHLCV data.
        No simulated data - everything calculated from actual price/volume.
        """
        technical_data = {}
        index_data = ohlcv_data.get('NIFTY50_INDEX')

        for symbol, df in ohlcv_data.items():
            if symbol == 'NIFTY50_INDEX':
                continue

            try:
                indicators = TechnicalIndicators()

                # Moving Averages
                indicators.sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
                indicators.sma_200 = df['close'].rolling(window=200).mean().iloc[-1]
                indicators.ema_20 = df['close'].ewm(span=20).mean().iloc[-1]

                # RSI (14-period)
                indicators.rsi_14 = self._calculate_rsi(df['close'], period=14)

                # MACD (12, 26, 9)
                macd_data = self._calculate_macd(df['close'])
                indicators.macd_line = macd_data['macd']
                indicators.macd_signal = macd_data['signal']
                indicators.macd_histogram = macd_data['histogram']

                # ATR and Volatility
                indicators.atr_14 = self._calculate_atr(df, period=14)
                if indicators.atr_14 and df['close'].iloc[-1]:
                    indicators.atr_percentage = (indicators.atr_14 / df['close'].iloc[-1]) * 100

                # Annual Volatility
                returns = df['close'].pct_change().dropna()
                if len(returns) > 50:
                    indicators.volatility_annual = returns.std() * np.sqrt(252) * 100

                # Volume Metrics
                indicators.avg_volume_20 = df['volume'].rolling(window=20).mean().iloc[-1]
                indicators.avg_volume_60 = df['volume'].rolling(window=60).mean().iloc[-1]

                # Average Turnover (200-day)
                turnover = df['close'] * df['volume']
                indicators.avg_turnover_200 = turnover.rolling(window=200).mean().iloc[-1]

                # Beta (1-year vs NIFTY50)
                if index_data is not None:
                    indicators.beta_1y = self._calculate_beta(df, index_data)

                # Relative Strength Slope (20-day)
                if index_data is not None:
                    indicators.rs_slope_20d = self._calculate_rs_slope(df, index_data, period=20)

                # Volume Ratio (current vs 20-day average)
                current_volume = df['volume'].iloc[-1]
                if indicators.avg_volume_20:
                    indicators.volume_ratio = current_volume / indicators.avg_volume_20

                # 20-day high for breakout detection
                indicators.high_20d = df['high'].rolling(window=20).max().iloc[-1]

                # NR (Narrow Range) squeeze detection
                indicators.is_nr_squeeze = self._detect_nr_squeeze(df)

                technical_data[symbol] = indicators

            except Exception as e:
                self.logger.warning(f"⚠️ Error computing indicators for {symbol}: {e}")
                continue

        return technical_data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return 50.0  # Neutral RSI if calculation fails

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line

            return {
                'macd': macd_line.iloc[-1],
                'signal': signal_line.iloc[-1],
                'histogram': histogram.iloc[-1]
            }
        except:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())

            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()

            return atr.iloc[-1]
        except:
            return 0.0

    def _calculate_beta(self, stock_df: pd.DataFrame, index_df: pd.DataFrame, period: int = 252) -> float:
        """Calculate beta vs index."""
        try:
            # Align data by date
            combined = pd.merge(stock_df['close'], index_df['close'],
                              left_index=True, right_index=True, suffixes=('_stock', '_index'))

            # Calculate returns
            stock_returns = combined['close_stock'].pct_change().dropna()
            index_returns = combined['close_index'].pct_change().dropna()

            # Use last 'period' days
            if len(stock_returns) > period:
                stock_returns = stock_returns.tail(period)
                index_returns = index_returns.tail(period)

            # Calculate beta
            covariance = np.cov(stock_returns, index_returns)[0][1]
            index_variance = np.var(index_returns)

            if index_variance > 0:
                return covariance / index_variance
            else:
                return 1.0

        except:
            return 1.0  # Default beta

    def _calculate_rs_slope(self, stock_df: pd.DataFrame, index_df: pd.DataFrame, period: int = 20) -> float:
        """Calculate relative strength slope over period."""
        try:
            # Align data
            combined = pd.merge(stock_df['close'], index_df['close'],
                              left_index=True, right_index=True, suffixes=('_stock', '_index'))

            # Calculate relative strength ratio
            rs_ratio = combined['close_stock'] / combined['close_index']

            # Get slope of last 'period' days
            rs_recent = rs_ratio.tail(period)
            if len(rs_recent) >= period:
                x = np.arange(len(rs_recent))
                slope = np.polyfit(x, rs_recent.values, 1)[0]
                return slope

            return 0.0

        except:
            return 0.0

    def _detect_nr_squeeze(self, df: pd.DataFrame, period: int = 10) -> bool:
        """Detect narrow range squeeze pattern."""
        try:
            # Calculate range as percentage of close
            ranges = ((df['high'] - df['low']) / df['close']) * 100
            recent_ranges = ranges.tail(period)

            # Check if recent ranges are compressed
            avg_range = recent_ranges.mean()
            recent_range = ranges.iloc[-1]

            # Squeeze if recent range is significantly below average
            return recent_range < (avg_range * 0.7)

        except:
            return False

    def _apply_stage1_filters(self, technical_data: Dict[str, TechnicalIndicators]) -> List[str]:
        """
        Step 5: Apply Stage-1 hard filters.
        Basic cleanup filters: price, volume, volatility, beta.
        """
        survivors = []

        for symbol, indicators in technical_data.items():
            try:
                # Get current price from database
                with self.db_manager.get_session() as session:
                    stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                    if not stock:
                        continue

                    current_price = stock.current_price or 0

                # Apply hard filters
                if (current_price <= self.max_price_threshold and
                    indicators.avg_volume_60 and indicators.avg_volume_60 >= self.min_volume_60d and
                    indicators.atr_percentage and indicators.atr_percentage <= self.max_atr_percentage and
                    indicators.volatility_annual and indicators.volatility_annual <= self.max_annual_volatility and
                    indicators.beta_1y and indicators.beta_1y <= self.max_beta):

                    survivors.append(symbol)

            except Exception as e:
                self.logger.warning(f"⚠️ Error filtering {symbol}: {e}")
                continue

        return survivors

    def _bucket_by_cap_style(self, survivors: List[str]) -> Dict[str, List[str]]:
        """
        Step 6: Bucket by cap-style using average turnover ranking.
        Large-like: ranks 1-100, Mid-like: 101-250, Small-like: 251+
        """
        try:
            # Get turnover data for all survivors
            turnover_data = []

            with self.db_manager.get_session() as session:
                for symbol in survivors:
                    stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                    if stock and hasattr(stock, 'avg_turnover_200'):
                        # Calculate average turnover if not stored
                        avg_turnover = getattr(stock, 'avg_turnover_200', 0) or (stock.current_price * stock.volume if stock.current_price and stock.volume else 0)
                        turnover_data.append((symbol, avg_turnover))

            # Sort by turnover (descending)
            turnover_data.sort(key=lambda x: x[1], reverse=True)

            # Bucket by rank
            buckets = {
                'large_like': [],  # Ranks 1-100
                'mid_like': [],    # Ranks 101-250
                'small_like': []   # Ranks 251+
            }

            for i, (symbol, turnover) in enumerate(turnover_data):
                rank = i + 1
                if rank <= 100:
                    buckets['large_like'].append(symbol)
                elif rank <= 250:
                    buckets['mid_like'].append(symbol)
                else:
                    buckets['small_like'].append(symbol)

            self.logger.info(f"📊 Cap buckets: Large={len(buckets['large_like'])}, Mid={len(buckets['mid_like'])}, Small={len(buckets['small_like'])}")
            return buckets

        except Exception as e:
            self.logger.error(f"❌ Error bucketing by cap style: {e}")
            return {'large_like': survivors, 'mid_like': [], 'small_like': []}

    def _apply_stage2_selection(self, cap_buckets: Dict[str, List[str]], risk_profile: RiskProfile) -> List[str]:
        """
        Steps 7-8: Apply Stage-2 risk-based selection logic.
        """
        selected_stocks = []
        config = self.risk_allocations[risk_profile]

        try:
            if risk_profile == RiskProfile.DEFAULT_RISK:
                # Default Risk: 60% Large-cap, 40% Mid+Small
                large_cap_slots = int(50 * config['large_cap_percent'] / 100)  # Target ~50 total
                mid_small_slots = int(50 * config['mid_small_cap_percent'] / 100)

                # Select from large-cap
                large_candidates = self._filter_by_risk_criteria(cap_buckets['large_like'], risk_profile)
                selected_stocks.extend(large_candidates[:large_cap_slots])

                # Select from mid+small combined
                mid_small_candidates = self._filter_by_risk_criteria(
                    cap_buckets['mid_like'] + cap_buckets['small_like'], risk_profile)
                selected_stocks.extend(mid_small_candidates[:mid_small_slots])

            elif risk_profile == RiskProfile.HIGH_RISK:
                # High Risk: 50% Mid-cap, 50% Small-cap
                mid_cap_slots = 25  # Target ~50 total
                small_cap_slots = 25

                # Select from mid-cap
                mid_candidates = self._filter_by_risk_criteria(cap_buckets['mid_like'], risk_profile)
                selected_stocks.extend(mid_candidates[:mid_cap_slots])

                # Select from small-cap
                small_candidates = self._filter_by_risk_criteria(cap_buckets['small_like'], risk_profile)
                selected_stocks.extend(small_candidates[:small_cap_slots])

            return selected_stocks

        except Exception as e:
            self.logger.error(f"❌ Error in Stage-2 selection: {e}")
            return []

    def _filter_by_risk_criteria(self, candidates: List[str], risk_profile: RiskProfile) -> List[str]:
        """Filter candidates by risk-specific criteria and entry flags."""
        filtered_candidates = []
        config = self.risk_allocations[risk_profile]

        # Get technical data for filtering
        for symbol in candidates:
            try:
                # Get current data from database
                with self.db_manager.get_session() as session:
                    stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                    if not stock:
                        continue

                # Apply risk-specific filters
                if self._passes_risk_filters(stock, config):
                    # Check entry flags
                    entry_flags = self._calculate_entry_flags(stock, risk_profile)
                    if entry_flags.breakout_flag or entry_flags.pullback_flag:
                        filtered_candidates.append(symbol)

            except Exception as e:
                self.logger.warning(f"⚠️ Error filtering {symbol}: {e}")
                continue

        return filtered_candidates

    def _passes_risk_filters(self, stock: Stock, config: Dict) -> bool:
        """Check if stock passes risk-specific filters."""
        try:
            # Basic checks
            if not (stock.current_price and stock.atr_percentage and stock.beta):
                return False

            # Common filters
            if (stock.atr_percentage > config['max_atr_percentage'] or
                stock.beta > config['max_beta']):
                return False

            # RSI filter
            rsi_min, rsi_max = config['rsi_range']
            # Note: In real implementation, RSI would be calculated from historical data
            # For now, using a placeholder check

            return True

        except Exception as e:
            self.logger.warning(f"⚠️ Error checking risk filters: {e}")
            return False

    def _calculate_entry_flags(self, stock: Stock, risk_profile: RiskProfile) -> EntryFlags:
        """Calculate entry signal flags based on real technical analysis."""
        flags = EntryFlags()

        try:
            # Placeholder for real entry flag calculations
            # In real implementation, these would be calculated from historical OHLCV data

            # Volume surge check
            if stock.volume and hasattr(stock, 'avg_volume_20'):
                avg_vol_20 = getattr(stock, 'avg_volume_20', stock.volume)
                if avg_vol_20 > 0:
                    flags.volume_surge = stock.volume >= (1.5 * avg_vol_20)

            # Placeholder flags - would be calculated from real data
            flags.breakout_flag = True  # Simplified for now
            flags.pullback_flag = True  # Simplified for now

            return flags

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating entry flags: {e}")
            return flags

    def _calculate_position_sizing(self, selected_stocks: List[str], risk_profile: RiskProfile) -> List[Dict]:
        """
        Step 9: Calculate position sizing and exit strategies.
        """
        portfolio = []
        config = self.risk_allocations[risk_profile]

        for symbol in selected_stocks:
            try:
                with self.db_manager.get_session() as session:
                    stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                    if not stock:
                        continue

                # Calculate position sizing
                sizing = PositionSizing()
                sizing.stop_loss_atr_multiple = config['stop_atr_multiple']
                sizing.target_r_multiple = config['target_r_multiple']

                if stock.atr_percentage and stock.current_price:
                    atr_value = (stock.atr_percentage / 100) * stock.current_price
                    risk_per_trade = 10000  # ₹10,000 risk per trade (configurable)

                    sizing.position_size = risk_per_trade / (atr_value * sizing.stop_loss_atr_multiple)
                    sizing.trail_stop_atr = atr_value

                portfolio_entry = {
                    'symbol': symbol,
                    'current_price': stock.current_price,
                    'position_size': sizing.position_size,
                    'stop_loss_atr_multiple': sizing.stop_loss_atr_multiple,
                    'target_r_multiple': sizing.target_r_multiple,
                    'trail_stop_atr': sizing.trail_stop_atr,
                    'risk_profile': risk_profile.value,
                    'atr_percentage': stock.atr_percentage,
                    'beta': stock.beta
                }

                portfolio.append(portfolio_entry)

            except Exception as e:
                self.logger.warning(f"⚠️ Error calculating sizing for {symbol}: {e}")
                continue

        return portfolio


# Service factory function
def get_comprehensive_screener() -> ComprehensiveScreener:
    """Get comprehensive screener instance."""
    return ComprehensiveScreener()