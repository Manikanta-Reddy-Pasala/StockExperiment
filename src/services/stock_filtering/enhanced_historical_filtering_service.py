"""
Enhanced Historical Filtering Service
Integrates historical data and technical indicators for superior stock filtering
Provides comprehensive analysis using 1+ years of market data
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, date

logger = logging.getLogger(__name__)

try:
    from ...models.database import get_database_manager
    from ...models.historical_models import HistoricalData, TechnicalIndicators
    from ...models.stock_models import Stock
    from ..data.technical_indicators_service import TechnicalIndicatorsService
except ImportError:
    from src.models.database import get_database_manager
    from src.models.historical_models import HistoricalData, TechnicalIndicators
    from src.models.stock_models import Stock
    from src.services.data.technical_indicators_service import TechnicalIndicatorsService


class EnhancedHistoricalFilteringService:
    """Advanced stock filtering using historical data and technical indicators."""

    def __init__(self):
        self.db_manager = get_database_manager()
        self.indicators_service = TechnicalIndicatorsService()

        # Enhanced filtering thresholds based on historical analysis
        self.filtering_criteria = {
            # Trend Analysis (Moving Averages)
            'bullish_trend': {
                'price_above_sma_20': True,  # Current price above 20-day SMA
                'sma_20_above_sma_50': True,  # 20-day SMA above 50-day SMA
                'sma_50_above_sma_200': True,  # 50-day SMA above 200-day SMA (golden cross territory)
            },

            # Momentum Indicators
            'momentum_strength': {
                'rsi_range': (30, 70),  # RSI not overbought/oversold
                'macd_bullish': True,   # MACD above signal line
                'price_momentum_20d_min': -5,  # Max 5% decline in 20 days
                'price_momentum_5d_min': -2,   # Max 2% decline in 5 days
            },

            # Volatility and Risk Management
            'volatility_control': {
                'atr_percentage_max': 5.0,     # Max 5% daily volatility
                'volatility_rank_max': 80,     # Not in top 20% most volatile
                'bb_width_max': 15,            # Bollinger band width under 15%
            },

            # Volume and Liquidity
            'liquidity_requirements': {
                'volume_ratio_min': 0.5,       # At least 50% of average volume
                'volume_ratio_max': 3.0,       # Not more than 300% (unusual activity)
                'avg_volume_min': 10000,       # Minimum average daily volume
            },

            # Technical Strength
            'technical_strength': {
                'adx_min': 20,                 # Trending market (ADX > 20)
                'price_near_bb_upper': False,  # Not near upper Bollinger Band (avoid overbought)
                'rsi_trending_up': True,       # RSI showing upward momentum
            }
        }

    def filter_stocks_with_historical_analysis(self, symbols: List[str] = None,
                                             max_results: int = 50) -> Dict[str, Any]:
        """
        Filter stocks using comprehensive historical analysis.

        Args:
            symbols: List of symbols to analyze (None = all available)
            max_results: Maximum number of stocks to return

        Returns:
            Dict with filtered stocks and analysis details
        """
        start_time = datetime.now()

        try:
            if not symbols:
                symbols = self._get_symbols_with_indicators()

            if not symbols:
                return {
                    'success': True,
                    'filtered_stocks': [],
                    'total_analyzed': 0,
                    'message': 'No symbols with sufficient historical data found'
                }

            logger.info(f"🔍 Analyzing {len(symbols)} symbols with historical data")

            filtered_stocks = []
            analysis_results = {
                'total_analyzed': len(symbols),
                'passed_trend_analysis': 0,
                'passed_momentum_check': 0,
                'passed_volatility_check': 0,
                'passed_liquidity_check': 0,
                'passed_technical_strength': 0,
                'final_selections': 0
            }

            for symbol in symbols:
                try:
                    stock_analysis = self._analyze_single_stock(symbol)

                    if stock_analysis and stock_analysis.get('overall_score', 0) > 0:
                        filtered_stocks.append(stock_analysis)

                        # Update statistics
                        if stock_analysis.get('trend_score', 0) > 0.7:
                            analysis_results['passed_trend_analysis'] += 1
                        if stock_analysis.get('momentum_score', 0) > 0.6:
                            analysis_results['passed_momentum_check'] += 1
                        if stock_analysis.get('volatility_score', 0) > 0.6:
                            analysis_results['passed_volatility_check'] += 1
                        if stock_analysis.get('liquidity_score', 0) > 0.6:
                            analysis_results['passed_liquidity_check'] += 1
                        if stock_analysis.get('technical_score', 0) > 0.6:
                            analysis_results['passed_technical_strength'] += 1

                except Exception as e:
                    logger.warning(f"Error analyzing {symbol}: {e}")
                    continue

            # Sort by overall score and limit results
            filtered_stocks.sort(key=lambda x: x['overall_score'], reverse=True)
            final_stocks = filtered_stocks[:max_results]
            analysis_results['final_selections'] = len(final_stocks)

            duration = (datetime.now() - start_time).total_seconds()

            return {
                'success': True,
                'filtered_stocks': final_stocks,
                'analysis_statistics': analysis_results,
                'duration_seconds': duration,
                'message': f"Selected {len(final_stocks)} stocks from {len(symbols)} analyzed using historical data"
            }

        except Exception as e:
            logger.error(f"Error in enhanced historical filtering: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _get_symbols_with_indicators(self) -> List[str]:
        """Get symbols that have calculated technical indicators."""
        try:
            with self.db_manager.get_session() as session:
                # Get symbols with recent technical indicators
                cutoff_date = datetime.now().date() - timedelta(days=30)

                symbols = session.query(TechnicalIndicators.symbol).filter(
                    TechnicalIndicators.date >= cutoff_date
                ).distinct().all()

                return [symbol[0] for symbol in symbols]

        except Exception as e:
            logger.error(f"Error getting symbols with indicators: {e}")
            return []

    def _analyze_single_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Perform comprehensive analysis on a single stock."""
        try:
            # Get latest technical indicators
            indicators = self.indicators_service.get_latest_indicators(symbol)
            if not indicators:
                return None

            # Get current stock data
            stock_data = self._get_current_stock_data(symbol)
            if not stock_data:
                return None

            # Perform multi-dimensional analysis
            trend_analysis = self._analyze_trend(indicators, stock_data)
            momentum_analysis = self._analyze_momentum(indicators)
            volatility_analysis = self._analyze_volatility(indicators)
            liquidity_analysis = self._analyze_liquidity(indicators, stock_data)
            technical_analysis = self._analyze_technical_strength(indicators)

            # Calculate overall score
            overall_score = self._calculate_overall_score(
                trend_analysis, momentum_analysis, volatility_analysis,
                liquidity_analysis, technical_analysis
            )

            return {
                'symbol': symbol,
                'current_price': stock_data.get('current_price'),
                'market_cap_category': stock_data.get('market_cap_category'),

                # Individual scores
                'trend_score': trend_analysis['score'],
                'momentum_score': momentum_analysis['score'],
                'volatility_score': volatility_analysis['score'],
                'liquidity_score': liquidity_analysis['score'],
                'technical_score': technical_analysis['score'],
                'overall_score': overall_score,

                # Detailed analysis
                'trend_signals': trend_analysis['signals'],
                'momentum_signals': momentum_analysis['signals'],
                'risk_signals': volatility_analysis['signals'],
                'liquidity_signals': liquidity_analysis['signals'],
                'technical_signals': technical_analysis['signals'],

                # Key indicators for decision making
                'key_indicators': {
                    'rsi_14': indicators.get('rsi_14'),
                    'sma_20': indicators.get('sma_20'),
                    'sma_50': indicators.get('sma_50'),
                    'macd': indicators.get('macd'),
                    'atr_percentage': indicators.get('atr_percentage'),
                    'volume_ratio': indicators.get('volume_ratio'),
                    'price_momentum_20d': indicators.get('price_momentum_20d')
                },

                'analysis_date': indicators.get('date'),
                'recommendation': self._generate_recommendation(overall_score)
            }

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def _analyze_trend(self, indicators: Dict, stock_data: Dict) -> Dict[str, Any]:
        """Analyze trend strength using moving averages."""
        signals = []
        score = 0.0
        max_score = 4.0

        current_price = stock_data.get('current_price', 0)
        sma_20 = indicators.get('sma_20')
        sma_50 = indicators.get('sma_50')
        sma_200 = indicators.get('sma_200')

        # Price above SMA 20 (25% weight)
        if sma_20 and current_price > sma_20:
            score += 1.0
            signals.append("Price above 20-day moving average (bullish)")
        elif sma_20:
            signals.append("Price below 20-day moving average (bearish)")

        # SMA 20 above SMA 50 (25% weight)
        if sma_20 and sma_50 and sma_20 > sma_50:
            score += 1.0
            signals.append("Short-term trend stronger than medium-term (bullish)")
        elif sma_20 and sma_50:
            signals.append("Short-term trend weaker than medium-term (bearish)")

        # SMA 50 above SMA 200 (25% weight)
        if sma_50 and sma_200 and sma_50 > sma_200:
            score += 1.0
            signals.append("Medium-term trend above long-term (golden cross territory)")
        elif sma_50 and sma_200:
            signals.append("Medium-term trend below long-term (death cross territory)")

        # Price momentum (25% weight)
        momentum_20d = indicators.get('price_momentum_20d', 0)
        if momentum_20d > 5:
            score += 1.0
            signals.append(f"Strong 20-day momentum: +{momentum_20d:.1f}%")
        elif momentum_20d > 0:
            score += 0.5
            signals.append(f"Positive 20-day momentum: +{momentum_20d:.1f}%")
        else:
            signals.append(f"Negative 20-day momentum: {momentum_20d:.1f}%")

        return {
            'score': score / max_score,
            'signals': signals
        }

    def _analyze_momentum(self, indicators: Dict) -> Dict[str, Any]:
        """Analyze momentum indicators."""
        signals = []
        score = 0.0
        max_score = 3.0

        rsi_14 = indicators.get('rsi_14')
        macd = indicators.get('macd')
        momentum_5d = indicators.get('price_momentum_5d', 0)

        # RSI analysis (40% weight)
        if rsi_14:
            if 40 <= rsi_14 <= 60:
                score += 1.2
                signals.append(f"RSI in healthy range: {rsi_14:.1f} (not overbought/oversold)")
            elif 30 <= rsi_14 <= 70:
                score += 0.8
                signals.append(f"RSI acceptable: {rsi_14:.1f}")
            else:
                signals.append(f"RSI extreme: {rsi_14:.1f} ({'overbought' if rsi_14 > 70 else 'oversold'})")

        # MACD analysis (30% weight)
        if macd:
            if macd > 0:
                score += 0.9
                signals.append(f"MACD bullish: {macd:.4f}")
            else:
                signals.append(f"MACD bearish: {macd:.4f}")

        # Short-term momentum (30% weight)
        if momentum_5d > 1:
            score += 0.9
            signals.append(f"Strong 5-day momentum: +{momentum_5d:.1f}%")
        elif momentum_5d > -1:
            score += 0.5
            signals.append(f"Stable 5-day performance: {momentum_5d:.1f}%")
        else:
            signals.append(f"Weak 5-day momentum: {momentum_5d:.1f}%")

        return {
            'score': score / max_score,
            'signals': signals
        }

    def _analyze_volatility(self, indicators: Dict) -> Dict[str, Any]:
        """Analyze volatility and risk indicators."""
        signals = []
        score = 0.0
        max_score = 3.0

        atr_percentage = indicators.get('atr_percentage')
        bb_width = indicators.get('bb_width')
        volatility_rank = indicators.get('volatility_rank')

        # ATR percentage (40% weight)
        if atr_percentage:
            if atr_percentage < 2:
                score += 1.2
                signals.append(f"Low volatility: {atr_percentage:.1f}% ATR (stable)")
            elif atr_percentage < 4:
                score += 0.8
                signals.append(f"Moderate volatility: {atr_percentage:.1f}% ATR")
            else:
                signals.append(f"High volatility: {atr_percentage:.1f}% ATR (risky)")

        # Bollinger Band width (30% weight)
        if bb_width:
            if bb_width < 8:
                score += 0.9
                signals.append(f"Tight Bollinger Bands: {bb_width:.1f}% (low volatility)")
            elif bb_width < 12:
                score += 0.6
                signals.append(f"Normal Bollinger Bands: {bb_width:.1f}%")
            else:
                signals.append(f"Wide Bollinger Bands: {bb_width:.1f}% (high volatility)")

        # Volatility rank (30% weight)
        if volatility_rank:
            if volatility_rank < 50:
                score += 0.9
                signals.append(f"Below average volatility: {volatility_rank:.0f}th percentile")
            elif volatility_rank < 75:
                score += 0.5
                signals.append(f"Above average volatility: {volatility_rank:.0f}th percentile")
            else:
                signals.append(f"Very high volatility: {volatility_rank:.0f}th percentile")

        return {
            'score': score / max_score,
            'signals': signals
        }

    def _analyze_liquidity(self, indicators: Dict, stock_data: Dict) -> Dict[str, Any]:
        """Analyze liquidity and volume indicators."""
        signals = []
        score = 0.0
        max_score = 2.0

        volume_ratio = indicators.get('volume_ratio')
        avg_volume = stock_data.get('volume', 0)

        # Volume ratio analysis (60% weight)
        if volume_ratio:
            if 0.8 <= volume_ratio <= 1.5:
                score += 1.2
                signals.append(f"Normal volume activity: {volume_ratio:.1f}x average")
            elif 0.5 <= volume_ratio <= 2.0:
                score += 0.8
                signals.append(f"Acceptable volume activity: {volume_ratio:.1f}x average")
            else:
                signals.append(f"Unusual volume activity: {volume_ratio:.1f}x average")

        # Absolute volume check (40% weight)
        if avg_volume > 50000:
            score += 0.8
            signals.append(f"Good liquidity: {avg_volume:,} average volume")
        elif avg_volume > 10000:
            score += 0.5
            signals.append(f"Adequate liquidity: {avg_volume:,} average volume")
        else:
            signals.append(f"Low liquidity: {avg_volume:,} average volume")

        return {
            'score': score / max_score,
            'signals': signals
        }

    def _analyze_technical_strength(self, indicators: Dict) -> Dict[str, Any]:
        """Analyze overall technical strength."""
        signals = []
        score = 0.0
        max_score = 2.0

        adx_14 = indicators.get('adx_14')
        rsi_14 = indicators.get('rsi_14')

        # ADX trend strength (60% weight)
        if adx_14:
            if adx_14 > 25:
                score += 1.2
                signals.append(f"Strong trend: ADX {adx_14:.1f}")
            elif adx_14 > 20:
                score += 0.8
                signals.append(f"Developing trend: ADX {adx_14:.1f}")
            else:
                signals.append(f"Weak trend: ADX {adx_14:.1f}")

        # RSI momentum direction (40% weight)
        if rsi_14:
            if 45 <= rsi_14 <= 65:
                score += 0.8
                signals.append("RSI in momentum zone")
            elif rsi_14 > 50:
                score += 0.5
                signals.append("RSI above midpoint")
            else:
                signals.append("RSI below midpoint")

        return {
            'score': score / max_score,
            'signals': signals
        }

    def _calculate_overall_score(self, trend: Dict, momentum: Dict, volatility: Dict,
                               liquidity: Dict, technical: Dict) -> float:
        """Calculate weighted overall score."""
        # Weighted scoring
        weights = {
            'trend': 0.30,      # 30% - Most important for direction
            'momentum': 0.25,   # 25% - Important for timing
            'volatility': 0.20, # 20% - Risk management
            'liquidity': 0.15,  # 15% - Execution quality
            'technical': 0.10   # 10% - Confirmation
        }

        overall_score = (
            trend['score'] * weights['trend'] +
            momentum['score'] * weights['momentum'] +
            volatility['score'] * weights['volatility'] +
            liquidity['score'] * weights['liquidity'] +
            technical['score'] * weights['technical']
        )

        return round(overall_score, 4)

    def _generate_recommendation(self, overall_score: float) -> str:
        """Generate recommendation based on overall score."""
        if overall_score >= 0.8:
            return "STRONG BUY"
        elif overall_score >= 0.65:
            return "BUY"
        elif overall_score >= 0.5:
            return "HOLD"
        elif overall_score >= 0.35:
            return "WEAK HOLD"
        else:
            return "AVOID"

    def _get_current_stock_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current stock data from database."""
        try:
            with self.db_manager.get_session() as session:
                stock = session.query(Stock).filter(Stock.symbol == symbol).first()

                if not stock:
                    return None

                return {
                    'current_price': stock.current_price,
                    'volume': stock.volume,
                    'market_cap': stock.market_cap,
                    'market_cap_category': stock.market_cap_category,
                    'pe_ratio': stock.pe_ratio,
                    'pb_ratio': stock.pb_ratio,
                    'sector': stock.sector
                }

        except Exception as e:
            logger.error(f"Error getting current stock data for {symbol}: {e}")
            return None

    def get_filtering_summary(self) -> Dict[str, Any]:
        """Get summary of filtering capabilities and criteria."""
        return {
            'service_name': 'Enhanced Historical Filtering Service',
            'description': 'Advanced stock filtering using 1+ years of historical data and 20+ technical indicators',
            'filtering_dimensions': [
                'Trend Analysis (Moving Averages)',
                'Momentum Indicators (RSI, MACD)',
                'Volatility Control (ATR, Bollinger Bands)',
                'Liquidity Requirements (Volume Analysis)',
                'Technical Strength (ADX, Price Action)'
            ],
            'key_advantages': [
                'Uses comprehensive historical data for reliable analysis',
                'Combines multiple technical indicators for robust signals',
                'Provides weighted scoring system for objective ranking',
                'Includes risk management through volatility analysis',
                'Generates detailed explanations for each recommendation'
            ],
            'data_requirements': 'Minimum 200 trading days of historical OHLCV data',
            'update_frequency': 'Daily calculation of technical indicators'
        }