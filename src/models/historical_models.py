"""
Historical Data Models for Enhanced Technical Analysis
Stores OHLCV data for comprehensive technical indicator calculations
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, BigInteger, Date, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

try:
    from .stock_models import Base
except ImportError:
    from src.models.stock_models import Base


class HistoricalData(Base):
    """
    Historical OHLCV data for stocks - optimized for technical analysis
    Stores daily candle data for 2+ years for accurate indicator calculations
    """
    __tablename__ = 'historical_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # OHLCV Data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)

    # Additional useful data
    adj_close = Column(Float)  # Adjusted close for splits/dividends
    turnover = Column(Float)  # Daily turnover in crores

    # Data quality flags
    is_adjusted = Column(Boolean, default=False)
    data_source = Column(String(20), default='fyers')

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite indexes for performance
    __table_args__ = (
        Index('ix_historical_symbol_date', 'symbol', 'date'),
        Index('ix_historical_date_symbol', 'date', 'symbol'),
    )

    def __repr__(self):
        return f'<HistoricalData {self.symbol} {self.date}: {self.close}>'


class TechnicalIndicators(Base):
    """
    Calculated technical indicators - cached for performance
    Pre-calculated indicators avoid real-time computation overhead
    """
    __tablename__ = 'technical_indicators'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # Moving Averages
    sma_5 = Column(Float)
    sma_10 = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_100 = Column(Float)
    sma_200 = Column(Float)

    ema_12 = Column(Float)
    ema_26 = Column(Float)
    ema_50 = Column(Float)

    # Momentum Indicators
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)

    # Volatility Indicators
    atr_14 = Column(Float)
    atr_percentage = Column(Float)
    bb_upper = Column(Float)  # Bollinger Bands
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_width = Column(Float)

    # Trend Indicators
    adx_14 = Column(Float)

    # Volume Indicators
    obv = Column(Float)  # On Balance Volume
    volume_sma_20 = Column(Float)
    volume_ratio = Column(Float)  # Current volume vs 20-day average

    # Custom indicators
    price_momentum_5d = Column(Float)
    price_momentum_20d = Column(Float)
    volatility_rank = Column(Float)  # Percentile rank of volatility

    # Calculation metadata
    calculation_date = Column(DateTime, default=datetime.utcnow)
    data_points_used = Column(Integer)  # Number of historical points used

    # Composite indexes
    __table_args__ = (
        Index('ix_technical_symbol_date', 'symbol', 'date'),
    )

    def __repr__(self):
        return f'<TechnicalIndicators {self.symbol} {self.date}>'


class MarketBenchmarks(Base):
    """
    Market benchmark data (NIFTY, SENSEX) for beta calculations
    Essential for relative performance analysis
    """
    __tablename__ = 'market_benchmarks'

    id = Column(Integer, primary_key=True)
    benchmark = Column(String(20), nullable=False, index=True)  # NIFTY50, SENSEX
    date = Column(Date, nullable=False, index=True)

    # OHLCV for benchmarks
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger)

    # Additional metrics
    market_cap = Column(Float)  # Total market cap
    pe_ratio = Column(Float)    # Market PE

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Composite indexes
    __table_args__ = (
        Index('ix_benchmark_date', 'benchmark', 'date'),
    )

    def __repr__(self):
        return f'<MarketBenchmark {self.benchmark} {self.date}: {self.close}>'


class DataQualityMetrics(Base):
    """
    Track data quality and completeness for each symbol
    Ensures reliable technical analysis calculations
    """
    __tablename__ = 'data_quality_metrics'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), unique=True, nullable=False, index=True)

    # Data coverage
    earliest_date = Column(Date)
    latest_date = Column(Date)
    total_days = Column(Integer)
    missing_days = Column(Integer)
    data_completeness = Column(Float)  # Percentage (0-100)

    # Data quality scores
    price_consistency_score = Column(Float)  # No unrealistic gaps
    volume_consistency_score = Column(Float)  # Volume patterns
    overall_quality_score = Column(Float)    # Combined score

    # Specific requirements for filtering
    has_200_day_history = Column(Boolean, default=False)
    has_1_year_history = Column(Boolean, default=False)
    meets_min_quality = Column(Boolean, default=False)

    # Update tracking
    last_quality_check = Column(DateTime, default=datetime.utcnow)
    last_data_update = Column(DateTime)

    def __repr__(self):
        return f'<DataQuality {self.symbol}: {self.data_completeness:.1f}%>'