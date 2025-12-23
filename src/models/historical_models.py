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
    Stores ALL available data from Fyers API plus calculated fields
    """
    __tablename__ = 'historical_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # Core OHLCV Data from Fyers API (ALL 6 fields)
    timestamp = Column(BigInteger, nullable=False)  # Original Unix timestamp
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)

    # Calculated fields for enhanced analysis
    adj_close = Column(Float)  # Adjusted close for splits/dividends
    turnover = Column(Float)  # Daily turnover in INR (price * volume)
    price_change = Column(Float)  # Close - Open
    price_change_pct = Column(Float)  # (Close - Open) / Open * 100
    high_low_pct = Column(Float)  # (High - Low) / Close * 100
    body_pct = Column(Float)  # |Close - Open| / (High - Low) * 100
    upper_shadow_pct = Column(Float)  # Upper wick percentage
    lower_shadow_pct = Column(Float)  # Lower wick percentage

    # Volume analysis
    volume_sma_ratio = Column(Float)  # Volume / SMA(Volume, 20)
    price_volume_trend = Column(Float)  # PVT indicator value

    # Data quality and metadata
    is_adjusted = Column(Boolean, default=False)
    data_source = Column(String(20), default='fyers')
    api_resolution = Column(String(10))  # Original API resolution (1D, 5M, etc.)
    data_quality_score = Column(Float)  # 0-1 score for data completeness

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

    # 8-21 EMA Strategy Indicators (Core)
    ema_8 = Column(Float)      # Fast EMA (8-day) - REQUIRED for power zone
    ema_21 = Column(Float)     # Slow EMA (21-day) - REQUIRED for power zone
    demarker = Column(Float)   # DeMarker oscillator (0-1) - REQUIRED for entry timing

    # Context Indicators (Optional but useful)
    sma_50 = Column(Float)     # Medium-term trend confirmation
    sma_200 = Column(Float)    # Major trend identification (bull/bear market)

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