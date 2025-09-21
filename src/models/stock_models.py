"""
Enhanced Data Models for Stock Management
Adds comprehensive stock data storage and categorization
"""
from sqlalchemy import Column, Integer, BigInteger, String, Float, DateTime, Boolean, Text, ForeignKey, UniqueConstraint, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

# Create separate base to avoid circular imports
Base = declarative_base()


class MarketCapCategory(enum.Enum):
    """Market capitalization categories."""
    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"


class Stock(Base):
    """Master stock information with categorization."""
    __tablename__ = 'stocks'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    exchange = Column(String(20), nullable=False, default='NSE')
    sector = Column(String(100))

    # Market capitalization data
    market_cap = Column(Float)  # in crores
    market_cap_category = Column(String(20), index=True)  # Use String instead of Enum for compatibility

    # Current market data
    current_price = Column(Float)
    volume = Column(BigInteger)  # Use BigInteger to handle large volume values

    # Fundamental ratios (basic set that matches actual table)
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    roe = Column(Float)  # Return on Equity
    debt_to_equity = Column(Float)
    dividend_yield = Column(Float)
    beta = Column(Float)

    # Volatility and Risk Metrics for Stage 1 Filtering
    atr_14 = Column(Float)  # Average True Range (14 days)
    atr_percentage = Column(Float)  # ATR as percentage of price
    historical_volatility_1y = Column(Float)  # 1-year historical volatility
    avg_daily_volume_20d = Column(Float)  # 20-day average volume (matches schema)
    avg_daily_turnover = Column(Float)  # Average daily turnover in crores
    bid_ask_spread = Column(Float)  # Bid-ask spread percentage
    trades_per_day = Column(Integer)  # Average trades per day

    # Status and metadata
    is_active = Column(Boolean, default=True, index=True)
    is_tradeable = Column(Boolean, default=True)
    volatility_last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # For volatility update tracking
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # For stock data update tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    strategy_selections = relationship("StrategyStockSelection", back_populates="stock")
    ml_predictions = relationship("MLPrediction", back_populates="stock")
    
    def __repr__(self):
        return f'<Stock {self.symbol}: {self.name}>'


class StrategyType(Base):
    """Strategy definitions and configurations."""
    __tablename__ = 'strategy_types'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)  # 'default_risk', 'high_risk'
    display_name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Strategy configuration
    config_json = Column(Text)  # JSON string of strategy parameters
    
    # Market cap allocation (for default_risk strategy)
    large_cap_allocation = Column(Float, default=0.0)  # 0.0 to 1.0
    mid_cap_allocation = Column(Float, default=0.0)
    small_cap_allocation = Column(Float, default=0.0)
    
    # Risk parameters
    risk_level = Column(String(20))  # 'low', 'medium', 'high'
    max_position_size = Column(Float)  # Maximum % of portfolio per stock
    max_sector_allocation = Column(Float)  # Maximum % per sector
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    stock_selections = relationship("StrategyStockSelection", back_populates="strategy_type")


class StrategyStockSelection(Base):
    """Stocks selected by specific strategies."""
    __tablename__ = 'strategy_stock_selections'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    strategy_type_id = Column(Integer, ForeignKey('strategy_types.id'), nullable=False)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    
    # Selection details
    selection_date = Column(DateTime, default=datetime.utcnow)
    selection_price = Column(Float, nullable=False)
    selection_score = Column(Float)  # Algorithm confidence score
    
    # Position sizing
    recommended_quantity = Column(Integer)
    recommended_allocation = Column(Float)  # % of portfolio
    position_size_rationale = Column(Text)
    
    # Target prices and risk management
    target_price = Column(Float)
    stop_loss = Column(Float)
    expected_return = Column(Float)  # % expected return
    risk_score = Column(Float)  # 1-10 risk score
    
    # Status tracking
    status = Column(String(20), default='selected')  # selected, executed, exited, expired
    execution_date = Column(DateTime)
    exit_date = Column(DateTime)
    
    # Performance tracking
    current_price = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float)
    
    # Metadata
    selection_reason = Column(Text)
    algorithm_version = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    strategy_type = relationship("StrategyType", back_populates="stock_selections")
    stock = relationship("Stock", back_populates="strategy_selections")


class MLPrediction(Base):
    """Machine Learning predictions for stocks."""
    __tablename__ = 'ml_predictions'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    
    # Prediction details
    prediction_date = Column(DateTime, default=datetime.utcnow)
    prediction_horizon_days = Column(Integer, default=30)  # Prediction horizon
    
    # Price predictions from different models
    current_price = Column(Float, nullable=False)
    rf_predicted_price = Column(Float)  # Random Forest
    xgb_predicted_price = Column(Float)  # XGBoost
    lstm_predicted_price = Column(Float)  # LSTM
    ensemble_predicted_price = Column(Float)  # Final ensemble prediction
    
    # Prediction confidence and metrics
    prediction_confidence = Column(Float)  # 0.0 to 1.0
    model_accuracy = Column(Float)  # Historical accuracy %
    prediction_std = Column(Float)  # Standard deviation of prediction
    
    # Trading signals
    signal = Column(String(10))  # BUY, SELL, HOLD
    signal_strength = Column(Float)  # 0.0 to 1.0
    expected_return = Column(Float)  # % expected return
    risk_reward_ratio = Column(Float)
    
    # Model metadata
    model_version = Column(String(50))
    features_used = Column(Text)  # JSON array of features
    training_data_period = Column(String(20))  # e.g., "1y", "2y"
    
    # Validation tracking
    actual_price = Column(Float)  # Actual price after prediction horizon
    prediction_error = Column(Float)  # % error vs actual
    is_validated = Column(Boolean, default=False)
    validation_date = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="ml_predictions")
    user = relationship("User")
    
    # Index for efficient querying
    __table_args__ = (
        UniqueConstraint('stock_id', 'prediction_date', 'prediction_horizon_days', name='_stock_prediction_uc'),
    )


class PortfolioStrategy(Base):
    """User portfolio strategies and allocations."""
    __tablename__ = 'portfolio_strategies'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Strategy details
    strategy_name = Column(String(100), nullable=False)  # 'Conservative Growth', 'Aggressive Small Cap', etc.
    strategy_type = Column(String(20), nullable=False)  # 'default_risk', 'high_risk'
    
    # Capital allocation
    total_capital = Column(Float, nullable=False)
    allocated_capital = Column(Float, default=0.0)
    available_capital = Column(Float)
    
    # Market cap allocation (percentages)
    large_cap_allocation = Column(Float, default=0.6)  # 60%
    mid_cap_allocation = Column(Float, default=0.3)   # 30%
    small_cap_allocation = Column(Float, default=0.1) # 10%
    
    # Risk management
    max_position_size = Column(Float, default=0.05)  # Max 5% per stock
    max_sector_allocation = Column(Float, default=0.20)  # Max 20% per sector
    stop_loss_percentage = Column(Float, default=0.10)  # 10% stop loss
    
    # Rebalancing
    rebalance_frequency_days = Column(Integer, default=30)  # Monthly rebalancing
    last_rebalance_date = Column(DateTime)
    next_rebalance_date = Column(DateTime)
    
    # Performance tracking
    initial_value = Column(Float)
    current_value = Column(Float)
    total_return = Column(Float)
    return_percentage = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    positions = relationship("PortfolioPosition", back_populates="portfolio_strategy")


class PortfolioPosition(Base):
    """Individual stock positions in user portfolios."""
    __tablename__ = 'portfolio_positions'
    
    id = Column(Integer, primary_key=True)
    portfolio_strategy_id = Column(Integer, ForeignKey('portfolio_strategies.id'), nullable=False)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    
    # Position details
    entry_date = Column(DateTime, default=datetime.utcnow)
    entry_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    investment_amount = Column(Float, nullable=False)
    
    # Current position data
    current_price = Column(Float)
    current_value = Column(Float)
    unrealized_pnl = Column(Float)
    unrealized_pnl_percentage = Column(Float)
    
    # Target and risk management
    target_price = Column(Float)
    stop_loss = Column(Float)
    position_allocation = Column(Float)  # % of portfolio
    
    # Exit details (if position is closed)
    exit_date = Column(DateTime)
    exit_price = Column(Float)
    realized_pnl = Column(Float)
    realized_pnl_percentage = Column(Float)
    
    # Status
    status = Column(String(20), default='active')  # active, closed, partial
    
    # Metadata
    entry_reason = Column(Text)
    exit_reason = Column(Text)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio_strategy = relationship("PortfolioStrategy", back_populates="positions")
    stock = relationship("Stock")


class SymbolMaster(Base):
    """Raw symbol master data from broker APIs."""
    __tablename__ = 'symbol_master'

    # Symbol identification - fytoken is the primary key (TRUE UNIQUE IDENTIFIER)
    fytoken = Column(String(50), primary_key=True, nullable=False)  # Fyers unique token (PRIMARY KEY)
    symbol = Column(String(50), nullable=False, index=True)  # NSE:SYMBOL-EQ
    name = Column(String(200), nullable=False)
    exchange = Column(String(20), nullable=False, index=True)  # NSE, BSE
    segment = Column(String(20), nullable=False)  # CM (Capital Market)
    instrument_type = Column(String(20), nullable=False)  # EQ (Equity)

    # Trading parameters
    lot_size = Column(Integer, default=1)
    tick_size = Column(Float, default=0.05)
    isin = Column(String(20))  # ISIN code

    # Data source and versioning
    data_source = Column(String(20), default='fyers')
    source_updated = Column(String(20))  # Last updated timestamp from source
    download_date = Column(DateTime, default=datetime.utcnow)

    # Status flags
    is_active = Column(Boolean, default=True, index=True)
    is_equity = Column(Boolean, default=True, index=True)  # Only equity symbols

    # Verification status for Fyers API compatibility
    is_fyers_verified = Column(Boolean, default=False, index=True)
    verification_date = Column(DateTime)
    verification_error = Column(Text)
    last_quote_check = Column(DateTime)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Unique constraint to prevent duplicate symbol-exchange combinations
    __table_args__ = (
        UniqueConstraint('symbol', 'exchange', name='_symbol_exchange_uc'),
    )

    def __repr__(self):
        return f'<SymbolMaster {self.symbol}: {self.name}>'


class MarketDataSnapshot(Base):
    """Daily market data snapshots for analysis."""
    __tablename__ = 'market_data_snapshots'
    
    id = Column(Integer, primary_key=True)
    snapshot_date = Column(String(10), nullable=False, index=True)  # YYYY-MM-DD
    
    # Market indices
    nifty_50 = Column(Float)
    sensex = Column(Float)
    nifty_midcap = Column(Float)
    nifty_smallcap = Column(Float)
    
    # Market statistics
    total_stocks_tracked = Column(Integer)
    large_cap_avg_change = Column(Float)
    mid_cap_avg_change = Column(Float)
    small_cap_avg_change = Column(Float)
    
    # Volume data
    total_volume = Column(Integer)
    advance_decline_ratio = Column(Float)
    
    # Metadata
    data_source = Column(String(20), default='fyers')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('snapshot_date', name='_daily_snapshot_uc'),
    )
