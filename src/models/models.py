"""
Data Models for the Automated Trading System
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from flask_login import UserMixin

# Import enhanced stock models and use their Base
from .stock_models import (
    Stock, StrategyType, StrategyStockSelection, MLPrediction,
    PortfolioStrategy, PortfolioPosition, MarketDataSnapshot, MarketCapCategory,
    SymbolMaster, Base
)


class User(UserMixin, Base):
    """User authentication and profile information."""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    is_mock_trading_mode = Column(Boolean, default=True)  # Mock trading enabled by default
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    last_activity = Column(DateTime)
    
    # Relationships with other models
    orders = relationship("Order", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    positions = relationship("Position", back_populates="user")
    holdings = relationship("Holding", back_populates="user")
    strategies = relationship("Strategy", back_populates="user")
    configurations = relationship("Configuration", back_populates="user")
    logs = relationship("Log", back_populates="user")
    suggested_stocks = relationship("SuggestedStock", back_populates="user")
    broker_configurations = relationship("BrokerConfiguration", back_populates="user")
    strategy_settings = relationship("UserStrategySettings", back_populates="user")
    
    def __repr__(self):
        return f'<User {self.username}>'


class Instrument(Base):
    """Tradable securities with metadata."""
    __tablename__ = 'instruments'
    
    id = Column(Integer, primary_key=True)
    exchange_token = Column(String(50), unique=True, nullable=False)
    tradingsymbol = Column(String(50), nullable=False)
    name = Column(String(100))
    exchange = Column(String(20))
    instrument_type = Column(String(20))
    segment = Column(String(20))
    tick_size = Column(Float)
    lot_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with market data
    market_data = relationship("MarketData", back_populates="instrument")


class MarketData(Base):
    """Price, volume, and quote data."""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    instrument_id = Column(Integer, ForeignKey('instruments.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    last_price = Column(Float)
    last_quantity = Column(Integer)
    average_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with instrument
    instrument = relationship("Instrument", back_populates="market_data")


class Order(Base):
    """Order details and state tracking."""
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    order_id = Column(String(50), unique=True, nullable=False)
    parent_order_id = Column(String(50))
    exchange_order_id = Column(String(50))
    tradingsymbol = Column(String(50), nullable=False)
    exchange = Column(String(20))
    instrument_token = Column(String(50))
    product = Column(String(10))
    order_type = Column(String(10))
    transaction_type = Column(String(10))
    quantity = Column(Integer)
    disclosed_quantity = Column(Integer)
    price = Column(Float)
    trigger_price = Column(Float)
    average_price = Column(Float)
    filled_quantity = Column(Integer)
    pending_quantity = Column(Integer)
    order_status = Column(String(20))
    status_message = Column(Text)
    tag = Column(String(100))
    placed_at = Column(DateTime)
    placed_by = Column(String(50))
    variety = Column(String(20))
    is_mock_order = Column(Boolean, default=False)  # Mock order flag
    model_type = Column(String(20))  # 'traditional' or 'raw_lstm'
    strategy = Column(String(50))  # 'default_risk' or 'high_risk'
    ml_prediction_score = Column(Float)  # ML prediction at time of order
    ml_price_target = Column(Float)  # Price target from ML
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="orders")
    trades = relationship("Trade", back_populates="order")


class Trade(Base):
    """Executed trades with fill details."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    trade_id = Column(String(50), unique=True, nullable=False)
    order_id = Column(String(50), ForeignKey('orders.order_id'), nullable=False)
    exchange_order_id = Column(String(50))
    tradingsymbol = Column(String(50))
    exchange = Column(String(20))
    instrument_token = Column(String(50))
    transaction_type = Column(String(10))
    quantity = Column(Integer)
    price = Column(Float)
    filled_quantity = Column(Integer)
    order_price = Column(Float)
    trade_time = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="trades")
    order = relationship("Order", back_populates="trades")


class Position(Base):
    """Current portfolio positions."""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    tradingsymbol = Column(String(50), nullable=False)
    exchange = Column(String(20))
    instrument_token = Column(String(50))
    product = Column(String(10))
    quantity = Column(Integer)
    overnight_quantity = Column(Integer)
    multiplier = Column(Integer)
    average_price = Column(Float)
    close_price = Column(Float)
    last_price = Column(Float)
    value = Column(Float)
    pnl = Column(Float)
    m2m = Column(Float)
    unrealised = Column(Float)
    realised = Column(Float)
    buy_quantity = Column(Integer)
    buy_price = Column(Float)
    buy_value = Column(Float)
    buy_m2m = Column(Float)
    sell_quantity = Column(Integer)
    sell_price = Column(Float)
    sell_value = Column(Float)
    sell_m2m = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with user
    user = relationship("User", back_populates="positions")


class Holding(Base):
    """Portfolio holdings (long-term investments)."""
    __tablename__ = 'holdings'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    tradingsymbol = Column(String(50), nullable=False)
    exchange = Column(String(20))
    instrument_token = Column(String(50))
    product = Column(String(10))
    quantity = Column(Integer)
    average_price = Column(Float)
    last_price = Column(Float)
    market_value = Column(Float)
    invested_value = Column(Float)
    pnl = Column(Float)
    pnl_percentage = Column(Float)
    holding_type = Column(String(10))  # T1, T0, etc.
    authorized_date = Column(String(20))
    authorized_quantity = Column(Integer)
    opening_quantity = Column(Integer)
    holding_quantity = Column(Integer)
    collateral_quantity = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship with user
    user = relationship("User", back_populates="holdings")


class Strategy(Base):
    """Strategy selection parameters."""
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    parameters = Column(Text)  # JSON string of strategy parameters
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with user
    user = relationship("User", back_populates="strategies")


class Configuration(Base):
    """System settings and thresholds."""
    __tablename__ = 'configurations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)  # NULL for global configs
    key = Column(String(100), nullable=False)
    value = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with user (optional for global configs)
    user = relationship("User", back_populates="configurations")
    
    # Unique constraint: key should be unique per user (or globally if user_id is NULL)
    __table_args__ = (
        UniqueConstraint('user_id', 'key', name='_user_key_uc'),
    )


class UserStrategySettings(Base):
    """User-specific strategy settings and preferences."""
    __tablename__ = 'user_strategy_settings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    strategy_name = Column(String(100), nullable=False)  # 'default_risk', 'high_risk'
    is_active = Column(Boolean, default=True)
    is_enabled = Column(Boolean, default=True)  # User can enable/disable
    priority = Column(Integer, default=1)  # Display order
    custom_parameters = Column(Text)  # JSON string for custom strategy parameters
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    
    # Unique constraint to ensure one setting per user per strategy
    __table_args__ = (
        UniqueConstraint('user_id', 'strategy_name', name='unique_user_strategy'),
    )
    
    def __repr__(self):
        return f'<UserStrategySettings user_id={self.user_id} strategy={self.strategy_name} active={self.is_active}>'


class Log(Base):
    """Audit trail and system events."""
    __tablename__ = 'logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)  # NULL for system logs
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(20))
    module = Column(String(50))
    message = Column(Text)
    details = Column(Text)  # JSON string of additional details
    
    # Relationship with user (optional for system logs)
    user = relationship("User", back_populates="logs")


class SuggestedStock(Base):
    """Suggested stocks with performance tracking."""
    __tablename__ = 'suggested_stocks'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    symbol = Column(String(50), nullable=False)
    selection_date = Column(DateTime, default=datetime.utcnow)
    selection_price = Column(Float)
    current_price = Column(Float)
    quantity = Column(Integer)
    strategy_name = Column(String(100))
    status = Column(String(20), default='Active')  # Active, Sold, Expired
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with user
    user = relationship("User", back_populates="suggested_stocks")


class ScreenedStock(Base):
    """Stocks that passed screening criteria."""
    __tablename__ = 'screened_stocks'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    name = Column(String(100))
    sector = Column(String(50))
    exchange = Column(String(20))
    market_cap = Column(Float)
    current_price = Column(Float)
    screening_date = Column(DateTime, default=datetime.utcnow)
    screening_criteria = Column(Text)  # JSON string of criteria used
    financial_data = Column(Text)  # JSON string of financial metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    strategy_selections = relationship("StrategySelection", back_populates="screened_stock")


class StrategySelection(Base):
    """Stocks selected by specific strategies."""
    __tablename__ = 'strategy_selections'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    screened_stock_id = Column(Integer, ForeignKey('screened_stocks.id'), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    selection_date = Column(DateTime, default=datetime.utcnow)
    selection_score = Column(Float)  # Strategy-specific score
    allocation_percentage = Column(Float)  # Portfolio allocation percentage
    position_size = Column(Integer)  # Number of shares
    status = Column(String(20), default='Selected')  # Selected, Executed, Exited
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    screened_stock = relationship("ScreenedStock", back_populates="strategy_selections")
    dry_run_positions = relationship("DryRunPosition", back_populates="strategy_selection")


class DryRunPortfolio(Base):
    """Dry run portfolio for strategy testing."""
    __tablename__ = 'dry_run_portfolios'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    portfolio_id = Column(String(100), unique=True, nullable=False)
    strategy_name = Column(String(100), nullable=False)
    initial_capital = Column(Float, nullable=False)
    current_capital = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User")
    positions = relationship("DryRunPosition", back_populates="portfolio")
    performance_snapshots = relationship("DryRunPerformance", back_populates="portfolio")


class DryRunPosition(Base):
    """Positions in dry run portfolios."""
    __tablename__ = 'dry_run_positions'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('dry_run_portfolios.id'), nullable=False)
    strategy_selection_id = Column(Integer, ForeignKey('strategy_selections.id'), nullable=False)
    symbol = Column(String(50), nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    average_price = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("DryRunPortfolio", back_populates="positions")
    strategy_selection = relationship("StrategySelection", back_populates="dry_run_positions")


class DryRunPerformance(Base):
    """Performance snapshots for dry run portfolios."""
    __tablename__ = 'dry_run_performance'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('dry_run_portfolios.id'), nullable=False)
    snapshot_date = Column(DateTime, default=datetime.utcnow)
    portfolio_value = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    return_percentage = Column(Float, nullable=False)
    num_positions = Column(Integer, nullable=False)
    performance_metrics = Column(Text)  # JSON string of detailed metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("DryRunPortfolio", back_populates="performance_snapshots")


class ExecutionLog(Base):
    """Log of trading workflow executions."""
    __tablename__ = 'execution_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    execution_id = Column(String(100), unique=True, nullable=False)
    execution_type = Column(String(50), nullable=False)  # 'complete_workflow', 'dry_run', 'screening_only'
    status = Column(String(20), nullable=False)  # 'success', 'error', 'partial'
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    screened_stocks_count = Column(Integer, default=0)
    strategies_executed = Column(Text)  # JSON string of strategy names
    results_summary = Column(Text)  # JSON string of execution results
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")


class AIAnalysis(Base):
    """AI analysis results from ChatGPT."""
    __tablename__ = 'ai_analyses'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    analysis_type = Column(String(50), nullable=False)  # 'stock', 'portfolio', 'strategy_comparison'
    target_id = Column(String(100))  # Stock symbol, strategy name, etc.
    analysis_data = Column(Text, nullable=False)  # JSON string of analysis results
    confidence_score = Column(Float)
    recommendation = Column(String(20))  # BUY, HOLD, SELL
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")


class BrokerConfiguration(Base):
    """Broker configuration and credentials."""
    __tablename__ = 'broker_configurations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)  # NULL for global configs
    broker_name = Column(String(50), nullable=False)  # 'fyers', 'zerodha', etc.
    client_id = Column(String(100))
    access_token = Column(Text)
    refresh_token = Column(Text)
    api_key = Column(String(200))
    api_secret = Column(Text)
    redirect_url = Column(String(500))
    app_type = Column(String(20))  # '100' for web, '2' for mobile
    is_active = Column(Boolean, default=True)
    is_connected = Column(Boolean, default=False)
    last_connection_test = Column(DateTime)
    connection_status = Column(String(20))  # 'connected', 'disconnected', 'error'
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with user (optional for global configs)
    user = relationship("User")
    
    # Unique constraint: broker_name should be unique per user (or globally if user_id is NULL)
    __table_args__ = (
        UniqueConstraint('user_id', 'broker_name', name='_user_broker_uc'),
    )


class MLTrainingJob(Base):
    """ML training job tracking."""
    __tablename__ = 'ml_training_jobs'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    symbol = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # 'ensemble', 'random_forest', 'xgboost', 'lstm'
    start_date = Column(DateTime, nullable=False)  # Training data start date
    end_date = Column(DateTime, nullable=False)    # Training data end date
    duration = Column(String(10), nullable=False)   # '1M', '3M', '6M', '1Y', etc.
    use_technical_indicators = Column(Boolean, default=True)
    status = Column(String(20), default='pending')  # 'pending', 'running', 'completed', 'failed'
    progress = Column(Float, default=0.0)  # 0.0 to 100.0
    accuracy = Column(Float)  # Model accuracy after training
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Relationships
    user = relationship("User")


class MLTrainedModel(Base):
    """Trained ML model metadata."""
    __tablename__ = 'ml_trained_models'

    id = Column(Integer, primary_key=True)
    training_job_id = Column(Integer, ForeignKey('ml_training_jobs.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    symbol = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    model_file_path = Column(Text)  # Path to saved model file
    scaler_file_path = Column(Text)  # Path to saved scaler file
    feature_columns = Column(Text)  # JSON array of feature columns (matches DB)
    target_column = Column(String(50))  # Target column name
    model_version = Column(String(50))  # Model version
    accuracy = Column(Float)
    training_start_date = Column(DateTime, nullable=False)  # Matches DB
    training_end_date = Column(DateTime, nullable=False)    # Matches DB
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    training_job = relationship("MLTrainingJob")
    user = relationship("User")
