"""
Data Models for the Automated Trading System
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from flask_login import UserMixin

Base = declarative_base()


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
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    last_activity = Column(DateTime)
    
    # Relationships with other models
    orders = relationship("Order", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    positions = relationship("Position", back_populates="user")
    strategies = relationship("Strategy", back_populates="user")
    configurations = relationship("Configuration", back_populates="user")
    logs = relationship("Log", back_populates="user")
    selected_stocks = relationship("SelectedStock", back_populates="user")
    
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


class Strategy(Base):
    """Momentum selection parameters."""
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
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with user (optional for global configs)
    user = relationship("User", back_populates="configurations")
    
    # Unique constraint: key should be unique per user (or globally if user_id is NULL)
    __table_args__ = (
        UniqueConstraint('user_id', 'key', name='_user_key_uc'),
    )


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


class SelectedStock(Base):
    """Selected stocks with performance tracking."""
    __tablename__ = 'selected_stocks'
    
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
    user = relationship("User", back_populates="selected_stocks")
