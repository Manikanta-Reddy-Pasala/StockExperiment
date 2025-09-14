"""
Enhanced Portfolio Strategy Engine with 4-Step Logic
=====================================================
Broker: FYERS
Steps:
1. Filtering (Initial Stock Universe)
2. Risk Strategy Allocation  
3. Entry Rules
4. Exit Rules

Machine Learning API integration is preserved and used for enhanced predictions.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import time

# Existing ML service import - DO NOT CHANGE
from .ml.prediction_service import get_prediction
from .fyers_api_service import FyersAPIService
from .broker_service import get_broker_service
from .cache_service import CacheService

logger = logging.getLogger(__name__)


# ============= Configuration Classes =============

class MarketCapCategory(Enum):
    """Market cap categories in INR Crores."""
    LARGE_CAP = "large_cap"     # > ₹50,000 Cr
    MID_CAP = "mid_cap"         # ₹10,000–50,000 Cr  
    SMALL_CAP = "small_cap"     # < ₹10,000 Cr


class RiskStrategy(Enum):
    """Risk strategy allocation buckets."""
    SAFE = "safe"               # 50% Large-cap + 50% Mid-cap
    HIGH_RISK = "high_risk"     # 50% Mid-cap + 50% Small-cap


@dataclass
class FilteringCriteria:
    """Step 1: Filtering criteria to remove junk stocks."""
    min_price: float = 50.0                      # Exclude stocks < ₹50
    min_avg_volume_20d: int = 500000            # Liquidity filter
    max_atr_percent: float = 10.0               # Volatility filter (ATR % of price)
    
    def to_dict(self) -> Dict:
        return {
            'min_price': self.min_price,
            'min_avg_volume_20d': self.min_avg_volume_20d,
            'max_atr_percent': self.max_atr_percent
        }


@dataclass
class RiskAllocation:
    """Step 2: Risk strategy allocation rules."""
    large_cap_threshold: float = 50000.0        # Crores
    mid_cap_min: float = 10000.0               # Crores
    mid_cap_max: float = 50000.0               # Crores
    
    safe_large_cap_percent: float = 0.5         # 50% Large-cap
    safe_mid_cap_percent: float = 0.5           # 50% Mid-cap
    
    high_risk_mid_cap_percent: float = 0.5      # 50% Mid-cap
    high_risk_small_cap_percent: float = 0.5    # 50% Small-cap


@dataclass
class EntryRules:
    """Step 3: Entry signal validation rules."""
    ema_20_enabled: bool = True                 # Price above 20-day EMA
    ema_50_enabled: bool = True                 # Price above 50-day EMA
    breakout_days: int = 20                     # Breakout period
    volume_multiplier: float = 1.5              # Volume ≥ 1.5× average
    rsi_min: float = 50.0                       # RSI lower bound
    rsi_max: float = 70.0                       # RSI upper bound (not overbought)
    
    def to_dict(self) -> Dict:
        return {
            'ema_20': self.ema_20_enabled,
            'ema_50': self.ema_50_enabled,
            'breakout_days': self.breakout_days,
            'volume_multiplier': self.volume_multiplier,
            'rsi_range': [self.rsi_min, self.rsi_max]
        }


@dataclass  
class ExitRules:
    """Step 4: Exit rules configuration."""
    profit_target_1_percent: float = 5.0        # Sell 50% at +5%
    profit_target_1_qty_percent: float = 50.0   # Quantity to sell at target 1
    profit_target_2_percent: float = 10.0       # Sell remaining at +10%
    stop_loss_percent: float = 3.0              # Exit if falls 2-4% (default 3%)
    max_holding_days: int = 10                  # Time stop at day 10
    trailing_stop_enabled: bool = True          # Enable trailing stop
    trailing_stop_percent: float = 3.0          # Trail by 3% below current
    
    def to_dict(self) -> Dict:
        return {
            'targets': [
                {'percent': self.profit_target_1_percent, 'qty': self.profit_target_1_qty_percent},
                {'percent': self.profit_target_2_percent, 'qty': 100 - self.profit_target_1_qty_percent}
            ],
            'stop_loss': self.stop_loss_percent,
            'max_days': self.max_holding_days,
            'trailing_stop': self.trailing_stop_percent if self.trailing_stop_enabled else None
        }


@dataclass
class StockSignal:
    """Complete stock signal with all indicators."""
    symbol: str
    name: str
    current_price: float
    market_cap: float
    market_cap_category: MarketCapCategory
    
    # Step 1: Filtering indicators
    avg_volume_20d: float
    atr_14: float
    atr_percent: float
    passes_filter: bool
    
    # Step 3: Entry indicators
    ema_20: float
    ema_50: float
    high_20d: float
    current_volume: float
    avg_volume: float
    rsi_14: float
    
    # Entry signals
    price_above_ema20: bool = False
    price_above_ema50: bool = False
    breakout_20d: bool = False
    volume_confirmation: bool = False
    rsi_valid: bool = False
    entry_signal: bool = False
    
    # ML prediction (preserved)
    ml_prediction: Optional[Dict] = None
    ml_confidence: float = 0.0
    
    # Risk allocation
    risk_strategy: Optional[RiskStrategy] = None
    
    # Metadata
    scan_timestamp: datetime = field(default_factory=datetime.now)
    entry_reason: str = ""


@dataclass
class Position:
    """Position tracking for exit management."""
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: int
    remaining_quantity: int
    
    # Current status
    current_price: float = 0.0
    high_since_entry: float = 0.0
    days_held: int = 0
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    realized_pnl: float = 0.0
    
    # Exit tracking
    partial_exit_done: bool = False
    partial_exit_price: float = 0.0
    trailing_stop_price: float = 0.0
    
    # Rules and strategy
    exit_rules: ExitRules = field(default_factory=ExitRules)
    risk_strategy: RiskStrategy = RiskStrategy.SAFE
    market_cap_category: MarketCapCategory = MarketCapCategory.MID_CAP
    
    # ML prediction at entry
    ml_prediction: Optional[Dict] = None
    
    # Exit reason
    exit_reason: str = ""
    status: str = "ACTIVE"  # ACTIVE, PARTIAL_EXIT, CLOSED
    
    def update_metrics(self, current_price: float):
        """Update position metrics."""
        self.current_price = current_price
        self.high_since_entry = max(self.high_since_entry, current_price)
        self.days_held = (datetime.now() - self.entry_date).days
        
        # Update P&L
        self.unrealized_pnl = (current_price - self.entry_price) * self.remaining_quantity
        self.unrealized_pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
        
        # Update trailing stop if enabled
        if self.exit_rules.trailing_stop_enabled:
            new_trailing_stop = self.high_since_entry * (1 - self.exit_rules.trailing_stop_percent / 100)
            self.trailing_stop_price = max(self.trailing_stop_price, new_trailing_stop)


# ============= Main Strategy Engine =============

class EnhancedPortfolioStrategy:
    """
    Enhanced Portfolio Strategy Engine implementing the 4-step strategy.
    Integrates with FYERS broker and preserves ML prediction capabilities.
    """
    
    def __init__(self, user_id: int = 1, capital: float = 100000.0):
        """Initialize the enhanced strategy engine."""
        self.user_id = user_id
        self.capital = capital
        self.available_capital = capital
        
        # Initialize services
        self.broker_service = get_broker_service()
        self.fyers_api = FyersAPIService()
        self.cache_service = CacheService()
        
        # Strategy configuration
        self.filtering_criteria = FilteringCriteria()
        self.risk_allocation = RiskAllocation()
        self.entry_rules = EntryRules()
        self.exit_rules = ExitRules()
        
        # Portfolio tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.watch_list: List[StockSignal] = []
        
        # Performance metrics
        self.total_realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        logger.info(f"Enhanced Portfolio Strategy initialized with capital: ₹{capital:,.2f}")
    
    # ============= Step 1: Filtering =============
    
    def apply_filtering(self, stock_universe: List[str]) -> List[Dict]:
        """
        Step 1: Apply filtering criteria to remove junk stocks.
        
        Args:
            stock_universe: List of stock symbols to filter
            
        Returns:
            List of stocks that pass all filters with their data
        """
        logger.info(f"Step 1: Filtering {len(stock_universe)} stocks...")
        filtered_stocks = []
        
        for symbol in stock_universe:
            try:
                stock_data = self._get_stock_data(symbol)
                
                if not stock_data:
                    continue
                
                # Apply filters
                price_pass = stock_data['current_price'] >= self.filtering_criteria.min_price
                volume_pass = stock_data['avg_volume_20d'] >= self.filtering_criteria.min_avg_volume_20d
                volatility_pass = stock_data['atr_percent'] <= self.filtering_criteria.max_atr_percent
                
                if price_pass and volume_pass and volatility_pass:
                    stock_data['passes_filter'] = True
                    filtered_stocks.append(stock_data)
                    logger.debug(f"✓ {symbol} passed all filters")
                else:
                    logger.debug(f"✗ {symbol} failed: Price={price_pass}, Volume={volume_pass}, Volatility={volatility_pass}")
                    
            except Exception as e:
                logger.error(f"Error filtering {symbol}: {e}")
                continue
        
        logger.info(f"Filtering complete: {len(filtered_stocks)}/{len(stock_universe)} stocks passed")
        return filtered_stocks
    
    # ============= Step 2: Risk Strategy Allocation =============
    
    def allocate_risk_strategy(self, filtered_stocks: List[Dict], risk_preference: RiskStrategy = RiskStrategy.SAFE) -> Dict[str, List[Dict]]:
        """
        Step 2: Allocate stocks to risk buckets based on market cap.
        
        Args:
            filtered_stocks: Stocks that passed filtering
            risk_preference: User's risk preference (SAFE or HIGH_RISK)
            
        Returns:
            Dictionary with categorized stocks by market cap and selected stocks for strategy
        """
        logger.info(f"Step 2: Risk allocation for {risk_preference.value} strategy...")
        
        # Categorize by market cap
        large_cap = []
        mid_cap = []
        small_cap = []
        
        for stock in filtered_stocks:
            market_cap_cr = stock.get('market_cap', 0) / 10000000  # Convert to Crores
            stock['market_cap_cr'] = market_cap_cr
            
            if market_cap_cr > self.risk_allocation.large_cap_threshold:
                stock['market_cap_category'] = MarketCapCategory.LARGE_CAP
                large_cap.append(stock)
            elif market_cap_cr >= self.risk_allocation.mid_cap_min:
                stock['market_cap_category'] = MarketCapCategory.MID_CAP
                mid_cap.append(stock)
            else:
                stock['market_cap_category'] = MarketCapCategory.SMALL_CAP
                small_cap.append(stock)
        
        # Select stocks based on risk strategy
        selected_stocks = []
        
        if risk_preference == RiskStrategy.SAFE:
            # 50% Large-cap + 50% Mid-cap
            large_cap_count = int(len(large_cap) * self.risk_allocation.safe_large_cap_percent)
            mid_cap_count = int(len(mid_cap) * self.risk_allocation.safe_mid_cap_percent)
            
            selected_stocks.extend(large_cap[:large_cap_count])
            selected_stocks.extend(mid_cap[:mid_cap_count])
            
        else:  # HIGH_RISK
            # 50% Mid-cap + 50% Small-cap
            mid_cap_count = int(len(mid_cap) * self.risk_allocation.high_risk_mid_cap_percent)
            small_cap_count = int(len(small_cap) * self.risk_allocation.high_risk_small_cap_percent)
            
            selected_stocks.extend(mid_cap[:mid_cap_count])
            selected_stocks.extend(small_cap[:small_cap_count])
        
        logger.info(f"Risk allocation complete: Large={len(large_cap)}, Mid={len(mid_cap)}, Small={len(small_cap)}")
        logger.info(f"Selected {len(selected_stocks)} stocks for {risk_preference.value} strategy")
        
        return {
            'large_cap': large_cap,
            'mid_cap': mid_cap,
            'small_cap': small_cap,
            'selected': selected_stocks,
            'strategy': risk_preference
        }
    
    # ============= Step 3: Entry Rules =============
    
    def check_entry_signals(self, selected_stocks: List[Dict]) -> List[StockSignal]:
        """
        Step 3: Check entry signals for selected stocks.
        
        Args:
            selected_stocks: Stocks selected based on risk strategy
            
        Returns:
            List of StockSignal objects with entry validation
        """
        logger.info(f"Step 3: Checking entry signals for {len(selected_stocks)} stocks...")
        entry_signals = []
        
        for stock in selected_stocks:
            try:
                symbol = stock['symbol']
                
                # Get technical indicators
                indicators = self._get_technical_indicators(symbol)
                if not indicators:
                    continue
                
                # Create stock signal
                signal = StockSignal(
                    symbol=symbol,
                    name=stock.get('name', symbol),
                    current_price=stock['current_price'],
                    market_cap=stock['market_cap'],
                    market_cap_category=stock['market_cap_category'],
                    avg_volume_20d=stock['avg_volume_20d'],
                    atr_14=stock['atr_14'],
                    atr_percent=stock['atr_percent'],
                    passes_filter=True,
                    ema_20=indicators['ema_20'],
                    ema_50=indicators['ema_50'],
                    high_20d=indicators['high_20d'],
                    current_volume=indicators['current_volume'],
                    avg_volume=indicators['avg_volume'],
                    rsi_14=indicators['rsi_14']
                )
                
                # Validate entry conditions
                signal.price_above_ema20 = stock['current_price'] > indicators['ema_20']
                signal.price_above_ema50 = stock['current_price'] > indicators['ema_50']
                signal.breakout_20d = stock['current_price'] > indicators['high_20d']
                signal.volume_confirmation = indicators['current_volume'] >= (indicators['avg_volume'] * self.entry_rules.volume_multiplier)
                signal.rsi_valid = self.entry_rules.rsi_min <= indicators['rsi_14'] <= self.entry_rules.rsi_max
                
                # Check if all entry conditions are met
                conditions_met = []
                if self.entry_rules.ema_20_enabled and signal.price_above_ema20:
                    conditions_met.append("EMA20")
                if self.entry_rules.ema_50_enabled and signal.price_above_ema50:
                    conditions_met.append("EMA50")
                if signal.breakout_20d:
                    conditions_met.append("Breakout")
                if signal.volume_confirmation:
                    conditions_met.append("Volume")
                if signal.rsi_valid:
                    conditions_met.append("RSI")
                
                # Entry signal is valid if all required conditions are met
                required_conditions = 5  # All conditions must be met
                signal.entry_signal = len(conditions_met) >= required_conditions
                signal.entry_reason = f"Conditions met: {', '.join(conditions_met)}"
                
                # Get ML prediction (preserving existing ML integration)
                try:
                    ml_result = get_prediction(symbol)
                    if ml_result:
                        signal.ml_prediction = ml_result
                        signal.ml_confidence = ml_result.get('confidence', 0.0)
                        logger.debug(f"ML prediction for {symbol}: {ml_result.get('prediction')} (confidence: {signal.ml_confidence:.2%})")
                except Exception as e:
                    logger.warning(f"ML prediction failed for {symbol}: {e}")
                
                if signal.entry_signal:
                    entry_signals.append(signal)
                    logger.info(f"✓ {symbol} - Entry signal confirmed! {signal.entry_reason}")
                else:
                    logger.debug(f"✗ {symbol} - Entry conditions not met: {signal.entry_reason}")
                    
            except Exception as e:
                logger.error(f"Error checking entry signal for {stock.get('symbol')}: {e}")
        
        logger.info(f"Entry signals found: {len(entry_signals)}/{len(selected_stocks)} stocks")
        return entry_signals
    
    # ============= Step 4: Exit Rules =============
    
    def manage_exits(self):
        """
        Step 4: Monitor and execute exit rules for all positions.
        Checks profit targets, stop loss, time stop, and trailing stop.
        """
        logger.info("Step 4: Managing exits for active positions...")
        
        for symbol, position in list(self.positions.items()):
            try:
                # Get current price
                current_price = self._get_current_price(symbol)
                if not current_price:
                    continue
                
                # Update position metrics
                position.update_metrics(current_price)
                
                # Check exit conditions
                exit_triggered = False
                exit_reason = ""
                exit_quantity = 0
                
                # 1. Check Time Stop (Day 10)
                if position.days_held >= self.exit_rules.max_holding_days:
                    exit_triggered = True
                    exit_quantity = position.remaining_quantity
                    exit_reason = f"TIME_STOP: Held for {position.days_held} days (max: {self.exit_rules.max_holding_days})"
                
                # 2. Check Stop Loss
                elif position.unrealized_pnl_percent <= -self.exit_rules.stop_loss_percent:
                    exit_triggered = True
                    exit_quantity = position.remaining_quantity
                    exit_reason = f"STOP_LOSS: Down {abs(position.unrealized_pnl_percent):.2f}% (stop: {self.exit_rules.stop_loss_percent}%)"
                
                # 3. Check Trailing Stop (if enabled)
                elif self.exit_rules.trailing_stop_enabled and current_price <= position.trailing_stop_price:
                    exit_triggered = True
                    exit_quantity = position.remaining_quantity
                    exit_reason = f"TRAILING_STOP: Price {current_price:.2f} hit trailing stop {position.trailing_stop_price:.2f}"
                
                # 4. Check Profit Target 1 (5% - Sell 50%)
                elif not position.partial_exit_done and position.unrealized_pnl_percent >= self.exit_rules.profit_target_1_percent:
                    exit_triggered = True
                    exit_quantity = int(position.quantity * (self.exit_rules.profit_target_1_qty_percent / 100))
                    exit_reason = f"PROFIT_TARGET_1: Up {position.unrealized_pnl_percent:.2f}% - Selling {self.exit_rules.profit_target_1_qty_percent}%"
                    position.partial_exit_done = True
                    position.partial_exit_price = current_price
                
                # 5. Check Profit Target 2 (10% - Sell remaining)
                elif position.unrealized_pnl_percent >= self.exit_rules.profit_target_2_percent:
                    exit_triggered = True
                    exit_quantity = position.remaining_quantity
                    exit_reason = f"PROFIT_TARGET_2: Up {position.unrealized_pnl_percent:.2f}% - Selling remaining"
                
                # Execute exit if triggered
                if exit_triggered and exit_quantity > 0:
                    success = self._execute_exit(position, exit_quantity, current_price, exit_reason)
                    if success:
                        logger.info(f"✓ Exit executed for {symbol}: {exit_reason}")
                        
                        # Update position status
                        if position.remaining_quantity == 0:
                            position.status = "CLOSED"
                            position.exit_reason = exit_reason
                            self.closed_positions.append(position)
                            del self.positions[symbol]
                        else:
                            position.status = "PARTIAL_EXIT"
                    
            except Exception as e:
                logger.error(f"Error managing exit for {symbol}: {e}")
        
        logger.info(f"Exit management complete. Active positions: {len(self.positions)}")
    
    # ============= Helper Methods =============
    
    def _get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get stock data with technical indicators from FYERS."""
        try:
            # Try cache first
            cache_key = f"stock_data_{symbol}"
            cached_data = self.cache_service.get(cache_key)
            if cached_data:
                return cached_data
            
            # Fetch from FYERS API
            data = self.fyers_api.get_stock_data(symbol)
            if data:
                # Calculate additional metrics
                data['atr_percent'] = (data.get('atr_14', 0) / data.get('current_price', 1)) * 100
                
                # Cache for 5 minutes
                self.cache_service.set(cache_key, data, expire=300)
                return data
                
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
        return None
    
    def _get_technical_indicators(self, symbol: str) -> Optional[Dict]:
        """Get technical indicators for entry signal validation."""
        try:
            # Fetch from FYERS API
            indicators = self.fyers_api.get_technical_indicators(symbol)
            return indicators
        except Exception as e:
            logger.error(f"Error getting indicators for {symbol}: {e}")
        return None
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            quote = self.fyers_api.get_quote(symbol)
            if quote:
                return quote.get('current_price', 0)
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
        return None
    
    def _execute_entry(self, signal: StockSignal, quantity: int) -> bool:
        """Execute entry order through FYERS."""
        try:
            order = {
                'symbol': signal.symbol,
                'qty': quantity,
                'type': 2,  # Market order
                'side': 1,  # Buy
                'productType': 'CNC',  # Cash and Carry
                'limitPrice': 0,
                'stopPrice': 0,
                'validity': 'DAY',
                'disclosedQty': 0,
                'offlineOrder': False
            }
            
            result = self.fyers_api.place_order(order)
            if result and result.get('status') == 'success':
                # Create position record
                position = Position(
                    symbol=signal.symbol,
                    entry_date=datetime.now(),
                    entry_price=signal.current_price,
                    quantity=quantity,
                    remaining_quantity=quantity,
                    current_price=signal.current_price,
                    high_since_entry=signal.current_price,
                    exit_rules=self.exit_rules,
                    risk_strategy=signal.risk_strategy,
                    market_cap_category=signal.market_cap_category,
                    ml_prediction=signal.ml_prediction
                )
                
                self.positions[signal.symbol] = position
                self.available_capital -= (signal.current_price * quantity)
                return True
                
        except Exception as e:
            logger.error(f"Error executing entry for {signal.symbol}: {e}")
        return False
    
    def _execute_exit(self, position: Position, quantity: int, price: float, reason: str) -> bool:
        """Execute exit order through FYERS."""
        try:
            order = {
                'symbol': position.symbol,
                'qty': quantity,
                'type': 2,  # Market order
                'side': -1,  # Sell
                'productType': 'CNC',
                'limitPrice': 0,
                'stopPrice': 0,
                'validity': 'DAY',
                'disclosedQty': 0,
                'offlineOrder': False
            }
            
            result = self.fyers_api.place_order(order)
            if result and result.get('status') == 'success':
                # Update position
                position.remaining_quantity -= quantity
                realized_pnl = (price - position.entry_price) * quantity
                position.realized_pnl += realized_pnl
                
                # Update portfolio metrics
                self.total_realized_pnl += realized_pnl
                self.available_capital += (price * quantity)
                
                if position.remaining_quantity == 0:
                    self.total_trades += 1
                    if position.realized_pnl > 0:
                        self.winning_trades += 1
                
                return True
                
        except Exception as e:
            logger.error(f"Error executing exit for {position.symbol}: {e}")
        return False
    
    # ============= Main Execution Flow =============
    
    def run_strategy(self, stock_universe: List[str], risk_preference: RiskStrategy = RiskStrategy.SAFE):
        """
        Execute the complete 4-step strategy.
        
        Args:
            stock_universe: List of stock symbols to analyze
            risk_preference: Risk strategy preference
            
        Returns:
            Dictionary with execution results
        """
        logger.info("="*60)
        logger.info("STARTING ENHANCED PORTFOLIO STRATEGY")
        logger.info(f"Capital: ₹{self.capital:,.2f} | Risk: {risk_preference.value}")
        logger.info("="*60)
        
        results = {
            'timestamp': datetime.now(),
            'capital': self.capital,
            'risk_preference': risk_preference.value,
            'steps': {}
        }
        
        try:
            # Step 1: Filtering
            filtered_stocks = self.apply_filtering(stock_universe)
            results['steps']['filtering'] = {
                'input_count': len(stock_universe),
                'output_count': len(filtered_stocks),
                'criteria': self.filtering_criteria.to_dict()
            }
            
            if not filtered_stocks:
                logger.warning("No stocks passed filtering. Exiting strategy.")
                return results
            
            # Step 2: Risk Strategy Allocation
            allocation_result = self.allocate_risk_strategy(filtered_stocks, risk_preference)
            selected_stocks = allocation_result['selected']
            results['steps']['risk_allocation'] = {
                'large_cap_count': len(allocation_result['large_cap']),
                'mid_cap_count': len(allocation_result['mid_cap']),
                'small_cap_count': len(allocation_result['small_cap']),
                'selected_count': len(selected_stocks),
                'strategy': risk_preference.value
            }
            
            # Step 3: Entry Rules
            entry_signals = self.check_entry_signals(selected_stocks)
            results['steps']['entry_signals'] = {
                'checked_count': len(selected_stocks),
                'signals_found': len(entry_signals),
                'rules': self.entry_rules.to_dict()
            }
            
            # Execute entries for valid signals
            if entry_signals:
                self._execute_entries(entry_signals)
                results['steps']['entries_executed'] = len([s for s in entry_signals if s.symbol in self.positions])
            
            # Step 4: Exit Rules (for existing positions)
            self.manage_exits()
            results['steps']['exit_management'] = {
                'active_positions': len(self.positions),
                'closed_positions': len(self.closed_positions),
                'rules': self.exit_rules.to_dict()
            }
            
            # Generate performance report
            results['performance'] = self.get_performance_metrics()
            
        except Exception as e:
            logger.error(f"Strategy execution error: {e}")
            results['error'] = str(e)
        
        logger.info("="*60)
        logger.info("STRATEGY EXECUTION COMPLETE")
        logger.info("="*60)
        
        return results
    
    def _execute_entries(self, entry_signals: List[StockSignal]):
        """Execute entry orders for valid signals."""
        max_positions = 10  # Maximum concurrent positions
        position_size_percent = 0.1  # 10% of capital per position
        
        for signal in entry_signals:
            if len(self.positions) >= max_positions:
                logger.warning("Maximum positions reached. Skipping remaining signals.")
                break
            
            # Calculate position size
            position_value = self.available_capital * position_size_percent
            quantity = int(position_value / signal.current_price)
            
            if quantity > 0 and position_value <= self.available_capital:
                success = self._execute_entry(signal, quantity)
                if success:
                    logger.info(f"✓ Entry executed: {signal.symbol} - {quantity} shares @ ₹{signal.current_price:.2f}")
                else:
                    logger.error(f"✗ Entry failed for {signal.symbol}")
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        total_value = self.available_capital
        unrealized_pnl = 0.0
        
        for position in self.positions.values():
            total_value += (position.current_price * position.remaining_quantity)
            unrealized_pnl += position.unrealized_pnl
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'total_value': total_value,
            'available_capital': self.available_capital,
            'total_realized_pnl': self.total_realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': self.total_realized_pnl + unrealized_pnl,
            'return_percent': ((total_value - self.capital) / self.capital * 100),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'active_positions': len(self.positions)
        }
    
    def run_continuous_monitoring(self, interval_seconds: int = 60):
        """
        Run continuous monitoring for exit management.
        
        Args:
            interval_seconds: Check interval in seconds
        """
        logger.info(f"Starting continuous monitoring (interval: {interval_seconds}s)...")
        
        try:
            while True:
                if self.positions:
                    self.manage_exits()
                    self.display_portfolio_status()
                else:
                    logger.info("No active positions to monitor.")
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user.")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    def display_portfolio_status(self):
        """Display current portfolio status."""
        metrics = self.get_performance_metrics()
        
        print("\n" + "="*60)
        print("PORTFOLIO STATUS")
        print("="*60)
        print(f"Total Value: ₹{metrics['total_value']:,.2f}")
        print(f"Available Capital: ₹{metrics['available_capital']:,.2f}")
        print(f"Total P&L: ₹{metrics['total_pnl']:,.2f} ({metrics['return_percent']:.2f}%)")
        print(f"Realized P&L: ₹{metrics['total_realized_pnl']:,.2f}")
        print(f"Unrealized P&L: ₹{metrics['unrealized_pnl']:,.2f}")
        print(f"Win Rate: {metrics['win_rate']:.1f}% ({metrics['winning_trades']}/{metrics['total_trades']})")
        print(f"Active Positions: {metrics['active_positions']}")
        
        if self.positions:
            print("\n" + "-"*60)
            print("ACTIVE POSITIONS:")
            print("-"*60)
            for symbol, pos in self.positions.items():
                print(f"{symbol}: {pos.remaining_quantity} shares @ ₹{pos.entry_price:.2f}")
                print(f"  Current: ₹{pos.current_price:.2f} | P&L: ₹{pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_percent:.2f}%)")
                print(f"  Days Held: {pos.days_held} | Status: {pos.status}")
                if pos.trailing_stop_price > 0:
                    print(f"  Trailing Stop: ₹{pos.trailing_stop_price:.2f}")
        
        print("="*60 + "\n")


# ============= Main Execution =============

if __name__ == "__main__":
    # Example usage
    strategy = EnhancedPortfolioStrategy(capital=100000)
    
    # Define stock universe (these should come from your broker or database)
    stock_universe = [
        "NSE:RELIANCE-EQ",
        "NSE:TCS-EQ",
        "NSE:INFY-EQ",
        "NSE:HDFCBANK-EQ",
        "NSE:ICICIBANK-EQ",
        # Add more stocks as needed
    ]
    
    # Run the strategy
    results = strategy.run_strategy(stock_universe, RiskStrategy.SAFE)
    
    # Start continuous monitoring
    strategy.run_continuous_monitoring(interval_seconds=60)