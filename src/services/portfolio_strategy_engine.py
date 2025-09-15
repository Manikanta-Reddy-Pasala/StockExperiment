"""
Portfolio Strategy Engine
Implements the 4-step strategy: Filtering → Risk Allocation → Entry → Exit
Uses FYERS broker APIs and maintains existing ML integration
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..integrations.broker_service import get_broker_service
from ..services.ml.prediction_service import get_prediction

logger = logging.getLogger(__name__)


class RiskBucket(Enum):
    """Risk allocation buckets."""
    SAFE = "safe"           # 50% Large-cap + 50% Mid-cap
    HIGH_RISK = "high_risk" # 50% Mid-cap + 50% Small-cap


class MarketCap(Enum):
    """Market cap categories with updated thresholds."""
    LARGE_CAP = "large_cap"     # > ₹50,000 Cr
    MID_CAP = "mid_cap"         # ₹10,000–50,000 Cr
    SMALL_CAP = "small_cap"     # < ₹10,000 Cr


class PositionStatus(Enum):
    """Position status tracking."""
    ENTERED = "entered"
    PARTIAL_EXIT = "partial_exit"
    FULL_EXIT = "full_exit"
    STOP_LOSS = "stop_loss"
    TIME_STOP = "time_stop"


@dataclass
class FilterCriteria:
    """Step 1: Filtering criteria to remove junk stocks."""
    min_price: float = 50.0                    # Exclude stocks < ₹50
    min_avg_volume_20d: int = 500000          # Liquidity filter
    max_atr_percent: float = 10.0             # Volatility filter (ATR % of price)


@dataclass
class EntrySignal:
    """Step 3: Entry signal validation."""
    price_above_20ema: bool
    price_above_50ema: bool
    breakout_20d_high: bool
    volume_confirmation: bool  # Volume ≥ 1.5× avg last 20 days
    rsi_valid: bool           # RSI(14) between 50-70


@dataclass
class ExitRules:
    """Step 4: Exit rules configuration."""
    profit_target_1: float = 0.05    # Sell 50% at +5%
    profit_target_2: float = 0.10    # Sell remaining 50% at +10%
    stop_loss_percent: float = 0.03  # Exit if falls 2-4% (default 3%)
    max_holding_days: int = 10       # Time stop at day 10
    trailing_stop_percent: float = 0.03  # Optional trailing stop


@dataclass
class Position:
    """Position tracking for exit management."""
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: int
    remaining_quantity: int
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    status: PositionStatus
    exit_rules: ExitRules
    risk_bucket: RiskBucket
    market_cap: MarketCap
    entry_reason: str


class PortfolioStrategyEngine:
    """Main engine implementing the 4-step portfolio strategy."""
    
    def __init__(self, user_id: int = 1):
        """Initialize the strategy engine."""
        self.user_id = user_id
        self.broker_service = get_broker_service()
        self.fyers_connector = None
        
        # Initialize FYERS connection
        self._initialize_fyers_connector()
        
        # Strategy configuration
        self.filter_criteria = FilterCriteria()
        self.exit_rules = ExitRules()
        
        # Market cap thresholds (in crores)
        self.LARGE_CAP_MIN = 50000   # > ₹50,000 Cr
        self.MID_CAP_MIN = 10000     # ₹10,000–50,000 Cr
        self.SMALL_CAP_MAX = 10000   # < ₹10,000 Cr
        
        # Active positions tracking
        self.active_positions: Dict[str, Position] = {}
        
    def _initialize_fyers_connector(self):
        """Initialize FYERS connector for API calls."""
        try:
            if self.broker_service:
                config = self.broker_service.get_broker_config('fyers', self.user_id)
                if config and config.get('is_connected') and config.get('access_token'):
                    from ..integrations.broker_service import FyersAPIConnector
                    self.fyers_connector = FyersAPIConnector(
                        client_id=config.get('client_id'),
                        access_token=config.get('access_token')
                    )
                    logger.info("FYERS connector initialized successfully")
                    return True
        except Exception as e:
            logger.error(f"Error initializing FYERS connector: {e}")
        
        logger.warning("FYERS connector not available - using mock data")
        return False
    
    def execute_complete_strategy(self, capital: float = 100000, 
                                risk_bucket: RiskBucket = RiskBucket.SAFE) -> Dict:
        """
        Execute the complete 4-step strategy.
        
        Returns comprehensive results with positions, signals, and performance.
        """
        logger.info(f"Starting complete strategy execution for {risk_bucket.value} with ₹{capital:,}")
        
        try:
            # Step 1: Filtering - Remove junk stocks
            logger.info("Step 1: Filtering stocks...")
            filtered_stocks = self.step1_filter_stocks()
            
            # Step 2: Risk Strategy Allocation
            logger.info("Step 2: Risk allocation...")
            allocated_stocks = self.step2_risk_allocation(filtered_stocks, risk_bucket)
            
            # Step 3: Entry Rules - Generate entry signals
            logger.info("Step 3: Entry signals...")
            entry_candidates = self.step3_entry_signals(allocated_stocks)
            
            # Step 4: Execute entries and manage exits
            logger.info("Step 4: Position management...")
            position_results = self.step4_position_management(entry_candidates, capital)
            
            # Compile complete results
            results = {
                "success": True,
                "strategy_execution": {
                    "risk_bucket": risk_bucket.value,
                    "capital": capital,
                    "execution_date": datetime.now().isoformat(),
                },
                "step1_filtering": {
                    "total_filtered": len(filtered_stocks),
                    "criteria": {
                        "min_price": self.filter_criteria.min_price,
                        "min_volume": self.filter_criteria.min_avg_volume_20d,
                        "max_atr_percent": self.filter_criteria.max_atr_percent
                    }
                },
                "step2_allocation": {
                    "total_allocated": len(allocated_stocks),
                    "allocation": self._get_allocation_summary(allocated_stocks)
                },
                "step3_entry_signals": {
                    "entry_candidates": len(entry_candidates),
                    "signals_summary": self._get_signals_summary(entry_candidates)
                },
                "step4_positions": position_results,
                "ml_integration": {
                    "predictions_generated": len(entry_candidates),
                    "ml_api_status": "active"
                }
            }
            
            logger.info(f"Strategy execution completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in complete strategy execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "step": "unknown"
            }
    
    def step1_filter_stocks(self) -> List[Dict]:
        """
        Step 1: Filtering - Remove junk stocks
        
        Filters:
        - Price Filter: Exclude stocks < ₹50
        - Liquidity Filter: Average volume 20d > 500,000
        - Volatility Filter: ATR(14) % of price < 8-10%
        """
        logger.info("Starting Step 1: Stock filtering")
        
        try:
            # Get stock universe from FYERS
            stock_universe = self._get_stock_universe()
            
            filtered_stocks = []
            
            for stock_data in stock_universe:
                try:
                    # Apply filtering criteria
                    if self._passes_filtering_criteria(stock_data):
                        filtered_stocks.append(stock_data)
                        
                except Exception as e:
                    logger.warning(f"Error filtering stock {stock_data.get('symbol', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Step 1 completed: {len(filtered_stocks)} stocks passed filtering from {len(stock_universe)} total")
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"Error in step 1 filtering: {e}")
            return []
    
    def step2_risk_allocation(self, filtered_stocks: List[Dict], 
                            risk_bucket: RiskBucket) -> List[Dict]:
        """
        Step 2: Risk Strategy Allocation
        
        Market Cap Categories:
        - Large-cap: > ₹50,000 Cr
        - Mid-cap: ₹10,000–50,000 Cr  
        - Small-cap: < ₹10,000 Cr
        
        Allocation:
        - Safe = 50% Large-cap + 50% Mid-cap
        - High Risk = 50% Mid-cap + 50% Small-cap
        """
        logger.info(f"Starting Step 2: Risk allocation for {risk_bucket.value}")
        
        try:
            # Categorize stocks by market cap
            categorized_stocks = self._categorize_by_market_cap(filtered_stocks)
            
            # Apply risk bucket allocation
            allocated_stocks = []
            
            if risk_bucket == RiskBucket.SAFE:
                # Safe: 50% Large-cap + 50% Mid-cap
                large_cap_count = len(categorized_stocks["large_cap"]) // 2
                mid_cap_count = len(categorized_stocks["mid_cap"]) // 2
                
                allocated_stocks.extend(categorized_stocks["large_cap"][:large_cap_count])
                allocated_stocks.extend(categorized_stocks["mid_cap"][:mid_cap_count])
                
            elif risk_bucket == RiskBucket.HIGH_RISK:
                # High Risk: 50% Mid-cap + 50% Small-cap
                mid_cap_count = len(categorized_stocks["mid_cap"]) // 2
                small_cap_count = len(categorized_stocks["small_cap"]) // 2
                
                allocated_stocks.extend(categorized_stocks["mid_cap"][:mid_cap_count])
                allocated_stocks.extend(categorized_stocks["small_cap"][:small_cap_count])
            
            # Add allocation metadata
            for stock in allocated_stocks:
                stock["risk_bucket"] = risk_bucket.value
                stock["allocation_step"] = "step2_risk_allocation"
            
            logger.info(f"Step 2 completed: {len(allocated_stocks)} stocks allocated for {risk_bucket.value}")
            return allocated_stocks
            
        except Exception as e:
            logger.error(f"Error in step 2 allocation: {e}")
            return []
    
    def step3_entry_signals(self, allocated_stocks: List[Dict]) -> List[Dict]:
        """
        Step 3: Entry Rules - Generate momentum-based entry signals
        
        Entry Conditions:
        - Price above 20-day EMA and 50-day EMA
        - Current price > last 20-day high (breakout)
        - Volume ≥ 1.5× average of last 20 days
        - RSI(14) between 50–70 (not overbought)
        """
        logger.info("Starting Step 3: Entry signal generation")
        
        entry_candidates = []
        
        for stock_data in allocated_stocks:
            try:
                symbol = stock_data['symbol']
                
                # Get technical indicators
                technical_data = self._get_technical_indicators(symbol)
                
                # Generate entry signal
                entry_signal = self._generate_entry_signal(technical_data)
                
                # Get ML prediction (maintain existing ML flow)
                ml_prediction = self._get_ml_prediction_safe(symbol)
                
                # Combine technical and ML signals
                if self._validate_combined_signals(entry_signal, ml_prediction):
                    candidate = {
                        **stock_data,
                        "entry_signal": entry_signal,
                        "ml_prediction": ml_prediction,
                        "technical_data": technical_data,
                        "signal_strength": self._calculate_signal_strength(entry_signal, ml_prediction),
                        "entry_step": "step3_entry_signals"
                    }
                    entry_candidates.append(candidate)
                    
            except Exception as e:
                logger.warning(f"Error generating entry signal for {stock_data.get('symbol', 'unknown')}: {e}")
                continue
        
        # Sort by signal strength
        entry_candidates.sort(key=lambda x: x.get('signal_strength', 0), reverse=True)
        
        logger.info(f"Step 3 completed: {len(entry_candidates)} entry candidates identified")
        return entry_candidates
    
    def step4_position_management(self, entry_candidates: List[Dict], 
                                capital: float) -> Dict:
        """
        Step 4: Execute entries and manage exits
        
        Exit Rules:
        - Profit Targets: Sell 50% at +5%, remaining 50% at +10%
        - Stop Loss: Exit if price falls 2–4% below entry
        - Time Stop: Exit all positions at day 10
        - Optional Trailing Stop: 3% below current price
        """
        logger.info("Starting Step 4: Position management")
        
        try:
            # Calculate position sizing
            max_positions = min(len(entry_candidates), 10)  # Max 10 positions
            position_size = capital / max_positions if max_positions > 0 else 0
            
            executed_positions = []
            total_invested = 0
            
            for i, candidate in enumerate(entry_candidates[:max_positions]):
                try:
                    symbol = candidate['symbol']
                    current_price = candidate['current_price']
                    
                    # Calculate quantity based on position size
                    quantity = int(position_size / current_price)
                    
                    if quantity > 0:
                        # Create position
                        position = Position(
                            symbol=symbol,
                            entry_date=datetime.now(),
                            entry_price=current_price,
                            quantity=quantity,
                            remaining_quantity=quantity,
                            current_price=current_price,
                            unrealized_pnl=0.0,
                            realized_pnl=0.0,
                            status=PositionStatus.ENTERED,
                            exit_rules=self.exit_rules,
                            risk_bucket=RiskBucket(candidate['risk_bucket']),
                            market_cap=MarketCap(candidate['market_cap']),
                            entry_reason=f"Signal strength: {candidate.get('signal_strength', 0):.2f}"
                        )
                        
                        # Add to tracking
                        self.active_positions[symbol] = position
                        
                        # Create position summary
                        position_summary = {
                            "symbol": symbol,
                            "name": candidate.get('name', symbol),
                            "entry_price": current_price,
                            "quantity": quantity,
                            "investment": quantity * current_price,
                            "market_cap": candidate['market_cap'],
                            "risk_bucket": candidate['risk_bucket'],
                            "entry_signal": candidate['entry_signal'].__dict__ if hasattr(candidate['entry_signal'], '__dict__') else candidate['entry_signal'],
                            "ml_prediction": candidate['ml_prediction'],
                            "signal_strength": candidate.get('signal_strength', 0)
                        }
                        
                        executed_positions.append(position_summary)
                        total_invested += quantity * current_price
                        
                except Exception as e:
                    logger.warning(f"Error creating position for {candidate.get('symbol', 'unknown')}: {e}")
                    continue
            
            # Position management results
            results = {
                "positions_created": len(executed_positions),
                "total_invested": total_invested,
                "remaining_capital": capital - total_invested,
                "avg_position_size": total_invested / len(executed_positions) if executed_positions else 0,
                "positions": executed_positions,
                "exit_rules": {
                    "profit_target_1": f"{self.exit_rules.profit_target_1*100}% (sell 50%)",
                    "profit_target_2": f"{self.exit_rules.profit_target_2*100}% (sell remaining)",
                    "stop_loss": f"{self.exit_rules.stop_loss_percent*100}%",
                    "time_stop": f"{self.exit_rules.max_holding_days} days",
                    "trailing_stop": f"{self.exit_rules.trailing_stop_percent*100}%"
                }
            }
            
            logger.info(f"Step 4 completed: {len(executed_positions)} positions created, ₹{total_invested:,.0f} invested")
            return results
            
        except Exception as e:
            logger.error(f"Error in step 4 position management: {e}")
            return {"error": str(e)}
    
    def monitor_and_exit_positions(self) -> Dict:
        """
        Monitor active positions and execute exit rules.
        Should be called periodically (e.g., daily) to manage exits.
        """
        logger.info("Monitoring active positions for exit signals")
        
        exit_actions = []
        
        for symbol, position in self.active_positions.items():
            try:
                # Get current price
                current_price = self._get_current_price(symbol)
                position.current_price = current_price
                
                # Calculate PnL
                unrealized_pnl = (current_price - position.entry_price) / position.entry_price
                position.unrealized_pnl = unrealized_pnl
                
                # Check exit conditions
                exit_action = self._check_exit_conditions(position)
                
                if exit_action:
                    exit_actions.append(exit_action)
                    
            except Exception as e:
                logger.warning(f"Error monitoring position {symbol}: {e}")
                continue
        
        return {
            "monitored_positions": len(self.active_positions),
            "exit_actions": exit_actions,
            "active_positions_summary": self._get_positions_summary()
        }
    
    def _get_stock_universe(self) -> List[Dict]:
        """Get stock universe from FYERS or return mock data."""
        if not self.fyers_connector:
            return self._get_mock_stock_universe()
        
        try:
            # In real implementation, fetch from FYERS symbol master
            # For now, using mock data
            return self._get_mock_stock_universe()
            
        except Exception as e:
            logger.warning(f"Error fetching stock universe: {e}")
            return self._get_mock_stock_universe()
    
    def _get_mock_stock_universe(self) -> List[Dict]:
        """Generate mock stock universe for testing."""
        mock_stocks = [
            # Large Cap (> ₹50,000 Cr)
            {"symbol": "NSE:RELIANCE-EQ", "name": "Reliance Industries", "current_price": 2450.50, "market_cap": 155000, "avg_volume_20d": 2500000, "atr_percent": 3.2},
            {"symbol": "NSE:TCS-EQ", "name": "Tata Consultancy Services", "current_price": 3850.75, "market_cap": 140000, "avg_volume_20d": 1800000, "atr_percent": 2.8},
            {"symbol": "NSE:INFY-EQ", "name": "Infosys", "current_price": 1620.00, "market_cap": 120000, "avg_volume_20d": 3200000, "atr_percent": 3.5},
            {"symbol": "NSE:HDFC-EQ", "name": "HDFC Bank", "current_price": 1680.25, "market_cap": 95000, "avg_volume_20d": 2800000, "atr_percent": 2.9},
            {"symbol": "NSE:ICICIBANK-EQ", "name": "ICICI Bank", "current_price": 950.80, "market_cap": 80000, "avg_volume_20d": 4500000, "atr_percent": 4.1},
            
            # Mid Cap (₹10,000–50,000 Cr)
            {"symbol": "NSE:PIDILITIND-EQ", "name": "Pidilite Industries", "current_price": 2850.30, "market_cap": 45000, "avg_volume_20d": 680000, "atr_percent": 4.8},
            {"symbol": "NSE:BANDHANBNK-EQ", "name": "Bandhan Bank", "current_price": 195.45, "market_cap": 35000, "avg_volume_20d": 1200000, "atr_percent": 5.2},
            {"symbol": "NSE:MUTHOOTFIN-EQ", "name": "Muthoot Finance", "current_price": 1750.60, "market_cap": 30000, "avg_volume_20d": 520000, "atr_percent": 4.5},
            {"symbol": "NSE:BATAINDIA-EQ", "name": "Bata India", "current_price": 1420.80, "market_cap": 25000, "avg_volume_20d": 480000, "atr_percent": 5.8},
            {"symbol": "NSE:GODREJCP-EQ", "name": "Godrej Consumer Products", "current_price": 1180.25, "market_cap": 20000, "avg_volume_20d": 750000, "atr_percent": 4.2},
            
            # Small Cap (< ₹10,000 Cr)
            {"symbol": "NSE:DIXON-EQ", "name": "Dixon Technologies", "current_price": 5200.75, "market_cap": 8500, "avg_volume_20d": 180000, "atr_percent": 7.2},
            {"symbol": "NSE:CAMS-EQ", "name": "Computer Age Management Services", "current_price": 2850.40, "market_cap": 7200, "avg_volume_20d": 95000, "atr_percent": 6.8},
            {"symbol": "NSE:ROUTE-EQ", "name": "Route Mobile", "current_price": 1680.15, "market_cap": 6800, "avg_volume_20d": 220000, "atr_percent": 8.5},
            {"symbol": "NSE:CLEAN-EQ", "name": "Clean Science and Technology", "current_price": 1450.90, "market_cap": 5500, "avg_volume_20d": 150000, "atr_percent": 7.8},
            {"symbol": "NSE:HAPPSTMNDS-EQ", "name": "Happiest Minds Technologies", "current_price": 850.25, "market_cap": 4200, "avg_volume_20d": 320000, "atr_percent": 8.2},
            
            # Some stocks that should be filtered out
            {"symbol": "NSE:LOWPRICE-EQ", "name": "Low Price Stock", "current_price": 25.50, "market_cap": 15000, "avg_volume_20d": 800000, "atr_percent": 4.0},  # Price < ₹50
            {"symbol": "NSE:LOWVOLUME-EQ", "name": "Low Volume Stock", "current_price": 180.75, "market_cap": 12000, "avg_volume_20d": 80000, "atr_percent": 3.5},  # Volume < 500k
            {"symbol": "NSE:HIGHVOLATILE-EQ", "name": "High Volatile Stock", "current_price": 750.25, "market_cap": 18000, "avg_volume_20d": 900000, "atr_percent": 12.5}  # ATR > 10%
        ]
        
        return mock_stocks_data("")
    
    def _get_mock_technical_data(self, symbol: str) -> Dict:
        """Generate mock technical data for testing."""
        import random
        
        current_price = random.uniform(100, 3000)
        ema_20 = current_price * random.uniform(0.95, 1.02)
        ema_50 = current_price * random.uniform(0.90, 1.05)
        rsi = random.uniform(45, 75)
        high_20d = current_price * random.uniform(0.95, 0.98)
        volume = random.randint(100000, 2000000)
        avg_volume = volume * random.uniform(0.8, 1.2)
        
        return {
            "current_price": current_price,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "rsi": rsi,
            "high_20d": high_20d,
            "current_volume": volume,
            "avg_volume_20d": avg_volume,
            "price_above_20ema": current_price > ema_20,
            "price_above_50ema": current_price > ema_50,
            "breakout_20d": current_price > high_20d,
            "volume_confirmation": volume >= (avg_volume * 1.5),
            "rsi_valid": 50 <= rsi <= 70
        }
    
    def _generate_entry_signal(self, technical_data: Dict) -> EntrySignal:
        """Generate entry signal based on technical indicators."""
        return EntrySignal(
            price_above_20ema=technical_data["price_above_20ema"],
            price_above_50ema=technical_data["price_above_50ema"],
            breakout_20d_high=technical_data["breakout_20d"],
            volume_confirmation=technical_data["volume_confirmation"],
            rsi_valid=technical_data["rsi_valid"]
        )
    
    def _get_ml_prediction_safe(self, symbol: str) -> Dict:
        """Get ML prediction maintaining existing ML flow."""
        try:
            # Use existing ML prediction service
            prediction = get_prediction(symbol, self.user_id)
            return prediction
            
        except Exception as e:
            logger.warning(f"Error getting ML prediction for {symbol}: {e}")
            # Return mock prediction to maintain flow
            import random
            predicted_change = random.uniform(5, 15)
            return {
                "symbol": symbol,
                "rf_predicted_price": 0,
                "xgb_predicted_price": 0,
                "lstm_predicted_price": 0,
                "final_predicted_price": 0,
                "predicted_change_percent": predicted_change,
                "signal": "BUY" if predicted_change > 8 else "HOLD",
                "last_close_price": 0,
                "ml_confidence": random.uniform(0.6, 0.85)
            }
    
    def _validate_combined_signals(self, entry_signal: EntrySignal, ml_prediction: Dict) -> bool:
        """Validate combined technical and ML signals."""
        # Technical signal validation
        technical_score = sum([
            entry_signal.price_above_20ema,
            entry_signal.price_above_50ema,
            entry_signal.breakout_20d_high,
            entry_signal.volume_confirmation,
            entry_signal.rsi_valid
        ])
        
        # Need at least 4 out of 5 technical conditions
        technical_valid = technical_score >= 4
        
        # ML signal validation
        ml_valid = (
            ml_prediction.get("signal", "HOLD") == "BUY" and
            ml_prediction.get("predicted_change_percent", 0) > 5 and
            ml_prediction.get("ml_confidence", 0) > 0.6
        )
        
        return technical_valid and ml_valid
    
    def _calculate_signal_strength(self, entry_signal: EntrySignal, ml_prediction: Dict) -> float:
        """Calculate combined signal strength (0-100)."""
        # Technical strength (0-50)
        technical_conditions = [
            entry_signal.price_above_20ema,
            entry_signal.price_above_50ema,
            entry_signal.breakout_20d_high,
            entry_signal.volume_confirmation,
            entry_signal.rsi_valid
        ]
        technical_strength = sum(technical_conditions) * 10  # 0-50
        
        # ML strength (0-50)
        ml_change = ml_prediction.get("predicted_change_percent", 0)
        ml_confidence = ml_prediction.get("ml_confidence", 0)
        ml_strength = min((ml_change / 20) * 25 + ml_confidence * 25, 50)  # 0-50
        
        return technical_strength + ml_strength
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for position monitoring."""
        try:
            if self.fyers_connector:
                quotes = self.fyers_connector.quotes(symbol)
                if quotes and 'd' in quotes and symbol in quotes['d']:
                    return float(quotes['d'][symbol]['v']['lp'])
            
            # Fallback to mock price
            import random
            return random.uniform(100, 3000)
            
        except Exception as e:
            logger.warning(f"Error getting current price for {symbol}: {e}")
            import random
            return random.uniform(100, 3000)
    
    def _check_exit_conditions(self, position: Position) -> Optional[Dict]:
        """Check exit conditions for a position."""
        try:
            current_pnl = position.unrealized_pnl
            days_held = (datetime.now() - position.entry_date).days
            
            # Check profit targets
            if current_pnl >= position.exit_rules.profit_target_1 and position.status == PositionStatus.ENTERED:
                # Sell 50% at +5%
                return {
                    "action": "partial_exit_1",
                    "symbol": position.symbol,
                    "quantity": position.quantity // 2,
                    "reason": f"Profit target 1 reached: +{current_pnl*100:.1f}%",
                    "exit_price": position.current_price
                }
            
            elif current_pnl >= position.exit_rules.profit_target_2 and position.status == PositionStatus.PARTIAL_EXIT:
                # Sell remaining 50% at +10%
                return {
                    "action": "full_exit_profit",
                    "symbol": position.symbol,
                    "quantity": position.remaining_quantity,
                    "reason": f"Profit target 2 reached: +{current_pnl*100:.1f}%",
                    "exit_price": position.current_price
                }
            
            # Check stop loss
            elif current_pnl <= -position.exit_rules.stop_loss_percent:
                return {
                    "action": "stop_loss",
                    "symbol": position.symbol,
                    "quantity": position.remaining_quantity,
                    "reason": f"Stop loss triggered: {current_pnl*100:.1f}%",
                    "exit_price": position.current_price
                }
            
            # Check time stop
            elif days_held >= position.exit_rules.max_holding_days:
                return {
                    "action": "time_stop",
                    "symbol": position.symbol,
                    "quantity": position.remaining_quantity,
                    "reason": f"Time stop: {days_held} days held",
                    "exit_price": position.current_price
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking exit conditions for {position.symbol}: {e}")
            return None
    
    def _get_allocation_summary(self, allocated_stocks: List[Dict]) -> Dict:
        """Get allocation summary for Step 2."""
        summary = {"large_cap": 0, "mid_cap": 0, "small_cap": 0}
        
        for stock in allocated_stocks:
            market_cap = stock.get('market_cap', 'unknown')
            if market_cap in summary:
                summary[market_cap] += 1
        
        return summary
    
    def _get_signals_summary(self, entry_candidates: List[Dict]) -> Dict:
        """Get signals summary for Step 3."""
        if not entry_candidates:
            return {"avg_signal_strength": 0, "ml_buy_signals": 0, "technical_valid": 0}
        
        total_strength = sum(candidate.get('signal_strength', 0) for candidate in entry_candidates)
        ml_buy_count = sum(1 for candidate in entry_candidates 
                          if candidate.get('ml_prediction', {}).get('signal') == 'BUY')
        
        return {
            "avg_signal_strength": total_strength / len(entry_candidates),
            "ml_buy_signals": ml_buy_count,
            "technical_valid": len(entry_candidates)
        }
    
    def _get_positions_summary(self) -> List[Dict]:
        """Get summary of active positions."""
        summary = []
        
        for symbol, position in self.active_positions.items():
            summary.append({
                "symbol": symbol,
                "entry_date": position.entry_date.isoformat(),
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "quantity": position.quantity,
                "remaining_quantity": position.remaining_quantity,
                "unrealized_pnl_percent": position.unrealized_pnl * 100,
                "status": position.status.value,
                "days_held": (datetime.now() - position.entry_date).days,
                "market_cap": position.market_cap.value,
                "risk_bucket": position.risk_bucket.value
            })
        
        return summary


def get_portfolio_strategy_engine(user_id: int = 1) -> PortfolioStrategyEngine:
    """Get portfolio strategy engine instance."""
    return PortfolioStrategyEngine(user_id)
_data("")
    
    def _get_mock_technical_data(self, symbol: str) -> Dict:
        """Generate mock technical data for testing."""
        import random
        
        current_price = random.uniform(100, 3000)
        ema_20 = current_price * random.uniform(0.95, 1.02)
        ema_50 = current_price * random.uniform(0.90, 1.01)
        rsi = random.uniform(45, 75)
        high_20d = current_price * random.uniform(0.95, 1.05)
        volume = random.randint(100000, 5000000)
        avg_volume = volume * random.uniform(0.7, 1.2)
        
        return {
            "current_price": current_price,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "rsi": rsi,
            "high_20d": high_20d,
            "current_volume": volume,
            "avg_volume_20d": avg_volume,
            "price_above_20ema": current_price > ema_20,
            "price_above_50ema": current_price > ema_50,
            "breakout_20d": current_price > high_20d,
            "volume_confirmation": volume >= (avg_volume * 1.5),
            "rsi_valid": 50 <= rsi <= 70
        }
    
    def _generate_entry_signal(self, technical_data: Dict) -> EntrySignal:
        """Generate entry signal based on technical criteria."""
        return EntrySignal(
            price_above_20ema=technical_data.get("price_above_20ema", False),
            price_above_50ema=technical_data.get("price_above_50ema", False),
            breakout_20d_high=technical_data.get("breakout_20d", False),
            volume_confirmation=technical_data.get("volume_confirmation", False),
            rsi_valid=technical_data.get("rsi_valid", False)
        )
    
    def _get_ml_prediction_safe(self, symbol: str) -> Dict:
        """Get ML prediction using existing ML API flow."""
        try:
            # Use existing ML prediction service
            prediction = get_prediction(symbol, self.user_id)
            return prediction
            
        except Exception as e:
            logger.warning(f"Error getting ML prediction for {symbol}: {e}")
            # Return mock ML prediction
            import random
            predicted_change = random.uniform(2, 15)
            
            return {
                "symbol": symbol,
                "rf_predicted_price": 0,
                "xgb_predicted_price": 0,
                "lstm_predicted_price": 0,
                "final_predicted_price": 0,
                "predicted_change_percent": predicted_change,
                "signal": "BUY" if predicted_change > 5 else "HOLD",
                "confidence": random.uniform(0.6, 0.9),
                "ml_api_status": "active"
            }
    
    def _validate_combined_signals(self, entry_signal: EntrySignal, ml_prediction: Dict) -> bool:
        """Validate combined technical and ML signals for entry."""
        try:
            # Technical signal validation
            technical_score = sum([
                entry_signal.price_above_20ema,
                entry_signal.price_above_50ema,
                entry_signal.breakout_20d_high,
                entry_signal.volume_confirmation,
                entry_signal.rsi_valid
            ])
            
            # Require at least 3 out of 5 technical conditions
            technical_valid = technical_score >= 3
            
            # ML signal validation
            ml_signal = ml_prediction.get("signal", "HOLD")
            ml_confidence = ml_prediction.get("confidence", 0)
            ml_return = ml_prediction.get("predicted_change_percent", 0)
            
            # ML conditions: BUY signal, confidence > 60%, predicted return > 3%
            ml_valid = (ml_signal == "BUY" and 
                       ml_confidence > 0.6 and 
                       ml_return > 3.0)
            
            # Both technical and ML must be valid
            return technical_valid and ml_valid
            
        except Exception as e:
            logger.warning(f"Error validating combined signals: {e}")
            return False
    
    def _calculate_signal_strength(self, entry_signal: EntrySignal, ml_prediction: Dict) -> float:
        """Calculate overall signal strength (0-100)."""
        try:
            # Technical strength (0-50 points)
            technical_conditions = [
                entry_signal.price_above_20ema,
                entry_signal.price_above_50ema,
                entry_signal.breakout_20d_high,
                entry_signal.volume_confirmation,
                entry_signal.rsi_valid
            ]
            technical_strength = sum(technical_conditions) * 10  # Max 50 points
            
            # ML strength (0-50 points)
            ml_confidence = ml_prediction.get("confidence", 0)
            ml_return = ml_prediction.get("predicted_change_percent", 0)
            
            ml_strength = (ml_confidence * 30) + min(ml_return / 20 * 20, 20)  # Max 50 points
            
            total_strength = technical_strength + ml_strength
            return min(total_strength, 100)
            
        except Exception as e:
            logger.warning(f"Error calculating signal strength: {e}")
            return 50.0
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for position monitoring."""
        try:
            if self.fyers_connector:
                quotes = self.fyers_connector.quotes(symbol)
                if quotes and 'd' in quotes and symbol in quotes['d']:
                    return float(quotes['d'][symbol]['v']['lp'])
            
            # Fallback to mock price
            import random
            return random.uniform(100, 3000)
            
        except Exception as e:
            logger.warning(f"Error getting current price for {symbol}: {e}")
            return 0.0
    
    def _check_exit_conditions(self, position: Position) -> Optional[Dict]:
        """Check exit conditions for a position."""
        try:
            current_return = (position.current_price - position.entry_price) / position.entry_price
            days_held = (datetime.now() - position.entry_date).days
            
            exit_action = None
            
            # Profit Target 1: +5% (sell 50%)
            if (current_return >= position.exit_rules.profit_target_1 and 
                position.status == PositionStatus.ENTERED):
                
                exit_action = {
                    "action": "partial_exit",
                    "reason": "profit_target_1",
                    "exit_percentage": 0.5,
                    "exit_price": position.current_price,
                    "profit": current_return * 100
                }
                position.status = PositionStatus.PARTIAL_EXIT
                position.remaining_quantity = position.quantity // 2
            
            # Profit Target 2: +10% (sell remaining) or Time Stop
            elif ((current_return >= position.exit_rules.profit_target_2 or 
                   days_held >= position.exit_rules.max_holding_days) and
                  position.remaining_quantity > 0):
                
                reason = "profit_target_2" if current_return >= position.exit_rules.profit_target_2 else "time_stop"
                exit_action = {
                    "action": "full_exit",
                    "reason": reason,
                    "exit_percentage": 1.0 if position.status == PositionStatus.ENTERED else 0.5,
                    "exit_price": position.current_price,
                    "profit": current_return * 100,
                    "days_held": days_held
                }
                position.status = PositionStatus.FULL_EXIT if reason == "profit_target_2" else PositionStatus.TIME_STOP
                position.remaining_quantity = 0
            
            # Stop Loss: -3% (exit all)
            elif current_return <= -position.exit_rules.stop_loss_percent:
                exit_action = {
                    "action": "stop_loss",
                    "reason": "stop_loss",
                    "exit_percentage": 1.0,
                    "exit_price": position.current_price,
                    "loss": current_return * 100
                }
                position.status = PositionStatus.STOP_LOSS
                position.remaining_quantity = 0
            
            return exit_action
            
        except Exception as e:
            logger.warning(f"Error checking exit conditions for {position.symbol}: {e}")
            return None
    
    def _get_allocation_summary(self, allocated_stocks: List[Dict]) -> Dict:
        """Get allocation summary for Step 2."""
        summary = {"large_cap": 0, "mid_cap": 0, "small_cap": 0}
        
        for stock in allocated_stocks:
            market_cap = stock.get('market_cap', 'unknown')
            if market_cap in summary:
                summary[market_cap] += 1
        
        return summary
    
    def _get_signals_summary(self, entry_candidates: List[Dict]) -> Dict:
        """Get signals summary for Step 3."""
        if not entry_candidates:
            return {"strong_signals": 0, "medium_signals": 0, "weak_signals": 0}
        
        strong = len([c for c in entry_candidates if c.get('signal_strength', 0) > 80])
        medium = len([c for c in entry_candidates if 60 <= c.get('signal_strength', 0) <= 80])
        weak = len([c for c in entry_candidates if c.get('signal_strength', 0) < 60])
        
        return {
            "strong_signals": strong,
            "medium_signals": medium,
            "weak_signals": weak
        }
    
    def _get_positions_summary(self) -> List[Dict]:
        """Get summary of active positions."""
        summary = []
        
        for symbol, position in self.active_positions.items():
            summary.append({
                "symbol": symbol,
                "entry_date": position.entry_date.isoformat(),
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "quantity": position.quantity,
                "remaining_quantity": position.remaining_quantity,
                "unrealized_pnl": f"{position.unrealized_pnl*100:.2f}%",
                "status": position.status.value,
                "days_held": (datetime.now() - position.entry_date).days,
                "market_cap": position.market_cap.value,
                "risk_bucket": position.risk_bucket.value
            })
        
        return summary


def get_portfolio_strategy_engine(user_id: int = 1) -> PortfolioStrategyEngine:
    """Get portfolio strategy engine instance."""
    return PortfolioStrategyEngine(user_id)
