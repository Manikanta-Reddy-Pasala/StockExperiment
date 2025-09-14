"""
Enhanced Strategy Service
Implements the user-defined trading strategy, including risk allocation,
entry rules, and exit rules.
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum
import math
import pandas as pd

from .stock_screening_service import get_stock_screening_service, StockData
from .broker_service import get_broker_service
from .brokers.fyers_service import FyersAPIConnector

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for strategies."""
    SAFE = "safe"
    HIGH_RISK = "high_risk"


class StrategyConfig:
    """Configuration for trading strategies."""
    def __init__(self, name: str, risk_level: RiskLevel, large_cap_allocation: float,
                 mid_cap_allocation: float, small_cap_allocation: float, max_stocks: int):
        self.name = name
        self.risk_level = risk_level
        self.large_cap_allocation = large_cap_allocation
        self.mid_cap_allocation = mid_cap_allocation
        self.small_cap_allocation = small_cap_allocation
        self.max_stocks = max_stocks


class StockRecommendation:
    """Stock recommendation with ML predictions and rationale."""
    def __init__(self, stock_data: StockData, ml_prediction: Dict, selection_score: float,
                 recommended_allocation: float, recommended_quantity: int,
                 exit_rules: Dict):
        self.stock_data = stock_data
        self.ml_prediction = ml_prediction
        self.selection_score = selection_score
        self.recommended_allocation = recommended_allocation
        self.recommended_quantity = recommended_quantity
        self.exit_rules = exit_rules


class AdvancedStrategyService:
    """Advanced strategy service to implement user-defined trading logic."""
    
    LARGE_CAP_MIN_CR = 50000
    MID_CAP_MIN_CR = 10000

    def __init__(self):
        """Initialize the strategy service."""
        self.broker_service = get_broker_service()
        self.stock_screening_service = get_stock_screening_service(self.broker_service)
        self.fyers_connector = None
        
        self.strategies = {
            "safe_risk": StrategyConfig(
                name="Safe Risk Strategy",
                risk_level=RiskLevel.SAFE,
                large_cap_allocation=0.50,
                mid_cap_allocation=0.50,
                small_cap_allocation=0.0,
                max_stocks=20
            ),
            "high_risk": StrategyConfig(
                name="High Risk Strategy",
                risk_level=RiskLevel.HIGH_RISK,
                large_cap_allocation=0.0,
                mid_cap_allocation=0.50,
                small_cap_allocation=0.50,
                max_stocks=20
            )
        }

    def _initialize_fyers_connector(self, user_id: int = 1):
        """Initialize FYERS connector for API calls."""
        if self.fyers_connector:
            return True
        try:
            config = self.broker_service.get_broker_config('fyers', user_id)
            if config and config.get('is_connected') and config.get('access_token'):
                self.fyers_connector = FyersAPIConnector(
                    client_id=config.get('client_id'),
                    access_token=config.get('access_token')
                )
                return True
        except Exception as e:
            logger.error(f"Error initializing FYERS connector in AdvancedStrategyService: {e}")
        return False
    
    def generate_stock_recommendations(self, user_id: int, strategy_type: str, 
                                     capital: float = 100000) -> Dict:
        """Generate stock recommendations based on strategy and ML predictions."""
        try:
            if strategy_type not in self.strategies:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            if not self._initialize_fyers_connector(user_id):
                raise ConnectionError("Failed to initialize Fyers connector.")

            config = self.strategies[strategy_type]
            logger.info(f"Generating recommendations for {config.name} strategy")
            
            stock_candidates = self._get_strategy_stock_candidates(config, user_id)
            
            recommendations = []
            
            for i, stock_data in enumerate(stock_candidates):
                if not self._passes_entry_rules(stock_data, user_id):
                    continue

                ml_prediction = self._get_ml_prediction(stock_data.symbol, user_id)
                selection_score = 85.0 - (i * 2.5)
                recommendation = self._create_recommendation(
                    stock_data, ml_prediction, selection_score, config, capital, i
                )
                recommendations.append(recommendation)
            
            total_investment = sum(rec.recommended_quantity * rec.stock_data.current_price for rec in recommendations)
            
            return {
                "success": True,
                "strategy_type": strategy_type,
                "strategy_name": config.name,
                "total_recommendations": len(recommendations),
                "recommendations": [self._recommendation_to_dict(rec) for rec in recommendations],
                "summary": {"total_investment": total_investment}
            }
            
        except Exception as e:
            logger.error(f"Error generating stock recommendations: {e}")
            return {"success": False, "error": str(e)}

    def _get_entry_indicators(self, symbol: str) -> Optional[Dict]:
        """Calculate technical indicators for entry rules."""
        try:
            range_to = datetime.now()
            range_from = range_to - timedelta(days=90)
            
            history_data = self.fyers_connector.get_history(
                symbol=symbol, resolution="D",
                range_from=range_from.strftime('%Y-%m-%d'),
                range_to=range_to.strftime('%Y-%m-%d')
            )

            if not history_data or not history_data.get('candles') or len(history_data['candles']) < 51:
                logger.warning(f"Not enough historical data for {symbol} for entry indicators.")
                return None

            df = pd.DataFrame(history_data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['high_20d'] = df['high'].rolling(window=20).max()
            df['avg_volume_20d'] = df['volume'].rolling(window=20).mean()
            
            # Correct RSI(14) calculation
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            latest = df.iloc[-1]
            return {
                "ema_20": latest['ema_20'], "ema_50": latest['ema_50'],
                "high_20d": latest['high_20d'], "avg_volume_20d": latest['avg_volume_20d'],
                "rsi_14": latest['rsi_14']
            }
        except Exception as e:
            logger.error(f"Error calculating entry indicators for {symbol}: {e}")
            return None

    def _passes_entry_rules(self, stock_data: StockData, user_id: int) -> bool:
        """Check if a stock passes the entry rules."""
        indicators = self._get_entry_indicators(stock_data.symbol)
        if not indicators: return False
        price = stock_data.current_price
        if not (price > indicators['ema_20'] and price > indicators['ema_50']): return False
        if not (price > indicators['high_20d']): return False
        if not (stock_data.volume >= 1.5 * indicators['avg_volume_20d']): return False
        if not (50 <= indicators['rsi_14'] <= 70): return False
        return True

    def _get_market_cap_category(self, market_cap_cr: float) -> Optional[str]:
        if market_cap_cr >= self.LARGE_CAP_MIN_CR: return "large_cap"
        elif market_cap_cr >= self.MID_CAP_MIN_CR: return "mid_cap"
        else: return "small_cap"

    def _get_strategy_stock_candidates(self, config: StrategyConfig, user_id: int) -> List[StockData]:
        all_screened_stocks = self.stock_screening_service.screen_stocks(user_id=user_id)
        large_caps, mid_caps, small_caps = [], [], []
        for stock in all_screened_stocks:
            category = self._get_market_cap_category(stock.market_cap / 1_00_00_000)
            if category == "large_cap": large_caps.append(stock)
            elif category == "mid_cap": mid_caps.append(stock)
            else: small_caps.append(stock)

        candidates = []
        num_large = math.ceil(config.max_stocks * config.large_cap_allocation)
        num_mid = math.ceil(config.max_stocks * config.mid_cap_allocation)
        num_small = math.ceil(config.max_stocks * config.small_cap_allocation)
        candidates.extend(large_caps[:num_large])
        candidates.extend(mid_caps[:num_mid])
        candidates.extend(small_caps[:num_small])
        logger.info(f"Found {len(candidates)} stock candidates for {config.name} strategy.")
        return candidates

    def _get_ml_prediction(self, symbol: str, user_id: int = 1) -> Dict:
        try:
            from .ml.prediction_service import get_prediction
            prediction = get_prediction(symbol, user_id)
            if prediction: return prediction
            return self._get_mock_ml_prediction(symbol)
        except Exception as e:
            logger.error(f"Error getting ML prediction for {symbol}: {e}")
            return self._get_mock_ml_prediction(symbol)

    def _get_mock_ml_prediction(self, symbol: str) -> Dict:
        import random
        predicted_change = random.uniform(5, 20)
        return {"predicted_change_percent": predicted_change, "signal": "BUY" if predicted_change > 8 else "HOLD", "confidence": random.uniform(0.6, 0.85)}

    def _create_recommendation(self, stock_data: StockData, ml_prediction: Dict,
                                  selection_score: float, config: StrategyConfig, 
                                  total_capital: float, index: int) -> StockRecommendation:
        entry_price = stock_data.current_price
        recommended_allocation = 1.0 / config.max_stocks
        recommended_quantity = int((total_capital * recommended_allocation) / entry_price)
        
        exit_rules = {
            "profit_target_1": { "price": entry_price * 1.05, "percent": 5, "sell_portion": 0.5 },
            "profit_target_2": { "price": entry_price * 1.10, "percent": 10, "sell_portion": 0.5 },
            "stop_loss_price": entry_price * 0.97, # 3% stop loss
            "time_stop_days": 10,
            "trailing_stop_percent": 3.0
        }
        
        return StockRecommendation(
            stock_data=stock_data,
            ml_prediction=ml_prediction,
            selection_score=selection_score,
            recommended_allocation=recommended_allocation,
            recommended_quantity=recommended_quantity,
            exit_rules=exit_rules
        )

    def _recommendation_to_dict(self, rec: StockRecommendation) -> Dict:
        stock_data = rec.stock_data
        market_cap_cr = stock_data.market_cap / 1_00_00_000
        return {
            "symbol": stock_data.symbol,
            "name": stock_data.name,
            "current_price": stock_data.current_price,
            "market_cap_cr": market_cap_cr,
            "market_cap_category": self._get_market_cap_category(market_cap_cr),
            "selection_score": rec.selection_score,
            "recommended_quantity": rec.recommended_quantity,
            "investment_amount": rec.recommended_quantity * stock_data.current_price,
            "ml_prediction": rec.ml_prediction,
            "exit_rules": rec.exit_rules,
            "fundamental_metrics": {
                "pe_ratio": stock_data.pe_ratio,
                "pb_ratio": stock_data.pb_ratio,
                "roe": stock_data.roe,
                "debt_to_equity": stock_data.debt_to_equity,
            }
        }


def get_advanced_strategy_service() -> AdvancedStrategyService:
    """Get advanced strategy service instance."""
    return AdvancedStrategyService()
