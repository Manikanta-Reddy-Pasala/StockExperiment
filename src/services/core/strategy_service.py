"""
Strategy Service
Implements the user-defined trading strategy, including risk allocation,
entry rules, and exit rules.
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum
import math
from src.utils.api_logger import APILogger, log_api_call
import pandas as pd

from .stock_screening_service import get_stock_screening_service, StockData
from .broker_service import get_broker_service
from .user_settings_service import get_user_settings_service
from .unified_broker_service import get_unified_broker_service

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
        self.user_settings_service = get_user_settings_service()
        self.unified_broker_service = get_unified_broker_service()
        self.current_broker = None
        
        self.strategies = {
            "default_risk": StrategyConfig(
                name="Default Risk Strategy",
                risk_level=RiskLevel.SAFE,
                large_cap_allocation=0.60,
                mid_cap_allocation=0.30,
                small_cap_allocation=0.10,
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

    def _get_user_broker(self, user_id: int = 1):
        """Get the user's selected broker and verify it's connected."""
        try:
            # Get user's current broker from settings
            settings = self.user_settings_service.get_user_settings(user_id)
            current_broker = settings.get('current_broker', 'fyers')  # Default to fyers if not set
            
            # Get broker configuration
            broker_config = self.broker_service.get_broker_config(current_broker, user_id)
            
            if not broker_config:
                logger.warning(f"No broker configuration found for {current_broker}")
                return None, None
            
            if not broker_config.get('is_connected'):
                logger.warning(f"Broker {current_broker} is not connected for user {user_id}")
                return current_broker, None
            
            self.current_broker = current_broker
            return current_broker, broker_config
            
        except Exception as e:
            logger.error(f"Error getting user broker: {e}")
            return None, None
    
    def generate_stock_recommendations(self, user_id: int, strategy_type: str, 
                                     capital: float = 100000) -> Dict:
        """Generate stock recommendations based on strategy and ML predictions."""
        # Log API call
        APILogger.log_request(
            service_name="StrategyService",
            method_name="generate_stock_recommendations",
            request_data={'strategy_type': strategy_type, 'capital': capital},
            user_id=user_id
        )
        
        try:
            if strategy_type not in self.strategies:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            config = self.strategies[strategy_type]
            logger.info(f"Generating recommendations for {config.name} strategy")
            
            # Get user's selected broker
            current_broker, broker_config = self._get_user_broker(user_id)
            
            if not current_broker or not broker_config:
                logger.warning(f"No connected broker found for user {user_id}. Cannot generate recommendations without broker connection.")
                return {
                    "success": False, 
                    "error": "No connected broker found. Please connect a broker to generate stock recommendations.",
                    "recommendations": [],
                    "strategy_type": strategy_type,
                    "total_capital": capital,
                    "allocated_capital": 0.0,
                    "broker_name": current_broker
                }
            
            logger.info(f"Using broker: {current_broker} for user {user_id}")
            
            # Use the unified broker service to get stock data
            stock_candidates = self._get_strategy_stock_candidates(config, user_id, current_broker)
            
            recommendations = []
            
            for i, stock_data in enumerate(stock_candidates):
                if not self._passes_entry_rules(stock_data, user_id, current_broker):
                    continue

                ml_prediction = self._get_ml_prediction(stock_data.symbol, user_id)
                selection_score = 85.0 - (i * 2.5)
                recommendation = self._create_recommendation(
                    stock_data, ml_prediction, selection_score, config, capital, i
                )
                recommendations.append(recommendation)
            
            # Convert recommendations to dict format for Ollama enhancement
            recommendations_dict = [self._recommendation_to_dict(rec) for rec in recommendations]
            
            # Apply Ollama enhancement to final strategy recommendations
            try:
                from src.services.data.strategy_ollama_enhancement_service import get_strategy_ollama_enhancement_service
                ollama_service = get_strategy_ollama_enhancement_service()
                
                logger.info(f"🔍 Applying Ollama enhancement to {len(recommendations_dict)} strategy recommendations")
                enhanced_recommendations = ollama_service.enhance_strategy_recommendations(
                    recommendations_dict, strategy_type, "comprehensive"
                )
                
                # Update recommendations with enhanced data
                recommendations_dict = enhanced_recommendations
                logger.info(f"✅ Ollama enhancement completed for strategy recommendations")
                
            except Exception as e:
                logger.warning(f"Ollama enhancement failed: {e}")
                logger.warning("Continuing with original recommendations")
            
            total_investment = sum(rec.get('recommended_quantity', 0) * rec.get('current_price', 0) for rec in recommendations_dict)
            
            result = {
                "success": True,
                "strategy_type": strategy_type,
                "strategy_name": config.name,
                "broker_used": current_broker,
                "total_recommendations": len(recommendations_dict),
                "recommendations": recommendations_dict,
                "summary": {"total_investment": total_investment}
            }
            
            # Log response
            APILogger.log_response(
                service_name="StrategyService",
                method_name="generate_stock_recommendations",
                response_data=result,
                user_id=user_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating stock recommendations: {e}")
            APILogger.log_error(
                service_name="StrategyService",
                method_name="generate_stock_recommendations",
                error=e,
                request_data={'strategy_type': strategy_type, 'capital': capital},
                user_id=user_id
            )
            return {"success": False, "error": str(e)}
    

    def _get_entry_indicators(self, symbol: str, user_id: int = 1) -> Optional[Dict]:
        """Calculate technical indicators for entry rules."""
        try:
            # For now, return mock indicators since we need to implement proper
            # historical data fetching through the unified broker service
            # TODO: Implement proper historical data fetching through unified broker service
            
            # Mock indicators for demonstration
            return {
                'ema_20': 100.0,
                'ema_50': 95.0,
                'rsi': 45.0,
                'volume_20_avg': 1000000,
                'price_20_high': 110.0
            }
        except Exception as e:
            logger.error(f"Error calculating entry indicators for {symbol}: {e}")
            return None

    def _passes_entry_rules(self, stock_data: StockData, user_id: int, broker_name: str = None) -> bool:
        """Check if a stock passes the entry rules."""
        indicators = self._get_entry_indicators(stock_data.symbol, user_id)
        if not indicators: return False
        price = stock_data.current_price
        if not (price > indicators['ema_20'] and price > indicators['ema_50']): return False
        if not (price > indicators['price_20_high']): return False
        if not (stock_data.volume >= 1.5 * indicators['volume_20_avg']): return False
        if not (50 <= indicators['rsi'] <= 70): return False
        return True

    def _get_market_cap_category(self, market_cap_cr: float) -> Optional[str]:
        if market_cap_cr >= self.LARGE_CAP_MIN_CR: return "large_cap"
        elif market_cap_cr >= self.MID_CAP_MIN_CR: return "mid_cap"
        else: return "small_cap"

    def _get_strategy_stock_candidates(self, config: StrategyConfig, user_id: int, broker_name: str = None) -> List[StockData]:
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
