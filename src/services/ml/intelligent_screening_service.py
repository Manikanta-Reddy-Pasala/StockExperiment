"""
Intelligent ML-Powered Stock Screening Service

This service combines dynamic stock discovery with advanced ML models to provide
intelligent stock recommendations based on risk profiles and market conditions.
"""

import logging
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import concurrent.futures

logger = logging.getLogger(__name__)

try:
    from .stock_discovery_service import get_stock_discovery_service, StockInfo, MarketCap
    from .data_service import get_stock_data, create_features
    from .prediction_service import get_prediction
    from .training_service import train_and_tune_models
    from ...utils.ml_helpers import load_model, load_lstm_model, load_scaler
    from ...services.brokers.fyers_service import get_fyers_service
except ImportError:
    from stock_discovery_service import get_stock_discovery_service, StockInfo, MarketCap
    from data_service import get_stock_data, create_features
    from prediction_service import get_prediction
    from training_service import train_and_tune_models
    from utils.ml_helpers import load_model, load_lstm_model, load_scaler
    from services.brokers.fyers_service import get_fyers_service


class RiskProfile(Enum):
    DEFAULT = "default"
    HIGH_RISK = "high_risk"


@dataclass
class MLStockAnalysis:
    stock_info: StockInfo
    ml_prediction: float
    prediction_confidence: float
    technical_score: float
    risk_score: float
    overall_score: float
    recommendation: str  # BUY, HOLD, SELL
    target_price: float
    stop_loss: float
    expected_return: float
    model_status: str  # TRAINED, TRAINING, NO_MODEL
    reasons: List[str]


@dataclass
class PortfolioRecommendation:
    risk_profile: RiskProfile
    total_stocks: int
    large_cap_stocks: List[MLStockAnalysis]
    mid_cap_stocks: List[MLStockAnalysis]
    small_cap_stocks: List[MLStockAnalysis]
    expected_return: float
    portfolio_risk: float
    diversification_score: float
    allocation_summary: Dict[str, float]


class IntelligentScreeningService:
    """ML-powered intelligent stock screening service."""

    def __init__(self):
        self.stock_discovery = get_stock_discovery_service()
        self.fyers_service = get_fyers_service()

        # Risk profile configurations
        self.risk_profiles = {
            RiskProfile.DEFAULT: {
                "name": "Balanced Portfolio",
                "allocation": {"large_cap": 0.60, "mid_cap": 0.30, "small_cap": 0.10},
                "min_confidence": 0.70,
                "max_risk_score": 0.60,
                "min_technical_score": 0.65,
                "max_stocks_per_category": {"large_cap": 8, "mid_cap": 5, "small_cap": 2}
            },
            RiskProfile.HIGH_RISK: {
                "name": "Aggressive Growth Portfolio",
                "allocation": {"large_cap": 0.00, "mid_cap": 0.50, "small_cap": 0.50},
                "min_confidence": 0.60,
                "max_risk_score": 0.85,
                "min_technical_score": 0.55,
                "max_stocks_per_category": {"large_cap": 0, "mid_cap": 8, "small_cap": 7}
            }
        }

        # Model paths
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                      'api', 'v1', 'ml', 'trained')

    def analyze_stock_with_ml(self, stock_info: StockInfo, user_id: int = 1,
                            auto_train: bool = True) -> Optional[MLStockAnalysis]:
        """Analyze stock using ML models with auto-training capability."""
        try:
            logger.info(f"Analyzing {stock_info.symbol} with ML models")

            # Check if models exist
            model_status = self._check_model_availability(stock_info.symbol)

            # Auto-train if models missing and enabled
            if model_status == "NO_MODEL" and auto_train:
                logger.info(f"Auto-training models for {stock_info.symbol}")
                model_status = self._auto_train_stock_models(stock_info.symbol, user_id)

            # Get historical data for technical analysis
            df = get_stock_data(stock_info.symbol, period="1y", user_id=user_id)
            if df is None or len(df) < 60:
                logger.warning(f"Insufficient data for {stock_info.symbol}")
                return None

            # Create features
            df, features = create_features(df)
            if len(df) < 30:
                logger.warning(f"Insufficient data after feature engineering for {stock_info.symbol}")
                return None

            # Calculate technical scores
            technical_score = self._calculate_technical_score(df, features)
            risk_score = self._calculate_risk_score(df, features)

            # Get ML prediction
            ml_prediction = stock_info.current_price
            prediction_confidence = 0.5

            if model_status == "TRAINED":
                try:
                    prediction_result = get_prediction(stock_info.symbol, user_id)
                    if prediction_result and prediction_result.get('success'):
                        ml_prediction = prediction_result.get('predicted_price', stock_info.current_price)
                        prediction_confidence = prediction_result.get('confidence', 0.5)
                except Exception as e:
                    logger.warning(f"ML prediction failed for {stock_info.symbol}: {e}")

            # Calculate expected return
            expected_return = (ml_prediction - stock_info.current_price) / stock_info.current_price

            # Calculate overall score
            overall_score = self._calculate_overall_score(
                technical_score, prediction_confidence, risk_score,
                stock_info.liquidity_score, expected_return
            )

            # Generate recommendation
            recommendation, target_price, stop_loss, reasons = self._generate_recommendation(
                stock_info, ml_prediction, overall_score, expected_return, technical_score, risk_score
            )

            return MLStockAnalysis(
                stock_info=stock_info,
                ml_prediction=ml_prediction,
                prediction_confidence=prediction_confidence,
                technical_score=technical_score,
                risk_score=risk_score,
                overall_score=overall_score,
                recommendation=recommendation,
                target_price=target_price,
                stop_loss=stop_loss,
                expected_return=expected_return,
                model_status=model_status,
                reasons=reasons
            )

        except Exception as e:
            logger.error(f"Error analyzing {stock_info.symbol}: {e}")
            return None

    def screen_stocks_by_risk_profile(self, risk_profile: RiskProfile,
                                   user_id: int = 1,
                                   auto_train: bool = True) -> PortfolioRecommendation:
        """Screen stocks based on risk profile using ML analysis."""
        try:
            logger.info(f"Screening stocks for {risk_profile.value} risk profile")

            # Get dynamic stock universe
            stocks_by_category = self.stock_discovery.get_stocks_by_category(user_id)
            profile_config = self.risk_profiles[risk_profile]

            # Analyze stocks in parallel for each category
            analysis_results = {
                MarketCap.LARGE_CAP: [],
                MarketCap.MID_CAP: [],
                MarketCap.SMALL_CAP: []
            }

            for market_cap, stocks in stocks_by_category.items():
                if profile_config["allocation"].get(market_cap.value, 0) <= 0:
                    continue  # Skip categories not in allocation

                max_stocks = profile_config["max_stocks_per_category"].get(market_cap.value, 5)

                # Take top liquid stocks for analysis
                top_stocks = stocks[:max_stocks * 2]  # Analyze more, filter later

                logger.info(f"Analyzing {len(top_stocks)} {market_cap.value} stocks")

                # Parallel analysis
                category_analyses = self._analyze_stocks_parallel(top_stocks, user_id, auto_train)

                # Filter by risk profile criteria
                filtered_analyses = self._filter_by_risk_criteria(category_analyses, profile_config)

                # Sort by overall score and take top stocks
                filtered_analyses.sort(key=lambda x: x.overall_score, reverse=True)
                analysis_results[market_cap] = filtered_analyses[:max_stocks]

            # Create portfolio recommendation
            return self._create_portfolio_recommendation(
                risk_profile, analysis_results, profile_config
            )

        except Exception as e:
            logger.error(f"Error screening stocks: {e}")
            return self._create_empty_portfolio_recommendation(risk_profile)

    def get_top_ml_picks(self, risk_profile: RiskProfile, count: int = 10,
                        user_id: int = 1) -> List[MLStockAnalysis]:
        """Get top ML picks across all categories."""
        try:
            portfolio = self.screen_stocks_by_risk_profile(risk_profile, user_id)

            all_stocks = (
                portfolio.large_cap_stocks +
                portfolio.mid_cap_stocks +
                portfolio.small_cap_stocks
            )

            # Filter for BUY recommendations
            buy_stocks = [s for s in all_stocks if s.recommendation == "BUY"]

            # Sort by combined score
            buy_stocks.sort(
                key=lambda x: (x.overall_score * 0.6 + x.expected_return * 0.4),
                reverse=True
            )

            return buy_stocks[:count]

        except Exception as e:
            logger.error(f"Error getting top picks: {e}")
            return []

    def _analyze_stocks_parallel(self, stocks: List[StockInfo], user_id: int,
                               auto_train: bool, max_workers: int = 5) -> List[MLStockAnalysis]:
        """Analyze stocks in parallel for better performance."""
        results = []

        # Use ThreadPoolExecutor for I/O bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {
                executor.submit(self.analyze_stock_with_ml, stock, user_id, auto_train): stock
                for stock in stocks
            }

            for future in concurrent.futures.as_completed(future_to_stock):
                try:
                    result = future.result(timeout=30)  # 30 second timeout per stock
                    if result:
                        results.append(result)
                except Exception as e:
                    stock = future_to_stock[future]
                    logger.warning(f"Failed to analyze {stock.symbol}: {e}")
                    continue

        return results

    def _check_model_availability(self, symbol: str) -> str:
        """Check if ML models are available for the symbol."""
        try:
            rf_path = os.path.join(self.models_dir, f"{symbol}_rf.pkl")
            xgb_path = os.path.join(self.models_dir, f"{symbol}_xgb.pkl")
            lstm_path = os.path.join(self.models_dir, f"{symbol}_lstm.h5")

            if all(os.path.exists(path) for path in [rf_path, xgb_path, lstm_path]):
                return "TRAINED"
            else:
                return "NO_MODEL"

        except Exception as e:
            logger.error(f"Error checking model availability for {symbol}: {e}")
            return "NO_MODEL"

    def _auto_train_stock_models(self, symbol: str, user_id: int) -> str:
        """Auto-train ML models for a stock with 5-year data."""
        try:
            logger.info(f"Auto-training models for {symbol} with 5-year data")

            # Set training dates for 5 years
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=1825)  # 5 years

            # Start training in background
            result = train_and_tune_models(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                job_id=None  # No job tracking for auto-training
            )

            if result and result.get('success'):
                logger.info(f"Successfully trained models for {symbol}")
                return "TRAINED"
            else:
                logger.warning(f"Training failed for {symbol}")
                return "NO_MODEL"

        except Exception as e:
            logger.error(f"Auto-training failed for {symbol}: {e}")
            return "NO_MODEL"

    def _calculate_technical_score(self, df: pd.DataFrame, features: List[str]) -> float:
        """Calculate technical analysis score using our 70 features."""
        try:
            if df.empty or len(df) < 10:
                return 0.5

            latest_data = df.iloc[-1]
            score_components = []

            # Trend indicators (30%)
            trend_score = 0.0
            if 'Trend_Regime' in features:
                trend_score += latest_data.get('Trend_Regime', 0) * 0.4
            if 'MA5_MA20_Ratio' in features:
                ma_ratio = latest_data.get('MA5_MA20_Ratio', 1.0)
                trend_score += (0.3 if ma_ratio > 1.02 else -0.2 if ma_ratio < 0.98 else 0.0)
            if 'Short_Long_MA_Diff' in features:
                ma_diff = latest_data.get('Short_Long_MA_Diff', 0.0)
                trend_score += (0.3 if ma_diff > 0.05 else -0.1 if ma_diff < -0.05 else 0.0)
            score_components.append(min(max(trend_score, 0), 1) * 0.30)

            # Momentum indicators (25%)
            momentum_score = 0.0
            if 'RSI' in features:
                rsi = latest_data.get('RSI', 50)
                momentum_score += (0.4 if 30 < rsi < 70 else 0.2 if rsi > 70 else -0.3)
            if 'MACD_Histogram' in features:
                macd_hist = latest_data.get('MACD_Histogram', 0)
                momentum_score += (0.3 if macd_hist > 0 else -0.2)
            if 'Momentum_Acceleration' in features:
                mom_accel = latest_data.get('Momentum_Acceleration', 0)
                momentum_score += (0.3 if mom_accel > 0 else -0.1)
            score_components.append(min(max(momentum_score, 0), 1) * 0.25)

            # Volume indicators (20%)
            volume_score = 0.0
            if 'Volume_Regime' in features:
                volume_score += latest_data.get('Volume_Regime', 0) * 0.5
            if 'Volume_Momentum' in features:
                vol_momentum = latest_data.get('Volume_Momentum', 0)
                volume_score += (0.3 if vol_momentum > 0.1 else 0.0)
            if 'OBV_Ratio' in features:
                obv_ratio = latest_data.get('OBV_Ratio', 1.0)
                volume_score += (0.2 if obv_ratio > 1.1 else 0.0)
            score_components.append(min(max(volume_score, 0), 1) * 0.20)

            # Volatility indicators (15%)
            volatility_score = 0.0
            if 'Volatility_Regime' in features:
                vol_regime = latest_data.get('Volatility_Regime', 0)
                volatility_score += (0.5 if vol_regime == 0 else 0.3)
            if 'Realized_Volatility' in features:
                realized_vol = latest_data.get('Realized_Volatility', 0.3)
                volatility_score += (0.5 if realized_vol < 0.25 else 0.2)
            score_components.append(min(max(volatility_score, 0), 1) * 0.15)

            # Pattern recognition (10%)
            pattern_score = 0.0
            if 'Hammer_Pattern' in features:
                pattern_score += latest_data.get('Hammer_Pattern', 0) * 0.4
            if 'Price_Position' in features:
                price_pos = latest_data.get('Price_Position', 0.5)
                pattern_score += (0.4 if 0.6 < price_pos < 0.9 else 0.2)
            if 'BB_Position' in features:
                bb_pos = latest_data.get('BB_Position', 0.5)
                pattern_score += (0.2 if 0.2 < bb_pos < 0.8 else 0.0)
            score_components.append(min(max(pattern_score, 0), 1) * 0.10)

            return sum(score_components)

        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0.5

    def _calculate_risk_score(self, df: pd.DataFrame, features: List[str]) -> float:
        """Calculate risk score using advanced volatility and stress indicators."""
        try:
            if df.empty:
                return 0.5

            latest_data = df.iloc[-1]
            risk_factors = []

            # Volatility risk (40%)
            if 'Realized_Volatility' in features:
                vol = latest_data.get('Realized_Volatility', 0.2)
                risk_factors.append(min(vol / 0.5, 1.0) * 0.40)

            # Market stress (25%)
            if 'Market_Stress' in features:
                stress = latest_data.get('Market_Stress', 0)
                normalized_stress = min(stress / 1000000, 1.0)
                risk_factors.append(normalized_stress * 0.25)

            # Liquidity risk (20%)
            if 'Liquidity_Index' in features:
                liquidity = latest_data.get('Liquidity_Index', 1000)
                liquidity_risk = max(0, 1 - (liquidity / 10000))
                risk_factors.append(liquidity_risk * 0.20)

            # Technical risk (15%)
            if 'Volatility_Skew' in features:
                skew = abs(latest_data.get('Volatility_Skew', 0))
                risk_factors.append(min(skew / 2.0, 1.0) * 0.15)

            return sum(risk_factors) if risk_factors else 0.5

        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5

    def _calculate_overall_score(self, technical_score: float, prediction_confidence: float,
                               risk_score: float, liquidity_score: float, expected_return: float) -> float:
        """Calculate overall investment score."""
        # Normalize expected return (cap at ±50%)
        normalized_return = max(-0.5, min(expected_return, 0.5)) + 0.5

        overall = (
            technical_score * 0.30 +
            prediction_confidence * 0.25 +
            (1 - risk_score) * 0.20 +  # Lower risk = higher score
            liquidity_score * 0.15 +
            normalized_return * 0.10
        )

        return min(max(overall, 0.0), 1.0)

    def _generate_recommendation(self, stock_info: StockInfo, ml_prediction: float,
                               overall_score: float, expected_return: float,
                               technical_score: float, risk_score: float) -> Tuple[str, float, float, List[str]]:
        """Generate trading recommendation with target and stop loss."""
        current_price = stock_info.current_price
        reasons = []

        # Base recommendation logic
        if overall_score >= 0.75 and expected_return >= 0.08:
            recommendation = "BUY"
            target_price = ml_prediction * 1.15  # 15% above prediction
            stop_loss = current_price * 0.92    # 8% stop loss
            reasons.append("Strong technical and ML signals")
        elif overall_score >= 0.65 and expected_return >= 0.05:
            recommendation = "BUY"
            target_price = ml_prediction * 1.10  # 10% above prediction
            stop_loss = current_price * 0.95    # 5% stop loss
            reasons.append("Good technical setup with positive ML outlook")
        elif overall_score <= 0.35 or expected_return <= -0.10:
            recommendation = "SELL"
            target_price = ml_prediction * 0.90  # 10% below prediction
            stop_loss = current_price * 1.05    # 5% upside stop for short
            reasons.append("Weak fundamentals or high risk detected")
        else:
            recommendation = "HOLD"
            target_price = ml_prediction
            stop_loss = current_price * 0.95
            reasons.append("Mixed signals - monitor for better entry")

        # Add specific reasons
        if technical_score >= 0.80:
            reasons.append("Excellent technical indicators")
        if risk_score <= 0.30:
            reasons.append("Low risk profile")
        if stock_info.liquidity_score >= 0.80:
            reasons.append("High liquidity")

        return recommendation, target_price, stop_loss, reasons

    def _filter_by_risk_criteria(self, analyses: List[MLStockAnalysis],
                               profile_config: Dict) -> List[MLStockAnalysis]:
        """Filter stocks based on risk profile criteria."""
        return [
            analysis for analysis in analyses
            if (
                analysis.prediction_confidence >= profile_config["min_confidence"] and
                analysis.risk_score <= profile_config["max_risk_score"] and
                analysis.technical_score >= profile_config["min_technical_score"]
            )
        ]

    def _create_portfolio_recommendation(self, risk_profile: RiskProfile,
                                       analysis_results: Dict[MarketCap, List[MLStockAnalysis]],
                                       profile_config: Dict) -> PortfolioRecommendation:
        """Create portfolio recommendation from analysis results."""
        large_cap_stocks = analysis_results[MarketCap.LARGE_CAP]
        mid_cap_stocks = analysis_results[MarketCap.MID_CAP]
        small_cap_stocks = analysis_results[MarketCap.SMALL_CAP]

        all_stocks = large_cap_stocks + mid_cap_stocks + small_cap_stocks

        # Calculate portfolio metrics
        if all_stocks:
            expected_return = sum(s.expected_return for s in all_stocks) / len(all_stocks)
            portfolio_risk = sum(s.risk_score for s in all_stocks) / len(all_stocks)

            # Diversification score based on sectors
            unique_sectors = len(set(s.stock_info.sector for s in all_stocks))
            diversification_score = min(unique_sectors / 8.0, 1.0)
        else:
            expected_return = 0.0
            portfolio_risk = 0.5
            diversification_score = 0.0

        allocation_summary = {
            "large_cap": len(large_cap_stocks),
            "mid_cap": len(mid_cap_stocks),
            "small_cap": len(small_cap_stocks),
            "total": len(all_stocks)
        }

        return PortfolioRecommendation(
            risk_profile=risk_profile,
            total_stocks=len(all_stocks),
            large_cap_stocks=large_cap_stocks,
            mid_cap_stocks=mid_cap_stocks,
            small_cap_stocks=small_cap_stocks,
            expected_return=expected_return,
            portfolio_risk=portfolio_risk,
            diversification_score=diversification_score,
            allocation_summary=allocation_summary
        )

    def _create_empty_portfolio_recommendation(self, risk_profile: RiskProfile) -> PortfolioRecommendation:
        """Create empty portfolio recommendation."""
        return PortfolioRecommendation(
            risk_profile=risk_profile,
            total_stocks=0,
            large_cap_stocks=[],
            mid_cap_stocks=[],
            small_cap_stocks=[],
            expected_return=0.0,
            portfolio_risk=0.5,
            diversification_score=0.0,
            allocation_summary={"large_cap": 0, "mid_cap": 0, "small_cap": 0, "total": 0}
        )


# Global service instance
_intelligent_screening_service = None

def get_intelligent_screening_service() -> IntelligentScreeningService:
    """Get the global intelligent screening service instance."""
    global _intelligent_screening_service
    if _intelligent_screening_service is None:
        _intelligent_screening_service = IntelligentScreeningService()
    return _intelligent_screening_service