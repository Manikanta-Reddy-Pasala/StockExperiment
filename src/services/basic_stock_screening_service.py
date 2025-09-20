"""
Basic Stock Screening Service

Simple stock filtering based on fundamental criteria without ML dependency.
Shows top stocks based on price, volume, and basic risk metrics.
"""

import logging
import random
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from .ml.stock_discovery_service import get_stock_discovery_service, StockInfo, MarketCap
except ImportError:
    from services.ml.stock_discovery_service import get_stock_discovery_service, StockInfo, MarketCap


class RiskLevel(Enum):
    LOW = "low_risk"
    MEDIUM = "medium_risk"
    HIGH = "high_risk"


class BasicStockScreeningService:
    """Simple stock screening without ML complexity."""

    def __init__(self):
        self.stock_discovery = get_stock_discovery_service()

        # Risk criteria configurations
        self.risk_criteria = {
            RiskLevel.LOW: {
                "price_range": (100, 5000),  # Conservative price range
                "min_liquidity": 0.6,
                "market_caps": [MarketCap.LARGE_CAP, MarketCap.MID_CAP],
                "allocation": {"large_cap": 70, "mid_cap": 30, "small_cap": 0}
            },
            RiskLevel.MEDIUM: {
                "price_range": (50, 3000),
                "min_liquidity": 0.4,
                "market_caps": [MarketCap.LARGE_CAP, MarketCap.MID_CAP, MarketCap.SMALL_CAP],
                "allocation": {"large_cap": 50, "mid_cap": 35, "small_cap": 15}
            },
            RiskLevel.HIGH: {
                "price_range": (10, 2000),
                "min_liquidity": 0.3,
                "market_caps": [MarketCap.MID_CAP, MarketCap.SMALL_CAP],
                "allocation": {"large_cap": 20, "mid_cap": 40, "small_cap": 40}
            }
        }

    def get_top_stocks(self, risk_level: str = "medium_risk", limit: int = 20, user_id: int = 1) -> Dict:
        """Get top stocks based on basic screening criteria."""
        try:
            logger.info(f"Basic screening for {limit} stocks with {risk_level} risk level")

            # Parse risk level
            if risk_level == "low_risk":
                risk = RiskLevel.LOW
            elif risk_level == "high_risk":
                risk = RiskLevel.HIGH
            else:
                risk = RiskLevel.MEDIUM

            # Get all tradeable stocks
            all_stocks = self.stock_discovery.discover_tradeable_stocks(user_id)
            logger.info(f"Retrieved {len(all_stocks)} stocks from discovery service")

            if not all_stocks:
                logger.warning("No stocks available from discovery service")
                return self._create_empty_response(risk_level)

            # Apply basic filters
            filtered_stocks = self._apply_basic_filters(all_stocks, risk)
            logger.info(f"After basic filtering: {len(filtered_stocks)} stocks")

            # Score and rank stocks
            scored_stocks = self._score_stocks(filtered_stocks, risk)

            # Get top stocks by category
            result = self._create_portfolio(scored_stocks, risk, limit)

            logger.info(f"Basic screening completed: {result['total_stocks']} stocks selected")
            return result

        except Exception as e:
            logger.error(f"Error in basic stock screening: {e}")
            return self._create_empty_response(risk_level)

    def _apply_basic_filters(self, stocks: List[StockInfo], risk: RiskLevel) -> List[StockInfo]:
        """Apply basic filtering criteria."""
        criteria = self.risk_criteria[risk]
        filtered = []

        for stock in stocks:
            # Price range filter
            price_min, price_max = criteria["price_range"]
            if not (price_min <= stock.current_price <= price_max):
                continue

            # Liquidity filter
            if stock.liquidity_score < criteria["min_liquidity"]:
                continue

            # Market cap filter
            if stock.market_cap_category not in criteria["market_caps"]:
                continue

            # Must be tradeable
            if not stock.is_tradeable:
                continue

            filtered.append(stock)

        return filtered

    def _score_stocks(self, stocks: List[StockInfo], risk: RiskLevel) -> List[Dict]:
        """Score stocks based on basic metrics."""
        scored_stocks = []

        for stock in stocks:
            # Simple scoring based on liquidity, market cap, and volume
            score = 0.0

            # Liquidity score (40% weight)
            score += stock.liquidity_score * 0.4

            # Market cap score (30% weight)
            if stock.market_cap_category == MarketCap.LARGE_CAP:
                score += 0.9 * 0.3
            elif stock.market_cap_category == MarketCap.MID_CAP:
                score += 0.7 * 0.3
            else:
                score += 0.5 * 0.3

            # Volume score (20% weight) - normalized
            volume_score = min(stock.volume / 1000000, 1.0)  # Normalize to 1M volume
            score += volume_score * 0.2

            # Price stability score (10% weight) - simplified
            price_score = 0.8 if 100 <= stock.current_price <= 1000 else 0.6
            score += price_score * 0.1

            # Create stock result
            stock_result = {
                "symbol": stock.symbol,
                "name": stock.name,
                "current_price": stock.current_price,
                "market_cap_crores": stock.market_cap_crores,
                "market_cap_category": stock.market_cap_category.value,
                "volume": stock.volume,
                "liquidity_score": stock.liquidity_score,
                "sector": stock.sector,
                "basic_score": round(score, 3),
                "recommendation": "BUY" if score > 0.6 else "HOLD",
                "risk_level": risk.value,
                "screening_method": "basic_criteria"
            }

            scored_stocks.append(stock_result)

        # Sort by score descending
        scored_stocks.sort(key=lambda x: x["basic_score"], reverse=True)
        return scored_stocks

    def _create_portfolio(self, scored_stocks: List[Dict], risk: RiskLevel, limit: int) -> Dict:
        """Create portfolio allocation based on risk level."""
        criteria = self.risk_criteria[risk]
        allocation = criteria["allocation"]

        # Separate stocks by market cap
        large_cap = [s for s in scored_stocks if s["market_cap_category"] == "large_cap"]
        mid_cap = [s for s in scored_stocks if s["market_cap_category"] == "mid_cap"]
        small_cap = [s for s in scored_stocks if s["market_cap_category"] == "small_cap"]

        # Calculate allocation counts
        large_count = int(limit * allocation["large_cap"] / 100)
        mid_count = int(limit * allocation["mid_cap"] / 100)
        small_count = int(limit * allocation["small_cap"] / 100)

        # Adjust for rounding
        remaining = limit - (large_count + mid_count + small_count)
        if remaining > 0:
            if allocation["large_cap"] > 0:
                large_count += remaining
            elif allocation["mid_cap"] > 0:
                mid_count += remaining
            else:
                small_count += remaining

        # Select top stocks from each category
        selected_large = large_cap[:large_count]
        selected_mid = mid_cap[:mid_count]
        selected_small = small_cap[:small_count]

        total_selected = len(selected_large) + len(selected_mid) + len(selected_small)

        return {
            "success": True,
            "method": "basic_screening",
            "risk_level": risk.value,
            "total_stocks": total_selected,
            "portfolio": {
                "large_cap": selected_large,
                "mid_cap": selected_mid,
                "small_cap": selected_small
            },
            "allocation_summary": {
                "large_cap": len(selected_large),
                "mid_cap": len(selected_mid),
                "small_cap": len(selected_small),
                "total": total_selected
            },
            "filtering_statistics": {
                "data_source": "Basic Screening Algorithm",
                "total_discovered": len(scored_stocks),
                "filtering_stages": f"Total {len(scored_stocks)} → Risk filtered → Top {total_selected} selected",
                "criteria_applied": f"Price range, liquidity ≥ {criteria['min_liquidity']}, market cap filter"
            }
        }

    def _create_empty_response(self, risk_level: str) -> Dict:
        """Create empty response when no stocks found."""
        return {
            "success": False,
            "method": "basic_screening",
            "risk_level": risk_level,
            "total_stocks": 0,
            "portfolio": {
                "large_cap": [],
                "mid_cap": [],
                "small_cap": []
            },
            "allocation_summary": {
                "large_cap": 0,
                "mid_cap": 0,
                "small_cap": 0,
                "total": 0
            },
            "filtering_statistics": {
                "data_source": "Basic Screening Algorithm",
                "total_discovered": 0,
                "filtering_stages": "No stocks available",
                "criteria_applied": "None - no data available"
            },
            "error": "No stocks available for screening"
        }


# Global service instance
_basic_screening_service = None

def get_basic_stock_screening_service() -> BasicStockScreeningService:
    """Get the global basic stock screening service instance."""
    global _basic_screening_service
    if _basic_screening_service is None:
        _basic_screening_service = BasicStockScreeningService()
    return _basic_screening_service