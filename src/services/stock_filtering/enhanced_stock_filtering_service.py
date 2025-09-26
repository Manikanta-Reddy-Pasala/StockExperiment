"""
Enhanced Stock Filtering Service
Implements comprehensive stock filtering with Stage 1 and Stage 2 filters
Supports all features from the enhanced YAML configuration
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
from dataclasses import dataclass

from .enhanced_config_loader import (
    EnhancedFilteringConfig, get_enhanced_filtering_config,
    UniverseConfig, Stage1Filters, TechnicalIndicators, 
    FundamentalRatios, RiskMetrics, ScoringWeights, 
    FilteringThresholds, SelectionConfig
)

logger = logging.getLogger(__name__)


@dataclass
class StockScore:
    """Stock scoring results."""
    symbol: str
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    risk_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    total_score: float = 0.0
    filters_passed: List[str] = None
    reject_reasons: List[str] = None
    
    def __post_init__(self):
        if self.filters_passed is None:
            self.filters_passed = []
        if self.reject_reasons is None:
            self.reject_reasons = []


@dataclass
class FilteringResult:
    """Complete filtering result."""
    selected_stocks: List[StockScore]
    rejected_stocks: List[StockScore]
    total_processed: int
    stage1_passed: int
    stage2_passed: int
    final_selected: int
    execution_time: float
    config_used: EnhancedFilteringConfig


class EnhancedStockFilteringService:
    """Enhanced stock filtering service with comprehensive filtering capabilities."""
    
    def __init__(self, config: Optional[EnhancedFilteringConfig] = None):
        """Initialize the enhanced filtering service."""
        self.config = config or get_enhanced_filtering_config()
        self.filter_stats = {
            'total_processed': 0,
            'stage1_passed': 0,
            'stage2_passed': 0,
            'final_selected': 0,
            'execution_time': 0.0
        }
    
    def filter_stocks(self, stocks: List[Any], user_id: int = 1) -> FilteringResult:
        """
        Comprehensive stock filtering with Stage 1 and Stage 2 filters.
        
        Args:
            stocks: List of stock objects to filter
            user_id: User ID for personalized filtering
            
        Returns:
            FilteringResult with selected and rejected stocks
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting enhanced stock filtering for {len(stocks)} stocks")
            
            # Stage 1: Basic filtering
            stage1_stocks, stage1_rejected = self._apply_stage1_filters(stocks)
            logger.info(f"Stage 1: {len(stage1_stocks)} stocks passed, {len(stage1_rejected)} rejected")
            
            # Stage 2: Advanced analysis and scoring
            stage2_stocks, stage2_rejected = self._apply_stage2_filters(stage1_stocks)
            logger.info(f"Stage 2: {len(stage2_stocks)} stocks passed, {len(stage2_rejected)} rejected")
            
            # Final selection with guardrails
            final_stocks, final_rejected = self._apply_final_selection(stage2_stocks)
            logger.info(f"Final selection: {len(final_stocks)} stocks selected, {len(final_rejected)} rejected")
            
            # Combine all rejected stocks
            all_rejected = stage1_rejected + stage2_rejected + final_rejected
            
            execution_time = time.time() - start_time
            
            # Update statistics
            self.filter_stats.update({
                'total_processed': len(stocks),
                'stage1_passed': len(stage1_stocks),
                'stage2_passed': len(stage2_stocks),
                'final_selected': len(final_stocks),
                'execution_time': execution_time
            })
            
            result = FilteringResult(
                selected_stocks=final_stocks,
                rejected_stocks=all_rejected,
                total_processed=len(stocks),
                stage1_passed=len(stage1_stocks),
                stage2_passed=len(stage2_stocks),
                final_selected=len(final_stocks),
                execution_time=execution_time,
                config_used=self.config
            )
            
            logger.info(f"Enhanced filtering completed in {execution_time:.2f}s")
            logger.info(f"Final result: {len(final_stocks)} stocks selected from {len(stocks)} processed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced stock filtering: {e}")
            # Return empty result on error
            return FilteringResult(
                selected_stocks=[],
                rejected_stocks=[],
                total_processed=len(stocks),
                stage1_passed=0,
                stage2_passed=0,
                final_selected=0,
                execution_time=time.time() - start_time,
                config_used=self.config
            )
    
    def _apply_stage1_filters(self, stocks: List[Any]) -> Tuple[List[StockScore], List[StockScore]]:
        """Apply Stage 1 basic filtering criteria."""
        passed_stocks = []
        rejected_stocks = []
        
        stage1_config = self.config.stage_1_filters
        
        for stock in stocks:
            try:
                score = StockScore(symbol=getattr(stock, 'symbol', 'UNKNOWN'))
                score.stock_object = stock  # Add stock object for scoring calculations
                reject_reasons = []
                
                # Price range filter
                current_price = getattr(stock, 'current_price', 0) or 0
                if current_price <= 0:
                    reject_reasons.append(f"Invalid price {current_price}")
                elif current_price < stage1_config.minimum_price:
                    reject_reasons.append(f"Price {current_price} below minimum {stage1_config.minimum_price}")
                elif current_price > stage1_config.maximum_price:
                    reject_reasons.append(f"Price {current_price} above maximum {stage1_config.maximum_price}")
                else:
                    score.filters_passed.append("price_range")
                
                # Volume and turnover filter
                volume = getattr(stock, 'volume', 0) or 0
                daily_turnover = current_price * volume if current_price and volume else 0
                
                if daily_turnover < stage1_config.minimum_daily_turnover_inr:
                    if volume < stage1_config.fallback_minimum_volume:
                        reject_reasons.append(f"Volume {volume} below fallback minimum {stage1_config.fallback_minimum_volume}")
                    else:
                        score.filters_passed.append("volume_fallback")
                else:
                    score.filters_passed.append("turnover")
                
                # Liquidity score filter (use volume as proxy if liquidity_score not available)
                liquidity_score = getattr(stock, 'liquidity_score', None)
                if liquidity_score is None:
                    # Use volume as proxy for liquidity
                    liquidity_score = min(1.0, volume / 1000000) if volume > 0 else 0
                
                if liquidity_score < stage1_config.minimum_liquidity_score:
                    reject_reasons.append(f"Liquidity score {liquidity_score} below minimum {stage1_config.minimum_liquidity_score}")
                else:
                    score.filters_passed.append("liquidity")
                
                # Trading status filter
                is_active = getattr(stock, 'is_active', False)
                is_tradeable = getattr(stock, 'is_tradeable', False)
                
                if not is_active:
                    reject_reasons.append("Stock not active")
                elif not is_tradeable:
                    reject_reasons.append("Stock not tradeable")
                else:
                    score.filters_passed.append("trading_status")
                
                # ATR volatility filter
                atr_percentage = getattr(stock, 'atr_percentage', None)
                if atr_percentage is None:
                    # Skip ATR filter if not available
                    score.filters_passed.append("atr_volatility_skip")
                elif atr_percentage < stage1_config.min_atr_pct_of_price:
                    reject_reasons.append(f"ATR {atr_percentage}% below minimum {stage1_config.min_atr_pct_of_price}%")
                elif atr_percentage > stage1_config.max_atr_pct_of_price:
                    reject_reasons.append(f"ATR {atr_percentage}% above maximum {stage1_config.max_atr_pct_of_price}%")
                else:
                    score.filters_passed.append("atr_volatility")
                
                # Apply rejection logic
                if reject_reasons:
                    score.reject_reasons = reject_reasons
                    rejected_stocks.append(score)
                else:
                    passed_stocks.append(score)
                    
            except Exception as e:
                logger.warning(f"Error processing stock {getattr(stock, 'symbol', 'UNKNOWN')} in Stage 1: {e}")
                score = StockScore(symbol=getattr(stock, 'symbol', 'UNKNOWN'))
                score.reject_reasons = [f"Processing error: {str(e)}"]
                rejected_stocks.append(score)
        
        return passed_stocks, rejected_stocks
    
    def _apply_stage2_filters(self, stocks: List[StockScore]) -> Tuple[List[StockScore], List[StockScore]]:
        """Apply Stage 2 advanced analysis and scoring."""
        passed_stocks = []
        rejected_stocks = []
        
        for stock_score in stocks:
            try:
                # Calculate technical score
                technical_score = self._calculate_technical_score(stock_score)
                stock_score.technical_score = technical_score
                
                # Calculate fundamental score
                fundamental_score = self._calculate_fundamental_score(stock_score)
                stock_score.fundamental_score = fundamental_score
                
                # Calculate risk score
                risk_score = self._calculate_risk_score(stock_score)
                stock_score.risk_score = risk_score
                
                # Calculate momentum score
                momentum_score = self._calculate_momentum_score(stock_score)
                stock_score.momentum_score = momentum_score
                
                # Calculate volume score
                volume_score = self._calculate_volume_score(stock_score)
                stock_score.volume_score = volume_score
                
                # Calculate total weighted score
                weights = self.config.scoring_weights
                total_score = (
                    technical_score * weights.technical_score +
                    fundamental_score * weights.fundamental_score +
                    risk_score * weights.risk_score +
                    momentum_score * weights.momentum_score +
                    volume_score * weights.volume_score
                ) * 100  # Convert to 0-100 scale
                
                # Convert individual scores to 0-100 scale for comparison
                technical_score = technical_score * 100
                fundamental_score = fundamental_score * 100
                risk_score = risk_score * 100
                momentum_score = momentum_score * 100
                volume_score = volume_score * 100
                
                stock_score.total_score = total_score
                
                # Apply filtering thresholds
                thresholds = self.config.filtering_thresholds
                reject_reasons = []
                
                # Debug logging for first few stocks
                if len(passed_stocks) < 3:
                    logger.info(f"Stock {stock_score.symbol} scores: total={total_score:.1f}, technical={technical_score:.1f}, fundamental={fundamental_score:.1f}, risk={risk_score:.1f}")
                
                if total_score < thresholds.minimum_total_score:
                    reject_reasons.append(f"Total score {total_score:.1f} below minimum {thresholds.minimum_total_score}")
                
                if technical_score < thresholds.minimum_technical_score:
                    reject_reasons.append(f"Technical score {technical_score:.1f} below minimum {thresholds.minimum_technical_score}")
                
                if fundamental_score < thresholds.minimum_fundamental_score:
                    reject_reasons.append(f"Fundamental score {fundamental_score:.1f} below minimum {thresholds.minimum_fundamental_score}")
                
                if risk_score < thresholds.minimum_risk_score:
                    reject_reasons.append(f"Risk score {risk_score:.1f} below minimum {thresholds.minimum_risk_score}")
                
                # Apply rejection logic
                if reject_reasons:
                    stock_score.reject_reasons.extend(reject_reasons)
                    rejected_stocks.append(stock_score)
                else:
                    stock_score.filters_passed.append("stage2_scoring")
                    passed_stocks.append(stock_score)
                    
            except Exception as e:
                logger.warning(f"Error processing stock {stock_score.symbol} in Stage 2: {e}")
                stock_score.reject_reasons.append(f"Stage 2 processing error: {str(e)}")
                rejected_stocks.append(stock_score)
        
        return passed_stocks, rejected_stocks
    
    def _apply_final_selection(self, stocks: List[StockScore]) -> Tuple[List[StockScore], List[StockScore]]:
        """Apply final selection with guardrails."""
        if not stocks:
            return [], []
        
        selection_config = self.config.selection
        
        # Sort by total score (descending)
        sorted_stocks = sorted(stocks, key=lambda x: x.total_score, reverse=True)
        
        # Apply tie-breaker priority
        if selection_config.tie_breaker_priority:
            # Additional sorting by tie-breaker criteria
            for criteria in reversed(selection_config.tie_breaker_priority):
                if criteria == "momentum_score":
                    sorted_stocks.sort(key=lambda x: x.momentum_score, reverse=True)
                elif criteria == "risk_score":
                    sorted_stocks.sort(key=lambda x: x.risk_score, reverse=True)
                elif criteria == "technical_score":
                    sorted_stocks.sort(key=lambda x: x.technical_score, reverse=True)
        
        # Apply maximum selection limit
        max_stocks = selection_config.max_suggested_stocks
        selected_stocks = sorted_stocks[:max_stocks]
        
        # Apply sector concentration limit
        selected_stocks = self._apply_sector_concentration_limit(selected_stocks)
        
        # Apply market cap mix requirement
        selected_stocks = self._apply_market_cap_mix(selected_stocks)
        
        # Apply blacklist/whitelist
        selected_stocks = self._apply_blacklist_whitelist(selected_stocks)
        
        # Apply resistance distance filter
        selected_stocks = self._apply_resistance_distance_filter(selected_stocks)
        
        # Determine rejected stocks
        rejected_stocks = []
        for stock in stocks:
            if stock not in selected_stocks:
                stock.reject_reasons.append("Final selection limit or guardrails")
                rejected_stocks.append(stock)
        
        return selected_stocks, rejected_stocks
    
    def _calculate_technical_score(self, stock_score: StockScore) -> float:
        """Calculate technical analysis score (0-1) based on real data."""
        try:
            # Get the stock object from the score
            stock = getattr(stock_score, 'stock_object', None)
            if not stock:
                return 0.0
            
            score = 0.0
            factors = 0
            
            # ATR-based volatility score (lower ATR% is better for stability)
            atr_pct = getattr(stock, 'atr_percentage', None)
            if atr_pct is not None and atr_pct > 0:
                # Score inversely to ATR% - lower ATR% gets higher score
                # More selective: only 0-3% ATR gets high score
                atr_score = max(0, min(1, (3 - atr_pct) / 3))  # 0-3% ATR range
                score += atr_score * 0.3
                factors += 0.3
            
            # Historical volatility score (lower volatility is better)
            hist_vol = getattr(stock, 'historical_volatility_1y', None)
            if hist_vol is not None and hist_vol > 0:
                # Score inversely to volatility - lower volatility gets higher score
                # More selective: only 0-30% volatility gets high score
                vol_score = max(0, min(1, (0.3 - hist_vol) / 0.3))  # 0-30% volatility range
                score += vol_score * 0.3
                factors += 0.3
            
            # Volume consistency score (higher average volume is better)
            avg_volume = getattr(stock, 'avg_daily_volume_20d', None)
            if avg_volume is not None and avg_volume > 0:
                # Score based on volume - higher volume gets higher score
                vol_score = min(1, avg_volume / 1000000)  # Normalize to 1M volume
                score += vol_score * 0.2
                factors += 0.2
            
            # Bid-ask spread score (lower spread is better)
            spread = getattr(stock, 'bid_ask_spread', None)
            if spread is not None and spread > 0:
                # Score inversely to spread - lower spread gets higher score
                spread_score = max(0, min(1, (0.1 - spread) / 0.1))  # 0-10% spread range
                score += spread_score * 0.2
                factors += 0.2
            
            # Normalize by factors used
            if factors > 0:
                return score / factors
            else:
                return 0.2  # Low score if no data available (penalize missing data)
                
        except Exception as e:
            logger.warning(f"Error calculating technical score: {e}")
            return 0.2
    
    def _calculate_fundamental_score(self, stock_score: StockScore) -> float:
        """Calculate fundamental analysis score (0-1) based on real data."""
        try:
            # Get the stock object from the score
            stock = getattr(stock_score, 'stock_object', None)
            if not stock:
                return 0.0
            
            score = 0.0
            factors = 0
            
            # P/E ratio score (moderate P/E is better)
            pe_ratio = getattr(stock, 'pe_ratio', None)
            if pe_ratio is not None and pe_ratio > 0:
                # Score based on P/E - 10-25 range gets highest score
                if 10 <= pe_ratio <= 25:
                    pe_score = 1.0
                elif pe_ratio < 10:
                    pe_score = pe_ratio / 10  # Lower P/E gets lower score
                else:
                    pe_score = max(0, (50 - pe_ratio) / 25)  # Higher P/E gets lower score
                score += pe_score * 0.3
                factors += 0.3
            
            # P/B ratio score (moderate P/B is better)
            pb_ratio = getattr(stock, 'pb_ratio', None)
            if pb_ratio is not None and pb_ratio > 0:
                # Score based on P/B - 1-3 range gets highest score
                if 1 <= pb_ratio <= 3:
                    pb_score = 1.0
                elif pb_ratio < 1:
                    pb_score = pb_ratio  # Lower P/B gets lower score
                else:
                    pb_score = max(0, (6 - pb_ratio) / 3)  # Higher P/B gets lower score
                score += pb_score * 0.3
                factors += 0.3
            
            # ROE score (higher ROE is better)
            roe = getattr(stock, 'roe', None)
            if roe is not None:
                # Score based on ROE - 15%+ gets highest score
                roe_score = min(1, roe / 20)  # Normalize to 20% ROE
                score += roe_score * 0.2
                factors += 0.2
            
            # Debt-to-equity score (lower D/E is better)
            debt_equity = getattr(stock, 'debt_to_equity', None)
            if debt_equity is not None and debt_equity >= 0:
                # Score inversely to D/E - lower D/E gets higher score
                de_score = max(0, min(1, (2 - debt_equity) / 2))  # 0-2 D/E range
                score += de_score * 0.2
                factors += 0.2
            
            # Normalize by factors used
            if factors > 0:
                return score / factors
            else:
                return 0.1  # Very low score if no fundamental data available
                
        except Exception as e:
            logger.warning(f"Error calculating fundamental score: {e}")
            return 0.1
    
    def _calculate_risk_score(self, stock_score: StockScore) -> float:
        """Calculate risk assessment score (0-1) based on real data."""
        try:
            # Get the stock object from the score
            stock = getattr(stock_score, 'stock_object', None)
            if not stock:
                return 0.0
            
            score = 0.0
            factors = 0
            
            # Beta score (moderate beta is better)
            beta = getattr(stock, 'beta', None)
            if beta is not None:
                # Score based on beta - 0.8-1.2 range gets highest score
                if 0.8 <= beta <= 1.2:
                    beta_score = 1.0
                elif beta < 0.8:
                    beta_score = beta / 0.8  # Lower beta gets lower score
                else:
                    beta_score = max(0, (2 - beta) / 0.8)  # Higher beta gets lower score
                score += beta_score * 0.4
                factors += 0.4
            
            # Historical volatility score (lower volatility is better)
            hist_vol = getattr(stock, 'historical_volatility_1y', None)
            if hist_vol is not None and hist_vol > 0:
                # Score inversely to volatility - lower volatility gets higher score
                vol_score = max(0, min(1, (0.3 - hist_vol) / 0.3))  # 0-30% volatility range
                score += vol_score * 0.3
                factors += 0.3
            
            # Market cap category score (larger cap is better for risk)
            market_cap_cat = getattr(stock, 'market_cap_category', None)
            if market_cap_cat:
                if market_cap_cat == 'large_cap':
                    cap_score = 1.0
                elif market_cap_cat == 'mid_cap':
                    cap_score = 0.7
                elif market_cap_cat == 'small_cap':
                    cap_score = 0.4
                else:
                    cap_score = 0.5
                score += cap_score * 0.3
                factors += 0.3
            
            # Normalize by factors used
            if factors > 0:
                return score / factors
            else:
                return 0.2  # Low score if no data available (penalize missing data)
                
        except Exception as e:
            logger.warning(f"Error calculating risk score: {e}")
            return 0.2
    
    def _calculate_momentum_score(self, stock_score: StockScore) -> float:
        """Calculate momentum score (0-1) based on real data."""
        try:
            # Get the stock object from the score
            stock = getattr(stock_score, 'stock_object', None)
            if not stock:
                return 0.0
            
            score = 0.0
            factors = 0
            
            # Volume momentum score (higher recent volume is better)
            current_volume = getattr(stock, 'volume', 0) or 0
            avg_volume = getattr(stock, 'avg_daily_volume_20d', None)
            if current_volume > 0 and avg_volume and avg_volume > 0:
                # Score based on volume ratio - higher ratio gets higher score
                volume_ratio = current_volume / avg_volume
                momentum_score = min(1, volume_ratio / 2)  # Normalize to 2x average
                score += momentum_score * 0.5
                factors += 0.5
            
            # Price momentum (using current price vs market cap as proxy)
            current_price = getattr(stock, 'current_price', 0) or 0
            market_cap = getattr(stock, 'market_cap', None)
            if current_price > 0 and market_cap and market_cap > 0:
                # Score based on price stability - moderate price gets higher score
                if 50 <= current_price <= 500:
                    price_score = 1.0
                elif current_price < 50:
                    price_score = current_price / 50
                else:
                    price_score = max(0, (1000 - current_price) / 500)
                score += price_score * 0.5
                factors += 0.5
            
            # Normalize by factors used
            if factors > 0:
                return score / factors
            else:
                return 0.2  # Low score if no data available (penalize missing data)
                
        except Exception as e:
            logger.warning(f"Error calculating momentum score: {e}")
            return 0.2
    
    def _calculate_volume_score(self, stock_score: StockScore) -> float:
        """Calculate volume analysis score (0-1) based on real data."""
        try:
            # Get the stock object from the score
            stock = getattr(stock_score, 'stock_object', None)
            if not stock:
                return 0.0
            
            score = 0.0
            factors = 0
            
            # Average volume score (higher volume is better)
            avg_volume = getattr(stock, 'avg_daily_volume_20d', None)
            if avg_volume and avg_volume > 0:
                # Score based on volume - higher volume gets higher score
                vol_score = min(1, avg_volume / 500000)  # Normalize to 500K volume
                score += vol_score * 0.4
                factors += 0.4
            
            # Average turnover score (higher turnover is better)
            avg_turnover = getattr(stock, 'avg_daily_turnover', None)
            if avg_turnover and avg_turnover > 0:
                # Score based on turnover - higher turnover gets higher score
                turnover_score = min(1, avg_turnover / 100)  # Normalize to 100 crores
                score += turnover_score * 0.3
                factors += 0.3
            
            # Trades per day score (higher trades is better)
            trades_per_day = getattr(stock, 'trades_per_day', None)
            if trades_per_day and trades_per_day > 0:
                # Score based on trades - higher trades gets higher score
                trades_score = min(1, trades_per_day / 10000)  # Normalize to 10K trades
                score += trades_score * 0.3
                factors += 0.3
            
            # Normalize by factors used
            if factors > 0:
                return score / factors
            else:
                return 0.2  # Low score if no data available (penalize missing data)
                
        except Exception as e:
            logger.warning(f"Error calculating volume score: {e}")
            return 0.2
    
    def _apply_sector_concentration_limit(self, stocks: List[StockScore]) -> List[StockScore]:
        """Apply sector concentration limit."""
        # This is a simplified implementation
        # In a real implementation, you would group by sector and apply limits
        return stocks
    
    def _apply_market_cap_mix(self, stocks: List[StockScore]) -> List[StockScore]:
        """Apply market cap mix requirements."""
        # This is a simplified implementation
        # In a real implementation, you would ensure minimum large/mid-cap percentage
        return stocks
    
    def _apply_blacklist_whitelist(self, stocks: List[StockScore]) -> List[StockScore]:
        """Apply blacklist and whitelist filters."""
        selection_config = self.config.selection
        
        # Apply blacklist
        if selection_config.blacklist_symbols:
            stocks = [s for s in stocks if s.symbol not in selection_config.blacklist_symbols]
        
        # Apply whitelist (if specified, only include whitelisted symbols)
        if selection_config.whitelist_symbols:
            stocks = [s for s in stocks if s.symbol in selection_config.whitelist_symbols]
        
        return stocks
    
    def _apply_resistance_distance_filter(self, stocks: List[StockScore]) -> List[StockScore]:
        """Apply resistance distance filter."""
        # This is a simplified implementation
        # In a real implementation, you would calculate distance from resistance levels
        return stocks
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get current filter statistics."""
        return self.filter_stats.copy()
    
    def reset_statistics(self):
        """Reset filter statistics."""
        self.filter_stats = {
            'total_processed': 0,
            'stage1_passed': 0,
            'stage2_passed': 0,
            'final_selected': 0,
            'execution_time': 0.0
        }


# Global service instance
_enhanced_filtering_service = None


def get_enhanced_filtering_service() -> EnhancedStockFilteringService:
    """Get the global enhanced filtering service instance."""
    global _enhanced_filtering_service
    if _enhanced_filtering_service is None:
        _enhanced_filtering_service = EnhancedStockFilteringService()
    return _enhanced_filtering_service
