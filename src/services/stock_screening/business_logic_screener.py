"""
Business Logic Screener - Advanced Risk-Based Strategy Filtering

Implements comprehensive multi-strategy business logic screening with:
- Default Risk (Balanced): Large/Mid-cap focused with conservative filters
- High Risk (Aggressive Momentum): Mid/Small-cap focused with momentum filters
- Trend analysis, momentum indicators, volume participation
- Risk management and portfolio allocation rules
"""

import logging
import os
from typing import List, Dict, Any, Optional
from enum import Enum
import random

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Risk-based strategy types for swing trading."""
    DEFAULT_RISK = "default_risk"     # Balanced: 50-60% Large-cap, 40-50% Mid/Small-cap
    HIGH_RISK = "high_risk"           # Aggressive: 50% Mid-cap, 50% Small-cap


class BusinessLogicScreener:
    """
    Advanced Risk-Based Strategy Screener

    Implements two distinct risk profiles with comprehensive technical and fundamental filters:

    📊 DEFAULT RISK (Balanced Strategy):
    - Allocation: 50-60% Large-cap, 40-50% Mid+Small-cap
    - Trend: Close > SMA50 > SMA200, Close > EMA20
    - Momentum: RSI 45-70, MACD > Signal
    - Volatility: ATR ≤ 4-5%, Beta ≤ 1.0-1.2
    - Volume: Breakout ≥ 1.5x 20-day avg
    - Risk: 1-2% per trade, Stop at 1.5x ATR

    🚀 HIGH RISK (Aggressive Momentum):
    - Allocation: 50% Mid-cap, 50% Small-cap (No large-cap)
    - Trend: Close > SMA50 ≥ SMA200
    - Momentum: RSI 50-75, MACD rising 3-5 bars, Relative Strength > 0
    - Volatility: ATR ≤ 6-8%, Beta ≤ 1.5
    - Volume: Breakout ≥ 2x avg, Pullback ≥ 1.5x avg
    - Risk: 1-2% per trade, Stop at 2x ATR
    """

    def __init__(self):
        # Global configuration from environment
        self.config = {
            'max_sector_allocation': float(os.getenv('SCREENING_MAX_SECTOR_ALLOCATION', '30.0')),
            'min_market_cap_crores': int(os.getenv('SCREENING_MIN_MARKET_CAP_CRORES', '500')),
            'max_pe_ratio': float(os.getenv('SCREENING_MAX_PE_RATIO', '50.0')),
            'min_roe': float(os.getenv('SCREENING_MIN_ROE', '5.0')),
            'max_debt_equity': float(os.getenv('SCREENING_MAX_DEBT_EQUITY', '2.0')),
            'max_position_size': float(os.getenv('SCREENING_MAX_POSITION_SIZE', '5.0')),
            'max_final_stocks': int(os.getenv('SCREENING_MAX_FINAL_STOCKS', '50'))
        }

        # Market cap categories (using existing market_cap_category field)
        self.market_cap_categories = ['large_cap', 'mid_cap', 'small_cap']

        # Strategy-specific criteria with comprehensive technical filters
        self.strategy_criteria = {
            StrategyType.DEFAULT_RISK: {
                # Market Cap Allocation
                'large_cap_allocation': (0.5, 0.6),    # 50-60%
                'mid_cap_allocation': (0.25, 0.35),    # 25-35%
                'small_cap_allocation': (0.1, 0.25),   # 10-25%

                # Volatility Controls
                'max_atr_percentage': 5.0,              # ATR ≤ 5%
                'max_beta': 1.2,                        # Beta ≤ 1.2
                'min_beta': 0.3,                        # Avoid too defensive

                # Trend Filters (simulated technical indicators)
                'trend_strength_min': 0.6,              # Strong trend required
                'momentum_strength_min': 0.5,           # Moderate momentum

                # Volume Participation
                'volume_surge_multiplier': 1.5,         # Breakout volume ≥ 1.5x avg

                # Risk Management
                'max_position_risk': 0.02,              # 2% max risk per trade
                'stop_loss_atr_multiplier': 1.5         # Stop at 1.5x ATR
            },

            StrategyType.HIGH_RISK: {
                # Market Cap Allocation (No Large-cap)
                'large_cap_allocation': (0.0, 0.0),     # 0% Large-cap
                'mid_cap_allocation': (0.45, 0.55),     # 45-55% Mid-cap
                'small_cap_allocation': (0.45, 0.55),   # 45-55% Small-cap

                # Volatility Controls (Higher tolerance)
                'max_atr_percentage': 8.0,              # ATR ≤ 8%
                'max_beta': 1.5,                        # Beta ≤ 1.5
                'min_beta': 0.7,                        # Prefer more responsive stocks

                # Trend Filters (More aggressive)
                'trend_strength_min': 0.5,              # Moderate trend OK
                'momentum_strength_min': 0.7,           # Strong momentum required

                # Volume Participation (Higher requirements)
                'volume_surge_multiplier': 2.0,         # Breakout volume ≥ 2x avg
                'pullback_volume_multiplier': 1.5,      # Pullback volume ≥ 1.5x avg

                # Additional momentum requirements
                'rsi_min': 50,                          # RSI ≥ 50
                'rsi_max': 75,                          # RSI ≤ 75
                'relative_strength_min': 0.0,           # Outperforming market

                # Risk Management
                'max_position_risk': 0.02,              # 2% max risk per trade
                'stop_loss_atr_multiplier': 2.0         # Stop at 2x ATR
            }
        }

    def screen_stocks(self, market_data_candidates: List, strategies: List[StrategyType]) -> List:
        """
        Apply advanced risk-based business logic screening.

        Args:
            market_data_candidates: Stocks that passed Stage 1 (Market Data Screening)
            strategies: List of strategy types to apply (DEFAULT_RISK, HIGH_RISK, or both)

        Returns:
            List of stocks optimized for specified risk strategies
        """
        print(f"🎯 STAGE 2: BUSINESS LOGIC SCREENING (RISK-BASED STRATEGIES)")
        print(f"=" * 70)
        print(f"   📊 Input: {len(market_data_candidates)} candidates from Stage 1")
        print(f"   🎯 Strategies: {[s.value for s in strategies] if strategies else ['default_risk']}")
        print()

        if not strategies:
            strategies = [StrategyType.DEFAULT_RISK]

        # Initialize results tracking
        self.results = {
            'total_input': len(market_data_candidates),
            'sector_filtered': 0,
            'fundamental_filtered': 0,
            'technical_filtered': 0,
            'allocation_optimized': 0,
            'strategy_portfolios': {},
            'combined_portfolio': []
        }

        # Step 1: Fundamental Screening (Quality Filter)
        print(f"📈 STAGE 2 STEP 1: Fundamental Screening")
        print(f"-" * 45)
        fundamental_qualified = self._apply_fundamental_screening(market_data_candidates)
        print(f"   ✅ {len(fundamental_qualified)} stocks passed fundamental screening")
        print()

        # Step 2: Technical Analysis Screening (Momentum Filter)
        print(f"🔍 STAGE 2 STEP 2: Technical Analysis Screening")
        print(f"-" * 45)
        technical_qualified = self._apply_technical_screening(fundamental_qualified)
        print(f"   ✅ {len(technical_qualified)} stocks passed technical screening")
        print()

        # Step 3: Strategy-Specific Portfolio Allocation
        print(f"🎯 STAGE 2 STEP 3: Strategy-Specific Portfolio Allocation")
        print(f"-" * 45)
        strategy_portfolios = self._apply_strategy_allocation(technical_qualified, strategies)
        print(f"   ✅ Portfolio allocation completed for {len(strategies)} strategy/strategies")
        print()

        # Step 4: Sector Allocation Control (Diversification Filter)
        print(f"📊 STAGE 2 STEP 4: Sector Allocation Control")
        print(f"-" * 45)
        # Combine all strategy portfolio stocks for sector filtering
        combined_strategy_stocks = []
        for strategy, stocks in strategy_portfolios.items():
            combined_strategy_stocks.extend(stocks)

        sector_controlled = self._apply_sector_allocation_control(combined_strategy_stocks)
        print(f"   ✅ {len(sector_controlled)} stocks passed sector allocation control")
        print()

        # Update strategy portfolios with sector-filtered stocks
        filtered_strategy_portfolios = {}
        for strategy, stocks in strategy_portfolios.items():
            filtered_stocks = [stock for stock in stocks if stock in sector_controlled]
            filtered_strategy_portfolios[strategy] = filtered_stocks

        # Step 5: Final Portfolio Optimization
        print(f"⚖️ STAGE 2 STEP 5: Final Portfolio Optimization")
        print(f"-" * 45)
        final_portfolio = self._optimize_final_portfolio(filtered_strategy_portfolios, strategies)
        print(f"   ✅ {len(final_portfolio)} stocks in final optimized portfolio")
        print()

        self.results['combined_portfolio'] = final_portfolio

        # Comprehensive results logging
        self._log_comprehensive_results(strategy_portfolios)
        self._log_portfolio_allocation_analysis(final_portfolio, strategies)

        print(f"✅ STAGE 2 COMPLETE: {len(final_portfolio)} stocks ready for swing trading")
        print(f"=" * 70)
        print()

        # Store final portfolio for metrics retrieval
        self.last_final_portfolio = final_portfolio

        logger.info(f"✅ Advanced risk-based screening complete: {len(final_portfolio)} stocks across {len(strategies)} strategies")
        return final_portfolio

    # ========================================
    # ADVANCED RISK-BASED SCREENING METHODS
    # ========================================

    def _apply_sector_allocation_control(self, stocks: List) -> List:
        """Apply sector allocation limits to prevent over-concentration."""
        sector_counts = {}
        for stock in stocks:
            sector = stock.sector or 'Unknown'
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        print(f"      📈 Current sector distribution:")
        for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / len(stocks)) * 100
            print(f"         {sector}: {count} stocks ({percentage:.1f}%)")

        # Apply sector diversification limits
        max_stocks_per_sector = max(1, int(len(stocks) * self.config['max_sector_allocation'] / 100))
        print(f"      🎯 Max stocks per sector: {max_stocks_per_sector} ({self.config['max_sector_allocation']}% limit)")

        sector_selected = {}
        sector_filtered_stocks = []
        sector_skipped = {}

        for stock in stocks:
            sector = stock.sector or 'Unknown'
            current_count = sector_selected.get(sector, 0)

            if current_count < max_stocks_per_sector:
                sector_selected[sector] = current_count + 1
                sector_filtered_stocks.append(stock)
            else:
                self.results['sector_filtered'] += 1
                # Track skipped stocks by sector for detailed logging
                if sector not in sector_skipped:
                    sector_skipped[sector] = []
                sector_skipped[sector].append(stock.symbol)

        print(f"      ✅ Sector control result: {len(sector_filtered_stocks)} stocks passed")

        # Log detailed information about skipped stocks
        total_skipped = sum(len(symbols) for symbols in sector_skipped.values())
        if total_skipped > 0:
            print(f"      🚫 {total_skipped} stocks skipped due to sector limits:")
            for sector, symbols in sector_skipped.items():
                if len(symbols) > 0:
                    print(f"         {sector}: {len(symbols)} stocks skipped")
                    # Show first few skipped symbols for each sector
                    sample_symbols = symbols[:5]
                    if len(symbols) > 5:
                        print(f"            Sample: {', '.join(sample_symbols)} (+{len(symbols)-5} more)")
                    else:
                        print(f"            Skipped: {', '.join(sample_symbols)}")

        return sector_filtered_stocks

    def _apply_fundamental_screening(self, stocks: List) -> List:
        """Apply fundamental analysis filters for quality stocks."""
        print(f"      📊 Applying fundamental quality filters...")

        fundamental_stocks = []
        excluded_stocks = []
        exclusion_reasons = {}

        for stock in stocks:
            # STRICT Quality checks - require fundamental data to be present and valid
            exclude_reasons = []

            # Require basic volatility data (from Stage 1)
            if not stock.atr_percentage or stock.atr_percentage <= 0:
                exclude_reasons.append("Missing ATR data")
            elif stock.atr_percentage > self.config.get('max_atr_percentage', 5.0):
                exclude_reasons.append(f"High ATR: {stock.atr_percentage:.1f}%")

            # Require beta data
            if not stock.beta:
                exclude_reasons.append("Missing Beta data")
            elif stock.beta > self.config.get('max_beta', 1.5):
                exclude_reasons.append(f"High Beta: {stock.beta:.2f}")

            # P/E Ratio check - only filter if data exists and is extreme
            if stock.pe_ratio and (stock.pe_ratio <= 0 or stock.pe_ratio > self.config['max_pe_ratio']):
                exclude_reasons.append(f"Invalid P/E: {stock.pe_ratio:.1f}")

            # ROE check - only filter if data exists and is poor
            if stock.roe and stock.roe < self.config['min_roe']:
                exclude_reasons.append(f"Low ROE: {stock.roe:.1f}%")

            # Market cap category check - require classification
            if not hasattr(stock, 'market_cap_category') or not stock.market_cap_category:
                exclude_reasons.append("Missing market cap classification")

            # Current price check - require reasonable price
            if not stock.current_price or stock.current_price <= 0:
                exclude_reasons.append("Invalid current price")
            elif stock.current_price > 1000:  # Filter extremely high-priced stocks
                exclude_reasons.append(f"Very high price: ₹{stock.current_price:.0f}")

            if exclude_reasons:
                excluded_stocks.append(stock.symbol)
                exclusion_reasons[stock.symbol] = exclude_reasons
                self.results['fundamental_filtered'] += 1
            else:
                fundamental_stocks.append(stock)

        self.results['fundamental_filtered'] = len(excluded_stocks)
        print(f"      ✅ Fundamental screening result: {len(fundamental_stocks)} stocks passed")

        # Log detailed information about excluded stocks
        if excluded_stocks:
            print(f"      🚫 {len(excluded_stocks)} stocks excluded for fundamental reasons:")
            # Group by exclusion reason for better readability
            reason_groups = {}
            for symbol in excluded_stocks:
                reasons = exclusion_reasons[symbol]
                for reason in reasons:
                    if reason not in reason_groups:
                        reason_groups[reason] = []
                    reason_groups[reason].append(symbol)

            for reason, symbols in reason_groups.items():
                print(f"         {reason}: {len(symbols)} stocks")
                sample_symbols = symbols[:3]
                if len(symbols) > 3:
                    print(f"            Sample: {', '.join(sample_symbols)} (+{len(symbols)-3} more)")
                else:
                    print(f"            Excluded: {', '.join(sample_symbols)}")

        return fundamental_stocks

    def _apply_technical_screening(self, stocks: List) -> List:
        """Apply technical analysis filters with simulated indicators."""
        print(f"      🔍 Applying technical analysis filters...")

        technical_stocks = []
        excluded_stocks = []
        technical_scores = {}

        for stock in stocks:
            # STRICT Technical analysis based on real volatility data
            exclude_reasons = []
            technical_score = self._calculate_technical_score(stock)

            # Require minimum technical strength (increased threshold)
            if technical_score < 0.65:  # Much stricter threshold
                exclude_reasons.append(f"Low technical score: {technical_score:.2f}")

            # Volume requirements - require reasonable volume
            if not stock.volume or stock.volume < 100000:  # Minimum daily volume
                exclude_reasons.append(f"Low volume: {stock.volume or 0:,}")

            # ATR percentage check for momentum
            if stock.atr_percentage and (stock.atr_percentage < 1.0 or stock.atr_percentage > 8.0):
                exclude_reasons.append(f"ATR out of range: {stock.atr_percentage:.1f}%")

            # Beta check for reasonable volatility vs market
            if stock.beta and (stock.beta < 0.5 or stock.beta > 2.0):
                exclude_reasons.append(f"Beta out of range: {stock.beta:.2f}")

            if not exclude_reasons:
                # Add simulated technical data for display
                stock.technical_score = technical_score
                stock.simulated_rsi = self._simulate_rsi(stock)
                stock.simulated_trend_strength = self._simulate_trend_strength(stock)
                stock.simulated_volume_ratio = self._simulate_volume_ratio(stock)

                technical_stocks.append(stock)
            else:
                excluded_stocks.append(stock.symbol)
                technical_scores[stock.symbol] = {'score': technical_score, 'reasons': exclude_reasons}
                self.results['technical_filtered'] += 1

        print(f"      ✅ Technical screening result: {len(technical_stocks)} stocks passed")

        # Log detailed information about excluded stocks
        if excluded_stocks:
            print(f"      🚫 {len(excluded_stocks)} stocks excluded for technical reasons:")

            # Group by exclusion reason for better readability
            reason_groups = {}
            for symbol in excluded_stocks:
                data = technical_scores[symbol]
                reasons = data['reasons']
                for reason in reasons:
                    if reason not in reason_groups:
                        reason_groups[reason] = []
                    reason_groups[reason].append(symbol)

            for reason, symbols in reason_groups.items():
                print(f"         {reason}: {len(symbols)} stocks")
                sample_symbols = symbols[:3]
                if len(symbols) > 3:
                    print(f"            Sample: {', '.join(sample_symbols)} (+{len(symbols)-3} more)")
                else:
                    print(f"            Excluded: {', '.join(sample_symbols)}")

        return technical_stocks

    def _apply_strategy_allocation(self, stocks: List, strategies: List[StrategyType]) -> Dict:
        """Apply strategy-specific allocation rules."""
        print(f"      🎯 Applying strategy-specific allocation...")

        strategy_portfolios = {}

        for strategy in strategies:
            print(f"         📊 Processing {strategy.value} strategy...")

            # Classify stocks by market cap
            classified_stocks = self._classify_stocks_by_market_cap(stocks)

            # Show market cap classification breakdown
            total_classified = sum(len(stocks_list) for stocks_list in classified_stocks.values())
            print(f"            Market cap breakdown: {total_classified} stocks classified")
            for cap_type, cap_stocks in classified_stocks.items():
                if cap_stocks:
                    print(f"               {cap_type}: {len(cap_stocks)} stocks")

            # Apply strategy-specific filters and allocation
            strategy_stocks = self._build_strategy_portfolio(classified_stocks, strategy)

            strategy_portfolios[strategy] = strategy_stocks
            print(f"         ✅ {strategy.value}: {len(strategy_stocks)} stocks selected")

        self.results['strategy_portfolios'] = strategy_portfolios
        return strategy_portfolios

    def _optimize_final_portfolio(self, strategy_portfolios: Dict, strategies: List[StrategyType]) -> List:
        """Optimize final portfolio combining all strategies."""
        print(f"      ⚖️ Optimizing final portfolio...")

        all_portfolio_stocks = []

        # Combine all strategy portfolios
        for strategy, stocks in strategy_portfolios.items():
            for stock in stocks:
                if not hasattr(stock, 'assigned_strategies'):
                    stock.assigned_strategies = []
                stock.assigned_strategies.append(strategy)

                # Add to portfolio if not already included
                if stock not in all_portfolio_stocks:
                    all_portfolio_stocks.append(stock)

        # Sort by composite score for final selection
        optimized_stocks = sorted(
            all_portfolio_stocks,
            key=lambda s: self._calculate_composite_score(s),
            reverse=True
        )

        # Apply final size limit
        final_stocks = optimized_stocks[:self.config['max_final_stocks']]
        dropped_stocks = optimized_stocks[self.config['max_final_stocks']:]

        self.results['allocation_optimized'] = len(all_portfolio_stocks) - len(final_stocks)
        print(f"      ✅ Portfolio optimization result: {len(final_stocks)} final stocks")

        # Log information about dropped stocks
        if dropped_stocks:
            print(f"      🚫 {len(dropped_stocks)} stocks dropped in final optimization:")
            print(f"         Reason: Portfolio size limit ({self.config['max_final_stocks']} max)")

            # Show sample of dropped stocks with their composite scores
            sample_dropped = dropped_stocks[:5]
            for stock in sample_dropped:
                score = self._calculate_composite_score(stock)
                cap_category = getattr(stock, 'market_cap_category', 'Unknown')
                print(f"            {stock.symbol} ({cap_category}): composite score {score:.3f}")

            if len(dropped_stocks) > 5:
                print(f"            (+{len(dropped_stocks)-5} more with lower composite scores)")

        return final_stocks

    # ========================================
    # HELPER METHODS
    # ========================================

    def _classify_stocks_by_market_cap(self, stocks: List) -> Dict:
        """Classify stocks by existing market cap categories."""
        classified = {
            'large_cap': [],
            'mid_cap': [],
            'small_cap': [],
            'unknown': []
        }

        for stock in stocks:
            category = getattr(stock, 'market_cap_category', None)
            if category and category in self.market_cap_categories:
                classified[category].append(stock)
            else:
                classified['unknown'].append(stock)

        return classified

    def _build_strategy_portfolio(self, classified_stocks: Dict, strategy: StrategyType) -> List:
        """Build portfolio for specific strategy based on allocation rules."""
        criteria = self.strategy_criteria[strategy]
        portfolio_stocks = []

        # Calculate target allocations
        total_available = sum(len(stocks) for stocks in classified_stocks.values())
        if total_available == 0:
            return []

        target_large = int(total_available * criteria['large_cap_allocation'][1])
        target_mid = int(total_available * criteria['mid_cap_allocation'][1])
        target_small = int(total_available * criteria['small_cap_allocation'][1])

        print(f"            Strategy allocation targets:")
        print(f"               Large-cap: {target_large} stocks (max {criteria['large_cap_allocation'][1]*100:.0f}%)")
        print(f"               Mid-cap: {target_mid} stocks (max {criteria['mid_cap_allocation'][1]*100:.0f}%)")
        print(f"               Small-cap: {target_small} stocks (max {criteria['small_cap_allocation'][1]*100:.0f}%)")

        # Select stocks for each category
        if target_large > 0:
            large_cap_candidates = self._filter_stocks_by_strategy(classified_stocks['large_cap'], strategy)
            selected_large = large_cap_candidates[:target_large]
            portfolio_stocks.extend(selected_large)

            available_large = len(classified_stocks['large_cap'])
            filtered_large = len(large_cap_candidates)
            print(f"               Large-cap: {len(selected_large)}/{available_large} stocks selected ({available_large-filtered_large} filtered out)")

        if target_mid > 0:
            mid_cap_candidates = self._filter_stocks_by_strategy(classified_stocks['mid_cap'], strategy)
            selected_mid = mid_cap_candidates[:target_mid]
            portfolio_stocks.extend(selected_mid)

            available_mid = len(classified_stocks['mid_cap'])
            filtered_mid = len(mid_cap_candidates)
            print(f"               Mid-cap: {len(selected_mid)}/{available_mid} stocks selected ({available_mid-filtered_mid} filtered out)")

        if target_small > 0:
            small_cap_candidates = self._filter_stocks_by_strategy(classified_stocks['small_cap'], strategy)
            selected_small = small_cap_candidates[:target_small]
            portfolio_stocks.extend(selected_small)

            available_small = len(classified_stocks['small_cap'])
            filtered_small = len(small_cap_candidates)
            print(f"               Small-cap: {len(selected_small)}/{available_small} stocks selected ({available_small-filtered_small} filtered out)")

        return portfolio_stocks

    def _filter_stocks_by_strategy(self, stocks: List, strategy: StrategyType) -> List:
        """Filter stocks based on strategy-specific criteria."""
        criteria = self.strategy_criteria[strategy]
        filtered_stocks = []

        for stock in stocks:
            if self._meets_strategy_requirements(stock, strategy):
                filtered_stocks.append(stock)

        # Sort by strategy-specific scoring
        return sorted(filtered_stocks, key=lambda s: self._calculate_strategy_score(s, strategy), reverse=True)

    def _meets_strategy_requirements(self, stock, strategy: StrategyType) -> bool:
        """Check if stock meets strategy requirements."""
        criteria = self.strategy_criteria[strategy]

        # Volatility requirements
        if stock.atr_percentage:
            if stock.atr_percentage > criteria['max_atr_percentage']:
                return False

        if stock.beta:
            if stock.beta > criteria['max_beta'] or stock.beta < criteria['min_beta']:
                return False

        # Strategy-specific requirements
        if strategy == StrategyType.HIGH_RISK:
            # High risk requires stronger momentum
            if hasattr(stock, 'simulated_rsi'):
                if stock.simulated_rsi < criteria.get('rsi_min', 50) or stock.simulated_rsi > criteria.get('rsi_max', 75):
                    return False

        return True

    def _calculate_technical_score(self, stock) -> float:
        """Calculate technical analysis score based on available data."""
        score = 0.5  # Base score

        # Use real volatility data if available
        if stock.atr_percentage:
            # Lower ATR% is better for most strategies
            if stock.atr_percentage <= 3.0:
                score += 0.2
            elif stock.atr_percentage <= 5.0:
                score += 0.1

        if stock.beta:
            # Prefer moderate beta values
            if 0.8 <= stock.beta <= 1.3:
                score += 0.2
            elif 0.5 <= stock.beta <= 1.6:
                score += 0.1

        if stock.volume and stock.volume > 100000:
            score += 0.1

        return min(1.0, score)

    def _simulate_rsi(self, stock) -> float:
        """Simulate RSI based on volatility characteristics."""
        # Use price and volatility to simulate reasonable RSI
        base_rsi = 50.0

        if stock.atr_percentage:
            # Higher volatility might indicate oversold/overbought conditions
            volatility_factor = (stock.atr_percentage - 3.0) * 2
            base_rsi += volatility_factor

        # Add some controlled randomness
        random.seed(hash(stock.symbol) % 1000)
        rsi_noise = random.uniform(-5, 5)

        simulated_rsi = base_rsi + rsi_noise
        return max(10, min(90, simulated_rsi))

    def _simulate_trend_strength(self, stock) -> float:
        """Simulate trend strength based on beta and volatility."""
        trend_strength = 0.5

        if stock.beta:
            # Higher beta suggests stronger trends
            trend_strength += (stock.beta - 1.0) * 0.2

        if stock.atr_percentage:
            # Moderate volatility suggests good trend potential
            if 2.0 <= stock.atr_percentage <= 4.0:
                trend_strength += 0.2

        return max(0.1, min(1.0, trend_strength))

    def _simulate_volume_ratio(self, stock) -> float:
        """Simulate volume ratio based on actual volume data."""
        if not stock.volume:
            return 1.0

        # Simulate volume surge based on actual volume
        base_ratio = 1.0
        if stock.volume > 500000:
            base_ratio = 1.3
        elif stock.volume > 200000:
            base_ratio = 1.1

        return base_ratio

    def _calculate_strategy_score(self, stock, strategy: StrategyType) -> float:
        """Calculate stock score for specific strategy."""
        score = stock.technical_score if hasattr(stock, 'technical_score') else 0.5

        # Add strategy-specific scoring
        if strategy == StrategyType.DEFAULT_RISK:
            # Prefer stable, quality stocks
            if hasattr(stock, 'market_cap_category') and stock.market_cap_category == 'large_cap':
                score += 0.2
            if stock.beta and 0.8 <= stock.beta <= 1.1:
                score += 0.1

        elif strategy == StrategyType.HIGH_RISK:
            # Prefer momentum and growth potential
            if hasattr(stock, 'simulated_rsi') and 55 <= stock.simulated_rsi <= 70:
                score += 0.2
            if stock.atr_percentage and 4.0 <= stock.atr_percentage <= 7.0:
                score += 0.1

        return score

    def _calculate_composite_score(self, stock) -> float:
        """Calculate composite score for final ranking."""
        score = 0.0

        # Technical score
        if hasattr(stock, 'technical_score'):
            score += stock.technical_score * 0.4

        # Volume score
        if stock.volume:
            volume_score = min(1.0, stock.volume / 1000000)  # Normalize to millions
            score += volume_score * 0.2

        # Market cap score (larger = more stable)
        if hasattr(stock, 'market_cap_category'):
            # Score based on market cap category (large_cap = highest score)
            cap_scores = {'large_cap': 1.0, 'mid_cap': 0.7, 'small_cap': 0.4}
            cap_score = cap_scores.get(stock.market_cap_category, 0.2)
            score += cap_score * 0.2

        # Strategy diversification bonus
        if hasattr(stock, 'assigned_strategies') and len(stock.assigned_strategies) > 1:
            score += 0.2

        return score

    def _log_comprehensive_results(self, strategy_portfolios: Dict):
        """Log comprehensive screening results."""
        print(f"   📊 COMPREHENSIVE SCREENING RESULTS:")
        print(f"      📈 Input candidates: {self.results['total_input']}")
        print(f"      🚫 Sector filtered: {self.results['sector_filtered']}")
        print(f"      🚫 Fundamental filtered: {self.results['fundamental_filtered']}")
        print(f"      🚫 Technical filtered: {self.results['technical_filtered']}")
        print(f"      ⚖️ Portfolio optimized: {self.results['allocation_optimized']}")
        print()

        for strategy, stocks in strategy_portfolios.items():
            print(f"      🎯 {strategy.value} Portfolio: {len(stocks)} stocks")
            if stocks:
                # Show market cap distribution using category field
                cap_dist = {'large_cap': 0, 'mid_cap': 0, 'small_cap': 0, 'unknown': 0}
                for stock in stocks:
                    category = getattr(stock, 'market_cap_category', None)
                    if category in self.market_cap_categories:
                        cap_dist[category] += 1
                    else:
                        cap_dist['unknown'] += 1

                print(f"         Large-cap: {cap_dist['large_cap']}, Mid-cap: {cap_dist['mid_cap']}, Small-cap: {cap_dist['small_cap']}")

    def _log_portfolio_allocation_analysis(self, final_portfolio: List, strategies: List[StrategyType]):
        """Log detailed portfolio allocation analysis."""
        if not final_portfolio:
            return

        print(f"   📋 FINAL PORTFOLIO ALLOCATION ANALYSIS:")

        # Market cap distribution
        cap_distribution = {'large_cap': 0, 'mid_cap': 0, 'small_cap': 0, 'unknown': 0}
        sector_distribution = {}
        strategy_coverage = {}

        for stock in final_portfolio:
            # Market cap classification using existing category field
            category = getattr(stock, 'market_cap_category', None)
            if category in self.market_cap_categories:
                cap_distribution[category] += 1
            else:
                cap_distribution['unknown'] += 1

            # Sector distribution
            sector = stock.sector or 'Unknown'
            sector_distribution[sector] = sector_distribution.get(sector, 0) + 1

            # Strategy coverage
            if hasattr(stock, 'assigned_strategies'):
                for strategy in stock.assigned_strategies:
                    strategy_key = strategy.value if hasattr(strategy, 'value') else str(strategy)
                    strategy_coverage[strategy_key] = strategy_coverage.get(strategy_key, 0) + 1

        # Display results
        total_stocks = len(final_portfolio)

        print(f"      💰 Market Cap Distribution:")
        for cap_type, count in cap_distribution.items():
            if count > 0:
                percentage = (count / total_stocks) * 100
                print(f"         {cap_type.replace('_', ' ').title()}: {count} stocks ({percentage:.1f}%)")

        print(f"      🏭 Top Sector Distribution:")
        for sector, count in sorted(sector_distribution.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / total_stocks) * 100
            print(f"         {sector}: {count} stocks ({percentage:.1f}%)")

        print(f"      🎯 Strategy Coverage:")
        for strategy, count in strategy_coverage.items():
            percentage = (count / total_stocks) * 100
            print(f"         {strategy}: {count} stocks ({percentage:.1f}%)")

    def get_filter_stats(self) -> Dict[str, Any]:
        """Get detailed filter statistics."""
        return {
            'stage': 'AdvancedBusinessLogicScreening',
            'config': self.config,
            'market_cap_categories': self.market_cap_categories,
            'strategy_criteria': {k.value: v for k, v in self.strategy_criteria.items()},
            'results': self.results
        }

    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio metrics and allocation analysis."""
        if not hasattr(self, 'last_final_portfolio') or not self.last_final_portfolio:
            return {
                'total_stocks': 0,
                'market_cap_distribution': {},
                'sector_distribution': {},
                'strategy_coverage': {},
                'risk_metrics': {}
            }

        portfolio = self.last_final_portfolio
        total_stocks = len(portfolio)

        # Market cap distribution
        cap_distribution = {'large_cap': 0, 'mid_cap': 0, 'small_cap': 0, 'unknown': 0}
        sector_distribution = {}
        strategy_coverage = {}
        atr_values = []
        beta_values = []

        for stock in portfolio:
            # Market cap classification
            category = getattr(stock, 'market_cap_category', None)
            if category in self.market_cap_categories:
                cap_distribution[category] += 1
            else:
                cap_distribution['unknown'] += 1

            # Sector distribution
            sector = stock.sector or 'Unknown'
            sector_distribution[sector] = sector_distribution.get(sector, 0) + 1

            # Strategy coverage
            if hasattr(stock, 'assigned_strategies'):
                for strategy in stock.assigned_strategies:
                    strategy_coverage[strategy] = strategy_coverage.get(strategy, 0) + 1

            # Risk metrics
            if hasattr(stock, 'atr_percentage') and stock.atr_percentage:
                atr_values.append(stock.atr_percentage)
            if hasattr(stock, 'beta') and stock.beta:
                beta_values.append(stock.beta)

        # Calculate risk metrics
        risk_metrics = {}
        if atr_values:
            risk_metrics['avg_atr_percentage'] = sum(atr_values) / len(atr_values)
            risk_metrics['max_atr_percentage'] = max(atr_values)
            risk_metrics['min_atr_percentage'] = min(atr_values)

        if beta_values:
            risk_metrics['avg_beta'] = sum(beta_values) / len(beta_values)
            risk_metrics['max_beta'] = max(beta_values)
            risk_metrics['min_beta'] = min(beta_values)

        return {
            'total_stocks': total_stocks,
            'market_cap_distribution': cap_distribution,
            'sector_distribution': sector_distribution,
            'strategy_coverage': strategy_coverage,
            'risk_metrics': risk_metrics
        }