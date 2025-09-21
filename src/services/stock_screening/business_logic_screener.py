"""
Business Logic Screener

Handles comprehensive business logic screening for swing trading:
- Sector allocation limits
- Market cap requirements
- Fundamental ratio filters
- Strategy-specific criteria
- Portfolio optimization
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy types for business logic screening."""
    DEFAULT_RISK = "default_risk"
    HIGH_RISK = "high_risk"
    MEDIUM_RISK = "medium_risk"


class BusinessLogicScreener:
    """
    Business Logic Screener: Multi-criteria business logic screening for swing trading.

    Responsibilities:
    - Apply sector allocation limits (max 30% per sector)
    - Filter by market cap requirements (min ₹500cr)
    - Apply fundamental ratio filters (P/E, ROE, Debt/Equity)
    - Match stocks to strategy-specific criteria
    - Optimize final portfolio composition
    """

    def __init__(self):
        # Business filter criteria
        self.config = {
            'max_sector_allocation': 30.0,      # Max 30% in any sector
            'min_market_cap_crores': 500,       # Minimum ₹500cr market cap
            'max_pe_ratio': 50.0,               # P/E ratio filter
            'min_roe': 5.0,                     # Minimum ROE %
            'max_debt_equity': 2.0,             # Maximum debt-to-equity
            'min_dividend_yield': 0.0,          # Minimum dividend yield
            'max_position_size': 5.0,           # Max 5% position size
            'max_final_stocks': 50              # Final portfolio limit
        }

        # Strategy-specific criteria
        self.strategy_criteria = {
            StrategyType.DEFAULT_RISK: {
                'min_market_cap': 1000,         # ₹1000cr minimum
                'max_atr_percentage': 4.0,      # Max 4% ATR
                'max_beta': 1.1,                # Conservative beta
                'max_pe_ratio': 40.0,           # Reasonable P/E
                'max_debt_equity': 1.5          # Conservative debt
            },
            StrategyType.HIGH_RISK: {
                'min_market_cap': 500,          # ₹500cr minimum
                'max_atr_percentage': 6.0,      # Higher ATR tolerance
                'max_beta': 1.5,                # Higher beta tolerance
                'min_volume': 50000             # Good liquidity required
            },
            StrategyType.MEDIUM_RISK: {
                'min_market_cap': 750,          # ₹750cr minimum
                'max_atr_percentage': 5.0,      # Moderate ATR
                'max_beta': 1.3                 # Moderate beta
            }
        }

    def screen_stocks(self, market_data_candidates: List, strategies: List[StrategyType]) -> List:
        """
        Apply business logic screening.

        Args:
            market_data_candidates: Stocks that passed market data screening
            strategies: List of strategy types to apply

        Returns:
            List of stocks optimized for swing trading portfolio
        """
        print(f"🎯 STAGE 2: BUSINESS LOGIC SCREENING")
        print(f"=" * 60)
        print(f"   📊 Input: {len(market_data_candidates)} candidates from Stage 1")
        print(f"   🎯 Strategies: {[s.value for s in strategies] if strategies else ['default_risk']}")
        print()

        if not strategies:
            strategies = [StrategyType.DEFAULT_RISK]

        # Results tracking for each screening filter
        self.results = {
            'total_input': len(market_data_candidates),
            'sector_allocation_filtered': 0,
            'market_cap_filtered': 0,
            'pe_ratio_filtered': 0,
            'roe_filtered': 0,
            'debt_equity_filtered': 0,
            'strategy_filtered': 0,
            'final_stocks': []
        }

        # STAGE 2 STEP 1: Sector allocation filter
        print(f"📊 STAGE 2 STEP 1: Sector Allocation Filter")
        print(f"-" * 40)
        sector_filtered = self._apply_sector_allocation_filter(market_data_candidates)
        print(f"   ✅ {len(sector_filtered)} stocks passed sector allocation filter")
        print()

        # STAGE 2 STEP 2: Market cap filter
        print(f"💰 STAGE 2 STEP 2: Market Cap Filter")
        print(f"-" * 40)
        market_cap_filtered = self._apply_market_cap_filter(sector_filtered)
        print(f"   ✅ {len(market_cap_filtered)} stocks passed market cap filter")
        print()

        # STAGE 2 STEP 3: Fundamental filters
        print(f"📈 STAGE 2 STEP 3: Fundamental Analysis Filter")
        print(f"-" * 40)
        fundamental_filtered = self._apply_fundamental_filters(market_cap_filtered)
        print(f"   ✅ {len(fundamental_filtered)} stocks passed fundamental filters")
        print()

        # STAGE 2 STEP 4: Strategy-specific filters
        print(f"🎯 STAGE 2 STEP 4: Strategy-Specific Filter")
        print(f"-" * 40)
        strategy_filtered = self._apply_strategy_filters(fundamental_filtered, strategies)
        print(f"   ✅ {len(strategy_filtered)} stocks passed strategy filters")
        print()

        # STAGE 2 STEP 5: Portfolio optimization
        print(f"⚖️ STAGE 2 STEP 5: Portfolio Optimization")
        print(f"-" * 40)
        final_optimized = self._apply_portfolio_optimization(strategy_filtered)
        print(f"   ✅ {len(final_optimized)} stocks in final optimized portfolio")
        print()

        self.results['final_stocks'] = final_optimized

        # Log comprehensive results
        self._log_business_logic_results()
        self._log_portfolio_composition(final_optimized)

        print(f"✅ STAGE 2 COMPLETE: {len(final_optimized)} stocks ready for swing trading")
        print(f"=" * 60)
        print()

        logger.info(f"✅ Business Logic Screening complete: {len(final_optimized)} stocks ready for swing trading")
        return final_optimized

    def _apply_sector_allocation_filter(self, stocks: List) -> List:
        """Apply sector allocation limits to prevent over-concentration."""

        # Analyze sector distribution
        sector_counts = {}
        for stock in stocks:
            sector = stock.sector or 'Unknown'
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        print(f"      📈 Sector distribution:")
        for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(stocks)) * 100
            print(f"         {sector}: {count} stocks ({percentage:.1f}%)")

        # Calculate sector allocation limits
        max_stocks_per_sector = max(1, int(len(stocks) * self.config['max_sector_allocation'] / 100))
        print(f"      🎯 Max stocks per sector: {max_stocks_per_sector} ({self.config['max_sector_allocation']}% limit)")

        # Apply sector limits
        sector_selected = {}
        sector_filtered_stocks = []

        for stock in stocks:
            sector = stock.sector or 'Unknown'
            current_count = sector_selected.get(sector, 0)

            if current_count < max_stocks_per_sector:
                sector_selected[sector] = current_count + 1
                sector_filtered_stocks.append(stock)
            else:
                self.results['sector_allocation_filtered'] += 1
                if self.results['sector_allocation_filtered'] <= 5:  # Show first 5
                    print(f"         🚫 SECTOR LIMIT: {stock.symbol} ({sector} - limit reached)")

        print(f"      ✅ Sector filter result: {len(sector_filtered_stocks)} stocks passed")
        return sector_filtered_stocks

    def _apply_market_cap_filter(self, stocks: List) -> List:
        """Apply market cap requirements."""
        print(f"   💰 BUSINESS FILTER 2: Market cap filtering...")
        market_cap_stocks = []

        for stock in stocks:
            market_cap_crores = stock.market_cap

            # Allow stocks with NULL market cap (data not available yet)
            if market_cap_crores is not None and market_cap_crores < self.config['min_market_cap_crores']:
                self.results['market_cap_filtered'] += 1
                if self.results['market_cap_filtered'] <= 5:  # Show first 5
                    print(f"         🚫 LOW MARKET CAP: {stock.symbol} - ₹{market_cap_crores:.0f}cr (< ₹{self.config['min_market_cap_crores']}cr)")
            else:
                # Include stocks with NULL market cap or above threshold
                market_cap_stocks.append(stock)
                if market_cap_crores is None:
                    print(f"         ✅ PASSED (market cap data not available): {stock.symbol}")
                else:
                    print(f"         ✅ PASSED: {stock.symbol} - ₹{market_cap_crores:.0f}cr")

        print(f"      ✅ Market cap filter result: {len(market_cap_stocks)} stocks passed")
        return market_cap_stocks

    def _apply_fundamental_filters(self, stocks: List) -> List:
        """Apply fundamental ratio filters."""
        print(f"   📈 BUSINESS FILTER 3: Fundamental ratio filtering...")
        fundamental_stocks = []

        for stock in stocks:
            excluded = False
            exclusion_reason = ""

            # P/E Ratio filter
            if stock.pe_ratio and stock.pe_ratio > self.config['max_pe_ratio']:
                self.results['pe_ratio_filtered'] += 1
                excluded = True
                exclusion_reason = f"HIGH P/E: {stock.pe_ratio:.1f} (> {self.config['max_pe_ratio']})"

            # ROE filter
            elif stock.roe and stock.roe < self.config['min_roe']:
                self.results['roe_filtered'] += 1
                excluded = True
                exclusion_reason = f"LOW ROE: {stock.roe:.1f}% (< {self.config['min_roe']}%)"

            # Debt-to-Equity filter
            elif stock.debt_to_equity and stock.debt_to_equity > self.config['max_debt_equity']:
                self.results['debt_equity_filtered'] += 1
                excluded = True
                exclusion_reason = f"HIGH DEBT/EQUITY: {stock.debt_to_equity:.1f} (> {self.config['max_debt_equity']})"

            if excluded:
                total_fundamental_filtered = (self.results['pe_ratio_filtered'] +
                                            self.results['roe_filtered'] +
                                            self.results['debt_equity_filtered'])
                if total_fundamental_filtered <= 10:  # Show first 10
                    print(f"         🚫 FUNDAMENTAL: {stock.symbol} - {exclusion_reason}")
            else:
                fundamental_stocks.append(stock)

        print(f"      ✅ Fundamental filter result: {len(fundamental_stocks)} stocks passed")
        return fundamental_stocks

    def _apply_strategy_filters(self, stocks: List, strategies: List[StrategyType]) -> List:
        """Apply strategy-specific filtering."""
        print(f"   🎯 BUSINESS FILTER 4: Strategy-specific filtering...")
        strategy_stocks = []

        for stock in stocks:
            suitable_strategies = []

            for strategy in strategies:
                if self._meets_strategy_criteria(stock, strategy):
                    suitable_strategies.append(strategy)

            if suitable_strategies:
                # Assign the most suitable strategy
                stock.assigned_strategy = suitable_strategies[0]
                strategy_stocks.append(stock)
                print(f"         ✅ STRATEGY FIT: {stock.symbol} - {stock.assigned_strategy.value}")
            else:
                self.results['strategy_filtered'] += 1
                if self.results['strategy_filtered'] <= 5:  # Show first 5
                    print(f"         🚫 STRATEGY: {stock.symbol} - No suitable strategy match")

        print(f"      ✅ Strategy filter result: {len(strategy_stocks)} stocks passed")
        return strategy_stocks

    def _apply_portfolio_optimization(self, stocks: List) -> List:
        """Apply final portfolio optimization."""
        print(f"   📊 BUSINESS FILTER 5: Portfolio optimization...")

        # Sort by multiple criteria for best selection - ONLY use actual database data
        # Filter out stocks with missing critical data first
        valid_stocks = []
        for stock in stocks:
            if (stock.volume is not None and stock.volume > 0):
                valid_stocks.append(stock)
            else:
                print(f"         ❌ EXCLUDED: {stock.symbol} - Missing critical volume data")

        optimized_stocks = sorted(
            valid_stocks,
            key=lambda s: (
                -(s.market_cap) if s.market_cap else float('-inf'),     # Prefer larger market cap (only if available)
                -(s.volume),                                           # Volume is required - already filtered
                -(s.roe) if s.roe else float('-inf'),                  # Prefer higher ROE (only if available)
                abs(s.pe_ratio - 15) if s.pe_ratio else float('inf'), # Prefer P/E around 15 (only if available)
                s.atr_percentage if s.atr_percentage else float('inf') # Prefer lower ATR% (only if available)
            )
        )

        # Apply final limit
        final_stocks = optimized_stocks[:self.config['max_final_stocks']]

        print(f"      ✅ Portfolio optimization result: {len(final_stocks)} final stocks")
        return final_stocks

    def _meets_strategy_criteria(self, stock, strategy: StrategyType) -> bool:
        """Check if stock meets specific strategy criteria."""
        try:
            criteria = self.strategy_criteria.get(strategy, {})

            if strategy == StrategyType.DEFAULT_RISK:
                # NO MOCK DATA - Only use actual database values, skip if data missing
                market_cap_ok = (stock.market_cap is None or stock.market_cap >= criteria.get('min_market_cap', 0))

                # ATR check - skip stock if no real data
                if stock.atr_percentage is not None:
                    atr_ok = stock.atr_percentage <= criteria.get('max_atr_percentage', 999)
                else:
                    atr_ok = True  # Allow if no ATR data available

                # Beta check - skip stock if no real data
                if stock.beta is not None:
                    beta_ok = stock.beta <= criteria.get('max_beta', 999)
                else:
                    beta_ok = True  # Allow if no beta data available

                pe_ok = (stock.pe_ratio is None or stock.pe_ratio <= criteria.get('max_pe_ratio', 999))

                # Debt check - skip stock if no real data
                if stock.debt_to_equity is not None:
                    debt_ok = stock.debt_to_equity <= criteria.get('max_debt_equity', 999)
                else:
                    debt_ok = True  # Allow if no debt data available

                return market_cap_ok and atr_ok and beta_ok and pe_ok and debt_ok

            elif strategy == StrategyType.HIGH_RISK:
                # NO MOCK DATA - Only use actual database values
                market_cap_ok = (stock.market_cap is None or stock.market_cap >= criteria.get('min_market_cap', 0))

                # ATR check - skip stock if no real data
                if stock.atr_percentage is not None:
                    atr_ok = stock.atr_percentage <= criteria.get('max_atr_percentage', 999)
                else:
                    atr_ok = True  # Allow if no ATR data available

                # Beta check - skip stock if no real data
                if stock.beta is not None:
                    beta_ok = stock.beta <= criteria.get('max_beta', 999)
                else:
                    beta_ok = True  # Allow if no beta data available

                # Volume check - REQUIRE actual volume data
                if stock.volume is not None:
                    volume_ok = stock.volume >= criteria.get('min_volume', 0)
                else:
                    return False  # Reject if no volume data for high risk strategy

                return market_cap_ok and atr_ok and beta_ok and volume_ok

            else:  # MEDIUM_RISK or default
                # NO MOCK DATA - Only use actual database values
                market_cap_ok = (stock.market_cap is None or stock.market_cap >= criteria.get('min_market_cap', 0))

                # ATR check - skip stock if no real data
                if stock.atr_percentage is not None:
                    atr_ok = stock.atr_percentage <= criteria.get('max_atr_percentage', 999)
                else:
                    atr_ok = True  # Allow if no ATR data available

                # Beta check - skip stock if no real data
                if stock.beta is not None:
                    beta_ok = stock.beta <= criteria.get('max_beta', 999)
                else:
                    beta_ok = True  # Allow if no beta data available

                return market_cap_ok and atr_ok and beta_ok

        except Exception as e:
            logger.warning(f"Error checking strategy criteria for {stock.symbol}: {e}")
            return False

    def _log_business_logic_results(self):
        """Log comprehensive business logic screening results."""
        print(f"   📊 BUSINESS LOGIC SCREENING RESULTS:")
        print(f"      📈 Input from market data screening: {self.results['total_input']}")
        print(f"      🚫 Sector allocation filtered: {self.results['sector_allocation_filtered']}")
        print(f"      🚫 Market cap filtered: {self.results['market_cap_filtered']}")
        print(f"      🚫 P/E ratio filtered: {self.results['pe_ratio_filtered']}")
        print(f"      🚫 ROE filtered: {self.results['roe_filtered']}")
        print(f"      🚫 Debt/Equity filtered: {self.results['debt_equity_filtered']}")
        print(f"      🚫 Strategy filtered: {self.results['strategy_filtered']}")
        print(f"      ✅ Final optimized portfolio: {len(self.results['final_stocks'])} stocks")

    def _log_portfolio_composition(self, final_stocks: List):
        """Log final portfolio composition by sector."""
        if not final_stocks:
            return

        print(f"   📋 FINAL PORTFOLIO COMPOSITION:")
        final_sector_counts = {}
        for stock in final_stocks:
            sector = stock.sector or 'Unknown'
            final_sector_counts[sector] = final_sector_counts.get(sector, 0) + 1

        for sector, count in sorted(final_sector_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(final_stocks)) * 100
            print(f"      {sector}: {count} stocks ({percentage:.1f}%)")

    def get_filter_stats(self) -> Dict[str, Any]:
        """Get detailed filter statistics."""
        return {
            'stage': 'BusinessLogicScreening',
            'config': self.config,
            'strategy_criteria': {k.value: v for k, v in self.strategy_criteria.items()},
            'results': self.results,
            'filter_effectiveness': {
                'sector_filter_rate': (self.results['sector_allocation_filtered'] / self.results['total_input']) * 100 if self.results['total_input'] > 0 else 0,
                'market_cap_filter_rate': (self.results['market_cap_filtered'] / self.results['total_input']) * 100 if self.results['total_input'] > 0 else 0,
                'fundamental_filter_rate': ((self.results['pe_ratio_filtered'] + self.results['roe_filtered'] + self.results['debt_equity_filtered']) / self.results['total_input']) * 100 if self.results['total_input'] > 0 else 0,
                'strategy_filter_rate': (self.results['strategy_filtered'] / self.results['total_input']) * 100 if self.results['total_input'] > 0 else 0,
                'final_selection_rate': (len(self.results['final_stocks']) / self.results['total_input']) * 100 if self.results['total_input'] > 0 else 0
            }
        }

    def update_config(self, new_config: Dict[str, Any]):
        """Update filter configuration."""
        self.config.update(new_config)
        logger.info(f"Updated BusinessLogicScreener config: {new_config}")

    def update_strategy_criteria(self, strategy: StrategyType, new_criteria: Dict[str, Any]):
        """Update strategy-specific criteria."""
        if strategy in self.strategy_criteria:
            self.strategy_criteria[strategy].update(new_criteria)
            logger.info(f"Updated {strategy.value} criteria: {new_criteria}")

    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio composition metrics."""
        if not self.results['final_stocks']:
            return {}

        # Sector distribution
        sector_distribution = {}
        strategy_distribution = {}
        market_cap_distribution = {'large_cap': 0, 'mid_cap': 0, 'small_cap': 0, 'unknown': 0}

        for stock in self.results['final_stocks']:
            # Sector analysis
            sector = stock.sector or 'Unknown'
            sector_distribution[sector] = sector_distribution.get(sector, 0) + 1

            # Strategy analysis
            strategy = getattr(stock, 'assigned_strategy', 'Unknown')
            if hasattr(strategy, 'value'):
                strategy = strategy.value
            strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1

            # Market cap analysis - NO MOCK DATA
            if stock.market_cap is not None:
                market_cap = stock.market_cap
                if market_cap >= 20000:  # ₹20,000cr+
                    market_cap_distribution['large_cap'] += 1
                elif market_cap >= 5000:  # ₹5,000cr - ₹20,000cr
                    market_cap_distribution['mid_cap'] += 1
                else:  # < ₹5,000cr
                    market_cap_distribution['small_cap'] += 1
            else:
                market_cap_distribution['unknown'] += 1

        return {
            'total_stocks': len(self.results['final_stocks']),
            'sector_distribution': sector_distribution,
            'strategy_distribution': strategy_distribution,
            'market_cap_distribution': market_cap_distribution
        }