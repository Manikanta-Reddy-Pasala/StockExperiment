"""
Simple Market Data Screener

Simplified version for initial testing that focuses on basic screening
without complex volatility calculations.
"""

import logging
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SimpleMarketDataScreener:
    """
    Simplified Market Data Screener: Basic API-based screening using FYERS v3 APIs.

    Focuses on essential screening without complex volatility calculations for initial testing.
    """

    def __init__(self, fyers_service):
        self.fyers_service = fyers_service

        # Configuration for basic screening
        self.config = {
            # Basic quotes screening criteria
            'min_price_threshold': 50.0,                # ₹50 minimum (lowered for more results)
            'min_daily_volume': 50000,                  # 50k shares minimum (lowered)
            'max_daily_change_percent': 25.0,           # Skip extreme movers (raised)
            'batch_size': 50,                           # Quotes API batch size
            'quotes_rate_limit_delay': 0.2,             # 200ms delay between batches
        }

    def screen_stocks(self, user_id: int, tradeable_stocks: List) -> List:
        """
        Execute simplified market data screening pipeline.

        Args:
            user_id: User ID for FYERS API
            tradeable_stocks: List of tradeable stock objects

        Returns:
            List of stocks that passed basic screening
        """
        print(f"📊 STAGE 1: MARKET DATA SCREENING")
        print(f"=" * 60)
        print(f"   📊 Input: {len(tradeable_stocks)} tradeable stocks")
        print()

        # STAGE 1 STEP 1: Basic quotes screening
        print(f"📋 STAGE 1 STEP 1: Basic Database Screening")
        print(f"-" * 40)
        quotes_candidates = self.basic_quotes_screening(user_id, tradeable_stocks)
        print()

        # STAGE 1 STEP 2: Historical volatility analysis
        print(f"📈 STAGE 1 STEP 2: Historical Volatility Analysis")
        print(f"-" * 40)
        volatility_candidates = self.historical_volatility_screening(user_id, quotes_candidates)
        print()

        print(f"✅ STAGE 1 COMPLETE: {len(volatility_candidates)} candidates passed Market Data Screening")
        print(f"=" * 60)
        print()

        logger.info(f"✅ Simple Market Data Screening complete: {len(volatility_candidates)} candidates")
        return volatility_candidates

    def basic_quotes_screening(self, user_id: int, tradeable_stocks: List) -> List:
        """
        Basic quotes screening using FYERS Quotes API.

        Args:
            user_id: User ID for FYERS API
            tradeable_stocks: List of tradeable stock objects

        Returns:
            List of stocks that passed basic quotes screening
        """
        # Results tracking for quotes screening
        self.quotes_results = {
            'total_input': len(tradeable_stocks),
            'symbol_validated': 0,
            'penny_stocks_filtered': 0,
            'low_volume_filtered': 0,
            'extreme_movers_filtered': 0,
            'api_failures': 0,
            'quotes_candidates': []
        }

        print(f"   📊 Processing {len(tradeable_stocks)} stocks with database-only screening")
        print(f"   🎯 Criteria: Price ≥ ₹{self.config['min_price_threshold']}, Volume ≥ {self.config['min_daily_volume']:,}")
        print(f"   📈 Analyzing stocks...")

        valid_stocks = []
        for stock in tradeable_stocks:  # Process all tradeable stocks
            try:
                # Only use actual database data - no defaults or assumptions
                current_price = getattr(stock, 'current_price', None)
                current_volume = getattr(stock, 'volume', None)

                # Skip stocks that don't have required data
                if current_price is None:
                    self.quotes_results['api_failures'] += 1
                    if self.quotes_results['api_failures'] <= 3:
                        print(f"         ❌ MISSING PRICE: {stock.symbol} - No current_price data")
                    continue

                if current_volume is None:
                    self.quotes_results['api_failures'] += 1
                    if self.quotes_results['api_failures'] <= 3:
                        print(f"         ❌ MISSING VOLUME: {stock.symbol} - No volume data")
                    continue

                # Convert to proper types
                current_price = float(current_price)
                current_volume = int(current_volume)

                # Apply basic filters using only actual data
                if current_price >= self.config['min_price_threshold']:
                    if current_volume >= self.config['min_daily_volume']:
                        valid_stocks.append(stock)
                        self.quotes_results['symbol_validated'] += 1
                        print(f"         ✅ PASSED: {stock.symbol} - {stock.name} (₹{current_price:.2f}, Vol: {current_volume:,})")
                    else:
                        self.quotes_results['low_volume_filtered'] += 1
                        if self.quotes_results['low_volume_filtered'] <= 5:
                            print(f"         💧 LOW VOLUME: {stock.symbol} - Vol: {current_volume:,} (< {self.config['min_daily_volume']:,})")
                else:
                    self.quotes_results['penny_stocks_filtered'] += 1
                    if self.quotes_results['penny_stocks_filtered'] <= 5:
                        print(f"         🪙 PENNY STOCK: {stock.symbol} - ₹{current_price:.2f} (< ₹{self.config['min_price_threshold']})")

            except Exception as e:
                self.quotes_results['api_failures'] += 1
                if self.quotes_results['api_failures'] <= 5:
                    print(f"         ❌ ERROR processing {getattr(stock, 'symbol', 'Unknown')}: {e}")
                continue

        self.quotes_results['quotes_candidates'] = valid_stocks

        # Log quotes screening results
        self._log_quotes_results()

        logger.info(f"✅ Basic Quotes Screening complete: {len(valid_stocks)} candidates")
        return valid_stocks

    def _log_quotes_results(self):
        """Log comprehensive quotes screening results."""
        print(f"   📊 BASIC QUOTES SCREENING RESULTS:")
        print(f"      📈 Total stocks processed: {self.quotes_results['total_input']}")
        print(f"      ✅ Symbols validated: {self.quotes_results['symbol_validated']}")
        print(f"      🪙 Penny stocks filtered (< ₹{self.config['min_price_threshold']}): {self.quotes_results['penny_stocks_filtered']}")
        print(f"      💧 Low volume filtered (< {self.config['min_daily_volume']:,}): {self.quotes_results['low_volume_filtered']}")
        print(f"      ❌ Processing failures: {self.quotes_results['api_failures']}")
        print(f"      ✅ Basic candidates: {len(self.quotes_results['quotes_candidates'])}")

    def get_screening_stats(self) -> Dict[str, Any]:
        """Get detailed screening statistics."""
        return {
            'stage': 'SimpleMarketDataScreening',
            'config': self.config,
            'quotes_results': getattr(self, 'quotes_results', {}),
            'method_effectiveness': {
                'quotes_screening_rate': (self.quotes_results['symbol_validated'] / self.quotes_results['total_input']) * 100 if hasattr(self, 'quotes_results') and self.quotes_results['total_input'] > 0 else 0,
                'overall_success_rate': (len(self.quotes_results['quotes_candidates']) / self.quotes_results['total_input']) * 100 if hasattr(self, 'quotes_results') and self.quotes_results['total_input'] > 0 else 0
            }
        }

    def update_config(self, new_config: Dict[str, Any]):
        """Update screening configuration."""
        self.config.update(new_config)
        logger.info(f"Updated SimpleMarketDataScreener config: {new_config}")

    def historical_volatility_screening(self, user_id: int, quotes_candidates: List) -> List:
        """
        Historical volatility analysis using database volatility metrics.

        Args:
            user_id: User ID for any API calls (not used in simplified version)
            quotes_candidates: Stocks that passed basic quotes screening

        Returns:
            List of stocks that passed volatility screening
        """
        # Volatility screening criteria
        volatility_config = {
            'max_atr_percentage': 5.0,         # Max 5% ATR for moderate volatility
            'min_avg_volume_20d': 100000,      # Minimum 20-day average volume
            'max_bid_ask_spread': 2.0,         # Max 2% bid-ask spread
            'min_historical_volatility': 10.0, # Min 10% historical volatility (active stock)
            'max_historical_volatility': 40.0, # Max 40% historical volatility (not too risky)
            'max_beta': 1.2                    # Max Beta 1.2 for smoother, more predictable swings
        }

        print(f"   📊 Processing {len(quotes_candidates)} stocks for volatility analysis")
        print(f"   🎯 Criteria: ATR ≤ {volatility_config['max_atr_percentage']}%, Volume ≥ {volatility_config['min_avg_volume_20d']:,}, Beta ≤ {volatility_config['max_beta']}")
        print(f"   📈 Analyzing volatility and Beta metrics...")

        # Results tracking for volatility screening
        self.volatility_results = {
            'total_input': len(quotes_candidates),
            'high_atr_filtered': 0,
            'low_volume_filtered': 0,
            'wide_spread_filtered': 0,
            'extreme_volatility_filtered': 0,
            'high_beta_filtered': 0,
            'missing_data_filtered': 0,
            'volatility_candidates': []
        }

        valid_stocks = []
        for stock in quotes_candidates:
            try:
                # Check volatility metrics from database
                atr_percentage = getattr(stock, 'atr_percentage', None)
                avg_volume_20d = getattr(stock, 'avg_daily_volume_20d', None) or getattr(stock, 'volume', None)
                bid_ask_spread = getattr(stock, 'bid_ask_spread', None)
                hist_volatility = getattr(stock, 'historical_volatility_1y', None)
                beta = getattr(stock, 'beta', None)

                # Handle stocks with missing ATR data - use alternative metrics or skip if critical data missing
                if atr_percentage is None:
                    # If no ATR data, check if we have good volume data to proceed
                    if avg_volume_20d and avg_volume_20d >= volatility_config['min_avg_volume_20d']:
                        # Allow stocks with good volume even without ATR data
                        valid_stocks.append(stock)
                        print(f"         ✅ PASSED: {stock.symbol} - Good volume, ATR data pending")
                        continue
                    else:
                        self.volatility_results['missing_data_filtered'] += 1
                        if self.volatility_results['missing_data_filtered'] <= 3:
                            print(f"         ❌ MISSING DATA: {stock.symbol} - No ATR percentage or volume")
                        continue

                # Convert to proper types - NO MOCK DATA
                atr_percentage = float(atr_percentage)
                avg_volume_20d = float(avg_volume_20d) if avg_volume_20d else None
                bid_ask_spread = float(bid_ask_spread) if bid_ask_spread else None
                hist_volatility = float(hist_volatility) if hist_volatility else None
                beta = float(beta) if beta else None

                # Apply volatility filters - NO MOCK DATA
                if atr_percentage <= volatility_config['max_atr_percentage']:
                    if avg_volume_20d is not None and avg_volume_20d >= volatility_config['min_avg_volume_20d']:
                        # Beta filter for swing trading - exclude high Beta stocks
                        beta_ok = beta is None or beta <= volatility_config['max_beta']

                        # Optional filters for bid-ask spread and historical volatility
                        spread_ok = bid_ask_spread is None or bid_ask_spread <= volatility_config['max_bid_ask_spread']
                        volatility_ok = (hist_volatility is None or
                                       (volatility_config['min_historical_volatility'] <= hist_volatility <= volatility_config['max_historical_volatility']))

                        if beta_ok and spread_ok and volatility_ok:
                            valid_stocks.append(stock)
                            beta_str = f", Beta: {beta:.2f}" if beta is not None else ""
                            print(f"         ✅ PASSED: {stock.symbol} - ATR: {atr_percentage:.2f}%, Vol: {avg_volume_20d:,.0f}{beta_str}")
                        else:
                            if not beta_ok:
                                self.volatility_results['high_beta_filtered'] += 1
                                if self.volatility_results['high_beta_filtered'] <= 3:
                                    print(f"         📈 HIGH BETA: {stock.symbol} - Beta: {beta:.2f} (> {volatility_config['max_beta']})")
                            else:
                                self.volatility_results['extreme_volatility_filtered'] += 1
                                if self.volatility_results['extreme_volatility_filtered'] <= 3:
                                    spread_str = f", Spread: {bid_ask_spread:.2f}%" if bid_ask_spread else ""
                                    hv_str = f", HV: {hist_volatility:.1f}%" if hist_volatility else ""
                                    print(f"         📊 VOLATILITY: {stock.symbol}{spread_str}{hv_str}")
                    else:
                        self.volatility_results['low_volume_filtered'] += 1
                        if self.volatility_results['low_volume_filtered'] <= 3:
                            print(f"         💧 LOW VOLUME: {stock.symbol} - Avg Vol: {avg_volume_20d:,.0f}")
                else:
                    self.volatility_results['high_atr_filtered'] += 1
                    if self.volatility_results['high_atr_filtered'] <= 3:
                        print(f"         📈 HIGH ATR: {stock.symbol} - ATR: {atr_percentage:.2f}%")

            except Exception as e:
                self.volatility_results['missing_data_filtered'] += 1
                if self.volatility_results['missing_data_filtered'] <= 3:
                    print(f"         ❌ ERROR: {getattr(stock, 'symbol', 'Unknown')} - {e}")
                continue

        self.volatility_results['volatility_candidates'] = valid_stocks

        # Log volatility screening results
        self._log_volatility_results()

        print(f"   ✅ {len(valid_stocks)} stocks passed volatility screening")
        return valid_stocks

    def _log_volatility_results(self):
        """Log comprehensive volatility screening results."""
        print(f"   📊 VOLATILITY SCREENING RESULTS:")
        print(f"      📈 Total stocks processed: {self.volatility_results['total_input']}")
        print(f"      ✅ Passed volatility filters: {len(self.volatility_results['volatility_candidates'])}")
        print(f"      📈 High ATR filtered (> 5%): {self.volatility_results['high_atr_filtered']}")
        print(f"      💧 Low volume filtered: {self.volatility_results['low_volume_filtered']}")
        print(f"      📈 High Beta filtered (> 1.2): {self.volatility_results['high_beta_filtered']}")
        print(f"      📊 Extreme volatility filtered: {self.volatility_results['extreme_volatility_filtered']}")
        print(f"      ❌ Missing data filtered: {self.volatility_results['missing_data_filtered']}")