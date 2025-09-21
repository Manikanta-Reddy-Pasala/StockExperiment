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

        # Track detailed filtering stats with stock examples
        self.quotes_results['detailed_examples'] = {
            'passed_stocks': [],
            'penny_stocks': [],
            'low_volume_stocks': [],
            'missing_data_stocks': []
        }

        for stock in tradeable_stocks:  # Process all tradeable stocks
            try:
                # Only use actual database data - no defaults or assumptions
                current_price = getattr(stock, 'current_price', None)
                current_volume = getattr(stock, 'volume', None)

                # Skip stocks that don't have required data
                if current_price is None:
                    self.quotes_results['api_failures'] += 1
                    self.quotes_results['detailed_examples']['missing_data_stocks'].append({
                        'symbol': stock.symbol,
                        'reason': 'Missing current_price',
                        'price': None,
                        'volume': current_volume
                    })
                    print(f"         ❌ MISSING PRICE: {stock.symbol} - No current_price data")
                    continue

                if current_volume is None:
                    self.quotes_results['api_failures'] += 1
                    self.quotes_results['detailed_examples']['missing_data_stocks'].append({
                        'symbol': stock.symbol,
                        'reason': 'Missing volume',
                        'price': current_price,
                        'volume': None
                    })
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
                        self.quotes_results['detailed_examples']['passed_stocks'].append({
                            'symbol': stock.symbol,
                            'name': getattr(stock, 'name', 'Unknown'),
                            'price': current_price,
                            'volume': current_volume
                        })
                        print(f"         ✅ PASSED: {stock.symbol} - {stock.name} (₹{current_price:.2f}, Vol: {current_volume:,})")
                    else:
                        self.quotes_results['low_volume_filtered'] += 1
                        self.quotes_results['detailed_examples']['low_volume_stocks'].append({
                            'symbol': stock.symbol,
                            'price': current_price,
                            'volume': current_volume,
                            'threshold': self.config['min_daily_volume']
                        })
                        print(f"         💧 LOW VOLUME: {stock.symbol} - Vol: {current_volume:,} (< {self.config['min_daily_volume']:,}), Price: ₹{current_price:.2f}")
                else:
                    self.quotes_results['penny_stocks_filtered'] += 1
                    self.quotes_results['detailed_examples']['penny_stocks'].append({
                        'symbol': stock.symbol,
                        'price': current_price,
                        'volume': current_volume,
                        'threshold': self.config['min_price_threshold']
                    })
                    print(f"         🪙 PENNY STOCK: {stock.symbol} - ₹{current_price:.2f} (< ₹{self.config['min_price_threshold']}), Vol: {current_volume:,}")

            except Exception as e:
                self.quotes_results['api_failures'] += 1
                self.quotes_results['detailed_examples']['missing_data_stocks'].append({
                    'symbol': getattr(stock, 'symbol', 'Unknown'),
                    'reason': f'Processing error: {e}',
                    'price': None,
                    'volume': None
                })
                print(f"         ❌ ERROR processing {getattr(stock, 'symbol', 'Unknown')}: {e}")
                continue

        self.quotes_results['quotes_candidates'] = valid_stocks

        # Log quotes screening results
        self._log_quotes_results()

        logger.info(f"✅ Basic Quotes Screening complete: {len(valid_stocks)} candidates")
        return valid_stocks

    def _log_quotes_results(self):
        """Log comprehensive quotes screening results with detailed examples."""
        print(f"   📊 BASIC QUOTES SCREENING RESULTS:")
        print(f"      📈 Total stocks processed: {self.quotes_results['total_input']}")
        print(f"      ✅ Symbols validated: {self.quotes_results['symbol_validated']}")
        print(f"      🪙 Penny stocks filtered (< ₹{self.config['min_price_threshold']}): {self.quotes_results['penny_stocks_filtered']}")
        print(f"      💧 Low volume filtered (< {self.config['min_daily_volume']:,}): {self.quotes_results['low_volume_filtered']}")
        print(f"      ❌ Processing failures: {self.quotes_results['api_failures']}")
        print(f"      ✅ Basic candidates: {len(self.quotes_results['quotes_candidates'])}")
        print()

        # Show detailed examples with actual values
        examples = self.quotes_results.get('detailed_examples', {})

        if examples.get('passed_stocks'):
            print(f"   📋 PASSED STOCKS (Sample with Values):")
            for i, stock in enumerate(examples['passed_stocks'][:3]):  # Show first 3
                print(f"      ✅ {stock['symbol']}: ₹{stock['price']:.2f}, Vol: {stock['volume']:,}")
            if len(examples['passed_stocks']) > 3:
                print(f"      ... and {len(examples['passed_stocks']) - 3} more passed stocks")
            print()

        if examples.get('penny_stocks'):
            print(f"   🪙 PENNY STOCKS FILTERED (Sample with Values):")
            for i, stock in enumerate(examples['penny_stocks'][:3]):  # Show first 3
                print(f"      🪙 {stock['symbol']}: ₹{stock['price']:.2f} (< ₹{stock['threshold']:.1f}), Vol: {stock['volume']:,}")
            if len(examples['penny_stocks']) > 3:
                print(f"      ... and {len(examples['penny_stocks']) - 3} more penny stocks")
            print()

        if examples.get('low_volume_stocks'):
            print(f"   💧 LOW VOLUME STOCKS FILTERED (Sample with Values):")
            for i, stock in enumerate(examples['low_volume_stocks'][:3]):  # Show first 3
                print(f"      💧 {stock['symbol']}: Vol: {stock['volume']:,} (< {stock['threshold']:,}), ₹{stock['price']:.2f}")
            if len(examples['low_volume_stocks']) > 3:
                print(f"      ... and {len(examples['low_volume_stocks']) - 3} more low volume stocks")
            print()

        if examples.get('missing_data_stocks'):
            print(f"   ❌ MISSING DATA STOCKS (Sample with Reasons):")
            for i, stock in enumerate(examples['missing_data_stocks'][:3]):  # Show first 3
                price_str = f"₹{stock['price']:.2f}" if stock['price'] is not None else "N/A"
                vol_str = f"{stock['volume']:,}" if stock['volume'] is not None else "N/A"
                print(f"      ❌ {stock['symbol']}: {stock['reason']} (Price: {price_str}, Vol: {vol_str})")
            if len(examples['missing_data_stocks']) > 3:
                print(f"      ... and {len(examples['missing_data_stocks']) - 3} more missing data stocks")
            print()

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
            'volatility_candidates': [],
            'detailed_examples': {
                'passed_stocks': [],
                'high_atr_stocks': [],
                'low_volume_stocks': [],
                'high_beta_stocks': [],
                'extreme_volatility_stocks': [],
                'missing_data_stocks': []
            }
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
                        self.volatility_results['detailed_examples']['passed_stocks'].append({
                            'symbol': stock.symbol,
                            'atr_percentage': None,
                            'avg_volume_20d': avg_volume_20d,
                            'beta': beta,
                            'reason': 'Good volume, ATR data pending'
                        })
                        print(f"         ✅ PASSED: {stock.symbol} - Good volume (Vol: {avg_volume_20d:,.0f}), ATR data pending")
                        continue
                    else:
                        self.volatility_results['missing_data_filtered'] += 1
                        self.volatility_results['detailed_examples']['missing_data_stocks'].append({
                            'symbol': stock.symbol,
                            'reason': 'No ATR percentage or sufficient volume',
                            'atr_percentage': None,
                            'avg_volume_20d': avg_volume_20d,
                            'beta': beta,
                            'volume_threshold': volatility_config['min_avg_volume_20d']
                        })
                        vol_str = f"Vol: {avg_volume_20d:,.0f}" if avg_volume_20d else "Vol: N/A"
                        print(f"         ❌ MISSING DATA: {stock.symbol} - No ATR percentage, insufficient volume ({vol_str} < {volatility_config['min_avg_volume_20d']:,})")
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
                            self.volatility_results['detailed_examples']['passed_stocks'].append({
                                'symbol': stock.symbol,
                                'atr_percentage': atr_percentage,
                                'avg_volume_20d': avg_volume_20d,
                                'beta': beta,
                                'bid_ask_spread': bid_ask_spread,
                                'hist_volatility': hist_volatility,
                                'reason': 'Passed all volatility criteria'
                            })
                            beta_str = f", Beta: {beta:.2f}" if beta is not None else ", Beta: N/A"
                            spread_str = f", Spread: {bid_ask_spread:.2f}%" if bid_ask_spread is not None else ""
                            print(f"         ✅ PASSED: {stock.symbol} - ATR: {atr_percentage:.2f}%, Vol: {avg_volume_20d:,.0f}{beta_str}{spread_str}")
                        else:
                            if not beta_ok:
                                self.volatility_results['high_beta_filtered'] += 1
                                self.volatility_results['detailed_examples']['high_beta_stocks'].append({
                                    'symbol': stock.symbol,
                                    'atr_percentage': atr_percentage,
                                    'avg_volume_20d': avg_volume_20d,
                                    'beta': beta,
                                    'beta_threshold': volatility_config['max_beta'],
                                    'reason': f'High Beta: {beta:.2f} > {volatility_config["max_beta"]}'
                                })
                                print(f"         📈 HIGH BETA: {stock.symbol} - Beta: {beta:.2f} (> {volatility_config['max_beta']}), ATR: {atr_percentage:.2f}%, Vol: {avg_volume_20d:,.0f}")
                            else:
                                self.volatility_results['extreme_volatility_filtered'] += 1
                                self.volatility_results['detailed_examples']['extreme_volatility_stocks'].append({
                                    'symbol': stock.symbol,
                                    'atr_percentage': atr_percentage,
                                    'avg_volume_20d': avg_volume_20d,
                                    'beta': beta,
                                    'bid_ask_spread': bid_ask_spread,
                                    'hist_volatility': hist_volatility,
                                    'reason': 'Failed spread or historical volatility filters'
                                })
                                spread_str = f", Spread: {bid_ask_spread:.2f}%" if bid_ask_spread else ""
                                hv_str = f", HV: {hist_volatility:.1f}%" if hist_volatility else ""
                                print(f"         📊 VOLATILITY: {stock.symbol} - ATR: {atr_percentage:.2f}%, Vol: {avg_volume_20d:,.0f}{spread_str}{hv_str}")
                    else:
                        self.volatility_results['low_volume_filtered'] += 1
                        self.volatility_results['detailed_examples']['low_volume_stocks'].append({
                            'symbol': stock.symbol,
                            'atr_percentage': atr_percentage,
                            'avg_volume_20d': avg_volume_20d,
                            'volume_threshold': volatility_config['min_avg_volume_20d'],
                            'beta': beta,
                            'reason': f'Low volume: {avg_volume_20d:,.0f} < {volatility_config["min_avg_volume_20d"]:,}'
                        })
                        vol_str = f"{avg_volume_20d:,.0f}" if avg_volume_20d is not None else "N/A"
                        print(f"         💧 LOW VOLUME: {stock.symbol} - ATR: {atr_percentage:.2f}%, Vol: {vol_str} (< {volatility_config['min_avg_volume_20d']:,})")
                else:
                    self.volatility_results['high_atr_filtered'] += 1
                    self.volatility_results['detailed_examples']['high_atr_stocks'].append({
                        'symbol': stock.symbol,
                        'atr_percentage': atr_percentage,
                        'atr_threshold': volatility_config['max_atr_percentage'],
                        'avg_volume_20d': avg_volume_20d,
                        'beta': beta,
                        'reason': f'High ATR: {atr_percentage:.2f}% > {volatility_config["max_atr_percentage"]}%'
                    })
                    vol_str = f", Vol: {avg_volume_20d:,.0f}" if avg_volume_20d is not None else ""
                    print(f"         📈 HIGH ATR: {stock.symbol} - ATR: {atr_percentage:.2f}% (> {volatility_config['max_atr_percentage']}%){vol_str}")

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
        """Log comprehensive volatility screening results with detailed examples."""
        print(f"   📊 VOLATILITY SCREENING RESULTS:")
        print(f"      📈 Total stocks processed: {self.volatility_results['total_input']}")
        print(f"      ✅ Passed volatility filters: {len(self.volatility_results['volatility_candidates'])}")
        print(f"      📈 High ATR filtered (> 5%): {self.volatility_results['high_atr_filtered']}")
        print(f"      💧 Low volume filtered: {self.volatility_results['low_volume_filtered']}")
        print(f"      📈 High Beta filtered (> 1.2): {self.volatility_results['high_beta_filtered']}")
        print(f"      📊 Extreme volatility filtered: {self.volatility_results['extreme_volatility_filtered']}")
        print(f"      ❌ Missing data filtered: {self.volatility_results['missing_data_filtered']}")
        print()

        # Show detailed examples with actual values for each filter category
        examples = self.volatility_results.get('detailed_examples', {})

        if examples.get('passed_stocks'):
            print(f"   📋 PASSED STOCKS (Sample with Values):")
            for i, stock in enumerate(examples['passed_stocks'][:3]):  # Show first 3
                atr_str = f"ATR: {stock['atr_percentage']:.2f}%" if stock['atr_percentage'] is not None else "ATR: Pending"
                beta_str = f", Beta: {stock['beta']:.2f}" if stock['beta'] is not None else ", Beta: N/A"
                spread_str = f", Spread: {stock['bid_ask_spread']:.2f}%" if stock.get('bid_ask_spread') is not None else ""
                print(f"      ✅ {stock['symbol']}: {atr_str}, Vol: {stock['avg_volume_20d']:,.0f}{beta_str}{spread_str}")
            if len(examples['passed_stocks']) > 3:
                print(f"      ... and {len(examples['passed_stocks']) - 3} more passed stocks")
            print()

        if examples.get('high_atr_stocks'):
            print(f"   📈 HIGH ATR STOCKS FILTERED (Sample with Values):")
            for i, stock in enumerate(examples['high_atr_stocks'][:3]):  # Show first 3
                vol_str = f", Vol: {stock['avg_volume_20d']:,.0f}" if stock['avg_volume_20d'] is not None else ""
                print(f"      📈 {stock['symbol']}: ATR: {stock['atr_percentage']:.2f}% (> {stock['atr_threshold']:.1f}%){vol_str}")
            if len(examples['high_atr_stocks']) > 3:
                print(f"      ... and {len(examples['high_atr_stocks']) - 3} more high ATR stocks")
            print()

        if examples.get('high_beta_stocks'):
            print(f"   📈 HIGH BETA STOCKS FILTERED (Sample with Values):")
            for i, stock in enumerate(examples['high_beta_stocks'][:3]):  # Show first 3
                print(f"      📈 {stock['symbol']}: Beta: {stock['beta']:.2f} (> {stock['beta_threshold']:.1f}), ATR: {stock['atr_percentage']:.2f}%, Vol: {stock['avg_volume_20d']:,.0f}")
            if len(examples['high_beta_stocks']) > 3:
                print(f"      ... and {len(examples['high_beta_stocks']) - 3} more high Beta stocks")
            print()

        if examples.get('low_volume_stocks'):
            print(f"   💧 LOW VOLUME STOCKS FILTERED (Sample with Values):")
            for i, stock in enumerate(examples['low_volume_stocks'][:3]):  # Show first 3
                vol_str = f"{stock['avg_volume_20d']:,.0f}" if stock['avg_volume_20d'] is not None else "N/A"
                print(f"      💧 {stock['symbol']}: Vol: {vol_str} (< {stock['volume_threshold']:,}), ATR: {stock['atr_percentage']:.2f}%")
            if len(examples['low_volume_stocks']) > 3:
                print(f"      ... and {len(examples['low_volume_stocks']) - 3} more low volume stocks")
            print()

        if examples.get('extreme_volatility_stocks'):
            print(f"   📊 EXTREME VOLATILITY STOCKS FILTERED (Sample with Values):")
            for i, stock in enumerate(examples['extreme_volatility_stocks'][:3]):  # Show first 3
                spread_str = f", Spread: {stock['bid_ask_spread']:.2f}%" if stock['bid_ask_spread'] is not None else ""
                hv_str = f", HV: {stock['hist_volatility']:.1f}%" if stock['hist_volatility'] is not None else ""
                print(f"      📊 {stock['symbol']}: ATR: {stock['atr_percentage']:.2f}%, Vol: {stock['avg_volume_20d']:,.0f}{spread_str}{hv_str}")
            if len(examples['extreme_volatility_stocks']) > 3:
                print(f"      ... and {len(examples['extreme_volatility_stocks']) - 3} more extreme volatility stocks")
            print()

        if examples.get('missing_data_stocks'):
            print(f"   ❌ MISSING DATA STOCKS (Sample with Reasons):")
            for i, stock in enumerate(examples['missing_data_stocks'][:3]):  # Show first 3
                vol_str = f"Vol: {stock['avg_volume_20d']:,.0f}" if stock['avg_volume_20d'] is not None else "Vol: N/A"
                beta_str = f", Beta: {stock['beta']:.2f}" if stock['beta'] is not None else ", Beta: N/A"
                print(f"      ❌ {stock['symbol']}: {stock['reason']} ({vol_str}{beta_str})")
            if len(examples['missing_data_stocks']) > 3:
                print(f"      ... and {len(examples['missing_data_stocks']) - 3} more missing data stocks")
            print()