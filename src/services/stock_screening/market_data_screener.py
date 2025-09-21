"""
Market Data Screener

Comprehensive market data analysis and screening using FYERS v3 APIs:
- Real-time quotes validation and basic filtering
- Historical data analysis with volatility calculations
- Two-method approach: initial_quotes_screening() and historical_volatility_screening()
"""

import logging
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MarketDataScreener:
    """
    Market Data Screener: Comprehensive API-based screening using FYERS v3 APIs.

    Two main screening methods:
    1. initial_quotes_screening(): Symbol validation, price, volume, and movement filters
    2. historical_volatility_screening(): ATR, Beta, Historical Volatility analysis

    Responsibilities:
    - Validate symbols against FYERS Symbol Master
    - Filter by price, volume, and extreme movements using Quotes API
    - Calculate volatility metrics from historical OHLCV data
    - Apply advanced volatility and liquidity filters
    """

    def __init__(self, fyers_service, volatility_calculator_service):
        self.fyers_service = fyers_service
        self.volatility_service = volatility_calculator_service

        # Configuration for both screening methods
        self.config = {
            # Initial quotes screening criteria
            'min_price_threshold': 100.0,           # ₹100 minimum
            'min_daily_volume': 100000,             # 100k shares minimum
            'max_daily_change_percent': 20.0,       # Skip extreme movers
            'batch_size': 50,                       # Quotes API batch size
            'quotes_rate_limit_delay': 0.2,         # 200ms delay between batches

            # Historical volatility screening criteria
            'max_atr_percentage': 5.0,              # ATR% > 5% = too volatile
            'max_beta': 1.2,                        # Beta > 1.2 = too volatile vs market
            'max_historical_volatility': 60.0,      # > 60% annual volatility
            'min_avg_volume_20d': 50000,            # 20-day average volume
            'min_daily_turnover': 1.0,              # ₹1cr minimum daily turnover
            'days_lookback': 252,                   # 1 year of historical data
            'historical_rate_limit_delay': 0.1      # 100ms between API calls
        }

    def screen_stocks(self, user_id: int, tradeable_stocks: List) -> List:
        """
        Execute complete market data screening pipeline.

        Args:
            user_id: User ID for FYERS API
            tradeable_stocks: List of tradeable stock objects

        Returns:
            List of stocks that passed both screening methods with enhanced data
        """
        logger.info(f"📊 Starting Market Data Screening for {len(tradeable_stocks)} stocks")

        # Method 1: Initial quotes screening
        quotes_candidates = self.initial_quotes_screening(user_id, tradeable_stocks)

        if not quotes_candidates:
            logger.warning("No candidates passed quotes screening")
            return []

        # Method 2: Historical volatility screening
        final_candidates = self.historical_volatility_screening(user_id, quotes_candidates)

        logger.info(f"✅ Market Data Screening complete: {len(final_candidates)} candidates for business logic screening")
        return final_candidates

    def initial_quotes_screening(self, user_id: int, tradeable_stocks: List) -> List:
        """
        Method 1: Initial quotes screening using FYERS Symbol Master and Quotes APIs.

        Responsibilities:
        - Validate symbols against FYERS system
        - Filter by price threshold (₹100 minimum)
        - Filter by volume threshold (100k shares minimum)
        - Filter extreme price movements (±20% daily change)
        - Enhance stocks with real-time data from Quotes API

        Args:
            user_id: User ID for FYERS API
            tradeable_stocks: List of tradeable stock objects

        Returns:
            List of stocks that passed quotes screening with enhanced real-time data
        """
        logger.info(f"📋 Starting Method 1: Initial quotes screening for {len(tradeable_stocks)} stocks")

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

        # Process stocks in batches for Quotes API efficiency
        processed_count = 0
        print(f"   📊 Processing {len(tradeable_stocks)} stocks in batches of {self.config['batch_size']}...")

        for i in range(0, len(tradeable_stocks), self.config['batch_size']):
            batch_stocks = tradeable_stocks[i:i + self.config['batch_size']]
            batch_symbols = [stock.symbol for stock in batch_stocks]

            print(f"      🔄 Processing batch {i//self.config['batch_size'] + 1}: {len(batch_symbols)} symbols...")

            try:
                # Get real-time quotes for batch
                enhanced_stocks = self._process_quotes_batch(user_id, batch_stocks, batch_symbols)
                self.quotes_results['quotes_candidates'].extend(enhanced_stocks)
                processed_count += len(batch_stocks)

                # Progress update
                if processed_count % 100 == 0:
                    print(f"      📈 Processed {processed_count}/{len(tradeable_stocks)} stocks...")

                # Rate limiting between batches
                time.sleep(self.config['quotes_rate_limit_delay'])

            except Exception as e:
                logger.error(f"Error processing quotes batch: {e}")
                self.quotes_results['api_failures'] += len(batch_symbols)
                continue

        # Log quotes screening results
        self._log_quotes_results()

        logger.info(f"✅ Method 1 complete: {len(self.quotes_results['quotes_candidates'])} candidates for volatility screening")
        return self.quotes_results['quotes_candidates']

    def historical_volatility_screening(self, user_id: int, quotes_candidates: List) -> List:
        """
        Method 2: Historical volatility screening using FYERS v3 History API.

        Responsibilities:
        - Calculate ATR and ATR% from historical OHLCV data
        - Calculate Beta coefficient vs NIFTY50 index
        - Calculate historical volatility (annualized)
        - Calculate volume and turnover metrics
        - Apply advanced volatility filters

        Args:
            user_id: User ID for FYERS API
            quotes_candidates: Stocks that passed quotes screening

        Returns:
            List of stocks that passed volatility screening with calculated metrics
        """
        logger.info(f"📈 Starting Method 2: Historical volatility screening for {len(quotes_candidates)} candidates")

        # Results tracking for volatility screening
        self.volatility_results = {
            'total_input': len(quotes_candidates),
            'historical_data_fetched': 0,
            'atr_calculated': 0,
            'beta_calculated': 0,
            'volatility_calculated': 0,
            'high_atr_filtered': 0,
            'high_beta_filtered': 0,
            'high_volatility_filtered': 0,
            'low_turnover_filtered': 0,
            'api_failures': 0,
            'final_candidates': []
        }

        # Process candidates individually (History API is per-symbol)
        for i, stock in enumerate(quotes_candidates):
            try:
                print(f"      📊 Processing {i+1}/{len(quotes_candidates)}: {stock.symbol}...")

                enhanced_stock = self._process_volatility_single_stock(user_id, stock)
                if enhanced_stock:
                    self.volatility_results['final_candidates'].append(enhanced_stock)

                # Progress reporting every 10 stocks
                if (i + 1) % 10 == 0:
                    print(f"      📈 Progress: {i+1}/{len(quotes_candidates)} stocks analyzed...")

                # Rate limiting
                time.sleep(self.config['historical_rate_limit_delay'])

            except Exception as e:
                self.volatility_results['api_failures'] += 1
                logger.warning(f"Error processing {stock.symbol}: {e}")
                continue

        # Log volatility screening results
        self._log_volatility_results()

        logger.info(f"✅ Method 2 complete: {len(self.volatility_results['final_candidates'])} candidates for business logic screening")
        return self.volatility_results['final_candidates']

    def _process_quotes_batch(self, user_id: int, batch_stocks: List, batch_symbols: List[str]) -> List:
        """Process a batch of stocks through Quotes API."""
        enhanced_stocks = []

        quotes_response = self.fyers_service.quotes(user_id, batch_symbols)

        if quotes_response.get('success') and quotes_response.get('data'):
            quotes_data = quotes_response['data']

            for stock in batch_stocks:
                enhanced_stock = self._process_quotes_single_stock(stock, quotes_data)
                if enhanced_stock:
                    enhanced_stocks.append(enhanced_stock)
        else:
            logger.warning(f"Quotes API batch failed: {quotes_response.get('message', 'Unknown error')}")
            self.quotes_results['api_failures'] += len(batch_symbols)

        return enhanced_stocks

    def _process_quotes_single_stock(self, stock, quotes_data: Dict) -> Any:
        """Process a single stock with quote data and apply quotes filters."""
        # Check if symbol exists in FYERS quotes response
        quote_data = quotes_data.get(stock.symbol)
        if not quote_data or not quote_data.get('v'):
            self.quotes_results['api_failures'] += 1
            if self.quotes_results['api_failures'] <= 3:  # Show first 3 failures
                print(f"         ❌ SYMBOL NOT FOUND: {stock.symbol} (invalid/delisted)")
            return None

        self.quotes_results['symbol_validated'] += 1
        quote = quote_data['v']

        # Extract real-time data
        current_price = float(quote.get('lp', quote.get('ltp', 0)))
        current_volume = int(quote.get('volume', quote.get('vol', 0)))
        change_percent = float(quote.get('chp', 0))

        # Apply quotes filters
        if not self._apply_quotes_filters(stock, current_price, current_volume, change_percent):
            return None

        # Enhance stock with real-time data
        stock.current_price = current_price
        stock.volume = current_volume
        stock.change_percent = change_percent

        return stock

    def _apply_quotes_filters(self, stock, current_price: float, current_volume: int, change_percent: float) -> bool:
        """Apply quotes screening filter criteria."""

        # Filter 1: Price threshold
        if current_price < self.config['min_price_threshold']:
            self.quotes_results['penny_stocks_filtered'] += 1
            if self.quotes_results['penny_stocks_filtered'] <= 5:  # Show first 5
                print(f"         🪙 PENNY STOCK: {stock.symbol} - ₹{current_price:.2f} (< ₹{self.config['min_price_threshold']})")
            return False

        # Filter 2: Volume threshold
        if current_volume < self.config['min_daily_volume']:
            self.quotes_results['low_volume_filtered'] += 1
            if self.quotes_results['low_volume_filtered'] <= 5:  # Show first 5
                print(f"         💧 LOW VOLUME: {stock.symbol} - {current_volume:,} shares (< {self.config['min_daily_volume']:,})")
            return False

        # Filter 3: Extreme price movements (may indicate manipulation)
        if abs(change_percent) > self.config['max_daily_change_percent']:
            self.quotes_results['extreme_movers_filtered'] += 1
            if self.quotes_results['extreme_movers_filtered'] <= 3:  # Show first 3
                print(f"         🚨 EXTREME MOVER: {stock.symbol} - {change_percent:.1f}% change (> ±{self.config['max_daily_change_percent']}%)")
            return False

        return True

    def _process_volatility_single_stock(self, user_id: int, stock) -> Optional[Any]:
        """Process a single stock with historical data analysis."""
        # Get comprehensive volatility metrics using History API
        volatility_metrics = self.volatility_service.calculate_stock_volatility_metrics(
            user_id=user_id,
            symbol=stock.symbol,
            days_lookback=self.config['days_lookback']
        )

        if not volatility_metrics:
            self.volatility_results['api_failures'] += 1
            print(f"         ❌ Failed to get historical data for {stock.symbol}")
            return None

        self.volatility_results['historical_data_fetched'] += 1

        # Apply volatility filters using calculated metrics
        if not self._apply_volatility_filters(stock, volatility_metrics):
            return None

        # Merge calculated metrics with stock data
        self._enhance_stock_with_metrics(stock, volatility_metrics)

        return stock

    def _apply_volatility_filters(self, stock, volatility_metrics: Dict) -> bool:
        """Apply volatility screening filter criteria using calculated metrics."""
        # Filter 1: ATR Percentage
        atr_pct = volatility_metrics.get('atr_percentage')
        if atr_pct:
            self.volatility_results['atr_calculated'] += 1
            if atr_pct > self.config['max_atr_percentage']:
                self.volatility_results['high_atr_filtered'] += 1
                exclusion_reason = f"HIGH ATR: {atr_pct:.1f}% (> {self.config['max_atr_percentage']}%)"
                self._log_volatility_exclusion(stock, exclusion_reason)
                return False

        # Filter 2: Beta
        beta = volatility_metrics.get('beta')
        if beta:
            self.volatility_results['beta_calculated'] += 1
            if beta > self.config['max_beta']:
                self.volatility_results['high_beta_filtered'] += 1
                exclusion_reason = f"HIGH BETA: {beta:.2f} (> {self.config['max_beta']})"
                self._log_volatility_exclusion(stock, exclusion_reason)
                return False

        # Filter 3: Historical Volatility
        hist_vol = volatility_metrics.get('historical_volatility_1y')
        if hist_vol:
            self.volatility_results['volatility_calculated'] += 1
            if hist_vol > self.config['max_historical_volatility']:
                self.volatility_results['high_volatility_filtered'] += 1
                exclusion_reason = f"HIGH VOLATILITY: {hist_vol:.1f}% (> {self.config['max_historical_volatility']}%)"
                self._log_volatility_exclusion(stock, exclusion_reason)
                return False

        # Filter 4: Daily Turnover
        daily_turnover = volatility_metrics.get('avg_daily_turnover')
        if daily_turnover and daily_turnover < self.config['min_daily_turnover']:
            self.volatility_results['low_turnover_filtered'] += 1
            exclusion_reason = f"LOW TURNOVER: ₹{daily_turnover:.1f}cr (< ₹{self.config['min_daily_turnover']}cr)"
            self._log_volatility_exclusion(stock, exclusion_reason)
            return False

        # Stock passed all volatility filters
        print(f"         ✅ PASSED: {stock.symbol} - ATR: {atr_pct:.1f if atr_pct else 'N/A'}%, Beta: {beta:.2f if beta else 'N/A'}")
        return True

    def _enhance_stock_with_metrics(self, stock, volatility_metrics: Dict):
        """Enhance stock object with calculated volatility metrics."""
        stock.atr_14 = volatility_metrics.get('atr_14')
        stock.atr_percentage = volatility_metrics.get('atr_percentage')
        stock.beta = volatility_metrics.get('beta')
        stock.historical_volatility_1y = volatility_metrics.get('historical_volatility_1y')
        stock.avg_daily_volume_20d = volatility_metrics.get('avg_daily_volume_20d')
        stock.avg_daily_turnover = volatility_metrics.get('avg_daily_turnover')
        stock.bid_ask_spread = volatility_metrics.get('bid_ask_spread')

    def _log_volatility_exclusion(self, stock, reason: str):
        """Log stock exclusion with reason (limited to first 10)."""
        total_filtered = (self.volatility_results['high_atr_filtered'] + self.volatility_results['high_beta_filtered'] +
                         self.volatility_results['high_volatility_filtered'] + self.volatility_results['low_turnover_filtered'])
        if total_filtered <= 10:  # Show first 10 exclusions
            print(f"         🚫 FILTERED: {stock.symbol} - {reason}")

    def _log_quotes_results(self):
        """Log comprehensive quotes screening results."""
        print(f"   📊 METHOD 1 RESULTS (Quotes Screening):")
        print(f"      📈 Total stocks processed: {self.quotes_results['total_input']}")
        print(f"      ✅ Symbols validated in FYERS: {self.quotes_results['symbol_validated']}")
        print(f"      🪙 Penny stocks filtered (< ₹{self.config['min_price_threshold']}): {self.quotes_results['penny_stocks_filtered']}")
        print(f"      💧 Low volume filtered (< {self.config['min_daily_volume']:,}): {self.quotes_results['low_volume_filtered']}")
        print(f"      🚨 Extreme movers filtered (> ±{self.config['max_daily_change_percent']}%): {self.quotes_results['extreme_movers_filtered']}")
        print(f"      ❌ API failures: {self.quotes_results['api_failures']}")
        print(f"      ✅ Quotes candidates for volatility screening: {len(self.quotes_results['quotes_candidates'])}")

    def _log_volatility_results(self):
        """Log comprehensive volatility screening results."""
        print(f"   📊 METHOD 2 RESULTS (Volatility Screening):")
        print(f"      📈 Candidates from quotes screening: {self.volatility_results['total_input']}")
        print(f"      📊 Historical data fetched: {self.volatility_results['historical_data_fetched']}")
        print(f"      🧮 ATR calculations: {self.volatility_results['atr_calculated']}")
        print(f"      📈 Beta calculations: {self.volatility_results['beta_calculated']}")
        print(f"      📉 Volatility calculations: {self.volatility_results['volatility_calculated']}")
        print(f"      🚫 Filtered - High ATR: {self.volatility_results['high_atr_filtered']}")
        print(f"      🚫 Filtered - High Beta: {self.volatility_results['high_beta_filtered']}")
        print(f"      🚫 Filtered - High Volatility: {self.volatility_results['high_volatility_filtered']}")
        print(f"      🚫 Filtered - Low Turnover: {self.volatility_results['low_turnover_filtered']}")
        print(f"      ❌ API failures: {self.volatility_results['api_failures']}")
        print(f"      ✅ Final volatility candidates: {len(self.volatility_results['final_candidates'])}")

    def get_screening_stats(self) -> Dict[str, Any]:
        """Get detailed screening statistics."""
        return {
            'stage': 'MarketDataScreening',
            'config': self.config,
            'quotes_results': getattr(self, 'quotes_results', {}),
            'volatility_results': getattr(self, 'volatility_results', {}),
            'method_effectiveness': {
                'quotes_screening_rate': (self.quotes_results['symbol_validated'] / self.quotes_results['total_input']) * 100 if hasattr(self, 'quotes_results') and self.quotes_results['total_input'] > 0 else 0,
                'volatility_screening_rate': (self.volatility_results['historical_data_fetched'] / self.volatility_results['total_input']) * 100 if hasattr(self, 'volatility_results') and self.volatility_results['total_input'] > 0 else 0,
                'overall_success_rate': (len(self.volatility_results['final_candidates']) / self.quotes_results['total_input']) * 100 if hasattr(self, 'quotes_results') and hasattr(self, 'volatility_results') and self.quotes_results['total_input'] > 0 else 0
            }
        }

    def update_config(self, new_config: Dict[str, Any]):
        """Update screening configuration."""
        self.config.update(new_config)
        logger.info(f"Updated MarketDataScreener config: {new_config}")

    def get_market_data_analysis(self) -> Dict[str, Any]:
        """Get comprehensive market data analysis."""
        return {
            'quotes_analysis': getattr(self, 'quotes_results', {}),
            'volatility_analysis': getattr(self, 'volatility_results', {}),
            'api_efficiency': {
                'quotes_api_batches': self.quotes_results['total_input'] // self.config['batch_size'] if hasattr(self, 'quotes_results') else 0,
                'history_api_calls': self.volatility_results['total_input'] if hasattr(self, 'volatility_results') else 0,
                'overall_api_efficiency': 'High' if hasattr(self, 'volatility_results') and len(self.volatility_results['final_candidates']) > 0 else 'Low'
            }
        }