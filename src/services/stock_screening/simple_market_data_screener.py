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
        logger.info(f"📊 Starting Simple Market Data Screening for {len(tradeable_stocks)} stocks")

        # Basic quotes screening only (skip volatility for now)
        quotes_candidates = self.basic_quotes_screening(user_id, tradeable_stocks)

        logger.info(f"✅ Simple Market Data Screening complete: {len(quotes_candidates)} candidates")
        return quotes_candidates

    def basic_quotes_screening(self, user_id: int, tradeable_stocks: List) -> List:
        """
        Basic quotes screening using FYERS Quotes API.

        Args:
            user_id: User ID for FYERS API
            tradeable_stocks: List of tradeable stock objects

        Returns:
            List of stocks that passed basic quotes screening
        """
        logger.info(f"📋 Starting Basic Quotes Screening for {len(tradeable_stocks)} stocks")

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

        # For initial testing, let's just return the first 20 valid stocks without API calls
        # to avoid rate limiting issues
        print(f"   📊 Performing basic database-only screening...")

        valid_stocks = []
        for stock in tradeable_stocks[:50]:  # Limit to first 50 for testing
            try:
                # Basic checks using existing database data
                current_price = getattr(stock, 'current_price', None)
                current_volume = getattr(stock, 'volume', None)

                if current_price is None:
                    current_price = 100.0  # Default assumption
                else:
                    current_price = float(current_price)

                if current_volume is None:
                    current_volume = 100000  # Default assumption
                else:
                    current_volume = int(current_volume)

                # Apply basic filters
                if current_price >= self.config['min_price_threshold']:
                    if current_volume >= self.config['min_daily_volume']:
                        valid_stocks.append(stock)
                        self.quotes_results['symbol_validated'] += 1
                        print(f"         ✅ PASSED: {stock.symbol} - {stock.name} (₹{current_price:.2f}, Vol: {current_volume:,})")

                        if len(valid_stocks) >= 20:  # Limit for testing
                            break
                    else:
                        self.quotes_results['low_volume_filtered'] += 1
                        if self.quotes_results['low_volume_filtered'] <= 3:
                            print(f"         💧 LOW VOLUME: {stock.symbol} - Vol: {current_volume:,} (< {self.config['min_daily_volume']:,})")
                else:
                    self.quotes_results['penny_stocks_filtered'] += 1
                    if self.quotes_results['penny_stocks_filtered'] <= 3:
                        print(f"         🪙 PENNY STOCK: {stock.symbol} - ₹{current_price:.2f} (< ₹{self.config['min_price_threshold']})")

            except Exception as e:
                self.quotes_results['api_failures'] += 1
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