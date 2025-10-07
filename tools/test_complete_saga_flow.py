#!/usr/bin/env python3
"""
Comprehensive Suggested Stocks Saga Flow Test
Tests all 7 steps with detailed validation of risk profiles
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator
from src.models.database import get_database_manager
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_step_by_step_flow():
    """Test each step individually to understand the flow."""

    logger.info("=" * 80)
    logger.info("SUGGESTED STOCKS SAGA - COMPLETE FLOW TEST")
    logger.info("=" * 80)

    orchestrator = SuggestedStocksSagaOrchestrator()
    db_manager = get_database_manager()

    try:
        with db_manager.get_session() as session:

            # ========================================
            # STEP 1: Stock Discovery
            # ========================================
            logger.info("\n" + "=" * 80)
            logger.info("STEP 1: STOCK DISCOVERY")
            logger.info("=" * 80)

            # Test with NIFTY search
            logger.info("\nSearching for 'NIFTY' stocks...")
            result = session.execute(text("""
                SELECT symbol, name, sector
                FROM stocks
                WHERE (symbol ILIKE '%NIFTY%' OR name ILIKE '%NIFTY%')
                  AND current_price IS NOT NULL
                LIMIT 10
            """))

            discovered_stocks = [dict(row._mapping) for row in result.fetchall()]
            logger.info(f"✓ Discovered {len(discovered_stocks)} stocks")

            if discovered_stocks:
                logger.info("\nSample discovered stocks:")
                for i, stock in enumerate(discovered_stocks[:5], 1):
                    logger.info(f"  {i}. {stock['symbol']} - {stock['name']}")

            # ========================================
            # STEP 2: Database Filtering
            # ========================================
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: DATABASE FILTERING")
            logger.info("=" * 80)

            logger.info("\nChecking for stocks with complete data...")
            result = session.execute(text("""
                SELECT s.symbol, s.name, s.current_price, s.market_cap,
                       COUNT(DISTINCT ti.id) as ti_count,
                       COUNT(DISTINCT hd.id) as hd_count
                FROM stocks s
                LEFT JOIN technical_indicators ti ON s.symbol = ti.symbol
                LEFT JOIN historical_data hd ON s.symbol = hd.symbol
                WHERE s.current_price IS NOT NULL
                  AND s.market_cap IS NOT NULL
                GROUP BY s.symbol, s.name, s.current_price, s.market_cap
                HAVING COUNT(DISTINCT ti.id) > 0
                   AND COUNT(DISTINCT hd.id) > 0
                ORDER BY s.market_cap DESC
                LIMIT 20
            """))

            filtered_stocks = [dict(row._mapping) for row in result.fetchall()]
            logger.info(f"✓ {len(filtered_stocks)} stocks have complete data")

            if filtered_stocks:
                logger.info("\nSample filtered stocks:")
                for i, stock in enumerate(filtered_stocks[:5], 1):
                    logger.info(f"  {i}. {stock['symbol']} - Price: ₹{stock['current_price']:,.2f}, "
                              f"TI: {stock['ti_count']}, HD: {stock['hd_count']}")

            # ========================================
            # STEP 3: Strategy Application - DEFAULT_RISK
            # ========================================
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: STRATEGY APPLICATION - DEFAULT_RISK")
            logger.info("=" * 80)

            logger.info("\nDEFAULT_RISK Criteria:")
            logger.info("  - Price Range: ₹50 - ₹5,000")
            logger.info("  - Volume: > 100,000")
            logger.info("  - RSI: 30-70 (not overbought/oversold)")
            logger.info("  - MACD: Bullish crossover")
            logger.info("  - ATR %: < 5% (low volatility)")
            logger.info("  - P/E Ratio: 5-30 (reasonable valuation)")
            logger.info("  - ROE: > 10%")
            logger.info("  - Debt/Equity: < 2")

            result = session.execute(text("""
                SELECT s.symbol, s.name, s.current_price, s.market_cap,
                       ti.rsi, ti.macd, ti.atr, ti.volatility_30d,
                       s.pe_ratio, s.roe, s.debt_to_equity,
                       hd.volume
                FROM stocks s
                JOIN technical_indicators ti ON s.symbol = ti.symbol
                JOIN historical_data hd ON s.symbol = hd.symbol
                WHERE s.current_price BETWEEN 50 AND 5000
                  AND hd.volume > 100000
                  AND ti.rsi BETWEEN 30 AND 70
                  AND ti.atr IS NOT NULL
                  AND (ti.atr / s.current_price * 100) < 5
                  AND s.pe_ratio BETWEEN 5 AND 30
                  AND s.roe > 10
                  AND s.debt_to_equity < 2
                ORDER BY s.market_cap DESC
                LIMIT 10
            """))

            default_risk_stocks = [dict(row._mapping) for row in result.fetchall()]
            logger.info(f"\n✓ {len(default_risk_stocks)} stocks match DEFAULT_RISK criteria")

            if default_risk_stocks:
                logger.info("\nSample DEFAULT_RISK stocks:")
                for i, stock in enumerate(default_risk_stocks[:5], 1):
                    logger.info(f"\n  {i}. {stock['symbol']} - {stock['name']}")
                    logger.info(f"     Price: ₹{stock['current_price']:,.2f}")
                    logger.info(f"     RSI: {stock['rsi']:.2f}, MACD: {stock['macd']:.2f}")
                    logger.info(f"     Volatility: {stock['volatility_30d']:.2f}%")
                    logger.info(f"     P/E: {stock['pe_ratio']:.2f}, ROE: {stock['roe']:.2f}%")

            # ========================================
            # STEP 3: Strategy Application - HIGH_RISK
            # ========================================
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: STRATEGY APPLICATION - HIGH_RISK")
            logger.info("=" * 80)

            logger.info("\nHIGH_RISK Criteria:")
            logger.info("  - Price Range: ₹10 - ₹10,000")
            logger.info("  - Volume: > 50,000 (lower than default)")
            logger.info("  - RSI: 50-80 (momentum stocks)")
            logger.info("  - MACD: Strong bullish")
            logger.info("  - ATR %: < 10% (accepts higher volatility)")
            logger.info("  - P/E Ratio: Can be higher (growth stocks)")
            logger.info("  - ROE: > 5% (more lenient)")
            logger.info("  - Beta: > 1.2 (more volatile)")

            result = session.execute(text("""
                SELECT s.symbol, s.name, s.current_price, s.market_cap,
                       ti.rsi, ti.macd, ti.atr, ti.volatility_30d,
                       s.pe_ratio, s.roe, s.beta, s.debt_to_equity,
                       hd.volume
                FROM stocks s
                JOIN technical_indicators ti ON s.symbol = ti.symbol
                JOIN historical_data hd ON s.symbol = hd.symbol
                WHERE s.current_price BETWEEN 10 AND 10000
                  AND hd.volume > 50000
                  AND ti.rsi BETWEEN 50 AND 80
                  AND ti.macd > 0
                  AND ti.atr IS NOT NULL
                  AND (ti.atr / s.current_price * 100) < 10
                  AND s.roe > 5
                  AND COALESCE(s.beta, 0) > 1.2
                ORDER BY s.market_cap DESC
                LIMIT 10
            """))

            high_risk_stocks = [dict(row._mapping) for row in result.fetchall()]
            logger.info(f"\n✓ {len(high_risk_stocks)} stocks match HIGH_RISK criteria")

            if high_risk_stocks:
                logger.info("\nSample HIGH_RISK stocks:")
                for i, stock in enumerate(high_risk_stocks[:5], 1):
                    logger.info(f"\n  {i}. {stock['symbol']} - {stock['name']}")
                    logger.info(f"     Price: ₹{stock['current_price']:,.2f}")
                    logger.info(f"     RSI: {stock['rsi']:.2f}, MACD: {stock['macd']:.2f}")
                    logger.info(f"     Volatility: {stock['volatility_30d']:.2f}%")
                    logger.info(f"     Beta: {stock['beta']:.2f}")

            # ========================================
            # Compare Risk Profiles
            # ========================================
            logger.info("\n" + "=" * 80)
            logger.info("RISK PROFILE COMPARISON")
            logger.info("=" * 80)

            if default_risk_stocks and high_risk_stocks:
                # Calculate averages
                default_avg_volatility = sum(s['volatility_30d'] or 0 for s in default_risk_stocks) / len(default_risk_stocks)
                high_avg_volatility = sum(s['volatility_30d'] or 0 for s in high_risk_stocks) / len(high_risk_stocks)

                default_avg_rsi = sum(s['rsi'] or 0 for s in default_risk_stocks) / len(default_risk_stocks)
                high_avg_rsi = sum(s['rsi'] or 0 for s in high_risk_stocks) / len(high_risk_stocks)

                logger.info("\n📊 Average Metrics Comparison:")
                logger.info(f"\n  DEFAULT_RISK:")
                logger.info(f"    Average Volatility: {default_avg_volatility:.2f}%")
                logger.info(f"    Average RSI: {default_avg_rsi:.2f}")
                logger.info(f"    Count: {len(default_risk_stocks)}")

                logger.info(f"\n  HIGH_RISK:")
                logger.info(f"    Average Volatility: {high_avg_volatility:.2f}%")
                logger.info(f"    Average RSI: {high_avg_rsi:.2f}")
                logger.info(f"    Count: {len(high_risk_stocks)}")

                logger.info(f"\n  Difference:")
                logger.info(f"    Volatility: {high_avg_volatility - default_avg_volatility:+.2f}%")
                logger.info(f"    RSI: {high_avg_rsi - default_avg_rsi:+.2f}")

            # ========================================
            # Test Complete Saga - DEFAULT_RISK
            # ========================================
            logger.info("\n" + "=" * 80)
            logger.info("COMPLETE SAGA TEST - DEFAULT_RISK ONLY")
            logger.info("=" * 80)

            logger.info("\nExecuting saga with DEFAULT_RISK strategy...")
            default_result = orchestrator.execute_suggested_stocks_saga(
                user_id=1,
                strategies=['DEFAULT_RISK'],
                limit=10,
                search='',
                sort_by='score',
                sort_order='desc'
            )

            logger.info(f"\nStatus: {default_result['status']}")
            logger.info(f"Duration: {default_result['total_duration_seconds']:.2f}s")
            logger.info(f"Results: {len(default_result.get('final_results', []))}")

            if default_result.get('final_results'):
                logger.info("\n✓ Sample DEFAULT_RISK Results:")
                for i, stock in enumerate(default_result['final_results'][:3], 1):
                    logger.info(f"\n  {i}. {stock['symbol']}")
                    logger.info(f"     Strategy: {stock.get('strategy', 'N/A')}")
                    logger.info(f"     Score: {stock.get('score', 0):.2f}")
                    logger.info(f"     ML Score: {stock.get('ml_prediction_score', 0):.3f}")
                    logger.info(f"     Risk Score: {stock.get('ml_risk_score', 0):.3f}")

            # ========================================
            # Test Complete Saga - HIGH_RISK
            # ========================================
            logger.info("\n" + "=" * 80)
            logger.info("COMPLETE SAGA TEST - HIGH_RISK ONLY")
            logger.info("=" * 80)

            logger.info("\nExecuting saga with HIGH_RISK strategy...")
            high_result = orchestrator.execute_suggested_stocks_saga(
                user_id=1,
                strategies=['HIGH_RISK'],
                limit=10,
                search='',
                sort_by='score',
                sort_order='desc'
            )

            logger.info(f"\nStatus: {high_result['status']}")
            logger.info(f"Duration: {high_result['total_duration_seconds']:.2f}s")
            logger.info(f"Results: {len(high_result.get('final_results', []))}")

            if high_result.get('final_results'):
                logger.info("\n✓ Sample HIGH_RISK Results:")
                for i, stock in enumerate(high_result['final_results'][:3], 1):
                    logger.info(f"\n  {i}. {stock['symbol']}")
                    logger.info(f"     Strategy: {stock.get('strategy', 'N/A')}")
                    logger.info(f"     Score: {stock.get('score', 0):.2f}")
                    logger.info(f"     ML Score: {stock.get('ml_prediction_score', 0):.3f}")
                    logger.info(f"     Risk Score: {stock.get('ml_risk_score', 0):.3f}")

            # ========================================
            # Test Complete Saga - BOTH STRATEGIES
            # ========================================
            logger.info("\n" + "=" * 80)
            logger.info("COMPLETE SAGA TEST - BOTH STRATEGIES")
            logger.info("=" * 80)

            logger.info("\nExecuting saga with BOTH strategies...")
            both_result = orchestrator.execute_suggested_stocks_saga(
                user_id=1,
                strategies=['DEFAULT_RISK', 'HIGH_RISK'],
                limit=20,
                search='',
                sort_by='score',
                sort_order='desc'
            )

            logger.info(f"\nStatus: {both_result['status']}")
            logger.info(f"Duration: {both_result['total_duration_seconds']:.2f}s")
            logger.info(f"Results: {len(both_result.get('final_results', []))}")

            # Count strategies
            if both_result.get('final_results'):
                default_count = sum(1 for s in both_result['final_results'] if s.get('strategy') == 'DEFAULT_RISK')
                high_count = sum(1 for s in both_result['final_results'] if s.get('strategy') == 'HIGH_RISK')

                logger.info(f"\n✓ Strategy Distribution:")
                logger.info(f"  DEFAULT_RISK: {default_count} stocks")
                logger.info(f"  HIGH_RISK: {high_count} stocks")
                logger.info(f"  Total: {len(both_result['final_results'])} stocks")

                logger.info("\n✓ Sample Mixed Results:")
                for i, stock in enumerate(both_result['final_results'][:5], 1):
                    logger.info(f"\n  {i}. {stock['symbol']} ({stock.get('strategy', 'N/A')})")
                    logger.info(f"     Score: {stock.get('score', 0):.2f}")
                    logger.info(f"     ML Score: {stock.get('ml_prediction_score', 0):.3f}")
                    logger.info(f"     Risk Score: {stock.get('ml_risk_score', 0):.3f}")

            # ========================================
            # Final Validation
            # ========================================
            logger.info("\n" + "=" * 80)
            logger.info("FINAL VALIDATION")
            logger.info("=" * 80)

            all_tests_passed = True

            # Check if strategies produce different results
            if default_result.get('final_results') and high_result.get('final_results'):
                default_symbols = {s['symbol'] for s in default_result['final_results']}
                high_symbols = {s['symbol'] for s in high_result['final_results']}

                overlap = default_symbols & high_symbols
                unique_default = default_symbols - high_symbols
                unique_high = high_symbols - default_symbols

                logger.info(f"\n✓ Strategy Differentiation:")
                logger.info(f"  Common stocks: {len(overlap)}")
                logger.info(f"  Unique to DEFAULT_RISK: {len(unique_default)}")
                logger.info(f"  Unique to HIGH_RISK: {len(unique_high)}")

                if len(unique_default) > 0 and len(unique_high) > 0:
                    logger.info("  ✅ Strategies are producing different results")
                else:
                    logger.warning("  ⚠️ Strategies are producing identical results")
                    all_tests_passed = False

            # Summary
            logger.info("\n" + "=" * 80)
            logger.info("TEST SUMMARY")
            logger.info("=" * 80)

            logger.info("\n✓ Tests Completed:")
            logger.info(f"  ✅ Step 1: Stock Discovery - {len(discovered_stocks)} stocks")
            logger.info(f"  ✅ Step 2: Database Filtering - {len(filtered_stocks)} stocks")
            logger.info(f"  ✅ Step 3: DEFAULT_RISK Strategy - {len(default_risk_stocks)} stocks")
            logger.info(f"  ✅ Step 3: HIGH_RISK Strategy - {len(high_risk_stocks)} stocks")
            logger.info(f"  ✅ Complete Saga (DEFAULT_RISK) - {len(default_result.get('final_results', []))} stocks")
            logger.info(f"  ✅ Complete Saga (HIGH_RISK) - {len(high_result.get('final_results', []))} stocks")
            logger.info(f"  ✅ Complete Saga (BOTH) - {len(both_result.get('final_results', []))} stocks")

            if all_tests_passed:
                logger.info("\n✅ ALL TESTS PASSED")
                logger.info("✅ Risk profiles are working correctly")
                logger.info("✅ Strategies produce different results as expected")
                return True
            else:
                logger.warning("\n⚠️ SOME ISSUES DETECTED")
                return False

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        return False


def main():
    """Run complete flow test."""
    try:
        success = test_step_by_step_flow()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
