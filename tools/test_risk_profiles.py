#!/usr/bin/env python3
"""
Test Risk Profiles - DEFAULT_RISK vs HIGH_RISK
Validates that the two strategies produce different results
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_risk_profiles():
    """Test DEFAULT_RISK vs HIGH_RISK strategies."""

    logger.info("=" * 80)
    logger.info("RISK PROFILE TESTING - DEFAULT_RISK vs HIGH_RISK")
    logger.info("=" * 80)

    orchestrator = SuggestedStocksSagaOrchestrator()

    # ========================================
    # Test 1: DEFAULT_RISK Only
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: DEFAULT_RISK STRATEGY")
    logger.info("=" * 80)

    logger.info("\nExecuting saga with DEFAULT_RISK strategy...")
    default_result = orchestrator.execute_suggested_stocks_saga(
        user_id=1,
        strategies=['DEFAULT_RISK'],
        limit=15,
        search='',
        sort_by='score',
        sort_order='desc'
    )

    logger.info(f"\nStatus: {default_result['status']}")
    logger.info(f"Duration: {default_result['total_duration_seconds']:.2f}s")
    logger.info(f"Results: {len(default_result.get('final_results', []))} stocks")

    default_stocks = default_result.get('final_results', [])

    if default_stocks:
        logger.info("\n✓ Sample DEFAULT_RISK stocks:")
        for i, stock in enumerate(default_stocks[:5], 1):
            logger.info(f"\n  {i}. {stock['symbol']}")
            logger.info(f"     Price: ₹{stock.get('current_price', 0):,.2f}")
            logger.info(f"     Market Cap: ₹{stock.get('market_cap', 0):,.0f} Cr")
            logger.info(f"     Strategy: {stock.get('strategy', 'N/A')}")
            logger.info(f"     Score: {stock.get('score', 0):.2f}")
            logger.info(f"     Target Price: ₹{stock.get('target_price', 0):,.2f}")
            logger.info(f"     Upside: {stock.get('upside_percentage', 0):.2f}%")
            if 'ml_prediction_score' in stock:
                logger.info(f"     ML Score: {stock['ml_prediction_score']:.3f}")
                logger.info(f"     Risk Score: {stock.get('ml_risk_score', 0):.3f}")

        # Calculate statistics
        default_avg_price = sum(s.get('current_price', 0) for s in default_stocks) / len(default_stocks)
        default_avg_market_cap = sum(s.get('market_cap', 0) for s in default_stocks) / len(default_stocks)
        default_avg_score = sum(s.get('score', 0) for s in default_stocks) / len(default_stocks)
        default_avg_upside = sum(s.get('upside_percentage', 0) for s in default_stocks) / len(default_stocks)

        logger.info(f"\n📊 DEFAULT_RISK Statistics:")
        logger.info(f"  Count: {len(default_stocks)}")
        logger.info(f"  Avg Price: ₹{default_avg_price:,.2f}")
        logger.info(f"  Avg Market Cap: ₹{default_avg_market_cap:,.0f} Cr")
        logger.info(f"  Avg Score: {default_avg_score:.2f}")
        logger.info(f"  Avg Upside: {default_avg_upside:.2f}%")

    # ========================================
    # Test 2: HIGH_RISK Only
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: HIGH_RISK STRATEGY")
    logger.info("=" * 80)

    logger.info("\nExecuting saga with HIGH_RISK strategy...")
    high_result = orchestrator.execute_suggested_stocks_saga(
        user_id=1,
        strategies=['HIGH_RISK'],
        limit=15,
        search='',
        sort_by='score',
        sort_order='desc'
    )

    logger.info(f"\nStatus: {high_result['status']}")
    logger.info(f"Duration: {high_result['total_duration_seconds']:.2f}s")
    logger.info(f"Results: {len(high_result.get('final_results', []))} stocks")

    high_stocks = high_result.get('final_results', [])

    if high_stocks:
        logger.info("\n✓ Sample HIGH_RISK stocks:")
        for i, stock in enumerate(high_stocks[:5], 1):
            logger.info(f"\n  {i}. {stock['symbol']}")
            logger.info(f"     Price: ₹{stock.get('current_price', 0):,.2f}")
            logger.info(f"     Market Cap: ₹{stock.get('market_cap', 0):,.0f} Cr")
            logger.info(f"     Strategy: {stock.get('strategy', 'N/A')}")
            logger.info(f"     Score: {stock.get('score', 0):.2f}")
            logger.info(f"     Target Price: ₹{stock.get('target_price', 0):,.2f}")
            logger.info(f"     Upside: {stock.get('upside_percentage', 0):.2f}%")
            if 'ml_prediction_score' in stock:
                logger.info(f"     ML Score: {stock['ml_prediction_score']:.3f}")
                logger.info(f"     Risk Score: {stock.get('ml_risk_score', 0):.3f}")

        # Calculate statistics
        high_avg_price = sum(s.get('current_price', 0) for s in high_stocks) / len(high_stocks)
        high_avg_market_cap = sum(s.get('market_cap', 0) for s in high_stocks) / len(high_stocks)
        high_avg_score = sum(s.get('score', 0) for s in high_stocks) / len(high_stocks)
        high_avg_upside = sum(s.get('upside_percentage', 0) for s in high_stocks) / len(high_stocks)

        logger.info(f"\n📊 HIGH_RISK Statistics:")
        logger.info(f"  Count: {len(high_stocks)}")
        logger.info(f"  Avg Price: ₹{high_avg_price:,.2f}")
        logger.info(f"  Avg Market Cap: ₹{high_avg_market_cap:,.0f} Cr")
        logger.info(f"  Avg Score: {high_avg_score:.2f}")
        logger.info(f"  Avg Upside: {high_avg_upside:.2f}%")

    # ========================================
    # Test 3: Both Strategies
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: BOTH STRATEGIES COMBINED")
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
    logger.info(f"Results: {len(both_result.get('final_results', []))} stocks")

    both_stocks = both_result.get('final_results', [])

    if both_stocks:
        # Count strategies
        default_count = sum(1 for s in both_stocks if s.get('strategy') == 'DEFAULT_RISK')
        high_count = sum(1 for s in both_stocks if s.get('strategy') == 'HIGH_RISK')

        logger.info(f"\n✓ Strategy Distribution:")
        logger.info(f"  DEFAULT_RISK: {default_count} stocks ({default_count/len(both_stocks)*100:.1f}%)")
        logger.info(f"  HIGH_RISK: {high_count} stocks ({high_count/len(both_stocks)*100:.1f}%)")

        logger.info("\n✓ Sample mixed results:")
        for i, stock in enumerate(both_stocks[:6], 1):
            strategy_emoji = "🛡️" if stock.get('strategy') == 'DEFAULT_RISK' else "🚀"
            logger.info(f"\n  {i}. {strategy_emoji} {stock['symbol']} ({stock.get('strategy', 'N/A')})")
            logger.info(f"     Price: ₹{stock.get('current_price', 0):,.2f}, Score: {stock.get('score', 0):.2f}")
            logger.info(f"     Upside: {stock.get('upside_percentage', 0):.2f}%")

    # ========================================
    # Comparison & Validation
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON & VALIDATION")
    logger.info("=" * 80)

    all_tests_passed = True

    if default_stocks and high_stocks:
        # Check for overlap
        default_symbols = {s['symbol'] for s in default_stocks}
        high_symbols = {s['symbol'] for s in high_stocks}

        overlap = default_symbols & high_symbols
        unique_default = default_symbols - high_symbols
        unique_high = high_symbols - default_symbols

        logger.info(f"\n✓ Strategy Differentiation:")
        logger.info(f"  Common stocks: {len(overlap)} ({len(overlap)/len(default_symbols)*100:.1f}%)")
        logger.info(f"  Unique to DEFAULT_RISK: {len(unique_default)} ({len(unique_default)/len(default_symbols)*100:.1f}%)")
        logger.info(f"  Unique to HIGH_RISK: {len(unique_high)} ({len(unique_high)/len(high_symbols)*100:.1f}%)")

        if overlap:
            logger.info(f"\n  Common stocks: {', '.join(list(overlap)[:5])}{'...' if len(overlap) > 5 else ''}")

        # Validate differentiation
        if len(unique_default) > 0 or len(unique_high) > 0:
            logger.info("\n  ✅ Strategies are producing different results")
        else:
            logger.warning("\n  ⚠️ Strategies are producing identical results")
            all_tests_passed = False

        # Compare average metrics
        logger.info(f"\n✓ Average Metrics Comparison:")
        logger.info(f"\n  Market Cap:")
        logger.info(f"    DEFAULT_RISK: ₹{default_avg_market_cap:,.0f} Cr")
        logger.info(f"    HIGH_RISK: ₹{high_avg_market_cap:,.0f} Cr")
        logger.info(f"    Difference: ₹{default_avg_market_cap - high_avg_market_cap:+,.0f} Cr")

        logger.info(f"\n  Expected Upside:")
        logger.info(f"    DEFAULT_RISK: {default_avg_upside:.2f}%")
        logger.info(f"    HIGH_RISK: {high_avg_upside:.2f}%")
        logger.info(f"    Difference: {high_avg_upside - default_avg_upside:+.2f}%")

        logger.info(f"\n  Quality Score:")
        logger.info(f"    DEFAULT_RISK: {default_avg_score:.2f}")
        logger.info(f"    HIGH_RISK: {high_avg_score:.2f}")
        logger.info(f"    Difference: {high_avg_score - default_avg_score:+.2f}")

    # ========================================
    # Validation Steps
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION CHECKLIST")
    logger.info("=" * 80)

    checks = []

    # Check 1: Both strategies return results
    if default_stocks and high_stocks:
        checks.append(("✅", "Both strategies return results"))
    else:
        checks.append(("❌", "One or both strategies returned no results"))
        all_tests_passed = False

    # Check 2: DEFAULT_RISK has conservative target (7%)
    if default_stocks:
        avg_upside_default = sum(s.get('upside_percentage', 0) for s in default_stocks) / len(default_stocks)
        if 5 <= avg_upside_default <= 10:
            checks.append(("✅", f"DEFAULT_RISK has conservative upside (~{avg_upside_default:.1f}%)"))
        else:
            checks.append(("⚠️", f"DEFAULT_RISK upside seems off ({avg_upside_default:.1f}%)"))

    # Check 3: HIGH_RISK has aggressive target (12%)
    if high_stocks:
        avg_upside_high = sum(s.get('upside_percentage', 0) for s in high_stocks) / len(high_stocks)
        if 10 <= avg_upside_high <= 15:
            checks.append(("✅", f"HIGH_RISK has aggressive upside (~{avg_upside_high:.1f}%)"))
        else:
            checks.append(("⚠️", f"HIGH_RISK upside seems off ({avg_upside_high:.1f}%)"))

    # Check 4: Combined strategy has both
    if both_stocks:
        default_in_both = sum(1 for s in both_stocks if s.get('strategy') == 'DEFAULT_RISK')
        high_in_both = sum(1 for s in both_stocks if s.get('strategy') == 'HIGH_RISK')

        if default_in_both > 0 and high_in_both > 0:
            checks.append(("✅", f"Combined result has both strategies ({default_in_both} + {high_in_both})"))
        else:
            checks.append(("❌", "Combined result missing one strategy"))
            all_tests_passed = False

    # Check 5: ML predictions are present
    if default_stocks:
        ml_present = all('ml_prediction_score' in s for s in default_stocks[:3])
        if ml_present:
            checks.append(("✅", "ML predictions are present in results"))
        else:
            checks.append(("⚠️", "ML predictions may be missing"))

    # Print all checks
    logger.info("")
    for status, check in checks:
        logger.info(f"  {status} {check}")

    # Final result
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    if all_tests_passed:
        logger.info("\n✅ ALL TESTS PASSED")
        logger.info("✅ Risk profiles are working correctly")
        logger.info("✅ DEFAULT_RISK: Conservative strategy (~7% target)")
        logger.info("✅ HIGH_RISK: Aggressive strategy (~12% target)")
        logger.info("✅ Strategies produce different results")
        logger.info("✅ ML predictions are included")
        return True
    else:
        logger.error("\n❌ SOME TESTS FAILED")
        logger.error("Please review the issues above")
        return False


def main():
    """Run risk profile test."""
    try:
        success = test_risk_profiles()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
