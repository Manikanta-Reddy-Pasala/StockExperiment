#!/usr/bin/env python3
"""
Comprehensive Test for Suggested Stocks Saga
Tests all 7 steps with detailed validation
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_suggested_stocks_saga():
    """Test all 7 steps of suggested stocks saga."""

    logger.info("=" * 80)
    logger.info("SUGGESTED STOCKS SAGA - COMPREHENSIVE TEST")
    logger.info("=" * 80)
    logger.info("")

    # Initialize orchestrator
    orchestrator = SuggestedStocksSagaOrchestrator()

    # Execute saga with detailed tracking
    logger.info("Executing saga with parameters:")
    logger.info("  User ID: 1")
    logger.info("  Strategies: ['DEFAULT_RISK', 'HIGH_RISK']")
    logger.info("  Limit: 20")
    logger.info("")

    result = orchestrator.execute_suggested_stocks_saga(
        user_id=1,
        strategies=['DEFAULT_RISK', 'HIGH_RISK'],
        limit=20
    )

    logger.info("\n" + "=" * 80)
    logger.info("SAGA EXECUTION COMPLETE")
    logger.info("=" * 80)

    # Overall status
    logger.info(f"\nOverall Status: {result['status']}")
    logger.info(f"Total Duration: {result['total_duration_seconds']:.2f} seconds")
    logger.info(f"Final Result Count: {result['summary'].get('final_result_count', 0)}")

    # Analyze each step
    logger.info("\n" + "=" * 80)
    logger.info("STEP-BY-STEP VALIDATION")
    logger.info("=" * 80)

    steps = result['summary'].get('step_summary', [])

    all_passed = True

    for i, step in enumerate(steps, 1):
        step_id = step['step_id']
        step_name = step['name']
        status = step['status']
        duration = step.get('duration_seconds', 0)

        # Status emoji
        if status == 'completed':
            status_emoji = "✅"
        elif status == 'failed':
            status_emoji = "❌"
            all_passed = False
        elif status == 'skipped':
            status_emoji = "⏭️"
        else:
            status_emoji = "⚠️"
            all_passed = False

        logger.info(f"\n{status_emoji} Step {i}: {step_name} ({step_id})")
        logger.info(f"   Status: {status.upper()}")
        logger.info(f"   Duration: {duration:.2f}s")
        logger.info(f"   Input: {step.get('input_count', 0)} items")
        logger.info(f"   Output: {step.get('output_count', 0)} items")

        if step.get('filtered_count', 0) > 0:
            logger.info(f"   Filtered: {step['filtered_count']} items")

        if step.get('rejected_count', 0) > 0:
            logger.info(f"   Rejected: {step['rejected_count']} items")

        # Step-specific validation
        if step_id == 'step1_stock_discovery':
            validate_step1(step)
        elif step_id == 'step2_database_filtering':
            validate_step2(step)
        elif step_id == 'step3_strategy_application':
            validate_step3(step)
        elif step_id == 'step4_search_and_sort':
            validate_step4(step)
        elif step_id == 'step5_final_selection':
            validate_step5(step)
        elif step_id == 'step6_ml_prediction':
            validate_step6(step)
        elif step_id == 'step7_daily_snapshot':
            validate_step7(step)

        if step.get('error_message'):
            logger.error(f"   ERROR: {step['error_message']}")

    # Final results validation
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS VALIDATION")
    logger.info("=" * 80)

    final_results = result.get('final_results', [])
    logger.info(f"\nTotal Suggested Stocks: {len(final_results)}")

    if final_results:
        logger.info("\nSample Stock (First Result):")
        sample = final_results[0]
        logger.info(f"  Symbol: {sample.get('symbol', 'N/A')}")
        logger.info(f"  Current Price: ₹{sample.get('current_price', 0):,.2f}")
        logger.info(f"  Target Price: ₹{sample.get('target_price', 0):,.2f}")
        logger.info(f"  Upside: {sample.get('upside_percentage', 0):.2f}%")
        logger.info(f"  Strategy: {sample.get('strategy', 'N/A')}")
        logger.info(f"  Score: {sample.get('score', 0):.2f}")

        # ML predictions validation
        if 'ml_prediction_score' in sample:
            logger.info(f"\n  ML Predictions:")
            logger.info(f"    Prediction Score: {sample['ml_prediction_score']:.3f}")
            logger.info(f"    Price Target: ₹{sample.get('ml_price_target', 0):,.2f}")
            logger.info(f"    Confidence: {sample.get('ml_confidence', 0):.3f}")
            logger.info(f"    Risk Score: {sample.get('ml_risk_score', 0):.3f}")
            logger.info(f"    Predicted Change: {sample.get('predicted_change_pct', 0):.2f}%")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    if all_passed and result['status'] == 'completed':
        logger.info("\n✅ ALL STEPS PASSED")
        logger.info(f"✅ Generated {len(final_results)} suggested stocks")
        logger.info(f"✅ Total execution time: {result['total_duration_seconds']:.2f}s")
        return True
    else:
        logger.error("\n❌ SOME STEPS FAILED")
        if result.get('errors'):
            logger.error("Errors:")
            for error in result['errors']:
                logger.error(f"  - {error}")
        return False


def validate_step1(step):
    """Validate Step 1: Stock Discovery."""
    logger.info("   Validation: Stock Discovery")
    output_count = step.get('output_count', 0)

    if output_count > 0:
        logger.info(f"   ✓ Discovered {output_count} stocks from exchange")
    else:
        logger.warning("   ⚠ No stocks discovered")


def validate_step2(step):
    """Validate Step 2: Database Filtering."""
    logger.info("   Validation: Database Filtering")
    filtered = step.get('filtered_count', 0)
    output = step.get('output_count', 0)

    if output > 0:
        logger.info(f"   ✓ {output} stocks passed database filters")
        if filtered > 0:
            logger.info(f"   ✓ Filtered out {filtered} stocks (missing data, etc.)")
    else:
        logger.warning("   ⚠ No stocks passed database filtering")


def validate_step3(step):
    """Validate Step 3: Strategy Application."""
    logger.info("   Validation: Strategy Application")
    metadata = step.get('metadata', {})

    if 'strategies_applied' in metadata:
        strategies = metadata['strategies_applied']
        logger.info(f"   ✓ Applied {len(strategies)} strategies:")
        for strategy in strategies:
            logger.info(f"     - {strategy}")

    output = step.get('output_count', 0)
    if output > 0:
        logger.info(f"   ✓ {output} stocks matched strategies")


def validate_step4(step):
    """Validate Step 4: Search and Sort."""
    logger.info("   Validation: Search and Sort")
    metadata = step.get('metadata', {})

    if metadata.get('search_applied'):
        logger.info(f"   ✓ Search query applied: '{metadata.get('search_query')}'")

    if metadata.get('sort_applied'):
        logger.info(f"   ✓ Sorted by: {metadata.get('sort_by')} ({metadata.get('sort_order')})")


def validate_step5(step):
    """Validate Step 5: Final Selection."""
    logger.info("   Validation: Final Selection")
    output = step.get('output_count', 0)
    metadata = step.get('metadata', {})

    if output > 0:
        logger.info(f"   ✓ Selected top {output} stocks")
        if 'limit' in metadata:
            logger.info(f"   ✓ Applied limit: {metadata['limit']}")


def validate_step6(step):
    """Validate Step 6: ML Prediction (Enhanced ML)."""
    logger.info("   Validation: ML Prediction (Enhanced)")
    metadata = step.get('metadata', {})

    # Check if enhanced ML was used
    predictor_type = metadata.get('predictor_type', 'Unknown')
    logger.info(f"   Predictor: {predictor_type}")

    if 'EnhancedStockPredictor' in predictor_type:
        logger.info("   ✅ Using Enhanced ML Predictor")
        logger.info("      - RF + XGBoost ensemble")
        logger.info("      - 42 features + chaos theory")
        logger.info("      - Calibrated scoring")
    else:
        logger.warning(f"   ⚠️ Using: {predictor_type}")

    if metadata.get('predictions_applied'):
        count = metadata.get('predictions_count', 0)
        logger.info(f"   ✓ Applied ML predictions to {count} stocks")

        # Check for prediction metrics
        if 'avg_ml_score' in metadata:
            logger.info(f"   ✓ Average ML score: {metadata['avg_ml_score']:.3f}")
        if 'avg_confidence' in metadata:
            logger.info(f"   ✓ Average confidence: {metadata['avg_confidence']:.3f}")


def validate_step7(step):
    """Validate Step 7: Daily Snapshot."""
    logger.info("   Validation: Daily Snapshot")
    metadata = step.get('metadata', {})

    if metadata.get('snapshot_saved'):
        count = metadata.get('records_saved', 0)
        logger.info(f"   ✓ Saved {count} records to daily snapshot")

        if 'snapshot_date' in metadata:
            logger.info(f"   ✓ Snapshot date: {metadata['snapshot_date']}")


def main():
    """Run saga test."""
    try:
        success = test_suggested_stocks_saga()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
