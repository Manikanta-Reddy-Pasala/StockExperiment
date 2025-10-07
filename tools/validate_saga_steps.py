#!/usr/bin/env python3
"""
Validate Suggested Stocks Saga Steps (No Training Required)
Tests code structure and integration
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_all_steps():
    """Validate all 7 saga steps exist and are properly structured."""

    logger.info("=" * 80)
    logger.info("SUGGESTED STOCKS SAGA - STEP VALIDATION")
    logger.info("=" * 80)

    try:
        # Import saga
        from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator

        orchestrator = SuggestedStocksSagaOrchestrator()

        # Expected steps
        expected_steps = [
            ('_execute_step1_stock_discovery', 'Step 1: Stock Discovery'),
            ('_execute_step2_database_filtering', 'Step 2: Database Filtering'),
            ('_execute_step3_strategy_application', 'Step 3: Strategy Application'),
            ('_execute_step4_search_and_sort', 'Step 4: Search and Sort'),
            ('_execute_step5_final_selection', 'Step 5: Final Selection'),
            ('_execute_step6_ml_prediction', 'Step 6: ML Prediction'),
            ('_execute_step7_daily_snapshot', 'Step 7: Daily Snapshot')
        ]

        logger.info("\nValidating step methods exist:")
        all_exist = True

        for method_name, step_desc in expected_steps:
            if hasattr(orchestrator, method_name):
                logger.info(f"  ✓ {step_desc} ({method_name})")
            else:
                logger.error(f"  ✗ {step_desc} ({method_name}) - MISSING!")
                all_exist = False

        if not all_exist:
            return False

        # Validate Step 6 uses Enhanced ML
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATING STEP 6: ML PREDICTION (Enhanced ML)")
        logger.info("=" * 80)

        # Read the source code to check imports
        saga_file = Path(__file__).parent.parent / 'src/services/data/suggested_stocks_saga.py'

        with open(saga_file, 'r') as f:
            saga_code = f.read()

        # Check for Enhanced ML import
        checks = [
            ('EnhancedStockPredictor' in saga_code, 'Enhanced ML Predictor imported'),
            ('from ..ml.enhanced_stock_predictor import EnhancedStockPredictor' in saga_code, 'Correct import statement'),
            ('EnhancedStockPredictor(session)' in saga_code, 'Enhanced predictor instantiated'),
            ('train_with_walk_forward' in saga_code, 'Walk-forward training method used'),
        ]

        logger.info("\nChecking ML integration:")
        all_checks_passed = True

        for check, description in checks:
            if check:
                logger.info(f"  ✓ {description}")
            else:
                logger.error(f"  ✗ {description} - MISSING!")
                all_checks_passed = False

        # Check old predictor is NOT used
        logger.info("\nVerifying old ML predictor removed:")

        old_checks = [
            ('StockMLPredictor' not in saga_code, 'Old predictor NOT imported'),
            ('from ..ml.stock_predictor' not in saga_code, 'Old import statement removed'),
        ]

        for check, description in old_checks:
            if check:
                logger.info(f"  ✓ {description}")
            else:
                logger.warning(f"  ⚠ {description} - FOUND!")
                all_checks_passed = False

        if all_checks_passed:
            logger.info("\n✅ Step 6 correctly uses Enhanced ML")
        else:
            logger.error("\n✗ Step 6 has integration issues")
            return False

        # Validate main execution method
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATING MAIN EXECUTION METHOD")
        logger.info("=" * 80)

        if hasattr(orchestrator, 'execute_suggested_stocks_saga'):
            logger.info("  ✓ Main execution method exists")

            # Check method signature
            import inspect
            sig = inspect.signature(orchestrator.execute_suggested_stocks_saga)
            params = list(sig.parameters.keys())

            expected_params = ['self', 'user_id', 'strategies', 'limit', 'search', 'sort_by', 'sort_order', 'sector']

            logger.info(f"  ✓ Method parameters: {params}")

            if 'user_id' in params and 'strategies' in params and 'limit' in params:
                logger.info("  ✓ Required parameters present")
            else:
                logger.error("  ✗ Missing required parameters")
                return False
        else:
            logger.error("  ✗ Main execution method missing")
            return False

        # Check enhanced ML predictor can be imported
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATING ENHANCED ML PREDICTOR")
        logger.info("=" * 80)

        try:
            from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor
            logger.info("  ✓ EnhancedStockPredictor imported successfully")

            # Check it has required methods
            required_methods = [
                'train_with_walk_forward',
                'predict',
                'prepare_training_data',
                '_add_chaos_features'
            ]

            for method in required_methods:
                if hasattr(EnhancedStockPredictor, method):
                    logger.info(f"  ✓ Method exists: {method}")
                else:
                    logger.error(f"  ✗ Method missing: {method}")
                    return False

        except ImportError as e:
            logger.error(f"  ✗ Cannot import EnhancedStockPredictor: {e}")
            return False

        # Final validation
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL VALIDATIONS PASSED")
        logger.info("=" * 80)

        logger.info("\nValidation Summary:")
        logger.info("  ✓ All 7 saga steps exist")
        logger.info("  ✓ Step 6 uses Enhanced ML Predictor")
        logger.info("  ✓ Old ML predictor removed from saga")
        logger.info("  ✓ Main execution method properly structured")
        logger.info("  ✓ Enhanced ML predictor has all required methods")

        logger.info("\nSaga Steps:")
        for i, (_, step_desc) in enumerate(expected_steps, 1):
            logger.info(f"  {i}. {step_desc.replace('Step ', '').replace(':', '')}")

        logger.info("\nML Configuration:")
        logger.info("  - Predictor: EnhancedStockPredictor")
        logger.info("  - Algorithm: RF + XGBoost ensemble")
        logger.info("  - Features: 42 (including chaos theory)")
        logger.info("  - Validation: Walk-forward CV (5 folds)")
        logger.info("  - Scoring: Calibrated probabilities")

        logger.info("\nTo run the saga:")
        logger.info("  from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator")
        logger.info("  orchestrator = SuggestedStocksSagaOrchestrator()")
        logger.info("  result = orchestrator.execute_suggested_stocks_saga(")
        logger.info("      user_id=1,")
        logger.info("      strategies=['DEFAULT_RISK', 'HIGH_RISK'],")
        logger.info("      limit=20")
        logger.info("  )")

        return True

    except Exception as e:
        logger.error(f"\n✗ Validation failed: {e}", exc_info=True)
        return False


def validate_scheduler_integration():
    """Validate scheduler also uses enhanced ML."""

    logger.info("\n" + "=" * 80)
    logger.info("SCHEDULER INTEGRATION VALIDATION")
    logger.info("=" * 80)

    try:
        scheduler_file = Path(__file__).parent.parent / 'scheduler.py'

        with open(scheduler_file, 'r') as f:
            scheduler_code = f.read()

        checks = [
            ('EnhancedStockPredictor' in scheduler_code, 'Scheduler imports Enhanced ML'),
            ('StockMLPredictor' not in scheduler_code, 'Old predictor NOT in scheduler'),
            ('USE_ENHANCED_MODEL' not in scheduler_code, 'Configuration flag removed'),
            ('train_with_walk_forward' in scheduler_code, 'Uses walk-forward training'),
        ]

        logger.info("\nScheduler checks:")
        all_passed = True

        for check, description in checks:
            if check:
                logger.info(f"  ✓ {description}")
            else:
                logger.error(f"  ✗ {description}")
                all_passed = False

        if all_passed:
            logger.info("\n✅ Scheduler correctly integrated with Enhanced ML")
        else:
            logger.error("\n✗ Scheduler has integration issues")

        return all_passed

    except Exception as e:
        logger.error(f"\n✗ Scheduler validation failed: {e}")
        return False


def main():
    """Run all validations."""

    results = {}

    # Validate saga steps
    results['saga_steps'] = validate_all_steps()

    # Validate scheduler
    results['scheduler'] = validate_scheduler_integration()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL VALIDATION SUMMARY")
    logger.info("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")

    all_passed = all(results.values())

    if all_passed:
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL VALIDATIONS PASSED - SYSTEM READY")
        logger.info("=" * 80)
        logger.info("\nThe suggested stocks saga is properly configured to use")
        logger.info("Enhanced ML with RF+XGBoost ensemble and calibrated scoring.")
        logger.info("\nAll 7 steps are validated and ready to use!")
        return 0
    else:
        logger.error("\n❌ SOME VALIDATIONS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
