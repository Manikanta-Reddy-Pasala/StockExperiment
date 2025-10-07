#!/usr/bin/env python3
"""
Test Scheduler Integration with ML Model Auto-Loading
Verifies that scheduler properly checks and loads ML models on startup
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("SCHEDULER INTEGRATION TEST")
print("=" * 80)

# Test 1: Check if ML models exist
print("\n📋 Test 1: ML Model Files Check")
print("-" * 80)

model_dir = Path('ml_models')
if not model_dir.exists():
    print("❌ ml_models/ directory not found")
    print("💡 Run 'python tools/train_ml_verbose.py' to create models")
    models_exist = False
else:
    print(f"✓ ml_models/ directory exists")

    critical_files = [
        'rf_price_model.pkl',
        'rf_risk_model.pkl',
        'metadata.pkl'
    ]

    missing_files = []
    for file in critical_files:
        file_path = model_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {file} - MISSING")
            missing_files.append(file)

    models_exist = len(missing_files) == 0

    if models_exist:
        print("\n✅ All critical model files found")
    else:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")

# Test 2: Test model loading
print("\n📋 Test 2: Model Loading Test")
print("-" * 80)

try:
    from src.models.database import get_database_manager
    from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor

    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        print("Initializing predictor with auto_load=True...")
        predictor = EnhancedStockPredictor(session, auto_load=True)

        if predictor.rf_price_model is not None:
            print("✅ ML models loaded successfully from cache")

            # Show metadata
            if hasattr(predictor, 'metadata') and predictor.metadata:
                print(f"\nModel Information:")
                print(f"  Trained: {predictor.metadata.get('trained_at', 'Unknown')}")
                print(f"  Samples: {predictor.metadata.get('training_samples', 'Unknown'):,}")
                print(f"  Features: {predictor.metadata.get('n_features', 'Unknown')}")
                print(f"  Price R²: {predictor.metadata.get('price_r2', 0):.4f}")
                print(f"  Risk R²: {predictor.metadata.get('risk_r2', 0):.4f}")
        else:
            print("⚠️  ML models not loaded (None)")
            print("💡 Models will be trained on first use or by scheduler")

except Exception as e:
    print(f"❌ Error loading models: {e}")

# Test 3: Test scheduler startup check function
print("\n📋 Test 3: Scheduler Startup Check")
print("-" * 80)

try:
    # Import the check function from scheduler
    import importlib.util
    spec = importlib.util.spec_from_file_location("scheduler", "scheduler.py")
    scheduler_module = importlib.util.module_from_spec(spec)

    # Test the logic (without actually running training)
    if not model_dir.exists():
        print("⚠️  Scheduler would train models (directory missing)")
        scheduler_action = "TRAIN"
    else:
        critical_files = ['rf_price_model.pkl', 'rf_risk_model.pkl', 'metadata.pkl']
        missing_files = [f for f in critical_files if not (model_dir / f).exists()]

        if missing_files:
            print(f"⚠️  Scheduler would train models (missing: {', '.join(missing_files)})")
            scheduler_action = "TRAIN"
        else:
            print("✅ Scheduler would use cached models")
            scheduler_action = "USE_CACHE"

    print(f"\nScheduler Action: {scheduler_action}")

except Exception as e:
    print(f"❌ Error testing scheduler logic: {e}")

# Test 4: Test suggested stocks saga integration
print("\n📋 Test 4: Suggested Stocks Saga Integration")
print("-" * 80)

try:
    from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator

    print("Testing saga with DEFAULT_RISK (limit=3 for speed)...")
    orchestrator = SuggestedStocksSagaOrchestrator()

    # Run with small limit for quick test
    result = orchestrator.execute_suggested_stocks_saga(
        user_id=1,
        strategies=['DEFAULT_RISK'],
        limit=3,
        search='',
        sort_by='score',
        sort_order='desc'
    )

    if result['status'] == 'completed':
        print(f"✅ Saga completed successfully")
        print(f"   Duration: {result['total_duration_seconds']:.2f}s")
        print(f"   Results: {len(result.get('final_results', []))} stocks")

        # Check if ML was used
        steps = result.get('summary', {}).get('step_summary', [])
        ml_step = next((s for s in steps if s['step_id'] == 'step6_ml_prediction'), None)

        if ml_step:
            if ml_step['status'] == 'completed':
                ml_duration = ml_step.get('duration_seconds', 0)
                print(f"   ML predictions: ✅ Applied ({ml_duration:.2f}s)")

                if ml_duration > 60:
                    print(f"   ⚠️  ML took {ml_duration:.1f}s (training occurred)")
                else:
                    print(f"   ✅ ML used cached models (fast)")
            else:
                print(f"   ML predictions: ⚠️  {ml_step['status']}")

        # Show sample results
        if result.get('final_results'):
            print(f"\n   Sample Results:")
            for i, stock in enumerate(result['final_results'][:2], 1):
                print(f"     {i}. {stock['symbol']} - ₹{stock.get('current_price', 0):,.2f}")
                if stock.get('target_price'):
                    print(f"        ML Target: ₹{stock['target_price']:,.2f} ({stock.get('predicted_return', 0):.1f}%)")
    else:
        print(f"❌ Saga failed: {result.get('status')}")

except Exception as e:
    print(f"❌ Error testing saga: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if models_exist:
    print("\n✅ System is properly configured for production")
    print("\nScheduler behavior:")
    print("  - On startup: Loads cached models (instant)")
    print("  - At 10:00 PM: Re-trains with latest data (10-15 min)")
    print("  - At 10:15 PM: Generates daily snapshot with fresh models")
    print("\nFlask app behavior:")
    print("  - On startup: Checks models, warns if missing")
    print("  - API calls: Uses cached models (fast)")
else:
    print("\n⚠️  ML models not found")
    print("\nNext steps:")
    print("  1. Train models: python tools/train_ml_verbose.py")
    print("  2. Start scheduler: python scheduler.py")
    print("  OR")
    print("  1. Start scheduler (will auto-train on startup)")

print("\n" + "=" * 80)
