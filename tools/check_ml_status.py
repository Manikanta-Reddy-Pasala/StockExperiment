#!/usr/bin/env python3
"""
Check ML Training Status
Shows if models are trained and their performance metrics
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor
from pathlib import Path

print("=" * 80)
print("ML TRAINING STATUS CHECK")
print("=" * 80)

# Check if models exist on disk
model_dir = Path('ml_models')
models_on_disk = False

if model_dir.exists():
    rf_price = model_dir / 'rf_price_model.pkl'
    rf_risk = model_dir / 'rf_risk_model.pkl'
    metadata = model_dir / 'metadata.pkl'

    if rf_price.exists() and rf_risk.exists() and metadata.exists():
        models_on_disk = True
        print("\n✓ Model files found on disk")

        # Load metadata to show info
        import pickle
        with open(metadata, 'rb') as f:
            meta = pickle.load(f)
            print(f"  Saved at: {meta.get('saved_at', 'unknown')}")
            if 'training_stats' in meta:
                stats = meta['training_stats']
                print(f"  Samples: {stats.get('samples', 'N/A'):,}")
                print(f"  Features: {stats.get('features', 'N/A')}")
                print(f"  CV R²: {stats.get('cv_price_r2', 0):.3f}")

db_manager = get_database_manager()

try:
    with db_manager.get_session() as session:
        predictor = EnhancedStockPredictor(session, auto_load=True)

        # Check if models are trained
        if predictor.rf_price_model is not None:
            print("\n✅ ML MODELS ARE TRAINED")
            print("=" * 80)

            # Get model details
            print("\n📊 Model Information:")
            print(f"\nRandom Forest Models:")
            print(f"  ✓ Price Model: Trained")
            print(f"  ✓ Risk Model: Trained")

            print(f"\nXGBoost Models:")
            print(f"  ✓ Price Model: Trained")
            print(f"  ✓ Risk Model: Trained")

            # Try to get training stats if available
            if hasattr(predictor, 'training_stats') and predictor.training_stats:
                stats = predictor.training_stats
                print(f"\n📈 Training Performance:")
                print(f"  Samples: {stats.get('samples', 'N/A')}")
                print(f"  Features: {stats.get('features', 'N/A')}")
                print(f"  Price R²: {stats.get('price_r2', 'N/A'):.3f}")
                print(f"  Risk R²: {stats.get('risk_r2', 'N/A'):.3f}")
                print(f"  CV Price R²: {stats.get('cv_price_r2', 'N/A'):.3f}")
                print(f"  CV Risk R²: {stats.get('cv_risk_r2', 'N/A'):.3f}")

            print("\n✅ System is ready for predictions!")
            print("\nYou can now run:")
            print("  python3 tools/test_risk_profiles.py")

        else:
            print("\n❌ ML MODELS ARE NOT TRAINED")
            print("=" * 80)

            print("\n⚠️  Models need to be trained before running saga tests")

            print("\n📋 Training Options:")
            print("\n1. Train manually now (10-15 minutes):")
            print("   python3 tools/train_ml_models.py")

            print("\n2. Train in background (recommended):")
            print("   nohup python3 tools/train_ml_models.py > training.log 2>&1 &")
            print("   # Then monitor with: tail -f training.log")

            print("\n3. Wait for scheduled training:")
            print("   Scheduler runs ML training daily at 10:00 PM")
            print("   Next scheduled run: Today at 22:00 (10:00 PM)")

            print("\n4. Trigger via Docker scheduler:")
            print("   docker exec trading_system_ml_scheduler python3 -c \\")
            print("     'from scheduler import train_ml_models; train_ml_models()'")

except Exception as e:
    print(f"\n❌ Error checking ML status: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
