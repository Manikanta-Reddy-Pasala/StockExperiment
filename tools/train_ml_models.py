#!/usr/bin/env python3
"""
Train Enhanced ML Models
Trains RF + XGBoost ensemble with walk-forward cross-validation
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor

print("=" * 80)
print("ENHANCED ML MODEL TRAINING")
print("=" * 80)
print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nThis will take 10-15 minutes due to:")
print("  • 5-fold walk-forward cross-validation")
print("  • Training on ~500k samples")
print("  • RF + XGBoost ensemble (4 models total)")
print("  • Chaos theory feature engineering")

print("\n" + "=" * 80)
print("TRAINING IN PROGRESS...")
print("=" * 80)

start_time = time.time()

db_manager = get_database_manager()

try:
    with db_manager.get_session() as session:
        predictor = EnhancedStockPredictor(session)

        # Check if already trained
        if predictor.rf_price_model is not None:
            print("\n⚠️  Models are already trained!")
            print("\nDo you want to retrain? (This will overwrite existing models)")
            # Since this might run in background, we'll just proceed
            print("Proceeding with retraining...\n")

        # Train with walk-forward CV
        print("Training with walk-forward cross-validation...")
        stats = predictor.train_with_walk_forward(
            lookback_days=365,
            n_splits=5
        )

        elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)

        print(f"\n⏱️  Training Time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")

        print(f"\n📊 Training Statistics:")
        print(f"  • Training Samples: {stats['samples']:,}")
        print(f"  • Features Used: {stats['features']}")
        print(f"  • Cross-Validation Folds: 5")

        print(f"\n📈 Model Performance:")
        print(f"  Random Forest:")
        print(f"    • Price Model R²: {stats['price_r2']:.3f}")
        print(f"    • Risk Model R²: {stats['risk_r2']:.3f}")
        print(f"    • CV Price R² (avg): {stats['cv_price_r2']:.3f}")
        print(f"    • CV Risk R² (avg): {stats['cv_risk_r2']:.3f}")

        print(f"\n✅ Models saved and ready for predictions")

        print(f"\n📋 Next Steps:")
        print(f"  1. Check status: python3 tools/check_ml_status.py")
        print(f"  2. Test risk profiles: python3 tools/test_risk_profiles.py")
        print(f"  3. Run saga: python3 tools/test_complete_saga_flow.py")

        print("\n" + "=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        sys.exit(0)

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    print("Models are NOT trained. Please run again to complete training.")
    sys.exit(1)

except Exception as e:
    print(f"\n\n❌ Training failed with error:")
    print(f"{str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
