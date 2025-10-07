#!/usr/bin/env python3 -u
"""
Train Enhanced ML Models with VERBOSE output
Forces unbuffered output so you can see progress in real-time
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.ml.enhanced_stock_predictor import EnhancedStockPredictor
import logging

# Set up verbose logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
    force=True
)

logger = logging.getLogger()

def print_progress(msg, end='\n'):
    """Print with immediate flush."""
    print(msg, end=end, flush=True)

print_progress("=" * 80)
print_progress("ENHANCED ML MODEL TRAINING (VERBOSE MODE)")
print_progress("=" * 80)
print_progress(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

start_time = time.time()

db_manager = get_database_manager()

try:
    with db_manager.get_session() as session:
        print_progress("✓ Database connection established")

        predictor = EnhancedStockPredictor(session)
        print_progress("✓ Predictor initialized\n")

        # Check if already trained
        if predictor.rf_price_model is not None:
            print_progress("⚠️  Models are already trained!")
            print_progress("Proceeding with retraining...\n")

        # Prepare data
        print_progress("📊 Step 1/5: Preparing training data...")
        df = predictor.prepare_training_data(lookback_days=365)
        print_progress(f"✓ Loaded {len(df):,} training samples")

        if len(df) < 100:
            print_progress(f"❌ Error: Not enough data ({len(df)} samples)")
            sys.exit(1)

        elapsed = time.time() - start_time
        print_progress(f"   Time: {elapsed:.1f}s\n")

        # Add chaos features
        print_progress("🌀 Step 2/5: Adding chaos theory features...")
        print_progress("   (This may take a few minutes...)")
        chaos_start = time.time()

        df = predictor._add_chaos_features(df)

        chaos_time = time.time() - chaos_start
        print_progress(f"✓ Chaos features added")
        print_progress(f"   Time: {chaos_time:.1f}s\n")

        # Select features
        print_progress("🔍 Step 3/5: Selecting features...")
        X = predictor._select_features(df)

        # Filter valid targets
        valid_mask = df['price_change_pct'].notna() & df['max_drawdown_pct'].notna()
        X = X[valid_mask]
        df_valid = df[valid_mask]

        print_progress(f"✓ Selected {len(predictor.feature_columns)} features")
        print_progress(f"✓ {len(df_valid):,} valid samples")

        elapsed = time.time() - start_time
        print_progress(f"   Total time so far: {elapsed/60:.1f} minutes\n")

        # Prepare targets
        y_price = df_valid['price_change_pct'].values
        y_risk = df_valid['max_drawdown_pct'].values

        # Walk-forward cross-validation
        print_progress("🔄 Step 4/5: Walk-forward cross-validation (5 folds)...")
        print_progress("   This is the longest step (8-12 minutes)\n")

        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores_price = []
        cv_scores_risk = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            fold_start = time.time()
            print_progress(f"   Fold {fold}/5:")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train_price, y_val_price = y_price[train_idx], y_price[val_idx]
            y_train_risk, y_val_risk = y_risk[train_idx], y_risk[val_idx]

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train RF price model
            print_progress(f"     Training price model...", end='')
            rf_price = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            rf_price.fit(X_train_scaled, y_train_price)
            price_score = rf_price.score(X_val_scaled, y_val_price)
            cv_scores_price.append(price_score)
            print_progress(f" R²={price_score:.3f}")

            # Train RF risk model
            print_progress(f"     Training risk model...", end='')
            rf_risk = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            rf_risk.fit(X_train_scaled, y_train_risk)
            risk_score = rf_risk.score(X_val_scaled, y_val_risk)
            cv_scores_risk.append(risk_score)

            fold_time = time.time() - fold_start
            print_progress(f" R²={risk_score:.3f}")
            print_progress(f"     Fold complete in {fold_time:.1f}s\n")

        # CV summary
        print_progress(f"✓ Cross-validation complete:")
        print_progress(f"   Price R²: {np.mean(cv_scores_price):.3f} ± {np.std(cv_scores_price):.3f}")
        print_progress(f"   Risk R²: {np.mean(cv_scores_risk):.3f} ± {np.std(cv_scores_risk):.3f}\n")

        # Train final models
        print_progress("🎯 Step 5/5: Training final models on full dataset...")
        final_start = time.time()

        X_scaled = predictor.feature_scaler.fit_transform(X)

        print_progress("   Training Random Forest models...", end='')
        predictor.rf_price_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        predictor.rf_price_model.fit(X_scaled, y_price)

        predictor.rf_risk_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        predictor.rf_risk_model.fit(X_scaled, y_risk)
        print_progress(" Done!")

        # XGBoost if available
        try:
            import xgboost as xgb
            print_progress("   Training XGBoost models...", end='')

            predictor.xgb_price_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            predictor.xgb_price_model.fit(X_scaled, y_price)

            predictor.xgb_risk_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            predictor.xgb_risk_model.fit(X_scaled, y_risk)
            print_progress(" Done!")
        except ImportError:
            print_progress("   XGBoost not available, skipping")

        final_time = time.time() - final_start
        print_progress(f"✓ Final models trained in {final_time:.1f}s")

        # Store stats
        predictor.training_stats = {
            'samples': len(df_valid),
            'features': len(predictor.feature_columns),
            'price_r2': predictor.rf_price_model.score(X_scaled, y_price),
            'risk_r2': predictor.rf_risk_model.score(X_scaled, y_risk),
            'cv_price_r2': np.mean(cv_scores_price),
            'cv_risk_r2': np.mean(cv_scores_risk)
        }

        total_time = time.time() - start_time

        print_progress("\n" + "=" * 80)
        print_progress("✅ TRAINING COMPLETE!")
        print_progress("=" * 80)

        print_progress(f"\n⏱️  Total Training Time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")

        print_progress(f"\n📊 Final Statistics:")
        print_progress(f"  • Training Samples: {predictor.training_stats['samples']:,}")
        print_progress(f"  • Features Used: {predictor.training_stats['features']}")

        print_progress(f"\n📈 Model Performance:")
        print_progress(f"  Random Forest:")
        print_progress(f"    • Price Model R²: {predictor.training_stats['price_r2']:.3f}")
        print_progress(f"    • Risk Model R²: {predictor.training_stats['risk_r2']:.3f}")
        print_progress(f"  Cross-Validation (more realistic):")
        print_progress(f"    • CV Price R²: {predictor.training_stats['cv_price_r2']:.3f}")
        print_progress(f"    • CV Risk R²: {predictor.training_stats['cv_risk_r2']:.3f}")

        print_progress(f"\n✅ Models saved and ready for predictions!")

        print_progress(f"\n📋 Next Steps:")
        print_progress(f"  1. Check status: python3 tools/check_ml_status.py")
        print_progress(f"  2. Test risk profiles: python3 tools/test_risk_profiles.py")

        print_progress("\n" + "=" * 80)
        print_progress(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_progress("=" * 80)

        sys.exit(0)

except KeyboardInterrupt:
    print_progress("\n\n⚠️  Training interrupted by user")
    print_progress("Models are NOT trained. Run again to complete training.")
    sys.exit(1)

except Exception as e:
    print_progress(f"\n\n❌ Training failed with error:")
    print_progress(f"{str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
