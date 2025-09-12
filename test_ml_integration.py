#!/usr/bin/env python3
"""
Test script for ML integration in StockExperiment
"""
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all ML modules can be imported successfully"""
    print("Testing ML module imports...")
    
    try:
        # Test ML services imports
        from src.services.ml.data_service import get_stock_data, create_features
        print("✅ data_service imported successfully")
        
        from src.services.ml.training_service import train_and_tune_models
        print("✅ training_service imported successfully")
        
        from src.services.ml.prediction_service import get_prediction
        print("✅ prediction_service imported successfully")
        
        from src.services.ml.backtest_service import run_backtest, run_backtest_for_all_stocks
        print("✅ backtest_service imported successfully")
        
        # Test utilities
        from src.utils.ml_helpers import save_model, load_model, save_lstm_model, load_lstm_model
        print("✅ ml_helpers imported successfully")
        
        print("\n🎉 All ML modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_data_service():
    """Test the data service functionality"""
    print("\nTesting data service...")
    
    try:
        from src.services.ml.data_service import get_stock_data, create_features
        import pandas as pd
        import numpy as np
        
        # Create a simple test dataframe instead of relying on external API
        print("Creating sample test data...")
        
        # Create sample stock data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105,
            'Low': np.random.randn(100).cumsum() + 95,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        print(f"✅ Created sample data with {len(sample_data)} rows")
        
        # Test feature creation
        df_featured, features = create_features(sample_data)
        print(f"✅ Created {len(features)} features: {features}")
        print(f"✅ Featured dataframe has {len(df_featured)} rows")
        
        # Test that all required features are present
        expected_features = ['Return', 'MA5', 'MA10', 'MA20', 'STD5', 'Volume_Change', 'RSI']
        if all(feat in features for feat in expected_features):
            print("✅ All expected features are present")
            return True
        else:
            print("❌ Some expected features are missing")
            return False
        
    except Exception as e:
        print(f"❌ Data service test failed: {e}")
        return False

def test_flask_routes():
    """Test that Flask routes can be imported"""
    print("\nTesting Flask routes...")
    
    try:
        # Add the src path to be able to import the routes
        import sys
        import os
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from web.routes.ml import ml_bp
        print("✅ ML Blueprint imported successfully")
        
        # Check if blueprint is properly configured
        if hasattr(ml_bp, 'url_prefix'):
            print(f"✅ Blueprint URL prefix: {ml_bp.url_prefix}")
        
        # Check if blueprint has deferred functions (routes)
        if hasattr(ml_bp, 'deferred_functions') and ml_bp.deferred_functions:
            print(f"✅ Blueprint has {len(ml_bp.deferred_functions)} registered functions")
            return True
        else:
            print("❌ Blueprint has no registered functions")
            return False
        
    except Exception as e:
        print(f"❌ Flask routes test failed: {e}")
        return False

def test_model_directory():
    """Test that model directory is accessible"""
    print("\nTesting model directory...")
    
    try:
        from src.utils.ml_helpers import get_model_dir
        
        model_dir = get_model_dir()
        print(f"✅ Model directory: {model_dir}")
        
        if os.path.exists(model_dir):
            print("✅ Model directory exists")
        else:
            print("❌ Model directory does not exist")
            return False
            
        # Test if we can write to the directory
        test_file = os.path.join(model_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✅ Model directory is writable")
        
        return True
        
    except Exception as e:
        print(f"❌ Model directory test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting ML Integration Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_service,
        test_flask_routes,
        test_model_directory
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ML integration is ready.")
        print("\n📋 Next steps:")
        print("1. Install new dependencies: pip install -r requirements.txt")
        print("2. Start your Flask app: python src/main.py")
        print("3. Visit /ml-prediction page to test the interface")
        print("4. Train your first model via the web interface")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
