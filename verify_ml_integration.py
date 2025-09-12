#!/usr/bin/env python3
"""
Simple integration verification script
"""
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Quick verification that ML integration is working"""
    print("🔍 Quick ML Integration Verification")
    print("=" * 40)
    
    success_count = 0
    
    # Test 1: Basic imports
    try:
        # Add paths to sys.path for the imports to work
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        ml_services_path = os.path.join(src_path, 'services', 'ml')
        utils_path = os.path.join(src_path, 'utils')
        
        sys.path.insert(0, ml_services_path)
        sys.path.insert(0, utils_path)
        
        from data_service import get_stock_data, create_features
        import ml_helpers
        print("✅ All ML modules can be imported")
        success_count += 1
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Feature engineering
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.randn(50).cumsum() + 100,
            'High': np.random.randn(50).cumsum() + 105,
            'Low': np.random.randn(50).cumsum() + 95,
            'Close': np.random.randn(50).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 50)
        }, index=dates)
        
        df_featured, features = create_features(sample_data)
        assert len(features) == 7
        assert 'RSI' in features
        print("✅ Feature engineering works correctly")
        success_count += 1
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
    
    # Test 3: Model directory
    try:
        model_dir = ml_helpers.get_model_dir()
        assert os.path.exists(model_dir)
        print("✅ Model directory is accessible")
        success_count += 1
    except Exception as e:
        print(f"❌ Model directory test failed: {e}")
    
    print("=" * 40)
    print(f"📊 Verification: {success_count}/3 components working")
    
    if success_count == 3:
        print("🎉 ML integration is ready!")
        print("\n📋 Next steps:")
        print("1. Start your Flask app: python src/main.py")
        print("2. Visit: http://localhost:5001/ml-prediction")
        print("3. Train your first model using the web interface")
        print("\n💡 Example symbols to try: AAPL, MSFT, GOOGL, TSLA")
        return True
    else:
        print("❌ Some components need attention")
        return False

if __name__ == "__main__":
    main()
