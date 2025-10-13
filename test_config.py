#!/usr/bin/env python3
"""
Test script to verify stock_suggestions configuration loading
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from services.config.stock_suggestions_config import get_stock_suggestions_config


def test_config():
    """Test loading and accessing configuration"""
    print("🧪 Testing stock_suggestions configuration...\n")

    try:
        # Load config
        config = get_stock_suggestions_config()
        print("✅ Config loaded successfully\n")

        # Test model selection
        print("📊 MODEL SELECTION:")
        print(f"  Enabled models: {config.get_enabled_models()}")
        print(f"  Traditional min score: {config.get_minimum_score('traditional')}")
        print(f"  Raw LSTM min score: {config.get_minimum_score('raw_lstm')}")
        print(f"  Kronos min score: {config.get_minimum_score('kronos')}")
        print(f"  Allowed recommendations: {config.get_allowed_recommendations()}\n")

        # Test valuation filters
        print("💰 VALUATION FILTERS:")
        print(f"  Upside range: {config.get_minimum_upside_pct()}% - {config.get_maximum_upside_pct()}%")
        print(f"  PE ratio: {config.get_pe_ratio_range()}")
        print(f"  PB ratio: {config.get_pb_ratio_range()}")
        print(f"  ROE: {config.get_roe_range()}\n")

        # Test risk management
        print("⚠️  RISK MANAGEMENT:")
        print(f"  Max risk score: {config.get_maximum_risk_score()}")
        print(f"  Require stop loss: {config.require_stop_loss()}")
        print(f"  Stop loss range: {config.get_stop_loss_range()}")
        print(f"  Min risk/reward ratio: {config.get_minimum_risk_reward_ratio()}\n")

        # Test strategy configs
        print("🎯 STRATEGY CONFIGS:")
        for strategy_key in ['DEFAULT_RISK', 'HIGH_RISK']:
            strategy_config = config.get_strategy_config(strategy_key)
            print(f"  {strategy_key}:")
            print(f"    Name: {strategy_config.get('name')}")
            print(f"    Target gain: {strategy_config.get('target_gain_pct')}")
            print(f"    Stop loss: {strategy_config.get('stop_loss_pct')}%")
            print(f"    Min ML score: {strategy_config.get('min_ml_score')}\n")

        # Test display settings
        print("🖥️  DISPLAY SETTINGS:")
        print(f"  Default limit: {config.get_default_limit()}")
        print(f"  Maximum limit: {config.get_maximum_limit()}")
        print(f"  Sort settings: {config.get_sort_settings()}\n")

        # Test SQL filter building
        print("🔧 SQL FILTER BUILDING:")
        for model in ['traditional', 'raw_lstm', 'kronos']:
            for strategy in ['default_risk', 'high_risk']:
                filters = config.build_sql_filters(model, strategy)
                print(f"  {model}/{strategy}: min_score={filters['min_ml_score']}, min_confidence={filters['min_confidence']}")

        print("\n✅ All configuration tests passed!")

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    test_config()
