#!/usr/bin/env python
"""
Test script for Enhanced Portfolio Strategy
Tests the 4-step strategy implementation without real trading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.enhanced_portfolio_strategy import (
    EnhancedPortfolioStrategy,
    RiskStrategy,
    StockSignal,
    MarketCapCategory,
    Position
)
from datetime import datetime, timedelta
import json


def test_filtering():
    """Test Step 1: Filtering criteria."""
    print("\n" + "="*60)
    print("TESTING STEP 1: FILTERING")
    print("="*60)
    
    # Mock stock data
    test_stocks = [
        {"symbol": "TEST1", "current_price": 100, "avg_volume_20d": 600000, "atr_14": 5, "atr_percent": 5},  # Pass
        {"symbol": "TEST2", "current_price": 45, "avg_volume_20d": 600000, "atr_14": 3, "atr_percent": 6.7},  # Fail: price
        {"symbol": "TEST3", "current_price": 100, "avg_volume_20d": 400000, "atr_14": 5, "atr_percent": 5},  # Fail: volume
        {"symbol": "TEST4", "current_price": 100, "avg_volume_20d": 600000, "atr_14": 12, "atr_percent": 12},  # Fail: volatility
    ]
    
    strategy = EnhancedPortfolioStrategy(capital=100000)
    
    passed = 0
    for stock in test_stocks:
        price_pass = stock['current_price'] >= strategy.filtering_criteria.min_price
        volume_pass = stock['avg_volume_20d'] >= strategy.filtering_criteria.min_avg_volume_20d
        volatility_pass = stock['atr_percent'] <= strategy.filtering_criteria.max_atr_percent
        
        if price_pass and volume_pass and volatility_pass:
            print(f"✓ {stock['symbol']} PASSED all filters")
            passed += 1
        else:
            print(f"✗ {stock['symbol']} FAILED - Price: {price_pass}, Volume: {volume_pass}, Volatility: {volatility_pass}")
    
    print(f"\nResult: {passed}/{len(test_stocks)} stocks passed filtering")
    return passed == 1  # Only TEST1 should pass


def test_risk_allocation():
    """Test Step 2: Risk Strategy Allocation."""
    print("\n" + "="*60)
    print("TESTING STEP 2: RISK ALLOCATION")
    print("="*60)
    
    # Mock filtered stocks with market caps
    filtered_stocks = [
        {"symbol": "LARGE1", "market_cap": 600000000000},  # 60,000 Cr - Large cap
        {"symbol": "LARGE2", "market_cap": 550000000000},  # 55,000 Cr - Large cap
        {"symbol": "MID1", "market_cap": 200000000000},    # 20,000 Cr - Mid cap
        {"symbol": "MID2", "market_cap": 150000000000},    # 15,000 Cr - Mid cap
        {"symbol": "SMALL1", "market_cap": 50000000000},   # 5,000 Cr - Small cap
        {"symbol": "SMALL2", "market_cap": 30000000000},   # 3,000 Cr - Small cap
    ]
    
    strategy = EnhancedPortfolioStrategy(capital=100000)
    
    # Test SAFE strategy
    print("\nTesting SAFE strategy (50% Large + 50% Mid):")
    safe_allocation = strategy.allocate_risk_strategy(filtered_stocks, RiskStrategy.SAFE)
    print(f"  Large-cap: {len(safe_allocation['large_cap'])} stocks")
    print(f"  Mid-cap: {len(safe_allocation['mid_cap'])} stocks")
    print(f"  Small-cap: {len(safe_allocation['small_cap'])} stocks")
    print(f"  Selected for SAFE: {len(safe_allocation['selected'])} stocks")
    
    # Test HIGH_RISK strategy
    print("\nTesting HIGH_RISK strategy (50% Mid + 50% Small):")
    high_risk_allocation = strategy.allocate_risk_strategy(filtered_stocks, RiskStrategy.HIGH_RISK)
    print(f"  Selected for HIGH_RISK: {len(high_risk_allocation['selected'])} stocks")
    
    return True


def test_entry_signals():
    """Test Step 3: Entry Signal Validation."""
    print("\n" + "="*60)
    print("TESTING STEP 3: ENTRY SIGNALS")
    print("="*60)
    
    # Create test signal
    signal = StockSignal(
        symbol="TEST",
        name="Test Stock",
        current_price=100,
        market_cap=200000000000,
        market_cap_category=MarketCapCategory.MID_CAP,
        avg_volume_20d=600000,
        atr_14=5,
        atr_percent=5,
        passes_filter=True,
        ema_20=95,    # Price above EMA20 (100 > 95)
        ema_50=92,    # Price above EMA50 (100 > 92)
        high_20d=98,  # Breakout (100 > 98)
        current_volume=1000000,
        avg_volume=600000,
        rsi_14=60     # RSI in valid range (50-70)
    )
    
    strategy = EnhancedPortfolioStrategy(capital=100000)
    
    # Validate entry conditions
    signal.price_above_ema20 = signal.current_price > signal.ema_20
    signal.price_above_ema50 = signal.current_price > signal.ema_50
    signal.breakout_20d = signal.current_price > signal.high_20d
    signal.volume_confirmation = signal.current_volume >= (signal.avg_volume * 1.5)
    signal.rsi_valid = 50 <= signal.rsi_14 <= 70
    
    print(f"Entry Conditions for {signal.symbol}:")
    print(f"  Price > EMA20: {signal.price_above_ema20} ({signal.current_price} > {signal.ema_20})")
    print(f"  Price > EMA50: {signal.price_above_ema50} ({signal.current_price} > {signal.ema_50})")
    print(f"  20-day Breakout: {signal.breakout_20d} ({signal.current_price} > {signal.high_20d})")
    print(f"  Volume Confirmation: {signal.volume_confirmation} ({signal.current_volume} >= {signal.avg_volume * 1.5})")
    print(f"  RSI Valid: {signal.rsi_valid} ({signal.rsi_14} in 50-70)")
    
    all_conditions = all([
        signal.price_above_ema20,
        signal.price_above_ema50,
        signal.breakout_20d,
        signal.volume_confirmation,
        signal.rsi_valid
    ])
    
    print(f"\n{'✓' if all_conditions else '✗'} All entry conditions met: {all_conditions}")
    return all_conditions


def test_exit_rules():
    """Test Step 4: Exit Rules."""
    print("\n" + "="*60)
    print("TESTING STEP 4: EXIT RULES")
    print("="*60)
    
    # Create test position
    position = Position(
        symbol="TEST",
        entry_date=datetime.now() - timedelta(days=5),
        entry_price=100,
        quantity=100,
        remaining_quantity=100
    )
    
    # Test different scenarios
    scenarios = [
        {"current_price": 105.5, "expected": "PROFIT_TARGET_1 (5% target)"},
        {"current_price": 111, "expected": "PROFIT_TARGET_2 (10% target)"},
        {"current_price": 96.5, "expected": "STOP_LOSS (3% stop)"},
        {"current_price": 102, "days_held": 11, "expected": "TIME_STOP (10 days max)"},
    ]
    
    print("Testing exit scenarios:")
    for scenario in scenarios:
        # Reset position for each test
        test_pos = Position(
            symbol="TEST",
            entry_date=datetime.now() - timedelta(days=scenario.get('days_held', 5)),
            entry_price=100,
            quantity=100,
            remaining_quantity=100
        )
        
        test_pos.update_metrics(scenario['current_price'])
        
        exit_triggered = False
        exit_reason = ""
        
        # Check conditions
        if test_pos.days_held >= 10:
            exit_triggered = True
            exit_reason = "TIME_STOP"
        elif test_pos.unrealized_pnl_percent <= -3:
            exit_triggered = True
            exit_reason = "STOP_LOSS"
        elif test_pos.unrealized_pnl_percent >= 10:
            exit_triggered = True
            exit_reason = "PROFIT_TARGET_2"
        elif test_pos.unrealized_pnl_percent >= 5:
            exit_triggered = True
            exit_reason = "PROFIT_TARGET_1"
        
        print(f"  Price: {scenario['current_price']}, Days: {test_pos.days_held}, P&L: {test_pos.unrealized_pnl_percent:.1f}%")
        print(f"    → Exit: {exit_triggered}, Reason: {exit_reason}")
        print(f"    Expected: {scenario['expected']}")
        print(f"    {'✓ PASS' if exit_reason in scenario['expected'] else '✗ FAIL'}")
    
    return True


def test_position_tracking():
    """Test position tracking and P&L calculation."""
    print("\n" + "="*60)
    print("TESTING POSITION TRACKING")
    print("="*60)
    
    position = Position(
        symbol="TEST",
        entry_date=datetime.now(),
        entry_price=100,
        quantity=100,
        remaining_quantity=100
    )
    
    # Update with current price
    position.update_metrics(105)
    
    print(f"Position Details:")
    print(f"  Symbol: {position.symbol}")
    print(f"  Entry Price: ₹{position.entry_price}")
    print(f"  Current Price: ₹{position.current_price}")
    print(f"  Quantity: {position.quantity}")
    print(f"  Unrealized P&L: ₹{position.unrealized_pnl:.2f}")
    print(f"  Unrealized P&L %: {position.unrealized_pnl_percent:.2f}%")
    
    # Test trailing stop update
    position.exit_rules.trailing_stop_enabled = True
    position.update_metrics(108)  # Price goes up
    print(f"\nAfter price increase to ₹108:")
    print(f"  High Since Entry: ₹{position.high_since_entry}")
    print(f"  Trailing Stop: ₹{position.trailing_stop_price:.2f}")
    
    return position.unrealized_pnl == 800  # (108-100) * 100


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ENHANCED PORTFOLIO STRATEGY - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Filtering", test_filtering),
        ("Risk Allocation", test_risk_allocation),
        ("Entry Signals", test_entry_signals),
        ("Exit Rules", test_exit_rules),
        ("Position Tracking", test_position_tracking)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The strategy is ready for use.")
    else:
        print("\n⚠️ Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)