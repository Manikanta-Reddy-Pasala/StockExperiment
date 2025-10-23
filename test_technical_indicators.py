#!/usr/bin/env python3
"""
Test script for technical indicators calculator
Tests the simplified system without ML dependencies
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.database import get_database_manager
from src.services.technical.indicators_calculator import get_indicators_calculator
from sqlalchemy import text

def test_technical_indicators():
    """Test technical indicators calculation for a sample stock."""

    print("\n" + "=" * 80)
    print("TESTING TECHNICAL INDICATORS CALCULATOR")
    print("=" * 80)

    db_manager = get_database_manager()

    with db_manager.get_session() as session:
        # Get a sample stock symbol
        query = text("""
            SELECT symbol
            FROM stocks
            WHERE is_active = TRUE
            AND is_tradeable = TRUE
            LIMIT 1
        """)
        result = session.execute(query).fetchone()

        if not result:
            print("❌ No stocks found in database. Run data pipeline first.")
            return False

        test_symbol = result[0]
        print(f"\n📊 Testing with stock: {test_symbol}")

        # Initialize calculator
        calculator = get_indicators_calculator(session)

        # Calculate indicators
        print(f"\n🔄 Calculating technical indicators...")
        indicators = calculator.calculate_all_indicators(test_symbol, lookback_days=252)

        if indicators is None:
            print(f"⚠️  Could not calculate indicators for {test_symbol}")
            print("   This is normal if there's insufficient historical data")
            return False

        # Display results
        print(f"\n✅ Indicators calculated successfully!")
        print(f"\n📈 Technical Indicators for {test_symbol}:")
        print("-" * 80)
        print(f"  RS Rating:     {indicators['rs_rating']:.2f} / 99")
        print(f"  Fast Wave:     {indicators['fast_wave']:.4f}")
        print(f"  Slow Wave:     {indicators['slow_wave']:.4f}")
        print(f"  Delta:         {indicators['delta']:.4f} {'📈 Bullish' if indicators['delta'] > 0 else '📉 Bearish'}")
        print(f"  Buy Signal:    {'✅ YES' if indicators['buy_signal'] else '❌ NO'}")
        print(f"  Sell Signal:   {'⚠️  YES' if indicators['sell_signal'] else '✅ NO'}")
        print("-" * 80)

        # Interpret the signals
        print(f"\n💡 Interpretation:")
        if indicators['rs_rating'] > 67:
            print(f"  • Strong relative strength (RS Rating > 67)")
        elif indicators['rs_rating'] > 33:
            print(f"  • Average relative strength (RS Rating 33-67)")
        else:
            print(f"  • Weak relative strength (RS Rating < 33)")

        if indicators['delta'] > 0:
            print(f"  • Bullish momentum (positive Delta)")
        else:
            print(f"  • Bearish momentum (negative Delta)")

        if indicators['buy_signal']:
            print(f"  • 🎯 BUY SIGNAL: Fast Wave crossed above Slow Wave")
        elif indicators['sell_signal']:
            print(f"  • ⚠️  SELL SIGNAL: Fast Wave crossed below Slow Wave")
        else:
            print(f"  • No crossover signal")

        print("\n" + "=" * 80)
        print("✅ TEST PASSED: Technical indicators calculator is working!")
        print("=" * 80 + "\n")

        return True


def test_batch_calculation():
    """Test batch calculation for multiple stocks."""

    print("\n" + "=" * 80)
    print("TESTING BATCH INDICATORS CALCULATION")
    print("=" * 80)

    db_manager = get_database_manager()

    with db_manager.get_session() as session:
        # Get 5 sample stocks
        query = text("""
            SELECT symbol
            FROM stocks
            WHERE is_active = TRUE
            AND is_tradeable = TRUE
            ORDER BY market_cap DESC NULLS LAST
            LIMIT 5
        """)
        result = session.execute(query)
        symbols = [row[0] for row in result]

        if not symbols:
            print("❌ No stocks found in database. Run data pipeline first.")
            return False

        print(f"\n📊 Testing batch calculation for {len(symbols)} stocks:")
        for symbol in symbols:
            print(f"  - {symbol}")

        # Initialize calculator
        calculator = get_indicators_calculator(session)

        # Calculate indicators in batch
        print(f"\n🔄 Calculating indicators...")
        results = calculator.calculate_indicators_bulk(symbols, lookback_days=252)

        print(f"\n✅ Batch calculation complete!")
        print(f"   Successfully calculated: {len(results)}/{len(symbols)} stocks")

        # Show summary
        if results:
            print(f"\n📊 Summary of Results:")
            print("-" * 80)
            for symbol, indicators in results.items():
                signal = ""
                if indicators['buy_signal']:
                    signal = "🎯 BUY"
                elif indicators['sell_signal']:
                    signal = "⚠️  SELL"
                else:
                    signal = "⏸️  HOLD"

                print(f"  {symbol:20s} | RS: {indicators['rs_rating']:5.1f} | "
                      f"Delta: {indicators['delta']:7.4f} | {signal}")
            print("-" * 80)

        print("\n" + "=" * 80)
        print("✅ TEST PASSED: Batch calculation is working!")
        print("=" * 80 + "\n")

        return True


def test_database_schema():
    """Test that database has the required columns for technical indicators."""

    print("\n" + "=" * 80)
    print("TESTING DATABASE SCHEMA")
    print("=" * 80)

    db_manager = get_database_manager()

    with db_manager.get_session() as session:
        # Check stocks table has indicator columns
        query = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'stocks'
            AND column_name IN ('rs_rating', 'fast_wave', 'slow_wave', 'delta', 'buy_signal', 'sell_signal', 'indicators_last_updated')
            ORDER BY column_name
        """)
        result = session.execute(query)
        columns = [row[0] for row in result]

        required_columns = ['rs_rating', 'fast_wave', 'slow_wave', 'delta', 'buy_signal', 'sell_signal', 'indicators_last_updated']
        missing_columns = set(required_columns) - set(columns)

        print(f"\n📋 Checking stocks table columns:")
        for col in required_columns:
            if col in columns:
                print(f"  ✅ {col}")
            else:
                print(f"  ❌ {col} (MISSING)")

        if missing_columns:
            print(f"\n⚠️  Missing columns: {', '.join(missing_columns)}")
            print(f"   Run migration script: docker exec -it trading_system_db psql -U trader -d trading_system -f /docker-entrypoint-initdb.d/02-add-technical-indicators.sql")
            return False

        # Check daily_suggested_stocks table
        query = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'daily_suggested_stocks'
            AND column_name IN ('rs_rating', 'fast_wave', 'slow_wave', 'delta', 'buy_signal', 'sell_signal')
            ORDER BY column_name
        """)
        result = session.execute(query)
        columns = [row[0] for row in result]

        required_columns = ['rs_rating', 'fast_wave', 'slow_wave', 'delta', 'buy_signal', 'sell_signal']
        missing_columns = set(required_columns) - set(columns)

        print(f"\n📋 Checking daily_suggested_stocks table columns:")
        for col in required_columns:
            if col in columns:
                print(f"  ✅ {col}")
            else:
                print(f"  ❌ {col} (MISSING)")

        if missing_columns:
            print(f"\n⚠️  Missing columns: {', '.join(missing_columns)}")
            print(f"   Run migration script!")
            return False

        print("\n" + "=" * 80)
        print("✅ TEST PASSED: Database schema is correct!")
        print("=" * 80 + "\n")

        return True


if __name__ == '__main__':
    print("\n" + "🧪" * 40)
    print("TECHNICAL INDICATORS SYSTEM TEST")
    print("🧪" * 40)

    # Run tests
    test1 = test_database_schema()
    test2 = test_technical_indicators() if test1 else False
    test3 = test_batch_calculation() if test2 else False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"  Database Schema:           {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"  Single Stock Calculation:  {'✅ PASS' if test2 else '❌ FAIL (or skipped)'}")
    print(f"  Batch Calculation:         {'✅ PASS' if test3 else '❌ FAIL (or skipped)'}")
    print("=" * 80)

    if test1 and test2 and test3:
        print("\n🎉 ALL TESTS PASSED! System is ready to use.")
    elif test1:
        print("\n⚠️  Database schema is correct, but need historical data to calculate indicators.")
        print("   Run data pipeline: python3 run_pipeline.py")
    else:
        print("\n❌ TESTS FAILED. Run database migration first.")

    print()
