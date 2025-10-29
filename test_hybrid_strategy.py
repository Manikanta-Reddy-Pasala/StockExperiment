#!/usr/bin/env python3
"""
Test Hybrid Strategy Calculator
Tests the complete hybrid strategy on a few sample stocks
"""

import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_hybrid_strategy():
    """Test hybrid strategy calculator with sample stocks."""

    print("\n" + "=" * 80)
    print("HYBRID STRATEGY TEST")
    print("=" * 80)

    try:
        from src.models.database import get_database_manager
        from src.services.technical.hybrid_strategy_calculator import get_hybrid_strategy_calculator

        # Initialize database
        db_manager = get_database_manager()

        with db_manager.get_session() as session:
            # Test with a few popular stocks
            test_symbols = [
                'NSE:RELIANCE-EQ',
                'NSE:TCS-EQ',
                'NSE:INFY-EQ',
                'NSE:HDFCBANK-EQ',
                'NSE:ICICIBANK-EQ'
            ]

            print(f"\n📊 Testing hybrid strategy on {len(test_symbols)} stocks...")
            print("Symbols:", ', '.join(test_symbols))

            # Initialize hybrid calculator
            hybrid_calc = get_hybrid_strategy_calculator(session)

            # Calculate indicators
            print("\n🔧 Calculating hybrid indicators...")
            results = hybrid_calc.calculate_all_indicators(test_symbols, lookback_days=252)

            if not results:
                print("❌ No results calculated. Check if historical data exists.")
                return False

            print(f"\n✅ Successfully calculated indicators for {len(results)} stocks")

            # Display results
            print("\n" + "=" * 80)
            print("RESULTS")
            print("=" * 80)

            for symbol, indicators in results.items():
                print(f"\n📈 {symbol}")
                print("-" * 40)

                # Price & EMAs
                print(f"  Current Price: ₹{indicators.get('current_price', 0):.2f}")
                print(f"  EMA 8:         ₹{indicators.get('ema_8', 0):.2f}")
                print(f"  EMA 21:        ₹{indicators.get('ema_21', 0):.2f}")

                # Scores
                print(f"\n  📊 Scores:")
                print(f"    RS Rating:          {indicators.get('rs_rating', 50):.1f}/99")
                print(f"    EMA Trend Score:    {indicators.get('ema_trend_score', 50):.1f}/100")
                print(f"    Wave Momentum:      {indicators.get('wave_momentum_score', 50):.1f}/100")
                print(f"    HYBRID COMPOSITE:   {indicators.get('hybrid_composite_score', 50):.1f}/100")

                # Indicators
                print(f"\n  🔍 Indicators:")
                print(f"    DeMarker:           {indicators.get('demarker', 0.5):.3f}")
                print(f"    Delta:              {indicators.get('delta', 0):.6f}")

                # Fibonacci Targets
                print(f"\n  🎯 Fibonacci Targets:")
                print(f"    Target 1 (127.2%):  ₹{indicators.get('fib_target_1', 0):.2f}")
                print(f"    Target 2 (161.8%):  ₹{indicators.get('fib_target_2', 0):.2f}")
                print(f"    Target 3 (200%):    ₹{indicators.get('fib_target_3', 0):.2f}")

                # Signals
                buy = indicators.get('buy_signal', False)
                sell = indicators.get('sell_signal', False)
                quality = indicators.get('signal_quality', 'none')
                conditions = indicators.get('conditions_met', 0)

                print(f"\n  📡 Signals:")
                print(f"    Buy Signal:         {'🟢 YES' if buy else '⚪ NO'}")
                print(f"    Sell Signal:        {'🔴 YES' if sell else '⚪ NO'}")
                print(f"    Signal Quality:     {quality.upper()} ({conditions}/5 conditions)")

            # Summary statistics
            print("\n" + "=" * 80)
            print("SUMMARY STATISTICS")
            print("=" * 80)

            avg_rs = sum(r.get('rs_rating', 50) for r in results.values()) / len(results)
            avg_ema = sum(r.get('ema_trend_score', 50) for r in results.values()) / len(results)
            avg_wave = sum(r.get('wave_momentum_score', 50) for r in results.values()) / len(results)
            avg_composite = sum(r.get('hybrid_composite_score', 50) for r in results.values()) / len(results)

            buy_signals = sum(1 for r in results.values() if r.get('buy_signal'))
            sell_signals = sum(1 for r in results.values() if r.get('sell_signal'))
            high_quality = sum(1 for r in results.values() if r.get('signal_quality') == 'high')

            print(f"  Stocks analyzed:     {len(results)}")
            print(f"  Avg RS Rating:       {avg_rs:.1f}/99")
            print(f"  Avg EMA Score:       {avg_ema:.1f}/100")
            print(f"  Avg Wave Score:      {avg_wave:.1f}/100")
            print(f"  Avg Hybrid Score:    {avg_composite:.1f}/100")
            print(f"  Buy Signals:         {buy_signals}")
            print(f"  Sell Signals:        {sell_signals}")
            print(f"  High Quality Buys:   {high_quality}")

            print("\n✅ Test completed successfully!")
            return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    success = test_hybrid_strategy()
    sys.exit(0 if success else 1)
