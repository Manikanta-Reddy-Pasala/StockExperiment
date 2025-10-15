#!/usr/bin/env python3
"""
Quick Test Script for Individual Model Testing
Quickly test any single model + strategy combination through saga.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.data.suggested_stocks_saga import get_suggested_stocks_saga_orchestrator


def quick_test(model_type: str = 'traditional', strategy: str = 'default_risk', limit: int = 5):
    """
    Quick test a single model + strategy combination.

    Args:
        model_type: 'traditional', 'raw_lstm', or 'kronos'
        strategy: 'default_risk' or 'high_risk'
        limit: Number of stocks to select (default: 5)
    """
    print(f"\n{'=' * 80}")
    print(f"QUICK TEST: {model_type.upper()} × {strategy.upper()}")
    print(f"{'=' * 80}\n")

    orchestrator = get_suggested_stocks_saga_orchestrator()

    try:
        result = orchestrator.execute_suggested_stocks_saga(
            user_id=1,
            strategies=[strategy],
            limit=limit,
            model_type=model_type
        )

        if result['status'] == 'completed':
            print(f"\n✅ SUCCESS!")
            print(f"   Stocks: {result['summary']['final_result_count']}")
            print(f"   Duration: {result['total_duration_seconds']:.1f}s")

            # Show top stocks
            if result['final_results']:
                print(f"\n   Top Stocks:")
                for idx, stock in enumerate(result['final_results'][:5], 1):
                    print(f"     {idx}. {stock['symbol']} - {stock['name']}")
                    print(f"        Price: ₹{stock['current_price']:.2f}, "
                          f"ML Score: {stock.get('ml_prediction_score', 0):.3f}")
        else:
            print(f"\n❌ FAILED: {result.get('errors', [])}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Quick test any model + strategy')
    parser.add_argument('model', type=str, choices=['traditional', 'raw_lstm', 'kronos'],
                       help='Model to test')
    parser.add_argument('strategy', type=str, choices=['default_risk', 'high_risk'],
                       help='Strategy to test')
    parser.add_argument('--limit', type=int, default=5,
                       help='Number of stocks (default: 5)')

    args = parser.parse_args()
    quick_test(args.model, args.strategy, args.limit)
