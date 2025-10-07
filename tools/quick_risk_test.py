#!/usr/bin/env python3
"""
Quick Risk Profile Test - Uses existing data without ML training
Tests DEFAULT_RISK vs HIGH_RISK on a small sample
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.data.suggested_stocks_saga import SuggestedStocksSagaOrchestrator

print("=" * 80)
print("QUICK RISK PROFILE TEST")
print("=" * 80)

orchestrator = SuggestedStocksSagaOrchestrator()

# Test DEFAULT_RISK (without ML - will skip ML step)
print("\n1. Testing DEFAULT_RISK strategy...")
default_result = orchestrator.execute_suggested_stocks_saga(
    user_id=1,
    strategies=['DEFAULT_RISK'],
    limit=5,
    search='',
    sort_by='score',
    sort_order='desc'
)

print(f"   Status: {default_result['status']}")
print(f"   Duration: {default_result['total_duration_seconds']:.1f}s")
print(f"   Results: {len(default_result.get('final_results', []))} stocks")

if default_result.get('final_results'):
    print("\n   DEFAULT_RISK Stocks:")
    for i, stock in enumerate(default_result['final_results'][:3], 1):
        print(f"     {i}. {stock['symbol']} - ₹{stock.get('current_price', 0):,.2f}")
        print(f"        Target: ₹{stock.get('target_price', 0):,.2f} ({stock.get('upside_percentage', 0):.1f}%)")
        print(f"        Score: {stock.get('score', 0):.2f}")

# Test HIGH_RISK
print("\n2. Testing HIGH_RISK strategy...")
high_result = orchestrator.execute_suggested_stocks_saga(
    user_id=1,
    strategies=['HIGH_RISK'],
    limit=5,
    search='',
    sort_by='score',
    sort_order='desc'
)

print(f"   Status: {high_result['status']}")
print(f"   Duration: {high_result['total_duration_seconds']:.1f}s")
print(f"   Results: {len(high_result.get('final_results', []))} stocks")

if high_result.get('final_results'):
    print("\n   HIGH_RISK Stocks:")
    for i, stock in enumerate(high_result['final_results'][:3], 1):
        print(f"     {i}. {stock['symbol']} - ₹{stock.get('current_price', 0):,.2f}")
        print(f"        Target: ₹{stock.get('target_price', 0):,.2f} ({stock.get('upside_percentage', 0):.1f}%)")
        print(f"        Score: {stock.get('score', 0):.2f}")

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

if default_result.get('final_results') and high_result.get('final_results'):
    default_symbols = {s['symbol'] for s in default_result['final_results']}
    high_symbols = {s['symbol'] for s in high_result['final_results']}

    overlap = default_symbols & high_symbols

    print(f"\nDefault stocks: {len(default_symbols)}")
    print(f"High risk stocks: {len(high_symbols)}")
    print(f"Common stocks: {len(overlap)}")

    if len(overlap) < len(default_symbols):
        print("\n✅ Strategies produce different results")
    else:
        print("\n⚠️ All stocks are the same")

print("\n" + "=" * 80)
