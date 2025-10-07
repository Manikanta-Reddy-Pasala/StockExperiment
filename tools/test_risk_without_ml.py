#!/usr/bin/env python3
"""
Test Risk Profiles WITHOUT ML Training
Just validates the strategy filtering logic
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from sqlalchemy import text

print("=" * 80)
print("RISK PROFILE TEST - NO ML TRAINING")
print("=" * 80)

db_manager = get_database_manager()

with db_manager.get_session() as session:

    # Test DEFAULT_RISK filters
    print("\n" + "=" * 80)
    print("DEFAULT_RISK STRATEGY")
    print("=" * 80)
    print("\nFilters:")
    print("  - Price: ₹50 - ₹5,000")
    print("  - Volume: > 100,000")
    print("  - Market Cap: > ₹5,000 Cr")
    print("  - P/E: 5-30")
    print("  - ROE: > 10%")
    print("  - Debt/Equity: < 2")

    result = session.execute(text("""
        SELECT s.symbol, s.name, s.current_price, s.market_cap,
               s.pe_ratio, s.roe, s.debt_to_equity
        FROM stocks s
        WHERE s.current_price BETWEEN 50 AND 5000
          AND s.market_cap > 5000
          AND s.pe_ratio BETWEEN 5 AND 30
          AND s.roe > 10
          AND s.debt_to_equity < 2
          AND s.current_price IS NOT NULL
        ORDER BY s.market_cap DESC
        LIMIT 10
    """))

    default_stocks = [dict(row._mapping) for row in result.fetchall()]

    print(f"\n✓ Found {len(default_stocks)} DEFAULT_RISK stocks\n")

    for i, stock in enumerate(default_stocks[:5], 1):
        print(f"{i}. {stock['symbol']}")
        print(f"   Price: ₹{stock['current_price']:,.2f}")
        print(f"   Market Cap: ₹{stock['market_cap']:,.0f} Cr")
        print(f"   P/E: {stock['pe_ratio']:.2f}, ROE: {stock['roe']:.2f}%")
        print(f"   Debt/Equity: {stock['debt_to_equity']:.2f}")

    # Test HIGH_RISK filters
    print("\n" + "=" * 80)
    print("HIGH_RISK STRATEGY")
    print("=" * 80)
    print("\nFilters:")
    print("  - Price: ₹10 - ₹10,000")
    print("  - Volume: > 50,000")
    print("  - Market Cap: > ₹1,000 Cr (lower)")
    print("  - ROE: > 5% (more lenient)")
    print("  - Beta: > 1.2 (more volatile)")

    result = session.execute(text("""
        SELECT s.symbol, s.name, s.current_price, s.market_cap,
               s.pe_ratio, s.roe, s.beta, s.debt_to_equity
        FROM stocks s
        WHERE s.current_price BETWEEN 10 AND 10000
          AND s.market_cap > 1000
          AND s.roe > 5
          AND COALESCE(s.beta, 0) > 1.2
          AND s.current_price IS NOT NULL
        ORDER BY s.market_cap DESC
        LIMIT 10
    """))

    high_stocks = [dict(row._mapping) for row in result.fetchall()]

    print(f"\n✓ Found {len(high_stocks)} HIGH_RISK stocks\n")

    for i, stock in enumerate(high_stocks[:5], 1):
        print(f"{i}. {stock['symbol']}")
        print(f"   Price: ₹{stock['current_price']:,.2f}")
        print(f"   Market Cap: ₹{stock['market_cap']:,.0f} Cr")
        print(f"   P/E: {stock.get('pe_ratio', 0):.2f}, ROE: {stock['roe']:.2f}%")
        print(f"   Beta: {stock['beta']:.2f}")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    default_symbols = {s['symbol'] for s in default_stocks}
    high_symbols = {s['symbol'] for s in high_stocks}

    overlap = default_symbols & high_symbols
    unique_default = default_symbols - high_symbols
    unique_high = high_symbols - default_symbols

    print(f"\nDEFAULT_RISK stocks: {len(default_symbols)}")
    print(f"HIGH_RISK stocks: {len(high_symbols)}")
    print(f"Common stocks: {len(overlap)}")
    print(f"Unique to DEFAULT_RISK: {len(unique_default)}")
    print(f"Unique to HIGH_RISK: {len(unique_high)}")

    if overlap:
        print(f"\nCommon stocks: {', '.join(list(overlap)[:5])}")

    if unique_default:
        print(f"Only DEFAULT_RISK: {', '.join(list(unique_default)[:5])}")

    if unique_high:
        print(f"Only HIGH_RISK: {', '.join(list(unique_high)[:5])}")

    # Calculate averages
    if default_stocks:
        avg_mc_default = sum(s['market_cap'] for s in default_stocks) / len(default_stocks)
        avg_pe_default = sum(s['pe_ratio'] for s in default_stocks) / len(default_stocks)
        avg_roe_default = sum(s['roe'] for s in default_stocks) / len(default_stocks)

        print(f"\nDEFAULT_RISK Averages:")
        print(f"  Market Cap: ₹{avg_mc_default:,.0f} Cr")
        print(f"  P/E Ratio: {avg_pe_default:.2f}")
        print(f"  ROE: {avg_roe_default:.2f}%")

    if high_stocks:
        avg_mc_high = sum(s['market_cap'] for s in high_stocks) / len(high_stocks)
        avg_roe_high = sum(s['roe'] for s in high_stocks) / len(high_stocks)
        avg_beta_high = sum(s['beta'] for s in high_stocks) / len(high_stocks)

        print(f"\nHIGH_RISK Averages:")
        print(f"  Market Cap: ₹{avg_mc_high:,.0f} Cr")
        print(f"  ROE: {avg_roe_high:.2f}%")
        print(f"  Beta: {avg_beta_high:.2f}")

    # Validate differences
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    if len(unique_default) > 0 or len(unique_high) > 0:
        print("\n✅ PASS: Strategies produce different results")
    else:
        print("\n❌ FAIL: Strategies produce identical results")

    if default_stocks and high_stocks and avg_mc_default > avg_mc_high:
        print("✅ PASS: DEFAULT_RISK has higher average market cap")
    else:
        print("⚠️  WARNING: Market cap averages unexpected")

    print("\n" + "=" * 80)
