#!/usr/bin/env python3
"""
Manual trigger for technical indicators and stock picks calculation.
Run this to update stale data without waiting for scheduler.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import scheduler functions
from scheduler import calculate_technical_indicators, update_daily_snapshot

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 MANUAL CALCULATION TRIGGER")
    print("=" * 80)

    print("\n📊 Step 1: Calculating technical indicators...")
    calculate_technical_indicators()

    print("\n📈 Step 2: Generating daily stock picks...")
    update_daily_snapshot()

    print("\n" + "=" * 80)
    print("✅ MANUAL CALCULATION COMPLETE")
    print("=" * 80)
