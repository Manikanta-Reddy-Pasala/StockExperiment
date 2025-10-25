#!/usr/bin/env python3
"""
Backtest Runner
Run backtests on the technical indicator strategy to validate performance
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.database import get_database_manager
from src.services.backtesting.backtest_service import create_backtest_service


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Backtest trading strategy on historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest last 6 months with 5-day hold
  python tools/run_backtest.py --months 6 --hold-days 5

  # Backtest specific date range
  python tools/run_backtest.py --start-date 2023-01-01 --end-date 2023-12-31

  # Backtest HIGH_RISK strategy with stop loss
  python tools/run_backtest.py --strategy HIGH_RISK --stop-loss 10 --target 15

  # Quick backtest (last month)
  python tools/run_backtest.py --months 1 --hold-days 3
        """
    )

    # Date range
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD). Default: 6 months ago'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD). Default: yesterday'
    )
    parser.add_argument(
        '--months',
        type=int,
        help='Backtest period in months (alternative to start-date)'
    )

    # Strategy parameters
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['DEFAULT_RISK', 'HIGH_RISK'],
        default='DEFAULT_RISK',
        help='Strategy to test (default: DEFAULT_RISK)'
    )
    parser.add_argument(
        '--hold-days',
        type=int,
        default=5,
        help='Number of days to hold each position (default: 5)'
    )

    # Portfolio parameters
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial capital in rupees (default: 100000)'
    )
    parser.add_argument(
        '--position-size',
        type=float,
        default=20.0,
        help='Position size as %% of capital (default: 20)'
    )
    parser.add_argument(
        '--max-positions',
        type=int,
        default=5,
        help='Maximum concurrent positions (default: 5)'
    )

    # Risk management
    parser.add_argument(
        '--stop-loss',
        type=float,
        help='Stop loss percentage (e.g., 5 for 5%%)'
    )
    parser.add_argument(
        '--target',
        type=float,
        help='Target profit percentage (e.g., 10 for 10%%)'
    )

    return parser.parse_args()


def main():
    """Run backtest with specified parameters."""
    args = parse_args()

    # Determine date range
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now() - timedelta(days=1)  # Yesterday

    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    elif args.months:
        start_date = end_date - timedelta(days=30 * args.months)
    else:
        start_date = end_date - timedelta(days=180)  # Default 6 months

    print("\n" + "🎯" * 40)
    print("BACKTESTING TECHNICAL INDICATOR STRATEGY")
    print("🎯" * 40)

    print(f"\n📅 Backtest Period:")
    print(f"   From: {start_date.date()}")
    print(f"   To:   {end_date.date()}")
    print(f"   Days: {(end_date - start_date).days}")

    print(f"\n⚙️  Configuration:")
    print(f"   Strategy:       {args.strategy}")
    print(f"   Hold Days:      {args.hold_days}")
    print(f"   Initial Capital: ₹{args.capital:,.0f}")
    print(f"   Position Size:  {args.position_size:.1f}%")
    print(f"   Max Positions:  {args.max_positions}")
    if args.stop_loss:
        print(f"   Stop Loss:      {args.stop_loss:.1f}%")
    if args.target:
        print(f"   Target:         {args.target:.1f}%")

    print(f"\n🔄 Initializing backtest...")

    # Get database connection
    db_manager = get_database_manager()

    with db_manager.get_session() as session:
        # Create backtest service
        backtest_service = create_backtest_service(session)

        # Run backtest
        print(f"⏳ Running backtest (this may take a few minutes)...\n")

        backtest_service.run_backtest_and_print_report(
            start_date=start_date,
            end_date=end_date,
            strategy=args.strategy,
            hold_days=args.hold_days,
            initial_capital=args.capital,
            position_size_pct=args.position_size,
            max_positions=args.max_positions,
            stop_loss_pct=args.stop_loss,
            target_pct=args.target
        )

    print("\n✅ Backtest complete!\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted by user\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running backtest: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
