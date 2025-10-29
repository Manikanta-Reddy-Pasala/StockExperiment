#!/usr/bin/env python3
"""
6-Month Backtesting Script
Runs comprehensive backtesting for the technical indicator strategy

Usage:
    python3 run_6month_backtest.py                    # Run both strategies
    python3 run_6month_backtest.py --strategy DEFAULT_RISK  # Run DEFAULT_RISK only
    python3 run_6month_backtest.py --strategy HIGH_RISK     # Run HIGH_RISK only
    python3 run_6month_backtest.py --docker            # Run inside Docker container
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# Color codes
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color


def print_header():
    """Print script header."""
    print(f"{Colors.BLUE}╔════════════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.BLUE}║          6-MONTH BACKTESTING - TECHNICAL INDICATORS            ║{Colors.NC}")
    print(f"{Colors.BLUE}╚════════════════════════════════════════════════════════════════╝{Colors.NC}")
    print()


def print_config():
    """Print configuration."""
    print(f"{Colors.YELLOW}Configuration:{Colors.NC}")
    print(f"  Period:         6 months")
    print(f"  Hold Days:      5 days")
    print(f"  Initial Capital: ₹100,000")
    print(f"  Position Size:  20%")
    print(f"  Max Positions:  5")
    print()


def print_separator():
    """Print separator line."""
    print(f"{Colors.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.NC}")


def run_backtest(strategy: str, stop_loss: float, target: float, docker: bool = False):
    """
    Run backtest for a specific strategy.

    Args:
        strategy: Strategy name ('DEFAULT_RISK' or 'HIGH_RISK')
        stop_loss: Stop loss percentage
        target: Target profit percentage
        docker: Whether to run inside Docker container
    """
    print_separator()
    print(f"{Colors.GREEN}  Running: {strategy} Strategy{Colors.NC}")
    print_separator()
    print()

    # Build command
    if docker:
        cmd = [
            'docker', 'exec', 'trading_system_app',
            'python3', 'tools/run_backtest.py'
        ]
    else:
        cmd = ['python3', 'tools/run_backtest.py']

    # Add arguments
    cmd.extend([
        '--months', '6',
        '--strategy', strategy,
        '--hold-days', '5',
        '--capital', '100000',
        '--position-size', '20',
        '--max-positions', '5',
        '--stop-loss', str(stop_loss),
        '--target', str(target)
    ])

    # Run backtest
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error running backtest for {strategy}: {e}{Colors.NC}")
        return False

    print()
    print_separator()
    print()

    return True


def print_footer():
    """Print footer."""
    print()
    print(f"{Colors.BLUE}╔════════════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.BLUE}║                    BACKTESTING COMPLETE                        ║{Colors.NC}")
    print(f"{Colors.BLUE}╚════════════════════════════════════════════════════════════════╝{Colors.NC}")
    print()
    print(f"{Colors.YELLOW}💡 Tips:{Colors.NC}")
    print(f"  - Compare win rates and Sharpe ratios between strategies")
    print(f"  - Check maximum drawdown to understand worst-case scenarios")
    print(f"  - Review trade distribution across different market conditions")
    print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run 6-month backtesting for technical indicator strategies'
    )
    parser.add_argument(
        '--strategy',
        choices=['DEFAULT_RISK', 'HIGH_RISK', 'both'],
        default='both',
        help='Strategy to test (default: both)'
    )
    parser.add_argument(
        '--docker',
        action='store_true',
        help='Run inside Docker container'
    )

    args = parser.parse_args()

    # Print header
    print_header()

    # Print mode
    if args.docker:
        print(f"{Colors.YELLOW}Running backtests inside Docker container...{Colors.NC}")
        print()

    # Print config
    if args.strategy == 'both':
        print_config()

    # Run backtests
    success = True

    if args.strategy in ['DEFAULT_RISK', 'both']:
        # DEFAULT_RISK: Conservative strategy
        # Stop loss: 5%, Target: 7%
        if not run_backtest('DEFAULT_RISK', 5.0, 7.0, docker=args.docker):
            success = False

    if args.strategy in ['HIGH_RISK', 'both']:
        # HIGH_RISK: Aggressive strategy
        # Stop loss: 10%, Target: 12%
        if not run_backtest('HIGH_RISK', 10.0, 12.0, docker=args.docker):
            success = False

    # Print footer
    print_footer()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.RED}⚠️  Backtest interrupted by user{Colors.NC}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.NC}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
