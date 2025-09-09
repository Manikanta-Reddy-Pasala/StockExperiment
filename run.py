#!/usr/bin/env python3
"""
Application Runner Script
"""
import argparse
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import main as trading_system_main


def main():
    """Main entry point for the application runner."""
    parser = argparse.ArgumentParser(description='Automated Trading System Runner')
    parser.add_argument(
        '--mode', 
        choices=['development', 'production'], 
        default='development',
        help='Running mode (development or production)'
    )
    parser.add_argument(
        '--config', 
        default='unified',
        help='Configuration environment (unified is the only supported environment now)'
    )
    
    args = parser.parse_args()
    
    # Run the main trading system
    try:
        # Pass the arguments to the main function
        import src.main
        src.main.main()
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()