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
        '--multi-user',
        action='store_true',
        default=True,
        help='Enable multi-user mode (default: True)'
    )
    parser.add_argument(
        '--single-user',
        action='store_true',
        help='Disable multi-user mode (use single-user mode)'
    )
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Enable development mode with auto-reloading'
    )
    
    args = parser.parse_args()
    
    # Run the FastAPI application
    try:
        import uvicorn
        
        if args.dev:
            # Use import string for reload to work
            uvicorn.run(
                "src.web.app:app",
                host="0.0.0.0",
                port=5001,
                reload=True,
                log_level="info"
            )
        else:
            # Import app directly for production
            from src.web.app import app
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=5001,
                log_level="info"
            )
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()