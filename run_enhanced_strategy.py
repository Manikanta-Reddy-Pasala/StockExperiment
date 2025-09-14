#!/usr/bin/env python
"""
Main Integration Script for Enhanced Portfolio Strategy
========================================================
This script integrates the 4-step portfolio strategy with:
- FYERS broker APIs for live trading
- Existing ML prediction service
- Real-time monitoring and execution
"""

import sys
import os
import logging
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.enhanced_portfolio_strategy import (
    EnhancedPortfolioStrategy,
    RiskStrategy,
    FilteringCriteria,
    EntryRules,
    ExitRules
)
from src.services.fyers_api_service import FyersAPIService
from src.services.ml.prediction_service import get_prediction
from src.config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PortfolioManager:
    """Main portfolio manager that orchestrates the strategy execution."""
    
    def __init__(self, capital: float = 100000.0):
        """Initialize the portfolio manager."""
        self.capital = capital
        self.strategy = EnhancedPortfolioStrategy(capital=capital)
        self.fyers_api = FyersAPIService()
        
        logger.info(f"Portfolio Manager initialized with capital: ₹{capital:,.2f}")
    
    def get_stock_universe(self) -> list:
        """
        Get the stock universe for analysis.
        This can be customized based on your requirements.
        """
        # Option 1: Predefined watchlist
        watchlist = [
            "NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:INFY-EQ",
            "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:WIPRO-EQ",
            "NSE:BHARTIARTL-EQ", "NSE:SBIN-EQ", "NSE:HDFC-EQ",
            "NSE:ASIANPAINT-EQ", "NSE:ITC-EQ", "NSE:AXISBANK-EQ",
            "NSE:LT-EQ", "NSE:DMART-EQ", "NSE:SUNPHARMA-EQ",
            "NSE:ULTRACEMCO-EQ", "NSE:TITAN-EQ", "NSE:BAJFINANCE-EQ",
            "NSE:MARUTI-EQ", "NSE:TATAMOTORS-EQ", "NSE:NESTLEIND-EQ",
            "NSE:KOTAKBANK-EQ", "NSE:HINDUNILVR-EQ", "NSE:DIVISLAB-EQ",
            "NSE:JSWSTEEL-EQ", "NSE:TECHM-EQ", "NSE:BAJAJFINSV-EQ",
            "NSE:ADANIENT-EQ", "NSE:TATASTEEL-EQ", "NSE:INDUSINDBK-EQ"
        ]
        
        # Option 2: Get from NIFTY indices
        # You can fetch from NIFTY 50, NIFTY MIDCAP 100, etc.
        # watchlist = self.fyers_api.get_index_constituents("NSE:NIFTY50")
        
        return watchlist
    
    def configure_strategy(self, config: dict = None):
        """
        Configure strategy parameters.
        
        Args:
            config: Optional configuration dictionary
        """
        if config:
            # Configure filtering criteria
            if 'filtering' in config:
                self.strategy.filtering_criteria.min_price = config['filtering'].get('min_price', 50.0)
                self.strategy.filtering_criteria.min_avg_volume_20d = config['filtering'].get('min_volume', 500000)
                self.strategy.filtering_criteria.max_atr_percent = config['filtering'].get('max_atr_percent', 10.0)
            
            # Configure entry rules
            if 'entry' in config:
                self.strategy.entry_rules.rsi_min = config['entry'].get('rsi_min', 50.0)
                self.strategy.entry_rules.rsi_max = config['entry'].get('rsi_max', 70.0)
                self.strategy.entry_rules.volume_multiplier = config['entry'].get('volume_multiplier', 1.5)
            
            # Configure exit rules
            if 'exit' in config:
                self.strategy.exit_rules.profit_target_1_percent = config['exit'].get('target1', 5.0)
                self.strategy.exit_rules.profit_target_2_percent = config['exit'].get('target2', 10.0)
                self.strategy.exit_rules.stop_loss_percent = config['exit'].get('stop_loss', 3.0)
                self.strategy.exit_rules.max_holding_days = config['exit'].get('max_days', 10)
                self.strategy.exit_rules.trailing_stop_enabled = config['exit'].get('trailing_stop', True)
        
        logger.info("Strategy configuration updated")
    
    def run_backtest(self, start_date: str, end_date: str):
        """
        Run backtest on historical data.
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        # Implement backtesting logic here using historical data
        # This would use the ML service's backtest_service
        pass
    
    def run_live_trading(self, risk_preference: str = "SAFE"):
        """
        Run live trading with real-time monitoring.
        
        Args:
            risk_preference: "SAFE" or "HIGH_RISK"
        """
        try:
            # Get stock universe
            stock_universe = self.get_stock_universe()
            logger.info(f"Analyzing {len(stock_universe)} stocks...")
            
            # Determine risk strategy
            risk_strategy = RiskStrategy.SAFE if risk_preference == "SAFE" else RiskStrategy.HIGH_RISK
            
            # Run the 4-step strategy
            results = self.strategy.run_strategy(stock_universe, risk_strategy)
            
            # Save results
            self.save_results(results)
            
            # Display results
            self.display_results(results)
            
            # Start continuous monitoring
            if self.strategy.positions:
                logger.info("Starting continuous monitoring of positions...")
                self.strategy.run_continuous_monitoring(interval_seconds=60)
            else:
                logger.info("No positions to monitor. Strategy execution complete.")
                
        except Exception as e:
            logger.error(f"Error in live trading: {e}")
            raise
    
    def save_results(self, results: dict):
        """Save strategy results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
    
    def display_results(self, results: dict):
        """Display strategy execution results."""
        print("\n" + "="*60)
        print("STRATEGY EXECUTION RESULTS")
        print("="*60)
        
        for step_name, step_data in results.get('steps', {}).items():
            print(f"\n{step_name.upper()}:")
            for key, value in step_data.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        
        if 'performance' in results:
            print("\nPERFORMANCE METRICS:")
            for key, value in results['performance'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        print("="*60 + "\n")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Portfolio Strategy with FYERS Integration')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital for trading (default: 100000)')
    parser.add_argument('--risk', choices=['SAFE', 'HIGH_RISK'], default='SAFE',
                       help='Risk preference (default: SAFE)')
    parser.add_argument('--mode', choices=['live', 'backtest', 'paper'], default='paper',
                       help='Execution mode (default: paper)')
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON)')
    
    args = parser.parse_args()
    
    # Initialize portfolio manager
    manager = PortfolioManager(capital=args.capital)
    
    # Load configuration if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            manager.configure_strategy(config)
    
    # Execute based on mode
    if args.mode == 'live':
        print("\n" + "="*60)
        print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK")
        print("="*60)
        confirmation = input("Are you sure you want to proceed? (yes/no): ")
        if confirmation.lower() == 'yes':
            manager.run_live_trading(args.risk)
        else:
            print("Live trading cancelled.")
    
    elif args.mode == 'backtest':
        print("\n" + "="*60)
        print("📊 BACKTEST MODE")
        print("="*60)
        # Implement backtest with dates
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")
        manager.run_backtest(start_date, end_date)
    
    else:  # paper trading
        print("\n" + "="*60)
        print("📝 PAPER TRADING MODE (Simulation)")
        print("="*60)
        manager.run_live_trading(args.risk)
    
    print("\nExecution complete!")


if __name__ == "__main__":
    main()