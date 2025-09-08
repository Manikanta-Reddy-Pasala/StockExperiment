"""
Backtesting Example with ASOF_DATE functionality
"""
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtesting.backtester import Backtester
from backtesting.strategies import simple_momentum_strategy, mean_reversion_strategy
from data_provider.data_manager import get_data_manager


def fetch_sample_data(symbols, period="6mo"):
    """
    Fetch sample historical data for backtesting.
    
    Args:
        symbols (List[str]): List of symbols to fetch data for
        period (str): Time period for data
        
    Returns:
        pd.DataFrame: Combined historical data
    """
    data_manager = get_data_manager()
    all_data = []
    
    for symbol in symbols:
        try:
            # Fetch historical data
            data = data_manager.get_historical_data(symbol, period=period, interval="1d")
            if not data.empty:
                # Add symbol column
                data['Symbol'] = symbol
                all_data.append(data)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data


def run_backtesting_example():
    """Run a complete backtesting example."""
    print("Running Backtesting Example")
    print("=" * 50)
    
    # Define symbols to test
    symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    
    # Fetch historical data
    print("Fetching historical data...")
    historical_data = fetch_sample_data(symbols, period="6mo")
    
    if historical_data.empty:
        print("No data available for backtesting")
        return
    
    print(f"Fetched data for {len(symbols)} symbols")
    print(f"Data shape: {historical_data.shape}")
    
    # Initialize backtester
    backtester = Backtester(initial_capital=100000.0)
    
    # Test 1: Simple momentum strategy
    print("\n1. Testing Simple Momentum Strategy")
    print("-" * 30)
    
    # Run backtest
    results1 = backtester.run_backtest(
        strategy_func=simple_momentum_strategy,
        historical_data=historical_data,
        symbols=symbols,
        lookback_period=20,
        max_positions=3
    )
    
    # Print results
    print(f"Initial Capital: ₹{results1['initial_capital']:,.2f}")
    print(f"Final Capital: ₹{results1['final_capital']:,.2f}")
    print(f"Total Return: {results1['total_return_percentage']:.2f}%")
    print(f"Sharpe Ratio: {results1['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {results1['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {results1['max_drawdown_percentage']:.2f}%")
    print(f"Win Rate: {results1['win_rate_percentage']:.2f}%")
    print(f"Total Trades: {results1['total_trades']}")
    
    # Test 2: Mean reversion strategy
    print("\n2. Testing Mean Reversion Strategy")
    print("-" * 30)
    
    # Reset backtester
    backtester = Backtester(initial_capital=100000.0)
    
    # Run backtest
    results2 = backtester.run_backtest(
        strategy_func=mean_reversion_strategy,
        historical_data=historical_data,
        symbols=symbols
    )
    
    # Print results
    print(f"Initial Capital: ₹{results2['initial_capital']:,.2f}")
    print(f"Final Capital: ₹{results2['final_capital']:,.2f}")
    print(f"Total Return: {results2['total_return_percentage']:.2f}%")
    print(f"Sharpe Ratio: {results2['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {results2['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {results2['max_drawdown_percentage']:.2f}%")
    print(f"Win Rate: {results2['win_rate_percentage']:.2f}%")
    print(f"Total Trades: {results2['total_trades']}")
    
    # Test 3: ASOF_DATE functionality
    print("\n3. Testing ASOF_DATE Functionality")
    print("-" * 30)
    
    # Set ASOF date to 3 months ago
    asof_date = datetime.now() - timedelta(days=90)
    backtester.set_asof_date(asof_date)
    
    print(f"ASOF Date: {asof_date.strftime('%Y-%m-%d')}")
    
    # Run backtest with ASOF date
    results3 = backtester.run_backtest(
        strategy_func=simple_momentum_strategy,
        historical_data=historical_data,
        symbols=symbols,
        lookback_period=20,
        max_positions=3
    )
    
    # Print results
    print(f"Initial Capital: ₹{results3['initial_capital']:,.2f}")
    print(f"Final Capital: ₹{results3['final_capital']:,.2f}")
    print(f"Total Return: {results3['total_return_percentage']:.2f}%")
    print(f"Sharpe Ratio: {results3['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {results3['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {results3['max_drawdown_percentage']:.2f}%")
    print(f"Win Rate: {results3['win_rate_percentage']:.2f}%")
    print(f"Total Trades: {results3['total_trades']}")
    
    print("\nBacktesting Example Completed!")


if __name__ == "__main__":
    run_backtesting_example()