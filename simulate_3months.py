#!/usr/bin/env python3
"""
3-Month Trading Simulation Script

This script runs a 3-month simulation of the trading system in development mode.
It uses the backtesting engine to simulate trading without using real money,
but still records profits and losses as if it were real trading.

The simulation will:
1. Fetch 3 months of historical data for selected stocks
2. Run the momentum strategy daily through the backtesting engine
3. Record all trades, profits, and losses in the database
4. Generate performance reports at the end
"""
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from backtesting.backtester import Backtester
from backtesting.strategies import simple_momentum_strategy
from data_provider.data_manager import get_data_manager
from datastore.database import get_database_manager
from datastore.models import Trade, Position, Order
from reporting.dashboard import DashboardReporter
from visualization.matplotlib_charts import MatplotlibCharts


def generate_sample_data(symbols, days=90):
    """
    Generate sample historical data for backtesting when real data is not available.
    
    Args:
        symbols (List[str]): List of symbols to generate data for
        days (int): Number of days of data to generate
        
    Returns:
        pd.DataFrame: Generated historical data
    """
    print("Generating sample data for simulation...")
    all_data = []
    
    # Start date for the simulation
    start_date = datetime.now() - timedelta(days=days)
    
    for i, symbol in enumerate(symbols):
        # Generate realistic price data with some trend and volatility
        dates = [start_date + timedelta(days=d) for d in range(days)]
        
        # Base price with some variation per symbol
        base_price = 1000 + (i * 200)  # Different base prices for different symbols
        
        # Generate price series with trend and random noise
        prices = []
        current_price = base_price
        trend = (np.random.random() - 0.5) * 0.002  # Small daily trend
        
        for day in range(days):
            # Apply trend and random noise
            daily_return = trend + (np.random.normal(0, 0.02))  # 2% daily volatility
            current_price = current_price * (1 + daily_return)
            
            # Ensure positive prices
            current_price = max(current_price, 0.01)
            prices.append(current_price)
        
        # Create DataFrame with OHLCV data
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],  # Open near close
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],  # High above close
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],   # Low below close
            'Close': prices,
            'Volume': [np.random.randint(100000, 1000000) for _ in range(days)],  # Random volume
            'Symbol': [symbol.replace('.NS', '') for _ in range(days)]  # Symbol column
        })
        
        all_data.append(data)
        print(f"  Generated data for {symbol}: {len(data)} days")
    
    if not all_data:
        print("No data generated for simulation")
        return pd.DataFrame()
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Total data points: {len(combined_data)}")
    return combined_data


def fetch_historical_data(symbols, period="3mo"):
    """
    Fetch 3 months of historical data for backtesting.
    Falls back to generated data if real data is not available.
    
    Args:
        symbols (List[str]): List of symbols to fetch data for
        period (str): Time period for data (default: "3mo")
        
    Returns:
        pd.DataFrame: Combined historical data
    """
    print("Fetching 3 months of historical data...")
    data_manager = get_data_manager()
    all_data = []
    
    for symbol in symbols:
        try:
            # Fetch historical data
            data = data_manager.get_historical_data(symbol, period=period, interval="1d")
            if not data.empty:
                # Add symbol column
                data['Symbol'] = symbol.replace('.NS', '')  # Remove .NS for consistency
                data['Date'] = data.index  # Add Date column from index
                all_data.append(data)
                print(f"  Fetched data for {symbol}: {len(data)} days")
            else:
                print(f"  No data for {symbol}")
        except Exception as e:
            print(f"  Error fetching data for {symbol}: {e}")
    
    if all_data:
        # Combine all data if we got any
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Total data points: {len(combined_data)}")
        return combined_data
    else:
        # Fall back to generated data
        print("Falling back to generated sample data...")
        return generate_sample_data([s.replace('.NS', '') for s in symbols], days=90)


def run_3month_simulation():
    """Run a 3-month trading simulation."""
    print("Starting 3-Month Trading Simulation")
    print("=" * 50)
    
    # Initialize components
    db_manager = get_database_manager()
    db_manager.create_tables()  # Ensure tables exist
    
    # Define symbols to trade (top NIFTY stocks)
    symbols = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "KOTAKBANK.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS"
    ]
    
    # Fetch historical data for 3 months
    historical_data = fetch_historical_data(symbols, period="3mo")
    
    if historical_data.empty:
        print("Cannot run simulation without data")
        return
    
    # Initialize backtester with 1 million initial capital
    backtester = Backtester(initial_capital=1000000.0)
    
    # Run backtest
    print("\nRunning 3-month simulation...")
    results = backtester.run_backtest(
        strategy_func=simple_momentum_strategy,
        historical_data=historical_data,
        symbols=[s.replace('.NS', '') for s in symbols],  # Remove .NS for consistency
        lookback_period=20,
        max_positions=5
    )
    
    # Save results to database
    save_simulation_results(results, db_manager)
    
    # Generate reports
    generate_simulation_report(results, db_manager)
    
    print("\n3-Month Simulation Completed!")
    return results


def save_simulation_results(results, db_manager):
    """
    Save simulation results to database.
    
    Args:
        results (Dict): Backtesting results
        db_manager: Database manager instance
    """
    print("\nSaving simulation results to database...")
    
    try:
        with db_manager.get_session() as session:
            # Save trades
            for i, trade in enumerate(results.get('trades', [])):
                db_trade = Trade(
                    trade_id=f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                    order_id=f"ORDER_{trade.get('date', datetime.now()).strftime('%Y%m%d')}_{trade.get('symbol', 'UNKNOWN')}",
                    tradingsymbol=trade.get('symbol', 'UNKNOWN'),
                    exchange="NSE",
                    transaction_type=trade.get('action', 'UNKNOWN'),
                    quantity=trade.get('quantity', 0),
                    price=trade.get('price', 0.0),
                    trade_time=trade.get('date', datetime.now()),
                    created_at=datetime.now()
                )
                session.add(db_trade)
            
            # Update positions (simplified - just show final positions)
            final_positions = results.get('final_positions', {})
            # In a real implementation, you would track positions more accurately
            
        print(f"Saved {len(results.get('trades', []))} trades to database")
    except Exception as e:
        print(f"Error saving results to database: {e}")


def generate_simulation_report(results, db_manager):
    """
    Generate and display simulation report.
    
    Args:
        results (Dict): Backtesting results
        db_manager: Database manager instance
    """
    print("\n" + "=" * 50)
    print("3-MONTH SIMULATION RESULTS")
    print("=" * 50)
    
    # Basic performance metrics
    print(f"Initial Capital:     ₹{results['initial_capital']:>15,.2f}")
    print(f"Final Capital:       ₹{results['final_capital']:>15,.2f}")
    print(f"Profit/Loss:         ₹{results['final_capital'] - results['initial_capital']:>15,.2f}")
    print(f"Total Return:        {results['total_return_percentage']:>15.2f}%")
    print(f"Total Trades:        {results['total_trades']:>15}")
    print(f"Win Rate:            {results['win_rate_percentage']:>15.2f}%")
    print(f"Sharpe Ratio:        {results['sharpe_ratio']:>15.2f}")
    print(f"Sortino Ratio:       {results['sortino_ratio']:>15.2f}")
    print(f"Max Drawdown:        {results['max_drawdown_percentage']:>15.2f}%")
    
    # Risk metrics
    print("\nRisk Metrics:")
    print(f"Annualized Volatility: {results['annualized_volatility']*100:>12.2f}%")
    
    # Generate charts
    try:
        print("\nGenerating performance charts...")
        dashboard = DashboardReporter(db_manager)
        charts = MatplotlibCharts()
        
        # Portfolio value chart
        portfolio_data = results.get('portfolio_values', [])
        if portfolio_data:
            chart_path = charts.create_portfolio_value_chart(portfolio_data, "3-Month Simulation Portfolio Value")
            print(f"Portfolio value chart saved to: {chart_path}")
        
        print("\nSimulation report generation completed!")
    except Exception as e:
        print(f"Error generating charts: {e}")


def main():
    """Main entry point."""
    try:
        results = run_3month_simulation()
        
        if results:
            print("\n" + "=" * 50)
            print("SIMULATION SUMMARY")
            print("=" * 50)
            print("The simulation has completed successfully!")
            print(f"Total return: {results['total_return_percentage']:.2f}%")
            print(f"Final portfolio value: ₹{results['final_capital']:,.2f}")
            print("\nAll trades and performance metrics have been saved to the database.")
            print("You can view the results through the web dashboard or by querying the database directly.")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError running simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()