"""
Database Charting Example
"""
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from charting.db_charts import DatabaseCharts
from datastore.database import get_database_manager


def run_charting_example():
    """Run a database charting example."""
    print("Running Database Charting Example")
    print("=" * 50)
    
    # Initialize database and charting system
    db_manager = get_database_manager()
    db_manager.create_tables()  # Ensure tables exist
    charts = DatabaseCharts(db_manager)
    
    # Example 1: Get portfolio value chart data
    print("1. Portfolio Value Chart Data")
    print("-" * 30)
    portfolio_data = charts.get_portfolio_value_chart_data(days=30)
    print(f"Retrieved {len(portfolio_data)} data points")
    if portfolio_data:
        print("Sample data:")
        for i, data in enumerate(portfolio_data[:5]):
            print(f"  {data['date']}: ₹{data['value']:,.2f} (P&L: ₹{data['pnl']:,.2f})")
    
    # Example 2: Get P&L chart data
    print("\n2. P&L Chart Data")
    print("-" * 30)
    pnl_data = charts.get_pnl_chart_data(days=30)
    print(f"Retrieved {len(pnl_data)} data points")
    if pnl_data:
        print("Sample data:")
        for i, data in enumerate(pnl_data[:5]):
            print(f"  {data['date']}: Daily P&L ₹{data['daily_pnl']:,.2f}, Cumulative P&L ₹{data['cumulative_pnl']:,.2f}")
    
    # Example 3: Get trades chart data
    print("\n3. Trades Chart Data")
    print("-" * 30)
    trades_data = charts.get_trades_chart_data(days=30)
    print(f"Retrieved {len(trades_data)} trades")
    if trades_data:
        print("Sample data:")
        for i, data in enumerate(trades_data[:3]):
            print(f"  {data['date']}: {data['transaction_type']} {data['quantity']} {data['symbol']} @ ₹{data['price']:.2f}")
    
    # Example 4: Get positions chart data
    print("\n4. Positions Chart Data")
    print("-" * 30)
    positions_data = charts.get_positions_chart_data()
    print(f"Retrieved {len(positions_data)} positions")
    if positions_data:
        print("Sample data:")
        for i, data in enumerate(positions_data[:3]):
            print(f"  {data['symbol']}: {data['quantity']} shares @ ₹{data['average_price']:.2f}, Value: ₹{data['market_value']:,.2f}")
    
    # Example 5: Get orders chart data
    print("\n5. Orders Chart Data")
    print("-" * 30)
    orders_data = charts.get_orders_chart_data()
    print(f"Retrieved {len(orders_data)} orders")
    if orders_data:
        print("Sample data:")
        for i, data in enumerate(orders_data[:3]):
            print(f"  {data['created_at']}: {data['transaction_type']} {data['quantity']} {data['symbol']} ({data['status']})")
    
    # Example 6: Get market data chart
    print("\n6. Market Data Chart")
    print("-" * 30)
    market_data = charts.get_market_data_chart("INFY", days=30)
    print(f"Retrieved {len(market_data)} market data points")
    if market_data:
        print("Sample data:")
        for i, data in enumerate(market_data[:3]):
            print(f"  {data['timestamp']}: O={data['open']:.2f}, H={data['high']:.2f}, L={data['low']:.2f}, C={data['close']:.2f}")
    
    # Example 7: Get performance summary
    print("\n7. Performance Summary")
    print("-" * 30)
    performance = charts.get_performance_summary()
    print("Performance Summary:")
    print(f"  Total Trades: {performance.get('total_trades', 0)}")
    print(f"  Total Orders: {performance.get('total_orders', 0)}")
    print(f"  Open Positions: {performance.get('open_positions', 0)}")
    print(f"  Total P&L: ₹{performance.get('total_pnl', 0):,.2f}")
    print(f"  Win Rate: {performance.get('win_rate', 0):.1f}%")
    print(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {performance.get('max_drawdown', 0):.1f}%")
    
    print("\nDatabase Charting Example Completed!")


if __name__ == "__main__":
    run_charting_example()