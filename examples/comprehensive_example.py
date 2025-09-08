"""
Comprehensive Example - All Features Working Together
"""
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_provider.data_manager import get_data_manager
from selector.selector_engine import SelectorEngine
from risk.risk_manager import RiskManager
from order.order_router import OrderRouter, Order, OrderType, ProductType
from simulator.simulator import Simulator
from backtesting.backtester import Backtester
from backtesting.strategies import simple_momentum_strategy
from analytics.performance_analytics import PerformanceAnalytics
from trade_logging.trade_logger import TradeLogger
from charting.db_charts import DatabaseCharts


def run_comprehensive_example():
    """Run a comprehensive example demonstrating all new features."""
    print("Running Comprehensive Example - All Features Working Together")
    print("=" * 70)
    
    # 1. Initialize components
    print("1. Initializing System Components")
    print("-" * 30)
    
    # Data provider manager (with yFinance as primary provider)
    data_manager = get_data_manager()
    print("✓ Data provider manager initialized")
    
    # Selector engine with data manager
    selector_engine = SelectorEngine()
    selector_engine.set_data_manager(data_manager)
    print("✓ Selector engine initialized with data manager")
    
    # Risk manager
    risk_config = {
        'max_capital_per_trade': 0.01,
        'max_concurrent_trades': 10,
        'daily_loss_limit': 0.02,
        'single_name_exposure_limit': 0.05,
        'stop_loss_percent': 0.05
    }
    risk_manager = RiskManager(risk_config)
    risk_manager.set_total_equity(1000000.0)
    print("✓ Risk manager initialized")
    
    # Simulator as broker
    simulator = Simulator(initial_balance=1000000.0)
    print("✓ Simulator initialized")
    
    # Order router
    order_router = OrderRouter(simulator)
    print("✓ Order router initialized")
    
    # Trade logger
    trade_logger = TradeLogger()
    print("✓ Trade logger initialized")
    
    # Performance analytics
    analytics = PerformanceAnalytics(risk_free_rate=0.05)
    print("✓ Performance analytics initialized")
    
    # Charting system
    charts = DatabaseCharts()
    print("✓ Charting system initialized")
    
    # 2. Demonstrate dual data sources
    print("\n2. Demonstrating Dual Data Sources")
    print("-" * 30)
    
    symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS"]
    
    for symbol in symbols:
        # Get current price from primary provider (yFinance)
        price = data_manager.get_current_price(symbol)
        print(f"  {symbol}: ₹{price:.2f}")
    
    # 3. Demonstrate stock selection with data manager
    print("\n3. Demonstrating Stock Selection with Data Manager")
    print("-" * 30)
    
    # Select stocks using momentum strategy with real data
    selected_stocks = selector_engine.select_stocks(symbols=symbols)
    print(f"  Selected {len(selected_stocks)} stocks:")
    for stock in selected_stocks:
        print(f"    - {stock.get('symbol', 'UNKNOWN')}: {stock.get('signal', 'HOLD')}")
    
    # 4. Demonstrate risk management with stop-loss
    print("\n4. Demonstrating Risk Management with Stop-Loss")
    print("-" * 30)
    
    for symbol in symbols:
        # Calculate position size
        atr = 50.0  # Simulated ATR
        position_size = risk_manager.calculate_position_size(atr)
        print(f"  {symbol} position size: {position_size} shares")
        
        # Check stop-loss price
        entry_price = data_manager.get_current_price(symbol)
        stop_loss_price = risk_manager.calculate_stop_loss_price(entry_price, "BUY")
        print(f"  {symbol} stop-loss price: ₹{stop_loss_price:.2f} ({risk_config['stop_loss_percent']*100:.1f}% stop-loss)")
    
    # 5. Demonstrate Market-on-Open order
    print("\n5. Demonstrating Market-on-Open Order")
    print("-" * 30)
    
    if selected_stocks:
        symbol = selected_stocks[0].get('symbol', symbols[0])
        price = data_manager.get_current_price(symbol)
        position_size = risk_manager.calculate_position_size(50.0)  # Simulated ATR
        
        # Create MOO order
        moo_order = Order(
            symbol=symbol,
            quantity=position_size,
            order_type=OrderType.MARKET_ON_OPEN,
            transaction_type="BUY",
            product_type=ProductType.MIS,
            tag="COMPREHENSIVE_EXAMPLE_MOO"
        )
        
        # Place MOO order
        order_id = order_router.place_order(moo_order)
        print(f"  Placed MOO order for {symbol}")
        print(f"    Order ID: {order_id}")
        print(f"    Quantity: {position_size}")
        print(f"    Status: {moo_order.status.name}")
    
    # 6. Demonstrate backtesting with ASOF_DATE
    print("\n6. Demonstrating Backtesting with ASOF_DATE")
    print("-" * 30)
    
    # Fetch historical data for backtesting
    historical_data = None
    for symbol in symbols:
        try:
            data = data_manager.get_historical_data(symbol, period="3mo", interval="1d")
            if not data.empty:
                data['Symbol'] = symbol
                if historical_data is None:
                    historical_data = data
                else:
                    historical_data = pd.concat([historical_data, data], ignore_index=True)
        except Exception as e:
            print(f"    Error fetching data for {symbol}: {e}")
    
    if historical_data is not None and not historical_data.empty:
        # Initialize backtester
        backtester = Backtester(initial_capital=100000.0)
        
        # Set ASOF date to 1 month ago
        asof_date = datetime.now() - timedelta(days=30)
        backtester.set_asof_date(asof_date)
        print(f"  ASOF Date set to: {asof_date.strftime('%Y-%m-%d')}")
        
        # Run backtest
        results = backtester.run_backtest(
            strategy_func=simple_momentum_strategy,
            historical_data=historical_data,
            symbols=symbols,
            lookback_period=20,
            max_positions=3
        )
        
        print(f"  Backtest Results:")
        print(f"    Initial Capital: ₹{results['initial_capital']:,.2f}")
        print(f"    Final Capital: ₹{results['final_capital']:,.2f}")
        print(f"    Total Return: {results['total_return_percentage']:.2f}%")
        print(f"    Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"    Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"    Max Drawdown: {results['max_drawdown_percentage']:.2f}%")
        print(f"    Win Rate: {results['win_rate_percentage']:.2f}%")
        print(f"    Total Trades: {results['total_trades']}")
    
    # 7. Demonstrate performance analytics
    print("\n7. Demonstrating Performance Analytics")
    print("-" * 30)
    
    # Generate sample portfolio data
    portfolio_values = []
    trades = []
    
    # Simulate 30 days of portfolio values
    current_value = 100000.0
    for i in range(30):
        date = datetime.now() - timedelta(days=29-i)
        daily_return = (i % 5 - 2) * 0.001  # Simulate some volatility
        current_value *= (1 + daily_return)
        
        portfolio_values.append({
            'date': date,
            'value': current_value,
            'capital': current_value * 0.8,
            'positions_value': current_value * 0.2
        })
        
        # Simulate some trades
        if i % 7 == 0:  # Every 7 days
            action = "SELL" if i % 14 == 0 else "BUY"
            trades.append({
                'action': action,
                'quantity': 100,
                'price': 1500 + (i * 5),
                'proceeds': 150000 if action == "SELL" else 0,
                'cost': 150000 if action == "BUY" else 0
            })
    
    # Calculate performance metrics
    returns = analytics.calculate_returns(portfolio_values)
    sharpe_ratio = analytics.calculate_sharpe_ratio(returns)
    sortino_ratio = analytics.calculate_sortino_ratio(returns)
    drawdown_metrics = analytics.calculate_max_drawdown(portfolio_values)
    win_rate = analytics.calculate_win_rate(trades)
    volatility = analytics.calculate_volatility(returns)
    
    print(f"  Performance Metrics:")
    print(f"    Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"    Sortino Ratio: {sortino_ratio:.4f}")
    print(f"    Annualized Volatility: {volatility*100:.2f}%")
    print(f"    Max Drawdown: {drawdown_metrics['max_drawdown_percentage']:.2f}%")
    print(f"    Win Rate: {win_rate*100:.2f}%")
    
    # 8. Demonstrate trade logging
    print("\n8. Demonstrating Trade Logging")
    print("-" * 30)
    
    # Log a sample trade
    trade_data = {
        'order_id': 'EXAMPLE_ORDER_001',
        'symbol': 'INFY',
        'quantity': 100,
        'price': 1500.50,
        'transaction_type': 'BUY',
        'timestamp': datetime.now().isoformat()
    }
    
    log_id = trade_logger.log_trade_execution(trade_data)
    print(f"  Trade logged with ID: {log_id}")
    
    # Retrieve recent logs
    recent_logs = trade_logger.get_trade_logs(limit=3)
    print(f"  Recent trade logs ({len(recent_logs)} retrieved):")
    for log in recent_logs:
        print(f"    - [{log['timestamp']}] {log['module']}: {log['message']}")
    
    # 9. Demonstrate database charting
    print("\n9. Demonstrating Database Charting")
    print("-" * 30)
    
    # Get portfolio chart data
    portfolio_chart = charts.get_portfolio_value_chart_data(days=30)
    print(f"  Portfolio chart data points: {len(portfolio_chart)}")
    
    # Get P&L chart data
    pnl_chart = charts.get_pnl_chart_data(days=30)
    print(f"  P&L chart data points: {len(pnl_chart)}")
    
    # Get performance summary
    performance_summary = charts.get_performance_summary()
    print(f"  Performance Summary:")
    print(f"    Total Trades: {performance_summary.get('total_trades', 0)}")
    print(f"    Total P&L: ₹{performance_summary.get('total_pnl', 0):,.2f}")
    print(f"    Win Rate: {performance_summary.get('win_rate', 0):.1f}%")
    
    print("\n" + "=" * 70)
    print("Comprehensive Example Completed Successfully!")
    print("All requested features have been implemented and demonstrated:")
    print("  ✓ Dual data sources (yFinance as secondary provider)")
    print("  ✓ Stop-loss functionality in risk manager")
    print("  ✓ Backtesting with ASOF_DATE override")
    print("  ✓ Market-on-Open orders")
    print("  ✓ Performance analytics (Sharpe/Sortino ratios, drawdown metrics)")
    print("  ✓ Complete trade logging")
    print("  ✓ Database-based charting system")
    print("=" * 70)


if __name__ == "__main__":
    run_comprehensive_example()