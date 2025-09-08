"""
Performance Analytics Example
"""
import sys
import os
import random
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analytics.performance_analytics import PerformanceAnalytics


def generate_sample_data():
    """Generate sample portfolio data for testing."""
    # Generate dates for the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Generate portfolio values with some randomness
    portfolio_values = []
    trades = []
    
    current_value = 100000.0  # Starting value
    current_date = start_date
    
    while current_date <= end_date:
        # Add some randomness to the value
        daily_return = random.uniform(-0.02, 0.02)  # -2% to +2% daily return
        current_value *= (1 + daily_return)
        
        portfolio_values.append({
            'date': current_date,
            'value': current_value,
            'capital': current_value * random.uniform(0.8, 1.0),
            'positions_value': current_value * random.uniform(0.0, 0.2)
        })
        
        # Occasionally generate trades
        if random.random() < 0.05:  # 5% chance of a trade
            action = random.choice(['BUY', 'SELL'])
            quantity = random.randint(10, 100)
            price = random.uniform(100, 2000)
            
            if action == 'SELL':
                proceeds = quantity * price * random.uniform(0.99, 1.01)
                cost = quantity * price * random.uniform(0.98, 1.02)
                trades.append({
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'proceeds': proceeds,
                    'cost': cost
                })
            else:
                trades.append({
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'cost': quantity * price
                })
        
        # Move to next day (skip weekends)
        current_date += timedelta(days=1)
        if current_date.weekday() > 4:  # Saturday or Sunday
            current_date += timedelta(days=1)
    
    return portfolio_values, trades


def run_performance_analytics_example():
    """Run a performance analytics example."""
    print("Running Performance Analytics Example")
    print("=" * 50)
    
    # Generate sample data
    print("Generating sample portfolio data...")
    portfolio_values, trades = generate_sample_data()
    
    print(f"Generated {len(portfolio_values)} portfolio values")
    print(f"Generated {len(trades)} trades")
    
    # Initialize performance analytics
    analytics = PerformanceAnalytics(risk_free_rate=0.05)  # 5% risk-free rate
    
    # Calculate returns
    print("\n1. Calculating Returns")
    print("-" * 20)
    returns = analytics.calculate_returns(portfolio_values)
    print(f"Calculated {len(returns)} returns")
    print(f"Average daily return: {returns.mean()*100:.4f}%")
    print(f"Return volatility: {returns.std()*100:.4f}%")
    
    # Calculate Sharpe ratio
    print("\n2. Calculating Sharpe Ratio")
    print("-" * 20)
    sharpe_ratio = analytics.calculate_sharpe_ratio(returns)
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    # Calculate Sortino ratio
    print("\n3. Calculating Sortino Ratio")
    print("-" * 20)
    sortino_ratio = analytics.calculate_sortino_ratio(returns)
    print(f"Sortino Ratio: {sortino_ratio:.4f}")
    
    # Calculate maximum drawdown
    print("\n4. Calculating Maximum Drawdown")
    print("-" * 20)
    drawdown_metrics = analytics.calculate_max_drawdown(portfolio_values)
    print(f"Maximum Drawdown: {drawdown_metrics['max_drawdown_percentage']:.2f}%")
    print(f"Drawdown Duration: {drawdown_metrics['drawdown_duration']} days")
    
    # Calculate win rate
    print("\n5. Calculating Win Rate")
    print("-" * 20)
    win_rate = analytics.calculate_win_rate(trades)
    print(f"Win Rate: {win_rate*100:.2f}%")
    
    # Calculate volatility
    print("\n6. Calculating Volatility")
    print("-" * 20)
    volatility = analytics.calculate_volatility(returns)
    print(f"Annualized Volatility: {volatility*100:.2f}%")
    
    # Generate comprehensive performance report
    print("\n7. Generating Performance Report")
    print("-" * 20)
    report = analytics.generate_performance_report(portfolio_values, trades)
    
    print("Performance Report:")
    print(f"  Initial Value: ₹{report['initial_value']:,.2f}")
    print(f"  Final Value: ₹{report['final_value']:,.2f}")
    print(f"  Total Return: {report['total_return_percentage']:.2f}%")
    print(f"  Sharpe Ratio: {report['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio: {report['sortino_ratio']:.4f}")
    print(f"  Annualized Volatility: {report['volatility']*100:.2f}%")
    print(f"  Maximum Drawdown: {report['max_drawdown_percentage']:.2f}%")
    print(f"  Drawdown Duration: {report['drawdown_duration']} days")
    print(f"  Win Rate: {report['win_rate_percentage']:.2f}%")
    print(f"  Total Trades: {report['total_trades']}")
    
    print("\nPerformance Analytics Example Completed!")


if __name__ == "__main__":
    run_performance_analytics_example()