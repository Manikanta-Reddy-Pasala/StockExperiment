"""
Complete Feature Demonstration
This example shows all the newly implemented features working together.
"""
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_provider.data_manager import get_data_manager
from selector.selector_engine import SelectorEngine
from ai.chatgpt_validator import ChatGPTValidator
from visualization.matplotlib_charts import MatplotlibCharts
from datastore.database import get_database_manager


def demo_complete_features():
    """Demonstrate all newly implemented features."""
    print("Complete Feature Demonstration")
    print("=" * 50)
    
    # 1. Initialize components
    print("1. Initializing Components")
    print("-" * 30)
    
    # Data provider manager
    data_manager = get_data_manager()
    print("✓ Data provider manager initialized")
    
    # Selector engine
    selector = SelectorEngine()
    selector.set_data_manager(data_manager)
    print("✓ Selector engine initialized")
    
    # ChatGPT validator (without API key for demo)
    chatgpt_validator = ChatGPTValidator()
    selector.set_chatgpt_validator(chatgpt_validator)
    print("✓ ChatGPT validator initialized")
    
    # Matplotlib charts
    matplotlib_charts = MatplotlibCharts()
    print("✓ Matplotlib charts initialized")
    
    # Database
    db_manager = get_database_manager()
    db_manager.create_tables()
    print("✓ Database initialized")
    
    # 2. Demonstrate dual data sources
    print("\n2. Dual Data Sources")
    print("-" * 30)
    
    symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS"]
    
    for symbol in symbols:
        price = data_manager.get_current_price(symbol)
        print(f"  {symbol}: ₹{price:.2f}")
    
    # 3. Demonstrate stock selection with ChatGPT validation
    print("\n3. Stock Selection with ChatGPT Validation")
    print("-" * 30)
    
    # Select stocks using momentum strategy
    selected_stocks = selector.select_stocks(symbols=symbols)
    print(f"  Selected {len(selected_stocks)} stocks:")
    
    for stock in selected_stocks:
        print(f"    - {stock.get('symbol', 'UNKNOWN')}:")
        print(f"        Signal: {stock.get('signal', 'HOLD')}")
        print(f"        Momentum: {stock.get('momentum', 0):.4f}")
        print(f"        ChatGPT Score: {stock.get('chatgpt_score', 0)}")
        print(f"        Recommendation: {stock.get('chatgpt_recommendation', 'N/A')}")
    
    # 4. Demonstrate Matplotlib visualizations
    print("\n4. Matplotlib Visualizations")
    print("-" * 30)
    
    # Generate sample portfolio data
    portfolio_data = []
    current_value = 100000.0
    for i in range(30):
        date = datetime.now() - timedelta(days=29-i)
        daily_return = (i % 5 - 2) * 0.001
        current_value *= (1 + daily_return)
        portfolio_data.append({
            'date': date,
            'value': current_value
        })
    
    # Create portfolio chart
    portfolio_chart = matplotlib_charts.create_portfolio_value_chart(
        portfolio_data, 
        "Sample Portfolio Performance"
    )
    print(f"  Portfolio value chart created (size: {len(portfolio_chart)} characters)")
    
    # Create performance comparison chart
    # Sample ChatGPT strategy data
    chatgpt_data = []
    value = 100
    for i in range(60):
        date = datetime.now() - timedelta(days=59-i)
        value *= (1 + (i % 7 - 3) * 0.002)
        chatgpt_data.append({
            'date': date,
            'value': value
        })
    
    # Sample index data (NIFTY 50)
    index_data = []
    value = 100
    for i in range(60):
        date = datetime.now() - timedelta(days=59-i)
        value *= (1 + (i % 5 - 2) * 0.0015)
        index_data.append({
            'date': date,
            'value': value
        })
    
    comparison_chart = matplotlib_charts.create_performance_comparison_chart(
        chatgpt_data,
        index_data,
        "ChatGPT Strategy vs NIFTY 50 Index"
    )
    print(f"  Performance comparison chart created (size: {len(comparison_chart)} characters)")
    
    # 5. Demonstrate database integration
    print("\n5. Database Integration")
    print("-" * 30)
    
    # This would normally save data to the database
    print("  Database integration ready for production use")
    print("  Supports both SQLite (development) and PostgreSQL (production)")
    
    print("\n" + "=" * 50)
    print("Complete Feature Demonstration Finished!")
    print("=" * 50)
    print("\nTo run the full application:")
    print("  Local development: ./run_local.sh")
    print("  Production: ./run_production.sh")
    print("  With Docker: docker-compose up")
    print("\nTo access the web interface:")
    print("  Open http://localhost:5000 in your browser")


if __name__ == "__main__":
    demo_complete_features()