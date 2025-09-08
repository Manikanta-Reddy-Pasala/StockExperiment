#!/usr/bin/env python3
"""
Test chart generation
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from visualization.matplotlib_charts import MatplotlibCharts

def test_chart():
    """Test chart generation."""
    print("Testing chart generation...")
    
    # Generate sample data
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    portfolio_data = []
    value = 1000000.0
    for date in dates:
        # Simulate some growth and volatility
        value = value * (1 + (0.001 * (0.5 - np.random.random())))
        portfolio_data.append({
            'date': date,
            'value': value,
            'capital': value * 0.8,
            'positions_value': value * 0.2
        })
    
    # Create chart
    charts = MatplotlibCharts()
    try:
        chart_path = charts.create_portfolio_value_chart(portfolio_data, "Test Portfolio Value")
        print(f"Chart saved to: {chart_path}")
    except Exception as e:
        print(f"Error generating chart: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chart()