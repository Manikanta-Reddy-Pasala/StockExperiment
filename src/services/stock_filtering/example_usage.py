"""
Example Usage of Enhanced Stock Filtering System
Demonstrates how to use the comprehensive stock filtering with Stage 1 and Stage 2 analysis
"""

import logging
from typing import List, Dict, Any
from .enhanced_stock_discovery_service import get_enhanced_discovery_service
from .enhanced_config_loader import get_enhanced_filtering_config
from .enhanced_stock_filtering_service import get_enhanced_filtering_service
from .technical_indicators_calculator import get_technical_calculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_discovery():
    """Example of basic stock discovery."""
    print("=== Basic Stock Discovery Example ===")
    
    try:
        # Get the enhanced discovery service
        discovery_service = get_enhanced_discovery_service()
        
        # Discover stocks with default configuration
        result = discovery_service.discover_stocks(user_id=1)
        
        print(f"Total processed: {result.total_processed}")
        print(f"Stage 1 passed: {result.stage1_passed}")
        print(f"Stage 2 passed: {result.stage2_passed}")
        print(f"Final selected: {result.final_selected}")
        print(f"Execution time: {result.execution_time:.2f}s")
        
        # Display selected stocks
        if result.selected_stocks:
            print("\n=== Selected Stocks ===")
            for i, stock in enumerate(result.selected_stocks[:5], 1):
                print(f"{i}. {stock['symbol']}")
                print(f"   Total Score: {stock['scores']['total']:.1f}")
                print(f"   Technical: {stock['scores']['technical']:.1f}")
                print(f"   Fundamental: {stock['scores']['fundamental']:.1f}")
                print(f"   Risk: {stock['scores']['risk']:.1f}")
                print(f"   Momentum: {stock['scores']['momentum']:.1f}")
                print(f"   Volume: {stock['scores']['volume']:.1f}")
                print()
        
        # Display summary
        if result.summary:
            print("=== Discovery Summary ===")
            print(f"Success rate: {result.summary.get('success_rate', 0):.1f}%")
            print(f"Stage 1 pass rate: {result.summary.get('stage1_pass_rate', 0):.1f}%")
            print(f"Stage 2 pass rate: {result.summary.get('stage2_pass_rate', 0):.1f}%")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in basic discovery: {e}")
        return None


def example_custom_configuration():
    """Example of using custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    try:
        # Get the enhanced configuration
        config = get_enhanced_filtering_config()
        
        # Modify configuration parameters
        config.stage_1_filters.minimum_price = 10.0  # Increase minimum price
        config.stage_1_filters.maximum_price = 5000.0  # Decrease maximum price
        config.stage_1_filters.minimum_daily_turnover_inr = 100000000  # Increase turnover requirement
        
        # Modify scoring weights
        config.scoring_weights.technical_score = 0.40  # Increase technical weight
        config.scoring_weights.fundamental_score = 0.30  # Increase fundamental weight
        config.scoring_weights.risk_score = 0.15  # Decrease risk weight
        config.scoring_weights.momentum_score = 0.10  # Decrease momentum weight
        config.scoring_weights.volume_score = 0.05  # Keep volume weight
        
        # Modify selection criteria
        config.selection.max_suggested_stocks = 5  # Reduce to 5 stocks
        config.selection.sector_concentration_limit_pct = 30  # Reduce sector concentration
        
        # Get the discovery service with custom config
        discovery_service = get_enhanced_discovery_service()
        discovery_service.update_config(config)
        
        # Discover stocks with custom configuration
        result = discovery_service.discover_stocks(user_id=1)
        
        print(f"Custom config - Final selected: {result.final_selected}")
        print(f"Custom config - Execution time: {result.execution_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in custom configuration: {e}")
        return None


def example_technical_indicators():
    """Example of calculating technical indicators."""
    print("\n=== Technical Indicators Example ===")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate sample price data
        base_price = 100
        returns = np.random.normal(0, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create sample DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates))
        })
        
        # Get technical calculator
        calculator = get_technical_calculator()
        
        # Configuration for indicators
        config = {
            'rsi': {'enabled': True, 'period': 14},
            'macd': {'enabled': True, 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'bollinger_bands': {'enabled': True, 'period': 20, 'std_dev': 2.0},
            'moving_averages': {'enabled': True, 'sma_periods': [5, 10, 20, 50], 'ema_periods': [12, 26]},
            'atr': {'enabled': True, 'period': 14},
            'adx': {'enabled': True, 'period': 14},
            'stochastic': {'enabled': True, 'k_period': 14, 'd_period': 3},
            'williams_r': {'enabled': True, 'period': 14},
            'obv': {'enabled': True},
            'vpt': {'enabled': True},
            'mfi': {'enabled': True, 'period': 14}
        }
        
        # Calculate indicators
        indicators = calculator.calculate_all_indicators(df, config)
        
        print("Technical Indicators Calculated:")
        print(f"RSI: {indicators.rsi:.2f}" if indicators.rsi else "RSI: N/A")
        print(f"MACD: {indicators.macd:.4f}" if indicators.macd else "MACD: N/A")
        print(f"MACD Signal: {indicators.macd_signal:.4f}" if indicators.macd_signal else "MACD Signal: N/A")
        print(f"MACD Histogram: {indicators.macd_histogram:.4f}" if indicators.macd_histogram else "MACD Histogram: N/A")
        print(f"Bollinger Upper: {indicators.bb_upper:.2f}" if indicators.bb_upper else "Bollinger Upper: N/A")
        print(f"Bollinger Middle: {indicators.bb_middle:.2f}" if indicators.bb_middle else "Bollinger Middle: N/A")
        print(f"Bollinger Lower: {indicators.bb_lower:.2f}" if indicators.bb_lower else "Bollinger Lower: N/A")
        print(f"SMA 20: {indicators.sma_20:.2f}" if indicators.sma_20 else "SMA 20: N/A")
        print(f"EMA 12: {indicators.ema_12:.2f}" if indicators.ema_12 else "EMA 12: N/A")
        print(f"ATR: {indicators.atr:.2f}" if indicators.atr else "ATR: N/A")
        print(f"ADX: {indicators.adx:.2f}" if indicators.adx else "ADX: N/A")
        print(f"Stochastic K: {indicators.stochastic_k:.2f}" if indicators.stochastic_k else "Stochastic K: N/A")
        print(f"Williams %R: {indicators.williams_r:.2f}" if indicators.williams_r else "Williams %R: N/A")
        print(f"OBV: {indicators.obv:.0f}" if indicators.obv else "OBV: N/A")
        print(f"VPT: {indicators.vpt:.0f}" if indicators.vpt else "VPT: N/A")
        print(f"MFI: {indicators.mfi:.2f}" if indicators.mfi else "MFI: N/A")
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error in technical indicators: {e}")
        return None


def example_filtering_service():
    """Example of using the enhanced filtering service directly."""
    print("\n=== Enhanced Filtering Service Example ===")
    
    try:
        # Get the enhanced filtering service
        filtering_service = get_enhanced_filtering_service()
        
        # Get current configuration
        config = filtering_service.config
        
        print("Current Configuration:")
        print(f"Minimum price: {config.stage_1_filters.minimum_price}")
        print(f"Maximum price: {config.stage_1_filters.maximum_price}")
        print(f"Minimum turnover: {config.stage_1_filters.minimum_daily_turnover_inr:,}")
        print(f"Minimum liquidity score: {config.stage_1_filters.minimum_liquidity_score}")
        print(f"Max suggested stocks: {config.selection.max_suggested_stocks}")
        print(f"Sector concentration limit: {config.selection.sector_concentration_limit_pct}%")
        
        # Get filter statistics
        stats = filtering_service.get_filter_statistics()
        print(f"\nFilter Statistics:")
        print(f"Total processed: {stats['total_processed']}")
        print(f"Stage 1 passed: {stats['stage1_passed']}")
        print(f"Stage 2 passed: {stats['stage2_passed']}")
        print(f"Final selected: {stats['final_selected']}")
        print(f"Execution time: {stats['execution_time']:.2f}s")
        
        return filtering_service
        
    except Exception as e:
        logger.error(f"Error in filtering service: {e}")
        return None


def main():
    """Main example function."""
    print("Enhanced Stock Filtering System - Example Usage")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_discovery()
        example_custom_configuration()
        example_technical_indicators()
        example_filtering_service()
        
        print("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Error in main example: {e}")


if __name__ == "__main__":
    main()
