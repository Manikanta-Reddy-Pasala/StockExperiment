# Enhanced Stock Filtering System

A comprehensive stock filtering and discovery system with Stage 1 and Stage 2 analysis, supporting all features from the enhanced YAML configuration.

## Features

### 🎯 **Stage 1: Basic Filtering**
- **Price Range**: Minimum/maximum price filters
- **Volume & Turnover**: Daily turnover and volume requirements
- **Liquidity**: Liquidity score and bid-ask spread filters
- **Trading Status**: Active, tradeable, and listing requirements
- **Volatility**: ATR-based volatility filters

### 🔍 **Stage 2: Advanced Analysis**
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, ATR, ADX, Stochastic, Williams %R
- **Volume Analysis**: OBV, VPT, MFI, volume surge detection
- **Fundamental Ratios**: P/E, P/B, PEG, ROE, ROA, profit margin, debt-to-equity, current ratio
- **Risk Metrics**: Beta, volatility, Sharpe ratio, maximum drawdown, VaR
- **Momentum**: Price momentum, ROC, relative strength
- **Trend Analysis**: Trend direction, support/resistance, chart patterns, ADX

### 📊 **Scoring & Ranking**
- **Weighted Scoring**: Technical (30%), Fundamental (20%), Risk (20%), Momentum (25%), Volume (5%)
- **Filtering Thresholds**: Minimum scores for each category
- **Tie-Breaking**: Priority-based ranking system

### 🛡️ **Portfolio Guardrails**
- **Sector Concentration**: Maximum 40% in one sector
- **Market Cap Mix**: Minimum 60% large/mid-cap
- **Blacklist/Whitelist**: Symbol-based filtering
- **Resistance Distance**: Avoid buying near resistance levels

## Quick Start

### Basic Usage

```python
from src.services.stock_filtering import get_enhanced_discovery_service

# Get the discovery service
discovery_service = get_enhanced_discovery_service()

# Discover stocks with default configuration
result = discovery_service.discover_stocks(user_id=1)

print(f"Selected {result.final_selected} stocks from {result.total_processed} processed")
print(f"Execution time: {result.execution_time:.2f}s")

# Display selected stocks
for stock in result.selected_stocks:
    print(f"{stock['symbol']}: {stock['scores']['total']:.1f} total score")
```

### Custom Configuration

```python
from src.services.stock_filtering import get_enhanced_filtering_config, get_enhanced_discovery_service

# Get configuration
config = get_enhanced_filtering_config()

# Modify parameters
config.stage_1_filters.minimum_price = 10.0
config.stage_1_filters.maximum_price = 5000.0
config.stage_1_filters.minimum_daily_turnover_inr = 100000000

# Update scoring weights
config.scoring_weights.technical_score = 0.40
config.scoring_weights.fundamental_score = 0.30
config.scoring_weights.risk_score = 0.15
config.scoring_weights.momentum_score = 0.10
config.scoring_weights.volume_score = 0.05

# Update selection criteria
config.selection.max_suggested_stocks = 5
config.selection.sector_concentration_limit_pct = 30

# Apply custom configuration
discovery_service = get_enhanced_discovery_service()
discovery_service.update_config(config)

# Discover stocks with custom settings
result = discovery_service.discover_stocks(user_id=1)
```

### Technical Indicators

```python
from src.services.stock_filtering import get_technical_calculator
import pandas as pd

# Get technical calculator
calculator = get_technical_calculator()

# Configuration for indicators
config = {
    'rsi': {'enabled': True, 'period': 14},
    'macd': {'enabled': True, 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
    'bollinger_bands': {'enabled': True, 'period': 20, 'std_dev': 2.0},
    'moving_averages': {'enabled': True, 'sma_periods': [5, 10, 20, 50], 'ema_periods': [12, 26]},
    'atr': {'enabled': True, 'period': 14},
    'adx': {'enabled': True, 'period': 14}
}

# Calculate indicators for OHLCV data
indicators = calculator.calculate_all_indicators(df, config)

print(f"RSI: {indicators.rsi:.2f}")
print(f"MACD: {indicators.macd:.4f}")
print(f"Bollinger Upper: {indicators.bb_upper:.2f}")
```

## Configuration

The system uses a comprehensive YAML configuration file (`config/stock_filters.yaml`) with the following sections:

### Universe Controls
```yaml
universe:
  exchanges: ["NSE"]
  instrument_types: ["EQ"]
  min_history_days: 220
  min_non_null_ratio: 0.98
  max_price_gap_pct: 20.0
```

### Stage 1 Filters
```yaml
stage_1_filters:
  tradeability:
    minimum_price: 5.0
    maximum_price: 10000.0
    minimum_daily_turnover_inr: 50000000
    minimum_liquidity_score: 0.3
  trading_status:
    min_listing_days: 180
  liquidity:
    max_bid_ask_spread_pct: 1.0
  baseline_volatility:
    min_atr_pct_of_price: 1.0
    max_atr_pct_of_price: 7.0
```

### Stage 2 Filters
```yaml
stage_2_filters:
  technical_indicators:
    rsi:
      enabled: true
      period: 14
      oversold_threshold: 30
      overbought_threshold: 70
    macd:
      enabled: true
      fast_period: 12
      slow_period: 26
      signal_period: 9
    bollinger_bands:
      enabled: true
      period: 20
      std_dev: 2
  fundamental_ratios:
    pe_ratio:
      enabled: true
      minimum: 0
      maximum: 50
    roe:
      enabled: true
      minimum: 12
  risk_metrics:
    beta:
      enabled: true
      minimum: 0.7
      maximum: 1.6
    volatility:
      enabled: true
      maximum_daily: 4.0
      maximum_annual: 45.0
  scoring_weights:
    technical_score: 0.30
    fundamental_score: 0.20
    risk_score: 0.20
    momentum_score: 0.25
    volume_score: 0.05
```

### Selection Criteria
```yaml
selection:
  max_suggested_stocks: 10
  tie_breaker_priority: ["momentum_score", "risk_score", "technical_score"]
  sector_concentration_limit_pct: 40
  min_large_mid_pct: 60
  blacklist_symbols: []
  whitelist_symbols: []
  min_distance_from_resistance_pct: 2.0
```

## API Reference

### EnhancedStockDiscoveryService

Main service for stock discovery with comprehensive filtering.

#### Methods

- `discover_stocks(user_id: int = 1, limit: Optional[int] = None) -> DiscoveryResult`
  - Discover stocks using comprehensive filtering
  - Returns selected and rejected stocks with detailed analysis

- `get_discovery_statistics() -> Dict[str, Any]`
  - Get discovery statistics and performance metrics

- `reset_statistics()`
  - Reset discovery statistics

- `get_config() -> EnhancedFilteringConfig`
  - Get current configuration

- `update_config(new_config: EnhancedFilteringConfig)`
  - Update configuration

### EnhancedStockFilteringService

Core filtering service with Stage 1 and Stage 2 analysis.

#### Methods

- `filter_stocks(stocks: List[Any], user_id: int = 1) -> FilteringResult`
  - Apply comprehensive filtering to stocks
  - Returns filtering result with scores and reasons

- `get_filter_statistics() -> Dict[str, Any]`
  - Get filter statistics

- `reset_statistics()`
  - Reset filter statistics

### TechnicalIndicatorsCalculator

Calculator for technical analysis indicators.

#### Methods

- `calculate_all_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> TechnicalIndicators`
  - Calculate all technical indicators for OHLCV data
  - Returns comprehensive indicators object

### EnhancedFilteringConfig

Configuration class with all filtering parameters.

#### Properties

- `universe`: UniverseConfig - Data quality and universe controls
- `stage_1_filters`: Stage1Filters - Basic filtering criteria
- `technical_indicators`: TechnicalIndicators - Technical analysis settings
- `fundamental_ratios`: FundamentalRatios - Fundamental analysis settings
- `risk_metrics`: RiskMetrics - Risk assessment settings
- `scoring_weights`: ScoringWeights - Scoring weights for ranking
- `filtering_thresholds`: FilteringThresholds - Minimum score thresholds
- `selection`: SelectionConfig - Final selection criteria

## Data Structures

### DiscoveryResult
```python
@dataclass
class DiscoveryResult:
    selected_stocks: List[Dict[str, Any]]
    rejected_stocks: List[Dict[str, Any]]
    total_processed: int
    stage1_passed: int
    stage2_passed: int
    final_selected: int
    execution_time: float
    config_used: Dict[str, Any]
    summary: Dict[str, Any]
```

### StockScore
```python
@dataclass
class StockScore:
    symbol: str
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    risk_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    total_score: float = 0.0
    filters_passed: List[str] = None
    reject_reasons: List[str] = None
```

### TechnicalIndicators
```python
@dataclass
class TechnicalIndicators:
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_squeeze: Optional[bool] = None
    sma_5: Optional[float] = None
    sma_10: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_100: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    ema_50: Optional[float] = None
    atr: Optional[float] = None
    adx: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    williams_r: Optional[float] = None
    obv: Optional[float] = None
    vpt: Optional[float] = None
    mfi: Optional[float] = None
```

## Examples

See `example_usage.py` for comprehensive examples of:
- Basic stock discovery
- Custom configuration
- Technical indicators calculation
- Enhanced filtering service usage

## Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `pyyaml`: YAML configuration parsing
- `ta` or `pandas_ta`: Technical analysis indicators (optional)

## Performance

The system is optimized for performance with:
- Efficient data processing
- Caching of technical indicators
- Batch processing capabilities
- Configurable rate limiting
- Memory-efficient operations

## Error Handling

Comprehensive error handling with:
- Graceful degradation on errors
- Detailed error logging
- Fallback mechanisms
- Statistics tracking

## Logging

Extensive logging support with:
- Configurable log levels
- Performance metrics
- Error tracking
- Debug information

## Future Enhancements

- Machine learning integration
- Real-time data feeds
- Advanced pattern recognition
- Custom indicator support
- Portfolio optimization
- Risk management tools
