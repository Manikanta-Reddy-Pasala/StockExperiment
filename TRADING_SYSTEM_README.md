# Advanced Trading System - Stock Screening & Strategy Testing

## 🎯 Overview

This comprehensive trading system implements a complete workflow for stock screening, strategy testing, and performance evaluation. It includes:

- **Stock Screening**: Advanced filtering for mid-cap and small-cap stocks
- **Multiple Trading Strategies**: Momentum, Value, and Growth strategies
- **Dry Run Testing**: Virtual portfolio management for strategy validation
- **AI Analysis**: ChatGPT integration for intelligent stock analysis
- **Performance Tracking**: Comprehensive analytics and visualization
- **Scheduled Execution**: Automated workflow execution

## 🏗️ System Architecture

### Core Components

1. **Stock Screener** (`src/screening/`)
   - Filters stocks based on comprehensive criteria
   - Market cap range: 5000-20000 crores (mid-cap/small-cap)
   - Financial health indicators
   - Technical analysis metrics

2. **Strategy Engine** (`src/strategies/`)
   - **Momentum Strategy**: Price and volume momentum
   - **Value Strategy**: Fundamental value metrics
   - **Growth Strategy**: Revenue and profit growth

3. **Dry Run Manager** (`src/portfolio/`)
   - Virtual portfolio management
   - Strategy testing with fake capital
   - Performance tracking and evaluation

4. **Trading Executor** (`src/execution/`)
   - Orchestrates complete workflow
   - Scheduled execution capabilities
   - Multi-user support

5. **AI Analyzer** (`src/analysis/`)
   - ChatGPT integration for stock analysis
   - Portfolio recommendations
   - Strategy comparison

6. **Performance Tracker** (`src/analytics/`)
   - Historical performance analysis
   - Risk metrics calculation
   - Visualization and reporting

## 📊 Stock Screening Criteria

The system applies the following comprehensive screening criteria:

### Market Capitalization
- **Range**: 5000 - 20000 crores (mid-cap/small-cap focus)
- **Current Price**: > ₹50
- **Price vs Low**: Current price > Low price
- **Price vs DMA**: Current price > 50-day moving average

### Volume Analysis
- **Volume**: > 1-week average volume

### Financial Health
- **Sales Growth**: Latest quarter > Preceding quarter
- **Operating Profit**: Latest quarter > Preceding quarter
- **Year-over-Year**: Current year sales > Preceding year sales

### Valuation Metrics
- **Intrinsic Value**: Current price < Intrinsic value
- **Value Multiplier**: Intrinsic value > Current price × 2
- **High Price**: High price < Current price × 2

### Risk Metrics
- **Debt-to-Equity**: < 0.2
- **Piotroski Score**: > 5

## 🎯 Trading Strategies

### 1. Momentum Strategy
**Focus**: Price and volume momentum
- **Criteria**: Price momentum > 5%, Volume > 1.5x average
- **Position Size**: 10% per position (max 10 positions)
- **Exit Signal**: Momentum turns negative

### 2. Value Strategy
**Focus**: Undervalued stocks with strong fundamentals
- **Criteria**: PE < 15, PB < 2, Debt/Equity < 0.3, ROE > 15%
- **Position Size**: 12.5% per position (max 8 positions)
- **Exit Signal**: Stock becomes overvalued

### 3. Growth Strategy
**Focus**: High-growth companies
- **Criteria**: Revenue growth > 20%, Profit growth > 15%, ROE > 20%
- **Position Size**: 16.7% per position (max 6 positions)
- **Exit Signal**: Growth slows significantly

## 🔄 Dry Run System

### Features
- **Virtual Capital**: ₹1,00,000 starting capital
- **Real-time Tracking**: Portfolio value and performance
- **Strategy Testing**: Test multiple strategies simultaneously
- **Performance Metrics**: Returns, risk, drawdown analysis
- **Clean Exit**: Fake portfolios can be completely removed

### Portfolio Management
- **Equal Weight**: Equal allocation across selected stocks
- **Market Cap Weight**: Allocation based on market capitalization
- **Custom Allocation**: Strategy-specific allocation logic

## 🤖 AI Integration

### ChatGPT Analysis
- **Stock Analysis**: Individual stock recommendations
- **Portfolio Analysis**: Overall portfolio assessment
- **Strategy Comparison**: AI-powered strategy ranking
- **Risk Assessment**: Intelligent risk analysis

### Analysis Types
1. **Investment Recommendation**: BUY/HOLD/SELL with confidence
2. **Key Strengths**: 3-5 positive factors
3. **Key Risks**: 3-5 risk factors
4. **Price Target**: 12-month target price
5. **Investment Thesis**: Brief investment rationale

## 📈 Performance Tracking

### Metrics Tracked
- **Total Return**: Absolute and percentage returns
- **Daily Returns**: Day-by-day performance
- **Volatility**: Risk measurement
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable days
- **Position Analysis**: Individual stock performance

### Visualization
- **Portfolio Value Chart**: Value over time
- **Returns Distribution**: Daily returns histogram
- **Risk-Return Scatter**: Risk vs return analysis
- **Drawdown Chart**: Drawdown over time
- **Strategy Comparison**: Multi-metric comparison

## 🚀 API Endpoints

### Trading Operations
- `POST /api/trading/run-screening` - Run stock screening
- `POST /api/trading/run-strategies` - Apply trading strategies
- `POST /api/trading/run-dry-run` - Execute dry run mode
- `POST /api/trading/run-complete-workflow` - Full workflow execution

### Scheduling
- `POST /api/trading/start-scheduled-execution` - Start scheduled execution
- `POST /api/trading/stop-scheduled-execution` - Stop scheduled execution
- `GET /api/trading/execution-status` - Get execution status

### Analytics
- `GET /api/analytics/performance-report` - Generate performance report
- `GET /api/analytics/strategy-comparison` - Compare strategies
- `POST /api/analytics/generate-charts` - Generate visualization charts

### AI Analysis
- `POST /api/ai/analyze-stock` - Analyze individual stock
- `POST /api/ai/analyze-portfolio` - Analyze portfolio
- `POST /api/ai/compare-strategies` - Compare strategies with AI

## 🔧 Configuration

### Environment Variables
```bash
# OpenAI API for ChatGPT integration
OPENAI_API_KEY=your_openai_api_key_here

# Database configuration
DATABASE_URL=postgresql://user:password@localhost:5432/trading_system

# Admin user configuration
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_admin_password
```

### Strategy Parameters
Each strategy can be customized through parameters:
- **Momentum**: Lookback period, momentum threshold, volume threshold
- **Value**: PE/PB ratios, debt limits, ROE requirements
- **Growth**: Growth rates, ROE requirements

## 📊 Usage Examples

### 1. Run Complete Workflow
```python
from execution.trading_executor import TradingExecutor

executor = TradingExecutor(user_id=1)
result = executor.run_complete_workflow()

print(f"Selected {len(result['screened_stocks']['stocks'])} stocks")
print(f"Strategies executed: {list(result['strategy_results'].keys())}")
```

### 2. Test Single Strategy
```python
from execution.trading_executor import TradingExecutor

executor = TradingExecutor(user_id=1)
result = executor.run_dry_run_only('momentum')

print(f"Momentum strategy selected {len(result['strategy_results']['momentum'])} stocks")
```

### 3. Generate Performance Report
```python
from analytics.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()
report = tracker.generate_performance_report(['momentum', 'value', 'growth'])

print(f"Best performing strategy: {report['summary']['best_performing_strategy']}")
```

### 4. AI Stock Analysis
```python
from analysis.chatgpt_analyzer import ChatGPTAnalyzer

analyzer = ChatGPTAnalyzer()
analysis = analyzer.analyze_stock(stock_data)

print(f"Recommendation: {analysis['recommendation']}")
print(f"Confidence: {analysis['confidence']}/10")
```

## 🎯 Key Benefits

### 1. Comprehensive Screening
- **Multi-criteria filtering** ensures only quality stocks
- **Financial health checks** prevent risky investments
- **Technical analysis** identifies momentum opportunities

### 2. Strategy Diversification
- **Multiple strategies** reduce single-strategy risk
- **Different approaches** capture various market conditions
- **Backtesting capability** validates strategy effectiveness

### 3. Risk Management
- **Dry run testing** prevents real money losses
- **Position sizing** controls individual stock risk
- **Performance monitoring** tracks strategy effectiveness

### 4. AI-Powered Insights
- **Intelligent analysis** provides expert-level recommendations
- **Objective assessment** removes emotional bias
- **Continuous learning** improves over time

### 5. Performance Analytics
- **Detailed tracking** monitors all aspects of performance
- **Visualization** makes data easy to understand
- **Historical analysis** identifies patterns and trends

## 🔮 Future Enhancements

### Planned Features
1. **Real-time Data Integration**: Live market data feeds
2. **Advanced Strategies**: Machine learning-based strategies
3. **Risk Management**: Advanced risk controls and stop-losses
4. **Portfolio Optimization**: Modern portfolio theory implementation
5. **Mobile App**: Mobile interface for monitoring and control

### Integration Opportunities
1. **Broker APIs**: Direct order execution
2. **News Analysis**: Sentiment analysis integration
3. **Economic Indicators**: Macro-economic factor integration
4. **Social Trading**: Community-based strategy sharing

## 📝 Notes

- **Dry Run Mode**: Always test strategies in dry run mode before live trading
- **Data Quality**: Ensure reliable data sources for accurate screening
- **Risk Management**: Never risk more than you can afford to lose
- **Regular Monitoring**: Review performance regularly and adjust strategies
- **Market Conditions**: Strategies may perform differently in different market conditions

This system provides a robust foundation for systematic stock selection and strategy testing, combining quantitative analysis with AI-powered insights for informed investment decisions.


### Reset Everything

To completely reset the system:

```bash
./run.sh cleanup
./run.sh start
```