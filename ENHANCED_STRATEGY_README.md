# Enhanced Portfolio Strategy - 4-Step Trading System

## Overview
This enhanced portfolio strategy implements a comprehensive 4-step trading system integrated with FYERS broker APIs and machine learning predictions.

## 📊 Strategy Steps

### Step 1: Filtering (Initial Stock Universe)
**Goal:** Remove junk stocks before applying strategies.

**Indicators & Rules:**
- **Price Filter:** Exclude stocks < ₹50
- **Liquidity Filter:** Keep only if average_volume_20d > 500,000
- **Volatility Filter:** Remove stocks with ATR(14) % of price > 8-10%

### Step 2: Risk Strategy Allocation
**Goal:** Decide Safe vs High Risk bucket.

**Market Cap Categories:**
- **Large-cap:** > ₹50,000 Cr
- **Mid-cap:** ₹10,000-50,000 Cr
- **Small-cap:** < ₹10,000 Cr

**Allocation:**
- **Safe Strategy:** 50% Large-cap + 50% Mid-cap
- **High Risk Strategy:** 50% Mid-cap + 50% Small-cap

### Step 3: Entry Rules
**Goal:** Enter only when momentum is valid.

**Indicators & Rules:**
- **Moving Averages:** Price above 20-day EMA and 50-day EMA
- **Breakout:** Current price > last 20-day high
- **Volume Confirmation:** Volume ≥ 1.5× average of last 20 days
- **RSI:** RSI(14) between 50-70 (not overbought)

### Step 4: Exit Rules
**Goal:** Protect capital and lock profits within 10 days.

**Indicators & Rules:**
- **Profit Targets:**
  - Sell 50% at +5%
  - Sell remaining 50% at +10% (or before 10th day)
- **Stop Loss:** Exit if price falls 2-4% below entry
- **Time Stop:** Exit all positions at day 10, even if flat
- **Trailing Stop (Optional):** Move stop up as stock rises (e.g., 3% below current price)

## 🚀 Installation & Setup

### Prerequisites
```bash
# Ensure Python 3.8+ is installed
python --version

# Install required packages
pip install -r requirements.txt
```

### Configuration
1. Set up FYERS API credentials in `.env` file:
```
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key
FYERS_REDIRECT_URI=your_redirect_uri
```

2. (Optional) Customize strategy parameters in `strategy_config.json`

## 💻 Usage

### Basic Usage
```bash
# Run with default settings (Paper trading, Safe strategy)
python run_enhanced_strategy.py

# Run with custom capital and risk preference
python run_enhanced_strategy.py --capital 200000 --risk HIGH_RISK

# Run with configuration file
python run_enhanced_strategy.py --config strategy_config.json
```

### Execution Modes

#### 1. Paper Trading (Default)
Simulates trading without real money:
```bash
python run_enhanced_strategy.py --mode paper
```

#### 2. Live Trading
Executes real trades with actual capital:
```bash
python run_enhanced_strategy.py --mode live --capital 100000
```
⚠️ **WARNING:** Live mode uses real money. Double-check all settings before running.

#### 3. Backtesting
Test strategy on historical data:
```bash
python run_enhanced_strategy.py --mode backtest
```

## 📁 Project Structure
```
StockExperiment/
├── src/
│   └── services/
│       ├── enhanced_portfolio_strategy.py  # Main 4-step strategy engine
│       ├── fyers_api_service.py           # FYERS broker integration
│       └── ml/
│           └── prediction_service.py      # ML predictions (preserved)
├── run_enhanced_strategy.py               # Main execution script
├── strategy_config.json                   # Strategy configuration
└── ENHANCED_STRATEGY_README.md           # This file
```

## 🤖 Machine Learning Integration
The strategy preserves and utilizes the existing ML prediction service:
- ML predictions are called for each stock during entry signal validation
- Predictions enhance decision-making but don't override rule-based signals
- ML confidence scores are tracked and logged for analysis

## 📊 Performance Metrics
The system tracks:
- Total P&L (Realized + Unrealized)
- Win Rate (Winning trades / Total trades)
- Return Percentage
- Active Positions
- Individual position performance

## 🔄 Continuous Monitoring
Once positions are entered, the system:
- Monitors prices every 60 seconds (configurable)
- Checks all exit conditions automatically
- Updates trailing stops if enabled
- Executes exits when conditions are met

## ⚙️ Configuration Options

### Filtering Criteria
- `min_price`: Minimum stock price (default: ₹50)
- `min_volume`: Minimum 20-day average volume (default: 500,000)
- `max_atr_percent`: Maximum ATR as % of price (default: 10%)

### Entry Rules
- `rsi_min/max`: RSI range (default: 50-70)
- `volume_multiplier`: Volume spike requirement (default: 1.5x)
- `breakout_days`: Breakout period (default: 20 days)

### Exit Rules
- `target1/target2`: Profit targets (default: 5%, 10%)
- `stop_loss`: Stop loss percentage (default: 3%)
- `max_days`: Maximum holding period (default: 10 days)
- `trailing_stop`: Enable/disable trailing stop (default: true)

## 📝 Logging
All activities are logged to:
- Console output for real-time monitoring
- `portfolio_strategy.log` for detailed debugging
- `strategy_results_[timestamp].json` for each execution

## ⚠️ Risk Management
- Maximum 10 concurrent positions
- 10% of capital per position (configurable)
- Automatic stop-loss on all positions
- Time-based exit after 10 days
- Trailing stop option for profit protection

## 🔍 Troubleshooting
1. **FYERS connection issues:** Check API credentials in `.env`
2. **No stocks passing filters:** Adjust filtering criteria in config
3. **ML predictions failing:** Check ML service logs
4. **Orders not executing:** Verify market hours and stock liquidity

## 📞 Support
For issues or questions:
1. Check the logs in `portfolio_strategy.log`
2. Review the FYERS_INTEGRATION_GUIDE.md
3. Consult ML_INTEGRATION_SUCCESS.md for ML-related issues

## 🎯 Best Practices
1. Start with paper trading to test the strategy
2. Use conservative capital allocation initially
3. Monitor the first few trades closely
4. Adjust parameters based on market conditions
5. Keep the ML model updated with recent data

---
**Disclaimer:** This trading system involves financial risk. Past performance does not guarantee future results. Always conduct your own research and consider consulting with a financial advisor.