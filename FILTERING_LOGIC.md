# Stock Filtering Logic Documentation

## Overview
The stock discovery system uses a two-stage filtering approach to identify tradeable stocks from the entire market universe, progressively narrowing down candidates based on increasingly sophisticated criteria.

## Stage 1 Filters: Basic Market Screening

**Purpose**: Reduce the entire stock universe to a manageable set of liquid, tradeable stocks.

### Steps in Stage 1:

1. **Price Range Filter**
   - Logic: Eliminate penny stocks and extremely expensive stocks
   - Criteria: ₹5 ≤ Price ≤ ₹10,000
   - Rationale: Ensures adequate price movement and reduces manipulation risk

2. **Volume Filter**
   - Logic: Ensure sufficient trading activity for entry/exit
   - Criteria: Minimum 10,000 shares daily volume
   - Rationale: Liquidity requirement for smooth order execution

3. **Market Cap Classification**
   - Logic: Categorize stocks by size for risk assessment
   - Categories:
     - Large Cap: >₹20,000 crores (Lower risk, stable)
     - Mid Cap: ₹5,000-20,000 crores (Moderate risk, growth potential)
     - Small Cap: <₹5,000 crores (Higher risk, high growth potential)

4. **Liquidity Score Filter**
   - Logic: Measure ease of trading based on volume and spreads
   - Criteria: Minimum liquidity score of 0.3
   - Calculation: Weighted combination of normalized volume (70%) and bid-ask spread (30%)

5. **Trading Status Verification**
   - Logic: Ensure stocks are actively tradeable
   - Criteria: Exclude suspended, delisted, or halt stocks
   - Additional: Minimum 30 days since listing

**Output**: ~200-500 stocks that meet basic tradeable criteria

## Stage 2 Filters: Advanced Analysis

**Purpose**: Apply sophisticated technical and fundamental analysis to identify high-probability trading opportunities.

### Steps in Stage 2:

1. **Technical Indicator Analysis**
   - **RSI (Relative Strength Index)**
     - Logic: Identify overbought/oversold conditions
     - Buy Signal: RSI < 30 (oversold)
     - Sell Signal: RSI > 70 (overbought)

   - **MACD (Moving Average Convergence Divergence)**
     - Logic: Detect trend changes and momentum
     - Buy Signal: MACD line crosses above signal line
     - Sell Signal: MACD line crosses below signal line

   - **Bollinger Bands**
     - Logic: Identify volatility and mean reversion opportunities
     - Buy Signal: Price touches lower band
     - Sell Signal: Price touches upper band

2. **Volume Analysis**
   - **Volume Surge Detection**
     - Logic: Identify unusual trading activity
     - Criteria: Current volume > 1.5x average volume

   - **On-Balance Volume (OBV)**
     - Logic: Confirm price movements with volume
     - Signal: OBV trend should align with price trend

3. **Fundamental Ratio Screening**
   - **Valuation Metrics**
     - P/E Ratio: < 25 (not overvalued)
     - P/B Ratio: < 3 (reasonable book value)
     - PEG Ratio: < 1.5 (growth at reasonable price)

   - **Profitability Metrics**
     - ROE: > 15% (efficient equity usage)
     - ROA: > 5% (effective asset utilization)
     - Net Profit Margin: > 10% (healthy profitability)

   - **Financial Health**
     - Debt-to-Equity: < 1.0 (manageable debt)
     - Current Ratio: > 1.2 (adequate liquidity)

4. **Risk Assessment**
   - **Volatility Analysis**
     - Logic: Measure price stability
     - Criteria: 30-day volatility within acceptable range

   - **Beta Calculation**
     - Logic: Assess systematic risk relative to market
     - Preference: Beta between 0.8-1.5 (moderate risk)

5. **Momentum Indicators**
   - **Price Momentum**
     - Logic: Identify trending stocks
     - Calculation: Price change over multiple timeframes

   - **Relative Strength**
     - Logic: Compare stock performance vs. market
     - Signal: Outperforming stocks preferred

6. **Trend Analysis**
   - **Moving Average Convergence**
     - Logic: Confirm trend direction
     - Signal: Price above 20-day and 50-day MA (uptrend)

   - **Support/Resistance Levels**
     - Logic: Identify key price levels
     - Signal: Buy near support, sell near resistance

**Output**: ~20-50 high-quality trading candidates

## Scoring and Ranking System

### Weighted Scoring Logic:
- **Technical Score (40%)**
  - Combines all technical indicators
  - Range: 0-100 points

- **Fundamental Score (35%)**
  - Weighted average of financial ratios
  - Range: 0-100 points

- **Risk Score (15%)**
  - Lower risk = higher score
  - Range: 0-100 points

- **Momentum Score (10%)**
  - Recent price and volume momentum
  - Range: 0-100 points

### Final Selection Criteria:
- Minimum composite score: 60/100
- Must pass at least 3 out of 4 category minimums
- Risk-adjusted ranking for final prioritization

## Configuration Flexibility

All thresholds, weights, and criteria are configurable through `config/stock_filters.yaml`, allowing for:
- Strategy-specific parameter tuning
- Market condition adaptations
- Risk tolerance adjustments
- Backtesting different configurations

This systematic approach ensures consistent, objective stock selection while maintaining flexibility for different trading strategies and market conditions.