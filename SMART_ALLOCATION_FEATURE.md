# Smart Allocation Feature - ML Score-Based Investment

## Overview
Implemented intelligent allocation system that automatically calculates optimal stock quantities based on ML prediction scores. Higher confidence predictions receive larger allocations.

## Key Features

### 1. ML Score-Based Allocation
**Principle**: Investment is distributed proportionally based on ML prediction confidence scores.

**Formula**:
```
Stock Allocation % = (Stock ML Score / Total ML Scores) × 100%
Stock Investment = Total Investment × Stock Allocation %
Stock Quantity = Floor(Stock Investment / Stock Price)
```

**Example**:
```
Total Investment: ₹50,000
Stocks:
- RELIANCE: ML Score 0.75 (75%) → Gets 33.3% = ₹16,650
- TCS: ML Score 0.68 (68%) → Gets 30.2% = ₹15,100
- INFY: ML Score 0.52 (52%) → Gets 23.1% = ₹11,550
- HDFC: ML Score 0.30 (30%) → Gets 13.3% = ₹6,650

Higher ML scores = More investment allocation
```

### 2. Investment Amount Input
**Location**: Bulk order modal
**Field**: "Investment Amount (₹)"
**Purpose**: User specifies total amount to invest
**Behavior**:
- Default: ₹50,000
- Minimum: ₹1,000
- Step: ₹1,000
- Real-time recalculation on change

### 3. Use Account Balance Option
**Location**: Checkbox next to investment amount
**Label**: "Use Account Balance"
**Behavior**:
- Shows current balance: ₹100,000 (from user settings)
- When checked: Auto-fills investment amount with balance
- Immediately recalculates allocation

**Future Enhancement**: Load balance from database/settings API

### 4. Automatic Quantity Calculation
**Process**:
1. Calculate allocation percentage for each stock based on ML score
2. Determine investment amount per stock
3. Divide by stock price to get quantity
4. Round down to whole shares (no fractional shares)
5. Calculate actual cost based on whole shares

**Smart Handling**:
- Stocks with 0 ML score get 0 allocation
- Rounding ensures no over-spending
- Actual total may be slightly less than planned (due to rounding)

## Modal Layout

### Input Section
```
┌─────────────────────────────────────────────────────────┐
│ Model Type: TRADITIONAL                                  │
│ Strategy: DEFAULT RISK                                   │
├─────────────────────────────────────────────────────────┤
│ Total Stocks: 10                                         │
│ Investment Amount: [50000] ₹                             │
│ □ Use Account Balance (Balance: ₹100,000)               │
├─────────────────────────────────────────────────────────┤
│ ℹ Smart Allocation: Quantities automatically calculated │
│   based on ML prediction scores. Higher scores get      │
│   larger allocations.                                    │
└─────────────────────────────────────────────────────────┘
```

### Stock Table
```
#  Symbol    Company      Score   Price      Qty  Cost        Allocation
─────────────────────────────────────────────────────────────────────────
1  RELIANCE  Reliance...  75.2%✓  ₹2,450.00  6   ₹14,700.00  33.3%
2  TCS       TCS Ltd...   72.1%✓  ₹3,200.00  4   ₹12,800.00  29.0%
3  INFY      Infosys...   68.5%   ₹1,500.00  7   ₹10,500.00  23.8%
4  HDFC      HDFC Bank    62.3%   ₹1,650.00  3   ₹4,950.00   11.2%
5  ICICI     ICICI Bk     58.9%   ₹800.00    1   ₹800.00     1.8%
─────────────────────────────────────────────────────────────────────────
                                  Total:     21  ₹44,150.00  100%
```

✓ = High confidence (≥70% ML score) - shown in bold green

### Recommendation Section
```
💡 Allocation Recommendation:
High Confidence (≥70%): RELIANCE (6 shares, ₹14,700), TCS (4 shares, ₹12,800).
Medium Confidence (60-70%): INFY (7 shares), HDFC (3 shares).
```

## Use Cases

### Use Case 1: Standard Investment
**Scenario**: User has ₹50,000 to invest

**Steps**:
1. Click "Buy All" on Traditional + Default Risk
2. Modal opens with default ₹50,000
3. See allocation:
   - High score stocks get more shares
   - Low score stocks get fewer shares
4. Total shows ~₹48,000 (due to rounding)
5. Confirm order

**Result**:
- Orders placed with quantities based on ML confidence
- Higher confidence stocks have larger positions

### Use Case 2: Use Full Balance
**Scenario**: User wants to invest entire account balance

**Steps**:
1. Click "Buy All"
2. Check "Use Account Balance"
3. Investment amount auto-fills with ₹100,000
4. Quantities recalculate instantly
5. Review allocation
6. Confirm

**Result**:
- All available balance invested
- Allocation still proportional to ML scores

### Use Case 3: Custom Amount
**Scenario**: User wants to invest specific amount (₹25,000)

**Steps**:
1. Click "Buy All"
2. Enter ₹25,000 in investment field
3. Table updates in real-time
4. See smaller quantities across all stocks
5. Confirm

**Result**:
- Exact budget control
- Proportional allocation maintained

### Use Case 4: Compare Strategies
**Scenario**: User wants to see allocation across all 4 strategies

**Steps**:
1. Click "Buy All" on Traditional + Default Risk → See allocation
2. Cancel, switch to Traditional + High Risk → See different stocks, different allocation
3. Cancel, switch to Raw LSTM + Default Risk → Compare
4. Cancel, switch to Raw LSTM + High Risk → Final comparison
5. Choose best strategy based on:
   - Stock selection
   - ML confidence levels
   - Investment distribution

**Result**:
- Informed decision across all 4 model/strategy combinations

## Technical Implementation

### Allocation Algorithm
```javascript
// 1. Calculate total score weight
const totalScoreWeight = stocks.reduce((sum, stock) => {
    return sum + (stock.ml_prediction_score || 0);
}, 0);

// 2. For each stock, calculate allocation
stocks.forEach(stock => {
    const score = stock.ml_prediction_score || 0;

    // Percentage of total investment
    const allocationPercent = (score / totalScoreWeight) * 100;

    // Amount allocated to this stock
    const allocatedAmount = (totalInvestment * score) / totalScoreWeight;

    // Quantity (round down)
    const quantity = Math.floor(allocatedAmount / stock.price);

    // Actual cost
    const cost = quantity * stock.price;
});
```

### Real-Time Updates
```javascript
// Investment amount change
document.getElementById('bulk-investment-amount').addEventListener('input', function() {
    const amount = parseFloat(this.value) || 0;
    if (amount > 0) {
        calculateSmartAllocation(amount);
    }
});

// Use balance checkbox
document.getElementById('bulk-use-balance').addEventListener('change', function() {
    if (this.checked) {
        document.getElementById('bulk-investment-amount').value = userBalance;
        calculateSmartAllocation(userBalance);
    }
});
```

## Benefits

### For Users
1. **No Manual Calculation**: System does complex math automatically
2. **Data-Driven**: Allocation based on ML confidence, not guesswork
3. **Flexible Budget**: Works with any investment amount
4. **Transparent**: See exactly how much goes to each stock
5. **Risk Managed**: Higher confidence stocks get more allocation

### For Portfolio Strategy
1. **Optimized Returns**: More money in high-confidence predictions
2. **Risk Distribution**: Still diversified across multiple stocks
3. **Consistent Logic**: Same allocation formula across all strategies
4. **Measurable**: Can track performance of allocation strategy

## Example Scenarios

### Scenario 1: High Confidence Portfolio
```
Investment: ₹100,000
Stocks (Traditional + Default Risk):
- Score 75%: Gets ₹30,000 (30% allocation)
- Score 72%: Gets ₹28,800 (28.8%)
- Score 68%: Gets ₹27,200 (27.2%)
- Score 52%: Gets ₹14,000 (14%)

Top 3 stocks get 86% of total investment
```

### Scenario 2: Balanced Confidence
```
Investment: ₹50,000
Stocks (Raw LSTM + High Risk):
- Score 65%: Gets ₹13,000 (26%)
- Score 63%: Gets ₹12,600 (25.2%)
- Score 61%: Gets ₹12,200 (24.4%)
- Score 60%: Gets ₹12,000 (24%)
- Score 51%: Gets ₹10,200 (20.4%)

More evenly distributed due to similar scores
```

## Order Placement Logic

```javascript
// Only place orders for stocks with quantity > 0
for (const stock of allocatedStocks) {
    if (stock.quantity === 0) continue;

    await fetch('/api/mock-trading/order', {
        method: 'POST',
        body: JSON.stringify({
            symbol: stock.symbol,
            quantity: stock.quantity,  // Calculated quantity
            model_type: modelType,
            strategy: strategy,
            ml_prediction_score: stock.ml_prediction_score,
            ml_price_target: stock.ml_price_target
        })
    });
}
```

## Success Summary

After order placement:
```
✅ Smart allocation order completed!

Total stocks: 10
✅ Successful: 10
❌ Failed: 0
💰 Total Invested: ₹48,150.00
📊 Planned Investment: ₹50,000.00

Model: TRADITIONAL
Strategy: DEFAULT RISK

Allocation based on ML prediction scores.
```

## Future Enhancements

1. **Advanced Allocation Strategies**:
   - Square root of ML score (less aggressive)
   - Exponential weighting (more aggressive)
   - Risk-adjusted allocation (ML score × risk score)

2. **Constraints**:
   - Minimum investment per stock (e.g., ₹5,000)
   - Maximum allocation per stock (e.g., 30%)
   - Sector diversification limits

3. **User Preferences**:
   - Save preferred investment amount
   - Auto-buy settings with balance
   - Scheduled recurring investments

4. **Visualization**:
   - Pie chart showing allocation breakdown
   - Comparison chart across strategies
   - Historical performance of allocation strategy

5. **Integration**:
   - Load balance from user settings API
   - Real-time balance updates
   - Transaction history with allocation details
