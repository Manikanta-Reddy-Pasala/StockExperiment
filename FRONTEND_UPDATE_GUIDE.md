# Frontend Update Guide - Dual Model View

## Current Problem

The current `suggested_stocks.html` shows:
- ❌ Only ONE model at a time (user doesn't know which model)
- ❌ Only ONE risk level at a time (user has to toggle)
- ❌ No comparison between models
- ❌ No visibility into dual model system

## Solution

Update the page to show ALL 4 combinations simultaneously:

```
┌─────────────────────────────────┬─────────────────────────────────┐
│   TRADITIONAL MODEL             │   RAW LSTM MODEL                │
├─────────────────────────────────┼─────────────────────────────────┤
│ DEFAULT RISK  │  HIGH RISK      │ DEFAULT RISK  │  HIGH RISK      │
└─────────────────────────────────┴─────────────────────────────────┘
```

## Step-by-Step Update

### Step 1: Update JavaScript API Call

**Current (Line 141):**
```javascript
const response = await fetch(`/api/suggested-stocks/?strategy=${riskStrategy}&limit=20`, ...);
```

**New:**
```javascript
const response = await fetch(`/api/suggested-stocks/dual-model-view?limit=10`, ...);
```

### Step 2: Update Data Structure

**Current:**
```javascript
let basicStocks = [];  // Single array
```

**New:**
```javascript
let dualModelData = {
    traditional: {
        default_risk: [],
        high_risk: []
    },
    raw_lstm: {
        default_risk: [],
        high_risk: []
    }
};
```

### Step 3: Update HTML Layout

Replace the single table with 4 sections:

```html
<div class="row">
    <!-- Traditional Model -->
    <div class="col-md-6 mb-4">
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h5>🏛️ Traditional Model</h5>
                <small>Feature-Engineered Ensemble (RF + XGBoost + LSTM)</small>
            </div>
            <div class="card-body">
                <!-- Tabs for Default/High Risk -->
                <ul class="nav nav-tabs" role="tablist">
                    <li class="nav-item">
                        <a class="nav-link active" data-bs-toggle="tab" href="#trad-default">
                            🛡️ Default Risk
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" data-bs-toggle="tab" href="#trad-high">
                            ⚡ High Risk
                        </a>
                    </li>
                </ul>
                <div class="tab-content">
                    <div class="tab-pane fade show active" id="trad-default">
                        <table class="table" id="traditional-default-table">
                            <!-- Data here -->
                        </table>
                    </div>
                    <div class="tab-pane fade" id="trad-high">
                        <table class="table" id="traditional-high-table">
                            <!-- Data here -->
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Raw LSTM Model -->
    <div class="col-md-6 mb-4">
        <div class="card">
            <div class="card-header bg-success text-white">
                <h5>🤖 Raw LSTM Model</h5>
                <small>Research-Based Raw OHLCV (5 features only)</small>
            </div>
            <div class="card-body">
                <!-- Tabs for Default/High Risk -->
                <ul class="nav nav-tabs" role="tablist">
                    <li class="nav-item">
                        <a class="nav-link active" data-bs-toggle="tab" href="#lstm-default">
                            🛡️ Default Risk
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" data-bs-toggle="tab" href="#lstm-high">
                            ⚡ High Risk
                        </a>
                    </li>
                </ul>
                <div class="tab-content">
                    <div class="tab-pane fade show active" id="lstm-default">
                        <table class="table" id="lstm-default-table">
                            <!-- Data here -->
                        </table>
                    </div>
                    <div class="tab-pane fade" id="lstm-high">
                        <table class="table" id="lstm-high-table">
                            <!-- Data here -->
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
```

### Step 4: Update JavaScript Functions

**New fetch function:**
```javascript
async function fetchDualModelView() {
    try {
        const response = await fetch('/api/suggested-stocks/dual-model-view?limit=10');
        const data = await response.json();

        if (data.success) {
            dualModelData = data.data;

            // Populate all 4 tables
            populateTable('traditional-default-table', dualModelData.traditional.default_risk);
            populateTable('traditional-high-table', dualModelData.traditional.high_risk);
            populateTable('lstm-default-table', dualModelData.raw_lstm.default_risk);
            populateTable('lstm-high-table', dualModelData.raw_lstm.high_risk);

            // Show statistics
            console.log('Statistics:', data.statistics);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}
```

**New populate function:**
```javascript
function populateTable(tableId, stocks) {
    const table = document.getElementById(tableId);
    const tbody = table.getElementsByTagName('tbody')[0];
    tbody.innerHTML = '';

    if (stocks.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted">
                    No predictions available
                </td>
            </tr>
        `;
        return;
    }

    stocks.forEach((stock, index) => {
        const row = tbody.insertRow();

        // Rank
        row.insertCell(0).textContent = index + 1;

        // Symbol
        row.insertCell(1).innerHTML = `<strong>${stock.symbol}</strong>`;

        // Company
        row.insertCell(2).textContent = stock.stock_name;

        // Score
        const scoreCell = row.insertCell(3);
        scoreCell.textContent = (stock.ml_prediction_score * 100).toFixed(1) + '%';
        scoreCell.className = stock.ml_prediction_score > 0.6 ? 'text-success fw-bold' : 'text-warning';

        // Target
        row.insertCell(4).textContent = '₹' + stock.ml_price_target.toFixed(2);

        // Recommendation
        const recCell = row.insertCell(5);
        const rec = stock.recommendation;
        const color = rec === 'BUY' ? 'success' : rec === 'SELL' ? 'danger' : 'secondary';
        recCell.innerHTML = `<span class="badge bg-${color}">${rec}</span>`;
    });
}
```

## Quick Implementation

### Option 1: Minimal Change (Easiest)

Just change line 141 in `suggested_stocks.html`:

```javascript
// OLD
const response = await fetch(`/api/suggested-stocks/?strategy=${riskStrategy}&limit=20`, ...);

// NEW
const response = await fetch(`/api/suggested-stocks/dual-model-view?limit=10`, ...);
```

Then update the display logic to show all 4 groups.

### Option 2: Complete Redesign (Recommended)

Replace the entire `suggested_stocks.html` template with the new dual-model layout.

I can create the complete new template if you want.

## Testing

After updating:

1. **Restart Flask server** (if running)
2. **Clear browser cache** (Ctrl+Shift+R or Cmd+Shift+R)
3. **Login and navigate to Suggested Stocks page**
4. **You should see:**
   - Traditional Model section (left)
   - Raw LSTM Model section (right)
   - Each with Default Risk and High Risk tabs

## Current Data Status

Based on database check:
- ✅ Traditional + Default Risk: 10 stocks
- ❌ Traditional + High Risk: 0 stocks (need to run)
- ✅ Raw LSTM + Default Risk: 5 stocks
- ❌ Raw LSTM + High Risk: 0 stocks (need to run)

## Need Help?

Let me know if you want me to:
1. Create the complete new template
2. Create just the JavaScript changes
3. Create a migration guide

The new endpoint `/api/suggested-stocks/dual-model-view` is ready and working - you just need to update the frontend to use it!
