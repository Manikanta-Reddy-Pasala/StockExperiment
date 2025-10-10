# Dual Model View API - Complete Guide

## Overview

The dual model view API endpoint provides a comprehensive view of stock suggestions from **BOTH machine learning models** (Traditional and Raw LSTM) and **BOTH risk levels** (Default Risk and High Risk).

**Endpoint:** `GET /api/suggested-stocks/dual-model-view`

---

## What You'll See

The API returns 4 separate lists of suggested stocks:

1. **Traditional Model + Default Risk** - Conservative stocks from feature-engineered ensemble
2. **Traditional Model + High Risk** - Aggressive stocks from feature-engineered ensemble
3. **Raw LSTM Model + Default Risk** - Conservative stocks from raw OHLCV LSTM
4. **Raw LSTM Model + High Risk** - Aggressive stocks from raw OHLCV LSTM

---

## API Request

### URL
```
GET http://localhost:5001/api/suggested-stocks/dual-model-view
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 10 | Number of stocks to return per model/risk combination |
| `date` | string (YYYY-MM-DD) | today | Date to fetch suggestions for |

### Authentication
Requires login (`@login_required` decorator)

---

## API Response Format

```json
{
  "success": true,
  "date": "2025-10-10",
  "limit_per_group": 10,
  "total_stocks": 20,
  "data": {
    "traditional": {
      "default_risk": [
        {
          "symbol": "NSE:RELIANCE-EQ",
          "stock_name": "Reliance Industries",
          "current_price": 2450.50,
          "model_type": "traditional",
          "strategy": "default_risk",
          "ml_prediction_score": 0.7543,
          "ml_price_target": 2621.04,
          "ml_confidence": 0.8234,
          "ml_risk_score": 0.2156,
          "recommendation": "BUY",
          "target_price": 2621.04,
          "stop_loss": 2327.98,
          "rank": 1,
          "pe_ratio": 24.5,
          "pb_ratio": 3.2,
          "roe": 15.6,
          "market_cap": 165000.0,
          "sector": "Energy",
          "market_cap_category": "Large Cap"
        }
        // ... more stocks
      ],
      "high_risk": [
        // ... stocks for high risk strategy
      ]
    },
    "raw_lstm": {
      "default_risk": [
        {
          "symbol": "NSE:HONAUT-EQ",
          "stock_name": "Honeywell Automation",
          "current_price": 38923.90,
          "model_type": "raw_lstm",
          "strategy": "raw_lstm",
          "ml_prediction_score": 0.3446,
          "ml_price_target": 38923.90,
          "ml_confidence": 0.3446,
          "ml_risk_score": 0.3323,
          "recommendation": "HOLD",
          "target_price": 38923.90,
          "stop_loss": null,
          "rank": 1,
          "pe_ratio": 45.2,
          "pb_ratio": 12.5,
          "roe": 18.3,
          "market_cap": 35650.0,
          "sector": "Industrials",
          "market_cap_category": "Mid Cap"
        }
        // ... more stocks
      ],
      "high_risk": [
        // ... stocks for high risk strategy
      ]
    }
  },
  "statistics": {
    "traditional": {
      "default_risk": {
        "count": 10,
        "avg_score": 0.6234,
        "avg_confidence": 0.7543
      },
      "high_risk": {
        "count": 10,
        "avg_score": 0.5987,
        "avg_confidence": 0.6892
      }
    },
    "raw_lstm": {
      "default_risk": {
        "count": 5,
        "avg_score": 0.2680,
        "avg_confidence": 0.4413
      },
      "high_risk": {
        "count": 0,
        "avg_score": 0,
        "avg_confidence": 0
      }
    }
  }
}
```

---

## Usage Examples

### Example 1: Fetch Top 5 Stocks Per Group

```bash
curl -X GET \
  'http://localhost:5001/api/suggested-stocks/dual-model-view?limit=5' \
  -H 'Cookie: session=your_session_cookie'
```

### Example 2: Fetch Specific Date

```bash
curl -X GET \
  'http://localhost:5001/api/suggested-stocks/dual-model-view?date=2025-10-09&limit=10' \
  -H 'Cookie: session=your_session_cookie'
```

### Example 3: JavaScript/Fetch

```javascript
fetch('/api/suggested-stocks/dual-model-view?limit=10')
  .then(response => response.json())
  .then(data => {
    console.log('Traditional Default Risk:', data.data.traditional.default_risk);
    console.log('Traditional High Risk:', data.data.traditional.high_risk);
    console.log('Raw LSTM Default Risk:', data.data.raw_lstm.default_risk);
    console.log('Raw LSTM High Risk:', data.data.raw_lstm.high_risk);
    console.log('Statistics:', data.statistics);
  });
```

---

## Frontend Integration

### React/Vue Component Example

```javascript
// Fetch dual model view
const fetchDualModelView = async () => {
  try {
    const response = await fetch('/api/suggested-stocks/dual-model-view?limit=10');
    const data = await response.json();

    if (data.success) {
      // Display in 4 separate sections
      displaySection('Traditional Model - Low Risk', data.data.traditional.default_risk);
      displaySection('Traditional Model - High Risk', data.data.traditional.high_risk);
      displaySection('Raw LSTM Model - Low Risk', data.data.raw_lstm.default_risk);
      displaySection('Raw LSTM Model - High Risk', data.data.raw_lstm.high_risk);

      // Show statistics
      console.log('Stats:', data.statistics);
    }
  } catch (error) {
    console.error('Error:', error);
  }
};

// Display a section
const displaySection = (title, stocks) => {
  console.log(`\n=== ${title} ===`);
  console.log(`Count: ${stocks.length}`);

  stocks.forEach((stock, index) => {
    console.log(`${index + 1}. ${stock.symbol} - ${stock.stock_name}`);
    console.log(`   Score: ${stock.ml_prediction_score}, Target: ₹${stock.ml_price_target}`);
    console.log(`   Recommendation: ${stock.recommendation}`);
  });
};
```

### HTML Table View

```html
<div class="dual-model-view">
  <!-- Traditional Model Section -->
  <div class="model-section">
    <h2>Traditional Model (Feature Engineered)</h2>

    <div class="risk-sections">
      <!-- Default Risk -->
      <div class="risk-section">
        <h3>Default Risk (Conservative)</h3>
        <table id="traditional-default-risk">
          <!-- Table will be populated by JavaScript -->
        </table>
      </div>

      <!-- High Risk -->
      <div class="risk-section">
        <h3>High Risk (Aggressive)</h3>
        <table id="traditional-high-risk">
          <!-- Table will be populated by JavaScript -->
        </table>
      </div>
    </div>
  </div>

  <!-- Raw LSTM Model Section -->
  <div class="model-section">
    <h2>Raw LSTM Model (Research-Based)</h2>

    <div class="risk-sections">
      <!-- Default Risk -->
      <div class="risk-section">
        <h3>Default Risk (Conservative)</h3>
        <table id="raw-lstm-default-risk">
          <!-- Table will be populated by JavaScript -->
        </table>
      </div>

      <!-- High Risk -->
      <div class="risk-section">
        <h3>High Risk (Aggressive)</h3>
        <table id="raw-lstm-high-risk">
          <!-- Table will be populated by JavaScript -->
        </table>
      </div>
    </div>
  </div>
</div>
```

---

## Data Fields

Each stock in the response contains:

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Stock symbol (e.g., "NSE:RELIANCE-EQ") |
| `stock_name` | string | Company name |
| `current_price` | float | Current market price |
| `model_type` | string | "traditional" or "raw_lstm" |
| `strategy` | string | Strategy name |
| `ml_prediction_score` | float | ML prediction score (0-1, higher is better) |
| `ml_price_target` | float | Predicted price target |
| `ml_confidence` | float | Model confidence (0-1) |
| `ml_risk_score` | float | Risk assessment (0-1, higher is riskier) |
| `recommendation` | string | "BUY", "HOLD", or "SELL" |
| `target_price` | float | Suggested target price |
| `stop_loss` | float | Suggested stop loss price |
| `rank` | integer | Ranking within the group (1 is best) |
| `pe_ratio` | float | Price-to-Earnings ratio |
| `pb_ratio` | float | Price-to-Book ratio |
| `roe` | float | Return on Equity (%) |
| `market_cap` | float | Market capitalization (crores) |
| `sector` | string | Sector name |
| `market_cap_category` | string | "Large Cap", "Mid Cap", or "Small Cap" |

---

## Statistics Object

The `statistics` section provides aggregate metrics for each model/risk combination:

```json
{
  "traditional": {
    "default_risk": {
      "count": 10,              // Number of stocks
      "avg_score": 0.6234,      // Average ML prediction score
      "avg_confidence": 0.7543  // Average confidence
    },
    "high_risk": {
      "count": 10,
      "avg_score": 0.5987,
      "avg_confidence": 0.6892
    }
  },
  "raw_lstm": {
    "default_risk": {
      "count": 5,
      "avg_score": 0.2680,
      "avg_confidence": 0.4413
    },
    "high_risk": {
      "count": 0,
      "avg_score": 0,
      "avg_confidence": 0
    }
  }
}
```

---

## UI Design Recommendations

### Layout

```
+------------------------------------------+------------------------------------------+
|     TRADITIONAL MODEL                    |        RAW LSTM MODEL                   |
|  (Feature Engineered Ensemble)           |    (Research-Based OHLCV)               |
+------------------------------------------+------------------------------------------+
| DEFAULT RISK      | HIGH RISK            | DEFAULT RISK      | HIGH RISK           |
| (Conservative)    | (Aggressive)         | (Conservative)    | (Aggressive)        |
+-------------------+----------------------+-------------------+---------------------+
| 1. RELIANCE       | 1. TATAPOWER        | 1. HONAUT         | (No stocks yet)     |
|    Score: 0.75    |    Score: 0.68      |    Score: 0.34    |                     |
|    Target: 2621   |    Target: 245      |    Target: 38924  |                     |
|    Rec: BUY       |    Rec: BUY         |    Rec: HOLD      |                     |
+-------------------+----------------------+-------------------+---------------------+
| 2. TCS            | 2. ZOMATO           | 2. PAGEIND        |                     |
|    ...            |    ...              |    ...            |                     |
+-------------------+----------------------+-------------------+---------------------+
```

### Color Coding

- **Traditional Model**: Blue theme
- **Raw LSTM Model**: Green theme
- **Default Risk**: Light background
- **High Risk**: Darker/warmer background

### Icons

- 🏛️ Traditional Model
- 🤖 Raw LSTM Model
- 🛡️ Default Risk (Conservative)
- ⚡ High Risk (Aggressive)

---

## Error Handling

### Possible Error Responses

```json
{
  "success": false,
  "error": "Invalid date format. Use YYYY-MM-DD"
}
```

```json
{
  "success": false,
  "error": "No predictions available for the specified date"
}
```

```json
{
  "success": false,
  "error": "Internal server error: Connection refused"
}
```

---

## Testing

### Quick Test

```bash
# Check if endpoint works
curl -X GET 'http://localhost:5001/api/suggested-stocks/dual-model-view?limit=2'
```

### Expected Behavior

1. **If both models have data**: Returns 4 populated lists
2. **If only traditional model has data**: Raw LSTM lists will be empty
3. **If only raw LSTM has data**: Traditional lists will be empty
4. **If no data**: All lists empty, counts = 0

---

## Notes

1. **Data Availability**: The endpoint will only return data for stocks that exist in the `daily_suggested_stocks` table for the specified date.

2. **Strategy Mapping**: The endpoint intelligently maps strategy names:
   - "DEFAULT_RISK", "default_risk", "balanced" → Default Risk
   - "HIGH_RISK", "high_risk", "growth" → High Risk
   - "raw_lstm" (without risk suffix) → Default Risk

3. **Limit Application**: The `limit` parameter applies to EACH of the 4 groups independently. So `limit=10` returns up to 40 total stocks (10 per group).

4. **Missing Data**: If a particular model/risk combination has no stocks, that section will be empty (`[]`) with count=0 in statistics.

---

## Frontend Implementation Checklist

- [ ] Update suggested stocks page to call `/dual-model-view` endpoint
- [ ] Create 2x2 grid layout (2 models × 2 risk levels)
- [ ] Add model type badges/icons
- [ ] Add risk level indicators
- [ ] Display statistics for each group
- [ ] Implement model comparison features
- [ ] Add filters/sorting within each group
- [ ] Handle empty states gracefully
- [ ] Add loading states
- [ ] Implement error handling

---

**Status:** ✅ Endpoint implemented and ready to use
**Last Updated:** October 10, 2025
