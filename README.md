# NSE Stock Trading System

**Automated stock trading system for NSE (National Stock Exchange) using pure technical analysis.**

---

## Quick Start

```bash
# 1. Start all services
./run.sh dev

# 2. Access web interface
http://localhost:5001

# 3. Check system status
./tools/check_all_schedulers.sh
```

**First run takes ~30 minutes** to collect data for 2,259 NSE stocks. After that, it runs fully automated with zero manual intervention!

---

## Table of Contents

- [System Overview](#system-overview)
- [How It Works](#how-it-works)
- [Hybrid Strategy Explained](#hybrid-strategy-explained)
- [Daily Automation Schedule](#daily-automation-schedule)
- [How Schedulers Work](#how-schedulers-work)
- [How to Trade](#how-to-trade)
- [Installation](#installation)
- [System Commands](#system-commands)
- [Database Tables](#database-tables)
- [Troubleshooting](#troubleshooting)

---

## System Overview

This is a **100% automated trading system** that:

- Collects data for **2,259+ NSE stocks** daily
- Analyzes using **5 technical indicators** (Hybrid Strategy)
- Generates **daily stock picks** with buy/sell signals
- Places **automatic orders** with stop-loss and targets
- Tracks **performance** and manages positions
- Runs **two strategies**: DEFAULT_RISK (conservative) and HIGH_RISK (aggressive)

**Key Features:**
- Zero manual intervention required
- Pure technical analysis (no fundamental analysis)
- Multi-broker support (Fyers, Zerodha)
- Paper trading simulator included
- Complete automation with saga pattern
- Self-healing with retry logic

---

## How It Works

### The Daily Cycle

```
Morning (Before Market Open):
└─ 06:00 AM → Update NSE symbols (Monday only)

Market Hours:
└─ 09:20 AM → Auto-trading places orders (if enabled)

Evening (After Market Close):
├─ 06:00 PM → Track performance, update P&L
├─ 09:00 PM → Data pipeline (fetch prices, history)
├─ 09:30 PM → Fill missing data + Calculate metrics
├─ 10:00 PM → Calculate technical indicators (Hybrid Strategy)
├─ 10:15 PM → Generate daily stock picks (50 per strategy)
└─ 10:00 PM → Export CSV files (parallel)

Night:
└─ 03:00 AM → Cleanup old data (Sunday only)
```

### Architecture

```
┌─────────────────────────────────────────┐
│         Docker Services                  │
├─────────────────────────────────────────┤
│                                          │
│  trading_system (Flask - Port 5001)    │
│  ├─ Web Interface                       │
│  ├─ REST API                            │
│  └─ Broker Integration                  │
│                                          │
│  ml_scheduler (Technical Indicators)    │
│  ├─ 10:00 PM: Calculate indicators     │
│  ├─ 10:15 PM: Generate stock picks     │
│  └─ 03:00 AM: Cleanup (Sunday)         │
│                                          │
│  data_scheduler (Data Pipeline)         │
│  ├─ 06:00 AM: Symbol update (Monday)   │
│  ├─ 09:00 PM: Data pipeline (daily)    │
│  └─ 09:30 PM: Fill data + Metrics      │
│                                          │
│  database (PostgreSQL 15)               │
│  └─ 11 tables with ~1.6M records       │
│                                          │
│  dragonfly (Redis Cache)                │
│  └─ API response caching                │
└─────────────────────────────────────────┘
```

---

## Hybrid Strategy Explained

The system uses a **hybrid technical strategy** combining 5 indicators with specific roles:

### The Complete Flow

```
STEP 1: FILTER (Remove weak stocks)
├─ RS Rating > 70 ✓ (Top 30% momentum)
├─ Price > EMA8 > EMA21 ✓ (Confirmed uptrend)
└─ Basic liquidity ✓

STEP 2: RANK (Order by strength)
└─ EMA Trend Score (70-100 for strong uptrends)

STEP 3: SIGNAL (When to buy)
├─ Wave crossover: Delta > 0 (REQUIRED)
├─ EMA confirmation: Price > EMA8 > EMA21 (REQUIRED)
└─ DeMarker < 0.30 (OPTIONAL for "high" quality)

STEP 4: TARGET (Where to exit)
├─ Target 1: Fibonacci 127.2%
├─ Target 2: Fibonacci 161.8%
└─ Target 3: Fibonacci 200%
```

### 1. RS Rating (Relative Strength) - **FILTER**

**Purpose:** Filter out weak momentum stocks

**How it works:**
- Compares each stock's performance vs NIFTY 50 benchmark
- Scale: 1-99 (percentile ranking)
- **Filter threshold: RS Rating > 70** (top 30% only)

**Example:**
- RS 95 = Stock beat 95% of all stocks ✓ PASS
- RS 65 = Stock is below average ✗ REJECTED

**NOT used for scoring** - Only for filtering!

### 2. Wave Indicators - **BUY/SELL SIGNALS**

**Purpose:** Tell you WHEN to enter or exit

**Components:**
- **Fast Wave**: 9-day momentum (short-term)
- **Slow Wave**: 21-day trend (long-term)
- **Delta**: Fast Wave - Slow Wave

**Signals:**
- **BUY**: Delta > 0 (fast crossed above slow)
- **SELL**: Delta < 0 (fast crossed below slow)

**Example:**
```
Day 1: Delta = -0.002 (no signal)
Day 2: Delta = +0.001 (BUY SIGNAL! Fast crossed above slow)
Day 3: Delta = +0.005 (still bullish, hold)
```

**NOT used for scoring** - Only for signals!

### 3. 8-21 EMA Strategy - **RANKING & CONFIRMATION**

**Purpose:** Primary ranking metric + trend confirmation

**Components:**
- **EMA 8**: Short-term momentum average
- **EMA 21**: Institutional holding period
- **Power Zone**: Price > EMA 8 > EMA 21 = Strong uptrend

**How it works:**
1. **Filter**: Price must be > EMA8 > EMA21 (uptrend required)
2. **Rank**: Stronger separation = Higher score
   - Strong Bull: 70-100 points (wide separation)
   - Early Bull: 60-70 points (just crossed up)
   - Weak/Bear: < 60 points (downtrend or flat)

**Example:**
```
Stock A: Price ₹100, EMA8 ₹98, EMA21 ₹95 → Score: 85 (strong uptrend)
Stock B: Price ₹100, EMA8 ₹99, EMA21 ₹98 → Score: 72 (early uptrend)
Stock C: Price ₹100, EMA8 ₹101, EMA21 ₹99 → REJECTED (downtrend)
```

**This is the ONLY metric used for ranking!**

### 4. DeMarker Oscillator - **ENTRY TIMING**

**Purpose:** Find the best moment to enter within an uptrend

**How it works:**
- Range: 0.0 to 1.0
- **< 0.30**: Oversold (good entry point)
- **0.30-0.70**: Neutral
- **> 0.70**: Overbought (avoid entry)

**Signal Quality:**
- **HIGH**: Wave + EMA + DeMarker < 0.30 (perfect timing)
- **MEDIUM**: Wave + EMA (good timing)
- **LOW**: Wave only (risky timing)

**Example:**
- DeMarker = 0.25 → "Oversold, great entry!" (HIGH quality)
- DeMarker = 0.50 → "Neutral, okay entry" (MEDIUM quality)
- DeMarker = 0.75 → "Overbought, wait!" (avoid)

### 5. Fibonacci Extension Targets - **EXIT STRATEGY**

**Purpose:** Dynamic profit targets based on price structure

**Targets:**
- **Target 1 (127.2%)**: Conservative exit (book 30% profit)
- **Target 2 (161.8%)**: Golden ratio (book 40% more)
- **Target 3 (200%)**: Aggressive target (let remainder ride)

**How it's calculated:**
1. Find recent swing low (bottom)
2. Find recent swing high (top)
3. Calculate extensions above current price

**Example:**
```
Swing Low: ₹90
Swing High: ₹110
Current: ₹100

Target 1: ₹115 (127.2% extension)
Target 2: ₹122 (161.8% extension)
Target 3: ₹130 (200% extension)
```

### The Ranking Formula

**Ranking Score = EMA Trend Score (70-100)**

That's it! No weighted composite. Just pure EMA trend strength.

### Buy Signal Logic

**BUY Signal Requirements:**

**HIGH Quality (3/3 conditions):**
1. Wave signal: Delta > 0 ✓ (REQUIRED)
2. EMA confirmation: Price > EMA8 > EMA21 ✓ (REQUIRED)
3. DeMarker timing: < 0.30 ✓ (perfect entry)

**MEDIUM Quality (2/3 conditions):**
1. Wave signal: Delta > 0 ✓ (REQUIRED)
2. EMA confirmation: Price > EMA8 > EMA21 ✓ (REQUIRED)
3. DeMarker timing: > 0.30 (not optimal, but okay)

**LOW Quality (1/3 conditions):**
1. Wave signal: Delta > 0 ✓ (REQUIRED)
2. EMA: No uptrend ✗ (risky!)
3. DeMarker: Not oversold ✗

**NONE:**
- No wave signal (Delta ≤ 0) = No buy

### Sell Signal Logic

**SELL Signal:**
- Wave turns negative: Delta < 0
- Downtrend confirmed: Price < EMA8 < EMA21

**Stop Loss:**
- Placed below EMA 21 (dynamic)
- Protects against trend reversal
- Typically 5-10% below entry

---

## Daily Automation Schedule

### Morning Tasks

**06:00 AM - Symbol Master Update (Monday Only)**
- Fetches ~2,259 NSE symbols from Fyers API
- Updates symbol_master table
- Duration: 1-2 minutes

**09:20 AM - Auto-Trading Execution (Daily, if enabled)**
- Checks today's suggested stocks
- Applies weekly trade limits
- Places orders with stop-loss and targets
- Duration: 2-3 minutes
- **Configurable**: Enable/disable per user

### Evening Tasks (After Market Close)

**06:00 PM - Performance Tracking**
- Updates order performance
- Creates daily P&L snapshots
- Checks stop-loss and target hits
- Closes orders automatically if targets met
- Duration: 1-2 minutes

**09:00 PM - Data Pipeline (6-step saga)**
1. Fetch current prices for all stocks
2. Download 1-year historical OHLCV data
3. Store data in database
4. Validate data quality
- Duration: 20-30 minutes
- **Records processed:** 2,259 stocks, ~820K history records

**09:30 PM - Data Processing (Parallel Tasks)**
- Fill missing adjusted_close prices
- Calculate ATR (Average True Range)
- Calculate volatility metrics
- Calculate liquidity scores
- Compute fundamental ratios
- Duration: 5-10 minutes

**10:00 PM - Technical Indicators Calculation**
- Calculate RS Rating for all stocks
- Calculate Wave indicators (Fast, Slow, Delta)
- Calculate 8-21 EMA values
- Calculate DeMarker oscillator
- Calculate Fibonacci targets
- Compute hybrid composite scores
- Generate buy/sell signals
- Duration: 3-5 minutes

**10:15 PM - Daily Stock Selection**
- Filter stocks by strategy criteria
- Rank by hybrid composite score
- Select top 50 for DEFAULT_RISK strategy
- Select top 50 for HIGH_RISK strategy
- Store in daily_suggested_stocks table
- Duration: 2-3 minutes
- **Result:** 100 stocks ready for next day

**10:00 PM - CSV Export (Parallel)**
- Export stocks.csv
- Export historical_data.csv
- Export technical_indicators.csv
- Export suggested_stocks.csv
- Duration: 2-3 minutes

### Night Tasks

**03:00 AM - Cleanup (Sunday Only)**
- Remove snapshots older than 90 days
- Remove CSV exports older than 30 days
- Duration: < 1 minute

**Total automation time per day: ~45-60 minutes**
**Manual intervention required: ZERO!**

---

## How Schedulers Work

The system runs **two independent schedulers** as Docker containers:

### 1. Data Scheduler (`data_scheduler.py`)

**Purpose:** Collects and processes market data

**Container:** `trading_system_data_scheduler`

**Schedule:**
```python
schedule.every().monday.at("06:00").do(update_symbol_master)  # Weekly
schedule.every().day.at("21:00").do(run_data_pipeline)        # 9 PM
schedule.every().day.at("21:30").do(fill_missing_data)        # 9:30 PM (parallel)
schedule.every().day.at("21:30").do(calculate_business_logic) # 9:30 PM (parallel)
schedule.every().day.at("22:00").do(export_csv_files)         # 10 PM (parallel)
schedule.every().day.at("22:00").do(validate_data_quality)    # 10 PM (parallel)
```

**Data Pipeline Steps (Saga Pattern):**
1. **SYMBOL_MASTER**: Update NSE symbols from Fyers
2. **STOCKS**: Create/update stock records with prices
3. **HISTORICAL_DATA**: Download 1-year OHLCV data
4. **TECHNICAL_INDICATORS**: Calculate base indicators
5. **COMPREHENSIVE_METRICS**: Calculate volatility, ratios
6. **PIPELINE_VALIDATION**: Verify data quality

**Retry Logic:**
- Each step retries up to 3 times on failure
- 60-second delay between retries
- Tracks failures in pipeline_tracking table
- Stops after 10 consecutive failures

**Logs:** `logs/data_scheduler.log`

### 2. Technical Indicators Scheduler (`scheduler.py`)

**Purpose:** Calculates hybrid strategy indicators and generates stock picks

**Container:** `trading_system_ml_scheduler`

**Schedule:**
```python
schedule.every().day.at("22:00").do(calculate_technical_indicators)  # 10 PM
schedule.every().day.at("22:15").do(generate_daily_snapshot)         # 10:15 PM
schedule.every().day.at("09:20").do(execute_auto_trading)            # 9:20 AM
schedule.every().day.at("18:00").do(track_order_performance)         # 6 PM
schedule.every().sunday.at("03:00").do(cleanup_old_snapshots)        # 3 AM Sunday
```

**Technical Indicators Calculation:**
1. Fetch historical data for all stocks (252 days)
2. Calculate RS Rating (vs NIFTY 50 benchmark)
3. Calculate Wave indicators (Fast, Slow, Delta)
4. Calculate 8-21 EMA values and trend scores
5. Calculate DeMarker oscillator
6. Calculate Fibonacci extension targets
7. Compute hybrid composite scores
8. Generate buy/sell signals with quality ratings

**Stock Selection Process:**
1. **Stage 1 - Market Data Filters:**
   - Price: ₹50 - ₹10,000
   - Minimum volume: 50,000 shares/day
   - Minimum turnover: ₹5 crore/day

2. **Stage 2 - Strategy-Specific Filters:**

   **DEFAULT_RISK (Conservative):**
   - Market cap: > ₹20,000 crore (large-cap)
   - PE ratio: 5-40
   - Good liquidity
   - Target gain: 7%, Stop loss: 5%

   **HIGH_RISK (Aggressive):**
   - Market cap: ₹1,000-20,000 crore (small/mid-cap)
   - Broader PE range
   - Medium liquidity
   - Target gain: 12%, Stop loss: 10%

3. **Stage 3 - Technical Scoring:**
   - Calculate hybrid composite score for each stock
   - Rank by score (highest = best opportunity)
   - Select top 50 stocks per strategy

4. **Stage 4 - Save Results:**
   - Store in daily_suggested_stocks table
   - Include all indicator values and signals
   - Ready for API queries and trading

**Logs:** `logs/scheduler.log`

### Monitoring Schedulers

```bash
# Check both schedulers
./tools/check_all_schedulers.sh

# View data scheduler logs
docker compose logs -f data_scheduler

# View technical indicators scheduler logs
docker compose logs -f ml_scheduler

# Check if schedulers are running
docker compose ps

# Restart schedulers if needed
docker compose restart data_scheduler ml_scheduler
```

---

## How to Trade

The system supports **three trading modes**:

### 1. Automatic Trading

**Best for:** Set-it-and-forget-it investors

**How it works:**
- System automatically places orders at 9:20 AM daily
- Uses today's suggested stocks (generated at 10:15 PM previous night)
- Applies buy signals with "high" quality only
- Sets stop-loss and target prices automatically
- Tracks performance and closes orders when targets hit
- Respects weekly trade limits (configurable per user)

**Setup:**
1. Configure broker API credentials (Fyers/Zerodha)
2. Enable auto-trading in settings
3. Set weekly trade limit (e.g., 5 trades/week)
4. Set capital allocation per trade
5. System handles the rest!

**Configuration:**
```sql
-- Enable auto-trading for your user
INSERT INTO auto_trading_settings (user_id, enabled, weekly_trade_limit, capital_per_trade)
VALUES (1, TRUE, 5, 10000.00);
```

**Access:** `http://localhost:5001/settings/auto-trading`

### 2. Manual Trading (Recommended for beginners)

**Best for:** Traders who want control and oversight

**How it works:**
1. View today's suggested stocks on dashboard
2. Review hybrid scores, indicators, and signals
3. Check Fibonacci targets and stop-loss levels
4. Manually place orders through broker platform
5. Track performance in system (optional)

**Daily workflow:**
```bash
# 1. Check system status
./tools/check_all_schedulers.sh

# 2. View suggested stocks via API
curl "http://localhost:5001/api/suggested-stocks/?strategy=default_risk&limit=10"

# 3. Or access web interface
http://localhost:5001/dashboard

# 4. Review indicators:
#    - Hybrid Score (0-100): Higher = better opportunity
#    - Signal Quality (high/medium/low): Only trade "high" quality
#    - RS Rating (1-99): Prefer > 80
#    - Wave Delta: Positive = momentum
#    - DeMarker: < 0.30 = oversold (good entry)
#    - Fibonacci Targets: Set your profit targets

# 5. Place orders on your broker platform
#    - Entry: Current price or near EMA 8
#    - Stop Loss: Below EMA 21 (shown in data)
#    - Target 1: Fibonacci 127.2%
#    - Target 2: Fibonacci 161.8%
#    - Target 3: Fibonacci 200%
```

**Dashboard Access:** `http://localhost:5001/dashboard`

### 3. Paper Trading (Simulator)

**Best for:** Testing strategies without risking real money

**How it works:**
- Simulates order placement and execution
- Uses real market data
- Tracks P&L as if trading real money
- No risk, perfect for learning
- Can run alongside automatic or manual trading

**Setup:**
```sql
-- Add simulator broker configuration
INSERT INTO broker_configurations (user_id, broker_name, is_default)
VALUES (1, 'simulator', TRUE);
```

**Usage:**
1. Enable simulator broker in settings
2. Set virtual capital (e.g., ₹100,000)
3. System simulates all trades
4. Review performance in dashboard
5. Adjust strategy based on results
6. Switch to real broker when confident

**Access:** `http://localhost:5001/settings/broker`

### Strategy Comparison

| Feature | DEFAULT_RISK | HIGH_RISK |
|---------|--------------|-----------|
| Market Cap | > ₹20,000 Cr | ₹1,000-20,000 Cr |
| Stock Type | Large-cap | Small/Mid-cap |
| PE Ratio | 5-40 | Flexible |
| Price Range | ₹100-10,000 | Flexible |
| Target Gain | 7% | 12% |
| Stop Loss | 5% | 10% |
| Risk Level | Low | High |
| Typical Stocks | RELIANCE, TCS, HDFC | Growth stocks |

### API Endpoints

**Get suggested stocks:**
```bash
# Conservative picks
curl "http://localhost:5001/api/suggested-stocks/?strategy=default_risk&limit=10"

# Aggressive picks
curl "http://localhost:5001/api/suggested-stocks/?strategy=high_risk&limit=20"

# Search by sector
curl "http://localhost:5001/api/suggested-stocks/?sector=Technology&limit=10"

# Search by symbol
curl "http://localhost:5001/api/suggested-stocks/?search=RELIANCE"
```

**Response format:**
```json
{
  "success": true,
  "count": 10,
  "strategy": "default_risk",
  "stocks": [
    {
      "symbol": "NSE:RELIANCE-EQ",
      "stock_name": "Reliance Industries Ltd",
      "rank": 1,
      "current_price": 2450.50,
      "market_cap": 16500000.0,

      "hybrid_composite_score": 85.4,
      "rs_rating": 92,
      "wave_momentum_score": 78.2,
      "ema_trend_score": 88.5,
      "demarker": 0.25,

      "ema_8": 2465.30,
      "ema_21": 2420.10,

      "fib_target_1": 2550.20,
      "fib_target_2": 2625.80,
      "fib_target_3": 2700.00,

      "buy_signal": true,
      "sell_signal": false,
      "signal_quality": "high"
    }
  ]
}
```

### Trading Best Practices

1. **Always check signal quality** - Only trade "high" quality signals
2. **Respect stop losses** - Exit immediately if price falls below stop loss
3. **Take partial profits** - Book profits at Fibonacci targets progressively
4. **Don't overtrade** - Stick to weekly limits (5-10 trades/week max)
5. **Review performance** - Check P&L daily, adjust strategy as needed
6. **Start with paper trading** - Test for 2-4 weeks before real money
7. **Diversify** - Don't put all capital in one stock
8. **Use both strategies** - Mix conservative and aggressive picks

---

## Installation

### Prerequisites

- **Docker & Docker Compose** (v20.10+)
- **Python 3.10+** (for local scripts)
- **Fyers API Credentials** (get from https://myapi.fyers.in)
- **4GB RAM minimum**, 8GB recommended
- **5GB free disk space**

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/StockExperiment.git
cd StockExperiment
```

### Step 2: Configure Environment

Create `.env` file:

```bash
# Fyers API Credentials (REQUIRED)
FYERS_CLIENT_ID=your_client_id_here
FYERS_ACCESS_TOKEN=your_access_token_here

# Database (Default - don't change)
DATABASE_URL=postgresql://trader:trader_password@database:5432/trading_system
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_password
POSTGRES_DB=trading_system

# Redis Cache
REDIS_HOST=dragonfly
REDIS_PORT=6379

# Application
FLASK_ENV=development
FLASK_DEBUG=1
LOG_LEVEL=INFO

# Optional: Adjust rate limits if needed
SCREENING_QUOTES_RATE_LIMIT_DELAY=0.2
VOLATILITY_MAX_WORKERS=5
```

### Step 3: Start Services

```bash
# Start all Docker containers
./run.sh dev

# Or use docker compose directly
docker compose up -d

# Monitor first-time setup (takes ~30 minutes)
docker compose logs -f
```

### Step 4: Verify Installation

```bash
# Check all services are running
docker compose ps

# Should show 5 containers:
# - trading_system_db (PostgreSQL)
# - trading_system_redis (Dragonfly)
# - trading_system_app (Flask)
# - trading_system_data_scheduler
# - trading_system_ml_scheduler

# Check system status
./tools/check_all_schedulers.sh

# Access web interface
http://localhost:5001
```

### Step 5: Wait for Initial Data Collection

On first run, system needs to:
1. Fetch NSE symbol list (~2,259 stocks)
2. Download 1-year historical data
3. Calculate technical indicators
4. Generate first daily picks

**This takes ~30 minutes.** After that, daily updates take 5-10 minutes.

### Step 6: Configure Broker (Optional)

For live trading:

```bash
# Access broker settings
http://localhost:5001/settings/broker

# Add Fyers credentials
# Or use simulator for testing
```

---

## System Commands

### Docker Operations

```bash
# Start services (development mode)
./run.sh dev

# Start services (production mode)
./run.sh prod

# Stop all services
docker compose down

# Restart specific service
docker compose restart trading_system
docker compose restart data_scheduler
docker compose restart ml_scheduler

# View logs (real-time)
docker compose logs -f                  # All services
docker compose logs -f trading_system   # Web app only
docker compose logs -f data_scheduler   # Data pipeline only
docker compose logs -f ml_scheduler     # Technical indicators only

# Check service status
docker compose ps

# Remove all data (WARNING: Deletes everything!)
docker compose down -v
```

### Database Access

```bash
# Connect to PostgreSQL
docker exec -it trading_system_db psql -U trader -d trading_system

# Run SQL query
docker exec -it trading_system_db psql -U trader -d trading_system -c "SELECT COUNT(*) FROM stocks;"

# Check database size
docker exec -it trading_system_db psql -U trader -d trading_system -c "SELECT pg_size_pretty(pg_database_size('trading_system'));"

# Export data
docker exec -it trading_system_db pg_dump -U trader trading_system > backup.sql
```

### Manual Tasks

```bash
# Run data pipeline manually
python3 run_pipeline.py

# Calculate technical indicators manually
docker exec trading_system python3 -c "
from src.models.database import get_database_manager
from src.services.technical.hybrid_strategy_calculator import get_hybrid_strategy_calculator

db_manager = get_database_manager()
with db_manager.get_session() as session:
    calculator = get_hybrid_strategy_calculator(session)
    result = calculator.calculate_all_indicators(['NSE:RELIANCE-EQ'], lookback_days=252)
    print(result)
"

# Fill missing data
python3 fill_data_sql.py

# Calculate business metrics
python3 fix_business_logic.py

# Export CSV files
python3 tools/export_csv.py
```

### System Health Checks

```bash
# Complete system status
./tools/check_all_schedulers.sh

# Check data pipeline status
cat logs/data_scheduler.log | grep -A 10 "Pipeline"

# Check technical indicators status
cat logs/scheduler.log | grep -A 5 "Technical Indicators"

# Check today's stock picks
docker exec -it trading_system_db psql -U trader -d trading_system -c "
SELECT COUNT(*), strategy
FROM daily_suggested_stocks
WHERE date = CURRENT_DATE
GROUP BY strategy;
"

# Check last pipeline run
docker exec -it trading_system_db psql -U trader -d trading_system -c "
SELECT * FROM pipeline_tracking
ORDER BY updated_at DESC
LIMIT 5;
"
```

---

## Database Tables

### Core Data Tables

**1. stocks** (~2,259 records)
- Current prices, market cap, volume
- Fundamental ratios: PE, PB, ROE, EPS
- Volatility: ATR, historical volatility
- Sector, industry classification

**2. historical_data** (~820,000 records)
- 1-year OHLCV data per stock
- Date, open, high, low, close, volume
- Adjusted close, turnover

**3. technical_indicators** (~820,000 records)
- RSI (14-day), MACD, SMA, EMA
- ATR, Bollinger Bands
- **Hybrid Strategy Fields:**
  - ema_8, ema_21 (8-21 EMA strategy)
  - demarker (oscillator)

**4. daily_suggested_stocks** (growing daily)
- Top 50 stocks per strategy per day
- Hybrid scores and indicators:
  - hybrid_composite_score (0-100)
  - rs_rating (1-99)
  - wave_momentum_score (0-100)
  - ema_trend_score (0-100)
  - demarker (0-1)
  - fib_target_1, fib_target_2, fib_target_3
  - buy_signal, sell_signal, signal_quality
- Strategy: DEFAULT_RISK or HIGH_RISK
- Rank: 1-50 per strategy

### Trading Tables

**5. users**
- User accounts and authentication

**6. broker_configurations**
- API credentials for brokers
- Fyers, Zerodha, or Simulator

**7. strategies**
- Trading strategy definitions

**8. orders**
- Order history with signals
- Entry price, stop loss, targets
- Order status, fill status

**9. trades**
- Executed trades
- Entry/exit prices, P&L

**10. positions**
- Current open positions
- Unrealized P&L

**11. auto_trading_settings**
- Per-user trading preferences
- Weekly limits, capital allocation

### System Tables

**12. pipeline_tracking**
- Saga step status
- Retry count, error messages
- Records processed

**13. symbol_master** (~2,259 records)
- Complete NSE symbol list

### Useful Queries

```sql
-- Today's top picks
SELECT symbol, stock_name, strategy, rank,
       hybrid_composite_score, signal_quality
FROM daily_suggested_stocks
WHERE date = CURRENT_DATE
ORDER BY strategy, rank
LIMIT 20;

-- High quality buy signals
SELECT symbol, stock_name,
       hybrid_composite_score, rs_rating,
       ema_trend_score, wave_momentum_score,
       fib_target_1, fib_target_2, fib_target_3
FROM daily_suggested_stocks
WHERE date = CURRENT_DATE
  AND buy_signal = TRUE
  AND signal_quality = 'high'
ORDER BY hybrid_composite_score DESC
LIMIT 10;

-- Pipeline status
SELECT step, status, records_processed, error_message
FROM pipeline_tracking
ORDER BY updated_at DESC
LIMIT 10;

-- Trading performance today
SELECT COUNT(*) as total_orders,
       SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as filled,
       SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected
FROM orders
WHERE DATE(created_at) = CURRENT_DATE;
```

---

## Troubleshooting

### Common Issues

#### 1. Pipeline Fails at HISTORICAL_DATA Step

**Symptoms:**
- Logs show "Rate limit exceeded" or timeouts
- Pipeline stuck in RETRYING status

**Solution:**
```bash
# Increase rate limit delay (default 0.2s → 0.5s)
export SCREENING_QUOTES_RATE_LIMIT_DELAY=0.5

# Reduce parallel workers (default 5 → 3)
export VOLATILITY_MAX_WORKERS=3

# Restart data scheduler
docker compose restart data_scheduler
```

#### 2. No Stock Picks Generated

**Symptoms:**
- daily_suggested_stocks table is empty
- Logs show "No stocks meet criteria"

**Check:**
```bash
# 1. Verify historical data exists
docker exec -it trading_system_db psql -U trader -d trading_system -c "
SELECT COUNT(*) FROM historical_data
WHERE date >= CURRENT_DATE - INTERVAL '252 days';
"

# 2. Check technical indicators calculated
docker exec -it trading_system_db psql -U trader -d trading_system -c "
SELECT COUNT(*) FROM technical_indicators
WHERE ema_8 IS NOT NULL AND ema_21 IS NOT NULL;
"

# 3. Run manual calculation
python3 test_hybrid_strategy.py
```

#### 3. Schedulers Not Running

**Symptoms:**
- Tasks not executing at scheduled times
- Logs show no activity

**Solution:**
```bash
# Check if schedulers are running
docker compose ps

# Restart schedulers
docker compose restart data_scheduler ml_scheduler

# Check logs for errors
docker compose logs data_scheduler | tail -100
docker compose logs ml_scheduler | tail -100

# Verify schedule configuration
docker exec ml_scheduler python3 -c "import schedule; print(schedule.jobs)"
```

#### 4. Database Connection Errors

**Symptoms:**
- Logs show "Connection refused" or "Could not connect"
- API returns 500 errors

**Solution:**
```bash
# Check database is running
docker compose ps database

# Check database logs
docker compose logs database | tail -50

# Restart database (WARNING: May cause data loss if not properly shut down)
docker compose restart database

# Wait 30 seconds for database to start
sleep 30

# Restart app and schedulers
docker compose restart trading_system data_scheduler ml_scheduler
```

#### 5. Fyers API Token Expired

**Symptoms:**
- Logs show "Invalid token" or "Unauthorized"
- Data pipeline fails at symbol fetch

**Solution:**
```bash
# 1. Get new access token from Fyers
#    Visit: https://myapi.fyers.in

# 2. Update .env file
nano .env
# Update FYERS_ACCESS_TOKEN=new_token_here

# 3. Restart services
docker compose restart trading_system data_scheduler
```

#### 6. Disk Space Full

**Symptoms:**
- Logs show "No space left on device"
- Database stops accepting writes

**Solution:**
```bash
# Check disk usage
df -h

# Clean old CSV exports
rm -rf exports/*.csv

# Clean old logs
rm -rf logs/*.log

# Remove old Docker images
docker system prune -a

# Restart services
docker compose up -d
```

#### 7. High Memory Usage

**Symptoms:**
- System slows down
- Docker containers crash with OOM errors

**Solution:**
```bash
# Check memory usage
docker stats

# Reduce parallel workers
export VOLATILITY_MAX_WORKERS=3

# Restart services with more memory
docker compose down
docker compose up -d

# Monitor memory
watch -n 5 docker stats
```

### Getting Help

1. **Check logs first:**
   ```bash
   docker compose logs -f | grep ERROR
   ```

2. **Check system status:**
   ```bash
   ./tools/check_all_schedulers.sh
   ```

3. **Review pipeline tracking:**
   ```bash
   docker exec -it trading_system_db psql -U trader -d trading_system -c "
   SELECT * FROM pipeline_tracking
   ORDER BY updated_at DESC
   LIMIT 10;
   "
   ```

4. **Test hybrid strategy:**
   ```bash
   python3 test_hybrid_strategy.py
   ```

5. **Contact support:**
   - Check CLAUDE.md for detailed technical documentation
   - Review init-scripts/01-init-db.sql for database schema
   - Check config/stock_filters.yaml for screening criteria

---

## Performance Metrics

**Daily Processing:**
- Data pipeline: 20-30 minutes
- Technical indicators: 3-5 minutes
- Stock selection: 2-3 minutes
- CSV export: 2-3 minutes
- **Total: ~30-45 minutes per day**

**Database Size:**
- Records: ~1.6 million
- Storage: ~500 MB
- Query speed: < 100ms (with indexes)

**API Performance:**
- Response time: < 100ms (with cache)
- Cache hit rate: > 90%
- Concurrent users: 50+

**System Requirements:**
- RAM: 4GB minimum, 8GB recommended
- Disk: 5GB minimum (10GB with logs)
- CPU: 2 cores minimum, 4 cores recommended

---

## Summary

✅ **100% Automated** - Zero manual intervention
✅ **2,259+ NSE Stocks** - Complete market coverage
✅ **Hybrid Strategy** - 5 technical indicators combined
✅ **Dual Strategies** - Conservative + Aggressive
✅ **Daily Stock Picks** - Top 50 per strategy
✅ **Multi-Broker Support** - Fyers, Zerodha, Simulator
✅ **Auto-Trading** - Automated order placement
✅ **Performance Tracking** - Daily P&L and analytics
✅ **Saga Pattern** - Reliable with retry logic
✅ **Production-Ready** - Docker, logging, monitoring

**Set it up once, let it run forever!** 🚀

---

## Support

**Documentation:**
- Technical details: CLAUDE.md
- Database schema: init-scripts/01-init-db.sql
- Screening criteria: config/stock_filters.yaml

**Commands:**
```bash
./run.sh dev                    # Start system
docker compose logs -f          # View logs
./tools/check_all_schedulers.sh # Check status
python3 test_hybrid_strategy.py # Test strategy
```

**Web Interface:**
- Dashboard: http://localhost:5001
- Settings: http://localhost:5001/settings
- Broker: http://localhost:5001/settings/broker

---

**Built for automated technical analysis trading** ❤️

Last Updated: October 29, 2025
