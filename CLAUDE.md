# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **automated swing trading system** for NSE (National Stock Exchange) stocks using **pure 8-21 EMA technical analysis**. The system features:

**⏱️ Trading Style: SWING TRADING (2-4 week holding period)**
- NOT for day trading or scalping
- Positions held for 2-4 weeks (10-20 trading days)
- Target gains: 10-15% to Fibonacci targets
- Multi-day trend following with power zone methodology

- **100% automated data pipeline** with scheduled tasks
- **2,259+ NSE stocks** with fundamental and technical data
- **Pure 8-21 EMA Strategy**: Power Zone (8 EMA > 21 EMA) + DeMarker (timing) + Fibonacci (targets)
- **Two-strategy approach**: DEFAULT_RISK (conservative, large-cap) and HIGH_RISK (aggressive, small/mid-cap)
- **Saga pattern** for reliable multi-step data processing with failure tracking and retry logic
- **Multi-broker support**: Fyers API, Zerodha (future), with trading simulator
- **NO ML MODELS** - Simple, battle-tested technical indicators only

The system is containerized with Docker Compose and runs three main services: Flask web app, technical indicators scheduler, and data pipeline scheduler.

## Key Commands

### Development & Deployment

```bash
# Start all services (development mode)
./run.sh dev

# Start production mode
./run.sh prod

# Stop all services
docker compose down

# View logs (real-time)
docker compose logs -f                    # All services
docker compose logs -f trading_system      # Web app only
docker compose logs -f technical_scheduler  # Technical indicators scheduler only
docker compose logs -f data_scheduler      # Data pipeline only

# Restart specific service
docker compose restart trading_system
docker compose restart technical_scheduler
docker compose restart data_scheduler
```

### Database Access

```bash
# Connect to PostgreSQL
docker exec -it trading_system_db psql -U trader -d trading_system

# Run SQL queries
docker exec -it trading_system_db psql -U trader -d trading_system -c "SELECT COUNT(*) FROM stocks;"

# Check database size
docker exec -it trading_system_db psql -U trader -d trading_system -c "SELECT pg_size_pretty(pg_database_size('trading_system'));"
```

### Manual Task Execution

```bash
# Run data pipeline manually (6-step saga: symbols → stocks → history → indicators → metrics → validation)
python3 run_pipeline.py

# Calculate 8-21 EMA strategy indicators manually
docker exec trading_system python3 -c "
from src.models.database import get_database_manager
from src.services.technical.ema_strategy_calculator import get_ema_strategy_calculator

db_manager = get_database_manager()
with db_manager.get_session() as session:
    calculator = get_ema_strategy_calculator(session)
    symbols = ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ']  # Example symbols
    result = calculator.calculate_all_indicators(symbols, lookback_days=252)
    print(f'Calculated 8-21 EMA indicators for {len(result)} stocks')
    for symbol, indicators in result.items():
        print(f'{symbol}: Power Zone={indicators[\"power_zone_status\"]}, DeMarker={indicators[\"demarker\"]:.2f}')
"

# Fill missing data fields (adj_close, volatility, liquidity, etc.)
python3 fill_data_sql.py

# Calculate business logic metrics (EPS, ROE, margins, ratios, etc.)
python3 fix_business_logic.py

# Generate daily snapshot with technical indicators
python3 tools/generate_daily_snapshot.py
```

### Fyers Token Management

```bash
# Check token status in logs (auto-checked every 6 hours)
docker compose logs -f technical_scheduler | grep -i "token"  # Note: technical_scheduler is the container name

# Manual token refresh (when expired)
# 1. Navigate to http://localhost:5001/brokers/fyers
# 2. Click "Refresh Token" button
# 3. Login to Fyers and authorize
# 4. System will automatically save new token

# Token monitoring features:
# - Auto-refresh checks every 30 minutes (background thread)
# - Status check every 6 hours (scheduler task)
# - Warns when token expires within 12 hours
# - Logs expiry time and re-auth URL
```

### System Health Checks

```bash
# Check all schedulers status
./tools/check_all_schedulers.sh

# Check technical indicators scheduler
./tools/check_scheduler.sh

# View pipeline status
cat logs/data_scheduler.log | grep -A 10 "Pipeline"

# Check technical indicators calculation status
cat logs/scheduler.log | grep -A 5 "Technical Indicators"

# Check Fyers token status
cat logs/scheduler.log | grep -A 5 "Token Status"
```

## Architecture Overview

### Technology Stack

**Backend:**
- Flask 3.0 (web framework)
- SQLAlchemy 2.0 (ORM with PostgreSQL)
- PostgreSQL 15 (primary database)
- Dragonfly (Redis-compatible cache)
- pandas + pandas-ta (technical indicator calculations)
- Schedule (Python task scheduler)
- **NO ML libraries** - Pure technical analysis

**DevOps:**
- Docker Compose (multi-service orchestration)
- 5 containers: database, dragonfly, app, data_scheduler, technical_scheduler

### Core Services Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Docker Services                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  trading_system (Flask App - Port 5001)                       │
│  ├─ Web Interface (dashboard, admin panel)                     │
│  ├─ REST API endpoints                                         │
│  ├─ Multi-broker integration (Fyers, Zerodha)                 │
│  └─ Real-time data caching                                     │
│                                                                 │
│  technical_scheduler (Python scheduler.py - 8-21 EMA Strategy)│
│  ├─ Startup: Initialize token monitoring (auto-refresh)       │
│  ├─ Every 6 hours: Check Fyers token status (warn if expiring)│
│  ├─ 10:00 PM: Calculate 8-21 EMA indicators for all stocks   │
│  │   ├─ 8 & 21 EMA (power zone identification)               │
│  │   ├─ DeMarker Oscillator (pullback timing, 0-1 range)     │
│  │   └─ Fibonacci Extensions (127.2%, 161.8%, 200% targets)  │
│  ├─ 10:15 PM: Generate daily stock picks (DUAL STRATEGY)     │
│  │   ├─ DEFAULT_RISK strategy (conservative, large-cap)      │
│  │   └─ HIGH_RISK strategy (aggressive, small/mid-cap)       │
│  └─ 03:00 AM (Sunday): Cleanup old snapshots (>90 days)       │
│                                                                 │
│  data_scheduler (Python data_scheduler.py)                     │
│  ├─ 06:00 AM (Monday): Update symbol master (~2,259 symbols)  │
│  ├─ 09:00 PM: Data pipeline (6-step saga)                     │
│  ├─ 09:30 PM: Fill missing data + Business logic calc         │
│  ├─ 10:00 PM: CSV export + Data quality validation            │
│  └─ Uses saga pattern with retry logic and failure tracking   │
│                                                                 │
│  database (PostgreSQL 15 - Port 5432)                          │
│  ├─ 11 tables: stocks, historical_data, technical_indicators, │
│  │             daily_suggested_stocks, pipeline_tracking, etc. │
│  └─ ~1.6M records (~500MB)                                     │
│                                                                 │
│  dragonfly (Redis cache - Port 6379)                           │
│  └─ API response caching, session management                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Pipeline Flow (6-Step Saga)

The pipeline uses a **saga pattern** for reliability:

```
Step 1: SYMBOL_MASTER
├─ Fetch NSE symbols from Fyers API
├─ Store in symbol_master table (~2,259 symbols)
└─ Retry on failure with exponential backoff

Step 2: STOCKS
├─ Create/update stock records with basic data
├─ Fetch current prices and market cap from Fyers
└─ Skip fundamental data for speed (done in Step 5)

Step 3: HISTORICAL_DATA
├─ Download 1-year OHLCV data for each stock
├─ Use historical_data_service with smart logic:
│   ├─ Check last trading day (skip weekends)
│   ├─ Classify API responses (rate limit, timeout, success)
│   ├─ Create placeholder records for holidays/weekends
│   └─ Retry with exponential backoff
├─ Process in parallel with configurable workers
└─ Store ~820K records

Step 4: TECHNICAL_INDICATORS
├─ Calculate RSI (14-day), MACD, SMA (20/50/200), EMA (12/26)
├─ Calculate ATR (14-day), Bollinger Bands
└─ Store per-date technical indicators

Step 5: COMPREHENSIVE_METRICS
├─ Calculate volatility (annualized, using √252 days)
├─ Calculate/estimate fundamental ratios:
│   ├─ PE, PB, ROE, ROA, PEG ratios
│   ├─ Current/Quick ratios, Debt-to-Equity
│   ├─ Operating/Net/Profit margins
│   └─ Revenue/Earnings growth, Dividend yield
├─ Uses sector-based estimation when real data unavailable
└─ ⚠️ Estimated values flagged with data_source='estimated_enhanced'

Step 6: PIPELINE_VALIDATION
├─ Verify all tables populated
├─ Check data quality and consistency
├─ Log validation results
└─ Mark pipeline as completed
```

**Retry Logic:**
- Each step retries up to 3 times on failure
- 60-second delay between retries
- Stops after 10 consecutive failures to prevent infinite loops
- Tracks failures in `pipeline_tracking` table

### Dual Strategy System

The 8-21 EMA strategy runs with two risk profiles:

1. **DEFAULT_RISK (Conservative)**:
   - Target: Large-cap stocks (>20,000 Cr market cap)
   - Focus: Stability, established companies
   - Filtering: PE 5-40, price 100-10000, good liquidity
   - Signal Quality: HIGH or MEDIUM only (strict)
   - Target gain: 10-15% (Fibonacci targets)
   - Stop loss: Below 21 EMA (typically 8-10%)

2. **HIGH_RISK (Aggressive)**:
   - Target: Small/Mid-cap stocks (1,000-20,000 Cr)
   - Focus: Growth potential, momentum
   - Filtering: Broader criteria, accepts more volatility
   - Signal Quality: MEDIUM or LOW accepted
   - Target gain: 12-18% (extended Fibonacci targets)
   - Stop loss: Below 21 EMA (typically 10-12%)

**Result:** 2 daily stock lists (50 stocks per strategy)

## Critical Code Patterns

### 1. Saga Pattern Implementation

The codebase uses **saga pattern** for multi-step operations with failure tracking:

```python
# Example: Data pipeline saga (src/services/data/pipeline_saga.py)
class PipelineSaga:
    def execute_step_with_retry(self, step, step_function):
        """Execute a step with retry logic and failure tracking"""
        for attempt in range(self.max_retries + 1):
            # Update status to in_progress
            self.update_step_status(step, PipelineStatus.IN_PROGRESS)

            # Execute the step
            result = step_function()

            if result.get('success'):
                # Success - mark completed
                self.update_step_status(step, PipelineStatus.COMPLETED)
                return result
            else:
                # Failure - retry or mark failed
                if attempt < self.max_retries:
                    self.update_step_status(step, PipelineStatus.RETRYING)
                    time.sleep(self.retry_delay)
                else:
                    self.update_step_status(step, PipelineStatus.FAILED)
                    return result
```

**Key files:**
- `src/services/data/pipeline_saga.py` - Data pipeline saga (6 steps)
- `src/services/data/suggested_stocks_saga.py` - Stock selection saga (7 steps)

### 2. Database Session Management

Always use context managers for database sessions:

```python
from src.models.database import get_database_manager

db_manager = get_database_manager()
with db_manager.get_session() as session:
    # Your database operations here
    stock = session.query(Stock).filter(Stock.symbol == 'NSE:RELIANCE-EQ').first()
    session.commit()  # Explicit commit if needed
# Session automatically closed and cleaned up
```

**Never:**
- Create sessions directly without context manager
- Forget to commit transactions
- Keep sessions open longer than necessary

### 3. Technical Indicator Caching

The EMA strategy calculator efficiently calculates indicators in bulk:

```python
# Example from ema_strategy_calculator.py
class EMAStrategyCalculator:
    def __init__(self, session: Session):
        self.session = session

    def calculate_all_indicators(self, symbols: List[str], lookback_days: int = 252):
        """Calculate 8-21 EMA + DeMarker + Fibonacci for all stocks"""
        results = {}

        for symbol in symbols:
            # Get historical data
            stock_data = self._get_historical_data(symbol, lookback_days)

            # Calculate EMAs
            df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()

            # Calculate DeMarker
            demarker = self._calculate_demarker(df, period=14)

            # Calculate Fibonacci targets
            fib_targets = self._calculate_fibonacci_extensions(df)

            results[symbol] = {
                'ema_8': ema_8,
                'ema_21': ema_21,
                'demarker': demarker,
                'fib_targets': fib_targets,
                'power_zone_status': 'bullish' if price > ema_8 > ema_21 else 'bearish'
            }

        return results
```

**No Model Storage Required:** Pure mathematical indicators calculated on-demand from price data.

### 4. Last Trading Day Logic

Critical pattern for handling market holidays and weekends:

```python
def _get_last_trading_day(self) -> date:
    """Get the last expected trading day (skip weekends)"""
    today = datetime.now().date()

    # If Saturday/Sunday, go back to Friday
    if today.weekday() == 5:  # Saturday
        return today - timedelta(days=1)
    elif today.weekday() == 6:  # Sunday
        return today - timedelta(days=2)
    else:
        # Weekday - check market close time (3:30 PM IST)
        now = datetime.now()
        market_close = now.replace(hour=15, minute=30)

        if now >= market_close:
            return today  # Market closed, today is last trading day
        else:
            # Market still open, yesterday is last complete day
            return today - timedelta(days=1)
```

**Usage:** Used in historical data pipeline to determine which data to fetch.

### 5. Environment-Based Configuration

Configuration uses environment variables with sensible defaults:

```python
# From config.py
DATABASE_URL = os.environ.get('DATABASE_URL',
    'postgresql://trader:trader_password@database:5432/trading_system')

# From pipeline_saga.py
self.rate_limit_delay = float(os.getenv('SCREENING_QUOTES_RATE_LIMIT_DELAY', '0.2'))
self.max_workers = int(os.getenv('VOLATILITY_MAX_WORKERS', '5'))
self.max_stocks = int(os.getenv('VOLATILITY_MAX_STOCKS', '500'))
```

**Key environment variables:**
- `DATABASE_URL` - PostgreSQL connection string
- `FYERS_CLIENT_ID`, `FYERS_ACCESS_TOKEN` - Fyers API credentials
- `SCREENING_*` - Stock screening parameters
- `VOLATILITY_*` - Volatility calculation configuration

## Database Schema (11 Tables)

### Core Data Tables

1. **stocks** (~2,259 records)
   - Current prices, market cap, volume
   - Fundamental ratios: PE, PB, ROE, ROA, EPS, beta
   - Growth metrics: revenue_growth, earnings_growth
   - Profitability: operating_margin, net_margin, profit_margin
   - Liquidity: current_ratio, quick_ratio, debt_to_equity
   - Volatility: historical_volatility_1y, atr_14

2. **historical_data** (~820,000 records)
   - 1-year OHLCV data per stock
   - Calculated fields: price_change_pct, turnover, high_low_pct
   - Candlestick metrics: body_pct, upper_shadow_pct, lower_shadow_pct
   - Metadata: data_source, api_resolution, data_quality_score

3. **technical_indicators** (~820,000 records)
   - RSI (14-day), MACD (histogram, signal line)
   - SMA (20, 50, 200), EMA (12, 26)
   - ATR (14-day), Bollinger Bands (upper, middle, lower)

4. **daily_suggested_stocks** (~4,500 records, growing daily)
   - Top 50 stocks per strategy per day
   - Includes: strategy, rank, selection_score
   - ML predictions: ml_prediction_score, ml_price_target, ml_confidence, ml_risk_score
   - Trading signals: target_price, stop_loss, recommendation, reason
   - **UNIQUE constraint:** (date, symbol, strategy) - allows upsert

### Tracking & System Tables

5. **pipeline_tracking**
   - Saga step status, retry count, failure reasons
   - Records processed, timestamps

6. **symbol_master** (~2,259 records)
   - Complete NSE symbol list from Fyers

7. **users**, **broker_configurations**, **strategies**
   - Multi-user support, broker API credentials

8. **orders**, **trades**, **positions**
   - Order execution history, current positions

## Important File Locations

### Entry Points
- `app.py` - Flask app entry point
- `run.py` - Application launcher
- `scheduler.py` - Technical indicators scheduler (10 PM calculations, 10:15 PM snapshot, 3 AM cleanup)
- `data_scheduler.py` - Data pipeline scheduler (6 AM symbols, 9 PM pipeline, 9:30 PM calcs, 10 PM export)

### Core Services

**Data Pipeline:**
- `src/services/data/pipeline_saga.py` - 6-step saga for data collection
- `src/services/data/suggested_stocks_saga.py` - 7-step saga for stock selection (8-21 EMA strategy)
- `src/services/data/historical_data_service.py` - Smart historical data fetching
- `src/services/data/fyers_symbol_service.py` - Symbol master management

**Technical Analysis Services:**
- `src/services/technical/ema_strategy_calculator.py` - Pure 8-21 EMA strategy calculator
- `src/services/technical/hybrid_strategy_calculator.py` - (DEPRECATED) Old hybrid strategy
- `src/services/technical/indicators_calculator.py` - (DEPRECATED) Old indicators

**Broker Integration:**
- `src/services/brokers/fyers_service.py` - Fyers API wrapper (DEPRECATED, use unified)
- `src/services/core/unified_broker_service.py` - Multi-broker abstraction

### Database Models
- `src/models/models.py` - User, Strategy, Order, Trade, Position models
- `src/models/stock_models.py` - Stock, SymbolMaster models
- `src/models/historical_models.py` - HistoricalData, TechnicalIndicators models
- `src/models/database.py` - Database manager, session factory

### Configuration Files
- `config.py` - Application configuration (database, logging, etc.)
- `docker-compose.yml` - Docker services definition
- `requirements.txt` - Python dependencies

## Common Development Tasks

### Adding a New Pipeline Step

1. Define step in `PipelineStep` enum:
```python
class PipelineStep(Enum):
    NEW_STEP = 7  # Add after existing steps
```

2. Implement step function in `PipelineSaga`:
```python
def step_new_feature(self) -> Dict[str, Any]:
    """Step 7: New feature description."""
    try:
        # Implementation
        return {
            'success': True,
            'records_processed': count,
            'message': 'New feature completed'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
```

3. Add to pipeline execution in `run_pipeline`:
```python
steps = [
    # ... existing steps ...
    (PipelineStep.NEW_STEP, self.step_new_feature),
]
```

### Modifying Stock Screening Criteria

Stock screening uses `config/stock_filters.yaml`:

```yaml
stage_1_filters:  # Market data screening
  tradeability:
    minimum_price: 50.0
    maximum_price: 10000.0
    minimum_daily_volume: 50000
    minimum_daily_turnover_inr: 50000000

stage_2_filters:  # Business logic screening
  filtering_thresholds:
    minimum_total_score: 25  # Out of 100
  fundamental_ratios:
    max_pe_ratio: 50.0
    min_roe: 5.0
    max_debt_equity: 2.0
```

Modify these values to adjust screening strictness.

## Testing & Debugging

### Manual Testing

```bash
# Test data pipeline (all steps)
python3 run_pipeline.py

# Test EMA strategy indicators
python3 -c "
from src.models.database import get_database_manager
from src.services.technical.ema_strategy_calculator import get_ema_strategy_calculator

db_manager = get_database_manager()
with db_manager.get_session() as session:
    calc = get_ema_strategy_calculator(session)
    result = calc.calculate_all_indicators(['NSE:RELIANCE-EQ'], lookback_days=252)
    print(result)
"

# Test stock selection saga
python3 -c "
from src.services.data.suggested_stocks_saga import get_suggested_stocks_saga_orchestrator
orchestrator = get_suggested_stocks_saga_orchestrator()
result = orchestrator.execute_suggested_stocks_saga(
    user_id=1,
    strategies=['DEFAULT_RISK'],
    limit=10
)
print(result)
"
```

### Debugging Pipeline Issues

1. **Check pipeline tracking:**
```sql
SELECT * FROM pipeline_tracking ORDER BY updated_at DESC;
```

2. **Check logs:**
```bash
cat logs/data_scheduler.log | grep ERROR
cat logs/scheduler.log | grep ERROR
```

3. **Check data coverage:**
```sql
SELECT
    COUNT(*) as total_stocks,
    COUNT(DISTINCT h.symbol) as with_history,
    COUNT(DISTINCT ti.symbol) as with_indicators
FROM stocks s
LEFT JOIN historical_data h ON s.symbol = h.symbol
LEFT JOIN technical_indicators ti ON s.symbol = ti.symbol;
```

### Common Issues

1. **Pipeline fails at HISTORICAL_DATA step:**
   - Usually rate limiting from Fyers API
   - Check `SCREENING_QUOTES_RATE_LIMIT_DELAY` (default 0.2s, increase to 0.5s)
   - Reduce `VOLATILITY_MAX_WORKERS` (default 5, try 3)

2. **EMA indicator calculation fails:**
   - Check historical data exists: `SELECT COUNT(*) FROM historical_data;`
   - Ensure at least 60 days of data per stock (minimum for reliable EMAs)
   - Check database connectivity

3. **Scheduler not running:**
   - Check process: `docker compose ps`
   - Restart: `docker compose restart technical_scheduler data_scheduler`
   - Check logs for errors

## Performance Considerations

- **Data pipeline:** ~20-30 minutes for 2,259 stocks
- **EMA indicator calculation:** ~2-5 minutes for all stocks
- **API response time:** <100ms (with Dragonfly cache)
- **Database size:** ~1.6M records, ~500MB

**Optimization tips:**
1. Increase cache TTL for frequently accessed data
2. Add database indexes for slow queries
3. Batch process stocks instead of one-by-one
4. Adjust SQLAlchemy connection pool size
5. Use Redis caching for expensive operations

## Important Notes

1. **Estimated Fundamental Data:**
   - When real fundamental data unavailable, system generates sector-based estimates
   - These are flagged with `data_source='estimated_enhanced'`
   - Do NOT use for actual trading without verification

2. **Market Hours:**
   - NSE market: 9:15 AM - 3:30 PM IST (Monday-Friday)
   - Pipeline runs at 9:00 PM (after market close)
   - Symbol update runs Monday 6:00 AM

3. **Rate Limiting:**
   - Fyers API has rate limits
   - Configured via `SCREENING_QUOTES_RATE_LIMIT_DELAY` (default 0.2s)
   - Adjust if you see rate limit errors

4. **Saga Pattern:**
   - All multi-step operations use saga pattern
   - Steps can retry up to 3 times
   - Failures tracked in `pipeline_tracking` table
   - Use saga pattern for new multi-step features

5. **Docker Volumes:**
   - PostgreSQL data persisted in `postgres_data` volume
   - Logs mounted at `./logs`
   - Exports mounted at `./exports`

## References

- **API Documentation:** Check `src/services/brokers/fyers/api.py` for Fyers API methods
- **Database Schema:** See `init-scripts/01-init-db.sql` for complete DDL
- **EMA Strategy:** See `src/services/technical/ema_strategy_calculator.py` for implementation
- **Stock Filtering:** See `config/stock_filters.yaml` for screening criteria
