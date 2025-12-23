# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an **automated NSE stock trading system** using the **8-21 EMA swing trading strategy** (pure technical analysis, no ML/AI). The system analyzes 2,259+ NSE stocks daily, generates top 50 stock picks, and executes automated trades via broker APIs (Fyers).

**Trading Style**: Swing trading (5-14 day holds), targeting 7-12% gains with 5-7% stop losses.

## Core Architecture

### 5-Container Docker Setup

```
trading_system       - Flask web app (port 5001)
technical_scheduler  - Daily technical indicators & stock picks (10 PM)
data_scheduler       - Daily data pipeline (9 PM)
database            - PostgreSQL 15
dragonfly           - Redis-compatible cache
```

### Key Design Patterns

**1. Saga Pattern**: Fault-tolerant multi-step workflows with rollback capability
- `src/services/data/pipeline_saga.py` - Data collection pipeline (6 steps)
- `src/services/data/suggested_stocks_saga.py` - Stock selection pipeline (7 steps)
- Each saga tracks step status, retries on failure, and maintains transaction boundaries

**2. Database Manager Singleton**: Global connection pool with context manager
- `src/models/database.py` - `get_database_manager()` returns singleton instance
- Always use `with db_manager.get_session() as session:` for automatic commit/rollback

**3. Service Layer Pattern**: Business logic isolated from routes/models
- `src/services/core/` - Core services (broker, order, user)
- `src/services/data/` - Data pipelines (historical, technical, snapshots)
- `src/services/technical/` - Technical analysis (EMA calculator)
- `src/services/trading/` - Trading automation & performance tracking

**4. Strategy Pattern**: Interface-based broker abstraction
- `src/services/interfaces/` - Abstract interfaces (orders, portfolio, dashboard)
- `src/services/implementations/` - Broker-specific implementations (Fyers)
- `src/services/core/unified_broker_service.py` - Unified multi-broker facade

## 8-21 EMA Strategy Implementation

### Core Concept
Price > 8 EMA > 21 EMA = "Power Zone" (bullish trend)
DeMarker < 0.30 = Oversold pullback (entry timing)
Fibonacci 127.2%, 161.8%, 200% = Profit targets

### Key Files
- `src/services/technical/ema_strategy_calculator.py` - EMA, DeMarker, Fibonacci calculations
- `src/models/stock_models.py` - Stock model with `ema_8`, `ema_21`, `demarker`, `buy_signal`, `sell_signal`
- `scheduler.py:123-188` - `calculate_technical_indicators()` runs nightly at 10 PM

### Strategy Flow
1. Calculate 8 & 21 EMA from 252-day historical data
2. Calculate DeMarker oscillator (< 0.30 = oversold)
3. Generate buy signal: `is_bullish AND demarker < 0.30`
4. Calculate Fibonacci targets from recent swing low
5. Update `stocks` table with indicators and signals

## Database Schema

### 11 Core Tables (~1.6M records, ~500MB)

**Stock Data**:
- `stocks` - Current stock info (2,259 stocks) with EMA fields
- `historical_data` - 1-year OHLCV data (820K records)
- `technical_indicators` - Daily calculations (820K records)
- `daily_suggested_stocks` - Daily top 50 picks
- `symbol_master` - NSE symbol master list

**Trading**:
- `users`, `broker_configurations`, `orders`, `positions`, `trades`

**System**:
- `pipeline_tracking` - Saga execution status for data pipeline

### Important Schema Notes
- `stocks.ema_8`, `stocks.ema_21`, `stocks.demarker` - Strategy indicators (updated nightly)
- `stocks.buy_signal`, `stocks.sell_signal` - Boolean flags for trade signals
- `historical_data.date` - Trading dates only (no weekends/holidays)
- `daily_suggested_stocks.snapshot_date` - One snapshot per day (upserted)

## Daily Automation Schedule

**Data Pipeline** (9:00 PM):
1. Update symbol master (Monday only)
2. Fetch current quotes for all stocks
3. Fetch 1-year historical data
4. Calculate comprehensive metrics (volatility, beta, fundamentals)

**Technical Analysis** (10:00 PM):
1. Calculate 8-21 EMA indicators for all stocks
2. Generate buy/sell signals
3. Update `stocks` table with latest indicators

**Stock Selection** (10:15 PM):
1. Run unified 8-21 EMA saga (filters ~2,259 stocks to top 50)
2. Save to `daily_suggested_stocks` table

**Auto-Trading** (9:20 AM):
1. Execute orders for users with auto-trading enabled
2. Place market orders via broker API

**Performance Tracking** (6:00 PM):
1. Update order performance snapshots
2. Close positions hitting targets/stops

## Common Commands

### Development
```bash
# Start all services (development)
./run.sh dev

# Start all services (production)
./run.sh prod

# View logs
docker compose logs -f                          # All services
docker compose logs -f trading_system           # Web app only
docker compose logs -f technical_scheduler      # Indicators
docker compose logs -f data_scheduler           # Data pipeline

# Database access
docker exec -it trading_system_db psql -U trader -d trading_system

# Check scheduler status
docker compose ps
```

### Manual Operations
```bash
# Run data pipeline manually
python run_pipeline.py

# Generate daily stock picks manually
docker exec trading_system python tools/generate_daily_snapshot.py

# Calculate technical indicators manually
docker exec trading_system python -c "from scheduler import calculate_technical_indicators; calculate_technical_indicators()"
```

## Critical Implementation Details

### 1. Database Session Management
**ALWAYS** use context manager for sessions:
```python
from src.models.database import get_database_manager

db_manager = get_database_manager()
with db_manager.get_session() as session:
    # Your code here
    session.commit()  # Auto-committed on success, auto-rolled back on exception
```

### 2. Saga Pattern Usage
When implementing multi-step processes:
- Create saga with steps: `saga.add_step(SagaStep(...))`
- Update status: `saga.update_step_status(step_id, status, metadata)`
- Track failures: `saga.fail_saga(error_message)`
- Complete: `saga.complete_saga(results, summary)`

### 3. Broker Service Access
```python
from src.services.core.unified_broker_service import get_unified_broker_service

broker_service = get_unified_broker_service()
result = broker_service.get_quotes(user_id, ["NSE:RELIANCE-EQ"])
```

### 4. Technical Indicator Calculation
```python
from src.services.technical.ema_strategy_calculator import get_ema_strategy_calculator

with db_manager.get_session() as session:
    calculator = get_ema_strategy_calculator(session)
    indicators = calculator.calculate_all_indicators(['NSE:RELIANCE-EQ'], lookback_days=252)
    # Returns: {symbol: {ema_8, ema_21, demarker, buy_signal, sell_signal, ...}}
```

### 5. Rate Limiting
- All broker API calls must respect rate limits (configurable via env vars)
- Use `time.sleep(rate_limit_delay)` between batch operations
- Default: `SCREENING_QUOTES_RATE_LIMIT_DELAY=0.3` (300ms)

## Environment Variables

### Critical Configuration
```bash
# Database
DATABASE_URL=postgresql+psycopg://trader:trader_password@database:5432/trading_system

# Broker (Fyers)
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key

# Rate Limiting (adjust if hitting API limits)
SCREENING_QUOTES_RATE_LIMIT_DELAY=0.3        # Delay between quote fetches
VOLATILITY_MAX_WORKERS=3                     # Parallel workers
```

### Stock Screening Thresholds
See `docker-compose.yml` lines 42-208 for all screening configuration.

## Project Structure

```
src/
├── models/                      # SQLAlchemy models
│   ├── database.py             # DatabaseManager singleton
│   ├── models.py               # User, Order, Position, Trade
│   ├── stock_models.py         # Stock, SymbolMaster
│   └── historical_models.py    # HistoricalData, TechnicalIndicators
├── services/
│   ├── core/                   # Core business logic
│   │   ├── unified_broker_service.py  # Multi-broker facade
│   │   ├── order_service.py    # Order management
│   │   └── user_service.py     # User management
│   ├── data/                   # Data pipelines
│   │   ├── pipeline_saga.py    # Data collection saga (6 steps)
│   │   ├── suggested_stocks_saga.py  # Stock selection saga (7 steps)
│   │   ├── historical_data_service.py
│   │   ├── technical_indicators_service.py
│   │   └── daily_snapshot_service.py
│   ├── technical/               # Technical analysis
│   │   └── ema_strategy_calculator.py  # 8-21 EMA strategy
│   ├── trading/                # Trading automation
│   │   ├── auto_trading_service.py
│   │   └── order_performance_tracking_service.py
│   ├── brokers/                # Broker integrations
│   │   └── fyers_service.py
│   ├── interfaces/             # Abstract interfaces
│   └── implementations/        # Broker-specific implementations
├── web/
│   ├── routes/                 # Flask routes
│   ├── templates/              # Jinja2 templates
│   └── static/                 # CSS, JS, images
└── main.py                     # Flask app entry point

scheduler.py                    # Technical scheduler (runs at 10 PM)
data_scheduler.py               # Data pipeline scheduler (runs at 9 PM)
run_pipeline.py                 # Manual pipeline runner
run.py                          # Flask app launcher
config.py                       # Configuration management
```

## Testing Patterns

When testing changes:
1. **Database changes**: Test with `docker exec -it trading_system_db psql -U trader -d trading_system`
2. **Service changes**: Run manual operations to verify logic
3. **Scheduler changes**: Check logs with `docker compose logs -f technical_scheduler`
4. **API changes**: Test endpoints via `curl` or web interface

## Common Pitfalls

1. **Session Management**: Never create sessions without context manager - causes connection leaks
2. **Rate Limiting**: Always respect broker API rate limits or risk account suspension
3. **Date Handling**: Market data only available on trading days (Mon-Fri, excluding holidays)
4. **Symbol Format**: NSE symbols format is `NSE:SYMBOL-EQ` (e.g., `NSE:RELIANCE-EQ`)
5. **Saga Rollback**: If a saga step fails, ensure proper cleanup in compensating transactions
6. **Token Expiry**: Fyers tokens expire - auto-refresh runs every 6 hours via scheduler

## Strategy Selection Logic

The system uses a **unified 8-21 EMA strategy** with multi-stage filtering:

**Stage 1: Basic Filters** (from database)
- Price: 50-10,000
- Volume: > 50,000
- Market cap: > 500 Cr
- Active & tradeable stocks only

**Stage 2: EMA Strategy Filters**
- Power Zone: `ema_8 > ema_21` (bullish trend)
- Buy Signal: `buy_signal = true` (from calculator)
- Signal Quality: high or medium (based on DeMarker)

**Stage 3: Scoring & Ranking**
- Score = EMA strategy score (0-100)
- Sort by score descending
- Select top 50 stocks

**Stage 4: Daily Snapshot**
- Save to `daily_suggested_stocks` table
- Upsert: replace same-day data

## Deployment Notes

- System runs 24/7 in Docker containers
- Schedulers use `schedule` library (not cron)
- Health checks monitor container status
- Logs written to `./logs/` volume
- Database persisted in `postgres_data` volume

## Additional Resources

See comprehensive documentation:
- `README.md` - Quick start and overview
- `FEATURES.md` - Complete feature documentation
- `HOW_TO_RUN.md` - Detailed setup guide
