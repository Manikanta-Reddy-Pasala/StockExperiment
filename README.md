# NSE Stock Trading System

**Automated swing trading system for NSE (National Stock Exchange) using the 8-21 EMA strategy.**

---

## What Is This?

A production-ready, fully automated stock trading system that:

- **Analyzes 2,259+ NSE stocks** daily
- **Uses pure 8-21 EMA technical strategy** (no AI/ML)
- **Generates daily stock picks** with buy/sell signals
- **Runs completely automated** via Docker containers
- **Supports multiple brokers** (Fyers API, Zerodha planned)

### Trading Style

- **Swing Trading**: 5-14 day holding periods
- **Target Gains**: 7-12% per trade
- **Stop Loss**: 5-7% below entry
- **NOT for day trading** - this is a swing trading system

---

## Quick Start

Get the system running in 5 minutes:

```bash
# 1. Clone repository
git clone https://github.com/your-username/StockExperiment.git
cd StockExperiment

# 2. Create .env file with your Fyers credentials
cp .env.example .env
nano .env  # Add your FYERS_CLIENT_ID and FYERS_SECRET_KEY

# 3. Start all services
./run.sh dev

# 4. Access web interface
http://localhost:5001
```

**Default Login:**
- Email: `admin@example.com`
- Password: `admin123`
- ⚠️ Change password immediately in production!

---

## Documentation

### Comprehensive Guides

- **[FEATURES.md](FEATURES.md)** - Complete system features and how everything works
  - 8-21 EMA strategy explained in detail
  - Technical architecture
  - Data pipeline
  - API endpoints
  - Database schema
  - Frontend features

- **[HOW_TO_RUN.md](HOW_TO_RUN.md)** - Setup and deployment guide
  - Installation steps
  - Configuration options
  - Docker deployment
  - Broker setup (Fyers)
  - Troubleshooting
  - Maintenance

- **[CLAUDE.md](CLAUDE.md)** - Developer guide (for Claude Code AI assistant)
  - Code patterns
  - Development commands
  - Critical implementation details

---

## System Overview

### Architecture

```
Docker Services (5 containers):
├── trading_system       - Flask web app (Port 5001)
├── technical_scheduler  - Indicators & stock picks (daily 10 PM)
├── data_scheduler       - Data collection pipeline (daily 9 PM)
├── database             - PostgreSQL 15
└── dragonfly            - Redis-compatible cache
```

### Daily Automation Schedule

```
09:00 PM → Data pipeline (fetch prices, history)
10:00 PM → Calculate technical indicators (EMA, RSI, MACD)
10:15 PM → Generate daily stock picks (top 50 stocks)
03:00 AM → Cleanup old data (Sunday only)
06:00 AM → Update symbol master (Monday only)
```

### 8-21 EMA Strategy

The system uses a proven **8-21 EMA crossover strategy** for swing trading:

**BUY Signal:**
- 8-day EMA crosses above 21-day EMA
- Price above both EMAs
- RSI between 30-70
- Positive MACD histogram
- Volume confirmation

**SELL Signal:**
- 8-day EMA crosses below 21-day EMA
- Price breaks below 50-day SMA
- Stop loss hit

**Target & Stop Loss:**
- Fibonacci extensions for targets (127.2%, 161.8%, 200%)
- Stop loss below 21 EMA or recent swing low

---

## Key Features

- ✅ **Pure Technical Analysis** - No AI/ML complexity
- ✅ **Single Unified Strategy** - 8-21 EMA crossover
- ✅ **Fully Automated** - Zero manual intervention
- ✅ **2,259+ NSE Stocks** - Complete market coverage
- ✅ **Daily Stock Picks** - Top 50 recommendations daily
- ✅ **Multi-Broker Support** - Fyers API (Zerodha planned)
- ✅ **Saga Pattern** - Fault-tolerant data pipeline
- ✅ **Production Ready** - Docker, logging, monitoring
- ✅ **Web Dashboard** - Interactive charts and trading interface

---

## Technology Stack

**Backend:**
- Python 3.11
- Flask 3.0 (web framework)
- SQLAlchemy 2.0 (ORM)
- PostgreSQL 15 (database)
- pandas + pandas-ta (technical analysis)

**Frontend:**
- Bootstrap 5 (UI framework)
- Chart.js (charts)
- Vanilla JavaScript

**DevOps:**
- Docker & Docker Compose
- 5 containerized services
- Volume persistence
- Health checks

---

## Essential Commands

```bash
# Start system
./run.sh dev                                    # Development mode
./run.sh prod                                   # Production mode

# Stop system
docker compose down

# View logs
docker compose logs -f                          # All services
docker compose logs -f trading_system           # Web app
docker compose logs -f technical_scheduler      # Indicators
docker compose logs -f data_scheduler           # Data pipeline

# Check status
docker compose ps                               # Container status
./tools/check_all_schedulers.sh                # Scheduler status

# Database access
docker exec -it trading_system_db psql -U trader -d trading_system

# Manual operations
python run_pipeline.py                          # Run data pipeline
docker exec trading_system python tools/generate_daily_snapshot.py  # Generate picks
```

---

## Database Schema

**11 tables** with ~1.6M records, ~500MB total:

**Core Data:**
- `stocks` - Current stock info (2,259 stocks)
- `historical_data` - 1-year OHLCV data (820K records)
- `technical_indicators` - Daily indicators (820K records)
- `daily_suggested_stocks` - Daily picks (50/day, growing)
- `symbol_master` - NSE symbol list (2,259 symbols)

**Trading:**
- `users` - User accounts
- `broker_configurations` - Broker API credentials
- `orders` - Order history
- `positions` - Open positions
- `trades` - Completed trades

**System:**
- `pipeline_tracking` - Saga execution status

---

## API Endpoints

```bash
# Get suggested stocks
curl "http://localhost:5001/api/suggested-stocks?limit=10"

# Get stock chart data
curl "http://localhost:5001/api/charts/stock/NSE:RELIANCE-EQ"

# Dashboard metrics
curl "http://localhost:5001/api/dashboard/metrics"

# Broker status
curl "http://localhost:5001/api/broker/status"
```

---

## Web Interface

Access at: **http://localhost:5001**

**Pages:**
- `/` - Dashboard (metrics, charts, positions)
- `/api/suggested-stocks` - Daily stock picks
- `/settings` - Trading and broker settings
- `/brokers/fyers` - Fyers broker integration

---

## Prerequisites

- **Docker & Docker Compose** (20.10+)
- **4GB RAM** minimum, 8GB recommended
- **5GB disk space** minimum
- **Fyers Trading Account** with API access
- **Stable internet** for market data

---

## Broker Setup

### Fyers API

1. **Create Fyers Account**: https://fyers.in/
2. **Enable API Access**: https://myapi.fyers.in/
3. **Create App** and get credentials:
   - App ID (Client ID)
   - Secret Key
4. **Add to `.env` file**:
   ```env
   FYERS_CLIENT_ID=your_app_id
   FYERS_SECRET_KEY=your_secret_key
   ```
5. **Authorize in web interface**: http://localhost:5001/brokers/fyers

---

## Troubleshooting

### Common Issues

**1. Containers won't start**
```bash
docker compose down -v
docker compose up -d --build
```

**2. Database connection errors**
```bash
docker compose restart database
sleep 10
docker compose restart trading_system
```

**3. Fyers token expired**
- Visit: http://localhost:5001/brokers/fyers
- Click "Refresh Token"
- Re-authorize

**4. Pipeline failures**
```bash
# Check logs
docker compose logs data_scheduler | grep ERROR

# Adjust rate limits in .env
SCREENING_QUOTES_RATE_LIMIT_DELAY=0.5
VOLATILITY_MAX_WORKERS=3

# Restart
docker compose restart data_scheduler
```

**5. No stock picks generated**
```bash
# Check data exists
docker exec -it trading_system_db psql -U trader -d trading_system -c "
SELECT COUNT(*) FROM historical_data WHERE date >= CURRENT_DATE - 365;
"

# Run manual calculation
docker exec trading_system python tools/generate_daily_snapshot.py
```

For more troubleshooting, see **[HOW_TO_RUN.md](HOW_TO_RUN.md)**.

---

## Performance

- **Data Pipeline**: 20-30 minutes (daily)
- **Indicators Calculation**: 5-7 minutes (daily)
- **Stock Selection**: 2-3 minutes (daily)
- **API Response Time**: <100ms (cached)
- **Database Size**: ~500MB, ~1.6M records
- **Memory Usage**: ~2GB total (all containers)

---

## System Requirements

**Minimum:**
- 2 CPU cores
- 4GB RAM
- 10GB disk space
- Ubuntu 20.04+ or macOS 10.15+

**Recommended:**
- 4 CPU cores
- 8GB RAM
- 20GB disk space
- Ubuntu 22.04 LTS or macOS 12+

---

## Project Structure

```
StockExperiment/
├── init-scripts/
│   └── 01-init-db.sql          # Database schema
├── src/
│   ├── models/                  # SQLAlchemy models
│   ├── services/                # Business logic
│   │   ├── core/               # Core services
│   │   ├── brokers/            # Broker integrations
│   │   ├── data/               # Data pipeline
│   │   ├── technical/          # Technical analysis
│   │   └── trading/            # Trading services
│   └── web/
│       ├── routes/             # API endpoints
│       ├── templates/          # HTML templates
│       └── static/             # CSS, JS, images
├── tools/                       # Utility scripts
├── logs/                        # Application logs
├── exports/                     # CSV exports
├── config.py                    # Configuration
├── run.py                       # Flask launcher
├── scheduler.py                 # Technical indicators scheduler
├── data_scheduler.py            # Data pipeline scheduler
├── run_pipeline.py              # Manual pipeline runner
├── docker-compose.yml           # Docker services
├── Dockerfile                   # Container image
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
├── README.md                    # This file
├── FEATURES.md                  # Comprehensive features guide
├── HOW_TO_RUN.md               # Setup & deployment guide
└── CLAUDE.md                    # Developer guide
```

---

## Support

**Documentation:**
- **Features Guide**: [FEATURES.md](FEATURES.md)
- **Setup Guide**: [HOW_TO_RUN.md](HOW_TO_RUN.md)
- **Developer Guide**: [CLAUDE.md](CLAUDE.md)

**Quick Help:**
```bash
# Check system status
./tools/check_all_schedulers.sh

# View logs
docker compose logs -f

# Database access
docker exec -it trading_system_db psql -U trader -d trading_system

# Manual pipeline
python run_pipeline.py
```

**Web Interface:**
- Dashboard: http://localhost:5001
- Broker Setup: http://localhost:5001/brokers/fyers
- Settings: http://localhost:5001/settings

---

## Contributing

This is a personal trading system. For questions or issues:
1. Check documentation first (FEATURES.md, HOW_TO_RUN.md)
2. Review logs: `docker compose logs -f`
3. Check database: `docker exec -it trading_system_db psql`

---

## License

Private use only. Not for redistribution.

---

## Disclaimer

⚠️ **Trading Risk Disclaimer**

This software is for educational and informational purposes only. Stock trading involves substantial risk of loss and is not suitable for everyone. Past performance is not indicative of future results.

**Key Risks:**
- You may lose all invested capital
- Historical backtests do not guarantee future performance
- Market conditions change and strategies may become ineffective
- Technical analysis has limitations and is not always accurate
- System errors, bugs, or outages may result in losses

**Your Responsibilities:**
- Understand the risks before trading
- Only invest capital you can afford to lose
- Monitor your positions regularly
- Set appropriate stop losses
- Diversify your portfolio
- Consult with a licensed financial advisor

**No Warranty:**
This software is provided "as is" without warranties of any kind. The developers are not liable for any losses incurred through use of this system.

**By using this system, you acknowledge and accept all risks associated with stock trading.**

---

**Built for automated swing trading with pure technical analysis** 🚀

Last Updated: October 31, 2025
