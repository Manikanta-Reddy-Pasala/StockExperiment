# Automated Trading System

This is a Python-based automated trading system that selects momentum stocks and places, modifies, and exits trades on Indian exchanges through the Fyers API.

## Features

- Integration with Fyers for market data and order execution
- Pluggable momentum stock selection engine with multiple strategies
- Comprehensive risk management with position sizing and exposure controls
- Dual-mode operation (development/simulation and production)
- Compliance with SEBI retail algo trading regulations
- Real-time monitoring, alerting, and reporting capabilities
- Web-based user interface with real-time dashboard (Flask)
- Detailed performance reporting (daily, weekly, and cumulative P&L)
- Order execution tracking and status monitoring
- Stock selection visualization and strategy insights
- Email alerting for critical events and system notifications
- Manual override capabilities through web interface
- Configuration management via web UI
- **Dual data sources** (Fyers and yFinance with fallback capabilities)
- **Automated stop-loss functionality**
- **Backtesting engine with ASOF_DATE override**
- **Market-on-Open order support**
- **Performance analytics** (Sharpe/Sortino ratios, drawdown metrics)
- **Complete trade logging**
- **Database-based charting system**
- **Matplotlib-based visualizations**
- **ChatGPT validation for stock selections**
- **Docker deployment support**
- **Shell scripts for local and production running**

## System Architecture

The system consists of the following core modules:

1. **BrokerConnector**: Interface with Fyers API for market data and order execution
2. **DataStore**: Persistent storage for market data, trades, configurations, and logs
3. **SelectorEngine**: Momentum stock selection with pluggable strategies
4. **RiskManager**: Position sizing, exposure controls, and risk limits enforcement
5. **OrderRouter**: Order placement, modification, and state management
6. **Simulator**: Paper-trading engine for development mode
7. **Scheduler**: Market-aware job scheduling with holiday awareness
8. **Reporting & Alerts**: Dashboard metrics, reports, and alerting system
9. **ComplianceLogger**: Immutable audit trail for regulatory compliance
10. **WebInterface**: Flask-based web application for UI and API endpoints
11. **EmailAlerting**: Email notification system for critical events and reports
12. **DataProvider**: Dual data source management (Fyers + yFinance)
13. **Backtesting**: Strategy backtesting engine with performance metrics
14. **Analytics**: Performance analytics and risk metrics
15. **Logging**: Comprehensive trade execution logging
16. **Visualization**: Database and Matplotlib-based charting
17. **AI**: ChatGPT validation for stock selections

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure the system using files in the `config/` directory
4. Set up environment variables for production mode (FYERS_CLIENT_ID, FYERS_ACCESS_TOKEN, OPENAI_API_KEY)
5. Run the application: `python src/main.py`

## Usage

The system operates in two modes:

1. **Development Mode**: Local execution with simulated order fills and P&L
2. **Production Mode**: Live trading with real order execution and full risk controls

### Running the Application

#### Local Development
```bash
./run_local.sh
```

#### Production
```bash
./run_production.sh
```

#### With Docker
```bash
docker-compose up
```

### Accessing the Web Interface

For development mode: http://localhost:5001
For production mode: http://localhost:8000

### Environment Variables

- `FYERS_CLIENT_ID`: Fyers API Client ID (or App ID) for your trading app (production only).
- `FYERS_ACCESS_TOKEN`: Fyers API Access Token. This needs to be generated via a browser-based login. See Fyers API documentation for more details. (production only)
- `OPENAI_API_KEY`: OpenAI API key for ChatGPT validation (optional)
- `DATABASE_URL`: Database connection URL (optional, defaults to SQLite)

## New Features Implemented

### Dual Data Sources
The system now supports two data sources:
- Primary: Fyers API
- Secondary: yFinance with automatic fallback

### Automated Stop-Loss
Risk manager includes configurable stop-loss functionality with automatic position exit.

### Backtesting Engine
Complete backtesting system with:
- ASOF_DATE override for historical analysis
- Multiple strategy support
- Performance metrics (Sharpe/Sortino ratios, drawdown)

### Interactive Trading Features
- Market-on-Open order support
- Enhanced order management

### Performance Analytics
Comprehensive performance metrics including:
- Sharpe and Sortino ratios
- Maximum drawdown calculations
- Win rate tracking

### Trade Logging
Complete execution logging for compliance and analysis.

### Visualization
Both database-based charts and Matplotlib visualizations for:
- Portfolio performance
- Trade executions
- Performance comparisons

### ChatGPT Validation
Stock selections are validated using ChatGPT for enhanced decision making.

### Deployment Options
Multiple deployment methods:
- Shell scripts for local/production
- Docker Compose for containerized deployment