# Trading System

High-performance automated trading system with live market data integration.

## Quick Start

### Start Web Application
```bash
python3 run.py
```
Launch the web interface for portfolio management and trading.

### Development Mode
```bash
./run.sh dev
```
Start in development mode with auto-reloading for faster development.

## Key Features

- **Ultra-Fast Sync**: Complete stock synchronization in ~20 seconds (vs 20+ minutes)
- **Live Market Data**: Real-time quotes and price updates
- **Multi-Broker Support**: FYERS, Zerodha, and simulator integration
- **ML Predictions**: Machine learning-powered stock price forecasting
- **Portfolio Management**: Advanced strategies with risk management
- **Web Interface**: Real-time dashboard and trading controls

## Project Structure

```
/src                    # Core application code
├── services/           # Business logic and services
├── models/            # Database models and schemas
├── web/               # Flask web application
└── integrations/      # External API integrations

/config.py             # Configuration settings
/run.py                # Main application entry point
/app.py                # Alternative entry point
```

## Performance

- **Stock Sync**: 2,248 symbols processed in 19.9 seconds
- **Success Rate**: 50.9% quote verification
- **Processing Speed**: 113 symbols/second
- **Database**: PostgreSQL with optimized schema

## Requirements

- Python 3.10+
- PostgreSQL database
- Valid broker API credentials (FYERS/Zerodha)

## Documentation

-   **[Architecture Document](PROMPTS/architecture.md)**: An overview of the system architecture, modules, dependencies, and design patterns.
-   **[Coding Guidelines](PROMPTS/guidelines.md)**: A set of coding style patterns and error handling rules to be followed when contributing to the project.


  🎯 CORE STATIC TABLES

  | Table         | Final Count | Description                              |
  |---------------|-------------|------------------------------------------|
  | symbol_master | 2,253       | All available NSE symbols from Fyers     |
  | stocks        | 2,253       | Stock records with prices & fundamentals |

  📈 HISTORICAL & ANALYTICS TABLES

  | Table                | Final Count | Calculation                       |
  |----------------------|-------------|-----------------------------------|
  | historical_data      | ~806,574    | 2,253 symbols × ~358 trading days |
  | technical_indicators | ~806,574    | 2,253 symbols × ~358 trading days |

  🔄 DYNAMIC TABLES (Variable)

  | Table            | Typical Range | Updated                |
  |------------------|---------------|------------------------|
  | suggested_stocks | 0 - 50        | On filtering requests  |
  | screened_stocks  | 0 - 200       | On screening runs      |
  | ml_predictions   | 0 - 2,253     | On ML model runs       |
  | market_data      | 0 - 5,000     | Real-time market feeds |