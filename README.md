# Automated Trading System

A multi-user automated trading system with Docker Compose deployment.

## Features

- **Multi-User Support**: Each user has isolated trading sessions and data
- **Authentication**: Secure user registration and login system
- **Trading Engine**: Automated trading with configurable strategies
- **Web Interface**: Modern dashboard for monitoring and control
- **Risk Management**: Built-in risk controls and position limits
- **Compliance Logging**: Complete audit trail of all trading activities
- **Real-time Monitoring**: Live updates on positions, orders, and performance
- **Multiple Data Sources**: Yahoo Finance + FYERS API for Indian stocks
- **Email Alerts**: Stock pick notifications and portfolio alerts
- **Order Management**: Buy/sell orders with stop-loss and take-profit

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Git (to clone the repository)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd StockExperiment
   ```

2. **Set up environment variables**:
   ```bash
   # Create .env file with your configuration
   # See Configuration section below for required variables
   ```

3. **Start the application**:
   ```bash
   ./run.sh
   ```

4. **Access the web interface**:
   - Open your browser and go to `http://localhost:5001`
   - Register a new account or login

### Using the Run Script

The `run.sh` script provides easy management of the trading system:

```bash
# Start the system (default)
./run.sh start

# Stop the system
./run.sh stop

# Restart the system
./run.sh restart

# View logs
./run.sh logs

# Check status
./run.sh status

# Clean up everything (removes all data)
./run.sh cleanup
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Fyers API Credentials (for live trading)
FYERS_CLIENT_ID=your_client_id
FYERS_ACCESS_TOKEN=your_access_token

# Database Configuration
POSTGRES_DB=trading_system
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_password

# Application Configuration
FLASK_ENV=production
PYTHONPATH=/app
```

### Configuration

All configuration is done through environment variables in the `.env` file:

- Market timings
- Risk management parameters
- Trading strategies
- Email notifications
- Database settings
- Broker API credentials

## Architecture

### Services

- **trading_system**: Main application (Flask web app + trading engine)
- **database**: PostgreSQL database for data persistence
- **redis**: Redis for caching and session management

### Ports

- **5001**: Web interface
- **5432**: PostgreSQL database
- **6379**: Redis cache

## Development

### Running in Development Mode

The system runs in a unified environment. To modify the application:

1. Make your changes to the source code
2. Restart the system: `./run.sh restart`
3. The changes will be automatically applied

### Database Access

To access the database directly:

```bash
# Connect to the database container
docker exec -it trading_system_db psql -U trader -d trading_system
```

### Viewing Logs

```bash
# View all logs
./run.sh logs

# View specific service logs
docker-compose logs -f trading_system
docker-compose logs -f database
```

## API Endpoints

### Authentication
- `POST /login` - User login
- `POST /register` - User registration
- `GET /logout` - User logout

### Trading Data
- `GET /api/positions` - Get user positions
- `GET /api/orders` - Get user orders
- `GET /api/trades` - Get user trades
- `GET /api/strategies` - Get user strategies

### Trading Engine
- `GET /api/trading_engine/status` - Get engine status
- `GET /api/trading_engine/user_session` - Get user session status
- `POST /api/trading_engine/start_session` - Start trading session
- `POST /api/trading_engine/stop_session` - Stop trading session

### Health Check
- `GET /health` - System health status

## Security

- All API endpoints require authentication
- User data is isolated by user ID
- Passwords are hashed using bcrypt
- Database connections use environment variables

## Monitoring

### Health Checks

The system includes health checks for:
- Database connectivity
- Application status
- Service availability

### Logging

All activities are logged including:
- User authentication
- Trading activities
- System events
- Error conditions

## Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Check what's using the port
   lsof -i :5001
   # Kill the process or change the port in docker-compose.yml
   ```

2. **Database connection issues**:
   ```bash
   # Check database status
   docker-compose ps database
   # View database logs
   docker-compose logs database
   ```

3. **Application won't start**:
   ```bash
   # Check application logs
   docker-compose logs trading_system
   # Rebuild the container
   docker-compose up --build
   ```

### Reset Everything

To completely reset the system:

```bash
./run.sh cleanup
./run.sh start
```

## Configuration

Create a `.env` file in the root directory with the following configuration:

```bash
# Database Configuration
DATABASE_URL=postgresql://trader:trader_password@localhost:5432/trading_system

# Flask Configuration
FLASK_ENV=production
SECRET_KEY=your_secret_key_here

# FYERS API Configuration (for Indian stock data)
FYERS_CLIENT_ID=your_fyers_client_id
FYERS_ACCESS_TOKEN=your_fyers_access_token

# Email Configuration (for alerts)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password

# OpenAI API (for ChatGPT integration)
OPENAI_API_KEY=your_openai_api_key

# Admin User Configuration
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
ADMIN_EMAIL=admin@example.com

# System Configuration
LOG_LEVEL=INFO
```

### Data Sources

The system supports multiple data sources for Indian stocks:

1. **Yahoo Finance** - Free, always available (fallback)
2. **FYERS API** - Premium Indian stock data (primary for Indian stocks)

**FYERS API Setup:**
1. Get your FYERS API credentials from [FYERS Developer Portal](https://api-docs.fyers.in/)
2. Add your `FYERS_CLIENT_ID` and `FYERS_ACCESS_TOKEN` to the `.env` file
3. The system will automatically use FYERS for Indian stocks and fall back to Yahoo Finance if needed

## Support

For issues and questions:
1. Check the logs: `./run.sh logs`
2. Verify configuration in `.env` file
3. Ensure Docker and Docker Compose are properly installed
4. Check system resources (memory, disk space)

## License

[Add your license information here]