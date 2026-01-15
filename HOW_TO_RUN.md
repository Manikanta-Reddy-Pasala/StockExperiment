# How to Run - Setup & Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (5 Minutes)](#quick-start-5-minutes)
3. [Detailed Installation](#detailed-installation)
4. [Configuration](#configuration)
5. [Docker Deployment](#docker-deployment)
6. [Development Setup](#development-setup)
7. [Production Deployment](#production-deployment)
8. [Broker Setup (Fyers)](#broker-setup-fyers)
9. [Manual Operations](#manual-operations)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance](#maintenance)

---

## Prerequisites

### Required Software

1. **Docker & Docker Compose**
   - Docker Engine 20.10+
   - Docker Compose 2.0+
   - Install: https://docs.docker.com/get-docker/

2. **Git**
   - For cloning the repository
   - Install: https://git-scm.com/downloads

3. **Fyers Trading Account**
   - Active Fyers trading account
   - API access enabled
   - Get API credentials from: https://myapi.fyers.in/

### Optional (for development)

4. **Python 3.11+**
   - Only needed for local development without Docker
   - Install: https://www.python.org/downloads/

5. **PostgreSQL 15+**
   - Only needed for local database (Docker recommended)

### System Requirements

- **OS**: Linux, macOS, or Windows (with WSL2)
- **RAM**: Minimum 4GB, Recommended 8GB
- **Storage**: 10GB free space
- **Network**: Stable internet connection for market data

---

## Quick Start (5 Minutes)

Get the system running in 5 minutes:

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/StockExperiment.git
cd StockExperiment
```

### Step 2: Create Environment File

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Fyers credentials
nano .env  # or use any text editor
```

**Required environment variables:**
```env
# Fyers API Credentials
FYERS_CLIENT_ID=your_client_id_here
FYERS_SECRET_KEY=your_secret_key_here
FYERS_REDIRECT_URI=http://localhost:5001/brokers/fyers/callback

# Database (defaults are fine for quick start)
DATABASE_URL=postgresql://trader:trader_password@database:5432/trading_system
```

### Step 3: Start Services

```bash
# Start in development mode
./run.sh dev

# Wait 30-60 seconds for all services to start
```

### Step 4: Access Application

Open browser: **http://localhost:5001**

**Default Login**:
- Email: `admin@example.com`
- Password: `admin123`

⚠️ **Change default password immediately in production!**

### Step 5: Connect Fyers Broker

1. Navigate to: http://localhost:5001/brokers/fyers
2. Click **"Refresh Token"**
3. Login to Fyers and authorize
4. Token will be automatically saved

**That's it!** The system is now running.

---

## Detailed Installation

### Step-by-Step Installation Guide

#### 1. Clone Repository

```bash
git clone https://github.com/your-username/StockExperiment.git
cd StockExperiment
```

#### 2. Verify Docker Installation

```bash
# Check Docker version
docker --version
# Expected: Docker version 20.10.0 or higher

# Check Docker Compose version
docker compose version
# Expected: Docker Compose version 2.0.0 or higher

# Test Docker
docker run hello-world
```

#### 3. Create Directory Structure

```bash
# Create required directories (if not exist)
mkdir -p logs exports
```

#### 4. Set Permissions (Linux/Mac)

```bash
# Make run script executable
chmod +x run.sh

# Make tool scripts executable
chmod +x tools/*.sh
```

#### 5. Configure Environment

Create `.env` file in root directory:

```bash
touch .env
nano .env
```

**Minimum Configuration**:
```env
# Flask Configuration
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here-change-in-production

# Database Configuration
DATABASE_URL=postgresql://trader:trader_password@database:5432/trading_system
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_password
POSTGRES_DB=trading_system

# Fyers API Configuration
FYERS_CLIENT_ID=your_fyers_client_id
FYERS_SECRET_KEY=your_fyers_secret_key
FYERS_REDIRECT_URI=http://localhost:5001/brokers/fyers/callback

# Redis Cache (Dragonfly)
REDIS_URL=redis://dragonfly:6379/0

# Scheduler Configuration
SCHEDULER_ENABLED=true
```

**Full Configuration Options** (see [Configuration section](#configuration))

#### 6. Build Docker Images

```bash
# Build all images
docker compose build

# Or build specific service
docker compose build trading_system
```

#### 7. Initialize Database

```bash
# Start database container
docker compose up -d database

# Wait 10 seconds for database to be ready
sleep 10

# Initialize database schema
docker exec -i trading_system_db psql -U trader -d trading_system < init-scripts/01-init-db.sql

# Verify tables created
docker exec trading_system_db psql -U trader -d trading_system -c "\dt"
```

#### 8. Start All Services

```bash
# Start in development mode
./run.sh dev

# Or manually with docker compose
docker compose up -d

# Check all containers are running
docker compose ps
```

#### 9. View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f trading_system
docker compose logs -f technical_scheduler
docker compose logs -f data_scheduler
```

#### 10. Access Application

Open browser: **http://localhost:5001**

---

## Configuration

### Environment Variables Reference

#### Flask Application

```env
# Flask Settings
FLASK_APP=run.py                     # Flask entry point
FLASK_ENV=development                # development or production
SECRET_KEY=change-this-secret-key    # Flask secret key (change!)
DEBUG=true                           # Enable debug mode (dev only)
```

#### Database

```env
# PostgreSQL Configuration
DATABASE_URL=postgresql://trader:trader_password@database:5432/trading_system
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_password
POSTGRES_DB=trading_system

# Connection Pool Settings (optional)
DB_POOL_SIZE=10                      # SQLAlchemy pool size
DB_MAX_OVERFLOW=20                   # Max overflow connections
```

#### Fyers API

```env
# Fyers Credentials
FYERS_CLIENT_ID=your_client_id       # From Fyers API dashboard
FYERS_SECRET_KEY=your_secret_key     # From Fyers API dashboard
FYERS_REDIRECT_URI=http://localhost:5001/brokers/fyers/callback

# Fyers Access Token (auto-populated, don't set manually)
FYERS_ACCESS_TOKEN=auto_populated_by_oauth
```

#### Cache (Redis/Dragonfly)

```env
# Redis Configuration
REDIS_URL=redis://dragonfly:6379/0
CACHE_TTL=3600                       # Cache TTL in seconds
```

#### Schedulers

```env
# Enable/disable schedulers
SCHEDULER_ENABLED=true               # Master switch for all schedulers
DATA_SCHEDULER_ENABLED=true          # Data pipeline scheduler
TECHNICAL_SCHEDULER_ENABLED=true     # Technical indicators scheduler
```

#### Data Pipeline Configuration

```env
# Symbol Update
SYMBOL_UPDATE_DAY=monday             # Day to update symbols (monday-sunday)
SYMBOL_UPDATE_TIME=06:00             # Time to update (HH:MM)

# Historical Data Pipeline
PIPELINE_RUN_TIME=21:00              # 9:00 PM
PIPELINE_MAX_RETRIES=3               # Retry attempts per step
PIPELINE_RETRY_DELAY=60              # Seconds between retries

# Rate Limiting
SCREENING_QUOTES_RATE_LIMIT_DELAY=0.2  # Seconds between API calls
VOLATILITY_MAX_WORKERS=5             # Parallel workers for data fetch
VOLATILITY_MAX_STOCKS=500            # Max stocks per batch
```

#### Technical Indicators

```env
# Indicator Calculation
INDICATORS_RUN_TIME=22:00            # 10:00 PM
INDICATORS_LOOKBACK_DAYS=365         # Days of history for calculations

# EMA Strategy
EMA_FAST_PERIOD=8                    # Fast EMA period
EMA_SLOW_PERIOD=21                   # Slow EMA period

# Stock Selection
DAILY_STOCKS_RUN_TIME=22:15          # 10:15 PM
DAILY_STOCKS_LIMIT=50                # Top N stocks to select

# Cleanup
CLEANUP_RUN_DAY=sunday               # Day for cleanup
CLEANUP_RUN_TIME=03:00               # 3:00 AM
CLEANUP_RETENTION_DAYS=90            # Keep data for N days
```

#### Logging

```env
# Log Configuration
LOG_LEVEL=INFO                       # DEBUG, INFO, WARNING, ERROR
LOG_DIR=logs                         # Log directory
LOG_MAX_BYTES=10485760              # 10MB per log file
LOG_BACKUP_COUNT=5                   # Keep 5 backup files
```

---

## Docker Deployment

### Using run.sh Script (Recommended)

The `run.sh` script provides easy deployment commands:

#### Development Mode

```bash
# Start in development mode (hot reload, debug logs)
./run.sh dev

# Features:
# - Flask debug mode enabled
# - Detailed logging
# - Code changes reflected immediately
# - All schedulers running
```

#### Production Mode

```bash
# Start in production mode (optimized, minimal logs)
./run.sh prod

# Features:
# - Flask production mode
# - INFO level logging
# - Optimized performance
# - Security hardening
```

#### Stop Services

```bash
# Stop all containers
./run.sh stop

# Or manually
docker compose down
```

#### Restart Services

```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart trading_system
docker compose restart technical_scheduler
```

### Manual Docker Compose Commands

#### Start Services

```bash
# Start all services in background
docker compose up -d

# Start specific service
docker compose up -d trading_system

# Start with build (rebuild images)
docker compose up -d --build

# Start in foreground (see logs)
docker compose up
```

#### Stop Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (WARNING: deletes data!)
docker compose down -v

# Stop specific service
docker compose stop trading_system
```

#### View Status

```bash
# Check service status
docker compose ps

# View resource usage
docker stats

# View logs
docker compose logs -f                    # All services
docker compose logs -f trading_system      # Specific service
docker compose logs --tail=100 trading_system  # Last 100 lines
```

#### Update Services

```bash
# Pull latest images
docker compose pull

# Rebuild and restart
docker compose up -d --build

# Or force recreate
docker compose up -d --force-recreate
```

### Docker Compose Services

The system consists of 5 Docker services:

1. **database** (PostgreSQL 15)
   - Port: 5432
   - Volume: `postgres_data`
   - Health check enabled

2. **dragonfly** (Redis-compatible cache)
   - Port: 6379
   - In-memory cache
   - No persistence (by design)

3. **trading_system** (Flask web app)
   - Port: 5001
   - Depends on: database, dragonfly
   - Volumes: logs, exports

4. **technical_scheduler** (Indicators scheduler)
   - No exposed ports
   - Depends on: database
   - Volumes: logs

5. **data_scheduler** (Data pipeline)
   - No exposed ports
   - Depends on: database
   - Volumes: logs

---

## Development Setup

### Local Development (Without Docker)

For development without Docker:

#### 1. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 2. Setup Local PostgreSQL

```bash
# Install PostgreSQL 15
# (varies by OS - see PostgreSQL docs)

# Create database and user
psql -U postgres
CREATE USER trader WITH PASSWORD 'trader_password';
CREATE DATABASE trading_system OWNER trader;
\q

# Initialize schema
psql -U trader -d trading_system < init-scripts/01-init-db.sql
```

#### 3. Setup Local Redis

```bash
# Install Redis
# (varies by OS - see Redis docs)

# Start Redis server
redis-server

# Or use Docker for Redis only
docker run -d -p 6379:6379 redis:7-alpine
```

#### 4. Configure Environment

```bash
# Create .env file for local development
cat > .env << EOF
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=dev-secret-key
DATABASE_URL=postgresql://trader:trader_password@localhost:5432/trading_system
REDIS_URL=redis://localhost:6379/0
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key
FYERS_REDIRECT_URI=http://localhost:5001/brokers/fyers/callback
EOF
```

#### 5. Run Flask App

```bash
# Activate virtual environment
source venv/bin/activate

# Run Flask development server
python run.py

# Or with flask command
flask run --host=0.0.0.0 --port=5001
```

#### 6. Run Schedulers (Separate Terminals)

```bash
# Terminal 1: Data scheduler
python data_scheduler.py

# Terminal 2: Technical scheduler
python scheduler.py
```

### Development Workflow

#### Hot Reload

Flask development server supports hot reload:
- Edit Python files
- Save changes
- Flask automatically reloads

#### Debug Mode

```python
# In run.py or app.py
app.run(debug=True, host='0.0.0.0', port=5001)

# Or set environment variable
export FLASK_ENV=development
```

#### Database Migrations

```bash
# Make changes to models
# Then run manual SQL updates or use migration tool

# Example: Add new column
docker exec -it trading_system_db psql -U trader -d trading_system
ALTER TABLE stocks ADD COLUMN new_field VARCHAR(100);
```

#### Testing Database Changes

```bash
# Backup database
docker exec trading_system_db pg_dump -U trader trading_system > backup.sql

# Make changes
# Test

# Restore if needed
cat backup.sql | docker exec -i trading_system_db psql -U trader -d trading_system
```

---

## Production Deployment

### Pre-Production Checklist

- [ ] Change all default passwords
- [ ] Generate strong SECRET_KEY
- [ ] Configure production database (managed PostgreSQL recommended)
- [ ] Setup HTTPS/SSL certificate
- [ ] Configure firewall rules
- [ ] Setup monitoring and alerting
- [ ] Configure automated backups
- [ ] Test disaster recovery
- [ ] Setup logging aggregation
- [ ] Configure rate limiting
- [ ] Review security settings

### Production Configuration

Create production `.env` file:

```env
# Production Environment
FLASK_ENV=production
DEBUG=false
SECRET_KEY=generate-strong-random-key-here

# Production Database (managed service recommended)
DATABASE_URL=postgresql://user:password@prod-db-host:5432/trading_system

# Production Redis (managed service recommended)
REDIS_URL=redis://prod-redis-host:6379/0

# Fyers Production Credentials
FYERS_CLIENT_ID=prod_client_id
FYERS_SECRET_KEY=prod_secret_key
FYERS_REDIRECT_URI=https://yourdomain.com/brokers/fyers/callback

# Logging
LOG_LEVEL=WARNING
LOG_DIR=/var/log/trading_system

# Security
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=Lax
```

### Production Deployment Steps

#### Option 1: Docker on VPS/Cloud Instance

1. **Provision Server**:
   - AWS EC2, DigitalOcean Droplet, or similar
   - Minimum: 2 vCPU, 4GB RAM, 50GB SSD
   - Ubuntu 22.04 LTS recommended

2. **Setup Server**:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
```

3. **Deploy Application**:
```bash
# Clone repository
git clone https://github.com/your-username/StockExperiment.git
cd StockExperiment

# Copy production config
cp .env.production .env
nano .env  # Edit with production values

# Start services
./run.sh prod

# Verify
docker compose ps
```

4. **Security Hardening - Bind Ports to Localhost**:

⚠️ **CRITICAL**: By default, Docker exposes ports to `0.0.0.0` (all interfaces), making databases publicly accessible. This is a major security vulnerability.

Update `docker-compose.yml` to bind sensitive services to localhost only:

```yaml
services:
  database:
    ports:
      - "127.0.0.1:5432:5432"  # PostgreSQL - localhost only

  dragonfly:
    # REMOVE ports section entirely - only accessible within Docker network
    # ports:
    #   - "6379:6379"  # DANGEROUS - exposed to internet!

  trading_system:
    ports:
      - "127.0.0.1:5001:5001"  # App - localhost only (nginx will proxy)
```

**Before (VULNERABLE)**:
```
0.0.0.0:6379 → Redis EXPOSED TO INTERNET ❌
0.0.0.0:5432 → PostgreSQL EXPOSED TO INTERNET ❌
0.0.0.0:5001 → App EXPOSED TO INTERNET ❌
```

**After (SECURE)**:
```
127.0.0.1:5432 → PostgreSQL (localhost only) ✅
Internal only  → Redis/Dragonfly (Docker network) ✅
127.0.0.1:5001 → App (localhost only, nginx proxies) ✅
```

5. **Setup Nginx Reverse Proxy**:
```bash
# Install Nginx
sudo apt install nginx

# Configure Nginx
sudo nano /etc/nginx/sites-available/trading
```

Nginx configuration (HTTP only - for initial setup):
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/trading /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

6. **Setup HTTPS with Self-Signed Certificate**:

For internal/development servers without a domain:

```bash
# Create SSL directory
sudo mkdir -p /etc/nginx/ssl

# Generate self-signed certificate (valid for 365 days)
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/trading.key \
  -out /etc/nginx/ssl/trading.crt \
  -subj '/C=IN/ST=Karnataka/L=Bangalore/O=YourOrg/CN=your-server-ip'

# Update Nginx config for HTTPS
sudo nano /etc/nginx/sites-available/trading
```

**Full HTTPS Nginx configuration**:
```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-server-ip;
    return 301 https://$host$request_uri;
}

# HTTPS server
server {
    listen 443 ssl;
    server_name your-server-ip;

    # SSL certificates
    ssl_certificate /etc/nginx/ssl/trading.crt;
    ssl_certificate_key /etc/nginx/ssl/trading.key;

    # SSL configuration (modern settings)
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 90;
    }
}
```

```bash
# Test and restart Nginx
sudo nginx -t
sudo systemctl restart nginx
```

**Access**: https://your-server-ip (browser will show certificate warning for self-signed cert - this is expected)

7. **Setup SSL with Let's Encrypt** (for domains with DNS):
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal is configured automatically
```

6. **Setup Systemd Service** (optional, for auto-start):
```bash
sudo nano /etc/systemd/system/trading-system.service
```

```ini
[Unit]
Description=Trading System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/StockExperiment
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable trading-system
sudo systemctl start trading-system
```

#### Option 2: Kubernetes Deployment

(Advanced - requires Kubernetes knowledge)

```bash
# Create Kubernetes manifests
kubectl apply -f k8s/

# Monitor deployment
kubectl get pods
kubectl logs -f deployment/trading-system
```

### Production Monitoring

#### Setup Monitoring

```bash
# Install monitoring tools
# Option 1: Prometheus + Grafana
# Option 2: Datadog
# Option 3: New Relic
```

#### Health Checks

```bash
# Check service health
curl http://localhost:5001/health

# Check database
docker exec trading_system_db pg_isready -U trader

# Check schedulers
curl http://localhost:5001/api/scheduler/status
```

### Backup Strategy

#### Database Backups

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR=/backups/postgres
DATE=$(date +%Y%m%d_%H%M%S)
docker exec trading_system_db pg_dump -U trader trading_system | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Keep last 30 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
```

```bash
# Setup cron job
crontab -e

# Add line (daily at 2 AM)
0 2 * * * /path/to/backup-script.sh
```

#### Application Backups

```bash
# Backup logs and exports
tar -czf backup_$(date +%Y%m%d).tar.gz logs/ exports/
```

---

## Broker Setup (Fyers)

### Step 1: Create Fyers Account

1. Visit: https://fyers.in/
2. Open trading account
3. Complete KYC verification
4. Activate account

### Step 2: Enable API Access

1. Login to Fyers
2. Navigate to: https://myapi.fyers.in/
3. Click "Create App"
4. Fill details:
   - App Name: "Trading System"
   - Redirect URL: `http://localhost:5001/brokers/fyers/callback`
   - App Type: "Web App"
5. Submit and get:
   - **App ID** (Client ID)
   - **Secret Key**

### Step 3: Configure System

Add to `.env` file:
```env
FYERS_CLIENT_ID=your_app_id_here
FYERS_SECRET_KEY=your_secret_key_here
FYERS_REDIRECT_URI=http://localhost:5001/brokers/fyers/callback
```

### Step 4: Authorize Application

1. Start system: `./run.sh dev`
2. Open browser: http://localhost:5001
3. Login with default credentials
4. Navigate to: http://localhost:5001/brokers/fyers
5. Click **"Refresh Token"**
6. Login to Fyers and authorize
7. Token saved automatically

### Step 5: Verify Connection

```bash
# Check logs
docker compose logs technical_scheduler | grep -i "token"

# Should see:
# ✓ Fyers token is valid
# Token expires at: 2024-12-25 10:30:00
```

### Token Management

**Token Validity**: 24 hours

**Auto-Refresh**: Background thread checks every 30 minutes

**Manual Refresh**:
1. Go to: http://localhost:5001/brokers/fyers
2. Click "Refresh Token"
3. Re-authorize

**Expiry Warnings**: System logs warning 12 hours before expiry

---

## Manual Operations

### Run Data Pipeline Manually

```bash
# Full 6-step pipeline
python run_pipeline.py

# Or via Docker
docker exec trading_system python run_pipeline.py
```

### Calculate Technical Indicators Manually

```bash
docker exec trading_system python -c "
from src.models.database import get_database_manager
from src.services.technical.ema_strategy_calculator import get_ema_strategy_calculator

db_manager = get_database_manager()
with db_manager.get_session() as session:
    calculator = get_ema_strategy_calculator(session)
    symbols = ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ']
    result = calculator.calculate_all_indicators(symbols, lookback_days=252)
    print(f'Calculated indicators for {len(result)} stocks')
"
```

### Generate Daily Stock Picks Manually

```bash
docker exec trading_system python tools/generate_daily_snapshot.py
```

### Update Symbol Master Manually

```bash
docker exec data_scheduler python -c "
from src.services.data.fyers_symbol_service import get_fyers_symbol_service

service = get_fyers_symbol_service()
result = service.update_symbol_master()
print(f'Updated {result[\"symbols_updated\"]} symbols')
"
```

### Database Operations

```bash
# Connect to database
docker exec -it trading_system_db psql -U trader -d trading_system

# Common queries
\dt                                  # List tables
\d stocks                            # Describe table
SELECT COUNT(*) FROM stocks;         # Count stocks
SELECT * FROM daily_suggested_stocks ORDER BY date DESC, rank LIMIT 10;
```

### Export Data

```bash
# Export suggested stocks to CSV
docker exec trading_system python -c "
import pandas as pd
from src.models.database import get_database_manager
from src.models.historical_models import DailySuggestedStock

db_manager = get_database_manager()
with db_manager.get_session() as session:
    stocks = session.query(DailySuggestedStock).filter_by(date='2024-12-24').all()
    df = pd.DataFrame([{
        'symbol': s.symbol,
        'rank': s.rank,
        'price': s.current_price,
        'target': s.target_price,
        'stop_loss': s.stop_loss
    } for s in stocks])
    df.to_csv('exports/stocks_2024-12-24.csv', index=False)
    print('Exported to exports/stocks_2024-12-24.csv')
"
```

---

## Troubleshooting

### Common Issues

#### 1. Containers Won't Start

**Problem**: Services fail to start

**Solutions**:
```bash
# Check Docker is running
docker ps

# Check logs for errors
docker compose logs

# Remove old containers and volumes
docker compose down -v
docker compose up -d --build

# Check disk space
df -h

# Check port conflicts
sudo lsof -i :5001
sudo lsof -i :5432
```

#### 2. Database Connection Errors

**Problem**: "psycopg2.OperationalError: could not connect to server"

**Solutions**:
```bash
# Check database is running
docker compose ps database

# Check database logs
docker compose logs database

# Verify connection string in .env
echo $DATABASE_URL

# Test connection
docker exec trading_system_db pg_isready -U trader

# Restart database
docker compose restart database
sleep 10
```

#### 3. Fyers Token Expired

**Problem**: "Invalid access token" or "Token expired"

**Solutions**:
1. Open http://localhost:5001/brokers/fyers
2. Click "Refresh Token"
3. Login and authorize
4. Check logs: `docker compose logs technical_scheduler | grep token`

#### 4. Pipeline Failures

**Problem**: Data pipeline fails at certain steps

**Solutions**:
```bash
# Check pipeline tracking
docker exec -it trading_system_db psql -U trader -d trading_system -c "
SELECT * FROM pipeline_tracking ORDER BY updated_at DESC LIMIT 10;
"

# Check rate limiting errors
docker compose logs data_scheduler | grep "rate"

# Increase rate limit delay in .env
SCREENING_QUOTES_RATE_LIMIT_DELAY=0.5

# Reduce parallel workers
VOLATILITY_MAX_WORKERS=3

# Restart data scheduler
docker compose restart data_scheduler
```

#### 5. Scheduler Not Running

**Problem**: Daily tasks not executing

**Solutions**:
```bash
# Check scheduler status
docker compose ps technical_scheduler data_scheduler

# Check scheduler logs
docker compose logs technical_scheduler
docker compose logs data_scheduler

# Verify SCHEDULER_ENABLED in .env
grep SCHEDULER_ENABLED .env

# Restart schedulers
docker compose restart technical_scheduler data_scheduler

# Check system time is correct
date
```

#### 6. High Memory Usage

**Problem**: System consuming too much memory

**Solutions**:
```bash
# Check container memory usage
docker stats

# Reduce parallel workers in .env
VOLATILITY_MAX_WORKERS=3

# Reduce database pool size
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10

# Restart services
docker compose restart

# Or allocate more memory to Docker
# (Docker Desktop -> Settings -> Resources -> Memory)
```

#### 7. Slow Performance

**Problem**: Queries and operations are slow

**Solutions**:
```bash
# Check database size
docker exec trading_system_db psql -U trader -d trading_system -c "
SELECT pg_size_pretty(pg_database_size('trading_system'));
"

# Run VACUUM
docker exec trading_system_db psql -U trader -d trading_system -c "VACUUM ANALYZE;"

# Check slow queries
docker exec trading_system_db psql -U trader -d trading_system -c "
SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;
"

# Clear Redis cache
docker exec trading_system redis-cli FLUSHALL

# Restart services
docker compose restart
```

#### 8. Missing Data

**Problem**: No stocks or historical data

**Solutions**:
```bash
# Check data counts
docker exec -it trading_system_db psql -U trader -d trading_system -c "
SELECT
  (SELECT COUNT(*) FROM stocks) as stocks,
  (SELECT COUNT(*) FROM historical_data) as history,
  (SELECT COUNT(*) FROM technical_indicators) as indicators;
"

# Run pipeline manually
docker exec trading_system python run_pipeline.py

# Check for errors
docker compose logs data_scheduler | grep ERROR
```

### Debug Mode

Enable debug logging:

```bash
# In .env
LOG_LEVEL=DEBUG
FLASK_ENV=development
DEBUG=true

# Restart services
docker compose restart
```

### Getting Help

1. **Check Logs**: Always check logs first
   ```bash
   docker compose logs -f
   ```

2. **Check Documentation**: Review FEATURES.md and CLAUDE.md

3. **Check GitHub Issues**: https://github.com/your-username/StockExperiment/issues

4. **Create Issue**: If problem persists, create detailed GitHub issue with:
   - Problem description
   - Steps to reproduce
   - Error messages from logs
   - Environment details (OS, Docker version)

---

## Maintenance

### Daily Maintenance

**Automated** (via schedulers):
- ✓ Data collection (9:00 PM)
- ✓ Indicator calculation (10:00 PM)
- ✓ Stock picks generation (10:15 PM)
- ✓ Token status check (every 6 hours)

**Manual** (weekly):
- [ ] Check logs for errors
- [ ] Verify Fyers token valid
- [ ] Check disk space

### Weekly Maintenance

```bash
# Check system health
docker compose ps
docker stats --no-stream

# Review logs
tail -100 logs/scheduler.log
tail -100 logs/data_scheduler.log

# Check database size
docker exec trading_system_db psql -U trader -d trading_system -c "
SELECT pg_size_pretty(pg_database_size('trading_system'));
"

# Verify data freshness
docker exec -it trading_system_db psql -U trader -d trading_system -c "
SELECT MAX(date) as latest_date FROM historical_data;
SELECT MAX(date) as latest_picks FROM daily_suggested_stocks;
"
```

### Monthly Maintenance

```bash
# Backup database
docker exec trading_system_db pg_dump -U trader trading_system | gzip > backups/backup_$(date +%Y%m%d).sql.gz

# Vacuum database
docker exec trading_system_db psql -U trader -d trading_system -c "VACUUM FULL ANALYZE;"

# Clean old logs (keep last 30 days)
find logs/ -name "*.log" -mtime +30 -delete

# Update Docker images
docker compose pull
docker compose up -d --build

# Review and update Python dependencies
pip list --outdated
```

### Quarterly Maintenance

```bash
# Review and optimize database indexes
# Review and archive old data (>1 year)
# Security updates
# Performance tuning
# Capacity planning
```

### Monitoring Checklist

Weekly monitoring:
- [ ] All containers running
- [ ] Database size <5GB
- [ ] Memory usage <80%
- [ ] CPU usage <50% average
- [ ] Disk space >20% free
- [ ] Logs no critical errors
- [ ] Fyers token valid
- [ ] Daily pipeline success
- [ ] Stock picks generated daily

---

## Support

### Resources

- **Features Documentation**: FEATURES.md
- **Developer Guide**: CLAUDE.md
- **API Reference**: /api/docs (when running)
- **Logs**: logs/ directory

### Contact

- **GitHub Issues**: https://github.com/your-username/StockExperiment/issues
- **Email**: support@yourdomain.com

---

## Quick Reference

### Essential Commands

```bash
# Start system
./run.sh dev                                    # Development
./run.sh prod                                   # Production

# Stop system
docker compose down

# View logs
docker compose logs -f

# Restart service
docker compose restart <service_name>

# Check status
docker compose ps

# Database access
docker exec -it trading_system_db psql -U trader -d trading_system

# Run pipeline
docker exec trading_system python run_pipeline.py

# Check scheduler logs
tail -f logs/scheduler.log
tail -f logs/data_scheduler.log
```

### Important URLs

- **Application**: http://localhost:5001
- **Broker Setup**: http://localhost:5001/brokers/fyers
- **Dashboard**: http://localhost:5001/
- **Suggested Stocks**: http://localhost:5001/api/suggested-stocks

### Default Credentials

- **Email**: admin@example.com
- **Password**: admin123
- ⚠️ **Change immediately in production!**

---

## Success Indicators

You know the system is running correctly when:

✅ All 5 containers are running (`docker compose ps`)
✅ Application accessible at http://localhost:5001
✅ Login works with default credentials
✅ Fyers token shows as valid
✅ Database has ~2,259 stocks
✅ Historical data being collected daily
✅ Technical indicators calculated daily (10 PM)
✅ Daily stock picks generated (10:15 PM)
✅ No ERROR messages in logs
✅ Dashboard shows metrics

---

**System is now ready for trading!** 🚀
