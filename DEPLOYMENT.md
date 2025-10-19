# Production Deployment Guide

## Server Information

- **Server IP**: 135.181.34.74
- **Location**: Germany
- **Server Timezone**: Europe/Berlin (CET/CEST)
- **Container Timezone**: Asia/Kolkata (IST) - For NSE trading hours
- **User**: root (SSH key authentication)

## Important: Timezone Configuration

The server is physically located in Germany (Europe/Berlin timezone), but all Docker containers run in **Asia/Kolkata (IST)** timezone. This ensures:

- Schedulers run at correct Indian market times
- Data pipeline executes after NSE market close (9:00 PM IST)
- ML training happens before market open (6:00 AM IST)
- Trading execution at market open (9:20 AM IST)

**Do NOT change the `TZ=Asia/Kolkata` environment variable in docker-compose.yml**

## Scheduled Tasks (All times in IST)

### Data Pipeline Scheduler
- **06:00 AM** (Monday) - Symbol master update (~2,259 NSE stocks)
- **09:00 PM** (Daily) - Data pipeline (6-step saga: OHLCV, indicators, metrics)
- **09:30 PM** (Daily) - Fill missing data + Business logic calculations
- **10:00 PM** (Daily) - CSV export + Data quality validation

### ML Scheduler
- **06:00 AM** (Daily) - Train all 3 ML models (Traditional, LSTM, Kronos)
- **07:00 AM** (Daily) - Generate daily stock picks (3 models × 2 strategies = 6 combinations)
- **09:20 AM** (Daily) - Auto-trading execution (5 min after market open)
- **06:00 PM** (Daily) - Performance tracking (after market close)
- **03:00 AM** (Sunday) - Cleanup old snapshots (>90 days)

## Prerequisites

1. **SSH Access**:
   ```bash
   # Make sure you have SSH key access to the server
   ssh-add ~/.ssh/your_private_key

   # Test SSH connection
   ssh root@135.181.34.74
   ```

2. **Fyers API Credentials** (Optional - Can configure after deployment):
   - Client ID (App ID)
   - API Secret
   - Access Token

   **Note**: Credentials are stored in the **database** (not `.env` file). You can configure them via web UI after deployment. See `FYERS_SETUP.md` for details.

## Deployment Methods

This project supports **3 deployment strategies**. See `DEPLOYMENT_METHODS.md` for detailed comparison.

### Quick Comparison

| Method | Command | Speed | Best For |
|--------|---------|-------|----------|
| **Image Export** (Recommended) | `./deploy.sh image` | Fast | Production |
| **Rsync + Build** | `./deploy.sh rsync` | Slow | Development |
| **Docker Registry** | `./deploy.sh registry` | Fastest | CI/CD |

**For production → Use Method 1 (Image Export)**

---

## Step-by-Step Deployment (Recommended Method)

### Step 1: Deploy Using Docker Image Export (Recommended)

```bash
# Deploy with pre-built Docker image (fast, reliable)
./deploy.sh image
```

**OR use the dedicated script:**

```bash
# Same as above, dedicated script
./deploy_docker_image.sh
```

**OR use rsync method (slower, but smaller transfer):**

```bash
# Transfer source code and build on server
./deploy.sh rsync

# OR
./deploy_to_germany.sh
```

### What Happens During Deployment

**Image Export Method** (Recommended):
1. Builds Docker image locally (5 min)
2. Exports to tar file (~500MB-1GB)
3. Transfers to server (3-5 min)
4. Loads image on server (instant)
5. Starts containers (instant)

**Total time: ~15 min first deploy, ~5-8 min for updates**

**Rsync Method**:
1. Transfers source code (~50MB, 2 min)
2. Builds Docker image on server (8-13 min)
3. Starts containers (instant)

**Total time: ~10-15 min every deploy**

### Step 2: Verify Deployment

Check container status:
```bash
./manage_production.sh status
```

Check timezone configuration:
```bash
./manage_production.sh time
```

View logs:
```bash
./manage_production.sh logs
```

### Step 3: Configure Fyers Credentials (After Deployment)

Credentials are stored in the **database**, not `.env` files. Configure via web UI:

1. **Access web interface**: http://135.181.34.74:5001
2. **Login**: Username `admin`, Password `admin123`
3. **Go to Settings** → **Broker Configuration**
4. **Select Fyers** and enter:
   - Client ID (App ID)
   - API Secret
   - Access Token
5. **Save** and **Test Connection**

**Alternative (SSH method)**: See `FYERS_SETUP.md` for SQL commands to insert credentials directly.

**Note**: System will work without credentials, but trading features require valid Fyers credentials.

## Production Management

Use the `manage_production.sh` script for common operations:

```bash
# Show all available commands
./manage_production.sh

# View logs (real-time)
./manage_production.sh logs          # All services
./manage_production.sh logs-app      # Trading app only
./manage_production.sh logs-ml       # ML scheduler only
./manage_production.sh logs-data     # Data scheduler only

# Restart services
./manage_production.sh restart       # All services
./manage_production.sh restart-app   # Trading app only
./manage_production.sh restart-ml    # ML scheduler only
./manage_production.sh restart-data  # Data scheduler only

# Start/Stop services
./manage_production.sh start
./manage_production.sh stop

# System health check
./manage_production.sh health

# Check timezone
./manage_production.sh time

# Connect to database
./manage_production.sh db

# SSH into server
./manage_production.sh ssh
```

## Accessing the Application

### Web Interface
```
http://135.181.34.74:5001
```

### Database (PostgreSQL)
```bash
# From your local machine
ssh root@135.181.34.74
cd /opt/trading_system
docker compose exec database psql -U trader -d trading_system
```

### Redis Cache (Dragonfly)
```bash
# Port 6379 exposed on server
redis-cli -h 135.181.34.74 -p 6379
```

## Monitoring

### View Logs
```bash
# Real-time logs
./manage_production.sh logs

# Last 100 lines
ssh root@135.181.34.74 'cd /opt/trading_system && docker compose logs --tail=100'

# Specific service
./manage_production.sh logs-ml
```

### Check Data Freshness
```bash
./manage_production.sh db

# In psql:
SELECT MAX(date) as latest_data, COUNT(DISTINCT symbol) as stocks
FROM historical_data;
```

### Check Daily Stock Picks
```bash
./manage_production.sh db

# In psql:
SELECT strategy, model_type, COUNT(*) as picks
FROM daily_suggested_stocks
WHERE date = CURRENT_DATE
GROUP BY strategy, model_type;
```

## Troubleshooting

### Schedulers Not Running at Expected Times

**Check container timezone:**
```bash
./manage_production.sh time
```

**Expected output:**
```
Server time (Germany):    2025-10-19 15:30:00 CEST
Container time (should be IST):   2025-10-19 19:00:00 IST
```

If timezone is incorrect, check docker-compose.yml for `TZ=Asia/Kolkata`

### Data Pipeline Failures

**Common issues:**
1. **Fyers API Rate Limiting**:
   - Increase `SCREENING_QUOTES_RATE_LIMIT_DELAY` in .env (default: 0.3s)
   - Reduce `VOLATILITY_MAX_WORKERS` (default: 3)

2. **Stale Data**:
   - Check if it's a market holiday (NSE closed)
   - Manually run pipeline: `ssh root@135.181.34.74 'cd /opt/trading_system && docker compose exec trading_system python3 run_pipeline.py'`

3. **Database Connection Issues**:
   - Restart database: `./manage_production.sh restart`
   - Check logs: `./manage_production.sh logs-data`

### ML Training Issues

**Check training status:**
```bash
./manage_production.sh logs-ml | grep "ML Training"
```

**Common issues:**
1. **Insufficient Historical Data**:
   - Need at least 20 days of data per stock
   - Check: `./manage_production.sh db` → `SELECT COUNT(*) FROM historical_data;`

2. **Disk Space**:
   - ML models can be large (LSTM models ~10MB each)
   - Check: `ssh root@135.181.34.74 'df -h'`

3. **Memory Issues**:
   - Training can use 2-4GB RAM
   - Check: `ssh root@135.181.34.74 'free -h'`

### Container Health Issues

**Check container status:**
```bash
./manage_production.sh ps
```

**Restart unhealthy containers:**
```bash
./manage_production.sh restart
```

**View container resource usage:**
```bash
ssh root@135.181.34.74 'docker stats'
```

## Updating the Application

### Full Redeployment

```bash
# Pull latest changes locally
git pull

# Redeploy to server
./deploy_to_germany.sh
```

### Quick Update (without rebuild)

```bash
# Sync files only
rsync -avz --exclude='.git' --exclude='__pycache__' ./ root@135.181.34.74:/opt/trading_system/

# Restart services
./manage_production.sh restart
```

### Update Environment Variables

```bash
# Edit .env.production locally
nano .env.production

# Copy to server
scp .env.production root@135.181.34.74:/opt/trading_system/.env

# Restart services
./manage_production.sh restart
```

## Security Considerations

1. **Firewall Configuration**:
   ```bash
   # Allow only necessary ports
   ufw allow 22/tcp    # SSH
   ufw allow 5001/tcp  # Web interface
   ufw enable
   ```

2. **Database Password**:
   - Change default password in .env.production:
     ```
     POSTGRES_PASSWORD=your_strong_password
     DATABASE_URL=postgresql://trader:your_strong_password@database:5432/trading_system
     ```

3. **API Credentials**:
   - Never commit .env.production to git
   - Keep Fyers credentials secure
   - Rotate access tokens periodically

## Backup Strategy

### Database Backup

```bash
# Create backup
ssh root@135.181.34.74 'cd /opt/trading_system && docker compose exec -T database pg_dump -U trader trading_system > backup_$(date +%Y%m%d).sql'

# Download backup
scp root@135.181.34.74:/opt/trading_system/backup_*.sql ./backups/
```

### ML Models Backup

```bash
# Backup ML models
ssh root@135.181.34.74 'cd /opt/trading_system && tar -czf ml_models_backup_$(date +%Y%m%d).tar.gz ml_models/'

# Download backup
scp root@135.181.34.74:/opt/trading_system/ml_models_backup_*.tar.gz ./backups/
```

## Performance Optimization

1. **Database Indexing**: Already optimized in init scripts
2. **Redis Caching**: Using Dragonfly (faster than Redis)
3. **API Rate Limiting**: Configured for 3.3 req/s (safe for Fyers 10 req/s limit)
4. **Parallel Processing**: 3 workers for volatility calculations

## Support

For issues or questions:
1. Check logs: `./manage_production.sh logs`
2. Run health check: `./manage_production.sh health`
3. Review this guide
4. Check CLAUDE.md for detailed system architecture

## Quick Reference

```bash
# Deploy to production
./deploy_to_germany.sh

# View logs
./manage_production.sh logs

# Check status
./manage_production.sh status

# Restart all
./manage_production.sh restart

# Health check
./manage_production.sh health

# SSH to server
./manage_production.sh ssh

# Database access
./manage_production.sh db
```
