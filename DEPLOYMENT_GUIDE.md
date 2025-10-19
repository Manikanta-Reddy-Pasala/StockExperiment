# Stock Trading System - Deployment Guide

**One Script. One Command. Production Ready.**

---

## Quick Start (New Deployment)

```bash
./DEPLOY_PRODUCTION.sh
```

That's it! The script will:
1. ✅ Build Docker image locally
2. ✅ Transfer to production server
3. ✅ Start all containers
4. ✅ Verify deployment

**Time**: ~15-20 minutes

---

## Prerequisites

### 1. SSH Access

```bash
# Add your SSH key
ssh-add ~/.ssh/your_private_key

# Test connection
ssh root@135.181.34.74
```

### 2. Docker Installed Locally

```bash
# Check Docker version
docker --version
docker compose version
```

That's all you need! The script will install Docker on the server automatically.

---

## What The Script Does

```
Step 1: Test SSH connection
Step 2: Build Docker image locally (~6 minutes)
Step 3: Export image to tar file (~11GB)
Step 4: Create directory structure on server
Step 5: Transfer files to server (~20 minutes)
Step 6: Install Docker on server (if needed)
Step 7: Load image and start containers
Step 8: Verify deployment and show status
```

**Output**: All 5 containers running, web UI accessible

---

## After Deployment

### 1. Access Web UI

**URL**: http://135.181.34.74:5001

**Login**:
- Username: `admin`
- Password: See `ADMIN_CREDENTIALS.md`

### 2. Configure Fyers Credentials

**Via Web UI** (Recommended):
1. Login to http://135.181.34.74:5001
2. Go to: Settings → Broker Configuration
3. Select: Fyers
4. Enter:
   - Client ID (App ID)
   - API Secret
   - Access Token
5. Click: Save and Test

**See**: `FYERS_SETUP.md` for detailed instructions

### 3. Monitor System

```bash
# View all logs
./manage_production.sh logs

# Check container status
./manage_production.sh status

# Check timezone (should be IST)
./manage_production.sh time

# System health check
./manage_production.sh health
```

### 4. Wait for Scheduled Tasks

**Tonight at 9:00 PM IST**:
- Data pipeline runs (downloads ~2,259 NSE stocks)

**Tomorrow at 6:00 AM IST**:
- ML models trained

**Tomorrow at 7:00 AM IST**:
- Daily stock picks generated

---

## Management Commands

All commands use `manage_production.sh`:

```bash
# View logs
./manage_production.sh logs           # All services
./manage_production.sh logs-app       # Web app only
./manage_production.sh logs-ml        # ML scheduler
./manage_production.sh logs-data      # Data pipeline

# Restart services
./manage_production.sh restart        # All services
./manage_production.sh restart-app    # Specific service

# Start/Stop
./manage_production.sh start
./manage_production.sh stop

# Status checks
./manage_production.sh status         # Container status
./manage_production.sh health         # Health check
./manage_production.sh time          # Verify timezone

# Database access
./manage_production.sh db            # Connect to PostgreSQL

# SSH to server
./manage_production.sh ssh
```

---

## Updating Production

**For updates after initial deployment**:

```bash
# Same script - it will update existing deployment
./DEPLOY_PRODUCTION.sh
```

The script automatically:
- Stops old containers
- Loads new image
- Starts updated containers
- Preserves database and ML models

---

## File Structure

### Essential Files (Don't Delete!)

```
StockExperiment/
├── DEPLOY_PRODUCTION.sh          ← Main deployment script
├── manage_production.sh           ← Production management
├── docker-compose.yml             ← Container configuration
├── .env.production                ← Production environment vars
├── init-scripts/
│   └── 01-init-db.sql            ← Database schema
├── scheduler.py                   ← ML scheduler
├── data_scheduler.py              ← Data pipeline scheduler
└── run_pipeline.py                ← Manual pipeline runner
```

### Documentation Files

```
├── DEPLOYMENT_GUIDE.md            ← This file (simple guide)
├── ADMIN_CREDENTIALS.md           ← Login credentials (secured)
├── FYERS_SETUP.md                ← Broker credentials setup
├── DEPLOYMENT_SUMMARY.md          ← Latest deployment status
├── ML_MODELS_FIX.md              ← ML models architecture
└── DEPLOYMENT.md                  ← Detailed reference (advanced)
```

### Other Scripts (Not Needed for Deployment)

```
├── deploy.sh                      ← Old unified script (deprecated)
├── deploy_docker_image.sh         ← Old image script (deprecated)
├── deploy_to_germany.sh          ← Old rsync script (deprecated)
├── deploy_test.sh                ← Testing script (optional)
└── run.sh                        ← Local development only
```

**For production, use only**: `DEPLOY_PRODUCTION.sh` and `manage_production.sh`

---

## Troubleshooting

### SSH Connection Failed

```bash
# Error: "SSH connection failed"
# Fix: Add SSH key
ssh-add ~/.ssh/your_private_key
```

### Docker Build Failed

```bash
# Error during docker build
# Fix: Ensure Docker is running locally
docker ps
```

### Transfer Failed

```bash
# Error during file transfer
# Fix: Check network connection and server accessibility
ping 135.181.34.74
ssh root@135.181.34.74
```

### Containers Not Starting

```bash
# Check logs on server
./manage_production.sh logs

# Restart containers
./manage_production.sh restart
```

### Web UI Not Accessible

```bash
# Check container status
./manage_production.sh status

# Check web app logs
./manage_production.sh logs-app

# Test from server
ssh root@135.181.34.74 'curl -I http://localhost:5001/'
```

---

## Architecture

### Server Details

| Parameter | Value |
|-----------|-------|
| IP Address | 135.181.34.74 |
| Location | Germany |
| Server Timezone | UTC/Europe/Berlin |
| Container Timezone | **Asia/Kolkata (IST)** |
| OS | Linux |

### Containers (5 Total)

| Container | Port | Purpose |
|-----------|------|---------|
| trading_system_app | 5001 | Web UI + REST API |
| database | 5432 | PostgreSQL 15 |
| dragonfly | 6379 | Redis cache |
| ml_scheduler | - | ML training scheduler |
| data_scheduler | - | Data pipeline scheduler |

### Scheduled Tasks (IST)

| Time | Task | Frequency |
|------|------|-----------|
| 06:00 AM | Symbol Update | Monday |
| 06:00 AM | ML Training | Daily |
| 07:00 AM | Stock Picks | Daily |
| 09:20 AM | Auto Trading | Daily |
| 09:00 PM | Data Pipeline | Daily |
| 09:30 PM | Data Cleanup | Daily |
| 10:00 PM | CSV Export | Daily |
| 03:00 AM | Old Data Cleanup | Sunday |

### Persistent Storage

**Volumes on remote server**:
- `/opt/trading_system/postgres_data/` - Database (auto-created)
- `/opt/trading_system/ml_models/` - ML models
- `/opt/trading_system/logs/` - Application logs
- `/opt/trading_system/exports/` - CSV exports

---

## Security Notes

### Default Admin Password

⚠️ **Default password has been changed** to a complex password.

**See**: `ADMIN_CREDENTIALS.md` for current password

**Recommendation**: Change password after first login via web UI

### Firewall (Optional but Recommended)

```bash
# SSH to server
ssh root@135.181.34.74

# Configure firewall
ufw allow 22/tcp      # SSH
ufw allow 5001/tcp    # Web UI
ufw enable
```

### Backup Strategy

**Database backup**:
```bash
ssh root@135.181.34.74 'cd /opt/trading_system && \
  docker compose exec -T database pg_dump -U trader trading_system \
  > backup_$(date +%Y%m%d).sql'
```

**ML models backup**:
```bash
ssh root@135.181.34.74 'cd /opt/trading_system && \
  tar -czf ml_models_backup_$(date +%Y%m%d).tar.gz ml_models/'
```

**Recommended**: Weekly backups on Sunday at 4:00 AM IST

---

## Quick Reference

```bash
# Deploy to production
./DEPLOY_PRODUCTION.sh

# View logs
./manage_production.sh logs

# Check status
./manage_production.sh status

# Restart all
./manage_production.sh restart

# SSH to server
./manage_production.sh ssh

# Database access
./manage_production.sh db
```

---

## Support

### Key URLs

- **Web UI**: http://135.181.34.74:5001
- **Database**: postgresql://trader:***@135.181.34.74:5432/trading_system
- **Redis**: redis://135.181.34.74:6379

### Documentation

- **Quick Start**: This file (DEPLOYMENT_GUIDE.md)
- **Credentials**: ADMIN_CREDENTIALS.md
- **Fyers Setup**: FYERS_SETUP.md
- **ML Models**: ML_MODELS_FIX.md
- **Detailed Docs**: DEPLOYMENT.md

### Server Access

```bash
# SSH to server
ssh root@135.181.34.74

# Navigate to app directory
cd /opt/trading_system

# View container logs
docker compose logs -f
```

---

## Summary

**One Command Deployment**:
```bash
./DEPLOY_PRODUCTION.sh
```

**Everything Else**:
```bash
./manage_production.sh [command]
```

**That's it!** Simple, clean, production-ready.

---

*Last Updated: October 20, 2025*
*Server: 135.181.34.74 (Germany)*
*Timezone: Asia/Kolkata (IST)*
