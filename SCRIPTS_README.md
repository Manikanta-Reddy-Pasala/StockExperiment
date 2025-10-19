# Deployment Scripts - Quick Reference

## Scripts Available

### Production Deployment
```
DEPLOY_PRODUCTION.sh     (6.9K) - Main deployment script
```
**Purpose**: Deploy to production server (135.181.34.74)
**Usage**: `./DEPLOY_PRODUCTION.sh`
**When**: New deployment or updates

### Production Management
```
manage_production.sh     (7.0K) - Production management
```
**Purpose**: Manage running production system
**Usage**: `./manage_production.sh [command]`
**Commands**:
- `logs` - View all logs
- `logs-app` - Web app logs only
- `logs-ml` - ML scheduler logs
- `logs-data` - Data pipeline logs
- `status` - Container status
- `restart` - Restart all services
- `restart-app` - Restart web app only
- `restart-ml` - Restart ML scheduler
- `restart-data` - Restart data scheduler
- `start` - Start all services
- `stop` - Stop all services
- `health` - Health check
- `time` - Verify timezone (IST)
- `db` - Connect to database
- `ssh` - SSH to server

### Local Development
```
run.sh                   (0.5K) - Local development server
```
**Purpose**: Run application locally for development
**Usage**: `./run.sh dev` or `./run.sh prod`
**When**: Local testing only (NOT for production)

---

## Deleted Scripts (No Longer Needed)

The following scripts have been deleted as they're replaced by `DEPLOY_PRODUCTION.sh`:

- ❌ `deploy.sh` - Old unified script (replaced)
- ❌ `deploy_docker_image.sh` - Old image deployment (replaced)
- ❌ `deploy_to_germany.sh` - Old rsync deployment (replaced)
- ❌ `deploy_test.sh` - Testing script (not needed)

---

## Quick Usage

### Deploy to Production
```bash
./DEPLOY_PRODUCTION.sh
```

### View Production Logs
```bash
./manage_production.sh logs
```

### Check Production Status
```bash
./manage_production.sh status
```

### Restart Production Services
```bash
./manage_production.sh restart
```

### Run Locally (Development)
```bash
./run.sh dev
```

---

## Summary

**Production**: Use `DEPLOY_PRODUCTION.sh` and `manage_production.sh`
**Development**: Use `run.sh`

That's it! Simple and clean. ✨

---

*Last Updated: October 20, 2025*
