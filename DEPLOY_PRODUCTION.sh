#!/bin/bash

################################################################################
# Production Deployment — Stock Trading System
# Server: 77.42.45.12  ·  Timezone: Asia/Kolkata (IST)
#
# WORKFLOW (locked 2026-05-27):
#   - LOCAL git is the source of truth. Commit before you deploy.
#   - The VM NEVER runs git. Source is rsync'd to /opt/trading_system and baked
#     into images via `docker compose build` on the VM (the "docker model").
#   - .env on the VM holds live secrets (rotated Fyers token, LIVE_TRADING,
#     TOTP, Telegram) — it is NEVER synced or overwritten.
#
# Three separate images / five containers:
#   trading_system  ·  technical_scheduler  ·  data_scheduler   (built here)
#   db  ·  dragonfly                                            (untouched)
# App-only change (web / templates / admin_routes) -> `--app-only` rebuilds
# just trading_system. Shared / model / executor / scheduler code -> rebuild
# all three (default).
#
# Usage:
#   ./DEPLOY_PRODUCTION.sh              # sync + build all 3 + recreate
#   ./DEPLOY_PRODUCTION.sh --app-only   # sync + build trading_system only
#   ./DEPLOY_PRODUCTION.sh --no-build   # sync source only (hotfix already cp'd)
#   ./DEPLOY_PRODUCTION.sh -h
################################################################################

set -euo pipefail

SERVER_IP="77.42.45.12"
SERVER_USER="root"
REMOTE_DIR="/opt/trading_system"
ALL_SERVICES="trading_system technical_scheduler data_scheduler"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

SERVICES="$ALL_SERVICES"
DO_BUILD=1
case "${1:-}" in
  -h|--help)
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '2,40p'; exit 0 ;;
  --app-only) SERVICES="trading_system" ;;
  --no-build) DO_BUILD=0 ;;
  "") ;;
  *) echo -e "${RED}Unknown arg: $1 (use -h)${NC}"; exit 1 ;;
esac

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Stock Trading System — Deploy (sync + build on VM)       ║${NC}"
echo -e "${BLUE}║   Server: $SERVER_IP   ·   build: ${SERVICES}${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Warn on uncommitted local changes — local git is the source of truth.
if command -v git >/dev/null && git rev-parse --git-dir >/dev/null 2>&1; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo -e "${YELLOW}⚠ Uncommitted local changes — local git is the source of truth.${NC}"
    echo -e "${YELLOW}  Commit first so git matches what you deploy. Continuing anyway.${NC}"
  fi
  echo -e "${BLUE}Local HEAD:${NC} $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
fi

# [1/4] SSH check
echo -e "\n${BLUE}[1/4] SSH to $SERVER_IP...${NC}"
if ssh -o ConnectTimeout=5 -o BatchMode=yes "$SERVER_USER@$SERVER_IP" 'echo ok' >/dev/null 2>&1; then
  echo -e "${GREEN}✓ SSH ok${NC}"
else
  echo -e "${RED}✗ SSH failed — run: ssh-add ~/.ssh/your_key${NC}"; exit 1
fi

# [2/4] Sync source (NO --delete: preserves VM-only files like scripts/, .env*,
# logs/, exports/). .env and runtime/build artefacts are excluded.
echo -e "\n${BLUE}[2/4] Syncing source -> $REMOTE_DIR ...${NC}"
# Flags chosen deliberately:
#   -rlz          recurse + symlinks + compress (NO -a: -a implies -ogtp which,
#                 run as root, would chown VM files to the local macOS uid/gid
#                 and rewrite perms — we must leave VM ownership/perms intact).
#   --checksum    decide by content hash, not mtime — local(macOS) vs VM mtimes
#                 always differ, so without this every file re-transfers each
#                 run; with it, identical files are skipped (idempotent).
#   --no-owner --no-group --no-perms   never touch VM ownership / permissions.
#   (no --delete) VM-only files survive: scripts/, .env, .env.bak.*, ml_models/.
rsync -rlz --checksum --no-owner --no-group --no-perms \
  --exclude='.git/' \
  --exclude='.env' --exclude='.env.*' \
  --exclude='logs/' --exclude='exports/' --exclude='ml_models/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pyo' --exclude='.pytest_cache/' \
  --exclude='.venv/' --exclude='venv/' --exclude='node_modules/' \
  --exclude='*.tar' --exclude='.DS_Store' \
  ./ "$SERVER_USER@$SERVER_IP:$REMOTE_DIR/"
echo -e "${GREEN}✓ Source synced (.env preserved, not touched)${NC}"

# [3/4] Build on VM + recreate. A stale image-pinning override from old
# tar-deploys would disable `build:` — remove it so build works.
if [[ "$DO_BUILD" -eq 1 ]]; then
  echo -e "\n${BLUE}[3/4] Building [$SERVICES] on VM + recreating...${NC}"
  echo -e "${YELLOW}Build runs on the server; a few minutes.${NC}"
  ssh "$SERVER_USER@$SERVER_IP" "cd $REMOTE_DIR && \
    if [ -f docker-compose.override.yml ] && grep -q 'build:' docker-compose.override.yml; then \
      echo 'Removing stale image-pinning docker-compose.override.yml'; rm -f docker-compose.override.yml; fi && \
    docker compose build $SERVICES && \
    docker compose up -d --force-recreate $SERVICES"
  echo -e "${GREEN}✓ Built + recreated${NC}"
else
  echo -e "\n${BLUE}[3/4] --no-build: recreating [$SERVICES] from current images...${NC}"
  ssh "$SERVER_USER@$SERVER_IP" "cd $REMOTE_DIR && docker compose up -d $SERVICES"
  echo -e "${GREEN}✓ Recreated${NC}"
fi

# [4/4] Verify
echo -e "\n${BLUE}[4/4] Verifying...${NC}"
ssh "$SERVER_USER@$SERVER_IP" "cd $REMOTE_DIR && docker compose ps --format 'table {{.Name}}\t{{.Status}}'"
HTTP=$(ssh "$SERVER_USER@$SERVER_IP" "curl -s -o /dev/null -w '%{http_code}' http://localhost:5001/ 2>/dev/null || echo 000")
echo -e "${BLUE}Web UI (localhost:5001 on VM):${NC} HTTP $HTTP"
echo -e "${BLUE}Scheduler boot (last lines):${NC}"
ssh "$SERVER_USER@$SERVER_IP" "docker logs trading_system_technical_scheduler --since 1m 2>&1 | grep -iE 'running|registered|error|NameError' | tail -5 || true"

echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                 DEPLOYMENT COMPLETE                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo -e "  Built: ${GREEN}${SERVICES}${NC}"
echo -e "  Reminder: the VM does NOT track git — local git is the record."
