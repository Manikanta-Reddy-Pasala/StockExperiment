#!/bin/bash

# Production Management Script for Server 77.42.45.12
# Quick commands to manage the production deployment

SERVER_IP="77.42.45.12"
SERVER_USER="root"
REMOTE_DIR="/opt/trading_system"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to show usage
show_usage() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     Production Management - Server $SERVER_IP             ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo -e "  ${GREEN}status${NC}      - Show container status"
    echo -e "  ${GREEN}logs${NC}        - View all logs (real-time)"
    echo -e "  ${GREEN}logs-app${NC}    - View trading app logs"
    echo -e "  ${GREEN}logs-tech${NC}   - View technical scheduler logs"
    echo -e "  ${GREEN}logs-data${NC}   - View data scheduler logs"
    echo -e "  ${GREEN}restart${NC}     - Restart all services"
    echo -e "  ${GREEN}stop${NC}        - Stop all services"
    echo -e "  ${GREEN}start${NC}       - Start all services"
    echo -e "  ${GREEN}ps${NC}          - Show process details"
    echo -e "  ${GREEN}ssh${NC}         - SSH into the server"
    echo -e "  ${GREEN}health${NC}      - Check system health"
    echo ""
}

# Check if command provided
if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

COMMAND=$1

case $COMMAND in
    status)
        echo -e "${BLUE}Checking container status...${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker compose ps"
        ;;

    logs)
        echo -e "${BLUE}Showing all logs (Ctrl+C to exit)...${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker compose logs -f"
        ;;

    logs-app)
        echo -e "${BLUE}Showing trading app logs (Ctrl+C to exit)...${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker compose logs -f trading_system"
        ;;

    logs-tech)
        echo -e "${BLUE}Showing technical scheduler logs (Ctrl+C to exit)...${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker compose logs -f technical_scheduler"
        ;;

    logs-data)
        echo -e "${BLUE}Showing data scheduler logs (Ctrl+C to exit)...${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker compose logs -f data_scheduler"
        ;;

    restart)
        echo -e "${BLUE}Restarting all services...${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker compose restart"
        echo -e "${GREEN}✓ All services restarted${NC}"
        ;;

    stop)
        echo -e "${BLUE}Stopping all services...${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker compose down"
        echo -e "${GREEN}✓ All services stopped${NC}"
        ;;

    start)
        echo -e "${BLUE}Starting all services...${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker compose up -d"
        echo -e "${GREEN}✓ All services started${NC}"
        ;;

    ps)
        echo -e "${BLUE}Container process details...${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker compose ps -a"
        echo ""
        echo -e "${BLUE}Docker stats (Ctrl+C to exit)...${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker stats"
        ;;

    ssh)
        echo -e "${BLUE}Connecting to server via SSH...${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP
        ;;

    health)
        echo -e "${BLUE}System Health Check${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

        echo ""
        echo -e "${YELLOW}1. Container Status:${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker compose ps"

        echo ""
        echo -e "${YELLOW}2. Database Records:${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker compose exec -T database psql -U trader -d trading_system -c 'SELECT
            (SELECT COUNT(*) FROM stocks) as stocks,
            (SELECT COUNT(*) FROM historical_data) as historical_records,
            (SELECT COUNT(*) FROM technical_indicators) as indicators,
            (SELECT COUNT(*) FROM daily_suggested_stocks WHERE date = CURRENT_DATE) as todays_picks;'"

        echo ""
        echo -e "${YELLOW}3. Latest Data Date:${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && docker compose exec -T database psql -U trader -d trading_system -c 'SELECT MAX(date) as latest_data_date FROM historical_data;'"

        echo ""
        echo -e "${YELLOW}4. Disk Usage:${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "df -h | grep -E 'Filesystem|/dev/'"

        echo ""
        echo -e "${YELLOW}5. Memory Usage:${NC}"
        ssh -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_IP "free -h"

        echo ""
        echo -e "${GREEN}✓ Health check complete${NC}"
        ;;

    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac
