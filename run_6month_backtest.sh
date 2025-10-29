#!/bin/bash
#
# 6-Month Backtesting Script
# Runs comprehensive backtesting for the technical indicator strategy
#
# Usage:
#   ./run_6month_backtest.sh                    # Run both strategies
#   ./run_6month_backtest.sh DEFAULT_RISK       # Run DEFAULT_RISK only
#   ./run_6month_backtest.sh HIGH_RISK          # Run HIGH_RISK only
#   ./run_6month_backtest.sh docker             # Run inside Docker container
#

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
MONTHS=6
HOLD_DAYS=5
CAPITAL=100000
POSITION_SIZE=20
MAX_POSITIONS=5

# Parse arguments
RUN_MODE=${1:-"all"}

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          6-MONTH BACKTESTING - TECHNICAL INDICATORS            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to run backtest
run_backtest() {
    local strategy=$1
    local stop_loss=$2
    local target=$3

    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  Running: ${strategy} Strategy${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    python3 tools/run_backtest.py \
        --months $MONTHS \
        --strategy $strategy \
        --hold-days $HOLD_DAYS \
        --capital $CAPITAL \
        --position-size $POSITION_SIZE \
        --max-positions $MAX_POSITIONS \
        --stop-loss $stop_loss \
        --target $target

    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Function to run backtest inside Docker
run_docker_backtest() {
    local strategy=$1
    local stop_loss=$2
    local target=$3

    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  Running: ${strategy} Strategy (Docker)${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    docker exec trading_system_app python3 tools/run_backtest.py \
        --months $MONTHS \
        --strategy $strategy \
        --hold-days $HOLD_DAYS \
        --capital $CAPITAL \
        --position-size $POSITION_SIZE \
        --max-positions $MAX_POSITIONS \
        --stop-loss $stop_loss \
        --target $target

    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Check if running in Docker mode
if [ "$RUN_MODE" == "docker" ]; then
    echo -e "${YELLOW}Running backtests inside Docker container...${NC}"
    echo ""

    # Run DEFAULT_RISK strategy
    run_docker_backtest "DEFAULT_RISK" 5.0 7.0

    # Run HIGH_RISK strategy
    run_docker_backtest "HIGH_RISK" 10.0 12.0

elif [ "$RUN_MODE" == "DEFAULT_RISK" ]; then
    # Run only DEFAULT_RISK strategy
    run_backtest "DEFAULT_RISK" 5.0 7.0

elif [ "$RUN_MODE" == "HIGH_RISK" ]; then
    # Run only HIGH_RISK strategy
    run_backtest "HIGH_RISK" 10.0 12.0

else
    # Run both strategies
    echo -e "${YELLOW}Configuration:${NC}"
    echo -e "  Period:         ${MONTHS} months"
    echo -e "  Hold Days:      ${HOLD_DAYS} days"
    echo -e "  Initial Capital: ₹${CAPITAL}"
    echo -e "  Position Size:  ${POSITION_SIZE}%"
    echo -e "  Max Positions:  ${MAX_POSITIONS}"
    echo ""

    # Run DEFAULT_RISK strategy (conservative)
    run_backtest "DEFAULT_RISK" 5.0 7.0

    # Wait a bit between runs
    sleep 2

    # Run HIGH_RISK strategy (aggressive)
    run_backtest "HIGH_RISK" 10.0 12.0
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    BACKTESTING COMPLETE                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo -e "  - Compare win rates and Sharpe ratios between strategies"
echo -e "  - Check maximum drawdown to understand worst-case scenarios"
echo -e "  - Review trade distribution across different market conditions"
echo ""
