#!/bin/bash

# Simple Docker Compose Runner Script for Trading System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker to run this application."
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not available. Please install Docker Compose to run this application."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs
mkdir -p init-scripts

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file with default values..."
    cat > .env << EOF
# Trading System Environment Variables
FYERS_CLIENT_ID=your_client_id
FYERS_ACCESS_TOKEN=your_access_token

# Database Configuration
POSTGRES_DB=trading_system
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_password

# Application Configuration
FLASK_ENV=production
PYTHONPATH=/app
EOF
    print_warning "Created .env file with default values. Please update with your actual credentials."
fi

# Function to start the application
start_app() {
    print_status "Starting Trading System with Docker Compose..."
    
    # Build and start services
    docker compose up --build -d
    
    print_success "Trading System started successfully!"
    print_status "Services running:"
    echo "  - Web Interface: http://localhost:5001"
    echo "  - Database: localhost:5432"
    echo "  - Redis: localhost:6379"
    echo ""
    print_status "To view logs: docker compose logs -f"
    print_status "To stop: docker compose down"
}

# Function to stop the application
stop_app() {
    print_status "Stopping Trading System..."
    docker compose down
    print_success "Trading System stopped successfully!"
}

# Function to show logs
show_logs() {
    print_status "Showing Trading System logs..."
    docker compose logs -f
}

# Function to show status
show_status() {
    print_status "Trading System Status:"
    docker compose ps
}

# Function to restart the application
restart_app() {
    print_status "Restarting Trading System..."
    docker compose restart
    print_success "Trading System restarted successfully!"
}

# Function to clean up everything
cleanup() {
    print_warning "This will remove all containers, volumes, and data. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up Trading System..."
        docker compose down -v --remove-orphans
        docker system prune -f
        print_success "Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Main script logic
case "${1:-start}" in
    start)
        start_app
        ;;
    stop)
        stop_app
        ;;
    restart)
        restart_app
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|cleanup}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the Trading System (default)"
        echo "  stop    - Stop the Trading System"
        echo "  restart - Restart the Trading System"
        echo "  logs    - Show application logs"
        echo "  status  - Show service status"
        echo "  cleanup - Remove all containers and data"
        exit 1
        ;;
esac
