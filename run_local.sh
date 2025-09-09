#!/bin/bash

# Local Development Runner Script

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed. Please install Python 3 to run this application."
    exit 1
fi

# Check if virtual environment exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create logs directory if it doesn't exist
if [ ! -d "logs" ]; then
    echo "Creating logs directory..."
    mkdir logs
fi

# Kill any existing processes on port 5001
echo "Checking for processes on port 5001..."
if lsof -i :5001 > /dev/null 2>&1; then
    echo "Killing existing processes on port 5001..."
    lsof -ti :5001 | xargs kill -9
    sleep 2
fi

# Set environment variables for development PostgreSQL
export DATABASE_URL="postgresql://trader_dev:trader_dev_password@localhost:5432/trading_system_dev"
export FLASK_ENV=development
export PYTHONPATH=$(pwd)

# Run database migrations/initialization
echo "Initializing database..."
python -c "
import sys
import os
sys.path.insert(0, 'src')
# Always use PostgreSQL
database_url = os.environ.get('DATABASE_URL')
from datastore.database import get_database_manager
db = get_database_manager(database_url)
db.create_tables()
print('Database tables created successfully')
"

# Start the application in development mode
echo "Starting application in development mode..."
echo "Access the web interface at http://localhost:5001"
python run.py --mode development --config development