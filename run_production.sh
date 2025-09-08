#!/bin/bash

# Production Runner Script

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

# Set environment variables for production
export FLASK_ENV=production
export PYTHONPATH=$(pwd)

# Run database migrations/initialization
echo "Initializing database..."
python -c "
import sys
import os
sys.path.insert(0, 'src')
# Use PostgreSQL for production if DATABASE_URL is set
database_url = os.environ.get('DATABASE_URL', 'sqlite:///trading_system.db')
from datastore.database import get_database_manager
db = get_database_manager(database_url)
db.create_tables()
print('Database tables created successfully')
"

# Start the application in production mode
echo "Starting application in production mode..."
echo "Access the web interface at http://localhost:5000"
python run.py --mode production --config production