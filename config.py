"""
Configuration management for the trading system
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src_new"
DATA_DIR = SRC_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', f'sqlite:///{DATA_DIR}/trading_system.db')

# Email configuration
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')

# Trading configuration
DEFAULT_EXCHANGE = 'NSE'
DEFAULT_PRODUCT = 'CNC'
DEFAULT_ORDER_TYPE = 'MARKET'

# Risk management
MAX_POSITION_SIZE = float(os.environ.get('MAX_POSITION_SIZE', '100000'))
MAX_DAILY_LOSS = float(os.environ.get('MAX_DAILY_LOSS', '5000'))
STOP_LOSS_PERCENTAGE = float(os.environ.get('STOP_LOSS_PERCENTAGE', '2.0'))

# Logging configuration
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Web configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', '5001'))
