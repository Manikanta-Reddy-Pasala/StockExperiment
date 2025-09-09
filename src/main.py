"""
Main Application for the Automated Trading System
Simplified for the new trading system architecture
"""
import argparse
import logging
import sys
import os
from typing import Dict, Any

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import get_config_manager
from datastore.database import get_database_manager
# Import FYERS connector for data provider
try:
    from broker.fyers_connector import FyersConnector
except ImportError:
    FyersConnector = None


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        logging.Logger: Configured logger
    """
    log_level = config.get('system', {}).get('log_level', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_system.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized with level {log_level}")
    return logger


def get_database_url(config: Dict[str, Any]) -> str:
    """
    Get database URL from configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        str: Database URL
    """
    # Get database URL from environment variable or config
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        # Fallback to config
        db_config = config.get('database', {})
        host = db_config.get('host', 'localhost')
        port = db_config.get('port', 5432)
        name = db_config.get('name', 'trading_system')
        user = db_config.get('user', 'trader')
        password = db_config.get('password', 'trader_password')
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
    
    return database_url


def initialize_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize system components for the new trading system.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Dict[str, Any]: Dictionary of initialized components
    """
    logger = logging.getLogger(__name__)
    logger.info("Initializing new trading system components")
    
    # Log environment variables for debugging
    logger.info(f"DATABASE_URL environment variable: {os.environ.get('DATABASE_URL', 'NOT SET')}")
    logger.info(f"FLASK_ENV environment variable: {os.environ.get('FLASK_ENV', 'NOT SET')}")
    
    # Initialize database with PostgreSQL URL
    database_url = get_database_url(config)
    logger.info(f"Using database URL: {database_url}")
    db_manager = get_database_manager(database_url)
    db_manager.create_tables()
    logger.info("Database initialized with PostgreSQL")
    
    # Create admin user
    try:
        from admin_setup import create_admin_user
        create_admin_user()
        logger.info("Admin user setup completed")
    except Exception as e:
        logger.error(f"Failed to create admin user: {e}")
    
    logger.info("New trading system components initialized")
    
    return {
        'config': config,
        'db_manager': db_manager
    }


def main():
    """Main entry point for the trading system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Automated Trading System')
    parser.add_argument(
        '--multi-user',
        action='store_true',
        default=True,
        help='Enable multi-user mode (default: True)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.get_config()
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting Automated Trading System")
    logger.info(f"Multi-user mode: {args.multi_user}")
    
    # Initialize components
    components = initialize_components(config)
    
    # Start the web application
    from web.app import create_app
    app = create_app()
    
    # Get configuration for web app
    host = config.get('web', {}).get('host', '0.0.0.0')
    port = config.get('web', {}).get('port', 5001)
    debug = config.get('web', {}).get('debug', False)
    
    logger.info(f"Starting web application on {host}:{port}")
    logger.info("Trading system is ready!")
    
    # Run the Flask app
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()