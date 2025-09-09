"""
Main Application for the Automated Trading System
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
try:
    from broker.fyers_connector import FyersConnector
except ImportError:
    FyersConnector = None
from simulator.simulator import Simulator
from selector.selector_engine import SelectorEngine
from risk.risk_manager import RiskManager
from order.order_router import OrderRouter
from scheduler.trading_scheduler import TradingScheduler
from scheduler.multi_user_trading_scheduler import MultiUserTradingScheduler
from trading_engine.multi_user_trading_engine import MultiUserTradingEngine
from reporting.dashboard import DashboardReporter, AlertManager
from compliance.compliance_logger import ComplianceLogger
from email_alerts.email_alerts import EmailAlertingSystem
# Add the data provider manager import
from data_provider.data_manager import get_data_manager


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
        str: Database connection URL
    """
    database_url = config.get('database', {}).get('url')
    if database_url:
        logging.getLogger(__name__).info(f"Using database URL from config: {database_url}")
        return database_url
    
    # Fallback: construct from individual components
    db_config = config.get('database', {})
    host = db_config.get('host', 'database')
    port = db_config.get('port', 5432)
    name = db_config.get('name', 'trading_system')
    user = db_config.get('user', 'trader')
    password = db_config.get('password', 'trader_password')
    
    constructed_url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
    logging.getLogger(__name__).info(f"Constructed database URL: {constructed_url}")
    return constructed_url


def initialize_components(config: Dict[str, Any], multi_user: bool = True):
    """
    Initialize all system components.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        multi_user (bool): Whether to initialize multi-user components
        
    Returns:
        dict: Dictionary of initialized components
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing components (multi_user={multi_user})")
    
    # Log environment variables for debugging
    logger.info(f"DATABASE_URL environment variable: {os.environ.get('DATABASE_URL', 'NOT SET')}")
    logger.info(f"FLASK_ENV environment variable: {os.environ.get('FLASK_ENV', 'NOT SET')}")
    
    # Initialize database with PostgreSQL URL
    database_url = get_database_url(config)
    logger.info(f"Using database URL: {database_url}")
    db_manager = get_database_manager(database_url)
    db_manager.create_tables()
    logger.info("Database initialized with PostgreSQL")
    
    # Initialize data provider manager
    data_manager = get_data_manager()
    logger.info("Data provider manager initialized")
    
    # Initialize broker connector - use Fyers if available, otherwise simulator
    if FyersConnector is not None:
        client_id = config.get('broker', {}).get('fyers_client_id', 'your_client_id')
        access_token = config.get('broker', {}).get('fyers_access_token', 'your_access_token')
        broker_connector = FyersConnector(client_id=client_id, access_token=access_token)
        logger.info("Initialized FyersConnector")
    else:
        # Fallback to simulator if FyersConnector is not available
        broker_connector = Simulator()
        logger.info("Initialized Simulator (FyersConnector not available)")
    
    # Initialize other components
    selector_engine = SelectorEngine()
    selector_engine.set_data_manager(data_manager)  # Set data manager
    risk_manager = RiskManager(config.get('risk', {}))
    order_router = OrderRouter(broker_connector)
    dashboard_reporter = DashboardReporter(db_manager)
    alert_manager = AlertManager(config)
    compliance_logger = ComplianceLogger(db_manager)
    email_alerting = EmailAlertingSystem(config)
    
    # Initialize scheduler based on multi_user flag
    if multi_user:
        # Initialize multi-user trading engine
        trading_engine = MultiUserTradingEngine(config, 'production')  # Always use production mode for multi-user
        scheduler = MultiUserTradingScheduler(config, trading_engine)
        logger.info("Initialized MultiUserTradingEngine and MultiUserTradingScheduler")
    else:
        # Initialize single-user scheduler
        scheduler = TradingScheduler(config)
        scheduler.set_order_router(order_router)  # Set order router for MOO processing
        trading_engine = None
        logger.info("Initialized single-user TradingScheduler")
    
    logger.info("All components initialized")
    
    return {
        'config': config,
        'db_manager': db_manager,
        'data_manager': data_manager,
        'broker_connector': broker_connector,
        'selector_engine': selector_engine,
        'risk_manager': risk_manager,
        'order_router': order_router,
        'scheduler': scheduler,
        'trading_engine': trading_engine,
        'dashboard_reporter': dashboard_reporter,
        'alert_manager': alert_manager,
        'compliance_logger': compliance_logger,
        'email_alerting': email_alerting
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
    parser.add_argument(
        '--single-user',
        action='store_true',
        help='Disable multi-user mode (use single-user mode)'
    )
    
    args = parser.parse_args()
    
    # Determine multi-user mode
    multi_user = args.multi_user and not args.single_user
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.config
    
    # Set up logging
    logger = setup_logging(config)
    logger.info(f"Starting Automated Trading System (multi_user={multi_user})")
    
    # Initialize components
    components = initialize_components(config, multi_user)
    
    # Log system start
    components['compliance_logger'].log_system_event(
        event_type="SYSTEM_START",
        message=f"System started (multi_user={multi_user})"
    )
    
    # Set up trading workflow based on mode
    if multi_user:
        logger.info("Setting up multi-user trading workflow")
        
        # Start the multi-user trading engine
        components['trading_engine'].start_engine()
        logger.info("Multi-user trading engine started")
        
        # Set up scheduled tasks for multi-user mode
        components['scheduler'].schedule_pre_open_job(components['scheduler'].run_pre_open_tasks)
        components['scheduler'].schedule_job(components['scheduler'].run_intraday_scan, 30)  # Every 30 minutes
        components['scheduler'].schedule_daily_job(components['scheduler'].run_eod_tasks, "15:35")
        
        # Start scheduler
        components['scheduler'].start_scheduler()
        logger.info("Multi-user scheduler started")
        
    else:
        logger.info("Setting up single-user trading workflow")
        
        # Register a momentum strategy for single-user mode
        from selector.momentum_strategy import MomentumStrategy
        momentum_strategy = MomentumStrategy()
        components['selector_engine'].register_strategy(momentum_strategy)
        components['selector_engine'].set_active_strategy("MomentumStrategy")
        
        # Log strategy registration
        components['compliance_logger'].log_strategy_event(
            strategy_name="MomentumStrategy",
            event_type="STRATEGY_REGISTERED",
            message="Momentum strategy registered and activated"
        )
        
        # Example: Set up scheduled tasks for single-user mode
        def morning_scan():
            """Perform morning market scan."""
            logger.info("Performing morning market scan")
            components['compliance_logger'].log_system_event(
                event_type="MORNING_SCAN",
                message="Morning market scan initiated"
            )
        
        def eod_process():
            """Perform end-of-day process."""
            logger.info("Performing end-of-day process")
            components['compliance_logger'].log_system_event(
                event_type="EOD_PROCESS",
                message="End-of-day process initiated"
            )
            
            # Generate and send daily report
            eod_report = components['dashboard_reporter'].generate_eod_report()
            components['email_alerting'].send_daily_summary(
                date=eod_report.get('generated_at', 'Unknown'),
                pnl=eod_report.get('performance', {}).get('total_pnl', 0),
                num_trades=eod_report.get('performance', {}).get('total_trades', 0),
                win_rate=eod_report.get('performance', {}).get('win_rate', 0)
            )
        
        # Schedule tasks
        components['scheduler'].schedule_pre_open_job(morning_scan)
        components['scheduler'].schedule_daily_job(eod_process, "15:35")
        
        # Start scheduler
        components['scheduler'].start_scheduler()
        logger.info("Single-user scheduler started")
    
    # Example: Send a test alert
    components['email_alerting'].send_test_email()
    
    # Start the web interface
    logger.info("Starting web interface")
    try:
        # Import and start Flask app
        from web.app import create_app
        app = create_app()
        app.run(
            host=config.get('web', {}).get('host', '0.0.0.0'),
            port=config.get('web', {}).get('port', 5001),
            debug=config.get('web', {}).get('debug', False)
        )
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
    
    # Keep the main thread alive
    try:
        while True:
            # In a real implementation, this would be replaced with actual trading logic
            import time
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down trading system")
        
        # Stop scheduler
        components['scheduler'].stop_scheduler()
        
        # Stop trading engine if in multi-user mode
        if multi_user and components['trading_engine']:
            components['trading_engine'].stop_engine()
        
        # Log system shutdown
        components['compliance_logger'].log_system_event(
            event_type="SYSTEM_SHUTDOWN",
            message="System shutdown initiated"
        )
        
        logger.info("Trading system shutdown complete")


if __name__ == "__main__":
    main()