"""
Scheduler Service - Background task scheduling for token refresh and other periodic tasks
"""
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any
import schedule

logger = logging.getLogger(__name__)

class SchedulerService:
    """Service for managing background scheduled tasks."""
    
    def __init__(self):
        self.running = False
        self.scheduler_thread = None
        self.tasks = {}  # Store task information
        self._stop_event = threading.Event()
    
    def start(self):
        """Start the scheduler service."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self._stop_event.clear()
        
        # Start the scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Scheduler service started")
    
    def stop(self):
        """Stop the scheduler service."""
        if not self.running:
            logger.warning("Scheduler is not running")
            return
        
        self.running = False
        self._stop_event.set()
        
        # Wait for scheduler thread to finish
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        # Clear all scheduled jobs
        schedule.clear()
        
        logger.info("Scheduler service stopped")
    
    def _run_scheduler(self):
        """Main scheduler loop."""
        logger.info("Scheduler thread started")
        
        while self.running and not self._stop_event.is_set():
            try:
                # Run pending jobs
                schedule.run_pending()
                
                # Sleep for 1 minute before checking again
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)  # Wait before retrying
        
        logger.info("Scheduler thread stopped")
    
    def schedule_token_refresh_check(self, user_id: int, broker_name: str, interval_minutes: int = 30):
        """
        Schedule periodic token refresh checks for a user and broker.
        
        Args:
            user_id (int): User ID
            broker_name (str): Broker name
            interval_minutes (int): Check interval in minutes
        """
        task_id = f"token_refresh_{broker_name}_{user_id}"
        
        def check_and_refresh_token():
            try:
                from .token_manager_service import get_token_manager
                
                token_manager = get_token_manager()
                token_data = token_manager.get_valid_token(user_id, broker_name)
                
                if token_data:
                    # Check if token will expire soon
                    expiry_time = token_manager.get_token_expiry_time(token_data['access_token'])
                    if expiry_time:
                        time_until_expiry = expiry_time - datetime.now()
                        check_interval = timedelta(minutes=interval_minutes)
                        
                        if time_until_expiry <= check_interval:
                            logger.info(f"Token for user {user_id}, broker {broker_name} will expire soon. Refreshing...")
                            token_manager.refresh_token(user_id, broker_name)
                
            except Exception as e:
                logger.error(f"Error in scheduled token refresh check for user {user_id}, broker {broker_name}: {e}")
        
        # Schedule the job
        schedule.every(interval_minutes).minutes.do(check_and_refresh_token)
        
        # Store task information
        self.tasks[task_id] = {
            'type': 'token_refresh',
            'user_id': user_id,
            'broker_name': broker_name,
            'interval_minutes': interval_minutes,
            'created_at': datetime.now()
        }
        
        logger.info(f"Scheduled token refresh check for user {user_id}, broker {broker_name} every {interval_minutes} minutes")
    
    def unschedule_token_refresh_check(self, user_id: int, broker_name: str):
        """
        Unschedule token refresh checks for a user and broker.
        
        Args:
            user_id (int): User ID
            broker_name (str): Broker name
        """
        task_id = f"token_refresh_{broker_name}_{user_id}"
        
        if task_id in self.tasks:
            # Remove the task
            del self.tasks[task_id]
            
            # Clear all jobs and reschedule remaining ones
            schedule.clear()
            for task_info in self.tasks.values():
                if task_info['type'] == 'token_refresh':
                    self.schedule_token_refresh_check(
                        task_info['user_id'],
                        task_info['broker_name'],
                        task_info['interval_minutes']
                    )
            
            logger.info(f"Unscheduled token refresh check for user {user_id}, broker {broker_name}")
        else:
            logger.warning(f"No scheduled token refresh check found for user {user_id}, broker {broker_name}")
    
    def schedule_api_health_check(self, endpoint: str, interval_minutes: int = 15):
        """
        Schedule periodic API health checks.
        
        Args:
            endpoint (str): API endpoint to check
            interval_minutes (int): Check interval in minutes
        """
        task_id = f"health_check_{endpoint}"
        
        def health_check():
            try:
                import requests
                
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    logger.debug(f"Health check passed for {endpoint}")
                else:
                    logger.warning(f"Health check failed for {endpoint}: HTTP {response.status_code}")
                
            except Exception as e:
                logger.error(f"Health check error for {endpoint}: {e}")
        
        # Schedule the job
        schedule.every(interval_minutes).minutes.do(health_check)
        
        # Store task information
        self.tasks[task_id] = {
            'type': 'health_check',
            'endpoint': endpoint,
            'interval_minutes': interval_minutes,
            'created_at': datetime.now()
        }
        
        logger.info(f"Scheduled health check for {endpoint} every {interval_minutes} minutes")
    
    def schedule_data_cleanup(self, interval_hours: int = 24):
        """
        Schedule periodic data cleanup tasks.

        Args:
            interval_hours (int): Cleanup interval in hours
        """
        task_id = "data_cleanup"

        def cleanup_data():
            try:
                from src.models.database import get_database_manager
                from src.models.models import Order, Trade
                from datetime import datetime, timedelta

                db_manager = get_database_manager()

                with db_manager.get_session() as session:
                    # Clean up old completed orders (older than 30 days)
                    cutoff_date = datetime.now() - timedelta(days=30)
                    old_orders = session.query(Order).filter(
                        Order.order_status == 'COMPLETE',
                        Order.created_at < cutoff_date
                    ).all()

                    for order in old_orders:
                        session.delete(order)

                    # Clean up old trades (older than 30 days)
                    old_trades = session.query(Trade).filter(
                        Trade.created_at < cutoff_date
                    ).all()

                    for trade in old_trades:
                        session.delete(trade)

                    session.commit()

                    logger.info(f"Data cleanup completed: removed {len(old_orders)} old orders and {len(old_trades)} old trades")

            except Exception as e:
                logger.error(f"Error in data cleanup: {e}")

        # Schedule the job
        schedule.every(interval_hours).hours.do(cleanup_data)

        # Store task information
        self.tasks[task_id] = {
            'type': 'data_cleanup',
            'interval_hours': interval_hours,
            'created_at': datetime.now()
        }

        logger.info(f"Scheduled data cleanup every {interval_hours} hours")

    def schedule_historical_data_update(self, user_id: int = 1, interval_hours: int = 6, max_stocks: int = 50):
        """
        Schedule periodic historical data updates.

        Args:
            user_id (int): User ID for API access
            interval_hours (int): Update interval in hours
            max_stocks (int): Maximum stocks to update per run
        """
        task_id = "historical_data_update"

        def update_historical_data():
            try:
                logger.info("🔄 Starting scheduled historical data update...")

                from ..data.historical_data_service import HistoricalDataService

                hist_service = HistoricalDataService()

                # Fetch historical data for active stocks
                result = hist_service.fetch_historical_data_bulk(
                    user_id=user_id,
                    days=365,  # Fetch up to 1 year of data
                    max_stocks=max_stocks
                )

                if result.get('success'):
                    logger.info(f"📊 Historical data update completed: {result['successful']} stocks updated out of {result['processed']} processed")
                else:
                    logger.error(f"❌ Historical data update failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error in scheduled historical data update: {e}")

        # Schedule the job
        schedule.every(interval_hours).hours.do(update_historical_data)

        # Store task information
        self.tasks[task_id] = {
            'type': 'historical_data_update',
            'user_id': user_id,
            'interval_hours': interval_hours,
            'max_stocks': max_stocks,
            'created_at': datetime.now()
        }

        logger.info(f"📅 Scheduled historical data updates every {interval_hours} hours for max {max_stocks} stocks")

    def schedule_technical_indicators_calculation(self, interval_hours: int = 4, max_symbols: int = 100):
        """
        Schedule periodic technical indicators calculation.

        Args:
            interval_hours (int): Calculation interval in hours
            max_symbols (int): Maximum symbols to process per run
        """
        task_id = "technical_indicators_calculation"

        def calculate_indicators():
            try:
                logger.info("🧮 Starting scheduled technical indicators calculation...")

                from ..data.technical_indicators_service import TechnicalIndicatorsService

                indicators_service = TechnicalIndicatorsService()

                # Calculate indicators for symbols with sufficient historical data
                result = indicators_service.calculate_indicators_bulk(
                    symbols=None,  # Auto-select symbols with sufficient data
                    max_symbols=max_symbols
                )

                if result.get('success'):
                    logger.info(f"📈 Technical indicators calculation completed: {result['successful']} symbols processed out of {result['processed']}")

                    # Log any errors for debugging
                    if result.get('errors'):
                        logger.warning(f"⚠️ Calculation errors occurred: {len(result['errors'])} errors")
                        for error in result['errors'][:5]:  # Log first 5 errors
                            logger.warning(f"  - {error}")
                else:
                    logger.error(f"❌ Technical indicators calculation failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error in scheduled technical indicators calculation: {e}")

        # Schedule the job
        schedule.every(interval_hours).hours.do(calculate_indicators)

        # Store task information
        self.tasks[task_id] = {
            'type': 'technical_indicators_calculation',
            'interval_hours': interval_hours,
            'max_symbols': max_symbols,
            'created_at': datetime.now()
        }

        logger.info(f"📅 Scheduled technical indicators calculation every {interval_hours} hours for max {max_symbols} symbols")

    def schedule_market_data_refresh(self, user_id: int = 1, interval_minutes: int = 30):
        """
        Schedule periodic market data refresh for current prices and volumes.

        Args:
            user_id (int): User ID for API access
            interval_minutes (int): Refresh interval in minutes
        """
        task_id = "market_data_refresh"

        def refresh_market_data():
            try:
                logger.info("💹 Starting scheduled market data refresh...")

                from ..data.stock_initialization_service import StockInitializationService

                init_service = StockInitializationService()

                # Update current prices and volumes for active stocks
                result = init_service.update_current_market_data(
                    user_id=user_id,
                    max_stocks=200  # Update current data for more stocks
                )

                if result.get('success'):
                    logger.info(f"💰 Market data refresh completed: {result.get('updated', 0)} stocks updated")
                else:
                    logger.warning(f"⚠️ Market data refresh had issues: {result.get('message', 'Unknown issue')}")

            except Exception as e:
                logger.error(f"Error in scheduled market data refresh: {e}")

        # Schedule the job
        schedule.every(interval_minutes).minutes.do(refresh_market_data)

        # Store task information
        self.tasks[task_id] = {
            'type': 'market_data_refresh',
            'user_id': user_id,
            'interval_minutes': interval_minutes,
            'created_at': datetime.now()
        }

        logger.info(f"📅 Scheduled market data refresh every {interval_minutes} minutes")

    def schedule_all_data_maintenance_tasks(self, user_id: int = 1):
        """
        Schedule all data maintenance tasks with optimal timing.

        Args:
            user_id (int): User ID for API access
        """
        logger.info("🔧 Setting up comprehensive data maintenance schedule...")

        # Market data refresh - every 30 minutes during market hours
        self.schedule_market_data_refresh(user_id=user_id, interval_minutes=30)

        # Technical indicators calculation - every 4 hours
        self.schedule_technical_indicators_calculation(interval_hours=4, max_symbols=100)

        # Historical data update - every 6 hours (to catch new daily data)
        self.schedule_historical_data_update(user_id=user_id, interval_hours=6, max_stocks=50)

        # Data cleanup - daily
        self.schedule_data_cleanup(interval_hours=24)

        logger.info("✅ All data maintenance tasks scheduled successfully!")
    
    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """
        Get information about all scheduled tasks.
        
        Returns:
            List[Dict]: List of task information
        """
        return list(self.tasks.values())
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a specific task.
        
        Args:
            task_id (str): Task ID
            
        Returns:
            Dict: Task status information
        """
        if task_id not in self.tasks:
            return {'error': 'Task not found'}
        
        task_info = self.tasks[task_id]
        
        return {
            'task_id': task_id,
            'type': task_info['type'],
            'created_at': task_info['created_at'].isoformat(),
            'status': 'scheduled',
            'next_run': 'N/A'  # schedule library doesn't provide easy access to next run time
        }


# Global scheduler instance
_scheduler = None

def get_scheduler() -> SchedulerService:
    """Get global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SchedulerService()
    return _scheduler
