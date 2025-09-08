"""
Trading Scheduler for the Automated Trading System
"""
import schedule
import time
import threading
from datetime import datetime, time as dt_time
from typing import Callable, List, Dict, Any
import pandas as pd


class TradingScheduler:
    """Manages market-aware job scheduling with holiday awareness."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading scheduler.
        
        Args:
            config (Dict[str, Any]): Configuration parameters
        """
        self.config = config
        self.jobs = []
        self.is_running = False
        self.scheduler_thread = None
        self.holidays = set()  # Set of holiday dates (YYYY-MM-DD)
        self.market_open_time = dt_time(9, 15)  # 09:15 IST
        self.market_close_time = dt_time(15, 30)  # 15:30 IST
        self.pre_open_start = dt_time(9, 0)  # 09:00 IST
        self.pre_open_end = dt_time(9, 15)  # 09:15 IST
        self.order_router = None  # Reference to order router for MOO processing
        
        # Parse times from config if available
        self._parse_config_times()
    
    def _parse_config_times(self):
        """Parse market timing configuration."""
        trading_config = self.config.get('trading', {})
        
        # Parse market open time
        market_open = trading_config.get('market_open', '09:15')
        if ':' in market_open:
            hour, minute = map(int, market_open.split(':'))
            self.market_open_time = dt_time(hour, minute)
        
        # Parse market close time
        market_close = trading_config.get('market_close', '15:30')
        if ':' in market_close:
            hour, minute = map(int, market_close.split(':'))
            self.market_close_time = dt_time(hour, minute)
        
        # Parse pre-open start time
        pre_open_start = trading_config.get('pre_open_start', '09:00')
        if ':' in pre_open_start:
            hour, minute = map(int, pre_open_start.split(':'))
            self.pre_open_start = dt_time(hour, minute)
        
        # Parse pre-open end time
        pre_open_end = trading_config.get('pre_open_end', '09:15')
        if ':' in pre_open_end:
            hour, minute = map(int, pre_open_end.split(':'))
            self.pre_open_end = dt_time(hour, minute)
    
    def add_holiday(self, date_str: str):
        """
        Add a holiday to the scheduler.
        
        Args:
            date_str (str): Holiday date in YYYY-MM-DD format
        """
        self.holidays.add(date_str)
    
    def remove_holiday(self, date_str: str):
        """
        Remove a holiday from the scheduler.
        
        Args:
            date_str (str): Holiday date in YYYY-MM-DD format
        """
        self.holidays.discard(date_str)
    
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')
        
        # Check if today is a holiday
        if today_str in self.holidays:
            return False
        
        # Check if today is a weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check market hours
        current_time = now.time()
        return self.market_open_time <= current_time <= self.market_close_time
    
    def is_pre_open(self) -> bool:
        """
        Check if the market is in pre-open session.
        
        Returns:
            bool: True if market is in pre-open session, False otherwise
        """
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')
        
        # Check if today is a holiday
        if today_str in self.holidays:
            return False
        
        # Check if today is a weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check pre-open hours
        current_time = now.time()
        return self.pre_open_start <= current_time <= self.pre_open_end
    
    def schedule_job(self, job_func: Callable, interval_minutes: int, 
                     start_time: str = None, end_time: str = None, 
                     days: List[str] = None) -> str:
        """
        Schedule a job to run at specified intervals.
        
        Args:
            job_func (Callable): Function to execute
            interval_minutes (int): Interval in minutes
            start_time (str, optional): Start time in HH:MM format
            end_time (str, optional): End time in HH:MM format
            days (List[str], optional): Days to run ('mon', 'tue', etc.)
            
        Returns:
            str: Job ID
        """
        job_id = f"job_{len(self.jobs) + 1}"
        
        # Create job wrapper that checks market hours
        def market_aware_job():
            if self.is_market_open():
                job_func()
        
        # Schedule the job
        if days:
            for day in days:
                if day.lower() == 'mon':
                    schedule.every(interval_minutes).minutes.monday.at(start_time or "09:15").do(market_aware_job)
                elif day.lower() == 'tue':
                    schedule.every(interval_minutes).minutes.tuesday.at(start_time or "09:15").do(market_aware_job)
                elif day.lower() == 'wed':
                    schedule.every(interval_minutes).minutes.wednesday.at(start_time or "09:15").do(market_aware_job)
                elif day.lower() == 'thu':
                    schedule.every(interval_minutes).minutes.thursday.at(start_time or "09:15").do(market_aware_job)
                elif day.lower() == 'fri':
                    schedule.every(interval_minutes).minutes.friday.at(start_time or "09:15").do(market_aware_job)
        else:
            # Schedule for all weekdays
            job = schedule.every(interval_minutes).minutes.do(market_aware_job)
            if start_time:
                job = job.at(start_time)
        
        self.jobs.append({
            'id': job_id,
            'function': job_func,
            'interval': interval_minutes,
            'start_time': start_time,
            'end_time': end_time,
            'days': days
        })
        
        return job_id
    
    def schedule_pre_open_job(self, job_func: Callable) -> str:
        """
        Schedule a job to run during pre-open session.
        
        Args:
            job_func (Callable): Function to execute
            
        Returns:
            str: Job ID
        """
        job_id = f"pre_open_job_{len(self.jobs) + 1}"
        
        # Schedule for 09:05 AM on weekdays
        schedule.every().monday.at("09:05").do(job_func)
        schedule.every().tuesday.at("09:05").do(job_func)
        schedule.every().wednesday.at("09:05").do(job_func)
        schedule.every().thursday.at("09:05").do(job_func)
        schedule.every().friday.at("09:05").do(job_func)
        
        self.jobs.append({
            'id': job_id,
            'function': job_func,
            'type': 'pre_open'
        })
        
        return job_id
    
    def schedule_daily_job(self, job_func: Callable, time_str: str = "09:00") -> str:
        """
        Schedule a daily job.
        
        Args:
            job_func (Callable): Function to execute
            time_str (str): Time to run in HH:MM format
            
        Returns:
            str: Job ID
        """
        job_id = f"daily_job_{len(self.jobs) + 1}"
        
        # Schedule for weekdays
        schedule.every().monday.at(time_str).do(job_func)
        schedule.every().tuesday.at(time_str).do(job_func)
        schedule.every().wednesday.at(time_str).do(job_func)
        schedule.every().thursday.at(time_str).do(job_func)
        schedule.every().friday.at(time_str).do(job_func)
        
        self.jobs.append({
            'id': job_id,
            'function': job_func,
            'type': 'daily',
            'time': time_str
        })
        
        return job_id
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a scheduled job.
        
        Args:
            job_id (str): Job ID to cancel
            
        Returns:
            bool: True if job was cancelled, False if not found
        """
        for i, job in enumerate(self.jobs):
            if job['id'] == job_id:
                # Remove from schedule
                schedule.clear(job_id)
                # Remove from jobs list
                del self.jobs[i]
                return True
        return False
    
    def get_scheduled_jobs(self) -> List[Dict[str, Any]]:
        """
        Get list of scheduled jobs.
        
        Returns:
            List[Dict[str, Any]]: List of scheduled jobs
        """
        return self.jobs.copy()
    
    def start_scheduler(self):
        """Start the scheduler in a separate thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        schedule.clear()
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    def set_order_router(self, order_router):
        """
        Set the order router for MOO order processing.
        
        Args:
            order_router: Order router instance
        """
        self.order_router = order_router
    
    def run_pre_open_tasks(self):
        """Run pre-open tasks."""
        print("Running pre-open tasks...")
        # Process MOO orders when market opens
        if self.order_router:
            self.order_router.process_moo_orders()
        # In a real implementation, this would run actual pre-open tasks
    
    def run_intraday_scan(self):
        """Run intraday scanning tasks."""
        print("Running intraday scan...")
        # In a real implementation, this would run actual scanning tasks
    
    def run_eod_tasks(self):
        """Run end-of-day tasks."""
        print("Running EOD tasks...")
        # In a real implementation, this would run actual EOD tasks


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'trading': {
            'market_open': '09:15',
            'market_close': '15:30',
            'pre_open_start': '09:00',
            'pre_open_end': '09:15'
        }
    }
    
    # Create scheduler
    scheduler = TradingScheduler(config)
    
    # Add some holidays
    scheduler.add_holiday('2023-01-26')  # Republic Day
    scheduler.add_holiday('2023-08-15')  # Independence Day
    
    # Schedule some jobs
    scheduler.schedule_pre_open_job(scheduler.run_pre_open_tasks)
    scheduler.schedule_job(scheduler.run_intraday_scan, 30)  # Every 30 minutes
    scheduler.schedule_daily_job(scheduler.run_eod_tasks, "15:35")  # After market close
    
    # Start scheduler
    scheduler.start_scheduler()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop scheduler on Ctrl+C
        scheduler.stop_scheduler()
        print("Scheduler stopped.")