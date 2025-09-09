"""
Multi-User Trading Scheduler for the Automated Trading System
"""
import schedule
import time
import threading
from datetime import datetime, time as dt_time
from typing import Callable, List, Dict, Any, Optional
import pandas as pd
from trading_engine.multi_user_trading_engine import MultiUserTradingEngine


class MultiUserTradingScheduler:
    """Manages market-aware job scheduling for multiple users."""
    
    def __init__(self, config: Dict[str, Any], trading_engine: MultiUserTradingEngine):
        """
        Initialize the multi-user trading scheduler.
        
        Args:
            config (Dict[str, Any]): Configuration parameters
            trading_engine (MultiUserTradingEngine): Multi-user trading engine instance
        """
        self.config = config
        self.trading_engine = trading_engine
        self.jobs = []
        self.is_running = False
        self.scheduler_thread = None
        self.holidays = set()  # Set of holiday dates (YYYY-MM-DD)
        self.market_open_time = dt_time(9, 15)  # 09:15 IST
        self.market_close_time = dt_time(15, 30)  # 15:30 IST
        self.pre_open_start = dt_time(9, 0)  # 09:00 IST
        self.pre_open_end = dt_time(9, 15)  # 09:15 IST
        
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
    
    def run_pre_open_tasks(self):
        """Run pre-open tasks for all active users."""
        print("Running pre-open tasks for all users...")
        
        # Get all active user sessions
        active_users = self.trading_engine.get_active_users()
        
        for user in active_users:
            user_session = self.trading_engine.get_user_session(user.id)
            if user_session and user_session.is_active:
                try:
                    # Process MOO orders for this user
                    if hasattr(user_session, 'order_router') and user_session.order_router:
                        user_session.order_router.process_moo_orders()
                    
                    # Log pre-open task completion
                    user_session.compliance_logger.log_system_event(
                        event_type="PRE_OPEN_TASKS_COMPLETED",
                        message=f"Pre-open tasks completed for user {user.username}",
                        user_id=user.id
                    )
                    
                except Exception as e:
                    print(f"Error running pre-open tasks for user {user.username}: {e}")
                    user_session.compliance_logger.log_system_event(
                        event_type="PRE_OPEN_TASKS_ERROR",
                        message=f"Error in pre-open tasks for user {user.username}: {str(e)}",
                        user_id=user.id
                    )
    
    def run_intraday_scan(self):
        """Run intraday scanning tasks for all active users."""
        print("Running intraday scan for all users...")
        
        # Get all active user sessions
        active_users = self.trading_engine.get_active_users()
        
        for user in active_users:
            user_session = self.trading_engine.get_user_session(user.id)
            if user_session and user_session.is_active:
                try:
                    # Trigger market scan for this user
                    self.trading_engine._run_market_scan(user_session)
                    
                    # Log intraday scan completion
                    user_session.compliance_logger.log_system_event(
                        event_type="INTRADAY_SCAN_COMPLETED",
                        message=f"Intraday scan completed for user {user.username}",
                        user_id=user.id
                    )
                    
                except Exception as e:
                    print(f"Error running intraday scan for user {user.username}: {e}")
                    user_session.compliance_logger.log_system_event(
                        event_type="INTRADAY_SCAN_ERROR",
                        message=f"Error in intraday scan for user {user.username}: {str(e)}",
                        user_id=user.id
                    )
    
    def run_eod_tasks(self):
        """Run end-of-day tasks for all active users."""
        print("Running EOD tasks for all users...")
        
        # Get all active user sessions
        active_users = self.trading_engine.get_active_users()
        
        for user in active_users:
            user_session = self.trading_engine.get_user_session(user.id)
            if user_session and user_session.is_active:
                try:
                    # Generate and send daily report for this user
                    from reporting.dashboard import DashboardReporter
                    db_manager = self.trading_engine.db_manager
                    dashboard_reporter = DashboardReporter(db_manager)
                    
                    # Generate EOD report for this user
                    eod_report = dashboard_reporter.generate_eod_report(user_id=user.id)
                    
                    # Send daily summary email for this user
                    user_session.email_alerting.send_daily_summary(
                        date=eod_report.get('generated_at', 'Unknown'),
                        pnl=eod_report.get('performance', {}).get('total_pnl', 0),
                        num_trades=eod_report.get('performance', {}).get('total_trades', 0),
                        win_rate=eod_report.get('performance', {}).get('win_rate', 0),
                        user_email=user.email
                    )
                    
                    # Log EOD task completion
                    user_session.compliance_logger.log_system_event(
                        event_type="EOD_TASKS_COMPLETED",
                        message=f"EOD tasks completed for user {user.username}",
                        user_id=user.id
                    )
                    
                except Exception as e:
                    print(f"Error running EOD tasks for user {user.username}: {e}")
                    user_session.compliance_logger.log_system_event(
                        event_type="EOD_TASKS_ERROR",
                        message=f"Error in EOD tasks for user {user.username}: {str(e)}",
                        user_id=user.id
                    )
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get the current status of the scheduler.
        
        Returns:
            Dict[str, Any]: Scheduler status information
        """
        return {
            'is_running': self.is_running,
            'scheduled_jobs_count': len(self.jobs),
            'market_open': self.is_market_open(),
            'pre_open': self.is_pre_open(),
            'holidays': list(self.holidays),
            'jobs': self.jobs
        }
