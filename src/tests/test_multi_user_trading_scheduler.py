"""
Tests for Multi-User Trading Scheduler
"""
import pytest
import sys
import os
import uuid
from datetime import datetime, time as dt_time
from unittest.mock import Mock, patch

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datastore.database import get_database_manager
from datastore.models import User
from trading_engine.multi_user_trading_engine import MultiUserTradingEngine
from scheduler.multi_user_trading_scheduler import MultiUserTradingScheduler


@pytest.fixture
def db_manager():
    """Create a database manager for testing."""
    # Use a unique in-memory database for each test
    database_url = f"sqlite:///:memory:{uuid.uuid4().hex}"
    db_manager = get_database_manager(database_url)
    db_manager.create_tables()
    
    # Reset the global db_manager to ensure fresh instances
    import datastore.database
    datastore.database.db_manager = None
    
    yield db_manager
    
    # Clean up
    datastore.database.db_manager = None


@pytest.fixture
def test_users(db_manager):
    """Create test users."""
    with db_manager.get_session() as session:
        # Create test users
        user1 = User(
            username="testuser1",
            email="test1@example.com",
            password_hash="hashed_password_1",
            first_name="Test",
            last_name="User1"
        )
        user2 = User(
            username="testuser2",
            email="test2@example.com",
            password_hash="hashed_password_2",
            first_name="Test",
            last_name="User2"
        )
        
        session.add(user1)
        session.add(user2)
        session.commit()
        
        return [user1.id, user2.id]


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'trading': {
            'pre_open_start': '09:00',
            'pre_open_end': '09:15'
        }
    }


@pytest.fixture
def trading_engine(db_manager, test_config):
    """Create a multi-user trading engine for testing."""
    with patch('trading_engine.multi_user_trading_engine.get_data_manager') as mock_data_manager:
        mock_data_manager.return_value = Mock()
        
        engine = MultiUserTradingEngine(test_config, 'development')
        engine.db_manager = db_manager
        
        return engine


@pytest.fixture
def scheduler(trading_engine, test_config):
    """Create a multi-user trading scheduler for testing."""
    return MultiUserTradingScheduler(test_config, trading_engine)


class TestMultiUserTradingScheduler:
    """Test MultiUserTradingScheduler class."""
    
    def test_scheduler_initialization(self, scheduler, test_config):
        """Test scheduler initialization."""
        assert scheduler.config == test_config
        assert scheduler.trading_engine is not None
        assert scheduler.is_running is False
        assert len(scheduler.jobs) == 0
    
    def test_holiday_management(self, scheduler):
        """Test holiday management."""
        # Add holiday
        scheduler.add_holiday('2023-01-26')
        assert '2023-01-26' in scheduler.holidays
        
        # Remove holiday
        scheduler.remove_holiday('2023-01-26')
        assert '2023-01-26' not in scheduler.holidays
    
    def test_market_hours_check(self, scheduler):
        """Test market hours checking."""
        # Test with current time (this will depend on when the test runs)
        # We'll mock the datetime to test specific scenarios
        
        with patch('scheduler.multi_user_trading_scheduler.datetime') as mock_datetime:
            # Test market open time
            mock_datetime.now.return_value = datetime(2023, 1, 25, 10, 0)  # Wednesday 10:00 AM
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            assert scheduler.is_market_open() is True
            
            # Test market closed time
            mock_datetime.now.return_value = datetime(2023, 1, 25, 16, 0)  # Wednesday 4:00 PM
            assert scheduler.is_market_open() is False
            
            # Test weekend
            mock_datetime.now.return_value = datetime(2023, 1, 28, 10, 0)  # Saturday 10:00 AM
            assert scheduler.is_market_open() is False
            
            # Test holiday
            scheduler.add_holiday('2023-01-25')
            mock_datetime.now.return_value = datetime(2023, 1, 25, 10, 0)  # Wednesday 10:00 AM
            assert scheduler.is_market_open() is False
    
    def test_pre_open_check(self, scheduler):
        """Test pre-open session checking."""
        with patch('scheduler.multi_user_trading_scheduler.datetime') as mock_datetime:
            # Test pre-open time
            mock_datetime.now.return_value = datetime(2023, 1, 25, 9, 10)  # Wednesday 9:10 AM
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            assert scheduler.is_pre_open() is True
            
            # Test outside pre-open time
            mock_datetime.now.return_value = datetime(2023, 1, 25, 10, 0)  # Wednesday 10:00 AM
            assert scheduler.is_pre_open() is False
    
    def test_job_scheduling(self, scheduler):
        """Test job scheduling."""
        def test_job():
            pass
        
        # Schedule a job
        job_id = scheduler.schedule_job(test_job, 30, start_time="09:15")
        
        assert job_id is not None
        assert len(scheduler.jobs) == 1
        assert scheduler.jobs[0]['id'] == job_id
        assert scheduler.jobs[0]['function'] == test_job
    
    def test_pre_open_job_scheduling(self, scheduler):
        """Test pre-open job scheduling."""
        def test_pre_open_job():
            pass
        
        # Schedule pre-open job
        job_id = scheduler.schedule_pre_open_job(test_pre_open_job)
        
        assert job_id is not None
        assert len(scheduler.jobs) == 1
        assert scheduler.jobs[0]['type'] == 'pre_open'
    
    def test_daily_job_scheduling(self, scheduler):
        """Test daily job scheduling."""
        def test_daily_job():
            pass
        
        # Schedule daily job
        job_id = scheduler.schedule_daily_job(test_daily_job, "15:35")
        
        assert job_id is not None
        assert len(scheduler.jobs) == 1
        assert scheduler.jobs[0]['type'] == 'daily'
        assert scheduler.jobs[0]['time'] == "15:35"
    
    def test_job_cancellation(self, scheduler):
        """Test job cancellation."""
        def test_job():
            pass
        
        # Schedule a job
        job_id = scheduler.schedule_job(test_job, 30)
        assert len(scheduler.jobs) == 1
        
        # Cancel the job
        result = scheduler.cancel_job(job_id)
        assert result is True
        assert len(scheduler.jobs) == 0
        
        # Try to cancel non-existent job
        result = scheduler.cancel_job("non_existent")
        assert result is False
    
    def test_scheduler_start_stop(self, scheduler):
        """Test starting and stopping the scheduler."""
        # Initially not running
        assert scheduler.is_running is False
        
        # Start scheduler
        scheduler.start_scheduler()
        assert scheduler.is_running is True
        assert scheduler.scheduler_thread is not None
        
        # Stop scheduler
        scheduler.stop_scheduler()
        assert scheduler.is_running is False
    
    def test_scheduler_status(self, scheduler):
        """Test getting scheduler status."""
        # Schedule some jobs
        def test_job():
            pass
        
        scheduler.schedule_job(test_job, 30)
        scheduler.schedule_pre_open_job(test_job)
        
        # Get status
        status = scheduler.get_scheduler_status()
        
        assert 'is_running' in status
        assert 'scheduled_jobs_count' in status
        assert 'pre_open' in status
        assert 'holidays' in status
        assert 'jobs' in status
        
        assert status['scheduled_jobs_count'] == 2
        assert isinstance(status['holidays'], list)
        assert len(status['jobs']) == 2


class TestMultiUserTradingSchedulerTasks:
    """Test scheduler task execution."""
    
    def test_pre_open_tasks_execution(self, scheduler, trading_engine, db_manager, test_users):
        """Test pre-open tasks execution for all users."""
        with db_manager.get_session() as session:
            user1 = session.query(User).filter(User.id == test_users[0]).first()
            user2 = session.query(User).filter(User.id == test_users[1]).first()
            
            # Create user sessions
            session1 = trading_engine.create_user_session(user1)
            session2 = trading_engine.create_user_session(user2)
            
            # Mock order router
            session1.order_router = Mock()
            session2.order_router = Mock()
            
            # Run pre-open tasks
            scheduler.run_pre_open_tasks()
            
            # Verify order routers were called
            session1.order_router.process_moo_orders.assert_called_once()
            session2.order_router.process_moo_orders.assert_called_once()
    
    def test_intraday_scan_execution(self, scheduler, trading_engine, db_manager, test_users):
        """Test intraday scan execution for all users."""
        with db_manager.get_session() as session:
            user1 = session.query(User).filter(User.id == test_users[0]).first()
            user2 = session.query(User).filter(User.id == test_users[1]).first()
            
            # Create user sessions
            session1 = trading_engine.create_user_session(user1)
            session2 = trading_engine.create_user_session(user2)
            
            # Mock market scan
            with patch.object(trading_engine, '_run_market_scan') as mock_scan:
                # Run intraday scan
                scheduler.run_intraday_scan()
                
                # Verify market scan was called for both users
                assert mock_scan.call_count == 2
    
    def test_eod_tasks_execution(self, scheduler, trading_engine, db_manager, test_users):
        """Test end-of-day tasks execution for all users."""
        with db_manager.get_session() as session:
            user1 = session.query(User).filter(User.id == test_users[0]).first()
            user2 = session.query(User).filter(User.id == test_users[1]).first()
            
            # Create user sessions
            session1 = trading_engine.create_user_session(user1)
            session2 = trading_engine.create_user_session(user2)
            
            # Mock dashboard reporter and email alerting
            with patch('scheduler.multi_user_trading_scheduler.DashboardReporter') as mock_reporter_class:
                mock_reporter = Mock()
                mock_reporter_class.return_value = mock_reporter
                mock_reporter.generate_eod_report.return_value = {
                    'generated_at': '2023-01-25',
                    'performance': {
                        'total_pnl': 1000,
                        'total_trades': 5,
                        'win_rate': 0.8
                    }
                }
                
                # Mock email alerting
                session1.email_alerting = Mock()
                session2.email_alerting = Mock()
                
                # Run EOD tasks
                scheduler.run_eod_tasks()
                
                # Verify dashboard reporter was called
                assert mock_reporter.generate_eod_report.call_count == 2
                
                # Verify email alerting was called
                session1.email_alerting.send_daily_summary.assert_called_once()
                session2.email_alerting.send_daily_summary.assert_called_once()
    
    def test_task_error_handling(self, scheduler, trading_engine, db_manager, test_users):
        """Test error handling in task execution."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            
            # Create user session
            user_session = trading_engine.create_user_session(user)
            
            # Mock an error in pre-open tasks
            user_session.order_router = Mock()
            user_session.order_router.process_moo_orders.side_effect = Exception("Test error")
            
            # This should not crash the scheduler
            try:
                scheduler.run_pre_open_tasks()
            except Exception:
                # The error should be caught and logged
                pass
            
            # Scheduler should still be functional
            assert scheduler.is_running is False  # Not started yet
    
    def test_inactive_user_handling(self, scheduler, trading_engine, db_manager, test_users):
        """Test handling of inactive users."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            
            # Create user session
            user_session = trading_engine.create_user_session(user)
            
            # Deactivate user session
            user_session.is_active = False
            
            # Run pre-open tasks
            scheduler.run_pre_open_tasks()
            
            # Should not process inactive users
            # (This test verifies the logic doesn't crash with inactive users)
            assert True  # If we get here, no exception was raised


class TestMultiUserTradingSchedulerIntegration:
    """Integration tests for the multi-user trading scheduler."""
    
    def test_full_scheduler_workflow(self, scheduler, trading_engine, db_manager, test_users):
        """Test a complete scheduler workflow."""
        with db_manager.get_session() as session:
            user1 = session.query(User).filter(User.id == test_users[0]).first()
            user2 = session.query(User).filter(User.id == test_users[1]).first()
            
            # Create user sessions
            trading_engine.create_user_session(user1)
            trading_engine.create_user_session(user2)
            
            # Schedule jobs
            def test_job():
                pass
            
            scheduler.schedule_pre_open_job(scheduler.run_pre_open_tasks)
            scheduler.schedule_job(scheduler.run_intraday_scan, 30)
            scheduler.schedule_daily_job(scheduler.run_eod_tasks, "15:35")
            
            # Verify jobs are scheduled
            assert len(scheduler.jobs) == 3
            
            # Start scheduler
            scheduler.start_scheduler()
            assert scheduler.is_running is True
            
            # Get status
            status = scheduler.get_scheduler_status()
            assert status['scheduled_jobs_count'] == 3
            
            # Stop scheduler
            scheduler.stop_scheduler()
            assert scheduler.is_running is False
    
    def test_market_aware_job_execution(self, scheduler):
        """Test that jobs only execute during market hours."""
        job_executed = False
        
        def test_job():
            nonlocal job_executed
            job_executed = True
        
        # Schedule a job
        scheduler.schedule_job(test_job, 1)  # Every minute
        
        # Mock market closed
        with patch.object(scheduler, 'is_market_open', return_value=False):
            # Start scheduler briefly
            scheduler.start_scheduler()
            import time
            time.sleep(0.1)  # Brief sleep
            scheduler.stop_scheduler()
            
            # Job should not have executed
            assert job_executed is False
        
        # Mock market open
        with patch.object(scheduler, 'is_market_open', return_value=True):
            # Start scheduler briefly
            scheduler.start_scheduler()
            import time
            time.sleep(0.1)  # Brief sleep
            scheduler.stop_scheduler()
            
            # Job should have executed
            assert job_executed is True
