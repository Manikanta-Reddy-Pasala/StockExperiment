"""
Tests for Multi-User Trading Engine
"""
import pytest
import sys
import os
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datastore.database import get_database_manager, DatabaseManager
from datastore.models import User, Configuration
from trading_engine.multi_user_trading_engine import MultiUserTradingEngine, UserTradingSession


@pytest.fixture
def db_manager():
    """Create a database manager for testing."""
    # Use a unique in-memory database for each test
    database_url = f"sqlite:///:memory:{uuid.uuid4().hex}"
    
    # Set the environment variable to override any existing DATABASE_URL
    import os
    old_database_url = os.environ.get('DATABASE_URL')
    os.environ['DATABASE_URL'] = database_url
    
    # Reset the global db_manager to ensure fresh instances
    import datastore.database
    datastore.database.db_manager = None
    
    db_manager = get_database_manager()
    db_manager.create_tables()
    
    yield db_manager
    
    # Clean up
    datastore.database.db_manager = None
    if old_database_url:
        os.environ['DATABASE_URL'] = old_database_url
    else:
        os.environ.pop('DATABASE_URL', None)


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
        'risk': {
            'max_capital_per_trade': 0.01,
            'max_concurrent_trades': 10,
            'daily_loss_limit': 0.02,
            'single_name_exposure_limit': 0.05
        },
        'trading': {
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


class TestUserTradingSession:
    """Test UserTradingSession class."""
    
    def test_user_trading_session_creation(self, db_manager, test_config, test_users):
        """Test creating a user trading session."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            # Detach the user from the session to avoid DetachedInstanceError
            session.expunge(user)
            
            with patch('trading_engine.multi_user_trading_engine.get_data_manager') as mock_data_manager:
                mock_data_manager.return_value = Mock()
                
                user_session = UserTradingSession(user, test_config, 'development')
                
                assert user_session.user.id == user.id
                assert user_session.config == test_config
                assert user_session.mode == 'development'
                assert user_session.is_active is True
                assert user_session.trading_state is not None
    
    def test_user_trading_session_activity_update(self, db_manager, test_config, test_users):
        """Test updating user activity."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            # Detach the user from the session to avoid DetachedInstanceError
            session.expunge(user)
            
            with patch('trading_engine.multi_user_trading_engine.get_data_manager') as mock_data_manager:
                mock_data_manager.return_value = Mock()
                
                user_session = UserTradingSession(user, test_config, 'development')
                initial_activity = user_session.last_activity
                
                # Update activity
                user_session.update_activity()
                
                assert user_session.last_activity > initial_activity
    
    def test_user_trading_session_deactivation(self, db_manager, test_config, test_users):
        """Test deactivating a user trading session."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            # Detach the user from the session to avoid DetachedInstanceError
            session.expunge(user)
            
            with patch('trading_engine.multi_user_trading_engine.get_data_manager') as mock_data_manager:
                mock_data_manager.return_value = Mock()
                
                user_session = UserTradingSession(user, test_config, 'development')
                assert user_session.is_active is True
                
                user_session.deactivate()
                assert user_session.is_active is False
    
    def test_user_trading_session_state_management(self, db_manager, test_config, test_users):
        """Test trading state management."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            # Detach the user from the session to avoid DetachedInstanceError
            session.expunge(user)
            
            with patch('trading_engine.multi_user_trading_engine.get_data_manager') as mock_data_manager:
                mock_data_manager.return_value = Mock()
                
                user_session = UserTradingSession(user, test_config, 'development')
                
                # Test initial state
                initial_state = user_session.get_trading_state()
                assert 'last_scan_time' in initial_state
                assert 'selected_stocks' in initial_state
                
                # Test state update
                updates = {'selected_stocks': ['RELIANCE.NS', 'TCS.NS']}
                user_session.update_trading_state(updates)
                
                updated_state = user_session.get_trading_state()
                assert updated_state['selected_stocks'] == ['RELIANCE.NS', 'TCS.NS']


class TestMultiUserTradingEngine:
    """Test MultiUserTradingEngine class."""
    
    def test_engine_initialization(self, trading_engine):
        """Test engine initialization."""
        assert trading_engine.config is not None
        assert trading_engine.mode == 'development'
        assert trading_engine.is_running is False
        assert len(trading_engine.user_sessions) == 0
    
    def test_create_user_session(self, trading_engine, db_manager, test_users):
        """Test creating a user session."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            user_id = user.id
            
            # Create user session
            user_session = trading_engine.create_user_session(user)
            
            assert user_session is not None
            assert user_session.user_id == user_id
            assert user_id in trading_engine.user_sessions
            assert trading_engine.user_sessions[user_id] == user_session
    
    def test_remove_user_session(self, trading_engine, db_manager, test_users):
        """Test removing a user session."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            user_id = user.id
            
            # Create user session
            user_session = trading_engine.create_user_session(user)
            assert user_id in trading_engine.user_sessions
            
            # Remove user session
            trading_engine.remove_user_session(user_id)
            assert user_id not in trading_engine.user_sessions
    
    def test_get_user_session(self, trading_engine, db_manager, test_users):
        """Test getting a user session."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            user_id = user.id
            
            # Initially no session
            assert trading_engine.get_user_session(user_id) is None
            
            # Create user session
            user_session = trading_engine.create_user_session(user)
            
            # Now should be able to get it
            retrieved_session = trading_engine.get_user_session(user_id)
            assert retrieved_session == user_session
    
    def test_get_active_users(self, trading_engine, db_manager, test_users):
        """Test getting active users."""
        with db_manager.get_session() as session:
            user1 = session.query(User).filter(User.id == test_users[0]).first()
            user2 = session.query(User).filter(User.id == test_users[1]).first()
            user1_id = user1.id
            user2_id = user2.id
            
            # Create user sessions
            trading_engine.create_user_session(user1)
            trading_engine.create_user_session(user2)
            
            # Get active users
            active_users = trading_engine.get_active_users()
            assert len(active_users) == 2
            assert user1 in active_users
            assert user2 in active_users
    
    def test_engine_status(self, trading_engine, db_manager, test_users):
        """Test getting engine status."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            
            # Create user session
            trading_engine.create_user_session(user)
            
            # Get engine status
            status = trading_engine.get_engine_status()
            
            assert 'is_running' in status
            assert 'active_users_count' in status
            assert 'total_users_count' in status
            assert 'mode' in status
            assert 'users' in status
            
            assert status['is_running'] is False
            assert status['active_users_count'] == 1
            assert status['total_users_count'] == 1
            assert status['mode'] == 'development'
            assert len(status['users']) == 1
    
    def test_engine_start_stop(self, trading_engine):
        """Test starting and stopping the engine."""
        # Initially not running
        assert trading_engine.is_running is False
        
        # Start engine
        trading_engine.start_engine()
        assert trading_engine.is_running is True
        assert trading_engine.engine_thread is not None
        
        # Stop engine
        trading_engine.stop_engine()
        assert trading_engine.is_running is False
    
    def test_user_isolation(self, trading_engine, db_manager, test_users):
        """Test that user sessions are properly isolated."""
        with db_manager.get_session() as session:
            user1 = session.query(User).filter(User.id == test_users[0]).first()
            user2 = session.query(User).filter(User.id == test_users[1]).first()
            
            # Create user sessions
            session1 = trading_engine.create_user_session(user1)
            session2 = trading_engine.create_user_session(user2)
            
            # Update trading state for user1
            session1.update_trading_state({'selected_stocks': ['RELIANCE.NS']})
            
            # Verify user2's state is not affected
            user2_state = session2.get_trading_state()
            assert user2_state['selected_stocks'] == []
            
            # Verify user1's state is correct
            user1_state = session1.get_trading_state()
            assert user1_state['selected_stocks'] == ['RELIANCE.NS']
    
    def test_market_scan_execution(self, trading_engine, db_manager, test_users):
        """Test market scan execution for a user."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            
            # Create user session
            user_session = trading_engine.create_user_session(user)
            
            # Mock the selector engine
            with patch.object(user_session.selector_engine, 'get_active_strategy') as mock_get_strategy:
                mock_strategy = Mock()
                mock_get_strategy.return_value = mock_strategy
                
                with patch.object(user_session.selector_engine, 'select_stocks') as mock_select_stocks:
                    mock_select_stocks.return_value = [{'symbol': 'RELIANCE.NS', 'price': 2500}]
                    
                    # Run market scan
                    trading_engine._run_market_scan(user_session)
                    
                    # Verify the scan was executed
                    mock_select_stocks.assert_called_once()
                    
                    # Verify trading state was updated
                    state = user_session.get_trading_state()
                    assert len(state['selected_stocks']) == 1
                    assert state['selected_stocks'][0]['symbol'] == 'RELIANCE.NS'


class TestMultiUserTradingEngineIntegration:
    """Integration tests for the multi-user trading engine."""
    
    def test_full_workflow(self, trading_engine, db_manager, test_users):
        """Test a complete workflow with multiple users."""
        with db_manager.get_session() as session:
            user1 = session.query(User).filter(User.id == test_users[0]).first()
            user2 = session.query(User).filter(User.id == test_users[1]).first()
            
            # Create user sessions
            session1 = trading_engine.create_user_session(user1)
            session2 = trading_engine.create_user_session(user2)
            
            # Start engine
            trading_engine.start_engine()
            
            # Verify both users are active
            active_users = trading_engine.get_active_users()
            assert len(active_users) == 2
            
            # Get engine status
            status = trading_engine.get_engine_status()
            assert status['active_users_count'] == 2
            
            # Stop engine
            trading_engine.stop_engine()
            
            # Verify engine is stopped
            assert trading_engine.is_running is False
    
    def test_user_session_cleanup(self, trading_engine, db_manager, test_users):
        """Test proper cleanup when user sessions are removed."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            
            # Create user session
            user_session = trading_engine.create_user_session(user)
            assert user.id in trading_engine.user_sessions
            
            # Deactivate user in database
            user.is_active = False
            session.commit()
            
            # Remove user session
            trading_engine.remove_user_session(user.id)
            
            # Verify session is removed
            assert user.id not in trading_engine.user_sessions
            assert user_session.is_active is False
    
    def test_error_handling(self, trading_engine, db_manager, test_users):
        """Test error handling in the trading engine."""
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == test_users[0]).first()
            
            # Create user session
            user_session = trading_engine.create_user_session(user)
            
            # Mock an error in market scan
            with patch.object(trading_engine, '_run_market_scan', side_effect=Exception("Test error")):
                # This should not crash the engine
                try:
                    trading_engine._process_user_trading(user_session)
                except Exception:
                    # The error should be caught and logged
                    pass
                
                # Engine should still be functional
                assert trading_engine.get_user_session(user.id) is not None
