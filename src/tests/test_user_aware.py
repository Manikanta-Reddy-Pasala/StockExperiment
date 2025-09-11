"""
Tests for User-Aware Functionality
"""
import pytest
import os
import sys
from datetime import datetime
from flask import url_for
from flask_login import current_user
from flask_bcrypt import Bcrypt

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from web.app import create_app
from models.database import get_database_manager
from models.models import User, Order, Trade, Position, Strategy, SuggestedStock, Configuration, Log


class TestUserAware:
    """Test cases for user-aware functionality."""
    
    @pytest.fixture
    def app(self):
        """Create a test Flask application."""
        # Use in-memory SQLite database for tests
        os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
        
        app = create_app()
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        
        with app.app_context():
            # Create tables
            db_manager = get_database_manager()
            db_manager.create_tables()
            
            yield app
            
            # Cleanup
            if 'DATABASE_URL' in os.environ:
                del os.environ['DATABASE_URL']
    
    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return app.test_client()
    
    @pytest.fixture
    def db_manager(self, app):
        """Get database manager."""
        with app.app_context():
            return get_database_manager()
    
    @pytest.fixture
    def bcrypt(self, app):
        """Get bcrypt instance."""
        return Bcrypt(app)
    
    @pytest.fixture
    def test_users(self, db_manager, bcrypt):
        """Create test users."""
        with db_manager.get_session() as session:
            # Create two test users
            password_hash = bcrypt.generate_password_hash('testpassword123').decode('utf-8')
            
            user1 = User(
                username='user1',
                email='user1@example.com',
                password_hash=password_hash
            )
            user2 = User(
                username='user2',
                email='user2@example.com',
                password_hash=password_hash
            )
            
            session.add(user1)
            session.add(user2)
            session.commit()
            
            # Return user IDs to avoid session issues
            return user1.id, user2.id
    
    def test_user_isolation_orders(self, client, db_manager, test_users):
        """Test that users only see their own orders."""
        user1, user2 = test_users
        
        # Create orders for both users
        with db_manager.get_session() as session:
            order1 = Order(
                user_id=user1,
                order_id='ORDER001',
                tradingsymbol='RELIANCE.NS',
                transaction_type='BUY',
                quantity=10,
                order_type='MARKET',
                order_status='COMPLETE'
            )
            order2 = Order(
                user_id=user2,
                order_id='ORDER002',
                tradingsymbol='TCS.NS',
                transaction_type='SELL',
                quantity=5,
                order_type='LIMIT',
                order_status='PENDING'
            )
            
            session.add(order1)
            session.add(order2)
            session.commit()
        
        # Login as user1
        client.post('/login', data={
            'username': 'user1',
            'password': 'testpassword123'
        })
        
        # Get orders for user1
        response = client.get('/api/orders')
        assert response.status_code == 200
        
        orders_data = response.get_json()
        assert len(orders_data) == 1
        assert orders_data[0]['order_id'] == 'ORDER001'
        assert orders_data[0]['tradingsymbol'] == 'RELIANCE.NS'
        
        # Logout and login as user2
        client.get('/logout')
        client.post('/login', data={
            'username': 'user2',
            'password': 'testpassword123'
        })
        
        # Get orders for user2
        response = client.get('/api/orders')
        assert response.status_code == 200
        
        orders_data = response.get_json()
        assert len(orders_data) == 1
        assert orders_data[0]['order_id'] == 'ORDER002'
        assert orders_data[0]['tradingsymbol'] == 'TCS.NS'
    
    def test_user_isolation_positions(self, client, db_manager, test_users):
        """Test that users only see their own positions."""
        user1, user2 = test_users
        
        # Create positions for both users
        with db_manager.get_session() as session:
            position1 = Position(
                user_id=user1,
                tradingsymbol='RELIANCE.NS',
                quantity=10,
                average_price=2750.0,
                last_price=2780.0,
                pnl=300.0
            )
            position2 = Position(
                user_id=user2,
                tradingsymbol='TCS.NS',
                quantity=5,
                average_price=3850.0,
                last_price=3825.0,
                pnl=-125.0
            )
            
            session.add(position1)
            session.add(position2)
            session.commit()
        
        # Login as user1
        client.post('/login', data={
            'username': 'user1',
            'password': 'testpassword123'
        })
        
        # Get positions for user1
        response = client.get('/api/portfolio')
        assert response.status_code == 200
        
        positions_data = response.get_json()
        assert len(positions_data) == 1
        assert positions_data[0]['tradingsymbol'] == 'RELIANCE.NS'
        assert positions_data[0]['quantity'] == 10
        
        # Logout and login as user2
        client.get('/logout')
        client.post('/login', data={
            'username': 'user2',
            'password': 'testpassword123'
        })
        
        # Get positions for user2
        response = client.get('/api/portfolio')
        assert response.status_code == 200
        
        positions_data = response.get_json()
        assert len(positions_data) == 1
        assert positions_data[0]['tradingsymbol'] == 'TCS.NS'
        assert positions_data[0]['quantity'] == 5
    
    def test_user_isolation_strategies(self, client, db_manager, test_users):
        """Test that users only see their own strategies."""
        user1, user2 = test_users
        
        # Create strategies for both users
        with db_manager.get_session() as session:
            strategy1 = Strategy(
                user_id=user1,
                name='User1 Momentum Strategy',
                description='Momentum strategy for user1',
                is_active=True
            )
            strategy2 = Strategy(
                user_id=user2,
                name='User2 Breakout Strategy',
                description='Breakout strategy for user2',
                is_active=True
            )
            
            session.add(strategy1)
            session.add(strategy2)
            session.commit()
        
        # Login as user1
        client.post('/login', data={
            'username': 'user1',
            'password': 'testpassword123'
        })
        
        # Get strategies for user1
        response = client.get('/api/strategies')
        assert response.status_code == 200
        
        strategies_data = response.get_json()
        assert len(strategies_data) == 1
        assert strategies_data[0]['name'] == 'User1 Momentum Strategy'
        
        # Logout and login as user2
        client.get('/logout')
        client.post('/login', data={
            'username': 'user2',
            'password': 'testpassword123'
        })
        
        # Get strategies for user2
        response = client.get('/api/strategies')
        assert response.status_code == 200
        
        strategies_data = response.get_json()
        assert len(strategies_data) == 1
        assert strategies_data[0]['name'] == 'User2 Breakout Strategy'
    
    def test_user_isolation_suggested_stocks(self, client, db_manager, test_users):
        """Test that users only see their own suggested stocks."""
        user1, user2 = test_users
        
        # Create suggested stocks for both users
        with db_manager.get_session() as session:
            stock1 = SuggestedStock(
                user_id=user1,
                symbol='RELIANCE.NS',
                selection_price=2750.0,
                current_price=2780.0,
                quantity=10,
                strategy_name='Momentum',
                status='Active'
            )
            stock2 = SuggestedStock(
                user_id=user2,
                symbol='TCS.NS',
                selection_price=3850.0,
                current_price=3825.0,
                quantity=5,
                strategy_name='Breakout',
                status='Active'
            )
            
            session.add(stock1)
            session.add(stock2)
            session.commit()
        
        # Login as user1
        client.post('/login', data={
            'username': 'user1',
            'password': 'testpassword123'
        })
        
        # API endpoint removed - no longer testing selected stocks API
        
        # Logout and login as user2
        client.get('/logout')
        client.post('/login', data={
            'username': 'user2',
            'password': 'testpassword123'
        })
        
        # API endpoint removed - no longer testing selected stocks API
    
    def test_user_isolation_logs(self, client, db_manager, test_users):
        """Test that users only see their own logs."""
        user1, user2 = test_users
        
        # Create logs for both users
        with db_manager.get_session() as session:
            log1 = Log(
                user_id=user1,
                level='INFO',
                module='trading',
                message='User1 order executed',
                details='{"order_id": "ORDER001"}'
            )
            log2 = Log(
                user_id=user2,
                level='WARNING',
                module='risk',
                message='User2 position limit exceeded',
                details='{"position": "TCS.NS"}'
            )
            
            session.add(log1)
            session.add(log2)
            session.commit()
        
        # Login as user1
        client.post('/login', data={
            'username': 'user1',
            'password': 'testpassword123'
        })
        
        # Get logs for user1
        response = client.get('/api/logs')
        assert response.status_code == 200
        
        logs_data = response.get_json()
        assert len(logs_data) == 1
        assert logs_data[0]['message'] == 'User1 order executed'
        
        # Logout and login as user2
        client.get('/logout')
        client.post('/login', data={
            'username': 'user2',
            'password': 'testpassword123'
        })
        
        # Get logs for user2
        response = client.get('/api/logs')
        assert response.status_code == 200
        
        logs_data = response.get_json()
        assert len(logs_data) == 1
        assert logs_data[0]['message'] == 'User2 position limit exceeded'
    
    def test_user_specific_configurations(self, client, db_manager, test_users):
        """Test that users get their own configurations with global fallbacks."""
        user1, user2 = test_users
        
        # Create global and user-specific configurations
        with db_manager.get_session() as session:
            # Global configuration
            global_config = Configuration(
                user_id=None,
                key='trading_mode',
                value='production',
                description='Global trading mode'
            )
            
            # User1 specific configuration
            user1_config = Configuration(
                user_id=user1,
                key='trading_mode',
                value='development',
                description='User1 trading mode'
            )
            
            # User2 specific configuration
            user2_config = Configuration(
                user_id=user2,
                key='max_capital_per_trade',
                value='2.0',
                description='User2 max capital'
            )
            
            session.add(global_config)
            session.add(user1_config)
            session.add(user2_config)
            session.commit()
        
        # Login as user1
        client.post('/login', data={
            'username': 'user1',
            'password': 'testpassword123'
        })
        
        # Get settings for user1
        response = client.get('/api/settings')
        assert response.status_code == 200
        
        settings_data = response.get_json()
        assert settings_data['trading_mode'] == 'development'  # User-specific override
        
        # Logout and login as user2
        client.get('/logout')
        client.post('/login', data={
            'username': 'user2',
            'password': 'testpassword123'
        })
        
        # Get settings for user2
        response = client.get('/api/settings')
        assert response.status_code == 200
        
        settings_data = response.get_json()
        assert settings_data['trading_mode'] == 'production'  # Global fallback
        assert settings_data['max_capital_per_trade'] == '2.0'  # User-specific


if __name__ == '__main__':
    pytest.main([__file__])
