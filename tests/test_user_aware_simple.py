"""
Simple Tests for User-Aware Database Queries
"""
import pytest
import os
import sys
from datetime import datetime
from flask_bcrypt import Bcrypt

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from datastore.database import get_database_manager
from datastore.models import User, Order, Trade, Position, Strategy, SuggestedStock, Configuration, Log


class TestUserAwareDatabase:
    """Test cases for user-aware database functionality."""
    
    @pytest.fixture
    def db_manager(self):
        """Get database manager."""
        # Use a unique in-memory database for each test
        import uuid
        db_name = f'sqlite:///:memory:{uuid.uuid4().hex}'
        os.environ['DATABASE_URL'] = db_name
        
        # Reset the global database manager to force recreation
        import datastore.database
        datastore.database.db_manager = None
        
        db_manager = get_database_manager()
        db_manager.create_tables()
        
        yield db_manager
        
        # Cleanup
        if 'DATABASE_URL' in os.environ:
            del os.environ['DATABASE_URL']
        # Reset the global database manager
        datastore.database.db_manager = None
    
    @pytest.fixture
    def bcrypt(self):
        """Get bcrypt instance."""
        return Bcrypt()
    
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
    
    def test_user_isolation_orders(self, db_manager, test_users):
        """Test that orders are properly isolated by user."""
        user1_id, user2_id = test_users
        
        # Create orders for both users
        with db_manager.get_session() as session:
            order1 = Order(
                user_id=user1_id,
                order_id='ORDER001',
                tradingsymbol='RELIANCE.NS',
                transaction_type='BUY',
                quantity=10,
                order_type='MARKET',
                order_status='COMPLETE'
            )
            order2 = Order(
                user_id=user2_id,
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
        
        # Test user1 can only see their orders
        with db_manager.get_session() as session:
            user1_orders = session.query(Order).filter(Order.user_id == user1_id).all()
            assert len(user1_orders) == 1
            assert user1_orders[0].order_id == 'ORDER001'
            assert user1_orders[0].tradingsymbol == 'RELIANCE.NS'
        
        # Test user2 can only see their orders
        with db_manager.get_session() as session:
            user2_orders = session.query(Order).filter(Order.user_id == user2_id).all()
            assert len(user2_orders) == 1
            assert user2_orders[0].order_id == 'ORDER002'
            assert user2_orders[0].tradingsymbol == 'TCS.NS'
    
    def test_user_isolation_positions(self, db_manager, test_users):
        """Test that positions are properly isolated by user."""
        user1_id, user2_id = test_users
        
        # Create positions for both users
        with db_manager.get_session() as session:
            position1 = Position(
                user_id=user1_id,
                tradingsymbol='RELIANCE.NS',
                quantity=10,
                average_price=2750.0,
                last_price=2780.0,
                pnl=300.0
            )
            position2 = Position(
                user_id=user2_id,
                tradingsymbol='TCS.NS',
                quantity=5,
                average_price=3850.0,
                last_price=3825.0,
                pnl=-125.0
            )
            
            session.add(position1)
            session.add(position2)
            session.commit()
        
        # Test user1 can only see their positions
        with db_manager.get_session() as session:
            user1_positions = session.query(Position).filter(Position.user_id == user1_id).all()
            assert len(user1_positions) == 1
            assert user1_positions[0].tradingsymbol == 'RELIANCE.NS'
            assert user1_positions[0].quantity == 10
        
        # Test user2 can only see their positions
        with db_manager.get_session() as session:
            user2_positions = session.query(Position).filter(Position.user_id == user2_id).all()
            assert len(user2_positions) == 1
            assert user2_positions[0].tradingsymbol == 'TCS.NS'
            assert user2_positions[0].quantity == 5
    
    def test_user_isolation_strategies(self, db_manager, test_users):
        """Test that strategies are properly isolated by user."""
        user1_id, user2_id = test_users
        
        # Create strategies for both users
        with db_manager.get_session() as session:
            strategy1 = Strategy(
                user_id=user1_id,
                name='User1 Momentum Strategy',
                description='Momentum strategy for user1',
                is_active=True
            )
            strategy2 = Strategy(
                user_id=user2_id,
                name='User2 Breakout Strategy',
                description='Breakout strategy for user2',
                is_active=True
            )
            
            session.add(strategy1)
            session.add(strategy2)
            session.commit()
        
        # Test user1 can only see their strategies
        with db_manager.get_session() as session:
            user1_strategies = session.query(Strategy).filter(Strategy.user_id == user1_id).all()
            assert len(user1_strategies) == 1
            assert user1_strategies[0].name == 'User1 Momentum Strategy'
        
        # Test user2 can only see their strategies
        with db_manager.get_session() as session:
            user2_strategies = session.query(Strategy).filter(Strategy.user_id == user2_id).all()
            assert len(user2_strategies) == 1
            assert user2_strategies[0].name == 'User2 Breakout Strategy'
    
    def test_user_isolation_suggested_stocks(self, db_manager, test_users):
        """Test that suggested stocks are properly isolated by user."""
        user1_id, user2_id = test_users
        
        # Create suggested stocks for both users
        with db_manager.get_session() as session:
            stock1 = SuggestedStock(
                user_id=user1_id,
                symbol='RELIANCE.NS',
                selection_price=2750.0,
                current_price=2780.0,
                quantity=10,
                strategy_name='Momentum',
                status='Active'
            )
            stock2 = SuggestedStock(
                user_id=user2_id,
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
        
        # Test user1 can only see their suggested stocks
        with db_manager.get_session() as session:
            user1_stocks = session.query(SuggestedStock).filter(SuggestedStock.user_id == user1_id).all()
            assert len(user1_stocks) == 1
            assert user1_stocks[0].symbol == 'RELIANCE.NS'
        
        # Test user2 can only see their suggested stocks
        with db_manager.get_session() as session:
            user2_stocks = session.query(SuggestedStock).filter(SuggestedStock.user_id == user2_id).all()
            assert len(user2_stocks) == 1
            assert user2_stocks[0].symbol == 'TCS.NS'
    
    def test_user_isolation_logs(self, db_manager, test_users):
        """Test that logs are properly isolated by user."""
        user1_id, user2_id = test_users
        
        # Create logs for both users
        with db_manager.get_session() as session:
            log1 = Log(
                user_id=user1_id,
                level='INFO',
                module='trading',
                message='User1 order executed',
                details='{"order_id": "ORDER001"}'
            )
            log2 = Log(
                user_id=user2_id,
                level='WARNING',
                module='risk',
                message='User2 position limit exceeded',
                details='{"position": "TCS.NS"}'
            )
            
            session.add(log1)
            session.add(log2)
            session.commit()
        
        # Test user1 can only see their logs
        with db_manager.get_session() as session:
            user1_logs = session.query(Log).filter(Log.user_id == user1_id).all()
            assert len(user1_logs) == 1
            assert user1_logs[0].message == 'User1 order executed'
        
        # Test user2 can only see their logs
        with db_manager.get_session() as session:
            user2_logs = session.query(Log).filter(Log.user_id == user2_id).all()
            assert len(user2_logs) == 1
            assert user2_logs[0].message == 'User2 position limit exceeded'
    
    def test_user_specific_configurations(self, db_manager, test_users):
        """Test that configurations work with user-specific and global settings."""
        user1_id, user2_id = test_users
        
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
                user_id=user1_id,
                key='trading_mode',
                value='development',
                description='User1 trading mode'
            )
            
            # User2 specific configuration
            user2_config = Configuration(
                user_id=user2_id,
                key='max_capital_per_trade',
                value='2.0',
                description='User2 max capital'
            )
            
            session.add(global_config)
            session.add(user1_config)
            session.add(user2_config)
            session.commit()
        
        # Test user1 gets their specific config
        with db_manager.get_session() as session:
            user1_configs = session.query(Configuration).filter(Configuration.user_id == user1_id).all()
            assert len(user1_configs) == 1
            assert user1_configs[0].key == 'trading_mode'
            assert user1_configs[0].value == 'development'
        
        # Test user2 gets their specific config
        with db_manager.get_session() as session:
            user2_configs = session.query(Configuration).filter(Configuration.user_id == user2_id).all()
            assert len(user2_configs) == 1
            assert user2_configs[0].key == 'max_capital_per_trade'
            assert user2_configs[0].value == '2.0'
        
        # Test global configs exist
        with db_manager.get_session() as session:
            global_configs = session.query(Configuration).filter(Configuration.user_id.is_(None)).all()
            assert len(global_configs) == 1
            assert global_configs[0].key == 'trading_mode'
            assert global_configs[0].value == 'production'


if __name__ == '__main__':
    pytest.main([__file__])
