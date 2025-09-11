"""
Tests for the Authentication System
"""
import pytest
import os
import sys
import tempfile
from datetime import datetime
from flask import url_for
from flask_login import current_user
from flask_bcrypt import Bcrypt

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from web.app import create_app
from datastore.database import get_database_manager
from datastore.models import User, Base


class TestAuthentication:
    """Test cases for authentication functionality."""
    
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
    
    def test_user_model_creation(self, db_manager, bcrypt):
        """Test User model creation and password hashing."""
        with db_manager.get_session() as session:
            # Create a test user
            password_hash = bcrypt.generate_password_hash('testpassword123').decode('utf-8')
            user = User(
                username='testuser',
                email='test@example.com',
                password_hash=password_hash,
                first_name='Test',
                last_name='User'
            )
            
            session.add(user)
            session.commit()
            
            # Verify user was created
            retrieved_user = session.query(User).filter_by(username='testuser').first()
            assert retrieved_user is not None
            assert retrieved_user.username == 'testuser'
            assert retrieved_user.email == 'test@example.com'
            assert retrieved_user.first_name == 'Test'
            assert retrieved_user.last_name == 'User'
            assert retrieved_user.is_active is True
            assert retrieved_user.is_admin is False
            assert retrieved_user.created_at is not None
            
            # Verify password hash
            assert bcrypt.check_password_hash(retrieved_user.password_hash, 'testpassword123')
            assert not bcrypt.check_password_hash(retrieved_user.password_hash, 'wrongpassword')
    
    def test_user_model_repr(self, db_manager, bcrypt):
        """Test User model string representation."""
        with db_manager.get_session() as session:
            password_hash = bcrypt.generate_password_hash('testpassword123').decode('utf-8')
            user = User(
                username='testuser',
                email='test@example.com',
                password_hash=password_hash
            )
            
            assert repr(user) == '<User testuser>'
    
    def test_login_page_get(self, client):
        """Test login page GET request."""
        response = client.get('/login')
        assert response.status_code == 200
        assert b'Sign in to your account' in response.data
        assert b'username' in response.data
        assert b'password' in response.data
    
    def test_login_page_post_success(self, client, db_manager, bcrypt):
        """Test successful login."""
        # Create a test user
        with db_manager.get_session() as session:
            password_hash = bcrypt.generate_password_hash('testpassword123').decode('utf-8')
            user = User(
                username='testuser',
                email='test@example.com',
                password_hash=password_hash
            )
            session.add(user)
            session.commit()
            user_id = user.id
        
        # Test login
        response = client.post('/login', data={
            'username': 'testuser',
            'password': 'testpassword123'
        }, follow_redirects=True)
        
        assert response.status_code == 200
        # Check that we're redirected to dashboard (which requires login)
        assert b'Dashboard' in response.data or b'dashboard' in response.data
    
    def test_login_page_post_invalid_credentials(self, client, db_manager, bcrypt):
        """Test login with invalid credentials."""
        # Create a test user
        with db_manager.get_session() as session:
            password_hash = bcrypt.generate_password_hash('testpassword123').decode('utf-8')
            user = User(
                username='testuser',
                email='test@example.com',
                password_hash=password_hash
            )
            session.add(user)
            session.commit()
        
        # Test login with wrong password
        response = client.post('/login', data={
            'username': 'testuser',
            'password': 'wrongpassword'
        })
        
        assert response.status_code == 200
        assert b'Invalid username or password' in response.data
    
    def test_login_page_post_missing_fields(self, client):
        """Test login with missing fields."""
        response = client.post('/login', data={
            'username': 'testuser'
            # Missing password
        })
        
        assert response.status_code == 200
        assert b'Please fill in all fields' in response.data
    
    def test_login_page_post_inactive_user(self, client, db_manager, bcrypt):
        """Test login with inactive user."""
        # Create an inactive user
        with db_manager.get_session() as session:
            password_hash = bcrypt.generate_password_hash('testpassword123').decode('utf-8')
            user = User(
                username='testuser',
                email='test@example.com',
                password_hash=password_hash,
                is_active=False
            )
            session.add(user)
            session.commit()
        
        # Test login
        response = client.post('/login', data={
            'username': 'testuser',
            'password': 'testpassword123'
        })
        
        assert response.status_code == 200
        assert b'Your account has been deactivated' in response.data
    
    def test_register_page_get(self, client):
        """Test registration page GET request."""
        response = client.get('/register')
        assert response.status_code == 200
        assert b'Create your account' in response.data
        assert b'username' in response.data
        assert b'email' in response.data
        assert b'password' in response.data
    
    def test_register_page_post_success(self, client, db_manager):
        """Test successful registration."""
        response = client.post('/register', data={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'newpassword123',
            'confirm_password': 'newpassword123',
            'first_name': 'New',
            'last_name': 'User'
        }, follow_redirects=True)
        
        assert response.status_code == 200
        assert b'Registration successful' in response.data
        
        # Verify user was created in database
        with db_manager.get_session() as session:
            user = session.query(User).filter_by(username='newuser').first()
            assert user is not None
            assert user.email == 'newuser@example.com'
            assert user.first_name == 'New'
            assert user.last_name == 'User'
    
    def test_register_page_post_password_mismatch(self, client):
        """Test registration with password mismatch."""
        response = client.post('/register', data={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'newpassword123',
            'confirm_password': 'differentpassword',
            'first_name': 'New',
            'last_name': 'User'
        })
        
        assert response.status_code == 200
        assert b'Passwords do not match' in response.data
    
    def test_register_page_post_short_password(self, client):
        """Test registration with short password."""
        response = client.post('/register', data={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': '123',
            'confirm_password': '123',
            'first_name': 'New',
            'last_name': 'User'
        })
        
        assert response.status_code == 200
        assert b'Password must be at least 6 characters long' in response.data
    
    def test_register_page_post_duplicate_username(self, client, db_manager, bcrypt):
        """Test registration with duplicate username."""
        # Create existing user
        with db_manager.get_session() as session:
            password_hash = bcrypt.generate_password_hash('testpassword123').decode('utf-8')
            user = User(
                username='existinguser',
                email='existing@example.com',
                password_hash=password_hash
            )
            session.add(user)
            session.commit()
        
        # Try to register with same username
        response = client.post('/register', data={
            'username': 'existinguser',
            'email': 'newuser@example.com',
            'password': 'newpassword123',
            'confirm_password': 'newpassword123',
            'first_name': 'New',
            'last_name': 'User'
        })
        
        assert response.status_code == 200
        assert b'Username already exists' in response.data
    
    def test_register_page_post_duplicate_email(self, client, db_manager, bcrypt):
        """Test registration with duplicate email."""
        # Create existing user
        with db_manager.get_session() as session:
            password_hash = bcrypt.generate_password_hash('testpassword123').decode('utf-8')
            user = User(
                username='existinguser',
                email='existing@example.com',
                password_hash=password_hash
            )
            session.add(user)
            session.commit()
        
        # Try to register with same email
        response = client.post('/register', data={
            'username': 'newuser',
            'email': 'existing@example.com',
            'password': 'newpassword123',
            'confirm_password': 'newpassword123',
            'first_name': 'New',
            'last_name': 'User'
        })
        
        assert response.status_code == 200
        assert b'Email already registered' in response.data
    
    def test_register_page_post_missing_fields(self, client):
        """Test registration with missing required fields."""
        response = client.post('/register', data={
            'username': 'newuser',
            'email': 'newuser@example.com'
            # Missing password and confirm_password
        })
        
        assert response.status_code == 200
        assert b'Please fill in all required fields' in response.data
    
    def test_logout(self, client, db_manager, bcrypt):
        """Test logout functionality."""
        # Create and login user
        with db_manager.get_session() as session:
            password_hash = bcrypt.generate_password_hash('testpassword123').decode('utf-8')
            user = User(
                username='testuser',
                email='test@example.com',
                password_hash=password_hash
            )
            session.add(user)
            session.commit()
        
        # Login
        client.post('/login', data={
            'username': 'testuser',
            'password': 'testpassword123'
        })
        
        # Logout
        response = client.get('/logout', follow_redirects=True)
        assert response.status_code == 200
        assert b'You have been logged out successfully' in response.data
    
    def test_protected_route_redirect(self, client):
        """Test that protected routes redirect to login."""
        response = client.get('/', follow_redirects=False)
        assert response.status_code == 302
        assert '/login' in response.location
    
    def test_dashboard_requires_login(self, client):
        """Test that dashboard requires login."""
        response = client.get('/', follow_redirects=True)
        assert response.status_code == 200
        assert b'Sign in to your account' in response.data
    
    def test_remember_me_functionality(self, client, db_manager, bcrypt):
        """Test remember me functionality."""
        # Create a test user
        with db_manager.get_session() as session:
            password_hash = bcrypt.generate_password_hash('testpassword123').decode('utf-8')
            user = User(
                username='testuser',
                email='test@example.com',
                password_hash=password_hash
            )
            session.add(user)
            session.commit()
        
        # Login with remember me
        response = client.post('/login', data={
            'username': 'testuser',
            'password': 'testpassword123',
            'remember': 'on'
        }, follow_redirects=True)
        
        assert response.status_code == 200
        
        # Check that remember me cookie is set
        cookies = [cookie for cookie in client.cookie_jar if cookie.name == 'remember_token']
        assert len(cookies) > 0
    
    def test_user_last_login_update(self, client, db_manager, bcrypt):
        """Test that last_login is updated on login."""
        # Create a test user
        with db_manager.get_session() as session:
            password_hash = bcrypt.generate_password_hash('testpassword123').decode('utf-8')
            user = User(
                username='testuser',
                email='test@example.com',
                password_hash=password_hash
            )
            session.add(user)
            session.commit()
            user_id = user.id
        
        # Login
        client.post('/login', data={
            'username': 'testuser',
            'password': 'testpassword123'
        })
        
        # Check that last_login was updated
        with db_manager.get_session() as session:
            user = session.query(User).get(user_id)
            assert user.last_login is not None
            assert user.last_login <= datetime.utcnow()


if __name__ == '__main__':
    pytest.main([__file__])
