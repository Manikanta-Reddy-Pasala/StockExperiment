"""
Admin User Setup for the Trading System
"""
import os
from datastore.database import get_database_manager
from datastore.models import User

# Make Flask-Bcrypt optional for FastAPI compatibility
try:
    from flask_bcrypt import Bcrypt
except ImportError:
    # Use passlib for FastAPI
    from passlib.context import CryptContext
    Bcrypt = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_admin_user():
    """Create admin user if it doesn't exist."""
    # Get admin credentials from environment variables
    admin_username = os.getenv('ADMIN_USERNAME', 'admin')
    admin_password = os.getenv('ADMIN_PASSWORD', 'admin123')
    admin_email = os.getenv('ADMIN_EMAIL', 'admin@tradingsystem.com')
    admin_first_name = os.getenv('ADMIN_FIRST_NAME', 'System')
    admin_last_name = os.getenv('ADMIN_LAST_NAME', 'Administrator')
    
    # Initialize bcrypt for password hashing
    bcrypt = Bcrypt()
    
    # Get database manager
    db_manager = get_database_manager()
    
    # Create tables if they don't exist
    db_manager.create_tables()
    
    with db_manager.get_session() as session:
        # Check if admin user already exists
        existing_admin = session.query(User).filter(
            (User.username == admin_username) | (User.email == admin_email)
        ).first()
        
        if existing_admin:
            # Update existing admin user
            existing_admin.is_admin = True
            existing_admin.password_hash = bcrypt.generate_password_hash(admin_password).decode('utf-8')
            existing_admin.first_name = admin_first_name
            existing_admin.last_name = admin_last_name
            existing_admin.is_active = True
            session.commit()
            print(f"Updated existing admin user: {admin_username}")
        else:
            # Create new admin user
            password_hash = bcrypt.generate_password_hash(admin_password).decode('utf-8')
            admin_user = User(
                username=admin_username,
                email=admin_email,
                password_hash=password_hash,
                first_name=admin_first_name,
                last_name=admin_last_name,
                is_admin=True,
                is_active=True
            )
            
            session.add(admin_user)
            session.commit()
            print(f"Created admin user: {admin_username}")
        
        return True


def get_admin_user():
    """Get the admin user."""
    admin_username = os.getenv('ADMIN_USERNAME', 'admin')
    db_manager = get_database_manager()
    
    with db_manager.get_session() as session:
        admin_user = session.query(User).filter(
            User.username == admin_username,
            User.is_admin == True
        ).first()
        
        return admin_user


if __name__ == "__main__":
    create_admin_user()
