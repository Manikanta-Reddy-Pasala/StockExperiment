"""
Broker Service for managing broker connections and API interactions
"""
import os
import time
import requests
import json
import base64
from datetime import datetime
from typing import Dict, Optional, Any
from cryptography.fernet import Fernet

try:
    from ..models.database import get_database_manager
    from ..models.models import BrokerConfiguration, Order, Trade
except ImportError:
    from models.database import get_database_manager
    from models.models import BrokerConfiguration, Order, Trade


class BrokerService:
    """Service for managing broker configurations and connections."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for storing sensitive data."""
        with self.db_manager.get_session() as session:
            # Try to get encryption key from database
            from ..models.models import Configuration
            config = session.query(Configuration).filter_by(
                user_id=None,  # Global config
                key='broker_encryption_key'
            ).first()
            
            if config and config.value:
                try:
                    return base64.b64decode(config.value.encode())
                except Exception:
                    # If decoding fails, generate new key
                    pass
            
            # Generate new encryption key
            key = Fernet.generate_key()
            key_b64 = base64.b64encode(key).decode()
            
            # Save to database
            if config:
                config.value = key_b64
                config.updated_at = datetime.utcnow()
            else:
                config = Configuration(
                    user_id=None,  # Global config
                    key='broker_encryption_key',
                    value=key_b64,
                    description='Encryption key for broker configuration data'
                )
                session.add(config)
            
            session.commit()
            return key
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not data:
            return ""
        return self.cipher.encrypt(data.encode()).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not encrypted_data:
            return ""
        try:
            return self.cipher.decrypt(encrypted_data.encode()).decode()
        except Exception:
            return ""
    
    def get_broker_config(self, broker_name: str, user_id: Optional[int] = None) -> Optional[BrokerConfiguration]:
        """Get broker configuration from database."""
        with self.db_manager.get_session() as session:
            query = session.query(BrokerConfiguration).filter_by(broker_name=broker_name)
            if user_id:
                query = query.filter_by(user_id=user_id)
            else:
                query = query.filter_by(user_id=None)  # Global config
            return query.first()
    
    def save_broker_config(self, broker_name: str, config_data: Dict[str, Any], user_id: Optional[int] = None) -> BrokerConfiguration:
        """Save broker configuration to database."""
        with self.db_manager.get_session() as session:
            # Check if config exists
            existing_config = self.get_broker_config(broker_name, user_id)
            
            if existing_config:
                # Update existing config
                config = existing_config
            else:
                # Create new config
                config = BrokerConfiguration(
                    broker_name=broker_name,
                    user_id=user_id
                )
                session.add(config)
            
            # Update fields
            config.client_id = config_data.get('client_id', '')
            config.access_token = self._encrypt_data(config_data.get('access_token', ''))
            config.refresh_token = self._encrypt_data(config_data.get('refresh_token', ''))
            config.api_key = config_data.get('api_key', '')
            config.api_secret = self._encrypt_data(config_data.get('api_secret', ''))
            config.redirect_url = config_data.get('redirect_url', '')
            config.app_type = config_data.get('app_type', '100')
            config.is_active = config_data.get('is_active', True)
            config.updated_at = datetime.utcnow()
            
            session.commit()
            return config
    
    def get_broker_stats(self, broker_name: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get broker statistics from database."""
        with self.db_manager.get_session() as session:
            # Get order statistics
            query = session.query(Order)
            if user_id:
                query = query.filter_by(user_id=user_id)
            
            total_orders = query.count()
            successful_orders = query.filter_by(order_status='COMPLETE').count()
            pending_orders = query.filter_by(order_status='PENDING').count()
            failed_orders = query.filter_by(order_status='REJECTED').count()
            
            # Get last order time
            last_order = query.order_by(Order.created_at.desc()).first()
            last_order_time = last_order.created_at.strftime('%Y-%m-%d %H:%M:%S') if last_order else '-'
            
            return {
                'total_orders': total_orders,
                'successful_orders': successful_orders,
                'pending_orders': pending_orders,
                'failed_orders': failed_orders,
                'last_order_time': last_order_time,
                'api_response_time': '-'
            }


class FyersAPIConnector:
    """FYERS API connector for real-time connection testing and operations."""
    
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.base_url = "https://api-t1.fyers.in/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        })
    
    def test_connection(self) -> Dict[str, Any]:
        """Test FYERS API connection by making a real API call."""
        try:
            start_time = time.time()
            
            # Test with profile API call (lightweight endpoint)
            response = self.session.get(f"{self.base_url}/profile")
            response_time = round((time.time() - start_time) * 1000, 2)  # in milliseconds
            
            if response.status_code == 200:
                data = response.json()
                if data.get('s') == 'ok':
                    return {
                        'success': True,
                        'message': 'Connection successful',
                        'response_time': f"{response_time}ms",
                        'profile_data': data.get('profile', {}),
                        'status_code': response.status_code
                    }
                else:
                    return {
                        'success': False,
                        'message': f"API Error: {data.get('message', 'Unknown error')}",
                        'response_time': f"{response_time}ms",
                        'status_code': response.status_code
                    }
            else:
                return {
                    'success': False,
                    'message': f"HTTP Error: {response.status_code} - {response.text}",
                    'response_time': f"{response_time}ms",
                    'status_code': response.status_code
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'message': 'Connection failed: Unable to reach FYERS API',
                'response_time': '-',
                'status_code': 0
            }
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'message': 'Connection failed: Request timeout',
                'response_time': '-',
                'status_code': 0
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection failed: {str(e)}',
                'response_time': '-',
                'status_code': 0
            }
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile information."""
        try:
            response = self.session.get(f"{self.base_url}/profile")
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}: {response.text}'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_holdings(self) -> Dict[str, Any]:
        """Get user holdings."""
        try:
            response = self.session.get(f"{self.base_url}/holdings")
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}: {response.text}'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_positions(self) -> Dict[str, Any]:
        """Get user positions."""
        try:
            response = self.session.get(f"{self.base_url}/positions")
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}: {response.text}'}
        except Exception as e:
            return {'error': str(e)}


def get_broker_service() -> BrokerService:
    """Get broker service instance."""
    return BrokerService()
