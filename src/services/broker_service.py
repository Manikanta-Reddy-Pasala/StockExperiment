"""
Broker Service for managing broker connections and API interactions
"""
import os
import time
import requests
import json
from datetime import datetime
from typing import Dict, Optional, Any

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
    
    def get_broker_config(self, broker_name: str, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get broker configuration from database."""
        with self.db_manager.get_session() as session:
            query = session.query(BrokerConfiguration).filter_by(broker_name=broker_name)
            if user_id:
                query = query.filter_by(user_id=user_id)
            else:
                query = query.filter_by(user_id=None)  # Global config
            
            config = query.first()
            if not config:
                return None
            
            # Return data dictionary instead of SQLAlchemy object
            return {
                'id': config.id,
                'user_id': config.user_id,
                'broker_name': config.broker_name,
                'client_id': config.client_id,
                'access_token': config.access_token,
                'refresh_token': config.refresh_token,
                'api_key': config.api_key,
                'api_secret': config.api_secret,
                'redirect_url': config.redirect_url,
                'app_type': config.app_type,
                'is_active': config.is_active,
                'is_connected': config.is_connected,
                'last_connection_test': config.last_connection_test,
                'connection_status': config.connection_status,
                'error_message': config.error_message,
                'created_at': config.created_at,
                'updated_at': config.updated_at
            }
    
    def save_broker_config(self, broker_name: str, config_data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
        """Save broker configuration to database."""
        with self.db_manager.get_session() as session:
            # Check if config exists within this session
            query = session.query(BrokerConfiguration).filter_by(broker_name=broker_name)
            if user_id:
                query = query.filter_by(user_id=user_id)
            else:
                query = query.filter_by(user_id=None)  # Global config
            
            existing_config = query.first()
            
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
            
            # Update fields (storing directly without encryption for simplicity)
            config.client_id = config_data.get('client_id', '')
            config.access_token = config_data.get('access_token', '')
            config.refresh_token = config_data.get('refresh_token', '')
            config.api_key = config_data.get('api_key', '')
            config.api_secret = config_data.get('api_secret', '')
            config.redirect_url = config_data.get('redirect_url', '')
            config.app_type = config_data.get('app_type', '100')
            config.is_active = config_data.get('is_active', True)
            config.updated_at = datetime.utcnow()
            
            session.commit()
            session.refresh(config)  # Refresh to get the updated object
            
            # Return data dictionary instead of SQLAlchemy object
            return {
                'id': config.id,
                'user_id': config.user_id,
                'broker_name': config.broker_name,
                'client_id': config.client_id,
                'access_token': config.access_token,
                'refresh_token': config.refresh_token,
                'api_key': config.api_key,
                'api_secret': config.api_secret,
                'redirect_url': config.redirect_url,
                'app_type': config.app_type,
                'is_active': config.is_active,
                'is_connected': config.is_connected,
                'last_connection_test': config.last_connection_test,
                'connection_status': config.connection_status,
                'error_message': config.error_message,
                'created_at': config.created_at,
                'updated_at': config.updated_at
            }
    
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
