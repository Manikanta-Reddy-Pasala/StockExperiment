"""
Zerodha Broker Service - Dedicated service for Zerodha Kite Connect operations
"""
import os
import time
import requests
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

try:
    from ...models.database import get_database_manager
    from ...models.models import BrokerConfiguration, Order, Trade
except ImportError:
    from models.database import get_database_manager
    from models.models import BrokerConfiguration, Order, Trade

try:
    from kiteconnect import KiteConnect
    ZERODHA_AVAILABLE = True
except ImportError:
    ZERODHA_AVAILABLE = False


class ZerodhaService:
    """Dedicated service for Zerodha Kite Connect operations."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.broker_name = 'zerodha'

    def _get_kite_connector(self, user_id: int) -> 'KiteAPIConnector':
        """Helper to get an initialized KiteAPIConnector for a user."""
        config = self.get_broker_config(user_id)
        if not config or not config.get('api_key') or not config.get('access_token'):
            raise ValueError('Zerodha credentials not configured or access token missing.')
        return KiteAPIConnector(config['api_key'], config['access_token'])

    def test_connection(self, user_id: int):
        """Test Zerodha broker connection."""
        config = self.get_broker_config(user_id)
        if not config or not config.get('api_key') or not config.get('access_token'):
            raise ValueError('Zerodha credentials not configured.')

        connector = KiteAPIConnector(config.get('api_key'), config.get('access_token'))
        result = connector.test_connection()

        with self.db_manager.get_session() as session:
            db_config = session.query(BrokerConfiguration).filter_by(broker_name=self.broker_name, user_id=user_id).first()
            if db_config:
                db_config.is_connected = result['success']
                db_config.connection_status = 'connected' if result['success'] else 'disconnected'
                db_config.last_connection_test = datetime.utcnow()
                db_config.error_message = result.get('message', '') if not result['success'] else None
                session.commit()

        return result

    def generate_login_url(self, user_id: int) -> str:
        """Generate Zerodha login URL for authentication."""
        config = self.get_broker_config(user_id)
        if not config or not config.get('api_key'):
            raise ValueError('Zerodha API key not configured.')

        # Zerodha login URL format
        login_url = f"https://kite.trade/connect/login?api_key={config.get('api_key')}"
        return login_url

    def get_funds(self, user_id: int):
        """Get user funds."""
        connector = self._get_kite_connector(user_id)
        return connector.get_funds()

    def get_holdings(self, user_id: int):
        """Get user holdings."""
        connector = self._get_kite_connector(user_id)
        return connector.get_holdings()

    def get_positions(self, user_id: int):
        """Get user positions."""
        connector = self._get_kite_connector(user_id)
        return connector.get_positions()

    def get_orderbook(self, user_id: int):
        """Get user orderbook."""
        connector = self._get_kite_connector(user_id)
        return connector.get_orderbook()

    def get_tradebook(self, user_id: int):
        """Get user tradebook."""
        connector = self._get_kite_connector(user_id)
        return connector.get_tradebook()

    def get_quotes(self, user_id: int, symbols: str):
        """Get market quotes for symbols."""
        connector = self._get_kite_connector(user_id)
        return connector.get_quotes(symbols)

    def get_instruments(self, user_id: int, exchange: str = None):
        """Get instruments list."""
        connector = self._get_kite_connector(user_id)
        return connector.get_instruments(exchange)

    def get_profile(self, user_id: int):
        """Get user profile."""
        connector = self._get_kite_connector(user_id)
        return connector.get_profile()
    
    def get_broker_config(self, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get Zerodha broker configuration from database."""
        with self.db_manager.get_session() as session:
            query = session.query(BrokerConfiguration).filter_by(broker_name=self.broker_name)
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
                'is_token_expired': False,  # Zerodha tokens don't expire like FYERS
                'last_connection_test': config.last_connection_test,
                'connection_status': config.connection_status,
                'error_message': config.error_message,
                'created_at': config.created_at,
                'updated_at': config.updated_at
            }
    
    def save_broker_config(self, config_data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
        """Save Zerodha broker configuration to database."""
        with self.db_manager.get_session() as session:
            # Check if config exists within this session
            query = session.query(BrokerConfiguration).filter_by(broker_name=self.broker_name)
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
                    broker_name=self.broker_name,
                    user_id=user_id
                )
                session.add(config)
            
            # Update fields
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
            session.refresh(config)
            
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
    
    def get_broker_stats(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get Zerodha broker statistics from database."""
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


class KiteAPIConnector:
    """Zerodha Kite Connect API connector for real-time connection testing and operations."""
    
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        
        # Initialize Kite Connect client if available
        if ZERODHA_AVAILABLE:
            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
        else:
            self.kite = None
            logger.warning("kiteconnect library not available, falling back to requests")
        
        # Fallback session for direct API calls
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {api_key}:{access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Zerodha Kite Connect API connection."""
        try:
            logger.info(f"Testing Zerodha connection with api_key: {self.api_key[:10]}...")
            start_time = time.time()
            
            # Try using the kiteconnect library first
            if self.kite:
                try:
                    logger.info("Using kiteconnect library for connection test")
                    # Use the profile endpoint to test connection
                    profile = self.kite.profile()
                    response_time = round((time.time() - start_time) * 1000, 2)
                    
                    logger.info(f"Zerodha API response successful, time: {response_time}ms")
                    
                    return {
                        'success': True,
                        'message': 'Connection successful',
                        'response_time': f"{response_time}ms",
                        'profile_data': profile,
                        'status_code': 200
                    }
                except Exception as e:
                    logger.warning(f"kiteconnect library failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API calls
            logger.info("Using direct API calls for connection test")
            url = "https://api.kite.trade/user/profile"
            
            response = self.session.get(url)
            response_time = round((time.time() - start_time) * 1000, 2)
            
            logger.info(f"Zerodha API response status: {response.status_code}, time: {response_time}ms")
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Zerodha connection test successful using requests")
                return {
                    'success': True,
                    'message': 'Connection successful',
                    'response_time': f"{response_time}ms",
                    'profile_data': data.get('data', {}),
                    'status_code': response.status_code
                }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Zerodha API error: {error_msg}")
                return {
                    'success': False,
                    'message': error_msg,
                    'response_time': f"{response_time}ms",
                    'status_code': response.status_code
                }
                
        except Exception as e:
            error_msg = f'Connection failed: {str(e)}'
            logger.error(f"Zerodha unexpected error: {error_msg}")
            return {
                'success': False,
                'message': error_msg,
                'response_time': '-',
                'status_code': 0
            }
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile information."""
        try:
            logger.info("Fetching Zerodha user profile")
            
            # Use Kite Connect client if available
            if self.kite:
                try:
                    profile = self.kite.profile()
                    logger.info("Zerodha profile fetched successfully using kiteconnect")
                    return {'data': profile}
                except Exception as e:
                    logger.warning(f"kiteconnect profile fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = "https://api.kite.trade/user/profile"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                logger.info("Zerodha profile fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching Zerodha profile: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching Zerodha profile: {error_msg}")
            return {'error': error_msg}
    
    def get_funds(self) -> Dict[str, Any]:
        """Get user funds."""
        try:
            logger.info("Fetching Zerodha user funds")
            
            # Use Kite Connect client if available
            if self.kite:
                try:
                    margins = self.kite.margins()
                    logger.info("Zerodha funds fetched successfully using kiteconnect")
                    return {'data': margins}
                except Exception as e:
                    logger.warning(f"kiteconnect funds fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = "https://api.kite.trade/user/margins"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                logger.info("Zerodha funds fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching Zerodha funds: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching Zerodha funds: {error_msg}")
            return {'error': error_msg}
    
    def get_holdings(self) -> Dict[str, Any]:
        """Get user holdings."""
        try:
            logger.info("Fetching Zerodha user holdings")
            
            # Use Kite Connect client if available
            if self.kite:
                try:
                    holdings = self.kite.holdings()
                    logger.info("Zerodha holdings fetched successfully using kiteconnect")
                    return {'data': holdings}
                except Exception as e:
                    logger.warning(f"kiteconnect holdings fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = "https://api.kite.trade/portfolio/holdings"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                logger.info("Zerodha holdings fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching Zerodha holdings: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching Zerodha holdings: {error_msg}")
            return {'error': error_msg}
    
    def get_positions(self) -> Dict[str, Any]:
        """Get user positions."""
        try:
            logger.info("Fetching Zerodha user positions")
            
            # Use Kite Connect client if available
            if self.kite:
                try:
                    positions = self.kite.positions()
                    logger.info("Zerodha positions fetched successfully using kiteconnect")
                    return {'data': positions}
                except Exception as e:
                    logger.warning(f"kiteconnect positions fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = "https://api.kite.trade/portfolio/positions"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                logger.info("Zerodha positions fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching Zerodha positions: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching Zerodha positions: {error_msg}")
            return {'error': error_msg}
    
    def get_orderbook(self) -> Dict[str, Any]:
        """Get user orderbook."""
        try:
            logger.info("Fetching Zerodha user orderbook")
            
            # Use Kite Connect client if available
            if self.kite:
                try:
                    orders = self.kite.orders()
                    logger.info("Zerodha orderbook fetched successfully using kiteconnect")
                    return {'data': orders}
                except Exception as e:
                    logger.warning(f"kiteconnect orderbook fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = "https://api.kite.trade/orders"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                logger.info("Zerodha orderbook fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching Zerodha orderbook: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching Zerodha orderbook: {error_msg}")
            return {'error': error_msg}
    
    def get_tradebook(self) -> Dict[str, Any]:
        """Get user tradebook."""
        try:
            logger.info("Fetching Zerodha user tradebook")
            
            # Use Kite Connect client if available
            if self.kite:
                try:
                    trades = self.kite.trades()
                    logger.info("Zerodha tradebook fetched successfully using kiteconnect")
                    return {'data': trades}
                except Exception as e:
                    logger.warning(f"kiteconnect tradebook fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = "https://api.kite.trade/orders/trades"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                logger.info("Zerodha tradebook fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching Zerodha tradebook: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching Zerodha tradebook: {error_msg}")
            return {'error': error_msg}
    
    def get_quotes(self, symbols: str) -> Dict[str, Any]:
        """Get market quotes for symbols."""
        try:
            logger.info(f"Fetching Zerodha quotes for symbols: {symbols}")
            
            # Use Kite Connect client if available
            if self.kite:
                try:
                    # Convert symbols to list format expected by Kite
                    symbol_list = symbols.split(',')
                    quotes = self.kite.quote(symbol_list)
                    logger.info("Zerodha quotes fetched successfully using kiteconnect")
                    return {'data': quotes}
                except Exception as e:
                    logger.warning(f"kiteconnect quotes fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = "https://api.kite.trade/quote"
            params = {'i': symbols}
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                logger.info("Zerodha quotes fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching Zerodha quotes: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching Zerodha quotes: {error_msg}")
            return {'error': error_msg}
    
    def get_instruments(self, exchange: str = None) -> Dict[str, Any]:
        """Get instruments list."""
        try:
            logger.info(f"Fetching Zerodha instruments for exchange: {exchange or 'all'}")
            
            # Use Kite Connect client if available
            if self.kite:
                try:
                    instruments = self.kite.instruments(exchange)
                    logger.info("Zerodha instruments fetched successfully using kiteconnect")
                    return {'data': instruments}
                except Exception as e:
                    logger.warning(f"kiteconnect instruments fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = "https://api.kite.trade/instruments"
            if exchange:
                url += f"/{exchange}"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                logger.info("Zerodha instruments fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching Zerodha instruments: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching Zerodha instruments: {error_msg}")
            return {'error': error_msg}


def get_zerodha_service() -> ZerodhaService:
    """Get Zerodha service instance."""
    return ZerodhaService()
