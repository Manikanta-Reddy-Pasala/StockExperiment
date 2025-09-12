"""
FYERS Broker Service - Dedicated service for FYERS broker operations
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
    from fyers_apiv3 import fyersModel
    FYERS_AVAILABLE = True
except ImportError:
    FYERS_AVAILABLE = False


class FyersService:
    """Dedicated service for FYERS broker operations."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.broker_name = 'fyers'

    def _get_fyers_connector(self, user_id: int) -> 'FyersAPIConnector':
        """Helper to get an initialized FyersAPIConnector for a user."""
        config = self.get_broker_config(user_id)
        if not config or not config.get('client_id') or not config.get('access_token'):
            raise ValueError('FYERS credentials not configured or access token missing.')
        return FyersAPIConnector(config['client_id'], config['access_token'])

    def test_connection(self, user_id: int):
        """Test FYERS broker connection."""
        config = self.get_broker_config(user_id)
        if not config or not config.get('client_id') or not config.get('access_token'):
            raise ValueError('FYERS credentials not configured.')

        connector = FyersAPIConnector(config.get('client_id'), config.get('access_token'))
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

    def generate_auth_url(self, user_id: int) -> str:
        """Generate FYERS OAuth2 authorization URL."""
        config = self.get_broker_config(user_id)
        if not config or not config.get('client_id') or not config.get('api_secret'):
            raise ValueError('FYERS configuration not found. Please save your Client ID and Secret Key first.')

        oauth_flow = FyersOAuth2Flow(
            client_id=config.get('client_id'),
            secret_key=config.get('api_secret'),
            redirect_uri=config.get('redirect_url')
        )
        return oauth_flow.generate_auth_url(user_id)

    def exchange_auth_code(self, user_id: int, auth_code: str) -> dict:
        """Exchange FYERS auth code for an access token and save it."""
        config = self.get_broker_config(user_id)
        if not config or not config.get('client_id') or not config.get('api_secret'):
            raise ValueError('FYERS configuration not found.')

        oauth_flow = FyersOAuth2Flow(
            client_id=config.get('client_id'),
            secret_key=config.get('api_secret'),
            redirect_uri=config.get('redirect_url')
        )
        token_response = oauth_flow.exchange_auth_code_for_token(auth_code)

        if 'access_token' in token_response:
            access_token = token_response['access_token']

            # Save the new tokens
            self.save_broker_config({
                'access_token': access_token,
                'is_connected': True,
                'connection_status': 'connected'
            }, user_id)

            return {'success': True, 'access_token': access_token}
        else:
            raise ValueError(token_response.get('message', 'Failed to obtain access token'))

    def get_funds(self, user_id: int):
        """Get user funds."""
        connector = self._get_fyers_connector(user_id)
        return connector.get_funds()

    def get_holdings(self, user_id: int):
        """Get user holdings."""
        connector = self._get_fyers_connector(user_id)
        return connector.get_holdings()

    def get_positions(self, user_id: int):
        """Get user positions."""
        connector = self._get_fyers_connector(user_id)
        return connector.get_positions()

    def get_orderbook(self, user_id: int):
        """Get user orderbook."""
        connector = self._get_fyers_connector(user_id)
        return connector.get_orderbook()

    def get_tradebook(self, user_id: int):
        """Get user tradebook."""
        connector = self._get_fyers_connector(user_id)
        return connector.get_tradebook()

    def get_quotes(self, user_id: int, symbols: str):
        """Get market quotes for symbols."""
        connector = self._get_fyers_connector(user_id)
        return connector.get_quotes(symbols)

    def get_history(self, user_id: int, symbol: str, resolution: str, range_from: str, range_to: str):
        """Get historical data for a symbol."""
        connector = self._get_fyers_connector(user_id)
        return connector.get_history(symbol, resolution, range_from, range_to)

    def get_profile(self, user_id: int):
        """Get user profile."""
        connector = self._get_fyers_connector(user_id)
        return connector.get_profile()
    
    def is_token_expired(self, access_token: str) -> bool:
        """Check if FYERS access token is expired."""
        if not access_token:
            return True
        
        try:
            import jwt
            import datetime
            
            # Decode JWT token without verification to get expiration
            decoded = jwt.decode(access_token, options={"verify_signature": False})
            exp_timestamp = decoded.get('exp', 0)
            
            # Convert to datetime and check if expired
            exp_datetime = datetime.datetime.fromtimestamp(exp_timestamp)
            current_time = datetime.datetime.now()
            
            return current_time >= exp_datetime
        except Exception as e:
            logger.warning(f"Error checking token expiration: {e}")
            return True  # Assume expired if we can't check
    
    def get_broker_config(self, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get FYERS broker configuration from database."""
        with self.db_manager.get_session() as session:
            query = session.query(BrokerConfiguration).filter_by(broker_name=self.broker_name)
            if user_id:
                query = query.filter_by(user_id=user_id)
            else:
                query = query.filter_by(user_id=None)  # Global config
            
            config = query.first()
            if not config:
                return None
            
            # Check if token is expired
            is_expired = self.is_token_expired(config.access_token)
            
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
                'is_connected': config.is_connected and not is_expired,  # Mark as disconnected if token expired
                'is_token_expired': is_expired,
                'last_connection_test': config.last_connection_test,
                'connection_status': 'expired' if is_expired else config.connection_status,
                'error_message': config.error_message,
                'created_at': config.created_at,
                'updated_at': config.updated_at
            }
    
    def save_broker_config(self, config_data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
        """Save FYERS broker configuration to database."""
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
            
            # Update fields (storing directly without encryption for simplicity)
            # Only update fields that are provided in config_data
            if config_data.get('client_id') is not None:
                config.client_id = config_data.get('client_id', '')
            if config_data.get('access_token') is not None:
                config.access_token = config_data.get('access_token', '')
            if config_data.get('refresh_token') is not None:
                config.refresh_token = config_data.get('refresh_token', '')
            if config_data.get('api_key') is not None:
                config.api_key = config_data.get('api_key', '')
            if config_data.get('api_secret') is not None:
                config.api_secret = config_data.get('api_secret', '')
            if config_data.get('redirect_url') is not None:
                config.redirect_url = config_data.get('redirect_url', '')
            if config_data.get('app_type') is not None:
                config.app_type = config_data.get('app_type', '100')
            if 'is_active' in config_data:
                config.is_active = config_data.get('is_active', True)
            if 'is_connected' in config_data:
                config.is_connected = config_data.get('is_connected', False)
            if 'connection_status' in config_data:
                config.connection_status = config_data.get('connection_status', 'disconnected')
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
    
    def get_broker_stats(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get FYERS broker statistics from database."""
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


class FyersOAuth2Flow:
    """FYERS OAuth2 authentication flow handler."""
    
    def __init__(self, client_id: str, secret_key: str, redirect_uri: str):
        self.client_id = client_id
        self.secret_key = secret_key
        self.redirect_uri = redirect_uri
        self.grant_type = "authorization_code"
        self.response_type = "code"
        self.state = "sample"
    
    def generate_auth_url(self, user_id: int = 1) -> str:
        """Generate the authorization URL for user login."""
        if not FYERS_AVAILABLE:
            raise Exception("fyers-apiv3 library not available")
        
        try:
            # Use the configured redirect URI from initialization
            # The redirect_uri should be the one configured in the FYERS app
            
            # Create session model for OAuth flow
            app_session = fyersModel.SessionModel(
                client_id=self.client_id,
                redirect_uri=self.redirect_uri,  # Use the actual redirect URI from config
                response_type=self.response_type,
                state=str(user_id),  # Pass user_id in state for callback
                secret_key=self.secret_key,
                grant_type=self.grant_type
            )
            
            # Generate the authorization URL
            auth_url = app_session.generate_authcode()
            logger.info(f"Generated FYERS authorization URL: {auth_url}")
            return auth_url
            
        except Exception as e:
            logger.error(f"Error generating FYERS auth URL: {str(e)}")
            raise
    
    def exchange_auth_code_for_token(self, auth_code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        if not FYERS_AVAILABLE:
            raise Exception("fyers-apiv3 library not available")
        
        try:
            # Create session model for OAuth flow
            app_session = fyersModel.SessionModel(
                client_id=self.client_id,
                redirect_uri=self.redirect_uri,
                response_type=self.response_type,
                state=self.state,
                secret_key=self.secret_key,
                grant_type=self.grant_type
            )
            
            # Set the auth code and generate token
            app_session.set_token(auth_code)
            response = app_session.generate_token()
            
            logger.info("Successfully exchanged auth code for access token")
            return response
            
        except Exception as e:
            logger.error(f"Error exchanging auth code for token: {str(e)}")
            raise


class FyersAPIConnector:
    """FYERS API connector for real-time connection testing and operations."""
    
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.base_url = "https://api-t1.fyers.in/api/v3"
        
        # Initialize FYERS API client if available
        if FYERS_AVAILABLE:
            self.fyers_client = fyersModel.FyersModel(
                token=access_token,
                is_async=False,
                client_id=client_id,
                log_path=""
            )
        else:
            self.fyers_client = None
            logger.warning("fyers-apiv3 library not available, falling back to requests")
        
        # Fallback session for direct API calls
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'{client_id}:{access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def test_connection(self) -> Dict[str, Any]:
        """Test FYERS API connection by making a real API call."""
        try:
            logger.info(f"Testing FYERS connection with client_id: {self.client_id[:10]}...")
            start_time = time.time()
            
            # Try using the fyers-apiv3 library first
            if self.fyers_client:
                try:
                    logger.info("Using fyers-apiv3 library for connection test")
                    # Use the profile endpoint to test connection
                    response = self.fyers_client.get_profile()
                    response_time = round((time.time() - start_time) * 1000, 2)
                    
                    logger.info(f"FYERS API response status: {response.get('s', 'unknown')}, time: {response_time}ms")
                    
                    if response.get('s') == 'ok':
                        logger.info("FYERS connection test successful using fyers-apiv3")
                        return {
                            'success': True,
                            'message': 'Connection successful',
                            'response_time': f"{response_time}ms",
                            'profile_data': response.get('profile', {}),
                            'status_code': 200
                        }
                    else:
                        error_msg = f"API Error: {response.get('message', 'Unknown error')}"
                        logger.warning(f"FYERS API error: {error_msg}")
                        return {
                            'success': False,
                            'message': error_msg,
                            'response_time': f"{response_time}ms",
                            'status_code': 400
                        }
                except Exception as e:
                    logger.warning(f"fyers-apiv3 library failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API calls
            logger.info("Using direct API calls for connection test")
            url = f"{self.base_url}/profile"
            logger.info(f"Making request to FYERS API: {url}")
            
            # Try different authentication methods
            auth_methods = [
                {'Authorization': f'{self.client_id}:{self.access_token}'},
                {'Authorization': f'Bearer {self.client_id}:{self.access_token}'},
                {'Authorization': f'Bearer {self.access_token}'}
            ]
            
            for i, headers in enumerate(auth_methods):
                try:
                    logger.info(f"Trying authentication method {i+1}: {headers}")
                    test_session = requests.Session()
                    test_session.headers.update({
                        **headers,
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    })
                    
                    response = test_session.get(url)
                    response_time = round((time.time() - start_time) * 1000, 2)
                    
                    logger.info(f"FYERS API response status: {response.status_code}, time: {response_time}ms")
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('s') == 'ok':
                            logger.info(f"FYERS connection test successful with method {i+1}")
                            return {
                                'success': True,
                                'message': 'Connection successful',
                                'response_time': f"{response_time}ms",
                                'profile_data': data.get('profile', {}),
                                'status_code': response.status_code
                            }
                    
                    # Log the response for debugging
                    response_text = response.text
                    if len(response_text) > 500:
                        logger.info(f"FYERS API response content (first 500 chars): {response_text[:500]}...")
                    else:
                        logger.info(f"FYERS API response content: {response_text}")
                        
                except Exception as e:
                    logger.warning(f"Authentication method {i+1} failed: {str(e)}")
                    continue
            
            # If all methods failed
            error_msg = "All authentication methods failed"
            logger.error(f"FYERS connection failed: {error_msg}")
            return {
                'success': False,
                'message': error_msg,
                'response_time': f"{round((time.time() - start_time) * 1000, 2)}ms",
                'status_code': 0
            }
                
        except Exception as e:
            error_msg = f'Connection failed: {str(e)}'
            logger.error(f"FYERS unexpected error: {error_msg}")
            return {
                'success': False,
                'message': error_msg,
                'response_time': '-',
                'status_code': 0
            }
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile information."""
        try:
            logger.info("Fetching FYERS user profile")
            
            # Use FYERS API client if available
            if self.fyers_client:
                try:
                    response = self.fyers_client.get_profile()
                    logger.info("FYERS profile fetched successfully using fyers-apiv3")
                    return response
                except Exception as e:
                    logger.warning(f"fyers-apiv3 profile fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = f"{self.base_url}/profile"
            response = self.session.get(url, params={'access_token': self.access_token})
            if response.status_code == 200:
                data = response.json()
                logger.info("FYERS profile fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching FYERS profile: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching FYERS profile: {error_msg}")
            return {'error': error_msg}
    
    def get_funds(self) -> Dict[str, Any]:
        """Get user funds."""
        try:
            logger.info("Fetching FYERS user funds")
            
            # Use FYERS API client if available
            if self.fyers_client:
                try:
                    response = self.fyers_client.funds()
                    logger.info("FYERS funds fetched successfully using fyers-apiv3")
                    return response
                except Exception as e:
                    logger.warning(f"fyers-apiv3 funds fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = f"{self.base_url}/funds"
            response = self.session.get(url, params={'access_token': self.access_token})
            if response.status_code == 200:
                data = response.json()
                logger.info("FYERS funds fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching FYERS funds: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching FYERS funds: {error_msg}")
            return {'error': error_msg}
    
    def get_holdings(self) -> Dict[str, Any]:
        """Get user holdings."""
        try:
            logger.info("Fetching FYERS user holdings")
            
            # Use FYERS API client if available
            if self.fyers_client:
                try:
                    response = self.fyers_client.holdings()
                    logger.info("FYERS holdings fetched successfully using fyers-apiv3")
                    return response
                except Exception as e:
                    logger.warning(f"fyers-apiv3 holdings fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = f"{self.base_url}/holdings"
            response = self.session.get(url, params={'access_token': self.access_token})
            if response.status_code == 200:
                data = response.json()
                logger.info("FYERS holdings fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching FYERS holdings: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching FYERS holdings: {error_msg}")
            return {'error': error_msg}
    
    def get_positions(self) -> Dict[str, Any]:
        """Get user positions."""
        try:
            logger.info("Fetching FYERS user positions")
            
            # Use FYERS API client if available
            if self.fyers_client:
                try:
                    response = self.fyers_client.positions()
                    logger.info("FYERS positions fetched successfully using fyers-apiv3")
                    return response
                except Exception as e:
                    logger.warning(f"fyers-apiv3 positions fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = f"{self.base_url}/positions"
            response = self.session.get(url, params={'access_token': self.access_token})
            if response.status_code == 200:
                data = response.json()
                logger.info("FYERS positions fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching FYERS positions: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching FYERS positions: {error_msg}")
            return {'error': error_msg}
    
    def get_tradebook(self) -> Dict[str, Any]:
        """Get user tradebook."""
        try:
            logger.info("Fetching FYERS user tradebook")
            
            # Use FYERS API client if available
            if self.fyers_client:
                try:
                    response = self.fyers_client.tradebook()
                    logger.info("FYERS tradebook fetched successfully using fyers-apiv3")
                    return response
                except Exception as e:
                    logger.warning(f"fyers-apiv3 tradebook fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = f"{self.base_url}/tradebook"
            response = self.session.get(url, params={'access_token': self.access_token})
            if response.status_code == 200:
                data = response.json()
                logger.info("FYERS tradebook fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching FYERS tradebook: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching FYERS tradebook: {error_msg}")
            return {'error': error_msg}
    
    def get_orderbook(self) -> Dict[str, Any]:
        """Get user orderbook."""
        try:
            logger.info("Fetching FYERS user orderbook")
            
            # Use FYERS API client if available
            if self.fyers_client:
                try:
                    response = self.fyers_client.orderbook()
                    logger.info("FYERS orderbook fetched successfully using fyers-apiv3")
                    return response
                except Exception as e:
                    logger.warning(f"fyers-apiv3 orderbook fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = f"{self.base_url}/orderbook"
            response = self.session.get(url, params={'access_token': self.access_token})
            if response.status_code == 200:
                data = response.json()
                logger.info("FYERS orderbook fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching FYERS orderbook: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching FYERS orderbook: {error_msg}")
            return {'error': error_msg}
    
    def get_quotes(self, symbols: str) -> Dict[str, Any]:
        """Get market quotes for symbols."""
        try:
            logger.info(f"Fetching FYERS quotes for symbols: {symbols}")
            
            # Use FYERS API client if available
            if self.fyers_client:
                try:
                    data = {"symbols": symbols}
                    response = self.fyers_client.quotes(data)
                    logger.info("FYERS quotes fetched successfully using fyers-apiv3")
                    return response
                except Exception as e:
                    logger.warning(f"fyers-apiv3 quotes fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = f"{self.base_url}/quotes"
            params = {'symbols': symbols, 'access_token': self.access_token}
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                logger.info("FYERS quotes fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching FYERS quotes: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching FYERS quotes: {error_msg}")
            return {'error': error_msg}
    
    def get_history(self, symbol: str, resolution: str = "D", range_from: str = None, range_to: str = None) -> Dict[str, Any]:
        """Get historical data for a symbol."""
        try:
            logger.info(f"Fetching FYERS historical data for symbol: {symbol}")
            
            # Use FYERS API client if available
            if self.fyers_client:
                try:
                    data = {
                        "symbol": symbol,
                        "resolution": resolution,
                        "date_format": "0",
                        "range_from": range_from or "1622097600",
                        "range_to": range_to or "1622097685",
                        "cont_flag": "1"
                    }
                    response = self.fyers_client.history(data)
                    logger.info("FYERS historical data fetched successfully using fyers-apiv3")
                    return response
                except Exception as e:
                    logger.warning(f"fyers-apiv3 historical data fetch failed, falling back to requests: {str(e)}")
            
            # Fallback to direct API call
            url = f"{self.base_url}/history"
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'date_format': '0',
                'range_from': range_from or '1622097600',
                'range_to': range_to or '1622097685',
                'cont_flag': '1',
                'access_token': self.access_token
            }
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                logger.info("FYERS historical data fetched successfully using requests")
                return data
            else:
                error_msg = f'HTTP {response.status_code}: {response.text}'
                logger.error(f"Error fetching FYERS historical data: {error_msg}")
                return {'error': error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Exception while fetching FYERS historical data: {error_msg}")
            return {'error': error_msg}


def get_fyers_service() -> FyersService:
    """Get FYERS service instance."""
    return FyersService()
