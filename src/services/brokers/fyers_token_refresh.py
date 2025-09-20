"""
FYERS Token Refresh Service - Handles automatic token refresh for FYERS API
"""
import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class FyersTokenRefreshService:
    """Service for refreshing FYERS tokens automatically."""
    
    def __init__(self):
        # Official library handles everything internally
        pass
    
    def refresh_fyers_token(self, user_id: int, refresh_token: str) -> Optional[Dict[str, Any]]:
        """
        Refresh FYERS access token using refresh token.
        
        Note: FYERS doesn't provide a traditional refresh token flow.
        Instead, we need to re-authenticate using the stored credentials.
        This method will attempt to use the stored client_id and api_secret
        to generate a new auth URL, but the user will need to complete the OAuth flow.
        
        Args:
            user_id (int): User ID
            refresh_token (str): Refresh token (not used for FYERS, but kept for interface compatibility)
            
        Returns:
            Dict: New token data or None if refresh failed
        """
        logger.info(f"Attempting to refresh FYERS token for user {user_id}")
        
        # For FYERS, we need to get the stored credentials and generate a new auth URL
        # The actual token refresh requires user interaction through OAuth flow
        try:
            from ..token_manager_service import get_token_manager
            from src.models.database import get_database_manager
            from src.models.models import BrokerConfiguration
            
            db_manager = get_database_manager()
            
            with db_manager.get_session() as session:
                config = session.query(BrokerConfiguration).filter_by(
                    broker_name='fyers',
                    user_id=user_id
                ).first()
                
                if not config or not config.client_id or not config.api_secret:
                    logger.error(f"FYERS credentials not found for user {user_id}")
                    return None
                
                # Check if redirect URL is configured
                if not config.redirect_url:
                    logger.error(f"FYERS redirect URL not configured for user {user_id}")
                    return None
                
                # Generate new auth URL for re-authentication
                auth_url = self._generate_fyers_auth_url(
                    config.client_id,
                    config.api_secret,
                    config.redirect_url,
                    user_id
                )
                
                # Update the configuration to indicate re-authentication is needed
                config.connection_status = 'reauth_required'
                config.is_connected = False
                config.updated_at = datetime.utcnow()
                session.commit()
                
                logger.warning(f"FYERS token refresh requires re-authentication for user {user_id}")
                logger.info(f"Generated new auth URL: {auth_url}")
                
                # Return None to indicate manual re-authentication is required
                # The frontend should detect this and prompt the user to re-authenticate
                return None
                
        except Exception as e:
            logger.error(f"Error refreshing FYERS token for user {user_id}: {e}")
            return None
    
    def _generate_fyers_auth_url(self, client_id: str, api_secret: str, redirect_uri: str, user_id: int) -> str:
        """
        Generate FYERS OAuth2 authorization URL.
        
        Args:
            client_id (str): FYERS client ID
            api_secret (str): FYERS API secret
            redirect_uri (str): OAuth redirect URI
            user_id (int): User ID
            
        Returns:
            str: Authorization URL
        """
        try:
            from fyers_apiv3 import fyersModel
            
            # Create session model for OAuth flow
            app_session = fyersModel.SessionModel(
                client_id=client_id,
                redirect_uri=redirect_uri,
                response_type="code",
                state=str(user_id),
                secret_key=api_secret,
                grant_type="authorization_code"
            )
            
            # Generate the authorization URL
            auth_url = app_session.generate_authcode()
            logger.info(f"Generated FYERS authorization URL for user {user_id}")
            return auth_url
            
        except Exception as e:
            logger.error(f"Error generating FYERS auth URL: {e}")
            raise
    
    def check_token_validity(self, access_token: str) -> Dict[str, Any]:
        """
        Check if a FYERS token is valid by making a test API call.
        
        Args:
            access_token (str): FYERS access token
            
        Returns:
            Dict: Token validity information
        """
        try:
            # Make a test API call to check token validity
            headers = {
                'Authorization': f'{access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Try to get profile information
            response = requests.get(
                f"{self.base_url}/profile",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('s') == 'ok':
                    return {
                        'is_valid': True,
                        'message': 'Token is valid',
                        'profile_data': data.get('profile', {})
                    }
                else:
                    return {
                        'is_valid': False,
                        'message': f"API Error: {data.get('message', 'Unknown error')}",
                        'error_code': data.get('code')
                    }
            else:
                return {
                    'is_valid': False,
                    'message': f"HTTP {response.status_code}: {response.text}",
                    'status_code': response.status_code
                }
                
        except Exception as e:
            return {
                'is_valid': False,
                'message': f"Error checking token validity: {str(e)}",
                'error': str(e)
            }
    
    def get_token_info(self, access_token: str) -> Dict[str, Any]:
        """
        Get information about a FYERS token (expiration, etc.).
        
        Args:
            access_token (str): FYERS access token
            
        Returns:
            Dict: Token information
        """
        try:
            import jwt
            from datetime import datetime
            
            # Decode JWT token without verification
            decoded = jwt.decode(access_token, options={"verify_signature": False})
            
            exp_timestamp = decoded.get('exp', 0)
            iat_timestamp = decoded.get('iat', 0)
            
            exp_datetime = datetime.fromtimestamp(exp_timestamp) if exp_timestamp else None
            iat_datetime = datetime.fromtimestamp(iat_timestamp) if iat_timestamp else None
            
            current_time = datetime.now()
            is_expired = exp_datetime and current_time >= exp_datetime
            
            time_until_expiry = None
            if exp_datetime:
                time_until_expiry = exp_datetime - current_time
            
            return {
                'issued_at': iat_datetime.isoformat() if iat_datetime else None,
                'expires_at': exp_datetime.isoformat() if exp_datetime else None,
                'is_expired': is_expired,
                'time_until_expiry': str(time_until_expiry) if time_until_expiry else None,
                'token_type': decoded.get('token_type', 'Bearer'),
                'scope': decoded.get('scope', ''),
                'client_id': decoded.get('client_id', ''),
                'user_id': decoded.get('user_id', '')
            }
            
        except Exception as e:
            logger.error(f"Error getting token info: {e}")
            return {
                'error': str(e),
                'is_expired': True  # Assume expired if we can't decode
            }


def register_fyers_refresh_callback():
    """Register FYERS token refresh callback with the token manager."""
    try:
        from ..token_manager_service import get_token_manager
        
        token_manager = get_token_manager()
        fyers_refresh_service = FyersTokenRefreshService()
        
        # Register the refresh callback
        token_manager.register_refresh_callback(
            'fyers',
            fyers_refresh_service.refresh_fyers_token
        )
        
        logger.info("Successfully registered FYERS token refresh callback")
        
    except Exception as e:
        logger.error(f"Error registering FYERS refresh callback: {e}")


# Auto-register the callback when this module is imported
register_fyers_refresh_callback()
