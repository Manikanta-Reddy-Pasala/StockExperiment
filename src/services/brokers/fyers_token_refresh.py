"""
FYERS Token Refresh Service - Uses FYERS v3 validate-refresh-token API
to refresh access tokens without browser login.

Endpoint: POST https://api-t1.fyers.in/api/v3/validate-refresh-token
Required: appIdHash (SHA256 of client_id:secret_key), refresh_token, pin, grant_type
"""
import hashlib
import logging
import os
import requests
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

FYERS_REFRESH_TOKEN_URL = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
FYERS_PIN = os.getenv('FYERS_PIN', '')


class FyersTokenRefreshService:
    """Service for refreshing FYERS tokens using the v3 refresh token API."""

    def __init__(self):
        pass

    def _compute_app_id_hash(self, client_id: str, secret_key: str) -> str:
        """Compute SHA256 hash of client_id:secret_key as required by FYERS API."""
        input_string = f"{client_id}:{secret_key}"
        return hashlib.sha256(input_string.encode()).hexdigest()

    def refresh_fyers_token(self, user_id: int, refresh_token: str) -> Optional[Dict[str, Any]]:
        """
        Refresh FYERS access token using the v3 validate-refresh-token API.

        Args:
            user_id: User ID
            refresh_token: The refresh token from last auth

        Returns:
            Dict with new token data, or None if refresh failed
        """
        logger.info(f"Attempting API-based FYERS token refresh for user {user_id}")

        try:
            from src.models.database import get_database_manager
            from src.models.models import BrokerConfiguration

            db_manager = get_database_manager()

            with db_manager.get_session() as session:
                config = session.query(BrokerConfiguration).filter_by(
                    broker_name='fyers',
                    user_id=user_id
                ).first()

                if not config:
                    logger.error(f"FYERS config not found for user {user_id}")
                    return None

                client_id = config.client_id
                secret_key = config.api_secret
                stored_refresh_token = refresh_token or config.refresh_token
                pin = FYERS_PIN

                if not all([client_id, secret_key, stored_refresh_token]):
                    logger.error(f"Missing credentials for user {user_id}: "
                                 f"client_id={'set' if client_id else 'MISSING'}, "
                                 f"secret_key={'set' if secret_key else 'MISSING'}, "
                                 f"refresh_token={'set' if stored_refresh_token else 'MISSING'}")
                    self._mark_reauth_required(config, session, "Missing credentials for refresh")
                    return None

                if not pin:
                    logger.error(f"FYERS_PIN not set in environment for user {user_id}")
                    self._mark_reauth_required(config, session, "FYERS_PIN not configured")
                    return None

                # Call FYERS v3 refresh token API
                app_id_hash = self._compute_app_id_hash(client_id, secret_key)

                payload = {
                    "grant_type": "refresh_token",
                    "appIdHash": app_id_hash,
                    "refresh_token": stored_refresh_token,
                    "pin": pin
                }

                headers = {"Content-Type": "application/json"}

                logger.info(f"Calling FYERS validate-refresh-token API for user {user_id}")
                response = requests.post(
                    FYERS_REFRESH_TOKEN_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                result = response.json()
                logger.info(f"FYERS refresh response for user {user_id}: "
                            f"status={result.get('s')}, code={result.get('code')}")

                if result.get('s') == 'ok' and result.get('access_token'):
                    new_access_token = result['access_token']
                    new_refresh_token = result.get('refresh_token', stored_refresh_token)

                    # Update tokens in database
                    config.access_token = new_access_token
                    if new_refresh_token:
                        config.refresh_token = new_refresh_token
                    config.is_connected = True
                    config.connection_status = 'connected'
                    config.error_message = None
                    config.updated_at = datetime.utcnow()
                    session.commit()

                    logger.info(f"Successfully refreshed FYERS token for user {user_id} via API")

                    return {
                        'access_token': new_access_token,
                        'refresh_token': new_refresh_token,
                        'refreshed_at': datetime.utcnow().isoformat()
                    }
                else:
                    error_msg = result.get('message', 'Unknown error')
                    error_code = result.get('code', 'UNKNOWN')
                    logger.error(f"FYERS refresh failed for user {user_id}: "
                                 f"code={error_code}, message={error_msg}")

                    # If refresh token is invalid/expired, mark for re-auth
                    self._mark_reauth_required(config, session,
                                               f"Refresh failed: {error_msg} (code: {error_code})")
                    return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error refreshing FYERS token for user {user_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error refreshing FYERS token for user {user_id}: {e}", exc_info=True)
            return None

    def _mark_reauth_required(self, config, session, reason: str):
        """Mark broker config as requiring re-authentication."""
        config.connection_status = 'reauth_required'
        config.is_connected = False
        config.error_message = reason
        config.updated_at = datetime.utcnow()
        session.commit()
        logger.warning(f"Marked user {config.user_id} as reauth_required: {reason}")

    def check_token_validity(self, access_token: str) -> Dict[str, Any]:
        """Check if a FYERS token is valid by making a test API call."""
        try:
            headers = {
                'Authorization': f'{access_token}',
                'Content-Type': 'application/json'
            }

            response = requests.get(
                "https://api-t1.fyers.in/api/v3/profile",
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('s') == 'ok':
                    return {'is_valid': True, 'message': 'Token is valid'}
                else:
                    return {
                        'is_valid': False,
                        'message': f"API Error: {data.get('message', 'Unknown error')}",
                        'error_code': data.get('code')
                    }
            else:
                return {
                    'is_valid': False,
                    'message': f"HTTP {response.status_code}",
                    'status_code': response.status_code
                }

        except Exception as e:
            return {'is_valid': False, 'message': str(e)}

    def get_token_info(self, access_token: str) -> Dict[str, Any]:
        """Get expiration info from a FYERS JWT token."""
        try:
            import jwt

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
                'client_id': decoded.get('client_id', ''),
                'user_id': decoded.get('user_id', '')
            }

        except Exception as e:
            logger.error(f"Error getting token info: {e}")
            return {'error': str(e), 'is_expired': True}


def register_fyers_refresh_callback():
    """Register FYERS token refresh callback with the token manager."""
    try:
        from ..utils.token_manager_service import get_token_manager

        token_manager = get_token_manager()
        fyers_refresh_service = FyersTokenRefreshService()

        token_manager.register_refresh_callback(
            'fyers',
            fyers_refresh_service.refresh_fyers_token
        )

        logger.info("Registered FYERS API-based token refresh callback")

    except Exception as e:
        logger.error(f"Error registering FYERS refresh callback: {e}")


# Auto-register on import
register_fyers_refresh_callback()
