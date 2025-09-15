"""
Fyers Service

This service provides comprehensive Fyers broker integration with standardized
API endpoints and response formats.
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime

try:
    from ...models.database import get_database_manager
    from ...models.models import BrokerConfiguration
    from ...utils.api_logger import APILogger
    from .fyers import create_fyers_api, create_fyers_auth
except ImportError:
    from models.database import get_database_manager
    from models.models import BrokerConfiguration
    from utils.api_logger import APILogger
    from .fyers import create_fyers_api, create_fyers_auth

logger = logging.getLogger(__name__)


class FyersService:
    """
    Comprehensive Fyers service with standardized API implementation.
    """
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.broker_name = 'fyers'
    
    def _get_api_instance(self, user_id: int):
        """Get standardized API instance for user."""
        config = self.get_broker_config(user_id)
        if not config:
            raise ValueError('Fyers configuration not found')
        
        api_key = config.get('client_id')
        api_secret = config.get('api_secret')
        access_token = config.get('access_token')
        
        if not all([api_key, api_secret, access_token]):
            raise ValueError('Incomplete Fyers configuration')
        
        return create_fyers_api(api_key, api_secret, access_token)
    
    def get_broker_config(self, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get Fyers broker configuration from database."""
        with self.db_manager.get_session() as session:
            query = session.query(BrokerConfiguration).filter_by(broker_name=self.broker_name)
            if user_id:
                query = query.filter_by(user_id=user_id)
            else:
                query = query.filter_by(user_id=None)
            
            config = query.first()
            if not config:
                return None
            
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
    
    def save_broker_config(self, config_data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
        """Save Fyers broker configuration to database."""
        with self.db_manager.get_session() as session:
            query = session.query(BrokerConfiguration).filter_by(broker_name=self.broker_name)
            if user_id:
                query = query.filter_by(user_id=user_id)
            else:
                query = query.filter_by(user_id=None)
            
            existing_config = query.first()
            
            if existing_config:
                config = existing_config
            else:
                config = BrokerConfiguration(
                    broker_name=self.broker_name,
                    user_id=user_id
                )
                session.add(config)
            
            # Update fields
            for key, value in config_data.items():
                if hasattr(config, key) and value is not None:
                    setattr(config, key, value)
            
            config.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(config)
            
            return self.get_broker_config(user_id)
    
    # Authentication Methods
    def generate_auth_url(self, user_id: int) -> str:
        """Generate Fyers OAuth2 authorization URL."""
        config = self.get_broker_config(user_id)
        if not config or not config.get('client_id') or not config.get('api_secret'):
            raise ValueError('Fyers configuration not found. Please save your Client ID and Secret Key first.')
        
        auth = create_fyers_auth(
            client_id=config.get('client_id'),
            secret_key=config.get('api_secret'),
            redirect_uri=config.get('redirect_url')
        )
        
        return auth.generate_auth_url(str(user_id))
    
    def exchange_auth_code(self, user_id: int, auth_code: str) -> dict:
        """Exchange Fyers auth code for access token."""
        config = self.get_broker_config(user_id)
        if not config or not config.get('client_id') or not config.get('api_secret'):
            raise ValueError('Fyers configuration not found.')
        
        auth = create_fyers_auth(
            client_id=config.get('client_id'),
            secret_key=config.get('api_secret'),
            redirect_uri=config.get('redirect_url')
        )
        
        token_response = auth.generate_access_token(auth_code)
        
        if token_response.get('status') == 'success':
            access_token = token_response.get('access_token')
            
            # Save the new token
            self.save_broker_config({
                'access_token': access_token,
                'is_connected': True,
                'connection_status': 'connected'
            }, user_id)
            
            return {'success': True, 'access_token': access_token}
        else:
            raise ValueError(token_response.get('message', 'Failed to obtain access token'))
    
    def test_connection(self, user_id: int):
        """Test Fyers broker connection using standardized API."""
        try:
            api = self._get_api_instance(user_id)
            result = api.login()
            
            # Update connection status in database
            with self.db_manager.get_session() as session:
                db_config = session.query(BrokerConfiguration).filter_by(
                    broker_name=self.broker_name, user_id=user_id
                ).first()
                
                if db_config:
                    success = result.get('status') == 'success'
                    db_config.is_connected = success
                    db_config.connection_status = 'connected' if success else 'disconnected'
                    db_config.last_connection_test = datetime.utcnow()
                    db_config.error_message = result.get('message', '') if not success else None
                    session.commit()
            
            return {
                'success': result.get('status') == 'success',
                'message': result.get('message', ''),
                'response_time': '-'
            }
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return {
                'success': False,
                'message': str(e),
                'response_time': '-'
            }
    
    # Standardized API Methods
    def login(self, user_id: int):
        """Login to Fyers using standardized format."""
        APILogger.log_request("FyersService", "login", {}, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.login()
            APILogger.log_response("FyersService", "login", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "login", e, {}, user_id)
            raise
    
    def placeorder(self, user_id: int, symbol: str, quantity: str, action: str,
                   product: str, pricetype: str, price: str = "0",
                   trigger_price: str = "0", disclosed_quantity: str = "0",
                   validity: str = "DAY", tag: str = ""):
        """Place order using standardized format."""
        request_data = {
            'symbol': symbol, 'quantity': quantity, 'action': action,
            'product': product, 'pricetype': pricetype, 'price': price,
            'trigger_price': trigger_price, 'disclosed_quantity': disclosed_quantity,
            'validity': validity, 'tag': tag
        }
        
        APILogger.log_request("FyersService", "placeorder", request_data, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.placeorder(
                symbol=symbol, quantity=quantity, action=action,
                product=product, pricetype=pricetype, price=price,
                trigger_price=trigger_price, disclosed_quantity=disclosed_quantity,
                validity=validity, tag=tag
            )
            APILogger.log_response("FyersService", "placeorder", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "placeorder", e, request_data, user_id)
            raise
    
    def modifyorder(self, user_id: int, orderid: str, symbol: str = "", 
                    quantity: str = "", price: str = "", trigger_price: str = "",
                    disclosed_quantity: str = "", validity: str = ""):
        """Modify order using standardized format."""
        request_data = {
            'orderid': orderid, 'symbol': symbol, 'quantity': quantity,
            'price': price, 'trigger_price': trigger_price,
            'disclosed_quantity': disclosed_quantity, 'validity': validity
        }
        
        APILogger.log_request("FyersService", "modifyorder", request_data, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.modifyorder(
                orderid=orderid, symbol=symbol, quantity=quantity,
                price=price, trigger_price=trigger_price,
                disclosed_quantity=disclosed_quantity, validity=validity
            )
            APILogger.log_response("FyersService", "modifyorder", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "modifyorder", e, request_data, user_id)
            raise
    
    def cancelorder(self, user_id: int, orderid: str):
        """Cancel order using standardized format."""
        request_data = {'orderid': orderid}
        
        APILogger.log_request("FyersService", "cancelorder", request_data, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.cancelorder(orderid=orderid)
            APILogger.log_response("FyersService", "cancelorder", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "cancelorder", e, request_data, user_id)
            raise
    
    def placesmartorder(self, user_id: int, symbol: str, action: str, product: str,
                       quantity: str = "", position_size: str = "", price: str = "0",
                       trigger_price: str = "0", pricetype: str = "MARKET",
                       strategy: str = "", tag: str = ""):
        """Place smart order using standardized format."""
        request_data = {
            'symbol': symbol, 'action': action, 'product': product,
            'quantity': quantity, 'position_size': position_size,
            'price': price, 'trigger_price': trigger_price,
            'pricetype': pricetype, 'strategy': strategy, 'tag': tag
        }
        
        APILogger.log_request("FyersService", "placesmartorder", request_data, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.placesmartorder(
                symbol=symbol, action=action, product=product,
                quantity=quantity, position_size=position_size,
                price=price, trigger_price=trigger_price,
                pricetype=pricetype, strategy=strategy, tag=tag
            )
            APILogger.log_response("FyersService", "placesmartorder", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "placesmartorder", e, request_data, user_id)
            raise
    
    def orderbook(self, user_id: int):
        """Get orderbook using standardized format."""
        APILogger.log_request("FyersService", "orderbook", {}, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.orderbook()
            APILogger.log_response("FyersService", "orderbook", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "orderbook", e, {}, user_id)
            raise
    
    def tradebook(self, user_id: int):
        """Get tradebook using standardized format."""
        APILogger.log_request("FyersService", "tradebook", {}, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.tradebook()
            APILogger.log_response("FyersService", "tradebook", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "tradebook", e, {}, user_id)
            raise
    
    def positions(self, user_id: int):
        """Get positions using standardized format."""
        APILogger.log_request("FyersService", "positions", {}, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.positions()
            APILogger.log_response("FyersService", "positions", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "positions", e, {}, user_id)
            raise
    
    def holdings(self, user_id: int):
        """Get holdings using standardized format."""
        APILogger.log_request("FyersService", "holdings", {}, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.holdings()
            APILogger.log_response("FyersService", "holdings", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "holdings", e, {}, user_id)
            raise
    
    def funds(self, user_id: int):
        """Get funds using standardized format."""
        APILogger.log_request("FyersService", "funds", {}, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.funds()
            APILogger.log_response("FyersService", "funds", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "funds", e, {}, user_id)
            raise
    
    def quotes(self, user_id: int, symbol: str, exchange: str = ""):
        """Get quotes using standardized format."""
        request_data = {'symbol': symbol, 'exchange': exchange}
        
        APILogger.log_request("FyersService", "quotes", request_data, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.quotes(symbol=symbol, exchange=exchange)
            APILogger.log_response("FyersService", "quotes", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "quotes", e, request_data, user_id)
            raise
    
    def depth(self, user_id: int, symbol: str, exchange: str = ""):
        """Get market depth using standardized format."""
        request_data = {'symbol': symbol, 'exchange': exchange}
        
        APILogger.log_request("FyersService", "depth", request_data, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.depth(symbol=symbol, exchange=exchange)
            APILogger.log_response("FyersService", "depth", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "depth", e, request_data, user_id)
            raise
    
    def history(self, user_id: int, symbol: str, exchange: str, interval: str,
                start_date: str, end_date: str):
        """Get historical data using standardized format."""
        request_data = {
            'symbol': symbol, 'exchange': exchange, 'interval': interval,
            'start_date': start_date, 'end_date': end_date
        }
        
        APILogger.log_request("FyersService", "history", request_data, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.history(
                symbol=symbol, exchange=exchange, interval=interval,
                start_date=start_date, end_date=end_date
            )
            APILogger.log_response("FyersService", "history", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "history", e, request_data, user_id)
            raise
    
    def search(self, user_id: int, symbol: str, exchange: str = ""):
        """Search symbols using standardized format."""
        request_data = {'symbol': symbol, 'exchange': exchange}
        
        APILogger.log_request("FyersService", "search", request_data, user_id)
        try:
            api = self._get_api_instance(user_id)
            result = api.search(symbol=symbol, exchange=exchange)
            APILogger.log_response("FyersService", "search", result, user_id)
            return result
        except Exception as e:
            APILogger.log_error("FyersService", "search", e, request_data, user_id)
            raise
    
    # Legacy compatibility methods (for backward compatibility)
    def get_funds(self, user_id: int):
        """Legacy method - redirects to standardized format."""
        return self.funds(user_id)
    
    def get_holdings(self, user_id: int):
        """Legacy method - redirects to standardized format."""
        return self.holdings(user_id)
    
    def get_positions(self, user_id: int):
        """Legacy method - redirects to standardized format."""
        return self.positions(user_id)
    
    def get_orderbook(self, user_id: int):
        """Legacy method - redirects to standardized format."""
        return self.orderbook(user_id)
    
    def get_tradebook(self, user_id: int):
        """Legacy method - redirects to standardized format."""
        return self.tradebook(user_id)
    
    def get_quotes(self, user_id: int, symbols: str):
        """Legacy method - redirects to standardized format."""
        # Handle multiple symbols (legacy format)
        if ',' in symbols:
            symbol = symbols.split(',')[0]  # Take first symbol for now
        else:
            symbol = symbols
        
        return self.quotes(user_id, symbol)
    
    def get_history(self, user_id: int, symbol: str, resolution: str, 
                    range_from: str, range_to: str):
        """Legacy method - redirects to standardized format."""
        # Map legacy parameters to standard format
        exchange = "NSE"  # Default
        if ":" in symbol:
            exchange = symbol.split(":")[0]
        
        return self.history(user_id, symbol, exchange, resolution, range_from, range_to)
    
    def get_profile(self, user_id: int):
        """Legacy method - redirects to login."""
        return self.login(user_id)


# Global service instance
_fyers_service = None

def get_fyers_service() -> FyersService:
    """Get the global Fyers service instance."""
    global _fyers_service
    if _fyers_service is None:
        _fyers_service = FyersService()
    return _fyers_service
