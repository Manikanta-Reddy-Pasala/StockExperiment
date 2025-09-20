"""
Simulator Broker Service - Dedicated service for simulated trading operations
"""
import os
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

try:
    from src.models.database import get_database_manager
    from src.models.models import BrokerConfiguration, Order, Trade
except ImportError:
    from models.database import get_database_manager
    from models.models import BrokerConfiguration, Order, Trade


class SimulatorService:
    """Dedicated service for simulated trading operations."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.broker_name = 'simulator'

    def test_connection(self, user_id: int):
        """Test simulator connection (always successful)."""
        logger.info(f"Testing simulator connection for user {user_id}")
        
        # Simulator is always connected
        result = {
            'success': True,
            'message': 'Simulator connection successful',
            'response_time': f"{random.randint(10, 50)}ms",
            'simulator_data': {
                'mode': 'simulation',
                'virtual_balance': 100000.0,
                'trading_enabled': True
            },
            'status_code': 200
        }

        # Update database status
        with self.db_manager.get_session() as session:
            db_config = session.query(BrokerConfiguration).filter_by(broker_name=self.broker_name, user_id=user_id).first()
            if db_config:
                db_config.is_connected = True
                db_config.connection_status = 'connected'
                db_config.last_connection_test = datetime.utcnow()
                db_config.error_message = None
                session.commit()

        return result

    def get_funds(self, user_id: int):
        """Get simulated user funds."""
        logger.info(f"Fetching simulator funds for user {user_id}")
        
        # Generate simulated fund data
        funds = {
            'data': {
                'available': {
                    'cash': random.uniform(50000, 200000),
                    'opening_balance': 100000.0,
                    'live_balance': random.uniform(80000, 150000),
                    'collateral': random.uniform(20000, 50000),
                    'intraday_payin': 0.0,
                    'adhoc_margin': 0.0,
                    'notional_cash': 0.0
                },
                'utilised': {
                    'debits': random.uniform(1000, 10000),
                    'exposure': random.uniform(5000, 20000),
                    'm2m_realised': random.uniform(-1000, 2000),
                    'm2m_unrealised': random.uniform(-500, 1000),
                    'value_at_risk': random.uniform(1000, 5000)
                }
            }
        }
        
        return funds

    def get_holdings(self, user_id: int):
        """Get simulated user holdings."""
        logger.info(f"Fetching simulator holdings for user {user_id}")
        
        # Generate simulated holdings data
        holdings = {
            'data': [
                {
                    'tradingsymbol': 'RELIANCE',
                    'exchange': 'NSE',
                    'isin': 'INE002A01018',
                    'product': 'CNC',
                    'collateral_quantity': 0,
                    'collateral_type': '',
                    't1_quantity': 0,
                    'average_price': 2450.50,
                    'last_price': 2455.75,
                    'pnl': 125.50,
                    'day_change': 5.25,
                    'day_change_percentage': 0.21,
                    'quantity': 10,
                    'realised_quantity': 0,
                    'authorised_quantity': 0,
                    'discrepancy': False
                },
                {
                    'tradingsymbol': 'TCS',
                    'exchange': 'NSE',
                    'isin': 'INE467B01029',
                    'product': 'CNC',
                    'collateral_quantity': 0,
                    'collateral_type': '',
                    't1_quantity': 0,
                    'average_price': 3450.25,
                    'last_price': 3465.80,
                    'pnl': 155.50,
                    'day_change': 15.55,
                    'day_change_percentage': 0.45,
                    'quantity': 5,
                    'realised_quantity': 0,
                    'authorised_quantity': 0,
                    'discrepancy': False
                }
            ]
        }
        
        return holdings

    def get_positions(self, user_id: int):
        """Get simulated user positions."""
        logger.info(f"Fetching simulator positions for user {user_id}")
        
        # Generate simulated positions data
        positions = {
            'data': {
                'day': [
                    {
                        'tradingsymbol': 'NIFTY 50',
                        'exchange': 'NSE',
                        'product': 'MIS',
                        'quantity': 50,
                        'overnight_quantity': 0,
                        'multiplier': 1,
                        'average_price': 19500.0,
                        'close_price': 19525.50,
                        'last_price': 19530.25,
                        'value': 976512.50,
                        'pnl': 1512.50,
                        'm2m': 1512.50,
                        'unrealised': 1512.50,
                        'realised': 0.0,
                        'buy_quantity': 50,
                        'buy_price': 19500.0,
                        'buy_value': 975000.0,
                        'buy_m2m': 0.0,
                        'sell_quantity': 0,
                        'sell_price': 0.0,
                        'sell_value': 0.0,
                        'sell_m2m': 0.0
                    }
                ],
                'net': [
                    {
                        'tradingsymbol': 'NIFTY 50',
                        'exchange': 'NSE',
                        'product': 'MIS',
                        'quantity': 50,
                        'overnight_quantity': 0,
                        'multiplier': 1,
                        'average_price': 19500.0,
                        'close_price': 19525.50,
                        'last_price': 19530.25,
                        'value': 976512.50,
                        'pnl': 1512.50,
                        'm2m': 1512.50,
                        'unrealised': 1512.50,
                        'realised': 0.0,
                        'buy_quantity': 50,
                        'buy_price': 19500.0,
                        'buy_value': 975000.0,
                        'buy_m2m': 0.0,
                        'sell_quantity': 0,
                        'sell_price': 0.0,
                        'sell_value': 0.0,
                        'sell_m2m': 0.0
                    }
                ]
            }
        }
        
        return positions

    def get_orderbook(self, user_id: int):
        """Get simulated user orderbook."""
        logger.info(f"Fetching simulator orderbook for user {user_id}")
        
        # Generate simulated orderbook data
        orders = {
            'data': [
                {
                    'order_id': f'ORD{random.randint(100000, 999999)}',
                    'parent_order_id': None,
                    'exchange_order_id': f'EX{random.randint(100000, 999999)}',
                    'placed_by': 'simulator',
                    'variety': 'regular',
                    'status': 'COMPLETE',
                    'tradingsymbol': 'RELIANCE',
                    'exchange': 'NSE',
                    'instrument_token': 738561,
                    'transaction_type': 'BUY',
                    'product': 'CNC',
                    'order_type': 'MARKET',
                    'quantity': 10,
                    'disclosed_quantity': 0,
                    'price': 0.0,
                    'trigger_price': 0.0,
                    'average_price': 2455.75,
                    'filled_quantity': 10,
                    'pending_quantity': 0,
                    'cancelled_quantity': 0,
                    'order_timestamp': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                    'exchange_timestamp': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                    'status_message': 'Order filled',
                    'tag': 'simulator'
                },
                {
                    'order_id': f'ORD{random.randint(100000, 999999)}',
                    'parent_order_id': None,
                    'exchange_order_id': f'EX{random.randint(100000, 999999)}',
                    'placed_by': 'simulator',
                    'variety': 'regular',
                    'status': 'OPEN',
                    'tradingsymbol': 'TCS',
                    'exchange': 'NSE',
                    'instrument_token': 2953217,
                    'transaction_type': 'SELL',
                    'product': 'CNC',
                    'order_type': 'LIMIT',
                    'quantity': 5,
                    'disclosed_quantity': 0,
                    'price': 3500.0,
                    'trigger_price': 0.0,
                    'average_price': 0.0,
                    'filled_quantity': 0,
                    'pending_quantity': 5,
                    'cancelled_quantity': 0,
                    'order_timestamp': (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
                    'exchange_timestamp': (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
                    'status_message': 'Order pending',
                    'tag': 'simulator'
                }
            ]
        }
        
        return orders

    def get_tradebook(self, user_id: int):
        """Get simulated user tradebook."""
        logger.info(f"Fetching simulator tradebook for user {user_id}")
        
        # Generate simulated tradebook data
        trades = {
            'data': [
                {
                    'trade_id': f'TRD{random.randint(100000, 999999)}',
                    'order_id': f'ORD{random.randint(100000, 999999)}',
                    'exchange_order_id': f'EX{random.randint(100000, 999999)}',
                    'tradingsymbol': 'RELIANCE',
                    'exchange': 'NSE',
                    'instrument_token': 738561,
                    'transaction_type': 'BUY',
                    'product': 'CNC',
                    'quantity': 10,
                    'price': 2455.75,
                    'order_price': 0.0,
                    'average_price': 2455.75,
                    'trade_timestamp': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                    'exchange_timestamp': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
                }
            ]
        }
        
        return trades

    def get_quotes(self, user_id: int, symbols: str):
        """Get simulated market quotes for symbols."""
        logger.info(f"Fetching simulator quotes for symbols: {symbols}")
        
        # Generate simulated quotes data
        symbol_list = symbols.split(',')
        quotes = {
            'data': {}
        }
        
        for symbol in symbol_list:
            symbol = symbol.strip()
            base_price = random.uniform(100, 5000)
            change = random.uniform(-50, 50)
            
            quotes['data'][symbol] = {
                'instrument_token': random.randint(100000, 999999),
                'last_price': base_price,
                'last_quantity': random.randint(1, 1000),
                'average_price': base_price + random.uniform(-10, 10),
                'volume': random.randint(1000, 100000),
                'buy_quantity': random.randint(100, 5000),
                'sell_quantity': random.randint(100, 5000),
                'ohlc': {
                    'open': base_price + random.uniform(-20, 20),
                    'high': base_price + random.uniform(0, 30),
                    'low': base_price - random.uniform(0, 30),
                    'close': base_price + change
                },
                'net_change': change,
                'oi': random.randint(1000, 10000),
                'oi_day_high': random.randint(1000, 10000),
                'oi_day_low': random.randint(1000, 10000),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_trade_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return quotes

    def get_profile(self, user_id: int):
        """Get simulated user profile."""
        logger.info(f"Fetching simulator profile for user {user_id}")
        
        # Generate simulated profile data
        profile = {
            'data': {
                'user_id': f'USER{user_id}',
                'user_name': f'Simulator User {user_id}',
                'email': f'simulator{user_id}@example.com',
                'user_type': 'individual',
                'broker': 'SIMULATOR',
                'exchanges': ['NSE', 'BSE', 'NFO', 'CDS', 'BFO'],
                'products': ['CNC', 'MIS', 'NRML'],
                'avatar_url': '',
                'meta': {
                    'demat_consent': 'consent'
                },
                'order_types': ['MARKET', 'LIMIT', 'SL', 'SL-M'],
                'avatar_url': ''
            }
        }
        
        return profile

    def get_broker_config(self, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get Simulator broker configuration from database."""
        with self.db_manager.get_session() as session:
            query = session.query(BrokerConfiguration).filter_by(broker_name=self.broker_name)
            if user_id:
                query = query.filter_by(user_id=user_id)
            else:
                query = query.filter_by(user_id=None)  # Global config
            
            config = query.first()
            if not config:
                # Return default simulator config
                return {
                    'id': None,
                    'user_id': user_id,
                    'broker_name': self.broker_name,
                    'client_id': 'simulator',
                    'access_token': 'simulator_token',
                    'refresh_token': None,
                    'api_key': 'simulator_key',
                    'api_secret': 'simulator_secret',
                    'redirect_url': 'http://localhost:5001/simulator/callback',
                    'app_type': '100',
                    'is_active': True,
                    'is_connected': True,
                    'is_token_expired': False,
                    'last_connection_test': datetime.utcnow(),
                    'connection_status': 'connected',
                    'error_message': None,
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }
            
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
                'is_connected': True,  # Simulator is always connected
                'is_token_expired': False,  # Simulator tokens don't expire
                'last_connection_test': config.last_connection_test,
                'connection_status': 'connected',
                'error_message': config.error_message,
                'created_at': config.created_at,
                'updated_at': config.updated_at
            }
    
    def save_broker_config(self, config_data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
        """Save Simulator broker configuration to database."""
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
            
            # Update fields with simulator defaults
            config.client_id = config_data.get('client_id', 'simulator')
            config.access_token = config_data.get('access_token', 'simulator_token')
            config.refresh_token = config_data.get('refresh_token', '')
            config.api_key = config_data.get('api_key', 'simulator_key')
            config.api_secret = config_data.get('api_secret', 'simulator_secret')
            config.redirect_url = config_data.get('redirect_url', 'http://localhost:5001/simulator/callback')
            config.app_type = config_data.get('app_type', '100')
            config.is_active = config_data.get('is_active', True)
            config.is_connected = True  # Simulator is always connected
            config.connection_status = 'connected'
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
        """Get Simulator broker statistics from database."""
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
            
            # For simulator, add some simulated stats if no real data
            if total_orders == 0:
                total_orders = random.randint(10, 100)
                successful_orders = int(total_orders * 0.85)
                pending_orders = random.randint(0, 5)
                failed_orders = total_orders - successful_orders - pending_orders
                last_order_time = (datetime.now() - timedelta(hours=random.randint(1, 24))).strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'total_orders': total_orders,
                'successful_orders': successful_orders,
                'pending_orders': pending_orders,
                'failed_orders': failed_orders,
                'last_order_time': last_order_time,
                'api_response_time': f"{random.randint(5, 25)}ms"
            }


def get_simulator_service() -> SimulatorService:
    """Get Simulator service instance."""
    return SimulatorService()
