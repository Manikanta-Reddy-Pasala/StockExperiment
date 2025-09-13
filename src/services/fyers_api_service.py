"""
Enhanced FYERS Service with Complete API Integration

This service provides comprehensive FYERS API integration with all features
including search, sort, and filtering capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
from fyers_apiv3 import fyersModel
from ..broker_service import get_broker_service
from ..user_settings_service import get_user_settings_service

logger = logging.getLogger(__name__)


class FyersAPIService:
    """
    Comprehensive FYERS API service with full feature implementation.
    """
    
    def __init__(self):
        self.broker_service = get_broker_service()
        self.user_settings_service = get_user_settings_service()
        
    def _get_fyers_client(self, user_id: int):
        """Get authenticated FYERS client for the user."""
        try:
            # Get broker configuration for the user
            config = self.broker_service.get_broker_config('fyers', user_id)
            if not config or not config.get('access_token'):
                raise ValueError("FYERS access token not found for user")
            
            client_id = config.get('client_id')
            access_token = config.get('access_token')
            
            if not client_id or not access_token:
                raise ValueError("FYERS configuration incomplete")
            
            # Create FYERS client
            fyers = fyersModel.FyersModel(
                client_id=client_id,
                token=access_token,
                is_async=False,
                log_path=""
            )
            
            return fyers
            
        except Exception as e:
            logger.error(f"Failed to create FYERS client for user {user_id}: {str(e)}")
            raise
    
    # User Profile and Account Information
    def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Get user profile information."""
        try:
            fyers = self._get_fyers_client(user_id)
            response = fyers.get_profile()
            
            if response.get('s') == 'ok':
                return {
                    'success': True,
                    'data': response.get('data', {}),
                    'last_updated': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': response.get('message', 'Failed to fetch profile'),
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting user profile for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
    
    def get_funds(self, user_id: int) -> Dict[str, Any]:
        """Get user funds information."""
        try:
            fyers = self._get_fyers_client(user_id)
            response = fyers.funds()
            
            if response.get('s') == 'ok':
                return {
                    'success': True,
                    'data': response.get('fund_limit', []),
                    'last_updated': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': response.get('message', 'Failed to fetch funds'),
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting funds for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
    
    def get_holdings(self, user_id: int, search: str = None, sort_by: str = None, 
                    sort_order: str = 'desc') -> Dict[str, Any]:
        """Get user holdings with search and sort functionality."""
        try:
            fyers = self._get_fyers_client(user_id)
            response = fyers.holdings()
            
            if response.get('s') != 'ok':
                return {
                    'success': False,
                    'error': response.get('message', 'Failed to fetch holdings'),
                    'data': [],
                    'total': 0,
                    'last_updated': datetime.now().isoformat()
                }
            
            holdings = response.get('holdings', [])
            
            # Process and enrich holdings data
            processed_holdings = []
            for holding in holdings:
                processed_holding = {
                    'symbol': holding.get('symbol', ''),
                    'symbol_name': self._extract_symbol_name(holding.get('symbol', '')),
                    'isin': holding.get('isin', ''),
                    'quantity': holding.get('qty', 0),
                    'avg_price': holding.get('costPrice', 0),
                    'current_price': holding.get('ltp', 0),
                    'market_value': holding.get('marketVal', 0),
                    'pnl': holding.get('pl', 0),
                    'pnl_percent': holding.get('plPercent', 0),
                    'product_type': holding.get('product', ''),
                    'exchange': holding.get('ex', ''),
                    'segment': holding.get('segment', ''),
                    'side': holding.get('side', 1),
                    'remaining_quantity': holding.get('remainingQty', 0),
                    'cross_currency': holding.get('crossCurrency', ''),
                    'rbi_reference_rate': holding.get('rbiReferenceRate', 0),
                    'holding_type': holding.get('holdingType', ''),
                    'last_updated': datetime.now().isoformat()
                }
                processed_holdings.append(processed_holding)
            
            # Apply search filter
            if search:
                search_lower = search.lower()
                filtered_holdings = []
                for holding in processed_holdings:
                    if (search_lower in holding['symbol'].lower() or 
                        search_lower in holding['symbol_name'].lower() or
                        search_lower in holding['isin'].lower()):
                        filtered_holdings.append(holding)
                processed_holdings = filtered_holdings
            
            # Apply sorting
            if sort_by:
                reverse = sort_order.lower() == 'desc'
                if sort_by in ['symbol', 'symbol_name', 'isin', 'product_type', 'exchange']:
                    processed_holdings.sort(key=lambda x: x.get(sort_by, '').lower(), reverse=reverse)
                elif sort_by in ['quantity', 'avg_price', 'current_price', 'market_value', 'pnl', 'pnl_percent']:
                    processed_holdings.sort(key=lambda x: float(x.get(sort_by, 0)), reverse=reverse)
                else:
                    # Default sort by market value
                    processed_holdings.sort(key=lambda x: float(x.get('market_value', 0)), reverse=True)
            else:
                # Default sort by market value descending
                processed_holdings.sort(key=lambda x: float(x.get('market_value', 0)), reverse=True)
            
            return {
                'success': True,
                'data': processed_holdings,
                'total': len(processed_holdings),
                'search': search,
                'sort_by': sort_by,
                'sort_order': sort_order,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting holdings for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    def get_positions(self, user_id: int, search: str = None, sort_by: str = None, 
                     sort_order: str = 'desc') -> Dict[str, Any]:
        """Get user positions with search and sort functionality."""
        try:
            fyers = self._get_fyers_client(user_id)
            response = fyers.positions()
            
            if response.get('s') != 'ok':
                return {
                    'success': False,
                    'error': response.get('message', 'Failed to fetch positions'),
                    'data': [],
                    'total': 0,
                    'last_updated': datetime.now().isoformat()
                }
            
            positions = response.get('netPositions', [])
            
            # Process and enrich positions data
            processed_positions = []
            for position in positions:
                processed_position = {
                    'symbol': position.get('symbol', ''),
                    'symbol_name': self._extract_symbol_name(position.get('symbol', '')),
                    'side': position.get('side', 1),
                    'product_type': position.get('productType', ''),
                    'quantity': position.get('qty', 0),
                    'avg_price': position.get('avgPrice', 0),
                    'market_price': position.get('ltp', 0),
                    'pnl': position.get('unrealizedProfit', 0),
                    'pnl_percent': position.get('plPercent', 0),
                    'day_change': position.get('dayChange', 0),
                    'day_change_percent': position.get('dayChangePercent', 0),
                    'buy_qty': position.get('buyQty', 0),
                    'buy_avg': position.get('buyAvg', 0),
                    'buy_value': position.get('buyVal', 0),
                    'sell_qty': position.get('sellQty', 0),
                    'sell_avg': position.get('sellAvg', 0),
                    'sell_value': position.get('sellVal', 0),
                    'net_qty': position.get('netQty', 0),
                    'realized_profit': position.get('realized_profit', 0),
                    'cross_currency': position.get('crossCurrency', ''),
                    'rbi_reference_rate': position.get('rbiReferenceRate', 0),
                    'last_updated': datetime.now().isoformat()
                }
                processed_positions.append(processed_position)
            
            # Apply search filter
            if search:
                search_lower = search.lower()
                filtered_positions = []
                for position in processed_positions:
                    if (search_lower in position['symbol'].lower() or 
                        search_lower in position['symbol_name'].lower()):
                        filtered_positions.append(position)
                processed_positions = filtered_positions
            
            # Apply sorting
            if sort_by:
                reverse = sort_order.lower() == 'desc'
                if sort_by in ['symbol', 'symbol_name', 'product_type']:
                    processed_positions.sort(key=lambda x: x.get(sort_by, '').lower(), reverse=reverse)
                elif sort_by in ['quantity', 'avg_price', 'market_price', 'pnl', 'pnl_percent', 'day_change']:
                    processed_positions.sort(key=lambda x: float(x.get(sort_by, 0)), reverse=reverse)
                else:
                    # Default sort by PnL
                    processed_positions.sort(key=lambda x: float(x.get('pnl', 0)), reverse=True)
            else:
                # Default sort by PnL descending
                processed_positions.sort(key=lambda x: float(x.get('pnl', 0)), reverse=True)
            
            return {
                'success': True,
                'data': processed_positions,
                'total': len(processed_positions),
                'search': search,
                'sort_by': sort_by,
                'sort_order': sort_order,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting positions for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    # Orders and Trading
    def get_orderbook(self, user_id: int, search: str = None, status_filter: str = None,
                     sort_by: str = None, sort_order: str = 'desc') -> Dict[str, Any]:
        """Get user orderbook with search, filter, and sort functionality."""
        try:
            fyers = self._get_fyers_client(user_id)
            response = fyers.orderbook()
            
            if response.get('s') != 'ok':
                return {
                    'success': False,
                    'error': response.get('message', 'Failed to fetch orderbook'),
                    'data': [],
                    'total': 0,
                    'last_updated': datetime.now().isoformat()
                }
            
            orders = response.get('orderBook', [])
            
            # Process and enrich orders data
            processed_orders = []
            for order in orders:
                processed_order = {
                    'id': order.get('id', ''),
                    'symbol': order.get('symbol', ''),
                    'symbol_name': self._extract_symbol_name(order.get('symbol', '')),
                    'side': 'BUY' if order.get('side', 1) == 1 else 'SELL',
                    'type': self._get_order_type_name(order.get('type', 1)),
                    'product_type': order.get('productType', ''),
                    'status': self._get_order_status_name(order.get('status', 1)),
                    'quantity': order.get('qty', 0),
                    'remaining_quantity': order.get('remainingQty', 0),
                    'filled_quantity': order.get('filledQty', 0),
                    'limit_price': order.get('limitPrice', 0),
                    'stop_price': order.get('stopPrice', 0),
                    'disclosed_quantity': order.get('disclosedQty', 0),
                    'validity': order.get('validity', ''),
                    'order_date_time': order.get('orderDateTime', ''),
                    'exchange_order_no': order.get('exchOrdId', ''),
                    'dq_qty_rem': order.get('dqQtyRem', 0),
                    'order_num_status': order.get('orderNumStatus', ''),
                    'offline_order': order.get('offlineOrder', False),
                    'order_tag': order.get('orderTag', ''),
                    'last_updated': datetime.now().isoformat()
                }
                processed_orders.append(processed_order)
            
            # Apply search filter
            if search:
                search_lower = search.lower()
                filtered_orders = []
                for order in processed_orders:
                    if (search_lower in order['symbol'].lower() or 
                        search_lower in order['symbol_name'].lower() or
                        search_lower in order['id'].lower()):
                        filtered_orders.append(order)
                processed_orders = filtered_orders
            
            # Apply status filter
            if status_filter:
                filtered_orders = []
                for order in processed_orders:
                    if order['status'].lower() == status_filter.lower():
                        filtered_orders.append(order)
                processed_orders = filtered_orders
            
            # Apply sorting
            if sort_by:
                reverse = sort_order.lower() == 'desc'
                if sort_by in ['symbol', 'symbol_name', 'side', 'type', 'status', 'validity']:
                    processed_orders.sort(key=lambda x: x.get(sort_by, '').lower(), reverse=reverse)
                elif sort_by in ['quantity', 'limit_price', 'stop_price']:
                    processed_orders.sort(key=lambda x: float(x.get(sort_by, 0)), reverse=reverse)
                elif sort_by == 'order_date_time':
                    processed_orders.sort(key=lambda x: x.get(sort_by, ''), reverse=reverse)
                else:
                    # Default sort by order time
                    processed_orders.sort(key=lambda x: x.get('order_date_time', ''), reverse=True)
            else:
                # Default sort by order time descending
                processed_orders.sort(key=lambda x: x.get('order_date_time', ''), reverse=True)
            
            return {
                'success': True,
                'data': processed_orders,
                'total': len(processed_orders),
                'search': search,
                'status_filter': status_filter,
                'sort_by': sort_by,
                'sort_order': sort_order,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting orderbook for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0,
                'last_updated': datetime.now().isoformat()
            }
