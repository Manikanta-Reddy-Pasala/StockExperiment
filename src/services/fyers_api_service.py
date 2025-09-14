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
try:
    from ..broker_service import get_broker_service
except ImportError:
    from src.services.broker_service import get_broker_service
try:
    from ..user_settings_service import get_user_settings_service
except ImportError:
    from src.services.user_settings_service import get_user_settings_service

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
    
    def get_tradebook(self, user_id: int, search: str = None, sort_by: str = None,
                     sort_order: str = 'desc') -> Dict[str, Any]:
        """Get user tradebook with search and sort functionality."""
        try:
            fyers = self._get_fyers_client(user_id)
            response = fyers.tradebook()
            
            if response.get('s') != 'ok':
                return {
                    'success': False,
                    'error': response.get('message', 'Failed to fetch tradebook'),
                    'data': [],
                    'total': 0,
                    'last_updated': datetime.now().isoformat()
                }
            
            trades = response.get('tradeBook', [])
            
            # Process and enrich trades data
            processed_trades = []
            for trade in trades:
                processed_trade = {
                    'id': trade.get('id', ''),
                    'order_id': trade.get('orderNumber', ''),
                    'symbol': trade.get('symbol', ''),
                    'symbol_name': self._extract_symbol_name(trade.get('symbol', '')),
                    'side': 'BUY' if trade.get('side', 1) == 1 else 'SELL',
                    'product_type': trade.get('productType', ''),
                    'quantity': trade.get('qty', 0),
                    'price': trade.get('tradePrice', 0),
                    'trade_value': trade.get('tradeValue', 0),
                    'trade_date_time': trade.get('tradeDateTime', ''),
                    'exchange_trade_no': trade.get('exchTrdId', ''),
                    'pnl': trade.get('pnl', 0),
                    'charges': trade.get('charges', 0),
                    'net_amount': trade.get('netAmount', 0),
                    'order_tag': trade.get('orderTag', ''),
                    'last_updated': datetime.now().isoformat()
                }
                processed_trades.append(processed_trade)
            
            # Apply search filter
            if search:
                search_lower = search.lower()
                filtered_trades = []
                for trade in processed_trades:
                    if (search_lower in trade['symbol'].lower() or 
                        search_lower in trade['symbol_name'].lower() or
                        search_lower in trade['id'].lower() or
                        search_lower in trade['order_id'].lower()):
                        filtered_trades.append(trade)
                processed_trades = filtered_trades
            
            # Apply sorting
            if sort_by:
                reverse = sort_order.lower() == 'desc'
                if sort_by in ['symbol', 'symbol_name', 'side', 'product_type']:
                    processed_trades.sort(key=lambda x: x.get(sort_by, '').lower(), reverse=reverse)
                elif sort_by in ['quantity', 'price', 'trade_value', 'pnl', 'charges', 'net_amount']:
                    processed_trades.sort(key=lambda x: float(x.get(sort_by, 0)), reverse=reverse)
                elif sort_by == 'trade_date_time':
                    processed_trades.sort(key=lambda x: x.get(sort_by, ''), reverse=reverse)
                else:
                    # Default sort by trade time
                    processed_trades.sort(key=lambda x: x.get('trade_date_time', ''), reverse=True)
            else:
                # Default sort by trade time descending
                processed_trades.sort(key=lambda x: x.get('trade_date_time', ''), reverse=True)
            
            return {
                'success': True,
                'data': processed_trades,
                'total': len(processed_trades),
                'search': search,
                'sort_by': sort_by,
                'sort_order': sort_order,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting tradebook for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    # Order Operations
    def place_order(self, user_id: int, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Place a single order."""
        try:
            fyers = self._get_fyers_client(user_id)
            response = fyers.place_order(data=order_data)
            
            if response.get('s') == 'ok':
                return {
                    'success': True,
                    'order_id': response.get('id', ''),
                    'message': 'Order placed successfully',
                    'data': response
                }
            else:
                return {
                    'success': False,
                    'error': response.get('message', 'Failed to place order'),
                    'data': response
                }
                
        except Exception as e:
            logger.error(f"Error placing order for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def place_basket_orders(self, user_id: int, orders_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Place multiple orders as a basket (max 10)."""
        try:
            if len(orders_data) > 10:
                return {
                    'success': False,
                    'error': 'Maximum 10 orders allowed in basket'
                }
            
            fyers = self._get_fyers_client(user_id)
            response = fyers.place_basket_orders(data=orders_data)
            
            return {
                'success': response.get('s') == 'ok',
                'message': response.get('message', ''),
                'data': response
            }
                
        except Exception as e:
            logger.error(f"Error placing basket orders for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def modify_order(self, user_id: int, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an existing order."""
        try:
            fyers = self._get_fyers_client(user_id)
            response = fyers.modify_order(data=order_data)
            
            return {
                'success': response.get('s') == 'ok',
                'message': response.get('message', 'Order modification failed'),
                'data': response
            }
                
        except Exception as e:
            logger.error(f"Error modifying order for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def cancel_order(self, user_id: int, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        try:
            fyers = self._get_fyers_client(user_id)
            response = fyers.cancel_order(data={"id": order_id})
            
            return {
                'success': response.get('s') == 'ok',
                'message': response.get('message', 'Order cancellation failed'),
                'data': response
            }
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id} for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # Market Data and Research
    def get_quotes(self, user_id: int, symbols: Union[str, List[str]]) -> Dict[str, Any]:
        """Get real-time quotes for symbols."""
        try:
            fyers = self._get_fyers_client(user_id)
            
            if isinstance(symbols, list):
                symbols_str = ','.join(symbols)
            else:
                symbols_str = symbols
            
            response = fyers.quotes(data={"symbols": symbols_str})
            
            if response.get('s') == 'ok':
                return {
                    'success': True,
                    'data': response.get('d', {}),
                    'last_updated': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': response.get('message', 'Failed to fetch quotes'),
                    'data': {}
                }
                
        except Exception as e:
            logger.error(f"Error getting quotes for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': {}
            }
    
    def get_market_depth(self, user_id: int, symbols: Union[str, List[str]], 
                        ohlcv_flag: int = 1) -> Dict[str, Any]:
        """Get market depth for symbols."""
        try:
            fyers = self._get_fyers_client(user_id)
            
            if isinstance(symbols, list):
                symbols_list = symbols
            else:
                symbols_list = [symbols]
            
            data = {
                "symbol": symbols_list,
                "ohlcv_flag": ohlcv_flag
            }
            
            response = fyers.depth(data=data)
            
            if response.get('s') == 'ok':
                return {
                    'success': True,
                    'data': response.get('d', {}),
                    'last_updated': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': response.get('message', 'Failed to fetch market depth'),
                    'data': {}
                }
                
        except Exception as e:
            logger.error(f"Error getting market depth for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': {}
            }
    
    def get_historical_data(self, user_id: int, symbol: str, resolution: str = "D", 
                           range_from: str = None, range_to: str = None, 
                           date_format: int = 1, cont_flag: int = 1) -> Dict[str, Any]:
        """Get historical data for a symbol."""
        try:
            fyers = self._get_fyers_client(user_id)
            
            # Set default dates if not provided
            if not range_to:
                range_to = datetime.now().strftime('%Y-%m-%d')
            if not range_from:
                from_date = datetime.now() - timedelta(days=30)
                range_from = from_date.strftime('%Y-%m-%d')
            
            data = {
                "symbol": symbol,
                "resolution": resolution,
                "date_format": date_format,
                "range_from": range_from,
                "range_to": range_to,
                "cont_flag": cont_flag
            }
            
            response = fyers.history(data=data)
            
            if response.get('s') == 'ok':
                return {
                    'success': True,
                    'data': {
                        'candles': response.get('candles', []),
                        'symbol': symbol,
                        'resolution': resolution,
                        'from': range_from,
                        'to': range_to
                    },
                    'last_updated': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': response.get('message', 'Failed to fetch historical data'),
                    'data': {}
                }
                
        except Exception as e:
            logger.error(f"Error getting historical data for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': {}
            }
    
    # Stock Screening and Search
    def search_symbols(self, user_id: int, query: str, limit: int = 50) -> Dict[str, Any]:
        """Search for symbols by name or code."""
        try:
            fyers = self._get_fyers_client(user_id)
            
            # FYERS search API endpoint
            response = fyers.search_scrips(data={"symbol": query, "exchange": "ALL"})
            
            if response.get('s') == 'ok':
                results = response.get('d', [])
                
                # Process and enrich search results
                processed_results = []
                for result in results[:limit]:
                    processed_result = {
                        'symbol': result.get('symbol', ''),
                        'symbol_name': result.get('name', ''),
                        'exchange': result.get('exch', ''),
                        'segment': result.get('segment', ''),
                        'description': result.get('description', ''),
                        'instrument_type': result.get('instrument_type', ''),
                        'lot_size': result.get('lot_size', 1),
                        'tick_size': result.get('tick_size', 0.01),
                        'isin': result.get('isin', ''),
                        'last_updated': datetime.now().isoformat()
                    }
                    processed_results.append(processed_result)
                
                return {
                    'success': True,
                    'data': processed_results,
                    'total': len(processed_results),
                    'query': query,
                    'last_updated': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': response.get('message', 'Failed to search symbols'),
                    'data': [],
                    'total': 0
                }
                
        except Exception as e:
            logger.error(f"Error searching symbols for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0
            }
    
    def get_watchlist_suggestions(self, user_id: int, search: str = None, 
                                 sector: str = None, sort_by: str = 'volume',
                                 sort_order: str = 'desc', limit: int = 50) -> Dict[str, Any]:
        """Get suggested stocks for watchlist with search and filtering."""
        try:
            # TODO: Get popular stocks from broker API or database
            # For now, return empty suggestions as we don't want hardcoded stocks
            popular_symbols = []
            
            # Get quotes for popular symbols
            quotes_response = self.get_quotes(user_id, popular_symbols)
            
            if not quotes_response.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to fetch market data for suggestions',
                    'data': [],
                    'total': 0
                }
            
            quotes_data = quotes_response.get('data', {})
            
            # Process suggestions
            suggestions = []
            for symbol in popular_symbols:
                if symbol in quotes_data and quotes_data[symbol].get('v'):
                    quote = quotes_data[symbol]['v']
                    suggestion = {
                        'symbol': symbol,
                        'symbol_name': self._extract_symbol_name(symbol),
                        'price': quote.get('lp', 0),
                        'change': quote.get('ch', 0),
                        'change_percent': quote.get('chp', 0),
                        'volume': quote.get('volume', 0),
                        'high': quote.get('h', 0),
                        'low': quote.get('l', 0),
                        'open': quote.get('open_price', 0),
                        'prev_close': quote.get('prev_close_price', 0),
                        'sector': self._get_sector_for_symbol(symbol),
                        'market_cap_category': self._get_market_cap_category(quote.get('lp', 0)),
                        'last_updated': datetime.now().isoformat()
                    }
                    suggestions.append(suggestion)
            
            # Apply search filter
            if search:
                search_lower = search.lower()
                filtered_suggestions = []
                for suggestion in suggestions:
                    if (search_lower in suggestion['symbol'].lower() or 
                        search_lower in suggestion['symbol_name'].lower()):
                        filtered_suggestions.append(suggestion)
                suggestions = filtered_suggestions
            
            # Apply sector filter
            if sector:
                filtered_suggestions = []
                for suggestion in suggestions:
                    if suggestion['sector'].lower() == sector.lower():
                        filtered_suggestions.append(suggestion)
                suggestions = filtered_suggestions
            
            # Apply sorting
            if sort_by:
                reverse = sort_order.lower() == 'desc'
                if sort_by in ['symbol', 'symbol_name', 'sector']:
                    suggestions.sort(key=lambda x: x.get(sort_by, '').lower(), reverse=reverse)
                elif sort_by in ['price', 'change', 'change_percent', 'volume']:
                    suggestions.sort(key=lambda x: float(x.get(sort_by, 0)), reverse=reverse)
                else:
                    # Default sort by volume
                    suggestions.sort(key=lambda x: float(x.get('volume', 0)), reverse=True)
            else:
                # Default sort by volume descending
                suggestions.sort(key=lambda x: float(x.get('volume', 0)), reverse=True)
            
            return {
                'success': True,
                'data': suggestions[:limit],
                'total': len(suggestions),
                'search': search,
                'sector': sector,
                'sort_by': sort_by,
                'sort_order': sort_order,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting watchlist suggestions for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0
            }
    
    # Utility Methods
    def _extract_symbol_name(self, symbol: str) -> str:
        """Extract readable name from symbol."""
        if ':' in symbol and '-' in symbol:
            # Extract from NSE:RELIANCE-EQ format
            parts = symbol.split(':')
            if len(parts) > 1:
                name_part = parts[1].split('-')[0]
                return name_part.replace('_', ' ').title()
        return symbol
    
    def _get_order_type_name(self, order_type: int) -> str:
        """Convert order type number to name."""
        type_mapping = {
            1: 'LIMIT',
            2: 'MARKET',
            3: 'STOP_MARKET',
            4: 'STOP_LIMIT'
        }
        return type_mapping.get(order_type, 'UNKNOWN')
    
    def _get_order_status_name(self, status: int) -> str:
        """Convert order status number to name."""
        status_mapping = {
            1: 'PENDING',
            2: 'OPEN',
            3: 'CANCELLED',
            4: 'TRADED',
            5: 'REJECTED',
            6: 'EXPIRED'
        }
        return status_mapping.get(status, 'UNKNOWN')
    
    def _get_sector_for_symbol(self, symbol: str) -> str:
        """Get sector classification for symbol (simplified mapping)."""
        # TODO: Get sector information from broker API or database
        # For now, return generic sector
        sector_mapping = {}
        
        for key, sector in sector_mapping.items():
            if key in symbol.upper():
                return sector
        return 'Others'
    
    def _get_market_cap_category(self, price: float) -> str:
        """Categorize by market cap (simplified)."""
        # This is a simplified categorization - in reality you'd use actual market cap
        if price > 2000:
            return 'Large Cap'
        elif price > 500:
            return 'Mid Cap'
        else:
            return 'Small Cap'
    
    # Reports and Analytics
    def generate_portfolio_summary_report(self, user_id: int) -> Dict[str, Any]:
        """Generate comprehensive portfolio summary report."""
        try:
            # Get all required data
            profile = self.get_user_profile(user_id)
            funds = self.get_funds(user_id)
            holdings = self.get_holdings(user_id)
            positions = self.get_positions(user_id)
            
            # Calculate summary metrics
            total_portfolio_value = 0
            total_pnl = 0
            total_day_change = 0
            holdings_count = 0
            positions_count = 0
            
            if holdings.get('success'):
                holdings_data = holdings.get('data', [])
                holdings_count = len(holdings_data)
                for holding in holdings_data:
                    total_portfolio_value += holding.get('market_value', 0)
                    total_pnl += holding.get('pnl', 0)
            
            if positions.get('success'):
                positions_data = positions.get('data', [])
                positions_count = len(positions_data)
                for position in positions_data:
                    total_pnl += position.get('pnl', 0)
                    total_day_change += position.get('day_change', 0)
            
            # Get available cash
            available_cash = 0
            if funds.get('success'):
                fund_data = funds.get('data', [])
                for fund in fund_data:
                    if fund.get('title') == 'Available Cash':
                        available_cash = fund.get('equityAmount', 0)
                        break
            
            report = {
                'success': True,
                'generated_at': datetime.now().isoformat(),
                'user_profile': profile.get('data', {}) if profile.get('success') else {},
                'summary': {
                    'total_portfolio_value': round(total_portfolio_value, 2),
                    'total_pnl': round(total_pnl, 2),
                    'total_pnl_percent': round((total_pnl / total_portfolio_value * 100) if total_portfolio_value > 0 else 0, 2),
                    'total_day_change': round(total_day_change, 2),
                    'available_cash': round(available_cash, 2),
                    'holdings_count': holdings_count,
                    'positions_count': positions_count
                },
                'holdings': holdings.get('data', []) if holdings.get('success') else [],
                'positions': positions.get('data', []) if positions.get('success') else [],
                'funds': funds.get('data', []) if funds.get('success') else []
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating portfolio report for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def generate_trading_summary_report(self, user_id: int, from_date: str = None, 
                                       to_date: str = None) -> Dict[str, Any]:
        """Generate trading summary report for a date range."""
        try:
            # Set default date range if not provided
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Get trading data
            orderbook = self.get_orderbook(user_id)
            tradebook = self.get_tradebook(user_id)
            
            # Analyze orders
            total_orders = 0
            orders_by_status = {}
            orders_by_type = {}
            
            if orderbook.get('success'):
                orders_data = orderbook.get('data', [])
                total_orders = len(orders_data)
                
                for order in orders_data:
                    status = order.get('status', 'UNKNOWN')
                    order_type = order.get('type', 'UNKNOWN')
                    
                    orders_by_status[status] = orders_by_status.get(status, 0) + 1
                    orders_by_type[order_type] = orders_by_type.get(order_type, 0) + 1
            
            # Analyze trades
            total_trades = 0
            total_trade_value = 0
            total_pnl = 0
            trades_by_side = {'BUY': 0, 'SELL': 0}
            
            if tradebook.get('success'):
                trades_data = tradebook.get('data', [])
                total_trades = len(trades_data)
                
                for trade in trades_data:
                    total_trade_value += trade.get('trade_value', 0)
                    total_pnl += trade.get('pnl', 0)
                    side = trade.get('side', 'BUY')
                    trades_by_side[side] = trades_by_side.get(side, 0) + 1
            
            report = {
                'success': True,
                'generated_at': datetime.now().isoformat(),
                'period': {
                    'from_date': from_date,
                    'to_date': to_date
                },
                'orders_summary': {
                    'total_orders': total_orders,
                    'orders_by_status': orders_by_status,
                    'orders_by_type': orders_by_type
                },
                'trades_summary': {
                    'total_trades': total_trades,
                    'total_trade_value': round(total_trade_value, 2),
                    'total_pnl': round(total_pnl, 2),
                    'trades_by_side': trades_by_side,
                    'avg_trade_value': round(total_trade_value / total_trades if total_trades > 0 else 0, 2)
                },
                'detailed_orders': orderbook.get('data', []) if orderbook.get('success') else [],
                'detailed_trades': tradebook.get('data', []) if tradebook.get('success') else []
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating trading summary for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }


# Global service instance
_fyers_api_service = None

def get_fyers_api_service() -> FyersAPIService:
    """Get the global FYERS API service instance."""
    global _fyers_api_service
    if _fyers_api_service is None:
        _fyers_api_service = FyersAPIService()
    return _fyers_api_service
