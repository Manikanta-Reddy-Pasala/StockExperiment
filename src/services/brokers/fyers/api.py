"""
Fyers API Implementation

This module provides a comprehensive Fyers API implementation with standardized
response formats and error handling.
"""

import logging
import time
import hashlib
import hmac
import base64
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class FyersAPI:
    """
    Comprehensive Fyers API implementation with standardized response formats.
    """
    
    def __init__(self, api_key: str, api_secret: str, access_token: str):
        """Initialize Fyers API with credentials."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.base_url = "https://api-t1.fyers.in/api/v3"
        self.session = requests.Session()
        
        # Set default headers for all requests
        self.session.headers.update({
            'Authorization': f'{api_key}:{access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict[str, Any]:
        """Make authenticated request to Fyers API."""
        try:
            url = f"{self.base_url}/{endpoint}"
            
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=30
            )
            
            # Log request details
            logger.info(f"Fyers API {method} {endpoint} - Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Check Fyers API response status
                if result.get('s') == 'ok':
                    return {
                        'status': 'success',
                        'data': result.get('data', result),
                        'message': 'Request successful'
                    }
                else:
                    return {
                        'status': 'error',
                        'message': result.get('message', 'API request failed'),
                        'error_code': result.get('code', 'UNKNOWN_ERROR')
                    }
            else:
                return {
                    'status': 'error',
                    'message': f'HTTP {response.status_code}: {response.text}',
                    'error_code': f'HTTP_{response.status_code}'
                }
                
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'REQUEST_FAILED'
            }
    
    # Authentication and Session Management
    def login(self) -> Dict[str, Any]:
        """
        Validate login credentials and session.
        Check if access token is valid.
        """
        try:
            result = self._make_request('GET', 'profile')
            
            if result['status'] == 'success':
                return {
                    'status': 'success',
                    'message': 'Login successful',
                    'data': {
                        'login_status': True,
                        'profile': result.get('data', {})
                    }
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Login failed - Invalid credentials or expired token',
                    'data': {'login_status': False}
                }
                
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'data': {'login_status': False}
            }
    
    # Order Management
    def placeorder(self, symbol: str, quantity: str, action: str, 
                   product: str, pricetype: str, price: str = "0", 
                   trigger_price: str = "0", disclosed_quantity: str = "0",
                   validity: str = "DAY", tag: str = "") -> Dict[str, Any]:
        """
        Place a new order with standardized parameters.
        """
        try:
            # Map parameters to Fyers format
            side = 1 if action.upper() == 'BUY' else -1
            
            # Map price types
            order_type_map = {
                'MARKET': 2,
                'LIMIT': 1,
                'SL': 3,  # Stop Loss Market
                'SL-M': 3,  # Stop Loss Market  
                'SL-L': 4   # Stop Loss Limit
            }
            
            order_data = {
                "symbol": symbol,
                "qty": int(quantity),
                "type": order_type_map.get(pricetype.upper(), 2),
                "side": side,
                "productType": product.upper(),
                "validity": validity.upper(),
                "disclosedQty": int(disclosed_quantity) if disclosed_quantity else 0,
                "orderTag": tag
            }
            
            # Add price fields based on order type
            if pricetype.upper() != 'MARKET':
                order_data["limitPrice"] = float(price) if price and price != "0" else 0
                
            if pricetype.upper() in ['SL', 'SL-M', 'SL-L']:
                order_data["stopPrice"] = float(trigger_price) if trigger_price and trigger_price != "0" else 0
            
            result = self._make_request('POST', 'orders', data=order_data)
            
            if result['status'] == 'success':
                order_id = result.get('data', {}).get('id', '')
                return {
                    'status': 'success',
                    'message': 'Order placed successfully',
                    'data': {'orderid': order_id}
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Place order error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'PLACE_ORDER_FAILED'
            }
    
    def modifyorder(self, orderid: str, symbol: str = "", quantity: str = "",
                    price: str = "", trigger_price: str = "", 
                    disclosed_quantity: str = "", validity: str = "") -> Dict[str, Any]:
        """
        Modify an existing order.
        """
        try:
            modify_data = {"id": orderid}
            
            # Add fields to modify
            if quantity:
                modify_data["qty"] = int(quantity)
            if price and price != "0":
                modify_data["limitPrice"] = float(price)
            if trigger_price and trigger_price != "0":
                modify_data["stopPrice"] = float(trigger_price)
            if disclosed_quantity:
                modify_data["disclosedQty"] = int(disclosed_quantity)
            if validity:
                modify_data["validity"] = validity.upper()
            
            result = self._make_request('PUT', 'orders', data=modify_data)
            
            if result['status'] == 'success':
                return {
                    'status': 'success',
                    'message': 'Order modified successfully',
                    'data': {'orderid': orderid}
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Modify order error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'MODIFY_ORDER_FAILED'
            }
    
    def cancelorder(self, orderid: str) -> Dict[str, Any]:
        """
        Cancel an existing order.
        """
        try:
            cancel_data = {"id": orderid}
            result = self._make_request('DELETE', 'orders', data=cancel_data)
            
            if result['status'] == 'success':
                return {
                    'status': 'success',
                    'message': 'Order cancelled successfully',
                    'data': {'orderid': orderid}
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Cancel order error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'CANCEL_ORDER_FAILED'
            }
    
    # Smart Order Management  
    def placesmartorder(self, symbol: str, action: str, product: str,
                       quantity: str = "", position_size: str = "",
                       price: str = "0", trigger_price: str = "0",
                       pricetype: str = "MARKET", strategy: str = "",
                       tag: str = "") -> Dict[str, Any]:
        """
        Place smart order with position sizing.
        """
        try:
            # Calculate actual quantity if position_size is provided
            if position_size and not quantity:
                # Get current positions to calculate quantity based on position size
                positions_result = self.positions()
                if positions_result['status'] == 'success':
                    # Calculate quantity based on position size logic
                    # This is a simplified implementation
                    quantity = position_size
            
            # Use regular place order
            return self.placeorder(
                symbol=symbol,
                quantity=quantity or position_size,
                action=action,
                product=product,
                pricetype=pricetype,
                price=price,
                trigger_price=trigger_price,
                tag=f"{strategy}_{tag}" if strategy else tag
            )
            
        except Exception as e:
            logger.error(f"Place smart order error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'PLACE_SMART_ORDER_FAILED'
            }
    
    # Account Information
    def orderbook(self) -> Dict[str, Any]:
        """
        Get order book with standard format.
        """
        try:
            result = self._make_request('GET', 'orderbook')
            
            if result['status'] == 'success':
                orders = result.get('data', {}).get('orderBook', [])
                formatted_orders = []
                
                for order in orders:
                    formatted_order = {
                        'orderid': order.get('id', ''),
                        'symbol': order.get('symbol', ''),
                        'product': order.get('productType', ''),
                        'action': 'BUY' if order.get('side', 1) == 1 else 'SELL',
                        'quantity': str(order.get('qty', 0)),
                        'price': str(order.get('limitPrice', 0)),
                        'trigger_price': str(order.get('stopPrice', 0)),
                        'pricetype': self._get_order_type_name(order.get('type', 2)),
                        'status': self._get_order_status_name(order.get('status', 1)),
                        'timestamp': order.get('orderDateTime', ''),
                        'filled_quantity': str(order.get('filledQty', 0)),
                        'remaining_quantity': str(order.get('remainingQty', 0)),
                        'average_price': str(order.get('avgPrice', 0)),
                        'exchange': self._extract_exchange(order.get('symbol', '')),
                        'tag': order.get('orderTag', '')
                    }
                    formatted_orders.append(formatted_order)
                
                return {
                    'status': 'success',
                    'data': formatted_orders
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Order book error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'ORDERBOOK_FAILED'
            }
    
    def tradebook(self) -> Dict[str, Any]:
        """
        Get trade book with standard format.
        """
        try:
            result = self._make_request('GET', 'tradebook')
            
            if result['status'] == 'success':
                trades = result.get('data', {}).get('tradeBook', [])
                formatted_trades = []
                
                for trade in trades:
                    formatted_trade = {
                        'tradeid': trade.get('id', ''),
                        'orderid': trade.get('orderNumber', ''),
                        'symbol': trade.get('symbol', ''),
                        'product': trade.get('productType', ''),
                        'action': 'BUY' if trade.get('side', 1) == 1 else 'SELL',
                        'quantity': str(trade.get('qty', 0)),
                        'price': str(trade.get('tradePrice', 0)),
                        'timestamp': trade.get('tradeDateTime', ''),
                        'exchange': self._extract_exchange(trade.get('symbol', '')),
                        'tag': trade.get('orderTag', '')
                    }
                    formatted_trades.append(formatted_trade)
                
                return {
                    'status': 'success',
                    'data': formatted_trades
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Trade book error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'TRADEBOOK_FAILED'
            }
    
    def positions(self) -> Dict[str, Any]:
        """
        Get positions with standard format.
        """
        try:
            result = self._make_request('GET', 'positions')
            
            if result['status'] == 'success':
                positions = result.get('data', {}).get('netPositions', [])
                formatted_positions = []
                
                for position in positions:
                    formatted_position = {
                        'symbol': position.get('symbol', ''),
                        'product': position.get('productType', ''),
                        'quantity': str(position.get('netQty', 0)),
                        'average_price': str(position.get('avgPrice', 0)),
                        'last_price': str(position.get('ltp', 0)),
                        'pnl': str(position.get('unrealizedProfit', 0)),
                        'exchange': self._extract_exchange(position.get('symbol', ''))
                    }
                    formatted_positions.append(formatted_position)
                
                return {
                    'status': 'success',
                    'data': formatted_positions
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Positions error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'POSITIONS_FAILED'
            }
    
    def holdings(self) -> Dict[str, Any]:
        """
        Get holdings with standard format.
        """
        try:
            result = self._make_request('GET', 'holdings')
            
            if result['status'] == 'success':
                holdings = result.get('data', {}).get('holdings', [])
                formatted_holdings = []
                
                for holding in holdings:
                    formatted_holding = {
                        'symbol': holding.get('symbol', ''),
                        'quantity': str(holding.get('qty', 0)),
                        'average_price': str(holding.get('costPrice', 0)),
                        'last_price': str(holding.get('ltp', 0)),
                        'pnl': str(holding.get('pl', 0)),
                        'exchange': self._extract_exchange(holding.get('symbol', ''))
                    }
                    formatted_holdings.append(formatted_holding)
                
                return {
                    'status': 'success',
                    'data': formatted_holdings
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Holdings error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'HOLDINGS_FAILED'
            }
    
    def funds(self) -> Dict[str, Any]:
        """
        Get account funds with standard format.
        """
        try:
            result = self._make_request('GET', 'funds')
            
            if result['status'] == 'success':
                fund_data = result.get('data', {}).get('fund_limit', [])
                
                # Extract key fund information
                funds_info = {
                    'available_cash': '0',
                    'utilized_margin': '0',
                    'total_margin': '0'
                }
                
                for fund in fund_data:
                    if fund.get('title') == 'Available Cash':
                        funds_info['available_cash'] = str(fund.get('equityAmount', 0))
                    elif fund.get('title') == 'Total Balance':
                        funds_info['total_margin'] = str(fund.get('equityAmount', 0))
                    elif fund.get('title') == 'Utilized Margin':
                        funds_info['utilized_margin'] = str(fund.get('equityAmount', 0))
                
                return {
                    'status': 'success',
                    'data': funds_info
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Funds error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'FUNDS_FAILED'
            }
    
    # Market Data
    def quotes(self, symbol: str, exchange: str = "") -> Dict[str, Any]:
        """
        Get real-time quotes with standard format.
        """
        try:
            # Format symbol for Fyers API
            if exchange and ":" not in symbol:
                formatted_symbol = f"{exchange}:{symbol}"
            else:
                formatted_symbol = symbol
            
            params = {"symbols": formatted_symbol}
            result = self._make_request('GET', 'quotes', params=params)
            
            if result['status'] == 'success':
                quotes_data = result.get('data', {}).get('d', {})
                
                if formatted_symbol in quotes_data:
                    quote = quotes_data[formatted_symbol]['v']
                    
                    formatted_quote = {
                        'symbol': symbol,
                        'exchange': exchange or self._extract_exchange(formatted_symbol),
                        'ltp': str(quote.get('lp', 0)),
                        'open': str(quote.get('open_price', 0)),
                        'high': str(quote.get('h', 0)),
                        'low': str(quote.get('l', 0)),
                        'prev_close': str(quote.get('prev_close_price', 0)),
                        'change': str(quote.get('ch', 0)),
                        'change_percent': str(quote.get('chp', 0)),
                        'volume': str(quote.get('volume', 0)),
                        'bid': str(quote.get('bid', 0)),
                        'ask': str(quote.get('ask', 0)),
                        'timestamp': str(quote.get('timestamp', ''))
                    }
                    
                    return {
                        'status': 'success',
                        'data': formatted_quote
                    }
                else:
                    return {
                        'status': 'error',
                        'message': 'Symbol not found in quotes data',
                        'error_code': 'SYMBOL_NOT_FOUND'
                    }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Quotes error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'QUOTES_FAILED'
            }
    
    def depth(self, symbol: str, exchange: str = "") -> Dict[str, Any]:
        """
        Get market depth with standard format.
        """
        try:
            # Format symbol for Fyers API
            if exchange and ":" not in symbol:
                formatted_symbol = f"{exchange}:{symbol}"
            else:
                formatted_symbol = symbol
            
            depth_data = {
                "symbol": [formatted_symbol],
                "ohlcv_flag": 1
            }
            
            result = self._make_request('POST', 'depth', data=depth_data)
            
            if result['status'] == 'success':
                depth_info = result.get('data', {}).get('d', {})
                
                if formatted_symbol in depth_info:
                    market_depth = depth_info[formatted_symbol]['v']
                    
                    formatted_depth = {
                        'symbol': symbol,
                        'exchange': exchange or self._extract_exchange(formatted_symbol),
                        'ltp': str(market_depth.get('lp', 0)),
                        'bid': [],
                        'ask': []
                    }
                    
                    # Extract bid/ask data if available
                    if 'bid' in market_depth:
                        for i in range(5):  # Top 5 levels
                            bid_key = f"bid_price_{i+1}"
                            bid_qty_key = f"bid_size_{i+1}"
                            if bid_key in market_depth:
                                formatted_depth['bid'].append({
                                    'price': str(market_depth.get(bid_key, 0)),
                                    'quantity': str(market_depth.get(bid_qty_key, 0))
                                })
                    
                    if 'ask' in market_depth:
                        for i in range(5):  # Top 5 levels
                            ask_key = f"ask_price_{i+1}"
                            ask_qty_key = f"ask_size_{i+1}"
                            if ask_key in market_depth:
                                formatted_depth['ask'].append({
                                    'price': str(market_depth.get(ask_key, 0)),
                                    'quantity': str(market_depth.get(ask_qty_key, 0))
                                })
                    
                    return {
                        'status': 'success',
                        'data': formatted_depth
                    }
                else:
                    return {
                        'status': 'error',
                        'message': 'Symbol not found in depth data',
                        'error_code': 'SYMBOL_NOT_FOUND'
                    }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Depth error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'DEPTH_FAILED'
            }
    
    def history(self, symbol: str, exchange: str, interval: str,
                start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get historical data with standard format.
        """
        try:
            # Format symbol for Fyers API
            if ":" not in symbol:
                formatted_symbol = f"{exchange}:{symbol}"
            else:
                formatted_symbol = symbol
            
            # Map intervals to Fyers format
            interval_map = {
                '1m': '1',
                '3m': '3',
                '5m': '5',
                '10m': '10',
                '15m': '15',
                '30m': '30',
                '1h': '60',
                '2h': '120',
                '3h': '180',
                '4h': '240',
                '1d': 'D',
                '1D': 'D',
                'D': 'D'
            }
            
            fyers_interval = interval_map.get(interval, 'D')
            
            # Convert date format if needed
            try:
                start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
                end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            except:
                # Try with timestamp format
                start_ts = int(start_date) if start_date.isdigit() else int(datetime.now().timestamp()) - 86400*30
                end_ts = int(end_date) if end_date.isdigit() else int(datetime.now().timestamp())
            
            history_data = {
                "symbol": formatted_symbol,
                "resolution": fyers_interval,
                "date_format": "0",  # Unix timestamp
                "range_from": str(start_ts),
                "range_to": str(end_ts),
                "cont_flag": "1"
            }
            
            result = self._make_request('POST', 'history', data=history_data)
            
            if result['status'] == 'success':
                candles = result.get('data', {}).get('candles', [])
                
                formatted_candles = []
                for candle in candles:
                    if len(candle) >= 6:
                        formatted_candles.append({
                            'timestamp': str(candle[0]),
                            'open': str(candle[1]),
                            'high': str(candle[2]),
                            'low': str(candle[3]),
                            'close': str(candle[4]),
                            'volume': str(candle[5])
                        })
                
                return {
                    'status': 'success',
                    'data': {
                        'symbol': symbol,
                        'exchange': exchange,
                        'interval': interval,
                        'candles': formatted_candles
                    }
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"History error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'HISTORY_FAILED'
            }
    
    # Search and Symbol Information
    def search(self, symbol: str, exchange: str = "") -> Dict[str, Any]:
        """
        Search for symbols with standard format.
        """
        try:
            search_data = {
                "symbol": symbol,
                "exchange": exchange.upper() if exchange else "ALL"
            }
            
            result = self._make_request('POST', 'search_scrips', data=search_data)
            
            if result['status'] == 'success':
                search_results = result.get('data', {}).get('d', [])
                formatted_results = []
                
                for item in search_results:
                    formatted_result = {
                        'symbol': item.get('symbol', ''),
                        'name': item.get('name', ''),
                        'exchange': item.get('exch', ''),
                        'segment': item.get('segment', ''),
                        'instrument_type': item.get('instrument_type', ''),
                        'lot_size': str(item.get('lot_size', 1)),
                        'tick_size': str(item.get('tick_size', 0.01))
                    }
                    formatted_results.append(formatted_result)
                
                return {
                    'status': 'success',
                    'data': formatted_results
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'SEARCH_FAILED'
            }
    
    # Utility Methods
    def _get_order_type_name(self, order_type: int) -> str:
        """Convert Fyers order type number to standard format."""
        type_mapping = {
            1: 'LIMIT',
            2: 'MARKET',
            3: 'SL-M',
            4: 'SL-L'
        }
        return type_mapping.get(order_type, 'MARKET')
    
    def _get_order_status_name(self, status: int) -> str:
        """Convert Fyers order status number to standard format."""
        status_mapping = {
            1: 'PENDING',
            2: 'OPEN',
            3: 'CANCELLED',
            4: 'COMPLETE',
            5: 'REJECTED',
            6: 'EXPIRED'
        }
        return status_mapping.get(status, 'PENDING')
    
    def _extract_exchange(self, symbol: str) -> str:
        """Extract exchange from symbol."""
        if ':' in symbol:
            return symbol.split(':')[0]
        return 'NSE'  # Default
    
    def _extract_symbol_name(self, symbol: str) -> str:
        """Extract clean symbol name."""
        if ':' in symbol:
            parts = symbol.split(':')
            if len(parts) > 1:
                name_part = parts[1].split('-')[0]
                return name_part
        return symbol


# Authentication and session management utilities
class FyersAuth:
    """
    Fyers authentication utilities.
    """
    
    def __init__(self, client_id: str, secret_key: str, redirect_uri: str):
        self.client_id = client_id
        self.secret_key = secret_key
        self.redirect_uri = redirect_uri
        self.base_url = "https://api-t2.fyers.in/vagator/v2"
    
    def generate_auth_url(self, state: str = "trading") -> str:
        """Generate authorization URL for OAuth flow."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "state": state
        }
        
        auth_url = f"https://api.fyers.in/api/v2/generate-authcode?{urlencode(params)}"
        return auth_url
    
    def generate_access_token(self, auth_code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        try:
            url = f"{self.base_url}/generate_access_token"
            
            data = {
                "grant_type": "authorization_code",
                "appIdHash": hashlib.sha256(f"{self.client_id}:{self.secret_key}".encode()).hexdigest(),
                "code": auth_code
            }
            
            response = requests.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('s') == 'ok':
                    return {
                        'status': 'success',
                        'access_token': result.get('access_token'),
                        'refresh_token': result.get('refresh_token', ''),
                        'message': 'Token generated successfully'
                    }
                else:
                    return {
                        'status': 'error',
                        'message': result.get('message', 'Token generation failed'),
                        'error_code': result.get('code', 'TOKEN_ERROR')
                    }
            else:
                return {
                    'status': 'error',
                    'message': f'HTTP {response.status_code}: {response.text}',
                    'error_code': f'HTTP_{response.status_code}'
                }
                
        except Exception as e:
            logger.error(f"Token generation error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'error_code': 'TOKEN_GENERATION_FAILED'
            }


# Factory function for creating API instances
def create_fyers_api(api_key: str, api_secret: str, access_token: str) -> FyersAPI:
    """
    Factory function to create Fyers API instance.
    """
    return FyersAPI(api_key, api_secret, access_token)


def create_fyers_auth(client_id: str, secret_key: str, redirect_uri: str) -> FyersAuth:
    """
    Factory function to create Fyers authentication instance.
    """
    return FyersAuth(client_id, secret_key, redirect_uri)
