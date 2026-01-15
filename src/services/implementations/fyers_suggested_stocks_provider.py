"""
Enhanced FYERS Suggested Stocks Provider Implementation with Saga Pattern

Implements comprehensive stock screening with saga pattern for step-by-step updates.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ..interfaces.suggested_stocks_interface import ISuggestedStocksProvider, StrategyType, SuggestedStock
from ..brokers.fyers_service import get_fyers_service

logger = logging.getLogger(__name__)


class FyersSuggestedStocksProvider(ISuggestedStocksProvider):
    """Enhanced FYERS implementation with saga pattern for comprehensive stock screening."""

    def __init__(self):
        self.fyers_service = get_fyers_service()

    def discover_tradeable_stocks(self, user_id: int, exchange: str = "NSE") -> Dict[str, Any]:
        """Discover all tradeable stocks from the Fyers broker API."""
        try:
            logger.info(f"Discovering tradeable stocks from {exchange} via Fyers broker API")

            # Search for stocks using generic sector terms
            search_terms = [
                # Major indices components
                "NIFTY", "SENSEX", "BANKNIFTY", "NIFTYNXT50", "FINNIFTY",
                # Generic sector keywords only
                "BANK", "IT", "PHARMA", "AUTO", "FMCG", "METAL", "INFRA", "ENERGY",
                "FINANCE", "TECH", "HEALTHCARE", "CONSUMER", "COMMODITY",
                # Generic size/volume based searches
                "LTD", "LIMITED", "CORP", "INC", "INDUSTRIES"
            ]

            all_symbols = set()
            discovered_stocks = []
            api_failed = False

            for term in search_terms:
                try:
                    search_result = self.fyers_service.search(user_id, term, exchange)

                    if search_result.get('status') == 'success':
                        symbols = search_result.get('data', [])
                        logger.info(f"Search term '{term}' returned {len(symbols)} results")

                        for symbol_data in symbols:
                            if symbol_data.get('symbol') not in all_symbols:
                                all_symbols.add(symbol_data.get('symbol'))
                                discovered_stocks.append(symbol_data)
                    else:
                        logger.warning(f"Search failed for term '{term}': {search_result.get('error', 'Unknown error')}")
                        api_failed = True
                        continue

                except Exception as e:
                    logger.warning(f"Search failed for term '{term}': {e}")
                    api_failed = True
                    continue

            logger.info(f"Discovered {len(all_symbols)} potential stocks")

            # Categorize by market cap (simplified)
            large_cap = []
            mid_cap = []
            small_cap = []

            for stock in discovered_stocks:
                market_cap = stock.get('market_cap', 0)
                if market_cap > 50000:  # > ₹50,000 Cr
                    large_cap.append(stock)
                elif market_cap > 10000:  # ₹10,000–50,000 Cr
                    mid_cap.append(stock)
                else:  # < ₹10,000 Cr
                    small_cap.append(stock)

            return {
                'success': True,
                'data': {
                    'large_cap': large_cap,
                    'mid_cap': mid_cap,
                    'small_cap': small_cap
                },
                'total': len(discovered_stocks),
                'exchange': exchange,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error discovering stocks: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': {'large_cap': [], 'mid_cap': [], 'small_cap': []},
                'total': 0,
                'last_updated': datetime.now().isoformat()
            }

    def search_stocks(self, user_id: int, query: str, limit: int = 50,
                     filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search for stocks by name or symbol with advanced screening."""
        try:
            # Use FYERS search API
            search_response = self.fyers_service.search(user_id, query)
            
            if not search_response.get('success'):
                return {
                    'success': False,
                    'error': search_response.get('error', 'Failed to search stocks'),
                    'data': [],
                    'total': 0,
                    'query': query,
                    'last_updated': datetime.now().isoformat()
                }
            
            search_results = search_response.get('data', [])
            
            # Apply filters if provided
            if filters:
                filtered_results = []
                for result in search_results:
                    if self._applies_search_filters(result, filters):
                        filtered_results.append(result)
                search_results = filtered_results
            
            # Enrich results with real-time data
            enriched_results = []
            symbols = [result['symbol'] for result in search_results[:20]]  # Limit for quotes API
            
            if symbols:
                quotes_response = self.fyers_service.quotes(user_id, symbols)
                quotes_data = quotes_response.get('data', {}) if quotes_response.get('success') else {}
                
                for result in search_results:
                    symbol = result['symbol']
                    enriched_result = result.copy()
                    
                    if symbol in quotes_data and quotes_data[symbol].get('v'):
                        quote = quotes_data[symbol]['v']
                        enriched_result.update({
                            'current_price': quote.get('lp', 0),
                            'change': quote.get('ch', 0),
                            'change_percent': quote.get('chp', 0),
                            'volume': quote.get('vol', 0),
                            'high': quote.get('h', 0),
                            'low': quote.get('l', 0),
                            'open': quote.get('o', 0),
                            'previous_close': quote.get('prev_close_price', 0)
                        })
                    
                    enriched_results.append(enriched_result)
            else:
                enriched_results = search_results
            
            # Apply limit
            enriched_results = enriched_results[:limit]
            
            return {
                'success': True,
                'data': enriched_results,
                'total': len(enriched_results),
                'query': query,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error searching stocks for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0,
                'query': query,
                'last_updated': datetime.now().isoformat()
            }

    def get_suggested_stocks(self, user_id: int, strategies: List[StrategyType] = None,
                           limit: int = 50, search: str = None, sort_by: str = None,
                           sort_order: str = 'desc', sector: str = None, config=None) -> Dict[str, Any]:
        """Get suggested stocks using saga pattern with comprehensive step-by-step updates.

        Args:
            user_id: User ID for the request
            strategies: List of strategies to apply
            limit: Maximum number of stocks to return
            search: Search query string
            sort_by: Field to sort results by
            sort_order: Sort order ('asc' or 'desc')
            sector: Filter by specific sector
            config: Optional StockScreeningConfig for customizing the screening pipeline
                   If None, uses default configuration

        Returns:
            Dictionary with comprehensive saga results including step-by-step information
        """
        try:
            if not strategies:
                strategies = [StrategyType.UNIFIED]  # Single unified EMA strategy

            # Convert StrategyType enum to string for saga
            strategy_strings = [s.value for s in strategies]
            
            print(f"🚀 Starting Suggested Stocks Saga for strategies: {strategy_strings}")
            print(f"   📊 Parameters: limit={limit}, search='{search}', sort_by='{sort_by}', sector='{sector}'")
            
            # Execute saga pattern
            from ..data.suggested_stocks_saga import get_suggested_stocks_saga_orchestrator
            saga_orchestrator = get_suggested_stocks_saga_orchestrator()
            
            saga_results = saga_orchestrator.execute_suggested_stocks_saga(
                user_id=user_id,
                strategies=strategy_strings,
                limit=limit,
                search=search,
                sort_by=sort_by,
                sort_order=sort_order,
                sector=sector
            )
            
            # Convert saga results to expected format
            if saga_results['status'] == 'completed':
                return {
                    'success': True,
                    'data': saga_results['final_results'],
                    'total': len(saga_results['final_results']),
                    'strategies_applied': strategy_strings,
                    'last_updated': datetime.now().isoformat(),
                    'saga_results': saga_results  # Include full saga information
                }
            else:
                return {
                    'success': False,
                    'error': f"Saga execution failed: {saga_results.get('errors', ['Unknown error'])}",
                    'data': [],
                    'total': 0,
                    'strategies_applied': strategy_strings,
                    'last_updated': datetime.now().isoformat(),
                    'saga_results': saga_results  # Include saga information for debugging
                }
            
        except Exception as e:
            logger.error(f"Error getting suggested stocks for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0,
                'strategies_applied': [s.value for s in strategies],
                'last_updated': datetime.now().isoformat()
            }
    
    def get_technical_screener(self, user_id: int, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Screen stocks based on technical criteria.
        
        Args:
            user_id: The user ID for broker-specific authentication
            criteria: Technical screening criteria
            
        Returns:
            Dict containing:
            - success: bool
            - data: List of stocks matching technical criteria
            - criteria_applied: Applied screening criteria
            - last_updated: timestamp
        """
        try:
            logger.info(f"Technical screening request for user {user_id}")
            
            # Use the saga pattern for technical screening
            from ..data.suggested_stocks_saga import get_suggested_stocks_saga_orchestrator
            saga_orchestrator = get_suggested_stocks_saga_orchestrator()
            
            # Execute saga with technical focus
            saga_results = saga_orchestrator.execute_suggested_stocks_saga(
                user_id=user_id,
                strategies=['unified'],  # Unified 8-21 EMA strategy
                limit=criteria.get('limit', 50),
                search=criteria.get('search'),
                sort_by=criteria.get('sort_by', 'current_price'),
                sort_order=criteria.get('sort_order', 'desc'),
                sector=criteria.get('sector')
            )
            
            if saga_results['status'] == 'completed':
                return {
                    'success': True,
                    'data': saga_results['final_results'],
                    'criteria_applied': criteria,
                    'last_updated': datetime.now().isoformat(),
                    'saga_results': saga_results
                }
            else:
                return {
                    'success': False,
                    'error': f"Technical screening failed: {saga_results.get('errors', ['Unknown error'])}",
                    'data': [],
                    'criteria_applied': criteria,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Technical screening error for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'criteria_applied': criteria,
                'last_updated': datetime.now().isoformat()
            }
    
    def get_fundamental_screener(self, user_id: int, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Screen stocks based on fundamental criteria.
        
        Args:
            user_id: The user ID for broker-specific authentication
            criteria: Fundamental screening criteria
            
        Returns:
            Dict containing:
            - success: bool
            - data: List of stocks matching fundamental criteria
            - criteria_applied: Applied screening criteria
            - last_updated: timestamp
        """
        try:
            logger.info(f"Fundamental screening request for user {user_id}")
            
            # Use the saga pattern for fundamental screening
            from ..data.suggested_stocks_saga import get_suggested_stocks_saga_orchestrator
            saga_orchestrator = get_suggested_stocks_saga_orchestrator()
            
            # Execute saga with fundamental focus
            saga_results = saga_orchestrator.execute_suggested_stocks_saga(
                user_id=user_id,
                strategies=['unified'],  # Unified 8-21 EMA strategy
                limit=criteria.get('limit', 50),
                search=criteria.get('search'),
                sort_by=criteria.get('sort_by', 'pe_ratio'),
                sort_order=criteria.get('sort_order', 'asc'),
                sector=criteria.get('sector')
            )
            
            if saga_results['status'] == 'completed':
                return {
                    'success': True,
                    'data': saga_results['final_results'],
                    'criteria_applied': criteria,
                    'last_updated': datetime.now().isoformat(),
                    'saga_results': saga_results
                }
            else:
                return {
                    'success': False,
                    'error': f"Fundamental screening failed: {saga_results.get('errors', ['Unknown error'])}",
                    'data': [],
                    'criteria_applied': criteria,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Fundamental screening error for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'criteria_applied': criteria,
                'last_updated': datetime.now().isoformat()
            }

    def get_stock_analysis(self, symbol: str, user_id: int) -> Dict[str, Any]:
        """Get detailed analysis for a specific stock."""
        try:
            # Get real-time quote
            quotes_response = self.fyers_service.quotes(user_id, [symbol])
            
            if not quotes_response.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to get stock data',
                    'data': {},
                    'symbol': symbol,
                    'last_updated': datetime.now().isoformat()
                }
            
            quotes_data = quotes_response.get('data', {})
            if symbol not in quotes_data:
                return {
                    'success': False,
                    'error': 'Stock not found',
                    'data': {},
                    'symbol': symbol,
                    'last_updated': datetime.now().isoformat()
                }
            
            quote = quotes_data[symbol]['v']
            
            # Basic analysis
            current_price = quote.get('lp', 0)
            change = quote.get('ch', 0)
            change_percent = quote.get('chp', 0)
            volume = quote.get('vol', 0)
            
            analysis = {
                'symbol': symbol,
                'current_price': current_price,
                'change': change,
                'change_percent': change_percent,
                'volume': volume,
                'high': quote.get('h', 0),
                'low': quote.get('l', 0),
                'open': quote.get('o', 0),
                'previous_close': quote.get('prev_close_price', 0),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return {
                'success': True,
                'data': analysis,
                'symbol': symbol,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting stock analysis for {symbol}, user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': {},
                'symbol': symbol,
                'last_updated': datetime.now().isoformat()
            }

    def get_strategy_performance(self, user_id: int) -> Dict[str, Any]:
        """Get performance metrics for different strategies."""
        try:
            # This would typically query historical performance data
            # For now, return mock data
            performance_data = {
                'DEFAULT_RISK': {
                    'total_trades': 45,
                    'win_rate': 0.68,
                    'avg_return': 0.12,
                    'max_drawdown': 0.08
                },
                'HIGH_RISK': {
                    'total_trades': 32,
                    'win_rate': 0.56,
                    'avg_return': 0.18,
                    'max_drawdown': 0.15
                }
            }
            
            return {
                'success': True,
                'data': performance_data,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting strategy performance for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': {},
                'last_updated': datetime.now().isoformat()
            }

    def get_sector_analysis(self, user_id: int) -> Dict[str, Any]:
        """Get sector-wise analysis and recommendations."""
        try:
            # This would typically analyze sector performance
            # For now, return mock data
            sector_data = [
                {
                    'sector': 'Banking',
                    'performance': 0.15,
                    'volatility': 0.12,
                    'recommendation': 'BUY',
                    'top_stocks': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK']
                },
                {
                    'sector': 'IT',
                    'performance': 0.08,
                    'volatility': 0.10,
                    'recommendation': 'HOLD',
                    'top_stocks': ['TCS', 'INFY', 'HCLTECH']
                }
            ]
            
            return {
                'success': True,
                'data': sector_data,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting sector analysis for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }

    def run_technical_screener(self, user_id: int, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Run technical screening based on specified criteria."""
        try:
            # This would implement technical screening logic
            # For now, return mock data
            screened_stocks = [
                {
                    'symbol': 'RELIANCE',
                    'name': 'Reliance Industries Ltd',
                    'current_price': 2500.0,
                    'technical_score': 0.85,
                    'signals': ['RSI_OVERSOLD', 'MACD_BULLISH', 'BOLLINGER_SQUEEZE']
                }
            ]
            
            return {
                'success': True,
                'data': screened_stocks,
                'total': len(screened_stocks),
                'criteria': criteria,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error running technical screener for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }

    def run_fundamental_screener(self, user_id: int, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Run fundamental screening based on specified criteria."""
        try:
            # This would implement fundamental screening logic
            # For now, return mock data
            screened_stocks = [
                {
                    'symbol': 'TCS',
                    'name': 'Tata Consultancy Services Ltd',
                    'current_price': 3500.0,
                    'pe_ratio': 25.5,
                    'pb_ratio': 6.2,
                    'roe': 0.18,
                    'fundamental_score': 0.78
                }
            ]
            
            return {
                'success': True,
                'data': screened_stocks,
                'total': len(screened_stocks),
                'criteria': criteria,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error running fundamental screener for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'last_updated': datetime.now().isoformat()
            }

    def _applies_search_filters(self, stock: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if a stock applies to the search filters."""
        try:
            for key, value in filters.items():
                if key in stock:
                    if isinstance(value, dict):
                        # Range filter
                        if 'min' in value and stock[key] < value['min']:
                            return False
                        if 'max' in value and stock[key] > value['max']:
                            return False
                    elif isinstance(value, list):
                        # List filter
                        if stock[key] not in value:
                            return False
                    else:
                        # Exact match
                        if stock[key] != value:
                            return False
            return True
        except Exception as e:
            logger.warning(f"Error applying search filters: {e}")
            return True
