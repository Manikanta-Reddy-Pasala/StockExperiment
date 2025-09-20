"""
Enhanced FYERS Suggested Stocks Provider Implementation

Implements comprehensive stock screening with search and sort capabilities.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ..interfaces.suggested_stocks_interface import ISuggestedStocksProvider, StrategyType, SuggestedStock
from ..brokers.fyers_service import get_fyers_service

logger = logging.getLogger(__name__)


class FyersSuggestedStocksProvider(ISuggestedStocksProvider):
    """Enhanced FYERS implementation with full search and sort functionality."""

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

                        for symbol_info in symbols:
                            symbol = symbol_info.get('symbol', '')
                            name = symbol_info.get('symbol_name', '')

                            # Filter for equity stocks only
                            if ('-EQ' in symbol and
                                symbol.startswith(f"{exchange}:") and
                                len(name) > 2):
                                all_symbols.add(symbol)
                                discovered_stocks.append({
                                    'symbol': symbol,
                                    'name': name,
                                    'exchange': exchange,
                                    'search_term': term
                                })
                    else:
                        logger.warning(f"Search failed for term '{term}': {search_result.get('message')}")
                        api_failed = True

                    # Rate limiting
                    import time
                    time.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Search failed for term '{term}': {e}")
                    api_failed = True
                    continue

            logger.info(f"Discovered {len(all_symbols)} potential stocks")


            categorized = {
                'large_cap': [],
                'mid_cap': [],
                'small_cap': []
            }

            # Get quotes for discovered stocks and categorize
            if all_symbols:
                quotes_result = self.fyers_service.quotes_multiple(user_id, list(all_symbols)[:50])  # Limit for API

                if quotes_result.get('status') == 'success':
                    quotes_data = quotes_result.get('data', {})

                    for stock in discovered_stocks:
                        symbol = stock['symbol']
                        if symbol in quotes_data:
                            quote = quotes_data[symbol]
                            price = float(quote.get('lp', quote.get('ltp', 0)))

                            if price > 0:
                                # Estimate market cap category based on price (simplified)
                                if price > 1500:
                                    category = 'large_cap'
                                elif price > 500:
                                    category = 'mid_cap'
                                else:
                                    category = 'small_cap'

                                stock_info = {
                                    'symbol': symbol,
                                    'name': stock['name'],
                                    'exchange': stock['exchange'],
                                    'current_price': price,
                                    'volume': float(quote.get('volume', quote.get('vol', 0))),
                                    'market_cap_category': category,
                                    'is_tradeable': True,
                                    'sector': self._determine_sector(stock['name'])
                                }
                                categorized[category].append(stock_info)

            # Summary statistics
            summary = {
                'total_discovered': len(all_symbols),
                'tradeable_stocks': len(discovered_stocks),
                'filtered_stocks': sum(len(stocks) for stocks in categorized.values()),
                'large_cap_count': len(categorized['large_cap']),
                'mid_cap_count': len(categorized['mid_cap']),
                'small_cap_count': len(categorized['small_cap']),
                'discovery_time': datetime.now().isoformat(),
                'data_source': 'Real Fyers Broker API'
            }

            return {
                'success': True,
                'data': categorized,
                'summary': summary,
                'filtering_statistics': summary,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error discovering stocks: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': {'large_cap': [], 'mid_cap': [], 'small_cap': []},
                'summary': {'total_discovered': 0},
                'last_updated': datetime.now().isoformat()
            }

    def search_stocks(self, user_id: int, search_term: str, exchange: str = "NSE") -> Dict[str, Any]:
        """Search for stocks using a search term via Fyers broker API."""
        try:
            search_result = self.fyers_service.search(user_id, search_term, exchange)


            if search_result.get('status') == 'success':
                stocks = search_result.get('data', [])

                # Filter for equity stocks
                filtered_stocks = []
                for stock in stocks:
                    symbol = stock.get('symbol', '')
                    if '-EQ' in symbol and symbol.startswith(f"{exchange}:"):
                        filtered_stocks.append({
                            'symbol': symbol,
                            'name': stock.get('symbol_name', ''),
                            'exchange': exchange,
                            'search_term': search_term
                        })

                return {
                    'success': True,
                    'data': filtered_stocks,
                    'search_term': search_term,
                    'total_results': len(filtered_stocks)
                }
            else:
                return {
                    'success': False,
                    'error': search_result.get('message', 'Search failed'),
                    'data': [],
                    'search_term': search_term,
                    'total_results': 0
                }

        except Exception as e:
            logger.error(f"Error searching stocks: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'search_term': search_term,
                'total_results': 0
            }
    
    def get_suggested_stocks(self, user_id: int, strategies: List[StrategyType] = None, 
                           limit: int = 50, search: str = None, sort_by: str = None,
                           sort_order: str = 'desc', sector: str = None) -> Dict[str, Any]:
        """Get suggested stocks with comprehensive search and sort functionality."""
        try:
            if not strategies:
                strategies = [StrategyType.DEFAULT_RISK, StrategyType.HIGH_RISK]
            
            # Use dynamic stock discovery instead of hardcoded stocks
            try:
                from src.services.ml.stock_discovery_service import get_stock_discovery_service
                discovery_service = get_stock_discovery_service()
                discovered_stocks = discovery_service.get_top_liquid_stocks(user_id, count=50)
                popular_symbols = [stock.symbol for stock in discovered_stocks]
                logger.info(f"Discovered {len(popular_symbols)} stocks dynamically from broker API")
            except Exception as e:
                logger.warning(f"Failed to discover stocks dynamically: {e}")
                # Fallback to generic sector-based search if discovery fails
                popular_symbols = []
                for sector in ['BANK', 'IT', 'PHARMA', 'AUTO', 'FMCG', 'METAL']:
                    try:
                        result = self.fyers_service.search(user_id, sector, 'NSE')
                        if result.get('status') == 'success':
                            symbols = [s.get('symbol') for s in result.get('data', [])[:5]]
                            popular_symbols.extend([s for s in symbols if s and '-EQ' in s])
                    except Exception:
                        continue
            
            # Get quotes for popular stocks
            quotes_response = self.fyers_service.quotes_multiple(user_id, popular_symbols)

            if quotes_response.get('status') != 'success':
                return {
                    'success': False,
                    'error': quotes_response.get('message', 'Failed to fetch stock suggestions'),
                    'data': [],
                    'total': 0,
                    'strategies_applied': [s.value for s in strategies],
                    'last_updated': datetime.now().isoformat()
                }
            
            quotes_data = quotes_response.get('data', {})
            
            # Convert quotes to suggested stocks format
            suggestions = []
            for symbol, quote_data in quotes_data.items():
                suggestions.append({
                    'symbol': symbol,
                    'symbol_name': symbol.replace('NSE:', '').replace('-EQ', ''),
                    'price': float(quote_data.get('ltp', 0)),
                    'change': float(quote_data.get('change', 0)),
                    'change_percent': float(quote_data.get('change_percent', 0)),
                    'volume': int(quote_data.get('volume', 0)),
                    'high': float(quote_data.get('high', 0)),
                    'low': float(quote_data.get('low', 0)),
                    'open': float(quote_data.get('open', 0)),
                    'prev_close': float(quote_data.get('prev_close', 0))
                })
            
            # Apply strategy-based filtering and scoring
            suggested_stocks = []
            for suggestion in suggestions:
                for strategy in strategies:
                    stock = self._create_suggested_stock_from_suggestion(suggestion, strategy)
                    if self._meets_strategy_criteria(stock, strategy):
                        suggested_stocks.append(stock.to_dict())
                        break  # Only add once per stock
            
            # Apply additional sorting if specified
            if sort_by and sort_by != 'volume':
                reverse = sort_order.lower() == 'desc'
                if sort_by in ['symbol', 'name', 'strategy', 'recommendation', 'sector']:
                    suggested_stocks.sort(key=lambda x: x.get(sort_by, '').lower(), reverse=reverse)
                elif sort_by in ['current_price', 'target_price', 'stop_loss', 'market_cap', 
                               'pe_ratio', 'pb_ratio', 'roe', 'sales_growth']:
                    suggested_stocks.sort(key=lambda x: float(x.get(sort_by, 0) or 0), reverse=reverse)
            
            return {
                'success': True,
                'data': suggested_stocks[:limit],
                'total': len(suggested_stocks),
                'search': search,
                'sector': sector,
                'sort_by': sort_by,
                'sort_order': sort_order,
                'strategies_applied': [s.value for s in strategies],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting suggested stocks for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'total': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    def search_stocks(self, user_id: int, query: str, limit: int = 50,
                     filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search for stocks by name or symbol with advanced filtering."""
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
                            'volume': quote.get('volume', 0),
                            'high': quote.get('h', 0),
                            'low': quote.get('l', 0)
                        })
                    
                    enriched_results.append(enriched_result)
            else:
                enriched_results = search_results
            
            return {
                'success': True,
                'data': enriched_results,
                'total': len(enriched_results),
                'query': query,
                'filters_applied': filters,
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
    
    def get_stock_analysis(self, user_id: int, symbol: str) -> Dict[str, Any]:
        """Get comprehensive analysis for a specific stock."""
        try:
            # Get real-time quote
            quotes_response = self.fyers_service.quotes(user_id, symbol)
            
            # Get historical data for technical analysis
            historical_response = self.fyers_service.get_historical_data(
                user_id, symbol, resolution='D', 
                range_from=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            )
            
            # Get market depth
            depth_response = self.fyers_service.get_market_depth(user_id, symbol)
            
            analysis_data = {
                'symbol': symbol,
                'symbol_name': self.fyers_service._extract_symbol_name(symbol),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Process quote data
            if quotes_response.get('success'):
                quote_data = quotes_response.get('data', {})
                if symbol in quote_data and quote_data[symbol].get('v'):
                    quote = quote_data[symbol]['v']
                    analysis_data.update({
                        'current_price': quote.get('lp', 0),
                        'change': quote.get('ch', 0),
                        'change_percent': quote.get('chp', 0),
                        'volume': quote.get('volume', 0),
                        'high': quote.get('h', 0),
                        'low': quote.get('l', 0),
                        'open': quote.get('open_price', 0),
                        'prev_close': quote.get('prev_close_price', 0)
                    })
            
            # Process historical data for technical indicators
            if historical_response.get('success'):
                candles = historical_response.get('data', {}).get('candles', [])
                if candles:
                    analysis_data.update(self._calculate_technical_indicators(candles))
            
            # Process market depth
            if depth_response.get('success'):
                depth_data = depth_response.get('data', {})
                if symbol in depth_data:
                    analysis_data.update(self._process_market_depth(depth_data[symbol]))
            
            # Add fundamental analysis (simplified)
            analysis_data.update(self._get_fundamental_analysis(symbol, analysis_data.get('current_price', 0)))
            
            # Generate recommendation
            analysis_data.update(self._generate_recommendation(analysis_data))
            
            return {
                'success': True,
                'data': analysis_data,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting stock analysis for {symbol}, user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'data': {},
                'last_updated': datetime.now().isoformat()
            }
    
    def get_strategy_performance(self, user_id: int, strategy: StrategyType, 
                               period: str = '1M') -> Dict[str, Any]:
        """Get performance metrics for a specific strategy."""
        try:
            # Get suggested stocks for the strategy
            strategy_stocks = self.get_suggested_stocks(
                user_id, [strategy], limit=20
            )
            
            if not strategy_stocks.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to get strategy stocks for performance analysis',
                    'data': {},
                    'last_updated': datetime.now().isoformat()
                }
            
            stocks = strategy_stocks.get('data', [])
            
            # Calculate performance metrics
            total_return = 0
            winning_stocks = 0
            total_stocks = len(stocks)
            
            for stock in stocks:
                change_percent = stock.get('change_percent', 0)  # Assuming this represents recent performance
                total_return += change_percent
                if change_percent > 0:
                    winning_stocks += 1
            
            avg_return = total_return / total_stocks if total_stocks > 0 else 0
            win_rate = (winning_stocks / total_stocks * 100) if total_stocks > 0 else 0
            
            performance_data = {
                'strategy': strategy.value,
                'period': period,
                'total_return': round(avg_return, 2),
                'win_rate': round(win_rate, 2),
                'avg_return_per_stock': round(avg_return, 2),
                'max_gain': max([stock.get('change_percent', 0) for stock in stocks]) if stocks else 0,
                'max_loss': min([stock.get('change_percent', 0) for stock in stocks]) if stocks else 0,
                'total_stocks_analyzed': total_stocks,
                'winning_stocks': winning_stocks,
                'losing_stocks': total_stocks - winning_stocks,
                'sharpe_ratio': self._calculate_sharpe_ratio([stock.get('change_percent', 0) for stock in stocks]),
                'volatility': self._calculate_volatility([stock.get('change_percent', 0) for stock in stocks])
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
            # TODO: Get sector representatives from broker API or database
            # For now, return empty sector analysis as we don't want hardcoded stocks
            sector_stocks = {}
            
            sector_analysis = []
            
            for sector, symbols in sector_stocks.items():
                # Get quotes for sector stocks
                quotes_response = self.fyers_service.quotes(user_id, symbols)
                
                if quotes_response.get('success'):
                    quotes_data = quotes_response.get('data', {})
                    
                    # Calculate sector performance
                    sector_changes = []
                    sector_volumes = []
                    top_performers = []
                    
                    for symbol in symbols:
                        if symbol in quotes_data and quotes_data[symbol].get('v'):
                            quote = quotes_data[symbol]['v']
                            change_percent = quote.get('chp', 0)
                            volume = quote.get('volume', 0)
                            
                            sector_changes.append(change_percent)
                            sector_volumes.append(volume)
                            
                            top_performers.append({
                                'symbol': symbol,
                                'symbol_name': self.fyers_service._extract_symbol_name(symbol),
                                'change_percent': change_percent,
                                'price': quote.get('lp', 0)
                            })
                    
                    # Calculate sector metrics
                    avg_performance = sum(sector_changes) / len(sector_changes) if sector_changes else 0
                    avg_volume = sum(sector_volumes) / len(sector_volumes) if sector_volumes else 0
                    
                    # Sort top performers
                    top_performers.sort(key=lambda x: x['change_percent'], reverse=True)
                    
                    # Generate recommendation
                    if avg_performance > 2:
                        recommendation = 'BUY'
                    elif avg_performance > -1:
                        recommendation = 'HOLD'
                    else:
                        recommendation = 'SELL'
                    
                    sector_analysis.append({
                        'sector': sector,
                        'performance': round(avg_performance, 2),
                        'recommendation': recommendation,
                        'avg_volume': round(avg_volume, 0),
                        'top_performers': top_performers[:3],
                        'market_sentiment': self._get_market_sentiment(avg_performance),
                        'strength': self._calculate_sector_strength(sector_changes)
                    })
            
            # Sort by performance
            sector_analysis.sort(key=lambda x: x['performance'], reverse=True)
            
            return {
                'success': True,
                'data': sector_analysis,
                'analysis_timestamp': datetime.now().isoformat(),
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
    
    def get_technical_screener(self, user_id: int, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Screen stocks based on technical criteria."""
        try:
            # Get a broader list of stocks to screen
            suggestions_response = self.fyers_service.get_watchlist_suggestions(
                user_id, limit=100, sort_by='volume'
            )
            
            if not suggestions_response.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to get stocks for screening',
                    'data': [],
                    'last_updated': datetime.now().isoformat()
                }
            
            stocks = suggestions_response.get('data', [])
            screened_stocks = []
            
            for stock in stocks:
                # Get historical data for technical analysis
                historical_response = self.fyers_service.get_historical_data(
                    user_id, stock['symbol'], resolution='D',
                    range_from=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                )
                
                if historical_response.get('success'):
                    candles = historical_response.get('data', {}).get('candles', [])
                    if candles:
                        technical_indicators = self._calculate_technical_indicators(candles)
                        
                        # Apply screening criteria
                        if self._meets_technical_criteria(technical_indicators, criteria):
                            screened_stock = {
                                'symbol': stock['symbol'],
                                'symbol_name': stock['symbol_name'],
                                'current_price': stock['price'],
                                'change_percent': stock['change_percent'],
                                'volume': stock['volume'],
                                **technical_indicators
                            }
                            screened_stocks.append(screened_stock)
            
            return {
                'success': True,
                'data': screened_stocks,
                'criteria_applied': criteria,
                'total': len(screened_stocks),
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
    
    def get_fundamental_screener(self, user_id: int, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Screen stocks based on fundamental criteria."""
        try:
            # Get stocks for screening
            suggestions_response = self.fyers_service.get_watchlist_suggestions(
                user_id, limit=100, sort_by='volume'
            )
            
            if not suggestions_response.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to get stocks for screening',
                    'data': [],
                    'last_updated': datetime.now().isoformat()
                }
            
            stocks = suggestions_response.get('data', [])
            screened_stocks = []
            
            for stock in stocks:
                # Get fundamental data (simplified - in reality you'd need external data source)
                fundamental_data = self._get_fundamental_analysis(stock['symbol'], stock['price'])
                
                # Apply screening criteria
                if self._meets_fundamental_criteria(fundamental_data, criteria):
                    screened_stock = {
                        'symbol': stock['symbol'],
                        'symbol_name': stock['symbol_name'],
                        'current_price': stock['price'],
                        'change_percent': stock['change_percent'],
                        'sector': stock['sector'],
                        **fundamental_data
                    }
                    screened_stocks.append(screened_stock)
            
            return {
                'success': True,
                'data': screened_stocks,
                'criteria_applied': criteria,
                'total': len(screened_stocks),
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
    
    # Helper Methods
    def _create_suggested_stock_from_suggestion(self, suggestion: Dict[str, Any], 
                                              strategy: StrategyType) -> SuggestedStock:
        """Create SuggestedStock object from market suggestion."""
        stock = SuggestedStock(
            symbol=suggestion['symbol'],
            name=suggestion['symbol_name'],
            strategy=strategy,
            current_price=suggestion['price'],
            recommendation='BUY'  # Will be refined based on analysis
        )
        
        # Set risk-based strategy targets
        if strategy == StrategyType.DEFAULT_RISK:
            stock.target_price = suggestion['price'] * 1.12  # 12% upside (conservative)
            stock.stop_loss = suggestion['price'] * 0.92     # 8% stop loss
            stock.reason = "Balanced risk profile with stable fundamentals"
        elif strategy == StrategyType.HIGH_RISK:
            stock.target_price = suggestion['price'] * 1.25  # 25% upside (aggressive)
            stock.stop_loss = suggestion['price'] * 0.85     # 15% stop loss
            stock.reason = "High growth potential with higher risk-reward"
        
        return stock
    
    def _meets_strategy_criteria(self, stock: SuggestedStock, strategy: StrategyType) -> bool:
        """Check if stock meets risk-based strategy criteria."""
        # Risk-based criteria
        if strategy == StrategyType.DEFAULT_RISK:
            return stock.current_price > 100  # Stable, established stocks
        elif strategy == StrategyType.HIGH_RISK:
            return stock.current_price < 2000  # Smaller, growth-oriented stocks
        return True
    
    def _applies_search_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply search filters to stock results."""
        if not filters:
            return True
        
        # Exchange filter
        if 'exchange' in filters and filters['exchange']:
            if filters['exchange'].upper() not in result.get('exchange', '').upper():
                return False
        
        # Segment filter
        if 'segment' in filters and filters['segment']:
            if filters['segment'].upper() not in result.get('segment', '').upper():
                return False
        
        # Price range filter
        if 'min_price' in filters and filters['min_price']:
            if result.get('current_price', 0) < filters['min_price']:
                return False
        
        if 'max_price' in filters and filters['max_price']:
            if result.get('current_price', 0) > filters['max_price']:
                return False
        
        return True
    
    def _calculate_technical_indicators(self, candles: List[List]) -> Dict[str, Any]:
        """Calculate technical indicators from historical data."""
        if len(candles) < 20:
            return {}
        
        # Extract price data
        closes = [candle[4] for candle in candles]  # Close prices
        highs = [candle[2] for candle in candles]   # High prices
        lows = [candle[3] for candle in candles]    # Low prices
        volumes = [candle[5] if len(candle) > 5 else 0 for candle in candles]
        
        # Calculate simple moving averages
        sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else 0
        sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else 0
        
        # Calculate RSI (simplified)
        rsi = self._calculate_rsi(closes)
        
        # Calculate MACD (simplified)
        macd_signal = self._calculate_macd_signal(closes)
        
        # Calculate support and resistance
        support = min(lows[-20:]) if len(lows) >= 20 else 0
        resistance = max(highs[-20:]) if len(highs) >= 20 else 0
        
        return {
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'rsi': round(rsi, 2),
            'macd_signal': macd_signal,
            'support': round(support, 2),
            'resistance': round(resistance, 2),
            'avg_volume': round(sum(volumes[-10:]) / 10 if volumes else 0, 0),
            '52_week_high': max(highs) if highs else 0,
            '52_week_low': min(lows) if lows else 0
        }
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd_signal(self, prices: List[float]) -> str:
        """Calculate MACD signal."""
        if len(prices) < 26:
            return 'NEUTRAL'
        
        # Simplified MACD calculation
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        macd_line = ema_12 - ema_26
        
        if macd_line > 0:
            return 'BULLISH'
        elif macd_line < 0:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _process_market_depth(self, depth_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market depth data."""
        return {
            'bid_price': depth_data.get('bids', [{}])[0].get('price', 0) if depth_data.get('bids') else 0,
            'ask_price': depth_data.get('asks', [{}])[0].get('price', 0) if depth_data.get('asks') else 0,
            'bid_quantity': depth_data.get('bids', [{}])[0].get('qty', 0) if depth_data.get('bids') else 0,
            'ask_quantity': depth_data.get('asks', [{}])[0].get('qty', 0) if depth_data.get('asks') else 0,
            'spread': 0  # Calculate bid-ask spread
        }
    
    def _get_fundamental_analysis(self, symbol: str, price: float) -> Dict[str, Any]:
        """Get fundamental analysis data (simplified)."""
        # In a real implementation, you'd fetch this from external data sources
        # This is a simplified version with estimated values
        
        sector = self.fyers_service._get_sector_for_symbol(symbol)
        
        # Simplified fundamental metrics based on sector and price
        if 'BANK' in symbol.upper():
            pe_ratio = 12.5 + (price / 1000) * 2
            pb_ratio = 1.2 + (price / 1000) * 0.5
            roe = 15.0
        elif 'IT' in symbol.upper() or 'TECH' in symbol.upper():
            pe_ratio = 20.0 + (price / 1000) * 3
            pb_ratio = 3.5 + (price / 1000) * 0.8
            roe = 18.0
        else:
            pe_ratio = 15.0 + (price / 1000) * 2.5
            pb_ratio = 2.0 + (price / 1000) * 0.6
            roe = 12.0
        
        return {
            'pe_ratio': round(pe_ratio, 2),
            'pb_ratio': round(pb_ratio, 2),
            'roe': round(roe, 2),
            'debt_to_equity': round(0.3 + (price / 5000) * 0.5, 2),
            'dividend_yield': round(1.5 + (price / 3000) * 1.0, 2),
            'revenue_growth': round(8.0 + (price / 2000) * 5.0, 2),
            'profit_margin': round(10.0 + (price / 1500) * 3.0, 2),
            'market_cap_category': self.fyers_service._get_market_cap_category(price),
            'sector': sector
        }
    
    def _generate_recommendation(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate buy/sell recommendation based on analysis."""
        score = 0
        reasons = []
        
        # Technical analysis scoring
        rsi = analysis_data.get('rsi', 50)
        if 30 <= rsi <= 70:  # Good RSI range
            score += 1
            reasons.append("RSI in favorable range")
        
        macd_signal = analysis_data.get('macd_signal', 'NEUTRAL')
        if macd_signal == 'BULLISH':
            score += 1
            reasons.append("MACD showing bullish signal")
        
        # Price action scoring
        current_price = analysis_data.get('current_price', 0)
        sma_20 = analysis_data.get('sma_20', 0)
        if current_price > sma_20:
            score += 1
            reasons.append("Price above 20-day moving average")
        
        # Fundamental scoring
        pe_ratio = analysis_data.get('pe_ratio', 25)
        if pe_ratio < 20:
            score += 1
            reasons.append("Attractive P/E ratio")
        
        # Generate recommendation
        if score >= 3:
            recommendation = 'BUY'
            target_price = current_price * 1.15
        elif score >= 2:
            recommendation = 'HOLD'
            target_price = current_price * 1.08
        else:
            recommendation = 'SELL'
            target_price = current_price * 0.95
        
        return {
            'recommendation': recommendation,
            'target_price': round(target_price, 2),
            'stop_loss': round(current_price * 0.92, 2),
            'recommendation_score': score,
            'recommendation_reasons': reasons
        }
    
    def _meets_technical_criteria(self, indicators: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if stock meets technical screening criteria."""
        if not criteria:
            return True
        
        # RSI criteria
        if 'rsi_min' in criteria and indicators.get('rsi', 0) < criteria['rsi_min']:
            return False
        if 'rsi_max' in criteria and indicators.get('rsi', 100) > criteria['rsi_max']:
            return False
        
        # MACD criteria
        if 'macd_signal' in criteria and indicators.get('macd_signal') != criteria['macd_signal']:
            return False
        
        # Moving average criteria
        if 'price_above_sma20' in criteria and criteria['price_above_sma20']:
            if indicators.get('current_price', 0) <= indicators.get('sma_20', 0):
                return False
        
        return True
    
    def _meets_fundamental_criteria(self, fundamentals: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if stock meets fundamental screening criteria."""
        if not criteria:
            return True
        
        # P/E ratio criteria
        if 'pe_min' in criteria and fundamentals.get('pe_ratio', 0) < criteria['pe_min']:
            return False
        if 'pe_max' in criteria and fundamentals.get('pe_ratio', 1000) > criteria['pe_max']:
            return False
        
        # ROE criteria
        if 'roe_min' in criteria and fundamentals.get('roe', 0) < criteria['roe_min']:
            return False
        
        # Debt to equity criteria
        if 'debt_equity_max' in criteria and fundamentals.get('debt_to_equity', 0) > criteria['debt_equity_max']:
            return False
        
        return True
    
    def _get_market_sentiment(self, performance: float) -> str:
        """Get market sentiment based on performance."""
        if performance > 3:
            return 'Very Bullish'
        elif performance > 1:
            return 'Bullish'
        elif performance > -1:
            return 'Neutral'
        elif performance > -3:
            return 'Bearish'
        else:
            return 'Very Bearish'
    
    def _calculate_sector_strength(self, changes: List[float]) -> str:
        """Calculate sector strength based on price changes."""
        if not changes:
            return 'Weak'
        
        positive_count = sum(1 for change in changes if change > 0)
        strength_ratio = positive_count / len(changes)
        
        if strength_ratio > 0.8:
            return 'Very Strong'
        elif strength_ratio > 0.6:
            return 'Strong'
        elif strength_ratio > 0.4:
            return 'Moderate'
        else:
            return 'Weak'
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio for strategy performance."""
        if not returns or len(returns) < 2:
            return 0
        
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0
        
        # Assuming risk-free rate of 6% annually
        risk_free_rate = 6.0
        sharpe_ratio = (avg_return - risk_free_rate) / std_dev
        return round(sharpe_ratio, 2)
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate volatility of returns."""
        if not returns or len(returns) < 2:
            return 0
        
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        return round(volatility, 2)


    def _determine_sector(self, company_name: str) -> str:
        """Determine sector from company name using keywords."""
        name = company_name.upper()

        # Banking and Financial Services
        if any(keyword in name for keyword in ['BANK', 'FINANCIAL', 'FINANCE', 'CREDIT', 'LOAN', 'INSURANCE', 'MUTUAL', 'FUND']):
            return 'Banking & Financial Services'

        # Information Technology
        if any(keyword in name for keyword in ['TECH', 'SOFTWARE', 'SYSTEMS', 'INFOTECH', 'TECHNOLOGIES', 'COMPUTER', 'DATA', 'DIGITAL']):
            return 'Information Technology'

        # Pharmaceuticals and Healthcare
        if any(keyword in name for keyword in ['PHARMA', 'DRUG', 'MEDICINE', 'HEALTHCARE', 'HOSPITAL', 'MEDICAL', 'BIO', 'HEALTH']):
            return 'Pharmaceuticals & Healthcare'

        # Automotive
        if any(keyword in name for keyword in ['AUTO', 'MOTOR', 'VEHICLE', 'CAR', 'TRUCK', 'BIKE', 'TYRE', 'TIRE']):
            return 'Automotive'

        # Fast Moving Consumer Goods
        if any(keyword in name for keyword in ['CONSUMER', 'FOOD', 'BEVERAGE', 'PERSONAL', 'CARE', 'HOUSEHOLD', 'FMCG']):
            return 'FMCG'

        # Metals and Mining
        if any(keyword in name for keyword in ['STEEL', 'IRON', 'METAL', 'MINING', 'ALUMINIUM', 'COPPER', 'ZINC', 'COAL']):
            return 'Metals & Mining'

        # Infrastructure and Construction
        if any(keyword in name for keyword in ['CONSTRUCTION', 'INFRASTRUCTURE', 'ENGINEERING', 'BUILDING', 'CEMENT', 'REAL', 'ESTATE']):
            return 'Infrastructure & Construction'

        # Energy and Power
        if any(keyword in name for keyword in ['POWER', 'ENERGY', 'ELECTRICITY', 'SOLAR', 'WIND', 'COAL', 'OIL', 'GAS', 'PETROLEUM']):
            return 'Energy & Power'

        # Telecommunications
        if any(keyword in name for keyword in ['TELECOM', 'COMMUNICATION', 'NETWORK', 'WIRELESS', 'BROADBAND', 'MOBILE']):
            return 'Telecommunications'

        # Textiles
        if any(keyword in name for keyword in ['TEXTILE', 'COTTON', 'FABRIC', 'GARMENT', 'APPAREL', 'CLOTH']):
            return 'Textiles'

        # Media and Entertainment
        if any(keyword in name for keyword in ['MEDIA', 'ENTERTAINMENT', 'TELEVISION', 'BROADCASTING', 'FILM', 'NEWS']):
            return 'Media & Entertainment'

        # Chemicals
        if any(keyword in name for keyword in ['CHEMICAL', 'FERTILIZER', 'PESTICIDE', 'PLASTIC', 'POLYMER']):
            return 'Chemicals'

        # Agriculture
        if any(keyword in name for keyword in ['AGRO', 'AGRICULTURE', 'FARM', 'SEED', 'CROP', 'DAIRY']):
            return 'Agriculture'

        # Default sector if no keywords match
        return 'Diversified'

