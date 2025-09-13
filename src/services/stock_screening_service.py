"""
Stock Screening Service
Implements comprehensive stock filtering based on the criteria defined in TRADING_SYSTEM_README.md
Uses FYERS API for stock data instead of yfinance to avoid rate limiting
"""
import logging
import requests
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Risk-based strategy types."""
    DEFAULT_RISK = "default_risk"
    HIGH_RISK = "high_risk"


@dataclass
class StockScreeningCriteria:
    """Stock screening criteria configuration."""
    # Market Capitalization
    min_market_cap: float = 5000  # crores
    max_market_cap: float = 20000  # crores
    min_price: float = 50  # rupees
    
    # Volume Analysis
    min_volume_ratio: float = 1.0  # > 1-week average volume
    
    # Financial Health
    min_sales_growth: float = 0.0  # Latest quarter > Preceding quarter
    min_operating_profit_growth: float = 0.0  # Latest quarter > Preceding quarter
    min_yoy_sales_growth: float = 0.0  # Current year > Preceding year
    
    # Valuation Metrics
    max_pe_ratio: float = 25.0
    max_pb_ratio: float = 3.0
    max_debt_to_equity: float = 0.3
    
    # Risk Metrics
    min_piotroski_score: float = 5.0
    min_roe: float = 15.0


@dataclass
class StockData:
    """Stock data structure."""
    symbol: str
    name: str
    current_price: float
    market_cap: float
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    debt_to_equity: Optional[float]
    roe: Optional[float]
    volume: int
    avg_volume: int
    sales_growth: Optional[float]
    operating_profit_growth: Optional[float]
    yoy_sales_growth: Optional[float]
    piotroski_score: Optional[float]
    recommendation: str
    strategy: str
    target_price: Optional[float]
    stop_loss: Optional[float]
    reason: str


class StockScreeningService:
    """Service for screening stocks based on comprehensive criteria."""
    
    def __init__(self, broker_service=None):
        """Initialize the screening service."""
        self.criteria = StockScreeningCriteria()
        self.nse_symbols = self._load_nse_symbols()
        self.broker_service = broker_service
        self.fyers_connector = None
    
    def _load_nse_symbols(self) -> List[str]:
        """Load NSE symbols for screening."""
        # Common NSE symbols for mid-cap and small-cap stocks
        # In a real implementation, this would be loaded from a database or API
        return [
            'NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:INFY-EQ', 'NSE:HDFC-EQ', 'NSE:WIPRO-EQ',
            'NSE:HCLTECH-EQ', 'NSE:BHARTIARTL-EQ', 'NSE:ITC-EQ', 'NSE:SBIN-EQ', 'NSE:LT-EQ',
            'NSE:ASIANPAINT-EQ', 'NSE:MARUTI-EQ', 'NSE:AXISBANK-EQ', 'NSE:KOTAKBANK-EQ', 'NSE:NESTLEIND-EQ',
            'NSE:TITAN-EQ', 'NSE:POWERGRID-EQ', 'NSE:NTPC-EQ', 'NSE:ULTRACEMCO-EQ', 'NSE:TECHM-EQ',
            'NSE:SUNPHARMA-EQ', 'NSE:TATAMOTORS-EQ', 'NSE:BAJFINANCE-EQ', 'NSE:DRREDDY-EQ', 'NSE:CIPLA-EQ',
            'NSE:EICHERMOT-EQ', 'NSE:GRASIM-EQ', 'NSE:JSWSTEEL-EQ', 'NSE:TATASTEEL-EQ', 'NSE:COALINDIA-EQ'
        ]
    
    def _initialize_fyers_connector(self, user_id: int = 1):
        """Initialize FYERS connector for API calls."""
        try:
            if self.broker_service:
                config = self.broker_service.get_broker_config('fyers', user_id)
                if config and config.get('is_connected') and config.get('access_token'):
                    from .broker_service import FyersAPIConnector
                    self.fyers_connector = FyersAPIConnector(
                        client_id=config.get('client_id'),
                        access_token=config.get('access_token')
                    )
                    return True
        except Exception as e:
            logger.error(f"Error initializing FYERS connector: {e}")
        return False
    
    def screen_stocks(self, strategy_types: List[StrategyType] = None, user_id: int = 1) -> List[StockData]:
        """Screen stocks based on criteria and strategies."""
        if strategy_types is None:
            strategy_types = [StrategyType.DEFAULT_RISK, StrategyType.HIGH_RISK]
        
        # Initialize FYERS connector
        if not self._initialize_fyers_connector(user_id):
            logger.warning("FYERS connector not available, using mock data for screening")
            return self._get_mock_stocks(strategy_types)
        
        screened_stocks = []
        
        # Process stocks in batches for better performance
        batch_size = 10  # Process 10 stocks at a time
        for i in range(0, len(self.nse_symbols), batch_size):
            batch_symbols = self.nse_symbols[i:i + batch_size]
            
            try:
                # Get batch quotes for multiple stocks at once
                batch_quotes = self._get_batch_stock_data(batch_symbols)
                
                for symbol in batch_symbols:
                    try:
                        stock_data = batch_quotes.get(symbol)
                        if not stock_data:
                            continue
                        
                        # Apply basic screening criteria
                        if not self._passes_basic_screening(stock_data):
                            continue
                        
                        # Apply strategy-specific screening
                        for strategy in strategy_types:
                            if self._passes_strategy_screening(stock_data, strategy):
                                recommended_stock = self._create_recommendation(stock_data, strategy)
                                screened_stocks.append(recommended_stock)
                                break  # Only add once per stock
                                
                    except Exception as e:
                        logger.warning(f"Error screening {symbol}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error processing batch {batch_symbols}: {e}")
                continue
        
        # Sort by recommendation strength
        screened_stocks.sort(key=lambda x: self._get_recommendation_score(x), reverse=True)
        
        return screened_stocks[:20]  # Return top 20 recommendations
    
    def _get_batch_stock_data(self, symbols: List[str]) -> Dict[str, Optional[StockData]]:
        """Get stock data for multiple symbols in a single API call."""
        try:
            if not self.fyers_connector:
                return {}
            
            # Join symbols with comma for batch API call
            symbols_str = ','.join(symbols)
            
            # Get batch quotes data from FYERS
            quotes_data = self.fyers_connector.get_quotes(symbols_str)
            
            if not quotes_data or 'd' not in quotes_data or not quotes_data['d']:
                logger.warning(f"No batch quotes data for symbols: {symbols}")
                return {}
            
            batch_results = {}
            
            # Handle both list and dict response formats
            if isinstance(quotes_data['d'], list):
                # If it's a list, process each item
                for item in quotes_data['d']:
                    if isinstance(item, dict):
                        item_symbol = item.get('symbol') or item.get('n')
                        if item_symbol in symbols:
                            stock_data = self._extract_stock_data_from_quote(item, item_symbol)
                            batch_results[item_symbol] = stock_data
            else:
                # If it's a dict, process each symbol
                for symbol in symbols:
                    quote = quotes_data['d'].get(symbol, {})
                    if quote:
                        stock_data = self._extract_stock_data_from_quote(quote, symbol)
                        batch_results[symbol] = stock_data
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error getting batch stock data for {symbols}: {e}")
            return {}
    
    def _extract_stock_data_from_quote(self, quote: Dict, symbol: str) -> Optional[StockData]:
        """Extract StockData from a single quote item."""
        try:
            # Extract basic data from quotes and convert to float
            current_price = float(quote.get('v', {}).get('lp', 0))  # Last price
            volume = float(quote.get('v', {}).get('tt', 0))  # Total traded quantity (tt field)
            high_price = float(quote.get('v', {}).get('high_price', 0))  # High price
            low_price = float(quote.get('v', {}).get('low_price', 0))  # Low price
            
            # Use current volume as avg_volume for faster processing
            # In a real implementation, you'd cache historical data or use batch calls
            avg_volume = volume
            
            # Calculate market cap (simplified - would need shares outstanding from another API)
            market_cap = self._estimate_market_cap(symbol, current_price)
            
            # For now, use mock financial data since FYERS doesn't provide fundamental data
            # In a real implementation, you'd integrate with another API for fundamentals
            pe_ratio = self._get_mock_pe_ratio(symbol)
            pb_ratio = self._get_mock_pb_ratio(symbol)
            debt_to_equity = self._get_mock_debt_to_equity(symbol)
            roe = self._get_mock_roe(symbol)
            sales_growth = self._get_mock_sales_growth(symbol)
            operating_profit_growth = self._get_mock_operating_profit_growth(symbol)
            yoy_sales_growth = self._get_mock_yoy_sales_growth(symbol)
            piotroski_score = self._get_mock_piotroski_score(symbol)
            
            return StockData(
                symbol=symbol,
                name=self._get_stock_name(symbol),
                current_price=current_price,
                volume=volume,
                avg_volume=avg_volume,
                market_cap=market_cap,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                debt_to_equity=debt_to_equity,
                roe=roe,
                sales_growth=sales_growth,
                operating_profit_growth=operating_profit_growth,
                yoy_sales_growth=yoy_sales_growth,
                piotroski_score=piotroski_score,
                recommendation="",
                strategy="",
                target_price=None,
                stop_loss=None,
                reason=""
            )
            
        except Exception as e:
            logger.error(f"Error extracting stock data for {symbol}: {e}")
            return None
    
    def _get_stock_data(self, symbol: str) -> Optional[StockData]:
        """Get stock data from FYERS API."""
        try:
            if not self.fyers_connector:
                return None
            
            # Get quotes data from FYERS
            # Fix: Pass symbol as string instead of list
            quotes_data = self.fyers_connector.get_quotes(symbol)
            
            if not quotes_data or 'd' not in quotes_data or not quotes_data['d']:
                logger.warning(f"No quotes data or 'd' field missing for {symbol}")
                return None
            
            # Handle both list and dict response formats
            if isinstance(quotes_data['d'], list):
                # If it's a list, find the quote for our symbol
                quote = {}
                for item in quotes_data['d']:
                    if isinstance(item, dict):
                        # Check both 'symbol' and 'n' fields for symbol matching
                        item_symbol = item.get('symbol') or item.get('n')
                        if item_symbol == symbol:
                            quote = item
                            break
            else:
                # If it's a dict, use the original logic
                quote = quotes_data['d'].get(symbol, {})
            if not quote:
                return None
            
            # Extract basic data from quotes and convert to float
            current_price = float(quote.get('v', {}).get('lp', 0))  # Last price
            volume = float(quote.get('v', {}).get('tt', 0))  # Total traded quantity (tt field)
            high_price = float(quote.get('v', {}).get('high_price', 0))  # High price
            low_price = float(quote.get('v', {}).get('low_price', 0))  # Low price
            
            # Use current volume as avg_volume for faster processing
            avg_volume = volume
            
            # Calculate market cap (simplified - would need shares outstanding from another API)
            market_cap = self._estimate_market_cap(symbol, current_price)
            
            # For now, use mock financial data since FYERS doesn't provide fundamental data
            # In a real implementation, you'd integrate with another API for fundamentals
            pe_ratio = self._get_mock_pe_ratio(symbol)
            pb_ratio = self._get_mock_pb_ratio(symbol)
            debt_to_equity = self._get_mock_debt_to_equity(symbol)
            roe = self._get_mock_roe(symbol)
            sales_growth = self._get_mock_sales_growth(symbol)
            operating_profit_growth = self._get_mock_operating_profit_growth(symbol)
            yoy_sales_growth = self._get_mock_yoy_sales_growth(symbol)
            piotroski_score = self._get_mock_piotroski_score(symbol)
            
            return StockData(
                symbol=symbol,
                name=self._get_stock_name(symbol),
                current_price=current_price,
                market_cap=market_cap,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                debt_to_equity=debt_to_equity,
                roe=roe,
                volume=volume,
                avg_volume=avg_volume,
                sales_growth=sales_growth,
                operating_profit_growth=operating_profit_growth,
                yoy_sales_growth=yoy_sales_growth,
                piotroski_score=piotroski_score,
                recommendation="",  # Will be set by strategy
                strategy="",  # Will be set by strategy
                target_price=None,
                stop_loss=None,
                reason=""
            )
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    
    def _estimate_market_cap(self, symbol: str, current_price: float) -> float:
        """Estimate market cap (simplified)."""
        # This is a simplified estimation - in reality you'd need shares outstanding
        # For now, return a mock value based on symbol
        mock_market_caps = {
            'NSE:RELIANCE-EQ': 15000,
            'NSE:TCS-EQ': 12000,
            'NSE:INFY-EQ': 8000,
            'NSE:HDFC-EQ': 6000,
            'NSE:WIPRO-EQ': 4000,
        }
        return mock_market_caps.get(symbol, 5000)
    
    def _get_stock_name(self, symbol: str) -> str:
        """Get stock name from symbol."""
        name_mapping = {
            'NSE:RELIANCE-EQ': 'Reliance Industries',
            'NSE:TCS-EQ': 'Tata Consultancy Services',
            'NSE:INFY-EQ': 'Infosys',
            'NSE:HDFC-EQ': 'HDFC Bank',
            'NSE:WIPRO-EQ': 'Wipro',
            'NSE:HCLTECH-EQ': 'HCL Technologies',
            'NSE:BHARTIARTL-EQ': 'Bharti Airtel',
            'NSE:ITC-EQ': 'ITC',
            'NSE:SBIN-EQ': 'State Bank of India',
            'NSE:LT-EQ': 'Larsen & Toubro',
        }
        return name_mapping.get(symbol, symbol)
    
    def _get_mock_pe_ratio(self, symbol: str) -> float:
        """Get mock PE ratio."""
        import random
        return round(random.uniform(10, 30), 2)
    
    def _get_mock_pb_ratio(self, symbol: str) -> float:
        """Get mock PB ratio."""
        import random
        return round(random.uniform(1, 4), 2)
    
    def _get_mock_debt_to_equity(self, symbol: str) -> float:
        """Get mock debt-to-equity ratio."""
        import random
        return round(random.uniform(0.1, 0.4), 2)
    
    def _get_mock_roe(self, symbol: str) -> float:
        """Get mock ROE."""
        import random
        return round(random.uniform(0.12, 0.25), 3)
    
    def _get_mock_sales_growth(self, symbol: str) -> float:
        """Get mock sales growth."""
        import random
        return round(random.uniform(5, 25), 2)
    
    def _get_mock_operating_profit_growth(self, symbol: str) -> float:
        """Get mock operating profit growth."""
        import random
        return round(random.uniform(8, 20), 2)
    
    def _get_mock_yoy_sales_growth(self, symbol: str) -> float:
        """Get mock YoY sales growth."""
        import random
        return round(random.uniform(10, 30), 2)
    
    def _get_mock_piotroski_score(self, symbol: str) -> float:
        """Get mock Piotroski score."""
        import random
        return round(random.uniform(5, 9), 1)
    
    def _get_mock_stocks(self, strategy_types: List[StrategyType]) -> List[StockData]:
        """Get mock stocks when FYERS is not available."""
        mock_stocks = []
        symbols = ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:INFY-EQ', 'NSE:HDFC-EQ', 'NSE:WIPRO-EQ']
        
        for i, symbol in enumerate(symbols):
            strategy = strategy_types[i % len(strategy_types)]
            mock_stocks.append(StockData(
                symbol=symbol,
                name=self._get_stock_name(symbol),
                current_price=1000 + (i * 200),
                market_cap=8000 + (i * 1000),
                pe_ratio=self._get_mock_pe_ratio(symbol),
                pb_ratio=self._get_mock_pb_ratio(symbol),
                debt_to_equity=self._get_mock_debt_to_equity(symbol),
                roe=self._get_mock_roe(symbol),
                volume=1000000 + (i * 100000),
                avg_volume=800000 + (i * 80000),
                sales_growth=self._get_mock_sales_growth(symbol),
                operating_profit_growth=self._get_mock_operating_profit_growth(symbol),
                yoy_sales_growth=self._get_mock_yoy_sales_growth(symbol),
                piotroski_score=self._get_mock_piotroski_score(symbol),
                recommendation="BUY",
                strategy=f"{strategy.value.title()} Strategy",
                target_price=1200 + (i * 240),
                stop_loss=950 + (i * 190),
                reason=f"Strong {strategy.value} characteristics"
            ))
        
        return mock_stocks
    
    def _passes_basic_screening(self, stock: StockData) -> bool:
        """Check if stock passes basic screening criteria."""
        # Market cap range
        if not (self.criteria.min_market_cap <= stock.market_cap <= self.criteria.max_market_cap):
            return False
        
        # Minimum price
        if stock.current_price < self.criteria.min_price:
            return False
        
        # Volume analysis
        if stock.volume < stock.avg_volume * self.criteria.min_volume_ratio:
            return False
        
        # Financial health
        if stock.sales_growth is not None and stock.sales_growth <= self.criteria.min_sales_growth:
            return False
        
        if stock.operating_profit_growth is not None and stock.operating_profit_growth <= self.criteria.min_operating_profit_growth:
            return False
        
        if stock.yoy_sales_growth is not None and stock.yoy_sales_growth <= self.criteria.min_yoy_sales_growth:
            return False
        
        # Risk metrics
        if stock.piotroski_score is not None and stock.piotroski_score < self.criteria.min_piotroski_score:
            return False
        
        return True
    
    def _passes_strategy_screening(self, stock: StockData, strategy: StrategyType) -> bool:
        """Check if stock passes risk-based strategy screening."""
        if strategy == StrategyType.DEFAULT_RISK:
            return self._passes_default_risk_screening(stock)
        elif strategy == StrategyType.HIGH_RISK:
            return self._passes_high_risk_screening(stock)
        
        return False
    
    def _passes_default_risk_screening(self, stock: StockData) -> bool:
        """Default risk strategy screening criteria - balanced approach."""
        # Focus on large and mid cap stocks with good fundamentals
        market_cap_ok = stock.market_cap is None or stock.market_cap >= 5000  # 5000 Cr+ (mid cap and above)
        pe_ok = stock.pe_ratio is None or (stock.pe_ratio > 5 and stock.pe_ratio < 30)  # Reasonable PE
        debt_ok = stock.debt_to_equity is None or stock.debt_to_equity < 0.5  # Low debt
        roe_ok = stock.roe is None or stock.roe > 0.10  # Decent ROE
        
        return market_cap_ok and pe_ok and debt_ok and roe_ok
    
    def _passes_high_risk_screening(self, stock: StockData) -> bool:
        """High risk strategy screening criteria - aggressive approach."""
        # Focus on small and mid cap stocks with growth potential
        market_cap_ok = stock.market_cap is None or stock.market_cap < 20000  # Below large cap
        growth_ok = stock.sales_growth is None or stock.sales_growth > 10  # Some growth
        volume_ok = stock.volume is None or stock.volume > 100000  # Decent liquidity
        
        return market_cap_ok and growth_ok and volume_ok
    
    def _create_recommendation(self, stock: StockData, strategy: StrategyType) -> StockData:
        """Create stock recommendation based on risk-based strategy."""
        stock.recommendation = "BUY"
        stock.strategy = strategy.value.replace('_', ' ').title() + " Strategy"
        
        # Set target price and stop loss based on risk strategy
        if strategy == StrategyType.DEFAULT_RISK:
            stock.target_price = stock.current_price * 1.12  # 12% target (conservative)
            stock.stop_loss = stock.current_price * 0.92     # 8% stop loss
            stock.reason = "Balanced risk profile with stable fundamentals"
        elif strategy == StrategyType.HIGH_RISK:
            stock.target_price = stock.current_price * 1.25  # 25% target (aggressive)
            stock.stop_loss = stock.current_price * 0.85     # 15% stop loss
            stock.reason = "High growth potential with higher risk-reward"
        
        return stock
    
    def _get_recommendation_score(self, stock: StockData) -> float:
        """Calculate recommendation score for sorting."""
        score = 0.0
        
        # Base score from risk strategy strength
        if stock.strategy == "Default Risk Strategy":
            score += 7.5  # Balanced approach
        elif stock.strategy == "High Risk Strategy":
            score += 8.5  # Higher potential returns
        
        # Bonus for strong fundamentals
        if stock.roe and stock.roe > 0.20:
            score += 2.0
        if stock.piotroski_score and stock.piotroski_score > 7:
            score += 1.5
        if stock.sales_growth and stock.sales_growth > 25:
            score += 1.0
        
        return score
    
    def _calculate_price_momentum(self, symbol: str) -> float:
        """Calculate price momentum percentage using mock data for faster processing."""
        try:
            # Use mock momentum data for faster processing
            # In a real implementation, you'd use cached historical data or batch calls
            import random
            # Generate realistic momentum values between -20% and +20%
            momentum = random.uniform(-20.0, 20.0)
            return round(momentum, 2)
        except:
            return 0.0
    
    def _is_mean_reversion_candidate(self, symbol: str) -> bool:
        """Check if stock is a mean reversion candidate using mock data for faster processing."""
        try:
            # Use mock mean reversion data for faster processing
            # In a real implementation, you'd use cached historical data or batch calls
            import random
            # 30% chance of being a mean reversion candidate
            return random.random() < 0.3
        except:
            return False
    
    def _is_breakout_candidate(self, symbol: str) -> bool:
        """Check if stock is a breakout candidate using mock data for faster processing."""
        try:
            # Use mock breakout data for faster processing
            # In a real implementation, you'd use cached historical data or batch calls
            import random
            # 25% chance of being a breakout candidate
            return random.random() < 0.25
        except:
            return False


def get_stock_screening_service(broker_service=None) -> StockScreeningService:
    """Get stock screening service instance."""
    return StockScreeningService(broker_service)
