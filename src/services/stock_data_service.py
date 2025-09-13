"""
Simplified Stock Data Service
Basic implementation to avoid compilation errors
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MarketCapType(Enum):
    """Market capitalization categories."""
    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"


class StockInfo:
    """Simple stock information structure."""
    def __init__(self, symbol: str, name: str, exchange: str, sector: str, 
                 market_cap: float, market_cap_category: MarketCapType, 
                 current_price: float, volume: int, **kwargs):
        self.symbol = symbol
        self.name = name
        self.exchange = exchange
        self.sector = sector
        self.market_cap = market_cap
        self.market_cap_category = market_cap_category
        self.current_price = current_price
        self.volume = volume
        self.pe_ratio = kwargs.get('pe_ratio')
        self.pb_ratio = kwargs.get('pb_ratio')
        self.roe = kwargs.get('roe')
        self.debt_to_equity = kwargs.get('debt_to_equity')
        self.dividend_yield = kwargs.get('dividend_yield')
        self.beta = kwargs.get('beta')
        self.last_updated = kwargs.get('last_updated', datetime.utcnow())


class StockDataService:
    """Simplified service for managing stock data."""
    
    def __init__(self):
        """Initialize the stock data service."""
        self.LARGE_CAP_MIN = 20000
        self.MID_CAP_MIN = 5000
        self.SMALL_CAP_MIN = 1000
        
        # Mock stock data
        self.mock_stocks = self._create_mock_stocks()
        
    def _create_mock_stocks(self) -> List[StockInfo]:
        """Create mock stock data for testing."""
        stocks = []
        
        # Large cap stocks
        large_cap_data = [
            ("NSE:RELIANCE-EQ", "Reliance Industries Ltd", "Oil & Gas", 150000, 2450.50),
            ("NSE:TCS-EQ", "Tata Consultancy Services Ltd", "Information Technology", 120000, 3850.75),
            ("NSE:INFY-EQ", "Infosys Ltd", "Information Technology", 80000, 1620.00),
            ("NSE:HDFC-EQ", "HDFC Bank Ltd", "Financial Services", 60000, 1680.25),
            ("NSE:ICICIBANK-EQ", "ICICI Bank Ltd", "Financial Services", 55000, 950.80)
        ]
        
        for symbol, name, sector, market_cap, price in large_cap_data:
            stocks.append(StockInfo(
                symbol=symbol, name=name, exchange="NSE", sector=sector,
                market_cap=market_cap, market_cap_category=MarketCapType.LARGE_CAP,
                current_price=price, volume=100000,
                pe_ratio=15.5, pb_ratio=2.1, roe=0.18, debt_to_equity=0.25
            ))
        
        # Mid cap stocks  
        mid_cap_data = [
            ("NSE:PIDILITIND-EQ", "Pidilite Industries Ltd", "Chemicals", 8000, 2850.30),
            ("NSE:BANDHANBNK-EQ", "Bandhan Bank Ltd", "Financial Services", 7500, 195.45),
            ("NSE:MUTHOOTFIN-EQ", "Muthoot Finance Ltd", "Financial Services", 7000, 1750.60),
            ("NSE:BATAINDIA-EQ", "Bata India Ltd", "Consumer Durables", 6500, 1420.80),
            ("NSE:GODREJCP-EQ", "Godrej Consumer Products Ltd", "FMCG", 6000, 1180.25)
        ]
        
        for symbol, name, sector, market_cap, price in mid_cap_data:
            stocks.append(StockInfo(
                symbol=symbol, name=name, exchange="NSE", sector=sector,
                market_cap=market_cap, market_cap_category=MarketCapType.MID_CAP,
                current_price=price, volume=50000,
                pe_ratio=18.2, pb_ratio=2.8, roe=0.16, debt_to_equity=0.30
            ))
        
        # Small cap stocks
        small_cap_data = [
            ("NSE:DIXON-EQ", "Dixon Technologies Ltd", "Electronics", 4500, 5200.75),
            ("NSE:CAMS-EQ", "Computer Age Management Services Ltd", "Financial Services", 4200, 2850.40),
            ("NSE:ROUTE-EQ", "Route Mobile Ltd", "Telecommunications", 3800, 1680.15),
            ("NSE:CLEAN-EQ", "Clean Science and Technology Ltd", "Chemicals", 3500, 1450.90),
            ("NSE:HAPPSTMNDS-EQ", "Happiest Minds Technologies Ltd", "Information Technology", 3200, 850.25)
        ]
        
        for symbol, name, sector, market_cap, price in small_cap_data:
            stocks.append(StockInfo(
                symbol=symbol, name=name, exchange="NSE", sector=sector,
                market_cap=market_cap, market_cap_category=MarketCapType.SMALL_CAP,
                current_price=price, volume=25000,
                pe_ratio=22.5, pb_ratio=3.2, roe=0.14, debt_to_equity=0.35
            ))
            
        return stocks
    
    def initialize_stock_universe(self, user_id: int = 1) -> Dict[str, int]:
        """Initialize stock universe with real data from broker APIs."""
        logger.info("Initializing stock universe with real data...")
        
        try:
            # Try to get real stock data from broker APIs
            from .broker_service import get_broker_service
            
            broker_service = get_broker_service()
            if broker_service:
                # Get stock list from broker
                stocks_data = broker_service.get_stock_list(user_id)
                
                if stocks_data and stocks_data.get('success'):
                    stocks = stocks_data.get('data', [])
                    logger.info(f"Retrieved {len(stocks)} stocks from broker API")
                    
                    # Categorize stocks by market cap
                    large_cap_count = 0
                    mid_cap_count = 0
                    small_cap_count = 0
                    
                    for stock in stocks:
                        market_cap = stock.get('market_cap', 0)
                        if market_cap >= self.LARGE_CAP_MIN:
                            large_cap_count += 1
                        elif market_cap >= self.MID_CAP_MIN:
                            mid_cap_count += 1
                        else:
                            small_cap_count += 1
                    
                    results = {
                        "total_stocks": len(stocks),
                        "large_cap": large_cap_count,
                        "mid_cap": mid_cap_count,
                        "small_cap": small_cap_count,
                        "processed": len(stocks),
                        "errors": 0
                    }
                    
                    return results
            
            # Fallback to mock data if broker API fails
            logger.warning("Broker API not available, using mock data")
            
        except Exception as e:
            logger.error(f"Error initializing stock universe: {e}")
            logger.warning("Falling back to mock data")
        
        # Fallback to mock data
        large_cap_count = len([s for s in self.mock_stocks if s.market_cap_category == MarketCapType.LARGE_CAP])
        mid_cap_count = len([s for s in self.mock_stocks if s.market_cap_category == MarketCapType.MID_CAP])
        small_cap_count = len([s for s in self.mock_stocks if s.market_cap_category == MarketCapType.SMALL_CAP])
        
        results = {
            "total_stocks": len(self.mock_stocks),
            "large_cap": large_cap_count,
            "mid_cap": mid_cap_count,
            "small_cap": small_cap_count,
            "processed": len(self.mock_stocks),
            "errors": 0
        }
        
        logger.info(f"Mock stock universe initialized: {results}")
        return results
    
    def get_stocks_by_category(self, category: MarketCapType, limit: int = None) -> List[StockInfo]:
        """Get stocks filtered by market cap category."""
        stocks = [s for s in self.mock_stocks if s.market_cap_category == category]
        
        if limit:
            stocks = stocks[:limit]
            
        return stocks
    
    def get_mixed_portfolio_stocks(self, large_cap_pct: float = 0.6, 
                                 mid_cap_pct: float = 0.3, 
                                 small_cap_pct: float = 0.1,
                                 total_stocks: int = 20) -> Dict:
        """Get a mixed portfolio of stocks based on market cap allocation."""
        large_cap_count = int(total_stocks * large_cap_pct)
        mid_cap_count = int(total_stocks * mid_cap_pct) 
        small_cap_count = int(total_stocks * small_cap_pct)
        
        # Adjust for rounding
        remaining = total_stocks - (large_cap_count + mid_cap_count + small_cap_count)
        if remaining > 0:
            large_cap_count += remaining
        
        return {
            "large_cap": self.get_stocks_by_category(MarketCapType.LARGE_CAP, large_cap_count),
            "mid_cap": self.get_stocks_by_category(MarketCapType.MID_CAP, mid_cap_count),
            "small_cap": self.get_stocks_by_category(MarketCapType.SMALL_CAP, small_cap_count),
            "allocation": {
                "large_cap_pct": large_cap_pct,
                "mid_cap_pct": mid_cap_pct,
                "small_cap_pct": small_cap_pct,
                "large_cap_count": large_cap_count,
                "mid_cap_count": mid_cap_count,
                "small_cap_count": small_cap_count
            }
        }
    
    def get_small_cap_stocks(self, limit: int = 30) -> List[StockInfo]:
        """Get small cap stocks for High Risk strategy."""
        return self.get_stocks_by_category(MarketCapType.SMALL_CAP, limit)
    
    def update_stock_prices(self, symbols: List[str] = None, user_id: int = 1) -> Dict[str, int]:
        """Mock update stock prices."""
        logger.info("Mock updating stock prices...")
        return {"updated": len(symbols) if symbols else len(self.mock_stocks), "errors": 0}
    
    def get_stock_details(self, symbol: str) -> Optional[StockInfo]:
        """Get detailed information for a specific stock."""
        for stock in self.mock_stocks:
            if stock.symbol == symbol:
                return stock
        return None
    
    def search_stocks(self, query: str, category: MarketCapType = None, limit: int = 10) -> List[StockInfo]:
        """Search stocks by name or symbol."""
        results = []
        query_lower = query.lower()
        
        for stock in self.mock_stocks:
            if (query_lower in stock.symbol.lower() or 
                query_lower in stock.name.lower() or 
                query_lower in stock.sector.lower()):
                
                if category is None or stock.market_cap_category == category:
                    results.append(stock)
                    
                if len(results) >= limit:
                    break
                    
        return results


def get_stock_data_service() -> StockDataService:
    """Get stock data service instance."""
    return StockDataService()
