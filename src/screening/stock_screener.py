"""
Stock Screening Module for Mid-Cap and Small-Cap Stocks
Implements comprehensive screening criteria for Indian stock market
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from datastore.database import get_database_manager
from datastore.models import Instrument, MarketData
import logging

logger = logging.getLogger(__name__)


class StockScreener:
    """Stock screening engine for mid-cap and small-cap stocks."""
    
    def __init__(self):
        """Initialize the stock screener."""
        self.db_manager = get_database_manager()
        
        # Screening criteria
        self.criteria = {
            'market_cap_min': 5000,  # 5000 crores
            'market_cap_max': 20000,  # 20000 crores
            'current_price_min': 50,
            'debt_to_equity_max': 0.2,
            'piotroski_score_min': 5,
            'intrinsic_value_multiplier': 2.0,
            'high_price_multiplier': 2.0
        }
    
    def get_market_cap_range_stocks(self) -> List[Dict]:
        """
        Get stocks within mid-cap and small-cap market cap range.
        
        Returns:
            List[Dict]: List of stocks with market cap data
        """
        try:
            # This would typically fetch from a data provider like NSE/BSE
            # For now, we'll create a mock implementation
            # In production, this would connect to real market data APIs
            
            mock_stocks = [
                {
                    'symbol': 'RELIANCE',
                    'name': 'Reliance Industries Ltd',
                    'market_cap': 15000,
                    'current_price': 2500,
                    'sector': 'Oil & Gas',
                    'exchange': 'NSE'
                },
                {
                    'symbol': 'TCS',
                    'name': 'Tata Consultancy Services Ltd',
                    'market_cap': 12000,
                    'current_price': 3200,
                    'sector': 'IT',
                    'exchange': 'NSE'
                },
                {
                    'symbol': 'INFY',
                    'name': 'Infosys Ltd',
                    'market_cap': 8000,
                    'current_price': 1500,
                    'sector': 'IT',
                    'exchange': 'NSE'
                },
                {
                    'symbol': 'HDFC',
                    'name': 'HDFC Bank Ltd',
                    'market_cap': 18000,
                    'current_price': 1600,
                    'sector': 'Banking',
                    'exchange': 'NSE'
                },
                {
                    'symbol': 'WIPRO',
                    'name': 'Wipro Ltd',
                    'market_cap': 6000,
                    'current_price': 400,
                    'sector': 'IT',
                    'exchange': 'NSE'
                }
            ]
            
            # Filter by market cap range
            filtered_stocks = [
                stock for stock in mock_stocks
                if self.criteria['market_cap_min'] <= stock['market_cap'] <= self.criteria['market_cap_max']
            ]
            
            logger.info(f"Found {len(filtered_stocks)} stocks in market cap range")
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"Error fetching market cap stocks: {e}")
            return []
    
    def apply_screening_criteria(self, stocks: List[Dict]) -> List[Dict]:
        """
        Apply comprehensive screening criteria to filter stocks.
        
        Args:
            stocks (List[Dict]): List of stocks to screen
            
        Returns:
            List[Dict]: Filtered stocks that meet all criteria
        """
        filtered_stocks = []
        
        for stock in stocks:
            try:
                # Get additional data for screening
                stock_data = self.get_stock_financial_data(stock['symbol'])
                
                if not stock_data:
                    continue
                
                # Apply screening criteria
                if self._meets_screening_criteria(stock, stock_data):
                    stock['screening_data'] = stock_data
                    stock['screening_date'] = datetime.utcnow()
                    filtered_stocks.append(stock)
                    
            except Exception as e:
                logger.error(f"Error screening stock {stock['symbol']}: {e}")
                continue
        
        logger.info(f"Screening completed: {len(filtered_stocks)} stocks passed all criteria")
        return filtered_stocks
    
    def _meets_screening_criteria(self, stock: Dict, stock_data: Dict) -> bool:
        """
        Check if a stock meets all screening criteria.
        
        Args:
            stock (Dict): Basic stock information
            stock_data (Dict): Detailed financial data
            
        Returns:
            bool: True if stock meets all criteria
        """
        try:
            # Market Capitalization criteria (already filtered)
            market_cap = stock['market_cap']
            if not (self.criteria['market_cap_min'] <= market_cap <= self.criteria['market_cap_max']):
                return False
            
            # Current price criteria
            current_price = stock_data.get('current_price', 0)
            if current_price <= self.criteria['current_price_min']:
                return False
            
            # Current price > Low price
            low_price = stock_data.get('low_price', 0)
            if current_price <= low_price:
                return False
            
            # Current price > DMA 50
            dma_50 = stock_data.get('dma_50', 0)
            if current_price <= dma_50:
                return False
            
            # Volume criteria
            current_volume = stock_data.get('volume', 0)
            avg_volume_1week = stock_data.get('avg_volume_1week', 0)
            if current_volume <= avg_volume_1week:
                return False
            
            # Sales growth criteria
            sales_latest = stock_data.get('sales_latest_quarter', 0)
            sales_preceding = stock_data.get('sales_preceding_quarter', 0)
            if sales_latest <= sales_preceding:
                return False
            
            # Operating profit growth criteria
            op_profit_latest = stock_data.get('op_profit_latest_quarter', 0)
            op_profit_preceding = stock_data.get('op_profit_preceding_quarter', 0)
            op_profit_2quarters_back = stock_data.get('op_profit_2quarters_back', 0)
            
            if op_profit_latest <= op_profit_preceding or op_profit_latest <= op_profit_2quarters_back:
                return False
            
            # Year-over-year sales growth
            sales_current_year = stock_data.get('sales_current_year', 0)
            sales_preceding_year = stock_data.get('sales_preceding_year', 0)
            if sales_current_year <= sales_preceding_year:
                return False
            
            # Intrinsic value criteria
            intrinsic_value = stock_data.get('intrinsic_value', 0)
            if current_price >= intrinsic_value:
                return False
            
            # Intrinsic value > Current price * 2
            if intrinsic_value <= current_price * self.criteria['intrinsic_value_multiplier']:
                return False
            
            # High price < Current price * 2
            high_price = stock_data.get('high_price', 0)
            if high_price >= current_price * self.criteria['high_price_multiplier']:
                return False
            
            # Debt to equity ratio
            debt_to_equity = stock_data.get('debt_to_equity', 0)
            if debt_to_equity >= self.criteria['debt_to_equity_max']:
                return False
            
            # Piotroski score
            piotroski_score = stock_data.get('piotroski_score', 0)
            if piotroski_score <= self.criteria['piotroski_score_min']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking screening criteria for {stock['symbol']}: {e}")
            return False
    
    def get_stock_financial_data(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed financial data for a stock.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Optional[Dict]: Financial data or None if not available
        """
        try:
            # This would typically fetch from financial data providers
            # For now, we'll create mock data
            # In production, this would connect to APIs like:
            # - NSE/BSE APIs
            # - Financial data providers (Alpha Vantage, Yahoo Finance, etc.)
            # - Company financial reports
            
            mock_data = {
                'current_price': np.random.uniform(100, 3000),
                'low_price': np.random.uniform(50, 2000),
                'high_price': np.random.uniform(2000, 4000),
                'dma_50': np.random.uniform(80, 2500),
                'volume': np.random.randint(100000, 10000000),
                'avg_volume_1week': np.random.randint(50000, 5000000),
                'sales_latest_quarter': np.random.uniform(1000, 10000),
                'sales_preceding_quarter': np.random.uniform(800, 8000),
                'op_profit_latest_quarter': np.random.uniform(100, 2000),
                'op_profit_preceding_quarter': np.random.uniform(80, 1500),
                'op_profit_2quarters_back': np.random.uniform(60, 1200),
                'sales_current_year': np.random.uniform(4000, 40000),
                'sales_preceding_year': np.random.uniform(3000, 30000),
                'intrinsic_value': np.random.uniform(2000, 6000),
                'debt_to_equity': np.random.uniform(0.1, 0.3),
                'piotroski_score': np.random.randint(3, 9)
            }
            
            return mock_data
            
        except Exception as e:
            logger.error(f"Error fetching financial data for {symbol}: {e}")
            return None
    
    def store_screened_stocks(self, stocks: List[Dict]) -> bool:
        """
        Store screened stocks in the database.
        
        Args:
            stocks (List[Dict]): List of screened stocks
            
        Returns:
            bool: True if successful
        """
        try:
            with self.db_manager.get_session() as session:
                for stock in stocks:
                    # Store or update instrument data
                    instrument = session.query(Instrument).filter(
                        Instrument.tradingsymbol == stock['symbol']
                    ).first()
                    
                    if not instrument:
                        instrument = Instrument(
                            tradingsymbol=stock['symbol'],
                            name=stock['name'],
                            exchange=stock.get('exchange', 'NSE'),
                            instrument_type='EQ',
                            segment='EQ'
                        )
                        session.add(instrument)
                        session.flush()  # Get the ID
                    
                    # Store screening data as market data
                    screening_data = stock.get('screening_data', {})
                    market_data = MarketData(
                        instrument_id=instrument.id,
                        timestamp=datetime.utcnow(),
                        last_price=screening_data.get('current_price', 0),
                        high_price=screening_data.get('high_price', 0),
                        low_price=screening_data.get('low_price', 0),
                        volume=screening_data.get('volume', 0)
                    )
                    session.add(market_data)
                
                session.commit()
                logger.info(f"Stored {len(stocks)} screened stocks in database")
                return True
                
        except Exception as e:
            logger.error(f"Error storing screened stocks: {e}")
            return False
    
    def run_daily_screening(self) -> List[Dict]:
        """
        Run daily stock screening process.
        
        Returns:
            List[Dict]: List of stocks that passed screening
        """
        try:
            logger.info("Starting daily stock screening process")
            
            # Step 1: Get stocks in market cap range
            market_cap_stocks = self.get_market_cap_range_stocks()
            logger.info(f"Found {len(market_cap_stocks)} stocks in market cap range")
            
            # Step 2: Apply screening criteria
            screened_stocks = self.apply_screening_criteria(market_cap_stocks)
            logger.info(f"{len(screened_stocks)} stocks passed screening criteria")
            
            # Step 3: Store results
            if screened_stocks:
                self.store_screened_stocks(screened_stocks)
            
            # Step 4: Log results
            logger.info(f"Daily screening completed: {len(screened_stocks)} stocks selected")
            
            return screened_stocks
            
        except Exception as e:
            logger.error(f"Error in daily screening process: {e}")
            return []


if __name__ == "__main__":
    # Test the screener
    screener = StockScreener()
    results = screener.run_daily_screening()
    print(f"Screening completed: {len(results)} stocks selected")
