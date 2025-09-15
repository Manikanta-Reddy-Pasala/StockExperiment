"""
Stock Screening Service
Implements comprehensive stock filtering based on the criteria defined in TRADING_SYSTEM_README.md
Uses FYERS API for stock data instead of yfinance to avoid rate limiting
"""
import logging
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Risk-based strategy types."""
    DEFAULT_RISK = "default_risk"
    HIGH_RISK = "high_risk"


@dataclass
class StockScreeningCriteria:
    """Stock screening criteria configuration."""
    min_price: float = 50.0  # rupees
    min_avg_volume_20d: float = 500000
    max_atr_percent: float = 8.0


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
    avg_volume_20d: float
    atr_14: float
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
        # TODO: Load stock universe from database or broker API
        # For now, return empty list - stocks should be fetched from connected broker
        return []
    
    def _initialize_fyers_connector(self, user_id: int = 1):
        """Initialize FYERS connector for API calls."""
        try:
            if self.broker_service:
                config = self.broker_service.get_broker_config('fyers', user_id)
                logger.info(f"FYERS config for user {user_id}: {config}")
                
                # Check if we have the required credentials (don't require is_connected to be True)
                if config and config.get('client_id') and config.get('access_token'):
                    logger.info("FYERS credentials found, initializing connector")
                    from .brokers.fyers_service import FyersAPIConnector
                    self.fyers_connector = FyersAPIConnector(
                        client_id=config.get('client_id'),
                        access_token=config.get('access_token')
                    )
                    logger.info("FYERS connector initialized successfully")
                    return True
                else:
                    logger.warning(f"FYERS credentials missing: client_id={bool(config.get('client_id'))}, access_token={bool(config.get('access_token'))}")
        except Exception as e:
            logger.error(f"Error initializing FYERS connector: {e}")
        return False

    def _get_technical_indicators(self, symbol: str) -> Optional[Dict[str, float]]:
        """Calculate technical indicators for a stock."""
        try:
            if not self.fyers_connector:
                return None

            # Fetch last 30 days of historical data for calculations
            range_to = datetime.now()
            range_from = range_to - timedelta(days=45) # Fetch more to ensure we get 30 trading days

            history_data = self.fyers_connector.history(
                symbol=symbol,
                resolution="D",
                range_from=range_from.strftime('%Y-%m-%d'),
                range_to=range_to.strftime('%Y-%m-%d')
            )

            if not history_data or not history_data.get('candles'):
                logger.warning(f"Could not fetch historical data for {symbol}")
                return None

            df = pd.DataFrame(history_data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if len(df) < 21: # Need at least 21 days for 20-day avg and 14-day ATR
                logger.warning(f"Not enough historical data for {symbol} ({len(df)} days)")
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)

            # Calculate 20-day average volume
            avg_volume_20d = df['volume'].rolling(window=20).mean().iloc[-1]

            # Calculate ATR(14)
            df['high_low'] = df['high'] - df['low']
            df['high_prev_close'] = abs(df['high'] - df['close'].shift(1))
            df['low_prev_close'] = abs(df['low'] - df['close'].shift(1))
            df['true_range'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
            atr_14 = df['true_range'].rolling(window=14).mean().iloc[-1]

            return {
                "avg_volume_20d": avg_volume_20d,
                "atr_14": atr_14
            }

        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return None

    def screen_stocks(self, strategy_types: List[StrategyType] = None, user_id: int = 1) -> List[StockData]:
        """Screen stocks based on criteria and strategies."""
        if strategy_types is None:
            strategy_types = [StrategyType.DEFAULT_RISK, StrategyType.HIGH_RISK]
        
        if not self._initialize_fyers_connector(user_id):
            logger.warning("FYERS connector not available, cannot screen stocks without broker connection")
            return []
        
        screened_stocks = []
        
        for symbol in self.nse_symbols:
            try:
                # Get quote data
                quote_data = self._get_stock_data(symbol)
                if not quote_data:
                    continue

                # Get technical indicators
                tech_indicators = self._get_technical_indicators(symbol)
                if not tech_indicators:
                    continue
                
                quote_data.avg_volume_20d = tech_indicators['avg_volume_20d']
                quote_data.atr_14 = tech_indicators['atr_14']

                # Apply basic screening criteria
                if not self._passes_basic_screening(quote_data):
                    continue

                # Apply strategy-specific screening
                for strategy in strategy_types:
                    if self._passes_strategy_screening(quote_data, strategy):
                        recommended_stock = self._create_recommendation(quote_data, strategy)
                        screened_stocks.append(recommended_stock)
                        break
                        
            except Exception as e:
                logger.warning(f"Error screening {symbol}: {e}")
                continue
        
        screened_stocks.sort(key=lambda x: self._get_recommendation_score(x), reverse=True)
        return screened_stocks[:20]

    def _get_stock_data(self, symbol: str) -> Optional[StockData]:
        """Get stock data from FYERS API."""
        try:
            if not self.fyers_connector:
                return None
            
            quotes_data = self.fyers_connector.quotes(symbol)
            
            if not quotes_data or 'd' not in quotes_data or not quotes_data['d']:
                logger.warning(f"No quotes data for {symbol}")
                return None
            
            quote = quotes_data['d'][0] if isinstance(quotes_data['d'], list) else quotes_data['d'].get(symbol, {})
            if not quote or 'v' not in quote:
                return None

            current_price = float(quote.get('v', {}).get('lp', 0))
            volume = int(quote.get('v', {}).get('volume', 0))

            return StockData(
                symbol=symbol,
                name=self._get_stock_name(symbol),
                current_price=current_price,
                volume=volume,
                avg_volume_20d=0, # Will be calculated later
                atr_14=0, # Will be calculated later
                market_cap=self._estimate_market_cap(symbol, current_price),
                pe_ratio=self._get_mock_pe_ratio(symbol),
                pb_ratio=self._get_mock_pb_ratio(symbol),
                debt_to_equity=self._get_mock_debt_to_equity(symbol),
                roe=self._get_mock_roe(symbol),
                sales_growth=self._get_mock_sales_growth(symbol),
                operating_profit_growth=self._get_mock_operating_profit_growth(symbol),
                yoy_sales_growth=self._get_mock_yoy_sales_growth(symbol),
                piotroski_score=self._get_mock_piotroski_score(symbol),
                recommendation="",
                strategy="",
                target_price=None,
                stop_loss=None,
                reason=""
            )
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None

    def _passes_basic_screening(self, stock: StockData) -> bool:
        """Check if stock passes basic screening criteria."""
        # Price Filter
        if stock.current_price < self.criteria.min_price:
            logger.debug(f"Failed Price Filter: {stock.symbol} price {stock.current_price} < {self.criteria.min_price}")
            return False

        # Liquidity Filter
        if stock.avg_volume_20d < self.criteria.min_avg_volume_20d:
            logger.debug(f"Failed Liquidity Filter: {stock.symbol} avg_volume {stock.avg_volume_20d} < {self.criteria.min_avg_volume_20d}")
            return False

        # Volatility Filter
        if stock.current_price > 0:
            atr_percent = (stock.atr_14 / stock.current_price) * 100
            if atr_percent > self.criteria.max_atr_percent:
                logger.debug(f"Failed Volatility Filter: {stock.symbol} ATR % {atr_percent} > {self.criteria.max_atr_percent}")
                return False

        return True

    def _estimate_market_cap(self, symbol: str, current_price: float) -> float:
        """Estimate market cap (simplified)."""
        # TODO: Fetch real market cap from broker API or financial data provider
        # For now, return a basic estimation
        estimated_cap = current_price * 1000000  # Rough estimation
        return min(estimated_cap, 2000000)  # Cap at 2M cr

    def _get_stock_name(self, symbol: str) -> str:
        """Get stock name from symbol."""
        return symbol.split(':')[1].replace('-EQ', '')

    def _get_mock_pe_ratio(self, symbol: str) -> float:
        import random
        return round(random.uniform(10, 30), 2)
    
    def _get_mock_pb_ratio(self, symbol: str) -> float:
        import random
        return round(random.uniform(1, 4), 2)
    
    def _get_mock_debt_to_equity(self, symbol: str) -> float:
        import random
        return round(random.uniform(0.1, 0.4), 2)
    
    def _get_mock_roe(self, symbol: str) -> float:
        import random
        return round(random.uniform(0.12, 0.25), 3)
    
    def _get_mock_sales_growth(self, symbol: str) -> float:
        import random
        return round(random.uniform(5, 25), 2)
    
    def _get_mock_operating_profit_growth(self, symbol: str) -> float:
        import random
        return round(random.uniform(8, 20), 2)
    
    def _get_mock_yoy_sales_growth(self, symbol: str) -> float:
        import random
        return round(random.uniform(10, 30), 2)

    def _get_mock_piotroski_score(self, symbol: str) -> float:
        import random
        return round(random.uniform(5, 9), 1)


    def _passes_strategy_screening(self, stock: StockData, strategy: StrategyType) -> bool:
        # This will be used in the next step
        return True

    def _create_recommendation(self, stock: StockData, strategy: StrategyType) -> StockData:
        # This will be used in the next step
        stock.recommendation = "BUY"
        stock.strategy = strategy.value
        return stock

    def _get_recommendation_score(self, stock: StockData) -> float:
        # This will be used in the next step
        return 1.0


def get_stock_screening_service(broker_service=None) -> StockScreeningService:
    """Get stock screening service instance."""
    return StockScreeningService(broker_service)
