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

    def _get_symbols_for_screening(self, user_id: int) -> List[str]:
        """Get symbols for screening using stock discovery service."""
        try:
            # Import here to avoid circular imports
            from .ml.stock_discovery_service import get_stock_discovery_service

            # Get stock discovery service
            discovery_service = get_stock_discovery_service()

            # Get top liquid stocks for screening
            discovered_stocks = discovery_service.get_top_liquid_stocks(user_id, count=100)

            # Extract symbols
            symbols = [stock.symbol for stock in discovered_stocks if stock.is_tradeable]

            logger.info(f"Found {len(symbols)} symbols from stock discovery service")
            return symbols

        except Exception as e:
            logger.error(f"Error getting symbols from discovery service: {e}")
            # Fallback to empty list
            return []
    
    def _initialize_broker_service(self, user_id: int = 1):
        """Initialize unified broker service for API calls."""
        try:
            # Import here to avoid circular imports
            from .unified_broker_service import get_unified_broker_service

            self.unified_broker_service = get_unified_broker_service()
            logger.info("Unified broker service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing unified broker service: {e}")
            return False

    def _get_technical_indicators(self, symbol: str, user_id: int = 1) -> Optional[Dict[str, float]]:
        """Calculate technical indicators for a stock."""
        try:
            if not hasattr(self, 'unified_broker_service') or not self.unified_broker_service:
                return None

            # Fetch last 30 days of historical data for calculations
            range_to = datetime.now()
            range_from = range_to - timedelta(days=45) # Fetch more to ensure we get 30 trading days

            history_result = self.unified_broker_service.get_historical_data(
                user_id, symbol, resolution="D",
                range_from=range_from.strftime('%Y-%m-%d')
            )

            if not history_result.get('success') or not history_result.get('data', {}).get('candles'):
                logger.warning(f"Could not fetch historical data for {symbol}")
                return None

            history_data = history_result.get('data', {})
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

        if not self._initialize_broker_service(user_id):
            logger.warning("Unified broker service not available, cannot screen stocks without broker connection")
            return []

        # Get symbols from stock discovery service
        symbols_to_screen = self._get_symbols_for_screening(user_id)
        if not symbols_to_screen:
            logger.warning("No symbols found for screening")
            return []

        logger.info(f"Screening {len(symbols_to_screen)} symbols")
        screened_stocks = []

        for symbol in symbols_to_screen:
            try:
                # Get quote data
                quote_data = self._get_stock_data(symbol, user_id)
                if not quote_data:
                    continue

                # Get technical indicators
                tech_indicators = self._get_technical_indicators(symbol, user_id)
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

    def _get_stock_data(self, symbol: str, user_id: int = 1) -> Optional[StockData]:
        """Get stock data from unified broker service."""
        try:
            if not hasattr(self, 'unified_broker_service') or not self.unified_broker_service:
                return None

            quotes_result = self.unified_broker_service.get_quotes(user_id, [symbol])

            if not quotes_result.get('success') or not quotes_result.get('data'):
                logger.warning(f"No quotes data for {symbol}")
                return None

            quotes_data = quotes_result.get('data', {})
            if symbol not in quotes_data:
                logger.warning(f"Symbol {symbol} not found in quotes response")
                return None

            quote = quotes_data[symbol]
            current_price = float(quote.get('lp', quote.get('ltp', 0)))
            volume = int(quote.get('volume', quote.get('vol', 0)))

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
