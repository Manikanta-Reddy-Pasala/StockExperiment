"""
Suggested Stocks Saga Pattern
Implements a comprehensive saga pattern for the suggested stocks pipeline with step-by-step updates and additional information.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)

# ETFs and index funds unsuitable for EMA swing trading (near-zero volatility or index-tracking)
ETF_EXCLUSION_SYMBOLS = {
    'LIQUIDBEES', 'NIFTYBEES', 'BANKBEES', 'MON100', 'GOLDBEES',
    'SILVERBEES', 'JUNIORBEES', 'ITBEES', 'PSUBNKBEES', 'SETFNIF50',
    'SETFNIFBK', 'NETFIT', 'HABOREES', 'HNGSNGBEES', 'SHARIABEES',
    'DIVOPPBEES', 'INFRABEES', 'CPSEETF', 'CONSUMBEES', 'HEALTHY',
    'MOM100', 'MOM50', 'PHARMABEES', 'AUTOBEES', 'COMMOETF',
}

# Stop-loss cooldown: 30 days after any SL hit, exclude if 2+ SL hits in 90 days
STOP_LOSS_COOLDOWN_DAYS = 30
STOP_LOSS_MAX_HITS_PERIOD_DAYS = 90
STOP_LOSS_MAX_HITS = 2


class SagaStepStatus(Enum):
    """Status of each saga step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SagaStatus(Enum):
    """Overall saga status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SagaStep:
    """Individual saga step with status and metadata."""
    step_id: str
    name: str
    description: str
    status: SagaStepStatus = SagaStepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    input_count: int = 0
    output_count: int = 0
    filtered_count: int = 0
    rejected_count: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuggestedStocksSaga:
    """Saga pattern for suggested stocks pipeline."""
    saga_id: str
    user_id: int
    strategies: List[str]
    limit: int
    search_query: Optional[str] = None
    sort_by: Optional[str] = None
    sort_order: str = 'desc'
    sector: Optional[str] = None
    model_type: str = 'traditional'
    status: SagaStatus = SagaStatus.RUNNING
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    steps: List[SagaStep] = field(default_factory=list)
    final_results: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def add_step(self, step: SagaStep) -> None:
        """Add a step to the saga."""
        self.steps.append(step)
    
    def get_step(self, step_id: str) -> Optional[SagaStep]:
        """Get a step by ID."""
        return next((step for step in self.steps if step.step_id == step_id), None)
    
    def update_step_status(self, step_id: str, status: SagaStepStatus, 
                          error_message: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update step status and metadata."""
        step = self.get_step(step_id)
        if step:
            step.status = status
            if status == SagaStepStatus.IN_PROGRESS:
                step.start_time = datetime.now()
            elif status in [SagaStepStatus.COMPLETED, SagaStepStatus.FAILED, SagaStepStatus.SKIPPED]:
                step.end_time = datetime.now()
                if step.start_time:
                    step.duration_seconds = (step.end_time - step.start_time).total_seconds()
            
            if error_message:
                step.error_message = error_message
                self.errors.append(f"Step {step_id}: {error_message}")
            
            if metadata:
                step.metadata.update(metadata)
    
    def complete_saga(self, final_results: List[Dict[str, Any]], 
                     summary: Dict[str, Any]) -> None:
        """Complete the saga with final results."""
        self.status = SagaStatus.COMPLETED
        self.end_time = datetime.now()
        self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.final_results = final_results
        self.summary = summary
    
    def fail_saga(self, error_message: str) -> None:
        """Fail the saga with error message."""
        self.status = SagaStatus.FAILED
        self.end_time = datetime.now()
        self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.errors.append(error_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert saga to dictionary for serialization."""
        return {
            'saga_id': self.saga_id,
            'user_id': self.user_id,
            'strategies': self.strategies,
            'limit': self.limit,
            'search_query': self.search_query,
            'sort_by': self.sort_by,
            'sort_order': self.sort_order,
            'sector': self.sector,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration_seconds': self.total_duration_seconds,
            'steps': [
                {
                    'step_id': step.step_id,
                    'name': step.name,
                    'description': step.description,
                    'status': step.status.value,
                    'start_time': step.start_time.isoformat() if step.start_time else None,
                    'end_time': step.end_time.isoformat() if step.end_time else None,
                    'duration_seconds': step.duration_seconds,
                    'input_count': step.input_count,
                    'output_count': step.output_count,
                    'filtered_count': step.filtered_count,
                    'rejected_count': step.rejected_count,
                    'error_message': step.error_message,
                    'metadata': step.metadata,
                    'additional_info': step.additional_info
                }
                for step in self.steps
            ],
            'final_results': self.final_results,
            'summary': self.summary,
            'errors': self.errors
        }


class SuggestedStocksSagaOrchestrator:
    """Orchestrator for the suggested stocks saga pattern."""
    
    def __init__(self):
        self.fyers_service = None
        self.stock_filters_config = None
    
    def initialize_services(self, user_id: int) -> None:
        """Initialize required services."""
        try:
            from ..brokers.fyers_service import get_fyers_service
            import yaml
            import os

            self.fyers_service = get_fyers_service()

            # Load stock filters configuration
            config_path = os.path.join(os.path.dirname(__file__), '../../../config/stock_filters.yaml')
            with open(config_path, 'r') as f:
                self.stock_filters_config = yaml.safe_load(f)

            logger.info("Services and stock filters configuration initialized successfully for saga orchestrator")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

    def detect_market_regime(self) -> Dict[str, Any]:
        """
        Detect market regime using Nifty 50 EMA analysis.
        Uses historical_data table for NSE:NIFTY50-INDEX or falls back to
        computing from the broad market (% of stocks in bullish power zone).

        Returns:
            {
                'regime': 'bullish' | 'bearish' | 'neutral',
                'nifty_above_ema21': bool,
                'nifty_ema8_above_ema21': bool,
                'bullish_pct': float,  # % of stocks in bullish power zone
                'method': str
            }
        """
        try:
            from src.models.database import get_database_manager
            from sqlalchemy import text

            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                # Method 1: Try Nifty 50 from market_benchmarks table
                nifty_query = text("""
                    SELECT close FROM market_benchmarks
                    WHERE benchmark = 'NIFTY50'
                    ORDER BY date DESC LIMIT 30
                """)
                nifty_rows = session.execute(nifty_query).fetchall()

                if len(nifty_rows) >= 21:
                    closes = [float(r[0]) for r in reversed(nifty_rows)]
                    # Calculate 8 and 21 EMA
                    import pandas as pd
                    s = pd.Series(closes)
                    ema_8 = float(s.ewm(span=8, adjust=False).mean().iloc[-1])
                    ema_21 = float(s.ewm(span=21, adjust=False).mean().iloc[-1])
                    nifty_price = closes[-1]

                    nifty_bullish = nifty_price > ema_8 > ema_21
                    nifty_bearish = nifty_price < ema_8 < ema_21

                    regime = 'bullish' if nifty_bullish else ('bearish' if nifty_bearish else 'neutral')
                    logger.info(f"Market regime (Nifty 50): {regime} | Price={nifty_price:.0f} EMA8={ema_8:.0f} EMA21={ema_21:.0f}")
                    return {
                        'regime': regime,
                        'nifty_above_ema21': nifty_price > ema_21,
                        'nifty_ema8_above_ema21': ema_8 > ema_21,
                        'bullish_pct': 0.0,
                        'method': 'nifty50_ema'
                    }

                # Method 2: Fallback — compute from broad market
                # Count % of stocks in bullish vs bearish power zone from technical_indicators
                regime_query = text("""
                    SELECT
                        COUNT(*) FILTER (WHERE ti.ema_8 > ti.ema_21) as bullish_count,
                        COUNT(*) FILTER (WHERE ti.ema_8 < ti.ema_21) as bearish_count,
                        COUNT(*) as total
                    FROM technical_indicators ti
                    INNER JOIN (
                        SELECT symbol, MAX(date) as max_date
                        FROM technical_indicators
                        WHERE date >= CURRENT_DATE - 5
                        GROUP BY symbol
                    ) latest ON ti.symbol = latest.symbol AND ti.date = latest.max_date
                    WHERE ti.ema_8 IS NOT NULL AND ti.ema_21 IS NOT NULL
                """)
                row = session.execute(regime_query).fetchone()

                if row and row[2] > 0:
                    bullish_count = row[0]
                    bearish_count = row[1]
                    total = row[2]
                    bullish_pct = (bullish_count / total) * 100

                    if bullish_pct >= 55:
                        regime = 'bullish'
                    elif bullish_pct <= 35:
                        regime = 'bearish'
                    else:
                        regime = 'neutral'

                    logger.info(f"Market regime (breadth): {regime} | {bullish_pct:.1f}% bullish ({bullish_count}/{total})")
                    return {
                        'regime': regime,
                        'nifty_above_ema21': bullish_pct > 50,
                        'nifty_ema8_above_ema21': bullish_pct > 50,
                        'bullish_pct': bullish_pct,
                        'method': 'market_breadth'
                    }

                logger.warning("Could not determine market regime — defaulting to neutral")
                return {'regime': 'neutral', 'nifty_above_ema21': True, 'nifty_ema8_above_ema21': True, 'bullish_pct': 50.0, 'method': 'default'}

        except Exception as e:
            logger.error(f"Market regime detection failed: {e}")
            return {'regime': 'neutral', 'nifty_above_ema21': True, 'nifty_ema8_above_ema21': True, 'bullish_pct': 50.0, 'method': 'error'}

    def execute_suggested_stocks_saga(self, user_id: int, strategies: List[str] = None,
                                   limit: int = 5, search: str = None, sort_by: str = None,
                                   sort_order: str = 'desc', sector: str = None,
                                   model_type: str = 'hybrid') -> Dict[str, Any]:
        """
        Execute the complete suggested stocks saga with step-by-step updates.

        Args:
            user_id: User ID for the request
            strategies: List of strategies to apply
            limit: Maximum number of stocks to return
            search: Search query string
            sort_by: Field to sort results by
            sort_order: Sort order ('asc' or 'desc')
            sector: Filter by specific sector
            model_type: Model type (default: 'hybrid' - pure technical analysis)

        Returns:
            Complete saga results with step-by-step information
        """
        # Initialize saga
        saga_id = f"suggested_stocks_{model_type}_{user_id}_{int(datetime.now().timestamp())}"
        saga = SuggestedStocksSaga(
            saga_id=saga_id,
            user_id=user_id,
            strategies=strategies or ['unified'],  # Single unified EMA strategy
            limit=limit,
            search_query=search,
            sort_by=sort_by,
            sort_order=sort_order,
            sector=sector
        )

        # Store model_type in saga metadata
        saga.model_type = model_type

        try:
            # Initialize services
            self.initialize_services(user_id)

            # Detect market regime BEFORE stock selection
            market_regime = self.detect_market_regime()
            saga.market_regime = market_regime
            print(f"\n🌍 Market Regime: {market_regime['regime'].upper()} (method: {market_regime['method']})")

            # Step 1: Stock Discovery
            saga = self._execute_step1_stock_discovery(saga)
            if saga.status == SagaStatus.FAILED:
                return saga.to_dict()
            
            # Step 2: Database Filtering
            saga = self._execute_step2_database_filtering(saga)
            if saga.status == SagaStatus.FAILED:
                return saga.to_dict()
            
            # Step 3: Strategy Application
            saga = self._execute_step3_strategy_application(saga)
            if saga.status == SagaStatus.FAILED:
                return saga.to_dict()
            
            # Step 4: Search and Sort
            saga = self._execute_step4_search_and_sort(saga)
            if saga.status == SagaStatus.FAILED:
                return saga.to_dict()
            
            # Step 5: Final Selection
            saga = self._execute_step5_final_selection(saga)
            if saga.status == SagaStatus.FAILED:
                return saga.to_dict()

            # Step 6: ML Prediction
            saga = self._execute_step6_ml_prediction(saga)
            if saga.status == SagaStatus.FAILED:
                return saga.to_dict()

            # Step 7: Daily Snapshot Save
            saga = self._execute_step7_daily_snapshot(saga)
            if saga.status == SagaStatus.FAILED:
                return saga.to_dict()

            # Complete saga
            saga.complete_saga(saga.final_results, self._generate_saga_summary(saga))
            
            logger.info(f"Saga {saga_id} completed successfully")
            return saga.to_dict()
            
        except Exception as e:
            error_msg = f"Saga execution failed: {str(e)}"
            logger.error(error_msg)
            saga.fail_saga(error_msg)
            return saga.to_dict()
    
    def _execute_step1_stock_discovery(self, saga: SuggestedStocksSaga) -> SuggestedStocksSaga:
        """Step 1: Discover tradeable stocks from broker API."""
        step = SagaStep(
            step_id="step1_discovery",
            name="Stock Discovery",
            description="Discover tradeable stocks from broker API using search terms"
        )
        saga.add_step(step)
        saga.update_step_status("step1_discovery", SagaStepStatus.IN_PROGRESS)
        
        try:
            print(f"🔍 STAGE 1: Discovering tradeable stocks...")
            
            # Search terms for stock discovery
            search_terms = [
                "NIFTY", "SENSEX", "BANKNIFTY", "NIFTYNXT50", "FINNIFTY",
                "BANK", "IT", "PHARMA", "AUTO", "FMCG", "METAL", "INFRA", "ENERGY",
                "FINANCE", "TECH", "HEALTHCARE", "CONSUMER", "COMMODITY",
                "LTD", "LIMITED", "CORP", "INC", "INDUSTRIES"
            ]
            
            all_symbols = set()
            discovered_stocks = []
            successful_searches = 0
            failed_searches = 0
            
            for term in search_terms:
                try:
                    search_result = self.fyers_service.search(saga.user_id, term, "NSE")
                    
                    if search_result.get('status') == 'success':
                        symbols = search_result.get('data', [])
                        for symbol_data in symbols:
                            if symbol_data.get('symbol') not in all_symbols:
                                all_symbols.add(symbol_data.get('symbol'))
                                discovered_stocks.append(symbol_data)
                        successful_searches += 1
                    else:
                        failed_searches += 1
                        
                except Exception as e:
                    logger.warning(f"Search failed for term '{term}': {e}")
                    failed_searches += 1
            
            step.input_count = len(search_terms)
            step.output_count = len(discovered_stocks)
            step.metadata = {
                'successful_searches': successful_searches,
                'failed_searches': failed_searches,
                'unique_symbols': len(all_symbols),
                'search_terms_used': search_terms
            }
            step.results = discovered_stocks
            step.additional_info = {
                'discovery_method': 'broker_api_search',
                'exchange': 'NSE',
                'search_coverage': f"{successful_searches}/{len(search_terms)} terms successful"
            }
            
            saga.update_step_status("step1_discovery", SagaStepStatus.COMPLETED, 
                                  metadata=step.metadata)
            
            print(f"   ✅ Discovered {len(discovered_stocks)} unique stocks from {successful_searches} successful searches")
            
            return saga
            
        except Exception as e:
            error_msg = f"Stock discovery failed: {str(e)}"
            saga.update_step_status("step1_discovery", SagaStepStatus.FAILED, error_msg)
            return saga
    
    def _execute_step2_database_filtering(self, saga: SuggestedStocksSaga) -> SuggestedStocksSaga:
        """Step 2: Apply database-driven filtering pipeline."""
        step = SagaStep(
            step_id="step2_filtering",
            name="Database Filtering",
            description="Apply comprehensive filtering pipeline using database screening"
        )
        saga.add_step(step)
        saga.update_step_status("step2_filtering", SagaStepStatus.IN_PROGRESS)
        
        try:
            print(f"🎯 STAGE 2: Applying database filtering pipeline...")
            
            # Get stocks from database using the database manager
            from ...models.database import get_database_manager
            from ...models.stock_models import Stock
            
            db_manager = get_database_manager()
            
            with db_manager.get_session() as session:
                # Get all available stocks from database
                available_stocks = session.query(Stock).filter(
                    Stock.is_active == True,
                    Stock.current_price > 0
                ).all()
                
                if not available_stocks:
                    raise Exception("No active stocks found in database")
                
                # Convert to dictionary format for filtering with all calculated fields
                stock_data_list = []
                for stock in available_stocks:
                    stock_data = {
                        'symbol': stock.symbol,
                        'name': stock.name,
                        'current_price': float(stock.current_price),
                        'market_cap': float(stock.market_cap) if stock.market_cap else 0,
                        'pe_ratio': float(stock.pe_ratio) if stock.pe_ratio else None,
                        'pb_ratio': float(stock.pb_ratio) if stock.pb_ratio else None,
                        'roe': float(stock.roe) if stock.roe else None,
                        'eps': float(stock.eps) if stock.eps else None,
                        'book_value': float(stock.book_value) if stock.book_value else None,
                        'beta': float(stock.beta) if stock.beta else None,
                        'peg_ratio': float(stock.peg_ratio) if stock.peg_ratio else None,
                        'roa': float(stock.roa) if stock.roa else None,
                        'debt_to_equity': float(stock.debt_to_equity) if stock.debt_to_equity else None,
                        'current_ratio': float(stock.current_ratio) if stock.current_ratio else None,
                        'quick_ratio': float(stock.quick_ratio) if stock.quick_ratio else None,
                        'revenue_growth': float(stock.revenue_growth) if stock.revenue_growth else None,
                        'earnings_growth': float(stock.earnings_growth) if stock.earnings_growth else None,
                        'operating_margin': float(stock.operating_margin) if stock.operating_margin else None,
                        'net_margin': float(stock.net_margin) if stock.net_margin else None,
                        'profit_margin': float(stock.profit_margin) if stock.profit_margin else None,
                        'dividend_yield': float(stock.dividend_yield) if stock.dividend_yield else None,
                        'volume': int(stock.volume) if stock.volume else 0,
                        'sector': stock.sector,
                        'market_cap_category': stock.market_cap_category,
                        # 8-21 EMA Strategy Indicators
                        'ema_8': float(stock.ema_8) if stock.ema_8 else None,
                        'ema_21': float(stock.ema_21) if stock.ema_21 else None,
                        'demarker': float(stock.demarker) if stock.demarker else None,
                        'buy_signal': bool(stock.buy_signal) if stock.buy_signal is not None else False,
                        'sell_signal': bool(stock.sell_signal) if stock.sell_signal is not None else False,
                        # Derived indicators for strategy filtering
                        'is_bullish': bool(stock.ema_8 and stock.ema_21 and stock.current_price and
                                          stock.current_price > stock.ema_8 > stock.ema_21),
                        'signal_quality': 'high' if (stock.demarker and stock.demarker < 0.30) else 'medium' if (stock.demarker and stock.demarker < 0.70) else 'low',
                        'ema_strategy_score': self._compute_db_ema_score(stock)
                    }
                    stock_data_list.append(stock_data)
                
                # Apply filtering criteria from stock_filters.yaml
                filtered_stocks = []
                stage1_filters = self.stock_filters_config.get('stage_1_filters', {})
                tradeability = stage1_filters.get('tradeability', {})
                
                min_price = tradeability.get('minimum_price', 5.0)
                max_price = tradeability.get('maximum_price', 10000.0)
                min_volume = tradeability.get('fallback_minimum_volume', 50000)
                min_turnover = tradeability.get('minimum_daily_turnover_inr', 50000000)
                
                for stock_data in stock_data_list:
                    current_price = stock_data['current_price']
                    volume = stock_data['volume']
                    market_cap = stock_data['market_cap']

                    # Exclude ETFs - not suitable for EMA swing trading
                    stock_name = (stock_data.get('name') or '').upper()
                    raw_symbol = (stock_data.get('symbol') or '').upper()
                    # Extract clean symbol (e.g. "NSE:LIQUIDBEES-EQ" -> "LIQUIDBEES")
                    clean_symbol = raw_symbol.replace('NSE:', '').replace('BSE:', '').replace('-EQ', '').strip()
                    if ('ETF' in stock_name or 'ETF' in clean_symbol or
                        'BEES' in clean_symbol or clean_symbol in ETF_EXCLUSION_SYMBOLS):
                        continue

                    # Calculate daily turnover (price * volume)
                    daily_turnover = current_price * volume

                    # Apply stage 1 filtering criteria from config
                    if (min_price <= current_price <= max_price and
                        volume >= min_volume and
                        daily_turnover >= min_turnover and
                        market_cap > 0):
                        filtered_stocks.append(stock_data)
                
                step.input_count = len(stock_data_list)
                step.output_count = len(filtered_stocks)
                step.filtered_count = len(filtered_stocks)
                step.rejected_count = len(stock_data_list) - len(filtered_stocks)
                step.metadata = {
                    'total_available': len(stock_data_list),
                    'filtered_count': len(filtered_stocks),
                    'rejected_count': len(stock_data_list) - len(filtered_stocks),
                    'filtering_criteria': 'basic_price_volume_market_cap'
                }
                step.results = filtered_stocks
                step.additional_info = {
                    'filtering_criteria': 'basic_database_filtering',
                    'criteria_applied': ['min_price_10', 'min_volume_1000', 'min_market_cap_100'],
                    'database_source': 'stocks_table'
                }
                
                saga.update_step_status("step2_filtering", SagaStepStatus.COMPLETED, 
                                      metadata=step.metadata)
                
                print(f"   ✅ Filtered {len(filtered_stocks)} stocks from {len(stock_data_list)} available")
                
                return saga
            
        except Exception as e:
            error_msg = f"Database filtering failed: {str(e)}"
            saga.update_step_status("step2_filtering", SagaStepStatus.FAILED, error_msg)
            return saga
    
    def _execute_step3_strategy_application(self, saga: SuggestedStocksSaga) -> SuggestedStocksSaga:
        """Step 3: Apply strategy-specific business logic."""
        step = SagaStep(
            step_id="step3_strategy",
            name="Strategy Application",
            description="Apply strategy-specific business logic and scoring"
        )
        saga.add_step(step)
        saga.update_step_status("step3_strategy", SagaStepStatus.IN_PROGRESS)
        
        try:
            print(f"🎯 STAGE 3: Applying strategy-specific business logic...")
            
            # Get filtered stocks from previous step
            previous_step = saga.get_step("step2_filtering")
            if not previous_step or not previous_step.results:
                raise Exception("No filtered stocks available from previous step")
            
            filtered_stocks = previous_step.results
            suggested_stocks = []
            strategy_counts = {strategy: 0 for strategy in saga.strategies}
            
            # Get stop-loss blacklist to avoid re-recommending repeated losers
            stop_loss_blacklist = self._get_stop_loss_blacklist()
            if stop_loss_blacklist:
                logger.info(f"Stop-loss blacklist: {len(stop_loss_blacklist)} symbols excluded")

            # Apply strategy logic to each stock (ONE entry per stock - no duplicates)
            sl_rejected = 0
            for stock_data in filtered_stocks:
                try:
                    # Check stop-loss blacklist
                    raw_sym = (stock_data.get('symbol') or '').upper()
                    clean_sym = raw_sym.replace('NSE:', '').replace('BSE:', '').replace('-EQ', '').strip()
                    if clean_sym in stop_loss_blacklist or raw_sym in stop_loss_blacklist:
                        sl_rejected += 1
                        continue

                    # Apply unified strategy only - no need to loop through multiple strategies
                    strategy = 'unified'  # Always use unified strategy
                    strategy_result = self._apply_strategy_logic(stock_data, strategy, saga.model_type,
                                                                   market_regime=getattr(saga, 'market_regime', None))
                    if strategy_result:
                        suggested_stocks.append(strategy_result)
                        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

                except Exception as e:
                    logger.warning(f"Failed to apply strategy to stock {stock_data.get('symbol', 'Unknown')}: {e}")
                    continue

            if sl_rejected:
                logger.info(f"Rejected {sl_rejected} stocks due to stop-loss history")
            
            step.input_count = len(filtered_stocks)
            step.output_count = len(suggested_stocks)
            step.metadata = {
                'strategies_applied': saga.strategies,
                'strategy_counts': strategy_counts,
                'success_rate': len(suggested_stocks) / len(filtered_stocks) if filtered_stocks else 0,
                'avg_suggestions_per_stock': len(suggested_stocks) / len(filtered_stocks) if filtered_stocks else 0
            }
            step.results = suggested_stocks
            step.additional_info = {
                'strategy_logic': 'swing_trading_focused',
                'scoring_method': 'multi_factor_analysis',
                'risk_assessment': 'strategy_specific',
                'business_domain': 'swing_trading_2_week_hold'
            }
            
            saga.update_step_status("step3_strategy", SagaStepStatus.COMPLETED, 
                                  metadata=step.metadata)
            
            print(f"   ✅ Applied {len(saga.strategies)} strategies to {len(filtered_stocks)} stocks, generated {len(suggested_stocks)} suggestions")
            
            return saga
            
        except Exception as e:
            error_msg = f"Strategy application failed: {str(e)}"
            saga.update_step_status("step3_strategy", SagaStepStatus.FAILED, error_msg)
            return saga
    
    def _execute_step4_search_and_sort(self, saga: SuggestedStocksSaga) -> SuggestedStocksSaga:
        """Step 4: Apply search and sort operations."""
        step = SagaStep(
            step_id="step4_search_sort",
            name="Search and Sort",
            description="Apply search filtering and sorting operations"
        )
        saga.add_step(step)
        saga.update_step_status("step4_search_sort", SagaStepStatus.IN_PROGRESS)
        
        try:
            print(f"🎯 STAGE 4: Applying search and sort operations...")
            
            # Get suggested stocks from previous step
            previous_step = saga.get_step("step3_strategy")
            if not previous_step or not previous_step.results:
                raise Exception("No suggested stocks available from previous step")
            
            suggested_stocks = previous_step.results.copy()
            
            # Apply search filter if provided
            if saga.search_query:
                search_query = saga.search_query.lower()
                suggested_stocks = [
                    stock for stock in suggested_stocks
                    if (search_query in stock.get('symbol', '').lower() or
                        search_query in stock.get('name', '').lower())
                ]
            
            # Apply sector filter if provided
            if saga.sector:
                sector_filter = saga.sector.lower()
                suggested_stocks = [
                    stock for stock in suggested_stocks
                    if sector_filter in stock.get('sector', '').lower()
                ]
            
            # Apply sorting if provided
            if saga.sort_by:
                reverse = saga.sort_order == 'desc'
                try:
                    suggested_stocks.sort(key=lambda x: x.get(saga.sort_by, 0), reverse=reverse)
                except Exception as e:
                    logger.warning(f"Sorting failed: {e}")
            
            step.input_count = len(previous_step.results)
            step.output_count = len(suggested_stocks)
            step.metadata = {
                'search_applied': bool(saga.search_query),
                'sort_applied': bool(saga.sort_by),
                'sector_filter_applied': bool(saga.sector),
                'search_query': saga.search_query,
                'sort_by': saga.sort_by,
                'sort_order': saga.sort_order,
                'sector': saga.sector
            }
            step.results = suggested_stocks
            step.additional_info = {
                'search_method': 'text_matching',
                'sort_method': 'field_based',
                'filtering_applied': ['search', 'sector', 'sort']
            }
            
            saga.update_step_status("step4_search_sort", SagaStepStatus.COMPLETED, 
                                  metadata=step.metadata)
            
            print(f"   ✅ Applied search/sort operations, {len(suggested_stocks)} stocks remaining")
            
            return saga
            
        except Exception as e:
            error_msg = f"Search and sort failed: {str(e)}"
            saga.update_step_status("step4_search_sort", SagaStepStatus.FAILED, error_msg)
            return saga
    
    def _execute_step5_final_selection(self, saga: SuggestedStocksSaga) -> SuggestedStocksSaga:
        """Step 5: Final selection - pass all filtered stocks to ML without limit."""
        step = SagaStep(
            step_id="step5_final_selection",
            name="Final Selection",
            description="Prepare all filtered stocks for ML scoring (no limit)"
        )
        saga.add_step(step)
        saga.update_step_status("step5_final_selection", SagaStepStatus.IN_PROGRESS)

        try:
            print(f"🎯 STAGE 5: Preparing stocks for ML scoring...")

            # Get processed stocks from previous step
            previous_step = saga.get_step("step4_search_sort")
            if not previous_step or not previous_step.results:
                raise Exception("No processed stocks available from previous step")

            processed_stocks = previous_step.results

            # NO LIMIT - Pass all filtered stocks to ML for scoring
            # The limit will be applied AFTER ML scoring (top N by ML score)
            final_stocks = processed_stocks

            # Add metadata (but not rank yet - that comes after ML scoring)
            for stock in final_stocks:
                stock['selection_timestamp'] = datetime.now().isoformat()
                stock['saga_id'] = saga.saga_id

            step.input_count = len(processed_stocks)
            step.output_count = len(final_stocks)
            step.metadata = {
                'limit_applied': 'none',  # No limit at this stage
                'final_count': len(final_stocks),
                'selection_criteria': 'all_filtered_stocks'
            }
            step.results = final_stocks
            step.additional_info = {
                'selection_method': 'pass_all_to_ml',
                'limit_strategy': 'apply_after_ml_scoring',
                'ranking_criteria': 'to_be_applied_by_ml_score'
            }

            saga.update_step_status("step5_final_selection", SagaStepStatus.COMPLETED,
                                  metadata=step.metadata)

            # Set final results (all stocks - will be limited after ML scoring)
            saga.final_results = final_stocks

            print(f"   ✅ Prepared {len(final_stocks)} stocks for ML scoring (no limit applied)")

            return saga

        except Exception as e:
            error_msg = f"Final selection failed: {str(e)}"
            saga.update_step_status("step5_final_selection", SagaStepStatus.FAILED, error_msg)
            return saga
    
    def _apply_strategy_logic(self, stock_data: Dict[str, Any], strategy: str, model_type: str = 'ema',
                              market_regime: Dict = None) -> Optional[Dict[str, Any]]:
        """
        Apply 8-21 EMA swing trading strategy logic to a stock.

        Bullish regime: Price > 8 EMA > 21 EMA (long entries)
        Bearish regime: Price < 8 EMA < 21 EMA (short entries)
        """
        try:
            current_price = stock_data.get('current_price', 0.0)
            market_cap = stock_data.get('market_cap', 0.0)
            pe_ratio = stock_data.get('pe_ratio')
            volume = stock_data.get('volume', 0)

            # Get EMA strategy indicators
            ema_8 = stock_data.get('ema_8', 0.0)
            ema_21 = stock_data.get('ema_21', 0.0)
            demarker = stock_data.get('demarker', 0.5)
            is_bullish = stock_data.get('is_bullish', False)
            buy_signal = stock_data.get('buy_signal', False)
            sell_signal = stock_data.get('sell_signal', False)
            signal_quality = stock_data.get('signal_quality', 'none')
            ema_strategy_score = stock_data.get('ema_strategy_score', 0.0)

            regime = (market_regime or {}).get('regime', 'neutral')

            # ========================================
            # 8-21 EMA STRATEGY: REGIME-AWARE FILTER
            # ========================================
            # Bearish power zone: Price < 8 EMA < 21 EMA
            is_bearish_pz = current_price < ema_8 < ema_21 if (ema_8 and ema_21 and current_price) else False

            if regime == 'bearish' and is_bearish_pz and sell_signal:
                # BEARISH REGIME: accept short candidates
                signal_quality = stock_data.get('signal_quality', 'none')
                # For shorts, accept medium+ quality from short_quality
                short_quality = stock_data.get('short_quality', signal_quality)
                if short_quality not in ('high', 'medium'):
                    # In bearish regime, also accept bearish power zone stocks with medium+ demarker
                    if 0.30 <= demarker <= 0.65:
                        short_quality = 'medium'
                    else:
                        return None
                signal_quality = short_quality
                # signal_direction will be set to SHORT below
            elif is_bullish and buy_signal:
                # BULLISH or NEUTRAL: keep existing long logic
                pass
            else:
                return None  # Skip stocks not matching any power zone

            # Get filtering thresholds from configuration
            stage2_filters = self.stock_filters_config.get('stage_2_filters', {})
            filtering_thresholds = stage2_filters.get('filtering_thresholds', {})
            fundamental_ratios = stage2_filters.get('fundamental_ratios', {})

            # Get market cap categories from config
            market_cap_categories = self.stock_filters_config.get('stage_1_filters', {}).get('market_cap_categories', {})
            large_cap_min = market_cap_categories.get('large_cap', {}).get('minimum', 20000)
            mid_cap_min = market_cap_categories.get('mid_cap', {}).get('minimum', 5000)


            # Unified 8-21 EMA strategy filtering (ALL market caps)
            # Apply balanced filtering criteria for all stocks

            # Price range: ₹50 - ₹10,000 (exclude penny stocks and overly expensive stocks)
            if current_price < 50 or current_price > 10000:
                return None

            # Minimum volume for liquidity
            if volume < 50000:
                return None

            # Market cap: minimum ₹500 Cr (exclude micro-cap stocks)
            if market_cap < 500:
                return None

            # PE ratio: if available, should be reasonable (5-100)
            if pe_ratio and (pe_ratio < 5 or pe_ratio > 100):
                return None

            # EMA strategy score threshold (backtest-optimized)
            score = ema_strategy_score / 100.0  # Convert to 0-1 range
            min_score = 0.25  # Require score >= 25/100 (backtest optimal: filters toxic setups)
            if score < min_score:
                return None

            # Signal quality: Accept ONLY high and medium quality signals
            if signal_quality not in ('high', 'medium'):
                return None

            # Determine direction: LONG or SHORT
            is_short = regime == 'bearish' and is_bearish_pz

            # Create suggested stock with 8-21 EMA strategy fields
            if is_short:
                # SHORT entry: Fibonacci support targets (downside), stop above 21 EMA
                target_1 = current_price * 0.95  # 5% drop
                target_2 = current_price * 0.90  # 10% drop
                target_3 = current_price * 0.85  # 15% drop
                stop_loss = ema_21 * 1.02  # 2% above 21 EMA
                recommendation = 'SHORT'
                reason_text = f"Bearish Power Zone: Price < 8EMA < 21EMA | DeMarker={demarker:.2f}"
            else:
                # LONG entry (existing logic)
                target_1 = stock_data.get('fib_target_127', current_price * 1.05)
                target_2 = stock_data.get('fib_target_162', current_price * 1.10)
                target_3 = stock_data.get('fib_target_200', current_price * 1.15)
                stop_loss = stock_data.get('suggested_stop', ema_21 * 0.98)
                recommendation = 'BUY'
                reason_text = self._generate_ema_reason(stock_data, strategy, score, signal_quality)

            suggested_stock = {
                'symbol': stock_data.get('symbol', ''),
                'name': stock_data.get('name', ''),
                'current_price': current_price,
                'market_cap': market_cap,
                'pe_ratio': pe_ratio,
                'pb_ratio': stock_data.get('pb_ratio'),
                'roe': stock_data.get('roe'),
                'eps': stock_data.get('eps'),
                'book_value': stock_data.get('book_value'),
                'beta': stock_data.get('beta'),
                'peg_ratio': stock_data.get('peg_ratio'),
                'roa': stock_data.get('roa'),
                'debt_to_equity': stock_data.get('debt_to_equity'),
                'current_ratio': stock_data.get('current_ratio'),
                'quick_ratio': stock_data.get('quick_ratio'),
                'revenue_growth': stock_data.get('revenue_growth'),
                'earnings_growth': stock_data.get('earnings_growth'),
                'operating_margin': stock_data.get('operating_margin'),
                'net_margin': stock_data.get('net_margin'),
                'profit_margin': stock_data.get('profit_margin'),
                'dividend_yield': stock_data.get('dividend_yield'),
                'sector': stock_data.get('sector', ''),
                'market_cap_category': stock_data.get('market_cap_category', ''),
                'strategy': strategy,
                'selection_score': score,

                # 8-21 EMA Strategy fields
                'ema_8': ema_8,
                'ema_21': ema_21,
                'ema_trend_score': ema_strategy_score,
                'demarker': demarker,
                'signal_quality': signal_quality,

                # Targets and stops
                'fib_target_1': target_1,
                'fib_target_2': target_2,
                'fib_target_3': target_3,
                'target_price': target_2,
                'stop_loss': stop_loss,

                'buy_signal': buy_signal and not is_short,
                'sell_signal': sell_signal or is_short,
                'short_signal': is_short,
                'signal_direction': 'SHORT' if is_short else 'LONG',
                'recommendation': recommendation,
                'reason': reason_text,
                'market_regime': regime,
                'selection_timestamp': datetime.now().isoformat()
            }
            
            return suggested_stock
            
        except Exception as e:
            logger.warning(f"Failed to apply strategy logic: {e}")
            return None
    
    def _calculate_conservative_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate score for conservative strategy."""
        score = 0.0
        
        # Price stability (0-0.3)
        current_price = stock_data.get('current_price', 0)
        if 100 <= current_price <= 1000:
            score += 0.3
        elif 50 <= current_price <= 2000:
            score += 0.2
        
        # Market cap stability (0-0.3)
        market_cap = stock_data.get('market_cap', 0)
        if market_cap > 10000:  # Large cap
            score += 0.3
        elif market_cap > 5000:  # Mid cap
            score += 0.2
        
        # P/E ratio (0-0.2)
        pe_ratio = stock_data.get('pe_ratio')
        if pe_ratio and 10 <= pe_ratio <= 25:
            score += 0.2
        elif pe_ratio and 5 <= pe_ratio <= 35:
            score += 0.1
        
        # Volume (0-0.2)
        volume = stock_data.get('volume', 0)
        if volume > 100000:
            score += 0.2
        elif volume > 50000:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_conservative_score_with_config(self, stock_data: Dict[str, Any]) -> float:
        """Calculate conservative score using stock_filters.yaml configuration."""
        if not self.stock_filters_config:
            return self._calculate_conservative_score(stock_data)
        
        # Get scoring weights from configuration
        scoring_weights = self.stock_filters_config.get('stage_2_filters', {}).get('scoring_weights', {})
        technical_weight = scoring_weights.get('technical_score', 0.30)
        fundamental_weight = scoring_weights.get('fundamental_score', 0.20)
        risk_weight = scoring_weights.get('risk_score', 0.20)
        momentum_weight = scoring_weights.get('momentum_score', 0.25)
        volume_weight = scoring_weights.get('volume_score', 0.05)
        
        # Calculate individual scores
        technical_score = self._calculate_technical_score(stock_data)
        fundamental_score = self._calculate_fundamental_score(stock_data)
        risk_score = self._calculate_risk_score(stock_data)
        momentum_score = self._calculate_momentum_score(stock_data)
        volume_score = self._calculate_volume_score(stock_data)
        
        # Weighted total score
        total_score = (
            technical_score * technical_weight +
            fundamental_score * fundamental_weight +
            risk_score * risk_weight +
            momentum_score * momentum_weight +
            volume_score * volume_weight
        )
        
        return min(total_score, 1.0)
    
    def _calculate_aggressive_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate score for aggressive strategy."""
        score = 0.0
        
        # Price momentum (0-0.4)
        current_price = stock_data.get('current_price', 0)
        if 50 <= current_price <= 500:  # Sweet spot for growth
            score += 0.4
        elif 20 <= current_price <= 1000:
            score += 0.3
        
        # Market cap growth potential (0-0.3)
        market_cap = stock_data.get('market_cap', 0)
        if 1000 <= market_cap <= 10000:  # Mid cap growth
            score += 0.3
        elif 500 <= market_cap <= 50000:
            score += 0.2
        
        # Volume activity (0-0.3)
        volume = stock_data.get('volume', 0)
        if volume > 50000:
            score += 0.3
        elif volume > 20000:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_aggressive_score_with_config(self, stock_data: Dict[str, Any]) -> float:
        """Calculate aggressive score using stock_filters.yaml configuration."""
        if not self.stock_filters_config:
            return self._calculate_aggressive_score(stock_data)
        
        # Use same weighted scoring as conservative but with different thresholds
        return self._calculate_conservative_score_with_config(stock_data)
    
    def _calculate_technical_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate technical score based on configuration."""
        # Simplified technical scoring - in real implementation would use technical indicators
        current_price = stock_data.get('current_price', 0)
        volume = stock_data.get('volume', 0)
        
        score = 0.0
        
        # Price range scoring
        if 50 <= current_price <= 2000:
            score += 0.5
        
        # Volume scoring
        if volume > 100000:
            score += 0.5
        elif volume > 50000:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_fundamental_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate fundamental score based on configuration."""
        pe_ratio = stock_data.get('pe_ratio')
        pb_ratio = stock_data.get('pb_ratio')
        roe = stock_data.get('roe')
        
        score = 0.0
        
        # P/E ratio scoring
        if pe_ratio and 10 <= pe_ratio <= 25:
            score += 0.4
        elif pe_ratio and 5 <= pe_ratio <= 35:
            score += 0.2
        
        # P/B ratio scoring
        if pb_ratio and 1 <= pb_ratio <= 3:
            score += 0.3
        elif pb_ratio and 0.5 <= pb_ratio <= 5:
            score += 0.1
        
        # ROE scoring
        if roe and roe > 15:
            score += 0.3
        elif roe and roe > 10:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_risk_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate risk score based on configuration."""
        market_cap = stock_data.get('market_cap', 0)
        current_price = stock_data.get('current_price', 0)
        
        score = 0.0
        
        # Market cap stability (higher market cap = lower risk)
        if market_cap > 20000:  # Large cap
            score += 0.5
        elif market_cap > 5000:  # Mid cap
            score += 0.3
        
        # Price stability
        if 100 <= current_price <= 1000:
            score += 0.5
        elif 50 <= current_price <= 2000:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_momentum_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate momentum score based on configuration."""
        volume = stock_data.get('volume', 0)
        current_price = stock_data.get('current_price', 0)
        
        score = 0.0
        
        # Volume momentum
        if volume > 200000:
            score += 0.6
        elif volume > 100000:
            score += 0.4
        
        # Price momentum (simplified)
        if 50 <= current_price <= 500:
            score += 0.4
        elif 20 <= current_price <= 1000:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_volume_score(self, stock_data: Dict[str, Any]) -> float:
        """Calculate volume score based on configuration."""
        volume = stock_data.get('volume', 0)
        
        if volume > 500000:
            return 1.0
        elif volume > 200000:
            return 0.8
        elif volume > 100000:
            return 0.6
        elif volume > 50000:
            return 0.4
        else:
            return 0.2
    
    def _calculate_target_price(self, current_price: float, strategy: str) -> float:
        """Calculate target price based on strategy."""
        if strategy == 'DEFAULT_RISK':
            return current_price * 1.07  # 7% target
        elif strategy == 'HIGH_RISK':
            return current_price * 1.12  # 12% target
        return current_price * 1.05  # Default 5%
    
    def _calculate_stop_loss(self, current_price: float, strategy: str) -> float:
        """Calculate stop loss based on strategy."""
        if strategy == 'DEFAULT_RISK':
            return current_price * 0.95  # 5% stop loss
        elif strategy == 'HIGH_RISK':
            return current_price * 0.90  # 10% stop loss
        return current_price * 0.95  # Default 5%
    
    def _generate_reason(self, stock_data: Dict[str, Any], strategy: str, score: float) -> str:
        """Generate reason for stock selection (legacy method)."""
        if strategy == 'DEFAULT_RISK':
            return f"Conservative swing trading opportunity (Score: {score:.2f}, 2-week hold)"
        elif strategy == 'HIGH_RISK':
            return f"Aggressive growth potential (Score: {score:.2f}, 2-week hold)"
        return f"Stock selection (Score: {score:.2f})"

    def _generate_ema_reason(self, stock_data: Dict[str, Any], strategy: str, score: float, signal_quality: str) -> str:
        """
        Generate reason for 8-21 EMA strategy stock selection.

        Explains the power zone status, DeMarker timing, and signal quality.
        """
        ema_8 = stock_data.get('ema_8', 0)
        ema_21 = stock_data.get('ema_21', 0)
        current_price = stock_data.get('current_price', 0)
        demarker = stock_data.get('demarker', 0.5)

        # Calculate EMA separation
        ema_sep_pct = ((ema_8 - ema_21) / ema_21 * 100) if ema_21 > 0 else 0

        # Build reason based on signal quality
        if signal_quality == 'high':
            timing = f"PERFECT ENTRY: Oversold pullback (DeMarker: {demarker:.2f})"
        elif signal_quality == 'medium':
            timing = f"GOOD ENTRY: Mild pullback (DeMarker: {demarker:.2f})"
        else:
            timing = f"BASIC ENTRY: Power zone active (DeMarker: {demarker:.2f})"

        # Strategy-specific message
        if strategy.upper() == 'DEFAULT_RISK':
            return f"{timing}. Bullish trend: Price > 8 EMA > 21 EMA ({ema_sep_pct:+.1f}% separation). Large-cap stability. Score: {score:.2f}"
        elif strategy.upper() == 'HIGH_RISK':
            return f"{timing}. Strong momentum: Price > 8 EMA > 21 EMA ({ema_sep_pct:+.1f}% separation). Small/mid-cap growth. Score: {score:.2f}"

        return f"{timing}. Bullish power zone (Score: {score:.2f})"
    
    def _extract_rejection_reasons(self, rejected_stocks: List[Any]) -> Dict[str, int]:
        """Extract rejection reasons from rejected stocks."""
        reasons = {}
        for stock in rejected_stocks:
            if hasattr(stock, 'reject_reasons'):
                for reason in stock.reject_reasons:
                    reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

    def _execute_step6_ml_prediction(self, saga: SuggestedStocksSaga) -> SuggestedStocksSaga:
        """Step 6: Apply 8-21 EMA strategy technical indicators to suggested stocks."""
        step = SagaStep(
            step_id="step6_ml_prediction",
            name="8-21 EMA Technical Indicators",
            description="Apply 8-21 EMA Strategy (Power Zone + DeMarker + Fibonacci)"
        )
        saga.add_step(step)
        saga.update_step_status("step6_ml_prediction", SagaStepStatus.IN_PROGRESS)

        print(f"\n📊 Step 6: 8-21 EMA Technical Indicators")
        print(f"   Applying EMA strategy to {len(saga.final_results)} stocks...")

        try:
            from src.models.database import get_database_manager
            from src.services.technical.ema_strategy_calculator import get_ema_strategy_calculator

            db_manager = get_database_manager()

            with db_manager.get_session() as session:
                # Get symbols
                symbols = [stock.get('symbol') for stock in saga.final_results]

                if not symbols:
                    logger.warning("No symbols to get indicators for")
                    return saga

                # Initialize EMA strategy calculator
                print(f"   🔧 Calculating EMA indicators for {len(symbols)} stocks...")
                ema_calc = get_ema_strategy_calculator(session)

                # Calculate all indicators (8 & 21 EMA + DeMarker + Fibonacci)
                indicators_dict = ema_calc.calculate_all_indicators(symbols, lookback_days=252)

                # Apply indicators to stocks
                stocks_with_indicators = 0
                total_ema_score = 0
                total_demarker = 0
                buy_signals = 0
                sell_signals = 0
                high_quality_signals = 0
                medium_quality_signals = 0

                for stock in saga.final_results:
                    symbol = stock.get('symbol')

                    if symbol in indicators_dict:
                        indicators = indicators_dict[symbol]

                        # 8-21 EMA Strategy indicators
                        stock['ema_8'] = indicators.get('ema_8', 0.0)
                        stock['ema_21'] = indicators.get('ema_21', 0.0)
                        stock['ema_trend_score'] = indicators.get('ema_strategy_score', 50.0)
                        stock['demarker'] = indicators.get('demarker', 0.5)
                        stock['power_zone_status'] = indicators.get('power_zone_status', 'neutral')
                        stock['is_bullish'] = indicators.get('is_bullish', False)

                        # Fibonacci targets
                        stock['fib_target_1'] = indicators.get('fib_target_127', 0.0)
                        stock['fib_target_2'] = indicators.get('fib_target_162', 0.0)
                        stock['fib_target_3'] = indicators.get('fib_target_200', 0.0)

                        # EMA strategy score = selection score
                        stock['selection_score'] = indicators.get('ema_strategy_score', 50.0)

                        # Signals
                        stock['buy_signal'] = indicators.get('buy_signal', False)
                        stock['sell_signal'] = indicators.get('sell_signal', False)
                        stock['signal_quality'] = indicators.get('signal_quality', 'none')
                        stock['short_signal'] = indicators.get('short_signal', False) or stock.get('short_signal', False)
                        stock['is_bearish_power_zone'] = indicators.get('is_bearish_power_zone', False)
                        if stock.get('signal_direction') == 'SHORT':
                            stock['short_quality'] = indicators.get('short_quality', stock.get('signal_quality', 'none'))

                        # Count statistics
                        stocks_with_indicators += 1
                        total_ema_score += stock['ema_trend_score']
                        total_demarker += stock['demarker']

                        if stock['buy_signal']:
                            buy_signals += 1
                            if stock['signal_quality'] == 'high':
                                high_quality_signals += 1
                            elif stock['signal_quality'] == 'medium':
                                medium_quality_signals += 1
                        if stock['sell_signal']:
                            sell_signals += 1

                    else:
                        # No indicators available - use default values
                        self._set_default_technical_values(stock)

                print(f"   ✅ EMA indicators applied to {stocks_with_indicators}/{len(saga.final_results)} stocks")

                # ========================================
                # POST-CALCULATION RE-FILTER: Remove stocks that no longer qualify
                # Step 3 filtered on stale DB values; Step 6 recalculated fresh indicators.
                # Stocks may now have buy_signal=false or weak signal_quality.
                # ========================================
                pre_filter_count = len(saga.final_results)
                regime = getattr(saga, 'market_regime', {}).get('regime', 'neutral')
                saga.final_results = [
                    stock for stock in saga.final_results
                    if (
                        # Long signals: buy_signal + no sell + medium+
                        (stock.get('buy_signal', False) and not stock.get('sell_signal', False)
                         and stock.get('signal_quality') in ('high', 'medium'))
                        or
                        # Short signals in bearish regime
                        (stock.get('signal_direction') == 'SHORT'
                         and stock.get('signal_quality') in ('high', 'medium'))
                    )
                ]
                post_filter_removed = pre_filter_count - len(saga.final_results)
                if post_filter_removed > 0:
                    print(f"   🚫 Post-calculation filter removed {post_filter_removed} stocks (buy_signal=false or weak quality)")
                    logger.info(f"Post-Step6 re-filter: removed {post_filter_removed} stocks that no longer qualify")

                # ========================================
                # SORT AND APPLY LIMIT BASED ON EMA STRATEGY SCORE
                # ========================================
                print(f"\n   📊 Sorting and selecting top stocks by EMA strategy score...")

                # Sort by EMA strategy score (descending order - best scores first)
                saga.final_results.sort(key=lambda x: x.get('selection_score', 0), reverse=True)

                # Apply limit: select top N stocks
                stocks_before_limit = len(saga.final_results)
                if saga.limit and saga.limit > 0:
                    saga.final_results = saga.final_results[:saga.limit]
                    print(f"   ✂️  Applied limit: Selected top {len(saga.final_results)} stocks (from {stocks_before_limit} total)")
                else:
                    print(f"   ✅ No limit applied - keeping all {len(saga.final_results)} stocks")

                # Add ranks based on EMA score (1 = highest score)
                for rank, stock in enumerate(saga.final_results, 1):
                    stock['rank'] = rank

                print(f"   🏆 Final selection: {len(saga.final_results)} stocks ranked by EMA strategy score")

                # Store metadata
                avg_ema = total_ema_score / stocks_with_indicators if stocks_with_indicators > 0 else 50.0
                avg_demarker = total_demarker / stocks_with_indicators if stocks_with_indicators > 0 else 0.5

                step.metadata = {
                    'method': '8_21_ema_strategy',
                    'indicators_applied': stocks_with_indicators,
                    'avg_ema_strategy_score': avg_ema,
                    'avg_demarker': avg_demarker,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'high_quality_signals': high_quality_signals,
                    'medium_quality_signals': medium_quality_signals,
                    'stocks_before_limit': stocks_before_limit,
                    'stocks_after_limit': len(saga.final_results),
                    'limit_applied': saga.limit if saga.limit and saga.limit > 0 else 'none'
                }

                step.input_count = stocks_before_limit
                step.output_count = len(saga.final_results)

                saga.update_step_status("step6_ml_prediction", SagaStepStatus.COMPLETED,
                                      metadata=step.metadata)

                return saga

        except Exception as e:
            error_msg = f"Technical indicators application failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            saga.update_step_status("step6_ml_prediction", SagaStepStatus.FAILED, error_msg)
            # Don't fail the saga - continue without indicators
            return saga

    def _set_default_technical_values(self, stock: Dict[str, Any]) -> None:
        """Set default 8-21 EMA technical indicator values when data is not available."""
        # 8-21 EMA indicators
        stock['ema_8'] = 0.0
        stock['ema_21'] = 0.0
        stock['ema_trend_score'] = 50.0
        stock['demarker'] = 0.5

        # Fibonacci targets
        stock['fib_target_1'] = 0.0
        stock['fib_target_2'] = 0.0
        stock['fib_target_3'] = 0.0

        # EMA strategy score
        stock['selection_score'] = 50.0  # Neutral composite score

        # Signals
        stock['buy_signal'] = False
        stock['sell_signal'] = False
        stock['signal_quality'] = 'none'

    def _execute_step7_daily_snapshot(self, saga: SuggestedStocksSaga) -> SuggestedStocksSaga:
        """Step 7: Save daily snapshot with technical indicators."""
        step = SagaStep(
            step_id="step7_daily_snapshot",
            name="Daily Snapshot",
            description="Save daily snapshot of suggested stocks with technical indicators"
        )
        saga.add_step(step)
        saga.update_step_status("step7_daily_snapshot", SagaStepStatus.IN_PROGRESS)

        print(f"\n💾 Step 7: Daily Snapshot")
        print(f"   Saving daily snapshot of {len(saga.final_results)} stocks...")

        try:
            from .daily_snapshot_service import DailySnapshotService
            from src.models.database import get_database_manager
            from datetime import date

            # Initialize snapshot service
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                snapshot_service = DailySnapshotService(session)

                # Build 8-21 EMA technical indicators dict
                technical_indicators = {}
                for stock in saga.final_results:
                    symbol = stock.get('symbol')
                    technical_indicators[symbol] = {
                        'ema_8': stock.get('ema_8', 0.0),
                        'ema_21': stock.get('ema_21', 0.0),
                        'ema_trend_score': stock.get('ema_trend_score', 50.0),
                        'demarker': stock.get('demarker', 0.5),
                        'buy_signal': stock.get('buy_signal', False),
                        'sell_signal': stock.get('sell_signal', False),
                        'signal_quality': stock.get('signal_quality', 'none')
                    }

                # Save snapshot (will replace same-day data)
                stats = snapshot_service.save_daily_snapshot(
                    suggested_stocks=saga.final_results,
                    snapshot_date=date.today()
                )

                step.metadata = stats
                step.input_count = len(saga.final_results)
                step.output_count = stats['inserted'] + stats['updated']

                saga.update_step_status("step7_daily_snapshot", SagaStepStatus.COMPLETED,
                                      metadata=step.metadata)

                print(f"   ✅ Daily snapshot saved: {stats['inserted']} inserted, {stats['updated']} updated")

            return saga

        except Exception as e:
            error_msg = f"Daily snapshot save failed: {str(e)}"
            logger.error(error_msg)
            saga.update_step_status("step7_daily_snapshot", SagaStepStatus.FAILED, error_msg)
            # Don't fail the saga - snapshot is optional
            return saga

    def _compute_db_ema_score(self, stock, direction: str = 'LONG') -> float:
        """
        Compute EMA strategy score for both LONG and SHORT directions.
        Returns 0-100 score based on EMA separation, DeMarker, and price position.

        LONG (D_momentum): Higher DeMarker = stronger buying pressure = better
        SHORT (D_momentum inverted): Lower DeMarker = stronger selling pressure = better
        """
        try:
            ema_8 = float(stock.ema_8) if stock.ema_8 else 0.0
            ema_21 = float(stock.ema_21) if stock.ema_21 else 0.0
            price = float(stock.current_price) if stock.current_price else 0.0
            demarker = float(stock.demarker) if stock.demarker else 0.5

            if not ema_21 or ema_21 <= 0 or not price or price <= 0:
                return 0.0

            if direction == 'SHORT':
                # === BEARISH SCORING ===
                # Bearish power zone: Price < 8 EMA < 21 EMA
                if not (price < ema_8 < ema_21):
                    return 0.0

                # Component 1: Bearish EMA separation (0-20 points)
                bear_sep_pct = ((ema_21 - ema_8) / ema_21) * 100
                sep_score = min(bear_sep_pct / 5.0, 1.0) * 20.0

                # Component 2: DeMarker selling pressure (0-50 points)
                # For shorts: lower DeMarker = stronger selling = better
                if 0.30 <= demarker <= 0.45:
                    dm_score = 50.0   # Strong selling pressure
                elif 0.45 < demarker <= 0.55:
                    dm_score = 40.0   # Moderate selling
                elif 0.55 < demarker <= 0.65:
                    dm_score = 25.0   # Mild
                elif demarker > 0.65:
                    dm_score = 12.5   # Weak selling
                else:
                    dm_score = 0.0    # Deeply oversold — bounce likely

                # Component 3: Price distance below 8 EMA (0-30 points)
                price_below_pct = ((ema_8 - price) / ema_8) * 100
                if 0 <= price_below_pct <= 1.0:
                    dist_score = 30.0
                elif price_below_pct <= 2.0:
                    dist_score = 24.9
                elif price_below_pct <= 3.0:
                    dist_score = 18.0
                elif price_below_pct <= 5.0:
                    dist_score = 9.9
                else:
                    dist_score = 3.0

            else:
                # === BULLISH SCORING (existing D_momentum) ===
                if not (price > ema_8 > ema_21):
                    return 0.0

                ema_sep_pct = ((ema_8 - ema_21) / ema_21) * 100
                sep_score = min(ema_sep_pct / 5.0, 1.0) * 20.0

                if 0.55 <= demarker <= 0.70:
                    dm_score = 50.0
                elif 0.45 <= demarker < 0.55:
                    dm_score = 40.0
                elif 0.35 <= demarker < 0.45:
                    dm_score = 25.0
                elif demarker < 0.35:
                    dm_score = 12.5
                else:
                    dm_score = 0.0

                price_dist_pct = ((price - ema_8) / ema_8) * 100
                if 0 <= price_dist_pct <= 1.0:
                    dist_score = 30.0
                elif price_dist_pct <= 2.0:
                    dist_score = 24.9
                elif price_dist_pct <= 3.0:
                    dist_score = 18.0
                elif price_dist_pct <= 5.0:
                    dist_score = 9.9
                else:
                    dist_score = 3.0

            total = sep_score + dm_score + dist_score
            return round(max(0.0, min(100.0, total)), 2)

        except Exception as e:
            logger.warning(f"Error computing DB EMA score: {e}")
            return 0.0

    def _get_stop_loss_blacklist(self) -> set:
        """
        Get set of symbols that should be excluded due to recent stop-loss exits.
        - Any symbol with a stop-loss exit in the last 30 days (cooldown)
        - Any symbol with 2+ stop-loss exits in the last 90 days (repeated failure)
        """
        blacklist = set()
        try:
            from src.models.database import get_database_manager
            from sqlalchemy import text

            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                # Recent stop-loss cooldown (30 days)
                recent_sl = session.execute(text("""
                    SELECT DISTINCT symbol FROM order_performance
                    WHERE exit_reason = 'stop_loss'
                      AND closed_at >= CURRENT_DATE - INTERVAL :cooldown_days
                """), {'cooldown_days': f'{STOP_LOSS_COOLDOWN_DAYS} days'}).fetchall()

                for row in recent_sl:
                    sym = (row[0] or '').upper().replace('NSE:', '').replace('-EQ', '').strip()
                    if sym:
                        blacklist.add(sym)

                # Repeated stop-loss failures (2+ in 90 days)
                repeated_sl = session.execute(text("""
                    SELECT symbol, COUNT(*) as sl_count FROM order_performance
                    WHERE exit_reason = 'stop_loss'
                      AND closed_at >= CURRENT_DATE - INTERVAL :period_days
                    GROUP BY symbol
                    HAVING COUNT(*) >= :max_hits
                """), {
                    'period_days': f'{STOP_LOSS_MAX_HITS_PERIOD_DAYS} days',
                    'max_hits': STOP_LOSS_MAX_HITS
                }).fetchall()

                for row in repeated_sl:
                    sym = (row[0] or '').upper().replace('NSE:', '').replace('-EQ', '').strip()
                    if sym:
                        blacklist.add(sym)

                if blacklist:
                    logger.info(f"Stop-loss blacklist ({len(blacklist)} symbols): {sorted(blacklist)[:10]}...")

        except Exception as e:
            logger.warning(f"Could not query stop-loss history (table may not exist yet): {e}")

        return blacklist

    def _generate_saga_summary(self, saga: SuggestedStocksSaga) -> Dict[str, Any]:
        """Generate comprehensive saga summary."""
        return {
            'saga_id': saga.saga_id,
            'total_steps': len(saga.steps),
            'completed_steps': len([s for s in saga.steps if s.status == SagaStepStatus.COMPLETED]),
            'failed_steps': len([s for s in saga.steps if s.status == SagaStepStatus.FAILED]),
            'total_duration_seconds': saga.total_duration_seconds,
            'final_result_count': len(saga.final_results),
            'strategies_applied': saga.strategies,
            'search_applied': bool(saga.search_query),
            'sort_applied': bool(saga.sort_by),
            'sector_filter_applied': bool(saga.sector),
            'success_rate': len([s for s in saga.steps if s.status == SagaStepStatus.COMPLETED]) / len(saga.steps) if saga.steps else 0,
            'step_summary': [
                {
                    'step_id': step.step_id,
                    'name': step.name,
                    'status': step.status.value,
                    'duration_seconds': step.duration_seconds,
                    'input_count': step.input_count,
                    'output_count': step.output_count,
                    'success_rate': step.output_count / step.input_count if step.input_count > 0 else 0
                }
                for step in saga.steps
            ]
        }


# Global orchestrator instance
_saga_orchestrator = None


def get_suggested_stocks_saga_orchestrator() -> SuggestedStocksSagaOrchestrator:
    """Get global saga orchestrator instance."""
    global _saga_orchestrator
    if _saga_orchestrator is None:
        _saga_orchestrator = SuggestedStocksSagaOrchestrator()
    return _saga_orchestrator
