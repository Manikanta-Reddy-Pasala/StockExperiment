"""
Pipeline Saga - Simple retry pattern with failure tracking
Single file that handles the entire data pipeline with retry logic
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import text, func
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from ...models.database import get_database_manager
    from ...models.stock_models import Stock, SymbolMaster
    from ...models.historical_models import HistoricalData, TechnicalIndicators
    from ..core.unified_broker_service import get_unified_broker_service
except ImportError:
    from src.models.database import get_database_manager
    from src.models.stock_models import Stock, SymbolMaster
    from src.models.historical_models import HistoricalData, TechnicalIndicators
    from src.services.core.unified_broker_service import get_unified_broker_service


class PipelineStep(Enum):
    """Pipeline steps with their order and dependencies."""
    SYMBOL_MASTER = 1
    STOCKS = 2
    HISTORICAL_DATA = 3
    TECHNICAL_INDICATORS = 4
    VOLATILITY_CALCULATION = 5
    PIPELINE_VALIDATION = 6


class PipelineStatus(Enum):
    """Pipeline status tracking."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class PipelineSaga:
    """
    Simple saga pattern for data pipeline with retry logic and failure tracking.
    Tracks each step, retries on failure, and stores failure reasons.
    """
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.broker_service = get_unified_broker_service()
        self.rate_limit_delay = 0.5
        self.max_retries = 3
        self.retry_delay = 60  # 1 minute between retries
        
    def create_pipeline_tracking_table(self):
        """Create table to track pipeline progress and failures."""
        try:
            with self.db_manager.get_session() as session:
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS pipeline_tracking (
                        id SERIAL PRIMARY KEY,
                        step_name VARCHAR(50) NOT NULL,
                        status VARCHAR(20) NOT NULL DEFAULT 'pending',
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        retry_count INTEGER DEFAULT 0,
                        failure_reason TEXT,
                        records_processed INTEGER DEFAULT 0,
                        last_error TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                session.commit()
                logger.info("✅ Pipeline tracking table created/verified")
        except Exception as e:
            logger.error(f"Error creating pipeline tracking table: {e}")
    
    def get_step_status(self, step: PipelineStep) -> Dict[str, Any]:
        """Get current status of a pipeline step."""
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(text("""
                    SELECT status, retry_count, failure_reason, last_error, records_processed
                    FROM pipeline_tracking 
                    WHERE step_name = :step_name
                    ORDER BY created_at DESC 
                    LIMIT 1
                """), {'step_name': step.name}).fetchone()
                
                if result:
                    return {
                        'status': result.status,
                        'retry_count': result.retry_count,
                        'failure_reason': result.failure_reason,
                        'last_error': result.last_error,
                        'records_processed': result.records_processed
                    }
                return {'status': 'pending', 'retry_count': 0}
        except Exception as e:
            logger.error(f"Error getting step status: {e}")
            return {'status': 'pending', 'retry_count': 0}
    
    def update_step_status(self, step: PipelineStep, status: PipelineStatus, 
                          records_processed: int = 0, error: str = None):
        """Update pipeline step status."""
        try:
            with self.db_manager.get_session() as session:
                # Get current retry count
                current = session.execute(text("""
                    SELECT retry_count FROM pipeline_tracking 
                    WHERE step_name = :step_name 
                    ORDER BY created_at DESC LIMIT 1
                """), {'step_name': step.name}).fetchone()
                
                retry_count = (current.retry_count + 1) if current and status == PipelineStatus.RETRYING else (current.retry_count if current else 0)
                
                session.execute(text("""
                    INSERT INTO pipeline_tracking 
                    (step_name, status, started_at, completed_at, retry_count, failure_reason, 
                     records_processed, last_error, updated_at)
                    VALUES (:step_name, :status, :started_at, :completed_at, :retry_count, 
                            :failure_reason, :records_processed, :last_error, :updated_at)
                """), {
                    'step_name': step.name,
                    'status': status.value,
                    'started_at': datetime.utcnow() if status == PipelineStatus.IN_PROGRESS else None,
                    'completed_at': datetime.utcnow() if status == PipelineStatus.COMPLETED else None,
                    'retry_count': retry_count,
                    'failure_reason': error if status == PipelineStatus.FAILED else None,
                    'records_processed': records_processed,
                    'last_error': error,
                    'updated_at': datetime.utcnow()
                })
                session.commit()
        except Exception as e:
            logger.error(f"Error updating step status: {e}")
    
    def execute_step_with_retry(self, step: PipelineStep, step_function) -> Dict[str, Any]:
        """Execute a pipeline step with retry logic."""
        max_retries = self.max_retries
        consecutive_failures = 0
        max_consecutive_failures = 10  # Stop if we get 10 consecutive failures
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"🔄 Executing {step.name} (attempt {attempt + 1}/{max_retries + 1})")
                
                # Update status to in_progress
                self.update_step_status(step, PipelineStatus.IN_PROGRESS)
                
                # Execute the step
                result = step_function()
                
                if result.get('success', False):
                    # Success - reset failure counter
                    consecutive_failures = 0
                    self.update_step_status(step, PipelineStatus.COMPLETED, 
                                          result.get('records_processed', 0))
                    logger.info(f"✅ {step.name} completed successfully")
                    return result
                else:
                    # Failure
                    consecutive_failures += 1
                    error_msg = result.get('error', 'Unknown error')
                    logger.warning(f"⚠️ {step.name} failed: {error_msg}")
                    
                    # Check if we've hit the consecutive failure threshold
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"🛑 {step.name} stopped after {consecutive_failures} consecutive failures")
                        self.update_step_status(step, PipelineStatus.FAILED, 
                                              result.get('records_processed', 0), 
                                              f"Stopped after {consecutive_failures} consecutive failures")
                        return {'success': False, 'error': f'Stopped after {consecutive_failures} consecutive failures'}
                    
                    if attempt < max_retries:
                        # Retry
                        self.update_step_status(step, PipelineStatus.RETRYING, 
                                            result.get('records_processed', 0), error_msg)
                        logger.info(f"🔄 Retrying {step.name} in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                    else:
                        # Final failure
                        self.update_step_status(step, PipelineStatus.FAILED, 
                                            result.get('records_processed', 0), error_msg)
                        logger.error(f"❌ {step.name} failed after {max_retries} retries")
                        return result
                        
            except Exception as e:
                consecutive_failures += 1
                error_msg = str(e)
                logger.error(f"❌ Exception in {step.name}: {error_msg}")
                
                # Check if we've hit the consecutive failure threshold
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"🛑 {step.name} stopped after {consecutive_failures} consecutive exceptions")
                    self.update_step_status(step, PipelineStatus.FAILED, 0, 
                                          f"Stopped after {consecutive_failures} consecutive exceptions")
                    return {'success': False, 'error': f'Stopped after {consecutive_failures} consecutive exceptions'}
                
                if attempt < max_retries:
                    self.update_step_status(step, PipelineStatus.RETRYING, 0, error_msg)
                    time.sleep(self.retry_delay)
                else:
                    self.update_step_status(step, PipelineStatus.FAILED, 0, error_msg)
                    return {'success': False, 'error': error_msg}
        
        return {'success': False, 'error': 'Max retries exceeded'}
    
    def step_symbol_master(self) -> Dict[str, Any]:
        """Step 1: Ensure symbol master is populated."""
        try:
            with self.db_manager.get_session() as session:
                count = session.execute(text('SELECT COUNT(*) as count FROM symbol_master')).fetchone().count
                
                if count >= 2000:  # Expected ~2253 symbols
                    return {
                        'success': True,
                        'records_processed': count,
                        'message': f'Symbol master already has {count} records'
                    }
                else:
                    # Trigger symbol download
                    from ..data.stock_initialization_service import get_stock_initialization_service
                    init_service = get_stock_initialization_service()
                    result = init_service._load_symbol_master_from_fyers()
                    
                    if result.get('success'):
                        return {
                            'success': True,
                            'records_processed': result.get('total_symbols', 0),
                            'message': 'Symbol master populated successfully'
                        }
                    else:
                        return {
                            'success': False,
                            'error': result.get('error', 'Failed to load symbols')
                        }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def step_stocks(self) -> Dict[str, Any]:
        """Step 2: Ensure stocks table is populated."""
        try:
            with self.db_manager.get_session() as session:
                count = session.execute(text('SELECT COUNT(*) as count FROM stocks')).fetchone().count
                
                if count >= 2000:  # Expected ~2253 stocks
                    return {
                        'success': True,
                        'records_processed': count,
                        'message': f'Stocks already has {count} records'
                    }
                else:
                    # Trigger stock sync (volatility warning is expected here)
                    # Volatility will be calculated in Step 5 (VOLATILITY_CALCULATION)
                    from ..data.stock_initialization_service import get_stock_initialization_service
                    init_service = get_stock_initialization_service()
                    result = init_service.fast_sync_stocks(user_id=1)
                    
                    if result.get('success'):
                        return {
                            'success': True,
                            'records_processed': result.get('stocks_created', 0),
                            'message': 'Stocks populated successfully (volatility will be calculated in Step 5)'
                        }
                    else:
                        return {
                            'success': False,
                            'error': result.get('error', 'Failed to sync stocks')
                        }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def step_historical_data(self) -> Dict[str, Any]:
        """Step 3: Download historical data for stocks missing data."""
        try:
            # Get stocks that need historical data
            with self.db_manager.get_session() as session:
                stocks_needing_data = session.execute(text("""
                    SELECT s.symbol FROM stocks s 
                    LEFT JOIN (
                        SELECT symbol, COUNT(*) as hist_count 
                        FROM historical_data 
                        GROUP BY symbol
                    ) h ON s.symbol = h.symbol 
                    WHERE h.symbol IS NULL OR h.hist_count < 300
                    AND s.symbol IN (
                        'NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:HDFCBANK-EQ', 'NSE:INFY-EQ', 'NSE:ITC-EQ',
                        'NSE:HINDUNILVR-EQ', 'NSE:ICICIBANK-EQ', 'NSE:KOTAKBANK-EQ', 'NSE:SBIN-EQ', 'NSE:BHARTIARTL-EQ',
                        'NSE:ASIANPAINT-EQ', 'NSE:MARUTI-EQ', 'NSE:ULTRACEMCO-EQ', 'NSE:TITAN-EQ', 'NSE:SUNPHARMA-EQ',
                        'NSE:WIPRO-EQ', 'NSE:LT-EQ', 'NSE:BAJAJFINSV-EQ', 'NSE:AXISBANK-EQ', 'NSE:NESTLEIND-EQ'
                    )
                    ORDER BY s.volume DESC
                    LIMIT 20
                """)).fetchall()
                
                if not stocks_needing_data:
                    return {
                        'success': True,
                        'records_processed': 0,
                        'message': 'All stocks have sufficient historical data'
                    }
                
                symbols = [row.symbol for row in stocks_needing_data]
                logger.info(f"📊 Downloading historical data for {len(symbols)} stocks")
                
                # Download historical data using Fyers service directly
                from ..brokers.fyers_service import get_fyers_service
                fyers_service = get_fyers_service()
                
                total_records = 0
                successful_downloads = 0
                failed_downloads = 0
                
                for symbol in symbols:
                    try:
                        # Get historical data for the last 365 days (maximum Fyers API supports)
                        from datetime import datetime, timedelta
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=365)
                        
                        result = fyers_service.history(
                            user_id=1, 
                            symbol=symbol, 
                            exchange='NSE', 
                            interval='1D',
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d')
                        )
                        
                        if result.get('status') == 'success' and result.get('data', {}).get('candles'):
                            # Store the data
                            candles = result['data']['candles']
                            records_added = self._store_historical_data(symbol, candles)
                            total_records += records_added
                            successful_downloads += 1
                            logger.info(f"✅ Downloaded {records_added} records for {symbol}")
                        else:
                            failed_downloads += 1
                            error_msg = result.get('message', 'Unknown error')
                            logger.warning(f"⚠️ No data for {symbol}: {error_msg}")
                            
                            # If we get too many API failures, stop trying
                            if failed_downloads > 10:
                                logger.warning(f"🛑 Too many API failures, stopping download after {failed_downloads} failures")
                                break
                            
                        time.sleep(self.rate_limit_delay)
                    except Exception as e:
                        failed_downloads += 1
                        logger.warning(f"Error downloading data for {symbol}: {e}")
                        continue
                
                logger.info(f"📊 Download summary: {successful_downloads} successful, {failed_downloads} failed")
                
                return {
                    'success': True,
                    'records_processed': total_records,
                    'message': f'Downloaded historical data for {len(symbols)} stocks'
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def step_technical_indicators(self) -> Dict[str, Any]:
        """Step 4: Calculate technical indicators for stocks with historical data."""
        try:
            # Get stocks with historical data but no technical indicators
            with self.db_manager.get_session() as session:
                stocks_needing_indicators = session.execute(text("""
                    SELECT DISTINCT s.symbol FROM stocks s 
                    JOIN historical_data h ON s.symbol = h.symbol 
                    LEFT JOIN technical_indicators t ON s.symbol = t.symbol 
                    WHERE t.symbol IS NULL
                    LIMIT 50
                """)).fetchall()
                
                if not stocks_needing_indicators:
                    return {
                        'success': True,
                        'records_processed': 0,
                        'message': 'All stocks have technical indicators'
                    }
                
                symbols = [row.symbol for row in stocks_needing_indicators]
                logger.info(f"📊 Calculating technical indicators for {len(symbols)} stocks")
                
                # Calculate indicators
                from ..data.technical_indicators_service import TechnicalIndicatorsService
                tech_service = TechnicalIndicatorsService()
                
                result = tech_service.calculate_indicators_bulk(max_symbols=len(symbols))
                
                return {
                    'success': result.get('success', False),
                    'records_processed': result.get('symbols_processed', 0),
                    'message': 'Technical indicators calculated'
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def step_volatility_calculation(self) -> Dict[str, Any]:
        """Step 5: Calculate volatility data for stocks."""
        try:
            logger.info("🔄 Starting volatility calculation step...")
            
            # Get stocks that need volatility calculation
            stocks_needing_volatility = self._get_stocks_needing_volatility()
            
            if not stocks_needing_volatility:
                logger.info("✅ All stocks already have volatility data")
                return {
                    'success': True,
                    'records_processed': 0,
                    'message': 'All stocks already have volatility data'
                }
            
            logger.info(f"📊 Found {len(stocks_needing_volatility)} stocks needing volatility calculation")
            
            # Calculate volatility for each stock
            records_processed = 0
            for symbol in stocks_needing_volatility:
                try:
                    # Get historical data for volatility calculation
                    with self.db_manager.get_session() as session:
                        historical_data = session.query(HistoricalData).filter(
                            HistoricalData.symbol == symbol
                        ).order_by(HistoricalData.date.desc()).limit(30).all()
                        
                        if len(historical_data) < 20:  # Need at least 20 days for volatility
                            logger.warning(f"⚠️ Insufficient data for volatility calculation: {symbol}")
                            continue
                        
                        # Calculate volatility (standard deviation of returns)
                        returns = []
                        for i in range(1, len(historical_data)):
                            prev_close = historical_data[i].close
                            curr_close = historical_data[i-1].close
                            if prev_close > 0:
                                returns.append((curr_close - prev_close) / prev_close)
                        
                        if len(returns) < 10:
                            logger.warning(f"⚠️ Insufficient returns for volatility calculation: {symbol}")
                            continue
                        
                        import numpy as np
                        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
                        
                        # Update stock with volatility data
                        stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                        if stock:
                            stock.volatility = volatility
                            session.commit()
                            records_processed += 1
                            logger.info(f"✅ Updated volatility for {symbol}: {volatility:.4f}")
                    
                
                except Exception as e:
                    logger.error(f"❌ Error calculating volatility for {symbol}: {e}")
                    continue
            
            logger.info(f"✅ Volatility calculation completed: {records_processed} stocks updated")
            return {
                'success': True,
                'records_processed': records_processed,
                'message': f'Calculated volatility for {records_processed} stocks'
            }
            
        except Exception as e:
            logger.error(f"❌ Error in volatility calculation step: {e}")
            return {
                'success': False,
                'records_processed': 0,
                'message': f'Volatility calculation failed: {e}'
            }

    def step_pipeline_validation(self) -> Dict[str, Any]:
        """Step 6: Final validation to ensure all steps completed successfully."""
        try:
            logger.info("🔄 Starting pipeline validation step...")
            
            validation_results = {
                'symbol_master_count': 0,
                'stocks_count': 0,
                'historical_data_count': 0,
                'technical_indicators_count': 0,
                'volatility_calculated_count': 0,
                'issues': []
            }
            
            with self.db_manager.get_session() as session:
                # Check symbol_master table
                result = session.execute(text("SELECT COUNT(*) FROM symbol_master")).scalar()
                validation_results['symbol_master_count'] = result
                if result == 0:
                    validation_results['issues'].append("❌ Symbol master table is empty")
                
                # Check stocks table
                result = session.execute(text("SELECT COUNT(*) FROM stocks")).scalar()
                validation_results['stocks_count'] = result
                if result == 0:
                    validation_results['issues'].append("❌ Stocks table is empty")
                
                # Check historical_data table
                result = session.execute(text("SELECT COUNT(*) FROM historical_data")).scalar()
                validation_results['historical_data_count'] = result
                if result == 0:
                    validation_results['issues'].append("❌ Historical data table is empty")
                
                # Check technical_indicators table
                result = session.execute(text("SELECT COUNT(*) FROM technical_indicators")).scalar()
                validation_results['technical_indicators_count'] = result
                if result == 0:
                    validation_results['issues'].append("❌ Technical indicators table is empty")
                
                # Check stocks with volatility data
                result = session.execute(text("SELECT COUNT(*) FROM stocks WHERE volatility IS NOT NULL AND volatility > 0")).scalar()
                validation_results['volatility_calculated_count'] = result
                if result == 0:
                    validation_results['issues'].append("❌ No stocks have volatility data")
                
                # Check data quality
                symbols_with_historical = session.execute(text("""
                    SELECT COUNT(DISTINCT symbol) FROM historical_data
                """)).scalar()
                
                symbols_with_indicators = session.execute(text("""
                    SELECT COUNT(DISTINCT symbol) FROM technical_indicators
                """)).scalar()
                
                if symbols_with_historical != symbols_with_indicators:
                    validation_results['issues'].append(
                        f"⚠️ Data mismatch: {symbols_with_historical} symbols have historical data, "
                        f"but only {symbols_with_indicators} have technical indicators"
                    )
            
            # Determine overall success
            success = len(validation_results['issues']) == 0
            
            if success:
                logger.info("✅ Pipeline validation passed - all steps completed successfully")
                message = "Pipeline validation passed - all data is complete"
            else:
                logger.warning(f"⚠️ Pipeline validation found {len(validation_results['issues'])} issues")
                message = f"Pipeline validation found {len(validation_results['issues'])} issues"
            
            return {
                'success': success,
                'records_processed': 0,
                'message': message,
                'validation_results': validation_results
            }
            
        except Exception as e:
            logger.error(f"❌ Error in pipeline validation step: {e}")
            return {
                'success': False,
                'records_processed': 0,
                'message': f'Pipeline validation failed: {e}'
            }

    def _get_stocks_needing_volatility(self) -> List[str]:
        """Get stocks that need volatility calculation."""
        try:
            with self.db_manager.get_session() as session:
                # Get stocks that don't have volatility data or have NULL volatility
                result = session.execute(text("""
                    SELECT s.symbol FROM stocks s 
                    WHERE s.volatility IS NULL OR s.volatility = 0
                    ORDER BY s.volume DESC
                    LIMIT 100
                """))
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.error(f"❌ Error getting stocks needing volatility: {e}")
            return []
    
    def _store_historical_data(self, symbol: str, candles: list) -> int:
        """Store historical data from Fyers API response."""
        records_added = 0
        if not candles:
            return 0

        try:
            with self.db_manager.get_session() as session:
                # Get existing dates for the symbol to avoid duplicates
                existing_dates = set(
                    r.date for r in session.query(HistoricalData.date).filter(
                        HistoricalData.symbol == symbol
                    ).all()
                )

                for candle in candles:
                    try:
                        # Convert timestamp to date
                        from datetime import datetime
                        timestamp = int(candle['timestamp'])
                        record_date = datetime.fromtimestamp(timestamp).date()
                        
                        if record_date in existing_dates:
                            continue  # Skip if already exists

                        # Extract OHLCV data
                        open_price = float(candle['open'])
                        high_price = float(candle['high'])
                        low_price = float(candle['low'])
                        close_price = float(candle['close'])
                        volume_val = int(candle['volume'])

                        # Calculate additional fields
                        price_change = close_price - open_price
                        price_change_pct = (price_change / open_price * 100) if open_price > 0 else 0
                        high_low_range = high_price - low_price
                        high_low_pct = (high_low_range / close_price * 100) if close_price > 0 and high_low_range > 0 else 0
                        body_pct = (abs(close_price - open_price) / high_low_range * 100) if high_low_range > 0 else 0
                        upper_shadow_pct = ((high_price - max(open_price, close_price)) / high_low_range * 100) if high_low_range > 0 else 0
                        lower_shadow_pct = ((min(open_price, close_price) - low_price) / high_low_range * 100) if high_low_range > 0 else 0
                        turnover_inr = close_price * volume_val / 10000000 if close_price and volume_val else 0

                        historical_record = HistoricalData(
                            symbol=symbol,
                            date=record_date,
                            timestamp=timestamp,
                            open=open_price,
                            high=high_price,
                            low=low_price,
                            close=close_price,
                            volume=volume_val,
                            turnover=turnover_inr,
                            price_change=price_change,
                            price_change_pct=price_change_pct,
                            high_low_pct=high_low_pct,
                            body_pct=body_pct,
                            upper_shadow_pct=upper_shadow_pct,
                            lower_shadow_pct=lower_shadow_pct,
                            data_source='fyers',
                            api_resolution='1D',
                            data_quality_score=1.0,
                            is_adjusted=False
                        )
                        session.add(historical_record)
                        records_added += 1
                    except Exception as e:
                        logger.warning(f"Error storing record for {symbol} on {record_date}: {e}")
                        continue
                
                session.commit()
        except Exception as e:
            logger.error(f"Error storing historical data for {symbol}: {e}")
        
        return records_added
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline with saga pattern."""
        try:
            logger.info("🚀 Starting Pipeline Saga")
            
            # Create tracking table
            self.create_pipeline_tracking_table()
            
            results = {
                'success': True,
                'steps_completed': [],
                'steps_failed': [],
                'total_records_processed': 0
            }
            
            # Execute each step with retry logic
            steps = [
                (PipelineStep.SYMBOL_MASTER, self.step_symbol_master),
                (PipelineStep.STOCKS, self.step_stocks),
                (PipelineStep.HISTORICAL_DATA, self.step_historical_data),
                (PipelineStep.TECHNICAL_INDICATORS, self.step_technical_indicators),
                (PipelineStep.VOLATILITY_CALCULATION, self.step_volatility_calculation),
                (PipelineStep.PIPELINE_VALIDATION, self.step_pipeline_validation)
            ]
            
            for step, step_function in steps:
                logger.info(f"🔄 Executing {step.name}")
                
                result = self.execute_step_with_retry(step, step_function)
                
                if result.get('success'):
                    results['steps_completed'].append(step.name)
                    results['total_records_processed'] += result.get('records_processed', 0)
                else:
                    results['steps_failed'].append({
                        'step': step.name,
                        'error': result.get('error', 'Unknown error')
                    })
                    results['success'] = False
            
            logger.info(f"🎉 Pipeline Saga completed: {len(results['steps_completed'])}/{len(steps)} steps successful")
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline Saga failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'steps_completed': [],
                'steps_failed': []
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status from tracking table."""
        try:
            with self.db_manager.get_session() as session:
                results = session.execute(text("""
                    SELECT step_name, status, retry_count, failure_reason, 
                           records_processed, last_error, updated_at
                    FROM pipeline_tracking 
                    WHERE id IN (
                        SELECT MAX(id) FROM pipeline_tracking 
                        GROUP BY step_name
                    )
                    ORDER BY step_name
                """)).fetchall()
                
                status = {}
                for row in results:
                    status[row.step_name] = {
                        'status': row.status,
                        'retry_count': row.retry_count,
                        'failure_reason': row.failure_reason,
                        'records_processed': row.records_processed,
                        'last_error': row.last_error,
                        'updated_at': row.updated_at
                    }
                
                return status
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {}


# Global saga instance
_pipeline_saga = None

def get_pipeline_saga() -> PipelineSaga:
    """Get the global pipeline saga instance."""
    global _pipeline_saga
    if _pipeline_saga is None:
        _pipeline_saga = PipelineSaga()
    return _pipeline_saga
