"""
Trading API endpoints
"""
from flask_restx import Namespace, Resource
from flask import request
from flask_login import login_required, current_user
from datetime import datetime

# Create namespace for trading API
ns_trading = Namespace('trading', description='Trading operations')

@ns_trading.route('/trades')
class Trades(Resource):
    @login_required
    def get(self):
        """Get trades."""
        try:
            from datastore.database import get_database_manager
            from datastore.models import Trade
            
            db_manager = get_database_manager()
            
            with db_manager.get_session() as session:
                trades = session.query(Trade).filter(
                    Trade.user_id == current_user.id
                ).order_by(Trade.trade_time.desc()).limit(100).all()
                return [{
                    'id': trade.id,
                    'trade_id': trade.trade_id,
                    'tradingsymbol': trade.tradingsymbol,
                    'transaction_type': trade.transaction_type,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'trade_time': trade.trade_time.isoformat()
                } for trade in trades]
        except Exception as e:
            return {'error': str(e)}, 500

@ns_trading.route('/portfolio')
class Portfolio(Resource):
    @login_required
    def get(self):
        """Get portfolio."""
        try:
            from datastore.database import get_database_manager
            from datastore.models import Position
            
            db_manager = get_database_manager()
            
            with db_manager.get_session() as session:
                positions = session.query(Position).filter(
                    Position.user_id == current_user.id,
                    Position.quantity != 0
                ).all()
                return [{
                    'id': position.id,
                    'tradingsymbol': position.tradingsymbol,
                    'quantity': position.quantity,
                    'average_price': position.average_price,
                    'last_price': position.last_price,
                    'pnl': position.pnl,
                    'value': position.value
                } for position in positions]
        except Exception as e:
            return {'error': str(e)}, 500

@ns_trading.route('/strategies')
class Strategies(Resource):
    @login_required
    def get(self):
        """Get strategies."""
        try:
            from datastore.database import get_database_manager
            from datastore.models import Strategy
            
            db_manager = get_database_manager()
            
            with db_manager.get_session() as session:
                strategies = session.query(Strategy).filter(
                    Strategy.user_id == current_user.id
                ).all()
                
                strategies_data = [{
                    'id': strategy.id,
                    'name': strategy.name,
                    'description': strategy.description,
                    'is_active': strategy.is_active,
                    'created_at': strategy.created_at.isoformat()
                } for strategy in strategies]
            
            return strategies_data
        except Exception as e:
            return {'error': str(e)}, 500

@ns_trading.route('/run-screening')
class RunStockScreening(Resource):
    @login_required
    def post(self):
        """Run stock screening process."""
        try:
            from screening.stock_screener import StockScreener
            
            screener = StockScreener()
            screened_stocks = screener.run_daily_screening()
            
            return {
                'success': True,
                'screened_stocks': screened_stocks,
                'count': len(screened_stocks),
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to run screening: {str(e)}'}, 500

@ns_trading.route('/run-strategies')
class RunTradingStrategies(Resource):
    @login_required
    def post(self):
        """Run trading strategies on screened stocks."""
        try:
            from strategies.strategy_engine import StrategyEngine
            
            data = request.get_json()
            screened_stocks = data.get('screened_stocks', [])
            
            if not screened_stocks:
                return {'error': 'No screened stocks provided'}, 400
            
            strategy_engine = StrategyEngine()
            strategy_results = strategy_engine.run_strategies(screened_stocks)
            
            return {
                'success': True,
                'strategy_results': strategy_results,
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to run strategies: {str(e)}'}, 500

@ns_trading.route('/run-dry-run')
class RunDryRun(Resource):
    @login_required
    def post(self):
        """Run dry run mode for strategy testing."""
        try:
            from execution.trading_executor import TradingExecutor
            
            data = request.get_json()
            strategy_name = data.get('strategy_name')  # Optional: specific strategy
            
            executor = TradingExecutor(user_id=current_user.id)
            result = executor.run_dry_run_only(strategy_name)
            
            return {
                'success': True,
                'dry_run_result': result,
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to run dry run: {str(e)}'}, 500

@ns_trading.route('/run-complete-workflow')
class RunCompleteWorkflow(Resource):
    @login_required
    def post(self):
        """Run complete trading workflow."""
        try:
            from execution.trading_executor import TradingExecutor
            
            executor = TradingExecutor(user_id=current_user.id)
            result = executor.run_complete_workflow()
            
            return {
                'success': True,
                'execution_result': result,
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to run complete workflow: {str(e)}'}, 500

@ns_trading.route('/start-scheduled-execution')
class StartScheduledExecution(Resource):
    @login_required
    def post(self):
        """Start scheduled execution of trading workflow."""
        try:
            from execution.trading_executor import TradingExecutor
            
            data = request.get_json()
            interval_hours = data.get('interval_hours', 1)
            
            executor = TradingExecutor(user_id=current_user.id)
            executor.start_scheduled_execution(interval_hours)
            
            return {
                'success': True,
                'message': f'Scheduled execution started with {interval_hours} hour interval',
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to start scheduled execution: {str(e)}'}, 500

@ns_trading.route('/stop-scheduled-execution')
class StopScheduledExecution(Resource):
    @login_required
    def post(self):
        """Stop scheduled execution."""
        try:
            from execution.trading_executor import TradingExecutor
            
            executor = TradingExecutor(user_id=current_user.id)
            executor.stop_scheduled_execution()
            
            return {
                'success': True,
                'message': 'Scheduled execution stopped',
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to stop scheduled execution: {str(e)}'}, 500

@ns_trading.route('/execution-status')
class ExecutionStatus(Resource):
    @login_required
    def get(self):
        """Get current execution status."""
        try:
            from execution.trading_executor import TradingExecutor
            
            executor = TradingExecutor(user_id=current_user.id)
            status = executor.get_execution_status()
            
            return {
                'success': True,
                'status': status,
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to get execution status: {str(e)}'}, 500

@ns_trading.route('/cleanup-dry-run')
class CleanupDryRun(Resource):
    @login_required
    def post(self):
        """Clean up dry run portfolios."""
        try:
            from execution.trading_executor import TradingExecutor
            
            executor = TradingExecutor(user_id=current_user.id)
            executor.cleanup_dry_run_portfolios()
            
            return {
                'success': True,
                'message': 'Dry run portfolios cleaned up',
                'timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            return {'error': f'Failed to cleanup dry run: {str(e)}'}, 500