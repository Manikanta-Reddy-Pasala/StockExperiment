"""
Trading Strategies API
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required
from api.common.decorators import api_response, validate_json
from api.common.errors import ValidationError, InternalServerError
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

strategies_bp = Blueprint('strategies', __name__)

@strategies_bp.route('/run', methods=['POST'])
@login_required
@validate_json('screened_stocks')
@api_response
def run_strategies():
    """Run trading strategies on screened stocks."""
    try:
        data = request.get_json()
        screened_stocks = data.get('screened_stocks', [])
        
        if not screened_stocks:
            raise ValidationError("No screened stocks provided")
        
        from strategies.strategy_engine import StrategyEngine
        strategy_engine = StrategyEngine()
        strategy_results = strategy_engine.run_strategies(screened_stocks)
        
        return {
            'strategy_results': strategy_results,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to run strategies: {str(e)}")
        raise InternalServerError(f"Failed to run strategies: {str(e)}")

@strategies_bp.route('/list', methods=['GET'])
@login_required
@api_response
def get_strategies():
    """Get available trading strategies."""
    try:
        from strategies.strategy_engine import StrategyEngine
        strategy_engine = StrategyEngine()
        strategies = strategy_engine.get_available_strategies()
        
        return strategies
    except Exception as e:
        logger.error(f"Failed to get strategies: {str(e)}")
        raise InternalServerError(f"Failed to get strategies: {str(e)}")

@strategies_bp.route('/performance', methods=['GET'])
@login_required
@api_response
def get_strategy_performance():
    """Get strategy performance metrics."""
    try:
        from datastore.database import get_database_manager
        from datastore.models import Strategy, Trade
        
        db_manager = get_database_manager()
        
        with db_manager.get_session() as session:
            strategies = session.query(Strategy).all()
            
            performance_data = []
            for strategy in strategies:
                # Get trades for this strategy
                trades = session.query(Trade).filter(Trade.strategy_id == strategy.id).all()
                
                if trades:
                    total_trades = len(trades)
                    winning_trades = len([t for t in trades if t.pnl > 0])
                    total_pnl = sum(t.pnl for t in trades)
                    win_rate = (winning_trades / total_trades) * 100
                    
                    performance_data.append({
                        'strategy_id': strategy.id,
                        'strategy_name': strategy.name,
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'win_rate': round(win_rate, 2),
                        'total_pnl': round(total_pnl, 2),
                        'avg_pnl': round(total_pnl / total_trades, 2)
                    })
            
            return performance_data
    except Exception as e:
        logger.error(f"Failed to get strategy performance: {str(e)}")
        raise InternalServerError(f"Failed to get strategy performance: {str(e)}")

@strategies_bp.route('/backtest', methods=['POST'])
@login_required
@validate_json('strategy_id', 'start_date', 'end_date')
@api_response
def run_backtest():
    """Run backtest for a strategy."""
    try:
        data = request.get_json()
        strategy_id = data.get('strategy_id')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        from strategies.strategy_engine import StrategyEngine
        strategy_engine = StrategyEngine()
        backtest_results = strategy_engine.run_backtest(strategy_id, start_date, end_date)
        
        return backtest_results
    except Exception as e:
        logger.error(f"Failed to run backtest: {str(e)}")
        raise InternalServerError(f"Failed to run backtest: {str(e)}")
