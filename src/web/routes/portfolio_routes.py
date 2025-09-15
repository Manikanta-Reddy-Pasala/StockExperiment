"""
API Routes for Portfolio Strategy Engine
Implements 4-step strategy: Filtering → Risk Allocation → Entry → Exit
"""
import logging
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from datetime import datetime

from ...services.portfolio_strategy_engine import get_portfolio_strategy_engine, RiskBucket

logger = logging.getLogger(__name__)

# Create blueprint
portfolio_bp = Blueprint('portfolio', __name__, url_prefix='/api/portfolio')


@portfolio_bp.route('/strategy/execute', methods=['POST'])
@login_required
def api_execute_complete_strategy():
    """
    Execute the complete 4-step portfolio strategy.
    
    Expected payload:
    {
        "capital": 100000,
        "risk_bucket": "safe" | "high_risk"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'Request data is required'}), 400
        
        capital = data.get('capital', 100000)
        risk_bucket_str = data.get('risk_bucket', 'safe')
        
        # Validate capital
        if capital < 10000:
            return jsonify({
                'success': False, 
                'error': 'Minimum capital required: ₹10,000'
            }), 400
        
        # Validate risk bucket
        try:
            risk_bucket = RiskBucket(risk_bucket_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid risk_bucket. Must be "safe" or "high_risk"'
            }), 400
        
        current_app.logger.info(f"Executing complete strategy for user {current_user.id}: {risk_bucket_str}, ₹{capital:,}")
        
        # Initialize strategy engine
        engine = get_portfolio_strategy_engine(current_user.id)
        
        # Execute complete strategy
        result = engine.execute_complete_strategy(capital, risk_bucket)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400
        
    except Exception as e:
        current_app.logger.error(f"Error executing complete strategy: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/strategy/step1/filter', methods=['POST'])
@login_required
def api_step1_filter_stocks():
    """
    Execute Step 1: Stock Filtering
    
    Filters out junk stocks based on:
    - Price > ₹50
    - Volume > 500k
    - ATR % < 10%
    """
    try:
        current_app.logger.info(f"Executing Step 1 filtering for user {current_user.id}")
        
        engine = get_portfolio_strategy_engine(current_user.id)
        filtered_stocks = engine.step1_filter_stocks()
        
        return jsonify({
            'success': True,
            'step': 'step1_filtering',
            'total_filtered': len(filtered_stocks),
            'stocks': filtered_stocks[:20],  # Return first 20 for preview
            'criteria': {
                'min_price': engine.filter_criteria.min_price,
                'min_volume': engine.filter_criteria.min_avg_volume_20d,
                'max_atr_percent': engine.filter_criteria.max_atr_percent
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in step 1 filtering: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/strategy/step2/allocate', methods=['POST'])
@login_required
def api_step2_risk_allocation():
    """
    Execute Step 2: Risk Strategy Allocation
    
    Expected payload:
    {
        "risk_bucket": "safe" | "high_risk"
    }
    """
    try:
        data = request.get_json()
        risk_bucket_str = data.get('risk_bucket', 'safe') if data else 'safe'
        
        try:
            risk_bucket = RiskBucket(risk_bucket_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid risk_bucket. Must be "safe" or "high_risk"'
            }), 400
        
        current_app.logger.info(f"Executing Step 2 allocation for user {current_user.id}: {risk_bucket_str}")
        
        engine = get_portfolio_strategy_engine(current_user.id)
        
        # First filter stocks
        filtered_stocks = engine.step1_filter_stocks()
        
        # Then allocate by risk
        allocated_stocks = engine.step2_risk_allocation(filtered_stocks, risk_bucket)
        
        return jsonify({
            'success': True,
            'step': 'step2_allocation',
            'risk_bucket': risk_bucket_str,
            'total_allocated': len(allocated_stocks),
            'allocation_summary': engine._get_allocation_summary(allocated_stocks),
            'stocks': allocated_stocks[:15]  # Return first 15 for preview
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in step 2 allocation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/strategy/step3/signals', methods=['POST'])
@login_required
def api_step3_entry_signals():
    """
    Execute Step 3: Entry Signal Generation
    
    Expected payload:
    {
        "risk_bucket": "safe" | "high_risk"
    }
    """
    try:
        data = request.get_json()
        risk_bucket_str = data.get('risk_bucket', 'safe') if data else 'safe'
        
        try:
            risk_bucket = RiskBucket(risk_bucket_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid risk_bucket. Must be "safe" or "high_risk"'
            }), 400
        
        current_app.logger.info(f"Executing Step 3 signals for user {current_user.id}: {risk_bucket_str}")
        
        engine = get_portfolio_strategy_engine(current_user.id)
        
        # Execute steps 1-3
        filtered_stocks = engine.step1_filter_stocks()
        allocated_stocks = engine.step2_risk_allocation(filtered_stocks, risk_bucket)
        entry_candidates = engine.step3_entry_signals(allocated_stocks)
        
        return jsonify({
            'success': True,
            'step': 'step3_entry_signals',
            'risk_bucket': risk_bucket_str,
            'entry_candidates': len(entry_candidates),
            'signals_summary': engine._get_signals_summary(entry_candidates),
            'top_candidates': entry_candidates[:10],  # Top 10 candidates
            'ml_integration': {
                'predictions_generated': len(entry_candidates),
                'ml_api_active': True
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in step 3 signals: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/positions/monitor', methods=['GET'])
@login_required
def api_monitor_positions():
    """
    Monitor active positions and check exit conditions.
    """
    try:
        current_app.logger.info(f"Monitoring positions for user {current_user.id}")
        
        engine = get_portfolio_strategy_engine(current_user.id)
        monitoring_result = engine.monitor_and_exit_positions()
        
        return jsonify({
            'success': True,
            'monitoring_result': monitoring_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error monitoring positions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/positions/active', methods=['GET'])
@login_required
def api_get_active_positions():
    """
    Get all active positions for the user.
    """
    try:
        current_app.logger.info(f"Getting active positions for user {current_user.id}")
        
        engine = get_portfolio_strategy_engine(current_user.id)
        positions_summary = engine._get_positions_summary()
        
        return jsonify({
            'success': True,
            'active_positions': len(positions_summary),
            'positions': positions_summary
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting active positions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/strategy/config', methods=['GET'])
@login_required
def api_get_strategy_config():
    """
    Get strategy configuration and rules.
    """
    try:
        engine = get_portfolio_strategy_engine(current_user.id)
        
        config = {
            'filtering_criteria': {
                'min_price': engine.filter_criteria.min_price,
                'min_avg_volume_20d': engine.filter_criteria.min_avg_volume_20d,
                'max_atr_percent': engine.filter_criteria.max_atr_percent
            },
            'market_cap_thresholds': {
                'large_cap_min': engine.LARGE_CAP_MIN,
                'mid_cap_min': engine.MID_CAP_MIN,
                'small_cap_max': engine.SMALL_CAP_MAX
            },
            'risk_allocations': {
                'safe': {
                    'large_cap': '50%',
                    'mid_cap': '50%',
                    'small_cap': '0%'
                },
                'high_risk': {
                    'large_cap': '0%',
                    'mid_cap': '50%',
                    'small_cap': '50%'
                }
            },
            'entry_rules': {
                'price_above_20ema': 'Required',
                'price_above_50ema': 'Required',
                'breakout_20d_high': 'Required',
                'volume_confirmation': '≥ 1.5× avg volume',
                'rsi_range': '50-70 (not overbought)'
            },
            'exit_rules': {
                'profit_target_1': f'{engine.exit_rules.profit_target_1*100}% (sell 50%)',
                'profit_target_2': f'{engine.exit_rules.profit_target_2*100}% (sell remaining)',
                'stop_loss': f'{engine.exit_rules.stop_loss_percent*100}%',
                'time_stop': f'{engine.exit_rules.max_holding_days} days',
                'trailing_stop': f'{engine.exit_rules.trailing_stop_percent*100}%'
            },
            'ml_integration': {
                'required_confidence': '>60%',
                'required_return': '>3%',
                'signal_required': 'BUY',
                'api_status': 'active'
            }
        }
        
        return jsonify({
            'success': True,
            'configuration': config
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting strategy config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/strategy/backtest', methods=['POST'])
@login_required
def api_strategy_backtest():
    """
    Run strategy backtest (placeholder for future implementation).
    
    Expected payload:
    {
        "risk_bucket": "safe" | "high_risk",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_capital": 100000
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'Request data is required'}), 400
        
        # Not implemented yet – avoid returning mock data
        return jsonify({
            'success': False,
            'error': 'Backtest is not implemented yet. Please check back later.'
        }), 501
        
    except Exception as e:
        current_app.logger.error(f"Error in strategy backtest: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Health check for portfolio strategy engine
@portfolio_bp.route('/health', methods=['GET'])
def api_portfolio_health():
    """Portfolio strategy engine health check."""
    try:
        return jsonify({
            'success': True,
            'service': 'Portfolio Strategy Engine',
            'version': '1.0',
            'timestamp': datetime.utcnow().isoformat(),
            'available_strategies': ['safe', 'high_risk'],
            'steps': [
                'step1_filtering',
                'step2_risk_allocation', 
                'step3_entry_signals',
                'step4_position_management'
            ],
            'ml_integration': 'active',
            'broker_integration': 'fyers'
        })
        
    except Exception as e:
        logger.error(f"Error in portfolio health check: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
