"""
API Routes for Portfolio Strategy Engine
Implements the 4-step strategy: Filtering → Risk Allocation → Entry → Exit
"""
import logging
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprint
portfolio_bp = Blueprint('portfolio', __name__, url_prefix='/api/portfolio-strategy')


@portfolio_bp.route('/execute-complete-strategy', methods=['POST'])
@login_required
def api_execute_complete_strategy():
    """Execute the complete 4-step portfolio strategy."""
    try:
        from ...services.portfolio_strategy_engine import get_portfolio_strategy_engine, RiskBucket
        
        data = request.get_json() or {}
        
        # Parse parameters
        capital = data.get('capital', 100000)
        risk_bucket_str = data.get('risk_bucket', 'safe')
        
        # Validate inputs
        if capital <= 0:
            return jsonify({
                'success': False,
                'error': 'Capital must be greater than 0'
            }), 400
        
        if risk_bucket_str not in ['safe', 'high_risk']:
            return jsonify({
                'success': False,
                'error': 'risk_bucket must be "safe" or "high_risk"'
            }), 400
        
        risk_bucket = RiskBucket.SAFE if risk_bucket_str == 'safe' else RiskBucket.HIGH_RISK
        
        current_app.logger.info(f"Executing complete strategy for user {current_user.id}: {risk_bucket_str} with ₹{capital:,}")
        
        # Execute strategy
        engine = get_portfolio_strategy_engine(current_user.id)
        results = engine.execute_complete_strategy(capital, risk_bucket)
        
        if results.get('success'):
            return jsonify(results)
        else:
            return jsonify(results), 400
            
    except Exception as e:
        current_app.logger.error(f"Error executing complete strategy: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/step1-filter', methods=['POST'])
@login_required
def api_step1_filter():
    """Execute Step 1: Stock filtering."""
    try:
        from ...services.portfolio_strategy_engine import get_portfolio_strategy_engine
        
        current_app.logger.info(f"Executing Step 1 filtering for user {current_user.id}")
        
        engine = get_portfolio_strategy_engine(current_user.id)
        filtered_stocks = engine.step1_filter_stocks()
        
        return jsonify({
            'success': True,
            'step': 'step1_filtering',
            'total_filtered': len(filtered_stocks),
            'filtered_stocks': filtered_stocks[:20],  # Limit response size
            'filter_criteria': {
                'min_price': engine.filter_criteria.min_price,
                'min_avg_volume_20d': engine.filter_criteria.min_avg_volume_20d,
                'max_atr_percent': engine.filter_criteria.max_atr_percent
            },
            'execution_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in step 1 filtering: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/step2-allocate', methods=['POST'])
@login_required
def api_step2_allocate():
    """Execute Step 2: Risk allocation."""
    try:
        from ...services.portfolio_strategy_engine import get_portfolio_strategy_engine, RiskBucket
        
        data = request.get_json() or {}
        risk_bucket_str = data.get('risk_bucket', 'safe')
        
        if risk_bucket_str not in ['safe', 'high_risk']:
            return jsonify({
                'success': False,
                'error': 'risk_bucket must be "safe" or "high_risk"'
            }), 400
        
        risk_bucket = RiskBucket.SAFE if risk_bucket_str == 'safe' else RiskBucket.HIGH_RISK
        
        current_app.logger.info(f"Executing Step 2 allocation for user {current_user.id}: {risk_bucket_str}")
        
        engine = get_portfolio_strategy_engine(current_user.id)
        
        # First get filtered stocks
        filtered_stocks = engine.step1_filter_stocks()
        
        # Then allocate based on risk
        allocated_stocks = engine.step2_risk_allocation(filtered_stocks, risk_bucket)
        
        return jsonify({
            'success': True,
            'step': 'step2_risk_allocation',
            'risk_bucket': risk_bucket_str,
            'total_allocated': len(allocated_stocks),
            'allocated_stocks': allocated_stocks,
            'allocation_summary': engine._get_allocation_summary(allocated_stocks),
            'execution_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in step 2 allocation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/step3-entry-signals', methods=['POST'])
@login_required
def api_step3_entry_signals():
    """Execute Step 3: Entry signal generation."""
    try:
        from ...services.portfolio_strategy_engine import get_portfolio_strategy_engine, RiskBucket
        
        data = request.get_json() or {}
        risk_bucket_str = data.get('risk_bucket', 'safe')
        
        if risk_bucket_str not in ['safe', 'high_risk']:
            return jsonify({
                'success': False,
                'error': 'risk_bucket must be "safe" or "high_risk"'
            }), 400
        
        risk_bucket = RiskBucket.SAFE if risk_bucket_str == 'safe' else RiskBucket.HIGH_RISK
        
        current_app.logger.info(f"Executing Step 3 entry signals for user {current_user.id}: {risk_bucket_str}")
        
        engine = get_portfolio_strategy_engine(current_user.id)
        
        # Execute steps 1 and 2 first
        filtered_stocks = engine.step1_filter_stocks()
        allocated_stocks = engine.step2_risk_allocation(filtered_stocks, risk_bucket)
        
        # Generate entry signals
        entry_candidates = engine.step3_entry_signals(allocated_stocks)
        
        return jsonify({
            'success': True,
            'step': 'step3_entry_signals',
            'risk_bucket': risk_bucket_str,
            'entry_candidates': len(entry_candidates),
            'candidates': entry_candidates,
            'signals_summary': engine._get_signals_summary(entry_candidates),
            'entry_conditions': {
                'price_above_emas': 'Price above 20-day & 50-day EMA',
                'breakout': 'Price > 20-day high',
                'volume': 'Volume ≥ 1.5× avg (20d)',
                'rsi': 'RSI(14) between 50-70',
                'ml_signal': 'ML prediction BUY with >60% confidence'
            },
            'execution_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in step 3 entry signals: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/step4-execute-positions', methods=['POST'])
@login_required
def api_step4_execute_positions():
    """Execute Step 4: Position management."""
    try:
        from ...services.portfolio_strategy_engine import get_portfolio_strategy_engine, RiskBucket
        
        data = request.get_json() or {}
        capital = data.get('capital', 100000)
        risk_bucket_str = data.get('risk_bucket', 'safe')
        
        # Validate inputs
        if capital <= 0:
            return jsonify({
                'success': False,
                'error': 'Capital must be greater than 0'
            }), 400
        
        if risk_bucket_str not in ['safe', 'high_risk']:
            return jsonify({
                'success': False,
                'error': 'risk_bucket must be "safe" or "high_risk"'
            }), 400
        
        risk_bucket = RiskBucket.SAFE if risk_bucket_str == 'safe' else RiskBucket.HIGH_RISK
        
        current_app.logger.info(f"Executing Step 4 positions for user {current_user.id}: {risk_bucket_str} with ₹{capital:,}")
        
        engine = get_portfolio_strategy_engine(current_user.id)
        
        # Execute all previous steps
        filtered_stocks = engine.step1_filter_stocks()
        allocated_stocks = engine.step2_risk_allocation(filtered_stocks, risk_bucket)
        entry_candidates = engine.step3_entry_signals(allocated_stocks)
        
        # Execute position management
        position_results = engine.step4_position_management(entry_candidates, capital)
        
        return jsonify({
            'success': True,
            'step': 'step4_position_management',
            'risk_bucket': risk_bucket_str,
            'capital': capital,
            'position_results': position_results,
            'execution_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in step 4 position management: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/monitor-positions', methods=['GET'])
@login_required
def api_monitor_positions():
    """Monitor active positions and check exit conditions."""
    try:
        from ...services.portfolio_strategy_engine import get_portfolio_strategy_engine
        
        current_app.logger.info(f"Monitoring positions for user {current_user.id}")
        
        engine = get_portfolio_strategy_engine(current_user.id)
        monitoring_results = engine.monitor_and_exit_positions()
        
        return jsonify({
            'success': True,
            'monitoring_results': monitoring_results,
            'exit_rules': {
                'profit_target_1': '5% (sell 50%)',
                'profit_target_2': '10% (sell remaining)',
                'stop_loss': '2-4% below entry',
                'time_stop': '10 days maximum hold',
                'trailing_stop': '3% below current price'
            },
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error monitoring positions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/strategy-config', methods=['GET'])
@login_required
def api_get_strategy_config():
    """Get current strategy configuration."""
    try:
        from ...services.portfolio_strategy_engine import get_portfolio_strategy_engine
        
        engine = get_portfolio_strategy_engine(current_user.id)
        
        return jsonify({
            'success': True,
            'strategy_config': {
                'step1_filtering': {
                    'min_price': engine.filter_criteria.min_price,
                    'min_avg_volume_20d': engine.filter_criteria.min_avg_volume_20d,
                    'max_atr_percent': engine.filter_criteria.max_atr_percent,
                    'description': 'Remove junk stocks before strategy application'
                },
                'step2_allocation': {
                    'market_cap_thresholds': {
                        'large_cap': f'> ₹{engine.LARGE_CAP_MIN:,} Cr',
                        'mid_cap': f'₹{engine.MID_CAP_MIN:,}–{engine.LARGE_CAP_MIN:,} Cr',
                        'small_cap': f'< ₹{engine.MID_CAP_MIN:,} Cr'
                    },
                    'risk_buckets': {
                        'safe': '50% Large-cap + 50% Mid-cap',
                        'high_risk': '50% Mid-cap + 50% Small-cap'
                    },
                    'description': 'Allocate stocks based on risk appetite'
                },
                'step3_entry_rules': {
                    'technical_conditions': [
                        'Price above 20-day EMA',
                        'Price above 50-day EMA', 
                        'Breakout above 20-day high',
                        'Volume ≥ 1.5× average (20d)',
                        'RSI(14) between 50-70'
                    ],
                    'ml_conditions': [
                        'ML signal = BUY',
                        'Predicted return > 5%',
                        'ML confidence > 60%'
                    ],
                    'required': 'At least 4/5 technical + Valid ML signal',
                    'description': 'Enter only when momentum is confirmed'
                },
                'step4_exit_rules': {
                    'profit_targets': {
                        'target_1': 'Sell 50% at +5%',
                        'target_2': 'Sell remaining 50% at +10%'
                    },
                    'risk_management': {
                        'stop_loss': f'{engine.exit_rules.stop_loss_percent*100}% below entry',
                        'time_stop': f'{engine.exit_rules.max_holding_days} days maximum',
                        'trailing_stop': f'{engine.exit_rules.trailing_stop_percent*100}% below current price'
                    },
                    'description': 'Protect capital and lock profits within 10 days'
                }
            },
            'fyers_integration': {
                'status': 'connected' if engine.fyers_connector else 'disconnected',
                'data_sources': ['Real-time quotes', 'Historical data', 'Technical indicators'],
                'ml_integration': 'Existing ML API maintained'
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting strategy config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@portfolio_bp.route('/backtest', methods=['POST'])
@login_required
def api_backtest_strategy():
    """Backtest the strategy with historical data."""
    try:
        data = request.get_json() or {}
        
        # This would implement backtesting logic
        # For now, return mock backtest results
        
        return jsonify({
            'success': True,
            'backtest_results': {
                'period': '2024-01-01 to 2024-12-31',
                'total_trades': 45,
                'winning_trades': 28,
                'losing_trades': 17,
                'win_rate': 62.2,
                'avg_return_per_trade': 3.8,
                'max_drawdown': -8.5,
                'sharpe_ratio': 1.45,
                'total_return': 18.7,
                'note': 'Backtest feature coming soon with historical data'
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in backtesting: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Health check endpoint
@portfolio_bp.route('/health', methods=['GET'])
def api_portfolio_health():
    """Portfolio strategy engine health check."""
    try:
        return jsonify({
            'success': True,
            'service': 'Portfolio Strategy Engine',
            'version': '1.0',
            'timestamp': datetime.utcnow().isoformat(),
            'strategy_steps': [
                'Step 1: Filtering (Remove junk stocks)',
                'Step 2: Risk Allocation (Safe vs High Risk)',
                'Step 3: Entry Rules (Momentum validation)', 
                'Step 4: Exit Rules (Profit targets & stop losses)'
            ],
            'features': [
                'FYERS broker integration',
                'Existing ML API integration',
                'Real-time position monitoring',
                '10-day holding period with profit targets'
            ]
        })
        
    except Exception as e:
        logger.error(f"Error in portfolio health check: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
