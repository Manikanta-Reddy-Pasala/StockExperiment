"""
Trading Execution API
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required
from api.common.decorators import api_response, validate_json
from api.common.errors import ValidationError, InternalServerError
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

execution_bp = Blueprint('execution', __name__)

@execution_bp.route('/dry-run', methods=['POST'])
@login_required
@api_response
def run_dry_run():
    """Run dry run execution."""
    try:
        from trading_engine.multi_user_trading_engine import MultiUserTradingEngine
        
        trading_engine = MultiUserTradingEngine()
        dry_run_results = trading_engine.run_dry_run()
        
        return {
            'dry_run_results': dry_run_results,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to run dry run: {str(e)}")
        raise InternalServerError(f"Failed to run dry run: {str(e)}")

@execution_bp.route('/complete-workflow', methods=['POST'])
@login_required
@api_response
def run_complete_workflow():
    """Run complete trading workflow."""
    try:
        from trading_engine.multi_user_trading_engine import MultiUserTradingEngine
        
        trading_engine = MultiUserTradingEngine()
        workflow_results = trading_engine.run_complete_workflow()
        
        return {
            'workflow_results': workflow_results,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to run complete workflow: {str(e)}")
        raise InternalServerError(f"Failed to run complete workflow: {str(e)}")

@execution_bp.route('/start-scheduled', methods=['POST'])
@login_required
@api_response
def start_scheduled_execution():
    """Start scheduled execution."""
    try:
        from trading_engine.multi_user_trading_engine import MultiUserTradingEngine
        
        trading_engine = MultiUserTradingEngine()
        result = trading_engine.start_scheduled_execution()
        
        return {
            'message': 'Scheduled execution started',
            'status': result,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to start scheduled execution: {str(e)}")
        raise InternalServerError(f"Failed to start scheduled execution: {str(e)}")

@execution_bp.route('/stop-scheduled', methods=['POST'])
@login_required
@api_response
def stop_scheduled_execution():
    """Stop scheduled execution."""
    try:
        from trading_engine.multi_user_trading_engine import MultiUserTradingEngine
        
        trading_engine = MultiUserTradingEngine()
        result = trading_engine.stop_scheduled_execution()
        
        return {
            'message': 'Scheduled execution stopped',
            'status': result,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to stop scheduled execution: {str(e)}")
        raise InternalServerError(f"Failed to stop scheduled execution: {str(e)}")

@execution_bp.route('/status', methods=['GET'])
@login_required
@api_response
def get_execution_status():
    """Get execution status."""
    try:
        from trading_engine.multi_user_trading_engine import MultiUserTradingEngine
        
        trading_engine = MultiUserTradingEngine()
        status = trading_engine.get_execution_status()
        
        return status
    except Exception as e:
        logger.error(f"Failed to get execution status: {str(e)}")
        raise InternalServerError(f"Failed to get execution status: {str(e)}")

@execution_bp.route('/cleanup-dry-run', methods=['POST'])
@login_required
@api_response
def cleanup_dry_run():
    """Cleanup dry run data."""
    try:
        from trading_engine.multi_user_trading_engine import MultiUserTradingEngine
        
        trading_engine = MultiUserTradingEngine()
        result = trading_engine.cleanup_dry_run()
        
        return {
            'message': 'Dry run cleanup completed',
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to cleanup dry run: {str(e)}")
        raise InternalServerError(f"Failed to cleanup dry run: {str(e)}")
