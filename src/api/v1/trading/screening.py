"""
Stock Screening API
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required
from api.common.decorators import api_response
from api.common.errors import InternalServerError
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

screening_bp = Blueprint('screening', __name__)

@screening_bp.route('/run', methods=['POST'])
@login_required
@api_response
def run_screening():
    """Run stock screening process."""
    try:
        from screening.stock_screener import StockScreener
        
        screener = StockScreener()
        screened_stocks = screener.run_daily_screening()
        
        return {
            'screened_stocks': screened_stocks,
            'count': len(screened_stocks),
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to run screening: {str(e)}")
        raise InternalServerError(f"Failed to run screening: {str(e)}")

@screening_bp.route('/criteria', methods=['GET'])
@login_required
@api_response
def get_screening_criteria():
    """Get current screening criteria."""
    try:
        from screening.stock_screener import StockScreener
        
        screener = StockScreener()
        criteria = screener.get_screening_criteria()
        
        return criteria
    except Exception as e:
        logger.error(f"Failed to get screening criteria: {str(e)}")
        raise InternalServerError(f"Failed to get screening criteria: {str(e)}")

@screening_bp.route('/criteria', methods=['PUT'])
@login_required
@api_response
def update_screening_criteria():
    """Update screening criteria."""
    try:
        data = request.get_json()
        
        from screening.stock_screener import StockScreener
        screener = StockScreener()
        result = screener.update_screening_criteria(data)
        
        return result
    except Exception as e:
        logger.error(f"Failed to update screening criteria: {str(e)}")
        raise InternalServerError(f"Failed to update screening criteria: {str(e)}")

@screening_bp.route('/history', methods=['GET'])
@login_required
@api_response
def get_screening_history():
    """Get screening history."""
    try:
        from datastore.database import get_database_manager
        from datastore.models import SuggestedStock
        
        db_manager = get_database_manager()
        
        with db_manager.get_session() as session:
            # Get recent screening results
            recent_screenings = session.query(SuggestedStock).order_by(
                SuggestedStock.created_at.desc()
            ).limit(50).all()
            
            history = []
            for stock in recent_screenings:
                history.append({
                    'symbol': stock.symbol,
                    'name': stock.name,
                    'screened_at': stock.created_at.isoformat(),
                    'reason': stock.reason,
                    'score': stock.score
                })
            
            return history
    except Exception as e:
        logger.error(f"Failed to get screening history: {str(e)}")
        raise InternalServerError(f"Failed to get screening history: {str(e)}")
