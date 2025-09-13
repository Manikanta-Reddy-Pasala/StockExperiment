"""
Zerodha Portfolio Provider Implementation (Stub)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from ..interfaces.portfolio_interface import IPortfolioProvider


class ZerodhaPortfolioProvider(IPortfolioProvider):
    def get_holdings(self, user_id: int) -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available', 'data': []}
    
    def get_positions(self, user_id: int) -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available', 'data': []}
    
    def get_portfolio_summary(self, user_id: int) -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available', 'data': {}}
    
    def get_portfolio_allocation(self, user_id: int) -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available', 'data': []}
    
    def get_portfolio_performance(self, user_id: int, period: str = '1M') -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available', 'data': {}}
    
    def get_dividend_history(self, user_id: int, start_date: datetime = None, 
                           end_date: datetime = None) -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available', 'data': []}
    
    def get_portfolio_risk_metrics(self, user_id: int) -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available', 'data': {}}
