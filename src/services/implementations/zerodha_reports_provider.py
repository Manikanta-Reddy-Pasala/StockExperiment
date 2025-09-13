"""
Zerodha Reports Provider Implementation (Stub)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from ..interfaces.reports_interface import IReportsProvider, ReportType, ReportFormat


class ZerodhaReportsProvider(IReportsProvider):
    def generate_pnl_report(self, user_id: int, start_date: datetime, 
                           end_date: datetime, report_format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available'}
    
    def generate_tax_report(self, user_id: int, financial_year: str, 
                          report_format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available'}
    
    def generate_portfolio_report(self, user_id: int, report_type: ReportType, 
                                 report_format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available'}
    
    def generate_trading_summary(self, user_id: int, start_date: datetime, 
                               end_date: datetime, report_format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available'}
    
    def get_report_history(self, user_id: int, limit: int = 50) -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available', 'data': []}
    
    def download_report(self, user_id: int, report_id: str) -> Dict[str, Any]:
        return {'success': False, 'error': 'Zerodha implementation not available'}
