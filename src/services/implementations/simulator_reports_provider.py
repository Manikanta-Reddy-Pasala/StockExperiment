"""
Simulator Reports Provider - Basic stub implementation
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import random
from ..interfaces.reports_interface import IReportsProvider, ReportType, ReportFormat


class SimulatorReportsProvider(IReportsProvider):
    def generate_pnl_report(self, user_id: int, start_date: datetime, 
                           end_date: datetime, report_format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        return {'success': True, 'report_id': f'RPT{random.randint(1000, 9999)}', 'message': 'Simulated report generated'}
    
    def generate_tax_report(self, user_id: int, financial_year: str, 
                          report_format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        return {'success': True, 'report_id': f'TAX{random.randint(1000, 9999)}', 'message': 'Simulated tax report generated'}
    
    def generate_portfolio_report(self, user_id: int, report_type: ReportType, 
                                 report_format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        return {'success': True, 'report_id': f'POR{random.randint(1000, 9999)}', 'message': 'Simulated portfolio report generated'}
    
    def generate_trading_summary(self, user_id: int, start_date: datetime, 
                               end_date: datetime, report_format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        return {'success': True, 'report_id': f'TRD{random.randint(1000, 9999)}', 'message': 'Simulated trading summary generated'}
    
    def get_report_history(self, user_id: int, limit: int = 50) -> Dict[str, Any]:
        return {'success': True, 'data': [], 'message': 'Simulator reports - implementation pending'}
    
    def download_report(self, user_id: int, report_id: str) -> Dict[str, Any]:
        return {'success': True, 'file_path': f'/tmp/sim_report_{report_id}.json', 'message': 'Simulated report download'}
