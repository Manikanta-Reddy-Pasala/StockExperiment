"""
Updated Dashboard Service

This service now uses the UnifiedBrokerService to provide multi-broker support
while maintaining backward compatibility with existing code.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from .unified_broker_service import get_unified_broker_service

logger = logging.getLogger(__name__)


class MultiBrokerDashboardService:
    """
    Updated dashboard service that uses the unified broker service
    to provide multi-broker support based on user settings.
    """
    
    def __init__(self):
        self.unified_service = get_unified_broker_service()
    
    def get_dashboard_metrics(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive dashboard metrics using user's selected broker."""
        try:
            # Get all dashboard data using the unified service
            market_overview = self.unified_service.get_market_overview(user_id)
            portfolio_summary = self.unified_service.get_portfolio_summary(user_id)
            account_balance = self.unified_service.get_account_balance(user_id)
            recent_activity = self.unified_service.get_recent_activity(user_id, 5)
            
            # Combine all metrics into a single response
            metrics = {
                'market_overview': market_overview.get('data', []),
                'portfolio_summary': portfolio_summary.get('data', {}),
                'account_balance': account_balance.get('data', {}),
                'recent_activity': recent_activity.get('data', []),
                'success': True,
                'last_updated': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching dashboard metrics for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
    
    def get_portfolio_holdings(self, user_id: int) -> Dict[str, Any]:
        """Get portfolio holdings using user's selected broker."""
        return self.unified_service.get_top_holdings(user_id, 10)
    
    def get_pending_orders(self, user_id: int) -> Dict[str, Any]:
        """Get pending orders using user's selected broker."""
        return self.unified_service.get_pending_orders(user_id)
    
    def get_recent_orders(self, user_id: int, limit: int = 10) -> Dict[str, Any]:
        """Get recent orders using user's selected broker."""
        return self.unified_service.get_recent_activity(user_id, limit)
    
    def get_portfolio_performance(self, user_id: int, period: str = '1W') -> Dict[str, Any]:
        """Get portfolio performance data using user's selected broker."""
        return self.unified_service.get_portfolio_performance(user_id, period)
    
    # Maintain backward compatibility methods
    def get_market_overview(self, user_id: int) -> Dict[str, Any]:
        """Get market overview (backward compatibility)."""
        return self.unified_service.get_market_overview(user_id)
    
    def get_watchlist_quotes(self, user_id: int, symbols: List[str] = None) -> Dict[str, Any]:
        """Get watchlist quotes (backward compatibility)."""
        return self.unified_service.get_watchlist_quotes(user_id, symbols)


# Global service instance
_multi_broker_dashboard_service = None

def get_multi_broker_dashboard_service() -> MultiBrokerDashboardService:
    """Get the global multi-broker dashboard service instance."""
    global _multi_broker_dashboard_service
    if _multi_broker_dashboard_service is None:
        _multi_broker_dashboard_service = MultiBrokerDashboardService()
    return _multi_broker_dashboard_service


# Backward compatibility - update existing service to use multi-broker version
def get_dashboard_service():
    """Backward compatibility function."""
    return get_multi_broker_dashboard_service()
