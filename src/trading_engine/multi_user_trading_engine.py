"""
Multi-User Trading Engine for the Automated Trading System
"""
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
from datastore.database import get_database_manager
from datastore.models import User, Configuration
from selector.selector_engine import SelectorEngine
from risk.risk_manager import RiskManager
from order.order_router import OrderRouter
from simulator.simulator import Simulator
try:
    from broker.fyers_connector import FyersConnector
except ImportError:
    FyersConnector = None
from data_provider.data_manager import get_data_manager
from compliance.compliance_logger import ComplianceLogger
from email_alerts.email_alerts import EmailAlertingSystem


class UserTradingSession:
    """Represents a trading session for a single user."""
    
    def __init__(self, user: User, config: Dict[str, Any], mode: str = 'development'):
        """
        Initialize a user trading session.
        
        Args:
            user (User): User object
            config (Dict[str, Any]): Configuration dictionary
            mode (str): Trading mode ('development' or 'production')
        """
        self.user = user
        self.user_id = user.id
        self.username = user.username
        self.config = config
        self.mode = mode
        self.is_active = True
        self.last_activity = datetime.utcnow()
        
        # Initialize components for this user
        self._initialize_components()
        
        # Load user-specific configurations
        self._load_user_configurations()
        
        # Initialize trading state
        self.trading_state = {
            'last_scan_time': None,
            'selected_stocks': [],
            'active_orders': [],
            'positions': {},
            'daily_pnl': 0.0,
            'risk_metrics': {}
        }
    
    def _initialize_components(self):
        """Initialize trading components for this user."""
        # Initialize data manager
        self.data_manager = get_data_manager()
        
        # Initialize broker connector based on mode
        if self.mode == 'production':
            # In production, use real Fyers connector if available
            if FyersConnector is not None:
                import os
                client_id = os.environ.get('FYERS_CLIENT_ID', 'your_client_id')
                access_token = os.environ.get('FYERS_ACCESS_TOKEN', 'your_access_token')
                self.broker_connector = FyersConnector(client_id=client_id, access_token=access_token)
            else:
                # Fallback to simulator if FyersConnector is not available
                self.broker_connector = Simulator()
        else:
            # In development, use simulator
            self.broker_connector = Simulator()
        
        # Initialize other components
        self.selector_engine = SelectorEngine()
        self.selector_engine.set_data_manager(self.data_manager)
        
        # Initialize risk manager with user-specific config
        user_risk_config = self._get_user_risk_config()
        self.risk_manager = RiskManager(user_risk_config)
        
        # Initialize order router
        self.order_router = OrderRouter(self.broker_connector)
        
        # Initialize compliance logger
        db_manager = get_database_manager()
        self.compliance_logger = ComplianceLogger(db_manager)
        
        # Initialize email alerting
        self.email_alerting = EmailAlertingSystem(self.config)
    
    def _get_user_risk_config(self) -> Dict[str, Any]:
        """Get user-specific risk configuration."""
        # Start with default risk config
        risk_config = self.config.get('risk', {}).copy()
        
        # Override with user-specific settings if available
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            user_configs = session.query(Configuration).filter(
                Configuration.user_id == self.user_id,
                Configuration.key.like('risk.%')
            ).all()
            
            for config in user_configs:
                key = config.key.replace('risk.', '')
                risk_config[key] = config.value
        
        return risk_config
    
    def _load_user_configurations(self):
        """Load user-specific configurations from database."""
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            user_configs = session.query(Configuration).filter(
                Configuration.user_id == self.user_id
            ).all()
            
            for config in user_configs:
                # Update the config dictionary with user-specific values
                self.config[config.key] = config.value
    
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = datetime.utcnow()
        
        # Update user's last_activity in database
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == self.user_id).first()
            if user:
                user.last_activity = self.last_activity
                session.commit()
    
    def deactivate(self):
        """Deactivate the trading session."""
        self.is_active = False
        
        # Log session deactivation
        self.compliance_logger.log_system_event(
            event_type="USER_SESSION_DEACTIVATED",
            message=f"Trading session deactivated for user {self.username}",
            details={"user_id": self.user_id, "username": self.username}
        )
    
    def get_trading_state(self) -> Dict[str, Any]:
        """Get current trading state for this user."""
        return self.trading_state.copy()
    
    def update_trading_state(self, updates: Dict[str, Any]):
        """Update trading state for this user."""
        self.trading_state.update(updates)
        self.update_activity()


class MultiUserTradingEngine:
    """Manages trading sessions for multiple users."""
    
    def __init__(self, config: Dict[str, Any], mode: str = 'development'):
        """
        Initialize the multi-user trading engine.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            mode (str): Trading mode ('development' or 'production')
        """
        self.config = config
        self.mode = mode
        self.user_sessions: Dict[int, UserTradingSession] = {}
        self.is_running = False
        self.engine_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize database manager
        self.db_manager = get_database_manager()
        
        # Initialize global components
        self.data_manager = get_data_manager()
        
        # Load active users
        self._load_active_users()
    
    def _load_active_users(self):
        """Load all active users and create trading sessions for them."""
        with self.db_manager.get_session() as session:
            active_users = session.query(User).filter(
                User.is_active == True
            ).all()
            
            for user in active_users:
                self.create_user_session(user)
    
    def create_user_session(self, user: User) -> UserTradingSession:
        """
        Create a trading session for a user.
        
        Args:
            user (User): User object
            
        Returns:
            UserTradingSession: Created trading session
        """
        user_id = user.id
        username = user.username
        
        if user_id in self.user_sessions:
            return self.user_sessions[user_id]
        
        # Create new trading session
        session = UserTradingSession(user, self.config, self.mode)
        self.user_sessions[user_id] = session
        
        # Log session creation
        session.compliance_logger.log_system_event(
            event_type="USER_SESSION_CREATED",
            message=f"Trading session created for user {session.username}",
            details={"user_id": session.user_id, "username": session.username}
        )
        
        self.logger.info(f"Created trading session for user {username}")
        return session
    
    def remove_user_session(self, user_id: int):
        """
        Remove a user's trading session.
        
        Args:
            user_id (int): User ID
        """
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            session.deactivate()
            del self.user_sessions[user_id]
            
            self.logger.info(f"Removed trading session for user {user_id}")
    
    def get_user_session(self, user_id: int) -> Optional[UserTradingSession]:
        """
        Get a user's trading session.
        
        Args:
            user_id (int): User ID
            
        Returns:
            Optional[UserTradingSession]: User's trading session or None
        """
        return self.user_sessions.get(user_id)
    
    def get_active_users(self) -> List[User]:
        """
        Get list of active users with trading sessions.
        
        Returns:
            List[User]: List of active users
        """
        return [session.user for session in self.user_sessions.values() if session.is_active]
    
    def start_engine(self):
        """Start the multi-user trading engine."""
        if self.is_running:
            return
        
        self.is_running = True
        self.engine_thread = threading.Thread(target=self._run_engine, daemon=True)
        self.engine_thread.start()
        
        self.logger.info("Multi-user trading engine started")
    
    def stop_engine(self):
        """Stop the multi-user trading engine."""
        self.is_running = False
        if self.engine_thread:
            self.engine_thread.join()
        
        # Deactivate all user sessions
        for session in self.user_sessions.values():
            session.deactivate()
        
        self.logger.info("Multi-user trading engine stopped")
    
    def _run_engine(self):
        """Main engine loop."""
        while self.is_running:
            try:
                # Process each active user session
                for user_id, session in list(self.user_sessions.items()):
                    if not session.is_active:
                        continue
                    
                    try:
                        # Check if user session is still active
                        if not self._is_user_active(user_id):
                            session.deactivate()
                            continue
                        
                        # Update user activity
                        session.update_activity()
                        
                        # Run user-specific trading logic
                        self._process_user_trading(session)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing user {user_id}: {e}")
                        # Log the error
                        session.compliance_logger.log_system_event(
                            event_type="USER_TRADING_ERROR",
                            message=f"Error in trading logic for user {session.username}: {str(e)}",
                            details={"user_id": user_id, "username": session.username}
                        )
                
                # Sleep for a short interval
                import time
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in trading engine loop: {e}")
                import time
                time.sleep(60)
    
    def _is_user_active(self, user_id: int) -> bool:
        """Check if a user is still active."""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            return user is not None and user.is_active
    
    def _process_user_trading(self, user_session: UserTradingSession):
        """
        Process trading logic for a specific user.
        
        Args:
            user_session (UserTradingSession): User's trading session
        """
        # This is where the actual trading logic would be implemented
        # For now, we'll just log that we're processing the user
        
        # Example: Check if it's time for a market scan
        now = datetime.utcnow()
        last_scan = user_session.trading_state.get('last_scan_time')
        
        # If it's been more than 1 hour since last scan, run a new scan
        if last_scan is None or (now - last_scan).total_seconds() > 3600:
            self._run_market_scan(user_session)
            user_session.trading_state['last_scan_time'] = now
    
    def _run_market_scan(self, user_session: UserTradingSession):
        """
        Run a market scan for a specific user.
        
        Args:
            user_session (UserTradingSession): User's trading session
        """
        try:
            # Get user's active strategy
            active_strategy = user_session.selector_engine.get_active_strategy()
            if not active_strategy:
                return
            
            # Get list of symbols to scan (simplified - in real implementation, this would be more sophisticated)
            symbols = list(user_session.selector_engine.MID_CAP_STOCKS)[:10]  # Limit to 10 for demo
            
            # Run stock selection
            selected_stocks = user_session.selector_engine.select_stocks(symbols=symbols)
            
            # Update user's trading state
            user_session.update_trading_state({
                'selected_stocks': selected_stocks
            })
            
            # Log the scan
            user_session.compliance_logger.log_system_event(
                event_type="MARKET_SCAN_COMPLETED",
                message=f"Market scan completed for user {user_session.username}. Selected {len(selected_stocks)} stocks.",
                details={"user_id": user_session.user_id, "username": user_session.username}
            )
            
        except Exception as e:
            self.logger.error(f"Error running market scan for user {user_session.user_id}: {e}")
            user_session.compliance_logger.log_system_event(
                event_type="MARKET_SCAN_ERROR",
                message=f"Error in market scan for user {user_session.username}: {str(e)}",
                details={"user_id": user_session.user_id, "username": user_session.username}
            )
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get the current status of the trading engine.
        
        Returns:
            Dict[str, Any]: Engine status information
        """
        return {
            'is_running': self.is_running,
            'active_users_count': len([s for s in self.user_sessions.values() if s.is_active]),
            'total_users_count': len(self.user_sessions),
            'mode': self.mode,
            'users': [
                {
                    'user_id': session.user_id,
                    'username': session.username,
                    'is_active': session.is_active,
                    'last_activity': session.last_activity.isoformat() if session.last_activity else None
                }
                for session in self.user_sessions.values()
            ]
        }
