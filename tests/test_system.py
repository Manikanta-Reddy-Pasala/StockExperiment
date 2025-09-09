"""
System tests for the Automated Trading System
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config_manager import ConfigManager
from src.broker.fyers_connector import FyersConnector
from src.simulator.simulator import Simulator
from src.selector.selector_engine import SelectorEngine
from src.selector.momentum_strategy import MomentumStrategy
from src.risk.risk_manager import RiskManager
from src.order.order_router import OrderRouter, Order, OrderType, ProductType
from src.scheduler.trading_scheduler import TradingScheduler
from src.reporting.dashboard import DashboardReporter, AlertManager
from src.compliance.compliance_logger import ComplianceLogger
from src.email_alerts.email_alerts import EmailAlertingSystem


class TestSystemComponents(unittest.TestCase):
    """Test system components integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal config for testing
        self.config = {
            'system': {'log_level': 'DEBUG'},
            'trading': {
                'market_open': '09:15',
                'market_close': '15:30'
            },
            'risk': {
                'max_capital_per_trade': 0.01,
                'max_concurrent_trades': 10
            },
            'email': {
                'smtp_host': 'smtp.test.com',
                'smtp_port': 587,
                'sender_email': 'test@example.com',
                'recipients': ['test@example.com']
            }
        }
    
    def test_config_manager(self):
        """Test configuration manager."""
        # Test with development config
        config_manager = ConfigManager("development")
        
        # Verify some base configuration values
        self.assertEqual(config_manager.get("system.name"), "Automated Trading System")
        self.assertEqual(config_manager.get("trading.modes.development.live_trading"), False)
    
    def test_simulator(self):
        """Test simulator functionality."""
        simulator = Simulator(initial_balance=100000)
        
        # Set some market data
        simulator.set_market_data({"INFY": 1500.0, "RELIANCE": 2400.0})
        
        # Test profile
        profile = simulator.get_profile()
        self.assertEqual(profile["status"], "success")
        
        # Test placing an order
        order_params = {
            'tradingsymbol': 'INFY',
            'quantity': 10,
            'order_type': 'MARKET',
            'transaction_type': 'BUY',
            'product': 'MIS'
        }
        
        result = simulator.place_order(order_params)
        self.assertEqual(result["status"], "success")
        self.assertIn("order_id", result["data"])
    
    def test_selector_engine(self):
        """Test selector engine with momentum strategy."""
        selector = SelectorEngine()
        
        # Check that default strategies are registered
        strategies = selector.get_available_strategies()
        self.assertIn("MomentumStrategy", strategies)
        self.assertIn("BreakoutStrategy", strategies)
        
        # Set active strategy
        selector.set_active_strategy("MomentumStrategy")
        active = selector.get_active_strategy()
        self.assertIsNotNone(active)
        self.assertEqual(active.name, "MomentumStrategy")
    
    def test_risk_manager(self):
        """Test risk manager functionality."""
        risk_manager = RiskManager(self.config['risk'])
        risk_manager.set_total_equity(1000000)
        
        # Test position size calculation
        position_size = risk_manager.calculate_position_size(atr=10.0)
        self.assertGreater(position_size, 0)
        
        # Test trade limits
        approval, message = risk_manager.check_trade_limits("INFY", 100)
        self.assertTrue(approval)
    
    def test_order_router(self):
        """Test order router with simulator."""
        simulator = Simulator()
        simulator.set_market_data({"INFY": 1500.0})
        
        order_router = OrderRouter(simulator)
        
        # Create an order
        order = Order(
            symbol="INFY",
            quantity=10,
            order_type=OrderType.MARKET,
            transaction_type="BUY",
            product_type=ProductType.MIS
        )
        
        # Place the order
        order_id = order_router.place_order(order)
        
        # Check that order was placed
        self.assertIsNotNone(order_id)
        self.assertEqual(order.status.name, "PLACED")
    
    def test_scheduler(self):
        """Test scheduler functionality."""
        scheduler = TradingScheduler(self.config)
        
        # Test market open check
        is_open = scheduler.is_market_open()
        # This will depend on current time, so we just verify it returns a boolean
        self.assertIsInstance(is_open, bool)
        
        # Test holiday management
        scheduler.add_holiday("2023-01-26")
        self.assertIn("2023-01-26", scheduler.holidays)
    
    def test_compliance_logger(self):
        """Test compliance logger."""
        # Use an in-memory database for testing (PostgreSQL-compatible for testing)
        from src.datastore.database import DatabaseManager
        db_manager = DatabaseManager("postgresql://test:test@localhost/test")
        # For testing purposes, we'll mock the database operations
        # In a real test environment, you would use a test PostgreSQL instance
        
        # Since we can't connect to a real database in this test, we'll skip table creation
        # db_manager.create_tables()
        
        compliance_logger = ComplianceLogger(db_manager)
        
        # Mock the session to avoid database connection
        with patch.object(compliance_logger.db_manager, 'get_session') as mock_session:
            mock_session.return_value.__enter__.return_value = Mock()
            
            # Log an event
            event_id = compliance_logger.log_event(
                module="TEST",
                event_type="TEST_EVENT",
                message="Test compliance event"
            )
            
            # Verify event ID is returned
            self.assertIsNotNone(event_id)
            self.assertIsInstance(event_id, str)
    
    @patch("smtplib.SMTP")
    def test_email_alerting(self, mock_smtp):
        """Test email alerting system."""
        # Mock the SMTP connection
        mock_smtp_instance = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_smtp_instance
        
        email_system = EmailAlertingSystem(self.config)
        
        # Test alert deduplication
        result1 = email_system.send_alert(
            alert_type="TEST",
            message="Test message",
            deduplicate=True
        )
        
        result2 = email_system.send_alert(
            alert_type="TEST",
            message="Test message",
            deduplicate=True
        )
        
        # Both should succeed (second one deduplicated)
        self.assertTrue(result1)
        self.assertTrue(result2)


if __name__ == '__main__':
    unittest.main()