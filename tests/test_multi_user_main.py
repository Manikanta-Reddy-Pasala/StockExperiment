"""
Tests for Multi-User Main Application
"""
import pytest
import sys
import os
import uuid
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datastore.database import get_database_manager
from datastore.models import User
from main import initialize_components, get_database_url


@pytest.fixture
def db_manager():
    """Create a database manager for testing."""
    # Use a unique in-memory database for each test
    database_url = f"sqlite:///:memory:{uuid.uuid4().hex}"
    db_manager = get_database_manager(database_url)
    db_manager.create_tables()
    
    # Reset the global db_manager to ensure fresh instances
    import datastore.database
    datastore.database.db_manager = None
    
    yield db_manager
    
    # Clean up
    datastore.database.db_manager = None


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'risk': {
            'max_capital_per_trade': 0.01,
            'max_concurrent_trades': 10,
            'daily_loss_limit': 0.02,
            'single_name_exposure_limit': 0.05
        },
        'trading': {
            'market_open': '09:15',
            'market_close': '15:30'
        },
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'test_db',
            'user': 'test_user',
            'password': 'test_password'
        }
    }


class TestMainApplication:
    """Test main application functionality."""
    
    def test_get_database_url_from_environment(self, test_config):
        """Test getting database URL from environment variable."""
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://env_user:env_pass@env_host:5432/env_db'}):
            url = get_database_url(test_config)
            assert url == 'postgresql://env_user:env_pass@env_host:5432/env_db'
    
    def test_get_database_url_from_config(self, test_config):
        """Test getting database URL from configuration."""
        with patch.dict(os.environ, {}, clear=True):
            url = get_database_url(test_config)
            expected_url = 'postgresql://test_user:test_password@localhost:5432/test_db'
            assert url == expected_url
    
    def test_initialize_components_single_user(self, test_config):
        """Test initializing components in single-user mode."""
        with patch('main.get_database_manager') as mock_db_manager:
            mock_db_manager.return_value = Mock()
            
            with patch('main.get_data_manager') as mock_data_manager:
                mock_data_manager.return_value = Mock()
                
                with patch('main.Simulator') as mock_simulator:
                    mock_simulator.return_value = Mock()
                    
                    with patch('main.SelectorEngine') as mock_selector:
                        mock_selector.return_value = Mock()
                        
                        with patch('main.RiskManager') as mock_risk:
                            mock_risk.return_value = Mock()
                            
                            with patch('main.OrderRouter') as mock_order_router:
                                mock_order_router.return_value = Mock()
                                
                                with patch('main.TradingScheduler') as mock_scheduler:
                                    mock_scheduler.return_value = Mock()
                                    
                                    with patch('main.DashboardReporter') as mock_dashboard:
                                        mock_dashboard.return_value = Mock()
                                        
                                        with patch('main.AlertManager') as mock_alert:
                                            mock_alert.return_value = Mock()
                                            
                                            with patch('main.ComplianceLogger') as mock_compliance:
                                                mock_compliance.return_value = Mock()
                                                
                                                with patch('main.EmailAlertingSystem') as mock_email:
                                                    mock_email.return_value = Mock()
                                                    
                                                    # Initialize components in single-user mode
                                                    components = initialize_components(test_config, 'development', multi_user=False)
                                                    
                                                    # Verify components are initialized
                                                    assert 'config' in components
                                                    assert 'db_manager' in components
                                                    assert 'data_manager' in components
                                                    assert 'broker_connector' in components
                                                    assert 'selector_engine' in components
                                                    assert 'risk_manager' in components
                                                    assert 'order_router' in components
                                                    assert 'scheduler' in components
                                                    assert 'trading_engine' in components
                                                    assert 'dashboard_reporter' in components
                                                    assert 'alert_manager' in components
                                                    assert 'compliance_logger' in components
                                                    assert 'email_alerting' in components
                                                    
                                                    # Verify trading_engine is None for single-user mode
                                                    assert components['trading_engine'] is None
                                                    
                                                    # Verify scheduler is TradingScheduler
                                                    assert components['scheduler'] is not None
    
    def test_initialize_components_multi_user(self, test_config):
        """Test initializing components in multi-user mode."""
        with patch('main.get_database_manager') as mock_db_manager:
            mock_db_manager.return_value = Mock()
            
            with patch('main.get_data_manager') as mock_data_manager:
                mock_data_manager.return_value = Mock()
                
                with patch('main.Simulator') as mock_simulator:
                    mock_simulator.return_value = Mock()
                    
                    with patch('main.SelectorEngine') as mock_selector:
                        mock_selector.return_value = Mock()
                        
                        with patch('main.RiskManager') as mock_risk:
                            mock_risk.return_value = Mock()
                            
                            with patch('main.OrderRouter') as mock_order_router:
                                mock_order_router.return_value = Mock()
                                
                                with patch('main.MultiUserTradingEngine') as mock_trading_engine:
                                    mock_trading_engine.return_value = Mock()
                                    
                                    with patch('main.MultiUserTradingScheduler') as mock_scheduler:
                                        mock_scheduler.return_value = Mock()
                                        
                                        with patch('main.DashboardReporter') as mock_dashboard:
                                            mock_dashboard.return_value = Mock()
                                            
                                            with patch('main.AlertManager') as mock_alert:
                                                mock_alert.return_value = Mock()
                                                
                                                with patch('main.ComplianceLogger') as mock_compliance:
                                                    mock_compliance.return_value = Mock()
                                                    
                                                    with patch('main.EmailAlertingSystem') as mock_email:
                                                        mock_email.return_value = Mock()
                                                        
                                                        # Initialize components in multi-user mode
                                                        components = initialize_components(test_config, 'development', multi_user=True)
                                                        
                                                        # Verify components are initialized
                                                        assert 'config' in components
                                                        assert 'db_manager' in components
                                                        assert 'data_manager' in components
                                                        assert 'broker_connector' in components
                                                        assert 'selector_engine' in components
                                                        assert 'risk_manager' in components
                                                        assert 'order_router' in components
                                                        assert 'scheduler' in components
                                                        assert 'trading_engine' in components
                                                        assert 'dashboard_reporter' in components
                                                        assert 'alert_manager' in components
                                                        assert 'compliance_logger' in components
                                                        assert 'email_alerting' in components
                                                        
                                                        # Verify trading_engine is initialized for multi-user mode
                                                        assert components['trading_engine'] is not None
                                                        
                                                        # Verify scheduler is MultiUserTradingScheduler
                                                        assert components['scheduler'] is not None
    
    def test_initialize_components_production_mode(self, test_config):
        """Test initializing components in production mode."""
        with patch('main.get_database_manager') as mock_db_manager:
            mock_db_manager.return_value = Mock()
            
            with patch('main.get_data_manager') as mock_data_manager:
                mock_data_manager.return_value = Mock()
                
                with patch('main.FyersConnector') as mock_fyers:
                    mock_fyers.return_value = Mock()
                    
                    with patch('main.SelectorEngine') as mock_selector:
                        mock_selector.return_value = Mock()
                        
                        with patch('main.RiskManager') as mock_risk:
                            mock_risk.return_value = Mock()
                            
                            with patch('main.OrderRouter') as mock_order_router:
                                mock_order_router.return_value = Mock()
                                
                                with patch('main.MultiUserTradingEngine') as mock_trading_engine:
                                    mock_trading_engine.return_value = Mock()
                                    
                                    with patch('main.MultiUserTradingScheduler') as mock_scheduler:
                                        mock_scheduler.return_value = Mock()
                                        
                                        with patch('main.DashboardReporter') as mock_dashboard:
                                            mock_dashboard.return_value = Mock()
                                            
                                            with patch('main.AlertManager') as mock_alert:
                                                mock_alert.return_value = Mock()
                                                
                                                with patch('main.ComplianceLogger') as mock_compliance:
                                                    mock_compliance.return_value = Mock()
                                                    
                                                    with patch('main.EmailAlertingSystem') as mock_email:
                                                        mock_email.return_value = Mock()
                                                        
                                                        with patch.dict(os.environ, {
                                                            'FYERS_CLIENT_ID': 'test_client_id',
                                                            'FYERS_ACCESS_TOKEN': 'test_access_token'
                                                        }):
                                                            # Initialize components in production mode
                                                            components = initialize_components(test_config, 'production', multi_user=True)
                                                            
                                                            # Verify FyersConnector was used instead of Simulator
                                                            mock_fyers.assert_called_once_with(
                                                                client_id='test_client_id',
                                                                access_token='test_access_token'
                                                            )
                                                            
                                                            # Verify broker_connector is FyersConnector
                                                            assert components['broker_connector'] is not None


class TestMainApplicationIntegration:
    """Integration tests for the main application."""
    
    def test_multi_user_workflow(self, test_config):
        """Test a complete multi-user workflow."""
        with patch('main.get_database_manager') as mock_db_manager:
            mock_db_manager.return_value = Mock()
            
            with patch('main.get_data_manager') as mock_data_manager:
                mock_data_manager.return_value = Mock()
                
                with patch('main.Simulator') as mock_simulator:
                    mock_simulator.return_value = Mock()
                    
                    with patch('main.SelectorEngine') as mock_selector:
                        mock_selector.return_value = Mock()
                        
                        with patch('main.RiskManager') as mock_risk:
                            mock_risk.return_value = Mock()
                            
                            with patch('main.OrderRouter') as mock_order_router:
                                mock_order_router.return_value = Mock()
                                
                                with patch('main.MultiUserTradingEngine') as mock_trading_engine:
                                    mock_trading_engine_instance = Mock()
                                    mock_trading_engine.return_value = mock_trading_engine_instance
                                    
                                    with patch('main.MultiUserTradingScheduler') as mock_scheduler:
                                        mock_scheduler_instance = Mock()
                                        mock_scheduler.return_value = mock_scheduler_instance
                                        
                                        with patch('main.DashboardReporter') as mock_dashboard:
                                            mock_dashboard.return_value = Mock()
                                            
                                            with patch('main.AlertManager') as mock_alert:
                                                mock_alert.return_value = Mock()
                                                
                                                with patch('main.ComplianceLogger') as mock_compliance:
                                                    mock_compliance.return_value = Mock()
                                                    
                                                    with patch('main.EmailAlertingSystem') as mock_email:
                                                        mock_email.return_value = Mock()
                                                        
                                                        # Initialize components
                                                        components = initialize_components(test_config, 'development', multi_user=True)
                                                        
                                                        # Verify trading engine was created
                                                        mock_trading_engine.assert_called_once_with(test_config, 'development')
                                                        
                                                        # Verify scheduler was created with trading engine
                                                        mock_scheduler.assert_called_once_with(test_config, mock_trading_engine_instance)
                                                        
                                                        # Verify components are properly connected
                                                        assert components['trading_engine'] == mock_trading_engine_instance
                                                        assert components['scheduler'] == mock_scheduler_instance
    
    def test_single_user_workflow(self, test_config):
        """Test a complete single-user workflow."""
        with patch('main.get_database_manager') as mock_db_manager:
            mock_db_manager.return_value = Mock()
            
            with patch('main.get_data_manager') as mock_data_manager:
                mock_data_manager.return_value = Mock()
                
                with patch('main.Simulator') as mock_simulator:
                    mock_simulator.return_value = Mock()
                    
                    with patch('main.SelectorEngine') as mock_selector:
                        mock_selector.return_value = Mock()
                        
                        with patch('main.RiskManager') as mock_risk:
                            mock_risk.return_value = Mock()
                            
                            with patch('main.OrderRouter') as mock_order_router:
                                mock_order_router.return_value = Mock()
                                
                                with patch('main.TradingScheduler') as mock_scheduler:
                                    mock_scheduler_instance = Mock()
                                    mock_scheduler.return_value = mock_scheduler_instance
                                    
                                    with patch('main.DashboardReporter') as mock_dashboard:
                                        mock_dashboard.return_value = Mock()
                                        
                                        with patch('main.AlertManager') as mock_alert:
                                            mock_alert.return_value = Mock()
                                            
                                            with patch('main.ComplianceLogger') as mock_compliance:
                                                mock_compliance.return_value = Mock()
                                                
                                                with patch('main.EmailAlertingSystem') as mock_email:
                                                    mock_email.return_value = Mock()
                                                    
                                                    # Initialize components
                                                    components = initialize_components(test_config, 'development', multi_user=False)
                                                    
                                                    # Verify TradingScheduler was created
                                                    mock_scheduler.assert_called_once_with(test_config)
                                                    
                                                    # Verify order router was set
                                                    mock_scheduler_instance.set_order_router.assert_called_once()
                                                    
                                                    # Verify trading_engine is None
                                                    assert components['trading_engine'] is None
                                                    
                                                    # Verify scheduler is TradingScheduler
                                                    assert components['scheduler'] == mock_scheduler_instance


class TestMainApplicationErrorHandling:
    """Test error handling in the main application."""
    
    def test_database_initialization_error(self, test_config):
        """Test handling of database initialization errors."""
        with patch('main.get_database_manager', side_effect=Exception("Database error")):
            with pytest.raises(Exception, match="Database error"):
                initialize_components(test_config, 'development', multi_user=True)
    
    def test_component_initialization_error(self, test_config):
        """Test handling of component initialization errors."""
        with patch('main.get_database_manager') as mock_db_manager:
            mock_db_manager.return_value = Mock()
            
            with patch('main.get_data_manager', side_effect=Exception("Data manager error")):
                with pytest.raises(Exception, match="Data manager error"):
                    initialize_components(test_config, 'development', multi_user=True)
    
    def test_missing_environment_variables(self, test_config):
        """Test handling of missing environment variables in production mode."""
        with patch('main.get_database_manager') as mock_db_manager:
            mock_db_manager.return_value = Mock()
            
            with patch('main.get_data_manager') as mock_data_manager:
                mock_data_manager.return_value = Mock()
                
                with patch.dict(os.environ, {}, clear=True):
                    # Should use default values when environment variables are missing
                    components = initialize_components(test_config, 'production', multi_user=True)
                    
                    # Verify components were still initialized
                    assert 'broker_connector' in components
                    assert components['broker_connector'] is not None
