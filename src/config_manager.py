"""
Configuration Manager for the Automated Trading System
Uses environment variables with sensible defaults
"""
import os
from typing import Dict, Any


class ConfigManager:
    """Manages configuration using environment variables with defaults."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables with defaults."""
        return {
            'system': {
                'name': os.getenv('SYSTEM_NAME', 'Automated Trading System'),
                'version': os.getenv('SYSTEM_VERSION', '1.0.0'),
                'log_level': os.getenv('LOG_LEVEL', 'INFO')
            },
            'trading': {
                'market_open': os.getenv('MARKET_OPEN', '09:15'),
                'market_close': os.getenv('MARKET_CLOSE', '15:30'),
                'pre_open_start': os.getenv('PRE_OPEN_START', '09:00'),
                'pre_open_end': os.getenv('PRE_OPEN_END', '09:15'),
                'live_trading': os.getenv('LIVE_TRADING', 'false').lower() == 'true'
            },
            'risk': {
                'max_capital_per_trade': float(os.getenv('MAX_CAPITAL_PER_TRADE', '0.01')),
                'max_concurrent_trades': int(os.getenv('MAX_CONCURRENT_TRADES', '10')),
                'daily_loss_limit': float(os.getenv('DAILY_LOSS_LIMIT', '0.02')),
                'single_name_exposure_limit': float(os.getenv('SINGLE_NAME_EXPOSURE_LIMIT', '0.05'))
            },
            'momentum': {
                'universe': os.getenv('MOMENTUM_UNIVERSE', 'NIFTY_100'),
                'rebalance_cadence': os.getenv('MOMENTUM_REBALANCE_CADENCE', 'daily'),
                'lookback_period': int(os.getenv('MOMENTUM_LOOKBACK_PERIOD', '20')),
                'min_volume_filter': int(os.getenv('MOMENTUM_MIN_VOLUME_FILTER', '100000'))
            },
            'database': {
                'url': os.getenv('DATABASE_URL', 'postgresql://trader:trader_password@database:5432/trading_system'),
                'host': os.getenv('POSTGRES_HOST', 'database'),
                'port': int(os.getenv('POSTGRES_PORT', '5432')),
                'name': os.getenv('POSTGRES_DB', 'trading_system'),
                'user': os.getenv('POSTGRES_USER', 'trader'),
                'password': os.getenv('POSTGRES_PASSWORD', 'trader_password')
            },
            'web': {
                'host': os.getenv('WEB_HOST', '0.0.0.0'),
                'port': int(os.getenv('WEB_PORT', '5001')),
                'debug': os.getenv('WEB_DEBUG', 'false').lower() == 'true'
            },
            'email': {
                'smtp_host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                'use_tls': os.getenv('SMTP_USE_TLS', 'true').lower() == 'true',
                'username': os.getenv('SMTP_USERNAME', ''),
                'password': os.getenv('SMTP_PASSWORD', '')
            },
            'broker': {
                'fyers_client_id': os.getenv('FYERS_CLIENT_ID', 'your_client_id'),
                'fyers_access_token': os.getenv('FYERS_ACCESS_TOKEN', 'your_access_token')
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path (str): Dot-separated path to the configuration value
            default (Any): Default value if key is not found
            
        Returns:
            Any: Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path (str): Dot-separated path to the configuration value
            value (Any): Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value


# Global configuration manager instance
config_manager = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigManager: Configuration manager instance
    """
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager