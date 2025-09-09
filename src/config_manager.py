"""
Configuration Manager for the Automated Trading System
"""
import os
import yaml
from typing import Dict, Any


class ConfigManager:
    """Manages configuration loading and access for the trading system."""
    
    def __init__(self, environment: str = "unified"):
        """
        Initialize the configuration manager.
        
        Args:
            environment (str): The environment to load configuration for 
                             (unified is the only supported environment now)
        """
        self.environment = environment
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML files."""
        # Load base configuration
        base_config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "config", 
            "base.yaml"
        )
        
        if os.path.exists(base_config_path):
            with open(base_config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        
        # Load unified configuration
        env_config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "config", 
            f"{self.environment}.yaml"
        )
        
        if os.path.exists(env_config_path):
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f) or {}
                self.config = self._merge_configs(self.config, env_config)
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base (Dict): Base configuration
            override (Dict): Override configuration
            
        Returns:
            Dict: Merged configuration
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._merge_configs(base[key], value)
            else:
                base[key] = value
        return base
    
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


def get_config_manager(environment: str = "unified") -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        environment (str): The environment to load configuration for
                          (unified is the only supported environment now)
        
    Returns:
        ConfigManager: Configuration manager instance
    """
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager(environment)
    return config_manager