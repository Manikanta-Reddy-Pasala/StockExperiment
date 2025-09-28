"""
Ollama Configuration Loader
Loads and validates Ollama API configuration from YAML files.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class OllamaConfig:
    """Configuration loader for Ollama API settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Ollama configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        if config_path is None:
            # Default to config/ollama_config.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "ollama_config.yaml"
        
        self.config_path = Path(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
            
            logger.info(f"Ollama configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load Ollama configuration: {e}")
            raise
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self._config.get('api', {})
    
    def get_rate_limiting_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return self._config.get('rate_limiting', {})
    
    def get_enhancement_levels(self) -> Dict[str, Any]:
        """Get enhancement levels configuration."""
        return self._config.get('enhancement_levels', {})
    
    def get_query_templates(self) -> Dict[str, str]:
        """Get query templates."""
        return self._config.get('query_templates', {})
    
    def get_data_processing_config(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        return self._config.get('data_processing', {})
    
    def get_scoring_config(self) -> Dict[str, Any]:
        """Get scoring configuration."""
        return self._config.get('scoring', {})
    
    def get_error_handling_config(self) -> Dict[str, Any]:
        """Get error handling configuration."""
        return self._config.get('error_handling', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})
    
    def get_caching_config(self) -> Dict[str, Any]:
        """Get caching configuration."""
        return self._config.get('caching', {})
    
    @property
    def api_key(self) -> str:
        """Get API key."""
        return self.get_api_config().get('api_key', '')
    
    @property
    def base_url(self) -> str:
        """Get base URL."""
        return self.get_api_config().get('base_url', 'https://ollama.com/api/web_search')
    
    @property
    def timeout_seconds(self) -> int:
        """Get timeout in seconds."""
        return self.get_api_config().get('timeout_seconds', 30)
    
    @property
    def max_retries(self) -> int:
        """Get maximum retries."""
        return self.get_api_config().get('max_retries', 3)
    
    @property
    def rate_limit_delay(self) -> float:
        """Get rate limit delay in seconds."""
        return self.get_rate_limiting_config().get('delay_between_calls', 1.0)
    
    def get_enhancement_level_config(self, level: str) -> Dict[str, Any]:
        """Get configuration for specific enhancement level."""
        levels = self.get_enhancement_levels()
        return levels.get(level, {})
    
    def is_enhancement_level_enabled(self, level: str) -> bool:
        """Check if enhancement level is enabled."""
        config = self.get_enhancement_level_config(level)
        return config.get('enabled', False)
    
    def get_query_template(self, template_name: str) -> str:
        """Get query template by name."""
        templates = self.get_query_templates()
        return templates.get(template_name, '')
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        try:
            # Check required fields
            if not self.api_key:
                logger.error("API key is required")
                return False
            
            if not self.base_url:
                logger.error("Base URL is required")
                return False
            
            # Check enhancement levels
            levels = self.get_enhancement_levels()
            if not levels:
                logger.warning("No enhancement levels configured")
            
            # Check query templates
            templates = self.get_query_templates()
            if not templates:
                logger.warning("No query templates configured")
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


# Global configuration instance
_ollama_config = None


def get_ollama_config() -> OllamaConfig:
    """Get global Ollama configuration instance."""
    global _ollama_config
    if _ollama_config is None:
        _ollama_config = OllamaConfig()
    return _ollama_config


def reload_ollama_config() -> OllamaConfig:
    """Reload Ollama configuration."""
    global _ollama_config
    _ollama_config = OllamaConfig()
    return _ollama_config
