"""
User Strategy Settings Service

Manages user-specific strategy settings and preferences.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.database import get_database_manager
from ..models.models import UserStrategySettings
from ..utils.api_logger import APILogger, log_api_call

logger = logging.getLogger(__name__)


class UserStrategySettingsService:
    """Service for managing user strategy settings."""
    
    def __init__(self):
        """Initialize the service."""
        self.db_manager = get_database_manager()
        self.available_strategies = {
            'default_risk': {
                'name': 'Default Risk Strategy',
                'description': 'Balanced portfolio with 60% large cap, 30% mid cap, 10% small cap',
                'risk_level': 'Medium',
                'default_enabled': True
            },
            'high_risk': {
                'name': 'High Risk Strategy', 
                'description': 'Aggressive portfolio with 50% mid cap, 50% small cap',
                'risk_level': 'High',
                'default_enabled': True
            }
        }
    
    def get_user_strategy_settings(self, user_id: int) -> Dict[str, Any]:
        """Get all strategy settings for a user."""
        try:
            session = self.db_manager.get_session()
            
            # Get user's strategy settings
            user_settings = session.query(UserStrategySettings).filter(
                UserStrategySettings.user_id == user_id
            ).all()
            
            # Convert to dictionary for easy access
            settings_dict = {}
            for setting in user_settings:
                settings_dict[setting.strategy_name] = {
                    'id': setting.id,
                    'is_active': setting.is_active,
                    'is_enabled': setting.is_enabled,
                    'priority': setting.priority,
                    'custom_parameters': json.loads(setting.custom_parameters) if setting.custom_parameters else {},
                    'created_at': setting.created_at.isoformat(),
                    'updated_at': setting.updated_at.isoformat()
                }
            
            # Add default settings for strategies not yet configured
            for strategy_name, strategy_info in self.available_strategies.items():
                if strategy_name not in settings_dict:
                    settings_dict[strategy_name] = {
                        'id': None,
                        'is_active': strategy_info['default_enabled'],
                        'is_enabled': strategy_info['default_enabled'],
                        'priority': 1,
                        'custom_parameters': {},
                        'created_at': None,
                        'updated_at': None
                    }
            
            # Add strategy metadata
            result = {
                'user_id': user_id,
                'strategies': {}
            }
            
            for strategy_name, setting in settings_dict.items():
                strategy_info = self.available_strategies[strategy_name]
                result['strategies'][strategy_name] = {
                    **setting,
                    'metadata': strategy_info
                }
            
            session.close()
            
            APILogger.log_response(
                service_name="UserStrategySettingsService",
                method_name="get_user_strategy_settings",
                response_data=result,
                user_id=user_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting user strategy settings: {e}")
            APILogger.log_error(
                service_name="UserStrategySettingsService",
                method_name="get_user_strategy_settings",
                error=e,
                request_data={'user_id': user_id},
                user_id=user_id
            )
            raise
    
    def update_strategy_setting(self, user_id: int, strategy_name: str, 
                               is_active: bool = None, is_enabled: bool = None,
                               priority: int = None, custom_parameters: Dict = None) -> Dict[str, Any]:
        """Update a specific strategy setting for a user."""
        try:
            if strategy_name not in self.available_strategies:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            session = self.db_manager.get_session()
            
            # Find existing setting or create new one
            setting = session.query(UserStrategySettings).filter(
                UserStrategySettings.user_id == user_id,
                UserStrategySettings.strategy_name == strategy_name
            ).first()
            
            if not setting:
                # Create new setting
                setting = UserStrategySettings(
                    user_id=user_id,
                    strategy_name=strategy_name,
                    is_active=is_active if is_active is not None else True,
                    is_enabled=is_enabled if is_enabled is not None else True,
                    priority=priority if priority is not None else 1,
                    custom_parameters=json.dumps(custom_parameters) if custom_parameters else None
                )
                session.add(setting)
            else:
                # Update existing setting
                if is_active is not None:
                    setting.is_active = is_active
                if is_enabled is not None:
                    setting.is_enabled = is_enabled
                if priority is not None:
                    setting.priority = priority
                if custom_parameters is not None:
                    setting.custom_parameters = json.dumps(custom_parameters)
                setting.updated_at = datetime.utcnow()
            
            session.commit()
            
            result = {
                'success': True,
                'strategy_name': strategy_name,
                'is_active': setting.is_active,
                'is_enabled': setting.is_enabled,
                'priority': setting.priority,
                'custom_parameters': json.loads(setting.custom_parameters) if setting.custom_parameters else {}
            }
            
            session.close()
            
            APILogger.log_response(
                service_name="UserStrategySettingsService",
                method_name="update_strategy_setting",
                response_data=result,
                user_id=user_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating strategy setting: {e}")
            APILogger.log_error(
                service_name="UserStrategySettingsService",
                method_name="update_strategy_setting",
                error=e,
                request_data={
                    'user_id': user_id,
                    'strategy_name': strategy_name,
                    'is_active': is_active,
                    'is_enabled': is_enabled,
                    'priority': priority,
                    'custom_parameters': custom_parameters
                },
                user_id=user_id
            )
            raise
    
    def get_active_strategies(self, user_id: int) -> List[str]:
        """Get list of active strategy names for a user."""
        try:
            session = self.db_manager.get_session()
            
            # Get active strategies
            active_settings = session.query(UserStrategySettings).filter(
                UserStrategySettings.user_id == user_id,
                UserStrategySettings.is_active == True,
                UserStrategySettings.is_enabled == True
            ).order_by(UserStrategySettings.priority).all()
            
            active_strategies = [setting.strategy_name for setting in active_settings]
            
            # If no settings found, return default active strategies
            if not active_strategies:
                active_strategies = [
                    name for name, info in self.available_strategies.items()
                    if info['default_enabled']
                ]
            
            session.close()
            
            APILogger.log_response(
                service_name="UserStrategySettingsService",
                method_name="get_active_strategies",
                response_data={'active_strategies': active_strategies},
                user_id=user_id
            )
            
            return active_strategies
            
        except Exception as e:
            logger.error(f"Error getting active strategies: {e}")
            APILogger.log_error(
                service_name="UserStrategySettingsService",
                method_name="get_active_strategies",
                error=e,
                request_data={'user_id': user_id},
                user_id=user_id
            )
            raise
    
    def get_available_strategies(self) -> Dict[str, Any]:
        """Get all available strategies with metadata."""
        try:
            result = {
                'strategies': self.available_strategies,
                'total_count': len(self.available_strategies)
            }
            
            APILogger.log_response(
                service_name="UserStrategySettingsService",
                method_name="get_available_strategies",
                response_data=result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting available strategies: {e}")
            APILogger.log_error(
                service_name="UserStrategySettingsService",
                method_name="get_available_strategies",
                error=e
            )
            raise
    
    def bulk_update_strategy_settings(self, user_id: int, settings: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Update multiple strategy settings at once."""
        try:
            session = self.db_manager.get_session()
            
            updated_strategies = []
            
            for strategy_name, setting_data in settings.items():
                if strategy_name not in self.available_strategies:
                    continue
                
                # Find existing setting or create new one
                setting = session.query(UserStrategySettings).filter(
                    UserStrategySettings.user_id == user_id,
                    UserStrategySettings.strategy_name == strategy_name
                ).first()
                
                if not setting:
                    setting = UserStrategySettings(
                        user_id=user_id,
                        strategy_name=strategy_name
                    )
                    session.add(setting)
                
                # Update fields
                if 'is_active' in setting_data:
                    setting.is_active = setting_data['is_active']
                if 'is_enabled' in setting_data:
                    setting.is_enabled = setting_data['is_enabled']
                if 'priority' in setting_data:
                    setting.priority = setting_data['priority']
                if 'custom_parameters' in setting_data:
                    setting.custom_parameters = json.dumps(setting_data['custom_parameters'])
                
                setting.updated_at = datetime.utcnow()
                updated_strategies.append(strategy_name)
            
            session.commit()
            session.close()
            
            result = {
                'success': True,
                'updated_strategies': updated_strategies,
                'total_updated': len(updated_strategies)
            }
            
            APILogger.log_response(
                service_name="UserStrategySettingsService",
                method_name="bulk_update_strategy_settings",
                response_data=result,
                user_id=user_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error bulk updating strategy settings: {e}")
            APILogger.log_error(
                service_name="UserStrategySettingsService",
                method_name="bulk_update_strategy_settings",
                error=e,
                request_data={'user_id': user_id, 'settings': settings},
                user_id=user_id
            )
            raise


# Global service instance
_user_strategy_settings_service = None


def get_user_strategy_settings_service() -> UserStrategySettingsService:
    """Get the global user strategy settings service instance."""
    global _user_strategy_settings_service
    if _user_strategy_settings_service is None:
        _user_strategy_settings_service = UserStrategySettingsService()
    return _user_strategy_settings_service
