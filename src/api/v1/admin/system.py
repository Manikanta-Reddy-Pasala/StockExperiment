"""
System Administration API
"""
from flask import Blueprint, jsonify
from flask_login import login_required
from api.common.decorators import api_response, admin_required
from api.common.errors import InternalServerError
from datetime import datetime
import logging
import os
try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

system_bp = Blueprint('system', __name__)

@system_bp.route('/health', methods=['GET'])
@api_response
def system_health():
    """Get system health status."""
    try:
        # Check database connection
        from datastore.database import get_database_manager
        db_manager = get_database_manager()
        
        with db_manager.get_session() as session:
            session.execute("SELECT 1")
        
        # Get system metrics
        if psutil:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
        else:
            cpu_percent = 0
            memory = type('Memory', (), {'percent': 0, 'available': 0})()
            disk = type('Disk', (), {'percent': 0, 'free': 0})()
        
        return {
            'status': 'healthy',
            'database': 'connected',
            'system_metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_free_gb': round(disk.free / (1024**3), 2)
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise InternalServerError(f"Health check failed: {str(e)}")

@system_bp.route('/logs', methods=['GET'])
@login_required
@admin_required
@api_response
def get_system_logs():
    """Get system logs."""
    try:
        from datastore.database import get_database_manager
        from datastore.models import Log
        
        db_manager = get_database_manager()
        
        with db_manager.get_session() as session:
            logs = session.query(Log).order_by(Log.timestamp.desc()).limit(100).all()
            
            logs_data = []
            for log in logs:
                logs_data.append({
                    'id': log.id,
                    'level': log.level,
                    'message': log.message,
                    'timestamp': log.timestamp.isoformat(),
                    'user_id': log.user_id,
                    'module': log.module
                })
            
            return logs_data
    except Exception as e:
        logger.error(f"Failed to get logs: {str(e)}")
        raise InternalServerError(f"Failed to get logs: {str(e)}")

@system_bp.route('/config', methods=['GET'])
@login_required
@admin_required
@api_response
def get_system_config():
    """Get system configuration."""
    try:
        from datastore.database import get_database_manager
        from datastore.models import Configuration
        
        db_manager = get_database_manager()
        
        with db_manager.get_session() as session:
            configs = session.query(Configuration).all()
            
            config_data = {}
            for config in configs:
                config_data[config.key] = {
                    'value': config.value,
                    'description': config.description,
                    'updated_at': config.updated_at.isoformat() if config.updated_at else None
                }
            
            return config_data
    except Exception as e:
        logger.error(f"Failed to get config: {str(e)}")
        raise InternalServerError(f"Failed to get config: {str(e)}")
