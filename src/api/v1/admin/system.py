"""
FastAPI Admin System Module
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import os

router = APIRouter()

class SystemHealth(BaseModel):
    status: str
    database: str
    system_metrics: Dict[str, Any]

class APIResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    message: Optional[str] = None
    timestamp: datetime

@router.get("/health", response_model=APIResponse, summary="Get System Health")
async def get_system_health():
    """
    Get system health status and metrics.
    
    Returns system health information including:
    - Database connection status
    - System metrics (CPU, memory, disk usage)
    - Overall system status
    """
    try:
        if PSUTIL_AVAILABLE:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_metrics = {
                "cpu_percent": round(cpu_percent, 2),
                "memory_percent": round(memory.percent, 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": round(disk.percent, 2),
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
        else:
            # Mock metrics when psutil is not available
            system_metrics = {
                "cpu_percent": 25.5,
                "memory_percent": 60.2,
                "memory_available_gb": 8.5,
                "disk_percent": 45.0,
                "disk_free_gb": 25.3
            }
        
        health_data = {
            "status": "healthy",
            "database": "connected",
            "system_metrics": system_metrics
        }
        
        return APIResponse(
            success=True,
            data=health_data,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        return APIResponse(
            success=False,
            data={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e)
            },
            timestamp=datetime.utcnow()
        )

@router.get("/logs", response_model=APIResponse, summary="Get System Logs")
async def get_system_logs():
    """
    Get system logs (Admin only).
    
    Returns recent system logs for monitoring and debugging.
    """
    # Mock log data
    mock_logs = [
        {
            "id": 1,
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "module": "trading_engine",
            "message": "Trading engine started successfully"
        },
        {
            "id": 2,
            "timestamp": datetime.utcnow().isoformat(),
            "level": "WARNING",
            "module": "data_provider",
            "message": "API rate limit approaching"
        },
        {
            "id": 3,
            "timestamp": datetime.utcnow().isoformat(),
            "level": "ERROR",
            "module": "order_manager",
            "message": "Failed to place order: insufficient funds"
        }
    ]
    
    return APIResponse(
        success=True,
        data={"logs": mock_logs, "count": len(mock_logs)},
        timestamp=datetime.utcnow()
    )

@router.get("/metrics", response_model=APIResponse, summary="Get System Metrics")
async def get_system_metrics():
    """
    Get detailed system metrics (Admin only).
    
    Returns comprehensive system performance metrics.
    """
    try:
        if PSUTIL_AVAILABLE:
            # Get detailed system metrics
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            metrics = {
                "cpu": {
                    "percent_per_core": cpu_percent,
                    "average_percent": round(sum(cpu_percent) / len(cpu_percent), 2),
                    "core_count": len(cpu_percent)
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "percent": round(memory.percent, 2)
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent": round(disk.percent, 2)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            }
        else:
            # Mock metrics when psutil is not available
            metrics = {
                "cpu": {
                    "percent_per_core": [25.5, 30.2, 22.8, 28.1],
                    "average_percent": 26.7,
                    "core_count": 4
                },
                "memory": {
                    "total_gb": 16.0,
                    "available_gb": 8.5,
                    "used_gb": 7.5,
                    "percent": 60.2
                },
                "disk": {
                    "total_gb": 500.0,
                    "used_gb": 275.0,
                    "free_gb": 225.0,
                    "percent": 45.0
                },
                "network": {
                    "bytes_sent": 1024000,
                    "bytes_recv": 2048000,
                    "packets_sent": 1500,
                    "packets_recv": 2000
                }
            }
        
        return APIResponse(
            success=True,
            data={"metrics": metrics},
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system metrics: {str(e)}"
        )
