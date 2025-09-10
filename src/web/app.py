"""
FastAPI Application for Trading System
Auto-generated API documentation with Swagger UI
"""
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import API modules
from api.v1.admin.users import router as admin_users_router
from api.v1.admin.system import router as admin_system_router
from api.v1.trading.screening import router as trading_screening_router
from api.v1.trading.strategies import router as trading_strategies_router
from api.v1.trading.execution import router as trading_execution_router

# Create FastAPI app
app = FastAPI(
    title="Trading System API",
    description="Automated Trading System REST API with auto-generated documentation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Include API routers
app.include_router(admin_users_router, prefix="/api/v1/admin/users", tags=["Admin - Users"])
app.include_router(admin_system_router, prefix="/api/v1/admin/system", tags=["Admin - System"])
app.include_router(trading_screening_router, prefix="/api/v1/trading/screening", tags=["Trading - Screening"])
app.include_router(trading_strategies_router, prefix="/api/v1/trading/strategies", tags=["Trading - Strategies"])
app.include_router(trading_execution_router, prefix="/api/v1/trading/execution", tags=["Trading - Execution"])

# API info endpoint (moved to /api/info)
@app.get("/api/info", tags=["API"])
async def api_info():
    """API information endpoint."""
    return {
        "message": "Trading System API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json"
    }

# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "api_version": "v1"
    }

# API info endpoint
@app.get("/api/health", tags=["API"])
async def api_health():
    """API health and information endpoint."""
    return {
        "status": "healthy",
        "api_version": "v1",
        "endpoints": {
            "admin": "/api/v1/admin",
            "trading": "/api/v1/trading",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# Web UI Routes
@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
async def root(request: Request):
    """Root page - redirect to login."""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login", response_class=HTMLResponse, tags=["Web UI"])
async def login_page(request: Request):
    """Login page."""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse, tags=["Web UI"])
async def dashboard(request: Request):
    """Dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/admin/users", response_class=HTMLResponse, tags=["Web UI"])
async def admin_users(request: Request):
    """Admin users page."""
    return templates.TemplateResponse("admin/users.html", {"request": request})

@app.get("/portfolio", response_class=HTMLResponse, tags=["Web UI"])
async def portfolio(request: Request):
    """Portfolio page."""
    return templates.TemplateResponse("portfolio.html", {"request": request})

@app.get("/orders", response_class=HTMLResponse, tags=["Web UI"])
async def orders(request: Request):
    """Orders page."""
    return templates.TemplateResponse("orders.html", {"request": request})

@app.get("/trades", response_class=HTMLResponse, tags=["Web UI"])
async def trades(request: Request):
    """Trades page."""
    return templates.TemplateResponse("trades.html", {"request": request})

@app.get("/strategies", response_class=HTMLResponse, tags=["Web UI"])
async def strategies(request: Request):
    """Strategies page."""
    return templates.TemplateResponse("strategies.html", {"request": request})

@app.get("/reports", response_class=HTMLResponse, tags=["Web UI"])
async def reports(request: Request):
    """Reports page."""
    return templates.TemplateResponse("reports.html", {"request": request})
    
@app.get("/alerts", response_class=HTMLResponse, tags=["Web UI"])
async def alerts(request: Request):
    """Alerts page."""
    return templates.TemplateResponse("alerts.html", {"request": request})

@app.get("/settings", response_class=HTMLResponse, tags=["Web UI"])
async def settings(request: Request):
    """Settings page."""
    return templates.TemplateResponse("settings.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level="info"
    )
