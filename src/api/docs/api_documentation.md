# Trading System API Documentation

## Overview

The Trading System API provides a comprehensive REST API for managing automated trading operations, user administration, and system monitoring.

**Base URL:** `http://localhost:5001/api/v1`

**Authentication:** All endpoints require authentication via Flask-Login session.

## API Structure

```
/api/v1/
├── admin/          # Administrative functions
├── trading/        # Trading operations
├── analytics/      # Performance analytics
├── data/          # Data management
├── alerts/        # Alert system
└── orders/        # Order management
```

## Common Response Format

All API responses follow this standard format:

```json
{
  "success": true,
  "data": {...},
  "timestamp": "2024-01-01T12:00:00Z"
}
```

Error responses:

```json
{
  "success": false,
  "error": "Error message",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Error Codes

- `VALIDATION_ERROR` (400): Invalid input data
- `AUTHENTICATION_ERROR` (401): Authentication required
- `AUTHORIZATION_ERROR` (403): Insufficient permissions
- `NOT_FOUND` (404): Resource not found
- `CONFLICT` (409): Resource conflict
- `INTERNAL_ERROR` (500): Internal server error

---

## Admin API

### User Management

#### Get All Users
```http
GET /api/v1/admin/users
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "username": "admin",
      "email": "admin@example.com",
      "first_name": "Admin",
      "last_name": "User",
      "is_admin": true,
      "is_active": true,
      "created_at": "2024-01-01T12:00:00Z",
      "last_login": "2024-01-01T12:00:00Z"
    }
  ]
}
```

#### Create User
```http
POST /api/v1/admin/users
Content-Type: application/json

{
  "username": "newuser",
  "email": "user@example.com",
  "password": "password123",
  "first_name": "New",
  "last_name": "User",
  "is_admin": false
}
```

#### Update User
```http
PUT /api/v1/admin/users/{user_id}
Content-Type: application/json

{
  "first_name": "Updated",
  "is_active": false
}
```

#### Delete User
```http
DELETE /api/v1/admin/users/{user_id}
```

#### Reset User Password
```http
POST /api/v1/admin/users/{user_id}/reset-password
Content-Type: application/json

{
  "new_password": "newpassword123"
}
```

### System Management

#### System Health
```http
GET /api/v1/admin/system/health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "database": "connected",
    "system_metrics": {
      "cpu_percent": 25.5,
      "memory_percent": 60.2,
      "memory_available_gb": 8.5,
      "disk_percent": 45.0,
      "disk_free_gb": 25.3
    }
  }
}
```

#### Get System Logs
```http
GET /api/v1/admin/system/logs
```

#### Get System Configuration
```http
GET /api/v1/admin/system/config
```

---

## Trading API

### Stock Screening

#### Run Screening
```http
POST /api/v1/trading/screening/run
```

**Response:**
```json
{
  "success": true,
  "data": {
    "screened_stocks": [
      {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "score": 85.5,
        "reason": "Strong fundamentals"
      }
    ],
    "count": 1
  }
}
```

#### Get Screening Criteria
```http
GET /api/v1/trading/screening/criteria
```

#### Update Screening Criteria
```http
PUT /api/v1/trading/screening/criteria
Content-Type: application/json

{
  "min_volume": 1000000,
  "max_pe_ratio": 25,
  "min_market_cap": 1000000000
}
```

#### Get Screening History
```http
GET /api/v1/trading/screening/history
```

### Trading Strategies

#### Run Strategies
```http
POST /api/v1/trading/strategies/run
Content-Type: application/json

{
  "screened_stocks": ["AAPL", "GOOGL"]
}
```

#### Get Available Strategies
```http
GET /api/v1/trading/strategies/list
```

#### Get Strategy Performance
```http
GET /api/v1/trading/strategies/performance
```

#### Run Backtest
```http
POST /api/v1/trading/strategies/backtest
Content-Type: application/json

{
  "strategy_id": 1,
  "start_date": "2023-01-01",
  "end_date": "2023-12-31"
}
```

### Trading Execution

#### Run Dry Run
```http
POST /api/v1/trading/execution/dry-run
```

#### Run Complete Workflow
```http
POST /api/v1/trading/execution/complete-workflow
```

#### Start Scheduled Execution
```http
POST /api/v1/trading/execution/start-scheduled
```

#### Stop Scheduled Execution
```http
POST /api/v1/trading/execution/stop-scheduled
```

#### Get Execution Status
```http
GET /api/v1/trading/execution/status
```

#### Cleanup Dry Run
```http
POST /api/v1/trading/execution/cleanup-dry-run
```

---

## Analytics API

### Performance Analytics

#### Get Performance Report
```http
GET /api/v1/analytics/performance-report
```

#### AI Stock Analysis
```http
POST /api/v1/analytics/ai/analyze-stock
Content-Type: application/json

{
  "symbol": "AAPL",
  "analysis_type": "fundamental"
}
```

#### AI Portfolio Analysis
```http
POST /api/v1/analytics/ai/analyze-portfolio
Content-Type: application/json

{
  "portfolio_id": 1,
  "analysis_type": "risk"
}
```

---

## Data API

### Stock Data

#### Get Stock Data
```http
POST /api/v1/data/stock-data
Content-Type: application/json

{
  "symbols": ["AAPL", "GOOGL"],
  "start_date": "2024-01-01",
  "end_date": "2024-01-31"
}
```

#### Get Current Price
```http
POST /api/v1/data/current-price
Content-Type: application/json

{
  "symbols": ["AAPL", "GOOGL"]
}
```

#### Get Data Providers
```http
GET /api/v1/data/providers
```

---

## Alerts API

### Email Alerts

#### Send Stock Pick Alert
```http
POST /api/v1/alerts/send-stock-pick
Content-Type: application/json

{
  "symbol": "AAPL",
  "reason": "Strong buy signal",
  "recipients": ["user@example.com"]
}
```

#### Send Portfolio Alert
```http
POST /api/v1/alerts/send-portfolio-alert
Content-Type: application/json

{
  "alert_type": "performance",
  "message": "Portfolio performance update",
  "recipients": ["user@example.com"]
}
```

---

## Orders API

### Order Management

#### Create Buy Order
```http
POST /api/v1/orders/create-buy-order
Content-Type: application/json

{
  "symbol": "AAPL",
  "quantity": 100,
  "price": 150.00,
  "order_type": "limit"
}
```

#### Create Sell Order
```http
POST /api/v1/orders/create-sell-order
Content-Type: application/json

{
  "symbol": "AAPL",
  "quantity": 50,
  "price": 155.00,
  "order_type": "limit"
}
```

#### Cancel Order
```http
POST /api/v1/orders/cancel-order
Content-Type: application/json

{
  "order_id": 123
}
```

#### Get User Orders
```http
GET /api/v1/orders/user-orders
```

#### Get User Positions
```http
GET /api/v1/orders/user-positions
```

#### Get User Trades
```http
GET /api/v1/orders/user-trades
```

---

## Authentication

All API endpoints require authentication. The system uses Flask-Login for session management.

### Login
```http
POST /login
Content-Type: application/x-www-form-urlencoded

username=admin&password=password
```

### Logout
```http
GET /logout
```

---

## Rate Limiting

API endpoints are rate-limited to prevent abuse:
- **Default:** 100 requests per hour per user
- **Admin endpoints:** 200 requests per hour
- **Trading endpoints:** 50 requests per hour

---

## Versioning

The API uses URL versioning:
- Current version: `v1`
- Future versions: `v2`, `v3`, etc.

---

## Support

For API support and questions:
- Check the system logs: `GET /api/v1/admin/system/logs`
- Monitor system health: `GET /api/v1/admin/system/health`
- Review error codes and messages in responses
