# Paper Trading Guide

## Overview

Your trading system **already has paper trading (simulation mode) fully implemented**! This allows you to test automated trading strategies without risking real money.

## How Paper Trading Works

### Default Behavior
- **All new users** are in paper trading mode by default (`is_mock_trading_mode=True`)
- Orders are created in the database with `is_mock_order=True` flag
- **NO actual orders are sent to Fyers broker**
- You can track performance and test strategies safely

### Paper Trading vs Live Trading

| Feature | Paper Trading | Live Trading |
|---------|--------------|--------------|
| Orders sent to broker | ❌ No | ✅ Yes |
| Orders saved in database | ✅ Yes | ✅ Yes |
| Performance tracking | ✅ Yes | ✅ Yes |
| Risk | 🟢 Zero | 🔴 Real money |
| Order ID format | `MOCK_{user}_{symbol}_{timestamp}` | Broker order ID |
| Order status | Auto-set to `COMPLETE` | From broker API |

## How to Use Paper Trading

### 1. Enable Paper Trading (Default)

```python
# Check if paper trading is enabled
from src.models.database import get_database_manager
from src.models.models import User

db_manager = get_database_manager()
with db_manager.get_session() as session:
    user = session.query(User).filter_by(id=1).first()
    print(f"Paper trading: {user.is_mock_trading_mode}")
    # Output: True (enabled by default)
```

### 2. Enable/Disable via Database

```sql
-- Enable paper trading (SAFE - simulation only)
UPDATE users SET is_mock_trading_mode = TRUE WHERE id = 1;

-- Disable paper trading (DANGER - real money!)
UPDATE users SET is_mock_trading_mode = FALSE WHERE id = 1;
```

### 3. Enable/Disable via Python

```python
from src.models.database import get_database_manager
from src.models.models import User

db_manager = get_database_manager()
with db_manager.get_session() as session:
    user = session.query(User).filter_by(id=1).first()

    # Enable paper trading
    user.is_mock_trading_mode = True

    # Or disable (use with caution!)
    # user.is_mock_trading_mode = False

    session.commit()
```

## Testing Paper Trading

### 1. Verify Auto-Trading Settings

```python
from src.models.database import get_database_manager
from src.models.models import AutoTradingSettings

db_manager = get_database_manager()
with db_manager.get_session() as session:
    settings = session.query(AutoTradingSettings).filter_by(user_id=1).first()

    if not settings:
        # Create auto-trading settings
        settings = AutoTradingSettings(
            user_id=1,
            is_enabled=True,
            max_weekly_investment=10000,  # ₹10,000 per week
            max_weekly_buys=10,            # Max 10 stocks per week
            minimum_confidence_score=0.7,  # 70% minimum confidence
            preferred_strategies='["default_risk"]',
            preferred_model_types='["traditional"]'
        )
        session.add(settings)
        session.commit()
```

### 2. Run Auto-Trading (Paper Mode)

```bash
# Method 1: Via scheduler (automatic at 9:20 AM daily)
docker compose logs -f ml_scheduler | grep "auto-trading"

# Method 2: Manual execution
python3 -c "
from src.services.trading.auto_trading_service import AutoTradingService

service = AutoTradingService()
result = service.execute_auto_trading_for_user(user_id=1)
print(result)
"
```

### 3. Check Paper Trading Orders

```sql
-- View all mock orders
SELECT
    order_id, tradingsymbol, quantity, price,
    model_type, strategy, ml_prediction_score,
    placed_at
FROM orders
WHERE is_mock_order = TRUE
ORDER BY placed_at DESC;

-- Count paper vs real orders
SELECT
    is_mock_order,
    COUNT(*) as order_count,
    SUM(quantity * price) as total_value
FROM orders
GROUP BY is_mock_order;
```

### 4. View Auto-Trading Execution Log

```sql
-- Check recent auto-trading executions
SELECT
    execution_date, status, orders_created,
    total_amount_invested, error_message
FROM auto_trading_executions
ORDER BY execution_date DESC
LIMIT 10;
```

## Paper Trading Workflow

### Daily Auto-Trading Cycle (9:20 AM)

1. **Check if paper trading enabled** for user
2. **Check market sentiment** (AI confidence score)
3. **Check weekly limits** (max investment/buys)
4. **Check account balance** (simulated or real)
5. **Select top strategies** with highest ML confidence
6. **Create paper orders** (if `is_mock_trading_mode=True`):
   - Generate mock order ID: `MOCK_1_NSE:RELIANCE-EQ_1234567890`
   - Save to `orders` table with `is_mock_order=True`
   - Set status to `COMPLETE` immediately
   - **NO API call to broker**
7. **Track performance** (same as real trading)

### Performance Tracking (6:00 PM Daily)

Even paper orders get full performance tracking:
- Current profit/loss
- ROI percentage
- Peak profit/loss
- Stop-loss/target price monitoring
- Daily snapshots

## Switching to Live Trading

### ⚠️ WARNING: Only After Thorough Testing!

Before switching to live trading:

1. **Test extensively in paper mode** (1-2 months minimum)
2. **Verify strategy performance** is consistently profitable
3. **Review risk management** (stop-loss, position sizing)
4. **Confirm broker configuration** (Fyers API credentials)
5. **Start with small amounts** (₹1000-5000 initially)

### Steps to Switch to Live Trading

```sql
-- 1. Verify you're ready
SELECT * FROM auto_trading_settings WHERE user_id = 1;
SELECT * FROM orders WHERE is_mock_order = TRUE ORDER BY placed_at DESC LIMIT 20;

-- 2. Reduce limits for safety
UPDATE auto_trading_settings
SET max_weekly_investment = 1000,  -- Start small!
    max_weekly_buys = 2
WHERE user_id = 1;

-- 3. Enable live trading (DANGER!)
UPDATE users
SET is_mock_trading_mode = FALSE
WHERE id = 1;
```

## Monitoring Paper Trading

### View Logs

```bash
# Auto-trading execution logs
cat logs/scheduler.log | grep "auto-trading"

# Paper order creation
cat logs/scheduler.log | grep "Mock order"

# Performance tracking
cat logs/scheduler.log | grep "performance"
```

### Dashboard Access

```bash
# View orders in web interface
http://localhost:5001/orders

# View performance tracking
http://localhost:5001/performance

# View auto-trading settings
http://localhost:5001/settings/auto-trading
```

## Troubleshooting

### Issue: No orders being created

```python
# Check if auto-trading is enabled
from src.models.database import get_database_manager
from src.models.models import AutoTradingSettings, User

db_manager = get_database_manager()
with db_manager.get_session() as session:
    settings = session.query(AutoTradingSettings).filter_by(user_id=1).first()
    print(f"Auto-trading enabled: {settings.is_enabled if settings else 'No settings found'}")

    user = session.query(User).filter_by(id=1).first()
    print(f"Paper mode: {user.is_mock_trading_mode}")
```

### Issue: Orders being sent to broker in paper mode

This should NEVER happen if `is_mock_trading_mode=True`. If it does:

1. Verify user setting:
   ```sql
   SELECT is_mock_trading_mode FROM users WHERE id = 1;
   ```

2. Check code at `auto_trading_service.py:452`:
   ```python
   if user.is_mock_trading_mode:
       # Paper trading - ONLY creates DB record
   else:
       # Real trading - sends to broker
   ```

### Issue: Can't find mock orders

```sql
-- Check if mock orders exist
SELECT COUNT(*) FROM orders WHERE is_mock_order = TRUE;

-- If 0, check auto-trading execution log
SELECT * FROM auto_trading_executions ORDER BY execution_date DESC LIMIT 5;
```

## Best Practices

1. **Always start in paper mode** - Never skip simulation testing
2. **Track metrics** - Monitor win rate, average ROI, max drawdown
3. **Test different strategies** - Try DEFAULT_RISK vs HIGH_RISK
4. **Test all ML models** - Compare Traditional vs LSTM vs Kronos
5. **Review failed orders** - Understand why orders didn't execute
6. **Monitor market conditions** - Paper trading shows how strategies perform in different markets
7. **Keep paper mode enabled** - Until you have 90%+ confidence in strategy

## FAQ

**Q: Is paper trading enabled by default?**
A: Yes! All users have `is_mock_trading_mode=True` by default.

**Q: Do paper orders affect my broker account?**
A: No! Paper orders are ONLY saved in the database, never sent to Fyers.

**Q: Can I test with real market data?**
A: Yes! Paper orders use real-time prices from your data pipeline.

**Q: How long should I run paper trading?**
A: Minimum 1-2 months, ideally 3-6 months to test in different market conditions.

**Q: Can I switch back to paper mode after going live?**
A: Yes, absolutely! Just set `is_mock_trading_mode=TRUE` again.

**Q: Do paper orders have order IDs?**
A: Yes, format: `MOCK_{user_id}_{symbol}_{timestamp}`

**Q: Does performance tracking work for paper orders?**
A: Yes! Paper orders get full performance tracking just like real orders.

## Summary

✅ **Paper trading is already implemented and enabled by default**
✅ **Safe to test strategies without real money**
✅ **Full performance tracking included**
✅ **Easy to switch between paper and live mode**
⚠️ **Always test thoroughly before going live**

---

For questions or issues, check the auto-trading service code:
- `src/services/trading/auto_trading_service.py`
- `src/models/models.py` (User model, line 29)
