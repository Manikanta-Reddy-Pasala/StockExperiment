-- Migration: Add partial exit, day trading, and virtual capital columns
-- Run this on the deployed database BEFORE deploying the new code
-- Usage: docker exec -it trading_system_db psql -U trader -d trading_system -f /app/migrations/add_partial_exit_columns.sql

BEGIN;

-- ============================================================
-- OrderPerformance: partial exit & day trading columns
-- ============================================================
ALTER TABLE order_performance ADD COLUMN IF NOT EXISTS original_quantity INTEGER;
ALTER TABLE order_performance ADD COLUMN IF NOT EXISTS remaining_quantity INTEGER;
ALTER TABLE order_performance ADD COLUMN IF NOT EXISTS target_price_1 FLOAT;
ALTER TABLE order_performance ADD COLUMN IF NOT EXISTS target_price_2 FLOAT;
ALTER TABLE order_performance ADD COLUMN IF NOT EXISTS target_price_3 FLOAT;
ALTER TABLE order_performance ADD COLUMN IF NOT EXISTS trading_type VARCHAR(20) DEFAULT 'swing';
ALTER TABLE order_performance ADD COLUMN IF NOT EXISTS partial_exit_1_done BOOLEAN DEFAULT false;
ALTER TABLE order_performance ADD COLUMN IF NOT EXISTS partial_exit_2_done BOOLEAN DEFAULT false;
ALTER TABLE order_performance ADD COLUMN IF NOT EXISTS partial_exit_3_done BOOLEAN DEFAULT false;
ALTER TABLE order_performance ADD COLUMN IF NOT EXISTS partial_pnl_realized FLOAT DEFAULT 0.0;

-- Backfill existing records so remaining_quantity/original_quantity are not NULL
UPDATE order_performance
SET original_quantity = quantity,
    remaining_quantity = quantity
WHERE original_quantity IS NULL;

-- ============================================================
-- AutoTradingSettings: trading mode & virtual capital
-- ============================================================
ALTER TABLE auto_trading_settings ADD COLUMN IF NOT EXISTS trading_mode VARCHAR(20) DEFAULT 'swing';
ALTER TABLE auto_trading_settings ADD COLUMN IF NOT EXISTS virtual_capital FLOAT DEFAULT 100000.0;

-- ============================================================
-- DailySuggestedStock: new columns for sector & fib targets
-- ============================================================
ALTER TABLE daily_suggested_stocks ADD COLUMN IF NOT EXISTS sector VARCHAR(100);
ALTER TABLE daily_suggested_stocks ADD COLUMN IF NOT EXISTS fib_target_1 FLOAT;
ALTER TABLE daily_suggested_stocks ADD COLUMN IF NOT EXISTS fib_target_2 FLOAT;
ALTER TABLE daily_suggested_stocks ADD COLUMN IF NOT EXISTS fib_target_3 FLOAT;

COMMIT;

-- Verify migration
SELECT 'order_performance columns' AS check_type, count(*) AS col_count
FROM information_schema.columns
WHERE table_name = 'order_performance'
  AND column_name IN ('original_quantity', 'remaining_quantity', 'target_price_1', 'target_price_2', 'target_price_3', 'trading_type', 'partial_exit_1_done', 'partial_exit_2_done', 'partial_exit_3_done', 'partial_pnl_realized');

SELECT 'auto_trading_settings columns' AS check_type, count(*) AS col_count
FROM information_schema.columns
WHERE table_name = 'auto_trading_settings'
  AND column_name IN ('trading_mode', 'virtual_capital');

SELECT 'daily_suggested_stocks columns' AS check_type, count(*) AS col_count
FROM information_schema.columns
WHERE table_name = 'daily_suggested_stocks'
  AND column_name IN ('sector', 'fib_target_1', 'fib_target_2', 'fib_target_3');
