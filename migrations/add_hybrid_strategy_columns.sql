-- Migration: Add Hybrid Strategy columns to existing tables
-- Run this with: docker exec trading_system_db psql -U trader -d trading_system -f /path/to/this/file
-- Or: psql -U trader -d trading_system -f add_hybrid_strategy_columns.sql

-- ============================================================================
-- TECHNICAL_INDICATORS TABLE
-- Add EMA 8, 21 and DeMarker columns
-- ============================================================================

ALTER TABLE technical_indicators
ADD COLUMN IF NOT EXISTS ema_8 DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS ema_21 DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS demarker DOUBLE PRECISION;

COMMENT ON COLUMN technical_indicators.ema_8 IS '8-period Exponential Moving Average (short-term momentum)';
COMMENT ON COLUMN technical_indicators.ema_21 IS '21-period Exponential Moving Average (institutional holding period)';
COMMENT ON COLUMN technical_indicators.demarker IS 'DeMarker oscillator (0-1 range, <0.30 oversold, >0.70 overbought)';

-- ============================================================================
-- DAILY_SUGGESTED_STOCKS TABLE
-- Add Hybrid Strategy columns
-- ============================================================================

ALTER TABLE daily_suggested_stocks
-- Wave indicators (enhanced)
ADD COLUMN IF NOT EXISTS wave_momentum_score DECIMAL(10, 4),

-- 8-21 EMA Strategy
ADD COLUMN IF NOT EXISTS ema_8 DECIMAL(10, 2),
ADD COLUMN IF NOT EXISTS ema_21 DECIMAL(10, 2),
ADD COLUMN IF NOT EXISTS ema_trend_score DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS demarker DECIMAL(10, 4),

-- Fibonacci Targets
ADD COLUMN IF NOT EXISTS fib_target_1 DECIMAL(10, 2),
ADD COLUMN IF NOT EXISTS fib_target_2 DECIMAL(10, 2),
ADD COLUMN IF NOT EXISTS fib_target_3 DECIMAL(10, 2),

-- Hybrid Composite Score
ADD COLUMN IF NOT EXISTS hybrid_composite_score DECIMAL(10, 4),

-- Enhanced Signals
ADD COLUMN IF NOT EXISTS signal_quality VARCHAR(20);

-- Add comments for documentation
COMMENT ON COLUMN daily_suggested_stocks.rs_rating IS 'Relative Strength Rating (1-99 percentile vs NIFTY 50)';
COMMENT ON COLUMN daily_suggested_stocks.wave_momentum_score IS 'Wave momentum score (0-100, normalized from Delta)';
COMMENT ON COLUMN daily_suggested_stocks.ema_8 IS '8-period EMA value at selection time';
COMMENT ON COLUMN daily_suggested_stocks.ema_21 IS '21-period EMA value at selection time';
COMMENT ON COLUMN daily_suggested_stocks.ema_trend_score IS 'EMA trend score (0-100, based on 8-21 EMA configuration)';
COMMENT ON COLUMN daily_suggested_stocks.demarker IS 'DeMarker oscillator value (0-1 range)';
COMMENT ON COLUMN daily_suggested_stocks.fib_target_1 IS 'Fibonacci 127.2% extension target';
COMMENT ON COLUMN daily_suggested_stocks.fib_target_2 IS 'Fibonacci 161.8% (golden ratio) extension target';
COMMENT ON COLUMN daily_suggested_stocks.fib_target_3 IS 'Fibonacci 200-261.8% extension target';
COMMENT ON COLUMN daily_suggested_stocks.hybrid_composite_score IS 'Hybrid composite score (40% EMA + 30% Wave + 30% RS)';
COMMENT ON COLUMN daily_suggested_stocks.signal_quality IS 'Buy signal quality: high (5/5 conditions), medium (4/5), low (3/5), none';

-- Update the unique constraint to include model_type
-- First, drop the old constraint
ALTER TABLE daily_suggested_stocks DROP CONSTRAINT IF EXISTS daily_suggested_stocks_date_symbol_strategy_model_type_key;
ALTER TABLE daily_suggested_stocks DROP CONSTRAINT IF EXISTS daily_suggested_stocks_date_symbol_strategy_key;

-- Add the new constraint
ALTER TABLE daily_suggested_stocks
ADD CONSTRAINT daily_suggested_stocks_date_symbol_strategy_model_type_key
UNIQUE (date, symbol, strategy, model_type);

-- ============================================================================
-- VERIFICATION QUERIES
-- Run these to verify migration success
-- ============================================================================

-- Verify technical_indicators columns
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'technical_indicators'
AND column_name IN ('ema_8', 'ema_21', 'demarker')
ORDER BY column_name;

-- Verify daily_suggested_stocks columns
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'daily_suggested_stocks'
AND column_name IN (
    'wave_momentum_score', 'ema_8', 'ema_21', 'ema_trend_score', 'demarker',
    'fib_target_1', 'fib_target_2', 'fib_target_3',
    'hybrid_composite_score', 'signal_quality'
)
ORDER BY column_name;

-- Check unique constraints
SELECT conname, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conrelid = 'daily_suggested_stocks'::regclass
AND contype = 'u';

-- ============================================================================
-- DONE!
-- ============================================================================

SELECT 'Hybrid Strategy Migration Completed Successfully!' AS status;
