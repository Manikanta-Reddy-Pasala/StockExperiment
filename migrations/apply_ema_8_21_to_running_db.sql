-- Migration: Apply 8-21 EMA Strategy Columns and Indexes to Running Database
-- Date: 2025-10-31
-- Purpose: Update running database to match init-scripts/01-init-db.sql schema

-- =============================================================================
-- STEP 1: Add Missing Columns (if not exist)
-- =============================================================================

ALTER TABLE technical_indicators
ADD COLUMN IF NOT EXISTS ema_8 DOUBLE PRECISION;

ALTER TABLE technical_indicators
ADD COLUMN IF NOT EXISTS ema_21 DOUBLE PRECISION;

ALTER TABLE technical_indicators
ADD COLUMN IF NOT EXISTS demarker DOUBLE PRECISION;

-- =============================================================================
-- STEP 2: Create Indexes for Optimized Querying
-- =============================================================================

-- Individual column indexes (with WHERE clause for NULL filtering)
CREATE INDEX IF NOT EXISTS idx_technical_ema8
ON technical_indicators(ema_8)
WHERE ema_8 IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_technical_ema21
ON technical_indicators(ema_21)
WHERE ema_21 IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_technical_demarker
ON technical_indicators(demarker)
WHERE demarker IS NOT NULL;

-- Composite index for 8-21 EMA strategy queries
CREATE INDEX IF NOT EXISTS idx_technical_ema_strategy
ON technical_indicators(symbol, date, ema_8, ema_21, demarker)
WHERE ema_8 IS NOT NULL AND ema_21 IS NOT NULL;

-- =============================================================================
-- STEP 3: Add Table and Column Comments
-- =============================================================================

COMMENT ON TABLE technical_indicators IS 'Historical technical indicators calculated from OHLCV data. Includes moving averages, momentum, volatility, and volume indicators. Primary columns for 8-21 EMA strategy: ema_8, ema_21, demarker.';

COMMENT ON COLUMN technical_indicators.ema_8 IS '8-day Exponential Moving Average (fast EMA). Used to identify short-term trend. Part of 8-21 EMA strategy power zone (Price > EMA8 > EMA21 = bullish).';

COMMENT ON COLUMN technical_indicators.ema_21 IS '21-day Exponential Moving Average (slow EMA). Represents institutional holding period. Acts as dynamic support/resistance in 8-21 EMA strategy.';

COMMENT ON COLUMN technical_indicators.demarker IS 'DeMarker Oscillator (0-1 range, 14-period). Measures buying/selling pressure. <0.30 = oversold (ideal buy), >0.70 = overbought (avoid). Used for entry timing in pullbacks.';

-- =============================================================================
-- STEP 4: Verify Changes
-- =============================================================================

-- List all columns in technical_indicators table
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'technical_indicators'
AND column_name IN ('ema_8', 'ema_21', 'demarker')
ORDER BY column_name;

-- List indexes created for 8-21 EMA strategy
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'technical_indicators'
AND indexname LIKE '%ema%'
ORDER BY indexname;

-- Check data coverage
SELECT
    COUNT(*) as total_rows,
    COUNT(ema_8) as has_ema8,
    COUNT(ema_21) as has_ema21,
    COUNT(demarker) as has_demarker,
    ROUND(COUNT(ema_8)::numeric / COUNT(*)::numeric * 100, 1) as ema8_coverage_pct,
    ROUND(COUNT(ema_21)::numeric / COUNT(*)::numeric * 100, 1) as ema21_coverage_pct,
    ROUND(COUNT(demarker)::numeric / COUNT(*)::numeric * 100, 1) as demarker_coverage_pct
FROM technical_indicators
WHERE date >= CURRENT_DATE - INTERVAL '365 days';

-- =============================================================================
-- Migration Complete!
-- Next Step: Run python tools/populate_ema_8_21.py to populate data
-- =============================================================================
