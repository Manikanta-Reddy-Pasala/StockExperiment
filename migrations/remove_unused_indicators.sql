-- Migration: Remove Unused Technical Indicators
-- Date: 2025-10-31
-- Purpose: Keep only indicators needed for 8-21 EMA strategy

-- =============================================================================
-- IMPORTANT: This will permanently delete data!
-- Backup your database before running this migration.
-- =============================================================================

-- Backup command (run before this migration):
-- docker exec trading_system_db pg_dump -U trader trading_system > backup_before_cleanup.sql

-- =============================================================================
-- STEP 1: Drop unused SMA columns
-- =============================================================================

ALTER TABLE technical_indicators DROP COLUMN IF EXISTS sma_5;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS sma_10;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS sma_20;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS sma_100;
-- Keep: sma_50, sma_200 (for context)

-- =============================================================================
-- STEP 2: Drop unused EMA columns
-- =============================================================================

ALTER TABLE technical_indicators DROP COLUMN IF EXISTS ema_12;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS ema_26;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS ema_50;
-- Keep: ema_8, ema_21 (core strategy)

-- =============================================================================
-- STEP 3: Drop unused momentum indicators
-- =============================================================================

ALTER TABLE technical_indicators DROP COLUMN IF EXISTS rsi_14;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS macd;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS macd_signal;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS macd_histogram;
-- Keep: demarker (core strategy)

-- =============================================================================
-- STEP 4: Drop unused volatility indicators
-- =============================================================================

ALTER TABLE technical_indicators DROP COLUMN IF EXISTS atr_14;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS atr_percentage;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS bb_upper;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS bb_middle;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS bb_lower;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS bb_width;

-- =============================================================================
-- STEP 5: Drop unused trend indicators
-- =============================================================================

ALTER TABLE technical_indicators DROP COLUMN IF EXISTS adx_14;

-- =============================================================================
-- STEP 6: Drop unused volume indicators
-- =============================================================================

ALTER TABLE technical_indicators DROP COLUMN IF EXISTS obv;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS volume_sma_20;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS volume_ratio;

-- =============================================================================
-- STEP 7: Drop custom indicators
-- =============================================================================

ALTER TABLE technical_indicators DROP COLUMN IF EXISTS price_momentum_5d;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS price_momentum_20d;
ALTER TABLE technical_indicators DROP COLUMN IF EXISTS volatility_rank;

-- =============================================================================
-- STEP 8: Verify remaining columns
-- =============================================================================

-- List all columns in technical_indicators table
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'technical_indicators'
ORDER BY ordinal_position;

-- Expected columns remaining:
-- id, symbol, date
-- sma_50, sma_200
-- ema_8, ema_21
-- demarker
-- calculation_date, data_points_used

-- =============================================================================
-- STEP 9: Run VACUUM to reclaim disk space
-- =============================================================================

VACUUM FULL technical_indicators;

-- =============================================================================
-- STEP 10: Check database size reduction
-- =============================================================================

SELECT pg_size_pretty(pg_total_relation_size('technical_indicators')) as table_size;

-- =============================================================================
-- Migration complete!
-- Estimated space savings: 100-150MB
-- =============================================================================
