-- Migration: Add 8-21 EMA Strategy Columns
-- Date: 2025-10-31
-- Purpose: Add missing columns for pure 8-21 EMA strategy

-- Add EMA 8 column
ALTER TABLE technical_indicators
ADD COLUMN IF NOT EXISTS ema_8 DOUBLE PRECISION;

-- Add EMA 21 column
ALTER TABLE technical_indicators
ADD COLUMN IF NOT EXISTS ema_21 DOUBLE PRECISION;

-- Add DeMarker oscillator column
ALTER TABLE technical_indicators
ADD COLUMN IF NOT EXISTS demarker DOUBLE PRECISION;

-- Add indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_technical_ema8 ON technical_indicators(ema_8);
CREATE INDEX IF NOT EXISTS idx_technical_ema21 ON technical_indicators(ema_21);
CREATE INDEX IF NOT EXISTS idx_technical_demarker ON technical_indicators(demarker);

-- Add composite index for EMA strategy queries
CREATE INDEX IF NOT EXISTS idx_technical_ema_strategy
ON technical_indicators(symbol, date, ema_8, ema_21, demarker);

COMMENT ON COLUMN technical_indicators.ema_8 IS '8-day Exponential Moving Average (fast EMA for 8-21 strategy)';
COMMENT ON COLUMN technical_indicators.ema_21 IS '21-day Exponential Moving Average (slow EMA for 8-21 strategy)';
COMMENT ON COLUMN technical_indicators.demarker IS 'DeMarker oscillator (0-1 range, <0.30=oversold, >0.70=overbought)';

-- Verify columns added
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'technical_indicators'
AND column_name IN ('ema_8', 'ema_21', 'demarker')
ORDER BY column_name;
