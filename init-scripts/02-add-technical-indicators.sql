-- Migration: Add Technical Indicators and Remove ML Columns
-- This script converts the system from ML-based to technical indicator-based stock selection

-- Step 1: Add new technical indicator columns to daily_suggested_stocks
ALTER TABLE daily_suggested_stocks
ADD COLUMN IF NOT EXISTS rs_rating DECIMAL(10, 4),  -- Relative Strength Rating (1-99)
ADD COLUMN IF NOT EXISTS fast_wave DECIMAL(10, 4),  -- Fast Wave indicator (EMA-based)
ADD COLUMN IF NOT EXISTS slow_wave DECIMAL(10, 4),  -- Slow Wave indicator (EMA-based)
ADD COLUMN IF NOT EXISTS delta DECIMAL(10, 4),      -- Delta (Fast - Slow)
ADD COLUMN IF NOT EXISTS buy_signal BOOLEAN DEFAULT FALSE,   -- Buy signal flag
ADD COLUMN IF NOT EXISTS sell_signal BOOLEAN DEFAULT FALSE;  -- Sell signal flag

-- Step 2: Drop ML-related columns (done in separate statements for safety)
ALTER TABLE daily_suggested_stocks
DROP COLUMN IF EXISTS ml_prediction_score,
DROP COLUMN IF EXISTS ml_price_target,
DROP COLUMN IF EXISTS ml_confidence,
DROP COLUMN IF EXISTS ml_risk_score,
DROP COLUMN IF EXISTS model_type;

-- Step 3: Add new columns to stocks table for caching technical indicators
ALTER TABLE stocks
ADD COLUMN IF NOT EXISTS rs_rating DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS fast_wave DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS slow_wave DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS delta DECIMAL(10, 4),
ADD COLUMN IF NOT EXISTS buy_signal BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS sell_signal BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS indicators_last_updated TIMESTAMP;

-- Step 4: Create index for faster filtering by technical indicators
CREATE INDEX IF NOT EXISTS idx_daily_suggested_rs_rating ON daily_suggested_stocks(rs_rating);
CREATE INDEX IF NOT EXISTS idx_daily_suggested_buy_signal ON daily_suggested_stocks(buy_signal);
CREATE INDEX IF NOT EXISTS idx_daily_suggested_sell_signal ON daily_suggested_stocks(sell_signal);
CREATE INDEX IF NOT EXISTS idx_stocks_rs_rating ON stocks(rs_rating);
CREATE INDEX IF NOT EXISTS idx_stocks_buy_signal ON stocks(buy_signal);

-- Step 5: Drop ML predictions table if it exists (no longer needed)
DROP TABLE IF EXISTS ml_predictions CASCADE;
DROP TABLE IF EXISTS strategy_stock_selections CASCADE;

COMMENT ON COLUMN daily_suggested_stocks.rs_rating IS 'Relative Strength Rating (1-99) comparing stock performance to NIFTY 50';
COMMENT ON COLUMN daily_suggested_stocks.fast_wave IS 'Fast Wave indicator calculated from EMA';
COMMENT ON COLUMN daily_suggested_stocks.slow_wave IS 'Slow Wave indicator (3-period MA of Fast Wave)';
COMMENT ON COLUMN daily_suggested_stocks.delta IS 'Delta = Fast Wave - Slow Wave (positive = bullish momentum)';
COMMENT ON COLUMN daily_suggested_stocks.buy_signal IS 'Buy signal when Fast Wave crosses above Slow Wave';
COMMENT ON COLUMN daily_suggested_stocks.sell_signal IS 'Sell signal when Fast Wave crosses below Slow Wave';
