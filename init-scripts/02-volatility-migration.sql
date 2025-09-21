-- Volatility Schema Migration
-- This script adds volatility columns and indexes if they don't exist
-- Safe to run on existing databases

-- Add volatility columns to stocks table if they don't exist
DO $$
BEGIN
    -- Add atr_14 column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stocks' AND column_name = 'atr_14') THEN
        ALTER TABLE stocks ADD COLUMN atr_14 DOUBLE PRECISION;
        COMMENT ON COLUMN stocks.atr_14 IS 'Average True Range (14-day period)';
    END IF;

    -- Add atr_percentage column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stocks' AND column_name = 'atr_percentage') THEN
        ALTER TABLE stocks ADD COLUMN atr_percentage DOUBLE PRECISION;
        COMMENT ON COLUMN stocks.atr_percentage IS 'ATR as percentage of current price';
    END IF;

    -- Add historical_volatility_1y column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stocks' AND column_name = 'historical_volatility_1y') THEN
        ALTER TABLE stocks ADD COLUMN historical_volatility_1y DOUBLE PRECISION;
        COMMENT ON COLUMN stocks.historical_volatility_1y IS 'Annualized historical volatility';
    END IF;

    -- Add bid_ask_spread column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stocks' AND column_name = 'bid_ask_spread') THEN
        ALTER TABLE stocks ADD COLUMN bid_ask_spread DOUBLE PRECISION;
        COMMENT ON COLUMN stocks.bid_ask_spread IS 'Estimated bid-ask spread';
    END IF;

    -- Add avg_daily_volume_20d column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stocks' AND column_name = 'avg_daily_volume_20d') THEN
        ALTER TABLE stocks ADD COLUMN avg_daily_volume_20d DOUBLE PRECISION;
        COMMENT ON COLUMN stocks.avg_daily_volume_20d IS '20-day average daily volume';
    END IF;

    -- Add updated_at column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stocks' AND column_name = 'updated_at') THEN
        ALTER TABLE stocks ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
        COMMENT ON COLUMN stocks.updated_at IS 'For tracking volatility updates';
    END IF;
END
$$;

-- Create volatility indexes for better screening performance
CREATE INDEX IF NOT EXISTS idx_stocks_atr_percentage ON stocks(atr_percentage);
CREATE INDEX IF NOT EXISTS idx_stocks_beta ON stocks(beta);
CREATE INDEX IF NOT EXISTS idx_stocks_historical_volatility ON stocks(historical_volatility_1y);
CREATE INDEX IF NOT EXISTS idx_stocks_avg_volume_20d ON stocks(avg_daily_volume_20d);
CREATE INDEX IF NOT EXISTS idx_stocks_updated_at ON stocks(updated_at);

-- Update any existing stocks to have updated_at set to current timestamp if NULL
UPDATE stocks
SET updated_at = CURRENT_TIMESTAMP
WHERE updated_at IS NULL;