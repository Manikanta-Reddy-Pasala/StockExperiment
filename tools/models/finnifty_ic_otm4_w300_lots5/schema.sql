-- Option chain OHLC+OI storage, isolated from equity historical_data_*.
-- Used by tools/options/* prefetch + backtest pipeline.
--
-- Symbol example: NSE:NIFTY25N1322500CE
--   underlying = NIFTY
--   expiry     = 2025-11-13
--   strike     = 22500
--   opt_type   = CE
--
-- Granularity columns:
--   interval IN ('5m','15m','1h','D')
--   timestamp = epoch seconds (IST)
--   candle_time = naive IST datetime

CREATE TABLE IF NOT EXISTS historical_options (
    id            BIGSERIAL PRIMARY KEY,
    symbol        VARCHAR(64) NOT NULL,
    underlying    VARCHAR(16) NOT NULL,
    expiry        DATE        NOT NULL,
    strike        INTEGER     NOT NULL,
    opt_type      CHAR(2)     NOT NULL,
    interval      VARCHAR(4)  NOT NULL,
    timestamp     BIGINT      NOT NULL,
    candle_time   TIMESTAMP   NOT NULL,
    open          DOUBLE PRECISION NOT NULL,
    high          DOUBLE PRECISION NOT NULL,
    low           DOUBLE PRECISION NOT NULL,
    close         DOUBLE PRECISION NOT NULL,
    volume        BIGINT      NOT NULL DEFAULT 0,
    oi            BIGINT,
    created_at    TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_hopt_sym_int_ts
    ON historical_options (symbol, interval, timestamp);

CREATE INDEX IF NOT EXISTS ix_hopt_underlying_expiry
    ON historical_options (underlying, expiry, opt_type, strike);

CREATE INDEX IF NOT EXISTS ix_hopt_time
    ON historical_options (candle_time);

-- Universe table: known expiries + strikes we plan to fetch.
-- Populated by build_option_universe.py, read by prefetch_options.py.
CREATE TABLE IF NOT EXISTS option_universe (
    symbol        VARCHAR(64) PRIMARY KEY,
    underlying    VARCHAR(16) NOT NULL,
    expiry        DATE        NOT NULL,
    expiry_kind   VARCHAR(8)  NOT NULL,  -- weekly | monthly | daily
    strike        INTEGER     NOT NULL,
    opt_type      CHAR(2)     NOT NULL,
    fetched_d     BOOLEAN     DEFAULT FALSE,
    fetched_5m    BOOLEAN     DEFAULT FALSE,
    last_attempt  TIMESTAMP,
    last_error    TEXT
);

CREATE INDEX IF NOT EXISTS ix_optuniv_expiry
    ON option_universe (expiry, opt_type);
