-- Multi-holding support for K>1 models (e.g. momentum_retest_n500, K=3).
-- Single-holding models keep their one position in model_ledger.open_symbol;
-- multi-holding models store their N open positions here (one row per symbol).
-- Cash / realized_pnl / stats stay in model_ledger (shared). Non-disruptive:
-- the single-holding flow never reads or writes this table.

CREATE TABLE IF NOT EXISTS model_holdings (
    id          SERIAL PRIMARY KEY,
    model_name  VARCHAR(64) NOT NULL REFERENCES model_settings(model_name),
    symbol      VARCHAR(64) NOT NULL,           -- normalized NSE:SYM-EQ
    qty         INTEGER     NOT NULL,
    entry_px    NUMERIC(14,4) NOT NULL,
    entry_date  DATE        DEFAULT CURRENT_DATE,
    created_at  TIMESTAMP   DEFAULT now(),
    UNIQUE (model_name, symbol)
);

CREATE INDEX IF NOT EXISTS idx_model_holdings_model ON model_holdings(model_name);
