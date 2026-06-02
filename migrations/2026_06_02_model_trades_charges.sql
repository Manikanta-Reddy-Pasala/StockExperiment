-- 2026-06-02  Add approx broker-charges column to model_trades.
--
-- Surfaces per-trade / per-model / global approx brokerage + statutory charges
-- (Fyers schedule, tools/live/broker_charges.compute_charges). New rows are
-- auto-stamped by the ModelTrade before_insert listener; existing rows are
-- backfilled by tools/backfill_trade_charges.py.
--
-- charges_inr: total approx charges (₹) for THIS BUY/SELL leg. NULL for
-- pre-backfill legacy rows (the UI falls back to an on-the-fly estimate).

BEGIN;

ALTER TABLE model_trades
    ADD COLUMN IF NOT EXISTS charges_inr NUMERIC(14, 4);

COMMENT ON COLUMN model_trades.charges_inr IS
    'Approx broker charges (₹, Fyers rates) for this BUY/SELL leg. Stamped by '
    'the ModelTrade before_insert listener; backfilled for legacy rows.';

COMMIT;
