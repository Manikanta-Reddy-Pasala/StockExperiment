-- Per-model observe mode. When signals_only=true the model still emits signals
-- + ranking, but executors place no real orders and don't mutate the ledger.
-- Default false = enabled models trade live (no behavior change for existing rows).
ALTER TABLE model_settings
  ADD COLUMN IF NOT EXISTS signals_only BOOLEAN NOT NULL DEFAULT false;
