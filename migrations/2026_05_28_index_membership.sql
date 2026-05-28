-- Point-in-time NSE index membership.
--
-- Backtests must filter their universe with this table to avoid survivorship
-- bias. Today's ind_nifty100list.csv applied backward silently excludes
-- stocks that LEFT the index (YESBANK, IL&FS, DHFL...) and includes stocks
-- that hadn't yet ENTERED (ADANIENT, ZOMATO, NUVAMA, ADANIGREEN...).
--
-- Built from web.archive.org snapshots of niftyindices.com IndexConstituent
-- CSVs by tools/analysis/build_membership_table.py. The CSV exports live at
-- src/data/symbols/n{100,500}_membership.csv. This SQL is a load mirror so
-- the VM can run point-in-time queries without reading the CSVs.
--
-- Half-open interval semantics: start_date <= d < end_date.
-- end_date = 2099-12-31 means "still in the index as of the latest snapshot".

CREATE TABLE IF NOT EXISTS index_membership (
    index_name  VARCHAR(20) NOT NULL,   -- 'n100' or 'n500'
    symbol      VARCHAR(50) NOT NULL,   -- raw NSE symbol e.g. 'RELIANCE'
    start_date  DATE        NOT NULL,
    end_date    DATE        NOT NULL,
    PRIMARY KEY (index_name, symbol, start_date)
);

CREATE INDEX IF NOT EXISTS idx_index_membership_lookup
    ON index_membership (index_name, start_date, end_date);

-- Loader pattern (run from psql after \copy):
--   TRUNCATE index_membership;
--   \copy index_membership(symbol,start_date,end_date) FROM 'n100_membership.csv' WITH (FORMAT csv, HEADER true)
--   UPDATE index_membership SET index_name='n100' WHERE index_name IS NULL;  -- (when staged separately)
--
-- A self-contained one-shot loader script lives at:
--   tools/analysis/load_index_membership.py
