# 3M India Ltd. (3MINDIA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 32070.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 98 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 13 |
| STOP_HIT | 85 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 85
- **Target hits / Stop hits / Partials:** 13 / 85 / 39
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 9.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 26 | 37.7% | 6 | 43 | 20 | 0.11% | 7.4% |
| BUY @ 2nd Alert (retest1) | 69 | 26 | 37.7% | 6 | 43 | 20 | 0.11% | 7.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 68 | 26 | 38.2% | 7 | 42 | 19 | 0.03% | 2.0% |
| SELL @ 2nd Alert (retest1) | 68 | 26 | 38.2% | 7 | 42 | 19 | 0.03% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 137 | 52 | 38.0% | 13 | 85 | 39 | 0.07% | 9.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 10:55:00 | 29740.00 | 29542.04 | 0.00 | ORB-long ORB[29305.00,29660.00] vol=2.9x ATR=128.20 |
| Stop hit — per-position SL triggered | 2025-05-12 11:25:00 | 29611.80 | 29547.33 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:55:00 | 29485.00 | 29364.53 | 0.00 | ORB-long ORB[29100.00,29395.00] vol=2.4x ATR=99.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 10:00:00 | 29633.51 | 29422.65 | 0.00 | T1 1.5R @ 29633.51 |
| Stop hit — per-position SL triggered | 2025-05-14 10:10:00 | 29485.00 | 29425.56 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 09:40:00 | 29200.00 | 29299.38 | 0.00 | ORB-short ORB[29280.00,29405.00] vol=2.6x ATR=78.77 |
| Stop hit — per-position SL triggered | 2025-05-15 10:10:00 | 29278.77 | 29273.49 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:35:00 | 29535.00 | 29641.04 | 0.00 | ORB-short ORB[29640.00,29850.00] vol=5.0x ATR=73.56 |
| Stop hit — per-position SL triggered | 2025-05-19 09:45:00 | 29608.56 | 29638.18 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:35:00 | 29700.00 | 29537.16 | 0.00 | ORB-long ORB[29385.00,29580.00] vol=1.7x ATR=90.11 |
| Stop hit — per-position SL triggered | 2025-05-21 12:15:00 | 29609.89 | 29661.44 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 11:15:00 | 30335.00 | 30067.23 | 0.00 | ORB-long ORB[29960.00,30320.00] vol=3.9x ATR=59.51 |
| Stop hit — per-position SL triggered | 2025-05-27 11:20:00 | 30275.49 | 30070.13 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 10:00:00 | 29500.00 | 29387.58 | 0.00 | ORB-long ORB[29260.00,29480.00] vol=1.8x ATR=60.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 10:05:00 | 29590.50 | 29429.24 | 0.00 | T1 1.5R @ 29590.50 |
| Stop hit — per-position SL triggered | 2025-06-03 10:10:00 | 29500.00 | 29432.08 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 11:10:00 | 29500.00 | 29559.29 | 0.00 | ORB-short ORB[29505.00,29700.00] vol=5.2x ATR=43.45 |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 29543.45 | 29558.94 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 11:05:00 | 29400.00 | 29538.61 | 0.00 | ORB-short ORB[29455.00,29770.00] vol=3.8x ATR=58.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 14:05:00 | 29312.63 | 29453.16 | 0.00 | T1 1.5R @ 29312.63 |
| Stop hit — per-position SL triggered | 2025-06-09 15:10:00 | 29400.00 | 29423.29 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 11:00:00 | 29380.00 | 29484.05 | 0.00 | ORB-short ORB[29450.00,29645.00] vol=2.1x ATR=56.12 |
| Stop hit — per-position SL triggered | 2025-06-11 11:25:00 | 29436.12 | 29464.16 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 10:15:00 | 29530.00 | 29401.29 | 0.00 | ORB-long ORB[29190.00,29495.00] vol=4.0x ATR=63.01 |
| Stop hit — per-position SL triggered | 2025-06-12 10:20:00 | 29466.99 | 29407.66 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 10:50:00 | 29280.00 | 29137.42 | 0.00 | ORB-long ORB[28935.00,29220.00] vol=3.6x ATR=70.19 |
| Stop hit — per-position SL triggered | 2025-06-13 11:00:00 | 29209.81 | 29153.83 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-06-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 11:10:00 | 29055.00 | 29063.63 | 0.00 | ORB-short ORB[29085.00,29400.00] vol=2.7x ATR=49.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 11:20:00 | 28980.77 | 29059.54 | 0.00 | T1 1.5R @ 28980.77 |
| Stop hit — per-position SL triggered | 2025-06-16 12:30:00 | 29055.00 | 29043.05 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 10:55:00 | 28950.00 | 29092.00 | 0.00 | ORB-short ORB[29020.00,29315.00] vol=3.4x ATR=43.59 |
| Stop hit — per-position SL triggered | 2025-06-17 11:00:00 | 28993.59 | 29085.48 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-06-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 10:55:00 | 28880.00 | 28959.79 | 0.00 | ORB-short ORB[28900.00,29185.00] vol=6.0x ATR=48.74 |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 28928.74 | 28949.72 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-06-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-20 10:25:00 | 28620.00 | 28742.09 | 0.00 | ORB-short ORB[28670.00,28925.00] vol=2.4x ATR=59.74 |
| Stop hit — per-position SL triggered | 2025-06-20 10:30:00 | 28679.74 | 28739.94 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-06-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-24 11:05:00 | 28450.00 | 28517.07 | 0.00 | ORB-short ORB[28500.00,28640.00] vol=2.3x ATR=31.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 12:55:00 | 28402.36 | 28486.89 | 0.00 | T1 1.5R @ 28402.36 |
| Stop hit — per-position SL triggered | 2025-06-24 13:30:00 | 28450.00 | 28482.88 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-06-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 11:05:00 | 28590.00 | 28579.99 | 0.00 | ORB-long ORB[28445.00,28570.00] vol=17.5x ATR=35.71 |
| Stop hit — per-position SL triggered | 2025-06-25 11:15:00 | 28554.29 | 28579.48 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-06-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 10:05:00 | 28600.00 | 28677.19 | 0.00 | ORB-short ORB[28605.00,28865.00] vol=2.0x ATR=45.92 |
| Stop hit — per-position SL triggered | 2025-06-27 10:10:00 | 28645.92 | 28676.88 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:30:00 | 28555.00 | 28607.78 | 0.00 | ORB-short ORB[28635.00,28800.00] vol=5.2x ATR=57.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:45:00 | 28469.03 | 28567.72 | 0.00 | T1 1.5R @ 28469.03 |
| Target hit | 2025-07-02 15:20:00 | 28400.00 | 28445.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2025-07-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:20:00 | 28650.00 | 28649.22 | 0.00 | ORB-long ORB[28450.00,28635.00] vol=3.5x ATR=74.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 12:10:00 | 28761.93 | 28663.70 | 0.00 | T1 1.5R @ 28761.93 |
| Target hit | 2025-07-03 15:20:00 | 28730.00 | 28683.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:30:00 | 29050.00 | 28928.59 | 0.00 | ORB-long ORB[28720.00,28950.00] vol=3.7x ATR=72.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 09:40:00 | 29158.05 | 29068.94 | 0.00 | T1 1.5R @ 29158.05 |
| Target hit | 2025-07-04 10:40:00 | 29190.00 | 29203.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — SELL (started 2025-07-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 09:30:00 | 29350.00 | 29458.93 | 0.00 | ORB-short ORB[29425.00,29645.00] vol=1.9x ATR=76.13 |
| Stop hit — per-position SL triggered | 2025-07-14 09:40:00 | 29426.13 | 29443.70 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:45:00 | 29655.00 | 29827.97 | 0.00 | ORB-short ORB[29750.00,29930.00] vol=1.7x ATR=76.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 10:00:00 | 29539.64 | 29565.71 | 0.00 | T1 1.5R @ 29539.64 |
| Target hit | 2025-07-16 10:05:00 | 29570.00 | 29564.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 25 — BUY (started 2025-07-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:50:00 | 30035.00 | 29881.46 | 0.00 | ORB-long ORB[29660.00,30005.00] vol=2.2x ATR=66.11 |
| Stop hit — per-position SL triggered | 2025-07-21 11:25:00 | 29968.89 | 29973.57 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-07-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:20:00 | 31460.00 | 31212.27 | 0.00 | ORB-long ORB[30945.00,31295.00] vol=1.5x ATR=143.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 11:30:00 | 31674.88 | 31319.95 | 0.00 | T1 1.5R @ 31674.88 |
| Stop hit — per-position SL triggered | 2025-07-23 12:10:00 | 31460.00 | 31364.28 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-07-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 10:35:00 | 31840.00 | 31607.38 | 0.00 | ORB-long ORB[31310.00,31745.00] vol=1.9x ATR=100.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 10:45:00 | 31990.17 | 31671.96 | 0.00 | T1 1.5R @ 31990.17 |
| Stop hit — per-position SL triggered | 2025-07-24 10:50:00 | 31840.00 | 31686.68 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-07-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:50:00 | 30935.00 | 31019.43 | 0.00 | ORB-short ORB[30965.00,31145.00] vol=3.6x ATR=71.92 |
| Stop hit — per-position SL triggered | 2025-07-30 11:00:00 | 31006.92 | 31016.29 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 09:55:00 | 31145.00 | 30789.95 | 0.00 | ORB-long ORB[30620.00,30985.00] vol=1.6x ATR=149.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 10:00:00 | 31369.94 | 30901.29 | 0.00 | T1 1.5R @ 31369.94 |
| Target hit | 2025-08-01 12:40:00 | 31515.00 | 31518.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — SELL (started 2025-08-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:05:00 | 30775.00 | 30933.76 | 0.00 | ORB-short ORB[31030.00,31450.00] vol=1.6x ATR=73.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 11:45:00 | 30664.99 | 30913.02 | 0.00 | T1 1.5R @ 30664.99 |
| Stop hit — per-position SL triggered | 2025-08-12 13:05:00 | 30775.00 | 30809.16 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 09:35:00 | 30250.00 | 30411.58 | 0.00 | ORB-short ORB[30355.00,30690.00] vol=2.1x ATR=75.98 |
| Stop hit — per-position SL triggered | 2025-08-18 09:50:00 | 30325.98 | 30359.48 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:25:00 | 30490.00 | 30290.79 | 0.00 | ORB-long ORB[30175.00,30450.00] vol=1.6x ATR=92.04 |
| Stop hit — per-position SL triggered | 2025-08-19 10:50:00 | 30397.96 | 30334.32 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-08-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:35:00 | 31450.00 | 31204.29 | 0.00 | ORB-long ORB[30830.00,31170.00] vol=2.4x ATR=107.04 |
| Stop hit — per-position SL triggered | 2025-08-20 11:20:00 | 31342.96 | 31285.28 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:40:00 | 31010.00 | 30874.22 | 0.00 | ORB-long ORB[30510.00,30910.00] vol=2.1x ATR=112.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 14:15:00 | 31179.25 | 31010.60 | 0.00 | T1 1.5R @ 31179.25 |
| Target hit | 2025-09-01 15:20:00 | 31280.00 | 31091.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2025-09-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:35:00 | 31490.00 | 31343.34 | 0.00 | ORB-long ORB[31030.00,31400.00] vol=2.5x ATR=82.22 |
| Stop hit — per-position SL triggered | 2025-09-02 11:25:00 | 31407.78 | 31373.47 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:40:00 | 31405.00 | 31277.79 | 0.00 | ORB-long ORB[31095.00,31400.00] vol=1.7x ATR=102.43 |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 31302.57 | 31317.50 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 11:10:00 | 30780.00 | 30959.64 | 0.00 | ORB-short ORB[30875.00,31115.00] vol=2.3x ATR=63.23 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 30843.23 | 30954.05 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 10:10:00 | 30240.00 | 30312.46 | 0.00 | ORB-short ORB[30305.00,30520.00] vol=1.6x ATR=61.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 10:45:00 | 30147.41 | 30265.72 | 0.00 | T1 1.5R @ 30147.41 |
| Stop hit — per-position SL triggered | 2025-09-19 12:35:00 | 30240.00 | 30198.39 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:05:00 | 29900.00 | 29932.85 | 0.00 | ORB-short ORB[29940.00,30085.00] vol=1.9x ATR=45.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 13:25:00 | 29831.17 | 29894.58 | 0.00 | T1 1.5R @ 29831.17 |
| Target hit | 2025-09-23 15:20:00 | 29800.00 | 29872.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2025-10-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:40:00 | 29300.00 | 29234.97 | 0.00 | ORB-long ORB[29015.00,29230.00] vol=6.3x ATR=41.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 11:40:00 | 29362.40 | 29247.27 | 0.00 | T1 1.5R @ 29362.40 |
| Stop hit — per-position SL triggered | 2025-10-06 11:55:00 | 29300.00 | 29260.41 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:45:00 | 29175.00 | 29221.42 | 0.00 | ORB-short ORB[29200.00,29365.00] vol=2.1x ATR=47.53 |
| Stop hit — per-position SL triggered | 2025-10-07 10:55:00 | 29222.53 | 29217.01 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 10:50:00 | 29010.00 | 29108.53 | 0.00 | ORB-short ORB[29100.00,29225.00] vol=2.0x ATR=38.73 |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 29048.73 | 29104.42 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:40:00 | 29460.00 | 29407.65 | 0.00 | ORB-long ORB[29230.00,29360.00] vol=2.0x ATR=67.50 |
| Stop hit — per-position SL triggered | 2025-10-10 09:45:00 | 29392.50 | 29406.80 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 11:05:00 | 29350.00 | 29492.58 | 0.00 | ORB-short ORB[29360.00,29575.00] vol=2.4x ATR=51.61 |
| Stop hit — per-position SL triggered | 2025-10-13 11:40:00 | 29401.61 | 29448.91 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-14 09:55:00 | 29290.00 | 29151.28 | 0.00 | ORB-long ORB[28960.00,29135.00] vol=4.1x ATR=81.51 |
| Stop hit — per-position SL triggered | 2025-10-14 10:10:00 | 29208.49 | 29194.53 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:45:00 | 29250.00 | 29196.92 | 0.00 | ORB-long ORB[28990.00,29245.00] vol=2.6x ATR=53.02 |
| Stop hit — per-position SL triggered | 2025-10-15 10:50:00 | 29196.98 | 29200.81 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:35:00 | 29495.00 | 29393.54 | 0.00 | ORB-long ORB[29200.00,29430.00] vol=1.9x ATR=48.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:45:00 | 29567.50 | 29474.06 | 0.00 | T1 1.5R @ 29567.50 |
| Stop hit — per-position SL triggered | 2025-10-17 10:00:00 | 29495.00 | 29519.74 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:45:00 | 29590.00 | 29476.65 | 0.00 | ORB-long ORB[29385.00,29565.00] vol=1.7x ATR=62.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:05:00 | 29683.55 | 29519.08 | 0.00 | T1 1.5R @ 29683.55 |
| Stop hit — per-position SL triggered | 2025-10-20 10:35:00 | 29590.00 | 29561.79 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:35:00 | 29530.00 | 29623.18 | 0.00 | ORB-short ORB[29535.00,29900.00] vol=1.5x ATR=58.51 |
| Stop hit — per-position SL triggered | 2025-10-24 10:55:00 | 29588.51 | 29619.22 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-10-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:55:00 | 29960.00 | 29879.65 | 0.00 | ORB-long ORB[29800.00,29915.00] vol=2.7x ATR=55.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 10:00:00 | 30042.61 | 29918.23 | 0.00 | T1 1.5R @ 30042.61 |
| Stop hit — per-position SL triggered | 2025-10-27 10:45:00 | 29960.00 | 29987.60 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-10-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 11:05:00 | 29530.00 | 29628.80 | 0.00 | ORB-short ORB[29545.00,29740.00] vol=1.8x ATR=43.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 11:10:00 | 29465.23 | 29590.49 | 0.00 | T1 1.5R @ 29465.23 |
| Stop hit — per-position SL triggered | 2025-10-29 11:20:00 | 29530.00 | 29587.79 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-10-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 10:40:00 | 29915.00 | 29848.56 | 0.00 | ORB-long ORB[29650.00,29870.00] vol=11.0x ATR=54.34 |
| Stop hit — per-position SL triggered | 2025-10-30 11:00:00 | 29860.66 | 29845.86 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-10-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 09:55:00 | 29665.00 | 29744.70 | 0.00 | ORB-short ORB[29705.00,29965.00] vol=1.7x ATR=68.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:05:00 | 29562.40 | 29693.10 | 0.00 | T1 1.5R @ 29562.40 |
| Target hit | 2025-10-31 14:10:00 | 29505.00 | 29503.42 | 0.00 | Trail-exit close>VWAP |

### Cycle 54 — BUY (started 2025-11-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:45:00 | 29940.00 | 29806.29 | 0.00 | ORB-long ORB[29490.00,29835.00] vol=2.3x ATR=104.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:50:00 | 30096.69 | 29863.82 | 0.00 | T1 1.5R @ 30096.69 |
| Target hit | 2025-11-03 14:55:00 | 30530.00 | 30610.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 55 — BUY (started 2025-11-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 09:45:00 | 36360.00 | 36215.58 | 0.00 | ORB-long ORB[35895.00,36325.00] vol=1.9x ATR=110.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 09:50:00 | 36526.24 | 36256.15 | 0.00 | T1 1.5R @ 36526.24 |
| Stop hit — per-position SL triggered | 2025-11-11 10:00:00 | 36360.00 | 36263.98 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-11-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 11:00:00 | 36245.00 | 36072.21 | 0.00 | ORB-long ORB[35845.00,36165.00] vol=3.2x ATR=85.45 |
| Stop hit — per-position SL triggered | 2025-11-12 11:10:00 | 36159.55 | 36086.89 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-11-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:50:00 | 36250.00 | 36086.56 | 0.00 | ORB-long ORB[35820.00,36145.00] vol=1.7x ATR=121.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 09:55:00 | 36432.36 | 36163.71 | 0.00 | T1 1.5R @ 36432.36 |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 36250.00 | 36248.82 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-11-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:55:00 | 35970.00 | 36087.35 | 0.00 | ORB-short ORB[36035.00,36220.00] vol=2.6x ATR=59.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 11:20:00 | 35880.42 | 36060.24 | 0.00 | T1 1.5R @ 35880.42 |
| Stop hit — per-position SL triggered | 2025-11-19 11:55:00 | 35970.00 | 36040.60 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-11-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 11:05:00 | 35680.00 | 35879.67 | 0.00 | ORB-short ORB[35715.00,36240.00] vol=3.4x ATR=56.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 12:15:00 | 35594.78 | 35815.19 | 0.00 | T1 1.5R @ 35594.78 |
| Stop hit — per-position SL triggered | 2025-11-20 12:55:00 | 35680.00 | 35793.88 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-11-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 11:05:00 | 35585.00 | 35517.40 | 0.00 | ORB-long ORB[35285.00,35560.00] vol=1.7x ATR=65.22 |
| Stop hit — per-position SL triggered | 2025-11-24 11:25:00 | 35519.78 | 35520.09 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 09:35:00 | 34800.00 | 34896.32 | 0.00 | ORB-short ORB[34850.00,35175.00] vol=1.9x ATR=91.35 |
| Stop hit — per-position SL triggered | 2025-12-01 10:35:00 | 34891.35 | 34852.75 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:30:00 | 34680.00 | 34831.51 | 0.00 | ORB-short ORB[34785.00,35020.00] vol=1.8x ATR=70.36 |
| Stop hit — per-position SL triggered | 2025-12-02 09:55:00 | 34750.36 | 34776.51 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:15:00 | 34700.00 | 34743.65 | 0.00 | ORB-short ORB[34715.00,34870.00] vol=1.9x ATR=44.81 |
| Stop hit — per-position SL triggered | 2025-12-03 12:10:00 | 34744.81 | 34724.25 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-08 10:00:00 | 34305.00 | 34135.83 | 0.00 | ORB-long ORB[33890.00,34130.00] vol=1.9x ATR=111.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:05:00 | 34471.75 | 34348.61 | 0.00 | T1 1.5R @ 34471.75 |
| Target hit | 2025-12-08 12:55:00 | 34650.00 | 34665.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 65 — SELL (started 2025-12-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:55:00 | 34390.00 | 34598.83 | 0.00 | ORB-short ORB[34495.00,35000.00] vol=6.9x ATR=66.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 11:40:00 | 34290.48 | 34575.83 | 0.00 | T1 1.5R @ 34290.48 |
| Stop hit — per-position SL triggered | 2025-12-10 13:00:00 | 34390.00 | 34520.34 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 10:55:00 | 34090.00 | 34293.99 | 0.00 | ORB-short ORB[34240.00,34630.00] vol=4.0x ATR=76.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 11:05:00 | 33975.60 | 34183.69 | 0.00 | T1 1.5R @ 33975.60 |
| Stop hit — per-position SL triggered | 2025-12-11 11:10:00 | 34090.00 | 34179.19 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:00:00 | 35245.00 | 34999.29 | 0.00 | ORB-long ORB[34475.00,34735.00] vol=1.6x ATR=116.50 |
| Stop hit — per-position SL triggered | 2025-12-12 10:05:00 | 35128.50 | 35020.76 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:30:00 | 35735.00 | 35534.75 | 0.00 | ORB-long ORB[35190.00,35620.00] vol=2.0x ATR=84.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:35:00 | 35861.78 | 35588.07 | 0.00 | T1 1.5R @ 35861.78 |
| Stop hit — per-position SL triggered | 2025-12-17 09:40:00 | 35735.00 | 35617.18 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-12-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:40:00 | 35425.00 | 35278.89 | 0.00 | ORB-long ORB[35075.00,35320.00] vol=3.6x ATR=105.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 11:45:00 | 35582.88 | 35370.24 | 0.00 | T1 1.5R @ 35582.88 |
| Stop hit — per-position SL triggered | 2025-12-22 12:45:00 | 35425.00 | 35408.17 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 10:55:00 | 34910.00 | 34738.57 | 0.00 | ORB-long ORB[34610.00,34780.00] vol=2.2x ATR=63.97 |
| Stop hit — per-position SL triggered | 2025-12-26 11:05:00 | 34846.03 | 34754.13 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-12-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:05:00 | 34240.00 | 34407.44 | 0.00 | ORB-short ORB[34390.00,34700.00] vol=1.9x ATR=56.13 |
| Stop hit — per-position SL triggered | 2025-12-29 11:45:00 | 34296.13 | 34399.24 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 09:40:00 | 34705.00 | 34465.67 | 0.00 | ORB-long ORB[34175.00,34495.00] vol=1.6x ATR=89.87 |
| Stop hit — per-position SL triggered | 2025-12-30 09:45:00 | 34615.13 | 34486.70 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-01-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:00:00 | 34800.00 | 35064.10 | 0.00 | ORB-short ORB[35080.00,35325.00] vol=2.3x ATR=57.11 |
| Stop hit — per-position SL triggered | 2026-01-01 12:10:00 | 34857.11 | 35017.96 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 11:10:00 | 36365.00 | 35981.95 | 0.00 | ORB-long ORB[35600.00,36070.00] vol=15.5x ATR=128.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:15:00 | 36557.20 | 36312.45 | 0.00 | T1 1.5R @ 36557.20 |
| Stop hit — per-position SL triggered | 2026-01-02 12:10:00 | 36365.00 | 36429.41 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 11:15:00 | 34745.00 | 34840.66 | 0.00 | ORB-short ORB[34750.00,35010.00] vol=5.5x ATR=72.36 |
| Stop hit — per-position SL triggered | 2026-01-07 11:50:00 | 34817.36 | 34822.29 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-01-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 11:00:00 | 35225.00 | 35093.58 | 0.00 | ORB-long ORB[34840.00,35140.00] vol=9.1x ATR=94.69 |
| Stop hit — per-position SL triggered | 2026-01-08 11:20:00 | 35130.31 | 35108.52 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:35:00 | 34085.00 | 34234.17 | 0.00 | ORB-short ORB[34250.00,34710.00] vol=7.2x ATR=133.02 |
| Stop hit — per-position SL triggered | 2026-01-09 09:40:00 | 34218.02 | 34228.37 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 11:15:00 | 33770.00 | 33982.76 | 0.00 | ORB-short ORB[33830.00,34285.00] vol=1.6x ATR=83.14 |
| Stop hit — per-position SL triggered | 2026-01-12 12:10:00 | 33853.14 | 33951.42 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-01-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:05:00 | 34520.00 | 34597.24 | 0.00 | ORB-short ORB[34560.00,34975.00] vol=1.5x ATR=95.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:10:00 | 34376.91 | 34486.29 | 0.00 | T1 1.5R @ 34376.91 |
| Target hit | 2026-01-13 11:55:00 | 34460.00 | 34455.37 | 0.00 | Trail-exit close>VWAP |

### Cycle 80 — SELL (started 2026-01-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:40:00 | 34665.00 | 34845.52 | 0.00 | ORB-short ORB[34680.00,34990.00] vol=2.1x ATR=209.28 |
| Stop hit — per-position SL triggered | 2026-01-14 10:35:00 | 34874.28 | 34770.34 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-01-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 10:20:00 | 35280.00 | 35406.54 | 0.00 | ORB-short ORB[35305.00,35590.00] vol=4.1x ATR=96.94 |
| Stop hit — per-position SL triggered | 2026-01-16 10:25:00 | 35376.94 | 35400.55 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-02-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-02 10:10:00 | 34035.00 | 34168.83 | 0.00 | ORB-short ORB[34040.00,34525.00] vol=3.3x ATR=114.16 |
| Stop hit — per-position SL triggered | 2026-02-02 10:20:00 | 34149.16 | 34159.10 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-02-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 09:35:00 | 35155.00 | 34897.93 | 0.00 | ORB-long ORB[34750.00,34950.00] vol=1.8x ATR=112.04 |
| Stop hit — per-position SL triggered | 2026-02-04 09:55:00 | 35042.96 | 34925.72 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-02-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:10:00 | 37620.00 | 37035.21 | 0.00 | ORB-long ORB[36375.00,36815.00] vol=1.8x ATR=251.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:25:00 | 37996.53 | 37292.75 | 0.00 | T1 1.5R @ 37996.53 |
| Stop hit — per-position SL triggered | 2026-02-10 13:35:00 | 37620.00 | 37603.55 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:15:00 | 35660.00 | 35992.51 | 0.00 | ORB-short ORB[35800.00,36335.00] vol=12.2x ATR=101.12 |
| Stop hit — per-position SL triggered | 2026-02-17 11:35:00 | 35761.12 | 35970.54 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-02-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:30:00 | 36320.00 | 36250.86 | 0.00 | ORB-long ORB[36000.00,36300.00] vol=3.6x ATR=97.35 |
| Stop hit — per-position SL triggered | 2026-02-25 10:40:00 | 36222.65 | 36251.65 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:55:00 | 36460.00 | 36338.91 | 0.00 | ORB-long ORB[36175.00,36345.00] vol=2.1x ATR=77.25 |
| Stop hit — per-position SL triggered | 2026-02-27 10:40:00 | 36382.75 | 36375.98 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-03-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:50:00 | 33550.00 | 33621.27 | 0.00 | ORB-short ORB[33555.00,33795.00] vol=2.4x ATR=89.63 |
| Stop hit — per-position SL triggered | 2026-03-10 11:00:00 | 33639.63 | 33619.26 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 34500.00 | 34327.19 | 0.00 | ORB-long ORB[34045.00,34350.00] vol=5.8x ATR=109.56 |
| Stop hit — per-position SL triggered | 2026-03-11 13:05:00 | 34390.44 | 34396.52 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-03-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:30:00 | 32250.00 | 32639.58 | 0.00 | ORB-short ORB[32595.00,32985.00] vol=1.9x ATR=142.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:50:00 | 32036.35 | 32574.21 | 0.00 | T1 1.5R @ 32036.35 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 32250.00 | 32506.38 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2026-03-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:35:00 | 33595.00 | 33334.52 | 0.00 | ORB-long ORB[32820.00,33200.00] vol=1.7x ATR=137.42 |
| Stop hit — per-position SL triggered | 2026-03-17 13:00:00 | 33457.58 | 33460.53 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 33435.00 | 33142.09 | 0.00 | ORB-long ORB[32730.00,33105.00] vol=2.5x ATR=132.81 |
| Stop hit — per-position SL triggered | 2026-03-20 09:40:00 | 33302.19 | 33175.83 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2026-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:30:00 | 31260.00 | 31422.34 | 0.00 | ORB-short ORB[31330.00,31785.00] vol=3.8x ATR=143.66 |
| Stop hit — per-position SL triggered | 2026-03-27 09:45:00 | 31403.66 | 31390.40 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2026-05-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:00:00 | 33630.00 | 33524.26 | 0.00 | ORB-long ORB[33280.00,33550.00] vol=2.1x ATR=104.93 |
| Stop hit — per-position SL triggered | 2026-05-04 10:20:00 | 33525.07 | 33537.09 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 32755.00 | 32945.47 | 0.00 | ORB-short ORB[32890.00,33115.00] vol=3.5x ATR=79.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:45:00 | 32635.89 | 32866.81 | 0.00 | T1 1.5R @ 32635.89 |
| Target hit | 2026-05-05 15:20:00 | 32440.00 | 32580.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 96 — SELL (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 32295.00 | 32465.26 | 0.00 | ORB-short ORB[32410.00,32795.00] vol=2.6x ATR=83.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:10:00 | 32169.85 | 32341.71 | 0.00 | T1 1.5R @ 32169.85 |
| Stop hit — per-position SL triggered | 2026-05-06 12:25:00 | 32295.00 | 32216.90 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:15:00 | 32805.00 | 32710.41 | 0.00 | ORB-long ORB[32445.00,32800.00] vol=10.1x ATR=79.59 |
| Stop hit — per-position SL triggered | 2026-05-07 11:25:00 | 32725.41 | 32712.11 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2026-05-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:55:00 | 32270.00 | 32373.46 | 0.00 | ORB-short ORB[32425.00,32700.00] vol=3.9x ATR=93.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 12:30:00 | 32129.45 | 32267.41 | 0.00 | T1 1.5R @ 32129.45 |
| Target hit | 2026-05-08 13:30:00 | 32250.00 | 32248.87 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 10:55:00 | 29740.00 | 2025-05-12 11:25:00 | 29611.80 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-05-14 09:55:00 | 29485.00 | 2025-05-14 10:00:00 | 29633.51 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-05-14 09:55:00 | 29485.00 | 2025-05-14 10:10:00 | 29485.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-15 09:40:00 | 29200.00 | 2025-05-15 10:10:00 | 29278.77 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-19 09:35:00 | 29535.00 | 2025-05-19 09:45:00 | 29608.56 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-05-21 09:35:00 | 29700.00 | 2025-05-21 12:15:00 | 29609.89 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-27 11:15:00 | 30335.00 | 2025-05-27 11:20:00 | 30275.49 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-06-03 10:00:00 | 29500.00 | 2025-06-03 10:05:00 | 29590.50 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-06-03 10:00:00 | 29500.00 | 2025-06-03 10:10:00 | 29500.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-06 11:10:00 | 29500.00 | 2025-06-06 11:15:00 | 29543.45 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-06-09 11:05:00 | 29400.00 | 2025-06-09 14:05:00 | 29312.63 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-06-09 11:05:00 | 29400.00 | 2025-06-09 15:10:00 | 29400.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-11 11:00:00 | 29380.00 | 2025-06-11 11:25:00 | 29436.12 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-12 10:15:00 | 29530.00 | 2025-06-12 10:20:00 | 29466.99 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-06-13 10:50:00 | 29280.00 | 2025-06-13 11:00:00 | 29209.81 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-06-16 11:10:00 | 29055.00 | 2025-06-16 11:20:00 | 28980.77 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-06-16 11:10:00 | 29055.00 | 2025-06-16 12:30:00 | 29055.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-17 10:55:00 | 28950.00 | 2025-06-17 11:00:00 | 28993.59 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-06-18 10:55:00 | 28880.00 | 2025-06-18 11:15:00 | 28928.74 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-06-20 10:25:00 | 28620.00 | 2025-06-20 10:30:00 | 28679.74 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-06-24 11:05:00 | 28450.00 | 2025-06-24 12:55:00 | 28402.36 | PARTIAL | 0.50 | 0.17% |
| SELL | retest1 | 2025-06-24 11:05:00 | 28450.00 | 2025-06-24 13:30:00 | 28450.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-25 11:05:00 | 28590.00 | 2025-06-25 11:15:00 | 28554.29 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2025-06-27 10:05:00 | 28600.00 | 2025-06-27 10:10:00 | 28645.92 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-02 09:30:00 | 28555.00 | 2025-07-02 09:45:00 | 28469.03 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-02 09:30:00 | 28555.00 | 2025-07-02 15:20:00 | 28400.00 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2025-07-03 10:20:00 | 28650.00 | 2025-07-03 12:10:00 | 28761.93 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-07-03 10:20:00 | 28650.00 | 2025-07-03 15:20:00 | 28730.00 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2025-07-04 09:30:00 | 29050.00 | 2025-07-04 09:40:00 | 29158.05 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-07-04 09:30:00 | 29050.00 | 2025-07-04 10:40:00 | 29190.00 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2025-07-14 09:30:00 | 29350.00 | 2025-07-14 09:40:00 | 29426.13 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-16 09:45:00 | 29655.00 | 2025-07-16 10:00:00 | 29539.64 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-16 09:45:00 | 29655.00 | 2025-07-16 10:05:00 | 29570.00 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2025-07-21 09:50:00 | 30035.00 | 2025-07-21 11:25:00 | 29968.89 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-23 10:20:00 | 31460.00 | 2025-07-23 11:30:00 | 31674.88 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-07-23 10:20:00 | 31460.00 | 2025-07-23 12:10:00 | 31460.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-24 10:35:00 | 31840.00 | 2025-07-24 10:45:00 | 31990.17 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-07-24 10:35:00 | 31840.00 | 2025-07-24 10:50:00 | 31840.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-30 10:50:00 | 30935.00 | 2025-07-30 11:00:00 | 31006.92 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-01 09:55:00 | 31145.00 | 2025-08-01 10:00:00 | 31369.94 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-08-01 09:55:00 | 31145.00 | 2025-08-01 12:40:00 | 31515.00 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2025-08-12 11:05:00 | 30775.00 | 2025-08-12 11:45:00 | 30664.99 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-12 11:05:00 | 30775.00 | 2025-08-12 13:05:00 | 30775.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-18 09:35:00 | 30250.00 | 2025-08-18 09:50:00 | 30325.98 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-19 10:25:00 | 30490.00 | 2025-08-19 10:50:00 | 30397.96 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-08-20 10:35:00 | 31450.00 | 2025-08-20 11:20:00 | 31342.96 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-01 09:40:00 | 31010.00 | 2025-09-01 14:15:00 | 31179.25 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-09-01 09:40:00 | 31010.00 | 2025-09-01 15:20:00 | 31280.00 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2025-09-02 10:35:00 | 31490.00 | 2025-09-02 11:25:00 | 31407.78 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-09-03 09:40:00 | 31405.00 | 2025-09-03 10:15:00 | 31302.57 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-09-10 11:10:00 | 30780.00 | 2025-09-10 11:15:00 | 30843.23 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-19 10:10:00 | 30240.00 | 2025-09-19 10:45:00 | 30147.41 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-09-19 10:10:00 | 30240.00 | 2025-09-19 12:35:00 | 30240.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-23 11:05:00 | 29900.00 | 2025-09-23 13:25:00 | 29831.17 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-09-23 11:05:00 | 29900.00 | 2025-09-23 15:20:00 | 29800.00 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-06 10:40:00 | 29300.00 | 2025-10-06 11:40:00 | 29362.40 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-10-06 10:40:00 | 29300.00 | 2025-10-06 11:55:00 | 29300.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-07 10:45:00 | 29175.00 | 2025-10-07 10:55:00 | 29222.53 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-10-09 10:50:00 | 29010.00 | 2025-10-09 11:15:00 | 29048.73 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-10-10 09:40:00 | 29460.00 | 2025-10-10 09:45:00 | 29392.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-13 11:05:00 | 29350.00 | 2025-10-13 11:40:00 | 29401.61 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-10-14 09:55:00 | 29290.00 | 2025-10-14 10:10:00 | 29208.49 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-15 10:45:00 | 29250.00 | 2025-10-15 10:50:00 | 29196.98 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-10-17 09:35:00 | 29495.00 | 2025-10-17 09:45:00 | 29567.50 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-10-17 09:35:00 | 29495.00 | 2025-10-17 10:00:00 | 29495.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-20 09:45:00 | 29590.00 | 2025-10-20 10:05:00 | 29683.55 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-10-20 09:45:00 | 29590.00 | 2025-10-20 10:35:00 | 29590.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-24 10:35:00 | 29530.00 | 2025-10-24 10:55:00 | 29588.51 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-27 09:55:00 | 29960.00 | 2025-10-27 10:00:00 | 30042.61 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-10-27 09:55:00 | 29960.00 | 2025-10-27 10:45:00 | 29960.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-29 11:05:00 | 29530.00 | 2025-10-29 11:10:00 | 29465.23 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-10-29 11:05:00 | 29530.00 | 2025-10-29 11:20:00 | 29530.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-30 10:40:00 | 29915.00 | 2025-10-30 11:00:00 | 29860.66 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-31 09:55:00 | 29665.00 | 2025-10-31 10:05:00 | 29562.40 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-10-31 09:55:00 | 29665.00 | 2025-10-31 14:10:00 | 29505.00 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2025-11-03 09:45:00 | 29940.00 | 2025-11-03 09:50:00 | 30096.69 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-11-03 09:45:00 | 29940.00 | 2025-11-03 14:55:00 | 30530.00 | TARGET_HIT | 0.50 | 1.97% |
| BUY | retest1 | 2025-11-11 09:45:00 | 36360.00 | 2025-11-11 09:50:00 | 36526.24 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-11-11 09:45:00 | 36360.00 | 2025-11-11 10:00:00 | 36360.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-12 11:00:00 | 36245.00 | 2025-11-12 11:10:00 | 36159.55 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-17 09:50:00 | 36250.00 | 2025-11-17 09:55:00 | 36432.36 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-11-17 09:50:00 | 36250.00 | 2025-11-17 10:15:00 | 36250.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-19 10:55:00 | 35970.00 | 2025-11-19 11:20:00 | 35880.42 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-11-19 10:55:00 | 35970.00 | 2025-11-19 11:55:00 | 35970.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-20 11:05:00 | 35680.00 | 2025-11-20 12:15:00 | 35594.78 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-11-20 11:05:00 | 35680.00 | 2025-11-20 12:55:00 | 35680.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-24 11:05:00 | 35585.00 | 2025-11-24 11:25:00 | 35519.78 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-01 09:35:00 | 34800.00 | 2025-12-01 10:35:00 | 34891.35 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-02 09:30:00 | 34680.00 | 2025-12-02 09:55:00 | 34750.36 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-03 11:15:00 | 34700.00 | 2025-12-03 12:10:00 | 34744.81 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-12-08 10:00:00 | 34305.00 | 2025-12-08 10:05:00 | 34471.75 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-12-08 10:00:00 | 34305.00 | 2025-12-08 12:55:00 | 34650.00 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2025-12-10 10:55:00 | 34390.00 | 2025-12-10 11:40:00 | 34290.48 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-12-10 10:55:00 | 34390.00 | 2025-12-10 13:00:00 | 34390.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-11 10:55:00 | 34090.00 | 2025-12-11 11:05:00 | 33975.60 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-11 10:55:00 | 34090.00 | 2025-12-11 11:10:00 | 34090.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-12 10:00:00 | 35245.00 | 2025-12-12 10:05:00 | 35128.50 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-17 09:30:00 | 35735.00 | 2025-12-17 09:35:00 | 35861.78 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-12-17 09:30:00 | 35735.00 | 2025-12-17 09:40:00 | 35735.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-22 10:40:00 | 35425.00 | 2025-12-22 11:45:00 | 35582.88 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-12-22 10:40:00 | 35425.00 | 2025-12-22 12:45:00 | 35425.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-26 10:55:00 | 34910.00 | 2025-12-26 11:05:00 | 34846.03 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-29 11:05:00 | 34240.00 | 2025-12-29 11:45:00 | 34296.13 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-12-30 09:40:00 | 34705.00 | 2025-12-30 09:45:00 | 34615.13 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-01 11:00:00 | 34800.00 | 2026-01-01 12:10:00 | 34857.11 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-01-02 11:10:00 | 36365.00 | 2026-01-02 11:15:00 | 36557.20 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-01-02 11:10:00 | 36365.00 | 2026-01-02 12:10:00 | 36365.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-07 11:15:00 | 34745.00 | 2026-01-07 11:50:00 | 34817.36 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-08 11:00:00 | 35225.00 | 2026-01-08 11:20:00 | 35130.31 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-09 09:35:00 | 34085.00 | 2026-01-09 09:40:00 | 34218.02 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-01-12 11:15:00 | 33770.00 | 2026-01-12 12:10:00 | 33853.14 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-13 11:05:00 | 34520.00 | 2026-01-13 11:10:00 | 34376.91 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-01-13 11:05:00 | 34520.00 | 2026-01-13 11:55:00 | 34460.00 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2026-01-14 09:40:00 | 34665.00 | 2026-01-14 10:35:00 | 34874.28 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2026-01-16 10:20:00 | 35280.00 | 2026-01-16 10:25:00 | 35376.94 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-02 10:10:00 | 34035.00 | 2026-02-02 10:20:00 | 34149.16 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-04 09:35:00 | 35155.00 | 2026-02-04 09:55:00 | 35042.96 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-10 10:10:00 | 37620.00 | 2026-02-10 10:25:00 | 37996.53 | PARTIAL | 0.50 | 1.00% |
| BUY | retest1 | 2026-02-10 10:10:00 | 37620.00 | 2026-02-10 13:35:00 | 37620.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 11:15:00 | 35660.00 | 2026-02-17 11:35:00 | 35761.12 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-25 10:30:00 | 36320.00 | 2026-02-25 10:40:00 | 36222.65 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-27 09:55:00 | 36460.00 | 2026-02-27 10:40:00 | 36382.75 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-10 10:50:00 | 33550.00 | 2026-03-10 11:00:00 | 33639.63 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-11 10:55:00 | 34500.00 | 2026-03-11 13:05:00 | 34390.44 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-16 10:30:00 | 32250.00 | 2026-03-16 10:50:00 | 32036.35 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-03-16 10:30:00 | 32250.00 | 2026-03-16 11:15:00 | 32250.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:35:00 | 33595.00 | 2026-03-17 13:00:00 | 33457.58 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-20 09:35:00 | 33435.00 | 2026-03-20 09:40:00 | 33302.19 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-27 09:30:00 | 31260.00 | 2026-03-27 09:45:00 | 31403.66 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-05-04 10:00:00 | 33630.00 | 2026-05-04 10:20:00 | 33525.07 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-05 10:10:00 | 32755.00 | 2026-05-05 10:45:00 | 32635.89 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-05-05 10:10:00 | 32755.00 | 2026-05-05 15:20:00 | 32440.00 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2026-05-06 09:40:00 | 32295.00 | 2026-05-06 10:10:00 | 32169.85 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-05-06 09:40:00 | 32295.00 | 2026-05-06 12:25:00 | 32295.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 11:15:00 | 32805.00 | 2026-05-07 11:25:00 | 32725.41 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-08 09:55:00 | 32270.00 | 2026-05-08 12:30:00 | 32129.45 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-05-08 09:55:00 | 32270.00 | 2026-05-08 13:30:00 | 32250.00 | TARGET_HIT | 0.50 | 0.06% |
