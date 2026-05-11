# Usha Martin Ltd. (USHAMART)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 472.00
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
| ENTRY1 | 95 |
| ENTRY2 | 0 |
| PARTIAL | 43 |
| TARGET_HIT | 18 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 138 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 77
- **Target hits / Stop hits / Partials:** 18 / 77 / 43
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 27.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 33 | 43.4% | 9 | 43 | 24 | 0.26% | 19.7% |
| BUY @ 2nd Alert (retest1) | 76 | 33 | 43.4% | 9 | 43 | 24 | 0.26% | 19.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 62 | 28 | 45.2% | 9 | 34 | 19 | 0.12% | 7.7% |
| SELL @ 2nd Alert (retest1) | 62 | 28 | 45.2% | 9 | 34 | 19 | 0.12% | 7.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 138 | 61 | 44.2% | 18 | 77 | 43 | 0.20% | 27.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:35:00 | 313.30 | 312.28 | 0.00 | ORB-long ORB[309.85,313.15] vol=1.9x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-05-15 09:45:00 | 312.25 | 312.46 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 09:35:00 | 310.10 | 311.81 | 0.00 | ORB-short ORB[311.15,314.80] vol=1.6x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 11:50:00 | 308.52 | 310.31 | 0.00 | T1 1.5R @ 308.52 |
| Stop hit — per-position SL triggered | 2025-05-16 14:20:00 | 310.10 | 309.05 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 09:45:00 | 313.05 | 311.93 | 0.00 | ORB-long ORB[310.65,312.95] vol=2.9x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-19 10:15:00 | 314.97 | 312.43 | 0.00 | T1 1.5R @ 314.97 |
| Target hit | 2025-05-19 13:30:00 | 329.55 | 331.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2025-05-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 10:40:00 | 319.00 | 320.47 | 0.00 | ORB-short ORB[319.05,323.60] vol=1.6x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-22 11:30:00 | 317.70 | 320.14 | 0.00 | T1 1.5R @ 317.70 |
| Stop hit — per-position SL triggered | 2025-05-22 12:10:00 | 319.00 | 319.54 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:20:00 | 328.60 | 327.11 | 0.00 | ORB-long ORB[322.90,327.40] vol=3.7x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-05-23 13:25:00 | 327.32 | 328.12 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 10:55:00 | 316.60 | 317.22 | 0.00 | ORB-short ORB[317.00,318.70] vol=3.4x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-05-28 11:15:00 | 317.29 | 317.13 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:00:00 | 308.55 | 309.78 | 0.00 | ORB-short ORB[310.35,312.45] vol=1.7x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 309.42 | 309.79 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:00:00 | 312.90 | 311.47 | 0.00 | ORB-long ORB[306.85,311.00] vol=1.9x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-06-02 11:20:00 | 312.21 | 311.80 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:45:00 | 309.65 | 312.34 | 0.00 | ORB-short ORB[312.85,315.00] vol=3.7x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-06-04 10:00:00 | 310.71 | 311.92 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 10:00:00 | 307.80 | 308.49 | 0.00 | ORB-short ORB[308.30,309.95] vol=1.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-06-05 10:20:00 | 308.58 | 308.38 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:30:00 | 310.00 | 308.94 | 0.00 | ORB-long ORB[305.30,309.70] vol=5.5x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 09:55:00 | 311.63 | 309.96 | 0.00 | T1 1.5R @ 311.63 |
| Stop hit — per-position SL triggered | 2025-06-09 11:40:00 | 310.00 | 310.61 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:10:00 | 311.00 | 311.94 | 0.00 | ORB-short ORB[311.15,314.70] vol=2.2x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 311.82 | 312.01 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 10:45:00 | 309.50 | 307.72 | 0.00 | ORB-long ORB[303.05,307.45] vol=4.1x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-06-13 12:30:00 | 308.44 | 309.19 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 10:40:00 | 312.00 | 310.89 | 0.00 | ORB-long ORB[307.10,311.45] vol=2.4x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-06-18 10:45:00 | 310.97 | 310.93 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:40:00 | 313.20 | 311.24 | 0.00 | ORB-long ORB[307.55,312.15] vol=2.0x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:50:00 | 314.82 | 311.74 | 0.00 | T1 1.5R @ 314.82 |
| Stop hit — per-position SL triggered | 2025-06-20 09:55:00 | 313.20 | 311.81 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-06-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:45:00 | 318.50 | 316.21 | 0.00 | ORB-long ORB[313.25,317.45] vol=1.9x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 09:50:00 | 320.21 | 317.14 | 0.00 | T1 1.5R @ 320.21 |
| Stop hit — per-position SL triggered | 2025-06-24 09:55:00 | 318.50 | 318.33 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:45:00 | 351.10 | 346.90 | 0.00 | ORB-long ORB[343.55,347.75] vol=2.3x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:50:00 | 353.77 | 348.87 | 0.00 | T1 1.5R @ 353.77 |
| Stop hit — per-position SL triggered | 2025-06-27 09:55:00 | 351.10 | 349.12 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:00:00 | 359.30 | 362.59 | 0.00 | ORB-short ORB[361.80,365.00] vol=1.7x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 360.50 | 361.73 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:45:00 | 368.75 | 366.38 | 0.00 | ORB-long ORB[362.00,367.40] vol=4.2x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 09:50:00 | 371.33 | 369.14 | 0.00 | T1 1.5R @ 371.33 |
| Target hit | 2025-07-03 13:15:00 | 371.55 | 371.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — BUY (started 2025-07-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 10:45:00 | 369.05 | 366.59 | 0.00 | ORB-long ORB[365.10,368.80] vol=5.3x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-07-07 10:50:00 | 367.47 | 366.66 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:35:00 | 375.75 | 373.51 | 0.00 | ORB-long ORB[371.40,374.25] vol=2.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-07-09 09:40:00 | 374.67 | 373.54 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:55:00 | 374.15 | 371.02 | 0.00 | ORB-long ORB[367.60,371.70] vol=4.8x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 10:15:00 | 376.61 | 372.89 | 0.00 | T1 1.5R @ 376.61 |
| Target hit | 2025-07-14 15:20:00 | 389.35 | 385.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-07-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 10:05:00 | 386.55 | 389.16 | 0.00 | ORB-short ORB[389.90,393.20] vol=1.6x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-07-16 10:55:00 | 387.91 | 388.44 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 09:30:00 | 387.15 | 386.08 | 0.00 | ORB-long ORB[382.30,384.65] vol=6.6x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-07-17 09:35:00 | 385.51 | 386.08 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 380.20 | 383.20 | 0.00 | ORB-short ORB[384.30,386.75] vol=2.7x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 11:00:00 | 378.51 | 382.16 | 0.00 | T1 1.5R @ 378.51 |
| Stop hit — per-position SL triggered | 2025-07-18 11:05:00 | 380.20 | 382.07 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-07-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 11:10:00 | 385.65 | 384.00 | 0.00 | ORB-long ORB[379.95,384.45] vol=1.9x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-07-21 11:40:00 | 384.60 | 384.18 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-07-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:00:00 | 381.55 | 383.86 | 0.00 | ORB-short ORB[383.30,386.30] vol=1.8x ATR=1.23 |
| Stop hit — per-position SL triggered | 2025-07-22 10:35:00 | 382.78 | 383.20 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-07-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 11:00:00 | 378.70 | 380.20 | 0.00 | ORB-short ORB[380.40,383.60] vol=1.9x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 11:05:00 | 377.58 | 380.05 | 0.00 | T1 1.5R @ 377.58 |
| Target hit | 2025-07-23 14:35:00 | 378.40 | 378.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 29 — SELL (started 2025-07-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:00:00 | 378.10 | 380.13 | 0.00 | ORB-short ORB[379.20,382.90] vol=4.0x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-07-24 11:20:00 | 379.05 | 379.97 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-07-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:40:00 | 385.60 | 382.96 | 0.00 | ORB-long ORB[379.30,385.00] vol=5.5x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 10:45:00 | 388.00 | 383.93 | 0.00 | T1 1.5R @ 388.00 |
| Stop hit — per-position SL triggered | 2025-07-30 10:50:00 | 385.60 | 384.09 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-07-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 09:35:00 | 378.60 | 374.90 | 0.00 | ORB-long ORB[369.95,374.80] vol=1.7x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:50:00 | 381.29 | 376.34 | 0.00 | T1 1.5R @ 381.29 |
| Stop hit — per-position SL triggered | 2025-07-31 09:55:00 | 378.60 | 376.45 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-08-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 11:10:00 | 372.20 | 375.08 | 0.00 | ORB-short ORB[375.60,379.80] vol=2.6x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:35:00 | 370.86 | 374.68 | 0.00 | T1 1.5R @ 370.86 |
| Target hit | 2025-08-01 13:10:00 | 371.85 | 371.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 33 — SELL (started 2025-08-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 09:45:00 | 335.95 | 337.70 | 0.00 | ORB-short ORB[337.10,340.05] vol=2.0x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 10:10:00 | 334.00 | 335.71 | 0.00 | T1 1.5R @ 334.00 |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 335.95 | 335.72 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 11:15:00 | 346.05 | 342.55 | 0.00 | ORB-long ORB[340.10,344.90] vol=3.8x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 12:35:00 | 348.05 | 343.86 | 0.00 | T1 1.5R @ 348.05 |
| Target hit | 2025-08-12 14:00:00 | 348.85 | 349.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — BUY (started 2025-08-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:35:00 | 372.70 | 371.84 | 0.00 | ORB-long ORB[368.60,372.60] vol=8.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-08-20 10:40:00 | 371.79 | 372.07 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-08-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:10:00 | 368.15 | 365.28 | 0.00 | ORB-long ORB[363.50,366.35] vol=2.3x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-08-21 11:20:00 | 367.02 | 367.65 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-08-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:00:00 | 375.65 | 370.49 | 0.00 | ORB-long ORB[367.10,371.90] vol=2.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-08-22 10:05:00 | 374.43 | 371.81 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-08-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 09:45:00 | 390.60 | 389.75 | 0.00 | ORB-long ORB[382.30,388.20] vol=2.0x ATR=2.34 |
| Stop hit — per-position SL triggered | 2025-08-26 15:05:00 | 388.26 | 390.79 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:55:00 | 392.80 | 389.48 | 0.00 | ORB-long ORB[380.45,385.90] vol=10.1x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-09-01 10:00:00 | 390.77 | 391.46 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 10:10:00 | 385.35 | 383.34 | 0.00 | ORB-long ORB[380.55,384.00] vol=2.5x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 383.97 | 383.71 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:35:00 | 383.25 | 380.82 | 0.00 | ORB-long ORB[378.00,382.50] vol=1.5x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:40:00 | 384.76 | 382.64 | 0.00 | T1 1.5R @ 384.76 |
| Target hit | 2025-09-05 10:40:00 | 384.90 | 384.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — SELL (started 2025-09-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 11:00:00 | 388.75 | 392.08 | 0.00 | ORB-short ORB[390.15,396.00] vol=2.2x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 11:15:00 | 386.65 | 391.58 | 0.00 | T1 1.5R @ 386.65 |
| Target hit | 2025-09-10 15:20:00 | 384.10 | 387.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-09-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:45:00 | 385.80 | 383.50 | 0.00 | ORB-long ORB[381.95,385.45] vol=1.6x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 10:50:00 | 387.20 | 384.20 | 0.00 | T1 1.5R @ 387.20 |
| Stop hit — per-position SL triggered | 2025-09-12 10:55:00 | 385.80 | 384.27 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-09-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 11:00:00 | 406.20 | 408.35 | 0.00 | ORB-short ORB[407.60,411.25] vol=2.0x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-09-17 11:10:00 | 407.35 | 408.31 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-09-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:05:00 | 406.90 | 402.83 | 0.00 | ORB-long ORB[399.15,404.15] vol=3.1x ATR=2.05 |
| Stop hit — per-position SL triggered | 2025-09-18 10:10:00 | 404.85 | 403.05 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-09-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 10:20:00 | 420.45 | 416.24 | 0.00 | ORB-long ORB[412.45,417.30] vol=2.5x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 10:25:00 | 423.28 | 417.56 | 0.00 | T1 1.5R @ 423.28 |
| Stop hit — per-position SL triggered | 2025-09-19 10:30:00 | 420.45 | 417.85 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-09-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 09:45:00 | 451.90 | 449.19 | 0.00 | ORB-long ORB[445.05,450.65] vol=2.3x ATR=2.51 |
| Stop hit — per-position SL triggered | 2025-09-26 10:20:00 | 449.39 | 449.74 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-10-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:55:00 | 444.25 | 446.67 | 0.00 | ORB-short ORB[445.00,450.60] vol=1.8x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-10-08 10:00:00 | 445.86 | 446.71 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 11:05:00 | 461.50 | 465.67 | 0.00 | ORB-short ORB[462.55,468.90] vol=3.1x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 13:55:00 | 458.83 | 463.78 | 0.00 | T1 1.5R @ 458.83 |
| Target hit | 2025-10-13 15:20:00 | 458.30 | 462.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2025-10-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-14 09:55:00 | 460.40 | 457.56 | 0.00 | ORB-long ORB[454.40,458.90] vol=3.3x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-10-14 10:20:00 | 458.48 | 457.79 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:50:00 | 473.55 | 472.21 | 0.00 | ORB-long ORB[467.20,472.95] vol=2.9x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-10-16 11:20:00 | 472.11 | 472.22 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 11:15:00 | 453.00 | 456.15 | 0.00 | ORB-short ORB[454.60,460.70] vol=2.1x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-10-23 12:00:00 | 454.33 | 455.70 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-10-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 11:10:00 | 454.70 | 457.04 | 0.00 | ORB-short ORB[457.70,461.55] vol=2.7x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:35:00 | 452.37 | 456.52 | 0.00 | T1 1.5R @ 452.37 |
| Target hit | 2025-10-24 15:20:00 | 448.05 | 451.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2025-10-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:40:00 | 462.25 | 458.99 | 0.00 | ORB-long ORB[453.85,459.50] vol=1.8x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 460.03 | 460.97 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-10-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:30:00 | 466.35 | 464.87 | 0.00 | ORB-long ORB[462.45,465.55] vol=1.8x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 09:40:00 | 468.90 | 466.12 | 0.00 | T1 1.5R @ 468.90 |
| Stop hit — per-position SL triggered | 2025-10-29 09:45:00 | 466.35 | 466.34 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-10-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:20:00 | 459.65 | 462.11 | 0.00 | ORB-short ORB[462.70,467.00] vol=2.7x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-10-30 10:25:00 | 461.26 | 462.07 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:15:00 | 471.10 | 475.90 | 0.00 | ORB-short ORB[474.00,479.95] vol=2.2x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-11-04 12:20:00 | 473.05 | 472.78 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-11-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:05:00 | 461.80 | 471.31 | 0.00 | ORB-short ORB[474.25,478.70] vol=2.1x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 463.60 | 468.90 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-11-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:30:00 | 478.15 | 473.67 | 0.00 | ORB-long ORB[470.00,474.05] vol=4.3x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:40:00 | 481.49 | 477.97 | 0.00 | T1 1.5R @ 481.49 |
| Target hit | 2025-11-10 11:45:00 | 484.50 | 488.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — SELL (started 2025-11-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 09:40:00 | 463.85 | 467.13 | 0.00 | ORB-short ORB[466.65,471.00] vol=1.8x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-11-12 10:55:00 | 465.62 | 464.98 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-11-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:30:00 | 457.40 | 455.77 | 0.00 | ORB-long ORB[450.35,457.00] vol=1.5x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:45:00 | 459.57 | 456.35 | 0.00 | T1 1.5R @ 459.57 |
| Stop hit — per-position SL triggered | 2025-11-19 10:55:00 | 457.40 | 456.47 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-11-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 10:30:00 | 422.80 | 425.12 | 0.00 | ORB-short ORB[425.55,428.90] vol=2.0x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:00:00 | 420.88 | 424.62 | 0.00 | T1 1.5R @ 420.88 |
| Target hit | 2025-11-27 14:15:00 | 421.60 | 421.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 63 — BUY (started 2025-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:45:00 | 430.65 | 425.89 | 0.00 | ORB-long ORB[419.20,424.00] vol=6.7x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-11-28 09:50:00 | 428.88 | 427.99 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:10:00 | 418.20 | 421.34 | 0.00 | ORB-short ORB[421.30,426.90] vol=3.0x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 415.61 | 420.03 | 0.00 | T1 1.5R @ 415.61 |
| Stop hit — per-position SL triggered | 2025-12-03 11:40:00 | 418.20 | 419.90 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:45:00 | 425.20 | 427.94 | 0.00 | ORB-short ORB[428.10,432.80] vol=1.7x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:10:00 | 423.06 | 427.43 | 0.00 | T1 1.5R @ 423.06 |
| Target hit | 2025-12-08 15:20:00 | 417.20 | 421.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2025-12-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:40:00 | 412.20 | 415.29 | 0.00 | ORB-short ORB[413.80,418.10] vol=1.5x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-12-09 10:00:00 | 413.80 | 414.50 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 11:00:00 | 429.80 | 432.15 | 0.00 | ORB-short ORB[430.05,435.20] vol=1.6x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 11:50:00 | 427.63 | 431.42 | 0.00 | T1 1.5R @ 427.63 |
| Stop hit — per-position SL triggered | 2025-12-10 12:10:00 | 429.80 | 431.02 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:40:00 | 440.00 | 434.98 | 0.00 | ORB-long ORB[430.50,434.60] vol=2.2x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-12-12 09:45:00 | 438.32 | 435.92 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-12-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 10:55:00 | 441.95 | 444.67 | 0.00 | ORB-short ORB[444.20,448.35] vol=2.2x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 11:05:00 | 440.22 | 443.76 | 0.00 | T1 1.5R @ 440.22 |
| Stop hit — per-position SL triggered | 2025-12-19 12:30:00 | 441.95 | 443.13 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-22 11:15:00 | 450.30 | 454.08 | 0.00 | ORB-short ORB[451.00,457.70] vol=2.4x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 11:40:00 | 448.40 | 453.57 | 0.00 | T1 1.5R @ 448.40 |
| Stop hit — per-position SL triggered | 2025-12-22 11:50:00 | 450.30 | 453.36 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-12-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:50:00 | 455.30 | 453.77 | 0.00 | ORB-long ORB[449.00,454.35] vol=3.3x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 10:00:00 | 457.47 | 454.96 | 0.00 | T1 1.5R @ 457.47 |
| Stop hit — per-position SL triggered | 2025-12-23 10:05:00 | 455.30 | 455.07 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 09:35:00 | 448.95 | 451.49 | 0.00 | ORB-short ORB[450.05,454.90] vol=1.6x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:40:00 | 446.13 | 449.39 | 0.00 | T1 1.5R @ 446.13 |
| Target hit | 2025-12-26 12:05:00 | 447.80 | 447.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 73 — BUY (started 2025-12-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 11:00:00 | 450.80 | 444.30 | 0.00 | ORB-long ORB[440.00,444.70] vol=5.5x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:05:00 | 453.61 | 445.84 | 0.00 | T1 1.5R @ 453.61 |
| Stop hit — per-position SL triggered | 2025-12-29 11:10:00 | 450.80 | 446.36 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 10:15:00 | 447.50 | 450.76 | 0.00 | ORB-short ORB[453.50,456.70] vol=2.1x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 12:05:00 | 445.01 | 448.71 | 0.00 | T1 1.5R @ 445.01 |
| Target hit | 2026-01-01 15:20:00 | 443.40 | 446.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2026-01-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:20:00 | 457.70 | 454.77 | 0.00 | ORB-long ORB[451.70,457.20] vol=2.1x ATR=1.44 |
| Stop hit — per-position SL triggered | 2026-01-05 10:30:00 | 456.26 | 454.99 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:15:00 | 447.15 | 449.55 | 0.00 | ORB-short ORB[450.10,455.50] vol=3.4x ATR=1.51 |
| Stop hit — per-position SL triggered | 2026-01-08 11:05:00 | 448.66 | 448.89 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 11:15:00 | 435.20 | 438.57 | 0.00 | ORB-short ORB[437.05,441.40] vol=7.3x ATR=1.27 |
| Stop hit — per-position SL triggered | 2026-01-13 11:30:00 | 436.47 | 436.59 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-01-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 11:05:00 | 430.65 | 431.99 | 0.00 | ORB-short ORB[432.00,435.45] vol=2.7x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 11:10:00 | 429.16 | 431.65 | 0.00 | T1 1.5R @ 429.16 |
| Stop hit — per-position SL triggered | 2026-01-16 13:50:00 | 430.65 | 429.48 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-01-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 09:35:00 | 425.30 | 422.50 | 0.00 | ORB-long ORB[420.00,423.00] vol=2.0x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 10:20:00 | 428.04 | 424.38 | 0.00 | T1 1.5R @ 428.04 |
| Target hit | 2026-01-19 15:20:00 | 428.80 | 427.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — SELL (started 2026-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:40:00 | 421.70 | 423.02 | 0.00 | ORB-short ORB[422.40,428.10] vol=3.4x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-01-20 09:55:00 | 423.04 | 422.87 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-01-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:55:00 | 412.05 | 416.51 | 0.00 | ORB-short ORB[415.50,420.05] vol=2.9x ATR=1.94 |
| Stop hit — per-position SL triggered | 2026-01-21 11:00:00 | 413.99 | 416.42 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-01-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 11:05:00 | 414.60 | 417.08 | 0.00 | ORB-short ORB[416.60,421.30] vol=6.2x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:30:00 | 412.91 | 416.60 | 0.00 | T1 1.5R @ 412.91 |
| Stop hit — per-position SL triggered | 2026-01-23 12:20:00 | 414.60 | 415.88 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-01-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 10:10:00 | 408.85 | 406.37 | 0.00 | ORB-long ORB[401.80,407.90] vol=2.6x ATR=2.09 |
| Stop hit — per-position SL triggered | 2026-01-27 10:40:00 | 406.76 | 406.91 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:10:00 | 411.00 | 408.02 | 0.00 | ORB-long ORB[404.60,410.05] vol=8.7x ATR=0.97 |
| Stop hit — per-position SL triggered | 2026-02-01 11:25:00 | 410.03 | 408.29 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-02-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:45:00 | 422.75 | 419.47 | 0.00 | ORB-long ORB[414.40,420.00] vol=2.3x ATR=1.73 |
| Stop hit — per-position SL triggered | 2026-02-17 09:50:00 | 421.02 | 419.94 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-03-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 10:10:00 | 416.25 | 412.05 | 0.00 | ORB-long ORB[408.85,415.00] vol=3.1x ATR=2.23 |
| Stop hit — per-position SL triggered | 2026-03-04 10:40:00 | 414.02 | 412.71 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:15:00 | 420.30 | 417.78 | 0.00 | ORB-long ORB[414.35,420.00] vol=2.2x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:35:00 | 422.34 | 418.74 | 0.00 | T1 1.5R @ 422.34 |
| Stop hit — per-position SL triggered | 2026-03-05 11:00:00 | 420.30 | 419.58 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 417.35 | 422.56 | 0.00 | ORB-short ORB[420.50,425.45] vol=2.1x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 418.72 | 421.99 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 09:40:00 | 416.90 | 420.04 | 0.00 | ORB-short ORB[420.30,425.75] vol=7.1x ATR=2.29 |
| Stop hit — per-position SL triggered | 2026-03-11 09:50:00 | 419.19 | 419.81 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-03-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:50:00 | 405.10 | 403.97 | 0.00 | ORB-long ORB[399.50,404.90] vol=1.7x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:30:00 | 407.30 | 404.60 | 0.00 | T1 1.5R @ 407.30 |
| Target hit | 2026-03-18 15:20:00 | 412.10 | 409.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 91 — SELL (started 2026-03-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:00:00 | 400.15 | 403.66 | 0.00 | ORB-short ORB[402.80,407.20] vol=1.6x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-03-19 10:50:00 | 401.76 | 402.87 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2026-04-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 10:20:00 | 408.05 | 405.28 | 0.00 | ORB-long ORB[402.45,407.05] vol=3.0x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-06 10:30:00 | 411.01 | 405.99 | 0.00 | T1 1.5R @ 411.01 |
| Stop hit — per-position SL triggered | 2026-04-06 12:40:00 | 408.05 | 408.31 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2026-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 10:55:00 | 445.45 | 442.81 | 0.00 | ORB-long ORB[439.00,443.75] vol=6.0x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:00:00 | 447.50 | 443.69 | 0.00 | T1 1.5R @ 447.50 |
| Stop hit — per-position SL triggered | 2026-04-24 11:50:00 | 445.45 | 447.44 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 463.40 | 461.51 | 0.00 | ORB-long ORB[455.60,461.70] vol=4.1x ATR=2.53 |
| Stop hit — per-position SL triggered | 2026-04-29 14:20:00 | 460.87 | 463.09 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 462.70 | 461.39 | 0.00 | ORB-long ORB[456.45,462.10] vol=2.4x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:50:00 | 465.05 | 462.86 | 0.00 | T1 1.5R @ 465.05 |
| Target hit | 2026-05-06 15:20:00 | 470.35 | 467.95 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 09:35:00 | 313.30 | 2025-05-15 09:45:00 | 312.25 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-05-16 09:35:00 | 310.10 | 2025-05-16 11:50:00 | 308.52 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-05-16 09:35:00 | 310.10 | 2025-05-16 14:20:00 | 310.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-19 09:45:00 | 313.05 | 2025-05-19 10:15:00 | 314.97 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-05-19 09:45:00 | 313.05 | 2025-05-19 13:30:00 | 329.55 | TARGET_HIT | 0.50 | 5.27% |
| SELL | retest1 | 2025-05-22 10:40:00 | 319.00 | 2025-05-22 11:30:00 | 317.70 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-05-22 10:40:00 | 319.00 | 2025-05-22 12:10:00 | 319.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-23 10:20:00 | 328.60 | 2025-05-23 13:25:00 | 327.32 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-05-28 10:55:00 | 316.60 | 2025-05-28 11:15:00 | 317.29 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-05-30 10:00:00 | 308.55 | 2025-05-30 10:15:00 | 309.42 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-02 11:00:00 | 312.90 | 2025-06-02 11:20:00 | 312.21 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-06-04 09:45:00 | 309.65 | 2025-06-04 10:00:00 | 310.71 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-06-05 10:00:00 | 307.80 | 2025-06-05 10:20:00 | 308.58 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-09 09:30:00 | 310.00 | 2025-06-09 09:55:00 | 311.63 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-06-09 09:30:00 | 310.00 | 2025-06-09 11:40:00 | 310.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-12 11:10:00 | 311.00 | 2025-06-12 11:15:00 | 311.82 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-13 10:45:00 | 309.50 | 2025-06-13 12:30:00 | 308.44 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-18 10:40:00 | 312.00 | 2025-06-18 10:45:00 | 310.97 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-20 09:40:00 | 313.20 | 2025-06-20 09:50:00 | 314.82 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-06-20 09:40:00 | 313.20 | 2025-06-20 09:55:00 | 313.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-24 09:45:00 | 318.50 | 2025-06-24 09:50:00 | 320.21 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-06-24 09:45:00 | 318.50 | 2025-06-24 09:55:00 | 318.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 09:45:00 | 351.10 | 2025-06-27 09:50:00 | 353.77 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2025-06-27 09:45:00 | 351.10 | 2025-06-27 09:55:00 | 351.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-02 10:00:00 | 359.30 | 2025-07-02 11:15:00 | 360.50 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-07-03 09:45:00 | 368.75 | 2025-07-03 09:50:00 | 371.33 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-07-03 09:45:00 | 368.75 | 2025-07-03 13:15:00 | 371.55 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2025-07-07 10:45:00 | 369.05 | 2025-07-07 10:50:00 | 367.47 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-07-09 09:35:00 | 375.75 | 2025-07-09 09:40:00 | 374.67 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-14 09:55:00 | 374.15 | 2025-07-14 10:15:00 | 376.61 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-07-14 09:55:00 | 374.15 | 2025-07-14 15:20:00 | 389.35 | TARGET_HIT | 0.50 | 4.06% |
| SELL | retest1 | 2025-07-16 10:05:00 | 386.55 | 2025-07-16 10:55:00 | 387.91 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-07-17 09:30:00 | 387.15 | 2025-07-17 09:35:00 | 385.51 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-07-18 10:15:00 | 380.20 | 2025-07-18 11:00:00 | 378.51 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-07-18 10:15:00 | 380.20 | 2025-07-18 11:05:00 | 380.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-21 11:10:00 | 385.65 | 2025-07-21 11:40:00 | 384.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-22 10:00:00 | 381.55 | 2025-07-22 10:35:00 | 382.78 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-07-23 11:00:00 | 378.70 | 2025-07-23 11:05:00 | 377.58 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-23 11:00:00 | 378.70 | 2025-07-23 14:35:00 | 378.40 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2025-07-24 11:00:00 | 378.10 | 2025-07-24 11:20:00 | 379.05 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-30 10:40:00 | 385.60 | 2025-07-30 10:45:00 | 388.00 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-07-30 10:40:00 | 385.60 | 2025-07-30 10:50:00 | 385.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-31 09:35:00 | 378.60 | 2025-07-31 09:50:00 | 381.29 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-07-31 09:35:00 | 378.60 | 2025-07-31 09:55:00 | 378.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-01 11:10:00 | 372.20 | 2025-08-01 11:35:00 | 370.86 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-01 11:10:00 | 372.20 | 2025-08-01 13:10:00 | 371.85 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2025-08-11 09:45:00 | 335.95 | 2025-08-11 10:10:00 | 334.00 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-08-11 09:45:00 | 335.95 | 2025-08-11 10:15:00 | 335.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-12 11:15:00 | 346.05 | 2025-08-12 12:35:00 | 348.05 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-08-12 11:15:00 | 346.05 | 2025-08-12 14:00:00 | 348.85 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2025-08-20 10:35:00 | 372.70 | 2025-08-20 10:40:00 | 371.79 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-21 10:10:00 | 368.15 | 2025-08-21 11:20:00 | 367.02 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-08-22 10:00:00 | 375.65 | 2025-08-22 10:05:00 | 374.43 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-26 09:45:00 | 390.60 | 2025-08-26 15:05:00 | 388.26 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2025-09-01 09:55:00 | 392.80 | 2025-09-01 10:00:00 | 390.77 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2025-09-03 10:10:00 | 385.35 | 2025-09-03 10:15:00 | 383.97 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-05 09:35:00 | 383.25 | 2025-09-05 09:40:00 | 384.76 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-09-05 09:35:00 | 383.25 | 2025-09-05 10:40:00 | 384.90 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2025-09-10 11:00:00 | 388.75 | 2025-09-10 11:15:00 | 386.65 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-09-10 11:00:00 | 388.75 | 2025-09-10 15:20:00 | 384.10 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2025-09-12 10:45:00 | 385.80 | 2025-09-12 10:50:00 | 387.20 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-09-12 10:45:00 | 385.80 | 2025-09-12 10:55:00 | 385.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-17 11:00:00 | 406.20 | 2025-09-17 11:10:00 | 407.35 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-18 10:05:00 | 406.90 | 2025-09-18 10:10:00 | 404.85 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-09-19 10:20:00 | 420.45 | 2025-09-19 10:25:00 | 423.28 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-09-19 10:20:00 | 420.45 | 2025-09-19 10:30:00 | 420.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-26 09:45:00 | 451.90 | 2025-09-26 10:20:00 | 449.39 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2025-10-08 09:55:00 | 444.25 | 2025-10-08 10:00:00 | 445.86 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-13 11:05:00 | 461.50 | 2025-10-13 13:55:00 | 458.83 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-10-13 11:05:00 | 461.50 | 2025-10-13 15:20:00 | 458.30 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2025-10-14 09:55:00 | 460.40 | 2025-10-14 10:20:00 | 458.48 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-10-16 10:50:00 | 473.55 | 2025-10-16 11:20:00 | 472.11 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-23 11:15:00 | 453.00 | 2025-10-23 12:00:00 | 454.33 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-24 11:10:00 | 454.70 | 2025-10-24 11:35:00 | 452.37 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-10-24 11:10:00 | 454.70 | 2025-10-24 15:20:00 | 448.05 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2025-10-28 09:40:00 | 462.25 | 2025-10-28 10:15:00 | 460.03 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-10-29 09:30:00 | 466.35 | 2025-10-29 09:40:00 | 468.90 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-10-29 09:30:00 | 466.35 | 2025-10-29 09:45:00 | 466.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-30 10:20:00 | 459.65 | 2025-10-30 10:25:00 | 461.26 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-11-04 11:15:00 | 471.10 | 2025-11-04 12:20:00 | 473.05 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-11-06 10:05:00 | 461.80 | 2025-11-06 10:15:00 | 463.60 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-11-10 09:30:00 | 478.15 | 2025-11-10 09:40:00 | 481.49 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-11-10 09:30:00 | 478.15 | 2025-11-10 11:45:00 | 484.50 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2025-11-12 09:40:00 | 463.85 | 2025-11-12 10:55:00 | 465.62 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-11-19 10:30:00 | 457.40 | 2025-11-19 10:45:00 | 459.57 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-11-19 10:30:00 | 457.40 | 2025-11-19 10:55:00 | 457.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-27 10:30:00 | 422.80 | 2025-11-27 11:00:00 | 420.88 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-11-27 10:30:00 | 422.80 | 2025-11-27 14:15:00 | 421.60 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2025-11-28 09:45:00 | 430.65 | 2025-11-28 09:50:00 | 428.88 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-12-03 10:10:00 | 418.20 | 2025-12-03 11:15:00 | 415.61 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-12-03 10:10:00 | 418.20 | 2025-12-03 11:40:00 | 418.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 10:45:00 | 425.20 | 2025-12-08 11:10:00 | 423.06 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-12-08 10:45:00 | 425.20 | 2025-12-08 15:20:00 | 417.20 | TARGET_HIT | 0.50 | 1.88% |
| SELL | retest1 | 2025-12-09 09:40:00 | 412.20 | 2025-12-09 10:00:00 | 413.80 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-12-10 11:00:00 | 429.80 | 2025-12-10 11:50:00 | 427.63 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-12-10 11:00:00 | 429.80 | 2025-12-10 12:10:00 | 429.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-12 09:40:00 | 440.00 | 2025-12-12 09:45:00 | 438.32 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-12-19 10:55:00 | 441.95 | 2025-12-19 11:05:00 | 440.22 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-19 10:55:00 | 441.95 | 2025-12-19 12:30:00 | 441.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-22 11:15:00 | 450.30 | 2025-12-22 11:40:00 | 448.40 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-12-22 11:15:00 | 450.30 | 2025-12-22 11:50:00 | 450.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 09:50:00 | 455.30 | 2025-12-23 10:00:00 | 457.47 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-12-23 09:50:00 | 455.30 | 2025-12-23 10:05:00 | 455.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-26 09:35:00 | 448.95 | 2025-12-26 09:40:00 | 446.13 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-12-26 09:35:00 | 448.95 | 2025-12-26 12:05:00 | 447.80 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-12-29 11:00:00 | 450.80 | 2025-12-29 11:05:00 | 453.61 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-12-29 11:00:00 | 450.80 | 2025-12-29 11:10:00 | 450.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-01 10:15:00 | 447.50 | 2026-01-01 12:05:00 | 445.01 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-01-01 10:15:00 | 447.50 | 2026-01-01 15:20:00 | 443.40 | TARGET_HIT | 0.50 | 0.92% |
| BUY | retest1 | 2026-01-05 10:20:00 | 457.70 | 2026-01-05 10:30:00 | 456.26 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-08 10:15:00 | 447.15 | 2026-01-08 11:05:00 | 448.66 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-01-13 11:15:00 | 435.20 | 2026-01-13 11:30:00 | 436.47 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-01-16 11:05:00 | 430.65 | 2026-01-16 11:10:00 | 429.16 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-01-16 11:05:00 | 430.65 | 2026-01-16 13:50:00 | 430.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-19 09:35:00 | 425.30 | 2026-01-19 10:20:00 | 428.04 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-01-19 09:35:00 | 425.30 | 2026-01-19 15:20:00 | 428.80 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2026-01-20 09:40:00 | 421.70 | 2026-01-20 09:55:00 | 423.04 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-21 10:55:00 | 412.05 | 2026-01-21 11:00:00 | 413.99 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-01-23 11:05:00 | 414.60 | 2026-01-23 11:30:00 | 412.91 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-01-23 11:05:00 | 414.60 | 2026-01-23 12:20:00 | 414.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-27 10:10:00 | 408.85 | 2026-01-27 10:40:00 | 406.76 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-02-01 11:10:00 | 411.00 | 2026-02-01 11:25:00 | 410.03 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-17 09:45:00 | 422.75 | 2026-02-17 09:50:00 | 421.02 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-04 10:10:00 | 416.25 | 2026-03-04 10:40:00 | 414.02 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-03-05 10:15:00 | 420.30 | 2026-03-05 10:35:00 | 422.34 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-03-05 10:15:00 | 420.30 | 2026-03-05 11:00:00 | 420.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 417.35 | 2026-03-06 10:50:00 | 418.72 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-11 09:40:00 | 416.90 | 2026-03-11 09:50:00 | 419.19 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2026-03-18 09:50:00 | 405.10 | 2026-03-18 10:30:00 | 407.30 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-03-18 09:50:00 | 405.10 | 2026-03-18 15:20:00 | 412.10 | TARGET_HIT | 0.50 | 1.73% |
| SELL | retest1 | 2026-03-19 10:00:00 | 400.15 | 2026-03-19 10:50:00 | 401.76 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-06 10:20:00 | 408.05 | 2026-04-06 10:30:00 | 411.01 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-04-06 10:20:00 | 408.05 | 2026-04-06 12:40:00 | 408.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-24 10:55:00 | 445.45 | 2026-04-24 11:00:00 | 447.50 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-24 10:55:00 | 445.45 | 2026-04-24 11:50:00 | 445.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:40:00 | 463.40 | 2026-04-29 14:20:00 | 460.87 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2026-05-06 11:00:00 | 462.70 | 2026-05-06 11:50:00 | 465.05 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-05-06 11:00:00 | 462.70 | 2026-05-06 15:20:00 | 470.35 | TARGET_HIT | 0.50 | 1.65% |
