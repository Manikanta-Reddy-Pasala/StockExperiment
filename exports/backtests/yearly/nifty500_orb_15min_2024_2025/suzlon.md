# Suzlon Energy Ltd. (SUZLON)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-01-03 15:25:00 (12108 bars)
- **Last close:** 62.05
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 17
- **Target hits / Stop hits / Partials:** 2 / 17 / 7
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 2.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 6 | 35.3% | 1 | 11 | 5 | 0.12% | 2.1% |
| BUY @ 2nd Alert (retest1) | 17 | 6 | 35.3% | 1 | 11 | 5 | 0.12% | 2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.08% | 0.7% |
| SELL @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.08% | 0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 9 | 34.6% | 2 | 17 | 7 | 0.11% | 2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:55:00 | 50.29 | 49.95 | 0.00 | ORB-long ORB[49.50,50.20] vol=2.2x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 10:05:00 | 50.58 | 50.48 | 0.00 | T1 1.5R @ 50.58 |
| Stop hit — per-position SL triggered | 2024-06-14 10:25:00 | 50.29 | 50.52 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-06-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:35:00 | 52.50 | 51.67 | 0.00 | ORB-long ORB[51.00,51.50] vol=5.4x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 09:45:00 | 52.90 | 52.10 | 0.00 | T1 1.5R @ 52.90 |
| Stop hit — per-position SL triggered | 2024-06-21 09:55:00 | 52.50 | 52.16 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 52.83 | 53.14 | 0.00 | ORB-short ORB[53.00,53.38] vol=1.6x ATR=0.17 |
| Stop hit — per-position SL triggered | 2024-07-02 10:45:00 | 53.00 | 52.96 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-07-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 11:10:00 | 53.61 | 53.31 | 0.00 | ORB-long ORB[53.10,53.45] vol=6.8x ATR=0.16 |
| Stop hit — per-position SL triggered | 2024-07-03 11:15:00 | 53.45 | 53.31 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 54.68 | 55.67 | 0.00 | ORB-short ORB[55.79,56.00] vol=1.7x ATR=0.24 |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 54.92 | 55.66 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:15:00 | 54.48 | 55.14 | 0.00 | ORB-short ORB[54.81,55.45] vol=2.3x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 10:35:00 | 54.05 | 54.79 | 0.00 | T1 1.5R @ 54.05 |
| Stop hit — per-position SL triggered | 2024-07-09 10:50:00 | 54.48 | 54.73 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 10:55:00 | 54.19 | 54.39 | 0.00 | ORB-short ORB[54.41,55.00] vol=1.6x ATR=0.17 |
| Stop hit — per-position SL triggered | 2024-07-15 11:45:00 | 54.36 | 54.37 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 11:10:00 | 55.24 | 54.90 | 0.00 | ORB-long ORB[54.72,55.00] vol=3.2x ATR=0.17 |
| Stop hit — per-position SL triggered | 2024-07-16 11:35:00 | 55.07 | 54.93 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-19 09:30:00 | 55.60 | 55.19 | 0.00 | ORB-long ORB[54.55,55.38] vol=2.0x ATR=0.25 |
| Stop hit — per-position SL triggered | 2024-07-19 09:35:00 | 55.35 | 55.22 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:35:00 | 63.72 | 63.07 | 0.00 | ORB-long ORB[62.38,62.95] vol=7.1x ATR=0.31 |
| Stop hit — per-position SL triggered | 2024-07-29 10:25:00 | 63.41 | 63.38 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-08-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:40:00 | 78.71 | 78.21 | 0.00 | ORB-long ORB[77.66,78.59] vol=1.9x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 09:50:00 | 79.18 | 78.39 | 0.00 | T1 1.5R @ 79.18 |
| Stop hit — per-position SL triggered | 2024-08-28 10:05:00 | 78.71 | 78.46 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-10-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:30:00 | 81.23 | 80.62 | 0.00 | ORB-long ORB[79.90,80.75] vol=1.9x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-10-01 09:55:00 | 80.89 | 80.86 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-11-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:40:00 | 58.70 | 59.17 | 0.00 | ORB-short ORB[58.85,59.56] vol=1.8x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-11-18 09:45:00 | 58.99 | 59.17 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-12-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:45:00 | 66.23 | 65.91 | 0.00 | ORB-long ORB[65.34,66.13] vol=3.5x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 10:50:00 | 66.68 | 66.04 | 0.00 | T1 1.5R @ 66.68 |
| Target hit | 2024-12-04 14:05:00 | 67.18 | 67.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2024-12-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:50:00 | 67.00 | 66.46 | 0.00 | ORB-long ORB[66.05,66.75] vol=1.5x ATR=0.26 |
| Stop hit — per-position SL triggered | 2024-12-11 10:55:00 | 66.74 | 66.49 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 67.24 | 66.84 | 0.00 | ORB-long ORB[66.21,67.10] vol=1.9x ATR=0.20 |
| Stop hit — per-position SL triggered | 2024-12-17 09:35:00 | 67.04 | 67.31 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-12-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:55:00 | 64.93 | 64.36 | 0.00 | ORB-long ORB[63.77,64.43] vol=1.6x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 11:30:00 | 65.37 | 64.71 | 0.00 | T1 1.5R @ 65.37 |
| Stop hit — per-position SL triggered | 2024-12-24 12:10:00 | 64.93 | 64.75 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-12-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:30:00 | 64.03 | 64.20 | 0.00 | ORB-short ORB[64.05,64.73] vol=1.8x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:20:00 | 63.78 | 64.13 | 0.00 | T1 1.5R @ 63.78 |
| Target hit | 2024-12-27 15:20:00 | 63.15 | 63.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2025-01-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:55:00 | 62.10 | 62.75 | 0.00 | ORB-short ORB[62.83,63.25] vol=1.7x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-01-03 11:05:00 | 62.28 | 62.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-14 09:55:00 | 50.29 | 2024-06-14 10:05:00 | 50.58 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-06-14 09:55:00 | 50.29 | 2024-06-14 10:25:00 | 50.29 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-21 09:35:00 | 52.50 | 2024-06-21 09:45:00 | 52.90 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-06-21 09:35:00 | 52.50 | 2024-06-21 09:55:00 | 52.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 09:30:00 | 52.83 | 2024-07-02 10:45:00 | 53.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-03 11:10:00 | 53.61 | 2024-07-03 11:15:00 | 53.45 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-08 11:10:00 | 54.68 | 2024-07-08 11:15:00 | 54.92 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-07-09 10:15:00 | 54.48 | 2024-07-09 10:35:00 | 54.05 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2024-07-09 10:15:00 | 54.48 | 2024-07-09 10:50:00 | 54.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-15 10:55:00 | 54.19 | 2024-07-15 11:45:00 | 54.36 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-16 11:10:00 | 55.24 | 2024-07-16 11:35:00 | 55.07 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-19 09:30:00 | 55.60 | 2024-07-19 09:35:00 | 55.35 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-07-29 09:35:00 | 63.72 | 2024-07-29 10:25:00 | 63.41 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-08-28 09:40:00 | 78.71 | 2024-08-28 09:50:00 | 79.18 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-08-28 09:40:00 | 78.71 | 2024-08-28 10:05:00 | 78.71 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-01 09:30:00 | 81.23 | 2024-10-01 09:55:00 | 80.89 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-11-18 09:40:00 | 58.70 | 2024-11-18 09:45:00 | 58.99 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-12-04 10:45:00 | 66.23 | 2024-12-04 10:50:00 | 66.68 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-12-04 10:45:00 | 66.23 | 2024-12-04 14:05:00 | 67.18 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2024-12-11 10:50:00 | 67.00 | 2024-12-11 10:55:00 | 66.74 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-12-17 09:30:00 | 67.24 | 2024-12-17 09:35:00 | 67.04 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-24 09:55:00 | 64.93 | 2024-12-24 11:30:00 | 65.37 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-12-24 09:55:00 | 64.93 | 2024-12-24 12:10:00 | 64.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-27 10:30:00 | 64.03 | 2024-12-27 11:20:00 | 63.78 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-12-27 10:30:00 | 64.03 | 2024-12-27 15:20:00 | 63.15 | TARGET_HIT | 0.50 | 1.37% |
| SELL | retest1 | 2025-01-03 10:55:00 | 62.10 | 2025-01-03 11:05:00 | 62.28 | STOP_HIT | 1.00 | -0.29% |
