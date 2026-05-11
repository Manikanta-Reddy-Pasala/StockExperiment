# MMTC Ltd. (MMTC)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (15538 bars)
- **Last close:** 68.15
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
| ENTRY1 | 53 |
| ENTRY2 | 0 |
| PARTIAL | 28 |
| TARGET_HIT | 14 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 39
- **Target hits / Stop hits / Partials:** 14 / 39 / 28
- **Avg / median % per leg:** 0.48% / 0.32%
- **Sum % (uncompounded):** 38.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 20 | 51.3% | 7 | 19 | 13 | 0.64% | 24.9% |
| BUY @ 2nd Alert (retest1) | 39 | 20 | 51.3% | 7 | 19 | 13 | 0.64% | 24.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 22 | 52.4% | 7 | 20 | 15 | 0.33% | 13.9% |
| SELL @ 2nd Alert (retest1) | 42 | 22 | 52.4% | 7 | 20 | 15 | 0.33% | 13.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 81 | 42 | 51.9% | 14 | 39 | 28 | 0.48% | 38.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:15:00 | 59.90 | 59.47 | 0.00 | ORB-long ORB[59.00,59.87] vol=1.9x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-05-15 10:25:00 | 59.56 | 59.51 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:05:00 | 61.00 | 60.62 | 0.00 | ORB-long ORB[60.05,60.73] vol=3.9x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 10:20:00 | 61.46 | 60.72 | 0.00 | T1 1.5R @ 61.46 |
| Stop hit — per-position SL triggered | 2025-05-16 10:25:00 | 61.00 | 60.74 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 10:30:00 | 63.06 | 62.14 | 0.00 | ORB-long ORB[61.99,62.40] vol=3.5x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-05-19 10:35:00 | 62.67 | 62.26 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:45:00 | 61.64 | 61.27 | 0.00 | ORB-long ORB[60.69,61.49] vol=5.2x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 10:50:00 | 62.00 | 61.40 | 0.00 | T1 1.5R @ 62.00 |
| Stop hit — per-position SL triggered | 2025-05-23 10:55:00 | 61.64 | 61.51 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:45:00 | 65.66 | 64.73 | 0.00 | ORB-long ORB[63.92,64.63] vol=8.2x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 10:50:00 | 66.10 | 66.76 | 0.00 | T1 1.5R @ 66.10 |
| Target hit | 2025-05-28 14:50:00 | 70.00 | 70.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2025-06-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 09:30:00 | 67.02 | 66.67 | 0.00 | ORB-long ORB[66.10,67.00] vol=3.4x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 11:20:00 | 67.70 | 67.04 | 0.00 | T1 1.5R @ 67.70 |
| Target hit | 2025-06-23 15:20:00 | 69.96 | 69.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2025-06-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 10:55:00 | 70.80 | 71.11 | 0.00 | ORB-short ORB[71.01,71.87] vol=1.7x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-06-27 11:20:00 | 71.02 | 71.09 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:30:00 | 71.30 | 71.81 | 0.00 | ORB-short ORB[71.71,72.50] vol=1.6x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-07-02 09:35:00 | 71.62 | 71.75 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 09:35:00 | 70.38 | 70.82 | 0.00 | ORB-short ORB[70.60,71.60] vol=1.7x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 09:40:00 | 70.01 | 70.44 | 0.00 | T1 1.5R @ 70.01 |
| Stop hit — per-position SL triggered | 2025-07-03 09:45:00 | 70.38 | 70.44 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 11:15:00 | 70.50 | 70.84 | 0.00 | ORB-short ORB[70.56,71.35] vol=1.9x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 11:45:00 | 70.14 | 70.81 | 0.00 | T1 1.5R @ 70.14 |
| Target hit | 2025-07-04 15:20:00 | 70.05 | 70.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 69.55 | 70.04 | 0.00 | ORB-short ORB[69.98,70.94] vol=1.5x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-07-08 11:40:00 | 69.73 | 69.95 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:15:00 | 68.87 | 69.54 | 0.00 | ORB-short ORB[69.62,70.35] vol=3.5x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-07-10 10:25:00 | 69.12 | 69.45 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:50:00 | 68.18 | 68.85 | 0.00 | ORB-short ORB[68.60,69.59] vol=1.7x ATR=0.23 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 68.41 | 68.76 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 67.80 | 68.34 | 0.00 | ORB-short ORB[68.10,68.87] vol=1.6x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-07-18 13:45:00 | 68.05 | 68.05 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:50:00 | 68.80 | 68.15 | 0.00 | ORB-long ORB[67.59,68.39] vol=3.5x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 10:00:00 | 69.32 | 68.45 | 0.00 | T1 1.5R @ 69.32 |
| Target hit | 2025-07-21 11:45:00 | 71.10 | 71.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2025-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:40:00 | 67.75 | 68.01 | 0.00 | ORB-short ORB[67.80,68.50] vol=2.0x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:35:00 | 67.44 | 67.78 | 0.00 | T1 1.5R @ 67.44 |
| Target hit | 2025-07-25 15:20:00 | 67.01 | 67.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-07-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:50:00 | 65.90 | 66.30 | 0.00 | ORB-short ORB[66.20,66.95] vol=2.4x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-07-30 11:35:00 | 66.18 | 66.07 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:05:00 | 65.00 | 65.38 | 0.00 | ORB-short ORB[65.50,66.30] vol=1.6x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 65.18 | 65.31 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 11:10:00 | 63.42 | 64.05 | 0.00 | ORB-short ORB[64.20,64.90] vol=1.6x ATR=0.24 |
| Target hit | 2025-08-08 15:20:00 | 63.20 | 63.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-09-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:30:00 | 65.95 | 65.02 | 0.00 | ORB-long ORB[64.36,65.18] vol=4.3x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-09-08 09:35:00 | 65.60 | 65.62 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:35:00 | 66.17 | 65.26 | 0.00 | ORB-long ORB[64.56,65.33] vol=4.3x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-09-10 09:40:00 | 65.82 | 65.41 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:50:00 | 65.80 | 65.41 | 0.00 | ORB-long ORB[65.00,65.69] vol=3.2x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-09-11 09:55:00 | 65.55 | 65.46 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 09:55:00 | 70.49 | 69.11 | 0.00 | ORB-long ORB[67.83,68.85] vol=7.8x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-09-25 10:00:00 | 69.98 | 69.32 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:40:00 | 65.79 | 65.33 | 0.00 | ORB-long ORB[64.80,65.75] vol=1.7x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:45:00 | 66.30 | 65.45 | 0.00 | T1 1.5R @ 66.30 |
| Stop hit — per-position SL triggered | 2025-09-29 10:25:00 | 65.79 | 65.64 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-10-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:40:00 | 72.59 | 71.89 | 0.00 | ORB-long ORB[70.89,71.86] vol=5.2x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-10-10 09:50:00 | 72.14 | 72.00 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-10-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:45:00 | 67.89 | 67.52 | 0.00 | ORB-long ORB[67.25,67.85] vol=2.5x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:50:00 | 68.19 | 67.64 | 0.00 | T1 1.5R @ 68.19 |
| Target hit | 2025-10-15 14:45:00 | 68.07 | 68.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — BUY (started 2025-10-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:35:00 | 68.94 | 68.61 | 0.00 | ORB-long ORB[68.04,68.74] vol=1.5x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-10-16 09:55:00 | 68.73 | 68.80 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:35:00 | 68.89 | 68.23 | 0.00 | ORB-long ORB[67.78,68.39] vol=1.7x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-10-17 09:40:00 | 68.64 | 68.46 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-10-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:40:00 | 67.04 | 67.51 | 0.00 | ORB-short ORB[67.54,68.47] vol=2.1x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-10-20 10:55:00 | 67.28 | 67.49 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 11:00:00 | 66.99 | 67.32 | 0.00 | ORB-short ORB[67.16,67.60] vol=1.7x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-10-24 11:55:00 | 67.15 | 67.28 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 11:10:00 | 67.19 | 67.24 | 0.00 | ORB-short ORB[67.20,67.72] vol=1.8x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 12:10:00 | 66.97 | 67.22 | 0.00 | T1 1.5R @ 66.97 |
| Stop hit — per-position SL triggered | 2025-10-28 13:05:00 | 67.19 | 67.17 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-11-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:20:00 | 66.30 | 66.87 | 0.00 | ORB-short ORB[66.90,67.73] vol=1.7x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 15:15:00 | 65.95 | 66.50 | 0.00 | T1 1.5R @ 65.95 |
| Target hit | 2025-11-06 15:20:00 | 65.85 | 66.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-12-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:25:00 | 55.75 | 55.95 | 0.00 | ORB-short ORB[55.92,56.70] vol=1.7x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:50:00 | 55.14 | 55.76 | 0.00 | T1 1.5R @ 55.14 |
| Target hit | 2025-12-08 15:20:00 | 54.34 | 55.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2025-12-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:40:00 | 53.38 | 53.64 | 0.00 | ORB-short ORB[53.58,54.10] vol=1.7x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 10:00:00 | 53.11 | 53.52 | 0.00 | T1 1.5R @ 53.11 |
| Stop hit — per-position SL triggered | 2025-12-18 10:05:00 | 53.38 | 53.51 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:35:00 | 54.92 | 54.67 | 0.00 | ORB-long ORB[54.14,54.90] vol=2.6x ATR=0.20 |
| Stop hit — per-position SL triggered | 2025-12-19 09:50:00 | 54.72 | 54.68 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:30:00 | 58.46 | 57.94 | 0.00 | ORB-long ORB[57.25,58.09] vol=3.3x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:50:00 | 58.83 | 59.99 | 0.00 | T1 1.5R @ 58.83 |
| Target hit | 2025-12-26 10:10:00 | 61.70 | 61.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2026-01-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 10:45:00 | 66.02 | 66.11 | 0.00 | ORB-short ORB[66.33,67.01] vol=1.7x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 13:10:00 | 65.66 | 66.03 | 0.00 | T1 1.5R @ 65.66 |
| Stop hit — per-position SL triggered | 2026-01-01 14:40:00 | 66.02 | 66.00 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2026-01-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:55:00 | 66.67 | 67.10 | 0.00 | ORB-short ORB[66.80,67.70] vol=1.8x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:20:00 | 66.32 | 67.04 | 0.00 | T1 1.5R @ 66.32 |
| Stop hit — per-position SL triggered | 2026-01-06 13:10:00 | 66.67 | 66.99 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2026-01-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 09:50:00 | 68.46 | 67.39 | 0.00 | ORB-long ORB[65.90,66.84] vol=8.0x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 09:55:00 | 69.19 | 68.47 | 0.00 | T1 1.5R @ 69.19 |
| Target hit | 2026-01-07 10:15:00 | 68.70 | 68.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — BUY (started 2026-01-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 10:50:00 | 68.55 | 66.82 | 0.00 | ORB-long ORB[64.70,65.70] vol=6.5x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 10:55:00 | 69.29 | 67.67 | 0.00 | T1 1.5R @ 69.29 |
| Stop hit — per-position SL triggered | 2026-01-28 11:05:00 | 68.55 | 67.87 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-02-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:45:00 | 64.08 | 64.54 | 0.00 | ORB-short ORB[64.27,65.10] vol=1.6x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 10:40:00 | 63.69 | 64.38 | 0.00 | T1 1.5R @ 63.69 |
| Target hit | 2026-02-05 15:20:00 | 63.25 | 63.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:15:00 | 65.34 | 64.53 | 0.00 | ORB-long ORB[64.07,64.86] vol=2.1x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:20:00 | 65.74 | 64.71 | 0.00 | T1 1.5R @ 65.74 |
| Stop hit — per-position SL triggered | 2026-02-09 11:25:00 | 65.34 | 64.77 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2026-02-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:05:00 | 66.84 | 66.06 | 0.00 | ORB-long ORB[65.64,66.44] vol=3.2x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-02-11 10:10:00 | 66.54 | 66.13 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 63.70 | 63.94 | 0.00 | ORB-short ORB[63.71,64.26] vol=1.6x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:00:00 | 63.41 | 63.84 | 0.00 | T1 1.5R @ 63.41 |
| Stop hit — per-position SL triggered | 2026-02-18 10:50:00 | 63.70 | 63.78 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:40:00 | 63.05 | 63.35 | 0.00 | ORB-short ORB[63.30,63.81] vol=2.1x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:45:00 | 62.80 | 63.16 | 0.00 | T1 1.5R @ 62.80 |
| Stop hit — per-position SL triggered | 2026-02-19 15:05:00 | 63.05 | 63.01 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 61.45 | 61.74 | 0.00 | ORB-short ORB[61.62,62.30] vol=1.7x ATR=0.20 |
| Stop hit — per-position SL triggered | 2026-02-25 09:45:00 | 61.65 | 61.69 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:15:00 | 61.48 | 61.89 | 0.00 | ORB-short ORB[61.86,62.64] vol=2.4x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:40:00 | 61.28 | 61.81 | 0.00 | T1 1.5R @ 61.28 |
| Stop hit — per-position SL triggered | 2026-02-26 15:05:00 | 61.48 | 61.59 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:50:00 | 61.65 | 61.19 | 0.00 | ORB-long ORB[60.80,61.35] vol=2.6x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:55:00 | 62.01 | 62.55 | 0.00 | T1 1.5R @ 62.01 |
| Target hit | 2026-02-27 10:00:00 | 62.55 | 62.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — SELL (started 2026-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:40:00 | 54.88 | 55.13 | 0.00 | ORB-short ORB[54.95,55.62] vol=2.2x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:50:00 | 54.51 | 55.01 | 0.00 | T1 1.5R @ 54.51 |
| Target hit | 2026-03-13 15:20:00 | 53.20 | 53.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2026-03-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-24 09:30:00 | 57.17 | 56.60 | 0.00 | ORB-long ORB[55.93,56.73] vol=3.3x ATR=0.41 |
| Stop hit — per-position SL triggered | 2026-03-24 09:35:00 | 56.76 | 56.53 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 60.15 | 59.88 | 0.00 | ORB-long ORB[59.07,59.80] vol=8.8x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-04-10 09:35:00 | 59.86 | 59.89 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 67.35 | 67.09 | 0.00 | ORB-long ORB[66.67,67.21] vol=3.6x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:50:00 | 67.78 | 67.25 | 0.00 | T1 1.5R @ 67.78 |
| Stop hit — per-position SL triggered | 2026-04-28 09:55:00 | 67.35 | 67.24 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 65.94 | 66.27 | 0.00 | ORB-short ORB[66.10,66.60] vol=1.8x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:05:00 | 65.59 | 66.18 | 0.00 | T1 1.5R @ 65.59 |
| Stop hit — per-position SL triggered | 2026-05-06 14:25:00 | 65.94 | 65.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 10:15:00 | 59.90 | 2025-05-15 10:25:00 | 59.56 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2025-05-16 10:05:00 | 61.00 | 2025-05-16 10:20:00 | 61.46 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2025-05-16 10:05:00 | 61.00 | 2025-05-16 10:25:00 | 61.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-19 10:30:00 | 63.06 | 2025-05-19 10:35:00 | 62.67 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2025-05-23 10:45:00 | 61.64 | 2025-05-23 10:50:00 | 62.00 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-05-23 10:45:00 | 61.64 | 2025-05-23 10:55:00 | 61.64 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-28 10:45:00 | 65.66 | 2025-05-28 10:50:00 | 66.10 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-05-28 10:45:00 | 65.66 | 2025-05-28 14:50:00 | 70.00 | TARGET_HIT | 0.50 | 6.61% |
| BUY | retest1 | 2025-06-23 09:30:00 | 67.02 | 2025-06-23 11:20:00 | 67.70 | PARTIAL | 0.50 | 1.01% |
| BUY | retest1 | 2025-06-23 09:30:00 | 67.02 | 2025-06-23 15:20:00 | 69.96 | TARGET_HIT | 0.50 | 4.39% |
| SELL | retest1 | 2025-06-27 10:55:00 | 70.80 | 2025-06-27 11:20:00 | 71.02 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-07-02 09:30:00 | 71.30 | 2025-07-02 09:35:00 | 71.62 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-07-03 09:35:00 | 70.38 | 2025-07-03 09:40:00 | 70.01 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-07-03 09:35:00 | 70.38 | 2025-07-03 09:45:00 | 70.38 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-04 11:15:00 | 70.50 | 2025-07-04 11:45:00 | 70.14 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-07-04 11:15:00 | 70.50 | 2025-07-04 15:20:00 | 70.05 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2025-07-08 11:05:00 | 69.55 | 2025-07-08 11:40:00 | 69.73 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-10 10:15:00 | 68.87 | 2025-07-10 10:25:00 | 69.12 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-07-11 10:50:00 | 68.18 | 2025-07-11 11:10:00 | 68.41 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-07-18 10:15:00 | 67.80 | 2025-07-18 13:45:00 | 68.05 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-07-21 09:50:00 | 68.80 | 2025-07-21 10:00:00 | 69.32 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2025-07-21 09:50:00 | 68.80 | 2025-07-21 11:45:00 | 71.10 | TARGET_HIT | 0.50 | 3.34% |
| SELL | retest1 | 2025-07-25 09:40:00 | 67.75 | 2025-07-25 10:35:00 | 67.44 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-07-25 09:40:00 | 67.75 | 2025-07-25 15:20:00 | 67.01 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2025-07-30 09:50:00 | 65.90 | 2025-07-30 11:35:00 | 66.18 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-08-06 11:05:00 | 65.00 | 2025-08-06 12:15:00 | 65.18 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-08 11:10:00 | 63.42 | 2025-08-08 15:20:00 | 63.20 | TARGET_HIT | 1.00 | 0.35% |
| BUY | retest1 | 2025-09-08 09:30:00 | 65.95 | 2025-09-08 09:35:00 | 65.60 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-09-10 09:35:00 | 66.17 | 2025-09-10 09:40:00 | 65.82 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-09-11 09:50:00 | 65.80 | 2025-09-11 09:55:00 | 65.55 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-09-25 09:55:00 | 70.49 | 2025-09-25 10:00:00 | 69.98 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2025-09-29 09:40:00 | 65.79 | 2025-09-29 09:45:00 | 66.30 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2025-09-29 09:40:00 | 65.79 | 2025-09-29 10:25:00 | 65.79 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 09:40:00 | 72.59 | 2025-10-10 09:50:00 | 72.14 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2025-10-15 10:45:00 | 67.89 | 2025-10-15 10:50:00 | 68.19 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-10-15 10:45:00 | 67.89 | 2025-10-15 14:45:00 | 68.07 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2025-10-16 09:35:00 | 68.94 | 2025-10-16 09:55:00 | 68.73 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-17 09:35:00 | 68.89 | 2025-10-17 09:40:00 | 68.64 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-20 10:40:00 | 67.04 | 2025-10-20 10:55:00 | 67.28 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-24 11:00:00 | 66.99 | 2025-10-24 11:55:00 | 67.15 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-28 11:10:00 | 67.19 | 2025-10-28 12:10:00 | 66.97 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-10-28 11:10:00 | 67.19 | 2025-10-28 13:05:00 | 67.19 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-06 10:20:00 | 66.30 | 2025-11-06 15:15:00 | 65.95 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-11-06 10:20:00 | 66.30 | 2025-11-06 15:20:00 | 65.85 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2025-12-08 10:25:00 | 55.75 | 2025-12-08 11:50:00 | 55.14 | PARTIAL | 0.50 | 1.09% |
| SELL | retest1 | 2025-12-08 10:25:00 | 55.75 | 2025-12-08 15:20:00 | 54.34 | TARGET_HIT | 0.50 | 2.53% |
| SELL | retest1 | 2025-12-18 09:40:00 | 53.38 | 2025-12-18 10:00:00 | 53.11 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-12-18 09:40:00 | 53.38 | 2025-12-18 10:05:00 | 53.38 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-19 09:35:00 | 54.92 | 2025-12-19 09:50:00 | 54.72 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-12-26 09:30:00 | 58.46 | 2025-12-26 09:50:00 | 58.83 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-12-26 09:30:00 | 58.46 | 2025-12-26 10:10:00 | 61.70 | TARGET_HIT | 0.50 | 5.54% |
| SELL | retest1 | 2026-01-01 10:45:00 | 66.02 | 2026-01-01 13:10:00 | 65.66 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-01-01 10:45:00 | 66.02 | 2026-01-01 14:40:00 | 66.02 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-06 10:55:00 | 66.67 | 2026-01-06 11:20:00 | 66.32 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-01-06 10:55:00 | 66.67 | 2026-01-06 13:10:00 | 66.67 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-07 09:50:00 | 68.46 | 2026-01-07 09:55:00 | 69.19 | PARTIAL | 0.50 | 1.06% |
| BUY | retest1 | 2026-01-07 09:50:00 | 68.46 | 2026-01-07 10:15:00 | 68.70 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2026-01-28 10:50:00 | 68.55 | 2026-01-28 10:55:00 | 69.29 | PARTIAL | 0.50 | 1.08% |
| BUY | retest1 | 2026-01-28 10:50:00 | 68.55 | 2026-01-28 11:05:00 | 68.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-05 09:45:00 | 64.08 | 2026-02-05 10:40:00 | 63.69 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-05 09:45:00 | 64.08 | 2026-02-05 15:20:00 | 63.25 | TARGET_HIT | 0.50 | 1.30% |
| BUY | retest1 | 2026-02-09 11:15:00 | 65.34 | 2026-02-09 11:20:00 | 65.74 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-09 11:15:00 | 65.34 | 2026-02-09 11:25:00 | 65.34 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 10:05:00 | 66.84 | 2026-02-11 10:10:00 | 66.54 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-02-18 09:50:00 | 63.70 | 2026-02-18 10:00:00 | 63.41 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-18 09:50:00 | 63.70 | 2026-02-18 10:50:00 | 63.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 09:40:00 | 63.05 | 2026-02-19 11:45:00 | 62.80 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-19 09:40:00 | 63.05 | 2026-02-19 15:05:00 | 63.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 09:30:00 | 61.45 | 2026-02-25 09:45:00 | 61.65 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-26 11:15:00 | 61.48 | 2026-02-26 11:40:00 | 61.28 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-26 11:15:00 | 61.48 | 2026-02-26 15:05:00 | 61.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-27 09:50:00 | 61.65 | 2026-02-27 09:55:00 | 62.01 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-27 09:50:00 | 61.65 | 2026-02-27 10:00:00 | 62.55 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2026-03-13 09:40:00 | 54.88 | 2026-03-13 09:50:00 | 54.51 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-03-13 09:40:00 | 54.88 | 2026-03-13 15:20:00 | 53.20 | TARGET_HIT | 0.50 | 3.06% |
| BUY | retest1 | 2026-03-24 09:30:00 | 57.17 | 2026-03-24 09:35:00 | 56.76 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest1 | 2026-04-10 09:30:00 | 60.15 | 2026-04-10 09:35:00 | 59.86 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-04-28 09:45:00 | 67.35 | 2026-04-28 09:50:00 | 67.78 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-28 09:45:00 | 67.35 | 2026-04-28 09:55:00 | 67.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 09:40:00 | 65.94 | 2026-05-06 10:05:00 | 65.59 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-05-06 09:40:00 | 65.94 | 2026-05-06 14:25:00 | 65.94 | STOP_HIT | 0.50 | 0.00% |
