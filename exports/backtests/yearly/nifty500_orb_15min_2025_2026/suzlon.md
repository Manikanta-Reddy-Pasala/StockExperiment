# Suzlon Energy Ltd. (SUZLON)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 54.90
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
| ENTRY1 | 43 |
| ENTRY2 | 0 |
| PARTIAL | 14 |
| TARGET_HIT | 5 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 38
- **Target hits / Stop hits / Partials:** 5 / 38 / 14
- **Avg / median % per leg:** 0.11% / -0.14%
- **Sum % (uncompounded):** 6.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 10 | 33.3% | 3 | 20 | 7 | 0.18% | 5.3% |
| BUY @ 2nd Alert (retest1) | 30 | 10 | 33.3% | 3 | 20 | 7 | 0.18% | 5.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 9 | 33.3% | 2 | 18 | 7 | 0.04% | 1.0% |
| SELL @ 2nd Alert (retest1) | 27 | 9 | 33.3% | 2 | 18 | 7 | 0.04% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 57 | 19 | 33.3% | 5 | 38 | 14 | 0.11% | 6.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:50:00 | 57.90 | 57.21 | 0.00 | ORB-long ORB[56.71,57.46] vol=1.8x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-05-13 10:00:00 | 57.62 | 57.26 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 09:55:00 | 61.17 | 61.67 | 0.00 | ORB-short ORB[61.31,62.13] vol=1.9x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-05-16 10:00:00 | 61.42 | 61.66 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:35:00 | 60.48 | 59.92 | 0.00 | ORB-long ORB[59.35,60.20] vol=2.0x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 10:05:00 | 60.93 | 60.09 | 0.00 | T1 1.5R @ 60.93 |
| Target hit | 2025-05-21 15:20:00 | 61.26 | 60.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2025-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:30:00 | 66.49 | 66.07 | 0.00 | ORB-long ORB[65.51,66.40] vol=2.4x ATR=0.27 |
| Stop hit — per-position SL triggered | 2025-05-28 09:35:00 | 66.22 | 66.08 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 09:30:00 | 67.06 | 67.34 | 0.00 | ORB-short ORB[67.13,67.85] vol=2.0x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-06-10 09:40:00 | 67.24 | 67.30 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:50:00 | 68.78 | 68.47 | 0.00 | ORB-long ORB[67.91,68.75] vol=3.5x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-06-11 10:20:00 | 68.49 | 68.51 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:45:00 | 64.96 | 64.41 | 0.00 | ORB-long ORB[63.81,64.68] vol=2.0x ATR=0.27 |
| Stop hit — per-position SL triggered | 2025-06-18 09:50:00 | 64.69 | 64.43 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 11:05:00 | 63.68 | 63.84 | 0.00 | ORB-short ORB[63.77,64.30] vol=2.9x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:15:00 | 63.40 | 63.82 | 0.00 | T1 1.5R @ 63.40 |
| Target hit | 2025-06-19 15:20:00 | 62.65 | 62.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2025-06-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-25 11:05:00 | 64.13 | 64.41 | 0.00 | ORB-short ORB[64.23,64.90] vol=2.0x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 12:10:00 | 63.91 | 64.35 | 0.00 | T1 1.5R @ 63.91 |
| Stop hit — per-position SL triggered | 2025-06-25 13:00:00 | 64.13 | 64.32 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:30:00 | 65.05 | 64.58 | 0.00 | ORB-long ORB[64.08,64.64] vol=4.9x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-06-26 09:35:00 | 64.84 | 64.64 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:00:00 | 65.02 | 64.76 | 0.00 | ORB-long ORB[64.56,64.88] vol=2.8x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 11:15:00 | 65.31 | 64.91 | 0.00 | T1 1.5R @ 65.31 |
| Target hit | 2025-06-27 15:20:00 | 67.46 | 66.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-07-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:50:00 | 65.44 | 65.75 | 0.00 | ORB-short ORB[65.77,66.24] vol=2.1x ATR=0.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:25:00 | 65.27 | 65.70 | 0.00 | T1 1.5R @ 65.27 |
| Stop hit — per-position SL triggered | 2025-07-24 15:00:00 | 65.44 | 65.50 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:30:00 | 60.84 | 60.53 | 0.00 | ORB-long ORB[60.01,60.75] vol=1.6x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:50:00 | 61.31 | 60.67 | 0.00 | T1 1.5R @ 61.31 |
| Stop hit — per-position SL triggered | 2025-07-29 10:25:00 | 60.84 | 60.87 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 11:15:00 | 61.35 | 61.43 | 0.00 | ORB-short ORB[61.37,62.05] vol=1.7x ATR=0.15 |
| Stop hit — per-position SL triggered | 2025-07-30 11:35:00 | 61.50 | 61.43 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 09:30:00 | 66.10 | 65.54 | 0.00 | ORB-long ORB[64.87,65.70] vol=2.2x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-08-05 09:35:00 | 65.86 | 65.61 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-08-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 11:00:00 | 59.98 | 59.51 | 0.00 | ORB-long ORB[59.30,59.72] vol=3.2x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-08-20 11:05:00 | 59.82 | 59.55 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:35:00 | 55.77 | 56.04 | 0.00 | ORB-short ORB[55.94,56.65] vol=2.4x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-08-29 09:40:00 | 55.96 | 56.03 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:55:00 | 56.97 | 57.11 | 0.00 | ORB-short ORB[57.06,57.32] vol=1.8x ATR=0.08 |
| Stop hit — per-position SL triggered | 2025-09-12 11:05:00 | 57.05 | 57.10 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 11:15:00 | 59.12 | 59.39 | 0.00 | ORB-short ORB[59.25,59.66] vol=1.5x ATR=0.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 11:45:00 | 58.99 | 59.35 | 0.00 | T1 1.5R @ 58.99 |
| Stop hit — per-position SL triggered | 2025-09-18 15:00:00 | 59.12 | 59.16 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-09-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 10:20:00 | 59.94 | 59.54 | 0.00 | ORB-long ORB[59.06,59.70] vol=2.4x ATR=0.15 |
| Stop hit — per-position SL triggered | 2025-09-19 10:25:00 | 59.79 | 59.58 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-09-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:05:00 | 58.07 | 58.37 | 0.00 | ORB-short ORB[58.15,58.88] vol=2.0x ATR=0.12 |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 58.19 | 58.35 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-10-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 10:45:00 | 54.70 | 55.03 | 0.00 | ORB-short ORB[54.96,55.26] vol=1.8x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-10-01 11:30:00 | 54.87 | 54.98 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 54.50 | 54.31 | 0.00 | ORB-long ORB[54.05,54.42] vol=1.7x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 09:35:00 | 54.72 | 54.40 | 0.00 | T1 1.5R @ 54.72 |
| Stop hit — per-position SL triggered | 2025-10-07 09:45:00 | 54.50 | 54.43 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-10-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:40:00 | 53.44 | 53.82 | 0.00 | ORB-short ORB[53.85,54.14] vol=1.7x ATR=0.11 |
| Stop hit — per-position SL triggered | 2025-10-08 10:45:00 | 53.55 | 53.81 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-10-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:40:00 | 54.32 | 53.77 | 0.00 | ORB-long ORB[53.23,53.74] vol=2.8x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-10-10 09:50:00 | 54.14 | 53.86 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 53.89 | 54.20 | 0.00 | ORB-short ORB[54.03,54.48] vol=2.0x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-10-13 10:05:00 | 54.05 | 54.10 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-10-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 10:45:00 | 52.73 | 52.96 | 0.00 | ORB-short ORB[52.87,53.64] vol=1.8x ATR=0.11 |
| Stop hit — per-position SL triggered | 2025-10-17 10:50:00 | 52.84 | 52.91 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-10-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:45:00 | 55.05 | 54.81 | 0.00 | ORB-long ORB[54.51,54.86] vol=2.3x ATR=0.15 |
| Stop hit — per-position SL triggered | 2025-10-23 11:05:00 | 54.90 | 54.84 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:30:00 | 54.11 | 53.88 | 0.00 | ORB-long ORB[53.60,53.98] vol=3.6x ATR=0.12 |
| Stop hit — per-position SL triggered | 2025-10-28 09:40:00 | 53.99 | 53.90 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:50:00 | 59.74 | 59.18 | 0.00 | ORB-long ORB[58.70,59.47] vol=3.7x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-10-31 10:00:00 | 59.52 | 59.26 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-11-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 11:00:00 | 60.15 | 59.61 | 0.00 | ORB-long ORB[59.20,59.94] vol=6.0x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-11-04 11:05:00 | 59.83 | 59.64 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-11-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:30:00 | 56.88 | 57.32 | 0.00 | ORB-short ORB[57.20,57.85] vol=1.8x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-11-11 09:35:00 | 57.05 | 57.28 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-11-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:05:00 | 56.08 | 56.38 | 0.00 | ORB-short ORB[56.10,56.72] vol=1.6x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:25:00 | 55.86 | 56.28 | 0.00 | T1 1.5R @ 55.86 |
| Stop hit — per-position SL triggered | 2025-11-21 10:55:00 | 56.08 | 56.24 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-11-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 10:05:00 | 54.32 | 54.53 | 0.00 | ORB-short ORB[54.40,55.00] vol=2.2x ATR=0.14 |
| Stop hit — per-position SL triggered | 2025-11-28 11:00:00 | 54.46 | 54.49 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-12-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 10:30:00 | 53.81 | 53.53 | 0.00 | ORB-long ORB[53.24,53.77] vol=1.8x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 10:50:00 | 54.06 | 53.57 | 0.00 | T1 1.5R @ 54.06 |
| Stop hit — per-position SL triggered | 2025-12-02 11:40:00 | 53.81 | 53.62 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-12-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:55:00 | 52.95 | 53.26 | 0.00 | ORB-short ORB[53.25,53.65] vol=1.8x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:05:00 | 52.75 | 53.16 | 0.00 | T1 1.5R @ 52.75 |
| Target hit | 2025-12-03 15:20:00 | 52.66 | 52.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2025-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:30:00 | 53.10 | 52.90 | 0.00 | ORB-long ORB[52.68,53.05] vol=1.6x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-12-10 09:55:00 | 52.93 | 52.96 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-12-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 09:45:00 | 51.94 | 51.68 | 0.00 | ORB-long ORB[51.32,51.90] vol=1.8x ATR=0.20 |
| Stop hit — per-position SL triggered | 2025-12-11 10:00:00 | 51.74 | 51.78 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-12-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:55:00 | 53.83 | 53.57 | 0.00 | ORB-long ORB[53.35,53.74] vol=1.8x ATR=0.12 |
| Stop hit — per-position SL triggered | 2025-12-23 10:00:00 | 53.71 | 53.59 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:15:00 | 52.71 | 52.52 | 0.00 | ORB-long ORB[51.91,52.65] vol=1.5x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 11:35:00 | 52.91 | 52.53 | 0.00 | T1 1.5R @ 52.91 |
| Stop hit — per-position SL triggered | 2025-12-31 13:35:00 | 52.71 | 52.67 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2026-01-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:40:00 | 53.02 | 52.72 | 0.00 | ORB-long ORB[52.44,52.80] vol=2.7x ATR=0.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:50:00 | 53.18 | 52.82 | 0.00 | T1 1.5R @ 53.18 |
| Target hit | 2026-01-02 15:20:00 | 54.27 | 53.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2026-01-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:00:00 | 53.30 | 53.50 | 0.00 | ORB-short ORB[53.32,53.99] vol=2.1x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:05:00 | 53.10 | 53.46 | 0.00 | T1 1.5R @ 53.10 |
| Stop hit — per-position SL triggered | 2026-01-06 12:55:00 | 53.30 | 53.32 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 56.94 | 57.40 | 0.00 | ORB-short ORB[57.37,58.06] vol=1.5x ATR=0.22 |
| Stop hit — per-position SL triggered | 2026-04-29 10:15:00 | 57.16 | 57.34 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 09:50:00 | 57.90 | 2025-05-13 10:00:00 | 57.62 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-05-16 09:55:00 | 61.17 | 2025-05-16 10:00:00 | 61.42 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-05-21 09:35:00 | 60.48 | 2025-05-21 10:05:00 | 60.93 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2025-05-21 09:35:00 | 60.48 | 2025-05-21 15:20:00 | 61.26 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2025-05-28 09:30:00 | 66.49 | 2025-05-28 09:35:00 | 66.22 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-06-10 09:30:00 | 67.06 | 2025-06-10 09:40:00 | 67.24 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-11 09:50:00 | 68.78 | 2025-06-11 10:20:00 | 68.49 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-06-18 09:45:00 | 64.96 | 2025-06-18 09:50:00 | 64.69 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-06-19 11:05:00 | 63.68 | 2025-06-19 11:15:00 | 63.40 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-06-19 11:05:00 | 63.68 | 2025-06-19 15:20:00 | 62.65 | TARGET_HIT | 0.50 | 1.62% |
| SELL | retest1 | 2025-06-25 11:05:00 | 64.13 | 2025-06-25 12:10:00 | 63.91 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-06-25 11:05:00 | 64.13 | 2025-06-25 13:00:00 | 64.13 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-26 09:30:00 | 65.05 | 2025-06-26 09:35:00 | 64.84 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-27 10:00:00 | 65.02 | 2025-06-27 11:15:00 | 65.31 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-06-27 10:00:00 | 65.02 | 2025-06-27 15:20:00 | 67.46 | TARGET_HIT | 0.50 | 3.75% |
| SELL | retest1 | 2025-07-24 10:50:00 | 65.44 | 2025-07-24 11:25:00 | 65.27 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-07-24 10:50:00 | 65.44 | 2025-07-24 15:00:00 | 65.44 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-29 09:30:00 | 60.84 | 2025-07-29 09:50:00 | 61.31 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2025-07-29 09:30:00 | 60.84 | 2025-07-29 10:25:00 | 60.84 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-30 11:15:00 | 61.35 | 2025-07-30 11:35:00 | 61.50 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-05 09:30:00 | 66.10 | 2025-08-05 09:35:00 | 65.86 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-08-20 11:00:00 | 59.98 | 2025-08-20 11:05:00 | 59.82 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-29 09:35:00 | 55.77 | 2025-08-29 09:40:00 | 55.96 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-09-12 10:55:00 | 56.97 | 2025-09-12 11:05:00 | 57.05 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-09-18 11:15:00 | 59.12 | 2025-09-18 11:45:00 | 58.99 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-09-18 11:15:00 | 59.12 | 2025-09-18 15:00:00 | 59.12 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-19 10:20:00 | 59.94 | 2025-09-19 10:25:00 | 59.79 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-24 11:05:00 | 58.07 | 2025-09-24 11:15:00 | 58.19 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-01 10:45:00 | 54.70 | 2025-10-01 11:30:00 | 54.87 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-07 09:30:00 | 54.50 | 2025-10-07 09:35:00 | 54.72 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-10-07 09:30:00 | 54.50 | 2025-10-07 09:45:00 | 54.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-08 10:40:00 | 53.44 | 2025-10-08 10:45:00 | 53.55 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-10 09:40:00 | 54.32 | 2025-10-10 09:50:00 | 54.14 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-10-13 09:30:00 | 53.89 | 2025-10-13 10:05:00 | 54.05 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-17 10:45:00 | 52.73 | 2025-10-17 10:50:00 | 52.84 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-23 10:45:00 | 55.05 | 2025-10-23 11:05:00 | 54.90 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-28 09:30:00 | 54.11 | 2025-10-28 09:40:00 | 53.99 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-31 09:50:00 | 59.74 | 2025-10-31 10:00:00 | 59.52 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-11-04 11:00:00 | 60.15 | 2025-11-04 11:05:00 | 59.83 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2025-11-11 09:30:00 | 56.88 | 2025-11-11 09:35:00 | 57.05 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-21 10:05:00 | 56.08 | 2025-11-21 10:25:00 | 55.86 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-11-21 10:05:00 | 56.08 | 2025-11-21 10:55:00 | 56.08 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-28 10:05:00 | 54.32 | 2025-11-28 11:00:00 | 54.46 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-02 10:30:00 | 53.81 | 2025-12-02 10:50:00 | 54.06 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-12-02 10:30:00 | 53.81 | 2025-12-02 11:40:00 | 53.81 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 09:55:00 | 52.95 | 2025-12-03 10:05:00 | 52.75 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-12-03 09:55:00 | 52.95 | 2025-12-03 15:20:00 | 52.66 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2025-12-10 09:30:00 | 53.10 | 2025-12-10 09:55:00 | 52.93 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-11 09:45:00 | 51.94 | 2025-12-11 10:00:00 | 51.74 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-12-23 09:55:00 | 53.83 | 2025-12-23 10:00:00 | 53.71 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-31 11:15:00 | 52.71 | 2025-12-31 11:35:00 | 52.91 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-12-31 11:15:00 | 52.71 | 2025-12-31 13:35:00 | 52.71 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 10:40:00 | 53.02 | 2026-01-02 10:50:00 | 53.18 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-01-02 10:40:00 | 53.02 | 2026-01-02 15:20:00 | 54.27 | TARGET_HIT | 0.50 | 2.36% |
| SELL | retest1 | 2026-01-06 11:00:00 | 53.30 | 2026-01-06 11:05:00 | 53.10 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-01-06 11:00:00 | 53.30 | 2026-01-06 12:55:00 | 53.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 09:55:00 | 56.94 | 2026-04-29 10:15:00 | 57.16 | STOP_HIT | 1.00 | -0.39% |
