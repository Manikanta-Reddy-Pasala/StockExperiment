# IDBI Bank Ltd. (IDBI)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-10-01 15:25:00 (25983 bars)
- **Last close:** 92.35
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
| ENTRY1 | 79 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 16 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 114 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 62
- **Target hits / Stop hits / Partials:** 16 / 63 / 35
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 20.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 23 | 39.7% | 6 | 36 | 16 | 0.07% | 4.2% |
| BUY @ 2nd Alert (retest1) | 58 | 23 | 39.7% | 6 | 36 | 16 | 0.07% | 4.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 56 | 29 | 51.8% | 10 | 27 | 19 | 0.28% | 16.0% |
| SELL @ 2nd Alert (retest1) | 56 | 29 | 51.8% | 10 | 27 | 19 | 0.28% | 16.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 114 | 52 | 45.6% | 16 | 63 | 35 | 0.18% | 20.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:10:00 | 84.30 | 84.68 | 0.00 | ORB-short ORB[84.55,85.10] vol=1.6x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 12:05:00 | 83.94 | 84.55 | 0.00 | T1 1.5R @ 83.94 |
| Stop hit — per-position SL triggered | 2024-05-16 12:10:00 | 84.30 | 84.47 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 11:05:00 | 84.30 | 83.85 | 0.00 | ORB-long ORB[83.50,84.00] vol=1.6x ATR=0.22 |
| Stop hit — per-position SL triggered | 2024-05-17 11:30:00 | 84.08 | 83.88 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:45:00 | 85.70 | 85.22 | 0.00 | ORB-long ORB[84.75,85.35] vol=3.4x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-18 12:15:00 | 86.11 | 85.55 | 0.00 | T1 1.5R @ 86.11 |
| Stop hit — per-position SL triggered | 2024-05-21 09:15:00 | 86.50 | 0.00 | 0.00 | EOD overnight gap close |

### Cycle 4 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 87.25 | 88.26 | 0.00 | ORB-short ORB[88.30,89.00] vol=2.3x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-05-22 09:50:00 | 87.66 | 88.10 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:30:00 | 88.15 | 87.66 | 0.00 | ORB-long ORB[86.75,87.55] vol=5.5x ATR=0.30 |
| Stop hit — per-position SL triggered | 2024-05-27 09:45:00 | 87.85 | 87.83 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:40:00 | 84.05 | 84.39 | 0.00 | ORB-short ORB[84.10,85.05] vol=2.4x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:00:00 | 83.56 | 84.24 | 0.00 | T1 1.5R @ 83.56 |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 84.05 | 84.13 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:40:00 | 87.60 | 86.88 | 0.00 | ORB-long ORB[86.15,87.40] vol=2.1x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-06-11 11:00:00 | 87.26 | 86.95 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:15:00 | 87.99 | 87.46 | 0.00 | ORB-long ORB[87.04,87.67] vol=1.9x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:55:00 | 88.37 | 87.67 | 0.00 | T1 1.5R @ 88.37 |
| Stop hit — per-position SL triggered | 2024-06-12 11:25:00 | 87.99 | 87.71 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:15:00 | 86.72 | 87.37 | 0.00 | ORB-short ORB[87.51,87.98] vol=2.3x ATR=0.16 |
| Stop hit — per-position SL triggered | 2024-06-13 11:50:00 | 86.88 | 87.25 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:20:00 | 87.15 | 86.60 | 0.00 | ORB-long ORB[85.87,86.68] vol=3.4x ATR=0.30 |
| Stop hit — per-position SL triggered | 2024-06-14 10:45:00 | 86.85 | 86.65 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 10:00:00 | 87.25 | 86.43 | 0.00 | ORB-long ORB[86.02,87.19] vol=3.8x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 10:10:00 | 87.76 | 86.71 | 0.00 | T1 1.5R @ 87.76 |
| Stop hit — per-position SL triggered | 2024-06-19 10:15:00 | 87.25 | 86.73 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:40:00 | 88.40 | 87.56 | 0.00 | ORB-long ORB[86.71,87.73] vol=4.0x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-06-20 09:45:00 | 88.06 | 87.62 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:00:00 | 85.45 | 85.16 | 0.00 | ORB-long ORB[84.82,85.40] vol=1.7x ATR=0.22 |
| Stop hit — per-position SL triggered | 2024-06-24 10:15:00 | 85.23 | 85.20 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 85.36 | 86.05 | 0.00 | ORB-short ORB[86.01,86.59] vol=3.3x ATR=0.19 |
| Stop hit — per-position SL triggered | 2024-06-25 11:20:00 | 85.55 | 85.97 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-06-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:30:00 | 84.07 | 84.59 | 0.00 | ORB-short ORB[84.40,85.10] vol=1.9x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 13:35:00 | 83.80 | 84.31 | 0.00 | T1 1.5R @ 83.80 |
| Target hit | 2024-06-27 15:20:00 | 83.16 | 83.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2024-06-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:40:00 | 84.29 | 83.87 | 0.00 | ORB-long ORB[83.46,84.07] vol=2.5x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 09:50:00 | 84.66 | 84.19 | 0.00 | T1 1.5R @ 84.66 |
| Stop hit — per-position SL triggered | 2024-06-28 10:30:00 | 84.29 | 84.32 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-01 09:35:00 | 83.86 | 83.97 | 0.00 | ORB-short ORB[83.92,84.19] vol=3.8x ATR=0.16 |
| Stop hit — per-position SL triggered | 2024-07-01 09:45:00 | 84.02 | 83.97 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:30:00 | 83.53 | 84.12 | 0.00 | ORB-short ORB[84.14,84.47] vol=1.5x ATR=0.17 |
| Stop hit — per-position SL triggered | 2024-07-04 11:20:00 | 83.70 | 84.05 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:35:00 | 84.01 | 84.48 | 0.00 | ORB-short ORB[84.05,84.91] vol=1.7x ATR=0.23 |
| Stop hit — per-position SL triggered | 2024-07-05 10:35:00 | 84.24 | 84.29 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:40:00 | 85.20 | 84.81 | 0.00 | ORB-long ORB[84.45,85.06] vol=1.6x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 09:45:00 | 85.56 | 85.20 | 0.00 | T1 1.5R @ 85.56 |
| Target hit | 2024-07-09 10:15:00 | 86.07 | 86.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — SELL (started 2024-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 09:30:00 | 86.06 | 86.34 | 0.00 | ORB-short ORB[86.12,86.80] vol=1.5x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:15:00 | 85.72 | 86.11 | 0.00 | T1 1.5R @ 85.72 |
| Stop hit — per-position SL triggered | 2024-07-11 11:40:00 | 86.06 | 86.03 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:40:00 | 87.85 | 87.20 | 0.00 | ORB-long ORB[86.74,87.51] vol=3.6x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:55:00 | 88.32 | 87.49 | 0.00 | T1 1.5R @ 88.32 |
| Stop hit — per-position SL triggered | 2024-07-15 11:20:00 | 87.85 | 87.62 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 09:45:00 | 103.10 | 103.82 | 0.00 | ORB-short ORB[103.60,104.80] vol=1.8x ATR=0.38 |
| Stop hit — per-position SL triggered | 2024-07-31 10:00:00 | 103.48 | 103.70 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:50:00 | 102.19 | 102.88 | 0.00 | ORB-short ORB[102.90,104.29] vol=1.6x ATR=0.46 |
| Stop hit — per-position SL triggered | 2024-08-01 10:55:00 | 102.65 | 102.80 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 11:15:00 | 96.16 | 95.06 | 0.00 | ORB-long ORB[94.39,95.55] vol=3.3x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 11:35:00 | 96.81 | 95.33 | 0.00 | T1 1.5R @ 96.81 |
| Stop hit — per-position SL triggered | 2024-08-08 12:35:00 | 96.16 | 95.58 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 11:05:00 | 96.06 | 95.08 | 0.00 | ORB-long ORB[94.83,95.89] vol=4.2x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-08-09 11:10:00 | 95.67 | 95.29 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 11:05:00 | 93.44 | 93.97 | 0.00 | ORB-short ORB[93.65,94.49] vol=2.0x ATR=0.25 |
| Stop hit — per-position SL triggered | 2024-08-16 11:25:00 | 93.69 | 93.92 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:45:00 | 97.98 | 97.40 | 0.00 | ORB-long ORB[96.56,97.70] vol=3.0x ATR=0.32 |
| Stop hit — per-position SL triggered | 2024-08-21 09:50:00 | 97.66 | 97.42 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:50:00 | 98.64 | 97.96 | 0.00 | ORB-long ORB[97.09,98.20] vol=3.5x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:55:00 | 99.16 | 98.48 | 0.00 | T1 1.5R @ 99.16 |
| Target hit | 2024-08-22 12:40:00 | 100.03 | 100.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — SELL (started 2024-08-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:35:00 | 96.24 | 96.63 | 0.00 | ORB-short ORB[96.50,97.19] vol=1.7x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 09:40:00 | 95.85 | 96.44 | 0.00 | T1 1.5R @ 95.85 |
| Stop hit — per-position SL triggered | 2024-08-27 09:50:00 | 96.24 | 96.41 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:40:00 | 97.05 | 96.54 | 0.00 | ORB-long ORB[95.70,96.90] vol=2.2x ATR=0.35 |
| Stop hit — per-position SL triggered | 2024-08-28 10:05:00 | 96.70 | 96.69 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-08-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:50:00 | 94.80 | 95.93 | 0.00 | ORB-short ORB[95.80,96.72] vol=1.6x ATR=0.25 |
| Stop hit — per-position SL triggered | 2024-08-29 11:45:00 | 95.05 | 95.70 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 10:45:00 | 94.32 | 94.90 | 0.00 | ORB-short ORB[94.53,95.44] vol=1.5x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 11:25:00 | 94.01 | 94.74 | 0.00 | T1 1.5R @ 94.01 |
| Target hit | 2024-09-03 15:20:00 | 93.81 | 94.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2024-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:45:00 | 91.90 | 92.39 | 0.00 | ORB-short ORB[92.40,92.84] vol=2.6x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:50:00 | 91.59 | 92.28 | 0.00 | T1 1.5R @ 91.59 |
| Target hit | 2024-09-06 15:20:00 | 88.50 | 89.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2024-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 10:55:00 | 88.03 | 88.57 | 0.00 | ORB-short ORB[88.25,89.48] vol=2.4x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:45:00 | 87.73 | 88.47 | 0.00 | T1 1.5R @ 87.73 |
| Target hit | 2024-09-11 15:20:00 | 86.89 | 87.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2024-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:50:00 | 88.70 | 89.92 | 0.00 | ORB-short ORB[90.11,90.92] vol=3.4x ATR=0.35 |
| Stop hit — per-position SL triggered | 2024-09-19 10:00:00 | 89.05 | 89.76 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 11:05:00 | 89.57 | 88.94 | 0.00 | ORB-long ORB[88.34,89.55] vol=1.9x ATR=0.30 |
| Stop hit — per-position SL triggered | 2024-09-20 11:35:00 | 89.27 | 89.05 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:00:00 | 90.90 | 90.25 | 0.00 | ORB-long ORB[89.50,90.66] vol=2.4x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:05:00 | 91.36 | 90.83 | 0.00 | T1 1.5R @ 91.36 |
| Target hit | 2024-09-23 15:20:00 | 91.72 | 91.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2024-09-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 11:00:00 | 88.98 | 89.57 | 0.00 | ORB-short ORB[89.55,90.48] vol=1.9x ATR=0.22 |
| Stop hit — per-position SL triggered | 2024-09-25 11:05:00 | 89.20 | 89.55 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:40:00 | 87.71 | 88.27 | 0.00 | ORB-short ORB[88.09,89.33] vol=2.1x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 09:45:00 | 87.34 | 88.09 | 0.00 | T1 1.5R @ 87.34 |
| Stop hit — per-position SL triggered | 2024-09-26 10:00:00 | 87.71 | 88.03 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-09-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 09:30:00 | 87.46 | 87.85 | 0.00 | ORB-short ORB[87.52,88.64] vol=2.1x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:50:00 | 87.07 | 87.70 | 0.00 | T1 1.5R @ 87.07 |
| Stop hit — per-position SL triggered | 2024-09-30 09:55:00 | 87.46 | 87.68 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:30:00 | 84.02 | 83.74 | 0.00 | ORB-long ORB[83.41,83.99] vol=2.6x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 09:45:00 | 84.36 | 83.88 | 0.00 | T1 1.5R @ 84.36 |
| Stop hit — per-position SL triggered | 2024-10-10 10:35:00 | 84.02 | 84.05 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:35:00 | 83.31 | 83.63 | 0.00 | ORB-short ORB[83.45,84.18] vol=2.0x ATR=0.22 |
| Stop hit — per-position SL triggered | 2024-10-11 13:25:00 | 83.53 | 83.56 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 10:45:00 | 83.23 | 82.82 | 0.00 | ORB-long ORB[82.43,82.95] vol=1.6x ATR=0.22 |
| Stop hit — per-position SL triggered | 2024-10-15 10:55:00 | 83.01 | 82.84 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:55:00 | 81.95 | 82.43 | 0.00 | ORB-short ORB[82.52,83.20] vol=1.9x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:10:00 | 81.61 | 82.19 | 0.00 | T1 1.5R @ 81.61 |
| Target hit | 2024-10-17 15:20:00 | 81.15 | 81.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2024-10-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:35:00 | 81.65 | 81.06 | 0.00 | ORB-long ORB[80.40,81.50] vol=1.6x ATR=0.25 |
| Stop hit — per-position SL triggered | 2024-10-18 11:05:00 | 81.40 | 81.09 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-10-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:40:00 | 83.25 | 83.00 | 0.00 | ORB-long ORB[82.33,83.10] vol=2.0x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:55:00 | 83.76 | 83.35 | 0.00 | T1 1.5R @ 83.76 |
| Target hit | 2024-10-30 14:05:00 | 83.79 | 84.00 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — SELL (started 2024-11-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:25:00 | 81.66 | 82.55 | 0.00 | ORB-short ORB[82.62,83.70] vol=1.8x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 11:05:00 | 81.11 | 82.33 | 0.00 | T1 1.5R @ 81.11 |
| Stop hit — per-position SL triggered | 2024-11-04 12:00:00 | 81.66 | 82.21 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:50:00 | 84.09 | 83.53 | 0.00 | ORB-long ORB[82.95,83.73] vol=2.5x ATR=0.32 |
| Stop hit — per-position SL triggered | 2024-11-06 10:25:00 | 83.77 | 83.71 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 11:10:00 | 83.00 | 81.85 | 0.00 | ORB-long ORB[80.85,81.97] vol=1.7x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-11-11 12:00:00 | 82.66 | 82.09 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-11-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:50:00 | 81.70 | 82.04 | 0.00 | ORB-short ORB[81.84,82.60] vol=2.3x ATR=0.26 |
| Stop hit — per-position SL triggered | 2024-11-12 09:55:00 | 81.96 | 82.03 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 79.21 | 79.80 | 0.00 | ORB-short ORB[79.55,80.29] vol=1.6x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:40:00 | 78.80 | 79.51 | 0.00 | T1 1.5R @ 78.80 |
| Stop hit — per-position SL triggered | 2024-11-13 10:25:00 | 79.21 | 79.13 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-11-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 10:25:00 | 82.13 | 81.40 | 0.00 | ORB-long ORB[80.81,81.49] vol=1.9x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 10:30:00 | 82.61 | 81.66 | 0.00 | T1 1.5R @ 82.61 |
| Stop hit — per-position SL triggered | 2024-11-26 10:35:00 | 82.13 | 81.72 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-11-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:55:00 | 83.32 | 82.65 | 0.00 | ORB-long ORB[81.90,83.07] vol=1.6x ATR=0.35 |
| Stop hit — per-position SL triggered | 2024-11-29 11:05:00 | 82.97 | 82.67 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:20:00 | 86.41 | 85.60 | 0.00 | ORB-long ORB[85.10,86.17] vol=3.1x ATR=0.57 |
| Stop hit — per-position SL triggered | 2024-12-06 10:30:00 | 85.84 | 85.71 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:50:00 | 85.51 | 85.14 | 0.00 | ORB-long ORB[84.76,85.39] vol=2.7x ATR=0.28 |
| Stop hit — per-position SL triggered | 2024-12-10 10:00:00 | 85.23 | 85.16 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 83.40 | 83.77 | 0.00 | ORB-short ORB[83.60,84.19] vol=2.0x ATR=0.25 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 83.65 | 83.73 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:55:00 | 81.12 | 81.41 | 0.00 | ORB-short ORB[81.20,81.68] vol=1.9x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:25:00 | 80.83 | 81.31 | 0.00 | T1 1.5R @ 80.83 |
| Stop hit — per-position SL triggered | 2024-12-17 11:00:00 | 81.12 | 81.15 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-12-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:00:00 | 77.71 | 78.08 | 0.00 | ORB-short ORB[77.77,78.63] vol=1.8x ATR=0.25 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 77.96 | 78.05 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-12-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 11:10:00 | 76.24 | 75.82 | 0.00 | ORB-long ORB[75.52,75.95] vol=2.1x ATR=0.17 |
| Stop hit — per-position SL triggered | 2024-12-31 11:15:00 | 76.07 | 75.82 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:10:00 | 76.92 | 76.64 | 0.00 | ORB-long ORB[76.15,76.78] vol=2.5x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 10:50:00 | 77.30 | 76.73 | 0.00 | T1 1.5R @ 77.30 |
| Stop hit — per-position SL triggered | 2025-01-01 11:10:00 | 76.92 | 76.76 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:30:00 | 76.28 | 76.75 | 0.00 | ORB-short ORB[76.85,77.28] vol=1.9x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:45:00 | 76.03 | 76.59 | 0.00 | T1 1.5R @ 76.03 |
| Target hit | 2025-01-02 13:30:00 | 76.12 | 76.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 63 — BUY (started 2025-01-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 11:00:00 | 75.00 | 74.53 | 0.00 | ORB-long ORB[74.01,74.69] vol=1.8x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 13:30:00 | 75.47 | 74.76 | 0.00 | T1 1.5R @ 75.47 |
| Target hit | 2025-01-07 15:20:00 | 75.72 | 75.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2025-01-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 10:40:00 | 74.49 | 75.07 | 0.00 | ORB-short ORB[75.01,75.70] vol=2.8x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 10:45:00 | 74.13 | 74.85 | 0.00 | T1 1.5R @ 74.13 |
| Target hit | 2025-01-08 14:20:00 | 74.47 | 74.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 65 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:15:00 | 79.50 | 78.90 | 0.00 | ORB-long ORB[77.61,78.79] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-01-29 10:30:00 | 79.00 | 78.95 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-01-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:45:00 | 79.12 | 78.65 | 0.00 | ORB-long ORB[78.08,78.98] vol=2.7x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-01-30 10:45:00 | 78.72 | 78.76 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-01-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:50:00 | 80.00 | 79.37 | 0.00 | ORB-long ORB[78.74,79.64] vol=1.7x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-01-31 10:05:00 | 79.66 | 79.48 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-02-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:30:00 | 82.10 | 81.65 | 0.00 | ORB-long ORB[81.05,81.84] vol=2.3x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-02-01 09:40:00 | 81.76 | 81.73 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-02-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:35:00 | 79.92 | 79.49 | 0.00 | ORB-long ORB[78.50,79.62] vol=2.5x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-02-04 10:25:00 | 79.57 | 79.74 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:30:00 | 78.59 | 79.06 | 0.00 | ORB-short ORB[78.90,79.65] vol=1.8x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 10:00:00 | 78.06 | 78.72 | 0.00 | T1 1.5R @ 78.06 |
| Target hit | 2025-02-10 15:20:00 | 76.70 | 77.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2025-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:35:00 | 73.66 | 73.12 | 0.00 | ORB-long ORB[72.55,73.59] vol=2.2x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-02-20 09:45:00 | 73.32 | 73.19 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-03-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-10 10:30:00 | 71.90 | 72.37 | 0.00 | ORB-short ORB[71.92,72.98] vol=1.6x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 11:55:00 | 71.51 | 72.07 | 0.00 | T1 1.5R @ 71.51 |
| Stop hit — per-position SL triggered | 2025-03-10 13:05:00 | 71.90 | 71.95 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-03-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:00:00 | 75.73 | 75.54 | 0.00 | ORB-long ORB[74.71,75.68] vol=1.6x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:25:00 | 76.04 | 75.70 | 0.00 | T1 1.5R @ 76.04 |
| Target hit | 2025-03-21 11:45:00 | 75.82 | 75.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 74 — SELL (started 2025-04-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 10:35:00 | 74.88 | 75.42 | 0.00 | ORB-short ORB[75.34,76.40] vol=2.0x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-04-08 10:50:00 | 75.21 | 75.38 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-04-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:50:00 | 80.43 | 79.90 | 0.00 | ORB-long ORB[79.55,80.25] vol=2.5x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 11:50:00 | 80.80 | 80.15 | 0.00 | T1 1.5R @ 80.80 |
| Stop hit — per-position SL triggered | 2025-04-15 15:15:00 | 80.43 | 80.41 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 83.31 | 82.82 | 0.00 | ORB-long ORB[82.25,83.16] vol=2.8x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-04-21 10:00:00 | 83.05 | 83.00 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 11:15:00 | 84.01 | 84.56 | 0.00 | ORB-short ORB[84.92,85.99] vol=1.9x ATR=0.33 |
| Target hit | 2025-04-23 15:20:00 | 83.99 | 84.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 82.10 | 83.04 | 0.00 | ORB-short ORB[82.84,83.95] vol=2.7x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:40:00 | 81.70 | 82.84 | 0.00 | T1 1.5R @ 81.70 |
| Target hit | 2025-04-25 15:20:00 | 80.12 | 81.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2025-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:00:00 | 80.70 | 80.21 | 0.00 | ORB-long ORB[79.42,80.49] vol=5.6x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-05-05 11:25:00 | 80.38 | 80.27 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 11:10:00 | 84.30 | 2024-05-16 12:05:00 | 83.94 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-05-16 11:10:00 | 84.30 | 2024-05-16 12:10:00 | 84.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-17 11:05:00 | 84.30 | 2024-05-17 11:30:00 | 84.08 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-05-18 09:45:00 | 85.70 | 2024-05-18 12:15:00 | 86.11 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-05-18 09:45:00 | 85.70 | 2024-05-21 09:15:00 | 86.50 | STOP_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2024-05-22 09:40:00 | 87.25 | 2024-05-22 09:50:00 | 87.66 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-05-27 09:30:00 | 88.15 | 2024-05-27 09:45:00 | 87.85 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-31 09:40:00 | 84.05 | 2024-05-31 10:00:00 | 83.56 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-05-31 09:40:00 | 84.05 | 2024-05-31 11:15:00 | 84.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-11 10:40:00 | 87.60 | 2024-06-11 11:00:00 | 87.26 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-06-12 10:15:00 | 87.99 | 2024-06-12 10:55:00 | 88.37 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-06-12 10:15:00 | 87.99 | 2024-06-12 11:25:00 | 87.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 11:15:00 | 86.72 | 2024-06-13 11:50:00 | 86.88 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-06-14 10:20:00 | 87.15 | 2024-06-14 10:45:00 | 86.85 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-19 10:00:00 | 87.25 | 2024-06-19 10:10:00 | 87.76 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-06-19 10:00:00 | 87.25 | 2024-06-19 10:15:00 | 87.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 09:40:00 | 88.40 | 2024-06-20 09:45:00 | 88.06 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-06-24 10:00:00 | 85.45 | 2024-06-24 10:15:00 | 85.23 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-06-25 11:15:00 | 85.36 | 2024-06-25 11:20:00 | 85.55 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-06-27 10:30:00 | 84.07 | 2024-06-27 13:35:00 | 83.80 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-06-27 10:30:00 | 84.07 | 2024-06-27 15:20:00 | 83.16 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2024-06-28 09:40:00 | 84.29 | 2024-06-28 09:50:00 | 84.66 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-06-28 09:40:00 | 84.29 | 2024-06-28 10:30:00 | 84.29 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-01 09:35:00 | 83.86 | 2024-07-01 09:45:00 | 84.02 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-07-04 10:30:00 | 83.53 | 2024-07-04 11:20:00 | 83.70 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-07-05 09:35:00 | 84.01 | 2024-07-05 10:35:00 | 84.24 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-09 09:40:00 | 85.20 | 2024-07-09 09:45:00 | 85.56 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-09 09:40:00 | 85.20 | 2024-07-09 10:15:00 | 86.07 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2024-07-11 09:30:00 | 86.06 | 2024-07-11 10:15:00 | 85.72 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-11 09:30:00 | 86.06 | 2024-07-11 11:40:00 | 86.06 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-15 10:40:00 | 87.85 | 2024-07-15 10:55:00 | 88.32 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-15 10:40:00 | 87.85 | 2024-07-15 11:20:00 | 87.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-31 09:45:00 | 103.10 | 2024-07-31 10:00:00 | 103.48 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-08-01 10:50:00 | 102.19 | 2024-08-01 10:55:00 | 102.65 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-08-08 11:15:00 | 96.16 | 2024-08-08 11:35:00 | 96.81 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-08-08 11:15:00 | 96.16 | 2024-08-08 12:35:00 | 96.16 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-09 11:05:00 | 96.06 | 2024-08-09 11:10:00 | 95.67 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-08-16 11:05:00 | 93.44 | 2024-08-16 11:25:00 | 93.69 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-21 09:45:00 | 97.98 | 2024-08-21 09:50:00 | 97.66 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-22 10:50:00 | 98.64 | 2024-08-22 10:55:00 | 99.16 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-08-22 10:50:00 | 98.64 | 2024-08-22 12:40:00 | 100.03 | TARGET_HIT | 0.50 | 1.41% |
| SELL | retest1 | 2024-08-27 09:35:00 | 96.24 | 2024-08-27 09:40:00 | 95.85 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-08-27 09:35:00 | 96.24 | 2024-08-27 09:50:00 | 96.24 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-28 09:40:00 | 97.05 | 2024-08-28 10:05:00 | 96.70 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-29 10:50:00 | 94.80 | 2024-08-29 11:45:00 | 95.05 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-03 10:45:00 | 94.32 | 2024-09-03 11:25:00 | 94.01 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-09-03 10:45:00 | 94.32 | 2024-09-03 15:20:00 | 93.81 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2024-09-06 09:45:00 | 91.90 | 2024-09-06 09:50:00 | 91.59 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-09-06 09:45:00 | 91.90 | 2024-09-06 15:20:00 | 88.50 | TARGET_HIT | 0.50 | 3.70% |
| SELL | retest1 | 2024-09-11 10:55:00 | 88.03 | 2024-09-11 11:45:00 | 87.73 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-09-11 10:55:00 | 88.03 | 2024-09-11 15:20:00 | 86.89 | TARGET_HIT | 0.50 | 1.30% |
| SELL | retest1 | 2024-09-19 09:50:00 | 88.70 | 2024-09-19 10:00:00 | 89.05 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-09-20 11:05:00 | 89.57 | 2024-09-20 11:35:00 | 89.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-23 11:00:00 | 90.90 | 2024-09-23 11:05:00 | 91.36 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-09-23 11:00:00 | 90.90 | 2024-09-23 15:20:00 | 91.72 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2024-09-25 11:00:00 | 88.98 | 2024-09-25 11:05:00 | 89.20 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-26 09:40:00 | 87.71 | 2024-09-26 09:45:00 | 87.34 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-26 09:40:00 | 87.71 | 2024-09-26 10:00:00 | 87.71 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-30 09:30:00 | 87.46 | 2024-09-30 09:50:00 | 87.07 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-09-30 09:30:00 | 87.46 | 2024-09-30 09:55:00 | 87.46 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-10 09:30:00 | 84.02 | 2024-10-10 09:45:00 | 84.36 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-10-10 09:30:00 | 84.02 | 2024-10-10 10:35:00 | 84.02 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-11 10:35:00 | 83.31 | 2024-10-11 13:25:00 | 83.53 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-10-15 10:45:00 | 83.23 | 2024-10-15 10:55:00 | 83.01 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-17 09:55:00 | 81.95 | 2024-10-17 11:10:00 | 81.61 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-10-17 09:55:00 | 81.95 | 2024-10-17 15:20:00 | 81.15 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2024-10-18 10:35:00 | 81.65 | 2024-10-18 11:05:00 | 81.40 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-10-30 09:40:00 | 83.25 | 2024-10-30 09:55:00 | 83.76 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-10-30 09:40:00 | 83.25 | 2024-10-30 14:05:00 | 83.79 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2024-11-04 10:25:00 | 81.66 | 2024-11-04 11:05:00 | 81.11 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-11-04 10:25:00 | 81.66 | 2024-11-04 12:00:00 | 81.66 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-06 09:50:00 | 84.09 | 2024-11-06 10:25:00 | 83.77 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-11-11 11:10:00 | 83.00 | 2024-11-11 12:00:00 | 82.66 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-11-12 09:50:00 | 81.70 | 2024-11-12 09:55:00 | 81.96 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-13 09:30:00 | 79.21 | 2024-11-13 09:40:00 | 78.80 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-11-13 09:30:00 | 79.21 | 2024-11-13 10:25:00 | 79.21 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-26 10:25:00 | 82.13 | 2024-11-26 10:30:00 | 82.61 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-11-26 10:25:00 | 82.13 | 2024-11-26 10:35:00 | 82.13 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 10:55:00 | 83.32 | 2024-11-29 11:05:00 | 82.97 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-12-06 10:20:00 | 86.41 | 2024-12-06 10:30:00 | 85.84 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest1 | 2024-12-10 09:50:00 | 85.51 | 2024-12-10 10:00:00 | 85.23 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-12 09:40:00 | 83.40 | 2024-12-12 09:50:00 | 83.65 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-17 09:55:00 | 81.12 | 2024-12-17 10:25:00 | 80.83 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-17 09:55:00 | 81.12 | 2024-12-17 11:00:00 | 81.12 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 10:00:00 | 77.71 | 2024-12-20 10:15:00 | 77.96 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-31 11:10:00 | 76.24 | 2024-12-31 11:15:00 | 76.07 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-01-01 10:10:00 | 76.92 | 2025-01-01 10:50:00 | 77.30 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-01-01 10:10:00 | 76.92 | 2025-01-01 11:10:00 | 76.92 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-02 10:30:00 | 76.28 | 2025-01-02 11:45:00 | 76.03 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-01-02 10:30:00 | 76.28 | 2025-01-02 13:30:00 | 76.12 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2025-01-07 11:00:00 | 75.00 | 2025-01-07 13:30:00 | 75.47 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-01-07 11:00:00 | 75.00 | 2025-01-07 15:20:00 | 75.72 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2025-01-08 10:40:00 | 74.49 | 2025-01-08 10:45:00 | 74.13 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-01-08 10:40:00 | 74.49 | 2025-01-08 14:20:00 | 74.47 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2025-01-29 10:15:00 | 79.50 | 2025-01-29 10:30:00 | 79.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2025-01-30 09:45:00 | 79.12 | 2025-01-30 10:45:00 | 78.72 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-01-31 09:50:00 | 80.00 | 2025-01-31 10:05:00 | 79.66 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-02-01 09:30:00 | 82.10 | 2025-02-01 09:40:00 | 81.76 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-02-04 09:35:00 | 79.92 | 2025-02-04 10:25:00 | 79.57 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-02-10 09:30:00 | 78.59 | 2025-02-10 10:00:00 | 78.06 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2025-02-10 09:30:00 | 78.59 | 2025-02-10 15:20:00 | 76.70 | TARGET_HIT | 0.50 | 2.40% |
| BUY | retest1 | 2025-02-20 09:35:00 | 73.66 | 2025-02-20 09:45:00 | 73.32 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-03-10 10:30:00 | 71.90 | 2025-03-10 11:55:00 | 71.51 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-03-10 10:30:00 | 71.90 | 2025-03-10 13:05:00 | 71.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 10:00:00 | 75.73 | 2025-03-21 10:25:00 | 76.04 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-03-21 10:00:00 | 75.73 | 2025-03-21 11:45:00 | 75.82 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2025-04-08 10:35:00 | 74.88 | 2025-04-08 10:50:00 | 75.21 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-04-15 10:50:00 | 80.43 | 2025-04-15 11:50:00 | 80.80 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-04-15 10:50:00 | 80.43 | 2025-04-15 15:15:00 | 80.43 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 09:30:00 | 83.31 | 2025-04-21 10:00:00 | 83.05 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-04-23 11:15:00 | 84.01 | 2025-04-23 15:20:00 | 83.99 | TARGET_HIT | 1.00 | 0.02% |
| SELL | retest1 | 2025-04-25 09:35:00 | 82.10 | 2025-04-25 09:40:00 | 81.70 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-04-25 09:35:00 | 82.10 | 2025-04-25 15:20:00 | 80.12 | TARGET_HIT | 0.50 | 2.41% |
| BUY | retest1 | 2025-05-05 11:00:00 | 80.70 | 2025-05-05 11:25:00 | 80.38 | STOP_HIT | 1.00 | -0.40% |
