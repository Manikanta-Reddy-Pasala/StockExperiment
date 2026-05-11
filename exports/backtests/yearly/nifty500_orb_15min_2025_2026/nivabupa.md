# Niva Bupa Health Insurance Company Ltd. (NIVABUPA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 81.25
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
| ENTRY1 | 68 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 9 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 91 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 59
- **Target hits / Stop hits / Partials:** 9 / 59 / 23
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 6.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 16 | 37.2% | 5 | 27 | 11 | 0.08% | 3.2% |
| BUY @ 2nd Alert (retest1) | 43 | 16 | 37.2% | 5 | 27 | 11 | 0.08% | 3.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 16 | 33.3% | 4 | 32 | 12 | 0.06% | 2.8% |
| SELL @ 2nd Alert (retest1) | 48 | 16 | 33.3% | 4 | 32 | 12 | 0.06% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 91 | 32 | 35.2% | 9 | 59 | 23 | 0.07% | 6.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 11:05:00 | 82.51 | 82.94 | 0.00 | ORB-short ORB[82.78,83.60] vol=2.4x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-06-05 11:15:00 | 82.69 | 82.93 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-06-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 10:10:00 | 81.18 | 80.58 | 0.00 | ORB-long ORB[79.72,80.71] vol=1.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-06-18 10:25:00 | 80.90 | 80.62 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:40:00 | 81.03 | 80.54 | 0.00 | ORB-long ORB[80.01,80.85] vol=1.8x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:45:00 | 81.50 | 80.85 | 0.00 | T1 1.5R @ 81.50 |
| Stop hit — per-position SL triggered | 2025-06-20 09:50:00 | 81.03 | 80.86 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-24 11:15:00 | 81.47 | 81.79 | 0.00 | ORB-short ORB[81.66,82.49] vol=5.4x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-06-24 12:25:00 | 81.63 | 81.71 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:15:00 | 81.17 | 81.40 | 0.00 | ORB-short ORB[81.30,82.15] vol=2.3x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 11:05:00 | 80.86 | 81.27 | 0.00 | T1 1.5R @ 80.86 |
| Stop hit — per-position SL triggered | 2025-07-01 14:00:00 | 81.17 | 81.08 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:30:00 | 81.38 | 81.58 | 0.00 | ORB-short ORB[81.54,82.21] vol=1.7x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-07-02 09:45:00 | 81.67 | 81.51 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-07-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:00:00 | 82.62 | 82.41 | 0.00 | ORB-long ORB[81.81,82.59] vol=2.4x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 11:15:00 | 83.03 | 82.65 | 0.00 | T1 1.5R @ 83.03 |
| Stop hit — per-position SL triggered | 2025-07-04 11:45:00 | 82.62 | 82.83 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-07-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:45:00 | 90.22 | 89.65 | 0.00 | ORB-long ORB[88.06,89.40] vol=2.1x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-07-14 10:25:00 | 89.81 | 89.84 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 10:45:00 | 89.05 | 89.65 | 0.00 | ORB-short ORB[89.13,90.47] vol=2.4x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-07-16 10:50:00 | 89.37 | 89.58 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:30:00 | 89.58 | 90.07 | 0.00 | ORB-short ORB[89.79,90.90] vol=1.5x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-07-17 09:35:00 | 89.91 | 90.08 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 11:10:00 | 90.27 | 90.48 | 0.00 | ORB-short ORB[90.74,91.50] vol=1.9x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 12:35:00 | 89.85 | 90.43 | 0.00 | T1 1.5R @ 89.85 |
| Target hit | 2025-07-18 15:20:00 | 89.04 | 90.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-07-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:20:00 | 87.87 | 88.52 | 0.00 | ORB-short ORB[88.60,89.25] vol=2.7x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 10:25:00 | 87.55 | 88.27 | 0.00 | T1 1.5R @ 87.55 |
| Stop hit — per-position SL triggered | 2025-07-22 12:05:00 | 87.87 | 87.98 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:20:00 | 85.85 | 86.58 | 0.00 | ORB-short ORB[86.63,87.47] vol=1.5x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-07-24 10:30:00 | 86.10 | 86.52 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:35:00 | 85.04 | 85.73 | 0.00 | ORB-short ORB[85.51,86.56] vol=4.4x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 12:20:00 | 84.58 | 85.41 | 0.00 | T1 1.5R @ 84.58 |
| Stop hit — per-position SL triggered | 2025-07-25 13:10:00 | 85.04 | 85.31 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:35:00 | 83.90 | 84.63 | 0.00 | ORB-short ORB[84.51,85.48] vol=3.7x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-07-30 10:35:00 | 84.24 | 84.29 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:40:00 | 81.56 | 81.91 | 0.00 | ORB-short ORB[81.81,82.45] vol=4.1x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-08-05 09:55:00 | 81.77 | 81.79 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:30:00 | 81.58 | 81.94 | 0.00 | ORB-short ORB[81.73,82.45] vol=1.6x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:35:00 | 81.29 | 81.80 | 0.00 | T1 1.5R @ 81.29 |
| Target hit | 2025-08-06 15:20:00 | 80.64 | 80.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-08-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 09:40:00 | 79.80 | 80.18 | 0.00 | ORB-short ORB[79.83,80.99] vol=2.6x ATR=0.27 |
| Stop hit — per-position SL triggered | 2025-08-07 09:45:00 | 80.07 | 80.16 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 09:30:00 | 80.80 | 81.07 | 0.00 | ORB-short ORB[80.90,81.48] vol=1.5x ATR=0.23 |
| Stop hit — per-position SL triggered | 2025-08-13 09:50:00 | 81.03 | 81.00 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 10:45:00 | 81.94 | 81.68 | 0.00 | ORB-long ORB[81.17,81.75] vol=2.5x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-08-14 13:25:00 | 81.75 | 81.81 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 11:10:00 | 85.32 | 84.72 | 0.00 | ORB-long ORB[84.00,84.81] vol=12.4x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-08-20 11:20:00 | 85.06 | 84.77 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:05:00 | 86.75 | 85.89 | 0.00 | ORB-long ORB[85.01,85.71] vol=4.0x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 86.32 | 86.03 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:30:00 | 81.01 | 81.39 | 0.00 | ORB-short ORB[81.19,81.94] vol=2.3x ATR=0.27 |
| Stop hit — per-position SL triggered | 2025-08-29 09:40:00 | 81.28 | 81.31 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:35:00 | 83.90 | 83.46 | 0.00 | ORB-long ORB[82.70,83.72] vol=1.8x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:45:00 | 84.42 | 83.68 | 0.00 | T1 1.5R @ 84.42 |
| Stop hit — per-position SL triggered | 2025-09-03 10:05:00 | 83.90 | 84.09 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:40:00 | 82.10 | 81.85 | 0.00 | ORB-long ORB[81.50,82.00] vol=2.6x ATR=0.20 |
| Stop hit — per-position SL triggered | 2025-09-12 11:30:00 | 81.90 | 81.90 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 09:35:00 | 81.52 | 81.69 | 0.00 | ORB-short ORB[81.53,81.98] vol=2.5x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-09-15 09:40:00 | 81.69 | 81.68 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:55:00 | 82.66 | 82.29 | 0.00 | ORB-long ORB[82.00,82.50] vol=3.2x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-09-19 10:00:00 | 82.48 | 82.30 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:15:00 | 82.32 | 81.68 | 0.00 | ORB-long ORB[81.24,81.80] vol=2.2x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 11:10:00 | 82.72 | 82.11 | 0.00 | T1 1.5R @ 82.72 |
| Target hit | 2025-10-01 12:30:00 | 82.36 | 82.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2025-10-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:30:00 | 78.85 | 79.11 | 0.00 | ORB-short ORB[78.87,79.67] vol=2.9x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-10-08 09:55:00 | 79.01 | 79.07 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 10:35:00 | 78.32 | 78.71 | 0.00 | ORB-short ORB[78.65,79.31] vol=3.3x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:40:00 | 78.00 | 78.46 | 0.00 | T1 1.5R @ 78.00 |
| Target hit | 2025-10-13 15:20:00 | 77.19 | 77.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2025-10-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:05:00 | 75.36 | 75.68 | 0.00 | ORB-short ORB[75.91,76.90] vol=1.6x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:20:00 | 74.89 | 75.48 | 0.00 | T1 1.5R @ 74.89 |
| Stop hit — per-position SL triggered | 2025-10-20 10:30:00 | 75.36 | 75.46 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:50:00 | 74.03 | 74.45 | 0.00 | ORB-short ORB[74.50,74.99] vol=1.8x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:45:00 | 73.77 | 74.33 | 0.00 | T1 1.5R @ 73.77 |
| Stop hit — per-position SL triggered | 2025-10-24 12:20:00 | 74.03 | 74.27 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 11:05:00 | 73.76 | 74.17 | 0.00 | ORB-short ORB[73.83,74.87] vol=1.9x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-10-27 11:45:00 | 73.93 | 74.11 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 11:10:00 | 75.04 | 74.83 | 0.00 | ORB-long ORB[74.51,75.03] vol=1.8x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-10-28 12:00:00 | 74.87 | 74.85 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 11:10:00 | 74.92 | 74.60 | 0.00 | ORB-long ORB[73.81,74.87] vol=3.5x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 15:15:00 | 75.25 | 74.83 | 0.00 | T1 1.5R @ 75.25 |
| Target hit | 2025-10-29 15:20:00 | 75.11 | 74.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2025-10-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:20:00 | 74.73 | 75.03 | 0.00 | ORB-short ORB[74.76,75.70] vol=2.8x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-10-30 12:25:00 | 74.95 | 74.94 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:30:00 | 75.18 | 75.02 | 0.00 | ORB-long ORB[74.34,75.11] vol=2.0x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-10-31 09:35:00 | 75.01 | 75.04 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-11-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 11:05:00 | 76.52 | 76.17 | 0.00 | ORB-long ORB[75.80,76.22] vol=6.1x ATR=0.15 |
| Stop hit — per-position SL triggered | 2025-11-11 11:10:00 | 76.37 | 76.19 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-11-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 11:05:00 | 76.66 | 76.49 | 0.00 | ORB-long ORB[75.71,76.28] vol=13.5x ATR=0.17 |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 76.49 | 76.50 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:30:00 | 76.48 | 77.01 | 0.00 | ORB-short ORB[76.86,77.75] vol=2.7x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:40:00 | 76.20 | 76.91 | 0.00 | T1 1.5R @ 76.20 |
| Stop hit — per-position SL triggered | 2025-11-19 10:45:00 | 76.48 | 76.83 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-11-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 11:00:00 | 77.17 | 76.58 | 0.00 | ORB-long ORB[76.05,76.64] vol=3.6x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 11:20:00 | 77.42 | 76.74 | 0.00 | T1 1.5R @ 77.42 |
| Stop hit — per-position SL triggered | 2025-11-20 11:25:00 | 77.17 | 76.74 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:45:00 | 75.93 | 76.52 | 0.00 | ORB-short ORB[76.25,77.03] vol=1.5x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-11-21 09:50:00 | 76.14 | 76.47 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-11-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 09:40:00 | 75.00 | 75.18 | 0.00 | ORB-short ORB[75.01,75.57] vol=1.9x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:10:00 | 74.69 | 75.08 | 0.00 | T1 1.5R @ 74.69 |
| Target hit | 2025-11-24 15:20:00 | 74.22 | 74.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2025-11-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-25 09:55:00 | 75.37 | 74.81 | 0.00 | ORB-long ORB[74.10,74.99] vol=2.0x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-11-25 10:00:00 | 75.12 | 74.83 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-12-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 10:45:00 | 73.21 | 73.39 | 0.00 | ORB-short ORB[73.23,74.00] vol=3.5x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-12-02 11:05:00 | 73.39 | 73.38 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-12-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 09:40:00 | 74.42 | 74.19 | 0.00 | ORB-long ORB[73.90,74.36] vol=2.8x ATR=0.16 |
| Stop hit — per-position SL triggered | 2025-12-05 09:45:00 | 74.26 | 74.18 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:50:00 | 74.99 | 74.52 | 0.00 | ORB-long ORB[74.00,74.58] vol=2.3x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-12-10 09:55:00 | 74.77 | 74.54 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:15:00 | 73.93 | 73.71 | 0.00 | ORB-long ORB[73.45,73.91] vol=1.6x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 12:15:00 | 74.16 | 73.78 | 0.00 | T1 1.5R @ 74.16 |
| Stop hit — per-position SL triggered | 2025-12-11 13:10:00 | 73.93 | 73.86 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 10:50:00 | 74.70 | 74.89 | 0.00 | ORB-short ORB[74.88,75.50] vol=1.8x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 11:35:00 | 74.39 | 74.81 | 0.00 | T1 1.5R @ 74.39 |
| Stop hit — per-position SL triggered | 2025-12-15 11:55:00 | 74.70 | 74.78 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2026-01-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 09:50:00 | 78.44 | 77.76 | 0.00 | ORB-long ORB[77.48,77.94] vol=1.6x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 09:55:00 | 78.75 | 78.18 | 0.00 | T1 1.5R @ 78.75 |
| Target hit | 2026-01-14 15:20:00 | 79.00 | 78.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2026-01-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:30:00 | 79.39 | 78.99 | 0.00 | ORB-long ORB[78.71,79.25] vol=3.6x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-01-16 10:40:00 | 79.14 | 79.03 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-01-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 10:05:00 | 77.08 | 77.31 | 0.00 | ORB-short ORB[77.25,78.00] vol=1.6x ATR=0.18 |
| Stop hit — per-position SL triggered | 2026-01-23 10:40:00 | 77.26 | 77.24 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-01-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 10:40:00 | 78.00 | 77.50 | 0.00 | ORB-long ORB[77.02,77.70] vol=5.7x ATR=0.20 |
| Stop hit — per-position SL triggered | 2026-01-29 12:55:00 | 77.80 | 77.85 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-02-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-03 09:30:00 | 79.48 | 79.76 | 0.00 | ORB-short ORB[79.51,80.51] vol=1.8x ATR=0.49 |
| Stop hit — per-position SL triggered | 2026-02-03 10:30:00 | 79.97 | 79.58 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-02-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 10:00:00 | 77.55 | 78.01 | 0.00 | ORB-short ORB[77.67,78.55] vol=1.5x ATR=0.26 |
| Stop hit — per-position SL triggered | 2026-02-05 13:55:00 | 77.81 | 77.85 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-02-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:20:00 | 77.05 | 77.13 | 0.00 | ORB-short ORB[77.11,77.74] vol=1.8x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:25:00 | 76.79 | 77.07 | 0.00 | T1 1.5R @ 76.79 |
| Stop hit — per-position SL triggered | 2026-02-12 11:40:00 | 77.05 | 77.05 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 76.65 | 76.53 | 0.00 | ORB-long ORB[76.25,76.62] vol=2.8x ATR=0.18 |
| Stop hit — per-position SL triggered | 2026-02-18 09:45:00 | 76.47 | 76.52 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-02-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:55:00 | 75.88 | 76.16 | 0.00 | ORB-short ORB[76.12,76.67] vol=1.9x ATR=0.16 |
| Stop hit — per-position SL triggered | 2026-02-19 10:00:00 | 76.04 | 76.14 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-02-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:05:00 | 76.51 | 76.75 | 0.00 | ORB-short ORB[76.75,77.17] vol=1.9x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-02-23 10:10:00 | 76.70 | 76.75 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:40:00 | 76.76 | 76.70 | 0.00 | ORB-long ORB[76.32,76.71] vol=5.5x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-02-25 10:00:00 | 76.57 | 76.71 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:15:00 | 69.96 | 70.17 | 0.00 | ORB-short ORB[70.00,71.00] vol=1.6x ATR=0.20 |
| Stop hit — per-position SL triggered | 2026-03-12 10:40:00 | 70.16 | 70.12 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 70.84 | 70.75 | 0.00 | ORB-long ORB[70.11,70.74] vol=1.7x ATR=0.34 |
| Stop hit — per-position SL triggered | 2026-03-20 09:45:00 | 70.50 | 70.70 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-03-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:50:00 | 67.87 | 68.33 | 0.00 | ORB-short ORB[68.60,69.56] vol=1.6x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-03-24 11:40:00 | 68.17 | 68.26 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-03-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:10:00 | 70.93 | 70.34 | 0.00 | ORB-long ORB[69.33,70.31] vol=1.9x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:15:00 | 71.44 | 70.58 | 0.00 | T1 1.5R @ 71.44 |
| Target hit | 2026-03-25 13:50:00 | 72.31 | 72.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 65 — BUY (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 11:15:00 | 74.36 | 73.59 | 0.00 | ORB-long ORB[73.10,73.75] vol=11.5x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-04-09 11:45:00 | 74.04 | 73.84 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-04-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:00:00 | 73.11 | 72.57 | 0.00 | ORB-long ORB[71.86,72.67] vol=3.1x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:20:00 | 73.57 | 72.65 | 0.00 | T1 1.5R @ 73.57 |
| Target hit | 2026-04-13 15:20:00 | 74.03 | 73.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2026-04-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:35:00 | 76.46 | 75.65 | 0.00 | ORB-long ORB[74.78,75.84] vol=7.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2026-04-15 10:40:00 | 76.13 | 75.68 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 78.30 | 78.10 | 0.00 | ORB-long ORB[77.49,78.28] vol=2.1x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:10:00 | 78.61 | 78.28 | 0.00 | T1 1.5R @ 78.61 |
| Stop hit — per-position SL triggered | 2026-04-21 12:10:00 | 78.30 | 78.39 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-06-05 11:05:00 | 82.51 | 2025-06-05 11:15:00 | 82.69 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-18 10:10:00 | 81.18 | 2025-06-18 10:25:00 | 80.90 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-20 09:40:00 | 81.03 | 2025-06-20 09:45:00 | 81.50 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-06-20 09:40:00 | 81.03 | 2025-06-20 09:50:00 | 81.03 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-24 11:15:00 | 81.47 | 2025-06-24 12:25:00 | 81.63 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-01 10:15:00 | 81.17 | 2025-07-01 11:05:00 | 80.86 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-01 10:15:00 | 81.17 | 2025-07-01 14:00:00 | 81.17 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-02 09:30:00 | 81.38 | 2025-07-02 09:45:00 | 81.67 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-07-04 10:00:00 | 82.62 | 2025-07-04 11:15:00 | 83.03 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-07-04 10:00:00 | 82.62 | 2025-07-04 11:45:00 | 82.62 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-14 09:45:00 | 90.22 | 2025-07-14 10:25:00 | 89.81 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-07-16 10:45:00 | 89.05 | 2025-07-16 10:50:00 | 89.37 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-07-17 09:30:00 | 89.58 | 2025-07-17 09:35:00 | 89.91 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-07-18 11:10:00 | 90.27 | 2025-07-18 12:35:00 | 89.85 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-07-18 11:10:00 | 90.27 | 2025-07-18 15:20:00 | 89.04 | TARGET_HIT | 0.50 | 1.36% |
| SELL | retest1 | 2025-07-22 10:20:00 | 87.87 | 2025-07-22 10:25:00 | 87.55 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-22 10:20:00 | 87.87 | 2025-07-22 12:05:00 | 87.87 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 10:20:00 | 85.85 | 2025-07-24 10:30:00 | 86.10 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-25 10:35:00 | 85.04 | 2025-07-25 12:20:00 | 84.58 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-07-25 10:35:00 | 85.04 | 2025-07-25 13:10:00 | 85.04 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-30 09:35:00 | 83.90 | 2025-07-30 10:35:00 | 84.24 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-08-05 09:40:00 | 81.56 | 2025-08-05 09:55:00 | 81.77 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-06 09:30:00 | 81.58 | 2025-08-06 09:35:00 | 81.29 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-06 09:30:00 | 81.58 | 2025-08-06 15:20:00 | 80.64 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2025-08-07 09:40:00 | 79.80 | 2025-08-07 09:45:00 | 80.07 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-08-13 09:30:00 | 80.80 | 2025-08-13 09:50:00 | 81.03 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-14 10:45:00 | 81.94 | 2025-08-14 13:25:00 | 81.75 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-20 11:10:00 | 85.32 | 2025-08-20 11:20:00 | 85.06 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-08-22 10:05:00 | 86.75 | 2025-08-22 10:15:00 | 86.32 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-08-29 09:30:00 | 81.01 | 2025-08-29 09:40:00 | 81.28 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-03 09:35:00 | 83.90 | 2025-09-03 09:45:00 | 84.42 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-09-03 09:35:00 | 83.90 | 2025-09-03 10:05:00 | 83.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-12 10:40:00 | 82.10 | 2025-09-12 11:30:00 | 81.90 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-15 09:35:00 | 81.52 | 2025-09-15 09:40:00 | 81.69 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-19 09:55:00 | 82.66 | 2025-09-19 10:00:00 | 82.48 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-01 10:15:00 | 82.32 | 2025-10-01 11:10:00 | 82.72 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-10-01 10:15:00 | 82.32 | 2025-10-01 12:30:00 | 82.36 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2025-10-08 09:30:00 | 78.85 | 2025-10-08 09:55:00 | 79.01 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-13 10:35:00 | 78.32 | 2025-10-13 10:40:00 | 78.00 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-10-13 10:35:00 | 78.32 | 2025-10-13 15:20:00 | 77.19 | TARGET_HIT | 0.50 | 1.44% |
| SELL | retest1 | 2025-10-20 10:05:00 | 75.36 | 2025-10-20 10:20:00 | 74.89 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-10-20 10:05:00 | 75.36 | 2025-10-20 10:30:00 | 75.36 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-24 10:50:00 | 74.03 | 2025-10-24 11:45:00 | 73.77 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-10-24 10:50:00 | 74.03 | 2025-10-24 12:20:00 | 74.03 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-27 11:05:00 | 73.76 | 2025-10-27 11:45:00 | 73.93 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-28 11:10:00 | 75.04 | 2025-10-28 12:00:00 | 74.87 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-29 11:10:00 | 74.92 | 2025-10-29 15:15:00 | 75.25 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-10-29 11:10:00 | 74.92 | 2025-10-29 15:20:00 | 75.11 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2025-10-30 10:20:00 | 74.73 | 2025-10-30 12:25:00 | 74.95 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-31 09:30:00 | 75.18 | 2025-10-31 09:35:00 | 75.01 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-11 11:05:00 | 76.52 | 2025-11-11 11:10:00 | 76.37 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-14 11:05:00 | 76.66 | 2025-11-14 11:15:00 | 76.49 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-19 10:30:00 | 76.48 | 2025-11-19 10:40:00 | 76.20 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-11-19 10:30:00 | 76.48 | 2025-11-19 10:45:00 | 76.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-20 11:00:00 | 77.17 | 2025-11-20 11:20:00 | 77.42 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-11-20 11:00:00 | 77.17 | 2025-11-20 11:25:00 | 77.17 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 09:45:00 | 75.93 | 2025-11-21 09:50:00 | 76.14 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-11-24 09:40:00 | 75.00 | 2025-11-24 11:10:00 | 74.69 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-11-24 09:40:00 | 75.00 | 2025-11-24 15:20:00 | 74.22 | TARGET_HIT | 0.50 | 1.04% |
| BUY | retest1 | 2025-11-25 09:55:00 | 75.37 | 2025-11-25 10:00:00 | 75.12 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-02 10:45:00 | 73.21 | 2025-12-02 11:05:00 | 73.39 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-05 09:40:00 | 74.42 | 2025-12-05 09:45:00 | 74.26 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-10 09:50:00 | 74.99 | 2025-12-10 09:55:00 | 74.77 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-11 11:15:00 | 73.93 | 2025-12-11 12:15:00 | 74.16 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-12-11 11:15:00 | 73.93 | 2025-12-11 13:10:00 | 73.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-15 10:50:00 | 74.70 | 2025-12-15 11:35:00 | 74.39 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-12-15 10:50:00 | 74.70 | 2025-12-15 11:55:00 | 74.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-14 09:50:00 | 78.44 | 2026-01-14 09:55:00 | 78.75 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-01-14 09:50:00 | 78.44 | 2026-01-14 15:20:00 | 79.00 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2026-01-16 10:30:00 | 79.39 | 2026-01-16 10:40:00 | 79.14 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-23 10:05:00 | 77.08 | 2026-01-23 10:40:00 | 77.26 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-01-29 10:40:00 | 78.00 | 2026-01-29 12:55:00 | 77.80 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-03 09:30:00 | 79.48 | 2026-02-03 10:30:00 | 79.97 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2026-02-05 10:00:00 | 77.55 | 2026-02-05 13:55:00 | 77.81 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-12 10:20:00 | 77.05 | 2026-02-12 11:25:00 | 76.79 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-12 10:20:00 | 77.05 | 2026-02-12 11:40:00 | 77.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 09:40:00 | 76.65 | 2026-02-18 09:45:00 | 76.47 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-19 09:55:00 | 75.88 | 2026-02-19 10:00:00 | 76.04 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-23 10:05:00 | 76.51 | 2026-02-23 10:10:00 | 76.70 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-25 09:40:00 | 76.76 | 2026-02-25 10:00:00 | 76.57 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-12 10:15:00 | 69.96 | 2026-03-12 10:40:00 | 70.16 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-20 09:35:00 | 70.84 | 2026-03-20 09:45:00 | 70.50 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-03-24 10:50:00 | 67.87 | 2026-03-24 11:40:00 | 68.17 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-25 10:10:00 | 70.93 | 2026-03-25 10:15:00 | 71.44 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-03-25 10:10:00 | 70.93 | 2026-03-25 13:50:00 | 72.31 | TARGET_HIT | 0.50 | 1.95% |
| BUY | retest1 | 2026-04-09 11:15:00 | 74.36 | 2026-04-09 11:45:00 | 74.04 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-13 10:00:00 | 73.11 | 2026-04-13 10:20:00 | 73.57 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-13 10:00:00 | 73.11 | 2026-04-13 15:20:00 | 74.03 | TARGET_HIT | 0.50 | 1.26% |
| BUY | retest1 | 2026-04-15 10:35:00 | 76.46 | 2026-04-15 10:40:00 | 76.13 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-21 09:45:00 | 78.30 | 2026-04-21 11:10:00 | 78.61 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-21 09:45:00 | 78.30 | 2026-04-21 12:10:00 | 78.30 | STOP_HIT | 0.50 | 0.00% |
