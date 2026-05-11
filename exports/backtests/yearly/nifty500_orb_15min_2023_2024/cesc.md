# CESC Ltd. (CESC)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2025-10-27 15:25:00 (45604 bars)
- **Last close:** 180.00
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
| ENTRY1 | 89 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 12 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 120 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 77
- **Target hits / Stop hits / Partials:** 12 / 77 / 31
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 17.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 74 | 26 | 35.1% | 7 | 48 | 19 | 0.15% | 11.1% |
| BUY @ 2nd Alert (retest1) | 74 | 26 | 35.1% | 7 | 48 | 19 | 0.15% | 11.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 46 | 17 | 37.0% | 5 | 29 | 12 | 0.13% | 6.0% |
| SELL @ 2nd Alert (retest1) | 46 | 17 | 37.0% | 5 | 29 | 12 | 0.13% | 6.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 120 | 43 | 35.8% | 12 | 77 | 31 | 0.14% | 17.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 09:45:00 | 71.00 | 70.65 | 0.00 | ORB-long ORB[70.20,70.75] vol=4.3x ATR=0.19 |
| Stop hit — per-position SL triggered | 2023-05-15 09:50:00 | 70.81 | 70.68 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-17 09:35:00 | 71.45 | 71.04 | 0.00 | ORB-long ORB[70.05,70.85] vol=6.0x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 09:45:00 | 71.89 | 71.30 | 0.00 | T1 1.5R @ 71.89 |
| Stop hit — per-position SL triggered | 2023-05-17 10:00:00 | 71.45 | 71.36 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-18 09:35:00 | 70.85 | 70.55 | 0.00 | ORB-long ORB[69.80,70.80] vol=3.4x ATR=0.24 |
| Stop hit — per-position SL triggered | 2023-05-18 09:40:00 | 70.61 | 70.56 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 11:00:00 | 72.15 | 71.84 | 0.00 | ORB-long ORB[71.25,71.95] vol=2.8x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-22 11:15:00 | 72.40 | 72.04 | 0.00 | T1 1.5R @ 72.40 |
| Stop hit — per-position SL triggered | 2023-05-22 11:30:00 | 72.15 | 72.08 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-05-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-26 09:40:00 | 69.75 | 70.00 | 0.00 | ORB-short ORB[69.90,70.20] vol=1.9x ATR=0.16 |
| Stop hit — per-position SL triggered | 2023-05-26 09:45:00 | 69.91 | 69.98 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 09:45:00 | 70.60 | 70.20 | 0.00 | ORB-long ORB[69.55,70.30] vol=7.2x ATR=0.20 |
| Stop hit — per-position SL triggered | 2023-05-29 09:55:00 | 70.40 | 70.24 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-05-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 09:50:00 | 69.40 | 69.62 | 0.00 | ORB-short ORB[69.55,69.90] vol=3.1x ATR=0.16 |
| Target hit | 2023-05-30 15:20:00 | 69.30 | 69.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2023-05-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 09:35:00 | 69.45 | 69.14 | 0.00 | ORB-long ORB[68.75,69.20] vol=1.9x ATR=0.15 |
| Stop hit — per-position SL triggered | 2023-05-31 09:55:00 | 69.30 | 69.18 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 10:55:00 | 70.75 | 70.38 | 0.00 | ORB-long ORB[70.10,70.60] vol=2.1x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 11:05:00 | 70.97 | 70.57 | 0.00 | T1 1.5R @ 70.97 |
| Stop hit — per-position SL triggered | 2023-06-05 11:25:00 | 70.75 | 70.60 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 09:35:00 | 72.45 | 72.15 | 0.00 | ORB-long ORB[71.60,72.25] vol=3.0x ATR=0.21 |
| Stop hit — per-position SL triggered | 2023-06-08 09:45:00 | 72.24 | 72.18 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-09 11:05:00 | 71.95 | 71.48 | 0.00 | ORB-long ORB[71.30,71.80] vol=4.9x ATR=0.18 |
| Stop hit — per-position SL triggered | 2023-06-09 11:20:00 | 71.77 | 71.53 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 10:20:00 | 72.20 | 71.67 | 0.00 | ORB-long ORB[71.30,71.75] vol=4.6x ATR=0.19 |
| Stop hit — per-position SL triggered | 2023-06-13 10:25:00 | 72.01 | 71.68 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 10:00:00 | 71.80 | 72.02 | 0.00 | ORB-short ORB[72.00,72.35] vol=1.6x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-14 12:25:00 | 71.54 | 71.83 | 0.00 | T1 1.5R @ 71.54 |
| Stop hit — per-position SL triggered | 2023-06-14 13:15:00 | 71.80 | 71.83 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-06-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 11:05:00 | 71.80 | 72.32 | 0.00 | ORB-short ORB[72.00,72.90] vol=1.9x ATR=0.18 |
| Stop hit — per-position SL triggered | 2023-06-19 12:35:00 | 71.98 | 72.14 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-06-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 09:55:00 | 72.50 | 72.23 | 0.00 | ORB-long ORB[71.90,72.40] vol=1.7x ATR=0.20 |
| Stop hit — per-position SL triggered | 2023-06-20 10:25:00 | 72.30 | 72.29 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 10:15:00 | 72.90 | 73.21 | 0.00 | ORB-short ORB[73.10,73.65] vol=1.6x ATR=0.17 |
| Stop hit — per-position SL triggered | 2023-06-21 10:20:00 | 73.07 | 73.20 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-06-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 11:10:00 | 71.90 | 72.10 | 0.00 | ORB-short ORB[72.15,72.40] vol=1.6x ATR=0.12 |
| Stop hit — per-position SL triggered | 2023-06-27 11:25:00 | 72.02 | 72.09 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-06-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 09:35:00 | 71.75 | 71.99 | 0.00 | ORB-short ORB[71.90,72.35] vol=2.1x ATR=0.17 |
| Stop hit — per-position SL triggered | 2023-06-28 11:25:00 | 71.92 | 72.16 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 10:35:00 | 72.50 | 72.30 | 0.00 | ORB-long ORB[72.00,72.45] vol=1.8x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 15:05:00 | 72.73 | 72.47 | 0.00 | T1 1.5R @ 72.73 |
| Stop hit — per-position SL triggered | 2023-07-03 15:10:00 | 72.50 | 72.47 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 11:05:00 | 73.35 | 73.11 | 0.00 | ORB-long ORB[72.50,73.30] vol=2.4x ATR=0.15 |
| Stop hit — per-position SL triggered | 2023-07-04 11:35:00 | 73.20 | 73.13 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-07-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 09:40:00 | 74.45 | 73.90 | 0.00 | ORB-long ORB[72.80,73.60] vol=9.0x ATR=0.24 |
| Stop hit — per-position SL triggered | 2023-07-06 10:15:00 | 74.21 | 74.21 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 10:50:00 | 75.70 | 75.12 | 0.00 | ORB-long ORB[75.05,75.55] vol=1.6x ATR=0.25 |
| Stop hit — per-position SL triggered | 2023-07-10 10:55:00 | 75.45 | 75.15 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:40:00 | 75.50 | 75.34 | 0.00 | ORB-long ORB[75.05,75.40] vol=2.9x ATR=0.18 |
| Stop hit — per-position SL triggered | 2023-07-11 09:45:00 | 75.32 | 75.35 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-07-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 09:35:00 | 74.60 | 74.80 | 0.00 | ORB-short ORB[74.65,75.10] vol=1.6x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 09:50:00 | 74.31 | 74.69 | 0.00 | T1 1.5R @ 74.31 |
| Stop hit — per-position SL triggered | 2023-07-12 09:55:00 | 74.60 | 74.72 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-07-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 10:30:00 | 75.90 | 75.55 | 0.00 | ORB-long ORB[75.25,75.70] vol=5.3x ATR=0.22 |
| Stop hit — per-position SL triggered | 2023-07-14 11:45:00 | 75.68 | 75.63 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-17 11:15:00 | 75.40 | 75.80 | 0.00 | ORB-short ORB[75.45,76.35] vol=3.2x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 11:40:00 | 75.08 | 75.70 | 0.00 | T1 1.5R @ 75.08 |
| Stop hit — per-position SL triggered | 2023-07-17 13:45:00 | 75.40 | 75.44 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-07-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:50:00 | 75.50 | 75.33 | 0.00 | ORB-long ORB[75.05,75.45] vol=1.9x ATR=0.17 |
| Stop hit — per-position SL triggered | 2023-07-25 10:25:00 | 75.33 | 75.41 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 10:15:00 | 75.90 | 76.50 | 0.00 | ORB-short ORB[76.35,77.30] vol=3.4x ATR=0.22 |
| Stop hit — per-position SL triggered | 2023-07-28 10:20:00 | 76.12 | 76.48 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 09:30:00 | 76.75 | 76.32 | 0.00 | ORB-long ORB[75.50,76.55] vol=2.2x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 09:40:00 | 77.15 | 76.53 | 0.00 | T1 1.5R @ 77.15 |
| Target hit | 2023-07-31 15:20:00 | 78.00 | 77.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2023-08-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-03 09:45:00 | 78.15 | 77.49 | 0.00 | ORB-long ORB[76.85,78.00] vol=3.3x ATR=0.37 |
| Stop hit — per-position SL triggered | 2023-08-03 09:50:00 | 77.78 | 77.56 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-08-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 10:10:00 | 79.70 | 78.97 | 0.00 | ORB-long ORB[78.40,79.15] vol=2.9x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 10:15:00 | 80.19 | 79.21 | 0.00 | T1 1.5R @ 80.19 |
| Stop hit — per-position SL triggered | 2023-08-04 10:20:00 | 79.70 | 79.26 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 10:45:00 | 77.60 | 78.06 | 0.00 | ORB-short ORB[78.00,78.60] vol=3.8x ATR=0.20 |
| Stop hit — per-position SL triggered | 2023-08-08 11:00:00 | 77.80 | 78.03 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-08-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 11:10:00 | 78.00 | 78.29 | 0.00 | ORB-short ORB[78.10,78.85] vol=3.9x ATR=0.18 |
| Stop hit — per-position SL triggered | 2023-08-09 11:40:00 | 78.18 | 78.26 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 09:55:00 | 79.15 | 78.03 | 0.00 | ORB-long ORB[77.30,77.95] vol=5.7x ATR=0.28 |
| Stop hit — per-position SL triggered | 2023-08-11 10:00:00 | 78.87 | 78.24 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-08-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 09:45:00 | 78.25 | 77.89 | 0.00 | ORB-long ORB[77.60,78.05] vol=2.1x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-16 10:15:00 | 78.60 | 78.04 | 0.00 | T1 1.5R @ 78.60 |
| Stop hit — per-position SL triggered | 2023-08-16 11:45:00 | 78.25 | 78.29 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-08-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 10:00:00 | 78.60 | 78.30 | 0.00 | ORB-long ORB[78.00,78.55] vol=2.2x ATR=0.20 |
| Stop hit — per-position SL triggered | 2023-08-17 10:20:00 | 78.40 | 78.36 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 09:30:00 | 79.55 | 79.21 | 0.00 | ORB-long ORB[78.70,79.30] vol=5.1x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-23 09:35:00 | 79.96 | 79.80 | 0.00 | T1 1.5R @ 79.96 |
| Target hit | 2023-08-23 15:20:00 | 83.80 | 82.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2023-08-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:35:00 | 84.95 | 84.38 | 0.00 | ORB-long ORB[83.50,84.65] vol=1.7x ATR=0.41 |
| Stop hit — per-position SL triggered | 2023-08-24 09:50:00 | 84.54 | 84.49 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:30:00 | 84.15 | 83.61 | 0.00 | ORB-long ORB[83.25,83.85] vol=2.1x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 09:35:00 | 84.52 | 83.92 | 0.00 | T1 1.5R @ 84.52 |
| Stop hit — per-position SL triggered | 2023-08-30 10:05:00 | 84.15 | 84.18 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 11:10:00 | 89.25 | 88.26 | 0.00 | ORB-long ORB[88.10,89.00] vol=5.3x ATR=0.48 |
| Stop hit — per-position SL triggered | 2023-09-04 11:15:00 | 88.77 | 88.28 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 10:30:00 | 93.00 | 91.91 | 0.00 | ORB-long ORB[91.05,92.35] vol=5.4x ATR=0.52 |
| Stop hit — per-position SL triggered | 2023-09-07 10:35:00 | 92.48 | 91.99 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-09-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 09:45:00 | 92.05 | 91.31 | 0.00 | ORB-long ORB[90.90,91.50] vol=1.9x ATR=0.42 |
| Stop hit — per-position SL triggered | 2023-09-08 09:55:00 | 91.63 | 91.53 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-18 09:35:00 | 90.45 | 89.88 | 0.00 | ORB-long ORB[89.05,90.40] vol=1.9x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-18 09:50:00 | 91.08 | 90.45 | 0.00 | T1 1.5R @ 91.08 |
| Target hit | 2023-09-18 11:30:00 | 91.05 | 91.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — BUY (started 2023-09-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-25 09:35:00 | 92.25 | 91.46 | 0.00 | ORB-long ORB[90.45,91.60] vol=4.4x ATR=0.42 |
| Stop hit — per-position SL triggered | 2023-09-25 09:50:00 | 91.83 | 91.64 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-09-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 10:00:00 | 91.50 | 91.14 | 0.00 | ORB-long ORB[90.25,91.35] vol=3.7x ATR=0.32 |
| Stop hit — per-position SL triggered | 2023-09-26 10:05:00 | 91.18 | 91.15 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-09-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 10:35:00 | 89.90 | 90.67 | 0.00 | ORB-short ORB[90.80,91.40] vol=1.5x ATR=0.25 |
| Stop hit — per-position SL triggered | 2023-09-27 11:00:00 | 90.15 | 90.57 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-10-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 09:55:00 | 90.50 | 89.91 | 0.00 | ORB-long ORB[89.40,90.30] vol=4.9x ATR=0.30 |
| Stop hit — per-position SL triggered | 2023-10-03 10:00:00 | 90.20 | 90.03 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-10-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-04 09:35:00 | 90.20 | 90.47 | 0.00 | ORB-short ORB[90.25,91.50] vol=1.9x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 10:15:00 | 89.76 | 90.22 | 0.00 | T1 1.5R @ 89.76 |
| Target hit | 2023-10-04 14:20:00 | 88.90 | 88.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 49 — BUY (started 2023-10-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 10:20:00 | 89.65 | 88.87 | 0.00 | ORB-long ORB[88.25,89.30] vol=2.9x ATR=0.34 |
| Stop hit — per-position SL triggered | 2023-10-11 10:25:00 | 89.31 | 89.21 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 09:30:00 | 91.00 | 90.40 | 0.00 | ORB-long ORB[89.05,89.75] vol=10.8x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 09:35:00 | 91.60 | 90.78 | 0.00 | T1 1.5R @ 91.60 |
| Stop hit — per-position SL triggered | 2023-10-17 10:15:00 | 91.00 | 91.06 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-10-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 11:10:00 | 91.95 | 92.60 | 0.00 | ORB-short ORB[92.15,93.50] vol=2.1x ATR=0.37 |
| Stop hit — per-position SL triggered | 2023-10-18 11:15:00 | 92.32 | 92.59 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 09:55:00 | 85.20 | 84.49 | 0.00 | ORB-long ORB[83.75,84.90] vol=2.1x ATR=0.29 |
| Stop hit — per-position SL triggered | 2023-11-01 10:10:00 | 84.91 | 84.65 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-11-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 09:50:00 | 85.40 | 85.01 | 0.00 | ORB-long ORB[84.50,85.30] vol=2.6x ATR=0.25 |
| Stop hit — per-position SL triggered | 2023-11-02 10:00:00 | 85.15 | 85.08 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 10:00:00 | 88.00 | 87.61 | 0.00 | ORB-long ORB[87.20,87.90] vol=5.0x ATR=0.25 |
| Stop hit — per-position SL triggered | 2023-11-08 10:05:00 | 87.75 | 87.63 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-11-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 09:35:00 | 90.25 | 89.69 | 0.00 | ORB-long ORB[89.05,89.80] vol=2.0x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 09:50:00 | 90.72 | 90.17 | 0.00 | T1 1.5R @ 90.72 |
| Target hit | 2023-11-13 10:15:00 | 90.35 | 90.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — SELL (started 2023-11-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-17 11:00:00 | 95.80 | 96.30 | 0.00 | ORB-short ORB[96.10,97.30] vol=2.2x ATR=0.33 |
| Stop hit — per-position SL triggered | 2023-11-17 11:20:00 | 96.13 | 96.28 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-11-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 09:35:00 | 99.05 | 98.37 | 0.00 | ORB-long ORB[97.65,98.80] vol=1.8x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 09:40:00 | 99.80 | 98.80 | 0.00 | T1 1.5R @ 99.80 |
| Stop hit — per-position SL triggered | 2023-11-20 09:55:00 | 99.05 | 98.90 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2023-11-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 09:55:00 | 99.45 | 100.17 | 0.00 | ORB-short ORB[99.90,100.80] vol=2.1x ATR=0.46 |
| Stop hit — per-position SL triggered | 2023-11-23 10:00:00 | 99.91 | 100.15 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-11-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 10:45:00 | 98.40 | 99.04 | 0.00 | ORB-short ORB[98.75,99.65] vol=2.1x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 12:20:00 | 97.93 | 98.85 | 0.00 | T1 1.5R @ 97.93 |
| Stop hit — per-position SL triggered | 2023-11-24 12:50:00 | 98.40 | 98.79 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2023-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-28 09:35:00 | 97.35 | 97.92 | 0.00 | ORB-short ORB[97.75,98.60] vol=2.4x ATR=0.30 |
| Stop hit — per-position SL triggered | 2023-11-28 09:40:00 | 97.65 | 97.88 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-11-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 09:30:00 | 99.75 | 98.84 | 0.00 | ORB-long ORB[97.35,98.55] vol=6.9x ATR=0.49 |
| Stop hit — per-position SL triggered | 2023-11-29 09:35:00 | 99.26 | 98.92 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-11-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:35:00 | 97.80 | 98.37 | 0.00 | ORB-short ORB[98.40,99.40] vol=2.4x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 09:50:00 | 97.23 | 97.98 | 0.00 | T1 1.5R @ 97.23 |
| Target hit | 2023-11-30 15:20:00 | 97.10 | 97.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2023-12-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 10:05:00 | 99.10 | 98.53 | 0.00 | ORB-long ORB[97.65,98.75] vol=2.5x ATR=0.36 |
| Stop hit — per-position SL triggered | 2023-12-01 10:10:00 | 98.74 | 98.56 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2023-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 09:35:00 | 119.85 | 118.62 | 0.00 | ORB-long ORB[117.60,118.80] vol=2.8x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 09:40:00 | 120.65 | 119.57 | 0.00 | T1 1.5R @ 120.65 |
| Stop hit — per-position SL triggered | 2023-12-12 10:05:00 | 119.85 | 120.17 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2023-12-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 10:10:00 | 118.50 | 119.12 | 0.00 | ORB-short ORB[118.70,119.90] vol=1.8x ATR=0.54 |
| Stop hit — per-position SL triggered | 2023-12-13 10:15:00 | 119.04 | 119.06 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2023-12-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 09:50:00 | 126.25 | 127.06 | 0.00 | ORB-short ORB[126.70,128.20] vol=1.8x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 09:55:00 | 125.15 | 126.93 | 0.00 | T1 1.5R @ 125.15 |
| Target hit | 2023-12-14 15:20:00 | 121.70 | 124.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2023-12-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 10:50:00 | 120.00 | 119.39 | 0.00 | ORB-long ORB[118.45,119.90] vol=2.8x ATR=0.50 |
| Stop hit — per-position SL triggered | 2023-12-22 11:20:00 | 119.50 | 119.47 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2023-12-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 10:10:00 | 123.30 | 123.97 | 0.00 | ORB-short ORB[123.40,124.70] vol=2.5x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 10:35:00 | 122.62 | 123.83 | 0.00 | T1 1.5R @ 122.62 |
| Stop hit — per-position SL triggered | 2023-12-27 11:25:00 | 123.30 | 123.61 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-01-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 09:45:00 | 134.25 | 133.44 | 0.00 | ORB-long ORB[132.50,133.90] vol=2.1x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 12:05:00 | 135.39 | 134.31 | 0.00 | T1 1.5R @ 135.39 |
| Target hit | 2024-01-01 15:20:00 | 136.80 | 135.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2024-01-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:45:00 | 133.10 | 133.48 | 0.00 | ORB-short ORB[133.35,134.60] vol=1.9x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-01-03 10:00:00 | 133.65 | 133.46 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 09:30:00 | 134.65 | 135.34 | 0.00 | ORB-short ORB[134.70,136.25] vol=2.1x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-01-04 11:00:00 | 135.33 | 134.97 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 09:30:00 | 136.60 | 135.74 | 0.00 | ORB-long ORB[134.85,136.55] vol=1.6x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 10:00:00 | 137.37 | 136.57 | 0.00 | T1 1.5R @ 137.37 |
| Target hit | 2024-01-05 10:50:00 | 137.30 | 137.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — SELL (started 2024-01-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 11:00:00 | 135.75 | 137.00 | 0.00 | ORB-short ORB[137.80,139.15] vol=1.9x ATR=0.61 |
| Stop hit — per-position SL triggered | 2024-01-08 11:40:00 | 136.36 | 136.82 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 09:30:00 | 137.95 | 138.94 | 0.00 | ORB-short ORB[138.80,140.00] vol=3.3x ATR=0.54 |
| Stop hit — per-position SL triggered | 2024-01-12 09:45:00 | 138.49 | 138.73 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-01-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 09:30:00 | 141.15 | 140.15 | 0.00 | ORB-long ORB[139.15,140.80] vol=2.9x ATR=0.53 |
| Stop hit — per-position SL triggered | 2024-01-16 09:40:00 | 140.62 | 140.36 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-01-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:35:00 | 136.40 | 138.02 | 0.00 | ORB-short ORB[137.10,139.10] vol=2.2x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:45:00 | 135.13 | 137.02 | 0.00 | T1 1.5R @ 135.13 |
| Stop hit — per-position SL triggered | 2024-01-18 10:05:00 | 136.40 | 136.11 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-01-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 09:50:00 | 143.45 | 142.45 | 0.00 | ORB-long ORB[141.05,142.85] vol=2.1x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 12:50:00 | 144.81 | 143.59 | 0.00 | T1 1.5R @ 144.81 |
| Stop hit — per-position SL triggered | 2024-01-19 14:30:00 | 143.45 | 143.80 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-02-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 09:40:00 | 142.90 | 141.82 | 0.00 | ORB-long ORB[140.50,142.60] vol=2.6x ATR=0.73 |
| Stop hit — per-position SL triggered | 2024-02-08 09:45:00 | 142.17 | 141.90 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-16 09:30:00 | 132.80 | 133.72 | 0.00 | ORB-short ORB[133.20,134.80] vol=1.6x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 09:40:00 | 131.99 | 133.37 | 0.00 | T1 1.5R @ 131.99 |
| Target hit | 2024-02-16 13:45:00 | 132.65 | 132.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 80 — BUY (started 2024-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 09:30:00 | 132.85 | 132.18 | 0.00 | ORB-long ORB[131.40,132.80] vol=1.9x ATR=0.46 |
| Stop hit — per-position SL triggered | 2024-02-19 09:40:00 | 132.39 | 132.10 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 09:50:00 | 128.15 | 128.93 | 0.00 | ORB-short ORB[128.55,129.85] vol=1.8x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 09:55:00 | 127.38 | 128.85 | 0.00 | T1 1.5R @ 127.38 |
| Stop hit — per-position SL triggered | 2024-02-26 11:50:00 | 128.15 | 128.19 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-02-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:55:00 | 127.10 | 129.75 | 0.00 | ORB-short ORB[130.85,132.10] vol=1.8x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-02-28 11:00:00 | 127.54 | 129.56 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 09:45:00 | 127.70 | 128.87 | 0.00 | ORB-short ORB[128.60,130.40] vol=1.9x ATR=0.51 |
| Stop hit — per-position SL triggered | 2024-03-05 09:50:00 | 128.21 | 128.83 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-03-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 10:10:00 | 120.00 | 119.43 | 0.00 | ORB-long ORB[118.05,119.80] vol=3.6x ATR=0.53 |
| Stop hit — per-position SL triggered | 2024-03-26 10:20:00 | 119.47 | 119.49 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-04-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 09:55:00 | 128.25 | 127.71 | 0.00 | ORB-long ORB[126.70,128.20] vol=1.8x ATR=0.52 |
| Stop hit — per-position SL triggered | 2024-04-03 10:20:00 | 127.73 | 127.77 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 11:15:00 | 141.40 | 143.39 | 0.00 | ORB-short ORB[142.35,144.10] vol=3.5x ATR=0.48 |
| Stop hit — per-position SL triggered | 2024-04-10 11:40:00 | 141.88 | 143.26 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-05-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 09:35:00 | 149.80 | 148.86 | 0.00 | ORB-long ORB[146.95,148.40] vol=4.1x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 09:50:00 | 150.92 | 150.12 | 0.00 | T1 1.5R @ 150.92 |
| Target hit | 2024-05-02 13:20:00 | 153.35 | 153.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 88 — BUY (started 2024-05-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 09:55:00 | 155.10 | 153.64 | 0.00 | ORB-long ORB[152.05,154.20] vol=3.0x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 10:00:00 | 156.22 | 154.10 | 0.00 | T1 1.5R @ 156.22 |
| Stop hit — per-position SL triggered | 2024-05-03 10:35:00 | 155.10 | 155.53 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-05-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 09:45:00 | 142.20 | 143.08 | 0.00 | ORB-short ORB[142.50,144.25] vol=2.1x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:10:00 | 141.25 | 142.62 | 0.00 | T1 1.5R @ 141.25 |
| Stop hit — per-position SL triggered | 2024-05-09 11:15:00 | 142.20 | 141.98 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 09:45:00 | 71.00 | 2023-05-15 09:50:00 | 70.81 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-05-17 09:35:00 | 71.45 | 2023-05-17 09:45:00 | 71.89 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2023-05-17 09:35:00 | 71.45 | 2023-05-17 10:00:00 | 71.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-18 09:35:00 | 70.85 | 2023-05-18 09:40:00 | 70.61 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-05-22 11:00:00 | 72.15 | 2023-05-22 11:15:00 | 72.40 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-05-22 11:00:00 | 72.15 | 2023-05-22 11:30:00 | 72.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-26 09:40:00 | 69.75 | 2023-05-26 09:45:00 | 69.91 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-05-29 09:45:00 | 70.60 | 2023-05-29 09:55:00 | 70.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-05-30 09:50:00 | 69.40 | 2023-05-30 15:20:00 | 69.30 | TARGET_HIT | 1.00 | 0.14% |
| BUY | retest1 | 2023-05-31 09:35:00 | 69.45 | 2023-05-31 09:55:00 | 69.30 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-06-05 10:55:00 | 70.75 | 2023-06-05 11:05:00 | 70.97 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-06-05 10:55:00 | 70.75 | 2023-06-05 11:25:00 | 70.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-08 09:35:00 | 72.45 | 2023-06-08 09:45:00 | 72.24 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-06-09 11:05:00 | 71.95 | 2023-06-09 11:20:00 | 71.77 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-06-13 10:20:00 | 72.20 | 2023-06-13 10:25:00 | 72.01 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-06-14 10:00:00 | 71.80 | 2023-06-14 12:25:00 | 71.54 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-06-14 10:00:00 | 71.80 | 2023-06-14 13:15:00 | 71.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-19 11:05:00 | 71.80 | 2023-06-19 12:35:00 | 71.98 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-06-20 09:55:00 | 72.50 | 2023-06-20 10:25:00 | 72.30 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-06-21 10:15:00 | 72.90 | 2023-06-21 10:20:00 | 73.07 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-06-27 11:10:00 | 71.90 | 2023-06-27 11:25:00 | 72.02 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-06-28 09:35:00 | 71.75 | 2023-06-28 11:25:00 | 71.92 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-03 10:35:00 | 72.50 | 2023-07-03 15:05:00 | 72.73 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-07-03 10:35:00 | 72.50 | 2023-07-03 15:10:00 | 72.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-04 11:05:00 | 73.35 | 2023-07-04 11:35:00 | 73.20 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-06 09:40:00 | 74.45 | 2023-07-06 10:15:00 | 74.21 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-07-10 10:50:00 | 75.70 | 2023-07-10 10:55:00 | 75.45 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-07-11 09:40:00 | 75.50 | 2023-07-11 09:45:00 | 75.32 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-07-12 09:35:00 | 74.60 | 2023-07-12 09:50:00 | 74.31 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-07-12 09:35:00 | 74.60 | 2023-07-12 09:55:00 | 74.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-14 10:30:00 | 75.90 | 2023-07-14 11:45:00 | 75.68 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-07-17 11:15:00 | 75.40 | 2023-07-17 11:40:00 | 75.08 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-07-17 11:15:00 | 75.40 | 2023-07-17 13:45:00 | 75.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-25 09:50:00 | 75.50 | 2023-07-25 10:25:00 | 75.33 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-07-28 10:15:00 | 75.90 | 2023-07-28 10:20:00 | 76.12 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-31 09:30:00 | 76.75 | 2023-07-31 09:40:00 | 77.15 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-07-31 09:30:00 | 76.75 | 2023-07-31 15:20:00 | 78.00 | TARGET_HIT | 0.50 | 1.63% |
| BUY | retest1 | 2023-08-03 09:45:00 | 78.15 | 2023-08-03 09:50:00 | 77.78 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2023-08-04 10:10:00 | 79.70 | 2023-08-04 10:15:00 | 80.19 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2023-08-04 10:10:00 | 79.70 | 2023-08-04 10:20:00 | 79.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-08 10:45:00 | 77.60 | 2023-08-08 11:00:00 | 77.80 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-08-09 11:10:00 | 78.00 | 2023-08-09 11:40:00 | 78.18 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-08-11 09:55:00 | 79.15 | 2023-08-11 10:00:00 | 78.87 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-08-16 09:45:00 | 78.25 | 2023-08-16 10:15:00 | 78.60 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-08-16 09:45:00 | 78.25 | 2023-08-16 11:45:00 | 78.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-17 10:00:00 | 78.60 | 2023-08-17 10:20:00 | 78.40 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-08-23 09:30:00 | 79.55 | 2023-08-23 09:35:00 | 79.96 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-08-23 09:30:00 | 79.55 | 2023-08-23 15:20:00 | 83.80 | TARGET_HIT | 0.50 | 5.34% |
| BUY | retest1 | 2023-08-24 09:35:00 | 84.95 | 2023-08-24 09:50:00 | 84.54 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2023-08-30 09:30:00 | 84.15 | 2023-08-30 09:35:00 | 84.52 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-08-30 09:30:00 | 84.15 | 2023-08-30 10:05:00 | 84.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-04 11:10:00 | 89.25 | 2023-09-04 11:15:00 | 88.77 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2023-09-07 10:30:00 | 93.00 | 2023-09-07 10:35:00 | 92.48 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2023-09-08 09:45:00 | 92.05 | 2023-09-08 09:55:00 | 91.63 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-09-18 09:35:00 | 90.45 | 2023-09-18 09:50:00 | 91.08 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2023-09-18 09:35:00 | 90.45 | 2023-09-18 11:30:00 | 91.05 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2023-09-25 09:35:00 | 92.25 | 2023-09-25 09:50:00 | 91.83 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-09-26 10:00:00 | 91.50 | 2023-09-26 10:05:00 | 91.18 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-09-27 10:35:00 | 89.90 | 2023-09-27 11:00:00 | 90.15 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-10-03 09:55:00 | 90.50 | 2023-10-03 10:00:00 | 90.20 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-10-04 09:35:00 | 90.20 | 2023-10-04 10:15:00 | 89.76 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-10-04 09:35:00 | 90.20 | 2023-10-04 14:20:00 | 88.90 | TARGET_HIT | 0.50 | 1.44% |
| BUY | retest1 | 2023-10-11 10:20:00 | 89.65 | 2023-10-11 10:25:00 | 89.31 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-10-17 09:30:00 | 91.00 | 2023-10-17 09:35:00 | 91.60 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2023-10-17 09:30:00 | 91.00 | 2023-10-17 10:15:00 | 91.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-18 11:10:00 | 91.95 | 2023-10-18 11:15:00 | 92.32 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-11-01 09:55:00 | 85.20 | 2023-11-01 10:10:00 | 84.91 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-11-02 09:50:00 | 85.40 | 2023-11-02 10:00:00 | 85.15 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-11-08 10:00:00 | 88.00 | 2023-11-08 10:05:00 | 87.75 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-11-13 09:35:00 | 90.25 | 2023-11-13 09:50:00 | 90.72 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-11-13 09:35:00 | 90.25 | 2023-11-13 10:15:00 | 90.35 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2023-11-17 11:00:00 | 95.80 | 2023-11-17 11:20:00 | 96.13 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-11-20 09:35:00 | 99.05 | 2023-11-20 09:40:00 | 99.80 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2023-11-20 09:35:00 | 99.05 | 2023-11-20 09:55:00 | 99.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-23 09:55:00 | 99.45 | 2023-11-23 10:00:00 | 99.91 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2023-11-24 10:45:00 | 98.40 | 2023-11-24 12:20:00 | 97.93 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-11-24 10:45:00 | 98.40 | 2023-11-24 12:50:00 | 98.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-28 09:35:00 | 97.35 | 2023-11-28 09:40:00 | 97.65 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-11-29 09:30:00 | 99.75 | 2023-11-29 09:35:00 | 99.26 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2023-11-30 09:35:00 | 97.80 | 2023-11-30 09:50:00 | 97.23 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2023-11-30 09:35:00 | 97.80 | 2023-11-30 15:20:00 | 97.10 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2023-12-01 10:05:00 | 99.10 | 2023-12-01 10:10:00 | 98.74 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-12-12 09:35:00 | 119.85 | 2023-12-12 09:40:00 | 120.65 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2023-12-12 09:35:00 | 119.85 | 2023-12-12 10:05:00 | 119.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-13 10:10:00 | 118.50 | 2023-12-13 10:15:00 | 119.04 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2023-12-14 09:50:00 | 126.25 | 2023-12-14 09:55:00 | 125.15 | PARTIAL | 0.50 | 0.87% |
| SELL | retest1 | 2023-12-14 09:50:00 | 126.25 | 2023-12-14 15:20:00 | 121.70 | TARGET_HIT | 0.50 | 3.60% |
| BUY | retest1 | 2023-12-22 10:50:00 | 120.00 | 2023-12-22 11:20:00 | 119.50 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-12-27 10:10:00 | 123.30 | 2023-12-27 10:35:00 | 122.62 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2023-12-27 10:10:00 | 123.30 | 2023-12-27 11:25:00 | 123.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-01 09:45:00 | 134.25 | 2024-01-01 12:05:00 | 135.39 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2024-01-01 09:45:00 | 134.25 | 2024-01-01 15:20:00 | 136.80 | TARGET_HIT | 0.50 | 1.90% |
| SELL | retest1 | 2024-01-03 09:45:00 | 133.10 | 2024-01-03 10:00:00 | 133.65 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-01-04 09:30:00 | 134.65 | 2024-01-04 11:00:00 | 135.33 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-01-05 09:30:00 | 136.60 | 2024-01-05 10:00:00 | 137.37 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-01-05 09:30:00 | 136.60 | 2024-01-05 10:50:00 | 137.30 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2024-01-08 11:00:00 | 135.75 | 2024-01-08 11:40:00 | 136.36 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-01-12 09:30:00 | 137.95 | 2024-01-12 09:45:00 | 138.49 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-01-16 09:30:00 | 141.15 | 2024-01-16 09:40:00 | 140.62 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-01-18 09:35:00 | 136.40 | 2024-01-18 09:45:00 | 135.13 | PARTIAL | 0.50 | 0.93% |
| SELL | retest1 | 2024-01-18 09:35:00 | 136.40 | 2024-01-18 10:05:00 | 136.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-19 09:50:00 | 143.45 | 2024-01-19 12:50:00 | 144.81 | PARTIAL | 0.50 | 0.95% |
| BUY | retest1 | 2024-01-19 09:50:00 | 143.45 | 2024-01-19 14:30:00 | 143.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-08 09:40:00 | 142.90 | 2024-02-08 09:45:00 | 142.17 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-02-16 09:30:00 | 132.80 | 2024-02-16 09:40:00 | 131.99 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-02-16 09:30:00 | 132.80 | 2024-02-16 13:45:00 | 132.65 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2024-02-19 09:30:00 | 132.85 | 2024-02-19 09:40:00 | 132.39 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-02-26 09:50:00 | 128.15 | 2024-02-26 09:55:00 | 127.38 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-02-26 09:50:00 | 128.15 | 2024-02-26 11:50:00 | 128.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-28 10:55:00 | 127.10 | 2024-02-28 11:00:00 | 127.54 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-03-05 09:45:00 | 127.70 | 2024-03-05 09:50:00 | 128.21 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-03-26 10:10:00 | 120.00 | 2024-03-26 10:20:00 | 119.47 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-04-03 09:55:00 | 128.25 | 2024-04-03 10:20:00 | 127.73 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-04-10 11:15:00 | 141.40 | 2024-04-10 11:40:00 | 141.88 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-05-02 09:35:00 | 149.80 | 2024-05-02 09:50:00 | 150.92 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-05-02 09:35:00 | 149.80 | 2024-05-02 13:20:00 | 153.35 | TARGET_HIT | 0.50 | 2.37% |
| BUY | retest1 | 2024-05-03 09:55:00 | 155.10 | 2024-05-03 10:00:00 | 156.22 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-05-03 09:55:00 | 155.10 | 2024-05-03 10:35:00 | 155.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-09 09:45:00 | 142.20 | 2024-05-09 10:10:00 | 141.25 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-05-09 09:45:00 | 142.20 | 2024-05-09 11:15:00 | 142.20 | STOP_HIT | 0.50 | 0.00% |
