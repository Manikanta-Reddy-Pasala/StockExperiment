# Canara Bank (CANBK)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-12-31 15:25:00 (29018 bars)
- **Last close:** 100.06
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
| ENTRY1 | 130 |
| ENTRY2 | 0 |
| PARTIAL | 49 |
| TARGET_HIT | 20 |
| STOP_HIT | 110 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 179 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 69 / 110
- **Target hits / Stop hits / Partials:** 20 / 110 / 49
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 17.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 99 | 39 | 39.4% | 11 | 60 | 28 | 0.06% | 6.1% |
| BUY @ 2nd Alert (retest1) | 99 | 39 | 39.4% | 11 | 60 | 28 | 0.06% | 6.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 80 | 30 | 37.5% | 9 | 50 | 21 | 0.14% | 11.1% |
| SELL @ 2nd Alert (retest1) | 80 | 30 | 37.5% | 9 | 50 | 21 | 0.14% | 11.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 179 | 69 | 38.5% | 20 | 110 | 49 | 0.10% | 17.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 11:15:00 | 60.19 | 59.78 | 0.00 | ORB-long ORB[59.12,59.90] vol=2.3x ATR=0.15 |
| Stop hit — per-position SL triggered | 2023-05-15 12:05:00 | 60.04 | 59.87 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 09:30:00 | 61.63 | 61.37 | 0.00 | ORB-long ORB[60.88,61.57] vol=2.2x ATR=0.21 |
| Stop hit — per-position SL triggered | 2023-05-16 09:35:00 | 61.42 | 61.45 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 11:00:00 | 60.69 | 61.21 | 0.00 | ORB-short ORB[60.90,61.57] vol=1.8x ATR=0.17 |
| Stop hit — per-position SL triggered | 2023-05-17 11:05:00 | 60.86 | 61.19 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 10:55:00 | 60.55 | 60.71 | 0.00 | ORB-short ORB[60.57,61.10] vol=3.3x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-18 11:45:00 | 60.32 | 60.66 | 0.00 | T1 1.5R @ 60.32 |
| Target hit | 2023-05-18 15:20:00 | 59.11 | 59.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2023-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:35:00 | 58.55 | 59.10 | 0.00 | ORB-short ORB[59.08,59.53] vol=1.5x ATR=0.27 |
| Stop hit — per-position SL triggered | 2023-05-19 10:10:00 | 58.82 | 58.83 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 10:00:00 | 60.76 | 60.49 | 0.00 | ORB-long ORB[60.23,60.58] vol=2.1x ATR=0.14 |
| Stop hit — per-position SL triggered | 2023-05-23 10:55:00 | 60.62 | 60.59 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 09:30:00 | 60.89 | 60.62 | 0.00 | ORB-long ORB[60.03,60.75] vol=4.8x ATR=0.18 |
| Stop hit — per-position SL triggered | 2023-05-26 09:35:00 | 60.71 | 60.66 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-05-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-29 10:05:00 | 61.51 | 61.68 | 0.00 | ORB-short ORB[61.60,61.94] vol=1.7x ATR=0.13 |
| Stop hit — per-position SL triggered | 2023-05-29 10:10:00 | 61.64 | 61.68 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-05-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 11:00:00 | 61.22 | 61.50 | 0.00 | ORB-short ORB[61.26,61.74] vol=2.2x ATR=0.12 |
| Stop hit — per-position SL triggered | 2023-05-30 12:30:00 | 61.34 | 61.40 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-05-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 10:50:00 | 61.72 | 61.27 | 0.00 | ORB-long ORB[60.92,61.48] vol=2.3x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 11:00:00 | 61.91 | 61.57 | 0.00 | T1 1.5R @ 61.91 |
| Stop hit — per-position SL triggered | 2023-05-31 11:45:00 | 61.72 | 61.70 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:30:00 | 62.55 | 62.33 | 0.00 | ORB-long ORB[61.96,62.38] vol=3.1x ATR=0.14 |
| Stop hit — per-position SL triggered | 2023-06-02 10:30:00 | 62.41 | 62.49 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 10:30:00 | 62.66 | 62.93 | 0.00 | ORB-short ORB[62.93,63.17] vol=2.0x ATR=0.13 |
| Stop hit — per-position SL triggered | 2023-06-05 10:45:00 | 62.79 | 62.92 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 11:05:00 | 62.00 | 62.28 | 0.00 | ORB-short ORB[62.40,62.68] vol=3.5x ATR=0.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-06 11:20:00 | 61.84 | 62.21 | 0.00 | T1 1.5R @ 61.84 |
| Stop hit — per-position SL triggered | 2023-06-06 11:25:00 | 62.00 | 62.20 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 09:35:00 | 62.85 | 62.74 | 0.00 | ORB-long ORB[62.52,62.76] vol=3.3x ATR=0.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 09:55:00 | 63.02 | 62.80 | 0.00 | T1 1.5R @ 63.02 |
| Target hit | 2023-06-07 14:35:00 | 63.00 | 63.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2023-06-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 09:55:00 | 62.88 | 63.07 | 0.00 | ORB-short ORB[62.94,63.31] vol=1.7x ATR=0.11 |
| Stop hit — per-position SL triggered | 2023-06-08 10:00:00 | 62.99 | 63.06 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:30:00 | 62.16 | 62.34 | 0.00 | ORB-short ORB[62.22,62.65] vol=1.8x ATR=0.13 |
| Stop hit — per-position SL triggered | 2023-06-09 09:40:00 | 62.29 | 62.30 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-06-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 10:05:00 | 62.82 | 62.97 | 0.00 | ORB-short ORB[62.93,63.28] vol=1.7x ATR=0.11 |
| Stop hit — per-position SL triggered | 2023-06-13 10:25:00 | 62.93 | 62.95 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-06-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 11:05:00 | 60.66 | 61.08 | 0.00 | ORB-short ORB[60.70,61.40] vol=3.3x ATR=0.14 |
| Stop hit — per-position SL triggered | 2023-06-14 11:15:00 | 60.80 | 61.06 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-06-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-15 09:35:00 | 60.58 | 60.75 | 0.00 | ORB-short ORB[60.64,61.07] vol=1.5x ATR=0.11 |
| Stop hit — per-position SL triggered | 2023-06-15 09:45:00 | 60.69 | 60.71 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-06-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-19 09:45:00 | 61.27 | 60.87 | 0.00 | ORB-long ORB[60.51,61.02] vol=2.7x ATR=0.18 |
| Stop hit — per-position SL triggered | 2023-06-19 09:55:00 | 61.09 | 60.91 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-06-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 10:25:00 | 60.98 | 61.13 | 0.00 | ORB-short ORB[61.08,61.40] vol=1.8x ATR=0.10 |
| Stop hit — per-position SL triggered | 2023-06-21 10:45:00 | 61.08 | 61.11 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-06-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-26 10:35:00 | 58.83 | 59.10 | 0.00 | ORB-short ORB[59.00,59.47] vol=1.9x ATR=0.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 11:45:00 | 58.63 | 58.98 | 0.00 | T1 1.5R @ 58.63 |
| Target hit | 2023-06-26 14:55:00 | 58.77 | 58.70 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — BUY (started 2023-06-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 09:30:00 | 60.16 | 59.83 | 0.00 | ORB-long ORB[59.50,59.90] vol=1.7x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-30 09:35:00 | 60.43 | 60.06 | 0.00 | T1 1.5R @ 60.43 |
| Stop hit — per-position SL triggered | 2023-06-30 10:20:00 | 60.16 | 60.20 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 09:40:00 | 61.01 | 60.80 | 0.00 | ORB-long ORB[60.51,60.91] vol=2.8x ATR=0.14 |
| Stop hit — per-position SL triggered | 2023-07-03 09:45:00 | 60.87 | 60.81 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 10:15:00 | 64.78 | 64.14 | 0.00 | ORB-long ORB[63.65,64.58] vol=1.6x ATR=0.35 |
| Stop hit — per-position SL triggered | 2023-07-04 10:35:00 | 64.43 | 64.34 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-07-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 10:45:00 | 64.76 | 64.39 | 0.00 | ORB-long ORB[64.12,64.70] vol=2.5x ATR=0.21 |
| Stop hit — per-position SL triggered | 2023-07-05 10:50:00 | 64.55 | 64.40 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-07-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 11:15:00 | 65.03 | 64.71 | 0.00 | ORB-long ORB[64.05,64.83] vol=5.5x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 11:20:00 | 65.25 | 64.87 | 0.00 | T1 1.5R @ 65.25 |
| Target hit | 2023-07-06 12:40:00 | 65.20 | 65.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2023-07-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 09:30:00 | 65.59 | 65.13 | 0.00 | ORB-long ORB[64.57,65.34] vol=2.1x ATR=0.21 |
| Stop hit — per-position SL triggered | 2023-07-07 09:45:00 | 65.38 | 65.31 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-07-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-11 09:55:00 | 66.09 | 66.70 | 0.00 | ORB-short ORB[66.74,67.29] vol=2.0x ATR=0.23 |
| Stop hit — per-position SL triggered | 2023-07-11 10:00:00 | 66.32 | 66.66 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 09:30:00 | 66.91 | 66.65 | 0.00 | ORB-long ORB[66.24,66.84] vol=3.1x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 10:55:00 | 67.21 | 66.84 | 0.00 | T1 1.5R @ 67.21 |
| Stop hit — per-position SL triggered | 2023-07-12 11:30:00 | 66.91 | 66.88 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-07-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 09:30:00 | 66.00 | 66.29 | 0.00 | ORB-short ORB[66.20,67.00] vol=4.7x ATR=0.19 |
| Stop hit — per-position SL triggered | 2023-07-13 09:35:00 | 66.19 | 66.28 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 09:40:00 | 65.44 | 65.00 | 0.00 | ORB-long ORB[64.40,64.99] vol=1.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2023-07-14 09:55:00 | 65.16 | 65.06 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 09:35:00 | 66.18 | 66.49 | 0.00 | ORB-short ORB[66.40,67.14] vol=1.6x ATR=0.24 |
| Stop hit — per-position SL triggered | 2023-07-18 10:15:00 | 66.42 | 66.37 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-07-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:50:00 | 66.51 | 66.05 | 0.00 | ORB-long ORB[65.69,66.19] vol=3.1x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 10:15:00 | 66.86 | 66.19 | 0.00 | T1 1.5R @ 66.86 |
| Stop hit — per-position SL triggered | 2023-07-19 10:50:00 | 66.51 | 66.31 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-07-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 09:45:00 | 68.57 | 68.32 | 0.00 | ORB-long ORB[67.80,68.49] vol=1.5x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-07-20 10:05:00 | 68.31 | 68.34 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-07-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 09:30:00 | 68.20 | 68.08 | 0.00 | ORB-long ORB[67.70,68.18] vol=3.0x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 10:00:00 | 68.53 | 68.17 | 0.00 | T1 1.5R @ 68.53 |
| Target hit | 2023-07-27 11:15:00 | 69.00 | 69.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2023-08-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 10:30:00 | 68.50 | 68.91 | 0.00 | ORB-short ORB[68.90,69.20] vol=1.5x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 12:05:00 | 68.29 | 68.68 | 0.00 | T1 1.5R @ 68.29 |
| Target hit | 2023-08-01 14:35:00 | 68.36 | 68.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — SELL (started 2023-08-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 11:00:00 | 67.10 | 67.70 | 0.00 | ORB-short ORB[67.52,68.00] vol=1.5x ATR=0.19 |
| Stop hit — per-position SL triggered | 2023-08-02 11:20:00 | 67.29 | 67.58 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-08-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-07 09:40:00 | 66.36 | 65.93 | 0.00 | ORB-long ORB[65.60,66.09] vol=1.6x ATR=0.25 |
| Stop hit — per-position SL triggered | 2023-08-07 10:00:00 | 66.11 | 66.00 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-08-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 10:10:00 | 66.76 | 66.40 | 0.00 | ORB-long ORB[65.80,66.33] vol=3.2x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 10:25:00 | 67.04 | 66.51 | 0.00 | T1 1.5R @ 67.04 |
| Stop hit — per-position SL triggered | 2023-08-08 10:45:00 | 66.76 | 66.71 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-08-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:20:00 | 66.63 | 66.96 | 0.00 | ORB-short ORB[66.85,67.34] vol=2.2x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-08-10 10:55:00 | 66.89 | 66.83 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-08-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 09:40:00 | 66.69 | 66.33 | 0.00 | ORB-long ORB[65.94,66.53] vol=1.6x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 09:55:00 | 66.99 | 66.48 | 0.00 | T1 1.5R @ 66.99 |
| Target hit | 2023-08-11 12:30:00 | 66.87 | 67.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — BUY (started 2023-08-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 09:45:00 | 66.08 | 65.83 | 0.00 | ORB-long ORB[65.32,66.02] vol=1.5x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 09:55:00 | 66.32 | 65.93 | 0.00 | T1 1.5R @ 66.32 |
| Stop hit — per-position SL triggered | 2023-08-17 10:30:00 | 66.08 | 66.15 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-08-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 09:45:00 | 66.62 | 66.32 | 0.00 | ORB-long ORB[65.72,66.55] vol=1.8x ATR=0.19 |
| Stop hit — per-position SL triggered | 2023-08-18 10:05:00 | 66.43 | 66.41 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-21 09:45:00 | 65.68 | 65.87 | 0.00 | ORB-short ORB[65.70,66.13] vol=3.0x ATR=0.17 |
| Stop hit — per-position SL triggered | 2023-08-21 09:55:00 | 65.85 | 65.85 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-08-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 11:10:00 | 65.60 | 65.67 | 0.00 | ORB-short ORB[65.61,65.90] vol=2.8x ATR=0.09 |
| Stop hit — per-position SL triggered | 2023-08-22 13:00:00 | 65.69 | 65.65 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-08-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-23 10:50:00 | 65.35 | 65.44 | 0.00 | ORB-short ORB[65.36,65.60] vol=2.9x ATR=0.11 |
| Stop hit — per-position SL triggered | 2023-08-23 10:55:00 | 65.46 | 65.44 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-08-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:20:00 | 65.51 | 65.99 | 0.00 | ORB-short ORB[65.69,66.37] vol=2.2x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 10:30:00 | 65.23 | 65.86 | 0.00 | T1 1.5R @ 65.23 |
| Stop hit — per-position SL triggered | 2023-08-25 10:40:00 | 65.51 | 65.79 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-08-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 09:35:00 | 65.41 | 65.18 | 0.00 | ORB-long ORB[64.94,65.30] vol=1.8x ATR=0.15 |
| Stop hit — per-position SL triggered | 2023-08-31 10:05:00 | 65.26 | 65.23 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-09-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 10:55:00 | 64.78 | 64.42 | 0.00 | ORB-long ORB[63.94,64.55] vol=1.7x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 11:05:00 | 65.03 | 64.49 | 0.00 | T1 1.5R @ 65.03 |
| Target hit | 2023-09-01 15:20:00 | 65.70 | 65.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2023-09-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 10:55:00 | 66.23 | 65.95 | 0.00 | ORB-long ORB[65.71,66.14] vol=1.9x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 11:00:00 | 66.51 | 65.99 | 0.00 | T1 1.5R @ 66.51 |
| Target hit | 2023-09-04 15:20:00 | 67.29 | 66.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2023-09-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 11:05:00 | 67.30 | 67.68 | 0.00 | ORB-short ORB[67.35,68.26] vol=1.6x ATR=0.19 |
| Stop hit — per-position SL triggered | 2023-09-05 11:45:00 | 67.49 | 67.63 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-09-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-06 11:10:00 | 67.22 | 67.38 | 0.00 | ORB-short ORB[67.26,67.85] vol=2.0x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 11:30:00 | 66.98 | 67.31 | 0.00 | T1 1.5R @ 66.98 |
| Target hit | 2023-09-06 15:20:00 | 66.80 | 66.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2023-09-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 09:55:00 | 67.38 | 67.10 | 0.00 | ORB-long ORB[66.72,67.18] vol=2.3x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 10:45:00 | 67.61 | 67.28 | 0.00 | T1 1.5R @ 67.61 |
| Stop hit — per-position SL triggered | 2023-09-07 11:30:00 | 67.38 | 67.37 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-09-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 09:50:00 | 67.54 | 67.75 | 0.00 | ORB-short ORB[67.66,68.13] vol=1.5x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 10:20:00 | 67.28 | 67.63 | 0.00 | T1 1.5R @ 67.28 |
| Stop hit — per-position SL triggered | 2023-09-08 10:45:00 | 67.54 | 67.58 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 09:30:00 | 69.81 | 69.05 | 0.00 | ORB-long ORB[68.06,69.05] vol=4.1x ATR=0.29 |
| Stop hit — per-position SL triggered | 2023-09-11 09:40:00 | 69.52 | 69.45 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-09-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:35:00 | 73.68 | 73.17 | 0.00 | ORB-long ORB[72.56,73.40] vol=1.5x ATR=0.35 |
| Stop hit — per-position SL triggered | 2023-09-14 09:45:00 | 73.33 | 73.22 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-09-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-20 10:40:00 | 75.13 | 74.32 | 0.00 | ORB-long ORB[73.39,74.47] vol=1.8x ATR=0.31 |
| Stop hit — per-position SL triggered | 2023-09-20 10:50:00 | 74.82 | 74.37 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-09-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 10:05:00 | 73.31 | 73.81 | 0.00 | ORB-short ORB[73.51,74.34] vol=2.0x ATR=0.28 |
| Stop hit — per-position SL triggered | 2023-09-27 10:50:00 | 73.59 | 73.66 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-09-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 09:45:00 | 75.77 | 75.20 | 0.00 | ORB-long ORB[74.80,75.36] vol=1.9x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-09-28 10:00:00 | 75.51 | 75.33 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2023-09-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-29 10:50:00 | 74.62 | 75.28 | 0.00 | ORB-short ORB[75.01,75.62] vol=1.9x ATR=0.24 |
| Stop hit — per-position SL triggered | 2023-09-29 11:05:00 | 74.86 | 75.24 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-10-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 09:45:00 | 75.67 | 75.23 | 0.00 | ORB-long ORB[74.82,75.47] vol=1.9x ATR=0.27 |
| Stop hit — per-position SL triggered | 2023-10-03 10:20:00 | 75.40 | 75.40 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-10-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 10:20:00 | 75.22 | 74.67 | 0.00 | ORB-long ORB[74.20,74.72] vol=2.1x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 10:40:00 | 75.57 | 74.85 | 0.00 | T1 1.5R @ 75.57 |
| Stop hit — per-position SL triggered | 2023-10-06 11:15:00 | 75.22 | 75.21 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2023-10-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-11 09:55:00 | 74.36 | 74.61 | 0.00 | ORB-short ORB[74.41,75.00] vol=1.9x ATR=0.18 |
| Stop hit — per-position SL triggered | 2023-10-11 10:00:00 | 74.54 | 74.59 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2023-10-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 09:40:00 | 72.40 | 73.16 | 0.00 | ORB-short ORB[73.12,73.87] vol=4.3x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 09:45:00 | 72.02 | 72.87 | 0.00 | T1 1.5R @ 72.02 |
| Stop hit — per-position SL triggered | 2023-10-13 09:50:00 | 72.40 | 72.80 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 10:15:00 | 75.46 | 75.23 | 0.00 | ORB-long ORB[74.30,75.40] vol=2.3x ATR=0.19 |
| Stop hit — per-position SL triggered | 2023-10-17 13:35:00 | 75.27 | 75.37 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2023-10-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:40:00 | 74.88 | 75.29 | 0.00 | ORB-short ORB[75.21,75.66] vol=1.6x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 10:45:00 | 74.63 | 75.24 | 0.00 | T1 1.5R @ 74.63 |
| Stop hit — per-position SL triggered | 2023-10-18 10:55:00 | 74.88 | 75.16 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2023-10-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 11:05:00 | 74.68 | 74.27 | 0.00 | ORB-long ORB[73.42,74.52] vol=1.5x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 11:20:00 | 74.97 | 74.34 | 0.00 | T1 1.5R @ 74.97 |
| Stop hit — per-position SL triggered | 2023-10-19 13:20:00 | 74.68 | 74.62 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2023-10-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:10:00 | 73.10 | 73.57 | 0.00 | ORB-short ORB[73.64,74.29] vol=2.4x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-10-23 10:25:00 | 73.36 | 73.47 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2023-10-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 09:30:00 | 69.17 | 69.64 | 0.00 | ORB-short ORB[69.40,70.36] vol=1.8x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:40:00 | 68.68 | 69.23 | 0.00 | T1 1.5R @ 68.68 |
| Target hit | 2023-10-26 10:00:00 | 69.11 | 69.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 71 — SELL (started 2023-11-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 09:50:00 | 76.28 | 76.52 | 0.00 | ORB-short ORB[76.34,76.86] vol=4.1x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-11-01 10:05:00 | 76.54 | 76.50 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2023-11-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 09:40:00 | 78.12 | 77.75 | 0.00 | ORB-long ORB[77.28,77.99] vol=3.1x ATR=0.27 |
| Stop hit — per-position SL triggered | 2023-11-02 10:10:00 | 77.85 | 77.92 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2023-11-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 10:10:00 | 78.36 | 78.16 | 0.00 | ORB-long ORB[77.85,78.32] vol=1.7x ATR=0.20 |
| Stop hit — per-position SL triggered | 2023-11-03 10:30:00 | 78.16 | 78.17 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2023-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 09:30:00 | 77.36 | 77.82 | 0.00 | ORB-short ORB[77.70,78.22] vol=2.0x ATR=0.18 |
| Stop hit — per-position SL triggered | 2023-11-06 09:40:00 | 77.54 | 77.72 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2023-11-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 09:55:00 | 77.36 | 77.05 | 0.00 | ORB-long ORB[76.50,77.16] vol=2.2x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 10:10:00 | 77.66 | 77.13 | 0.00 | T1 1.5R @ 77.66 |
| Stop hit — per-position SL triggered | 2023-11-07 11:20:00 | 77.36 | 77.34 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2023-11-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 10:55:00 | 77.42 | 77.34 | 0.00 | ORB-long ORB[77.04,77.35] vol=1.7x ATR=0.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 11:10:00 | 77.64 | 77.41 | 0.00 | T1 1.5R @ 77.64 |
| Stop hit — per-position SL triggered | 2023-11-09 12:45:00 | 77.42 | 77.50 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2023-11-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 10:00:00 | 77.91 | 77.70 | 0.00 | ORB-long ORB[77.39,77.73] vol=2.1x ATR=0.15 |
| Stop hit — per-position SL triggered | 2023-11-13 10:10:00 | 77.76 | 77.72 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2023-11-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 11:00:00 | 81.96 | 81.66 | 0.00 | ORB-long ORB[81.50,81.94] vol=2.3x ATR=0.23 |
| Stop hit — per-position SL triggered | 2023-11-15 11:30:00 | 81.73 | 81.70 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2023-11-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-16 10:00:00 | 80.96 | 81.44 | 0.00 | ORB-short ORB[81.31,81.70] vol=2.8x ATR=0.19 |
| Stop hit — per-position SL triggered | 2023-11-16 10:05:00 | 81.15 | 81.40 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2023-11-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 09:35:00 | 80.33 | 80.03 | 0.00 | ORB-long ORB[79.42,80.18] vol=1.6x ATR=0.23 |
| Stop hit — per-position SL triggered | 2023-11-20 09:40:00 | 80.10 | 80.05 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2023-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 09:35:00 | 79.76 | 80.01 | 0.00 | ORB-short ORB[79.93,80.57] vol=2.4x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 10:10:00 | 79.47 | 79.90 | 0.00 | T1 1.5R @ 79.47 |
| Target hit | 2023-11-21 15:20:00 | 79.45 | 79.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — BUY (started 2023-11-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:50:00 | 80.05 | 79.48 | 0.00 | ORB-long ORB[79.07,79.74] vol=2.1x ATR=0.20 |
| Stop hit — per-position SL triggered | 2023-11-22 10:10:00 | 79.85 | 79.63 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2023-11-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 09:45:00 | 78.90 | 79.12 | 0.00 | ORB-short ORB[78.96,79.26] vol=1.6x ATR=0.16 |
| Stop hit — per-position SL triggered | 2023-11-23 09:50:00 | 79.06 | 79.11 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 11:15:00 | 78.59 | 78.85 | 0.00 | ORB-short ORB[78.72,79.08] vol=2.2x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 13:05:00 | 78.37 | 78.76 | 0.00 | T1 1.5R @ 78.37 |
| Target hit | 2023-11-24 15:20:00 | 77.91 | 78.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 85 — BUY (started 2023-11-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 10:05:00 | 80.72 | 79.88 | 0.00 | ORB-long ORB[79.52,79.99] vol=3.7x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-11-29 10:10:00 | 80.46 | 80.06 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2023-11-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:50:00 | 80.27 | 80.73 | 0.00 | ORB-short ORB[80.31,81.20] vol=1.5x ATR=0.25 |
| Stop hit — per-position SL triggered | 2023-11-30 10:15:00 | 80.52 | 80.62 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2023-12-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 09:35:00 | 82.49 | 81.81 | 0.00 | ORB-long ORB[81.03,81.58] vol=3.9x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 09:55:00 | 82.98 | 82.22 | 0.00 | T1 1.5R @ 82.98 |
| Target hit | 2023-12-01 11:10:00 | 82.66 | 82.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 88 — SELL (started 2023-12-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 09:40:00 | 85.90 | 86.37 | 0.00 | ORB-short ORB[86.25,87.25] vol=1.8x ATR=0.30 |
| Stop hit — per-position SL triggered | 2023-12-06 09:55:00 | 86.20 | 86.30 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2023-12-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 11:05:00 | 87.55 | 86.71 | 0.00 | ORB-long ORB[86.04,87.29] vol=3.1x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-12-07 11:30:00 | 87.29 | 86.87 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2023-12-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 10:40:00 | 87.90 | 87.23 | 0.00 | ORB-long ORB[86.54,87.33] vol=2.0x ATR=0.29 |
| Stop hit — per-position SL triggered | 2023-12-08 10:50:00 | 87.61 | 87.31 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2023-12-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 09:35:00 | 89.38 | 88.70 | 0.00 | ORB-long ORB[87.67,88.96] vol=2.2x ATR=0.39 |
| Stop hit — per-position SL triggered | 2023-12-11 10:40:00 | 88.99 | 89.09 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2023-12-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 09:45:00 | 89.76 | 89.35 | 0.00 | ORB-long ORB[88.90,89.53] vol=2.2x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-12-13 09:55:00 | 89.50 | 89.44 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2023-12-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-15 09:35:00 | 89.32 | 89.87 | 0.00 | ORB-short ORB[89.80,90.36] vol=2.7x ATR=0.23 |
| Stop hit — per-position SL triggered | 2023-12-15 09:45:00 | 89.55 | 89.75 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2023-12-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:50:00 | 86.23 | 85.73 | 0.00 | ORB-long ORB[85.30,85.76] vol=2.3x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 09:55:00 | 86.58 | 86.12 | 0.00 | T1 1.5R @ 86.58 |
| Target hit | 2023-12-27 11:05:00 | 86.31 | 86.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 95 — BUY (started 2024-01-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 09:55:00 | 88.58 | 88.17 | 0.00 | ORB-long ORB[87.60,88.26] vol=2.2x ATR=0.23 |
| Stop hit — per-position SL triggered | 2024-01-01 10:05:00 | 88.35 | 88.25 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2024-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:35:00 | 89.16 | 88.52 | 0.00 | ORB-long ORB[88.04,88.63] vol=1.9x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 09:40:00 | 89.62 | 89.16 | 0.00 | T1 1.5R @ 89.62 |
| Stop hit — per-position SL triggered | 2024-01-02 09:55:00 | 89.16 | 89.44 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2024-01-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 09:35:00 | 91.60 | 91.99 | 0.00 | ORB-short ORB[91.71,92.46] vol=1.8x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-01-04 09:40:00 | 91.89 | 91.96 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2024-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 09:30:00 | 92.88 | 93.31 | 0.00 | ORB-short ORB[93.02,93.85] vol=1.6x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 09:45:00 | 92.51 | 93.06 | 0.00 | T1 1.5R @ 92.51 |
| Stop hit — per-position SL triggered | 2024-01-05 09:50:00 | 92.88 | 93.04 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2024-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 09:30:00 | 91.38 | 92.04 | 0.00 | ORB-short ORB[91.82,92.60] vol=1.5x ATR=0.37 |
| Stop hit — per-position SL triggered | 2024-01-08 10:00:00 | 91.75 | 91.78 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2024-01-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 10:25:00 | 91.42 | 90.52 | 0.00 | ORB-long ORB[90.05,90.68] vol=3.2x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-01-11 10:30:00 | 91.13 | 90.59 | 0.00 | SL hit |

### Cycle 101 — BUY (started 2024-01-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 09:55:00 | 91.76 | 91.17 | 0.00 | ORB-long ORB[90.58,91.33] vol=3.8x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 10:40:00 | 92.15 | 91.50 | 0.00 | T1 1.5R @ 92.15 |
| Target hit | 2024-01-12 15:05:00 | 92.28 | 92.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 102 — BUY (started 2024-01-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 10:00:00 | 93.77 | 93.25 | 0.00 | ORB-long ORB[92.68,93.49] vol=2.0x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 10:10:00 | 94.13 | 93.50 | 0.00 | T1 1.5R @ 94.13 |
| Stop hit — per-position SL triggered | 2024-01-16 10:15:00 | 93.77 | 93.52 | 0.00 | SL hit |

### Cycle 103 — BUY (started 2024-01-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 10:40:00 | 92.96 | 92.50 | 0.00 | ORB-long ORB[92.04,92.70] vol=2.2x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 11:00:00 | 93.45 | 92.61 | 0.00 | T1 1.5R @ 93.45 |
| Stop hit — per-position SL triggered | 2024-01-19 11:05:00 | 92.96 | 92.62 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2024-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 09:40:00 | 93.13 | 93.49 | 0.00 | ORB-short ORB[93.43,93.98] vol=1.6x ATR=0.27 |
| Stop hit — per-position SL triggered | 2024-01-20 11:30:00 | 93.40 | 93.29 | 0.00 | SL hit |

### Cycle 105 — BUY (started 2024-01-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 10:55:00 | 95.73 | 95.09 | 0.00 | ORB-long ORB[94.41,95.58] vol=1.5x ATR=0.27 |
| Stop hit — per-position SL triggered | 2024-01-30 11:05:00 | 95.46 | 95.14 | 0.00 | SL hit |

### Cycle 106 — BUY (started 2024-02-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 09:55:00 | 97.11 | 96.52 | 0.00 | ORB-long ORB[95.87,96.66] vol=2.1x ATR=0.36 |
| Stop hit — per-position SL triggered | 2024-02-01 10:00:00 | 96.75 | 96.59 | 0.00 | SL hit |

### Cycle 107 — BUY (started 2024-02-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 09:35:00 | 102.30 | 101.81 | 0.00 | ORB-long ORB[101.30,101.99] vol=2.3x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 09:50:00 | 103.02 | 102.11 | 0.00 | T1 1.5R @ 103.02 |
| Target hit | 2024-02-02 14:55:00 | 103.10 | 103.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 108 — BUY (started 2024-02-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-14 10:55:00 | 111.40 | 110.15 | 0.00 | ORB-long ORB[108.60,109.87] vol=1.9x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-02-14 11:35:00 | 111.01 | 110.41 | 0.00 | SL hit |

### Cycle 109 — SELL (started 2024-02-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-16 09:40:00 | 115.98 | 116.60 | 0.00 | ORB-short ORB[116.32,117.96] vol=1.6x ATR=0.52 |
| Stop hit — per-position SL triggered | 2024-02-16 09:55:00 | 116.50 | 116.46 | 0.00 | SL hit |

### Cycle 110 — BUY (started 2024-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:40:00 | 115.28 | 114.80 | 0.00 | ORB-long ORB[113.90,115.08] vol=3.2x ATR=0.32 |
| Stop hit — per-position SL triggered | 2024-02-21 09:45:00 | 114.96 | 114.83 | 0.00 | SL hit |

### Cycle 111 — SELL (started 2024-02-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:05:00 | 113.13 | 113.72 | 0.00 | ORB-short ORB[114.00,114.95] vol=2.9x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 10:25:00 | 112.53 | 113.55 | 0.00 | T1 1.5R @ 112.53 |
| Target hit | 2024-02-28 15:20:00 | 109.88 | 111.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 112 — BUY (started 2024-03-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-02 09:30:00 | 117.23 | 116.72 | 0.00 | ORB-long ORB[116.00,116.47] vol=2.4x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-02 09:40:00 | 117.68 | 117.24 | 0.00 | T1 1.5R @ 117.68 |
| Stop hit — per-position SL triggered | 2024-03-02 09:50:00 | 117.23 | 117.31 | 0.00 | SL hit |

### Cycle 113 — SELL (started 2024-03-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:35:00 | 117.37 | 117.86 | 0.00 | ORB-short ORB[117.46,118.44] vol=1.7x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 09:45:00 | 116.80 | 117.70 | 0.00 | T1 1.5R @ 116.80 |
| Stop hit — per-position SL triggered | 2024-03-04 09:50:00 | 117.37 | 117.67 | 0.00 | SL hit |

### Cycle 114 — BUY (started 2024-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 10:50:00 | 119.88 | 118.75 | 0.00 | ORB-long ORB[117.62,118.92] vol=4.7x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 11:00:00 | 120.55 | 118.94 | 0.00 | T1 1.5R @ 120.55 |
| Stop hit — per-position SL triggered | 2024-03-05 11:15:00 | 119.88 | 119.14 | 0.00 | SL hit |

### Cycle 115 — SELL (started 2024-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:30:00 | 120.17 | 120.58 | 0.00 | ORB-short ORB[120.21,121.26] vol=1.8x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:35:00 | 119.55 | 120.49 | 0.00 | T1 1.5R @ 119.55 |
| Stop hit — per-position SL triggered | 2024-03-06 09:40:00 | 120.17 | 120.44 | 0.00 | SL hit |

### Cycle 116 — SELL (started 2024-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 09:35:00 | 112.21 | 113.15 | 0.00 | ORB-short ORB[112.70,114.33] vol=1.8x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:55:00 | 111.45 | 112.63 | 0.00 | T1 1.5R @ 111.45 |
| Target hit | 2024-03-13 15:20:00 | 106.90 | 109.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 117 — SELL (started 2024-03-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 09:45:00 | 109.49 | 109.98 | 0.00 | ORB-short ORB[109.68,110.56] vol=2.2x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:00:00 | 108.88 | 109.74 | 0.00 | T1 1.5R @ 108.88 |
| Stop hit — per-position SL triggered | 2024-03-20 12:25:00 | 109.49 | 108.83 | 0.00 | SL hit |

### Cycle 118 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:15:00 | 113.10 | 112.26 | 0.00 | ORB-long ORB[111.60,112.60] vol=1.8x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-03-21 12:05:00 | 112.67 | 112.69 | 0.00 | SL hit |

### Cycle 119 — BUY (started 2024-03-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 09:35:00 | 113.75 | 113.20 | 0.00 | ORB-long ORB[112.69,113.49] vol=2.0x ATR=0.37 |
| Stop hit — per-position SL triggered | 2024-03-22 10:10:00 | 113.38 | 113.33 | 0.00 | SL hit |

### Cycle 120 — BUY (started 2024-03-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:40:00 | 114.95 | 114.45 | 0.00 | ORB-long ORB[113.62,114.58] vol=2.5x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-03-28 09:50:00 | 114.56 | 114.49 | 0.00 | SL hit |

### Cycle 121 — SELL (started 2024-04-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:55:00 | 121.54 | 122.01 | 0.00 | ORB-short ORB[121.60,122.95] vol=1.7x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 11:10:00 | 120.86 | 121.95 | 0.00 | T1 1.5R @ 120.86 |
| Stop hit — per-position SL triggered | 2024-04-04 11:35:00 | 121.54 | 121.91 | 0.00 | SL hit |

### Cycle 122 — SELL (started 2024-04-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 09:30:00 | 122.09 | 122.77 | 0.00 | ORB-short ORB[122.35,123.79] vol=1.5x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-04-08 09:35:00 | 122.43 | 122.71 | 0.00 | SL hit |

### Cycle 123 — BUY (started 2024-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 09:40:00 | 118.67 | 118.04 | 0.00 | ORB-long ORB[116.89,118.58] vol=1.5x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 09:50:00 | 119.46 | 118.45 | 0.00 | T1 1.5R @ 119.46 |
| Target hit | 2024-04-22 15:20:00 | 119.94 | 119.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 124 — SELL (started 2024-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-24 09:45:00 | 120.06 | 120.66 | 0.00 | ORB-short ORB[120.52,121.09] vol=1.8x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 09:55:00 | 119.67 | 120.40 | 0.00 | T1 1.5R @ 119.67 |
| Stop hit — per-position SL triggered | 2024-04-24 10:40:00 | 120.06 | 120.29 | 0.00 | SL hit |

### Cycle 125 — BUY (started 2024-04-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 09:55:00 | 121.70 | 120.70 | 0.00 | ORB-long ORB[119.02,120.50] vol=3.3x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 10:00:00 | 122.15 | 121.01 | 0.00 | T1 1.5R @ 122.15 |
| Stop hit — per-position SL triggered | 2024-04-25 10:50:00 | 121.70 | 121.63 | 0.00 | SL hit |

### Cycle 126 — BUY (started 2024-04-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 09:35:00 | 124.41 | 123.61 | 0.00 | ORB-long ORB[122.82,123.89] vol=1.7x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-04-26 09:50:00 | 123.97 | 124.00 | 0.00 | SL hit |

### Cycle 127 — SELL (started 2024-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 09:45:00 | 123.42 | 124.30 | 0.00 | ORB-short ORB[124.00,124.95] vol=1.6x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-04-29 10:25:00 | 123.82 | 123.94 | 0.00 | SL hit |

### Cycle 128 — BUY (started 2024-05-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 09:55:00 | 125.85 | 125.15 | 0.00 | ORB-long ORB[124.61,125.50] vol=1.6x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-05-02 10:35:00 | 125.43 | 125.42 | 0.00 | SL hit |

### Cycle 129 — SELL (started 2024-05-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:20:00 | 125.45 | 126.08 | 0.00 | ORB-short ORB[126.00,126.58] vol=1.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2024-05-03 10:25:00 | 125.73 | 126.06 | 0.00 | SL hit |

### Cycle 130 — SELL (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 11:15:00 | 115.54 | 117.72 | 0.00 | ORB-short ORB[117.73,119.34] vol=2.7x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 12:35:00 | 114.75 | 117.03 | 0.00 | T1 1.5R @ 114.75 |
| Stop hit — per-position SL triggered | 2024-05-07 13:25:00 | 115.54 | 116.77 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 11:15:00 | 60.19 | 2023-05-15 12:05:00 | 60.04 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-05-16 09:30:00 | 61.63 | 2023-05-16 09:35:00 | 61.42 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-05-17 11:00:00 | 60.69 | 2023-05-17 11:05:00 | 60.86 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-05-18 10:55:00 | 60.55 | 2023-05-18 11:45:00 | 60.32 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-05-18 10:55:00 | 60.55 | 2023-05-18 15:20:00 | 59.11 | TARGET_HIT | 0.50 | 2.38% |
| SELL | retest1 | 2023-05-19 09:35:00 | 58.55 | 2023-05-19 10:10:00 | 58.82 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2023-05-23 10:00:00 | 60.76 | 2023-05-23 10:55:00 | 60.62 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-05-26 09:30:00 | 60.89 | 2023-05-26 09:35:00 | 60.71 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-05-29 10:05:00 | 61.51 | 2023-05-29 10:10:00 | 61.64 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-05-30 11:00:00 | 61.22 | 2023-05-30 12:30:00 | 61.34 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-05-31 10:50:00 | 61.72 | 2023-05-31 11:00:00 | 61.91 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-05-31 10:50:00 | 61.72 | 2023-05-31 11:45:00 | 61.72 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-02 09:30:00 | 62.55 | 2023-06-02 10:30:00 | 62.41 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-06-05 10:30:00 | 62.66 | 2023-06-05 10:45:00 | 62.79 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-06-06 11:05:00 | 62.00 | 2023-06-06 11:20:00 | 61.84 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-06-06 11:05:00 | 62.00 | 2023-06-06 11:25:00 | 62.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-07 09:35:00 | 62.85 | 2023-06-07 09:55:00 | 63.02 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-06-07 09:35:00 | 62.85 | 2023-06-07 14:35:00 | 63.00 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2023-06-08 09:55:00 | 62.88 | 2023-06-08 10:00:00 | 62.99 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-06-09 09:30:00 | 62.16 | 2023-06-09 09:40:00 | 62.29 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-06-13 10:05:00 | 62.82 | 2023-06-13 10:25:00 | 62.93 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-14 11:05:00 | 60.66 | 2023-06-14 11:15:00 | 60.80 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-06-15 09:35:00 | 60.58 | 2023-06-15 09:45:00 | 60.69 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-06-19 09:45:00 | 61.27 | 2023-06-19 09:55:00 | 61.09 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-06-21 10:25:00 | 60.98 | 2023-06-21 10:45:00 | 61.08 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-06-26 10:35:00 | 58.83 | 2023-06-26 11:45:00 | 58.63 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-06-26 10:35:00 | 58.83 | 2023-06-26 14:55:00 | 58.77 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2023-06-30 09:30:00 | 60.16 | 2023-06-30 09:35:00 | 60.43 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-06-30 09:30:00 | 60.16 | 2023-06-30 10:20:00 | 60.16 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-03 09:40:00 | 61.01 | 2023-07-03 09:45:00 | 60.87 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-04 10:15:00 | 64.78 | 2023-07-04 10:35:00 | 64.43 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2023-07-05 10:45:00 | 64.76 | 2023-07-05 10:50:00 | 64.55 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-07-06 11:15:00 | 65.03 | 2023-07-06 11:20:00 | 65.25 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-07-06 11:15:00 | 65.03 | 2023-07-06 12:40:00 | 65.20 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2023-07-07 09:30:00 | 65.59 | 2023-07-07 09:45:00 | 65.38 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-07-11 09:55:00 | 66.09 | 2023-07-11 10:00:00 | 66.32 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-07-12 09:30:00 | 66.91 | 2023-07-12 10:55:00 | 67.21 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-07-12 09:30:00 | 66.91 | 2023-07-12 11:30:00 | 66.91 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-13 09:30:00 | 66.00 | 2023-07-13 09:35:00 | 66.19 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-14 09:40:00 | 65.44 | 2023-07-14 09:55:00 | 65.16 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-07-18 09:35:00 | 66.18 | 2023-07-18 10:15:00 | 66.42 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-07-19 09:50:00 | 66.51 | 2023-07-19 10:15:00 | 66.86 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-07-19 09:50:00 | 66.51 | 2023-07-19 10:50:00 | 66.51 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-20 09:45:00 | 68.57 | 2023-07-20 10:05:00 | 68.31 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-07-27 09:30:00 | 68.20 | 2023-07-27 10:00:00 | 68.53 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-07-27 09:30:00 | 68.20 | 2023-07-27 11:15:00 | 69.00 | TARGET_HIT | 0.50 | 1.17% |
| SELL | retest1 | 2023-08-01 10:30:00 | 68.50 | 2023-08-01 12:05:00 | 68.29 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-08-01 10:30:00 | 68.50 | 2023-08-01 14:35:00 | 68.36 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2023-08-02 11:00:00 | 67.10 | 2023-08-02 11:20:00 | 67.29 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-08-07 09:40:00 | 66.36 | 2023-08-07 10:00:00 | 66.11 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-08-08 10:10:00 | 66.76 | 2023-08-08 10:25:00 | 67.04 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-08-08 10:10:00 | 66.76 | 2023-08-08 10:45:00 | 66.76 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-10 10:20:00 | 66.63 | 2023-08-10 10:55:00 | 66.89 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-08-11 09:40:00 | 66.69 | 2023-08-11 09:55:00 | 66.99 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-08-11 09:40:00 | 66.69 | 2023-08-11 12:30:00 | 66.87 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2023-08-17 09:45:00 | 66.08 | 2023-08-17 09:55:00 | 66.32 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-08-17 09:45:00 | 66.08 | 2023-08-17 10:30:00 | 66.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-18 09:45:00 | 66.62 | 2023-08-18 10:05:00 | 66.43 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-08-21 09:45:00 | 65.68 | 2023-08-21 09:55:00 | 65.85 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-08-22 11:10:00 | 65.60 | 2023-08-22 13:00:00 | 65.69 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-08-23 10:50:00 | 65.35 | 2023-08-23 10:55:00 | 65.46 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-08-25 10:20:00 | 65.51 | 2023-08-25 10:30:00 | 65.23 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-08-25 10:20:00 | 65.51 | 2023-08-25 10:40:00 | 65.51 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-31 09:35:00 | 65.41 | 2023-08-31 10:05:00 | 65.26 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-09-01 10:55:00 | 64.78 | 2023-09-01 11:05:00 | 65.03 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-09-01 10:55:00 | 64.78 | 2023-09-01 15:20:00 | 65.70 | TARGET_HIT | 0.50 | 1.42% |
| BUY | retest1 | 2023-09-04 10:55:00 | 66.23 | 2023-09-04 11:00:00 | 66.51 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-09-04 10:55:00 | 66.23 | 2023-09-04 15:20:00 | 67.29 | TARGET_HIT | 0.50 | 1.60% |
| SELL | retest1 | 2023-09-05 11:05:00 | 67.30 | 2023-09-05 11:45:00 | 67.49 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-09-06 11:10:00 | 67.22 | 2023-09-06 11:30:00 | 66.98 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-09-06 11:10:00 | 67.22 | 2023-09-06 15:20:00 | 66.80 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2023-09-07 09:55:00 | 67.38 | 2023-09-07 10:45:00 | 67.61 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-09-07 09:55:00 | 67.38 | 2023-09-07 11:30:00 | 67.38 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-08 09:50:00 | 67.54 | 2023-09-08 10:20:00 | 67.28 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-09-08 09:50:00 | 67.54 | 2023-09-08 10:45:00 | 67.54 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-11 09:30:00 | 69.81 | 2023-09-11 09:40:00 | 69.52 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-09-14 09:35:00 | 73.68 | 2023-09-14 09:45:00 | 73.33 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2023-09-20 10:40:00 | 75.13 | 2023-09-20 10:50:00 | 74.82 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-09-27 10:05:00 | 73.31 | 2023-09-27 10:50:00 | 73.59 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-09-28 09:45:00 | 75.77 | 2023-09-28 10:00:00 | 75.51 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-09-29 10:50:00 | 74.62 | 2023-09-29 11:05:00 | 74.86 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-10-03 09:45:00 | 75.67 | 2023-10-03 10:20:00 | 75.40 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-10-06 10:20:00 | 75.22 | 2023-10-06 10:40:00 | 75.57 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-10-06 10:20:00 | 75.22 | 2023-10-06 11:15:00 | 75.22 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-11 09:55:00 | 74.36 | 2023-10-11 10:00:00 | 74.54 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-10-13 09:40:00 | 72.40 | 2023-10-13 09:45:00 | 72.02 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-10-13 09:40:00 | 72.40 | 2023-10-13 09:50:00 | 72.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-17 10:15:00 | 75.46 | 2023-10-17 13:35:00 | 75.27 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-10-18 10:40:00 | 74.88 | 2023-10-18 10:45:00 | 74.63 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-10-18 10:40:00 | 74.88 | 2023-10-18 10:55:00 | 74.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-19 11:05:00 | 74.68 | 2023-10-19 11:20:00 | 74.97 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-10-19 11:05:00 | 74.68 | 2023-10-19 13:20:00 | 74.68 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-23 10:10:00 | 73.10 | 2023-10-23 10:25:00 | 73.36 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-10-26 09:30:00 | 69.17 | 2023-10-26 09:40:00 | 68.68 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2023-10-26 09:30:00 | 69.17 | 2023-10-26 10:00:00 | 69.11 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2023-11-01 09:50:00 | 76.28 | 2023-11-01 10:05:00 | 76.54 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-11-02 09:40:00 | 78.12 | 2023-11-02 10:10:00 | 77.85 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-11-03 10:10:00 | 78.36 | 2023-11-03 10:30:00 | 78.16 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-11-06 09:30:00 | 77.36 | 2023-11-06 09:40:00 | 77.54 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-07 09:55:00 | 77.36 | 2023-11-07 10:10:00 | 77.66 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-11-07 09:55:00 | 77.36 | 2023-11-07 11:20:00 | 77.36 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-09 10:55:00 | 77.42 | 2023-11-09 11:10:00 | 77.64 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-11-09 10:55:00 | 77.42 | 2023-11-09 12:45:00 | 77.42 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-13 10:00:00 | 77.91 | 2023-11-13 10:10:00 | 77.76 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-11-15 11:00:00 | 81.96 | 2023-11-15 11:30:00 | 81.73 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-16 10:00:00 | 80.96 | 2023-11-16 10:05:00 | 81.15 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-11-20 09:35:00 | 80.33 | 2023-11-20 09:40:00 | 80.10 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-21 09:35:00 | 79.76 | 2023-11-21 10:10:00 | 79.47 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-11-21 09:35:00 | 79.76 | 2023-11-21 15:20:00 | 79.45 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2023-11-22 09:50:00 | 80.05 | 2023-11-22 10:10:00 | 79.85 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-11-23 09:45:00 | 78.90 | 2023-11-23 09:50:00 | 79.06 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-11-24 11:15:00 | 78.59 | 2023-11-24 13:05:00 | 78.37 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-11-24 11:15:00 | 78.59 | 2023-11-24 15:20:00 | 77.91 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2023-11-29 10:05:00 | 80.72 | 2023-11-29 10:10:00 | 80.46 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-11-30 09:50:00 | 80.27 | 2023-11-30 10:15:00 | 80.52 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-12-01 09:35:00 | 82.49 | 2023-12-01 09:55:00 | 82.98 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2023-12-01 09:35:00 | 82.49 | 2023-12-01 11:10:00 | 82.66 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2023-12-06 09:40:00 | 85.90 | 2023-12-06 09:55:00 | 86.20 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-12-07 11:05:00 | 87.55 | 2023-12-07 11:30:00 | 87.29 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-12-08 10:40:00 | 87.90 | 2023-12-08 10:50:00 | 87.61 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-12-11 09:35:00 | 89.38 | 2023-12-11 10:40:00 | 88.99 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-12-13 09:45:00 | 89.76 | 2023-12-13 09:55:00 | 89.50 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-12-15 09:35:00 | 89.32 | 2023-12-15 09:45:00 | 89.55 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-27 09:50:00 | 86.23 | 2023-12-27 09:55:00 | 86.58 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-12-27 09:50:00 | 86.23 | 2023-12-27 11:05:00 | 86.31 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2024-01-01 09:55:00 | 88.58 | 2024-01-01 10:05:00 | 88.35 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-01-02 09:35:00 | 89.16 | 2024-01-02 09:40:00 | 89.62 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-01-02 09:35:00 | 89.16 | 2024-01-02 09:55:00 | 89.16 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-04 09:35:00 | 91.60 | 2024-01-04 09:40:00 | 91.89 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-01-05 09:30:00 | 92.88 | 2024-01-05 09:45:00 | 92.51 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-01-05 09:30:00 | 92.88 | 2024-01-05 09:50:00 | 92.88 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-08 09:30:00 | 91.38 | 2024-01-08 10:00:00 | 91.75 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-01-11 10:25:00 | 91.42 | 2024-01-11 10:30:00 | 91.13 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-01-12 09:55:00 | 91.76 | 2024-01-12 10:40:00 | 92.15 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-01-12 09:55:00 | 91.76 | 2024-01-12 15:05:00 | 92.28 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2024-01-16 10:00:00 | 93.77 | 2024-01-16 10:10:00 | 94.13 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-01-16 10:00:00 | 93.77 | 2024-01-16 10:15:00 | 93.77 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-19 10:40:00 | 92.96 | 2024-01-19 11:00:00 | 93.45 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-01-19 10:40:00 | 92.96 | 2024-01-19 11:05:00 | 92.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-20 09:40:00 | 93.13 | 2024-01-20 11:30:00 | 93.40 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-01-30 10:55:00 | 95.73 | 2024-01-30 11:05:00 | 95.46 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-02-01 09:55:00 | 97.11 | 2024-02-01 10:00:00 | 96.75 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-02-02 09:35:00 | 102.30 | 2024-02-02 09:50:00 | 103.02 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-02-02 09:35:00 | 102.30 | 2024-02-02 14:55:00 | 103.10 | TARGET_HIT | 0.50 | 0.78% |
| BUY | retest1 | 2024-02-14 10:55:00 | 111.40 | 2024-02-14 11:35:00 | 111.01 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-02-16 09:40:00 | 115.98 | 2024-02-16 09:55:00 | 116.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-02-21 09:40:00 | 115.28 | 2024-02-21 09:45:00 | 114.96 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-02-28 10:05:00 | 113.13 | 2024-02-28 10:25:00 | 112.53 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-02-28 10:05:00 | 113.13 | 2024-02-28 15:20:00 | 109.88 | TARGET_HIT | 0.50 | 2.87% |
| BUY | retest1 | 2024-03-02 09:30:00 | 117.23 | 2024-03-02 09:40:00 | 117.68 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-03-02 09:30:00 | 117.23 | 2024-03-02 09:50:00 | 117.23 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-04 09:35:00 | 117.37 | 2024-03-04 09:45:00 | 116.80 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-03-04 09:35:00 | 117.37 | 2024-03-04 09:50:00 | 117.37 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-05 10:50:00 | 119.88 | 2024-03-05 11:00:00 | 120.55 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-03-05 10:50:00 | 119.88 | 2024-03-05 11:15:00 | 119.88 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-06 09:30:00 | 120.17 | 2024-03-06 09:35:00 | 119.55 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-03-06 09:30:00 | 120.17 | 2024-03-06 09:40:00 | 120.17 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-13 09:35:00 | 112.21 | 2024-03-13 09:55:00 | 111.45 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-03-13 09:35:00 | 112.21 | 2024-03-13 15:20:00 | 106.90 | TARGET_HIT | 0.50 | 4.73% |
| SELL | retest1 | 2024-03-20 09:45:00 | 109.49 | 2024-03-20 10:00:00 | 108.88 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-03-20 09:45:00 | 109.49 | 2024-03-20 12:25:00 | 109.49 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-21 10:15:00 | 113.10 | 2024-03-21 12:05:00 | 112.67 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-03-22 09:35:00 | 113.75 | 2024-03-22 10:10:00 | 113.38 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-03-28 09:40:00 | 114.95 | 2024-03-28 09:50:00 | 114.56 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-04-04 10:55:00 | 121.54 | 2024-04-04 11:10:00 | 120.86 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-04-04 10:55:00 | 121.54 | 2024-04-04 11:35:00 | 121.54 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-08 09:30:00 | 122.09 | 2024-04-08 09:35:00 | 122.43 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-04-22 09:40:00 | 118.67 | 2024-04-22 09:50:00 | 119.46 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-04-22 09:40:00 | 118.67 | 2024-04-22 15:20:00 | 119.94 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2024-04-24 09:45:00 | 120.06 | 2024-04-24 09:55:00 | 119.67 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-04-24 09:45:00 | 120.06 | 2024-04-24 10:40:00 | 120.06 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-25 09:55:00 | 121.70 | 2024-04-25 10:00:00 | 122.15 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-04-25 09:55:00 | 121.70 | 2024-04-25 10:50:00 | 121.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-26 09:35:00 | 124.41 | 2024-04-26 09:50:00 | 123.97 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-04-29 09:45:00 | 123.42 | 2024-04-29 10:25:00 | 123.82 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-05-02 09:55:00 | 125.85 | 2024-05-02 10:35:00 | 125.43 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-03 10:20:00 | 125.45 | 2024-05-03 10:25:00 | 125.73 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-05-07 11:15:00 | 115.54 | 2024-05-07 12:35:00 | 114.75 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-05-07 11:15:00 | 115.54 | 2024-05-07 13:25:00 | 115.54 | STOP_HIT | 0.50 | 0.00% |
