# Samvardhana Motherson International Ltd. (MOTHERSON)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (33933 bars)
- **Last close:** 131.57
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
| ENTRY1 | 59 |
| ENTRY2 | 0 |
| PARTIAL | 27 |
| TARGET_HIT | 13 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 46
- **Target hits / Stop hits / Partials:** 13 / 46 / 27
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 16.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 19 | 45.2% | 6 | 23 | 13 | 0.14% | 5.9% |
| BUY @ 2nd Alert (retest1) | 42 | 19 | 45.2% | 6 | 23 | 13 | 0.14% | 5.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 44 | 21 | 47.7% | 7 | 23 | 14 | 0.23% | 10.3% |
| SELL @ 2nd Alert (retest1) | 44 | 21 | 47.7% | 7 | 23 | 14 | 0.23% | 10.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 86 | 40 | 46.5% | 13 | 46 | 27 | 0.19% | 16.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 11:15:00 | 84.27 | 83.75 | 0.00 | ORB-long ORB[83.17,84.20] vol=2.5x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 11:35:00 | 84.65 | 83.86 | 0.00 | T1 1.5R @ 84.65 |
| Stop hit — per-position SL triggered | 2024-05-14 12:15:00 | 84.27 | 84.07 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 09:30:00 | 84.50 | 84.93 | 0.00 | ORB-short ORB[84.77,85.53] vol=1.9x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:00:00 | 84.14 | 84.69 | 0.00 | T1 1.5R @ 84.14 |
| Stop hit — per-position SL triggered | 2024-05-15 11:20:00 | 84.50 | 84.25 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 84.23 | 84.77 | 0.00 | ORB-short ORB[84.60,85.23] vol=2.8x ATR=0.26 |
| Stop hit — per-position SL triggered | 2024-05-16 11:25:00 | 84.49 | 84.75 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 09:30:00 | 89.27 | 88.31 | 0.00 | ORB-long ORB[87.80,88.53] vol=1.5x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-05-22 09:35:00 | 88.88 | 88.37 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 09:35:00 | 92.47 | 92.11 | 0.00 | ORB-long ORB[91.03,92.23] vol=2.1x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 10:20:00 | 93.20 | 92.46 | 0.00 | T1 1.5R @ 93.20 |
| Stop hit — per-position SL triggered | 2024-05-24 10:40:00 | 92.47 | 92.56 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:20:00 | 108.67 | 109.28 | 0.00 | ORB-short ORB[108.73,109.90] vol=2.0x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-06-12 10:25:00 | 109.08 | 109.25 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:30:00 | 113.67 | 112.89 | 0.00 | ORB-long ORB[111.88,113.05] vol=1.6x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 10:40:00 | 114.20 | 113.09 | 0.00 | T1 1.5R @ 114.20 |
| Target hit | 2024-06-14 15:20:00 | 116.40 | 115.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:30:00 | 127.25 | 126.92 | 0.00 | ORB-long ORB[125.57,127.20] vol=3.4x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-06-25 09:35:00 | 126.85 | 126.93 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:30:00 | 129.06 | 128.57 | 0.00 | ORB-long ORB[127.15,128.97] vol=2.0x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 09:45:00 | 129.75 | 128.96 | 0.00 | T1 1.5R @ 129.75 |
| Target hit | 2024-06-26 13:10:00 | 129.83 | 129.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2024-07-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:00:00 | 133.89 | 132.44 | 0.00 | ORB-long ORB[131.27,132.60] vol=3.2x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 10:05:00 | 134.68 | 133.13 | 0.00 | T1 1.5R @ 134.68 |
| Target hit | 2024-07-02 12:10:00 | 134.75 | 135.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2024-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:30:00 | 138.79 | 137.90 | 0.00 | ORB-long ORB[136.67,138.43] vol=2.4x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-07-04 09:45:00 | 138.35 | 138.24 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:45:00 | 136.14 | 137.49 | 0.00 | ORB-short ORB[137.33,138.40] vol=2.6x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 09:50:00 | 135.40 | 136.79 | 0.00 | T1 1.5R @ 135.40 |
| Stop hit — per-position SL triggered | 2024-07-05 10:05:00 | 136.14 | 136.39 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 135.89 | 136.76 | 0.00 | ORB-short ORB[136.37,137.57] vol=1.7x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-07-08 09:45:00 | 136.29 | 136.72 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 133.75 | 135.29 | 0.00 | ORB-short ORB[135.13,136.20] vol=1.6x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 133.06 | 134.68 | 0.00 | T1 1.5R @ 133.06 |
| Target hit | 2024-07-10 11:45:00 | 133.45 | 133.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2024-07-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:40:00 | 133.02 | 131.92 | 0.00 | ORB-long ORB[131.07,132.17] vol=1.8x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 11:10:00 | 133.56 | 132.27 | 0.00 | T1 1.5R @ 133.56 |
| Target hit | 2024-07-15 15:20:00 | 134.46 | 133.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2024-07-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:30:00 | 135.32 | 134.71 | 0.00 | ORB-long ORB[133.82,135.31] vol=1.7x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:45:00 | 135.82 | 134.87 | 0.00 | T1 1.5R @ 135.82 |
| Stop hit — per-position SL triggered | 2024-07-16 11:15:00 | 135.32 | 135.02 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 10:55:00 | 126.58 | 126.98 | 0.00 | ORB-short ORB[126.85,128.30] vol=1.6x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:15:00 | 126.01 | 126.91 | 0.00 | T1 1.5R @ 126.01 |
| Stop hit — per-position SL triggered | 2024-07-23 11:20:00 | 126.58 | 126.90 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-24 11:10:00 | 126.45 | 127.41 | 0.00 | ORB-short ORB[127.44,128.49] vol=2.2x ATR=0.45 |
| Stop hit — per-position SL triggered | 2024-07-24 15:20:00 | 126.47 | 126.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 10:15:00 | 124.60 | 125.54 | 0.00 | ORB-short ORB[125.05,126.53] vol=1.7x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:25:00 | 123.96 | 125.17 | 0.00 | T1 1.5R @ 123.96 |
| Stop hit — per-position SL triggered | 2024-07-25 10:35:00 | 124.60 | 125.06 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:50:00 | 129.47 | 128.83 | 0.00 | ORB-long ORB[127.59,129.33] vol=1.6x ATR=0.36 |
| Stop hit — per-position SL triggered | 2024-07-26 11:35:00 | 129.11 | 129.15 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:55:00 | 131.52 | 130.37 | 0.00 | ORB-long ORB[129.67,130.90] vol=2.6x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-07-31 10:00:00 | 131.10 | 130.48 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 11:00:00 | 130.09 | 131.00 | 0.00 | ORB-short ORB[131.00,132.00] vol=2.5x ATR=0.36 |
| Stop hit — per-position SL triggered | 2024-08-01 11:05:00 | 130.45 | 130.99 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:55:00 | 121.31 | 120.22 | 0.00 | ORB-long ORB[119.07,120.66] vol=1.9x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-08-08 10:00:00 | 120.76 | 120.29 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:55:00 | 125.07 | 124.47 | 0.00 | ORB-long ORB[122.87,124.63] vol=1.9x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-08-09 10:10:00 | 124.52 | 124.52 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 10:45:00 | 124.77 | 124.16 | 0.00 | ORB-long ORB[123.80,124.67] vol=3.4x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-08-19 10:55:00 | 124.48 | 124.20 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:05:00 | 122.88 | 123.01 | 0.00 | ORB-short ORB[123.27,124.57] vol=3.6x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:35:00 | 122.38 | 122.93 | 0.00 | T1 1.5R @ 122.38 |
| Target hit | 2024-08-20 12:10:00 | 122.71 | 122.63 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — BUY (started 2024-08-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 11:00:00 | 130.67 | 130.04 | 0.00 | ORB-long ORB[128.75,130.43] vol=3.1x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:15:00 | 131.13 | 130.35 | 0.00 | T1 1.5R @ 131.13 |
| Stop hit — per-position SL triggered | 2024-08-23 11:35:00 | 130.67 | 130.44 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:55:00 | 131.33 | 131.99 | 0.00 | ORB-short ORB[131.75,132.93] vol=1.5x ATR=0.32 |
| Stop hit — per-position SL triggered | 2024-08-26 11:15:00 | 131.65 | 131.96 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:15:00 | 132.75 | 132.08 | 0.00 | ORB-long ORB[131.55,132.50] vol=1.6x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:30:00 | 133.21 | 132.32 | 0.00 | T1 1.5R @ 133.21 |
| Stop hit — per-position SL triggered | 2024-08-27 10:45:00 | 132.75 | 132.43 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 11:05:00 | 129.50 | 130.16 | 0.00 | ORB-short ORB[129.99,131.07] vol=1.7x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-08-30 11:25:00 | 129.79 | 129.94 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:45:00 | 127.07 | 127.58 | 0.00 | ORB-short ORB[128.35,129.07] vol=2.1x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 12:45:00 | 126.33 | 127.16 | 0.00 | T1 1.5R @ 126.33 |
| Target hit | 2024-09-06 15:20:00 | 125.24 | 126.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2024-10-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-30 10:00:00 | 125.30 | 126.06 | 0.00 | ORB-short ORB[126.31,127.30] vol=1.9x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 10:15:00 | 124.63 | 125.55 | 0.00 | T1 1.5R @ 124.63 |
| Target hit | 2024-10-30 15:20:00 | 123.26 | 124.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2024-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:30:00 | 124.58 | 124.05 | 0.00 | ORB-long ORB[123.17,124.29] vol=2.1x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 09:40:00 | 125.25 | 124.25 | 0.00 | T1 1.5R @ 125.25 |
| Target hit | 2024-11-06 10:15:00 | 124.81 | 124.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — SELL (started 2024-11-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 10:25:00 | 120.20 | 121.30 | 0.00 | ORB-short ORB[121.00,122.16] vol=1.6x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 11:00:00 | 119.49 | 120.95 | 0.00 | T1 1.5R @ 119.49 |
| Target hit | 2024-11-08 15:20:00 | 117.29 | 119.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2024-11-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:40:00 | 113.11 | 112.39 | 0.00 | ORB-long ORB[110.20,111.74] vol=4.8x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-11-19 11:05:00 | 112.72 | 112.45 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-12-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 10:45:00 | 108.96 | 109.56 | 0.00 | ORB-short ORB[110.67,111.47] vol=12.2x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-12-04 10:50:00 | 109.25 | 109.30 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:30:00 | 111.23 | 110.55 | 0.00 | ORB-long ORB[109.50,110.12] vol=8.1x ATR=0.33 |
| Stop hit — per-position SL triggered | 2024-12-16 09:35:00 | 110.90 | 110.60 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:40:00 | 111.25 | 111.88 | 0.00 | ORB-short ORB[111.53,112.94] vol=2.1x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:25:00 | 110.71 | 111.69 | 0.00 | T1 1.5R @ 110.71 |
| Target hit | 2024-12-17 15:20:00 | 109.43 | 110.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2024-12-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:35:00 | 103.78 | 104.10 | 0.00 | ORB-short ORB[103.80,105.16] vol=1.6x ATR=0.31 |
| Stop hit — per-position SL triggered | 2024-12-24 09:40:00 | 104.09 | 104.09 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:35:00 | 104.58 | 104.94 | 0.00 | ORB-short ORB[104.87,105.59] vol=2.6x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:10:00 | 104.15 | 104.70 | 0.00 | T1 1.5R @ 104.15 |
| Stop hit — per-position SL triggered | 2024-12-26 11:30:00 | 104.58 | 104.44 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:55:00 | 106.54 | 105.62 | 0.00 | ORB-long ORB[104.27,105.76] vol=2.2x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:10:00 | 107.14 | 106.03 | 0.00 | T1 1.5R @ 107.14 |
| Stop hit — per-position SL triggered | 2024-12-27 10:30:00 | 106.54 | 106.20 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 09:40:00 | 103.67 | 104.03 | 0.00 | ORB-short ORB[103.93,104.89] vol=2.0x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-12-30 09:50:00 | 104.01 | 104.04 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-01-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 11:05:00 | 100.72 | 100.26 | 0.00 | ORB-long ORB[99.65,100.60] vol=1.8x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-01-20 12:10:00 | 100.46 | 100.46 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-01-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:00:00 | 98.04 | 99.33 | 0.00 | ORB-short ORB[100.07,100.97] vol=1.6x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-01-21 11:40:00 | 98.33 | 99.13 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-01-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 10:35:00 | 95.15 | 96.00 | 0.00 | ORB-short ORB[96.33,97.73] vol=1.5x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-01-22 10:55:00 | 95.54 | 95.78 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:55:00 | 95.72 | 96.50 | 0.00 | ORB-short ORB[96.55,97.34] vol=1.7x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-01-24 10:25:00 | 96.03 | 96.30 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:05:00 | 92.20 | 93.08 | 0.00 | ORB-short ORB[93.10,94.31] vol=4.0x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 91.54 | 92.81 | 0.00 | T1 1.5R @ 91.54 |
| Stop hit — per-position SL triggered | 2025-01-27 12:00:00 | 92.20 | 92.31 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-01-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 10:20:00 | 91.38 | 91.81 | 0.00 | ORB-short ORB[91.73,92.81] vol=2.0x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:30:00 | 90.76 | 91.76 | 0.00 | T1 1.5R @ 90.76 |
| Stop hit — per-position SL triggered | 2025-01-28 10:40:00 | 91.38 | 91.73 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-31 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:10:00 | 93.41 | 93.09 | 0.00 | ORB-long ORB[92.33,93.32] vol=1.9x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 10:25:00 | 93.82 | 93.15 | 0.00 | T1 1.5R @ 93.82 |
| Target hit | 2025-01-31 14:15:00 | 93.69 | 93.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — BUY (started 2025-02-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 11:00:00 | 91.90 | 91.22 | 0.00 | ORB-long ORB[90.65,91.89] vol=2.5x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 12:00:00 | 92.45 | 91.53 | 0.00 | T1 1.5R @ 92.45 |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 91.90 | 91.64 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:35:00 | 86.65 | 85.70 | 0.00 | ORB-long ORB[85.00,85.78] vol=1.9x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-02-20 09:45:00 | 86.32 | 85.99 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-02-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 11:05:00 | 84.45 | 85.80 | 0.00 | ORB-short ORB[86.22,87.33] vol=1.6x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-02-21 11:35:00 | 84.82 | 85.67 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 84.11 | 83.47 | 0.00 | ORB-long ORB[82.72,83.97] vol=2.0x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-02-25 09:50:00 | 83.75 | 83.72 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:50:00 | 82.38 | 82.52 | 0.00 | ORB-short ORB[82.50,83.66] vol=10.1x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-02-27 10:50:00 | 82.64 | 82.48 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-03-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:40:00 | 82.47 | 81.77 | 0.00 | ORB-long ORB[81.00,82.17] vol=1.7x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-03-18 09:50:00 | 82.19 | 81.81 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-04-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-01 09:35:00 | 88.27 | 87.33 | 0.00 | ORB-long ORB[86.39,87.56] vol=1.7x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-04-01 10:25:00 | 87.81 | 87.60 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-04-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 10:00:00 | 85.40 | 86.35 | 0.00 | ORB-short ORB[85.81,87.09] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-04-03 10:05:00 | 85.81 | 86.33 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-04-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:45:00 | 89.26 | 89.38 | 0.00 | ORB-short ORB[90.13,91.07] vol=2.4x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:50:00 | 88.65 | 89.36 | 0.00 | T1 1.5R @ 88.65 |
| Target hit | 2025-04-25 12:10:00 | 89.11 | 89.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 59 — BUY (started 2025-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:45:00 | 90.07 | 89.11 | 0.00 | ORB-long ORB[88.37,89.00] vol=1.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-05-05 10:05:00 | 89.68 | 89.44 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 11:15:00 | 84.27 | 2024-05-14 11:35:00 | 84.65 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-05-14 11:15:00 | 84.27 | 2024-05-14 12:15:00 | 84.27 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-15 09:30:00 | 84.50 | 2024-05-15 10:00:00 | 84.14 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-15 09:30:00 | 84.50 | 2024-05-15 11:20:00 | 84.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 11:15:00 | 84.23 | 2024-05-16 11:25:00 | 84.49 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-05-22 09:30:00 | 89.27 | 2024-05-22 09:35:00 | 88.88 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-05-24 09:35:00 | 92.47 | 2024-05-24 10:20:00 | 93.20 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2024-05-24 09:35:00 | 92.47 | 2024-05-24 10:40:00 | 92.47 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-12 10:20:00 | 108.67 | 2024-06-12 10:25:00 | 109.08 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-06-14 10:30:00 | 113.67 | 2024-06-14 10:40:00 | 114.20 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-06-14 10:30:00 | 113.67 | 2024-06-14 15:20:00 | 116.40 | TARGET_HIT | 0.50 | 2.40% |
| BUY | retest1 | 2024-06-25 09:30:00 | 127.25 | 2024-06-25 09:35:00 | 126.85 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-06-26 09:30:00 | 129.06 | 2024-06-26 09:45:00 | 129.75 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-06-26 09:30:00 | 129.06 | 2024-06-26 13:10:00 | 129.83 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2024-07-02 10:00:00 | 133.89 | 2024-07-02 10:05:00 | 134.68 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-07-02 10:00:00 | 133.89 | 2024-07-02 12:10:00 | 134.75 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2024-07-04 09:30:00 | 138.79 | 2024-07-04 09:45:00 | 138.35 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-05 09:45:00 | 136.14 | 2024-07-05 09:50:00 | 135.40 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-07-05 09:45:00 | 136.14 | 2024-07-05 10:05:00 | 136.14 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 09:40:00 | 135.89 | 2024-07-08 09:45:00 | 136.29 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-10 10:05:00 | 133.75 | 2024-07-10 10:20:00 | 133.06 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-07-10 10:05:00 | 133.75 | 2024-07-10 11:45:00 | 133.45 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2024-07-15 10:40:00 | 133.02 | 2024-07-15 11:10:00 | 133.56 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-15 10:40:00 | 133.02 | 2024-07-15 15:20:00 | 134.46 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2024-07-16 10:30:00 | 135.32 | 2024-07-16 10:45:00 | 135.82 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-07-16 10:30:00 | 135.32 | 2024-07-16 11:15:00 | 135.32 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-23 10:55:00 | 126.58 | 2024-07-23 11:15:00 | 126.01 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-23 10:55:00 | 126.58 | 2024-07-23 11:20:00 | 126.58 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-24 11:10:00 | 126.45 | 2024-07-24 15:20:00 | 126.47 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest1 | 2024-07-25 10:15:00 | 124.60 | 2024-07-25 10:25:00 | 123.96 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-07-25 10:15:00 | 124.60 | 2024-07-25 10:35:00 | 124.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:50:00 | 129.47 | 2024-07-26 11:35:00 | 129.11 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-31 09:55:00 | 131.52 | 2024-07-31 10:00:00 | 131.10 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-08-01 11:00:00 | 130.09 | 2024-08-01 11:05:00 | 130.45 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-08 09:55:00 | 121.31 | 2024-08-08 10:00:00 | 120.76 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-08-09 09:55:00 | 125.07 | 2024-08-09 10:10:00 | 124.52 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-08-19 10:45:00 | 124.77 | 2024-08-19 10:55:00 | 124.48 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-20 10:05:00 | 122.88 | 2024-08-20 10:35:00 | 122.38 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-08-20 10:05:00 | 122.88 | 2024-08-20 12:10:00 | 122.71 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2024-08-23 11:00:00 | 130.67 | 2024-08-23 11:15:00 | 131.13 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-08-23 11:00:00 | 130.67 | 2024-08-23 11:35:00 | 130.67 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-26 10:55:00 | 131.33 | 2024-08-26 11:15:00 | 131.65 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-27 10:15:00 | 132.75 | 2024-08-27 10:30:00 | 133.21 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-08-27 10:15:00 | 132.75 | 2024-08-27 10:45:00 | 132.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-30 11:05:00 | 129.50 | 2024-08-30 11:25:00 | 129.79 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-06 10:45:00 | 127.07 | 2024-09-06 12:45:00 | 126.33 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-06 10:45:00 | 127.07 | 2024-09-06 15:20:00 | 125.24 | TARGET_HIT | 0.50 | 1.44% |
| SELL | retest1 | 2024-10-30 10:00:00 | 125.30 | 2024-10-30 10:15:00 | 124.63 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-10-30 10:00:00 | 125.30 | 2024-10-30 15:20:00 | 123.26 | TARGET_HIT | 0.50 | 1.63% |
| BUY | retest1 | 2024-11-06 09:30:00 | 124.58 | 2024-11-06 09:40:00 | 125.25 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-11-06 09:30:00 | 124.58 | 2024-11-06 10:15:00 | 124.81 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2024-11-08 10:25:00 | 120.20 | 2024-11-08 11:00:00 | 119.49 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-11-08 10:25:00 | 120.20 | 2024-11-08 15:20:00 | 117.29 | TARGET_HIT | 0.50 | 2.42% |
| BUY | retest1 | 2024-11-19 10:40:00 | 113.11 | 2024-11-19 11:05:00 | 112.72 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-04 10:45:00 | 108.96 | 2024-12-04 10:50:00 | 109.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-16 09:30:00 | 111.23 | 2024-12-16 09:35:00 | 110.90 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-17 10:40:00 | 111.25 | 2024-12-17 11:25:00 | 110.71 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-12-17 10:40:00 | 111.25 | 2024-12-17 15:20:00 | 109.43 | TARGET_HIT | 0.50 | 1.64% |
| SELL | retest1 | 2024-12-24 09:35:00 | 103.78 | 2024-12-24 09:40:00 | 104.09 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-26 09:35:00 | 104.58 | 2024-12-26 10:10:00 | 104.15 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-26 09:35:00 | 104.58 | 2024-12-26 11:30:00 | 104.58 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 09:55:00 | 106.54 | 2024-12-27 10:10:00 | 107.14 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-12-27 09:55:00 | 106.54 | 2024-12-27 10:30:00 | 106.54 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-30 09:40:00 | 103.67 | 2024-12-30 09:50:00 | 104.01 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-20 11:05:00 | 100.72 | 2025-01-20 12:10:00 | 100.46 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-21 11:00:00 | 98.04 | 2025-01-21 11:40:00 | 98.33 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-22 10:35:00 | 95.15 | 2025-01-22 10:55:00 | 95.54 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-24 09:55:00 | 95.72 | 2025-01-24 10:25:00 | 96.03 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-27 10:05:00 | 92.20 | 2025-01-27 10:15:00 | 91.54 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2025-01-27 10:05:00 | 92.20 | 2025-01-27 12:00:00 | 92.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-28 10:20:00 | 91.38 | 2025-01-28 10:30:00 | 90.76 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2025-01-28 10:20:00 | 91.38 | 2025-01-28 10:40:00 | 91.38 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-31 10:10:00 | 93.41 | 2025-01-31 10:25:00 | 93.82 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-31 10:10:00 | 93.41 | 2025-01-31 14:15:00 | 93.69 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2025-02-07 11:00:00 | 91.90 | 2025-02-07 12:00:00 | 92.45 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-02-07 11:00:00 | 91.90 | 2025-02-07 12:15:00 | 91.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 09:35:00 | 86.65 | 2025-02-20 09:45:00 | 86.32 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-02-21 11:05:00 | 84.45 | 2025-02-21 11:35:00 | 84.82 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-02-25 09:30:00 | 84.11 | 2025-02-25 09:50:00 | 83.75 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-02-27 09:50:00 | 82.38 | 2025-02-27 10:50:00 | 82.64 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-03-18 09:40:00 | 82.47 | 2025-03-18 09:50:00 | 82.19 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-01 09:35:00 | 88.27 | 2025-04-01 10:25:00 | 87.81 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2025-04-03 10:00:00 | 85.40 | 2025-04-03 10:05:00 | 85.81 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-04-25 10:45:00 | 89.26 | 2025-04-25 10:50:00 | 88.65 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2025-04-25 10:45:00 | 89.26 | 2025-04-25 12:10:00 | 89.11 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2025-05-05 09:45:00 | 90.07 | 2025-05-05 10:05:00 | 89.68 | STOP_HIT | 1.00 | -0.44% |
