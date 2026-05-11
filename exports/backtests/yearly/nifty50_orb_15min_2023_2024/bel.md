# BEL (BEL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-01-05 15:25:00 (12239 bars)
- **Last close:** 186.10
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
| PARTIAL | 27 |
| TARGET_HIT | 13 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 95 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 55
- **Target hits / Stop hits / Partials:** 13 / 55 / 27
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 13.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 23 | 41.1% | 7 | 33 | 16 | 0.10% | 5.8% |
| BUY @ 2nd Alert (retest1) | 56 | 23 | 41.1% | 7 | 33 | 16 | 0.10% | 5.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 17 | 43.6% | 6 | 22 | 11 | 0.19% | 7.3% |
| SELL @ 2nd Alert (retest1) | 39 | 17 | 43.6% | 6 | 22 | 11 | 0.19% | 7.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 95 | 40 | 42.1% | 13 | 55 | 27 | 0.14% | 13.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-17 09:50:00 | 109.50 | 108.98 | 0.00 | ORB-long ORB[108.30,109.05] vol=2.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2023-05-17 10:25:00 | 109.22 | 109.19 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 10:30:00 | 108.05 | 108.57 | 0.00 | ORB-short ORB[108.45,109.25] vol=1.5x ATR=0.25 |
| Stop hit — per-position SL triggered | 2023-05-18 10:35:00 | 108.30 | 108.53 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:40:00 | 106.15 | 106.80 | 0.00 | ORB-short ORB[106.75,107.85] vol=1.6x ATR=0.31 |
| Stop hit — per-position SL triggered | 2023-05-19 09:45:00 | 106.46 | 106.68 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-23 09:40:00 | 109.75 | 110.06 | 0.00 | ORB-short ORB[109.90,111.05] vol=3.0x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-23 09:55:00 | 109.30 | 109.86 | 0.00 | T1 1.5R @ 109.30 |
| Target hit | 2023-05-23 15:20:00 | 107.90 | 108.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2023-05-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:30:00 | 110.10 | 109.42 | 0.00 | ORB-long ORB[108.00,109.60] vol=2.3x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 09:35:00 | 110.60 | 109.69 | 0.00 | T1 1.5R @ 110.60 |
| Stop hit — per-position SL triggered | 2023-05-25 09:55:00 | 110.10 | 109.89 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-29 09:30:00 | 110.30 | 110.92 | 0.00 | ORB-short ORB[110.65,111.50] vol=1.5x ATR=0.28 |
| Stop hit — per-position SL triggered | 2023-05-29 09:35:00 | 110.58 | 110.86 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-05-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 10:55:00 | 111.15 | 110.80 | 0.00 | ORB-long ORB[110.20,111.00] vol=1.9x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 11:35:00 | 111.54 | 110.95 | 0.00 | T1 1.5R @ 111.54 |
| Target hit | 2023-05-31 15:20:00 | 112.20 | 111.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2023-06-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 09:40:00 | 114.85 | 114.43 | 0.00 | ORB-long ORB[113.35,114.70] vol=1.5x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 09:50:00 | 115.28 | 114.68 | 0.00 | T1 1.5R @ 115.28 |
| Target hit | 2023-06-05 13:40:00 | 117.15 | 117.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2023-06-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 11:10:00 | 117.15 | 117.28 | 0.00 | ORB-short ORB[117.55,118.30] vol=3.1x ATR=0.23 |
| Stop hit — per-position SL triggered | 2023-06-09 11:40:00 | 117.38 | 117.26 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 11:10:00 | 121.60 | 120.98 | 0.00 | ORB-long ORB[120.05,121.45] vol=8.9x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 11:20:00 | 121.97 | 121.03 | 0.00 | T1 1.5R @ 121.97 |
| Stop hit — per-position SL triggered | 2023-06-15 14:50:00 | 121.60 | 121.75 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 10:00:00 | 123.90 | 122.77 | 0.00 | ORB-long ORB[121.65,122.55] vol=2.0x ATR=0.39 |
| Stop hit — per-position SL triggered | 2023-06-16 10:10:00 | 123.51 | 122.98 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 10:20:00 | 125.50 | 124.99 | 0.00 | ORB-long ORB[124.00,125.30] vol=4.9x ATR=0.34 |
| Stop hit — per-position SL triggered | 2023-06-20 10:25:00 | 125.16 | 125.00 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 09:35:00 | 120.65 | 121.25 | 0.00 | ORB-short ORB[121.05,122.60] vol=2.2x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-27 09:45:00 | 120.02 | 121.01 | 0.00 | T1 1.5R @ 120.02 |
| Target hit | 2023-06-27 15:20:00 | 118.50 | 119.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2023-06-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 11:05:00 | 118.60 | 119.20 | 0.00 | ORB-short ORB[118.85,119.75] vol=3.2x ATR=0.29 |
| Stop hit — per-position SL triggered | 2023-06-28 11:15:00 | 118.89 | 119.04 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 11:15:00 | 124.00 | 124.80 | 0.00 | ORB-short ORB[124.70,126.00] vol=3.3x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 12:50:00 | 123.47 | 124.53 | 0.00 | T1 1.5R @ 123.47 |
| Target hit | 2023-07-03 15:20:00 | 123.95 | 124.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2023-07-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 11:00:00 | 122.90 | 123.34 | 0.00 | ORB-short ORB[123.25,124.20] vol=2.1x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 13:05:00 | 122.49 | 123.05 | 0.00 | T1 1.5R @ 122.49 |
| Target hit | 2023-07-04 15:20:00 | 122.25 | 122.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2023-07-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 10:35:00 | 124.20 | 123.15 | 0.00 | ORB-long ORB[121.90,123.00] vol=2.8x ATR=0.30 |
| Stop hit — per-position SL triggered | 2023-07-06 10:40:00 | 123.90 | 123.21 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:30:00 | 125.05 | 124.52 | 0.00 | ORB-long ORB[123.40,124.75] vol=2.0x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 09:35:00 | 125.62 | 124.80 | 0.00 | T1 1.5R @ 125.62 |
| Target hit | 2023-07-11 10:05:00 | 125.75 | 126.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — SELL (started 2023-07-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-14 10:00:00 | 125.45 | 125.90 | 0.00 | ORB-short ORB[125.50,126.60] vol=1.9x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 10:20:00 | 124.76 | 125.54 | 0.00 | T1 1.5R @ 124.76 |
| Target hit | 2023-07-14 11:25:00 | 125.25 | 125.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — SELL (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-17 11:15:00 | 126.60 | 127.28 | 0.00 | ORB-short ORB[127.00,128.25] vol=1.6x ATR=0.32 |
| Stop hit — per-position SL triggered | 2023-07-17 11:40:00 | 126.92 | 127.26 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 10:45:00 | 125.80 | 126.38 | 0.00 | ORB-short ORB[125.90,127.25] vol=1.7x ATR=0.25 |
| Stop hit — per-position SL triggered | 2023-07-20 11:00:00 | 126.05 | 126.33 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 09:45:00 | 127.30 | 126.42 | 0.00 | ORB-long ORB[125.00,126.35] vol=2.2x ATR=0.31 |
| Stop hit — per-position SL triggered | 2023-07-24 09:50:00 | 126.99 | 126.46 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-07-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-26 10:45:00 | 126.10 | 126.42 | 0.00 | ORB-short ORB[126.30,127.35] vol=2.2x ATR=0.23 |
| Stop hit — per-position SL triggered | 2023-07-26 10:50:00 | 126.33 | 126.42 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 09:35:00 | 129.10 | 129.70 | 0.00 | ORB-short ORB[129.25,130.70] vol=1.5x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 10:30:00 | 128.54 | 129.29 | 0.00 | T1 1.5R @ 128.54 |
| Target hit | 2023-08-02 15:20:00 | 125.45 | 126.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2023-08-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 09:50:00 | 126.95 | 126.16 | 0.00 | ORB-long ORB[125.50,126.75] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2023-08-04 09:55:00 | 126.45 | 126.19 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-08-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 10:35:00 | 129.20 | 128.50 | 0.00 | ORB-long ORB[127.60,128.90] vol=1.7x ATR=0.36 |
| Stop hit — per-position SL triggered | 2023-08-08 10:45:00 | 128.84 | 128.56 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 10:40:00 | 130.85 | 129.81 | 0.00 | ORB-long ORB[129.00,130.50] vol=2.5x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-09 10:45:00 | 131.41 | 130.13 | 0.00 | T1 1.5R @ 131.41 |
| Stop hit — per-position SL triggered | 2023-08-09 10:50:00 | 130.85 | 130.19 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 09:50:00 | 132.55 | 132.21 | 0.00 | ORB-long ORB[131.05,132.35] vol=1.5x ATR=0.42 |
| Stop hit — per-position SL triggered | 2023-08-10 10:20:00 | 132.13 | 132.38 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-08-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 10:05:00 | 131.25 | 130.64 | 0.00 | ORB-long ORB[129.55,130.60] vol=2.5x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 10:15:00 | 131.87 | 130.92 | 0.00 | T1 1.5R @ 131.87 |
| Target hit | 2023-08-11 15:00:00 | 132.25 | 132.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — SELL (started 2023-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-14 09:30:00 | 129.60 | 130.57 | 0.00 | ORB-short ORB[130.00,131.95] vol=1.6x ATR=0.52 |
| Stop hit — per-position SL triggered | 2023-08-14 09:35:00 | 130.12 | 130.51 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 11:15:00 | 129.85 | 130.56 | 0.00 | ORB-short ORB[130.60,131.60] vol=3.6x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 12:10:00 | 129.49 | 130.34 | 0.00 | T1 1.5R @ 129.49 |
| Stop hit — per-position SL triggered | 2023-08-17 12:25:00 | 129.85 | 130.29 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-08-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 09:40:00 | 131.15 | 130.50 | 0.00 | ORB-long ORB[129.35,130.75] vol=2.7x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-22 10:00:00 | 131.64 | 130.82 | 0.00 | T1 1.5R @ 131.64 |
| Stop hit — per-position SL triggered | 2023-08-22 10:10:00 | 131.15 | 130.85 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-08-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 11:05:00 | 133.90 | 134.56 | 0.00 | ORB-short ORB[134.15,135.50] vol=4.6x ATR=0.37 |
| Stop hit — per-position SL triggered | 2023-08-24 11:35:00 | 134.27 | 134.52 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-08-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 10:05:00 | 132.40 | 133.45 | 0.00 | ORB-short ORB[133.45,135.05] vol=1.9x ATR=0.33 |
| Stop hit — per-position SL triggered | 2023-08-31 10:40:00 | 132.73 | 133.18 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 10:00:00 | 136.00 | 135.05 | 0.00 | ORB-long ORB[133.90,135.00] vol=3.2x ATR=0.45 |
| Stop hit — per-position SL triggered | 2023-09-01 10:25:00 | 135.55 | 135.47 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 09:30:00 | 141.40 | 140.18 | 0.00 | ORB-long ORB[139.35,140.55] vol=2.4x ATR=0.62 |
| Stop hit — per-position SL triggered | 2023-09-04 09:35:00 | 140.78 | 140.59 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 10:15:00 | 139.45 | 138.48 | 0.00 | ORB-long ORB[137.60,138.90] vol=2.8x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 10:25:00 | 140.14 | 138.91 | 0.00 | T1 1.5R @ 140.14 |
| Stop hit — per-position SL triggered | 2023-09-07 10:50:00 | 139.45 | 139.17 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-09-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 09:40:00 | 141.85 | 141.25 | 0.00 | ORB-long ORB[140.20,141.45] vol=1.9x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 09:55:00 | 142.48 | 141.52 | 0.00 | T1 1.5R @ 142.48 |
| Target hit | 2023-09-08 15:20:00 | 143.25 | 142.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2023-09-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 10:35:00 | 134.00 | 134.66 | 0.00 | ORB-short ORB[134.85,136.35] vol=1.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2023-09-25 12:10:00 | 134.33 | 134.35 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 10:45:00 | 138.00 | 137.51 | 0.00 | ORB-long ORB[135.90,137.80] vol=1.5x ATR=0.31 |
| Stop hit — per-position SL triggered | 2023-09-26 11:15:00 | 137.69 | 137.62 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 11:15:00 | 137.55 | 136.84 | 0.00 | ORB-long ORB[136.55,137.40] vol=1.7x ATR=0.28 |
| Stop hit — per-position SL triggered | 2023-09-28 11:20:00 | 137.27 | 136.86 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-09-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 11:05:00 | 137.45 | 136.62 | 0.00 | ORB-long ORB[136.00,136.65] vol=1.6x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 12:45:00 | 137.97 | 137.03 | 0.00 | T1 1.5R @ 137.97 |
| Target hit | 2023-09-29 15:20:00 | 138.55 | 137.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2023-10-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 10:25:00 | 139.35 | 138.40 | 0.00 | ORB-long ORB[137.55,139.20] vol=1.6x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-03 10:35:00 | 140.02 | 138.79 | 0.00 | T1 1.5R @ 140.02 |
| Stop hit — per-position SL triggered | 2023-10-03 11:20:00 | 139.35 | 139.21 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 09:30:00 | 139.05 | 138.06 | 0.00 | ORB-long ORB[136.50,138.40] vol=2.1x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 09:45:00 | 139.83 | 138.96 | 0.00 | T1 1.5R @ 139.83 |
| Target hit | 2023-10-10 10:05:00 | 139.35 | 139.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — BUY (started 2023-10-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 11:10:00 | 138.55 | 138.03 | 0.00 | ORB-long ORB[137.40,138.40] vol=1.6x ATR=0.22 |
| Stop hit — per-position SL triggered | 2023-10-12 11:50:00 | 138.33 | 138.14 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 11:05:00 | 138.55 | 138.12 | 0.00 | ORB-long ORB[137.50,138.05] vol=1.8x ATR=0.21 |
| Stop hit — per-position SL triggered | 2023-10-17 11:15:00 | 138.34 | 138.14 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-10-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 11:00:00 | 137.50 | 138.07 | 0.00 | ORB-short ORB[138.15,138.95] vol=5.3x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 11:05:00 | 137.12 | 138.01 | 0.00 | T1 1.5R @ 137.12 |
| Stop hit — per-position SL triggered | 2023-10-18 11:30:00 | 137.50 | 137.88 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-10-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 09:30:00 | 128.25 | 129.00 | 0.00 | ORB-short ORB[128.60,130.20] vol=1.9x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:40:00 | 127.43 | 128.53 | 0.00 | T1 1.5R @ 127.43 |
| Stop hit — per-position SL triggered | 2023-10-26 10:00:00 | 128.25 | 128.19 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-11-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 11:05:00 | 134.35 | 133.82 | 0.00 | ORB-long ORB[133.30,134.00] vol=2.2x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 11:20:00 | 134.78 | 133.97 | 0.00 | T1 1.5R @ 134.78 |
| Stop hit — per-position SL triggered | 2023-11-02 11:35:00 | 134.35 | 134.03 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-11-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 10:40:00 | 136.15 | 135.42 | 0.00 | ORB-long ORB[134.75,135.50] vol=6.1x ATR=0.32 |
| Stop hit — per-position SL triggered | 2023-11-03 10:45:00 | 135.83 | 135.53 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-11-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 11:10:00 | 138.50 | 137.83 | 0.00 | ORB-long ORB[136.70,137.60] vol=2.7x ATR=0.24 |
| Stop hit — per-position SL triggered | 2023-11-07 11:25:00 | 138.26 | 138.05 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-11-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 11:15:00 | 138.10 | 138.70 | 0.00 | ORB-short ORB[138.65,139.95] vol=1.6x ATR=0.22 |
| Stop hit — per-position SL triggered | 2023-11-09 11:35:00 | 138.32 | 138.67 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-11-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 11:00:00 | 138.30 | 137.69 | 0.00 | ORB-long ORB[137.20,138.10] vol=2.4x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-10 12:05:00 | 138.71 | 137.91 | 0.00 | T1 1.5R @ 138.71 |
| Stop hit — per-position SL triggered | 2023-11-10 13:50:00 | 138.30 | 138.15 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 10:15:00 | 144.25 | 143.73 | 0.00 | ORB-long ORB[143.05,144.05] vol=2.1x ATR=0.32 |
| Stop hit — per-position SL triggered | 2023-11-16 10:20:00 | 143.93 | 143.75 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 09:30:00 | 143.30 | 143.78 | 0.00 | ORB-short ORB[143.50,144.55] vol=2.4x ATR=0.30 |
| Stop hit — per-position SL triggered | 2023-11-21 09:35:00 | 143.60 | 143.73 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-11-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-22 11:05:00 | 141.75 | 142.30 | 0.00 | ORB-short ORB[142.00,143.45] vol=1.7x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 11:45:00 | 141.40 | 142.20 | 0.00 | T1 1.5R @ 141.40 |
| Stop hit — per-position SL triggered | 2023-11-22 13:30:00 | 141.75 | 141.80 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-11-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 11:10:00 | 140.20 | 141.02 | 0.00 | ORB-short ORB[141.65,142.55] vol=9.1x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 12:10:00 | 139.78 | 140.58 | 0.00 | T1 1.5R @ 139.78 |
| Stop hit — per-position SL triggered | 2023-11-23 12:50:00 | 140.20 | 140.47 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2023-11-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:55:00 | 139.45 | 139.78 | 0.00 | ORB-short ORB[139.70,140.65] vol=1.9x ATR=0.37 |
| Stop hit — per-position SL triggered | 2023-11-24 10:00:00 | 139.82 | 139.76 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-11-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-29 09:50:00 | 140.90 | 141.32 | 0.00 | ORB-short ORB[141.10,142.65] vol=1.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2023-11-29 10:50:00 | 141.23 | 141.17 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-11-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 10:40:00 | 144.35 | 142.71 | 0.00 | ORB-long ORB[140.75,142.00] vol=2.1x ATR=0.52 |
| Stop hit — per-position SL triggered | 2023-11-30 10:45:00 | 143.83 | 142.81 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 09:45:00 | 149.65 | 148.16 | 0.00 | ORB-long ORB[146.75,148.50] vol=2.2x ATR=0.80 |
| Stop hit — per-position SL triggered | 2023-12-01 09:50:00 | 148.85 | 148.23 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 09:45:00 | 163.00 | 161.47 | 0.00 | ORB-long ORB[159.15,160.95] vol=5.4x ATR=0.64 |
| Stop hit — per-position SL triggered | 2023-12-11 09:50:00 | 162.36 | 161.74 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 11:00:00 | 161.35 | 160.44 | 0.00 | ORB-long ORB[159.40,160.90] vol=1.6x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 11:05:00 | 162.04 | 160.55 | 0.00 | T1 1.5R @ 162.04 |
| Stop hit — per-position SL triggered | 2023-12-13 11:10:00 | 161.35 | 160.57 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2023-12-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:40:00 | 175.40 | 174.34 | 0.00 | ORB-long ORB[173.50,174.80] vol=1.5x ATR=0.47 |
| Stop hit — per-position SL triggered | 2023-12-20 09:50:00 | 174.93 | 174.53 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2023-12-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 11:10:00 | 181.65 | 183.50 | 0.00 | ORB-short ORB[183.10,184.90] vol=1.5x ATR=0.54 |
| Stop hit — per-position SL triggered | 2023-12-27 11:40:00 | 182.19 | 183.33 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-12-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 09:55:00 | 183.05 | 182.05 | 0.00 | ORB-long ORB[181.10,182.20] vol=2.3x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-12-28 10:00:00 | 182.39 | 182.08 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2023-12-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 10:35:00 | 183.75 | 182.71 | 0.00 | ORB-long ORB[181.40,183.35] vol=4.8x ATR=0.75 |
| Stop hit — per-position SL triggered | 2023-12-29 10:45:00 | 183.00 | 182.77 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-01-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 09:50:00 | 184.60 | 184.06 | 0.00 | ORB-long ORB[182.50,183.90] vol=5.4x ATR=0.57 |
| Stop hit — per-position SL triggered | 2024-01-04 09:55:00 | 184.03 | 184.12 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-17 09:50:00 | 109.50 | 2023-05-17 10:25:00 | 109.22 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-05-18 10:30:00 | 108.05 | 2023-05-18 10:35:00 | 108.30 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-05-19 09:40:00 | 106.15 | 2023-05-19 09:45:00 | 106.46 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-05-23 09:40:00 | 109.75 | 2023-05-23 09:55:00 | 109.30 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-05-23 09:40:00 | 109.75 | 2023-05-23 15:20:00 | 107.90 | TARGET_HIT | 0.50 | 1.69% |
| BUY | retest1 | 2023-05-25 09:30:00 | 110.10 | 2023-05-25 09:35:00 | 110.60 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-05-25 09:30:00 | 110.10 | 2023-05-25 09:55:00 | 110.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-29 09:30:00 | 110.30 | 2023-05-29 09:35:00 | 110.58 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-05-31 10:55:00 | 111.15 | 2023-05-31 11:35:00 | 111.54 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-05-31 10:55:00 | 111.15 | 2023-05-31 15:20:00 | 112.20 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2023-06-05 09:40:00 | 114.85 | 2023-06-05 09:50:00 | 115.28 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-06-05 09:40:00 | 114.85 | 2023-06-05 13:40:00 | 117.15 | TARGET_HIT | 0.50 | 2.00% |
| SELL | retest1 | 2023-06-09 11:10:00 | 117.15 | 2023-06-09 11:40:00 | 117.38 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-06-15 11:10:00 | 121.60 | 2023-06-15 11:20:00 | 121.97 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-06-15 11:10:00 | 121.60 | 2023-06-15 14:50:00 | 121.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-16 10:00:00 | 123.90 | 2023-06-16 10:10:00 | 123.51 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-06-20 10:20:00 | 125.50 | 2023-06-20 10:25:00 | 125.16 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-06-27 09:35:00 | 120.65 | 2023-06-27 09:45:00 | 120.02 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-06-27 09:35:00 | 120.65 | 2023-06-27 15:20:00 | 118.50 | TARGET_HIT | 0.50 | 1.78% |
| SELL | retest1 | 2023-06-28 11:05:00 | 118.60 | 2023-06-28 11:15:00 | 118.89 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-07-03 11:15:00 | 124.00 | 2023-07-03 12:50:00 | 123.47 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-07-03 11:15:00 | 124.00 | 2023-07-03 15:20:00 | 123.95 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2023-07-04 11:00:00 | 122.90 | 2023-07-04 13:05:00 | 122.49 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-07-04 11:00:00 | 122.90 | 2023-07-04 15:20:00 | 122.25 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2023-07-06 10:35:00 | 124.20 | 2023-07-06 10:40:00 | 123.90 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-11 09:30:00 | 125.05 | 2023-07-11 09:35:00 | 125.62 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-07-11 09:30:00 | 125.05 | 2023-07-11 10:05:00 | 125.75 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2023-07-14 10:00:00 | 125.45 | 2023-07-14 10:20:00 | 124.76 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2023-07-14 10:00:00 | 125.45 | 2023-07-14 11:25:00 | 125.25 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2023-07-17 11:15:00 | 126.60 | 2023-07-17 11:40:00 | 126.92 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-07-20 10:45:00 | 125.80 | 2023-07-20 11:00:00 | 126.05 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-24 09:45:00 | 127.30 | 2023-07-24 09:50:00 | 126.99 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-07-26 10:45:00 | 126.10 | 2023-07-26 10:50:00 | 126.33 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-08-02 09:35:00 | 129.10 | 2023-08-02 10:30:00 | 128.54 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-08-02 09:35:00 | 129.10 | 2023-08-02 15:20:00 | 125.45 | TARGET_HIT | 0.50 | 2.83% |
| BUY | retest1 | 2023-08-04 09:50:00 | 126.95 | 2023-08-04 09:55:00 | 126.45 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-08-08 10:35:00 | 129.20 | 2023-08-08 10:45:00 | 128.84 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-08-09 10:40:00 | 130.85 | 2023-08-09 10:45:00 | 131.41 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-08-09 10:40:00 | 130.85 | 2023-08-09 10:50:00 | 130.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-10 09:50:00 | 132.55 | 2023-08-10 10:20:00 | 132.13 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-08-11 10:05:00 | 131.25 | 2023-08-11 10:15:00 | 131.87 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-08-11 10:05:00 | 131.25 | 2023-08-11 15:00:00 | 132.25 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2023-08-14 09:30:00 | 129.60 | 2023-08-14 09:35:00 | 130.12 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-08-17 11:15:00 | 129.85 | 2023-08-17 12:10:00 | 129.49 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-08-17 11:15:00 | 129.85 | 2023-08-17 12:25:00 | 129.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-22 09:40:00 | 131.15 | 2023-08-22 10:00:00 | 131.64 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-08-22 09:40:00 | 131.15 | 2023-08-22 10:10:00 | 131.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-24 11:05:00 | 133.90 | 2023-08-24 11:35:00 | 134.27 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-08-31 10:05:00 | 132.40 | 2023-08-31 10:40:00 | 132.73 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-01 10:00:00 | 136.00 | 2023-09-01 10:25:00 | 135.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-09-04 09:30:00 | 141.40 | 2023-09-04 09:35:00 | 140.78 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-09-07 10:15:00 | 139.45 | 2023-09-07 10:25:00 | 140.14 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-09-07 10:15:00 | 139.45 | 2023-09-07 10:50:00 | 139.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-08 09:40:00 | 141.85 | 2023-09-08 09:55:00 | 142.48 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-09-08 09:40:00 | 141.85 | 2023-09-08 15:20:00 | 143.25 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2023-09-25 10:35:00 | 134.00 | 2023-09-25 12:10:00 | 134.33 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-26 10:45:00 | 138.00 | 2023-09-26 11:15:00 | 137.69 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-09-28 11:15:00 | 137.55 | 2023-09-28 11:20:00 | 137.27 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-09-29 11:05:00 | 137.45 | 2023-09-29 12:45:00 | 137.97 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-09-29 11:05:00 | 137.45 | 2023-09-29 15:20:00 | 138.55 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2023-10-03 10:25:00 | 139.35 | 2023-10-03 10:35:00 | 140.02 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-10-03 10:25:00 | 139.35 | 2023-10-03 11:20:00 | 139.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-10 09:30:00 | 139.05 | 2023-10-10 09:45:00 | 139.83 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-10-10 09:30:00 | 139.05 | 2023-10-10 10:05:00 | 139.35 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2023-10-12 11:10:00 | 138.55 | 2023-10-12 11:50:00 | 138.33 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-10-17 11:05:00 | 138.55 | 2023-10-17 11:15:00 | 138.34 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-10-18 11:00:00 | 137.50 | 2023-10-18 11:05:00 | 137.12 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-10-18 11:00:00 | 137.50 | 2023-10-18 11:30:00 | 137.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-26 09:30:00 | 128.25 | 2023-10-26 09:40:00 | 127.43 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2023-10-26 09:30:00 | 128.25 | 2023-10-26 10:00:00 | 128.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-02 11:05:00 | 134.35 | 2023-11-02 11:20:00 | 134.78 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-11-02 11:05:00 | 134.35 | 2023-11-02 11:35:00 | 134.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-03 10:40:00 | 136.15 | 2023-11-03 10:45:00 | 135.83 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-07 11:10:00 | 138.50 | 2023-11-07 11:25:00 | 138.26 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-11-09 11:15:00 | 138.10 | 2023-11-09 11:35:00 | 138.32 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-11-10 11:00:00 | 138.30 | 2023-11-10 12:05:00 | 138.71 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-11-10 11:00:00 | 138.30 | 2023-11-10 13:50:00 | 138.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-16 10:15:00 | 144.25 | 2023-11-16 10:20:00 | 143.93 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-11-21 09:30:00 | 143.30 | 2023-11-21 09:35:00 | 143.60 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-11-22 11:05:00 | 141.75 | 2023-11-22 11:45:00 | 141.40 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-11-22 11:05:00 | 141.75 | 2023-11-22 13:30:00 | 141.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-23 11:10:00 | 140.20 | 2023-11-23 12:10:00 | 139.78 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-11-23 11:10:00 | 140.20 | 2023-11-23 12:50:00 | 140.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-24 09:55:00 | 139.45 | 2023-11-24 10:00:00 | 139.82 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-11-29 09:50:00 | 140.90 | 2023-11-29 10:50:00 | 141.23 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-30 10:40:00 | 144.35 | 2023-11-30 10:45:00 | 143.83 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-12-01 09:45:00 | 149.65 | 2023-12-01 09:50:00 | 148.85 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2023-12-11 09:45:00 | 163.00 | 2023-12-11 09:50:00 | 162.36 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-12-13 11:00:00 | 161.35 | 2023-12-13 11:05:00 | 162.04 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-12-13 11:00:00 | 161.35 | 2023-12-13 11:10:00 | 161.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-20 09:40:00 | 175.40 | 2023-12-20 09:50:00 | 174.93 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-12-27 11:10:00 | 181.65 | 2023-12-27 11:40:00 | 182.19 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-12-28 09:55:00 | 183.05 | 2023-12-28 10:00:00 | 182.39 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-12-29 10:35:00 | 183.75 | 2023-12-29 10:45:00 | 183.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-01-04 09:50:00 | 184.60 | 2024-01-04 09:55:00 | 184.03 | STOP_HIT | 1.00 | -0.31% |
