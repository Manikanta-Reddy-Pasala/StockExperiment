# TATASTEEL (TATASTEEL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55356 bars)
- **Last close:** 214.60
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
| ENTRY1 | 92 |
| ENTRY2 | 0 |
| PARTIAL | 48 |
| TARGET_HIT | 15 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 140 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 63 / 77
- **Target hits / Stop hits / Partials:** 15 / 77 / 48
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 22.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 82 | 41 | 50.0% | 10 | 41 | 31 | 0.24% | 19.3% |
| BUY @ 2nd Alert (retest1) | 82 | 41 | 50.0% | 10 | 41 | 31 | 0.24% | 19.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 58 | 22 | 37.9% | 5 | 36 | 17 | 0.06% | 3.5% |
| SELL @ 2nd Alert (retest1) | 58 | 22 | 37.9% | 5 | 36 | 17 | 0.06% | 3.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 140 | 63 | 45.0% | 15 | 77 | 48 | 0.16% | 22.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 10:00:00 | 104.50 | 104.84 | 0.00 | ORB-short ORB[104.65,105.70] vol=1.7x ATR=0.24 |
| Stop hit — per-position SL triggered | 2023-05-19 10:10:00 | 104.74 | 104.83 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 11:00:00 | 105.70 | 105.32 | 0.00 | ORB-long ORB[105.00,105.45] vol=2.1x ATR=0.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-23 12:55:00 | 105.90 | 105.48 | 0.00 | T1 1.5R @ 105.90 |
| Stop hit — per-position SL triggered | 2023-05-23 13:10:00 | 105.70 | 105.49 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 11:15:00 | 104.45 | 104.77 | 0.00 | ORB-short ORB[104.70,105.20] vol=3.0x ATR=0.16 |
| Stop hit — per-position SL triggered | 2023-05-25 12:25:00 | 104.61 | 104.70 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 10:15:00 | 107.35 | 106.89 | 0.00 | ORB-long ORB[106.50,107.30] vol=1.7x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 10:45:00 | 107.71 | 107.09 | 0.00 | T1 1.5R @ 107.71 |
| Target hit | 2023-05-29 15:20:00 | 108.40 | 107.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2023-06-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 11:10:00 | 108.35 | 109.05 | 0.00 | ORB-short ORB[109.10,109.85] vol=2.1x ATR=0.19 |
| Stop hit — per-position SL triggered | 2023-06-06 11:20:00 | 108.54 | 108.98 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 11:00:00 | 112.75 | 112.16 | 0.00 | ORB-long ORB[111.15,112.70] vol=1.5x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-06-08 11:25:00 | 112.49 | 112.25 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 10:45:00 | 110.05 | 109.63 | 0.00 | ORB-long ORB[109.15,109.85] vol=2.0x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-12 10:55:00 | 110.35 | 109.74 | 0.00 | T1 1.5R @ 110.35 |
| Stop hit — per-position SL triggered | 2023-06-12 12:20:00 | 110.05 | 109.90 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 10:00:00 | 113.20 | 112.68 | 0.00 | ORB-long ORB[111.40,113.00] vol=2.2x ATR=0.31 |
| Stop hit — per-position SL triggered | 2023-06-14 10:20:00 | 112.89 | 112.80 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 11:00:00 | 114.10 | 114.59 | 0.00 | ORB-short ORB[114.30,115.60] vol=1.8x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-19 11:50:00 | 113.80 | 114.47 | 0.00 | T1 1.5R @ 113.80 |
| Stop hit — per-position SL triggered | 2023-06-19 14:55:00 | 114.10 | 114.08 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 09:55:00 | 114.95 | 114.29 | 0.00 | ORB-long ORB[113.50,114.25] vol=1.6x ATR=0.29 |
| Stop hit — per-position SL triggered | 2023-06-20 10:30:00 | 114.66 | 114.46 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:05:00 | 113.40 | 113.81 | 0.00 | ORB-short ORB[113.80,114.90] vol=1.5x ATR=0.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-21 11:15:00 | 113.14 | 113.76 | 0.00 | T1 1.5R @ 113.14 |
| Stop hit — per-position SL triggered | 2023-06-21 12:20:00 | 113.40 | 113.60 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 11:15:00 | 113.90 | 113.30 | 0.00 | ORB-long ORB[112.05,113.70] vol=3.2x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 11:20:00 | 114.22 | 113.43 | 0.00 | T1 1.5R @ 114.22 |
| Stop hit — per-position SL triggered | 2023-07-03 11:25:00 | 113.90 | 113.45 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-07-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 10:20:00 | 112.45 | 111.85 | 0.00 | ORB-long ORB[111.50,112.25] vol=1.9x ATR=0.24 |
| Stop hit — per-position SL triggered | 2023-07-06 10:25:00 | 112.21 | 111.86 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 09:30:00 | 112.60 | 112.21 | 0.00 | ORB-long ORB[111.65,112.45] vol=1.7x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 10:05:00 | 112.96 | 112.81 | 0.00 | T1 1.5R @ 112.96 |
| Target hit | 2023-07-10 15:20:00 | 115.40 | 114.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2023-07-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:40:00 | 118.40 | 117.83 | 0.00 | ORB-long ORB[117.10,118.00] vol=2.3x ATR=0.34 |
| Stop hit — per-position SL triggered | 2023-07-17 09:45:00 | 118.06 | 117.87 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-21 09:35:00 | 115.70 | 116.11 | 0.00 | ORB-short ORB[115.95,116.70] vol=2.3x ATR=0.23 |
| Stop hit — per-position SL triggered | 2023-07-21 09:45:00 | 115.93 | 116.01 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-24 10:30:00 | 115.05 | 115.53 | 0.00 | ORB-short ORB[115.25,116.50] vol=2.2x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 11:55:00 | 114.62 | 115.35 | 0.00 | T1 1.5R @ 114.62 |
| Stop hit — per-position SL triggered | 2023-07-24 12:35:00 | 115.05 | 115.25 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-08-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 09:35:00 | 117.80 | 118.34 | 0.00 | ORB-short ORB[118.10,119.00] vol=2.0x ATR=0.39 |
| Stop hit — per-position SL triggered | 2023-08-03 09:40:00 | 118.19 | 118.31 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-08-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:20:00 | 117.60 | 117.98 | 0.00 | ORB-short ORB[117.80,118.50] vol=2.7x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-09 10:50:00 | 117.23 | 117.77 | 0.00 | T1 1.5R @ 117.23 |
| Stop hit — per-position SL triggered | 2023-08-09 13:10:00 | 117.60 | 117.60 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-08-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 10:25:00 | 120.50 | 119.82 | 0.00 | ORB-long ORB[119.20,120.30] vol=1.7x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 10:30:00 | 120.99 | 119.94 | 0.00 | T1 1.5R @ 120.99 |
| Stop hit — per-position SL triggered | 2023-08-11 10:55:00 | 120.50 | 120.01 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-08-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-16 10:55:00 | 116.30 | 116.84 | 0.00 | ORB-short ORB[116.40,117.95] vol=1.6x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-16 11:30:00 | 115.88 | 116.65 | 0.00 | T1 1.5R @ 115.88 |
| Target hit | 2023-08-16 15:20:00 | 115.90 | 116.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 11:15:00 | 115.30 | 116.03 | 0.00 | ORB-short ORB[115.70,116.65] vol=1.7x ATR=0.17 |
| Stop hit — per-position SL triggered | 2023-08-17 11:20:00 | 115.47 | 116.00 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 09:50:00 | 116.65 | 117.01 | 0.00 | ORB-short ORB[116.75,117.30] vol=1.8x ATR=0.22 |
| Stop hit — per-position SL triggered | 2023-08-22 09:55:00 | 116.87 | 116.97 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-08-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 10:15:00 | 120.00 | 119.62 | 0.00 | ORB-long ORB[119.10,119.80] vol=1.6x ATR=0.21 |
| Stop hit — per-position SL triggered | 2023-08-24 10:20:00 | 119.79 | 119.66 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 10:10:00 | 118.55 | 118.31 | 0.00 | ORB-long ORB[117.75,118.50] vol=2.1x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-08-29 10:20:00 | 118.29 | 118.33 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 10:15:00 | 122.00 | 121.38 | 0.00 | ORB-long ORB[120.25,121.65] vol=2.5x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 10:45:00 | 122.44 | 121.62 | 0.00 | T1 1.5R @ 122.44 |
| Stop hit — per-position SL triggered | 2023-08-30 11:35:00 | 122.00 | 121.77 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-31 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 10:20:00 | 123.30 | 122.74 | 0.00 | ORB-long ORB[122.20,123.10] vol=2.1x ATR=0.32 |
| Stop hit — per-position SL triggered | 2023-08-31 10:25:00 | 122.98 | 122.76 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-06 11:15:00 | 129.15 | 130.26 | 0.00 | ORB-short ORB[130.65,132.40] vol=1.5x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 11:40:00 | 128.68 | 130.05 | 0.00 | T1 1.5R @ 128.68 |
| Stop hit — per-position SL triggered | 2023-09-06 12:05:00 | 129.15 | 129.96 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 10:20:00 | 130.20 | 130.66 | 0.00 | ORB-short ORB[130.50,131.65] vol=1.9x ATR=0.33 |
| Stop hit — per-position SL triggered | 2023-09-08 11:30:00 | 130.53 | 130.46 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-09-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 10:20:00 | 130.95 | 129.87 | 0.00 | ORB-long ORB[129.20,130.00] vol=2.0x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 10:25:00 | 131.46 | 130.09 | 0.00 | T1 1.5R @ 131.46 |
| Stop hit — per-position SL triggered | 2023-09-11 11:10:00 | 130.95 | 130.37 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-09-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:45:00 | 129.25 | 131.37 | 0.00 | ORB-short ORB[131.45,133.15] vol=2.0x ATR=0.47 |
| Stop hit — per-position SL triggered | 2023-09-12 09:55:00 | 129.72 | 130.82 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:45:00 | 132.40 | 131.82 | 0.00 | ORB-long ORB[130.80,132.30] vol=2.1x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 09:50:00 | 133.07 | 132.41 | 0.00 | T1 1.5R @ 133.07 |
| Stop hit — per-position SL triggered | 2023-09-14 10:15:00 | 132.40 | 132.82 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-09-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-15 09:30:00 | 131.75 | 132.28 | 0.00 | ORB-short ORB[132.05,133.30] vol=2.5x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 09:50:00 | 131.05 | 132.00 | 0.00 | T1 1.5R @ 131.05 |
| Target hit | 2023-09-15 13:55:00 | 131.20 | 131.10 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — SELL (started 2023-09-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:55:00 | 126.00 | 126.96 | 0.00 | ORB-short ORB[126.75,127.85] vol=1.7x ATR=0.40 |
| Stop hit — per-position SL triggered | 2023-09-22 10:00:00 | 126.40 | 126.90 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 09:45:00 | 130.40 | 129.51 | 0.00 | ORB-long ORB[128.40,130.00] vol=2.6x ATR=0.49 |
| Stop hit — per-position SL triggered | 2023-09-26 10:30:00 | 129.91 | 129.77 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-10-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-04 09:30:00 | 125.80 | 126.29 | 0.00 | ORB-short ORB[125.85,127.10] vol=1.7x ATR=0.33 |
| Stop hit — per-position SL triggered | 2023-10-04 09:35:00 | 126.13 | 126.26 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-10-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 09:40:00 | 124.85 | 125.53 | 0.00 | ORB-short ORB[125.50,126.10] vol=2.8x ATR=0.35 |
| Stop hit — per-position SL triggered | 2023-10-05 10:45:00 | 125.20 | 125.10 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-10-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 09:30:00 | 126.50 | 125.80 | 0.00 | ORB-long ORB[125.10,125.70] vol=2.1x ATR=0.27 |
| Stop hit — per-position SL triggered | 2023-10-12 09:55:00 | 126.23 | 126.21 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 10:40:00 | 124.45 | 124.72 | 0.00 | ORB-short ORB[124.55,125.50] vol=3.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2023-10-13 10:50:00 | 124.73 | 124.71 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-10-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 11:10:00 | 127.25 | 127.95 | 0.00 | ORB-short ORB[127.50,128.70] vol=1.9x ATR=0.28 |
| Stop hit — per-position SL triggered | 2023-10-18 11:25:00 | 127.53 | 127.91 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-10-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 09:55:00 | 124.95 | 125.26 | 0.00 | ORB-short ORB[125.00,125.75] vol=1.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2023-10-20 10:00:00 | 125.23 | 125.25 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 09:30:00 | 119.40 | 119.80 | 0.00 | ORB-short ORB[119.65,120.15] vol=1.6x ATR=0.29 |
| Stop hit — per-position SL triggered | 2023-10-31 09:35:00 | 119.69 | 119.77 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-11-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 10:45:00 | 119.75 | 119.20 | 0.00 | ORB-long ORB[118.40,119.30] vol=1.6x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 11:20:00 | 120.18 | 119.42 | 0.00 | T1 1.5R @ 120.18 |
| Stop hit — per-position SL triggered | 2023-11-07 11:40:00 | 119.75 | 119.49 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-11-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-08 11:05:00 | 118.95 | 119.46 | 0.00 | ORB-short ORB[119.55,120.45] vol=2.0x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 11:20:00 | 118.64 | 119.30 | 0.00 | T1 1.5R @ 118.64 |
| Stop hit — per-position SL triggered | 2023-11-08 11:45:00 | 118.95 | 119.21 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-11-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 10:00:00 | 120.95 | 120.52 | 0.00 | ORB-long ORB[119.90,120.90] vol=1.9x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-11-13 10:20:00 | 120.69 | 120.61 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-11-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:15:00 | 124.25 | 123.58 | 0.00 | ORB-long ORB[123.05,124.00] vol=2.1x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 11:25:00 | 124.58 | 123.70 | 0.00 | T1 1.5R @ 124.58 |
| Target hit | 2023-11-16 15:10:00 | 124.45 | 124.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2023-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:30:00 | 126.15 | 125.72 | 0.00 | ORB-long ORB[124.95,126.10] vol=1.7x ATR=0.30 |
| Stop hit — per-position SL triggered | 2023-11-21 09:35:00 | 125.85 | 125.76 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-11-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 11:05:00 | 126.80 | 126.34 | 0.00 | ORB-long ORB[126.05,126.60] vol=2.4x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 12:25:00 | 127.16 | 126.57 | 0.00 | T1 1.5R @ 127.16 |
| Stop hit — per-position SL triggered | 2023-11-23 14:05:00 | 126.80 | 126.77 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-11-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 10:00:00 | 126.00 | 126.45 | 0.00 | ORB-short ORB[126.40,127.00] vol=1.5x ATR=0.26 |
| Stop hit — per-position SL triggered | 2023-11-24 11:25:00 | 126.26 | 126.26 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 10:15:00 | 127.65 | 127.11 | 0.00 | ORB-long ORB[126.70,127.45] vol=2.5x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 10:20:00 | 128.00 | 127.26 | 0.00 | T1 1.5R @ 128.00 |
| Stop hit — per-position SL triggered | 2023-11-29 10:40:00 | 127.65 | 127.35 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-11-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 10:50:00 | 128.00 | 127.26 | 0.00 | ORB-long ORB[127.00,127.90] vol=1.6x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 12:00:00 | 128.42 | 127.56 | 0.00 | T1 1.5R @ 128.42 |
| Stop hit — per-position SL triggered | 2023-11-30 12:30:00 | 128.00 | 127.62 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-12-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 10:50:00 | 131.35 | 130.74 | 0.00 | ORB-long ORB[130.00,131.10] vol=1.6x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 11:00:00 | 131.72 | 131.04 | 0.00 | T1 1.5R @ 131.72 |
| Stop hit — per-position SL triggered | 2023-12-05 11:45:00 | 131.35 | 131.34 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-12-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 10:20:00 | 131.00 | 131.29 | 0.00 | ORB-short ORB[131.15,132.50] vol=2.1x ATR=0.29 |
| Stop hit — per-position SL triggered | 2023-12-06 10:55:00 | 131.29 | 131.25 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-12-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 09:30:00 | 131.20 | 130.61 | 0.00 | ORB-long ORB[129.75,130.70] vol=2.0x ATR=0.29 |
| Stop hit — per-position SL triggered | 2023-12-08 09:35:00 | 130.91 | 130.68 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-12-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 10:00:00 | 133.10 | 132.36 | 0.00 | ORB-long ORB[131.50,132.70] vol=2.7x ATR=0.35 |
| Stop hit — per-position SL triggered | 2023-12-14 10:05:00 | 132.75 | 132.41 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-12-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 10:10:00 | 135.00 | 134.26 | 0.00 | ORB-long ORB[133.00,134.50] vol=2.8x ATR=0.41 |
| Stop hit — per-position SL triggered | 2023-12-15 10:15:00 | 134.59 | 134.29 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-12-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 09:40:00 | 135.15 | 135.59 | 0.00 | ORB-short ORB[135.45,136.15] vol=1.6x ATR=0.31 |
| Stop hit — per-position SL triggered | 2023-12-20 09:45:00 | 135.46 | 135.57 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-12-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:30:00 | 133.50 | 132.88 | 0.00 | ORB-long ORB[131.75,133.20] vol=2.9x ATR=0.47 |
| Stop hit — per-position SL triggered | 2023-12-22 09:35:00 | 133.03 | 132.94 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:40:00 | 136.60 | 136.16 | 0.00 | ORB-long ORB[135.50,136.35] vol=1.7x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 09:55:00 | 137.07 | 136.52 | 0.00 | T1 1.5R @ 137.07 |
| Target hit | 2023-12-27 12:10:00 | 137.50 | 137.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — SELL (started 2023-12-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 09:30:00 | 137.50 | 138.11 | 0.00 | ORB-short ORB[137.55,138.75] vol=1.9x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 09:40:00 | 136.89 | 137.85 | 0.00 | T1 1.5R @ 136.89 |
| Stop hit — per-position SL triggered | 2023-12-28 09:45:00 | 137.50 | 137.81 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 10:45:00 | 139.55 | 138.02 | 0.00 | ORB-long ORB[137.15,138.85] vol=4.9x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 10:55:00 | 140.23 | 138.56 | 0.00 | T1 1.5R @ 140.23 |
| Stop hit — per-position SL triggered | 2023-12-29 11:00:00 | 139.55 | 138.69 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 139.05 | 139.89 | 0.00 | ORB-short ORB[139.55,140.60] vol=1.7x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 10:20:00 | 138.45 | 139.67 | 0.00 | T1 1.5R @ 138.45 |
| Stop hit — per-position SL triggered | 2024-01-02 12:25:00 | 139.05 | 138.87 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-01-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 09:45:00 | 134.05 | 134.26 | 0.00 | ORB-short ORB[134.20,135.00] vol=1.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2024-01-05 09:50:00 | 134.33 | 134.56 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-01-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:50:00 | 133.90 | 133.27 | 0.00 | ORB-long ORB[133.00,133.65] vol=1.8x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 12:20:00 | 134.38 | 133.58 | 0.00 | T1 1.5R @ 134.38 |
| Stop hit — per-position SL triggered | 2024-01-09 14:15:00 | 133.90 | 133.86 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-01-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:45:00 | 135.05 | 135.68 | 0.00 | ORB-short ORB[135.75,136.25] vol=1.7x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 09:55:00 | 134.53 | 135.49 | 0.00 | T1 1.5R @ 134.53 |
| Stop hit — per-position SL triggered | 2024-01-15 10:00:00 | 135.05 | 135.44 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 09:55:00 | 134.10 | 134.60 | 0.00 | ORB-short ORB[134.30,135.20] vol=1.6x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 10:05:00 | 133.56 | 134.48 | 0.00 | T1 1.5R @ 133.56 |
| Stop hit — per-position SL triggered | 2024-01-23 10:10:00 | 134.10 | 134.43 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-01-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 10:45:00 | 132.05 | 131.54 | 0.00 | ORB-long ORB[130.10,131.85] vol=1.6x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-24 11:10:00 | 132.78 | 131.71 | 0.00 | T1 1.5R @ 132.78 |
| Stop hit — per-position SL triggered | 2024-01-24 11:30:00 | 132.05 | 131.83 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-01-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 10:30:00 | 137.05 | 136.41 | 0.00 | ORB-long ORB[135.40,136.85] vol=2.3x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 10:45:00 | 137.62 | 136.60 | 0.00 | T1 1.5R @ 137.62 |
| Stop hit — per-position SL triggered | 2024-01-30 10:50:00 | 137.05 | 136.61 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-02-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 09:40:00 | 136.65 | 136.13 | 0.00 | ORB-long ORB[134.80,136.40] vol=2.6x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 11:05:00 | 137.25 | 136.59 | 0.00 | T1 1.5R @ 137.25 |
| Stop hit — per-position SL triggered | 2024-02-02 12:30:00 | 136.65 | 136.87 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-02-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:00:00 | 143.55 | 144.83 | 0.00 | ORB-short ORB[144.50,145.50] vol=1.7x ATR=0.52 |
| Stop hit — per-position SL triggered | 2024-02-08 11:45:00 | 144.07 | 144.51 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-02-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 09:40:00 | 141.95 | 143.04 | 0.00 | ORB-short ORB[142.90,144.45] vol=1.6x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:50:00 | 141.14 | 142.76 | 0.00 | T1 1.5R @ 141.14 |
| Target hit | 2024-02-09 12:20:00 | 141.25 | 141.20 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — BUY (started 2024-02-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-14 11:10:00 | 138.20 | 137.33 | 0.00 | ORB-long ORB[135.90,137.35] vol=1.9x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-14 12:25:00 | 138.85 | 137.55 | 0.00 | T1 1.5R @ 138.85 |
| Target hit | 2024-02-14 15:20:00 | 141.30 | 138.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2024-02-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 10:20:00 | 143.10 | 142.37 | 0.00 | ORB-long ORB[141.65,142.75] vol=1.6x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-02-16 12:25:00 | 142.67 | 142.74 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-02-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 09:50:00 | 140.85 | 141.35 | 0.00 | ORB-short ORB[141.25,142.25] vol=1.6x ATR=0.36 |
| Stop hit — per-position SL triggered | 2024-02-20 12:05:00 | 141.21 | 141.02 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-02-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:30:00 | 143.95 | 143.00 | 0.00 | ORB-long ORB[141.45,143.20] vol=4.2x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 09:35:00 | 144.68 | 143.35 | 0.00 | T1 1.5R @ 144.68 |
| Target hit | 2024-02-21 10:40:00 | 144.30 | 144.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 76 — BUY (started 2024-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 10:40:00 | 143.50 | 142.85 | 0.00 | ORB-long ORB[142.00,143.15] vol=2.2x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-02-27 11:50:00 | 143.16 | 143.05 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-03-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 10:55:00 | 146.25 | 145.05 | 0.00 | ORB-long ORB[143.00,145.10] vol=3.6x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-01 11:20:00 | 146.99 | 145.62 | 0.00 | T1 1.5R @ 146.99 |
| Target hit | 2024-03-01 15:20:00 | 150.65 | 147.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — SELL (started 2024-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:30:00 | 150.50 | 151.45 | 0.00 | ORB-short ORB[151.10,152.65] vol=1.9x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:35:00 | 149.76 | 151.16 | 0.00 | T1 1.5R @ 149.76 |
| Target hit | 2024-03-06 13:50:00 | 150.20 | 149.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 79 — BUY (started 2024-03-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 09:30:00 | 155.20 | 153.97 | 0.00 | ORB-long ORB[152.25,154.20] vol=3.6x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 09:35:00 | 156.13 | 154.55 | 0.00 | T1 1.5R @ 156.13 |
| Target hit | 2024-03-07 10:25:00 | 155.50 | 155.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 80 — BUY (started 2024-03-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 11:10:00 | 151.55 | 150.23 | 0.00 | ORB-long ORB[148.35,150.45] vol=1.6x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 11:40:00 | 152.28 | 150.57 | 0.00 | T1 1.5R @ 152.28 |
| Stop hit — per-position SL triggered | 2024-03-22 15:10:00 | 151.55 | 151.28 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-03-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 10:55:00 | 154.50 | 153.66 | 0.00 | ORB-long ORB[152.70,154.05] vol=2.9x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 11:05:00 | 155.09 | 153.85 | 0.00 | T1 1.5R @ 155.09 |
| Target hit | 2024-03-28 15:20:00 | 156.30 | 155.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 162.10 | 164.39 | 0.00 | ORB-short ORB[164.55,165.95] vol=1.9x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 11:10:00 | 161.28 | 163.73 | 0.00 | T1 1.5R @ 161.28 |
| Stop hit — per-position SL triggered | 2024-04-04 11:30:00 | 162.10 | 163.49 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-04-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 09:55:00 | 167.80 | 166.38 | 0.00 | ORB-long ORB[165.20,166.20] vol=4.1x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-09 10:00:00 | 168.64 | 166.97 | 0.00 | T1 1.5R @ 168.64 |
| Stop hit — per-position SL triggered | 2024-04-09 12:45:00 | 167.80 | 168.16 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 11:15:00 | 165.35 | 168.00 | 0.00 | ORB-short ORB[167.30,169.20] vol=1.6x ATR=0.52 |
| Stop hit — per-position SL triggered | 2024-04-10 11:35:00 | 165.87 | 167.77 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:15:00 | 163.40 | 162.74 | 0.00 | ORB-long ORB[161.45,163.00] vol=1.7x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 10:55:00 | 163.93 | 162.97 | 0.00 | T1 1.5R @ 163.93 |
| Target hit | 2024-04-24 15:20:00 | 165.80 | 164.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — SELL (started 2024-04-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:50:00 | 164.15 | 165.32 | 0.00 | ORB-short ORB[164.55,165.80] vol=1.6x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-04-25 10:55:00 | 164.58 | 165.29 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 10:00:00 | 169.25 | 167.91 | 0.00 | ORB-long ORB[166.90,168.30] vol=1.8x ATR=0.61 |
| Stop hit — per-position SL triggered | 2024-04-29 10:10:00 | 168.64 | 168.11 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-04-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-30 09:45:00 | 166.95 | 167.51 | 0.00 | ORB-short ORB[167.30,168.40] vol=1.6x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 10:05:00 | 166.37 | 167.24 | 0.00 | T1 1.5R @ 166.37 |
| Target hit | 2024-04-30 15:20:00 | 165.05 | 165.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — BUY (started 2024-05-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 10:20:00 | 167.25 | 166.32 | 0.00 | ORB-long ORB[165.00,166.35] vol=1.6x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 11:05:00 | 167.93 | 166.69 | 0.00 | T1 1.5R @ 167.93 |
| Stop hit — per-position SL triggered | 2024-05-02 15:05:00 | 167.25 | 167.56 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-05-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 09:30:00 | 169.85 | 169.08 | 0.00 | ORB-long ORB[168.00,169.40] vol=1.6x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 09:55:00 | 170.62 | 169.70 | 0.00 | T1 1.5R @ 170.62 |
| Stop hit — per-position SL triggered | 2024-05-03 10:00:00 | 169.85 | 169.72 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 11:15:00 | 164.60 | 166.42 | 0.00 | ORB-short ORB[166.50,167.95] vol=1.6x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:25:00 | 163.78 | 166.08 | 0.00 | T1 1.5R @ 163.78 |
| Stop hit — per-position SL triggered | 2024-05-07 13:40:00 | 164.60 | 164.97 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 09:30:00 | 166.00 | 164.99 | 0.00 | ORB-long ORB[163.50,165.60] vol=1.5x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-05-08 10:35:00 | 165.32 | 165.56 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-19 10:00:00 | 104.50 | 2023-05-19 10:10:00 | 104.74 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-05-23 11:00:00 | 105.70 | 2023-05-23 12:55:00 | 105.90 | PARTIAL | 0.50 | 0.19% |
| BUY | retest1 | 2023-05-23 11:00:00 | 105.70 | 2023-05-23 13:10:00 | 105.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-25 11:15:00 | 104.45 | 2023-05-25 12:25:00 | 104.61 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-05-29 10:15:00 | 107.35 | 2023-05-29 10:45:00 | 107.71 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-05-29 10:15:00 | 107.35 | 2023-05-29 15:20:00 | 108.40 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2023-06-06 11:10:00 | 108.35 | 2023-06-06 11:20:00 | 108.54 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-06-08 11:00:00 | 112.75 | 2023-06-08 11:25:00 | 112.49 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-06-12 10:45:00 | 110.05 | 2023-06-12 10:55:00 | 110.35 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-06-12 10:45:00 | 110.05 | 2023-06-12 12:20:00 | 110.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-14 10:00:00 | 113.20 | 2023-06-14 10:20:00 | 112.89 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-06-19 11:00:00 | 114.10 | 2023-06-19 11:50:00 | 113.80 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-06-19 11:00:00 | 114.10 | 2023-06-19 14:55:00 | 114.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-20 09:55:00 | 114.95 | 2023-06-20 10:30:00 | 114.66 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-06-21 11:05:00 | 113.40 | 2023-06-21 11:15:00 | 113.14 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-06-21 11:05:00 | 113.40 | 2023-06-21 12:20:00 | 113.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-03 11:15:00 | 113.90 | 2023-07-03 11:20:00 | 114.22 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-07-03 11:15:00 | 113.90 | 2023-07-03 11:25:00 | 113.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-06 10:20:00 | 112.45 | 2023-07-06 10:25:00 | 112.21 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-10 09:30:00 | 112.60 | 2023-07-10 10:05:00 | 112.96 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-07-10 09:30:00 | 112.60 | 2023-07-10 15:20:00 | 115.40 | TARGET_HIT | 0.50 | 2.49% |
| BUY | retest1 | 2023-07-17 09:40:00 | 118.40 | 2023-07-17 09:45:00 | 118.06 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-07-21 09:35:00 | 115.70 | 2023-07-21 09:45:00 | 115.93 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-07-24 10:30:00 | 115.05 | 2023-07-24 11:55:00 | 114.62 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-07-24 10:30:00 | 115.05 | 2023-07-24 12:35:00 | 115.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-03 09:35:00 | 117.80 | 2023-08-03 09:40:00 | 118.19 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-08-09 10:20:00 | 117.60 | 2023-08-09 10:50:00 | 117.23 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-08-09 10:20:00 | 117.60 | 2023-08-09 13:10:00 | 117.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-11 10:25:00 | 120.50 | 2023-08-11 10:30:00 | 120.99 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-08-11 10:25:00 | 120.50 | 2023-08-11 10:55:00 | 120.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-16 10:55:00 | 116.30 | 2023-08-16 11:30:00 | 115.88 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-08-16 10:55:00 | 116.30 | 2023-08-16 15:20:00 | 115.90 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2023-08-17 11:15:00 | 115.30 | 2023-08-17 11:20:00 | 115.47 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-08-22 09:50:00 | 116.65 | 2023-08-22 09:55:00 | 116.87 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-08-24 10:15:00 | 120.00 | 2023-08-24 10:20:00 | 119.79 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-08-29 10:10:00 | 118.55 | 2023-08-29 10:20:00 | 118.29 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-08-30 10:15:00 | 122.00 | 2023-08-30 10:45:00 | 122.44 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-08-30 10:15:00 | 122.00 | 2023-08-30 11:35:00 | 122.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-31 10:20:00 | 123.30 | 2023-08-31 10:25:00 | 122.98 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-09-06 11:15:00 | 129.15 | 2023-09-06 11:40:00 | 128.68 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-09-06 11:15:00 | 129.15 | 2023-09-06 12:05:00 | 129.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-08 10:20:00 | 130.20 | 2023-09-08 11:30:00 | 130.53 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-09-11 10:20:00 | 130.95 | 2023-09-11 10:25:00 | 131.46 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-09-11 10:20:00 | 130.95 | 2023-09-11 11:10:00 | 130.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-12 09:45:00 | 129.25 | 2023-09-12 09:55:00 | 129.72 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-09-14 09:45:00 | 132.40 | 2023-09-14 09:50:00 | 133.07 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-09-14 09:45:00 | 132.40 | 2023-09-14 10:15:00 | 132.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-15 09:30:00 | 131.75 | 2023-09-15 09:50:00 | 131.05 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-09-15 09:30:00 | 131.75 | 2023-09-15 13:55:00 | 131.20 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2023-09-22 09:55:00 | 126.00 | 2023-09-22 10:00:00 | 126.40 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-09-26 09:45:00 | 130.40 | 2023-09-26 10:30:00 | 129.91 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-10-04 09:30:00 | 125.80 | 2023-10-04 09:35:00 | 126.13 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-10-05 09:40:00 | 124.85 | 2023-10-05 10:45:00 | 125.20 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-10-12 09:30:00 | 126.50 | 2023-10-12 09:55:00 | 126.23 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-10-13 10:40:00 | 124.45 | 2023-10-13 10:50:00 | 124.73 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-10-18 11:10:00 | 127.25 | 2023-10-18 11:25:00 | 127.53 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-10-20 09:55:00 | 124.95 | 2023-10-20 10:00:00 | 125.23 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-10-31 09:30:00 | 119.40 | 2023-10-31 09:35:00 | 119.69 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-11-07 10:45:00 | 119.75 | 2023-11-07 11:20:00 | 120.18 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-11-07 10:45:00 | 119.75 | 2023-11-07 11:40:00 | 119.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-08 11:05:00 | 118.95 | 2023-11-08 11:20:00 | 118.64 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-11-08 11:05:00 | 118.95 | 2023-11-08 11:45:00 | 118.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-13 10:00:00 | 120.95 | 2023-11-13 10:20:00 | 120.69 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-11-16 11:15:00 | 124.25 | 2023-11-16 11:25:00 | 124.58 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-11-16 11:15:00 | 124.25 | 2023-11-16 15:10:00 | 124.45 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2023-11-21 09:30:00 | 126.15 | 2023-11-21 09:35:00 | 125.85 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-23 11:05:00 | 126.80 | 2023-11-23 12:25:00 | 127.16 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-11-23 11:05:00 | 126.80 | 2023-11-23 14:05:00 | 126.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-24 10:00:00 | 126.00 | 2023-11-24 11:25:00 | 126.26 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-11-29 10:15:00 | 127.65 | 2023-11-29 10:20:00 | 128.00 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-11-29 10:15:00 | 127.65 | 2023-11-29 10:40:00 | 127.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 10:50:00 | 128.00 | 2023-11-30 12:00:00 | 128.42 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-11-30 10:50:00 | 128.00 | 2023-11-30 12:30:00 | 128.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-05 10:50:00 | 131.35 | 2023-12-05 11:00:00 | 131.72 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-12-05 10:50:00 | 131.35 | 2023-12-05 11:45:00 | 131.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-06 10:20:00 | 131.00 | 2023-12-06 10:55:00 | 131.29 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-12-08 09:30:00 | 131.20 | 2023-12-08 09:35:00 | 130.91 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-12-14 10:00:00 | 133.10 | 2023-12-14 10:05:00 | 132.75 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-15 10:10:00 | 135.00 | 2023-12-15 10:15:00 | 134.59 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-12-20 09:40:00 | 135.15 | 2023-12-20 09:45:00 | 135.46 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-12-22 09:30:00 | 133.50 | 2023-12-22 09:35:00 | 133.03 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-12-27 09:40:00 | 136.60 | 2023-12-27 09:55:00 | 137.07 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-12-27 09:40:00 | 136.60 | 2023-12-27 12:10:00 | 137.50 | TARGET_HIT | 0.50 | 0.66% |
| SELL | retest1 | 2023-12-28 09:30:00 | 137.50 | 2023-12-28 09:40:00 | 136.89 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-12-28 09:30:00 | 137.50 | 2023-12-28 09:45:00 | 137.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-29 10:45:00 | 139.55 | 2023-12-29 10:55:00 | 140.23 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-12-29 10:45:00 | 139.55 | 2023-12-29 11:00:00 | 139.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-02 09:55:00 | 139.05 | 2024-01-02 10:20:00 | 138.45 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-01-02 09:55:00 | 139.05 | 2024-01-02 12:25:00 | 139.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-05 09:45:00 | 134.05 | 2024-01-05 09:50:00 | 134.33 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-01-09 10:50:00 | 133.90 | 2024-01-09 12:20:00 | 134.38 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-01-09 10:50:00 | 133.90 | 2024-01-09 14:15:00 | 133.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-15 09:45:00 | 135.05 | 2024-01-15 09:55:00 | 134.53 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-01-15 09:45:00 | 135.05 | 2024-01-15 10:00:00 | 135.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-23 09:55:00 | 134.10 | 2024-01-23 10:05:00 | 133.56 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-01-23 09:55:00 | 134.10 | 2024-01-23 10:10:00 | 134.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-24 10:45:00 | 132.05 | 2024-01-24 11:10:00 | 132.78 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-01-24 10:45:00 | 132.05 | 2024-01-24 11:30:00 | 132.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-30 10:30:00 | 137.05 | 2024-01-30 10:45:00 | 137.62 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-01-30 10:30:00 | 137.05 | 2024-01-30 10:50:00 | 137.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-02 09:40:00 | 136.65 | 2024-02-02 11:05:00 | 137.25 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-02-02 09:40:00 | 136.65 | 2024-02-02 12:30:00 | 136.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-08 11:00:00 | 143.55 | 2024-02-08 11:45:00 | 144.07 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-02-09 09:40:00 | 141.95 | 2024-02-09 09:50:00 | 141.14 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-02-09 09:40:00 | 141.95 | 2024-02-09 12:20:00 | 141.25 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2024-02-14 11:10:00 | 138.20 | 2024-02-14 12:25:00 | 138.85 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-02-14 11:10:00 | 138.20 | 2024-02-14 15:20:00 | 141.30 | TARGET_HIT | 0.50 | 2.24% |
| BUY | retest1 | 2024-02-16 10:20:00 | 143.10 | 2024-02-16 12:25:00 | 142.67 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-02-20 09:50:00 | 140.85 | 2024-02-20 12:05:00 | 141.21 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-02-21 09:30:00 | 143.95 | 2024-02-21 09:35:00 | 144.68 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-02-21 09:30:00 | 143.95 | 2024-02-21 10:40:00 | 144.30 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2024-02-27 10:40:00 | 143.50 | 2024-02-27 11:50:00 | 143.16 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-03-01 10:55:00 | 146.25 | 2024-03-01 11:20:00 | 146.99 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-03-01 10:55:00 | 146.25 | 2024-03-01 15:20:00 | 150.65 | TARGET_HIT | 0.50 | 3.01% |
| SELL | retest1 | 2024-03-06 09:30:00 | 150.50 | 2024-03-06 09:35:00 | 149.76 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-03-06 09:30:00 | 150.50 | 2024-03-06 13:50:00 | 150.20 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2024-03-07 09:30:00 | 155.20 | 2024-03-07 09:35:00 | 156.13 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-03-07 09:30:00 | 155.20 | 2024-03-07 10:25:00 | 155.50 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2024-03-22 11:10:00 | 151.55 | 2024-03-22 11:40:00 | 152.28 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-03-22 11:10:00 | 151.55 | 2024-03-22 15:10:00 | 151.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-28 10:55:00 | 154.50 | 2024-03-28 11:05:00 | 155.09 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-03-28 10:55:00 | 154.50 | 2024-03-28 15:20:00 | 156.30 | TARGET_HIT | 0.50 | 1.17% |
| SELL | retest1 | 2024-04-04 10:50:00 | 162.10 | 2024-04-04 11:10:00 | 161.28 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-04-04 10:50:00 | 162.10 | 2024-04-04 11:30:00 | 162.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-09 09:55:00 | 167.80 | 2024-04-09 10:00:00 | 168.64 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-04-09 09:55:00 | 167.80 | 2024-04-09 12:45:00 | 167.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-10 11:15:00 | 165.35 | 2024-04-10 11:35:00 | 165.87 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-04-24 10:15:00 | 163.40 | 2024-04-24 10:55:00 | 163.93 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-04-24 10:15:00 | 163.40 | 2024-04-24 15:20:00 | 165.80 | TARGET_HIT | 0.50 | 1.47% |
| SELL | retest1 | 2024-04-25 10:50:00 | 164.15 | 2024-04-25 10:55:00 | 164.58 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-04-29 10:00:00 | 169.25 | 2024-04-29 10:10:00 | 168.64 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-04-30 09:45:00 | 166.95 | 2024-04-30 10:05:00 | 166.37 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-04-30 09:45:00 | 166.95 | 2024-04-30 15:20:00 | 165.05 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2024-05-02 10:20:00 | 167.25 | 2024-05-02 11:05:00 | 167.93 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-05-02 10:20:00 | 167.25 | 2024-05-02 15:05:00 | 167.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-03 09:30:00 | 169.85 | 2024-05-03 09:55:00 | 170.62 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-05-03 09:30:00 | 169.85 | 2024-05-03 10:00:00 | 169.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-07 11:15:00 | 164.60 | 2024-05-07 11:25:00 | 163.78 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-05-07 11:15:00 | 164.60 | 2024-05-07 13:40:00 | 164.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-08 09:30:00 | 166.00 | 2024-05-08 10:35:00 | 165.32 | STOP_HIT | 1.00 | -0.41% |
