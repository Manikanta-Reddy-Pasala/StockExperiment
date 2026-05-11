# Adani Power Ltd. (ADANIPOWER)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35371 bars)
- **Last close:** 225.02
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
| ENTRY1 | 48 |
| ENTRY2 | 0 |
| PARTIAL | 16 |
| TARGET_HIT | 10 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 38
- **Target hits / Stop hits / Partials:** 10 / 38 / 16
- **Avg / median % per leg:** 0.29% / 0.00%
- **Sum % (uncompounded):** 18.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 11 | 31.4% | 4 | 24 | 7 | 0.19% | 6.7% |
| BUY @ 2nd Alert (retest1) | 35 | 11 | 31.4% | 4 | 24 | 7 | 0.19% | 6.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 29 | 15 | 51.7% | 6 | 14 | 9 | 0.41% | 11.9% |
| SELL @ 2nd Alert (retest1) | 29 | 15 | 51.7% | 6 | 14 | 9 | 0.41% | 11.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 64 | 26 | 40.6% | 10 | 38 | 16 | 0.29% | 18.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:35:00 | 120.30 | 119.26 | 0.00 | ORB-long ORB[118.69,119.98] vol=3.9x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 10:50:00 | 121.30 | 119.82 | 0.00 | T1 1.5R @ 121.30 |
| Target hit | 2024-05-14 15:20:00 | 125.10 | 124.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2024-05-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 09:35:00 | 125.90 | 126.77 | 0.00 | ORB-short ORB[126.40,128.07] vol=2.1x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-05-17 09:40:00 | 126.33 | 126.69 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 139.48 | 140.73 | 0.00 | ORB-short ORB[140.37,142.36] vol=3.9x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 10:00:00 | 138.45 | 140.05 | 0.00 | T1 1.5R @ 138.45 |
| Target hit | 2024-05-28 15:20:00 | 135.53 | 137.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2024-06-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:35:00 | 151.56 | 152.61 | 0.00 | ORB-short ORB[152.02,153.76] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-06-12 10:40:00 | 151.97 | 152.58 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 11:10:00 | 150.07 | 150.11 | 0.00 | ORB-short ORB[150.10,151.70] vol=1.9x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 11:40:00 | 149.47 | 150.08 | 0.00 | T1 1.5R @ 149.47 |
| Stop hit — per-position SL triggered | 2024-06-18 12:05:00 | 150.07 | 150.03 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 146.42 | 146.94 | 0.00 | ORB-short ORB[147.17,148.48] vol=2.8x ATR=0.48 |
| Stop hit — per-position SL triggered | 2024-06-25 11:30:00 | 146.90 | 146.80 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 11:15:00 | 143.20 | 143.94 | 0.00 | ORB-short ORB[143.82,145.59] vol=1.6x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 11:20:00 | 142.74 | 143.88 | 0.00 | T1 1.5R @ 142.74 |
| Stop hit — per-position SL triggered | 2024-06-27 11:40:00 | 143.20 | 143.72 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:35:00 | 146.18 | 144.46 | 0.00 | ORB-long ORB[143.03,144.75] vol=7.5x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-07-02 10:40:00 | 145.53 | 144.81 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 140.00 | 140.95 | 0.00 | ORB-short ORB[140.44,142.49] vol=2.5x ATR=0.57 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 140.57 | 140.68 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:45:00 | 140.78 | 140.20 | 0.00 | ORB-long ORB[138.51,140.59] vol=1.8x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-07-24 11:25:00 | 139.77 | 140.30 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:10:00 | 140.45 | 138.75 | 0.00 | ORB-long ORB[137.23,138.80] vol=4.0x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:15:00 | 141.56 | 139.19 | 0.00 | T1 1.5R @ 141.56 |
| Stop hit — per-position SL triggered | 2024-07-25 10:20:00 | 140.45 | 139.29 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:40:00 | 142.61 | 141.11 | 0.00 | ORB-long ORB[139.72,141.60] vol=4.4x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-07-26 10:45:00 | 142.05 | 141.17 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-09-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 10:00:00 | 124.99 | 125.66 | 0.00 | ORB-short ORB[125.69,126.76] vol=1.5x ATR=0.45 |
| Stop hit — per-position SL triggered | 2024-09-12 12:05:00 | 125.44 | 125.42 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-09-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 11:00:00 | 134.29 | 133.98 | 0.00 | ORB-long ORB[132.52,134.26] vol=1.8x ATR=0.35 |
| Stop hit — per-position SL triggered | 2024-09-27 11:05:00 | 133.94 | 133.99 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-09-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 10:50:00 | 131.93 | 131.07 | 0.00 | ORB-long ORB[130.00,131.74] vol=2.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-09-30 13:05:00 | 131.29 | 131.29 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-10-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 11:10:00 | 127.53 | 128.24 | 0.00 | ORB-short ORB[128.00,129.14] vol=2.2x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 11:20:00 | 127.13 | 128.14 | 0.00 | T1 1.5R @ 127.13 |
| Stop hit — per-position SL triggered | 2024-10-14 11:45:00 | 127.53 | 128.02 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-10-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:05:00 | 125.54 | 126.41 | 0.00 | ORB-short ORB[126.31,127.16] vol=1.8x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:20:00 | 125.14 | 126.22 | 0.00 | T1 1.5R @ 125.14 |
| Stop hit — per-position SL triggered | 2024-10-16 14:20:00 | 125.54 | 125.50 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:15:00 | 123.99 | 124.42 | 0.00 | ORB-short ORB[124.48,125.74] vol=1.7x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:40:00 | 123.62 | 124.28 | 0.00 | T1 1.5R @ 123.62 |
| Target hit | 2024-10-17 15:20:00 | 121.68 | 123.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2024-10-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:25:00 | 118.04 | 119.95 | 0.00 | ORB-short ORB[119.90,121.43] vol=1.9x ATR=0.53 |
| Stop hit — per-position SL triggered | 2024-10-22 10:35:00 | 118.57 | 119.66 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-10-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:30:00 | 120.32 | 121.25 | 0.00 | ORB-short ORB[120.71,122.46] vol=1.8x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:40:00 | 119.33 | 120.99 | 0.00 | T1 1.5R @ 119.33 |
| Target hit | 2024-10-25 15:20:00 | 118.33 | 118.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2024-11-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-05 09:55:00 | 117.95 | 117.44 | 0.00 | ORB-long ORB[116.20,117.92] vol=1.5x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-11-05 10:20:00 | 117.52 | 117.56 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-11-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:55:00 | 115.38 | 116.14 | 0.00 | ORB-short ORB[115.94,117.42] vol=1.7x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 11:10:00 | 114.85 | 116.01 | 0.00 | T1 1.5R @ 114.85 |
| Target hit | 2024-11-12 15:20:00 | 111.76 | 113.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2024-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:30:00 | 108.38 | 107.57 | 0.00 | ORB-long ORB[106.76,108.20] vol=2.1x ATR=0.46 |
| Stop hit — per-position SL triggered | 2024-12-05 09:40:00 | 107.92 | 107.82 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:35:00 | 108.35 | 107.82 | 0.00 | ORB-long ORB[107.10,108.16] vol=1.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-12-06 09:40:00 | 107.96 | 107.85 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-12-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:30:00 | 106.22 | 104.10 | 0.00 | ORB-long ORB[102.90,104.09] vol=5.8x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-12-12 10:35:00 | 105.73 | 104.47 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:30:00 | 107.41 | 108.24 | 0.00 | ORB-short ORB[107.63,108.94] vol=1.6x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-12-13 09:40:00 | 108.03 | 108.08 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-12-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:10:00 | 106.38 | 107.10 | 0.00 | ORB-short ORB[106.81,107.88] vol=2.6x ATR=0.40 |
| Target hit | 2024-12-16 15:20:00 | 105.92 | 106.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 106.88 | 105.98 | 0.00 | ORB-long ORB[104.85,106.05] vol=3.9x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-12-17 09:35:00 | 106.47 | 106.07 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-12-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:30:00 | 101.60 | 100.70 | 0.00 | ORB-long ORB[100.06,101.39] vol=1.9x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-12-24 10:45:00 | 101.21 | 100.86 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-12-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 10:40:00 | 101.64 | 100.31 | 0.00 | ORB-long ORB[99.12,100.36] vol=4.8x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-12-26 10:45:00 | 101.15 | 100.39 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-12-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:25:00 | 103.17 | 102.34 | 0.00 | ORB-long ORB[101.35,102.79] vol=2.9x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-12-30 12:35:00 | 102.62 | 102.68 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:35:00 | 107.12 | 105.23 | 0.00 | ORB-long ORB[103.74,105.09] vol=4.0x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-01-23 09:40:00 | 106.49 | 105.56 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-01-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 10:55:00 | 102.85 | 103.96 | 0.00 | ORB-short ORB[103.53,104.90] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-01-30 11:10:00 | 103.35 | 103.92 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-01-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 11:00:00 | 102.62 | 101.86 | 0.00 | ORB-long ORB[100.85,102.17] vol=1.6x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-01-31 14:05:00 | 102.11 | 102.19 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 09:30:00 | 97.53 | 97.96 | 0.00 | ORB-short ORB[97.72,98.52] vol=1.8x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-02-18 09:35:00 | 97.88 | 97.94 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-03-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:40:00 | 101.58 | 100.77 | 0.00 | ORB-long ORB[99.80,101.29] vol=2.7x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 11:30:00 | 102.13 | 101.05 | 0.00 | T1 1.5R @ 102.13 |
| Stop hit — per-position SL triggered | 2025-03-07 11:45:00 | 101.58 | 101.11 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:30:00 | 103.70 | 103.15 | 0.00 | ORB-long ORB[102.60,103.59] vol=2.2x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-03-12 09:40:00 | 103.26 | 103.27 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:35:00 | 104.11 | 103.22 | 0.00 | ORB-long ORB[102.41,103.38] vol=3.1x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-03-13 09:40:00 | 103.56 | 103.29 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-03-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:35:00 | 105.36 | 104.65 | 0.00 | ORB-long ORB[103.62,104.97] vol=2.6x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-03-21 09:45:00 | 104.99 | 104.71 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-03-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 10:55:00 | 103.19 | 104.25 | 0.00 | ORB-short ORB[104.04,105.59] vol=1.9x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 11:25:00 | 102.68 | 104.15 | 0.00 | T1 1.5R @ 102.68 |
| Target hit | 2025-03-25 15:20:00 | 101.31 | 103.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2025-04-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:40:00 | 102.42 | 101.25 | 0.00 | ORB-long ORB[100.56,101.61] vol=3.3x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 10:45:00 | 102.98 | 101.52 | 0.00 | T1 1.5R @ 102.98 |
| Stop hit — per-position SL triggered | 2025-04-02 10:50:00 | 102.42 | 101.59 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-04-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:30:00 | 104.30 | 103.58 | 0.00 | ORB-long ORB[102.85,104.14] vol=3.2x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-04-11 09:55:00 | 103.84 | 103.86 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:40:00 | 108.81 | 108.44 | 0.00 | ORB-long ORB[107.62,108.71] vol=1.5x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-04-16 10:15:00 | 108.32 | 108.53 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-04-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:25:00 | 110.16 | 109.25 | 0.00 | ORB-long ORB[108.46,109.90] vol=3.1x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 10:35:00 | 110.82 | 110.02 | 0.00 | T1 1.5R @ 110.82 |
| Target hit | 2025-04-17 10:50:00 | 110.36 | 110.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — BUY (started 2025-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:45:00 | 112.88 | 111.75 | 0.00 | ORB-long ORB[109.81,111.49] vol=2.7x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:15:00 | 113.65 | 112.36 | 0.00 | T1 1.5R @ 113.65 |
| Target hit | 2025-04-21 12:35:00 | 114.50 | 114.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — BUY (started 2025-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:40:00 | 115.70 | 114.75 | 0.00 | ORB-long ORB[113.75,115.22] vol=1.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-04-22 09:45:00 | 115.22 | 114.77 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:40:00 | 110.78 | 111.74 | 0.00 | ORB-short ORB[111.22,112.40] vol=2.1x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-04-29 09:50:00 | 111.23 | 111.59 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:45:00 | 106.80 | 105.71 | 0.00 | ORB-long ORB[105.10,105.96] vol=1.6x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:55:00 | 107.47 | 106.38 | 0.00 | T1 1.5R @ 107.47 |
| Target hit | 2025-05-05 13:10:00 | 113.06 | 113.22 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 10:35:00 | 120.30 | 2024-05-14 10:50:00 | 121.30 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2024-05-14 10:35:00 | 120.30 | 2024-05-14 15:20:00 | 125.10 | TARGET_HIT | 0.50 | 3.99% |
| SELL | retest1 | 2024-05-17 09:35:00 | 125.90 | 2024-05-17 09:40:00 | 126.33 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-28 09:35:00 | 139.48 | 2024-05-28 10:00:00 | 138.45 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2024-05-28 09:35:00 | 139.48 | 2024-05-28 15:20:00 | 135.53 | TARGET_HIT | 0.50 | 2.83% |
| SELL | retest1 | 2024-06-12 10:35:00 | 151.56 | 2024-06-12 10:40:00 | 151.97 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-06-18 11:10:00 | 150.07 | 2024-06-18 11:40:00 | 149.47 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-06-18 11:10:00 | 150.07 | 2024-06-18 12:05:00 | 150.07 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 11:15:00 | 146.42 | 2024-06-25 11:30:00 | 146.90 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-27 11:15:00 | 143.20 | 2024-06-27 11:20:00 | 142.74 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-06-27 11:15:00 | 143.20 | 2024-06-27 11:40:00 | 143.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-02 10:35:00 | 146.18 | 2024-07-02 10:40:00 | 145.53 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-07-18 09:30:00 | 140.00 | 2024-07-18 09:40:00 | 140.57 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-07-24 09:45:00 | 140.78 | 2024-07-24 11:25:00 | 139.77 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2024-07-25 10:10:00 | 140.45 | 2024-07-25 10:15:00 | 141.56 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2024-07-25 10:10:00 | 140.45 | 2024-07-25 10:20:00 | 140.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:40:00 | 142.61 | 2024-07-26 10:45:00 | 142.05 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-09-12 10:00:00 | 124.99 | 2024-09-12 12:05:00 | 125.44 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-09-27 11:00:00 | 134.29 | 2024-09-27 11:05:00 | 133.94 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-30 10:50:00 | 131.93 | 2024-09-30 13:05:00 | 131.29 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-10-14 11:10:00 | 127.53 | 2024-10-14 11:20:00 | 127.13 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-10-14 11:10:00 | 127.53 | 2024-10-14 11:45:00 | 127.53 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 11:05:00 | 125.54 | 2024-10-16 11:20:00 | 125.14 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-10-16 11:05:00 | 125.54 | 2024-10-16 14:20:00 | 125.54 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 11:15:00 | 123.99 | 2024-10-17 11:40:00 | 123.62 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-10-17 11:15:00 | 123.99 | 2024-10-17 15:20:00 | 121.68 | TARGET_HIT | 0.50 | 1.86% |
| SELL | retest1 | 2024-10-22 10:25:00 | 118.04 | 2024-10-22 10:35:00 | 118.57 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-10-25 09:30:00 | 120.32 | 2024-10-25 09:40:00 | 119.33 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2024-10-25 09:30:00 | 120.32 | 2024-10-25 15:20:00 | 118.33 | TARGET_HIT | 0.50 | 1.65% |
| BUY | retest1 | 2024-11-05 09:55:00 | 117.95 | 2024-11-05 10:20:00 | 117.52 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-11-12 10:55:00 | 115.38 | 2024-11-12 11:10:00 | 114.85 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-11-12 10:55:00 | 115.38 | 2024-11-12 15:20:00 | 111.76 | TARGET_HIT | 0.50 | 3.14% |
| BUY | retest1 | 2024-12-05 09:30:00 | 108.38 | 2024-12-05 09:40:00 | 107.92 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-12-06 09:35:00 | 108.35 | 2024-12-06 09:40:00 | 107.96 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-12-12 10:30:00 | 106.22 | 2024-12-12 10:35:00 | 105.73 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-12-13 09:30:00 | 107.41 | 2024-12-13 09:40:00 | 108.03 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-12-16 10:10:00 | 106.38 | 2024-12-16 15:20:00 | 105.92 | TARGET_HIT | 1.00 | 0.43% |
| BUY | retest1 | 2024-12-17 09:30:00 | 106.88 | 2024-12-17 09:35:00 | 106.47 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-24 10:30:00 | 101.60 | 2024-12-24 10:45:00 | 101.21 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-12-26 10:40:00 | 101.64 | 2024-12-26 10:45:00 | 101.15 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-12-30 10:25:00 | 103.17 | 2024-12-30 12:35:00 | 102.62 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-01-23 09:35:00 | 107.12 | 2025-01-23 09:40:00 | 106.49 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2025-01-30 10:55:00 | 102.85 | 2025-01-30 11:10:00 | 103.35 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-01-31 11:00:00 | 102.62 | 2025-01-31 14:05:00 | 102.11 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-02-18 09:30:00 | 97.53 | 2025-02-18 09:35:00 | 97.88 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-07 10:40:00 | 101.58 | 2025-03-07 11:30:00 | 102.13 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-03-07 10:40:00 | 101.58 | 2025-03-07 11:45:00 | 101.58 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-12 09:30:00 | 103.70 | 2025-03-12 09:40:00 | 103.26 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-03-13 09:35:00 | 104.11 | 2025-03-13 09:40:00 | 103.56 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2025-03-21 09:35:00 | 105.36 | 2025-03-21 09:45:00 | 104.99 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-03-25 10:55:00 | 103.19 | 2025-03-25 11:25:00 | 102.68 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-03-25 10:55:00 | 103.19 | 2025-03-25 15:20:00 | 101.31 | TARGET_HIT | 0.50 | 1.82% |
| BUY | retest1 | 2025-04-02 10:40:00 | 102.42 | 2025-04-02 10:45:00 | 102.98 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-04-02 10:40:00 | 102.42 | 2025-04-02 10:50:00 | 102.42 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-11 09:30:00 | 104.30 | 2025-04-11 09:55:00 | 103.84 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-04-16 09:40:00 | 108.81 | 2025-04-16 10:15:00 | 108.32 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-04-17 10:25:00 | 110.16 | 2025-04-17 10:35:00 | 110.82 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-04-17 10:25:00 | 110.16 | 2025-04-17 10:50:00 | 110.36 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2025-04-21 09:45:00 | 112.88 | 2025-04-21 10:15:00 | 113.65 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-04-21 09:45:00 | 112.88 | 2025-04-21 12:35:00 | 114.50 | TARGET_HIT | 0.50 | 1.44% |
| BUY | retest1 | 2025-04-22 09:40:00 | 115.70 | 2025-04-22 09:45:00 | 115.22 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-04-29 09:40:00 | 110.78 | 2025-04-29 09:50:00 | 111.23 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-05-05 09:45:00 | 106.80 | 2025-05-05 09:55:00 | 107.47 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-05-05 09:45:00 | 106.80 | 2025-05-05 13:10:00 | 113.06 | TARGET_HIT | 0.50 | 5.86% |
