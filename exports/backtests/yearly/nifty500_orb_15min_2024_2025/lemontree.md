# Lemon Tree Hotels Ltd. (LEMONTREE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-01-30 15:25:00 (31996 bars)
- **Last close:** 130.58
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
| PARTIAL | 17 |
| TARGET_HIT | 8 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 40
- **Target hits / Stop hits / Partials:** 8 / 40 / 17
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 14.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 17 | 45.9% | 5 | 20 | 12 | 0.36% | 13.2% |
| BUY @ 2nd Alert (retest1) | 37 | 17 | 45.9% | 5 | 20 | 12 | 0.36% | 13.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 28 | 8 | 28.6% | 3 | 20 | 5 | 0.06% | 1.8% |
| SELL @ 2nd Alert (retest1) | 28 | 8 | 28.6% | 3 | 20 | 5 | 0.06% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 65 | 25 | 38.5% | 8 | 40 | 17 | 0.23% | 15.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:55:00 | 148.05 | 147.29 | 0.00 | ORB-long ORB[146.20,148.00] vol=1.6x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 10:15:00 | 148.97 | 147.87 | 0.00 | T1 1.5R @ 148.97 |
| Target hit | 2024-05-21 11:40:00 | 148.35 | 148.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2024-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 09:30:00 | 148.70 | 149.82 | 0.00 | ORB-short ORB[149.35,151.35] vol=1.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2024-05-23 09:35:00 | 149.15 | 149.69 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:30:00 | 142.60 | 143.18 | 0.00 | ORB-short ORB[142.65,144.00] vol=2.0x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:35:00 | 141.90 | 142.86 | 0.00 | T1 1.5R @ 141.90 |
| Target hit | 2024-05-28 15:20:00 | 140.30 | 141.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2024-06-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:35:00 | 141.00 | 139.96 | 0.00 | ORB-long ORB[138.10,140.20] vol=1.9x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 09:45:00 | 141.82 | 140.21 | 0.00 | T1 1.5R @ 141.82 |
| Stop hit — per-position SL triggered | 2024-06-07 10:10:00 | 141.00 | 140.42 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 142.43 | 142.90 | 0.00 | ORB-short ORB[142.60,143.80] vol=1.8x ATR=0.48 |
| Stop hit — per-position SL triggered | 2024-06-10 09:40:00 | 142.91 | 142.80 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 11:00:00 | 143.15 | 144.59 | 0.00 | ORB-short ORB[144.00,146.05] vol=2.4x ATR=0.37 |
| Stop hit — per-position SL triggered | 2024-06-11 11:10:00 | 143.52 | 144.51 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:50:00 | 145.39 | 144.05 | 0.00 | ORB-long ORB[142.74,143.40] vol=2.0x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-06-12 10:05:00 | 144.81 | 144.60 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:15:00 | 144.15 | 144.78 | 0.00 | ORB-short ORB[144.75,145.60] vol=1.7x ATR=0.38 |
| Stop hit — per-position SL triggered | 2024-06-13 11:20:00 | 144.53 | 144.76 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:15:00 | 146.20 | 145.10 | 0.00 | ORB-long ORB[144.18,145.49] vol=1.9x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 10:25:00 | 146.84 | 146.00 | 0.00 | T1 1.5R @ 146.84 |
| Target hit | 2024-06-14 15:20:00 | 150.10 | 149.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 09:30:00 | 146.03 | 146.72 | 0.00 | ORB-short ORB[146.35,148.45] vol=2.0x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-06-27 09:45:00 | 146.58 | 146.53 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 144.94 | 145.26 | 0.00 | ORB-short ORB[145.00,145.85] vol=4.5x ATR=0.35 |
| Stop hit — per-position SL triggered | 2024-07-02 09:40:00 | 145.29 | 145.22 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 11:00:00 | 143.51 | 145.01 | 0.00 | ORB-short ORB[145.28,146.80] vol=1.5x ATR=0.48 |
| Stop hit — per-position SL triggered | 2024-07-18 11:05:00 | 143.99 | 144.98 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 145.82 | 146.86 | 0.00 | ORB-short ORB[146.10,147.95] vol=2.0x ATR=0.46 |
| Stop hit — per-position SL triggered | 2024-07-26 09:40:00 | 146.28 | 146.67 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 11:05:00 | 147.21 | 146.74 | 0.00 | ORB-long ORB[146.37,147.18] vol=1.9x ATR=0.35 |
| Stop hit — per-position SL triggered | 2024-07-30 11:15:00 | 146.86 | 146.75 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 118.19 | 117.41 | 0.00 | ORB-long ORB[116.61,117.76] vol=1.6x ATR=0.51 |
| Stop hit — per-position SL triggered | 2024-08-16 10:00:00 | 117.68 | 117.81 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:55:00 | 125.80 | 124.11 | 0.00 | ORB-long ORB[122.70,124.50] vol=2.4x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:00:00 | 126.63 | 124.58 | 0.00 | T1 1.5R @ 126.63 |
| Target hit | 2024-08-21 15:20:00 | 132.00 | 128.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-08-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:45:00 | 134.10 | 132.37 | 0.00 | ORB-long ORB[131.25,132.70] vol=1.7x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-08-22 09:55:00 | 133.34 | 132.60 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 11:15:00 | 132.76 | 133.86 | 0.00 | ORB-short ORB[133.47,135.40] vol=3.5x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-08-27 11:20:00 | 133.15 | 133.83 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:35:00 | 135.50 | 134.33 | 0.00 | ORB-long ORB[133.26,134.88] vol=4.0x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 09:40:00 | 136.32 | 135.03 | 0.00 | T1 1.5R @ 136.32 |
| Stop hit — per-position SL triggered | 2024-08-29 10:10:00 | 135.50 | 135.60 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-09-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-04 09:35:00 | 131.63 | 132.26 | 0.00 | ORB-short ORB[132.51,133.99] vol=6.1x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-09-04 09:45:00 | 132.26 | 132.22 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 09:35:00 | 130.50 | 131.45 | 0.00 | ORB-short ORB[131.25,132.58] vol=2.1x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 10:00:00 | 129.76 | 131.09 | 0.00 | T1 1.5R @ 129.76 |
| Stop hit — per-position SL triggered | 2024-09-12 12:40:00 | 130.50 | 130.72 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:35:00 | 133.95 | 133.15 | 0.00 | ORB-long ORB[132.30,133.10] vol=4.5x ATR=0.35 |
| Stop hit — per-position SL triggered | 2024-09-16 09:45:00 | 133.60 | 133.27 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:55:00 | 129.65 | 130.82 | 0.00 | ORB-short ORB[130.56,131.48] vol=2.1x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-09-19 10:00:00 | 130.09 | 130.76 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:50:00 | 125.52 | 125.97 | 0.00 | ORB-short ORB[125.55,126.85] vol=2.1x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 14:10:00 | 124.90 | 125.53 | 0.00 | T1 1.5R @ 124.90 |
| Target hit | 2024-09-24 15:20:00 | 123.85 | 125.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2024-09-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:55:00 | 123.73 | 123.25 | 0.00 | ORB-long ORB[122.65,123.49] vol=2.6x ATR=0.36 |
| Stop hit — per-position SL triggered | 2024-09-26 11:05:00 | 123.37 | 123.26 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:05:00 | 122.71 | 122.24 | 0.00 | ORB-long ORB[121.60,122.25] vol=2.3x ATR=0.38 |
| Stop hit — per-position SL triggered | 2024-09-27 13:20:00 | 122.33 | 122.51 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 09:30:00 | 120.91 | 121.64 | 0.00 | ORB-short ORB[121.29,122.70] vol=2.5x ATR=0.33 |
| Stop hit — per-position SL triggered | 2024-09-30 09:35:00 | 121.24 | 121.61 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:30:00 | 116.69 | 117.58 | 0.00 | ORB-short ORB[117.25,118.50] vol=1.9x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:40:00 | 116.00 | 117.26 | 0.00 | T1 1.5R @ 116.00 |
| Target hit | 2024-10-07 15:20:00 | 114.00 | 114.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2024-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 09:35:00 | 125.23 | 124.74 | 0.00 | ORB-long ORB[124.10,125.03] vol=3.8x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:15:00 | 125.88 | 124.99 | 0.00 | T1 1.5R @ 125.88 |
| Stop hit — per-position SL triggered | 2024-10-17 10:20:00 | 125.23 | 124.99 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-10-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 10:50:00 | 118.00 | 116.89 | 0.00 | ORB-long ORB[115.10,116.75] vol=2.3x ATR=0.46 |
| Stop hit — per-position SL triggered | 2024-10-31 12:00:00 | 117.54 | 117.13 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-11-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:45:00 | 123.39 | 124.26 | 0.00 | ORB-short ORB[124.00,125.35] vol=2.1x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-11-12 11:00:00 | 123.80 | 124.21 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-11-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:05:00 | 130.80 | 130.32 | 0.00 | ORB-long ORB[129.43,130.42] vol=4.6x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-11-29 10:10:00 | 130.31 | 130.31 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 11:10:00 | 132.03 | 131.09 | 0.00 | ORB-long ORB[130.50,131.63] vol=5.2x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-12-02 11:50:00 | 131.60 | 131.39 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:50:00 | 129.44 | 130.40 | 0.00 | ORB-short ORB[130.41,131.22] vol=1.9x ATR=0.32 |
| Stop hit — per-position SL triggered | 2024-12-03 11:30:00 | 129.76 | 130.20 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:30:00 | 132.17 | 131.59 | 0.00 | ORB-long ORB[130.78,131.85] vol=1.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2024-12-04 09:45:00 | 131.69 | 131.65 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:15:00 | 136.51 | 135.75 | 0.00 | ORB-long ORB[135.17,136.45] vol=5.7x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 11:35:00 | 137.17 | 136.04 | 0.00 | T1 1.5R @ 137.17 |
| Target hit | 2024-12-06 15:20:00 | 137.70 | 136.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2024-12-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:30:00 | 142.19 | 139.81 | 0.00 | ORB-long ORB[138.29,140.30] vol=2.4x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 10:35:00 | 143.27 | 140.35 | 0.00 | T1 1.5R @ 143.27 |
| Stop hit — per-position SL triggered | 2024-12-09 10:45:00 | 142.19 | 140.65 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-12-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:50:00 | 141.85 | 141.06 | 0.00 | ORB-long ORB[139.46,140.78] vol=3.8x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 10:25:00 | 142.72 | 141.56 | 0.00 | T1 1.5R @ 142.72 |
| Stop hit — per-position SL triggered | 2024-12-11 10:50:00 | 141.85 | 141.70 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:15:00 | 149.04 | 148.05 | 0.00 | ORB-long ORB[147.46,148.71] vol=1.7x ATR=0.67 |
| Stop hit — per-position SL triggered | 2024-12-16 10:20:00 | 148.37 | 148.10 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-01-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 10:00:00 | 158.67 | 157.70 | 0.00 | ORB-long ORB[156.60,158.41] vol=2.1x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-01-03 10:15:00 | 158.04 | 157.80 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-01-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 11:00:00 | 136.80 | 139.25 | 0.00 | ORB-short ORB[139.05,141.05] vol=1.7x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-01-30 11:10:00 | 137.50 | 139.05 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:50:00 | 128.75 | 128.04 | 0.00 | ORB-long ORB[126.95,128.40] vol=1.6x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 09:55:00 | 129.51 | 128.16 | 0.00 | T1 1.5R @ 129.51 |
| Stop hit — per-position SL triggered | 2025-03-13 11:35:00 | 128.75 | 128.93 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-03-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:05:00 | 131.27 | 129.81 | 0.00 | ORB-long ORB[128.90,130.30] vol=2.4x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 11:15:00 | 132.07 | 130.53 | 0.00 | T1 1.5R @ 132.07 |
| Target hit | 2025-03-18 15:20:00 | 134.22 | 132.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2025-03-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:35:00 | 134.60 | 135.76 | 0.00 | ORB-short ORB[135.45,136.60] vol=1.9x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-03-26 09:40:00 | 135.18 | 135.63 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 09:30:00 | 130.91 | 131.55 | 0.00 | ORB-short ORB[131.25,132.75] vol=2.9x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-03-27 10:15:00 | 131.65 | 131.36 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-04-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 10:50:00 | 142.10 | 142.91 | 0.00 | ORB-short ORB[142.30,144.15] vol=1.7x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-04-16 11:05:00 | 142.47 | 142.79 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 145.57 | 146.66 | 0.00 | ORB-short ORB[146.20,147.99] vol=1.9x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 09:45:00 | 144.83 | 146.23 | 0.00 | T1 1.5R @ 144.83 |
| Stop hit — per-position SL triggered | 2025-04-23 09:50:00 | 145.57 | 146.21 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-04-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:40:00 | 138.88 | 138.08 | 0.00 | ORB-long ORB[137.05,138.59] vol=1.8x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 09:45:00 | 139.72 | 138.20 | 0.00 | T1 1.5R @ 139.72 |
| Stop hit — per-position SL triggered | 2025-04-30 10:25:00 | 138.88 | 138.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-21 09:55:00 | 148.05 | 2024-05-21 10:15:00 | 148.97 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-05-21 09:55:00 | 148.05 | 2024-05-21 11:40:00 | 148.35 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2024-05-23 09:30:00 | 148.70 | 2024-05-23 09:35:00 | 149.15 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-28 09:30:00 | 142.60 | 2024-05-28 09:35:00 | 141.90 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-05-28 09:30:00 | 142.60 | 2024-05-28 15:20:00 | 140.30 | TARGET_HIT | 0.50 | 1.61% |
| BUY | retest1 | 2024-06-07 09:35:00 | 141.00 | 2024-06-07 09:45:00 | 141.82 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-06-07 09:35:00 | 141.00 | 2024-06-07 10:10:00 | 141.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-10 09:30:00 | 142.43 | 2024-06-10 09:40:00 | 142.91 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-06-11 11:00:00 | 143.15 | 2024-06-11 11:10:00 | 143.52 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-12 09:50:00 | 145.39 | 2024-06-12 10:05:00 | 144.81 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-06-13 11:15:00 | 144.15 | 2024-06-13 11:20:00 | 144.53 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-14 10:15:00 | 146.20 | 2024-06-14 10:25:00 | 146.84 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-06-14 10:15:00 | 146.20 | 2024-06-14 15:20:00 | 150.10 | TARGET_HIT | 0.50 | 2.67% |
| SELL | retest1 | 2024-06-27 09:30:00 | 146.03 | 2024-06-27 09:45:00 | 146.58 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-02 09:30:00 | 144.94 | 2024-07-02 09:40:00 | 145.29 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-07-18 11:00:00 | 143.51 | 2024-07-18 11:05:00 | 143.99 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-26 09:30:00 | 145.82 | 2024-07-26 09:40:00 | 146.28 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-30 11:05:00 | 147.21 | 2024-07-30 11:15:00 | 146.86 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-16 09:30:00 | 118.19 | 2024-08-16 10:00:00 | 117.68 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-08-21 09:55:00 | 125.80 | 2024-08-21 10:00:00 | 126.63 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-08-21 09:55:00 | 125.80 | 2024-08-21 15:20:00 | 132.00 | TARGET_HIT | 0.50 | 4.93% |
| BUY | retest1 | 2024-08-22 09:45:00 | 134.10 | 2024-08-22 09:55:00 | 133.34 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-08-27 11:15:00 | 132.76 | 2024-08-27 11:20:00 | 133.15 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-29 09:35:00 | 135.50 | 2024-08-29 09:40:00 | 136.32 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-08-29 09:35:00 | 135.50 | 2024-08-29 10:10:00 | 135.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-04 09:35:00 | 131.63 | 2024-09-04 09:45:00 | 132.26 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-09-12 09:35:00 | 130.50 | 2024-09-12 10:00:00 | 129.76 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-09-12 09:35:00 | 130.50 | 2024-09-12 12:40:00 | 130.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-16 09:35:00 | 133.95 | 2024-09-16 09:45:00 | 133.60 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-19 09:55:00 | 129.65 | 2024-09-19 10:00:00 | 130.09 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-24 10:50:00 | 125.52 | 2024-09-24 14:10:00 | 124.90 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-09-24 10:50:00 | 125.52 | 2024-09-24 15:20:00 | 123.85 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2024-09-26 10:55:00 | 123.73 | 2024-09-26 11:05:00 | 123.37 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-27 10:05:00 | 122.71 | 2024-09-27 13:20:00 | 122.33 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-30 09:30:00 | 120.91 | 2024-09-30 09:35:00 | 121.24 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-07 09:30:00 | 116.69 | 2024-10-07 09:40:00 | 116.00 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-10-07 09:30:00 | 116.69 | 2024-10-07 15:20:00 | 114.00 | TARGET_HIT | 0.50 | 2.31% |
| BUY | retest1 | 2024-10-17 09:35:00 | 125.23 | 2024-10-17 10:15:00 | 125.88 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-10-17 09:35:00 | 125.23 | 2024-10-17 10:20:00 | 125.23 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 10:50:00 | 118.00 | 2024-10-31 12:00:00 | 117.54 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-11-12 10:45:00 | 123.39 | 2024-11-12 11:00:00 | 123.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-11-29 10:05:00 | 130.80 | 2024-11-29 10:10:00 | 130.31 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-02 11:10:00 | 132.03 | 2024-12-02 11:50:00 | 131.60 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-03 10:50:00 | 129.44 | 2024-12-03 11:30:00 | 129.76 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-04 09:30:00 | 132.17 | 2024-12-04 09:45:00 | 131.69 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-12-06 11:15:00 | 136.51 | 2024-12-06 11:35:00 | 137.17 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-12-06 11:15:00 | 136.51 | 2024-12-06 15:20:00 | 137.70 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2024-12-09 10:30:00 | 142.19 | 2024-12-09 10:35:00 | 143.27 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-12-09 10:30:00 | 142.19 | 2024-12-09 10:45:00 | 142.19 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 09:50:00 | 141.85 | 2024-12-11 10:25:00 | 142.72 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-12-11 09:50:00 | 141.85 | 2024-12-11 10:50:00 | 141.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-16 10:15:00 | 149.04 | 2024-12-16 10:20:00 | 148.37 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-01-03 10:00:00 | 158.67 | 2025-01-03 10:15:00 | 158.04 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-01-30 11:00:00 | 136.80 | 2025-01-30 11:10:00 | 137.50 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-03-13 09:50:00 | 128.75 | 2025-03-13 09:55:00 | 129.51 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-03-13 09:50:00 | 128.75 | 2025-03-13 11:35:00 | 128.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 10:05:00 | 131.27 | 2025-03-18 11:15:00 | 132.07 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-03-18 10:05:00 | 131.27 | 2025-03-18 15:20:00 | 134.22 | TARGET_HIT | 0.50 | 2.25% |
| SELL | retest1 | 2025-03-26 09:35:00 | 134.60 | 2025-03-26 09:40:00 | 135.18 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-03-27 09:30:00 | 130.91 | 2025-03-27 10:15:00 | 131.65 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2025-04-16 10:50:00 | 142.10 | 2025-04-16 11:05:00 | 142.47 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-04-23 09:35:00 | 145.57 | 2025-04-23 09:45:00 | 144.83 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-04-23 09:35:00 | 145.57 | 2025-04-23 09:50:00 | 145.57 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-30 09:40:00 | 138.88 | 2025-04-30 09:45:00 | 139.72 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-04-30 09:40:00 | 138.88 | 2025-04-30 10:25:00 | 138.88 | STOP_HIT | 0.50 | 0.00% |
