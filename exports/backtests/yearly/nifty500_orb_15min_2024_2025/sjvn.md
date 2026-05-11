# SJVN Ltd. (SJVN)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 78.69
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
| ENTRY1 | 47 |
| ENTRY2 | 0 |
| PARTIAL | 16 |
| TARGET_HIT | 7 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 40
- **Target hits / Stop hits / Partials:** 7 / 40 / 16
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 8.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 14 | 36.8% | 4 | 24 | 10 | 0.09% | 3.5% |
| BUY @ 2nd Alert (retest1) | 38 | 14 | 36.8% | 4 | 24 | 10 | 0.09% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 25 | 9 | 36.0% | 3 | 16 | 6 | 0.21% | 5.1% |
| SELL @ 2nd Alert (retest1) | 25 | 9 | 36.0% | 3 | 16 | 6 | 0.21% | 5.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 63 | 23 | 36.5% | 7 | 40 | 16 | 0.14% | 8.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 11:15:00 | 134.30 | 132.66 | 0.00 | ORB-long ORB[131.30,133.00] vol=8.0x ATR=0.53 |
| Stop hit — per-position SL triggered | 2024-05-17 11:20:00 | 133.77 | 132.74 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:35:00 | 146.95 | 146.21 | 0.00 | ORB-long ORB[144.85,146.70] vol=4.3x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 09:40:00 | 148.23 | 146.40 | 0.00 | T1 1.5R @ 148.23 |
| Target hit | 2024-05-23 12:25:00 | 148.45 | 148.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2024-06-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:35:00 | 132.83 | 133.89 | 0.00 | ORB-short ORB[133.50,135.40] vol=2.0x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-06-10 09:40:00 | 133.51 | 133.85 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:10:00 | 132.95 | 134.21 | 0.00 | ORB-short ORB[134.13,135.39] vol=3.4x ATR=0.38 |
| Stop hit — per-position SL triggered | 2024-06-13 11:50:00 | 133.33 | 134.00 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:25:00 | 133.30 | 132.39 | 0.00 | ORB-long ORB[131.22,132.95] vol=2.8x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:30:00 | 133.97 | 132.63 | 0.00 | T1 1.5R @ 133.97 |
| Stop hit — per-position SL triggered | 2024-06-20 10:45:00 | 133.30 | 132.74 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:35:00 | 133.39 | 132.65 | 0.00 | ORB-long ORB[131.80,133.05] vol=1.7x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 09:50:00 | 134.02 | 133.05 | 0.00 | T1 1.5R @ 134.02 |
| Stop hit — per-position SL triggered | 2024-06-21 10:10:00 | 133.39 | 133.15 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:10:00 | 130.80 | 131.53 | 0.00 | ORB-short ORB[131.51,132.50] vol=1.5x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:30:00 | 130.46 | 131.41 | 0.00 | T1 1.5R @ 130.46 |
| Stop hit — per-position SL triggered | 2024-06-25 11:45:00 | 130.80 | 131.37 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:40:00 | 132.20 | 131.38 | 0.00 | ORB-long ORB[130.80,131.62] vol=5.1x ATR=0.37 |
| Stop hit — per-position SL triggered | 2024-07-03 10:45:00 | 131.83 | 131.46 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:00:00 | 141.40 | 139.37 | 0.00 | ORB-long ORB[136.35,138.46] vol=1.6x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-07-04 10:15:00 | 140.44 | 139.62 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 09:50:00 | 141.52 | 142.37 | 0.00 | ORB-short ORB[141.54,143.60] vol=1.8x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-07-25 10:40:00 | 142.22 | 142.07 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-08-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:05:00 | 140.34 | 138.26 | 0.00 | ORB-long ORB[137.20,139.22] vol=2.9x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-08-12 11:15:00 | 139.75 | 138.49 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-08-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:45:00 | 134.59 | 133.12 | 0.00 | ORB-long ORB[131.90,133.38] vol=3.9x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 09:55:00 | 135.34 | 133.76 | 0.00 | T1 1.5R @ 135.34 |
| Stop hit — per-position SL triggered | 2024-08-23 11:10:00 | 134.59 | 134.12 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:35:00 | 134.99 | 134.36 | 0.00 | ORB-long ORB[133.60,134.79] vol=2.6x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-08-26 09:50:00 | 134.55 | 134.46 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 134.31 | 133.47 | 0.00 | ORB-long ORB[132.35,133.75] vol=4.9x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 09:35:00 | 134.85 | 133.92 | 0.00 | T1 1.5R @ 134.85 |
| Target hit | 2024-08-29 10:00:00 | 134.99 | 135.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 11:15:00 | 135.12 | 135.79 | 0.00 | ORB-short ORB[135.23,136.85] vol=1.8x ATR=0.30 |
| Stop hit — per-position SL triggered | 2024-09-03 11:25:00 | 135.42 | 135.78 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-09-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:35:00 | 131.69 | 132.41 | 0.00 | ORB-short ORB[132.25,133.25] vol=2.0x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:50:00 | 131.07 | 132.15 | 0.00 | T1 1.5R @ 131.07 |
| Target hit | 2024-09-06 15:20:00 | 130.55 | 130.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-09-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:05:00 | 131.55 | 130.53 | 0.00 | ORB-long ORB[129.56,131.02] vol=2.3x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-09-16 10:20:00 | 131.05 | 130.67 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-09-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 11:10:00 | 126.98 | 128.96 | 0.00 | ORB-short ORB[129.72,131.00] vol=1.6x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:20:00 | 126.30 | 128.80 | 0.00 | T1 1.5R @ 126.30 |
| Stop hit — per-position SL triggered | 2024-09-19 12:10:00 | 126.98 | 128.14 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-09-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:35:00 | 129.54 | 128.85 | 0.00 | ORB-long ORB[127.83,129.29] vol=1.6x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 10:45:00 | 130.29 | 128.98 | 0.00 | T1 1.5R @ 130.29 |
| Stop hit — per-position SL triggered | 2024-09-20 11:00:00 | 129.54 | 129.03 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-09-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:45:00 | 128.37 | 128.85 | 0.00 | ORB-short ORB[128.56,130.01] vol=1.6x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 10:05:00 | 127.81 | 128.57 | 0.00 | T1 1.5R @ 127.81 |
| Target hit | 2024-09-25 10:55:00 | 128.25 | 128.19 | 0.00 | Trail-exit close>VWAP |

### Cycle 21 — SELL (started 2024-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:30:00 | 125.54 | 126.41 | 0.00 | ORB-short ORB[125.80,127.50] vol=1.8x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:35:00 | 124.83 | 126.15 | 0.00 | T1 1.5R @ 124.83 |
| Target hit | 2024-10-07 15:20:00 | 118.10 | 120.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2024-10-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:45:00 | 119.99 | 119.80 | 0.00 | ORB-long ORB[118.38,119.49] vol=1.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-10-11 10:55:00 | 119.60 | 119.79 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-10-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:30:00 | 114.37 | 112.36 | 0.00 | ORB-long ORB[110.69,112.30] vol=1.9x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-10-24 10:45:00 | 113.68 | 112.54 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:35:00 | 117.10 | 116.04 | 0.00 | ORB-long ORB[114.88,116.41] vol=3.5x ATR=0.51 |
| Stop hit — per-position SL triggered | 2024-11-07 09:40:00 | 116.59 | 116.10 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-11-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:30:00 | 102.41 | 103.76 | 0.00 | ORB-short ORB[103.52,104.90] vol=2.3x ATR=0.51 |
| Stop hit — per-position SL triggered | 2024-11-18 09:45:00 | 102.92 | 103.26 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-11-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:10:00 | 113.43 | 112.55 | 0.00 | ORB-long ORB[111.56,112.95] vol=2.1x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-11-27 10:20:00 | 113.00 | 112.59 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-12-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:35:00 | 115.18 | 114.45 | 0.00 | ORB-long ORB[113.72,114.85] vol=1.6x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-12-02 09:40:00 | 114.78 | 114.51 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-12-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:45:00 | 118.89 | 118.08 | 0.00 | ORB-long ORB[117.42,118.40] vol=1.8x ATR=0.37 |
| Stop hit — per-position SL triggered | 2024-12-04 09:50:00 | 118.52 | 118.13 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 116.92 | 117.16 | 0.00 | ORB-short ORB[117.08,117.90] vol=1.9x ATR=0.33 |
| Stop hit — per-position SL triggered | 2024-12-05 12:10:00 | 117.25 | 117.10 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:35:00 | 118.35 | 117.60 | 0.00 | ORB-long ORB[116.71,117.60] vol=4.2x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 09:40:00 | 118.83 | 118.42 | 0.00 | T1 1.5R @ 118.83 |
| Target hit | 2024-12-06 11:35:00 | 120.71 | 120.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 118.72 | 119.70 | 0.00 | ORB-short ORB[119.79,120.60] vol=2.2x ATR=0.31 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 119.03 | 119.53 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-12-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:20:00 | 115.25 | 116.72 | 0.00 | ORB-short ORB[116.73,118.18] vol=1.7x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-12-13 10:50:00 | 115.66 | 116.43 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 10:45:00 | 118.66 | 117.46 | 0.00 | ORB-long ORB[117.05,117.86] vol=4.3x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-12-17 10:50:00 | 118.26 | 117.50 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 09:55:00 | 115.38 | 116.67 | 0.00 | ORB-short ORB[116.50,118.14] vol=1.9x ATR=0.47 |
| Stop hit — per-position SL triggered | 2024-12-18 10:15:00 | 115.85 | 116.42 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-12-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:55:00 | 109.43 | 110.09 | 0.00 | ORB-short ORB[109.55,111.10] vol=2.2x ATR=0.30 |
| Stop hit — per-position SL triggered | 2024-12-26 11:10:00 | 109.73 | 110.07 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-01-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:45:00 | 93.47 | 92.44 | 0.00 | ORB-long ORB[91.63,92.95] vol=1.8x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-01-30 10:25:00 | 93.08 | 92.73 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-01-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:40:00 | 96.06 | 95.10 | 0.00 | ORB-long ORB[93.82,95.20] vol=1.8x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 10:25:00 | 96.73 | 95.65 | 0.00 | T1 1.5R @ 96.73 |
| Target hit | 2025-01-31 15:20:00 | 97.51 | 96.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2025-02-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:45:00 | 97.70 | 97.20 | 0.00 | ORB-long ORB[95.85,97.30] vol=2.8x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 09:50:00 | 98.22 | 97.36 | 0.00 | T1 1.5R @ 98.22 |
| Stop hit — per-position SL triggered | 2025-02-05 10:05:00 | 97.70 | 97.45 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 96.26 | 96.95 | 0.00 | ORB-short ORB[96.51,97.64] vol=1.6x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-02-06 09:35:00 | 96.55 | 96.89 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-02-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 10:45:00 | 88.66 | 89.05 | 0.00 | ORB-short ORB[88.86,89.99] vol=2.6x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-02-27 10:50:00 | 89.04 | 89.07 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-03-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:30:00 | 88.00 | 87.55 | 0.00 | ORB-long ORB[86.77,87.99] vol=1.9x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-03-07 09:35:00 | 87.66 | 87.57 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:30:00 | 86.98 | 86.38 | 0.00 | ORB-long ORB[85.51,86.72] vol=2.0x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-03-17 09:35:00 | 86.67 | 86.43 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-03-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 11:10:00 | 87.28 | 87.06 | 0.00 | ORB-long ORB[86.40,87.25] vol=1.6x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-03-18 11:50:00 | 87.04 | 87.10 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:55:00 | 95.09 | 94.47 | 0.00 | ORB-long ORB[93.85,94.50] vol=1.6x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 10:00:00 | 95.49 | 94.65 | 0.00 | T1 1.5R @ 95.49 |
| Stop hit — per-position SL triggered | 2025-04-16 10:15:00 | 95.09 | 94.79 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-04-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:00:00 | 93.85 | 94.62 | 0.00 | ORB-short ORB[94.31,95.69] vol=1.8x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-04-17 10:20:00 | 94.18 | 94.40 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:35:00 | 95.30 | 94.57 | 0.00 | ORB-long ORB[93.81,94.88] vol=2.8x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-04-21 09:45:00 | 95.00 | 94.67 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-04-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:05:00 | 98.60 | 99.90 | 0.00 | ORB-short ORB[99.91,100.87] vol=1.7x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:15:00 | 97.94 | 99.58 | 0.00 | T1 1.5R @ 97.94 |
| Stop hit — per-position SL triggered | 2025-04-23 11:05:00 | 98.60 | 99.20 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-17 11:15:00 | 134.30 | 2024-05-17 11:20:00 | 133.77 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-05-23 09:35:00 | 146.95 | 2024-05-23 09:40:00 | 148.23 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2024-05-23 09:35:00 | 146.95 | 2024-05-23 12:25:00 | 148.45 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2024-06-10 09:35:00 | 132.83 | 2024-06-10 09:40:00 | 133.51 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-06-13 11:10:00 | 132.95 | 2024-06-13 11:50:00 | 133.33 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-20 10:25:00 | 133.30 | 2024-06-20 10:30:00 | 133.97 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-06-20 10:25:00 | 133.30 | 2024-06-20 10:45:00 | 133.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-21 09:35:00 | 133.39 | 2024-06-21 09:50:00 | 134.02 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-06-21 09:35:00 | 133.39 | 2024-06-21 10:10:00 | 133.39 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 11:10:00 | 130.80 | 2024-06-25 11:30:00 | 130.46 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-06-25 11:10:00 | 130.80 | 2024-06-25 11:45:00 | 130.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-03 10:40:00 | 132.20 | 2024-07-03 10:45:00 | 131.83 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-04 10:00:00 | 141.40 | 2024-07-04 10:15:00 | 140.44 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest1 | 2024-07-25 09:50:00 | 141.52 | 2024-07-25 10:40:00 | 142.22 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-08-12 11:05:00 | 140.34 | 2024-08-12 11:15:00 | 139.75 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-08-23 09:45:00 | 134.59 | 2024-08-23 09:55:00 | 135.34 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-08-23 09:45:00 | 134.59 | 2024-08-23 11:10:00 | 134.59 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-26 09:35:00 | 134.99 | 2024-08-26 09:50:00 | 134.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-29 09:30:00 | 134.31 | 2024-08-29 09:35:00 | 134.85 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-29 09:30:00 | 134.31 | 2024-08-29 10:00:00 | 134.99 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-03 11:15:00 | 135.12 | 2024-09-03 11:25:00 | 135.42 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-06 09:35:00 | 131.69 | 2024-09-06 09:50:00 | 131.07 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-09-06 09:35:00 | 131.69 | 2024-09-06 15:20:00 | 130.55 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2024-09-16 10:05:00 | 131.55 | 2024-09-16 10:20:00 | 131.05 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-19 11:10:00 | 126.98 | 2024-09-19 11:20:00 | 126.30 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-09-19 11:10:00 | 126.98 | 2024-09-19 12:10:00 | 126.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-20 10:35:00 | 129.54 | 2024-09-20 10:45:00 | 130.29 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-09-20 10:35:00 | 129.54 | 2024-09-20 11:00:00 | 129.54 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-25 09:45:00 | 128.37 | 2024-09-25 10:05:00 | 127.81 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-25 09:45:00 | 128.37 | 2024-09-25 10:55:00 | 128.25 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2024-10-07 09:30:00 | 125.54 | 2024-10-07 09:35:00 | 124.83 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-10-07 09:30:00 | 125.54 | 2024-10-07 15:20:00 | 118.10 | TARGET_HIT | 0.50 | 5.93% |
| BUY | retest1 | 2024-10-11 10:45:00 | 119.99 | 2024-10-11 10:55:00 | 119.60 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-24 10:30:00 | 114.37 | 2024-10-24 10:45:00 | 113.68 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2024-11-07 09:35:00 | 117.10 | 2024-11-07 09:40:00 | 116.59 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-11-18 09:30:00 | 102.41 | 2024-11-18 09:45:00 | 102.92 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-11-27 10:10:00 | 113.43 | 2024-11-27 10:20:00 | 113.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-02 09:35:00 | 115.18 | 2024-12-02 09:40:00 | 114.78 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-12-04 09:45:00 | 118.89 | 2024-12-04 09:50:00 | 118.52 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-05 10:55:00 | 116.92 | 2024-12-05 12:10:00 | 117.25 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-06 09:35:00 | 118.35 | 2024-12-06 09:40:00 | 118.83 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-12-06 09:35:00 | 118.35 | 2024-12-06 11:35:00 | 120.71 | TARGET_HIT | 0.50 | 1.99% |
| SELL | retest1 | 2024-12-12 09:40:00 | 118.72 | 2024-12-12 09:50:00 | 119.03 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-13 10:20:00 | 115.25 | 2024-12-13 10:50:00 | 115.66 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-12-17 10:45:00 | 118.66 | 2024-12-17 10:50:00 | 118.26 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-18 09:55:00 | 115.38 | 2024-12-18 10:15:00 | 115.85 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-12-26 10:55:00 | 109.43 | 2024-12-26 11:10:00 | 109.73 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-30 09:45:00 | 93.47 | 2025-01-30 10:25:00 | 93.08 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-01-31 09:40:00 | 96.06 | 2025-01-31 10:25:00 | 96.73 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-01-31 09:40:00 | 96.06 | 2025-01-31 15:20:00 | 97.51 | TARGET_HIT | 0.50 | 1.51% |
| BUY | retest1 | 2025-02-05 09:45:00 | 97.70 | 2025-02-05 09:50:00 | 98.22 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-02-05 09:45:00 | 97.70 | 2025-02-05 10:05:00 | 97.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-06 09:30:00 | 96.26 | 2025-02-06 09:35:00 | 96.55 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-02-27 10:45:00 | 88.66 | 2025-02-27 10:50:00 | 89.04 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-03-07 09:30:00 | 88.00 | 2025-03-07 09:35:00 | 87.66 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-17 09:30:00 | 86.98 | 2025-03-17 09:35:00 | 86.67 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-18 11:10:00 | 87.28 | 2025-03-18 11:50:00 | 87.04 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-04-16 09:55:00 | 95.09 | 2025-04-16 10:00:00 | 95.49 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-04-16 09:55:00 | 95.09 | 2025-04-16 10:15:00 | 95.09 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-17 10:00:00 | 93.85 | 2025-04-17 10:20:00 | 94.18 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-21 09:35:00 | 95.30 | 2025-04-21 09:45:00 | 95.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-23 10:05:00 | 98.60 | 2025-04-23 10:15:00 | 97.94 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-04-23 10:05:00 | 98.60 | 2025-04-23 11:05:00 | 98.60 | STOP_HIT | 0.50 | 0.00% |
