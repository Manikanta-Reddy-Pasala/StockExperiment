# Jammu & Kashmir Bank Ltd. (J&KBANK)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 141.24
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
| ENTRY1 | 74 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 12 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 103 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 62
- **Target hits / Stop hits / Partials:** 12 / 62 / 29
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 12.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 13 | 27.1% | 2 | 35 | 11 | 0.04% | 1.9% |
| BUY @ 2nd Alert (retest1) | 48 | 13 | 27.1% | 2 | 35 | 11 | 0.04% | 1.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 55 | 28 | 50.9% | 10 | 27 | 18 | 0.20% | 10.9% |
| SELL @ 2nd Alert (retest1) | 55 | 28 | 50.9% | 10 | 27 | 18 | 0.20% | 10.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 103 | 41 | 39.8% | 12 | 62 | 29 | 0.12% | 12.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 130.05 | 130.78 | 0.00 | ORB-short ORB[130.15,131.50] vol=1.9x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-05-16 09:45:00 | 130.49 | 130.70 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 128.95 | 130.28 | 0.00 | ORB-short ORB[129.50,131.40] vol=2.4x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-05-22 09:50:00 | 129.45 | 130.05 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 10:55:00 | 132.00 | 131.15 | 0.00 | ORB-long ORB[130.65,131.85] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-05-28 11:05:00 | 131.50 | 131.18 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:30:00 | 129.75 | 128.98 | 0.00 | ORB-long ORB[127.90,129.50] vol=2.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2024-05-29 10:00:00 | 129.27 | 129.14 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:40:00 | 125.70 | 126.35 | 0.00 | ORB-short ORB[126.00,127.15] vol=1.6x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:55:00 | 125.11 | 126.01 | 0.00 | T1 1.5R @ 125.11 |
| Stop hit — per-position SL triggered | 2024-05-31 10:45:00 | 125.70 | 125.65 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:45:00 | 123.12 | 122.68 | 0.00 | ORB-long ORB[122.01,123.00] vol=2.1x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 09:55:00 | 123.62 | 122.85 | 0.00 | T1 1.5R @ 123.62 |
| Stop hit — per-position SL triggered | 2024-06-12 10:05:00 | 123.12 | 122.92 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:15:00 | 121.99 | 122.93 | 0.00 | ORB-short ORB[123.05,123.80] vol=2.7x ATR=0.25 |
| Stop hit — per-position SL triggered | 2024-06-13 11:25:00 | 122.24 | 122.88 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 10:00:00 | 122.48 | 121.55 | 0.00 | ORB-long ORB[120.70,121.94] vol=4.0x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-06-19 10:15:00 | 121.92 | 121.67 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 120.24 | 120.95 | 0.00 | ORB-short ORB[120.92,121.60] vol=2.0x ATR=0.35 |
| Stop hit — per-position SL triggered | 2024-06-21 11:20:00 | 120.59 | 120.91 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:50:00 | 117.50 | 118.63 | 0.00 | ORB-short ORB[118.55,119.50] vol=1.9x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:00:00 | 116.85 | 118.23 | 0.00 | T1 1.5R @ 116.85 |
| Stop hit — per-position SL triggered | 2024-06-25 14:45:00 | 117.50 | 117.59 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:40:00 | 116.18 | 116.99 | 0.00 | ORB-short ORB[116.65,117.42] vol=2.2x ATR=0.38 |
| Stop hit — per-position SL triggered | 2024-06-27 10:50:00 | 116.56 | 116.93 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:45:00 | 117.45 | 116.52 | 0.00 | ORB-long ORB[115.43,116.55] vol=2.1x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 09:50:00 | 118.33 | 117.21 | 0.00 | T1 1.5R @ 118.33 |
| Stop hit — per-position SL triggered | 2024-06-28 09:55:00 | 117.45 | 117.34 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 11:05:00 | 116.04 | 117.13 | 0.00 | ORB-short ORB[117.50,118.10] vol=3.2x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:25:00 | 115.58 | 116.55 | 0.00 | T1 1.5R @ 115.58 |
| Target hit | 2024-07-03 15:20:00 | 114.44 | 115.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-07-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:50:00 | 110.80 | 109.00 | 0.00 | ORB-long ORB[106.63,108.27] vol=1.7x ATR=0.61 |
| Stop hit — per-position SL triggered | 2024-07-09 10:05:00 | 110.19 | 109.27 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 11:15:00 | 106.25 | 108.12 | 0.00 | ORB-short ORB[108.77,109.83] vol=2.1x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-07-10 11:20:00 | 106.84 | 107.89 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:35:00 | 107.30 | 107.59 | 0.00 | ORB-short ORB[107.50,108.90] vol=2.4x ATR=0.41 |
| Target hit | 2024-07-11 15:20:00 | 107.19 | 107.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-07-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:10:00 | 108.04 | 107.38 | 0.00 | ORB-long ORB[107.10,107.88] vol=2.8x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:30:00 | 108.54 | 107.56 | 0.00 | T1 1.5R @ 108.54 |
| Target hit | 2024-07-15 15:20:00 | 113.73 | 111.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 109.59 | 109.71 | 0.00 | ORB-short ORB[109.62,110.50] vol=1.5x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 108.99 | 109.57 | 0.00 | T1 1.5R @ 108.99 |
| Target hit | 2024-07-23 12:50:00 | 109.12 | 108.29 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — BUY (started 2024-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:30:00 | 110.17 | 109.20 | 0.00 | ORB-long ORB[108.30,109.80] vol=2.3x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-07-24 09:35:00 | 109.54 | 109.30 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 09:50:00 | 111.16 | 111.77 | 0.00 | ORB-short ORB[111.43,112.71] vol=1.6x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 09:55:00 | 110.68 | 111.53 | 0.00 | T1 1.5R @ 110.68 |
| Stop hit — per-position SL triggered | 2024-07-31 10:00:00 | 111.16 | 111.51 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 11:15:00 | 109.60 | 110.08 | 0.00 | ORB-short ORB[110.05,111.30] vol=1.8x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 11:30:00 | 109.18 | 109.94 | 0.00 | T1 1.5R @ 109.18 |
| Target hit | 2024-08-01 14:00:00 | 109.25 | 109.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 22 — BUY (started 2024-08-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 09:50:00 | 108.87 | 108.39 | 0.00 | ORB-long ORB[107.71,108.79] vol=1.9x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 10:10:00 | 109.44 | 108.53 | 0.00 | T1 1.5R @ 109.44 |
| Target hit | 2024-08-02 10:20:00 | 110.59 | 110.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 11:15:00 | 116.54 | 115.34 | 0.00 | ORB-long ORB[114.59,115.99] vol=4.0x ATR=0.48 |
| Stop hit — per-position SL triggered | 2024-08-08 11:20:00 | 116.06 | 115.38 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 11:05:00 | 110.05 | 110.79 | 0.00 | ORB-short ORB[110.18,111.39] vol=4.0x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 11:15:00 | 109.34 | 110.71 | 0.00 | T1 1.5R @ 109.34 |
| Stop hit — per-position SL triggered | 2024-08-14 12:55:00 | 110.05 | 110.41 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:40:00 | 111.14 | 110.38 | 0.00 | ORB-long ORB[109.56,110.90] vol=2.7x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-08-19 09:55:00 | 110.71 | 110.54 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:35:00 | 112.25 | 111.65 | 0.00 | ORB-long ORB[111.09,111.98] vol=1.9x ATR=0.32 |
| Stop hit — per-position SL triggered | 2024-08-22 10:45:00 | 111.93 | 111.69 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 111.00 | 111.20 | 0.00 | ORB-short ORB[111.09,111.58] vol=2.4x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:40:00 | 110.67 | 111.00 | 0.00 | T1 1.5R @ 110.67 |
| Target hit | 2024-08-23 15:20:00 | 110.55 | 110.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2024-08-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:35:00 | 109.04 | 109.35 | 0.00 | ORB-short ORB[109.06,109.85] vol=1.7x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 13:15:00 | 108.63 | 109.00 | 0.00 | T1 1.5R @ 108.63 |
| Target hit | 2024-08-27 15:20:00 | 108.29 | 108.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2024-08-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 11:10:00 | 110.71 | 111.03 | 0.00 | ORB-short ORB[110.80,112.20] vol=4.9x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 11:15:00 | 110.28 | 110.93 | 0.00 | T1 1.5R @ 110.28 |
| Target hit | 2024-08-30 15:20:00 | 110.00 | 110.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2024-09-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 10:45:00 | 109.05 | 109.83 | 0.00 | ORB-short ORB[109.67,110.56] vol=2.9x ATR=0.25 |
| Stop hit — per-position SL triggered | 2024-09-03 12:05:00 | 109.30 | 109.54 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:35:00 | 108.91 | 108.33 | 0.00 | ORB-long ORB[107.61,108.63] vol=1.8x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:55:00 | 109.28 | 108.57 | 0.00 | T1 1.5R @ 109.28 |
| Stop hit — per-position SL triggered | 2024-09-04 10:30:00 | 108.91 | 108.81 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:15:00 | 108.54 | 108.90 | 0.00 | ORB-short ORB[108.80,109.45] vol=1.8x ATR=0.21 |
| Stop hit — per-position SL triggered | 2024-09-05 11:45:00 | 108.75 | 108.80 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:55:00 | 107.81 | 108.91 | 0.00 | ORB-short ORB[108.79,109.89] vol=2.0x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 107.35 | 108.72 | 0.00 | T1 1.5R @ 107.35 |
| Stop hit — per-position SL triggered | 2024-09-06 12:15:00 | 107.81 | 108.02 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:15:00 | 105.77 | 104.98 | 0.00 | ORB-long ORB[104.51,105.40] vol=2.7x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 10:20:00 | 106.31 | 105.34 | 0.00 | T1 1.5R @ 106.31 |
| Stop hit — per-position SL triggered | 2024-09-12 10:50:00 | 105.77 | 105.58 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:30:00 | 107.56 | 107.13 | 0.00 | ORB-long ORB[106.26,107.40] vol=2.7x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 09:40:00 | 108.06 | 107.48 | 0.00 | T1 1.5R @ 108.06 |
| Stop hit — per-position SL triggered | 2024-09-16 10:00:00 | 107.56 | 107.58 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:40:00 | 105.06 | 105.46 | 0.00 | ORB-short ORB[105.20,106.68] vol=3.9x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:50:00 | 104.60 | 105.30 | 0.00 | T1 1.5R @ 104.60 |
| Stop hit — per-position SL triggered | 2024-09-17 13:05:00 | 105.06 | 104.96 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:30:00 | 105.50 | 104.84 | 0.00 | ORB-long ORB[103.61,104.39] vol=15.8x ATR=0.36 |
| Stop hit — per-position SL triggered | 2024-09-18 10:35:00 | 105.14 | 104.85 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:40:00 | 105.20 | 105.71 | 0.00 | ORB-short ORB[105.50,106.65] vol=2.4x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:00:00 | 104.74 | 105.49 | 0.00 | T1 1.5R @ 104.74 |
| Target hit | 2024-09-19 14:40:00 | 104.21 | 104.19 | 0.00 | Trail-exit close>VWAP |

### Cycle 39 — BUY (started 2024-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:00:00 | 106.80 | 105.71 | 0.00 | ORB-long ORB[104.90,105.80] vol=3.8x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:05:00 | 107.31 | 105.94 | 0.00 | T1 1.5R @ 107.31 |
| Stop hit — per-position SL triggered | 2024-09-23 11:50:00 | 106.80 | 106.46 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:30:00 | 110.00 | 109.19 | 0.00 | ORB-long ORB[108.76,109.17] vol=4.2x ATR=0.23 |
| Stop hit — per-position SL triggered | 2024-09-26 10:35:00 | 109.77 | 109.25 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:15:00 | 106.45 | 106.99 | 0.00 | ORB-short ORB[106.66,108.00] vol=1.7x ATR=0.38 |
| Stop hit — per-position SL triggered | 2024-10-01 10:30:00 | 106.83 | 106.85 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:55:00 | 99.88 | 101.72 | 0.00 | ORB-short ORB[102.21,103.17] vol=1.6x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:25:00 | 99.13 | 100.97 | 0.00 | T1 1.5R @ 99.13 |
| Target hit | 2024-10-07 15:20:00 | 97.36 | 99.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2024-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:30:00 | 100.90 | 100.16 | 0.00 | ORB-long ORB[99.37,99.95] vol=3.1x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 09:35:00 | 101.45 | 100.49 | 0.00 | T1 1.5R @ 101.45 |
| Stop hit — per-position SL triggered | 2024-10-10 09:40:00 | 100.90 | 100.53 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:45:00 | 100.25 | 99.66 | 0.00 | ORB-long ORB[99.30,99.65] vol=3.0x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 09:50:00 | 100.59 | 99.90 | 0.00 | T1 1.5R @ 100.59 |
| Stop hit — per-position SL triggered | 2024-10-14 09:55:00 | 100.25 | 99.93 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:00:00 | 98.10 | 98.63 | 0.00 | ORB-short ORB[98.55,99.06] vol=2.2x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 10:05:00 | 97.82 | 98.51 | 0.00 | T1 1.5R @ 97.82 |
| Target hit | 2024-10-15 15:20:00 | 96.90 | 97.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2024-10-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:30:00 | 97.61 | 97.19 | 0.00 | ORB-long ORB[96.57,97.50] vol=2.9x ATR=0.28 |
| Stop hit — per-position SL triggered | 2024-10-16 10:35:00 | 97.33 | 97.19 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:55:00 | 97.10 | 97.91 | 0.00 | ORB-short ORB[97.50,98.60] vol=1.8x ATR=0.31 |
| Stop hit — per-position SL triggered | 2024-10-17 10:45:00 | 97.41 | 97.58 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-10-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:40:00 | 99.20 | 97.87 | 0.00 | ORB-long ORB[96.26,97.70] vol=3.6x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 09:50:00 | 100.16 | 98.59 | 0.00 | T1 1.5R @ 100.16 |
| Stop hit — per-position SL triggered | 2024-10-30 09:55:00 | 99.20 | 98.71 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 09:55:00 | 103.23 | 102.71 | 0.00 | ORB-long ORB[102.00,103.00] vol=3.5x ATR=0.38 |
| Stop hit — per-position SL triggered | 2024-11-12 10:00:00 | 102.85 | 102.72 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:35:00 | 99.30 | 98.51 | 0.00 | ORB-long ORB[97.35,98.80] vol=2.7x ATR=0.28 |
| Stop hit — per-position SL triggered | 2024-11-28 09:40:00 | 99.02 | 98.58 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 11:15:00 | 96.95 | 97.81 | 0.00 | ORB-short ORB[97.40,98.39] vol=3.8x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:35:00 | 96.54 | 97.72 | 0.00 | T1 1.5R @ 96.54 |
| Stop hit — per-position SL triggered | 2024-11-29 13:20:00 | 96.95 | 97.48 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 10:00:00 | 97.52 | 97.19 | 0.00 | ORB-long ORB[96.20,97.45] vol=1.7x ATR=0.35 |
| Stop hit — per-position SL triggered | 2024-12-02 10:35:00 | 97.17 | 97.27 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:05:00 | 105.36 | 106.02 | 0.00 | ORB-short ORB[105.60,107.05] vol=3.0x ATR=0.52 |
| Stop hit — per-position SL triggered | 2024-12-06 10:20:00 | 105.88 | 106.16 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:50:00 | 105.99 | 105.33 | 0.00 | ORB-long ORB[104.70,105.67] vol=2.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-12-10 10:00:00 | 105.60 | 105.35 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:00:00 | 103.17 | 103.88 | 0.00 | ORB-short ORB[103.90,104.84] vol=1.6x ATR=0.23 |
| Stop hit — per-position SL triggered | 2024-12-12 11:20:00 | 103.40 | 103.77 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 10:30:00 | 101.11 | 100.01 | 0.00 | ORB-long ORB[99.30,100.43] vol=2.1x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-12-18 10:35:00 | 100.70 | 100.02 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:05:00 | 99.25 | 100.05 | 0.00 | ORB-short ORB[99.78,100.96] vol=1.6x ATR=0.35 |
| Stop hit — per-position SL triggered | 2024-12-20 10:10:00 | 99.60 | 100.04 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:30:00 | 101.56 | 100.92 | 0.00 | ORB-long ORB[100.42,101.20] vol=3.6x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-01-02 10:35:00 | 101.32 | 100.99 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:45:00 | 97.06 | 97.39 | 0.00 | ORB-short ORB[97.25,97.98] vol=2.1x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-01-08 09:50:00 | 97.39 | 97.39 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:20:00 | 95.30 | 94.36 | 0.00 | ORB-long ORB[93.45,94.60] vol=2.3x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-01-16 10:30:00 | 94.89 | 94.80 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:30:00 | 96.63 | 95.89 | 0.00 | ORB-long ORB[94.80,95.90] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-01-20 10:45:00 | 96.22 | 95.98 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:30:00 | 96.80 | 97.73 | 0.00 | ORB-short ORB[97.30,98.40] vol=1.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-01-21 09:35:00 | 97.25 | 97.70 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:40:00 | 91.90 | 91.23 | 0.00 | ORB-long ORB[90.46,91.65] vol=2.3x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-01-23 10:45:00 | 91.53 | 91.31 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:45:00 | 97.00 | 96.04 | 0.00 | ORB-long ORB[94.40,95.79] vol=6.5x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-01-30 09:50:00 | 96.55 | 96.27 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-02-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:45:00 | 105.00 | 103.75 | 0.00 | ORB-long ORB[101.00,102.47] vol=11.0x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-02-04 10:10:00 | 104.39 | 104.25 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-02-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 10:50:00 | 98.06 | 98.61 | 0.00 | ORB-short ORB[98.92,99.90] vol=1.5x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-02-10 11:20:00 | 98.39 | 98.57 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-02-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:45:00 | 96.99 | 97.23 | 0.00 | ORB-short ORB[97.00,97.88] vol=1.9x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 10:00:00 | 96.53 | 97.12 | 0.00 | T1 1.5R @ 96.53 |
| Stop hit — per-position SL triggered | 2025-02-11 10:25:00 | 96.99 | 97.02 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 09:30:00 | 95.41 | 95.87 | 0.00 | ORB-short ORB[95.68,96.55] vol=1.6x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-02-18 09:35:00 | 95.78 | 95.86 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-02-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 10:25:00 | 97.27 | 97.78 | 0.00 | ORB-short ORB[97.70,98.65] vol=2.5x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 12:10:00 | 96.69 | 97.43 | 0.00 | T1 1.5R @ 96.69 |
| Stop hit — per-position SL triggered | 2025-02-27 14:20:00 | 97.27 | 97.27 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:45:00 | 94.41 | 93.96 | 0.00 | ORB-long ORB[93.07,94.09] vol=3.4x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-03-05 10:00:00 | 94.06 | 93.99 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:30:00 | 92.64 | 91.85 | 0.00 | ORB-long ORB[91.03,92.29] vol=2.5x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-03-17 09:35:00 | 92.31 | 91.89 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-03-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 10:55:00 | 95.42 | 95.69 | 0.00 | ORB-short ORB[95.53,96.43] vol=1.7x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-03-26 11:10:00 | 95.72 | 95.67 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-03-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 09:45:00 | 93.81 | 92.99 | 0.00 | ORB-long ORB[92.21,93.50] vol=2.0x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-03-27 09:50:00 | 93.34 | 93.06 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-04-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:20:00 | 110.05 | 106.06 | 0.00 | ORB-long ORB[101.60,103.16] vol=9.0x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-04-21 10:25:00 | 108.94 | 106.81 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 09:30:00 | 130.05 | 2024-05-16 09:45:00 | 130.49 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-22 09:40:00 | 128.95 | 2024-05-22 09:50:00 | 129.45 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-28 10:55:00 | 132.00 | 2024-05-28 11:05:00 | 131.50 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-05-29 09:30:00 | 129.75 | 2024-05-29 10:00:00 | 129.27 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-05-31 09:40:00 | 125.70 | 2024-05-31 09:55:00 | 125.11 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-31 09:40:00 | 125.70 | 2024-05-31 10:45:00 | 125.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-12 09:45:00 | 123.12 | 2024-06-12 09:55:00 | 123.62 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-06-12 09:45:00 | 123.12 | 2024-06-12 10:05:00 | 123.12 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 11:15:00 | 121.99 | 2024-06-13 11:25:00 | 122.24 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-06-19 10:00:00 | 122.48 | 2024-06-19 10:15:00 | 121.92 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-06-21 10:45:00 | 120.24 | 2024-06-21 11:20:00 | 120.59 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-06-25 10:50:00 | 117.50 | 2024-06-25 11:00:00 | 116.85 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-06-25 10:50:00 | 117.50 | 2024-06-25 14:45:00 | 117.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-27 10:40:00 | 116.18 | 2024-06-27 10:50:00 | 116.56 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-06-28 09:45:00 | 117.45 | 2024-06-28 09:50:00 | 118.33 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-06-28 09:45:00 | 117.45 | 2024-06-28 09:55:00 | 117.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-03 11:05:00 | 116.04 | 2024-07-03 11:25:00 | 115.58 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-03 11:05:00 | 116.04 | 2024-07-03 15:20:00 | 114.44 | TARGET_HIT | 0.50 | 1.38% |
| BUY | retest1 | 2024-07-09 09:50:00 | 110.80 | 2024-07-09 10:05:00 | 110.19 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2024-07-10 11:15:00 | 106.25 | 2024-07-10 11:20:00 | 106.84 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2024-07-11 10:35:00 | 107.30 | 2024-07-11 15:20:00 | 107.19 | TARGET_HIT | 1.00 | 0.10% |
| BUY | retest1 | 2024-07-15 10:10:00 | 108.04 | 2024-07-15 10:30:00 | 108.54 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-15 10:10:00 | 108.04 | 2024-07-15 15:20:00 | 113.73 | TARGET_HIT | 0.50 | 5.27% |
| SELL | retest1 | 2024-07-23 11:15:00 | 109.59 | 2024-07-23 12:15:00 | 108.99 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-07-23 11:15:00 | 109.59 | 2024-07-23 12:50:00 | 109.12 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-24 09:30:00 | 110.17 | 2024-07-24 09:35:00 | 109.54 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-07-31 09:50:00 | 111.16 | 2024-07-31 09:55:00 | 110.68 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-07-31 09:50:00 | 111.16 | 2024-07-31 10:00:00 | 111.16 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-01 11:15:00 | 109.60 | 2024-08-01 11:30:00 | 109.18 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-08-01 11:15:00 | 109.60 | 2024-08-01 14:00:00 | 109.25 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2024-08-02 09:50:00 | 108.87 | 2024-08-02 10:10:00 | 109.44 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-08-02 09:50:00 | 108.87 | 2024-08-02 10:20:00 | 110.59 | TARGET_HIT | 0.50 | 1.58% |
| BUY | retest1 | 2024-08-08 11:15:00 | 116.54 | 2024-08-08 11:20:00 | 116.06 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-08-14 11:05:00 | 110.05 | 2024-08-14 11:15:00 | 109.34 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-08-14 11:05:00 | 110.05 | 2024-08-14 12:55:00 | 110.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-19 09:40:00 | 111.14 | 2024-08-19 09:55:00 | 110.71 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-22 10:35:00 | 112.25 | 2024-08-22 10:45:00 | 111.93 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-23 09:30:00 | 111.00 | 2024-08-23 11:40:00 | 110.67 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-08-23 09:30:00 | 111.00 | 2024-08-23 15:20:00 | 110.55 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2024-08-27 09:35:00 | 109.04 | 2024-08-27 13:15:00 | 108.63 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-08-27 09:35:00 | 109.04 | 2024-08-27 15:20:00 | 108.29 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2024-08-30 11:10:00 | 110.71 | 2024-08-30 11:15:00 | 110.28 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-08-30 11:10:00 | 110.71 | 2024-08-30 15:20:00 | 110.00 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2024-09-03 10:45:00 | 109.05 | 2024-09-03 12:05:00 | 109.30 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-04 09:35:00 | 108.91 | 2024-09-04 09:55:00 | 109.28 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-04 09:35:00 | 108.91 | 2024-09-04 10:30:00 | 108.91 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-05 10:15:00 | 108.54 | 2024-09-05 11:45:00 | 108.75 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-09-06 09:55:00 | 107.81 | 2024-09-06 10:05:00 | 107.35 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-09-06 09:55:00 | 107.81 | 2024-09-06 12:15:00 | 107.81 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-12 10:15:00 | 105.77 | 2024-09-12 10:20:00 | 106.31 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-09-12 10:15:00 | 105.77 | 2024-09-12 10:50:00 | 105.77 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-16 09:30:00 | 107.56 | 2024-09-16 09:40:00 | 108.06 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-09-16 09:30:00 | 107.56 | 2024-09-16 10:00:00 | 107.56 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-17 09:40:00 | 105.06 | 2024-09-17 09:50:00 | 104.60 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-09-17 09:40:00 | 105.06 | 2024-09-17 13:05:00 | 105.06 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-18 10:30:00 | 105.50 | 2024-09-18 10:35:00 | 105.14 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-19 09:40:00 | 105.20 | 2024-09-19 10:00:00 | 104.74 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-19 09:40:00 | 105.20 | 2024-09-19 14:40:00 | 104.21 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2024-09-23 11:00:00 | 106.80 | 2024-09-23 11:05:00 | 107.31 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-09-23 11:00:00 | 106.80 | 2024-09-23 11:50:00 | 106.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-26 10:30:00 | 110.00 | 2024-09-26 10:35:00 | 109.77 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-10-01 10:15:00 | 106.45 | 2024-10-01 10:30:00 | 106.83 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-07 09:55:00 | 99.88 | 2024-10-07 10:25:00 | 99.13 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2024-10-07 09:55:00 | 99.88 | 2024-10-07 15:20:00 | 97.36 | TARGET_HIT | 0.50 | 2.52% |
| BUY | retest1 | 2024-10-10 09:30:00 | 100.90 | 2024-10-10 09:35:00 | 101.45 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-10-10 09:30:00 | 100.90 | 2024-10-10 09:40:00 | 100.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-14 09:45:00 | 100.25 | 2024-10-14 09:50:00 | 100.59 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-10-14 09:45:00 | 100.25 | 2024-10-14 09:55:00 | 100.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-15 10:00:00 | 98.10 | 2024-10-15 10:05:00 | 97.82 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-10-15 10:00:00 | 98.10 | 2024-10-15 15:20:00 | 96.90 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2024-10-16 10:30:00 | 97.61 | 2024-10-16 10:35:00 | 97.33 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-17 09:55:00 | 97.10 | 2024-10-17 10:45:00 | 97.41 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-30 09:40:00 | 99.20 | 2024-10-30 09:50:00 | 100.16 | PARTIAL | 0.50 | 0.97% |
| BUY | retest1 | 2024-10-30 09:40:00 | 99.20 | 2024-10-30 09:55:00 | 99.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-12 09:55:00 | 103.23 | 2024-11-12 10:00:00 | 102.85 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-11-28 09:35:00 | 99.30 | 2024-11-28 09:40:00 | 99.02 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-29 11:15:00 | 96.95 | 2024-11-29 11:35:00 | 96.54 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-11-29 11:15:00 | 96.95 | 2024-11-29 13:20:00 | 96.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-02 10:00:00 | 97.52 | 2024-12-02 10:35:00 | 97.17 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-06 10:05:00 | 105.36 | 2024-12-06 10:20:00 | 105.88 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-12-10 09:50:00 | 105.99 | 2024-12-10 10:00:00 | 105.60 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-12-12 11:00:00 | 103.17 | 2024-12-12 11:20:00 | 103.40 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-12-18 10:30:00 | 101.11 | 2024-12-18 10:35:00 | 100.70 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-20 10:05:00 | 99.25 | 2024-12-20 10:10:00 | 99.60 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-02 10:30:00 | 101.56 | 2025-01-02 10:35:00 | 101.32 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-01-08 09:45:00 | 97.06 | 2025-01-08 09:50:00 | 97.39 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-16 10:20:00 | 95.30 | 2025-01-16 10:30:00 | 94.89 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-01-20 10:30:00 | 96.63 | 2025-01-20 10:45:00 | 96.22 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-01-21 09:30:00 | 96.80 | 2025-01-21 09:35:00 | 97.25 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-01-23 10:40:00 | 91.90 | 2025-01-23 10:45:00 | 91.53 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-01-30 09:45:00 | 97.00 | 2025-01-30 09:50:00 | 96.55 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-02-04 09:45:00 | 105.00 | 2025-02-04 10:10:00 | 104.39 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2025-02-10 10:50:00 | 98.06 | 2025-02-10 11:20:00 | 98.39 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-11 09:45:00 | 96.99 | 2025-02-11 10:00:00 | 96.53 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-02-11 09:45:00 | 96.99 | 2025-02-11 10:25:00 | 96.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-18 09:30:00 | 95.41 | 2025-02-18 09:35:00 | 95.78 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-02-27 10:25:00 | 97.27 | 2025-02-27 12:10:00 | 96.69 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-02-27 10:25:00 | 97.27 | 2025-02-27 14:20:00 | 97.27 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-05 09:45:00 | 94.41 | 2025-03-05 10:00:00 | 94.06 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-03-17 09:30:00 | 92.64 | 2025-03-17 09:35:00 | 92.31 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-03-26 10:55:00 | 95.42 | 2025-03-26 11:10:00 | 95.72 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-03-27 09:45:00 | 93.81 | 2025-03-27 09:50:00 | 93.34 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-04-21 10:20:00 | 110.05 | 2025-04-21 10:25:00 | 108.94 | STOP_HIT | 1.00 | -1.01% |
