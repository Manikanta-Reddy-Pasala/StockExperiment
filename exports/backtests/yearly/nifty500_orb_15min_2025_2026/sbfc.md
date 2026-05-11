# SBFC Finance Ltd. (SBFC)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 98.60
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
| ENTRY1 | 65 |
| ENTRY2 | 0 |
| PARTIAL | 30 |
| TARGET_HIT | 19 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 95 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 49 / 46
- **Target hits / Stop hits / Partials:** 19 / 46 / 30
- **Avg / median % per leg:** 0.26% / 0.11%
- **Sum % (uncompounded):** 24.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 21 | 46.7% | 9 | 24 | 12 | 0.31% | 13.8% |
| BUY @ 2nd Alert (retest1) | 45 | 21 | 46.7% | 9 | 24 | 12 | 0.31% | 13.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 50 | 28 | 56.0% | 10 | 22 | 18 | 0.21% | 10.7% |
| SELL @ 2nd Alert (retest1) | 50 | 28 | 56.0% | 10 | 22 | 18 | 0.21% | 10.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 95 | 49 | 51.6% | 19 | 46 | 30 | 0.26% | 24.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 10:00:00 | 108.00 | 108.65 | 0.00 | ORB-short ORB[108.62,110.00] vol=2.2x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 11:00:00 | 107.37 | 108.51 | 0.00 | T1 1.5R @ 107.37 |
| Target hit | 2025-05-16 14:55:00 | 107.67 | 107.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2025-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:35:00 | 107.18 | 108.18 | 0.00 | ORB-short ORB[108.00,109.33] vol=2.2x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-05-19 09:45:00 | 107.72 | 108.07 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 10:40:00 | 107.04 | 106.48 | 0.00 | ORB-long ORB[105.50,106.70] vol=2.1x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-05-27 10:55:00 | 106.61 | 106.49 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:55:00 | 113.10 | 114.09 | 0.00 | ORB-short ORB[113.61,115.28] vol=1.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-06-03 10:50:00 | 113.66 | 113.90 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:10:00 | 106.92 | 107.44 | 0.00 | ORB-short ORB[107.00,108.35] vol=4.5x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:35:00 | 106.31 | 107.36 | 0.00 | T1 1.5R @ 106.31 |
| Target hit | 2025-06-19 15:20:00 | 104.38 | 105.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2025-06-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:40:00 | 105.86 | 105.07 | 0.00 | ORB-long ORB[104.01,105.40] vol=2.8x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-06-24 09:45:00 | 105.39 | 105.09 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:20:00 | 105.64 | 106.46 | 0.00 | ORB-short ORB[106.21,107.40] vol=1.9x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 11:10:00 | 105.03 | 106.21 | 0.00 | T1 1.5R @ 105.03 |
| Stop hit — per-position SL triggered | 2025-06-26 12:45:00 | 105.64 | 105.90 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:50:00 | 116.44 | 116.91 | 0.00 | ORB-short ORB[116.65,117.74] vol=2.3x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:10:00 | 115.92 | 116.68 | 0.00 | T1 1.5R @ 115.92 |
| Stop hit — per-position SL triggered | 2025-07-18 11:10:00 | 116.44 | 115.90 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:50:00 | 117.18 | 117.53 | 0.00 | ORB-short ORB[118.08,119.16] vol=12.4x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 10:45:00 | 116.67 | 117.41 | 0.00 | T1 1.5R @ 116.67 |
| Target hit | 2025-07-24 15:20:00 | 114.45 | 116.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2025-08-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 09:40:00 | 104.11 | 104.74 | 0.00 | ORB-short ORB[104.44,105.47] vol=2.1x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 10:00:00 | 103.52 | 104.25 | 0.00 | T1 1.5R @ 103.52 |
| Stop hit — per-position SL triggered | 2025-08-01 10:25:00 | 104.11 | 104.22 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-08-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:55:00 | 102.25 | 103.23 | 0.00 | ORB-short ORB[103.39,104.55] vol=1.9x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-08-06 11:50:00 | 102.59 | 103.04 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-08-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 11:00:00 | 103.54 | 102.42 | 0.00 | ORB-long ORB[102.03,103.11] vol=2.1x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-08-13 15:05:00 | 103.19 | 102.77 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 09:30:00 | 104.19 | 103.91 | 0.00 | ORB-long ORB[103.31,104.13] vol=2.8x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 09:45:00 | 104.66 | 104.34 | 0.00 | T1 1.5R @ 104.66 |
| Target hit | 2025-08-14 12:55:00 | 104.88 | 104.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2025-08-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:55:00 | 110.47 | 109.54 | 0.00 | ORB-long ORB[108.67,110.30] vol=2.0x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-08-19 10:00:00 | 109.97 | 109.59 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-08-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:55:00 | 111.33 | 112.68 | 0.00 | ORB-short ORB[112.51,114.00] vol=2.1x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-08-25 11:10:00 | 111.77 | 112.65 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-09-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:45:00 | 106.93 | 106.11 | 0.00 | ORB-long ORB[105.35,106.79] vol=1.8x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-09-01 11:10:00 | 106.44 | 106.17 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-09-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 11:10:00 | 107.13 | 106.67 | 0.00 | ORB-long ORB[106.10,106.90] vol=3.2x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-09-12 11:25:00 | 106.92 | 106.69 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 09:50:00 | 106.96 | 107.30 | 0.00 | ORB-short ORB[107.30,108.30] vol=5.6x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 10:30:00 | 106.44 | 107.13 | 0.00 | T1 1.5R @ 106.44 |
| Stop hit — per-position SL triggered | 2025-09-19 10:45:00 | 106.96 | 107.10 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-09-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 09:45:00 | 107.99 | 107.71 | 0.00 | ORB-long ORB[106.80,107.56] vol=2.2x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 10:25:00 | 108.44 | 108.07 | 0.00 | T1 1.5R @ 108.44 |
| Target hit | 2025-09-22 15:20:00 | 109.50 | 108.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-09-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:00:00 | 112.59 | 112.11 | 0.00 | ORB-long ORB[110.94,112.48] vol=2.6x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-09-24 10:10:00 | 112.04 | 112.12 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:15:00 | 109.82 | 110.38 | 0.00 | ORB-short ORB[110.22,111.60] vol=1.9x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-10-08 10:25:00 | 110.18 | 110.35 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:15:00 | 109.51 | 109.12 | 0.00 | ORB-long ORB[108.38,109.15] vol=8.5x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-10-10 10:20:00 | 109.17 | 109.12 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-10-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:20:00 | 108.23 | 108.01 | 0.00 | ORB-long ORB[107.17,108.20] vol=7.6x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:30:00 | 108.69 | 108.13 | 0.00 | T1 1.5R @ 108.69 |
| Target hit | 2025-10-17 15:20:00 | 112.39 | 111.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2025-10-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:40:00 | 114.83 | 113.21 | 0.00 | ORB-long ORB[111.54,112.52] vol=3.2x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-10-24 09:45:00 | 114.24 | 113.66 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-10-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 11:05:00 | 114.05 | 115.17 | 0.00 | ORB-short ORB[115.00,116.39] vol=2.5x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-10-27 11:20:00 | 114.37 | 115.10 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 10:15:00 | 115.07 | 114.76 | 0.00 | ORB-long ORB[113.60,114.79] vol=5.5x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-10-28 11:45:00 | 114.71 | 114.79 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:30:00 | 115.27 | 114.68 | 0.00 | ORB-long ORB[113.74,114.83] vol=3.2x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 09:40:00 | 115.75 | 115.55 | 0.00 | T1 1.5R @ 115.75 |
| Target hit | 2025-10-31 09:50:00 | 115.74 | 115.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2025-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 09:30:00 | 112.82 | 112.15 | 0.00 | ORB-long ORB[111.02,112.50] vol=2.4x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-11-07 09:35:00 | 112.30 | 112.27 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-11-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 09:50:00 | 109.04 | 109.71 | 0.00 | ORB-short ORB[109.35,110.94] vol=1.9x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-11-12 09:55:00 | 109.42 | 109.68 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-11-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:35:00 | 111.35 | 110.36 | 0.00 | ORB-long ORB[109.10,110.25] vol=5.1x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 09:40:00 | 111.95 | 110.65 | 0.00 | T1 1.5R @ 111.95 |
| Stop hit — per-position SL triggered | 2025-11-17 09:55:00 | 111.35 | 112.06 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-11-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:00:00 | 107.57 | 108.33 | 0.00 | ORB-short ORB[108.43,109.58] vol=3.6x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:50:00 | 107.15 | 108.22 | 0.00 | T1 1.5R @ 107.15 |
| Target hit | 2025-11-27 15:00:00 | 106.89 | 106.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — BUY (started 2025-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:50:00 | 107.71 | 107.10 | 0.00 | ORB-long ORB[106.66,107.34] vol=1.6x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 09:55:00 | 108.35 | 107.38 | 0.00 | T1 1.5R @ 108.35 |
| Target hit | 2025-11-28 11:35:00 | 107.91 | 107.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2025-12-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 11:00:00 | 107.12 | 107.14 | 0.00 | ORB-short ORB[107.13,107.66] vol=3.1x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 11:15:00 | 106.77 | 107.13 | 0.00 | T1 1.5R @ 106.77 |
| Target hit | 2025-12-02 15:20:00 | 106.34 | 106.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2025-12-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:10:00 | 105.50 | 105.93 | 0.00 | ORB-short ORB[105.57,106.90] vol=1.8x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-12-03 11:10:00 | 105.86 | 105.83 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:30:00 | 104.88 | 105.38 | 0.00 | ORB-short ORB[105.10,106.10] vol=2.3x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:40:00 | 104.40 | 105.19 | 0.00 | T1 1.5R @ 104.40 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 104.88 | 105.04 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-12-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:20:00 | 105.91 | 105.51 | 0.00 | ORB-long ORB[104.88,105.71] vol=2.0x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-12-11 10:45:00 | 105.60 | 105.67 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-12-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 09:40:00 | 104.45 | 104.99 | 0.00 | ORB-short ORB[104.88,105.52] vol=2.3x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-12-15 09:45:00 | 104.74 | 104.91 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-12-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:40:00 | 103.82 | 103.35 | 0.00 | ORB-long ORB[102.99,103.73] vol=4.5x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-12-19 10:45:00 | 103.51 | 103.35 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-12-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:30:00 | 106.14 | 105.69 | 0.00 | ORB-long ORB[104.65,106.00] vol=3.1x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-12-23 09:40:00 | 105.84 | 105.75 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-12-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:00:00 | 105.05 | 105.48 | 0.00 | ORB-short ORB[105.42,106.10] vol=1.8x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 12:30:00 | 104.68 | 105.32 | 0.00 | T1 1.5R @ 104.68 |
| Target hit | 2025-12-26 15:20:00 | 104.64 | 104.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2025-12-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:05:00 | 103.70 | 104.12 | 0.00 | ORB-short ORB[104.11,104.74] vol=1.8x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:15:00 | 103.35 | 104.07 | 0.00 | T1 1.5R @ 103.35 |
| Stop hit — per-position SL triggered | 2025-12-29 11:20:00 | 103.70 | 104.05 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-12-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:35:00 | 104.77 | 104.29 | 0.00 | ORB-long ORB[103.80,104.61] vol=2.8x ATR=0.27 |
| Stop hit — per-position SL triggered | 2025-12-31 10:40:00 | 104.50 | 104.55 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2026-01-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 10:00:00 | 102.22 | 102.86 | 0.00 | ORB-short ORB[102.70,103.49] vol=1.9x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 13:05:00 | 101.88 | 102.37 | 0.00 | T1 1.5R @ 101.88 |
| Target hit | 2026-01-05 15:20:00 | 101.91 | 102.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2026-01-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:50:00 | 102.80 | 102.33 | 0.00 | ORB-long ORB[101.89,102.55] vol=1.6x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:00:00 | 103.23 | 102.48 | 0.00 | T1 1.5R @ 103.23 |
| Stop hit — per-position SL triggered | 2026-01-06 10:10:00 | 102.80 | 102.51 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-01-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:45:00 | 103.02 | 103.26 | 0.00 | ORB-short ORB[103.11,103.80] vol=2.2x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:00:00 | 102.68 | 103.23 | 0.00 | T1 1.5R @ 102.68 |
| Target hit | 2026-01-08 14:45:00 | 102.91 | 102.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — BUY (started 2026-01-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 09:35:00 | 102.54 | 102.15 | 0.00 | ORB-long ORB[101.50,102.10] vol=5.6x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:35:00 | 102.95 | 102.53 | 0.00 | T1 1.5R @ 102.95 |
| Target hit | 2026-01-16 14:50:00 | 107.13 | 107.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2026-01-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 10:45:00 | 103.31 | 102.39 | 0.00 | ORB-long ORB[101.29,102.48] vol=1.7x ATR=0.34 |
| Stop hit — per-position SL triggered | 2026-01-22 10:55:00 | 102.97 | 102.48 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2026-01-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 10:55:00 | 102.66 | 103.40 | 0.00 | ORB-short ORB[102.88,104.10] vol=1.7x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:00:00 | 102.21 | 103.31 | 0.00 | T1 1.5R @ 102.21 |
| Target hit | 2026-01-23 15:20:00 | 101.97 | 102.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2026-02-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 10:00:00 | 92.14 | 91.54 | 0.00 | ORB-long ORB[90.01,90.87] vol=4.4x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 12:40:00 | 92.63 | 91.90 | 0.00 | T1 1.5R @ 92.63 |
| Target hit | 2026-02-06 15:20:00 | 93.70 | 92.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2026-02-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:30:00 | 97.20 | 96.22 | 0.00 | ORB-long ORB[94.95,95.75] vol=1.9x ATR=0.34 |
| Stop hit — per-position SL triggered | 2026-02-17 15:20:00 | 97.07 | 96.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 96.17 | 96.96 | 0.00 | ORB-short ORB[96.91,97.50] vol=1.6x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-02-18 10:00:00 | 96.42 | 96.78 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-02-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:00:00 | 96.33 | 96.59 | 0.00 | ORB-short ORB[96.52,97.26] vol=1.9x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:05:00 | 96.03 | 96.46 | 0.00 | T1 1.5R @ 96.03 |
| Stop hit — per-position SL triggered | 2026-02-19 12:15:00 | 96.33 | 96.45 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:15:00 | 97.17 | 96.55 | 0.00 | ORB-long ORB[95.81,96.95] vol=1.6x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-02-24 12:45:00 | 96.79 | 96.79 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 96.70 | 96.33 | 0.00 | ORB-long ORB[96.05,96.62] vol=3.8x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:00:00 | 97.19 | 96.65 | 0.00 | T1 1.5R @ 97.19 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 96.70 | 96.84 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:55:00 | 95.13 | 95.59 | 0.00 | ORB-short ORB[95.39,96.69] vol=1.7x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:35:00 | 94.63 | 95.19 | 0.00 | T1 1.5R @ 94.63 |
| Stop hit — per-position SL triggered | 2026-02-27 12:45:00 | 95.13 | 94.97 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-03-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:30:00 | 94.77 | 93.69 | 0.00 | ORB-long ORB[92.49,93.52] vol=4.9x ATR=0.39 |
| Stop hit — per-position SL triggered | 2026-03-06 10:35:00 | 94.38 | 93.76 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-03-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:30:00 | 86.98 | 87.37 | 0.00 | ORB-short ORB[87.11,88.00] vol=1.9x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-03-19 09:50:00 | 87.33 | 87.34 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-04-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:40:00 | 92.91 | 92.18 | 0.00 | ORB-long ORB[91.42,92.30] vol=4.7x ATR=0.54 |
| Stop hit — per-position SL triggered | 2026-04-13 09:45:00 | 92.37 | 92.33 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 97.20 | 96.28 | 0.00 | ORB-long ORB[95.06,95.94] vol=3.4x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:25:00 | 97.84 | 97.24 | 0.00 | T1 1.5R @ 97.84 |
| Target hit | 2026-04-16 11:55:00 | 99.13 | 99.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — SELL (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 97.53 | 98.16 | 0.00 | ORB-short ORB[98.05,99.00] vol=2.4x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-04-22 12:40:00 | 97.96 | 97.65 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-04-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:20:00 | 95.17 | 95.64 | 0.00 | ORB-short ORB[95.41,96.75] vol=5.1x ATR=0.26 |
| Stop hit — per-position SL triggered | 2026-04-24 10:35:00 | 95.43 | 95.62 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-04-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:10:00 | 94.40 | 94.63 | 0.00 | ORB-short ORB[94.52,95.07] vol=1.5x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:25:00 | 94.16 | 94.61 | 0.00 | T1 1.5R @ 94.16 |
| Target hit | 2026-04-28 14:20:00 | 94.30 | 94.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 63 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:15:00 | 92.89 | 93.56 | 0.00 | ORB-short ORB[93.09,93.97] vol=2.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2026-05-05 11:25:00 | 93.17 | 93.56 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:55:00 | 94.93 | 94.56 | 0.00 | ORB-long ORB[93.68,94.71] vol=2.0x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-05-06 10:50:00 | 94.55 | 94.66 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 95.97 | 95.66 | 0.00 | ORB-long ORB[94.87,95.93] vol=2.6x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:40:00 | 96.49 | 95.84 | 0.00 | T1 1.5R @ 96.49 |
| Target hit | 2026-05-07 14:50:00 | 96.90 | 97.09 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-16 10:00:00 | 108.00 | 2025-05-16 11:00:00 | 107.37 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-05-16 10:00:00 | 108.00 | 2025-05-16 14:55:00 | 107.67 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-05-19 09:35:00 | 107.18 | 2025-05-19 09:45:00 | 107.72 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-05-27 10:40:00 | 107.04 | 2025-05-27 10:55:00 | 106.61 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-06-03 09:55:00 | 113.10 | 2025-06-03 10:50:00 | 113.66 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-06-19 10:10:00 | 106.92 | 2025-06-19 10:35:00 | 106.31 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-06-19 10:10:00 | 106.92 | 2025-06-19 15:20:00 | 104.38 | TARGET_HIT | 0.50 | 2.38% |
| BUY | retest1 | 2025-06-24 09:40:00 | 105.86 | 2025-06-24 09:45:00 | 105.39 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-06-26 10:20:00 | 105.64 | 2025-06-26 11:10:00 | 105.03 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-06-26 10:20:00 | 105.64 | 2025-06-26 12:45:00 | 105.64 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 09:50:00 | 116.44 | 2025-07-18 10:10:00 | 115.92 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-07-18 09:50:00 | 116.44 | 2025-07-18 11:10:00 | 116.44 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 09:50:00 | 117.18 | 2025-07-24 10:45:00 | 116.67 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-07-24 09:50:00 | 117.18 | 2025-07-24 15:20:00 | 114.45 | TARGET_HIT | 0.50 | 2.33% |
| SELL | retest1 | 2025-08-01 09:40:00 | 104.11 | 2025-08-01 10:00:00 | 103.52 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-08-01 09:40:00 | 104.11 | 2025-08-01 10:25:00 | 104.11 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-06 10:55:00 | 102.25 | 2025-08-06 11:50:00 | 102.59 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-08-13 11:00:00 | 103.54 | 2025-08-13 15:05:00 | 103.19 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-08-14 09:30:00 | 104.19 | 2025-08-14 09:45:00 | 104.66 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-08-14 09:30:00 | 104.19 | 2025-08-14 12:55:00 | 104.88 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2025-08-19 09:55:00 | 110.47 | 2025-08-19 10:00:00 | 109.97 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-08-25 10:55:00 | 111.33 | 2025-08-25 11:10:00 | 111.77 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-09-01 10:45:00 | 106.93 | 2025-09-01 11:10:00 | 106.44 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-09-12 11:10:00 | 107.13 | 2025-09-12 11:25:00 | 106.92 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-19 09:50:00 | 106.96 | 2025-09-19 10:30:00 | 106.44 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-09-19 09:50:00 | 106.96 | 2025-09-19 10:45:00 | 106.96 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-22 09:45:00 | 107.99 | 2025-09-22 10:25:00 | 108.44 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-09-22 09:45:00 | 107.99 | 2025-09-22 15:20:00 | 109.50 | TARGET_HIT | 0.50 | 1.40% |
| BUY | retest1 | 2025-09-24 10:00:00 | 112.59 | 2025-09-24 10:10:00 | 112.04 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-10-08 10:15:00 | 109.82 | 2025-10-08 10:25:00 | 110.18 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-10 10:15:00 | 109.51 | 2025-10-10 10:20:00 | 109.17 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-17 10:20:00 | 108.23 | 2025-10-17 11:30:00 | 108.69 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-10-17 10:20:00 | 108.23 | 2025-10-17 15:20:00 | 112.39 | TARGET_HIT | 0.50 | 3.84% |
| BUY | retest1 | 2025-10-24 09:40:00 | 114.83 | 2025-10-24 09:45:00 | 114.24 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-10-27 11:05:00 | 114.05 | 2025-10-27 11:20:00 | 114.37 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-28 10:15:00 | 115.07 | 2025-10-28 11:45:00 | 114.71 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-31 09:30:00 | 115.27 | 2025-10-31 09:40:00 | 115.75 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-10-31 09:30:00 | 115.27 | 2025-10-31 09:50:00 | 115.74 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2025-11-07 09:30:00 | 112.82 | 2025-11-07 09:35:00 | 112.30 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-11-12 09:50:00 | 109.04 | 2025-11-12 09:55:00 | 109.42 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-11-17 09:35:00 | 111.35 | 2025-11-17 09:40:00 | 111.95 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-11-17 09:35:00 | 111.35 | 2025-11-17 09:55:00 | 111.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-27 11:00:00 | 107.57 | 2025-11-27 11:50:00 | 107.15 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-11-27 11:00:00 | 107.57 | 2025-11-27 15:00:00 | 106.89 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-11-28 09:50:00 | 107.71 | 2025-11-28 09:55:00 | 108.35 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-11-28 09:50:00 | 107.71 | 2025-11-28 11:35:00 | 107.91 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2025-12-02 11:00:00 | 107.12 | 2025-12-02 11:15:00 | 106.77 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-12-02 11:00:00 | 107.12 | 2025-12-02 15:20:00 | 106.34 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2025-12-03 10:10:00 | 105.50 | 2025-12-03 11:10:00 | 105.86 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-05 09:30:00 | 104.88 | 2025-12-05 09:40:00 | 104.40 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-12-05 09:30:00 | 104.88 | 2025-12-05 10:00:00 | 104.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-11 10:20:00 | 105.91 | 2025-12-11 10:45:00 | 105.60 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-15 09:40:00 | 104.45 | 2025-12-15 09:45:00 | 104.74 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-19 10:40:00 | 103.82 | 2025-12-19 10:45:00 | 103.51 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-23 09:30:00 | 106.14 | 2025-12-23 09:40:00 | 105.84 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-26 11:00:00 | 105.05 | 2025-12-26 12:30:00 | 104.68 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-26 11:00:00 | 105.05 | 2025-12-26 15:20:00 | 104.64 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-29 11:05:00 | 103.70 | 2025-12-29 11:15:00 | 103.35 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-29 11:05:00 | 103.70 | 2025-12-29 11:20:00 | 103.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 10:35:00 | 104.77 | 2025-12-31 10:40:00 | 104.50 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-05 10:00:00 | 102.22 | 2026-01-05 13:05:00 | 101.88 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-05 10:00:00 | 102.22 | 2026-01-05 15:20:00 | 101.91 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2026-01-06 09:50:00 | 102.80 | 2026-01-06 10:00:00 | 103.23 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-01-06 09:50:00 | 102.80 | 2026-01-06 10:10:00 | 102.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 10:45:00 | 103.02 | 2026-01-08 11:00:00 | 102.68 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-08 10:45:00 | 103.02 | 2026-01-08 14:45:00 | 102.91 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2026-01-16 09:35:00 | 102.54 | 2026-01-16 10:35:00 | 102.95 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-01-16 09:35:00 | 102.54 | 2026-01-16 14:50:00 | 107.13 | TARGET_HIT | 0.50 | 4.48% |
| BUY | retest1 | 2026-01-22 10:45:00 | 103.31 | 2026-01-22 10:55:00 | 102.97 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-23 10:55:00 | 102.66 | 2026-01-23 11:00:00 | 102.21 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-01-23 10:55:00 | 102.66 | 2026-01-23 15:20:00 | 101.97 | TARGET_HIT | 0.50 | 0.67% |
| BUY | retest1 | 2026-02-06 10:00:00 | 92.14 | 2026-02-06 12:40:00 | 92.63 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-02-06 10:00:00 | 92.14 | 2026-02-06 15:20:00 | 93.70 | TARGET_HIT | 0.50 | 1.69% |
| BUY | retest1 | 2026-02-17 10:30:00 | 97.20 | 2026-02-17 15:20:00 | 97.07 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2026-02-18 09:45:00 | 96.17 | 2026-02-18 10:00:00 | 96.42 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-19 11:00:00 | 96.33 | 2026-02-19 12:05:00 | 96.03 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-19 11:00:00 | 96.33 | 2026-02-19 12:15:00 | 96.33 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 10:15:00 | 97.17 | 2026-02-24 12:45:00 | 96.79 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-26 09:30:00 | 96.70 | 2026-02-26 10:00:00 | 97.19 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-26 09:30:00 | 96.70 | 2026-02-26 11:30:00 | 96.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:55:00 | 95.13 | 2026-02-27 10:35:00 | 94.63 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-27 09:55:00 | 95.13 | 2026-02-27 12:45:00 | 95.13 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 10:30:00 | 94.77 | 2026-03-06 10:35:00 | 94.38 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-19 09:30:00 | 86.98 | 2026-03-19 09:50:00 | 87.33 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-13 09:40:00 | 92.91 | 2026-04-13 09:45:00 | 92.37 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-04-16 09:30:00 | 97.20 | 2026-04-16 10:25:00 | 97.84 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-16 09:30:00 | 97.20 | 2026-04-16 11:55:00 | 99.13 | TARGET_HIT | 0.50 | 1.99% |
| SELL | retest1 | 2026-04-22 09:45:00 | 97.53 | 2026-04-22 12:40:00 | 97.96 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-24 10:20:00 | 95.17 | 2026-04-24 10:35:00 | 95.43 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-28 11:10:00 | 94.40 | 2026-04-28 11:25:00 | 94.16 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-04-28 11:10:00 | 94.40 | 2026-04-28 14:20:00 | 94.30 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2026-05-05 11:15:00 | 92.89 | 2026-05-05 11:25:00 | 93.17 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-06 09:55:00 | 94.93 | 2026-05-06 10:50:00 | 94.55 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-07 09:35:00 | 95.97 | 2026-05-07 09:40:00 | 96.49 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-05-07 09:35:00 | 95.97 | 2026-05-07 14:50:00 | 96.90 | TARGET_HIT | 0.50 | 0.97% |
