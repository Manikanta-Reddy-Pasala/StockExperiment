# Bank of India (BANKINDIA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 139.85
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
| ENTRY1 | 91 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 17 |
| STOP_HIT | 74 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 122 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 74
- **Target hits / Stop hits / Partials:** 17 / 74 / 31
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 14.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 26 | 37.7% | 9 | 43 | 17 | 0.15% | 10.4% |
| BUY @ 2nd Alert (retest1) | 69 | 26 | 37.7% | 9 | 43 | 17 | 0.15% | 10.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 53 | 22 | 41.5% | 8 | 31 | 14 | 0.08% | 4.3% |
| SELL @ 2nd Alert (retest1) | 53 | 22 | 41.5% | 8 | 31 | 14 | 0.08% | 4.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 122 | 48 | 39.3% | 17 | 74 | 31 | 0.12% | 14.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 09:30:00 | 116.23 | 115.59 | 0.00 | ORB-long ORB[114.57,115.72] vol=3.7x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-05-19 09:35:00 | 115.92 | 115.62 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:35:00 | 116.45 | 116.06 | 0.00 | ORB-long ORB[115.11,116.40] vol=1.8x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 10:05:00 | 117.10 | 116.31 | 0.00 | T1 1.5R @ 117.10 |
| Target hit | 2025-05-21 12:00:00 | 117.00 | 117.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2025-05-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 10:30:00 | 116.81 | 117.57 | 0.00 | ORB-short ORB[116.93,118.55] vol=3.4x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-22 12:50:00 | 116.20 | 117.19 | 0.00 | T1 1.5R @ 116.20 |
| Target hit | 2025-05-22 15:20:00 | 116.40 | 116.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2025-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 117.60 | 116.92 | 0.00 | ORB-long ORB[116.26,117.18] vol=2.0x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-05-23 09:35:00 | 117.24 | 116.95 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:30:00 | 116.52 | 116.95 | 0.00 | ORB-short ORB[116.61,117.72] vol=2.4x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 10:00:00 | 116.13 | 116.80 | 0.00 | T1 1.5R @ 116.13 |
| Stop hit — per-position SL triggered | 2025-05-27 10:10:00 | 116.52 | 116.78 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:35:00 | 118.58 | 118.03 | 0.00 | ORB-long ORB[116.95,118.23] vol=3.8x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-05-28 09:50:00 | 118.26 | 118.13 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 11:10:00 | 118.32 | 118.49 | 0.00 | ORB-short ORB[118.65,119.68] vol=2.7x ATR=0.23 |
| Stop hit — per-position SL triggered | 2025-05-30 11:20:00 | 118.55 | 118.49 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 10:00:00 | 127.23 | 126.39 | 0.00 | ORB-long ORB[125.46,126.80] vol=2.0x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-06-03 10:05:00 | 126.84 | 126.43 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 123.67 | 124.75 | 0.00 | ORB-short ORB[124.25,125.79] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-06-04 10:35:00 | 124.17 | 124.19 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:05:00 | 126.36 | 125.13 | 0.00 | ORB-long ORB[123.48,124.68] vol=7.7x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-06-06 10:10:00 | 125.86 | 125.17 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 11:10:00 | 125.36 | 125.80 | 0.00 | ORB-short ORB[125.37,126.90] vol=5.7x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-06-11 11:35:00 | 125.58 | 125.75 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 09:50:00 | 117.56 | 118.29 | 0.00 | ORB-short ORB[117.88,119.30] vol=1.5x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 116.99 | 117.86 | 0.00 | T1 1.5R @ 116.99 |
| Target hit | 2025-06-19 15:20:00 | 115.67 | 116.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:30:00 | 117.15 | 116.39 | 0.00 | ORB-long ORB[115.68,116.48] vol=2.9x ATR=0.42 |
| Stop hit — per-position SL triggered | 2025-06-24 09:35:00 | 116.73 | 116.45 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:20:00 | 119.35 | 118.30 | 0.00 | ORB-long ORB[117.25,118.95] vol=1.8x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 10:30:00 | 120.00 | 118.78 | 0.00 | T1 1.5R @ 120.00 |
| Stop hit — per-position SL triggered | 2025-06-27 10:40:00 | 119.35 | 119.02 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 11:15:00 | 117.89 | 118.54 | 0.00 | ORB-short ORB[118.30,119.80] vol=3.6x ATR=0.23 |
| Stop hit — per-position SL triggered | 2025-07-03 11:20:00 | 118.12 | 118.53 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 10:20:00 | 118.08 | 118.60 | 0.00 | ORB-short ORB[118.60,119.35] vol=2.2x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 11:35:00 | 117.66 | 118.41 | 0.00 | T1 1.5R @ 117.66 |
| Target hit | 2025-07-04 15:10:00 | 118.00 | 117.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — SELL (started 2025-07-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:00:00 | 118.01 | 118.37 | 0.00 | ORB-short ORB[118.14,119.14] vol=3.3x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 118.34 | 118.35 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 116.41 | 117.55 | 0.00 | ORB-short ORB[117.66,118.58] vol=1.6x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-07-08 11:15:00 | 116.67 | 117.47 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 09:40:00 | 115.40 | 115.92 | 0.00 | ORB-short ORB[115.74,116.46] vol=2.1x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:00:00 | 115.08 | 115.69 | 0.00 | T1 1.5R @ 115.08 |
| Target hit | 2025-07-10 15:20:00 | 114.35 | 114.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:40:00 | 115.62 | 114.83 | 0.00 | ORB-long ORB[113.84,114.85] vol=2.3x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-07-14 09:45:00 | 115.26 | 114.98 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:00:00 | 116.05 | 115.36 | 0.00 | ORB-long ORB[115.19,115.70] vol=2.6x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-07-15 10:20:00 | 115.76 | 115.67 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:35:00 | 116.70 | 116.25 | 0.00 | ORB-long ORB[115.54,116.49] vol=2.5x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-07-16 09:40:00 | 116.44 | 116.31 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:30:00 | 116.77 | 117.59 | 0.00 | ORB-short ORB[117.18,118.47] vol=2.0x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 09:35:00 | 116.34 | 117.44 | 0.00 | T1 1.5R @ 116.34 |
| Stop hit — per-position SL triggered | 2025-07-17 09:50:00 | 116.77 | 117.25 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 11:00:00 | 115.22 | 115.75 | 0.00 | ORB-short ORB[115.76,116.75] vol=1.7x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 115.44 | 115.73 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:30:00 | 114.40 | 114.80 | 0.00 | ORB-short ORB[114.53,115.15] vol=1.8x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 09:40:00 | 114.04 | 114.64 | 0.00 | T1 1.5R @ 114.04 |
| Stop hit — per-position SL triggered | 2025-07-22 09:45:00 | 114.40 | 114.61 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-07-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:10:00 | 113.00 | 113.39 | 0.00 | ORB-short ORB[113.26,113.70] vol=1.8x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-07-24 10:30:00 | 113.21 | 113.35 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-07-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:25:00 | 111.00 | 111.77 | 0.00 | ORB-short ORB[111.46,112.71] vol=1.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-07-29 10:30:00 | 111.39 | 111.75 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:00:00 | 110.52 | 110.91 | 0.00 | ORB-short ORB[110.76,111.35] vol=1.8x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-08-06 10:20:00 | 110.81 | 110.88 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 09:50:00 | 111.92 | 111.34 | 0.00 | ORB-long ORB[110.55,111.54] vol=2.9x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-08-07 09:55:00 | 111.59 | 111.36 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:05:00 | 114.75 | 114.37 | 0.00 | ORB-long ORB[113.84,114.61] vol=1.7x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:40:00 | 115.14 | 114.57 | 0.00 | T1 1.5R @ 115.14 |
| Stop hit — per-position SL triggered | 2025-08-18 12:35:00 | 114.75 | 114.80 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-08-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:50:00 | 115.10 | 114.86 | 0.00 | ORB-long ORB[114.52,114.98] vol=3.0x ATR=0.20 |
| Stop hit — per-position SL triggered | 2025-08-19 09:55:00 | 114.90 | 114.87 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:05:00 | 117.04 | 116.52 | 0.00 | ORB-long ORB[115.90,116.70] vol=2.4x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-08-20 11:00:00 | 116.78 | 116.74 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-08-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:30:00 | 116.87 | 116.45 | 0.00 | ORB-long ORB[116.10,116.80] vol=1.8x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-08-21 10:40:00 | 116.69 | 116.49 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 116.45 | 116.09 | 0.00 | ORB-long ORB[115.52,116.44] vol=2.9x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 10:10:00 | 116.81 | 116.39 | 0.00 | T1 1.5R @ 116.81 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 116.45 | 116.40 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 113.27 | 113.78 | 0.00 | ORB-short ORB[113.41,115.00] vol=2.2x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:35:00 | 112.92 | 113.65 | 0.00 | T1 1.5R @ 112.92 |
| Stop hit — per-position SL triggered | 2025-08-26 09:40:00 | 113.27 | 113.51 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:35:00 | 112.62 | 112.12 | 0.00 | ORB-long ORB[111.42,112.38] vol=2.2x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:45:00 | 113.01 | 112.41 | 0.00 | T1 1.5R @ 113.01 |
| Target hit | 2025-09-02 13:10:00 | 113.10 | 113.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:15:00 | 111.79 | 112.43 | 0.00 | ORB-short ORB[112.39,113.03] vol=3.1x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-09-05 10:20:00 | 112.07 | 112.40 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:30:00 | 113.59 | 113.08 | 0.00 | ORB-long ORB[112.51,113.32] vol=1.8x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-09-08 09:55:00 | 113.28 | 113.25 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:40:00 | 114.19 | 113.82 | 0.00 | ORB-long ORB[112.90,114.09] vol=2.4x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-09-10 09:45:00 | 113.93 | 113.90 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:55:00 | 117.75 | 118.68 | 0.00 | ORB-short ORB[118.59,119.35] vol=2.0x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:15:00 | 117.42 | 118.58 | 0.00 | T1 1.5R @ 117.42 |
| Target hit | 2025-09-12 15:20:00 | 117.31 | 117.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2025-09-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:50:00 | 117.16 | 117.70 | 0.00 | ORB-short ORB[117.70,118.21] vol=1.9x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-09-17 10:05:00 | 117.40 | 117.64 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:00:00 | 120.66 | 119.84 | 0.00 | ORB-long ORB[118.72,120.28] vol=2.1x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-09-18 10:05:00 | 120.27 | 119.92 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:50:00 | 120.50 | 119.84 | 0.00 | ORB-long ORB[118.92,120.10] vol=2.8x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-09-19 10:00:00 | 120.15 | 119.91 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-09-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:00:00 | 122.90 | 122.22 | 0.00 | ORB-long ORB[121.70,122.85] vol=1.9x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:20:00 | 123.51 | 122.65 | 0.00 | T1 1.5R @ 123.51 |
| Stop hit — per-position SL triggered | 2025-09-24 10:35:00 | 122.90 | 122.71 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-09-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:35:00 | 118.05 | 117.36 | 0.00 | ORB-long ORB[116.26,117.75] vol=1.9x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:55:00 | 118.59 | 117.69 | 0.00 | T1 1.5R @ 118.59 |
| Target hit | 2025-09-29 11:35:00 | 118.45 | 118.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — SELL (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:15:00 | 124.15 | 124.79 | 0.00 | ORB-short ORB[124.45,125.33] vol=1.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-10-03 12:00:00 | 124.54 | 124.66 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:50:00 | 124.29 | 125.35 | 0.00 | ORB-short ORB[125.00,126.40] vol=3.4x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-10-06 11:00:00 | 124.72 | 125.31 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 10:55:00 | 126.03 | 125.38 | 0.00 | ORB-long ORB[124.72,125.73] vol=1.6x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-10-09 12:25:00 | 125.68 | 125.56 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-10-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:00:00 | 126.68 | 126.31 | 0.00 | ORB-long ORB[125.55,126.49] vol=3.1x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 10:10:00 | 127.21 | 126.60 | 0.00 | T1 1.5R @ 127.21 |
| Target hit | 2025-10-10 11:05:00 | 126.96 | 126.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — SELL (started 2025-10-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:00:00 | 124.80 | 125.64 | 0.00 | ORB-short ORB[125.50,127.24] vol=3.4x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:15:00 | 124.27 | 125.54 | 0.00 | T1 1.5R @ 124.27 |
| Target hit | 2025-10-14 15:20:00 | 124.32 | 124.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2025-10-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:05:00 | 126.17 | 125.36 | 0.00 | ORB-long ORB[124.36,125.48] vol=1.7x ATR=0.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 12:05:00 | 126.58 | 125.65 | 0.00 | T1 1.5R @ 126.58 |
| Stop hit — per-position SL triggered | 2025-10-15 14:05:00 | 126.17 | 126.05 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-10-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 11:10:00 | 126.01 | 124.94 | 0.00 | ORB-long ORB[124.20,125.81] vol=4.8x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-10-17 11:40:00 | 125.57 | 125.09 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-10-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:40:00 | 125.61 | 124.82 | 0.00 | ORB-long ORB[123.90,125.36] vol=2.1x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:25:00 | 126.50 | 125.39 | 0.00 | T1 1.5R @ 126.50 |
| Target hit | 2025-10-20 15:20:00 | 129.84 | 128.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2025-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:30:00 | 132.42 | 131.66 | 0.00 | ORB-long ORB[130.78,131.94] vol=2.0x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 09:50:00 | 133.06 | 132.26 | 0.00 | T1 1.5R @ 133.06 |
| Target hit | 2025-10-23 15:20:00 | 135.47 | 134.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 11:15:00 | 140.16 | 141.05 | 0.00 | ORB-short ORB[140.73,142.33] vol=1.7x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-10-29 11:25:00 | 140.63 | 141.04 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-10-31 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 10:10:00 | 141.00 | 140.00 | 0.00 | ORB-long ORB[138.40,139.90] vol=2.0x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-10-31 10:40:00 | 140.47 | 140.30 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-11-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:30:00 | 142.95 | 142.13 | 0.00 | ORB-long ORB[141.20,142.60] vol=4.1x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-11-04 09:40:00 | 142.52 | 142.29 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-11-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:45:00 | 140.53 | 141.34 | 0.00 | ORB-short ORB[141.01,142.46] vol=1.9x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-11-06 09:50:00 | 141.02 | 141.27 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:35:00 | 138.87 | 139.39 | 0.00 | ORB-short ORB[138.95,140.44] vol=3.5x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-11-07 09:45:00 | 139.32 | 139.32 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-11-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:35:00 | 142.49 | 143.82 | 0.00 | ORB-short ORB[144.52,146.37] vol=3.6x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-11-11 10:45:00 | 142.98 | 143.72 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-11-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:45:00 | 147.74 | 147.04 | 0.00 | ORB-long ORB[146.00,147.40] vol=1.6x ATR=0.42 |
| Stop hit — per-position SL triggered | 2025-11-12 09:50:00 | 147.32 | 147.03 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-11-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:30:00 | 149.81 | 148.89 | 0.00 | ORB-long ORB[147.25,149.45] vol=3.1x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-11-17 09:45:00 | 149.33 | 149.24 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-11-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 11:05:00 | 147.10 | 146.47 | 0.00 | ORB-long ORB[145.60,146.90] vol=9.1x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 146.67 | 146.53 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-11-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:55:00 | 146.71 | 147.04 | 0.00 | ORB-short ORB[147.24,148.26] vol=1.9x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:25:00 | 146.12 | 146.86 | 0.00 | T1 1.5R @ 146.12 |
| Target hit | 2025-11-21 12:35:00 | 146.18 | 146.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 65 — BUY (started 2025-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:30:00 | 149.74 | 148.67 | 0.00 | ORB-long ORB[147.52,148.99] vol=2.0x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 09:40:00 | 150.41 | 149.29 | 0.00 | T1 1.5R @ 150.41 |
| Stop hit — per-position SL triggered | 2025-11-26 11:20:00 | 149.74 | 150.26 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-11-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:45:00 | 147.21 | 148.13 | 0.00 | ORB-short ORB[148.50,149.69] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-11-27 10:05:00 | 147.62 | 147.95 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:10:00 | 142.64 | 141.70 | 0.00 | ORB-long ORB[141.06,141.96] vol=2.5x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-12-05 10:25:00 | 142.10 | 141.91 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 09:50:00 | 140.84 | 141.71 | 0.00 | ORB-short ORB[141.25,142.89] vol=1.9x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:30:00 | 140.18 | 141.29 | 0.00 | T1 1.5R @ 140.18 |
| Target hit | 2025-12-08 15:20:00 | 138.15 | 138.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — BUY (started 2025-12-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:10:00 | 140.04 | 138.26 | 0.00 | ORB-long ORB[136.75,138.72] vol=1.5x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 12:20:00 | 140.71 | 138.68 | 0.00 | T1 1.5R @ 140.71 |
| Target hit | 2025-12-09 15:20:00 | 141.46 | 139.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2025-12-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 09:55:00 | 140.45 | 141.04 | 0.00 | ORB-short ORB[140.62,142.00] vol=1.7x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-12-16 10:05:00 | 140.77 | 140.99 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-12-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:20:00 | 140.60 | 139.36 | 0.00 | ORB-long ORB[138.24,140.20] vol=2.1x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-12-30 10:40:00 | 140.16 | 139.79 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:15:00 | 144.06 | 143.73 | 0.00 | ORB-long ORB[141.20,143.21] vol=1.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-12-31 13:30:00 | 143.58 | 143.81 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:05:00 | 145.09 | 144.57 | 0.00 | ORB-long ORB[143.51,144.35] vol=1.8x ATR=0.42 |
| Stop hit — per-position SL triggered | 2026-01-01 10:10:00 | 144.67 | 144.60 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 11:15:00 | 151.24 | 150.31 | 0.00 | ORB-long ORB[149.71,150.94] vol=2.2x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-01-07 13:00:00 | 150.89 | 150.70 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:05:00 | 147.86 | 149.75 | 0.00 | ORB-short ORB[149.65,151.33] vol=2.9x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:10:00 | 147.20 | 149.62 | 0.00 | T1 1.5R @ 147.20 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 147.86 | 149.02 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-01-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 09:45:00 | 148.59 | 147.36 | 0.00 | ORB-long ORB[146.10,148.12] vol=1.7x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 10:05:00 | 149.55 | 147.83 | 0.00 | T1 1.5R @ 149.55 |
| Stop hit — per-position SL triggered | 2026-01-09 11:10:00 | 148.59 | 148.29 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 09:30:00 | 163.33 | 162.61 | 0.00 | ORB-long ORB[160.83,163.09] vol=1.9x ATR=0.53 |
| Stop hit — per-position SL triggered | 2026-02-05 09:35:00 | 162.80 | 162.70 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 161.64 | 162.62 | 0.00 | ORB-short ORB[162.04,164.45] vol=1.8x ATR=0.52 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 162.16 | 162.41 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-02-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:00:00 | 161.58 | 161.17 | 0.00 | ORB-long ORB[159.56,161.50] vol=1.8x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:15:00 | 162.49 | 161.41 | 0.00 | T1 1.5R @ 162.49 |
| Target hit | 2026-02-16 15:20:00 | 165.76 | 163.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 167.44 | 166.32 | 0.00 | ORB-long ORB[165.00,166.80] vol=1.7x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:30:00 | 168.24 | 166.80 | 0.00 | T1 1.5R @ 168.24 |
| Target hit | 2026-02-17 15:20:00 | 170.86 | 168.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — SELL (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 170.70 | 171.51 | 0.00 | ORB-short ORB[170.80,172.50] vol=1.8x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-02-18 09:40:00 | 171.27 | 171.48 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 175.81 | 174.46 | 0.00 | ORB-long ORB[173.02,174.60] vol=3.7x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:55:00 | 176.67 | 175.31 | 0.00 | T1 1.5R @ 176.67 |
| Stop hit — per-position SL triggered | 2026-02-24 11:45:00 | 175.81 | 176.08 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-03-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:30:00 | 155.20 | 156.38 | 0.00 | ORB-short ORB[156.25,157.70] vol=2.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2026-03-11 10:35:00 | 155.76 | 156.25 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-03-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:05:00 | 153.60 | 151.67 | 0.00 | ORB-long ORB[149.81,151.88] vol=1.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 152.95 | 151.74 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:40:00 | 152.89 | 153.46 | 0.00 | ORB-short ORB[152.98,154.65] vol=1.5x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:00:00 | 152.01 | 153.18 | 0.00 | T1 1.5R @ 152.01 |
| Stop hit — per-position SL triggered | 2026-03-13 11:25:00 | 152.89 | 152.48 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2026-03-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:25:00 | 148.24 | 149.93 | 0.00 | ORB-short ORB[148.52,150.20] vol=1.5x ATR=0.74 |
| Stop hit — per-position SL triggered | 2026-03-16 10:30:00 | 148.98 | 149.87 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 147.96 | 147.39 | 0.00 | ORB-long ORB[146.16,147.90] vol=1.7x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 147.35 | 147.65 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 146.97 | 148.10 | 0.00 | ORB-short ORB[147.73,149.47] vol=7.4x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-04-17 11:10:00 | 147.44 | 148.04 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 151.30 | 150.43 | 0.00 | ORB-long ORB[149.18,150.99] vol=1.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-04-22 09:50:00 | 150.85 | 150.47 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 141.83 | 141.14 | 0.00 | ORB-long ORB[140.01,141.70] vol=2.6x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 141.26 | 141.16 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 139.19 | 138.76 | 0.00 | ORB-long ORB[137.63,139.09] vol=2.1x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 138.83 | 138.78 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-19 09:30:00 | 116.23 | 2025-05-19 09:35:00 | 115.92 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-05-21 09:35:00 | 116.45 | 2025-05-21 10:05:00 | 117.10 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-05-21 09:35:00 | 116.45 | 2025-05-21 12:00:00 | 117.00 | TARGET_HIT | 0.50 | 0.47% |
| SELL | retest1 | 2025-05-22 10:30:00 | 116.81 | 2025-05-22 12:50:00 | 116.20 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-05-22 10:30:00 | 116.81 | 2025-05-22 15:20:00 | 116.40 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2025-05-23 09:30:00 | 117.60 | 2025-05-23 09:35:00 | 117.24 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-05-27 09:30:00 | 116.52 | 2025-05-27 10:00:00 | 116.13 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-05-27 09:30:00 | 116.52 | 2025-05-27 10:10:00 | 116.52 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-28 09:35:00 | 118.58 | 2025-05-28 09:50:00 | 118.26 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-30 11:10:00 | 118.32 | 2025-05-30 11:20:00 | 118.55 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-03 10:00:00 | 127.23 | 2025-06-03 10:05:00 | 126.84 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-06-04 09:30:00 | 123.67 | 2025-06-04 10:35:00 | 124.17 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-06 10:05:00 | 126.36 | 2025-06-06 10:10:00 | 125.86 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-06-11 11:10:00 | 125.36 | 2025-06-11 11:35:00 | 125.58 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-06-19 09:50:00 | 117.56 | 2025-06-19 10:15:00 | 116.99 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-06-19 09:50:00 | 117.56 | 2025-06-19 15:20:00 | 115.67 | TARGET_HIT | 0.50 | 1.61% |
| BUY | retest1 | 2025-06-24 09:30:00 | 117.15 | 2025-06-24 09:35:00 | 116.73 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-06-27 10:20:00 | 119.35 | 2025-06-27 10:30:00 | 120.00 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-06-27 10:20:00 | 119.35 | 2025-06-27 10:40:00 | 119.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-03 11:15:00 | 117.89 | 2025-07-03 11:20:00 | 118.12 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-04 10:20:00 | 118.08 | 2025-07-04 11:35:00 | 117.66 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-04 10:20:00 | 118.08 | 2025-07-04 15:10:00 | 118.00 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2025-07-07 10:00:00 | 118.01 | 2025-07-07 10:15:00 | 118.34 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-08 11:05:00 | 116.41 | 2025-07-08 11:15:00 | 116.67 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-10 09:40:00 | 115.40 | 2025-07-10 11:00:00 | 115.08 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-07-10 09:40:00 | 115.40 | 2025-07-10 15:20:00 | 114.35 | TARGET_HIT | 0.50 | 0.91% |
| BUY | retest1 | 2025-07-14 09:40:00 | 115.62 | 2025-07-14 09:45:00 | 115.26 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-07-15 10:00:00 | 116.05 | 2025-07-15 10:20:00 | 115.76 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-16 09:35:00 | 116.70 | 2025-07-16 09:40:00 | 116.44 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-17 09:30:00 | 116.77 | 2025-07-17 09:35:00 | 116.34 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-17 09:30:00 | 116.77 | 2025-07-17 09:50:00 | 116.77 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 11:00:00 | 115.22 | 2025-07-18 11:15:00 | 115.44 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-22 09:30:00 | 114.40 | 2025-07-22 09:40:00 | 114.04 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-22 09:30:00 | 114.40 | 2025-07-22 09:45:00 | 114.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 10:10:00 | 113.00 | 2025-07-24 10:30:00 | 113.21 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-29 10:25:00 | 111.00 | 2025-07-29 10:30:00 | 111.39 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-08-06 10:00:00 | 110.52 | 2025-08-06 10:20:00 | 110.81 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-08-07 09:50:00 | 111.92 | 2025-08-07 09:55:00 | 111.59 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-18 10:05:00 | 114.75 | 2025-08-18 10:40:00 | 115.14 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-08-18 10:05:00 | 114.75 | 2025-08-18 12:35:00 | 114.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-19 09:50:00 | 115.10 | 2025-08-19 09:55:00 | 114.90 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-08-20 10:05:00 | 117.04 | 2025-08-20 11:00:00 | 116.78 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-08-21 10:30:00 | 116.87 | 2025-08-21 10:40:00 | 116.69 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-08-22 09:30:00 | 116.45 | 2025-08-22 10:10:00 | 116.81 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-08-22 09:30:00 | 116.45 | 2025-08-22 10:15:00 | 116.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-26 09:30:00 | 113.27 | 2025-08-26 09:35:00 | 112.92 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-08-26 09:30:00 | 113.27 | 2025-08-26 09:40:00 | 113.27 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-02 09:35:00 | 112.62 | 2025-09-02 09:45:00 | 113.01 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-09-02 09:35:00 | 112.62 | 2025-09-02 13:10:00 | 113.10 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2025-09-05 10:15:00 | 111.79 | 2025-09-05 10:20:00 | 112.07 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-08 09:30:00 | 113.59 | 2025-09-08 09:55:00 | 113.28 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-10 09:40:00 | 114.19 | 2025-09-10 09:45:00 | 113.93 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-12 10:55:00 | 117.75 | 2025-09-12 11:15:00 | 117.42 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-09-12 10:55:00 | 117.75 | 2025-09-12 15:20:00 | 117.31 | TARGET_HIT | 0.50 | 0.37% |
| SELL | retest1 | 2025-09-17 09:50:00 | 117.16 | 2025-09-17 10:05:00 | 117.40 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-18 10:00:00 | 120.66 | 2025-09-18 10:05:00 | 120.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-19 09:50:00 | 120.50 | 2025-09-19 10:00:00 | 120.15 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-24 10:00:00 | 122.90 | 2025-09-24 10:20:00 | 123.51 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-09-24 10:00:00 | 122.90 | 2025-09-24 10:35:00 | 122.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-29 09:35:00 | 118.05 | 2025-09-29 09:55:00 | 118.59 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-09-29 09:35:00 | 118.05 | 2025-09-29 11:35:00 | 118.45 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-03 10:15:00 | 124.15 | 2025-10-03 12:00:00 | 124.54 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-06 10:50:00 | 124.29 | 2025-10-06 11:00:00 | 124.72 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-10-09 10:55:00 | 126.03 | 2025-10-09 12:25:00 | 125.68 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-10 10:00:00 | 126.68 | 2025-10-10 10:10:00 | 127.21 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-10-10 10:00:00 | 126.68 | 2025-10-10 11:05:00 | 126.96 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2025-10-14 11:00:00 | 124.80 | 2025-10-14 11:15:00 | 124.27 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-14 11:00:00 | 124.80 | 2025-10-14 15:20:00 | 124.32 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2025-10-15 11:05:00 | 126.17 | 2025-10-15 12:05:00 | 126.58 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-15 11:05:00 | 126.17 | 2025-10-15 14:05:00 | 126.17 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-17 11:10:00 | 126.01 | 2025-10-17 11:40:00 | 125.57 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-10-20 09:40:00 | 125.61 | 2025-10-20 10:25:00 | 126.50 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-10-20 09:40:00 | 125.61 | 2025-10-20 15:20:00 | 129.84 | TARGET_HIT | 0.50 | 3.37% |
| BUY | retest1 | 2025-10-23 09:30:00 | 132.42 | 2025-10-23 09:50:00 | 133.06 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-10-23 09:30:00 | 132.42 | 2025-10-23 15:20:00 | 135.47 | TARGET_HIT | 0.50 | 2.30% |
| SELL | retest1 | 2025-10-29 11:15:00 | 140.16 | 2025-10-29 11:25:00 | 140.63 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-31 10:10:00 | 141.00 | 2025-10-31 10:40:00 | 140.47 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-11-04 09:30:00 | 142.95 | 2025-11-04 09:40:00 | 142.52 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-06 09:45:00 | 140.53 | 2025-11-06 09:50:00 | 141.02 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-11-07 09:35:00 | 138.87 | 2025-11-07 09:45:00 | 139.32 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-11-11 10:35:00 | 142.49 | 2025-11-11 10:45:00 | 142.98 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-11-12 09:45:00 | 147.74 | 2025-11-12 09:50:00 | 147.32 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-11-17 09:30:00 | 149.81 | 2025-11-17 09:45:00 | 149.33 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-19 11:05:00 | 147.10 | 2025-11-19 11:15:00 | 146.67 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-11-21 09:55:00 | 146.71 | 2025-11-21 10:25:00 | 146.12 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-11-21 09:55:00 | 146.71 | 2025-11-21 12:35:00 | 146.18 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2025-11-26 09:30:00 | 149.74 | 2025-11-26 09:40:00 | 150.41 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-11-26 09:30:00 | 149.74 | 2025-11-26 11:20:00 | 149.74 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-27 09:45:00 | 147.21 | 2025-11-27 10:05:00 | 147.62 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-05 10:10:00 | 142.64 | 2025-12-05 10:25:00 | 142.10 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-12-08 09:50:00 | 140.84 | 2025-12-08 10:30:00 | 140.18 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-12-08 09:50:00 | 140.84 | 2025-12-08 15:20:00 | 138.15 | TARGET_HIT | 0.50 | 1.91% |
| BUY | retest1 | 2025-12-09 11:10:00 | 140.04 | 2025-12-09 12:20:00 | 140.71 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-12-09 11:10:00 | 140.04 | 2025-12-09 15:20:00 | 141.46 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2025-12-16 09:55:00 | 140.45 | 2025-12-16 10:05:00 | 140.77 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-30 10:20:00 | 140.60 | 2025-12-30 10:40:00 | 140.16 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-31 11:15:00 | 144.06 | 2025-12-31 13:30:00 | 143.58 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-01-01 10:05:00 | 145.09 | 2026-01-01 10:10:00 | 144.67 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-01-07 11:15:00 | 151.24 | 2026-01-07 13:00:00 | 150.89 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-01-08 11:05:00 | 147.86 | 2026-01-08 11:10:00 | 147.20 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-01-08 11:05:00 | 147.86 | 2026-01-08 11:35:00 | 147.86 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-09 09:45:00 | 148.59 | 2026-01-09 10:05:00 | 149.55 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-01-09 09:45:00 | 148.59 | 2026-01-09 11:10:00 | 148.59 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-05 09:30:00 | 163.33 | 2026-02-05 09:35:00 | 162.80 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-13 09:30:00 | 161.64 | 2026-02-13 09:40:00 | 162.16 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-16 10:00:00 | 161.58 | 2026-02-16 10:15:00 | 162.49 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-02-16 10:00:00 | 161.58 | 2026-02-16 15:20:00 | 165.76 | TARGET_HIT | 0.50 | 2.59% |
| BUY | retest1 | 2026-02-17 10:20:00 | 167.44 | 2026-02-17 10:30:00 | 168.24 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-17 10:20:00 | 167.44 | 2026-02-17 15:20:00 | 170.86 | TARGET_HIT | 0.50 | 2.04% |
| SELL | retest1 | 2026-02-18 09:30:00 | 170.70 | 2026-02-18 09:40:00 | 171.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-24 09:45:00 | 175.81 | 2026-02-24 09:55:00 | 176.67 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-24 09:45:00 | 175.81 | 2026-02-24 11:45:00 | 175.81 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:30:00 | 155.20 | 2026-03-11 10:35:00 | 155.76 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-12 11:05:00 | 153.60 | 2026-03-12 11:15:00 | 152.95 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-13 09:40:00 | 152.89 | 2026-03-13 10:00:00 | 152.01 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-13 09:40:00 | 152.89 | 2026-03-13 11:25:00 | 152.89 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:25:00 | 148.24 | 2026-03-16 10:30:00 | 148.98 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-10 09:30:00 | 147.96 | 2026-04-10 10:05:00 | 147.35 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-17 11:00:00 | 146.97 | 2026-04-17 11:10:00 | 147.44 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-22 09:45:00 | 151.30 | 2026-04-22 09:50:00 | 150.85 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-04 09:45:00 | 141.83 | 2026-05-04 09:50:00 | 141.26 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-05 11:00:00 | 139.19 | 2026-05-05 11:15:00 | 138.83 | STOP_HIT | 1.00 | -0.26% |
