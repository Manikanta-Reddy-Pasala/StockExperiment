# Zee Entertainment Enterprises Ltd. (ZEEL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 95.22
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
| ENTRY1 | 81 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 20 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 61
- **Target hits / Stop hits / Partials:** 20 / 61 / 32
- **Avg / median % per leg:** 0.30% / 0.00%
- **Sum % (uncompounded):** 34.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 12 | 30.0% | 4 | 28 | 8 | 0.11% | 4.3% |
| BUY @ 2nd Alert (retest1) | 40 | 12 | 30.0% | 4 | 28 | 8 | 0.11% | 4.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 73 | 40 | 54.8% | 16 | 33 | 24 | 0.41% | 30.0% |
| SELL @ 2nd Alert (retest1) | 73 | 40 | 54.8% | 16 | 33 | 24 | 0.41% | 30.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 113 | 52 | 46.0% | 20 | 61 | 32 | 0.30% | 34.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:30:00 | 128.20 | 127.34 | 0.00 | ORB-long ORB[126.00,127.88] vol=2.4x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 10:15:00 | 128.94 | 127.99 | 0.00 | T1 1.5R @ 128.94 |
| Target hit | 2025-05-23 15:05:00 | 128.48 | 128.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2025-05-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 09:35:00 | 126.87 | 127.60 | 0.00 | ORB-short ORB[127.28,128.60] vol=1.7x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-05-26 10:05:00 | 127.26 | 127.33 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:00:00 | 128.00 | 128.08 | 0.00 | ORB-short ORB[128.20,129.25] vol=1.9x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 128.35 | 128.06 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:00:00 | 126.88 | 127.80 | 0.00 | ORB-short ORB[127.60,128.70] vol=2.5x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 10:45:00 | 126.35 | 127.41 | 0.00 | T1 1.5R @ 126.35 |
| Stop hit — per-position SL triggered | 2025-05-30 10:55:00 | 126.88 | 127.38 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:30:00 | 129.62 | 128.97 | 0.00 | ORB-long ORB[128.10,129.16] vol=3.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-06-03 09:40:00 | 129.08 | 129.09 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 11:00:00 | 127.91 | 128.93 | 0.00 | ORB-short ORB[128.37,130.09] vol=1.8x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-06-05 11:10:00 | 128.20 | 128.84 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 09:30:00 | 126.87 | 127.46 | 0.00 | ORB-short ORB[126.90,128.76] vol=2.1x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-06-06 09:50:00 | 127.38 | 127.36 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 09:50:00 | 136.07 | 134.65 | 0.00 | ORB-long ORB[133.50,135.24] vol=5.8x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-06-12 10:00:00 | 135.32 | 134.80 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 11:00:00 | 131.00 | 132.66 | 0.00 | ORB-short ORB[132.52,134.01] vol=2.1x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:25:00 | 130.27 | 132.40 | 0.00 | T1 1.5R @ 130.27 |
| Target hit | 2025-06-19 15:20:00 | 127.51 | 129.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:15:00 | 144.13 | 145.50 | 0.00 | ORB-short ORB[145.25,147.35] vol=1.8x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 12:05:00 | 143.57 | 145.30 | 0.00 | T1 1.5R @ 143.57 |
| Stop hit — per-position SL triggered | 2025-06-26 14:15:00 | 144.13 | 144.90 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:35:00 | 145.00 | 144.14 | 0.00 | ORB-long ORB[143.16,144.98] vol=2.3x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-06-27 09:50:00 | 144.41 | 144.51 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:20:00 | 143.26 | 143.77 | 0.00 | ORB-short ORB[143.60,145.00] vol=1.7x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:45:00 | 142.67 | 143.61 | 0.00 | T1 1.5R @ 142.67 |
| Stop hit — per-position SL triggered | 2025-07-08 14:00:00 | 143.26 | 142.92 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 09:45:00 | 143.22 | 144.46 | 0.00 | ORB-short ORB[144.35,146.43] vol=1.6x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:35:00 | 142.11 | 143.51 | 0.00 | T1 1.5R @ 142.11 |
| Target hit | 2025-07-09 15:20:00 | 141.40 | 142.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2025-07-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:30:00 | 139.87 | 137.67 | 0.00 | ORB-long ORB[136.20,137.44] vol=3.1x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:40:00 | 140.95 | 138.69 | 0.00 | T1 1.5R @ 140.95 |
| Target hit | 2025-07-14 15:20:00 | 142.90 | 142.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 11:15:00 | 142.45 | 142.63 | 0.00 | ORB-short ORB[142.65,144.05] vol=1.6x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 12:20:00 | 141.74 | 142.54 | 0.00 | T1 1.5R @ 141.74 |
| Stop hit — per-position SL triggered | 2025-07-18 14:40:00 | 142.45 | 142.40 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:30:00 | 141.85 | 142.76 | 0.00 | ORB-short ORB[142.20,143.90] vol=2.5x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 10:00:00 | 141.15 | 142.01 | 0.00 | T1 1.5R @ 141.15 |
| Target hit | 2025-07-22 15:20:00 | 133.50 | 137.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:40:00 | 127.91 | 128.77 | 0.00 | ORB-short ORB[128.25,129.83] vol=2.5x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:05:00 | 127.17 | 128.42 | 0.00 | T1 1.5R @ 127.17 |
| Target hit | 2025-07-25 15:20:00 | 123.45 | 125.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-07-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 11:05:00 | 116.93 | 118.50 | 0.00 | ORB-short ORB[118.60,120.17] vol=3.9x ATR=0.42 |
| Stop hit — per-position SL triggered | 2025-07-29 11:20:00 | 117.35 | 118.44 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:35:00 | 117.67 | 118.35 | 0.00 | ORB-short ORB[117.72,119.36] vol=1.6x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 10:40:00 | 116.93 | 117.90 | 0.00 | T1 1.5R @ 116.93 |
| Target hit | 2025-07-30 15:20:00 | 116.46 | 117.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2025-08-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 10:00:00 | 116.20 | 116.86 | 0.00 | ORB-short ORB[116.51,117.69] vol=2.7x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-08-04 10:35:00 | 116.65 | 116.73 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 11:05:00 | 117.45 | 118.25 | 0.00 | ORB-short ORB[118.29,119.63] vol=1.7x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 13:30:00 | 116.99 | 117.96 | 0.00 | T1 1.5R @ 116.99 |
| Target hit | 2025-08-05 15:20:00 | 116.73 | 117.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-08-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:40:00 | 118.48 | 117.72 | 0.00 | ORB-long ORB[117.06,118.22] vol=1.5x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-08-18 10:50:00 | 118.02 | 117.74 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 11:05:00 | 117.51 | 117.12 | 0.00 | ORB-long ORB[116.20,117.27] vol=13.9x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-08-19 11:10:00 | 117.23 | 117.15 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 09:50:00 | 118.76 | 118.09 | 0.00 | ORB-long ORB[117.55,118.63] vol=2.1x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-08-20 10:00:00 | 118.41 | 118.19 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 11:00:00 | 119.31 | 118.56 | 0.00 | ORB-long ORB[117.45,118.65] vol=2.9x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 11:10:00 | 119.87 | 118.64 | 0.00 | T1 1.5R @ 119.87 |
| Stop hit — per-position SL triggered | 2025-08-28 11:50:00 | 119.31 | 118.72 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 09:40:00 | 115.96 | 116.49 | 0.00 | ORB-short ORB[116.20,117.38] vol=2.7x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:45:00 | 115.28 | 116.06 | 0.00 | T1 1.5R @ 115.28 |
| Target hit | 2025-09-01 15:20:00 | 113.99 | 115.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-09-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:40:00 | 116.99 | 116.13 | 0.00 | ORB-long ORB[115.38,116.39] vol=2.4x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-09-03 10:20:00 | 116.63 | 116.46 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:50:00 | 116.70 | 115.64 | 0.00 | ORB-long ORB[114.80,115.89] vol=2.7x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-09-05 09:55:00 | 116.30 | 115.69 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:45:00 | 116.22 | 116.69 | 0.00 | ORB-short ORB[116.55,117.40] vol=1.8x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-09-12 11:05:00 | 116.52 | 116.57 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 11:10:00 | 115.57 | 116.28 | 0.00 | ORB-short ORB[115.60,116.97] vol=2.6x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-09-16 11:20:00 | 115.78 | 116.24 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:50:00 | 116.78 | 116.38 | 0.00 | ORB-long ORB[115.53,116.38] vol=3.2x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-09-19 09:55:00 | 116.52 | 116.39 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:30:00 | 117.20 | 117.56 | 0.00 | ORB-short ORB[117.30,118.38] vol=3.8x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:40:00 | 116.72 | 117.42 | 0.00 | T1 1.5R @ 116.72 |
| Stop hit — per-position SL triggered | 2025-09-24 10:00:00 | 117.20 | 117.38 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 113.20 | 114.23 | 0.00 | ORB-short ORB[114.27,115.37] vol=3.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-09-26 09:35:00 | 113.61 | 114.12 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:35:00 | 113.91 | 113.86 | 0.00 | ORB-long ORB[112.99,113.81] vol=3.1x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-09-29 11:05:00 | 113.60 | 113.84 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 09:30:00 | 113.98 | 113.40 | 0.00 | ORB-long ORB[112.57,113.86] vol=1.6x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-10-01 10:00:00 | 113.63 | 113.65 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:10:00 | 113.57 | 114.04 | 0.00 | ORB-short ORB[114.02,114.67] vol=1.9x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 13:05:00 | 113.23 | 113.87 | 0.00 | T1 1.5R @ 113.23 |
| Target hit | 2025-10-06 15:20:00 | 112.90 | 113.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2025-10-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:10:00 | 112.76 | 113.10 | 0.00 | ORB-short ORB[112.90,113.59] vol=3.2x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 11:25:00 | 112.43 | 113.01 | 0.00 | T1 1.5R @ 112.43 |
| Stop hit — per-position SL triggered | 2025-10-07 14:10:00 | 112.76 | 112.70 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:15:00 | 112.36 | 112.98 | 0.00 | ORB-short ORB[112.61,113.49] vol=1.5x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 11:00:00 | 111.97 | 112.73 | 0.00 | T1 1.5R @ 111.97 |
| Target hit | 2025-10-08 15:20:00 | 109.20 | 110.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2025-10-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:05:00 | 109.85 | 108.93 | 0.00 | ORB-long ORB[108.21,109.68] vol=2.0x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-10-15 11:30:00 | 109.59 | 108.98 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 11:00:00 | 104.16 | 104.31 | 0.00 | ORB-short ORB[104.38,105.88] vol=1.8x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 15:15:00 | 103.75 | 104.15 | 0.00 | T1 1.5R @ 103.75 |
| Target hit | 2025-10-20 15:20:00 | 103.97 | 104.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2025-10-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:55:00 | 105.16 | 105.81 | 0.00 | ORB-short ORB[105.50,106.46] vol=1.5x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 13:25:00 | 104.83 | 105.36 | 0.00 | T1 1.5R @ 104.83 |
| Stop hit — per-position SL triggered | 2025-10-24 14:20:00 | 105.16 | 105.14 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:20:00 | 103.10 | 103.35 | 0.00 | ORB-short ORB[103.14,103.73] vol=2.0x ATR=0.20 |
| Stop hit — per-position SL triggered | 2025-10-28 10:30:00 | 103.30 | 103.35 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 09:45:00 | 102.10 | 102.50 | 0.00 | ORB-short ORB[102.37,103.19] vol=2.1x ATR=0.19 |
| Stop hit — per-position SL triggered | 2025-10-29 09:55:00 | 102.29 | 102.47 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:50:00 | 103.00 | 103.80 | 0.00 | ORB-short ORB[103.67,104.58] vol=1.8x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:20:00 | 102.56 | 103.38 | 0.00 | T1 1.5R @ 102.56 |
| Target hit | 2025-10-30 15:20:00 | 101.92 | 102.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2025-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 09:30:00 | 101.53 | 101.74 | 0.00 | ORB-short ORB[101.60,102.19] vol=2.9x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-10-31 09:55:00 | 101.71 | 101.66 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-11-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:30:00 | 102.24 | 101.72 | 0.00 | ORB-long ORB[100.80,102.01] vol=2.2x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:35:00 | 102.72 | 101.96 | 0.00 | T1 1.5R @ 102.72 |
| Stop hit — per-position SL triggered | 2025-11-04 10:00:00 | 102.24 | 102.12 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:35:00 | 100.94 | 101.26 | 0.00 | ORB-short ORB[101.11,102.00] vol=1.8x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:45:00 | 100.53 | 101.11 | 0.00 | T1 1.5R @ 100.53 |
| Target hit | 2025-11-06 15:20:00 | 99.75 | 100.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2025-11-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:40:00 | 98.50 | 99.04 | 0.00 | ORB-short ORB[98.66,99.85] vol=1.5x ATR=0.24 |
| Stop hit — per-position SL triggered | 2025-11-07 09:50:00 | 98.74 | 98.96 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 10:15:00 | 98.40 | 98.91 | 0.00 | ORB-short ORB[98.77,99.76] vol=1.8x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-11-10 10:25:00 | 98.62 | 98.89 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:15:00 | 100.28 | 99.33 | 0.00 | ORB-long ORB[98.74,99.37] vol=1.8x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 10:25:00 | 100.78 | 99.53 | 0.00 | T1 1.5R @ 100.78 |
| Target hit | 2025-11-12 15:20:00 | 103.54 | 102.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2025-11-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 09:50:00 | 98.76 | 99.00 | 0.00 | ORB-short ORB[98.80,99.68] vol=1.6x ATR=0.15 |
| Stop hit — per-position SL triggered | 2025-11-19 10:05:00 | 98.91 | 98.97 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:45:00 | 98.95 | 98.53 | 0.00 | ORB-long ORB[97.95,98.79] vol=2.2x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-11-27 10:10:00 | 98.70 | 98.66 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:45:00 | 98.90 | 98.07 | 0.00 | ORB-long ORB[97.35,98.32] vol=3.1x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-11-28 09:55:00 | 98.54 | 98.13 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:55:00 | 97.11 | 97.60 | 0.00 | ORB-short ORB[97.70,98.30] vol=4.2x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 97.37 | 97.56 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:00:00 | 96.48 | 97.49 | 0.00 | ORB-short ORB[97.40,98.39] vol=2.1x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:00:00 | 96.05 | 97.01 | 0.00 | T1 1.5R @ 96.05 |
| Target hit | 2025-12-08 15:20:00 | 93.76 | 95.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2025-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:30:00 | 92.20 | 93.17 | 0.00 | ORB-short ORB[92.77,94.08] vol=1.7x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-12-09 10:20:00 | 92.61 | 92.71 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:15:00 | 93.57 | 94.27 | 0.00 | ORB-short ORB[94.05,95.07] vol=4.4x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-12-15 11:20:00 | 93.75 | 94.26 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:10:00 | 93.46 | 93.77 | 0.00 | ORB-short ORB[93.50,94.18] vol=5.5x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:25:00 | 93.22 | 93.71 | 0.00 | T1 1.5R @ 93.22 |
| Stop hit — per-position SL triggered | 2025-12-16 11:30:00 | 93.46 | 93.69 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:30:00 | 91.44 | 92.19 | 0.00 | ORB-short ORB[92.00,93.18] vol=3.1x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:45:00 | 91.00 | 91.73 | 0.00 | T1 1.5R @ 91.00 |
| Target hit | 2025-12-18 10:30:00 | 90.98 | 90.88 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — BUY (started 2025-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:35:00 | 92.26 | 91.60 | 0.00 | ORB-long ORB[90.90,91.86] vol=1.6x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-12-19 09:50:00 | 91.95 | 91.68 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:30:00 | 93.26 | 92.75 | 0.00 | ORB-long ORB[91.70,92.90] vol=4.0x ATR=0.25 |
| Stop hit — per-position SL triggered | 2025-12-24 09:35:00 | 93.01 | 92.76 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:00:00 | 91.00 | 91.40 | 0.00 | ORB-short ORB[91.10,91.88] vol=1.8x ATR=0.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 12:50:00 | 90.72 | 91.27 | 0.00 | T1 1.5R @ 90.72 |
| Target hit | 2025-12-29 15:20:00 | 90.80 | 91.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2025-12-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:55:00 | 90.50 | 90.81 | 0.00 | ORB-short ORB[90.61,91.11] vol=2.0x ATR=0.18 |
| Stop hit — per-position SL triggered | 2025-12-30 11:00:00 | 90.68 | 90.79 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-01-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 09:50:00 | 92.71 | 92.17 | 0.00 | ORB-long ORB[91.74,92.38] vol=1.7x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-01-08 10:05:00 | 92.46 | 92.27 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:30:00 | 89.90 | 90.27 | 0.00 | ORB-short ORB[89.91,91.10] vol=2.2x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-01-09 09:45:00 | 90.19 | 90.19 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-01-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:55:00 | 89.75 | 89.99 | 0.00 | ORB-short ORB[89.83,90.69] vol=5.5x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 12:50:00 | 89.40 | 89.88 | 0.00 | T1 1.5R @ 89.40 |
| Target hit | 2026-01-13 15:20:00 | 89.24 | 89.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2026-01-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:30:00 | 90.35 | 90.07 | 0.00 | ORB-long ORB[89.18,90.24] vol=2.3x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:45:00 | 90.68 | 90.20 | 0.00 | T1 1.5R @ 90.68 |
| Stop hit — per-position SL triggered | 2026-01-14 13:45:00 | 90.35 | 90.29 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 82.36 | 81.77 | 0.00 | ORB-long ORB[81.25,82.14] vol=1.6x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:55:00 | 82.86 | 82.23 | 0.00 | T1 1.5R @ 82.86 |
| Target hit | 2026-01-30 15:20:00 | 84.35 | 83.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — BUY (started 2026-02-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:10:00 | 84.13 | 83.73 | 0.00 | ORB-long ORB[83.00,84.11] vol=2.3x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-02-01 10:15:00 | 83.88 | 83.75 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-02-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-02 11:10:00 | 80.44 | 81.42 | 0.00 | ORB-short ORB[81.11,82.13] vol=3.2x ATR=0.34 |
| Stop hit — per-position SL triggered | 2026-02-02 11:40:00 | 80.78 | 81.33 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 92.39 | 92.95 | 0.00 | ORB-short ORB[92.67,94.00] vol=2.1x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 92.70 | 92.89 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-02-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:00:00 | 93.44 | 92.52 | 0.00 | ORB-long ORB[91.89,93.25] vol=2.1x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 93.09 | 92.62 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-02-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:20:00 | 95.55 | 94.01 | 0.00 | ORB-long ORB[92.71,94.01] vol=2.1x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:25:00 | 96.43 | 94.34 | 0.00 | T1 1.5R @ 96.43 |
| Stop hit — per-position SL triggered | 2026-02-13 10:55:00 | 95.55 | 95.01 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-02-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:30:00 | 96.06 | 95.00 | 0.00 | ORB-long ORB[93.64,94.52] vol=6.2x ATR=0.40 |
| Stop hit — per-position SL triggered | 2026-02-17 11:25:00 | 95.66 | 95.18 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 94.32 | 94.86 | 0.00 | ORB-short ORB[94.35,95.54] vol=1.5x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-02-18 10:20:00 | 94.62 | 94.78 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:35:00 | 92.09 | 91.48 | 0.00 | ORB-long ORB[90.71,91.80] vol=1.6x ATR=0.39 |
| Stop hit — per-position SL triggered | 2026-02-20 10:10:00 | 91.70 | 91.65 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 89.47 | 90.09 | 0.00 | ORB-short ORB[89.76,91.05] vol=2.1x ATR=0.26 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 89.73 | 90.03 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-04-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:05:00 | 83.36 | 82.21 | 0.00 | ORB-long ORB[81.30,82.35] vol=4.5x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-04-15 10:25:00 | 82.92 | 82.38 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 89.62 | 88.81 | 0.00 | ORB-long ORB[88.10,89.20] vol=2.3x ATR=0.40 |
| Stop hit — per-position SL triggered | 2026-04-27 09:35:00 | 89.22 | 88.86 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 92.01 | 91.65 | 0.00 | ORB-long ORB[90.64,92.00] vol=1.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-04-28 09:50:00 | 91.56 | 91.67 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 90.87 | 91.26 | 0.00 | ORB-short ORB[91.03,92.23] vol=4.1x ATR=0.33 |
| Stop hit — per-position SL triggered | 2026-05-06 09:50:00 | 91.20 | 91.24 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-23 09:30:00 | 128.20 | 2025-05-23 10:15:00 | 128.94 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-05-23 09:30:00 | 128.20 | 2025-05-23 15:05:00 | 128.48 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2025-05-26 09:35:00 | 126.87 | 2025-05-26 10:05:00 | 127.26 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-05-29 11:00:00 | 128.00 | 2025-05-29 15:15:00 | 128.35 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-30 10:00:00 | 126.88 | 2025-05-30 10:45:00 | 126.35 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-05-30 10:00:00 | 126.88 | 2025-05-30 10:55:00 | 126.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-03 09:30:00 | 129.62 | 2025-06-03 09:40:00 | 129.08 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-06-05 11:00:00 | 127.91 | 2025-06-05 11:10:00 | 128.20 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-06-06 09:30:00 | 126.87 | 2025-06-06 09:50:00 | 127.38 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-12 09:50:00 | 136.07 | 2025-06-12 10:00:00 | 135.32 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2025-06-19 11:00:00 | 131.00 | 2025-06-19 11:25:00 | 130.27 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-06-19 11:00:00 | 131.00 | 2025-06-19 15:20:00 | 127.51 | TARGET_HIT | 0.50 | 2.66% |
| SELL | retest1 | 2025-06-26 11:15:00 | 144.13 | 2025-06-26 12:05:00 | 143.57 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-06-26 11:15:00 | 144.13 | 2025-06-26 14:15:00 | 144.13 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 09:35:00 | 145.00 | 2025-06-27 09:50:00 | 144.41 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-07-08 10:20:00 | 143.26 | 2025-07-08 10:45:00 | 142.67 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-07-08 10:20:00 | 143.26 | 2025-07-08 14:00:00 | 143.26 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-09 09:45:00 | 143.22 | 2025-07-09 10:35:00 | 142.11 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2025-07-09 09:45:00 | 143.22 | 2025-07-09 15:20:00 | 141.40 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2025-07-14 09:30:00 | 139.87 | 2025-07-14 09:40:00 | 140.95 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2025-07-14 09:30:00 | 139.87 | 2025-07-14 15:20:00 | 142.90 | TARGET_HIT | 0.50 | 2.17% |
| SELL | retest1 | 2025-07-18 11:15:00 | 142.45 | 2025-07-18 12:20:00 | 141.74 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-07-18 11:15:00 | 142.45 | 2025-07-18 14:40:00 | 142.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 09:30:00 | 141.85 | 2025-07-22 10:00:00 | 141.15 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-07-22 09:30:00 | 141.85 | 2025-07-22 15:20:00 | 133.50 | TARGET_HIT | 0.50 | 5.89% |
| SELL | retest1 | 2025-07-25 09:40:00 | 127.91 | 2025-07-25 10:05:00 | 127.17 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-07-25 09:40:00 | 127.91 | 2025-07-25 15:20:00 | 123.45 | TARGET_HIT | 0.50 | 3.49% |
| SELL | retest1 | 2025-07-29 11:05:00 | 116.93 | 2025-07-29 11:20:00 | 117.35 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-07-30 09:35:00 | 117.67 | 2025-07-30 10:40:00 | 116.93 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-07-30 09:35:00 | 117.67 | 2025-07-30 15:20:00 | 116.46 | TARGET_HIT | 0.50 | 1.03% |
| SELL | retest1 | 2025-08-04 10:00:00 | 116.20 | 2025-08-04 10:35:00 | 116.65 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-08-05 11:05:00 | 117.45 | 2025-08-05 13:30:00 | 116.99 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-08-05 11:05:00 | 117.45 | 2025-08-05 15:20:00 | 116.73 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2025-08-18 10:40:00 | 118.48 | 2025-08-18 10:50:00 | 118.02 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-08-19 11:05:00 | 117.51 | 2025-08-19 11:10:00 | 117.23 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-20 09:50:00 | 118.76 | 2025-08-20 10:00:00 | 118.41 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-28 11:00:00 | 119.31 | 2025-08-28 11:10:00 | 119.87 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-08-28 11:00:00 | 119.31 | 2025-08-28 11:50:00 | 119.31 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-01 09:40:00 | 115.96 | 2025-09-01 11:45:00 | 115.28 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-09-01 09:40:00 | 115.96 | 2025-09-01 15:20:00 | 113.99 | TARGET_HIT | 0.50 | 1.70% |
| BUY | retest1 | 2025-09-03 09:40:00 | 116.99 | 2025-09-03 10:20:00 | 116.63 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-05 09:50:00 | 116.70 | 2025-09-05 09:55:00 | 116.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-09-12 10:45:00 | 116.22 | 2025-09-12 11:05:00 | 116.52 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-16 11:10:00 | 115.57 | 2025-09-16 11:20:00 | 115.78 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-19 09:50:00 | 116.78 | 2025-09-19 09:55:00 | 116.52 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-24 09:30:00 | 117.20 | 2025-09-24 09:40:00 | 116.72 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-09-24 09:30:00 | 117.20 | 2025-09-24 10:00:00 | 117.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-26 09:30:00 | 113.20 | 2025-09-26 09:35:00 | 113.61 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-29 10:35:00 | 113.91 | 2025-09-29 11:05:00 | 113.60 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-01 09:30:00 | 113.98 | 2025-10-01 10:00:00 | 113.63 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-10-06 11:10:00 | 113.57 | 2025-10-06 13:05:00 | 113.23 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-10-06 11:10:00 | 113.57 | 2025-10-06 15:20:00 | 112.90 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2025-10-07 11:10:00 | 112.76 | 2025-10-07 11:25:00 | 112.43 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-10-07 11:10:00 | 112.76 | 2025-10-07 14:10:00 | 112.76 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-08 10:15:00 | 112.36 | 2025-10-08 11:00:00 | 111.97 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-10-08 10:15:00 | 112.36 | 2025-10-08 15:20:00 | 109.20 | TARGET_HIT | 0.50 | 2.81% |
| BUY | retest1 | 2025-10-15 11:05:00 | 109.85 | 2025-10-15 11:30:00 | 109.59 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-20 11:00:00 | 104.16 | 2025-10-20 15:15:00 | 103.75 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-20 11:00:00 | 104.16 | 2025-10-20 15:20:00 | 103.97 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2025-10-24 10:55:00 | 105.16 | 2025-10-24 13:25:00 | 104.83 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-10-24 10:55:00 | 105.16 | 2025-10-24 14:20:00 | 105.16 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-28 10:20:00 | 103.10 | 2025-10-28 10:30:00 | 103.30 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-10-29 09:45:00 | 102.10 | 2025-10-29 09:55:00 | 102.29 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-30 09:50:00 | 103.00 | 2025-10-30 10:20:00 | 102.56 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-10-30 09:50:00 | 103.00 | 2025-10-30 15:20:00 | 101.92 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2025-10-31 09:30:00 | 101.53 | 2025-10-31 09:55:00 | 101.71 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-11-04 09:30:00 | 102.24 | 2025-11-04 09:35:00 | 102.72 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-11-04 09:30:00 | 102.24 | 2025-11-04 10:00:00 | 102.24 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-06 09:35:00 | 100.94 | 2025-11-06 09:45:00 | 100.53 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-11-06 09:35:00 | 100.94 | 2025-11-06 15:20:00 | 99.75 | TARGET_HIT | 0.50 | 1.18% |
| SELL | retest1 | 2025-11-07 09:40:00 | 98.50 | 2025-11-07 09:50:00 | 98.74 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-10 10:15:00 | 98.40 | 2025-11-10 10:25:00 | 98.62 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-12 10:15:00 | 100.28 | 2025-11-12 10:25:00 | 100.78 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-11-12 10:15:00 | 100.28 | 2025-11-12 15:20:00 | 103.54 | TARGET_HIT | 0.50 | 3.25% |
| SELL | retest1 | 2025-11-19 09:50:00 | 98.76 | 2025-11-19 10:05:00 | 98.91 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-11-27 09:45:00 | 98.95 | 2025-11-27 10:10:00 | 98.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-28 09:45:00 | 98.90 | 2025-11-28 09:55:00 | 98.54 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-12-05 09:55:00 | 97.11 | 2025-12-05 10:00:00 | 97.37 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-08 10:00:00 | 96.48 | 2025-12-08 11:00:00 | 96.05 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-12-08 10:00:00 | 96.48 | 2025-12-08 15:20:00 | 93.76 | TARGET_HIT | 0.50 | 2.82% |
| SELL | retest1 | 2025-12-09 09:30:00 | 92.20 | 2025-12-09 10:20:00 | 92.61 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-12-15 11:15:00 | 93.57 | 2025-12-15 11:20:00 | 93.75 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-16 11:10:00 | 93.46 | 2025-12-16 11:25:00 | 93.22 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-12-16 11:10:00 | 93.46 | 2025-12-16 11:30:00 | 93.46 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-18 09:30:00 | 91.44 | 2025-12-18 09:45:00 | 91.00 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-12-18 09:30:00 | 91.44 | 2025-12-18 10:30:00 | 90.98 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2025-12-19 09:35:00 | 92.26 | 2025-12-19 09:50:00 | 91.95 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-12-24 09:30:00 | 93.26 | 2025-12-24 09:35:00 | 93.01 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-29 11:00:00 | 91.00 | 2025-12-29 12:50:00 | 90.72 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-29 11:00:00 | 91.00 | 2025-12-29 15:20:00 | 90.80 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2025-12-30 10:55:00 | 90.50 | 2025-12-30 11:00:00 | 90.68 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-01-08 09:50:00 | 92.71 | 2026-01-08 10:05:00 | 92.46 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-09 09:30:00 | 89.90 | 2026-01-09 09:45:00 | 90.19 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-13 10:55:00 | 89.75 | 2026-01-13 12:50:00 | 89.40 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-01-13 10:55:00 | 89.75 | 2026-01-13 15:20:00 | 89.24 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2026-01-14 10:30:00 | 90.35 | 2026-01-14 11:45:00 | 90.68 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-01-14 10:30:00 | 90.35 | 2026-01-14 13:45:00 | 90.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-30 09:30:00 | 82.36 | 2026-01-30 09:55:00 | 82.86 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-01-30 09:30:00 | 82.36 | 2026-01-30 15:20:00 | 84.35 | TARGET_HIT | 0.50 | 2.42% |
| BUY | retest1 | 2026-02-01 10:10:00 | 84.13 | 2026-02-01 10:15:00 | 83.88 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-02 11:10:00 | 80.44 | 2026-02-02 11:40:00 | 80.78 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-11 09:35:00 | 92.39 | 2026-02-11 09:45:00 | 92.70 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-12 10:00:00 | 93.44 | 2026-02-12 10:15:00 | 93.09 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-13 10:20:00 | 95.55 | 2026-02-13 10:25:00 | 96.43 | PARTIAL | 0.50 | 0.92% |
| BUY | retest1 | 2026-02-13 10:20:00 | 95.55 | 2026-02-13 10:55:00 | 95.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:30:00 | 96.06 | 2026-02-17 11:25:00 | 95.66 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-18 09:50:00 | 94.32 | 2026-02-18 10:20:00 | 94.62 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-20 09:35:00 | 92.09 | 2026-02-20 10:10:00 | 91.70 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-24 09:30:00 | 89.47 | 2026-02-24 09:35:00 | 89.73 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-15 10:05:00 | 83.36 | 2026-04-15 10:25:00 | 82.92 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-04-27 09:30:00 | 89.62 | 2026-04-27 09:35:00 | 89.22 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-28 09:40:00 | 92.01 | 2026-04-28 09:50:00 | 91.56 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-05-06 09:40:00 | 90.87 | 2026-05-06 09:50:00 | 91.20 | STOP_HIT | 1.00 | -0.36% |
