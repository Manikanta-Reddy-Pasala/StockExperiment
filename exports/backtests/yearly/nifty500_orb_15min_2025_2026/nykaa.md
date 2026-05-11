# FSN E-Commerce Ventures Ltd. (NYKAA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (15238 bars)
- **Last close:** 273.00
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
| ENTRY1 | 63 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 12 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 88 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 51
- **Target hits / Stop hits / Partials:** 12 / 51 / 25
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 12.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 18 | 43.9% | 7 | 23 | 11 | 0.15% | 6.1% |
| BUY @ 2nd Alert (retest1) | 41 | 18 | 43.9% | 7 | 23 | 11 | 0.15% | 6.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 19 | 40.4% | 5 | 28 | 14 | 0.13% | 6.1% |
| SELL @ 2nd Alert (retest1) | 47 | 19 | 40.4% | 5 | 28 | 14 | 0.13% | 6.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 88 | 37 | 42.0% | 12 | 51 | 25 | 0.14% | 12.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:30:00 | 200.05 | 198.84 | 0.00 | ORB-long ORB[197.57,199.40] vol=1.9x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 09:40:00 | 200.96 | 199.43 | 0.00 | T1 1.5R @ 200.96 |
| Stop hit — per-position SL triggered | 2025-05-13 09:45:00 | 200.05 | 199.49 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:55:00 | 201.02 | 200.33 | 0.00 | ORB-long ORB[197.00,199.84] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-05-16 10:00:00 | 200.20 | 200.34 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 10:20:00 | 204.36 | 202.77 | 0.00 | ORB-long ORB[200.67,203.00] vol=3.1x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-05-19 10:35:00 | 203.65 | 204.06 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:40:00 | 205.30 | 203.74 | 0.00 | ORB-long ORB[201.62,203.50] vol=1.7x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-05-28 09:50:00 | 204.60 | 204.13 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 10:40:00 | 195.20 | 196.55 | 0.00 | ORB-short ORB[195.30,197.95] vol=1.7x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-06-05 11:10:00 | 195.74 | 196.19 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:35:00 | 199.60 | 198.27 | 0.00 | ORB-long ORB[197.15,198.75] vol=1.5x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 10:45:00 | 200.43 | 198.94 | 0.00 | T1 1.5R @ 200.43 |
| Target hit | 2025-06-09 15:20:00 | 202.11 | 201.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 10:15:00 | 217.67 | 218.81 | 0.00 | ORB-short ORB[218.70,220.90] vol=2.3x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 11:25:00 | 216.77 | 218.27 | 0.00 | T1 1.5R @ 216.77 |
| Stop hit — per-position SL triggered | 2025-07-15 12:30:00 | 217.67 | 217.91 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-07-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:50:00 | 216.60 | 215.57 | 0.00 | ORB-long ORB[214.31,216.05] vol=1.9x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-07-16 10:05:00 | 215.94 | 215.68 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-07-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 09:35:00 | 218.00 | 217.18 | 0.00 | ORB-long ORB[215.15,217.22] vol=2.7x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-07-17 09:45:00 | 217.39 | 217.38 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 11:00:00 | 216.00 | 216.80 | 0.00 | ORB-short ORB[216.79,218.47] vol=2.0x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 11:35:00 | 215.04 | 216.31 | 0.00 | T1 1.5R @ 215.04 |
| Target hit | 2025-07-18 15:20:00 | 212.75 | 214.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2025-07-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 09:50:00 | 217.97 | 217.19 | 0.00 | ORB-long ORB[214.36,217.38] vol=1.8x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 10:10:00 | 219.01 | 217.75 | 0.00 | T1 1.5R @ 219.01 |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 217.97 | 217.78 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:00:00 | 216.37 | 217.29 | 0.00 | ORB-short ORB[217.50,218.70] vol=2.1x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-07-24 11:30:00 | 216.85 | 216.91 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:50:00 | 207.39 | 209.06 | 0.00 | ORB-short ORB[207.77,210.17] vol=3.7x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-07-29 11:00:00 | 208.15 | 208.74 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-08-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 11:10:00 | 210.09 | 210.59 | 0.00 | ORB-short ORB[210.11,212.34] vol=1.7x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:45:00 | 209.30 | 210.50 | 0.00 | T1 1.5R @ 209.30 |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 210.09 | 210.25 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-08-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:35:00 | 210.56 | 211.29 | 0.00 | ORB-short ORB[211.25,212.54] vol=1.5x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:00:00 | 209.73 | 210.83 | 0.00 | T1 1.5R @ 209.73 |
| Target hit | 2025-08-06 12:20:00 | 209.24 | 209.12 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — SELL (started 2025-08-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:30:00 | 203.54 | 204.49 | 0.00 | ORB-short ORB[203.68,205.50] vol=2.4x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 11:10:00 | 202.66 | 204.30 | 0.00 | T1 1.5R @ 202.66 |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 203.54 | 204.27 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:30:00 | 227.50 | 226.28 | 0.00 | ORB-long ORB[224.94,226.67] vol=2.2x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 09:40:00 | 228.49 | 227.04 | 0.00 | T1 1.5R @ 228.49 |
| Target hit | 2025-08-25 12:55:00 | 230.44 | 230.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2025-08-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 09:50:00 | 230.99 | 229.81 | 0.00 | ORB-long ORB[227.50,230.23] vol=1.9x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-08-26 10:35:00 | 230.05 | 230.20 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:45:00 | 233.94 | 233.22 | 0.00 | ORB-long ORB[231.35,233.89] vol=1.6x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:55:00 | 235.21 | 233.59 | 0.00 | T1 1.5R @ 235.21 |
| Target hit | 2025-09-03 15:20:00 | 237.29 | 236.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-09-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 10:45:00 | 243.00 | 242.36 | 0.00 | ORB-long ORB[240.49,242.12] vol=1.6x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 11:25:00 | 244.02 | 242.66 | 0.00 | T1 1.5R @ 244.02 |
| Stop hit — per-position SL triggered | 2025-09-05 11:45:00 | 243.00 | 242.96 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-10-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:00:00 | 256.25 | 257.12 | 0.00 | ORB-short ORB[257.37,260.30] vol=1.6x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-10-08 11:05:00 | 257.26 | 257.12 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 10:15:00 | 260.00 | 259.10 | 0.00 | ORB-long ORB[256.80,259.44] vol=2.8x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 10:25:00 | 261.56 | 259.21 | 0.00 | T1 1.5R @ 261.56 |
| Target hit | 2025-10-09 15:20:00 | 264.36 | 262.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2025-10-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:45:00 | 267.79 | 265.40 | 0.00 | ORB-long ORB[262.70,266.18] vol=3.2x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-10-10 10:50:00 | 266.82 | 265.49 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:40:00 | 260.26 | 260.91 | 0.00 | ORB-short ORB[260.50,263.23] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-10-14 09:45:00 | 261.06 | 260.89 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-10-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:50:00 | 264.98 | 262.86 | 0.00 | ORB-long ORB[260.71,263.79] vol=1.6x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:55:00 | 266.31 | 263.95 | 0.00 | T1 1.5R @ 266.31 |
| Stop hit — per-position SL triggered | 2025-10-17 10:00:00 | 264.98 | 264.12 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-10-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:45:00 | 254.27 | 256.12 | 0.00 | ORB-short ORB[256.00,257.80] vol=1.8x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 11:00:00 | 253.35 | 255.36 | 0.00 | T1 1.5R @ 253.35 |
| Stop hit — per-position SL triggered | 2025-10-31 11:25:00 | 254.27 | 255.17 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-11-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 10:25:00 | 245.65 | 246.91 | 0.00 | ORB-short ORB[246.66,249.90] vol=3.3x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-11-03 10:35:00 | 246.50 | 246.89 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-11-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:40:00 | 251.20 | 249.75 | 0.00 | ORB-long ORB[247.11,250.23] vol=2.0x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-11-04 09:45:00 | 250.43 | 249.86 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-11-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 09:35:00 | 255.15 | 255.56 | 0.00 | ORB-short ORB[255.20,258.20] vol=1.8x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-11-13 09:45:00 | 255.92 | 256.04 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-11-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:55:00 | 257.21 | 256.60 | 0.00 | ORB-long ORB[255.03,256.95] vol=5.1x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-11-14 10:20:00 | 256.26 | 256.67 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-11-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:40:00 | 263.52 | 260.83 | 0.00 | ORB-long ORB[257.69,260.80] vol=1.8x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-11-17 09:50:00 | 262.46 | 261.47 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-11-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:50:00 | 267.29 | 267.88 | 0.00 | ORB-short ORB[267.98,270.40] vol=3.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-11-19 12:05:00 | 267.94 | 267.82 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:30:00 | 271.40 | 269.85 | 0.00 | ORB-long ORB[266.77,270.49] vol=2.3x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-11-21 10:00:00 | 270.33 | 270.91 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-11-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:40:00 | 264.94 | 265.42 | 0.00 | ORB-short ORB[265.00,267.89] vol=3.9x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-11-27 10:40:00 | 265.80 | 265.35 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:30:00 | 266.66 | 265.97 | 0.00 | ORB-long ORB[264.37,265.62] vol=14.7x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 09:35:00 | 267.67 | 266.00 | 0.00 | T1 1.5R @ 267.67 |
| Target hit | 2025-11-28 13:05:00 | 267.22 | 267.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — SELL (started 2025-12-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:50:00 | 262.35 | 263.14 | 0.00 | ORB-short ORB[262.40,265.35] vol=1.6x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-12-02 11:10:00 | 263.15 | 262.62 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 258.85 | 259.79 | 0.00 | ORB-short ORB[259.20,262.15] vol=1.7x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-12-03 09:50:00 | 259.81 | 259.39 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-12-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:45:00 | 253.00 | 254.41 | 0.00 | ORB-short ORB[253.45,256.70] vol=1.7x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 10:55:00 | 251.96 | 254.15 | 0.00 | T1 1.5R @ 251.96 |
| Target hit | 2025-12-10 15:20:00 | 246.65 | 249.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2025-12-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 10:45:00 | 247.90 | 249.08 | 0.00 | ORB-short ORB[248.65,252.25] vol=9.6x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-12-15 11:25:00 | 248.73 | 248.39 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-12-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 11:05:00 | 244.55 | 245.79 | 0.00 | ORB-short ORB[245.90,248.25] vol=8.6x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-12-17 11:45:00 | 245.10 | 245.35 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-12-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 10:40:00 | 259.30 | 258.56 | 0.00 | ORB-long ORB[257.15,259.00] vol=2.8x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-12-29 11:30:00 | 258.78 | 258.65 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 09:40:00 | 260.40 | 258.50 | 0.00 | ORB-long ORB[256.00,258.80] vol=2.8x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-12-30 09:50:00 | 259.58 | 258.92 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2026-01-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 10:25:00 | 264.80 | 266.56 | 0.00 | ORB-short ORB[267.30,270.00] vol=1.7x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-01-07 10:30:00 | 265.65 | 266.46 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2026-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:55:00 | 263.00 | 264.78 | 0.00 | ORB-short ORB[265.15,267.70] vol=1.9x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 262.07 | 264.48 | 0.00 | T1 1.5R @ 262.07 |
| Target hit | 2026-01-08 12:40:00 | 262.00 | 261.82 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — SELL (started 2026-01-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:40:00 | 249.60 | 251.65 | 0.00 | ORB-short ORB[251.70,255.00] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-01-13 09:45:00 | 250.46 | 251.30 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-01-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:30:00 | 253.30 | 254.72 | 0.00 | ORB-short ORB[253.40,256.30] vol=2.8x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 09:45:00 | 251.68 | 253.91 | 0.00 | T1 1.5R @ 251.68 |
| Stop hit — per-position SL triggered | 2026-01-14 10:30:00 | 253.30 | 253.43 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2026-01-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:05:00 | 241.70 | 244.75 | 0.00 | ORB-short ORB[242.50,245.50] vol=2.1x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:30:00 | 240.38 | 244.02 | 0.00 | T1 1.5R @ 240.38 |
| Target hit | 2026-01-22 15:20:00 | 240.10 | 241.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 11:15:00 | 237.80 | 236.13 | 0.00 | ORB-long ORB[233.05,236.30] vol=2.0x ATR=0.74 |
| Stop hit — per-position SL triggered | 2026-01-30 11:55:00 | 237.06 | 236.46 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 277.73 | 276.44 | 0.00 | ORB-long ORB[275.01,277.45] vol=3.8x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:25:00 | 278.81 | 276.80 | 0.00 | T1 1.5R @ 278.81 |
| Target hit | 2026-02-12 15:20:00 | 279.80 | 278.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2026-02-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:25:00 | 275.51 | 277.13 | 0.00 | ORB-short ORB[277.55,280.25] vol=6.0x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:05:00 | 274.21 | 276.78 | 0.00 | T1 1.5R @ 274.21 |
| Stop hit — per-position SL triggered | 2026-02-13 11:50:00 | 275.51 | 276.22 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-02-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:40:00 | 269.82 | 272.59 | 0.00 | ORB-short ORB[272.66,276.38] vol=1.6x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:50:00 | 268.58 | 271.68 | 0.00 | T1 1.5R @ 268.58 |
| Stop hit — per-position SL triggered | 2026-02-17 13:05:00 | 269.82 | 270.59 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-02-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:55:00 | 268.25 | 271.14 | 0.00 | ORB-short ORB[271.20,274.40] vol=1.8x ATR=0.74 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 268.99 | 270.95 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2026-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 11:00:00 | 264.62 | 267.37 | 0.00 | ORB-short ORB[266.20,269.38] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-02-20 11:30:00 | 265.49 | 266.27 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 261.40 | 263.05 | 0.00 | ORB-short ORB[262.51,266.44] vol=2.1x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 262.25 | 262.45 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 265.40 | 265.17 | 0.00 | ORB-long ORB[262.66,265.39] vol=2.2x ATR=0.96 |
| Stop hit — per-position SL triggered | 2026-02-25 09:40:00 | 264.44 | 265.11 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 269.67 | 267.45 | 0.00 | ORB-long ORB[264.77,268.50] vol=2.2x ATR=0.77 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 268.90 | 267.96 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-03-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:10:00 | 256.10 | 257.60 | 0.00 | ORB-short ORB[257.25,259.45] vol=1.5x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:45:00 | 254.66 | 256.79 | 0.00 | T1 1.5R @ 254.66 |
| Stop hit — per-position SL triggered | 2026-03-05 12:25:00 | 256.10 | 256.20 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:15:00 | 254.40 | 252.41 | 0.00 | ORB-long ORB[251.25,253.15] vol=2.6x ATR=1.04 |
| Stop hit — per-position SL triggered | 2026-03-10 10:20:00 | 253.36 | 252.46 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-03-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:45:00 | 240.60 | 238.73 | 0.00 | ORB-long ORB[235.65,238.60] vol=1.5x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-03-16 09:55:00 | 239.39 | 238.82 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 240.20 | 241.48 | 0.00 | ORB-short ORB[241.00,243.65] vol=2.7x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:15:00 | 239.03 | 241.03 | 0.00 | T1 1.5R @ 239.03 |
| Stop hit — per-position SL triggered | 2026-03-27 13:15:00 | 240.20 | 239.31 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-04-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:05:00 | 264.22 | 265.14 | 0.00 | ORB-short ORB[265.00,267.20] vol=2.9x ATR=0.76 |
| Stop hit — per-position SL triggered | 2026-04-21 11:35:00 | 264.98 | 264.52 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-04-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:55:00 | 266.90 | 265.13 | 0.00 | ORB-long ORB[263.19,266.21] vol=3.1x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 12:00:00 | 268.11 | 265.98 | 0.00 | T1 1.5R @ 268.11 |
| Target hit | 2026-04-27 15:20:00 | 269.38 | 268.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2026-05-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:10:00 | 268.80 | 266.86 | 0.00 | ORB-long ORB[264.40,267.80] vol=2.1x ATR=0.71 |
| Stop hit — per-position SL triggered | 2026-05-04 11:35:00 | 268.09 | 267.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 09:30:00 | 200.05 | 2025-05-13 09:40:00 | 200.96 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-05-13 09:30:00 | 200.05 | 2025-05-13 09:45:00 | 200.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-16 09:55:00 | 201.02 | 2025-05-16 10:00:00 | 200.20 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-05-19 10:20:00 | 204.36 | 2025-05-19 10:35:00 | 203.65 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-05-28 09:40:00 | 205.30 | 2025-05-28 09:50:00 | 204.60 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-06-05 10:40:00 | 195.20 | 2025-06-05 11:10:00 | 195.74 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-09 10:35:00 | 199.60 | 2025-06-09 10:45:00 | 200.43 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-06-09 10:35:00 | 199.60 | 2025-06-09 15:20:00 | 202.11 | TARGET_HIT | 0.50 | 1.26% |
| SELL | retest1 | 2025-07-15 10:15:00 | 217.67 | 2025-07-15 11:25:00 | 216.77 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-07-15 10:15:00 | 217.67 | 2025-07-15 12:30:00 | 217.67 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-16 09:50:00 | 216.60 | 2025-07-16 10:05:00 | 215.94 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-17 09:35:00 | 218.00 | 2025-07-17 09:45:00 | 217.39 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-18 11:00:00 | 216.00 | 2025-07-18 11:35:00 | 215.04 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-07-18 11:00:00 | 216.00 | 2025-07-18 15:20:00 | 212.75 | TARGET_HIT | 0.50 | 1.50% |
| BUY | retest1 | 2025-07-22 09:50:00 | 217.97 | 2025-07-22 10:10:00 | 219.01 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-22 09:50:00 | 217.97 | 2025-07-22 10:15:00 | 217.97 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 11:00:00 | 216.37 | 2025-07-24 11:30:00 | 216.85 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-29 10:50:00 | 207.39 | 2025-07-29 11:00:00 | 208.15 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-08-01 11:10:00 | 210.09 | 2025-08-01 11:45:00 | 209.30 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-08-01 11:10:00 | 210.09 | 2025-08-01 13:15:00 | 210.09 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-06 09:35:00 | 210.56 | 2025-08-06 10:00:00 | 209.73 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-08-06 09:35:00 | 210.56 | 2025-08-06 12:20:00 | 209.24 | TARGET_HIT | 0.50 | 0.63% |
| SELL | retest1 | 2025-08-12 10:30:00 | 203.54 | 2025-08-12 11:10:00 | 202.66 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-08-12 10:30:00 | 203.54 | 2025-08-12 11:15:00 | 203.54 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-25 09:30:00 | 227.50 | 2025-08-25 09:40:00 | 228.49 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-08-25 09:30:00 | 227.50 | 2025-08-25 12:55:00 | 230.44 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2025-08-26 09:50:00 | 230.99 | 2025-08-26 10:35:00 | 230.05 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-09-03 09:45:00 | 233.94 | 2025-09-03 09:55:00 | 235.21 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-09-03 09:45:00 | 233.94 | 2025-09-03 15:20:00 | 237.29 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2025-09-05 10:45:00 | 243.00 | 2025-09-05 11:25:00 | 244.02 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-09-05 10:45:00 | 243.00 | 2025-09-05 11:45:00 | 243.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-08 11:00:00 | 256.25 | 2025-10-08 11:05:00 | 257.26 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-10-09 10:15:00 | 260.00 | 2025-10-09 10:25:00 | 261.56 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-10-09 10:15:00 | 260.00 | 2025-10-09 15:20:00 | 264.36 | TARGET_HIT | 0.50 | 1.68% |
| BUY | retest1 | 2025-10-10 10:45:00 | 267.79 | 2025-10-10 10:50:00 | 266.82 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-14 09:40:00 | 260.26 | 2025-10-14 09:45:00 | 261.06 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-17 09:50:00 | 264.98 | 2025-10-17 09:55:00 | 266.31 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-10-17 09:50:00 | 264.98 | 2025-10-17 10:00:00 | 264.98 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-31 10:45:00 | 254.27 | 2025-10-31 11:00:00 | 253.35 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-10-31 10:45:00 | 254.27 | 2025-10-31 11:25:00 | 254.27 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-03 10:25:00 | 245.65 | 2025-11-03 10:35:00 | 246.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-11-04 09:40:00 | 251.20 | 2025-11-04 09:45:00 | 250.43 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-13 09:35:00 | 255.15 | 2025-11-13 09:45:00 | 255.92 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-14 09:55:00 | 257.21 | 2025-11-14 10:20:00 | 256.26 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-11-17 09:40:00 | 263.52 | 2025-11-17 09:50:00 | 262.46 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-11-19 10:50:00 | 267.29 | 2025-11-19 12:05:00 | 267.94 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-21 09:30:00 | 271.40 | 2025-11-21 10:00:00 | 270.33 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-11-27 09:40:00 | 264.94 | 2025-11-27 10:40:00 | 265.80 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-28 09:30:00 | 266.66 | 2025-11-28 09:35:00 | 267.67 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-11-28 09:30:00 | 266.66 | 2025-11-28 13:05:00 | 267.22 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2025-12-02 09:50:00 | 262.35 | 2025-12-02 11:10:00 | 263.15 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-03 09:30:00 | 258.85 | 2025-12-03 09:50:00 | 259.81 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-12-10 10:45:00 | 253.00 | 2025-12-10 10:55:00 | 251.96 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-12-10 10:45:00 | 253.00 | 2025-12-10 15:20:00 | 246.65 | TARGET_HIT | 0.50 | 2.51% |
| SELL | retest1 | 2025-12-15 10:45:00 | 247.90 | 2025-12-15 11:25:00 | 248.73 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-17 11:05:00 | 244.55 | 2025-12-17 11:45:00 | 245.10 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-29 10:40:00 | 259.30 | 2025-12-29 11:30:00 | 258.78 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-30 09:40:00 | 260.40 | 2025-12-30 09:50:00 | 259.58 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-07 10:25:00 | 264.80 | 2026-01-07 10:30:00 | 265.65 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-08 10:55:00 | 263.00 | 2026-01-08 11:15:00 | 262.07 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-01-08 10:55:00 | 263.00 | 2026-01-08 12:40:00 | 262.00 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2026-01-13 09:40:00 | 249.60 | 2026-01-13 09:45:00 | 250.46 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-01-14 09:30:00 | 253.30 | 2026-01-14 09:45:00 | 251.68 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2026-01-14 09:30:00 | 253.30 | 2026-01-14 10:30:00 | 253.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-22 11:05:00 | 241.70 | 2026-01-22 11:30:00 | 240.38 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-01-22 11:05:00 | 241.70 | 2026-01-22 15:20:00 | 240.10 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2026-01-30 11:15:00 | 237.80 | 2026-01-30 11:55:00 | 237.06 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-12 10:55:00 | 277.73 | 2026-02-12 11:25:00 | 278.81 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-12 10:55:00 | 277.73 | 2026-02-12 15:20:00 | 279.80 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2026-02-13 10:25:00 | 275.51 | 2026-02-13 11:05:00 | 274.21 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-13 10:25:00 | 275.51 | 2026-02-13 11:50:00 | 275.51 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 10:40:00 | 269.82 | 2026-02-17 10:50:00 | 268.58 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-17 10:40:00 | 269.82 | 2026-02-17 13:05:00 | 269.82 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:55:00 | 268.25 | 2026-02-19 11:00:00 | 268.99 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-20 11:00:00 | 264.62 | 2026-02-20 11:30:00 | 265.49 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-24 09:30:00 | 261.40 | 2026-02-24 09:45:00 | 262.25 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-25 09:35:00 | 265.40 | 2026-02-25 09:40:00 | 264.44 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-26 10:50:00 | 269.67 | 2026-02-26 11:25:00 | 268.90 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-05 10:10:00 | 256.10 | 2026-03-05 11:45:00 | 254.66 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-05 10:10:00 | 256.10 | 2026-03-05 12:25:00 | 256.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:15:00 | 254.40 | 2026-03-10 10:20:00 | 253.36 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-16 09:45:00 | 240.60 | 2026-03-16 09:55:00 | 239.39 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-03-27 11:05:00 | 240.20 | 2026-03-27 11:15:00 | 239.03 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-27 11:05:00 | 240.20 | 2026-03-27 13:15:00 | 240.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-21 10:05:00 | 264.22 | 2026-04-21 11:35:00 | 264.98 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-27 10:55:00 | 266.90 | 2026-04-27 12:00:00 | 268.11 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-27 10:55:00 | 266.90 | 2026-04-27 15:20:00 | 269.38 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2026-05-04 11:10:00 | 268.80 | 2026-05-04 11:35:00 | 268.09 | STOP_HIT | 1.00 | -0.26% |
