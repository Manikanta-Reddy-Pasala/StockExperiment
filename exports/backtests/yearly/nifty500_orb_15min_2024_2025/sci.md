# Shipping Corporation of India Ltd. (SCI)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 339.60
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
| ENTRY1 | 46 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 7 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 70 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 39
- **Target hits / Stop hits / Partials:** 7 / 39 / 24
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 17.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 16 | 40.0% | 3 | 24 | 13 | 0.14% | 5.6% |
| BUY @ 2nd Alert (retest1) | 40 | 16 | 40.0% | 3 | 24 | 13 | 0.14% | 5.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 30 | 15 | 50.0% | 4 | 15 | 11 | 0.39% | 11.6% |
| SELL @ 2nd Alert (retest1) | 30 | 15 | 50.0% | 4 | 15 | 11 | 0.39% | 11.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 70 | 31 | 44.3% | 7 | 39 | 24 | 0.25% | 17.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:15:00 | 230.35 | 232.20 | 0.00 | ORB-short ORB[231.20,234.20] vol=1.5x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 10:20:00 | 228.61 | 232.01 | 0.00 | T1 1.5R @ 228.61 |
| Stop hit — per-position SL triggered | 2024-05-17 10:35:00 | 230.35 | 231.80 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:30:00 | 247.15 | 248.98 | 0.00 | ORB-short ORB[247.75,251.35] vol=2.1x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:35:00 | 244.94 | 247.31 | 0.00 | T1 1.5R @ 244.94 |
| Target hit | 2024-05-30 15:20:00 | 242.15 | 245.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:40:00 | 238.50 | 242.08 | 0.00 | ORB-short ORB[241.05,244.45] vol=2.8x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:00:00 | 236.48 | 239.21 | 0.00 | T1 1.5R @ 236.48 |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 238.50 | 238.52 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 11:05:00 | 245.31 | 245.99 | 0.00 | ORB-short ORB[245.99,249.40] vol=1.5x ATR=1.02 |
| Stop hit — per-position SL triggered | 2024-06-10 11:40:00 | 246.33 | 246.02 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:15:00 | 262.00 | 262.68 | 0.00 | ORB-short ORB[262.05,264.35] vol=2.0x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-06-25 10:55:00 | 263.15 | 262.69 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:40:00 | 260.17 | 258.35 | 0.00 | ORB-long ORB[255.50,259.00] vol=2.5x ATR=1.03 |
| Stop hit — per-position SL triggered | 2024-06-27 10:10:00 | 259.14 | 258.71 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:35:00 | 258.48 | 255.89 | 0.00 | ORB-long ORB[252.90,256.28] vol=1.9x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-06-28 10:00:00 | 256.94 | 257.14 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:15:00 | 255.80 | 253.80 | 0.00 | ORB-long ORB[251.55,254.70] vol=2.1x ATR=1.02 |
| Stop hit — per-position SL triggered | 2024-07-01 10:20:00 | 254.78 | 253.91 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:40:00 | 257.80 | 256.18 | 0.00 | ORB-long ORB[254.10,257.25] vol=2.6x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 09:45:00 | 259.59 | 257.15 | 0.00 | T1 1.5R @ 259.59 |
| Target hit | 2024-07-03 12:40:00 | 262.30 | 262.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2024-07-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:45:00 | 276.20 | 274.18 | 0.00 | ORB-long ORB[272.30,275.20] vol=2.9x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 09:50:00 | 278.21 | 275.24 | 0.00 | T1 1.5R @ 278.21 |
| Stop hit — per-position SL triggered | 2024-07-09 09:55:00 | 276.20 | 275.43 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:45:00 | 266.60 | 268.61 | 0.00 | ORB-short ORB[267.70,271.10] vol=1.7x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:00:00 | 264.75 | 267.82 | 0.00 | T1 1.5R @ 264.75 |
| Stop hit — per-position SL triggered | 2024-07-10 13:50:00 | 266.60 | 266.28 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 263.85 | 266.44 | 0.00 | ORB-short ORB[264.95,268.90] vol=1.6x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:35:00 | 261.56 | 265.73 | 0.00 | T1 1.5R @ 261.56 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 263.85 | 264.93 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-08-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:00:00 | 275.20 | 272.78 | 0.00 | ORB-long ORB[272.10,274.05] vol=2.1x ATR=1.03 |
| Stop hit — per-position SL triggered | 2024-08-22 10:05:00 | 274.17 | 272.83 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-08-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 10:10:00 | 277.00 | 275.69 | 0.00 | ORB-long ORB[273.50,276.40] vol=2.9x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 10:15:00 | 278.40 | 276.57 | 0.00 | T1 1.5R @ 278.40 |
| Stop hit — per-position SL triggered | 2024-08-23 10:50:00 | 277.00 | 277.10 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 268.15 | 268.99 | 0.00 | ORB-short ORB[268.55,271.00] vol=1.6x ATR=0.77 |
| Stop hit — per-position SL triggered | 2024-08-29 09:40:00 | 268.92 | 268.88 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:30:00 | 263.25 | 265.09 | 0.00 | ORB-short ORB[264.00,267.75] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2024-08-30 09:40:00 | 264.11 | 264.73 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-09-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:50:00 | 259.30 | 261.71 | 0.00 | ORB-short ORB[261.50,265.00] vol=2.4x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 257.97 | 260.69 | 0.00 | T1 1.5R @ 257.97 |
| Stop hit — per-position SL triggered | 2024-09-06 10:25:00 | 259.30 | 260.23 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 09:30:00 | 245.75 | 246.60 | 0.00 | ORB-short ORB[246.15,247.75] vol=1.8x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 09:50:00 | 244.72 | 246.08 | 0.00 | T1 1.5R @ 244.72 |
| Stop hit — per-position SL triggered | 2024-09-16 13:05:00 | 245.75 | 247.12 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-09-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:45:00 | 247.20 | 246.50 | 0.00 | ORB-long ORB[244.90,246.75] vol=1.7x ATR=0.77 |
| Stop hit — per-position SL triggered | 2024-09-18 10:30:00 | 246.43 | 246.61 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 09:45:00 | 241.85 | 242.84 | 0.00 | ORB-short ORB[242.00,245.00] vol=2.1x ATR=0.73 |
| Stop hit — per-position SL triggered | 2024-10-14 09:55:00 | 242.58 | 243.35 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-10-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 11:05:00 | 241.55 | 238.54 | 0.00 | ORB-long ORB[237.65,240.05] vol=4.6x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 11:10:00 | 243.09 | 239.12 | 0.00 | T1 1.5R @ 243.09 |
| Stop hit — per-position SL triggered | 2024-10-15 12:15:00 | 241.55 | 241.67 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-10-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:10:00 | 244.60 | 240.14 | 0.00 | ORB-long ORB[237.20,240.50] vol=4.9x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:15:00 | 246.90 | 244.34 | 0.00 | T1 1.5R @ 246.90 |
| Stop hit — per-position SL triggered | 2024-10-16 10:35:00 | 244.60 | 245.05 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-10-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:45:00 | 236.15 | 238.01 | 0.00 | ORB-short ORB[237.05,240.50] vol=1.9x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:50:00 | 234.51 | 237.65 | 0.00 | T1 1.5R @ 234.51 |
| Stop hit — per-position SL triggered | 2024-10-21 10:00:00 | 236.15 | 237.38 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-11-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:50:00 | 229.50 | 226.84 | 0.00 | ORB-long ORB[224.33,227.24] vol=2.0x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 10:00:00 | 230.98 | 228.50 | 0.00 | T1 1.5R @ 230.98 |
| Stop hit — per-position SL triggered | 2024-11-27 10:50:00 | 229.50 | 229.38 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:35:00 | 238.85 | 237.66 | 0.00 | ORB-long ORB[236.04,238.60] vol=1.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-12-06 09:50:00 | 237.77 | 238.15 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-12-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:45:00 | 240.09 | 239.00 | 0.00 | ORB-long ORB[237.55,239.80] vol=1.5x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-12-09 09:50:00 | 239.17 | 239.11 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-12-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:05:00 | 239.70 | 238.39 | 0.00 | ORB-long ORB[237.00,239.34] vol=1.8x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-12-11 10:55:00 | 238.61 | 238.71 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 235.50 | 237.05 | 0.00 | ORB-short ORB[236.80,238.92] vol=1.8x ATR=0.61 |
| Stop hit — per-position SL triggered | 2024-12-12 09:55:00 | 236.11 | 236.65 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-12-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:45:00 | 229.92 | 231.84 | 0.00 | ORB-short ORB[231.70,234.19] vol=4.3x ATR=0.77 |
| Stop hit — per-position SL triggered | 2024-12-17 09:55:00 | 230.69 | 231.62 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 11:15:00 | 220.60 | 218.73 | 0.00 | ORB-long ORB[216.84,219.47] vol=1.9x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-12-19 12:10:00 | 219.84 | 219.02 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:50:00 | 212.09 | 210.34 | 0.00 | ORB-long ORB[209.20,210.94] vol=1.6x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-12-24 11:35:00 | 211.13 | 211.15 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-12-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:45:00 | 208.35 | 210.03 | 0.00 | ORB-short ORB[210.00,212.38] vol=1.5x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-12-26 10:55:00 | 209.10 | 209.95 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 10:35:00 | 206.99 | 205.18 | 0.00 | ORB-long ORB[203.55,205.90] vol=1.6x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 10:40:00 | 208.37 | 205.73 | 0.00 | T1 1.5R @ 208.37 |
| Stop hit — per-position SL triggered | 2024-12-31 11:25:00 | 206.99 | 206.68 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 11:05:00 | 211.40 | 210.61 | 0.00 | ORB-long ORB[208.60,211.23] vol=2.3x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-01-01 11:10:00 | 210.81 | 210.63 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 217.00 | 215.24 | 0.00 | ORB-long ORB[212.69,215.45] vol=6.3x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:35:00 | 218.32 | 215.98 | 0.00 | T1 1.5R @ 218.32 |
| Stop hit — per-position SL triggered | 2025-01-02 09:40:00 | 217.00 | 216.10 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:35:00 | 208.12 | 206.97 | 0.00 | ORB-long ORB[205.02,207.80] vol=1.9x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 09:45:00 | 209.41 | 207.65 | 0.00 | T1 1.5R @ 209.41 |
| Stop hit — per-position SL triggered | 2025-01-09 09:50:00 | 208.12 | 207.70 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:35:00 | 198.99 | 197.42 | 0.00 | ORB-long ORB[195.43,198.07] vol=1.9x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:15:00 | 200.30 | 198.31 | 0.00 | T1 1.5R @ 200.30 |
| Target hit | 2025-01-23 11:50:00 | 199.02 | 199.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — SELL (started 2025-01-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:35:00 | 196.41 | 197.85 | 0.00 | ORB-short ORB[196.90,199.65] vol=1.7x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 09:45:00 | 195.10 | 197.39 | 0.00 | T1 1.5R @ 195.10 |
| Target hit | 2025-01-24 11:40:00 | 196.40 | 196.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 39 — BUY (started 2025-01-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:40:00 | 193.63 | 192.18 | 0.00 | ORB-long ORB[190.91,192.92] vol=1.9x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:25:00 | 194.82 | 193.38 | 0.00 | T1 1.5R @ 194.82 |
| Stop hit — per-position SL triggered | 2025-01-30 10:35:00 | 193.63 | 193.42 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-01-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:40:00 | 193.70 | 192.49 | 0.00 | ORB-long ORB[190.91,193.53] vol=1.7x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 09:50:00 | 194.99 | 192.96 | 0.00 | T1 1.5R @ 194.99 |
| Target hit | 2025-01-31 12:10:00 | 196.78 | 196.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2025-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:45:00 | 153.00 | 153.72 | 0.00 | ORB-short ORB[153.55,155.63] vol=2.8x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 11:15:00 | 151.91 | 153.11 | 0.00 | T1 1.5R @ 151.91 |
| Target hit | 2025-02-27 15:20:00 | 151.40 | 152.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2025-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:30:00 | 150.25 | 149.22 | 0.00 | ORB-long ORB[148.00,149.71] vol=1.7x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-03-17 09:35:00 | 149.66 | 149.22 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-03-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:00:00 | 181.05 | 179.31 | 0.00 | ORB-long ORB[170.95,173.69] vol=4.1x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-03-24 10:25:00 | 179.16 | 179.57 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-03-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 10:55:00 | 172.87 | 174.14 | 0.00 | ORB-short ORB[173.80,175.90] vol=3.4x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 11:55:00 | 171.55 | 173.80 | 0.00 | T1 1.5R @ 171.55 |
| Target hit | 2025-03-26 15:20:00 | 166.94 | 171.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2025-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:30:00 | 174.74 | 173.68 | 0.00 | ORB-long ORB[172.30,174.34] vol=2.8x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-04-15 09:45:00 | 173.93 | 173.90 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:55:00 | 177.50 | 175.34 | 0.00 | ORB-long ORB[173.10,175.32] vol=3.8x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 10:00:00 | 178.56 | 176.14 | 0.00 | T1 1.5R @ 178.56 |
| Stop hit — per-position SL triggered | 2025-04-16 10:05:00 | 177.50 | 176.24 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-17 10:15:00 | 230.35 | 2024-05-17 10:20:00 | 228.61 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2024-05-17 10:15:00 | 230.35 | 2024-05-17 10:35:00 | 230.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-30 09:30:00 | 247.15 | 2024-05-30 11:35:00 | 244.94 | PARTIAL | 0.50 | 0.89% |
| SELL | retest1 | 2024-05-30 09:30:00 | 247.15 | 2024-05-30 15:20:00 | 242.15 | TARGET_HIT | 0.50 | 2.02% |
| SELL | retest1 | 2024-05-31 09:40:00 | 238.50 | 2024-05-31 10:00:00 | 236.48 | PARTIAL | 0.50 | 0.85% |
| SELL | retest1 | 2024-05-31 09:40:00 | 238.50 | 2024-05-31 11:15:00 | 238.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-10 11:05:00 | 245.31 | 2024-06-10 11:40:00 | 246.33 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-06-25 10:15:00 | 262.00 | 2024-06-25 10:55:00 | 263.15 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-06-27 09:40:00 | 260.17 | 2024-06-27 10:10:00 | 259.14 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-28 09:35:00 | 258.48 | 2024-06-28 10:00:00 | 256.94 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2024-07-01 10:15:00 | 255.80 | 2024-07-01 10:20:00 | 254.78 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-07-03 09:40:00 | 257.80 | 2024-07-03 09:45:00 | 259.59 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-07-03 09:40:00 | 257.80 | 2024-07-03 12:40:00 | 262.30 | TARGET_HIT | 0.50 | 1.75% |
| BUY | retest1 | 2024-07-09 09:45:00 | 276.20 | 2024-07-09 09:50:00 | 278.21 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-07-09 09:45:00 | 276.20 | 2024-07-09 09:55:00 | 276.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 09:45:00 | 266.60 | 2024-07-10 10:00:00 | 264.75 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-07-10 09:45:00 | 266.60 | 2024-07-10 13:50:00 | 266.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-14 09:30:00 | 263.85 | 2024-08-14 09:35:00 | 261.56 | PARTIAL | 0.50 | 0.87% |
| SELL | retest1 | 2024-08-14 09:30:00 | 263.85 | 2024-08-14 09:45:00 | 263.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 10:00:00 | 275.20 | 2024-08-22 10:05:00 | 274.17 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-08-23 10:10:00 | 277.00 | 2024-08-23 10:15:00 | 278.40 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-08-23 10:10:00 | 277.00 | 2024-08-23 10:50:00 | 277.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-29 09:30:00 | 268.15 | 2024-08-29 09:40:00 | 268.92 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-30 09:30:00 | 263.25 | 2024-08-30 09:40:00 | 264.11 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-06 09:50:00 | 259.30 | 2024-09-06 10:05:00 | 257.97 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-06 09:50:00 | 259.30 | 2024-09-06 10:25:00 | 259.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-16 09:30:00 | 245.75 | 2024-09-16 09:50:00 | 244.72 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-16 09:30:00 | 245.75 | 2024-09-16 13:05:00 | 245.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-18 09:45:00 | 247.20 | 2024-09-18 10:30:00 | 246.43 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-14 09:45:00 | 241.85 | 2024-10-14 09:55:00 | 242.58 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-15 11:05:00 | 241.55 | 2024-10-15 11:10:00 | 243.09 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-10-15 11:05:00 | 241.55 | 2024-10-15 12:15:00 | 241.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-16 10:10:00 | 244.60 | 2024-10-16 10:15:00 | 246.90 | PARTIAL | 0.50 | 0.94% |
| BUY | retest1 | 2024-10-16 10:10:00 | 244.60 | 2024-10-16 10:35:00 | 244.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 09:45:00 | 236.15 | 2024-10-21 09:50:00 | 234.51 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-10-21 09:45:00 | 236.15 | 2024-10-21 10:00:00 | 236.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:50:00 | 229.50 | 2024-11-27 10:00:00 | 230.98 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-11-27 09:50:00 | 229.50 | 2024-11-27 10:50:00 | 229.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-06 09:35:00 | 238.85 | 2024-12-06 09:50:00 | 237.77 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-12-09 09:45:00 | 240.09 | 2024-12-09 09:50:00 | 239.17 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-11 10:05:00 | 239.70 | 2024-12-11 10:55:00 | 238.61 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-12-12 09:40:00 | 235.50 | 2024-12-12 09:55:00 | 236.11 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-17 09:45:00 | 229.92 | 2024-12-17 09:55:00 | 230.69 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-19 11:15:00 | 220.60 | 2024-12-19 12:10:00 | 219.84 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-12-24 09:50:00 | 212.09 | 2024-12-24 11:35:00 | 211.13 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-12-26 10:45:00 | 208.35 | 2024-12-26 10:55:00 | 209.10 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-12-31 10:35:00 | 206.99 | 2024-12-31 10:40:00 | 208.37 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-12-31 10:35:00 | 206.99 | 2024-12-31 11:25:00 | 206.99 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-01 11:05:00 | 211.40 | 2025-01-01 11:10:00 | 210.81 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-02 09:30:00 | 217.00 | 2025-01-02 09:35:00 | 218.32 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-01-02 09:30:00 | 217.00 | 2025-01-02 09:40:00 | 217.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-09 09:35:00 | 208.12 | 2025-01-09 09:45:00 | 209.41 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-01-09 09:35:00 | 208.12 | 2025-01-09 09:50:00 | 208.12 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-23 09:35:00 | 198.99 | 2025-01-23 10:15:00 | 200.30 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-01-23 09:35:00 | 198.99 | 2025-01-23 11:50:00 | 199.02 | TARGET_HIT | 0.50 | 0.02% |
| SELL | retest1 | 2025-01-24 09:35:00 | 196.41 | 2025-01-24 09:45:00 | 195.10 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2025-01-24 09:35:00 | 196.41 | 2025-01-24 11:40:00 | 196.40 | TARGET_HIT | 0.50 | 0.01% |
| BUY | retest1 | 2025-01-30 09:40:00 | 193.63 | 2025-01-30 10:25:00 | 194.82 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-01-30 09:40:00 | 193.63 | 2025-01-30 10:35:00 | 193.63 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-31 09:40:00 | 193.70 | 2025-01-31 09:50:00 | 194.99 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-01-31 09:40:00 | 193.70 | 2025-01-31 12:10:00 | 196.78 | TARGET_HIT | 0.50 | 1.59% |
| SELL | retest1 | 2025-02-27 09:45:00 | 153.00 | 2025-02-27 11:15:00 | 151.91 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2025-02-27 09:45:00 | 153.00 | 2025-02-27 15:20:00 | 151.40 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2025-03-17 09:30:00 | 150.25 | 2025-03-17 09:35:00 | 149.66 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-24 10:00:00 | 181.05 | 2025-03-24 10:25:00 | 179.16 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest1 | 2025-03-26 10:55:00 | 172.87 | 2025-03-26 11:55:00 | 171.55 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2025-03-26 10:55:00 | 172.87 | 2025-03-26 15:20:00 | 166.94 | TARGET_HIT | 0.50 | 3.43% |
| BUY | retest1 | 2025-04-15 09:30:00 | 174.74 | 2025-04-15 09:45:00 | 173.93 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-04-16 09:55:00 | 177.50 | 2025-04-16 10:00:00 | 178.56 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-04-16 09:55:00 | 177.50 | 2025-04-16 10:05:00 | 177.50 | STOP_HIT | 0.50 | 0.00% |
