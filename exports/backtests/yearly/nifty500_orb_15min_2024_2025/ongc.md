# Oil & Natural Gas Corporation Ltd. (ONGC)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 279.00
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
| ENTRY1 | 58 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 10 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 82 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 48
- **Target hits / Stop hits / Partials:** 10 / 48 / 24
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 15.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 19 | 45.2% | 7 | 23 | 12 | 0.22% | 9.3% |
| BUY @ 2nd Alert (retest1) | 42 | 19 | 45.2% | 7 | 23 | 12 | 0.22% | 9.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 40 | 15 | 37.5% | 3 | 25 | 12 | 0.15% | 5.9% |
| SELL @ 2nd Alert (retest1) | 40 | 15 | 37.5% | 3 | 25 | 12 | 0.15% | 5.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 82 | 34 | 41.5% | 10 | 48 | 24 | 0.19% | 15.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:40:00 | 278.35 | 277.28 | 0.00 | ORB-long ORB[275.25,278.25] vol=1.9x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-05-16 09:50:00 | 277.51 | 277.56 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:35:00 | 281.35 | 280.17 | 0.00 | ORB-long ORB[279.10,280.90] vol=1.6x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 10:30:00 | 282.58 | 280.98 | 0.00 | T1 1.5R @ 282.58 |
| Stop hit — per-position SL triggered | 2024-05-23 10:40:00 | 281.35 | 281.02 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 273.85 | 275.04 | 0.00 | ORB-short ORB[274.00,278.00] vol=1.5x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:50:00 | 272.69 | 274.47 | 0.00 | T1 1.5R @ 272.69 |
| Stop hit — per-position SL triggered | 2024-06-13 09:55:00 | 273.85 | 274.38 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 09:30:00 | 276.20 | 277.53 | 0.00 | ORB-short ORB[277.10,279.00] vol=1.8x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-06-18 09:45:00 | 276.90 | 277.33 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:30:00 | 272.40 | 271.15 | 0.00 | ORB-long ORB[270.10,271.60] vol=1.8x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 09:40:00 | 273.50 | 272.02 | 0.00 | T1 1.5R @ 273.50 |
| Target hit | 2024-06-21 10:45:00 | 272.75 | 272.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 11:15:00 | 271.00 | 268.72 | 0.00 | ORB-long ORB[267.00,270.00] vol=2.0x ATR=0.74 |
| Stop hit — per-position SL triggered | 2024-06-24 11:20:00 | 270.26 | 268.75 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:15:00 | 267.95 | 269.51 | 0.00 | ORB-short ORB[269.50,270.80] vol=2.0x ATR=0.54 |
| Stop hit — per-position SL triggered | 2024-06-25 10:30:00 | 268.49 | 269.11 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-07-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:25:00 | 273.70 | 275.12 | 0.00 | ORB-short ORB[274.70,276.60] vol=1.8x ATR=0.60 |
| Stop hit — per-position SL triggered | 2024-07-04 10:40:00 | 274.30 | 274.97 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:40:00 | 281.70 | 279.35 | 0.00 | ORB-long ORB[276.50,279.55] vol=2.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-07-05 09:50:00 | 280.60 | 280.13 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:35:00 | 289.40 | 295.61 | 0.00 | ORB-short ORB[297.50,299.80] vol=1.6x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 290.92 | 295.35 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 10:50:00 | 300.70 | 297.89 | 0.00 | ORB-long ORB[296.75,300.00] vol=5.0x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 11:00:00 | 302.54 | 298.55 | 0.00 | T1 1.5R @ 302.54 |
| Target hit | 2024-07-11 15:20:00 | 305.00 | 303.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2024-07-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:30:00 | 310.55 | 307.97 | 0.00 | ORB-long ORB[304.80,309.20] vol=2.3x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-07-12 10:55:00 | 309.32 | 308.47 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:20:00 | 314.00 | 312.02 | 0.00 | ORB-long ORB[308.65,313.25] vol=1.6x ATR=1.39 |
| Stop hit — per-position SL triggered | 2024-07-15 10:40:00 | 312.61 | 312.17 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:50:00 | 317.20 | 318.70 | 0.00 | ORB-short ORB[317.25,321.40] vol=1.5x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 10:00:00 | 315.45 | 318.37 | 0.00 | T1 1.5R @ 315.45 |
| Target hit | 2024-07-23 12:50:00 | 313.20 | 311.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2024-07-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:05:00 | 320.40 | 316.55 | 0.00 | ORB-long ORB[312.10,316.75] vol=1.5x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 10:20:00 | 323.33 | 317.99 | 0.00 | T1 1.5R @ 323.33 |
| Stop hit — per-position SL triggered | 2024-07-24 11:30:00 | 320.40 | 320.60 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:15:00 | 343.00 | 340.47 | 0.00 | ORB-long ORB[337.15,341.30] vol=2.4x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-08-01 11:40:00 | 341.57 | 341.65 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 11:15:00 | 329.80 | 331.64 | 0.00 | ORB-short ORB[331.45,335.75] vol=1.9x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-08-16 12:05:00 | 330.90 | 331.19 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 11:10:00 | 328.40 | 329.82 | 0.00 | ORB-short ORB[329.15,330.70] vol=1.6x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 12:00:00 | 327.43 | 329.47 | 0.00 | T1 1.5R @ 327.43 |
| Stop hit — per-position SL triggered | 2024-08-21 14:25:00 | 328.40 | 328.71 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 328.05 | 329.21 | 0.00 | ORB-short ORB[328.55,330.50] vol=1.9x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:15:00 | 326.78 | 328.57 | 0.00 | T1 1.5R @ 326.78 |
| Target hit | 2024-08-22 15:20:00 | 324.25 | 325.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2024-08-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 11:10:00 | 321.40 | 322.48 | 0.00 | ORB-short ORB[321.50,324.80] vol=1.6x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:30:00 | 320.30 | 322.19 | 0.00 | T1 1.5R @ 320.30 |
| Stop hit — per-position SL triggered | 2024-08-23 12:00:00 | 321.40 | 322.05 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-09-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 10:20:00 | 324.60 | 326.27 | 0.00 | ORB-short ORB[326.45,328.00] vol=3.1x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 10:45:00 | 323.66 | 325.57 | 0.00 | T1 1.5R @ 323.66 |
| Stop hit — per-position SL triggered | 2024-09-03 10:50:00 | 324.60 | 325.54 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 11:10:00 | 306.00 | 307.46 | 0.00 | ORB-short ORB[309.60,313.00] vol=1.8x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-09-06 11:20:00 | 307.20 | 307.39 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:40:00 | 297.90 | 296.29 | 0.00 | ORB-long ORB[294.70,297.40] vol=1.6x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-09-24 09:55:00 | 297.07 | 296.65 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-10-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:35:00 | 299.60 | 296.38 | 0.00 | ORB-long ORB[294.20,297.65] vol=1.6x ATR=1.13 |
| Stop hit — per-position SL triggered | 2024-10-04 10:50:00 | 298.47 | 296.65 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-10-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:45:00 | 284.05 | 290.14 | 0.00 | ORB-short ORB[292.55,296.30] vol=2.9x ATR=1.19 |
| Stop hit — per-position SL triggered | 2024-10-07 10:55:00 | 285.24 | 289.65 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-10-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:00:00 | 281.75 | 283.60 | 0.00 | ORB-short ORB[282.55,285.20] vol=1.8x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 12:30:00 | 280.89 | 282.97 | 0.00 | T1 1.5R @ 280.89 |
| Stop hit — per-position SL triggered | 2024-10-16 12:40:00 | 281.75 | 282.89 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-10-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 10:35:00 | 266.35 | 264.10 | 0.00 | ORB-long ORB[261.15,263.50] vol=2.1x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 10:50:00 | 267.49 | 264.72 | 0.00 | T1 1.5R @ 267.49 |
| Stop hit — per-position SL triggered | 2024-10-31 12:40:00 | 266.35 | 266.00 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-11-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 11:00:00 | 261.65 | 263.22 | 0.00 | ORB-short ORB[262.80,266.50] vol=1.9x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 11:50:00 | 260.54 | 262.75 | 0.00 | T1 1.5R @ 260.54 |
| Stop hit — per-position SL triggered | 2024-11-08 13:30:00 | 261.65 | 262.11 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-11-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 10:55:00 | 249.80 | 251.27 | 0.00 | ORB-short ORB[250.80,254.30] vol=1.7x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-11-14 11:55:00 | 250.71 | 250.75 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-11-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:30:00 | 248.95 | 250.04 | 0.00 | ORB-short ORB[249.25,251.95] vol=2.2x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:40:00 | 247.65 | 249.68 | 0.00 | T1 1.5R @ 247.65 |
| Stop hit — per-position SL triggered | 2024-11-18 10:00:00 | 248.95 | 249.33 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-11-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 10:25:00 | 257.10 | 255.95 | 0.00 | ORB-long ORB[254.35,255.70] vol=1.6x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-11-28 10:30:00 | 256.52 | 255.95 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-11-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 11:00:00 | 254.70 | 253.67 | 0.00 | ORB-long ORB[252.25,253.90] vol=1.9x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-11-29 11:10:00 | 254.11 | 253.74 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:40:00 | 264.05 | 261.16 | 0.00 | ORB-long ORB[259.40,261.80] vol=1.6x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-12-03 11:50:00 | 263.12 | 261.88 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 257.95 | 259.28 | 0.00 | ORB-short ORB[259.30,261.60] vol=1.8x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 11:30:00 | 256.98 | 258.99 | 0.00 | T1 1.5R @ 256.98 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 257.95 | 258.85 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 10:15:00 | 257.85 | 258.92 | 0.00 | ORB-short ORB[259.00,261.35] vol=1.9x ATR=0.51 |
| Stop hit — per-position SL triggered | 2024-12-09 10:30:00 | 258.36 | 258.63 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-12-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:05:00 | 255.20 | 256.60 | 0.00 | ORB-short ORB[257.10,258.60] vol=1.5x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-12-12 11:10:00 | 255.62 | 256.57 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:25:00 | 250.15 | 251.18 | 0.00 | ORB-short ORB[251.65,253.65] vol=1.6x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-12-13 10:35:00 | 250.74 | 251.03 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:00:00 | 251.65 | 253.24 | 0.00 | ORB-short ORB[252.90,256.30] vol=2.0x ATR=0.51 |
| Stop hit — per-position SL triggered | 2024-12-16 11:10:00 | 252.16 | 253.15 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 11:10:00 | 249.00 | 249.77 | 0.00 | ORB-short ORB[250.80,252.45] vol=2.0x ATR=0.48 |
| Stop hit — per-position SL triggered | 2024-12-17 11:35:00 | 249.48 | 249.68 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 09:35:00 | 241.65 | 240.55 | 0.00 | ORB-long ORB[239.25,241.25] vol=1.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-12-26 09:40:00 | 241.00 | 240.65 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:55:00 | 239.20 | 239.98 | 0.00 | ORB-short ORB[239.40,240.35] vol=3.2x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:20:00 | 238.52 | 239.80 | 0.00 | T1 1.5R @ 238.52 |
| Stop hit — per-position SL triggered | 2024-12-27 11:25:00 | 239.20 | 239.78 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 239.72 | 238.60 | 0.00 | ORB-long ORB[237.00,238.88] vol=1.6x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:40:00 | 240.56 | 239.20 | 0.00 | T1 1.5R @ 240.56 |
| Target hit | 2025-01-02 15:20:00 | 245.81 | 243.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-01-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 10:30:00 | 263.45 | 261.02 | 0.00 | ORB-long ORB[258.54,261.00] vol=2.3x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-01-15 11:15:00 | 262.63 | 261.91 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-01-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:25:00 | 265.17 | 262.40 | 0.00 | ORB-long ORB[260.00,262.45] vol=3.1x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-01-16 10:30:00 | 264.23 | 262.85 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:30:00 | 265.40 | 264.13 | 0.00 | ORB-long ORB[261.30,265.10] vol=1.8x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-01-17 09:35:00 | 264.66 | 264.27 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-01-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:35:00 | 254.98 | 253.56 | 0.00 | ORB-long ORB[250.94,254.70] vol=1.6x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 10:00:00 | 256.10 | 254.31 | 0.00 | T1 1.5R @ 256.10 |
| Target hit | 2025-01-30 13:30:00 | 257.25 | 257.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2025-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 10:50:00 | 240.00 | 239.03 | 0.00 | ORB-long ORB[236.50,239.55] vol=3.0x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 12:25:00 | 240.95 | 239.56 | 0.00 | T1 1.5R @ 240.95 |
| Target hit | 2025-02-20 15:20:00 | 241.85 | 240.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2025-02-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 09:35:00 | 242.65 | 242.27 | 0.00 | ORB-long ORB[240.05,242.50] vol=3.3x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-02-21 09:40:00 | 242.04 | 242.26 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 09:30:00 | 227.45 | 229.40 | 0.00 | ORB-short ORB[228.73,231.51] vol=1.9x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-03-06 09:35:00 | 228.19 | 229.13 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-03-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:55:00 | 235.24 | 233.25 | 0.00 | ORB-long ORB[231.86,234.90] vol=3.2x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 11:00:00 | 236.31 | 233.54 | 0.00 | T1 1.5R @ 236.31 |
| Stop hit — per-position SL triggered | 2025-03-07 11:05:00 | 235.24 | 233.62 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-03-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-10 11:05:00 | 231.17 | 232.83 | 0.00 | ORB-short ORB[231.61,235.00] vol=1.8x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 11:15:00 | 230.30 | 232.64 | 0.00 | T1 1.5R @ 230.30 |
| Target hit | 2025-03-10 15:20:00 | 223.00 | 226.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2025-03-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:25:00 | 234.19 | 233.60 | 0.00 | ORB-long ORB[231.60,233.44] vol=3.8x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-03-19 10:45:00 | 233.66 | 233.64 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-04-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:00:00 | 234.01 | 233.03 | 0.00 | ORB-long ORB[231.72,233.79] vol=1.6x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 11:05:00 | 234.88 | 233.67 | 0.00 | T1 1.5R @ 234.88 |
| Target hit | 2025-04-16 15:20:00 | 241.66 | 237.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 11:15:00 | 243.20 | 241.05 | 0.00 | ORB-long ORB[239.84,242.23] vol=4.0x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 11:40:00 | 244.42 | 241.60 | 0.00 | T1 1.5R @ 244.42 |
| Stop hit — per-position SL triggered | 2025-04-17 11:55:00 | 243.20 | 241.78 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-04-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:50:00 | 250.90 | 249.58 | 0.00 | ORB-long ORB[248.25,250.72] vol=1.5x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-04-23 09:55:00 | 250.07 | 249.62 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 247.93 | 249.00 | 0.00 | ORB-short ORB[248.60,250.68] vol=1.7x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-04-25 09:35:00 | 248.59 | 249.01 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-04-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 11:10:00 | 249.66 | 248.62 | 0.00 | ORB-long ORB[244.38,247.30] vol=1.5x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 11:55:00 | 250.75 | 248.91 | 0.00 | T1 1.5R @ 250.75 |
| Target hit | 2025-04-28 15:20:00 | 250.28 | 249.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2025-05-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 10:20:00 | 238.78 | 240.98 | 0.00 | ORB-short ORB[240.80,242.99] vol=2.0x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-05-06 10:35:00 | 239.54 | 240.88 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 09:40:00 | 278.35 | 2024-05-16 09:50:00 | 277.51 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-05-23 09:35:00 | 281.35 | 2024-05-23 10:30:00 | 282.58 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-05-23 09:35:00 | 281.35 | 2024-05-23 10:40:00 | 281.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 09:30:00 | 273.85 | 2024-06-13 09:50:00 | 272.69 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-06-13 09:30:00 | 273.85 | 2024-06-13 09:55:00 | 273.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-18 09:30:00 | 276.20 | 2024-06-18 09:45:00 | 276.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-06-21 09:30:00 | 272.40 | 2024-06-21 09:40:00 | 273.50 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-06-21 09:30:00 | 272.40 | 2024-06-21 10:45:00 | 272.75 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-06-24 11:15:00 | 271.00 | 2024-06-24 11:20:00 | 270.26 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-06-25 10:15:00 | 267.95 | 2024-06-25 10:30:00 | 268.49 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-07-04 10:25:00 | 273.70 | 2024-07-04 10:40:00 | 274.30 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-07-05 09:40:00 | 281.70 | 2024-07-05 09:50:00 | 280.60 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-07-10 10:35:00 | 289.40 | 2024-07-10 10:40:00 | 290.92 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-07-11 10:50:00 | 300.70 | 2024-07-11 11:00:00 | 302.54 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-07-11 10:50:00 | 300.70 | 2024-07-11 15:20:00 | 305.00 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2024-07-12 10:30:00 | 310.55 | 2024-07-12 10:55:00 | 309.32 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-07-15 10:20:00 | 314.00 | 2024-07-15 10:40:00 | 312.61 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-07-23 09:50:00 | 317.20 | 2024-07-23 10:00:00 | 315.45 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-07-23 09:50:00 | 317.20 | 2024-07-23 12:50:00 | 313.20 | TARGET_HIT | 0.50 | 1.26% |
| BUY | retest1 | 2024-07-24 10:05:00 | 320.40 | 2024-07-24 10:20:00 | 323.33 | PARTIAL | 0.50 | 0.92% |
| BUY | retest1 | 2024-07-24 10:05:00 | 320.40 | 2024-07-24 11:30:00 | 320.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-01 10:15:00 | 343.00 | 2024-08-01 11:40:00 | 341.57 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-08-16 11:15:00 | 329.80 | 2024-08-16 12:05:00 | 330.90 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-21 11:10:00 | 328.40 | 2024-08-21 12:00:00 | 327.43 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-08-21 11:10:00 | 328.40 | 2024-08-21 14:25:00 | 328.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-22 09:30:00 | 328.05 | 2024-08-22 10:15:00 | 326.78 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-08-22 09:30:00 | 328.05 | 2024-08-22 15:20:00 | 324.25 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2024-08-23 11:10:00 | 321.40 | 2024-08-23 11:30:00 | 320.30 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-08-23 11:10:00 | 321.40 | 2024-08-23 12:00:00 | 321.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-03 10:20:00 | 324.60 | 2024-09-03 10:45:00 | 323.66 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-09-03 10:20:00 | 324.60 | 2024-09-03 10:50:00 | 324.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 11:10:00 | 306.00 | 2024-09-06 11:20:00 | 307.20 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-09-24 09:40:00 | 297.90 | 2024-09-24 09:55:00 | 297.07 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-10-04 10:35:00 | 299.60 | 2024-10-04 10:50:00 | 298.47 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-07 10:45:00 | 284.05 | 2024-10-07 10:55:00 | 285.24 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-10-16 11:00:00 | 281.75 | 2024-10-16 12:30:00 | 280.89 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-10-16 11:00:00 | 281.75 | 2024-10-16 12:40:00 | 281.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 10:35:00 | 266.35 | 2024-10-31 10:50:00 | 267.49 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-10-31 10:35:00 | 266.35 | 2024-10-31 12:40:00 | 266.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-08 11:00:00 | 261.65 | 2024-11-08 11:50:00 | 260.54 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-11-08 11:00:00 | 261.65 | 2024-11-08 13:30:00 | 261.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-14 10:55:00 | 249.80 | 2024-11-14 11:55:00 | 250.71 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-11-18 09:30:00 | 248.95 | 2024-11-18 09:40:00 | 247.65 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-11-18 09:30:00 | 248.95 | 2024-11-18 10:00:00 | 248.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-28 10:25:00 | 257.10 | 2024-11-28 10:30:00 | 256.52 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-11-29 11:00:00 | 254.70 | 2024-11-29 11:10:00 | 254.11 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-03 10:40:00 | 264.05 | 2024-12-03 11:50:00 | 263.12 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-12-05 10:55:00 | 257.95 | 2024-12-05 11:30:00 | 256.98 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-05 10:55:00 | 257.95 | 2024-12-05 12:05:00 | 257.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-09 10:15:00 | 257.85 | 2024-12-09 10:30:00 | 258.36 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-12-12 11:05:00 | 255.20 | 2024-12-12 11:10:00 | 255.62 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-12-13 10:25:00 | 250.15 | 2024-12-13 10:35:00 | 250.74 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-16 11:00:00 | 251.65 | 2024-12-16 11:10:00 | 252.16 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-12-17 11:10:00 | 249.00 | 2024-12-17 11:35:00 | 249.48 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-12-26 09:35:00 | 241.65 | 2024-12-26 09:40:00 | 241.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-27 10:55:00 | 239.20 | 2024-12-27 11:20:00 | 238.52 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-12-27 10:55:00 | 239.20 | 2024-12-27 11:25:00 | 239.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-02 09:30:00 | 239.72 | 2025-01-02 09:40:00 | 240.56 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-01-02 09:30:00 | 239.72 | 2025-01-02 15:20:00 | 245.81 | TARGET_HIT | 0.50 | 2.54% |
| BUY | retest1 | 2025-01-15 10:30:00 | 263.45 | 2025-01-15 11:15:00 | 262.63 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-16 10:25:00 | 265.17 | 2025-01-16 10:30:00 | 264.23 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-17 09:30:00 | 265.40 | 2025-01-17 09:35:00 | 264.66 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-30 09:35:00 | 254.98 | 2025-01-30 10:00:00 | 256.10 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-30 09:35:00 | 254.98 | 2025-01-30 13:30:00 | 257.25 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2025-02-20 10:50:00 | 240.00 | 2025-02-20 12:25:00 | 240.95 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-02-20 10:50:00 | 240.00 | 2025-02-20 15:20:00 | 241.85 | TARGET_HIT | 0.50 | 0.77% |
| BUY | retest1 | 2025-02-21 09:35:00 | 242.65 | 2025-02-21 09:40:00 | 242.04 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-03-06 09:30:00 | 227.45 | 2025-03-06 09:35:00 | 228.19 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-07 10:55:00 | 235.24 | 2025-03-07 11:00:00 | 236.31 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-03-07 10:55:00 | 235.24 | 2025-03-07 11:05:00 | 235.24 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-10 11:05:00 | 231.17 | 2025-03-10 11:15:00 | 230.30 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-03-10 11:05:00 | 231.17 | 2025-03-10 15:20:00 | 223.00 | TARGET_HIT | 0.50 | 3.53% |
| BUY | retest1 | 2025-03-19 10:25:00 | 234.19 | 2025-03-19 10:45:00 | 233.66 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-04-16 10:00:00 | 234.01 | 2025-04-16 11:05:00 | 234.88 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-04-16 10:00:00 | 234.01 | 2025-04-16 15:20:00 | 241.66 | TARGET_HIT | 0.50 | 3.27% |
| BUY | retest1 | 2025-04-17 11:15:00 | 243.20 | 2025-04-17 11:40:00 | 244.42 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-17 11:15:00 | 243.20 | 2025-04-17 11:55:00 | 243.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-23 09:50:00 | 250.90 | 2025-04-23 09:55:00 | 250.07 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-04-25 09:30:00 | 247.93 | 2025-04-25 09:35:00 | 248.59 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-04-28 11:10:00 | 249.66 | 2025-04-28 11:55:00 | 250.75 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-04-28 11:10:00 | 249.66 | 2025-04-28 15:20:00 | 250.28 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2025-05-06 10:20:00 | 238.78 | 2025-05-06 10:35:00 | 239.54 | STOP_HIT | 1.00 | -0.32% |
