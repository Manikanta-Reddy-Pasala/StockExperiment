# JIOFIN (JIOFIN)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (12225 bars)
- **Last close:** 249.01
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
| ENTRY1 | 57 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 9 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 48
- **Target hits / Stop hits / Partials:** 9 / 48 / 24
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 8.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 21 | 43.8% | 7 | 27 | 14 | 0.14% | 6.7% |
| BUY @ 2nd Alert (retest1) | 48 | 21 | 43.8% | 7 | 27 | 14 | 0.14% | 6.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 33 | 12 | 36.4% | 2 | 21 | 10 | 0.07% | 2.2% |
| SELL @ 2nd Alert (retest1) | 33 | 12 | 36.4% | 2 | 21 | 10 | 0.07% | 2.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 81 | 33 | 40.7% | 9 | 48 | 24 | 0.11% | 8.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:35:00 | 265.20 | 263.47 | 0.00 | ORB-long ORB[261.50,264.70] vol=1.8x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 09:55:00 | 266.59 | 264.14 | 0.00 | T1 1.5R @ 266.59 |
| Stop hit — per-position SL triggered | 2025-05-13 11:10:00 | 265.20 | 265.42 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 09:45:00 | 266.45 | 267.08 | 0.00 | ORB-short ORB[266.60,268.40] vol=1.9x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-05-15 10:10:00 | 267.22 | 266.98 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-20 09:45:00 | 274.30 | 275.31 | 0.00 | ORB-short ORB[274.65,278.00] vol=1.7x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-20 09:50:00 | 273.08 | 275.01 | 0.00 | T1 1.5R @ 273.08 |
| Stop hit — per-position SL triggered | 2025-05-20 10:35:00 | 274.30 | 274.50 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:45:00 | 278.30 | 276.64 | 0.00 | ORB-long ORB[274.55,276.80] vol=1.8x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 12:10:00 | 279.39 | 277.86 | 0.00 | T1 1.5R @ 279.39 |
| Target hit | 2025-05-23 15:20:00 | 281.85 | 280.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2025-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:30:00 | 284.35 | 283.11 | 0.00 | ORB-long ORB[281.80,283.80] vol=1.7x ATR=0.73 |
| Stop hit — per-position SL triggered | 2025-05-26 09:50:00 | 283.62 | 283.47 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:05:00 | 285.90 | 288.63 | 0.00 | ORB-short ORB[288.20,292.05] vol=1.5x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-05-29 11:20:00 | 286.61 | 288.48 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:05:00 | 293.15 | 291.66 | 0.00 | ORB-long ORB[290.65,292.20] vol=2.5x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-06-06 10:10:00 | 292.34 | 291.72 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 290.10 | 291.98 | 0.00 | ORB-short ORB[291.35,294.05] vol=2.4x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:45:00 | 288.65 | 290.89 | 0.00 | T1 1.5R @ 288.65 |
| Stop hit — per-position SL triggered | 2025-06-16 10:15:00 | 290.10 | 290.39 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:30:00 | 300.85 | 299.44 | 0.00 | ORB-long ORB[297.10,300.45] vol=1.9x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 10:55:00 | 302.64 | 300.42 | 0.00 | T1 1.5R @ 302.64 |
| Target hit | 2025-06-24 13:45:00 | 301.65 | 301.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2025-06-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:40:00 | 320.00 | 315.55 | 0.00 | ORB-long ORB[313.50,316.85] vol=3.9x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 10:45:00 | 321.64 | 317.11 | 0.00 | T1 1.5R @ 321.64 |
| Target hit | 2025-06-27 15:05:00 | 322.75 | 323.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2025-06-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 09:40:00 | 330.35 | 328.64 | 0.00 | ORB-long ORB[325.40,330.00] vol=2.6x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-06-30 09:45:00 | 329.07 | 328.70 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 11:15:00 | 330.95 | 329.12 | 0.00 | ORB-long ORB[327.10,330.35] vol=3.0x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-07-01 11:25:00 | 330.07 | 329.17 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 326.85 | 328.30 | 0.00 | ORB-short ORB[327.50,330.95] vol=1.5x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:15:00 | 325.53 | 327.60 | 0.00 | T1 1.5R @ 325.53 |
| Stop hit — per-position SL triggered | 2025-07-02 10:25:00 | 326.85 | 327.53 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 11:10:00 | 327.35 | 325.72 | 0.00 | ORB-long ORB[323.90,326.50] vol=3.0x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 326.69 | 325.77 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:30:00 | 326.60 | 324.83 | 0.00 | ORB-long ORB[323.05,325.30] vol=2.9x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-07-07 09:35:00 | 325.85 | 324.99 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 11:00:00 | 333.85 | 331.71 | 0.00 | ORB-long ORB[328.80,332.75] vol=2.4x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 11:20:00 | 335.11 | 332.32 | 0.00 | T1 1.5R @ 335.11 |
| Stop hit — per-position SL triggered | 2025-07-09 11:45:00 | 333.85 | 332.56 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:45:00 | 330.85 | 329.72 | 0.00 | ORB-long ORB[328.60,330.00] vol=2.3x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-08-13 10:40:00 | 330.08 | 330.26 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 11:10:00 | 327.10 | 328.18 | 0.00 | ORB-short ORB[327.85,329.40] vol=1.8x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 11:50:00 | 326.34 | 327.79 | 0.00 | T1 1.5R @ 326.34 |
| Target hit | 2025-08-21 15:20:00 | 324.05 | 325.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2025-08-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:40:00 | 326.15 | 324.31 | 0.00 | ORB-long ORB[323.30,325.25] vol=1.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-08-22 09:50:00 | 325.37 | 324.61 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 315.95 | 317.46 | 0.00 | ORB-short ORB[316.70,320.15] vol=2.8x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:35:00 | 314.89 | 317.00 | 0.00 | T1 1.5R @ 314.89 |
| Stop hit — per-position SL triggered | 2025-08-26 09:55:00 | 315.95 | 316.57 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-09-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 10:20:00 | 310.10 | 310.73 | 0.00 | ORB-short ORB[310.30,311.75] vol=1.5x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-09-09 10:35:00 | 310.65 | 310.66 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-09-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 09:55:00 | 310.70 | 311.12 | 0.00 | ORB-short ORB[311.05,312.40] vol=5.4x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 10:35:00 | 310.09 | 310.95 | 0.00 | T1 1.5R @ 310.09 |
| Stop hit — per-position SL triggered | 2025-09-15 11:20:00 | 310.70 | 310.85 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 11:10:00 | 318.65 | 317.47 | 0.00 | ORB-long ORB[315.90,317.75] vol=2.4x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 318.14 | 317.50 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:05:00 | 317.75 | 316.86 | 0.00 | ORB-long ORB[316.10,317.25] vol=1.6x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-09-18 10:10:00 | 317.20 | 316.87 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:15:00 | 310.85 | 312.43 | 0.00 | ORB-short ORB[313.30,315.00] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-09-23 11:25:00 | 311.35 | 312.37 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-10-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 11:00:00 | 295.10 | 294.43 | 0.00 | ORB-long ORB[293.05,294.70] vol=2.9x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 11:30:00 | 296.18 | 294.59 | 0.00 | T1 1.5R @ 296.18 |
| Target hit | 2025-10-01 15:20:00 | 300.40 | 296.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-10-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 09:30:00 | 305.45 | 303.68 | 0.00 | ORB-long ORB[301.50,304.80] vol=1.9x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-10-06 09:40:00 | 304.77 | 304.07 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 309.85 | 308.29 | 0.00 | ORB-long ORB[306.30,308.40] vol=4.6x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-10-07 09:35:00 | 309.09 | 308.46 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-11-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:05:00 | 307.90 | 306.40 | 0.00 | ORB-long ORB[305.25,307.30] vol=2.1x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 10:15:00 | 308.87 | 306.93 | 0.00 | T1 1.5R @ 308.87 |
| Stop hit — per-position SL triggered | 2025-11-12 10:20:00 | 307.90 | 307.10 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-11-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:10:00 | 312.70 | 311.59 | 0.00 | ORB-long ORB[309.70,312.25] vol=1.9x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 10:25:00 | 313.62 | 312.75 | 0.00 | T1 1.5R @ 313.62 |
| Target hit | 2025-11-13 12:15:00 | 314.00 | 314.05 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2025-11-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:25:00 | 314.30 | 312.59 | 0.00 | ORB-long ORB[309.30,312.70] vol=2.7x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 11:10:00 | 315.66 | 313.69 | 0.00 | T1 1.5R @ 315.66 |
| Stop hit — per-position SL triggered | 2025-11-14 11:20:00 | 314.30 | 313.76 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-11-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:35:00 | 308.10 | 306.97 | 0.00 | ORB-long ORB[305.20,307.00] vol=1.6x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 10:50:00 | 308.84 | 307.16 | 0.00 | T1 1.5R @ 308.84 |
| Stop hit — per-position SL triggered | 2025-11-20 10:55:00 | 308.10 | 307.18 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-11-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:25:00 | 305.55 | 306.79 | 0.00 | ORB-short ORB[306.80,308.35] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-11-21 12:20:00 | 306.16 | 306.17 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-11-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 11:00:00 | 302.65 | 303.32 | 0.00 | ORB-short ORB[303.00,304.35] vol=1.6x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-11-24 11:25:00 | 303.17 | 303.27 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-11-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:45:00 | 305.15 | 304.25 | 0.00 | ORB-long ORB[301.50,304.50] vol=2.3x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 11:55:00 | 306.01 | 304.59 | 0.00 | T1 1.5R @ 306.01 |
| Target hit | 2025-11-26 15:20:00 | 308.10 | 306.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2025-11-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:30:00 | 307.70 | 306.63 | 0.00 | ORB-long ORB[305.70,307.10] vol=1.9x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-11-28 10:35:00 | 307.08 | 306.65 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-12-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 09:50:00 | 306.40 | 307.23 | 0.00 | ORB-short ORB[306.70,308.15] vol=1.5x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-12-01 10:30:00 | 306.89 | 306.97 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-12-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:00:00 | 299.65 | 301.73 | 0.00 | ORB-short ORB[302.05,305.70] vol=1.8x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:30:00 | 298.82 | 301.23 | 0.00 | T1 1.5R @ 298.82 |
| Stop hit — per-position SL triggered | 2025-12-03 11:35:00 | 299.65 | 301.20 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-12-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:00:00 | 302.90 | 301.43 | 0.00 | ORB-long ORB[301.20,302.85] vol=2.1x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-12-05 10:05:00 | 302.21 | 301.80 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-12-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 09:35:00 | 296.30 | 297.08 | 0.00 | ORB-short ORB[296.70,298.75] vol=1.6x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-12-16 09:40:00 | 296.92 | 296.99 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-12-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:55:00 | 293.00 | 294.42 | 0.00 | ORB-short ORB[294.65,296.00] vol=1.6x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-12-17 11:05:00 | 293.52 | 294.35 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-12-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:40:00 | 290.30 | 291.72 | 0.00 | ORB-short ORB[291.65,292.90] vol=1.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-12-18 10:05:00 | 290.86 | 291.17 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:30:00 | 297.00 | 295.93 | 0.00 | ORB-long ORB[293.35,296.80] vol=2.0x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-12-19 09:35:00 | 296.24 | 295.98 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-12-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:45:00 | 302.25 | 301.21 | 0.00 | ORB-long ORB[298.65,302.20] vol=1.7x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-12-24 09:55:00 | 301.66 | 301.36 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-01-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:40:00 | 300.45 | 299.09 | 0.00 | ORB-long ORB[295.50,299.70] vol=1.7x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:45:00 | 301.36 | 299.36 | 0.00 | T1 1.5R @ 301.36 |
| Stop hit — per-position SL triggered | 2026-01-02 10:55:00 | 300.45 | 299.46 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2026-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:40:00 | 241.40 | 239.35 | 0.00 | ORB-long ORB[237.50,240.20] vol=1.9x ATR=0.78 |
| Stop hit — per-position SL triggered | 2026-03-11 10:45:00 | 240.62 | 239.45 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:15:00 | 232.80 | 234.09 | 0.00 | ORB-short ORB[233.35,236.50] vol=1.8x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:30:00 | 231.85 | 233.84 | 0.00 | T1 1.5R @ 231.85 |
| Stop hit — per-position SL triggered | 2026-03-27 11:35:00 | 232.80 | 233.81 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 242.17 | 243.16 | 0.00 | ORB-short ORB[242.70,245.00] vol=2.3x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:45:00 | 241.41 | 242.80 | 0.00 | T1 1.5R @ 241.41 |
| Target hit | 2026-04-16 15:20:00 | 241.40 | 241.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:15:00 | 242.69 | 242.39 | 0.00 | ORB-long ORB[240.64,242.23] vol=5.0x ATR=0.55 |
| Stop hit — per-position SL triggered | 2026-04-17 12:00:00 | 242.14 | 242.43 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-04-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:20:00 | 235.00 | 236.05 | 0.00 | ORB-short ORB[235.64,237.60] vol=1.6x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-04-21 10:40:00 | 235.47 | 235.85 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:45:00 | 241.80 | 240.13 | 0.00 | ORB-long ORB[238.54,240.13] vol=4.0x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:50:00 | 242.94 | 241.61 | 0.00 | T1 1.5R @ 242.94 |
| Target hit | 2026-04-23 11:10:00 | 243.40 | 243.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 52 — BUY (started 2026-04-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:55:00 | 250.34 | 248.57 | 0.00 | ORB-long ORB[247.02,249.28] vol=1.7x ATR=0.97 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 249.37 | 248.76 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-04-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:10:00 | 254.64 | 253.03 | 0.00 | ORB-long ORB[251.21,254.46] vol=1.7x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-04-29 10:15:00 | 253.78 | 253.11 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-05-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:10:00 | 251.24 | 249.78 | 0.00 | ORB-long ORB[247.87,251.00] vol=2.9x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:15:00 | 252.27 | 250.03 | 0.00 | T1 1.5R @ 252.27 |
| Stop hit — per-position SL triggered | 2026-05-04 12:10:00 | 251.24 | 250.66 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-05-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:25:00 | 249.00 | 251.11 | 0.00 | ORB-short ORB[250.50,252.34] vol=2.4x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:55:00 | 247.67 | 250.36 | 0.00 | T1 1.5R @ 247.67 |
| Stop hit — per-position SL triggered | 2026-05-05 11:25:00 | 249.00 | 250.06 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:15:00 | 250.54 | 251.32 | 0.00 | ORB-short ORB[250.88,252.25] vol=1.6x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-05-06 12:00:00 | 251.11 | 251.17 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-05-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:55:00 | 251.24 | 252.04 | 0.00 | ORB-short ORB[251.53,254.15] vol=1.6x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-05-07 11:55:00 | 251.86 | 251.86 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 09:35:00 | 265.20 | 2025-05-13 09:55:00 | 266.59 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-05-13 09:35:00 | 265.20 | 2025-05-13 11:10:00 | 265.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-15 09:45:00 | 266.45 | 2025-05-15 10:10:00 | 267.22 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-20 09:45:00 | 274.30 | 2025-05-20 09:50:00 | 273.08 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-05-20 09:45:00 | 274.30 | 2025-05-20 10:35:00 | 274.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-23 10:45:00 | 278.30 | 2025-05-23 12:10:00 | 279.39 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-05-23 10:45:00 | 278.30 | 2025-05-23 15:20:00 | 281.85 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2025-05-26 09:30:00 | 284.35 | 2025-05-26 09:50:00 | 283.62 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-05-29 11:05:00 | 285.90 | 2025-05-29 11:20:00 | 286.61 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-06 10:05:00 | 293.15 | 2025-06-06 10:10:00 | 292.34 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-16 09:30:00 | 290.10 | 2025-06-16 09:45:00 | 288.65 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-06-16 09:30:00 | 290.10 | 2025-06-16 10:15:00 | 290.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-24 09:30:00 | 300.85 | 2025-06-24 10:55:00 | 302.64 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-06-24 09:30:00 | 300.85 | 2025-06-24 13:45:00 | 301.65 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2025-06-27 10:40:00 | 320.00 | 2025-06-27 10:45:00 | 321.64 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-06-27 10:40:00 | 320.00 | 2025-06-27 15:05:00 | 322.75 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2025-06-30 09:40:00 | 330.35 | 2025-06-30 09:45:00 | 329.07 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-07-01 11:15:00 | 330.95 | 2025-07-01 11:25:00 | 330.07 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-02 09:40:00 | 326.85 | 2025-07-02 10:15:00 | 325.53 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-02 09:40:00 | 326.85 | 2025-07-02 10:25:00 | 326.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-04 11:10:00 | 327.35 | 2025-07-04 11:15:00 | 326.69 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-07-07 09:30:00 | 326.60 | 2025-07-07 09:35:00 | 325.85 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-09 11:00:00 | 333.85 | 2025-07-09 11:20:00 | 335.11 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-07-09 11:00:00 | 333.85 | 2025-07-09 11:45:00 | 333.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-13 09:45:00 | 330.85 | 2025-08-13 10:40:00 | 330.08 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-08-21 11:10:00 | 327.10 | 2025-08-21 11:50:00 | 326.34 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-08-21 11:10:00 | 327.10 | 2025-08-21 15:20:00 | 324.05 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2025-08-22 09:40:00 | 326.15 | 2025-08-22 09:50:00 | 325.37 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-26 09:30:00 | 315.95 | 2025-08-26 09:35:00 | 314.89 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-08-26 09:30:00 | 315.95 | 2025-08-26 09:55:00 | 315.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-09 10:20:00 | 310.10 | 2025-09-09 10:35:00 | 310.65 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-15 09:55:00 | 310.70 | 2025-09-15 10:35:00 | 310.09 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-09-15 09:55:00 | 310.70 | 2025-09-15 11:20:00 | 310.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-17 11:10:00 | 318.65 | 2025-09-17 11:15:00 | 318.14 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-09-18 10:05:00 | 317.75 | 2025-09-18 10:10:00 | 317.20 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-23 11:15:00 | 310.85 | 2025-09-23 11:25:00 | 311.35 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-01 11:00:00 | 295.10 | 2025-10-01 11:30:00 | 296.18 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-10-01 11:00:00 | 295.10 | 2025-10-01 15:20:00 | 300.40 | TARGET_HIT | 0.50 | 1.80% |
| BUY | retest1 | 2025-10-06 09:30:00 | 305.45 | 2025-10-06 09:40:00 | 304.77 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-07 09:30:00 | 309.85 | 2025-10-07 09:35:00 | 309.09 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-12 10:05:00 | 307.90 | 2025-11-12 10:15:00 | 308.87 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-11-12 10:05:00 | 307.90 | 2025-11-12 10:20:00 | 307.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 10:10:00 | 312.70 | 2025-11-13 10:25:00 | 313.62 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-11-13 10:10:00 | 312.70 | 2025-11-13 12:15:00 | 314.00 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2025-11-14 10:25:00 | 314.30 | 2025-11-14 11:10:00 | 315.66 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-11-14 10:25:00 | 314.30 | 2025-11-14 11:20:00 | 314.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-20 10:35:00 | 308.10 | 2025-11-20 10:50:00 | 308.84 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-11-20 10:35:00 | 308.10 | 2025-11-20 10:55:00 | 308.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 10:25:00 | 305.55 | 2025-11-21 12:20:00 | 306.16 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-11-24 11:00:00 | 302.65 | 2025-11-24 11:25:00 | 303.17 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-26 10:45:00 | 305.15 | 2025-11-26 11:55:00 | 306.01 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-11-26 10:45:00 | 305.15 | 2025-11-26 15:20:00 | 308.10 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2025-11-28 10:30:00 | 307.70 | 2025-11-28 10:35:00 | 307.08 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-01 09:50:00 | 306.40 | 2025-12-01 10:30:00 | 306.89 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-12-03 11:00:00 | 299.65 | 2025-12-03 11:30:00 | 298.82 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-12-03 11:00:00 | 299.65 | 2025-12-03 11:35:00 | 299.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-05 10:00:00 | 302.90 | 2025-12-05 10:05:00 | 302.21 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-16 09:35:00 | 296.30 | 2025-12-16 09:40:00 | 296.92 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-17 10:55:00 | 293.00 | 2025-12-17 11:05:00 | 293.52 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-18 09:40:00 | 290.30 | 2025-12-18 10:05:00 | 290.86 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-19 09:30:00 | 297.00 | 2025-12-19 09:35:00 | 296.24 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-24 09:45:00 | 302.25 | 2025-12-24 09:55:00 | 301.66 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-01-02 10:40:00 | 300.45 | 2026-01-02 10:45:00 | 301.36 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-01-02 10:40:00 | 300.45 | 2026-01-02 10:55:00 | 300.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 10:40:00 | 241.40 | 2026-03-11 10:45:00 | 240.62 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-27 11:15:00 | 232.80 | 2026-03-27 11:30:00 | 231.85 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-27 11:15:00 | 232.80 | 2026-03-27 11:35:00 | 232.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 11:00:00 | 242.17 | 2026-04-16 11:45:00 | 241.41 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-04-16 11:00:00 | 242.17 | 2026-04-16 15:20:00 | 241.40 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-04-17 11:15:00 | 242.69 | 2026-04-17 12:00:00 | 242.14 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-04-21 10:20:00 | 235.00 | 2026-04-21 10:40:00 | 235.47 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-04-23 09:45:00 | 241.80 | 2026-04-23 09:50:00 | 242.94 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-23 09:45:00 | 241.80 | 2026-04-23 11:10:00 | 243.40 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-27 09:55:00 | 250.34 | 2026-04-27 10:05:00 | 249.37 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-29 10:10:00 | 254.64 | 2026-04-29 10:15:00 | 253.78 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-04 11:10:00 | 251.24 | 2026-05-04 11:15:00 | 252.27 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-05-04 11:10:00 | 251.24 | 2026-05-04 12:10:00 | 251.24 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:25:00 | 249.00 | 2026-05-05 10:55:00 | 247.67 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-05-05 10:25:00 | 249.00 | 2026-05-05 11:25:00 | 249.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 11:15:00 | 250.54 | 2026-05-06 12:00:00 | 251.11 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-07 10:55:00 | 251.24 | 2026-05-07 11:55:00 | 251.86 | STOP_HIT | 1.00 | -0.25% |
