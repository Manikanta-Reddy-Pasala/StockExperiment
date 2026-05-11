# Sapphire Foods India Ltd. (SAPPHIRE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 183.60
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
| ENTRY1 | 60 |
| ENTRY2 | 0 |
| PARTIAL | 21 |
| TARGET_HIT | 7 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 53
- **Target hits / Stop hits / Partials:** 7 / 53 / 21
- **Avg / median % per leg:** 0.03% / 0.00%
- **Sum % (uncompounded):** 2.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 19 | 41.3% | 6 | 27 | 13 | 0.08% | 3.6% |
| BUY @ 2nd Alert (retest1) | 46 | 19 | 41.3% | 6 | 27 | 13 | 0.08% | 3.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 35 | 9 | 25.7% | 1 | 26 | 8 | -0.05% | -1.6% |
| SELL @ 2nd Alert (retest1) | 35 | 9 | 25.7% | 1 | 26 | 8 | -0.05% | -1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 81 | 28 | 34.6% | 7 | 53 | 21 | 0.03% | 2.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:30:00 | 273.00 | 275.32 | 0.00 | ORB-short ORB[275.30,277.61] vol=2.0x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:40:00 | 271.86 | 274.28 | 0.00 | T1 1.5R @ 271.86 |
| Stop hit — per-position SL triggered | 2024-05-15 10:45:00 | 273.00 | 274.14 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:10:00 | 274.64 | 275.90 | 0.00 | ORB-short ORB[275.11,278.00] vol=1.9x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-05-16 11:35:00 | 275.23 | 275.72 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:40:00 | 275.52 | 277.12 | 0.00 | ORB-short ORB[276.61,280.46] vol=3.3x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-05-17 11:15:00 | 276.16 | 276.88 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:30:00 | 279.80 | 278.87 | 0.00 | ORB-long ORB[276.76,278.98] vol=8.5x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 10:50:00 | 281.13 | 279.85 | 0.00 | T1 1.5R @ 281.13 |
| Target hit | 2024-05-21 15:20:00 | 283.85 | 281.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-05-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 09:50:00 | 284.34 | 285.39 | 0.00 | ORB-short ORB[285.22,287.00] vol=2.1x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-05-23 10:10:00 | 285.04 | 285.18 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 281.57 | 281.50 | 0.00 | ORB-long ORB[279.65,281.49] vol=3.1x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-05-28 09:40:00 | 280.50 | 281.15 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-05-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:10:00 | 288.00 | 285.63 | 0.00 | ORB-long ORB[280.29,282.70] vol=3.0x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-05-29 10:15:00 | 286.88 | 285.85 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-05-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-30 10:40:00 | 284.49 | 283.04 | 0.00 | ORB-long ORB[281.73,283.80] vol=3.3x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-05-30 11:00:00 | 283.87 | 283.25 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-05-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:05:00 | 283.20 | 283.33 | 0.00 | ORB-short ORB[283.65,285.99] vol=1.7x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-05-31 10:15:00 | 283.99 | 283.35 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-03 10:55:00 | 286.00 | 289.02 | 0.00 | ORB-short ORB[290.21,293.58] vol=2.0x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-06-03 11:40:00 | 287.15 | 288.70 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:45:00 | 296.08 | 293.35 | 0.00 | ORB-long ORB[289.65,293.86] vol=2.5x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 11:45:00 | 297.89 | 294.63 | 0.00 | T1 1.5R @ 297.89 |
| Stop hit — per-position SL triggered | 2024-06-07 12:25:00 | 296.08 | 296.55 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 11:10:00 | 300.95 | 300.30 | 0.00 | ORB-long ORB[297.96,300.00] vol=1.7x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 14:40:00 | 302.01 | 300.80 | 0.00 | T1 1.5R @ 302.01 |
| Stop hit — per-position SL triggered | 2024-06-11 15:00:00 | 300.95 | 300.93 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-06-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 09:30:00 | 296.44 | 297.41 | 0.00 | ORB-short ORB[299.78,303.00] vol=8.8x ATR=1.19 |
| Stop hit — per-position SL triggered | 2024-06-12 09:40:00 | 297.63 | 297.12 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 11:15:00 | 301.48 | 301.06 | 0.00 | ORB-long ORB[298.85,301.19] vol=4.6x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:35:00 | 302.50 | 301.18 | 0.00 | T1 1.5R @ 302.50 |
| Stop hit — per-position SL triggered | 2024-06-13 11:50:00 | 301.48 | 301.24 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-06-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:55:00 | 307.98 | 309.73 | 0.00 | ORB-short ORB[308.20,312.80] vol=1.5x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 11:05:00 | 306.59 | 309.12 | 0.00 | T1 1.5R @ 306.59 |
| Stop hit — per-position SL triggered | 2024-06-27 11:20:00 | 307.98 | 308.74 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-06-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 10:05:00 | 311.60 | 310.55 | 0.00 | ORB-long ORB[309.14,310.99] vol=2.7x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 10:10:00 | 313.49 | 311.01 | 0.00 | T1 1.5R @ 313.49 |
| Target hit | 2024-06-28 12:05:00 | 312.99 | 313.27 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — SELL (started 2024-07-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:55:00 | 311.80 | 313.37 | 0.00 | ORB-short ORB[313.91,316.38] vol=1.7x ATR=0.97 |
| Stop hit — per-position SL triggered | 2024-07-02 10:10:00 | 312.77 | 313.09 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:45:00 | 313.66 | 312.79 | 0.00 | ORB-long ORB[310.02,313.56] vol=2.4x ATR=1.05 |
| Stop hit — per-position SL triggered | 2024-07-03 09:55:00 | 312.61 | 312.87 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:05:00 | 315.93 | 316.23 | 0.00 | ORB-short ORB[316.00,319.19] vol=5.6x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-07-08 10:10:00 | 317.01 | 316.25 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:30:00 | 320.56 | 319.47 | 0.00 | ORB-long ORB[312.77,317.58] vol=1.6x ATR=1.31 |
| Stop hit — per-position SL triggered | 2024-07-09 10:40:00 | 319.25 | 319.48 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:15:00 | 305.99 | 308.97 | 0.00 | ORB-short ORB[311.69,315.13] vol=2.0x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 304.08 | 307.98 | 0.00 | T1 1.5R @ 304.08 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 305.99 | 307.26 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:50:00 | 312.96 | 312.61 | 0.00 | ORB-long ORB[309.06,311.98] vol=14.7x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-07-11 10:05:00 | 312.07 | 312.62 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 10:50:00 | 306.40 | 307.96 | 0.00 | ORB-short ORB[308.01,311.40] vol=2.1x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-07-18 10:55:00 | 307.28 | 307.95 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 11:05:00 | 304.19 | 301.19 | 0.00 | ORB-long ORB[298.09,302.11] vol=1.9x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 12:05:00 | 305.60 | 302.82 | 0.00 | T1 1.5R @ 305.60 |
| Stop hit — per-position SL triggered | 2024-07-22 12:25:00 | 304.19 | 302.91 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-07-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 10:45:00 | 314.40 | 316.62 | 0.00 | ORB-short ORB[315.29,318.40] vol=3.3x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-07-30 10:55:00 | 315.50 | 316.54 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-07-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:45:00 | 321.52 | 320.16 | 0.00 | ORB-long ORB[316.80,320.98] vol=1.9x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-07-31 11:05:00 | 320.37 | 320.19 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:00:00 | 341.02 | 337.70 | 0.00 | ORB-long ORB[332.83,335.65] vol=3.7x ATR=2.44 |
| Stop hit — per-position SL triggered | 2024-08-07 10:15:00 | 338.58 | 338.27 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 306.90 | 309.23 | 0.00 | ORB-short ORB[307.17,311.40] vol=1.8x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:40:00 | 305.16 | 308.68 | 0.00 | T1 1.5R @ 305.16 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 306.90 | 308.55 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 311.40 | 312.59 | 0.00 | ORB-short ORB[312.20,315.68] vol=1.6x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-08-19 09:45:00 | 312.35 | 312.31 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:55:00 | 317.20 | 314.93 | 0.00 | ORB-long ORB[311.60,315.98] vol=1.6x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:00:00 | 319.02 | 315.47 | 0.00 | T1 1.5R @ 319.02 |
| Target hit | 2024-08-21 13:10:00 | 317.92 | 318.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2024-08-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 10:40:00 | 324.19 | 325.20 | 0.00 | ORB-short ORB[324.21,327.98] vol=1.7x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 12:30:00 | 323.27 | 324.82 | 0.00 | T1 1.5R @ 323.27 |
| Stop hit — per-position SL triggered | 2024-08-23 14:05:00 | 324.19 | 324.62 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 11:00:00 | 326.99 | 326.18 | 0.00 | ORB-long ORB[323.47,326.55] vol=6.8x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-08-27 11:05:00 | 326.10 | 326.16 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-08-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:35:00 | 322.67 | 324.25 | 0.00 | ORB-short ORB[324.42,326.79] vol=2.5x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 14:10:00 | 321.17 | 323.02 | 0.00 | T1 1.5R @ 321.17 |
| Target hit | 2024-08-28 15:20:00 | 320.50 | 322.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2024-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:35:00 | 318.65 | 319.72 | 0.00 | ORB-short ORB[318.74,322.40] vol=2.4x ATR=0.85 |
| Stop hit — per-position SL triggered | 2024-08-29 09:50:00 | 319.50 | 319.61 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 11:05:00 | 335.00 | 336.17 | 0.00 | ORB-short ORB[336.10,340.00] vol=3.2x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-09-11 12:30:00 | 336.10 | 335.86 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 10:55:00 | 329.30 | 333.08 | 0.00 | ORB-short ORB[333.45,337.75] vol=3.4x ATR=1.02 |
| Stop hit — per-position SL triggered | 2024-09-12 14:00:00 | 330.32 | 331.33 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 11:00:00 | 328.90 | 331.14 | 0.00 | ORB-short ORB[328.95,332.35] vol=8.7x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 11:35:00 | 327.33 | 330.50 | 0.00 | T1 1.5R @ 327.33 |
| Stop hit — per-position SL triggered | 2024-09-13 11:45:00 | 328.90 | 330.39 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:00:00 | 326.90 | 329.59 | 0.00 | ORB-short ORB[329.30,333.05] vol=1.7x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 10:30:00 | 325.22 | 328.66 | 0.00 | T1 1.5R @ 325.22 |
| Stop hit — per-position SL triggered | 2024-09-16 12:05:00 | 326.90 | 326.97 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-09-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:45:00 | 338.40 | 339.22 | 0.00 | ORB-short ORB[341.10,345.00] vol=6.5x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-09-19 11:00:00 | 339.94 | 339.22 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 09:30:00 | 343.55 | 340.53 | 0.00 | ORB-long ORB[337.10,340.75] vol=1.6x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-09-20 09:40:00 | 341.84 | 341.65 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-09-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:05:00 | 360.55 | 358.68 | 0.00 | ORB-long ORB[348.00,353.25] vol=16.1x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-09-24 10:10:00 | 358.36 | 358.69 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 10:55:00 | 360.15 | 357.57 | 0.00 | ORB-long ORB[352.30,357.00] vol=3.5x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-10-01 11:00:00 | 358.45 | 357.71 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:35:00 | 362.15 | 360.44 | 0.00 | ORB-long ORB[357.45,361.00] vol=2.3x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-10-11 09:40:00 | 360.63 | 360.40 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 11:05:00 | 345.00 | 345.63 | 0.00 | ORB-short ORB[345.20,350.30] vol=4.9x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-10-21 11:10:00 | 345.96 | 345.63 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-11-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-05 10:35:00 | 315.00 | 313.41 | 0.00 | ORB-long ORB[310.55,314.90] vol=2.4x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 10:50:00 | 316.72 | 313.93 | 0.00 | T1 1.5R @ 316.72 |
| Stop hit — per-position SL triggered | 2024-11-05 11:00:00 | 315.00 | 314.02 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 10:55:00 | 314.30 | 311.28 | 0.00 | ORB-long ORB[308.60,311.90] vol=3.9x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 11:10:00 | 316.30 | 312.52 | 0.00 | T1 1.5R @ 316.30 |
| Target hit | 2024-11-26 15:20:00 | 316.00 | 315.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2024-11-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:40:00 | 327.80 | 327.22 | 0.00 | ORB-long ORB[323.85,326.65] vol=13.0x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 09:45:00 | 329.64 | 327.62 | 0.00 | T1 1.5R @ 329.64 |
| Stop hit — per-position SL triggered | 2024-11-29 10:10:00 | 327.80 | 329.28 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:35:00 | 328.85 | 326.46 | 0.00 | ORB-long ORB[322.55,326.25] vol=2.3x ATR=1.35 |
| Stop hit — per-position SL triggered | 2024-12-03 11:20:00 | 327.50 | 328.58 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:00:00 | 347.40 | 344.67 | 0.00 | ORB-long ORB[340.00,345.00] vol=3.8x ATR=1.05 |
| Stop hit — per-position SL triggered | 2024-12-06 11:15:00 | 346.35 | 345.19 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:55:00 | 323.60 | 324.33 | 0.00 | ORB-short ORB[324.70,327.95] vol=3.1x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-12-26 11:10:00 | 324.61 | 324.36 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 11:10:00 | 326.95 | 324.92 | 0.00 | ORB-long ORB[321.95,326.40] vol=2.3x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-12-27 11:20:00 | 325.77 | 325.10 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 10:50:00 | 317.55 | 320.04 | 0.00 | ORB-short ORB[319.80,324.00] vol=1.7x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-01-17 11:10:00 | 318.45 | 319.82 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:30:00 | 312.35 | 315.37 | 0.00 | ORB-short ORB[315.20,319.80] vol=2.5x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-01-20 10:15:00 | 313.75 | 312.98 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-01-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 10:10:00 | 314.50 | 311.90 | 0.00 | ORB-long ORB[310.00,313.80] vol=2.0x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 10:15:00 | 316.17 | 313.13 | 0.00 | T1 1.5R @ 316.17 |
| Target hit | 2025-01-22 10:30:00 | 315.35 | 315.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 55 — BUY (started 2025-01-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:55:00 | 293.05 | 291.37 | 0.00 | ORB-long ORB[284.65,289.00] vol=4.2x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:05:00 | 295.42 | 291.52 | 0.00 | T1 1.5R @ 295.42 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 293.05 | 292.46 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-01-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:20:00 | 291.25 | 289.32 | 0.00 | ORB-long ORB[286.40,289.95] vol=3.1x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-01-30 10:50:00 | 290.10 | 289.48 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-03-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:25:00 | 304.00 | 301.26 | 0.00 | ORB-long ORB[295.10,299.30] vol=4.7x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-03-18 10:30:00 | 302.25 | 301.41 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-04-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:45:00 | 299.55 | 297.98 | 0.00 | ORB-long ORB[294.55,299.00] vol=1.6x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-11 10:05:00 | 301.89 | 298.76 | 0.00 | T1 1.5R @ 301.89 |
| Target hit | 2025-04-11 15:20:00 | 304.30 | 302.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2025-04-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 11:00:00 | 315.10 | 310.88 | 0.00 | ORB-long ORB[308.05,312.40] vol=4.2x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-04-15 11:05:00 | 313.89 | 310.97 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-05-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 10:25:00 | 317.75 | 315.96 | 0.00 | ORB-long ORB[313.20,317.00] vol=1.9x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-05-06 10:35:00 | 316.53 | 316.44 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 10:30:00 | 273.00 | 2024-05-15 10:40:00 | 271.86 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-05-15 10:30:00 | 273.00 | 2024-05-15 10:45:00 | 273.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 11:10:00 | 274.64 | 2024-05-16 11:35:00 | 275.23 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-05-17 10:40:00 | 275.52 | 2024-05-17 11:15:00 | 276.16 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-05-21 10:30:00 | 279.80 | 2024-05-21 10:50:00 | 281.13 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-05-21 10:30:00 | 279.80 | 2024-05-21 15:20:00 | 283.85 | TARGET_HIT | 0.50 | 1.45% |
| SELL | retest1 | 2024-05-23 09:50:00 | 284.34 | 2024-05-23 10:10:00 | 285.04 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-28 09:35:00 | 281.57 | 2024-05-28 09:40:00 | 280.50 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-05-29 10:10:00 | 288.00 | 2024-05-29 10:15:00 | 286.88 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-30 10:40:00 | 284.49 | 2024-05-30 11:00:00 | 283.87 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-05-31 10:05:00 | 283.20 | 2024-05-31 10:15:00 | 283.99 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-06-03 10:55:00 | 286.00 | 2024-06-03 11:40:00 | 287.15 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-07 10:45:00 | 296.08 | 2024-06-07 11:45:00 | 297.89 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-06-07 10:45:00 | 296.08 | 2024-06-07 12:25:00 | 296.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-11 11:10:00 | 300.95 | 2024-06-11 14:40:00 | 302.01 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-06-11 11:10:00 | 300.95 | 2024-06-11 15:00:00 | 300.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-12 09:30:00 | 296.44 | 2024-06-12 09:40:00 | 297.63 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-13 11:15:00 | 301.48 | 2024-06-13 11:35:00 | 302.50 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-06-13 11:15:00 | 301.48 | 2024-06-13 11:50:00 | 301.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-27 10:55:00 | 307.98 | 2024-06-27 11:05:00 | 306.59 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-06-27 10:55:00 | 307.98 | 2024-06-27 11:20:00 | 307.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-28 10:05:00 | 311.60 | 2024-06-28 10:10:00 | 313.49 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-06-28 10:05:00 | 311.60 | 2024-06-28 12:05:00 | 312.99 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-02 09:55:00 | 311.80 | 2024-07-02 10:10:00 | 312.77 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-03 09:45:00 | 313.66 | 2024-07-03 09:55:00 | 312.61 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-08 10:05:00 | 315.93 | 2024-07-08 10:10:00 | 317.01 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-09 10:30:00 | 320.56 | 2024-07-09 10:40:00 | 319.25 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-07-10 10:15:00 | 305.99 | 2024-07-10 10:20:00 | 304.08 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-07-10 10:15:00 | 305.99 | 2024-07-10 10:55:00 | 305.99 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-11 09:50:00 | 312.96 | 2024-07-11 10:05:00 | 312.07 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-18 10:50:00 | 306.40 | 2024-07-18 10:55:00 | 307.28 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-22 11:05:00 | 304.19 | 2024-07-22 12:05:00 | 305.60 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-22 11:05:00 | 304.19 | 2024-07-22 12:25:00 | 304.19 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-30 10:45:00 | 314.40 | 2024-07-30 10:55:00 | 315.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-31 10:45:00 | 321.52 | 2024-07-31 11:05:00 | 320.37 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-07 10:00:00 | 341.02 | 2024-08-07 10:15:00 | 338.58 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest1 | 2024-08-14 09:30:00 | 306.90 | 2024-08-14 09:40:00 | 305.16 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-08-14 09:30:00 | 306.90 | 2024-08-14 09:45:00 | 306.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-19 09:35:00 | 311.40 | 2024-08-19 09:45:00 | 312.35 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-21 09:55:00 | 317.20 | 2024-08-21 10:00:00 | 319.02 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-08-21 09:55:00 | 317.20 | 2024-08-21 13:10:00 | 317.92 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2024-08-23 10:40:00 | 324.19 | 2024-08-23 12:30:00 | 323.27 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-08-23 10:40:00 | 324.19 | 2024-08-23 14:05:00 | 324.19 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-27 11:00:00 | 326.99 | 2024-08-27 11:05:00 | 326.10 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-28 09:35:00 | 322.67 | 2024-08-28 14:10:00 | 321.17 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-08-28 09:35:00 | 322.67 | 2024-08-28 15:20:00 | 320.50 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2024-08-29 09:35:00 | 318.65 | 2024-08-29 09:50:00 | 319.50 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-11 11:05:00 | 335.00 | 2024-09-11 12:30:00 | 336.10 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-12 10:55:00 | 329.30 | 2024-09-12 14:00:00 | 330.32 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-13 11:00:00 | 328.90 | 2024-09-13 11:35:00 | 327.33 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-09-13 11:00:00 | 328.90 | 2024-09-13 11:45:00 | 328.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-16 10:00:00 | 326.90 | 2024-09-16 10:30:00 | 325.22 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-16 10:00:00 | 326.90 | 2024-09-16 12:05:00 | 326.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 10:45:00 | 338.40 | 2024-09-19 11:00:00 | 339.94 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-09-20 09:30:00 | 343.55 | 2024-09-20 09:40:00 | 341.84 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-09-24 10:05:00 | 360.55 | 2024-09-24 10:10:00 | 358.36 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2024-10-01 10:55:00 | 360.15 | 2024-10-01 11:00:00 | 358.45 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-10-11 09:35:00 | 362.15 | 2024-10-11 09:40:00 | 360.63 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-10-21 11:05:00 | 345.00 | 2024-10-21 11:10:00 | 345.96 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-11-05 10:35:00 | 315.00 | 2024-11-05 10:50:00 | 316.72 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-11-05 10:35:00 | 315.00 | 2024-11-05 11:00:00 | 315.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-26 10:55:00 | 314.30 | 2024-11-26 11:10:00 | 316.30 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-11-26 10:55:00 | 314.30 | 2024-11-26 15:20:00 | 316.00 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2024-11-29 09:40:00 | 327.80 | 2024-11-29 09:45:00 | 329.64 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-11-29 09:40:00 | 327.80 | 2024-11-29 10:10:00 | 327.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-03 09:35:00 | 328.85 | 2024-12-03 11:20:00 | 327.50 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-12-06 11:00:00 | 347.40 | 2024-12-06 11:15:00 | 346.35 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-26 10:55:00 | 323.60 | 2024-12-26 11:10:00 | 324.61 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-27 11:10:00 | 326.95 | 2024-12-27 11:20:00 | 325.77 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-01-17 10:50:00 | 317.55 | 2025-01-17 11:10:00 | 318.45 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-20 09:30:00 | 312.35 | 2025-01-20 10:15:00 | 313.75 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-01-22 10:10:00 | 314.50 | 2025-01-22 10:15:00 | 316.17 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-01-22 10:10:00 | 314.50 | 2025-01-22 10:30:00 | 315.35 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2025-01-29 09:55:00 | 293.05 | 2025-01-29 10:05:00 | 295.42 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2025-01-29 09:55:00 | 293.05 | 2025-01-29 11:20:00 | 293.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-30 10:20:00 | 291.25 | 2025-01-30 10:50:00 | 290.10 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-18 10:25:00 | 304.00 | 2025-03-18 10:30:00 | 302.25 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2025-04-11 09:45:00 | 299.55 | 2025-04-11 10:05:00 | 301.89 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2025-04-11 09:45:00 | 299.55 | 2025-04-11 15:20:00 | 304.30 | TARGET_HIT | 0.50 | 1.59% |
| BUY | retest1 | 2025-04-15 11:00:00 | 315.10 | 2025-04-15 11:05:00 | 313.89 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-05-06 10:25:00 | 317.75 | 2025-05-06 10:35:00 | 316.53 | STOP_HIT | 1.00 | -0.39% |
