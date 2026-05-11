# Chambal Fertilizers & Chemicals Ltd. (CHAMBLFERT)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55355 bars)
- **Last close:** 455.85
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
| ENTRY1 | 97 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 20 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 59 / 77
- **Target hits / Stop hits / Partials:** 20 / 77 / 39
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 21.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 25 | 37.9% | 7 | 41 | 18 | 0.08% | 5.1% |
| BUY @ 2nd Alert (retest1) | 66 | 25 | 37.9% | 7 | 41 | 18 | 0.08% | 5.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 70 | 34 | 48.6% | 13 | 36 | 21 | 0.23% | 16.2% |
| SELL @ 2nd Alert (retest1) | 70 | 34 | 48.6% | 13 | 36 | 21 | 0.23% | 16.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 136 | 59 | 43.4% | 20 | 77 | 39 | 0.16% | 21.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 09:30:00 | 294.35 | 293.83 | 0.00 | ORB-long ORB[291.05,293.90] vol=5.6x ATR=0.80 |
| Stop hit — per-position SL triggered | 2023-05-16 09:35:00 | 293.55 | 293.74 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 09:30:00 | 286.25 | 286.92 | 0.00 | ORB-short ORB[286.65,288.70] vol=4.0x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 09:35:00 | 285.06 | 286.30 | 0.00 | T1 1.5R @ 285.06 |
| Target hit | 2023-05-25 10:10:00 | 285.90 | 285.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2023-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 11:15:00 | 280.35 | 281.23 | 0.00 | ORB-short ORB[280.50,281.90] vol=2.3x ATR=0.39 |
| Stop hit — per-position SL triggered | 2023-06-02 11:20:00 | 280.74 | 281.20 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 11:15:00 | 283.20 | 281.96 | 0.00 | ORB-long ORB[280.90,282.25] vol=5.8x ATR=0.58 |
| Stop hit — per-position SL triggered | 2023-06-05 11:20:00 | 282.62 | 281.99 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 11:15:00 | 281.35 | 280.71 | 0.00 | ORB-long ORB[278.80,280.75] vol=2.0x ATR=0.48 |
| Stop hit — per-position SL triggered | 2023-06-07 12:00:00 | 280.87 | 280.86 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 10:25:00 | 278.90 | 276.35 | 0.00 | ORB-long ORB[273.55,275.45] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2023-06-12 10:30:00 | 278.04 | 276.44 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 10:35:00 | 280.25 | 278.71 | 0.00 | ORB-long ORB[277.00,279.50] vol=2.1x ATR=0.68 |
| Stop hit — per-position SL triggered | 2023-06-13 12:25:00 | 279.57 | 279.15 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 10:55:00 | 282.00 | 280.99 | 0.00 | ORB-long ORB[279.95,281.20] vol=3.3x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-14 11:10:00 | 283.08 | 281.24 | 0.00 | T1 1.5R @ 283.08 |
| Stop hit — per-position SL triggered | 2023-06-14 11:30:00 | 282.00 | 281.55 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 11:05:00 | 279.55 | 278.83 | 0.00 | ORB-long ORB[277.25,279.00] vol=1.7x ATR=0.54 |
| Stop hit — per-position SL triggered | 2023-06-16 11:15:00 | 279.01 | 278.85 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 10:20:00 | 279.85 | 278.62 | 0.00 | ORB-long ORB[277.20,279.00] vol=3.4x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-06-20 10:30:00 | 279.19 | 278.79 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:15:00 | 279.00 | 279.78 | 0.00 | ORB-short ORB[279.15,280.45] vol=1.8x ATR=0.42 |
| Stop hit — per-position SL triggered | 2023-06-21 11:20:00 | 279.42 | 279.77 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-23 09:35:00 | 270.00 | 271.32 | 0.00 | ORB-short ORB[270.55,273.95] vol=2.0x ATR=0.83 |
| Stop hit — per-position SL triggered | 2023-06-23 09:50:00 | 270.83 | 271.11 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 11:05:00 | 270.75 | 268.60 | 0.00 | ORB-long ORB[266.20,269.70] vol=1.6x ATR=0.89 |
| Stop hit — per-position SL triggered | 2023-06-26 11:55:00 | 269.86 | 268.92 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 09:35:00 | 276.10 | 274.86 | 0.00 | ORB-long ORB[273.15,275.65] vol=2.5x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-27 09:40:00 | 277.31 | 275.71 | 0.00 | T1 1.5R @ 277.31 |
| Stop hit — per-position SL triggered | 2023-06-27 10:35:00 | 276.10 | 276.05 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 10:00:00 | 272.95 | 274.36 | 0.00 | ORB-short ORB[274.20,276.35] vol=3.1x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-07-04 10:05:00 | 273.61 | 274.33 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 10:45:00 | 277.55 | 276.34 | 0.00 | ORB-long ORB[273.25,277.20] vol=3.0x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 11:40:00 | 278.71 | 276.79 | 0.00 | T1 1.5R @ 278.71 |
| Stop hit — per-position SL triggered | 2023-07-06 12:05:00 | 277.55 | 276.85 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:40:00 | 274.65 | 277.12 | 0.00 | ORB-short ORB[277.10,279.30] vol=1.9x ATR=0.68 |
| Stop hit — per-position SL triggered | 2023-07-07 11:10:00 | 275.33 | 276.74 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 11:00:00 | 272.90 | 273.89 | 0.00 | ORB-short ORB[273.00,275.25] vol=1.6x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 11:30:00 | 272.02 | 273.64 | 0.00 | T1 1.5R @ 272.02 |
| Stop hit — per-position SL triggered | 2023-07-10 13:00:00 | 272.90 | 273.26 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-11 11:15:00 | 273.00 | 274.31 | 0.00 | ORB-short ORB[273.25,274.75] vol=7.3x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 13:50:00 | 272.18 | 273.72 | 0.00 | T1 1.5R @ 272.18 |
| Target hit | 2023-07-11 15:20:00 | 272.40 | 273.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2023-07-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 10:45:00 | 271.95 | 272.89 | 0.00 | ORB-short ORB[272.30,273.80] vol=2.2x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 11:30:00 | 271.23 | 272.66 | 0.00 | T1 1.5R @ 271.23 |
| Target hit | 2023-07-12 15:20:00 | 271.00 | 271.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2023-07-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 10:55:00 | 270.75 | 272.51 | 0.00 | ORB-short ORB[271.35,273.90] vol=2.9x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 12:25:00 | 269.86 | 271.52 | 0.00 | T1 1.5R @ 269.86 |
| Target hit | 2023-07-13 15:20:00 | 265.40 | 268.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2023-07-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:40:00 | 269.00 | 268.26 | 0.00 | ORB-long ORB[266.05,268.40] vol=3.5x ATR=0.78 |
| Stop hit — per-position SL triggered | 2023-07-17 09:50:00 | 268.22 | 268.29 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-07-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 10:40:00 | 267.25 | 268.71 | 0.00 | ORB-short ORB[267.55,270.40] vol=1.5x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 11:10:00 | 266.39 | 268.41 | 0.00 | T1 1.5R @ 266.39 |
| Target hit | 2023-07-18 15:20:00 | 265.50 | 266.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2023-07-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:35:00 | 272.00 | 271.15 | 0.00 | ORB-long ORB[269.50,271.70] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2023-07-25 10:00:00 | 271.20 | 271.31 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-07-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:05:00 | 272.50 | 271.30 | 0.00 | ORB-long ORB[269.30,271.55] vol=2.2x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 10:10:00 | 273.42 | 272.25 | 0.00 | T1 1.5R @ 273.42 |
| Target hit | 2023-07-26 10:40:00 | 274.50 | 274.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — SELL (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 11:15:00 | 273.40 | 275.78 | 0.00 | ORB-short ORB[275.90,278.80] vol=1.9x ATR=0.61 |
| Stop hit — per-position SL triggered | 2023-07-27 11:45:00 | 274.01 | 275.66 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-07-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 09:50:00 | 275.30 | 273.93 | 0.00 | ORB-long ORB[272.15,275.00] vol=2.0x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 10:50:00 | 276.79 | 274.38 | 0.00 | T1 1.5R @ 276.79 |
| Stop hit — per-position SL triggered | 2023-07-28 13:40:00 | 275.30 | 275.59 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 09:40:00 | 277.45 | 276.36 | 0.00 | ORB-long ORB[275.15,277.15] vol=1.6x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 09:45:00 | 278.48 | 278.19 | 0.00 | T1 1.5R @ 278.48 |
| Target hit | 2023-07-31 10:50:00 | 278.95 | 279.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2023-08-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 09:40:00 | 271.30 | 273.27 | 0.00 | ORB-short ORB[271.90,275.35] vol=1.6x ATR=1.04 |
| Stop hit — per-position SL triggered | 2023-08-03 09:45:00 | 272.34 | 273.11 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 10:50:00 | 270.65 | 272.31 | 0.00 | ORB-short ORB[272.00,274.35] vol=1.6x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-08-07 11:25:00 | 271.31 | 272.10 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 10:15:00 | 275.55 | 274.04 | 0.00 | ORB-long ORB[272.40,275.15] vol=2.3x ATR=0.74 |
| Stop hit — per-position SL triggered | 2023-08-08 10:20:00 | 274.81 | 274.35 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:20:00 | 269.70 | 271.55 | 0.00 | ORB-short ORB[270.55,274.60] vol=2.0x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 10:30:00 | 268.31 | 270.94 | 0.00 | T1 1.5R @ 268.31 |
| Stop hit — per-position SL triggered | 2023-08-10 10:40:00 | 269.70 | 270.82 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-08-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 10:20:00 | 263.45 | 261.80 | 0.00 | ORB-long ORB[259.90,261.75] vol=2.2x ATR=0.64 |
| Stop hit — per-position SL triggered | 2023-08-22 10:45:00 | 262.81 | 262.27 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-08-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:35:00 | 269.10 | 270.75 | 0.00 | ORB-short ORB[270.00,272.95] vol=2.1x ATR=0.96 |
| Stop hit — per-position SL triggered | 2023-08-25 11:15:00 | 270.06 | 270.54 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-08-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 09:40:00 | 273.40 | 272.03 | 0.00 | ORB-long ORB[269.00,272.85] vol=3.6x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-28 10:30:00 | 275.17 | 273.64 | 0.00 | T1 1.5R @ 275.17 |
| Target hit | 2023-08-28 11:45:00 | 274.85 | 274.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — BUY (started 2023-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:30:00 | 282.35 | 280.90 | 0.00 | ORB-long ORB[278.95,281.65] vol=2.6x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 09:40:00 | 283.64 | 282.04 | 0.00 | T1 1.5R @ 283.64 |
| Target hit | 2023-08-30 11:05:00 | 283.25 | 283.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2023-08-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 11:05:00 | 277.15 | 278.70 | 0.00 | ORB-short ORB[278.30,282.00] vol=1.6x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-08-31 11:20:00 | 277.81 | 278.64 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 09:35:00 | 282.15 | 281.16 | 0.00 | ORB-long ORB[279.45,282.10] vol=2.0x ATR=0.78 |
| Stop hit — per-position SL triggered | 2023-09-05 10:20:00 | 281.37 | 281.92 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 10:40:00 | 284.70 | 283.00 | 0.00 | ORB-long ORB[281.35,283.85] vol=5.0x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 10:45:00 | 285.76 | 284.24 | 0.00 | T1 1.5R @ 285.76 |
| Stop hit — per-position SL triggered | 2023-09-06 11:05:00 | 284.70 | 285.04 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 10:55:00 | 289.60 | 287.04 | 0.00 | ORB-long ORB[284.60,286.85] vol=11.3x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 11:00:00 | 291.11 | 288.13 | 0.00 | T1 1.5R @ 291.11 |
| Stop hit — per-position SL triggered | 2023-09-07 11:05:00 | 289.60 | 288.32 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 11:15:00 | 283.35 | 286.89 | 0.00 | ORB-short ORB[288.50,291.40] vol=1.6x ATR=0.83 |
| Stop hit — per-position SL triggered | 2023-09-08 11:30:00 | 284.18 | 286.78 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:30:00 | 281.25 | 283.08 | 0.00 | ORB-short ORB[282.80,285.30] vol=2.0x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:35:00 | 280.06 | 281.74 | 0.00 | T1 1.5R @ 280.06 |
| Target hit | 2023-09-12 10:50:00 | 277.75 | 277.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — SELL (started 2023-09-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-14 10:05:00 | 284.00 | 285.73 | 0.00 | ORB-short ORB[284.75,287.20] vol=2.2x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 10:15:00 | 282.58 | 285.26 | 0.00 | T1 1.5R @ 282.58 |
| Stop hit — per-position SL triggered | 2023-09-14 10:25:00 | 284.00 | 285.18 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-09-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 09:55:00 | 289.20 | 287.06 | 0.00 | ORB-long ORB[285.00,287.25] vol=4.1x ATR=1.06 |
| Stop hit — per-position SL triggered | 2023-09-15 10:10:00 | 288.14 | 287.78 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-09-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-21 09:30:00 | 284.20 | 282.41 | 0.00 | ORB-long ORB[280.25,284.00] vol=1.8x ATR=1.02 |
| Stop hit — per-position SL triggered | 2023-09-21 09:40:00 | 283.18 | 282.61 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-09-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 10:00:00 | 277.35 | 280.61 | 0.00 | ORB-short ORB[279.75,283.40] vol=1.5x ATR=1.12 |
| Stop hit — per-position SL triggered | 2023-09-22 10:05:00 | 278.47 | 280.53 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-09-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 09:45:00 | 282.20 | 283.70 | 0.00 | ORB-short ORB[282.50,286.65] vol=1.6x ATR=1.02 |
| Stop hit — per-position SL triggered | 2023-09-25 09:55:00 | 283.22 | 283.56 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-09-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 10:55:00 | 279.10 | 279.52 | 0.00 | ORB-short ORB[279.25,280.55] vol=1.5x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 11:05:00 | 278.31 | 279.43 | 0.00 | T1 1.5R @ 278.31 |
| Target hit | 2023-09-28 15:20:00 | 273.05 | 275.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-04 10:15:00 | 273.40 | 274.16 | 0.00 | ORB-short ORB[273.65,275.60] vol=1.5x ATR=0.63 |
| Stop hit — per-position SL triggered | 2023-10-04 10:25:00 | 274.03 | 274.10 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 11:15:00 | 272.25 | 273.38 | 0.00 | ORB-short ORB[272.90,273.90] vol=3.1x ATR=0.48 |
| Stop hit — per-position SL triggered | 2023-10-06 11:25:00 | 272.73 | 273.35 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-10-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 09:45:00 | 275.20 | 274.22 | 0.00 | ORB-long ORB[273.00,274.95] vol=1.9x ATR=0.85 |
| Stop hit — per-position SL triggered | 2023-10-10 09:55:00 | 274.35 | 274.25 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-10-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:30:00 | 287.25 | 285.99 | 0.00 | ORB-long ORB[284.20,286.65] vol=2.8x ATR=1.28 |
| Stop hit — per-position SL triggered | 2023-10-11 09:45:00 | 285.97 | 286.27 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-10-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:45:00 | 293.25 | 295.74 | 0.00 | ORB-short ORB[295.20,297.90] vol=2.3x ATR=0.80 |
| Stop hit — per-position SL triggered | 2023-10-18 10:55:00 | 294.05 | 295.51 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 10:15:00 | 292.75 | 295.13 | 0.00 | ORB-short ORB[293.15,297.50] vol=2.1x ATR=1.01 |
| Stop hit — per-position SL triggered | 2023-10-20 10:45:00 | 293.76 | 294.44 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 09:30:00 | 286.70 | 288.96 | 0.00 | ORB-short ORB[287.30,291.50] vol=2.6x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 10:10:00 | 284.75 | 287.77 | 0.00 | T1 1.5R @ 284.75 |
| Target hit | 2023-10-23 15:20:00 | 276.65 | 283.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2023-10-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 09:30:00 | 275.00 | 273.81 | 0.00 | ORB-long ORB[271.05,274.80] vol=1.8x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-27 10:05:00 | 276.80 | 274.57 | 0.00 | T1 1.5R @ 276.80 |
| Stop hit — per-position SL triggered | 2023-10-27 10:20:00 | 275.00 | 274.67 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-30 11:15:00 | 277.80 | 279.57 | 0.00 | ORB-short ORB[278.90,281.00] vol=4.2x ATR=0.80 |
| Stop hit — per-position SL triggered | 2023-10-30 11:25:00 | 278.60 | 279.55 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2023-11-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-02 11:00:00 | 285.95 | 287.04 | 0.00 | ORB-short ORB[286.60,288.85] vol=1.5x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-11-02 11:30:00 | 286.66 | 286.90 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-11-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-03 10:25:00 | 287.05 | 287.88 | 0.00 | ORB-short ORB[287.30,289.25] vol=2.3x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-03 10:35:00 | 285.85 | 287.61 | 0.00 | T1 1.5R @ 285.85 |
| Target hit | 2023-11-03 12:15:00 | 285.20 | 285.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — SELL (started 2023-11-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:30:00 | 303.50 | 305.33 | 0.00 | ORB-short ORB[305.75,309.25] vol=2.2x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 12:20:00 | 301.90 | 304.42 | 0.00 | T1 1.5R @ 301.90 |
| Target hit | 2023-11-13 15:20:00 | 300.10 | 303.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2023-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:35:00 | 310.85 | 308.40 | 0.00 | ORB-long ORB[305.00,308.90] vol=3.6x ATR=1.07 |
| Stop hit — per-position SL triggered | 2023-11-21 09:45:00 | 309.78 | 309.06 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:30:00 | 308.35 | 310.22 | 0.00 | ORB-short ORB[309.30,311.95] vol=2.5x ATR=0.94 |
| Stop hit — per-position SL triggered | 2023-11-24 09:35:00 | 309.29 | 310.04 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 09:45:00 | 314.80 | 312.93 | 0.00 | ORB-long ORB[308.65,313.35] vol=5.5x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-11-28 09:50:00 | 313.58 | 313.70 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2023-12-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:55:00 | 344.85 | 347.67 | 0.00 | ORB-short ORB[347.55,351.25] vol=2.3x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 11:00:00 | 343.20 | 347.35 | 0.00 | T1 1.5R @ 343.20 |
| Stop hit — per-position SL triggered | 2023-12-08 11:25:00 | 344.85 | 346.83 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 09:35:00 | 360.35 | 357.91 | 0.00 | ORB-long ORB[353.90,358.50] vol=3.2x ATR=1.84 |
| Stop hit — per-position SL triggered | 2023-12-13 09:50:00 | 358.51 | 358.56 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2023-12-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-15 10:45:00 | 348.00 | 353.38 | 0.00 | ORB-short ORB[352.05,355.95] vol=2.1x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 11:10:00 | 345.84 | 351.95 | 0.00 | T1 1.5R @ 345.84 |
| Stop hit — per-position SL triggered | 2023-12-15 11:40:00 | 348.00 | 351.38 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2023-12-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 09:45:00 | 349.30 | 351.57 | 0.00 | ORB-short ORB[351.50,355.60] vol=1.7x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 09:50:00 | 347.31 | 350.77 | 0.00 | T1 1.5R @ 347.31 |
| Stop hit — per-position SL triggered | 2023-12-19 12:20:00 | 349.30 | 348.72 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2023-12-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 10:50:00 | 366.00 | 362.79 | 0.00 | ORB-long ORB[360.00,364.00] vol=3.3x ATR=1.50 |
| Stop hit — per-position SL triggered | 2023-12-26 10:55:00 | 364.50 | 362.93 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2023-12-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 10:35:00 | 368.15 | 371.93 | 0.00 | ORB-short ORB[372.35,375.00] vol=1.6x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-12-27 10:40:00 | 369.37 | 371.80 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 09:30:00 | 383.10 | 380.47 | 0.00 | ORB-long ORB[378.25,382.00] vol=1.9x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-01-03 09:35:00 | 381.59 | 380.52 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-01-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 09:35:00 | 386.90 | 384.34 | 0.00 | ORB-long ORB[381.55,385.40] vol=4.3x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 09:40:00 | 389.14 | 385.44 | 0.00 | T1 1.5R @ 389.14 |
| Stop hit — per-position SL triggered | 2024-01-12 09:55:00 | 386.90 | 385.79 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-01-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 11:05:00 | 380.05 | 383.94 | 0.00 | ORB-short ORB[382.35,387.00] vol=3.1x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 11:30:00 | 377.55 | 383.23 | 0.00 | T1 1.5R @ 377.55 |
| Target hit | 2024-01-19 15:20:00 | 373.65 | 376.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2024-01-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 10:55:00 | 365.10 | 368.60 | 0.00 | ORB-short ORB[367.05,370.05] vol=3.0x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-01-25 11:05:00 | 366.32 | 368.50 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-30 09:30:00 | 370.10 | 371.93 | 0.00 | ORB-short ORB[370.85,373.70] vol=2.2x ATR=1.48 |
| Stop hit — per-position SL triggered | 2024-01-30 10:25:00 | 371.58 | 370.95 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-02-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 09:40:00 | 370.60 | 368.82 | 0.00 | ORB-long ORB[366.05,369.70] vol=1.8x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 09:50:00 | 372.54 | 369.57 | 0.00 | T1 1.5R @ 372.54 |
| Stop hit — per-position SL triggered | 2024-02-01 10:15:00 | 370.60 | 370.48 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-02-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 10:45:00 | 353.20 | 357.16 | 0.00 | ORB-short ORB[360.05,364.15] vol=2.0x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 11:20:00 | 350.83 | 356.01 | 0.00 | T1 1.5R @ 350.83 |
| Stop hit — per-position SL triggered | 2024-02-09 11:35:00 | 353.20 | 355.84 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-02-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 10:30:00 | 369.90 | 367.89 | 0.00 | ORB-long ORB[364.70,369.00] vol=2.2x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-02-16 10:55:00 | 368.68 | 368.34 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-02-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:35:00 | 372.40 | 371.68 | 0.00 | ORB-long ORB[370.00,372.00] vol=3.2x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-02-21 10:05:00 | 371.44 | 372.10 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-02-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-22 10:40:00 | 370.00 | 368.32 | 0.00 | ORB-long ORB[364.95,369.15] vol=4.8x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-02-22 10:50:00 | 368.53 | 368.38 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 09:45:00 | 359.95 | 359.14 | 0.00 | ORB-long ORB[357.10,359.55] vol=2.0x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-02-27 10:05:00 | 359.16 | 359.32 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 352.10 | 354.66 | 0.00 | ORB-short ORB[353.55,355.95] vol=2.4x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 11:00:00 | 350.88 | 353.95 | 0.00 | T1 1.5R @ 350.88 |
| Stop hit — per-position SL triggered | 2024-02-28 11:15:00 | 352.10 | 353.78 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-03-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 10:00:00 | 350.60 | 352.47 | 0.00 | ORB-short ORB[352.40,355.45] vol=1.8x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-03-01 11:55:00 | 352.43 | 351.82 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:55:00 | 351.55 | 354.82 | 0.00 | ORB-short ORB[353.50,357.60] vol=2.4x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-03-06 10:05:00 | 352.76 | 353.94 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 09:30:00 | 344.65 | 346.21 | 0.00 | ORB-short ORB[345.15,347.75] vol=1.9x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-18 10:55:00 | 342.91 | 344.70 | 0.00 | T1 1.5R @ 342.91 |
| Target hit | 2024-03-18 11:45:00 | 344.30 | 344.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 85 — BUY (started 2024-03-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-19 09:35:00 | 344.85 | 343.88 | 0.00 | ORB-long ORB[342.30,344.45] vol=1.7x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-03-19 09:50:00 | 343.84 | 343.93 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-03-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:00:00 | 334.55 | 338.40 | 0.00 | ORB-short ORB[339.05,343.00] vol=1.7x ATR=1.36 |
| Stop hit — per-position SL triggered | 2024-03-20 10:05:00 | 335.91 | 338.21 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-03-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 09:55:00 | 343.70 | 342.52 | 0.00 | ORB-long ORB[339.00,342.90] vol=1.8x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-03-21 10:45:00 | 342.59 | 342.90 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-03-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 10:30:00 | 343.30 | 344.46 | 0.00 | ORB-short ORB[343.75,346.70] vol=3.4x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-03-27 10:40:00 | 344.18 | 344.44 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2024-04-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 09:45:00 | 360.40 | 358.66 | 0.00 | ORB-long ORB[355.40,359.40] vol=1.9x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 09:55:00 | 361.97 | 359.40 | 0.00 | T1 1.5R @ 361.97 |
| Stop hit — per-position SL triggered | 2024-04-02 10:00:00 | 360.40 | 359.49 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-04-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 09:45:00 | 365.10 | 362.84 | 0.00 | ORB-long ORB[358.00,362.95] vol=1.6x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 11:15:00 | 367.01 | 364.30 | 0.00 | T1 1.5R @ 367.01 |
| Target hit | 2024-04-03 15:20:00 | 374.30 | 372.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 91 — BUY (started 2024-04-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 09:30:00 | 373.80 | 372.43 | 0.00 | ORB-long ORB[369.30,373.55] vol=4.0x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-04-05 09:35:00 | 372.58 | 372.51 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-04-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 10:10:00 | 379.10 | 377.06 | 0.00 | ORB-long ORB[374.45,378.80] vol=1.9x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 10:15:00 | 380.94 | 377.81 | 0.00 | T1 1.5R @ 380.94 |
| Target hit | 2024-04-10 10:55:00 | 381.95 | 382.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 93 — BUY (started 2024-04-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 10:20:00 | 374.70 | 372.41 | 0.00 | ORB-long ORB[370.35,374.30] vol=2.6x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 10:25:00 | 376.24 | 372.97 | 0.00 | T1 1.5R @ 376.24 |
| Stop hit — per-position SL triggered | 2024-04-23 10:40:00 | 374.70 | 374.02 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2024-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 09:35:00 | 408.25 | 405.68 | 0.00 | ORB-long ORB[402.20,406.50] vol=2.2x ATR=1.97 |
| Stop hit — per-position SL triggered | 2024-04-25 09:50:00 | 406.28 | 406.42 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-04-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 10:30:00 | 430.40 | 427.44 | 0.00 | ORB-long ORB[424.00,428.90] vol=2.5x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 10:40:00 | 433.13 | 428.55 | 0.00 | T1 1.5R @ 433.13 |
| Target hit | 2024-04-29 13:40:00 | 432.20 | 432.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 96 — SELL (started 2024-05-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 09:35:00 | 411.95 | 413.40 | 0.00 | ORB-short ORB[412.80,417.70] vol=2.1x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 09:45:00 | 410.13 | 412.45 | 0.00 | T1 1.5R @ 410.13 |
| Target hit | 2024-05-03 15:10:00 | 406.00 | 405.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 97 — SELL (started 2024-05-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-10 09:35:00 | 382.75 | 387.87 | 0.00 | ORB-short ORB[386.30,391.95] vol=1.9x ATR=2.17 |
| Stop hit — per-position SL triggered | 2024-05-10 09:40:00 | 384.92 | 387.82 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-16 09:30:00 | 294.35 | 2023-05-16 09:35:00 | 293.55 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-05-25 09:30:00 | 286.25 | 2023-05-25 09:35:00 | 285.06 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-05-25 09:30:00 | 286.25 | 2023-05-25 10:10:00 | 285.90 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2023-06-02 11:15:00 | 280.35 | 2023-06-02 11:20:00 | 280.74 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-06-05 11:15:00 | 283.20 | 2023-06-05 11:20:00 | 282.62 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-07 11:15:00 | 281.35 | 2023-06-07 12:00:00 | 280.87 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-06-12 10:25:00 | 278.90 | 2023-06-12 10:30:00 | 278.04 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-06-13 10:35:00 | 280.25 | 2023-06-13 12:25:00 | 279.57 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-06-14 10:55:00 | 282.00 | 2023-06-14 11:10:00 | 283.08 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-06-14 10:55:00 | 282.00 | 2023-06-14 11:30:00 | 282.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-16 11:05:00 | 279.55 | 2023-06-16 11:15:00 | 279.01 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-06-20 10:20:00 | 279.85 | 2023-06-20 10:30:00 | 279.19 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-06-21 11:15:00 | 279.00 | 2023-06-21 11:20:00 | 279.42 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-06-23 09:35:00 | 270.00 | 2023-06-23 09:50:00 | 270.83 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-06-26 11:05:00 | 270.75 | 2023-06-26 11:55:00 | 269.86 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-06-27 09:35:00 | 276.10 | 2023-06-27 09:40:00 | 277.31 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-06-27 09:35:00 | 276.10 | 2023-06-27 10:35:00 | 276.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-04 10:00:00 | 272.95 | 2023-07-04 10:05:00 | 273.61 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-06 10:45:00 | 277.55 | 2023-07-06 11:40:00 | 278.71 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-07-06 10:45:00 | 277.55 | 2023-07-06 12:05:00 | 277.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-07 10:40:00 | 274.65 | 2023-07-07 11:10:00 | 275.33 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-07-10 11:00:00 | 272.90 | 2023-07-10 11:30:00 | 272.02 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-07-10 11:00:00 | 272.90 | 2023-07-10 13:00:00 | 272.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-11 11:15:00 | 273.00 | 2023-07-11 13:50:00 | 272.18 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-07-11 11:15:00 | 273.00 | 2023-07-11 15:20:00 | 272.40 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2023-07-12 10:45:00 | 271.95 | 2023-07-12 11:30:00 | 271.23 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-07-12 10:45:00 | 271.95 | 2023-07-12 15:20:00 | 271.00 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2023-07-13 10:55:00 | 270.75 | 2023-07-13 12:25:00 | 269.86 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-07-13 10:55:00 | 270.75 | 2023-07-13 15:20:00 | 265.40 | TARGET_HIT | 0.50 | 1.98% |
| BUY | retest1 | 2023-07-17 09:40:00 | 269.00 | 2023-07-17 09:50:00 | 268.22 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-07-18 10:40:00 | 267.25 | 2023-07-18 11:10:00 | 266.39 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-07-18 10:40:00 | 267.25 | 2023-07-18 15:20:00 | 265.50 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2023-07-25 09:35:00 | 272.00 | 2023-07-25 10:00:00 | 271.20 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-26 10:05:00 | 272.50 | 2023-07-26 10:10:00 | 273.42 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-07-26 10:05:00 | 272.50 | 2023-07-26 10:40:00 | 274.50 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2023-07-27 11:15:00 | 273.40 | 2023-07-27 11:45:00 | 274.01 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-28 09:50:00 | 275.30 | 2023-07-28 10:50:00 | 276.79 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2023-07-28 09:50:00 | 275.30 | 2023-07-28 13:40:00 | 275.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-31 09:40:00 | 277.45 | 2023-07-31 09:45:00 | 278.48 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-07-31 09:40:00 | 277.45 | 2023-07-31 10:50:00 | 278.95 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2023-08-03 09:40:00 | 271.30 | 2023-08-03 09:45:00 | 272.34 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-08-07 10:50:00 | 270.65 | 2023-08-07 11:25:00 | 271.31 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-08-08 10:15:00 | 275.55 | 2023-08-08 10:20:00 | 274.81 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-10 10:20:00 | 269.70 | 2023-08-10 10:30:00 | 268.31 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2023-08-10 10:20:00 | 269.70 | 2023-08-10 10:40:00 | 269.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-22 10:20:00 | 263.45 | 2023-08-22 10:45:00 | 262.81 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-08-25 10:35:00 | 269.10 | 2023-08-25 11:15:00 | 270.06 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-08-28 09:40:00 | 273.40 | 2023-08-28 10:30:00 | 275.17 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2023-08-28 09:40:00 | 273.40 | 2023-08-28 11:45:00 | 274.85 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2023-08-30 09:30:00 | 282.35 | 2023-08-30 09:40:00 | 283.64 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-08-30 09:30:00 | 282.35 | 2023-08-30 11:05:00 | 283.25 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2023-08-31 11:05:00 | 277.15 | 2023-08-31 11:20:00 | 277.81 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-09-05 09:35:00 | 282.15 | 2023-09-05 10:20:00 | 281.37 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-09-06 10:40:00 | 284.70 | 2023-09-06 10:45:00 | 285.76 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-09-06 10:40:00 | 284.70 | 2023-09-06 11:05:00 | 284.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-07 10:55:00 | 289.60 | 2023-09-07 11:00:00 | 291.11 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-09-07 10:55:00 | 289.60 | 2023-09-07 11:05:00 | 289.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-08 11:15:00 | 283.35 | 2023-09-08 11:30:00 | 284.18 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-09-12 09:30:00 | 281.25 | 2023-09-12 09:35:00 | 280.06 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-09-12 09:30:00 | 281.25 | 2023-09-12 10:50:00 | 277.75 | TARGET_HIT | 0.50 | 1.24% |
| SELL | retest1 | 2023-09-14 10:05:00 | 284.00 | 2023-09-14 10:15:00 | 282.58 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2023-09-14 10:05:00 | 284.00 | 2023-09-14 10:25:00 | 284.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-15 09:55:00 | 289.20 | 2023-09-15 10:10:00 | 288.14 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-09-21 09:30:00 | 284.20 | 2023-09-21 09:40:00 | 283.18 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-09-22 10:00:00 | 277.35 | 2023-09-22 10:05:00 | 278.47 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-09-25 09:45:00 | 282.20 | 2023-09-25 09:55:00 | 283.22 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-09-28 10:55:00 | 279.10 | 2023-09-28 11:05:00 | 278.31 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-09-28 10:55:00 | 279.10 | 2023-09-28 15:20:00 | 273.05 | TARGET_HIT | 0.50 | 2.17% |
| SELL | retest1 | 2023-10-04 10:15:00 | 273.40 | 2023-10-04 10:25:00 | 274.03 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-10-06 11:15:00 | 272.25 | 2023-10-06 11:25:00 | 272.73 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-10-10 09:45:00 | 275.20 | 2023-10-10 09:55:00 | 274.35 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-10-11 09:30:00 | 287.25 | 2023-10-11 09:45:00 | 285.97 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2023-10-18 10:45:00 | 293.25 | 2023-10-18 10:55:00 | 294.05 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-10-20 10:15:00 | 292.75 | 2023-10-20 10:45:00 | 293.76 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-10-23 09:30:00 | 286.70 | 2023-10-23 10:10:00 | 284.75 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2023-10-23 09:30:00 | 286.70 | 2023-10-23 15:20:00 | 276.65 | TARGET_HIT | 0.50 | 3.51% |
| BUY | retest1 | 2023-10-27 09:30:00 | 275.00 | 2023-10-27 10:05:00 | 276.80 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2023-10-27 09:30:00 | 275.00 | 2023-10-27 10:20:00 | 275.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-30 11:15:00 | 277.80 | 2023-10-30 11:25:00 | 278.60 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-02 11:00:00 | 285.95 | 2023-11-02 11:30:00 | 286.66 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-11-03 10:25:00 | 287.05 | 2023-11-03 10:35:00 | 285.85 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-11-03 10:25:00 | 287.05 | 2023-11-03 12:15:00 | 285.20 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2023-11-13 10:30:00 | 303.50 | 2023-11-13 12:20:00 | 301.90 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-11-13 10:30:00 | 303.50 | 2023-11-13 15:20:00 | 300.10 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2023-11-21 09:35:00 | 310.85 | 2023-11-21 09:45:00 | 309.78 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-11-24 09:30:00 | 308.35 | 2023-11-24 09:35:00 | 309.29 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-11-28 09:45:00 | 314.80 | 2023-11-28 09:50:00 | 313.58 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-12-08 10:55:00 | 344.85 | 2023-12-08 11:00:00 | 343.20 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-12-08 10:55:00 | 344.85 | 2023-12-08 11:25:00 | 344.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-13 09:35:00 | 360.35 | 2023-12-13 09:50:00 | 358.51 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2023-12-15 10:45:00 | 348.00 | 2023-12-15 11:10:00 | 345.84 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2023-12-15 10:45:00 | 348.00 | 2023-12-15 11:40:00 | 348.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-19 09:45:00 | 349.30 | 2023-12-19 09:50:00 | 347.31 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2023-12-19 09:45:00 | 349.30 | 2023-12-19 12:20:00 | 349.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 10:50:00 | 366.00 | 2023-12-26 10:55:00 | 364.50 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-12-27 10:35:00 | 368.15 | 2023-12-27 10:40:00 | 369.37 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-01-03 09:30:00 | 383.10 | 2024-01-03 09:35:00 | 381.59 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-01-12 09:35:00 | 386.90 | 2024-01-12 09:40:00 | 389.14 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-01-12 09:35:00 | 386.90 | 2024-01-12 09:55:00 | 386.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-19 11:05:00 | 380.05 | 2024-01-19 11:30:00 | 377.55 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-01-19 11:05:00 | 380.05 | 2024-01-19 15:20:00 | 373.65 | TARGET_HIT | 0.50 | 1.68% |
| SELL | retest1 | 2024-01-25 10:55:00 | 365.10 | 2024-01-25 11:05:00 | 366.32 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-01-30 09:30:00 | 370.10 | 2024-01-30 10:25:00 | 371.58 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-02-01 09:40:00 | 370.60 | 2024-02-01 09:50:00 | 372.54 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-02-01 09:40:00 | 370.60 | 2024-02-01 10:15:00 | 370.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-09 10:45:00 | 353.20 | 2024-02-09 11:20:00 | 350.83 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-02-09 10:45:00 | 353.20 | 2024-02-09 11:35:00 | 353.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-16 10:30:00 | 369.90 | 2024-02-16 10:55:00 | 368.68 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-02-21 09:35:00 | 372.40 | 2024-02-21 10:05:00 | 371.44 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-02-22 10:40:00 | 370.00 | 2024-02-22 10:50:00 | 368.53 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-02-27 09:45:00 | 359.95 | 2024-02-27 10:05:00 | 359.16 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-02-28 10:50:00 | 352.10 | 2024-02-28 11:00:00 | 350.88 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-02-28 10:50:00 | 352.10 | 2024-02-28 11:15:00 | 352.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-01 10:00:00 | 350.60 | 2024-03-01 11:55:00 | 352.43 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-03-06 09:55:00 | 351.55 | 2024-03-06 10:05:00 | 352.76 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-03-18 09:30:00 | 344.65 | 2024-03-18 10:55:00 | 342.91 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-03-18 09:30:00 | 344.65 | 2024-03-18 11:45:00 | 344.30 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2024-03-19 09:35:00 | 344.85 | 2024-03-19 09:50:00 | 343.84 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-03-20 10:00:00 | 334.55 | 2024-03-20 10:05:00 | 335.91 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-03-21 09:55:00 | 343.70 | 2024-03-21 10:45:00 | 342.59 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-03-27 10:30:00 | 343.30 | 2024-03-27 10:40:00 | 344.18 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-04-02 09:45:00 | 360.40 | 2024-04-02 09:55:00 | 361.97 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-04-02 09:45:00 | 360.40 | 2024-04-02 10:00:00 | 360.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-03 09:45:00 | 365.10 | 2024-04-03 11:15:00 | 367.01 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-04-03 09:45:00 | 365.10 | 2024-04-03 15:20:00 | 374.30 | TARGET_HIT | 0.50 | 2.52% |
| BUY | retest1 | 2024-04-05 09:30:00 | 373.80 | 2024-04-05 09:35:00 | 372.58 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-04-10 10:10:00 | 379.10 | 2024-04-10 10:15:00 | 380.94 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-04-10 10:10:00 | 379.10 | 2024-04-10 10:55:00 | 381.95 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2024-04-23 10:20:00 | 374.70 | 2024-04-23 10:25:00 | 376.24 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-04-23 10:20:00 | 374.70 | 2024-04-23 10:40:00 | 374.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-25 09:35:00 | 408.25 | 2024-04-25 09:50:00 | 406.28 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-04-29 10:30:00 | 430.40 | 2024-04-29 10:40:00 | 433.13 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-04-29 10:30:00 | 430.40 | 2024-04-29 13:40:00 | 432.20 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2024-05-03 09:35:00 | 411.95 | 2024-05-03 09:45:00 | 410.13 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-05-03 09:35:00 | 411.95 | 2024-05-03 15:10:00 | 406.00 | TARGET_HIT | 0.50 | 1.44% |
| SELL | retest1 | 2024-05-10 09:35:00 | 382.75 | 2024-05-10 09:40:00 | 384.92 | STOP_HIT | 1.00 | -0.57% |
