# Mahindra & Mahindra Financial Services Ltd. (M&MFIN)

## Backtest Summary

- **Window:** 2024-08-09 09:15:00 → 2026-05-08 15:25:00 (32275 bars)
- **Last close:** 339.00
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
| PARTIAL | 20 |
| TARGET_HIT | 8 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 85 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 57
- **Target hits / Stop hits / Partials:** 8 / 57 / 20
- **Avg / median % per leg:** 0.01% / -0.22%
- **Sum % (uncompounded):** 1.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 16 | 33.3% | 5 | 32 | 11 | 0.02% | 1.2% |
| BUY @ 2nd Alert (retest1) | 48 | 16 | 33.3% | 5 | 32 | 11 | 0.02% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 37 | 12 | 32.4% | 3 | 25 | 9 | -0.00% | -0.1% |
| SELL @ 2nd Alert (retest1) | 37 | 12 | 32.4% | 3 | 25 | 9 | -0.00% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 85 | 28 | 32.9% | 8 | 57 | 20 | 0.01% | 1.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 10:45:00 | 295.80 | 296.97 | 0.00 | ORB-short ORB[296.00,298.45] vol=1.5x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-08-19 11:20:00 | 296.44 | 296.79 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-08-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:40:00 | 299.10 | 298.02 | 0.00 | ORB-long ORB[296.80,298.45] vol=2.3x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:45:00 | 300.26 | 298.67 | 0.00 | T1 1.5R @ 300.26 |
| Target hit | 2024-08-20 15:20:00 | 303.85 | 302.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 312.70 | 311.79 | 0.00 | ORB-long ORB[307.55,309.50] vol=9.9x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-08-22 09:35:00 | 311.71 | 312.21 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-08-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:40:00 | 311.65 | 308.77 | 0.00 | ORB-long ORB[306.50,308.60] vol=1.9x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-08-26 09:50:00 | 310.50 | 309.41 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-08-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:00:00 | 317.80 | 315.50 | 0.00 | ORB-long ORB[313.65,317.70] vol=1.5x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:05:00 | 319.27 | 317.61 | 0.00 | T1 1.5R @ 319.27 |
| Target hit | 2024-08-27 12:40:00 | 319.35 | 319.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 317.45 | 320.19 | 0.00 | ORB-short ORB[317.95,322.45] vol=2.4x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 318.74 | 320.10 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-09-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:00:00 | 320.70 | 318.04 | 0.00 | ORB-long ORB[315.80,318.60] vol=2.1x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-09-02 10:20:00 | 319.47 | 319.03 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-09-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:50:00 | 326.50 | 324.88 | 0.00 | ORB-long ORB[323.30,325.65] vol=1.7x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-09-03 09:55:00 | 325.49 | 324.95 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-09-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 11:05:00 | 331.00 | 328.23 | 0.00 | ORB-long ORB[324.50,329.00] vol=6.9x ATR=1.00 |
| Stop hit — per-position SL triggered | 2024-09-05 11:25:00 | 330.00 | 329.36 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:45:00 | 327.45 | 330.07 | 0.00 | ORB-short ORB[327.70,332.40] vol=1.8x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 325.90 | 329.01 | 0.00 | T1 1.5R @ 325.90 |
| Stop hit — per-position SL triggered | 2024-09-06 10:20:00 | 327.45 | 328.49 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 09:30:00 | 320.80 | 322.63 | 0.00 | ORB-short ORB[322.15,325.75] vol=1.6x ATR=1.65 |
| Stop hit — per-position SL triggered | 2024-09-09 09:35:00 | 322.45 | 322.50 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-09-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:25:00 | 335.75 | 332.65 | 0.00 | ORB-long ORB[329.45,332.45] vol=3.7x ATR=1.17 |
| Stop hit — per-position SL triggered | 2024-09-13 10:30:00 | 334.58 | 332.98 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-09-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 11:00:00 | 328.60 | 329.73 | 0.00 | ORB-short ORB[329.20,333.85] vol=2.5x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-09-17 11:30:00 | 329.48 | 329.08 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-09-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:10:00 | 324.40 | 329.84 | 0.00 | ORB-short ORB[329.05,333.25] vol=1.9x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:25:00 | 322.36 | 328.71 | 0.00 | T1 1.5R @ 322.36 |
| Target hit | 2024-09-19 14:50:00 | 322.90 | 322.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2024-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:30:00 | 331.15 | 329.25 | 0.00 | ORB-long ORB[326.05,329.40] vol=2.6x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-09-25 09:35:00 | 330.01 | 329.33 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-09-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:45:00 | 342.35 | 340.04 | 0.00 | ORB-long ORB[337.50,340.15] vol=1.8x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-09-27 11:05:00 | 341.19 | 340.37 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-09-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 10:55:00 | 337.15 | 334.28 | 0.00 | ORB-long ORB[332.75,336.75] vol=1.9x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-09-30 11:05:00 | 335.87 | 334.42 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:15:00 | 330.35 | 334.39 | 0.00 | ORB-short ORB[335.75,339.90] vol=2.5x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-10-01 10:55:00 | 331.89 | 332.39 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 11:15:00 | 324.20 | 326.05 | 0.00 | ORB-short ORB[325.15,329.15] vol=2.2x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-10-03 11:20:00 | 325.19 | 325.88 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-10-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:55:00 | 294.55 | 297.84 | 0.00 | ORB-short ORB[300.95,304.95] vol=2.0x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:40:00 | 291.91 | 295.64 | 0.00 | T1 1.5R @ 291.91 |
| Stop hit — per-position SL triggered | 2024-10-07 11:05:00 | 294.55 | 294.71 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-10-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:40:00 | 293.90 | 291.81 | 0.00 | ORB-long ORB[289.60,293.00] vol=4.3x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 10:50:00 | 295.81 | 292.28 | 0.00 | T1 1.5R @ 295.81 |
| Stop hit — per-position SL triggered | 2024-10-09 11:00:00 | 293.90 | 292.33 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 09:30:00 | 285.45 | 287.73 | 0.00 | ORB-short ORB[287.50,289.60] vol=1.5x ATR=1.31 |
| Stop hit — per-position SL triggered | 2024-10-10 09:40:00 | 286.76 | 287.37 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-10-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 11:05:00 | 283.55 | 284.51 | 0.00 | ORB-short ORB[284.65,286.65] vol=2.3x ATR=0.60 |
| Stop hit — per-position SL triggered | 2024-10-14 11:15:00 | 284.15 | 284.46 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-10-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:45:00 | 287.55 | 285.98 | 0.00 | ORB-long ORB[283.55,286.00] vol=2.4x ATR=0.85 |
| Stop hit — per-position SL triggered | 2024-10-15 09:50:00 | 286.70 | 286.03 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-10-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:50:00 | 290.00 | 288.95 | 0.00 | ORB-long ORB[287.70,289.90] vol=1.7x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-10-16 10:25:00 | 289.24 | 289.63 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-10-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-21 11:00:00 | 294.65 | 292.42 | 0.00 | ORB-long ORB[290.55,293.45] vol=3.5x ATR=1.30 |
| Stop hit — per-position SL triggered | 2024-10-21 11:10:00 | 293.35 | 292.48 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-10-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:45:00 | 279.05 | 277.87 | 0.00 | ORB-long ORB[276.00,278.75] vol=1.8x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-10-30 10:40:00 | 278.16 | 278.48 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:00:00 | 272.05 | 273.65 | 0.00 | ORB-short ORB[273.50,275.55] vol=1.7x ATR=0.60 |
| Stop hit — per-position SL triggered | 2024-10-31 11:10:00 | 272.65 | 273.62 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-11-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:45:00 | 274.15 | 275.80 | 0.00 | ORB-short ORB[276.25,278.90] vol=4.0x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-11-07 10:50:00 | 274.99 | 275.77 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-11-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:35:00 | 268.05 | 269.23 | 0.00 | ORB-short ORB[268.50,271.65] vol=1.9x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-11-12 09:50:00 | 268.92 | 268.99 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 11:15:00 | 260.00 | 260.91 | 0.00 | ORB-short ORB[261.50,265.20] vol=5.1x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 11:40:00 | 258.48 | 260.65 | 0.00 | T1 1.5R @ 258.48 |
| Stop hit — per-position SL triggered | 2024-11-13 12:20:00 | 260.00 | 260.50 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-11-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 09:55:00 | 257.50 | 258.28 | 0.00 | ORB-short ORB[257.70,261.25] vol=1.6x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-11-14 10:00:00 | 258.64 | 258.30 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-11-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-18 11:10:00 | 257.70 | 256.30 | 0.00 | ORB-long ORB[255.15,257.40] vol=1.5x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 11:25:00 | 258.87 | 256.48 | 0.00 | T1 1.5R @ 258.87 |
| Target hit | 2024-11-18 15:05:00 | 258.30 | 258.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — BUY (started 2024-11-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:50:00 | 262.85 | 261.02 | 0.00 | ORB-long ORB[258.50,260.95] vol=1.9x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 10:35:00 | 264.20 | 262.08 | 0.00 | T1 1.5R @ 264.20 |
| Stop hit — per-position SL triggered | 2024-11-19 11:20:00 | 262.85 | 262.38 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-11-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:35:00 | 259.35 | 258.63 | 0.00 | ORB-long ORB[255.75,259.30] vol=1.5x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-11-22 09:40:00 | 258.56 | 258.67 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:30:00 | 274.55 | 273.81 | 0.00 | ORB-long ORB[272.35,273.95] vol=3.0x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 09:35:00 | 275.55 | 274.14 | 0.00 | T1 1.5R @ 275.55 |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 274.55 | 274.98 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-11-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 11:10:00 | 270.15 | 270.89 | 0.00 | ORB-short ORB[271.00,272.80] vol=1.9x ATR=0.67 |
| Stop hit — per-position SL triggered | 2024-11-29 12:05:00 | 270.82 | 270.68 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-12-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:35:00 | 274.80 | 273.18 | 0.00 | ORB-long ORB[271.00,273.65] vol=2.0x ATR=0.86 |
| Stop hit — per-position SL triggered | 2024-12-02 09:40:00 | 273.94 | 273.31 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:55:00 | 290.25 | 288.16 | 0.00 | ORB-long ORB[285.45,288.90] vol=2.4x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-12-04 10:15:00 | 289.13 | 288.89 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:05:00 | 283.35 | 284.79 | 0.00 | ORB-short ORB[284.60,287.90] vol=3.4x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 10:15:00 | 281.79 | 283.15 | 0.00 | T1 1.5R @ 281.79 |
| Stop hit — per-position SL triggered | 2024-12-06 10:20:00 | 283.35 | 282.99 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:30:00 | 284.00 | 282.47 | 0.00 | ORB-long ORB[279.15,283.25] vol=4.4x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 09:35:00 | 285.26 | 283.86 | 0.00 | T1 1.5R @ 285.26 |
| Target hit | 2024-12-10 10:10:00 | 285.40 | 285.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — SELL (started 2024-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:55:00 | 282.75 | 283.62 | 0.00 | ORB-short ORB[283.45,285.40] vol=2.5x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-12-11 11:05:00 | 283.30 | 283.61 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 280.70 | 282.06 | 0.00 | ORB-short ORB[281.50,283.60] vol=1.7x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 10:10:00 | 279.74 | 281.53 | 0.00 | T1 1.5R @ 279.74 |
| Stop hit — per-position SL triggered | 2024-12-12 10:50:00 | 280.70 | 280.47 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-12-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:55:00 | 275.10 | 276.86 | 0.00 | ORB-short ORB[276.35,280.25] vol=1.5x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:10:00 | 273.79 | 275.14 | 0.00 | T1 1.5R @ 273.79 |
| Target hit | 2024-12-13 11:55:00 | 273.90 | 273.34 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — BUY (started 2024-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:40:00 | 267.10 | 265.51 | 0.00 | ORB-long ORB[264.20,267.05] vol=1.6x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 10:55:00 | 268.54 | 266.50 | 0.00 | T1 1.5R @ 268.54 |
| Target hit | 2024-12-19 15:20:00 | 271.95 | 268.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:15:00 | 269.70 | 269.54 | 0.00 | ORB-long ORB[265.00,268.60] vol=3.7x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-12-27 10:20:00 | 268.69 | 269.44 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-01-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:05:00 | 279.90 | 278.59 | 0.00 | ORB-long ORB[276.30,279.50] vol=2.1x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:20:00 | 281.34 | 279.29 | 0.00 | T1 1.5R @ 281.34 |
| Stop hit — per-position SL triggered | 2025-01-09 10:50:00 | 279.90 | 279.77 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-01-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:50:00 | 276.00 | 276.51 | 0.00 | ORB-short ORB[276.35,279.75] vol=4.1x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-01-10 09:55:00 | 276.83 | 276.09 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 10:00:00 | 269.75 | 271.02 | 0.00 | ORB-short ORB[270.05,273.30] vol=2.1x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-01-13 10:15:00 | 271.17 | 270.86 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-01-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:05:00 | 274.15 | 270.96 | 0.00 | ORB-long ORB[268.00,271.95] vol=1.7x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-01-17 10:15:00 | 273.14 | 271.41 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:55:00 | 269.50 | 267.54 | 0.00 | ORB-long ORB[265.05,267.80] vol=2.1x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-01-23 10:10:00 | 268.54 | 269.11 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-01-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:25:00 | 281.70 | 278.77 | 0.00 | ORB-long ORB[276.05,278.80] vol=1.8x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-01-31 11:30:00 | 280.56 | 279.52 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-02-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:55:00 | 297.40 | 295.26 | 0.00 | ORB-long ORB[292.50,296.50] vol=1.7x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 10:20:00 | 299.37 | 296.31 | 0.00 | T1 1.5R @ 299.37 |
| Stop hit — per-position SL triggered | 2025-02-04 10:30:00 | 297.40 | 296.60 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-02-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:45:00 | 294.90 | 297.08 | 0.00 | ORB-short ORB[296.55,299.20] vol=1.5x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-02-07 10:55:00 | 296.36 | 296.96 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:00:00 | 279.80 | 278.11 | 0.00 | ORB-long ORB[275.55,279.00] vol=3.2x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-02-20 11:20:00 | 278.87 | 278.67 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 274.70 | 272.93 | 0.00 | ORB-long ORB[270.00,273.55] vol=2.0x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-02-25 10:10:00 | 273.65 | 274.04 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-03-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:45:00 | 280.00 | 278.81 | 0.00 | ORB-long ORB[275.05,278.90] vol=5.8x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-03-18 10:10:00 | 278.96 | 279.19 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-04-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 10:05:00 | 257.40 | 258.50 | 0.00 | ORB-short ORB[257.50,260.40] vol=1.6x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 10:15:00 | 255.91 | 257.83 | 0.00 | T1 1.5R @ 255.91 |
| Stop hit — per-position SL triggered | 2025-04-09 10:25:00 | 257.40 | 257.75 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:35:00 | 273.05 | 270.12 | 0.00 | ORB-long ORB[267.60,271.20] vol=2.2x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 10:05:00 | 274.39 | 271.14 | 0.00 | T1 1.5R @ 274.39 |
| Stop hit — per-position SL triggered | 2025-04-16 13:15:00 | 273.05 | 273.12 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:45:00 | 277.45 | 275.24 | 0.00 | ORB-long ORB[274.05,277.15] vol=3.1x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-04-21 11:05:00 | 276.43 | 276.17 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 267.70 | 268.78 | 0.00 | ORB-short ORB[268.20,270.75] vol=1.8x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:40:00 | 266.60 | 268.36 | 0.00 | T1 1.5R @ 266.60 |
| Target hit | 2025-04-25 12:10:00 | 266.25 | 266.13 | 0.00 | Trail-exit close>VWAP |

### Cycle 62 — SELL (started 2025-04-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 11:05:00 | 261.65 | 262.80 | 0.00 | ORB-short ORB[262.45,264.70] vol=4.0x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-04-29 11:15:00 | 262.42 | 262.76 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-04-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:50:00 | 261.90 | 260.33 | 0.00 | ORB-long ORB[258.35,261.65] vol=1.7x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-04-30 09:55:00 | 261.11 | 260.37 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:05:00 | 264.65 | 264.40 | 0.00 | ORB-long ORB[262.25,264.35] vol=1.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-05-05 11:55:00 | 264.00 | 264.44 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 10:45:00 | 261.55 | 263.09 | 0.00 | ORB-short ORB[262.00,264.10] vol=2.0x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-05-08 11:15:00 | 262.24 | 262.68 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-08-19 10:45:00 | 295.80 | 2024-08-19 11:20:00 | 296.44 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-20 09:40:00 | 299.10 | 2024-08-20 09:45:00 | 300.26 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-08-20 09:40:00 | 299.10 | 2024-08-20 15:20:00 | 303.85 | TARGET_HIT | 0.50 | 1.59% |
| BUY | retest1 | 2024-08-22 09:30:00 | 312.70 | 2024-08-22 09:35:00 | 311.71 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-26 09:40:00 | 311.65 | 2024-08-26 09:50:00 | 310.50 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-08-27 10:00:00 | 317.80 | 2024-08-27 10:05:00 | 319.27 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-08-27 10:00:00 | 317.80 | 2024-08-27 12:40:00 | 319.35 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2024-08-28 09:30:00 | 317.45 | 2024-08-28 09:35:00 | 318.74 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-09-02 10:00:00 | 320.70 | 2024-09-02 10:20:00 | 319.47 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-09-03 09:50:00 | 326.50 | 2024-09-03 09:55:00 | 325.49 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-05 11:05:00 | 331.00 | 2024-09-05 11:25:00 | 330.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-06 09:45:00 | 327.45 | 2024-09-06 10:05:00 | 325.90 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-09-06 09:45:00 | 327.45 | 2024-09-06 10:20:00 | 327.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-09 09:30:00 | 320.80 | 2024-09-09 09:35:00 | 322.45 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-09-13 10:25:00 | 335.75 | 2024-09-13 10:30:00 | 334.58 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-09-17 11:00:00 | 328.60 | 2024-09-17 11:30:00 | 329.48 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-19 10:10:00 | 324.40 | 2024-09-19 10:25:00 | 322.36 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-09-19 10:10:00 | 324.40 | 2024-09-19 14:50:00 | 322.90 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2024-09-25 09:30:00 | 331.15 | 2024-09-25 09:35:00 | 330.01 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-27 10:45:00 | 342.35 | 2024-09-27 11:05:00 | 341.19 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-30 10:55:00 | 337.15 | 2024-09-30 11:05:00 | 335.87 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-01 10:15:00 | 330.35 | 2024-10-01 10:55:00 | 331.89 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-10-03 11:15:00 | 324.20 | 2024-10-03 11:20:00 | 325.19 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-07 09:55:00 | 294.55 | 2024-10-07 10:40:00 | 291.91 | PARTIAL | 0.50 | 0.90% |
| SELL | retest1 | 2024-10-07 09:55:00 | 294.55 | 2024-10-07 11:05:00 | 294.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 10:40:00 | 293.90 | 2024-10-09 10:50:00 | 295.81 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-10-09 10:40:00 | 293.90 | 2024-10-09 11:00:00 | 293.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-10 09:30:00 | 285.45 | 2024-10-10 09:40:00 | 286.76 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-10-14 11:05:00 | 283.55 | 2024-10-14 11:15:00 | 284.15 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-10-15 09:45:00 | 287.55 | 2024-10-15 09:50:00 | 286.70 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-16 09:50:00 | 290.00 | 2024-10-16 10:25:00 | 289.24 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-10-21 11:00:00 | 294.65 | 2024-10-21 11:10:00 | 293.35 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-10-30 09:45:00 | 279.05 | 2024-10-30 10:40:00 | 278.16 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-10-31 11:00:00 | 272.05 | 2024-10-31 11:10:00 | 272.65 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-11-07 10:45:00 | 274.15 | 2024-11-07 10:50:00 | 274.99 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-11-12 09:35:00 | 268.05 | 2024-11-12 09:50:00 | 268.92 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-11-13 11:15:00 | 260.00 | 2024-11-13 11:40:00 | 258.48 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-11-13 11:15:00 | 260.00 | 2024-11-13 12:20:00 | 260.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-14 09:55:00 | 257.50 | 2024-11-14 10:00:00 | 258.64 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-11-18 11:10:00 | 257.70 | 2024-11-18 11:25:00 | 258.87 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-11-18 11:10:00 | 257.70 | 2024-11-18 15:05:00 | 258.30 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2024-11-19 09:50:00 | 262.85 | 2024-11-19 10:35:00 | 264.20 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-11-19 09:50:00 | 262.85 | 2024-11-19 11:20:00 | 262.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-22 09:35:00 | 259.35 | 2024-11-22 09:40:00 | 258.56 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-28 09:30:00 | 274.55 | 2024-11-28 09:35:00 | 275.55 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-11-28 09:30:00 | 274.55 | 2024-11-28 10:15:00 | 274.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-29 11:10:00 | 270.15 | 2024-11-29 12:05:00 | 270.82 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-12-02 09:35:00 | 274.80 | 2024-12-02 09:40:00 | 273.94 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-04 09:55:00 | 290.25 | 2024-12-04 10:15:00 | 289.13 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-12-06 10:05:00 | 283.35 | 2024-12-06 10:15:00 | 281.79 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-12-06 10:05:00 | 283.35 | 2024-12-06 10:20:00 | 283.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-10 09:30:00 | 284.00 | 2024-12-10 09:35:00 | 285.26 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-12-10 09:30:00 | 284.00 | 2024-12-10 10:10:00 | 285.40 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2024-12-11 10:55:00 | 282.75 | 2024-12-11 11:05:00 | 283.30 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-12-12 09:40:00 | 280.70 | 2024-12-12 10:10:00 | 279.74 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-12-12 09:40:00 | 280.70 | 2024-12-12 10:50:00 | 280.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 09:55:00 | 275.10 | 2024-12-13 10:10:00 | 273.79 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-12-13 09:55:00 | 275.10 | 2024-12-13 11:55:00 | 273.90 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2024-12-19 09:40:00 | 267.10 | 2024-12-19 10:55:00 | 268.54 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-12-19 09:40:00 | 267.10 | 2024-12-19 15:20:00 | 271.95 | TARGET_HIT | 0.50 | 1.82% |
| BUY | retest1 | 2024-12-27 10:15:00 | 269.70 | 2024-12-27 10:20:00 | 268.69 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-09 10:05:00 | 279.90 | 2025-01-09 10:20:00 | 281.34 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-01-09 10:05:00 | 279.90 | 2025-01-09 10:50:00 | 279.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-10 09:50:00 | 276.00 | 2025-01-10 09:55:00 | 276.83 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-13 10:00:00 | 269.75 | 2025-01-13 10:15:00 | 271.17 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-01-17 10:05:00 | 274.15 | 2025-01-17 10:15:00 | 273.14 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-23 09:55:00 | 269.50 | 2025-01-23 10:10:00 | 268.54 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-01-31 10:25:00 | 281.70 | 2025-01-31 11:30:00 | 280.56 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-02-04 09:55:00 | 297.40 | 2025-02-04 10:20:00 | 299.37 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-02-04 09:55:00 | 297.40 | 2025-02-04 10:30:00 | 297.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-07 10:45:00 | 294.90 | 2025-02-07 10:55:00 | 296.36 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-02-20 11:00:00 | 279.80 | 2025-02-20 11:20:00 | 278.87 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-02-25 09:30:00 | 274.70 | 2025-02-25 10:10:00 | 273.65 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-03-18 09:45:00 | 280.00 | 2025-03-18 10:10:00 | 278.96 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-04-09 10:05:00 | 257.40 | 2025-04-09 10:15:00 | 255.91 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-04-09 10:05:00 | 257.40 | 2025-04-09 10:25:00 | 257.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-16 09:35:00 | 273.05 | 2025-04-16 10:05:00 | 274.39 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-04-16 09:35:00 | 273.05 | 2025-04-16 13:15:00 | 273.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 10:45:00 | 277.45 | 2025-04-21 11:05:00 | 276.43 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-04-25 09:30:00 | 267.70 | 2025-04-25 09:40:00 | 266.60 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-04-25 09:30:00 | 267.70 | 2025-04-25 12:10:00 | 266.25 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2025-04-29 11:05:00 | 261.65 | 2025-04-29 11:15:00 | 262.42 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-04-30 09:50:00 | 261.90 | 2025-04-30 09:55:00 | 261.11 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-05 11:05:00 | 264.65 | 2025-05-05 11:55:00 | 264.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-05-08 10:45:00 | 261.55 | 2025-05-08 11:15:00 | 262.24 | STOP_HIT | 1.00 | -0.26% |
