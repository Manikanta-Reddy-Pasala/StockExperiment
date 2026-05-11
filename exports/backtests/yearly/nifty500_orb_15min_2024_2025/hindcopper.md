# Hindustan Copper Ltd. (HINDCOPPER)

## Backtest Summary

- **Window:** 2024-08-09 09:15:00 → 2026-05-08 15:25:00 (32275 bars)
- **Last close:** 568.90
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
| ENTRY1 | 39 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 3 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 36
- **Target hits / Stop hits / Partials:** 3 / 36 / 13
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 2.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 9 | 25.7% | 2 | 26 | 7 | -0.00% | -0.0% |
| BUY @ 2nd Alert (retest1) | 35 | 9 | 25.7% | 2 | 26 | 7 | -0.00% | -0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 7 | 41.2% | 1 | 10 | 6 | 0.14% | 2.4% |
| SELL @ 2nd Alert (retest1) | 17 | 7 | 41.2% | 1 | 10 | 6 | 0.14% | 2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 52 | 16 | 30.8% | 3 | 36 | 13 | 0.05% | 2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 320.25 | 317.91 | 0.00 | ORB-long ORB[315.00,319.25] vol=1.7x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 11:50:00 | 323.39 | 320.07 | 0.00 | T1 1.5R @ 323.39 |
| Stop hit — per-position SL triggered | 2024-08-19 12:20:00 | 320.25 | 320.25 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-08-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:25:00 | 319.25 | 317.91 | 0.00 | ORB-long ORB[316.50,318.75] vol=2.5x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-08-22 11:00:00 | 318.09 | 318.41 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-08-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:35:00 | 323.00 | 326.31 | 0.00 | ORB-short ORB[325.00,328.50] vol=1.8x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:05:00 | 320.94 | 325.27 | 0.00 | T1 1.5R @ 320.94 |
| Stop hit — per-position SL triggered | 2024-08-29 11:10:00 | 323.00 | 325.20 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-09-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:55:00 | 310.80 | 313.43 | 0.00 | ORB-short ORB[313.15,315.90] vol=1.5x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-09-05 11:00:00 | 311.71 | 313.27 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-09-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:40:00 | 315.80 | 313.51 | 0.00 | ORB-long ORB[311.00,313.95] vol=2.2x ATR=1.50 |
| Stop hit — per-position SL triggered | 2024-09-10 09:50:00 | 314.30 | 313.91 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-09-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 10:00:00 | 310.50 | 312.07 | 0.00 | ORB-short ORB[311.15,314.00] vol=2.6x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 13:55:00 | 308.67 | 310.92 | 0.00 | T1 1.5R @ 308.67 |
| Target hit | 2024-09-11 15:20:00 | 306.50 | 309.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:55:00 | 338.30 | 335.47 | 0.00 | ORB-long ORB[332.80,337.00] vol=3.2x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:10:00 | 340.60 | 336.86 | 0.00 | T1 1.5R @ 340.60 |
| Stop hit — per-position SL triggered | 2024-09-24 10:45:00 | 338.30 | 338.15 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:30:00 | 341.55 | 344.27 | 0.00 | ORB-short ORB[342.55,347.00] vol=1.6x ATR=1.78 |
| Stop hit — per-position SL triggered | 2024-09-25 09:40:00 | 343.33 | 344.04 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-10-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:10:00 | 337.60 | 340.64 | 0.00 | ORB-short ORB[338.85,343.90] vol=1.9x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-10-01 11:35:00 | 339.04 | 340.46 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-10-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 11:05:00 | 319.25 | 315.92 | 0.00 | ORB-long ORB[313.00,316.20] vol=1.5x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-10-09 11:10:00 | 317.98 | 315.97 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:30:00 | 325.90 | 321.70 | 0.00 | ORB-long ORB[318.30,321.75] vol=3.7x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-10-16 09:35:00 | 323.99 | 322.83 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-10-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:10:00 | 292.50 | 290.21 | 0.00 | ORB-long ORB[287.10,291.25] vol=1.5x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-10-30 10:45:00 | 291.03 | 290.75 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-10-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 11:00:00 | 294.15 | 291.14 | 0.00 | ORB-long ORB[286.50,290.65] vol=2.6x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-10-31 11:05:00 | 293.05 | 291.23 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-11-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-11 09:35:00 | 275.20 | 277.48 | 0.00 | ORB-short ORB[276.20,279.65] vol=1.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-11-11 09:40:00 | 276.87 | 277.22 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:30:00 | 268.00 | 266.63 | 0.00 | ORB-long ORB[263.75,267.65] vol=2.2x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 10:00:00 | 269.82 | 267.45 | 0.00 | T1 1.5R @ 269.82 |
| Stop hit — per-position SL triggered | 2024-11-19 12:05:00 | 268.00 | 268.33 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:15:00 | 264.40 | 262.90 | 0.00 | ORB-long ORB[260.60,263.50] vol=2.2x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-11-22 10:25:00 | 263.46 | 263.02 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-11-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:30:00 | 269.50 | 268.23 | 0.00 | ORB-long ORB[266.50,269.40] vol=2.5x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 10:00:00 | 271.26 | 269.20 | 0.00 | T1 1.5R @ 271.26 |
| Target hit | 2024-11-25 15:20:00 | 279.75 | 273.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2024-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 09:30:00 | 276.40 | 278.17 | 0.00 | ORB-short ORB[277.00,280.90] vol=1.5x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 10:10:00 | 274.81 | 276.78 | 0.00 | T1 1.5R @ 274.81 |
| Stop hit — per-position SL triggered | 2024-11-27 10:35:00 | 276.40 | 276.43 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-11-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 10:00:00 | 280.15 | 279.58 | 0.00 | ORB-long ORB[277.60,279.80] vol=3.0x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-11-28 10:05:00 | 279.25 | 279.57 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-12-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:05:00 | 282.95 | 285.77 | 0.00 | ORB-short ORB[285.05,287.45] vol=2.1x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:55:00 | 281.68 | 285.37 | 0.00 | T1 1.5R @ 281.68 |
| Stop hit — per-position SL triggered | 2024-12-04 12:05:00 | 282.95 | 285.20 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:30:00 | 292.05 | 290.64 | 0.00 | ORB-long ORB[287.40,290.70] vol=2.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-12-09 09:35:00 | 290.90 | 290.74 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-12-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:05:00 | 287.25 | 289.77 | 0.00 | ORB-short ORB[289.35,291.40] vol=1.6x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:25:00 | 285.98 | 289.21 | 0.00 | T1 1.5R @ 285.98 |
| Stop hit — per-position SL triggered | 2024-12-12 11:45:00 | 287.25 | 289.03 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 11:15:00 | 275.40 | 274.16 | 0.00 | ORB-long ORB[271.25,274.80] vol=1.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-12-19 12:10:00 | 274.49 | 274.34 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-12-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 11:00:00 | 280.00 | 277.39 | 0.00 | ORB-long ORB[275.00,278.65] vol=3.4x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 11:50:00 | 281.66 | 278.13 | 0.00 | T1 1.5R @ 281.66 |
| Stop hit — per-position SL triggered | 2024-12-20 13:05:00 | 280.00 | 279.07 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-01-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 09:45:00 | 245.97 | 246.69 | 0.00 | ORB-short ORB[246.26,247.90] vol=1.8x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-01-01 09:55:00 | 246.80 | 246.62 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-01-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 09:30:00 | 239.91 | 237.39 | 0.00 | ORB-long ORB[235.35,237.81] vol=1.7x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-01-07 10:00:00 | 238.63 | 238.49 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-01-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 10:00:00 | 229.15 | 227.16 | 0.00 | ORB-long ORB[225.75,227.85] vol=3.0x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-01-15 10:10:00 | 227.73 | 227.31 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-01-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:05:00 | 235.90 | 232.39 | 0.00 | ORB-long ORB[227.50,230.55] vol=2.1x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-01-29 10:10:00 | 234.70 | 232.67 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-01-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:35:00 | 234.80 | 233.42 | 0.00 | ORB-long ORB[228.60,232.00] vol=3.6x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-01-30 10:50:00 | 234.06 | 233.48 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-02-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:45:00 | 245.28 | 242.68 | 0.00 | ORB-long ORB[239.59,242.90] vol=2.4x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 11:10:00 | 247.04 | 244.26 | 0.00 | T1 1.5R @ 247.04 |
| Stop hit — per-position SL triggered | 2025-02-05 12:00:00 | 245.28 | 244.61 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-03-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 09:45:00 | 225.50 | 223.96 | 0.00 | ORB-long ORB[220.00,223.40] vol=2.8x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-03-06 11:25:00 | 223.60 | 224.63 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-03-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:30:00 | 225.96 | 225.60 | 0.00 | ORB-long ORB[223.00,225.95] vol=1.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-03-19 10:50:00 | 224.88 | 225.60 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-03-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:55:00 | 236.70 | 233.59 | 0.00 | ORB-long ORB[230.30,233.39] vol=1.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-03-21 10:25:00 | 235.48 | 234.94 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 11:00:00 | 217.90 | 220.18 | 0.00 | ORB-short ORB[218.66,221.82] vol=1.6x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 11:35:00 | 216.56 | 219.19 | 0.00 | T1 1.5R @ 216.56 |
| Stop hit — per-position SL triggered | 2025-04-01 12:10:00 | 217.90 | 218.95 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:35:00 | 216.49 | 215.02 | 0.00 | ORB-long ORB[213.00,215.95] vol=2.1x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-04-21 09:50:00 | 215.62 | 215.39 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:30:00 | 216.00 | 214.89 | 0.00 | ORB-long ORB[213.20,215.89] vol=1.6x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 09:55:00 | 217.72 | 215.70 | 0.00 | T1 1.5R @ 217.72 |
| Target hit | 2025-04-28 14:20:00 | 217.51 | 218.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — BUY (started 2025-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 09:30:00 | 222.54 | 221.04 | 0.00 | ORB-long ORB[219.00,221.35] vol=2.8x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-04-29 09:35:00 | 221.69 | 221.10 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:30:00 | 218.12 | 216.61 | 0.00 | ORB-long ORB[214.61,217.78] vol=1.8x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-04-30 12:45:00 | 217.06 | 217.56 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-05-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 09:50:00 | 218.70 | 215.93 | 0.00 | ORB-long ORB[212.70,215.85] vol=1.9x ATR=0.99 |
| Stop hit — per-position SL triggered | 2025-05-02 10:20:00 | 217.71 | 216.58 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-19 09:35:00 | 320.25 | 2024-08-19 11:50:00 | 323.39 | PARTIAL | 0.50 | 0.98% |
| BUY | retest1 | 2024-08-19 09:35:00 | 320.25 | 2024-08-19 12:20:00 | 320.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 10:25:00 | 319.25 | 2024-08-22 11:00:00 | 318.09 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-29 10:35:00 | 323.00 | 2024-08-29 11:05:00 | 320.94 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-08-29 10:35:00 | 323.00 | 2024-08-29 11:10:00 | 323.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-05 10:55:00 | 310.80 | 2024-09-05 11:00:00 | 311.71 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-10 09:40:00 | 315.80 | 2024-09-10 09:50:00 | 314.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-09-11 10:00:00 | 310.50 | 2024-09-11 13:55:00 | 308.67 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-09-11 10:00:00 | 310.50 | 2024-09-11 15:20:00 | 306.50 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2024-09-24 09:55:00 | 338.30 | 2024-09-24 10:10:00 | 340.60 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-09-24 09:55:00 | 338.30 | 2024-09-24 10:45:00 | 338.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-25 09:30:00 | 341.55 | 2024-09-25 09:40:00 | 343.33 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-10-01 11:10:00 | 337.60 | 2024-10-01 11:35:00 | 339.04 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-10-09 11:05:00 | 319.25 | 2024-10-09 11:10:00 | 317.98 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-10-16 09:30:00 | 325.90 | 2024-10-16 09:35:00 | 323.99 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-10-30 10:10:00 | 292.50 | 2024-10-30 10:45:00 | 291.03 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-10-31 11:00:00 | 294.15 | 2024-10-31 11:05:00 | 293.05 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-11-11 09:35:00 | 275.20 | 2024-11-11 09:40:00 | 276.87 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2024-11-19 09:30:00 | 268.00 | 2024-11-19 10:00:00 | 269.82 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-11-19 09:30:00 | 268.00 | 2024-11-19 12:05:00 | 268.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-22 10:15:00 | 264.40 | 2024-11-22 10:25:00 | 263.46 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-11-25 09:30:00 | 269.50 | 2024-11-25 10:00:00 | 271.26 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-11-25 09:30:00 | 269.50 | 2024-11-25 15:20:00 | 279.75 | TARGET_HIT | 0.50 | 3.80% |
| SELL | retest1 | 2024-11-27 09:30:00 | 276.40 | 2024-11-27 10:10:00 | 274.81 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-11-27 09:30:00 | 276.40 | 2024-11-27 10:35:00 | 276.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-28 10:00:00 | 280.15 | 2024-11-28 10:05:00 | 279.25 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-04 11:05:00 | 282.95 | 2024-12-04 11:55:00 | 281.68 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-12-04 11:05:00 | 282.95 | 2024-12-04 12:05:00 | 282.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-09 09:30:00 | 292.05 | 2024-12-09 09:35:00 | 290.90 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-12-12 11:05:00 | 287.25 | 2024-12-12 11:25:00 | 285.98 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-12 11:05:00 | 287.25 | 2024-12-12 11:45:00 | 287.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-19 11:15:00 | 275.40 | 2024-12-19 12:10:00 | 274.49 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-20 11:00:00 | 280.00 | 2024-12-20 11:50:00 | 281.66 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-12-20 11:00:00 | 280.00 | 2024-12-20 13:05:00 | 280.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-01 09:45:00 | 245.97 | 2025-01-01 09:55:00 | 246.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-07 09:30:00 | 239.91 | 2025-01-07 10:00:00 | 238.63 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-01-15 10:00:00 | 229.15 | 2025-01-15 10:10:00 | 227.73 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2025-01-29 10:05:00 | 235.90 | 2025-01-29 10:10:00 | 234.70 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-01-30 10:35:00 | 234.80 | 2025-01-30 10:50:00 | 234.06 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-02-05 09:45:00 | 245.28 | 2025-02-05 11:10:00 | 247.04 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-02-05 09:45:00 | 245.28 | 2025-02-05 12:00:00 | 245.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-06 09:45:00 | 225.50 | 2025-03-06 11:25:00 | 223.60 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest1 | 2025-03-19 10:30:00 | 225.96 | 2025-03-19 10:50:00 | 224.88 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-03-21 09:55:00 | 236.70 | 2025-03-21 10:25:00 | 235.48 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-04-01 11:00:00 | 217.90 | 2025-04-01 11:35:00 | 216.56 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-04-01 11:00:00 | 217.90 | 2025-04-01 12:10:00 | 217.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 09:35:00 | 216.49 | 2025-04-21 09:50:00 | 215.62 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-04-28 09:30:00 | 216.00 | 2025-04-28 09:55:00 | 217.72 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2025-04-28 09:30:00 | 216.00 | 2025-04-28 14:20:00 | 217.51 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2025-04-29 09:30:00 | 222.54 | 2025-04-29 09:35:00 | 221.69 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-04-30 09:30:00 | 218.12 | 2025-04-30 12:45:00 | 217.06 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-05-02 09:50:00 | 218.70 | 2025-05-02 10:20:00 | 217.71 | STOP_HIT | 1.00 | -0.45% |
