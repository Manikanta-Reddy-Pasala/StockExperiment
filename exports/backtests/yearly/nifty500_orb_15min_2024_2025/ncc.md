# NCC Ltd. (NCC)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35371 bars)
- **Last close:** 170.10
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
| ENTRY1 | 36 |
| ENTRY2 | 0 |
| PARTIAL | 17 |
| TARGET_HIT | 7 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 29
- **Target hits / Stop hits / Partials:** 7 / 29 / 17
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 11.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 13 | 46.4% | 4 | 15 | 9 | 0.24% | 6.8% |
| BUY @ 2nd Alert (retest1) | 28 | 13 | 46.4% | 4 | 15 | 9 | 0.24% | 6.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 25 | 11 | 44.0% | 3 | 14 | 8 | 0.21% | 5.1% |
| SELL @ 2nd Alert (retest1) | 25 | 11 | 44.0% | 3 | 14 | 8 | 0.21% | 5.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 53 | 24 | 45.3% | 7 | 29 | 17 | 0.22% | 11.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 276.25 | 280.57 | 0.00 | ORB-short ORB[280.55,284.70] vol=2.4x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-05-27 09:55:00 | 277.87 | 279.99 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-06-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:50:00 | 332.10 | 327.40 | 0.00 | ORB-long ORB[322.65,326.70] vol=2.7x ATR=2.08 |
| Stop hit — per-position SL triggered | 2024-06-21 09:55:00 | 330.02 | 328.03 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:30:00 | 328.75 | 326.39 | 0.00 | ORB-long ORB[323.40,328.20] vol=1.7x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 09:35:00 | 331.45 | 329.64 | 0.00 | T1 1.5R @ 331.45 |
| Target hit | 2024-06-25 10:00:00 | 332.85 | 332.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2024-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:40:00 | 332.40 | 330.43 | 0.00 | ORB-long ORB[327.15,331.95] vol=2.5x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-06-27 10:10:00 | 330.50 | 331.03 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-07-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:55:00 | 319.70 | 318.39 | 0.00 | ORB-long ORB[315.55,319.10] vol=4.5x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 14:05:00 | 322.20 | 319.70 | 0.00 | T1 1.5R @ 322.20 |
| Stop hit — per-position SL triggered | 2024-07-01 14:25:00 | 319.70 | 319.82 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-07-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:55:00 | 335.10 | 337.42 | 0.00 | ORB-short ORB[336.25,340.90] vol=1.9x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-07-05 10:15:00 | 336.77 | 337.17 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 330.30 | 336.01 | 0.00 | ORB-short ORB[335.25,339.95] vol=2.0x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:10:00 | 327.78 | 334.84 | 0.00 | T1 1.5R @ 327.78 |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 330.30 | 334.66 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-07-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:55:00 | 326.40 | 328.89 | 0.00 | ORB-short ORB[328.55,332.55] vol=1.6x ATR=1.46 |
| Stop hit — per-position SL triggered | 2024-07-11 11:15:00 | 327.86 | 328.80 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:30:00 | 335.30 | 333.13 | 0.00 | ORB-long ORB[331.25,335.25] vol=2.2x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-07-16 12:00:00 | 333.87 | 333.69 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:40:00 | 343.75 | 340.27 | 0.00 | ORB-long ORB[338.20,340.75] vol=3.3x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 09:45:00 | 345.91 | 342.14 | 0.00 | T1 1.5R @ 345.91 |
| Target hit | 2024-07-31 14:10:00 | 352.20 | 352.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2024-08-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 10:45:00 | 325.80 | 322.55 | 0.00 | ORB-long ORB[320.10,324.70] vol=1.9x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-08-09 10:50:00 | 324.27 | 322.67 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-08-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:40:00 | 316.00 | 314.15 | 0.00 | ORB-long ORB[312.30,315.50] vol=2.0x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-08-16 09:50:00 | 314.46 | 314.56 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:55:00 | 323.60 | 324.67 | 0.00 | ORB-short ORB[324.60,328.40] vol=2.9x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:20:00 | 321.63 | 324.41 | 0.00 | T1 1.5R @ 321.63 |
| Target hit | 2024-08-20 15:20:00 | 317.70 | 321.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-08-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:55:00 | 323.65 | 321.22 | 0.00 | ORB-long ORB[318.80,321.45] vol=1.6x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-08-23 10:05:00 | 322.56 | 321.45 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-09-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:00:00 | 321.95 | 323.89 | 0.00 | ORB-short ORB[322.30,326.90] vol=1.9x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:20:00 | 320.33 | 323.13 | 0.00 | T1 1.5R @ 320.33 |
| Target hit | 2024-09-06 15:20:00 | 316.00 | 318.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2024-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 09:30:00 | 309.65 | 312.15 | 0.00 | ORB-short ORB[312.00,316.20] vol=3.3x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-09-09 09:45:00 | 311.12 | 311.66 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:30:00 | 319.50 | 317.48 | 0.00 | ORB-long ORB[315.25,317.90] vol=2.5x ATR=1.35 |
| Stop hit — per-position SL triggered | 2024-09-11 09:35:00 | 318.15 | 317.56 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-09-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:50:00 | 314.95 | 314.10 | 0.00 | ORB-long ORB[312.45,314.65] vol=1.8x ATR=1.00 |
| Stop hit — per-position SL triggered | 2024-09-18 12:10:00 | 313.95 | 314.25 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 315.30 | 313.96 | 0.00 | ORB-long ORB[312.10,315.05] vol=4.2x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-09-19 09:35:00 | 314.19 | 314.00 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 308.60 | 309.76 | 0.00 | ORB-short ORB[308.85,311.90] vol=1.9x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:35:00 | 307.11 | 309.51 | 0.00 | T1 1.5R @ 307.11 |
| Stop hit — per-position SL triggered | 2024-10-21 10:00:00 | 308.60 | 308.61 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-11-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 10:55:00 | 305.60 | 307.85 | 0.00 | ORB-short ORB[306.05,309.85] vol=1.9x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-11-05 13:55:00 | 306.76 | 306.99 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-11-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 09:40:00 | 307.00 | 308.62 | 0.00 | ORB-short ORB[307.45,311.85] vol=2.2x ATR=1.38 |
| Stop hit — per-position SL triggered | 2024-11-06 13:00:00 | 308.38 | 307.41 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-12-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 09:35:00 | 311.00 | 312.96 | 0.00 | ORB-short ORB[311.85,315.65] vol=2.0x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 10:10:00 | 309.72 | 312.01 | 0.00 | T1 1.5R @ 309.72 |
| Stop hit — per-position SL triggered | 2024-12-10 11:45:00 | 311.00 | 311.16 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-12-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:50:00 | 309.80 | 312.10 | 0.00 | ORB-short ORB[312.30,315.80] vol=2.1x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 12:00:00 | 308.65 | 311.25 | 0.00 | T1 1.5R @ 308.65 |
| Target hit | 2024-12-11 15:20:00 | 306.60 | 308.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2024-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:40:00 | 312.30 | 310.65 | 0.00 | ORB-long ORB[307.35,311.50] vol=2.1x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 09:55:00 | 314.06 | 312.01 | 0.00 | T1 1.5R @ 314.06 |
| Stop hit — per-position SL triggered | 2024-12-17 10:25:00 | 312.30 | 312.36 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 11:15:00 | 299.60 | 297.39 | 0.00 | ORB-long ORB[295.05,298.65] vol=1.6x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 11:25:00 | 300.79 | 297.58 | 0.00 | T1 1.5R @ 300.79 |
| Stop hit — per-position SL triggered | 2024-12-19 12:45:00 | 299.60 | 298.10 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-12-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:00:00 | 294.70 | 298.10 | 0.00 | ORB-short ORB[298.55,301.20] vol=1.6x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:20:00 | 293.11 | 296.30 | 0.00 | T1 1.5R @ 293.11 |
| Stop hit — per-position SL triggered | 2024-12-20 10:40:00 | 294.70 | 295.74 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 282.00 | 283.72 | 0.00 | ORB-short ORB[283.10,285.95] vol=1.8x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-12-24 09:35:00 | 282.88 | 283.53 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:00:00 | 276.85 | 277.87 | 0.00 | ORB-short ORB[277.30,280.00] vol=1.8x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-01-02 10:20:00 | 277.59 | 277.59 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-02-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:05:00 | 257.75 | 255.01 | 0.00 | ORB-long ORB[252.40,254.80] vol=2.6x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 10:35:00 | 259.37 | 256.11 | 0.00 | T1 1.5R @ 259.37 |
| Stop hit — per-position SL triggered | 2025-02-01 11:00:00 | 257.75 | 257.25 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 10:15:00 | 188.16 | 186.35 | 0.00 | ORB-long ORB[184.17,186.70] vol=4.0x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-03-12 10:25:00 | 187.32 | 186.53 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-03-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:40:00 | 186.00 | 185.04 | 0.00 | ORB-long ORB[184.01,185.83] vol=1.5x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 09:50:00 | 186.92 | 185.76 | 0.00 | T1 1.5R @ 186.92 |
| Target hit | 2025-03-18 12:45:00 | 188.72 | 188.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2025-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 11:00:00 | 202.37 | 203.76 | 0.00 | ORB-short ORB[202.88,205.00] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-04-09 11:25:00 | 203.19 | 203.57 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:30:00 | 215.00 | 213.77 | 0.00 | ORB-long ORB[212.08,214.50] vol=2.3x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 09:45:00 | 216.31 | 214.39 | 0.00 | T1 1.5R @ 216.31 |
| Target hit | 2025-04-15 14:35:00 | 216.25 | 216.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — SELL (started 2025-04-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 10:10:00 | 218.16 | 219.75 | 0.00 | ORB-short ORB[219.05,222.19] vol=2.8x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 12:00:00 | 216.92 | 218.98 | 0.00 | T1 1.5R @ 216.92 |
| Stop hit — per-position SL triggered | 2025-04-17 13:50:00 | 218.16 | 218.59 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-05-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:35:00 | 220.08 | 218.02 | 0.00 | ORB-long ORB[216.05,218.45] vol=3.7x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 11:10:00 | 221.26 | 218.91 | 0.00 | T1 1.5R @ 221.26 |
| Stop hit — per-position SL triggered | 2025-05-05 14:05:00 | 220.08 | 219.92 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-27 09:45:00 | 276.25 | 2024-05-27 09:55:00 | 277.87 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-06-21 09:50:00 | 332.10 | 2024-06-21 09:55:00 | 330.02 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2024-06-25 09:30:00 | 328.75 | 2024-06-25 09:35:00 | 331.45 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2024-06-25 09:30:00 | 328.75 | 2024-06-25 10:00:00 | 332.85 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2024-06-27 09:40:00 | 332.40 | 2024-06-27 10:10:00 | 330.50 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-07-01 09:55:00 | 319.70 | 2024-07-01 14:05:00 | 322.20 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-07-01 09:55:00 | 319.70 | 2024-07-01 14:25:00 | 319.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-05 09:55:00 | 335.10 | 2024-07-05 10:15:00 | 336.77 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-07-10 10:05:00 | 330.30 | 2024-07-10 10:10:00 | 327.78 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2024-07-10 10:05:00 | 330.30 | 2024-07-10 10:15:00 | 330.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 10:55:00 | 326.40 | 2024-07-11 11:15:00 | 327.86 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-07-16 10:30:00 | 335.30 | 2024-07-16 12:00:00 | 333.87 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-07-31 09:40:00 | 343.75 | 2024-07-31 09:45:00 | 345.91 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-07-31 09:40:00 | 343.75 | 2024-07-31 14:10:00 | 352.20 | TARGET_HIT | 0.50 | 2.46% |
| BUY | retest1 | 2024-08-09 10:45:00 | 325.80 | 2024-08-09 10:50:00 | 324.27 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-08-16 09:40:00 | 316.00 | 2024-08-16 09:50:00 | 314.46 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-08-20 10:55:00 | 323.60 | 2024-08-20 11:20:00 | 321.63 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-08-20 10:55:00 | 323.60 | 2024-08-20 15:20:00 | 317.70 | TARGET_HIT | 0.50 | 1.82% |
| BUY | retest1 | 2024-08-23 09:55:00 | 323.65 | 2024-08-23 10:05:00 | 322.56 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-06 10:00:00 | 321.95 | 2024-09-06 10:20:00 | 320.33 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-09-06 10:00:00 | 321.95 | 2024-09-06 15:20:00 | 316.00 | TARGET_HIT | 0.50 | 1.85% |
| SELL | retest1 | 2024-09-09 09:30:00 | 309.65 | 2024-09-09 09:45:00 | 311.12 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-09-11 09:30:00 | 319.50 | 2024-09-11 09:35:00 | 318.15 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-09-18 10:50:00 | 314.95 | 2024-09-18 12:10:00 | 313.95 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-19 09:30:00 | 315.30 | 2024-09-19 09:35:00 | 314.19 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-21 09:30:00 | 308.60 | 2024-10-21 09:35:00 | 307.11 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-10-21 09:30:00 | 308.60 | 2024-10-21 10:00:00 | 308.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-05 10:55:00 | 305.60 | 2024-11-05 13:55:00 | 306.76 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-11-06 09:40:00 | 307.00 | 2024-11-06 13:00:00 | 308.38 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-12-10 09:35:00 | 311.00 | 2024-12-10 10:10:00 | 309.72 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-10 09:35:00 | 311.00 | 2024-12-10 11:45:00 | 311.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-11 10:50:00 | 309.80 | 2024-12-11 12:00:00 | 308.65 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-11 10:50:00 | 309.80 | 2024-12-11 15:20:00 | 306.60 | TARGET_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2024-12-17 09:40:00 | 312.30 | 2024-12-17 09:55:00 | 314.06 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-12-17 09:40:00 | 312.30 | 2024-12-17 10:25:00 | 312.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-19 11:15:00 | 299.60 | 2024-12-19 11:25:00 | 300.79 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-12-19 11:15:00 | 299.60 | 2024-12-19 12:45:00 | 299.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 10:00:00 | 294.70 | 2024-12-20 10:20:00 | 293.11 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-12-20 10:00:00 | 294.70 | 2024-12-20 10:40:00 | 294.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-24 09:30:00 | 282.00 | 2024-12-24 09:35:00 | 282.88 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-02 10:00:00 | 276.85 | 2025-01-02 10:20:00 | 277.59 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-02-01 10:05:00 | 257.75 | 2025-02-01 10:35:00 | 259.37 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-02-01 10:05:00 | 257.75 | 2025-02-01 11:00:00 | 257.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-12 10:15:00 | 188.16 | 2025-03-12 10:25:00 | 187.32 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-03-18 09:40:00 | 186.00 | 2025-03-18 09:50:00 | 186.92 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-03-18 09:40:00 | 186.00 | 2025-03-18 12:45:00 | 188.72 | TARGET_HIT | 0.50 | 1.46% |
| SELL | retest1 | 2025-04-09 11:00:00 | 202.37 | 2025-04-09 11:25:00 | 203.19 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-04-15 09:30:00 | 215.00 | 2025-04-15 09:45:00 | 216.31 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-04-15 09:30:00 | 215.00 | 2025-04-15 14:35:00 | 216.25 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-04-17 10:10:00 | 218.16 | 2025-04-17 12:00:00 | 216.92 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-04-17 10:10:00 | 218.16 | 2025-04-17 13:50:00 | 218.16 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 10:35:00 | 220.08 | 2025-05-05 11:10:00 | 221.26 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-05-05 10:35:00 | 220.08 | 2025-05-05 14:05:00 | 220.08 | STOP_HIT | 0.50 | 0.00% |
