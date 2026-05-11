# Blue Jet Healthcare Ltd. (BLUEJET)

## Backtest Summary

- **Window:** 2023-11-01 09:55:00 → 2026-05-08 15:25:00 (43496 bars)
- **Last close:** 491.00
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
| ENTRY1 | 29 |
| ENTRY2 | 0 |
| PARTIAL | 17 |
| TARGET_HIT | 11 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 18
- **Target hits / Stop hits / Partials:** 11 / 18 / 17
- **Avg / median % per leg:** 0.44% / 0.57%
- **Sum % (uncompounded):** 20.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 13 | 61.9% | 5 | 8 | 8 | 0.48% | 10.1% |
| BUY @ 2nd Alert (retest1) | 21 | 13 | 61.9% | 5 | 8 | 8 | 0.48% | 10.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 25 | 15 | 60.0% | 6 | 10 | 9 | 0.41% | 10.2% |
| SELL @ 2nd Alert (retest1) | 25 | 15 | 60.0% | 6 | 10 | 9 | 0.41% | 10.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 46 | 28 | 60.9% | 11 | 18 | 17 | 0.44% | 20.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 09:50:00 | 392.05 | 388.79 | 0.00 | ORB-long ORB[385.10,389.70] vol=4.1x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 10:00:00 | 395.80 | 391.40 | 0.00 | T1 1.5R @ 395.80 |
| Target hit | 2023-11-08 15:20:00 | 404.00 | 399.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2023-11-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-10 10:00:00 | 390.95 | 395.76 | 0.00 | ORB-short ORB[393.85,399.00] vol=1.6x ATR=2.78 |
| Stop hit — per-position SL triggered | 2023-11-10 10:10:00 | 393.73 | 395.52 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-11-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-15 09:55:00 | 399.10 | 402.49 | 0.00 | ORB-short ORB[400.95,406.90] vol=2.0x ATR=2.21 |
| Stop hit — per-position SL triggered | 2023-11-15 10:00:00 | 401.31 | 402.46 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-11-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-17 10:50:00 | 385.45 | 389.57 | 0.00 | ORB-short ORB[391.00,394.00] vol=2.8x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 12:25:00 | 383.22 | 386.49 | 0.00 | T1 1.5R @ 383.22 |
| Target hit | 2023-11-17 15:20:00 | 381.85 | 383.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2023-11-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 09:30:00 | 385.20 | 383.80 | 0.00 | ORB-long ORB[381.00,384.50] vol=1.5x ATR=1.48 |
| Stop hit — per-position SL triggered | 2023-11-20 10:05:00 | 383.72 | 384.33 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-11-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:45:00 | 398.40 | 395.97 | 0.00 | ORB-long ORB[392.60,396.00] vol=3.5x ATR=2.27 |
| Stop hit — per-position SL triggered | 2023-11-21 11:15:00 | 396.13 | 396.83 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 10:15:00 | 381.10 | 382.10 | 0.00 | ORB-short ORB[381.65,384.30] vol=3.3x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 11:20:00 | 379.27 | 381.30 | 0.00 | T1 1.5R @ 379.27 |
| Stop hit — per-position SL triggered | 2023-11-24 11:50:00 | 381.10 | 381.11 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-11-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 10:05:00 | 369.70 | 367.84 | 0.00 | ORB-long ORB[364.00,368.25] vol=2.0x ATR=1.52 |
| Stop hit — per-position SL triggered | 2023-11-29 10:25:00 | 368.18 | 368.00 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-11-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:45:00 | 357.80 | 362.82 | 0.00 | ORB-short ORB[363.00,366.80] vol=1.6x ATR=1.73 |
| Stop hit — per-position SL triggered | 2023-11-30 10:10:00 | 359.53 | 359.69 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-12-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 10:55:00 | 354.40 | 357.67 | 0.00 | ORB-short ORB[357.60,362.90] vol=1.5x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 11:25:00 | 352.54 | 356.39 | 0.00 | T1 1.5R @ 352.54 |
| Target hit | 2023-12-06 15:20:00 | 348.45 | 351.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2023-12-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 09:35:00 | 345.05 | 348.11 | 0.00 | ORB-short ORB[348.00,353.00] vol=1.9x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 10:15:00 | 342.58 | 343.44 | 0.00 | T1 1.5R @ 342.58 |
| Stop hit — per-position SL triggered | 2023-12-08 10:25:00 | 345.05 | 343.48 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-12-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 10:10:00 | 338.80 | 335.50 | 0.00 | ORB-long ORB[332.80,336.85] vol=1.5x ATR=1.46 |
| Stop hit — per-position SL triggered | 2023-12-15 10:50:00 | 337.34 | 336.64 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 09:30:00 | 358.00 | 355.21 | 0.00 | ORB-long ORB[352.30,355.10] vol=4.4x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 09:35:00 | 360.57 | 356.72 | 0.00 | T1 1.5R @ 360.57 |
| Stop hit — per-position SL triggered | 2023-12-19 09:40:00 | 358.00 | 356.88 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-12-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:55:00 | 363.10 | 360.35 | 0.00 | ORB-long ORB[357.25,361.00] vol=4.3x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 10:00:00 | 365.61 | 361.91 | 0.00 | T1 1.5R @ 365.61 |
| Stop hit — per-position SL triggered | 2023-12-22 10:15:00 | 363.10 | 362.50 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-01-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 10:45:00 | 378.25 | 383.68 | 0.00 | ORB-short ORB[382.95,387.80] vol=3.0x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-01-05 10:50:00 | 380.08 | 382.32 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-02-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 09:40:00 | 348.10 | 350.45 | 0.00 | ORB-short ORB[348.80,354.00] vol=2.1x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:50:00 | 345.60 | 349.61 | 0.00 | T1 1.5R @ 345.60 |
| Target hit | 2024-02-09 10:10:00 | 347.75 | 346.93 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — BUY (started 2024-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 09:35:00 | 348.90 | 347.84 | 0.00 | ORB-long ORB[344.05,347.90] vol=3.7x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 12:45:00 | 350.89 | 349.05 | 0.00 | T1 1.5R @ 350.89 |
| Target hit | 2024-02-20 15:20:00 | 354.00 | 352.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-02-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 10:00:00 | 363.50 | 361.10 | 0.00 | ORB-long ORB[351.50,356.95] vol=1.9x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-02-21 10:55:00 | 361.29 | 361.95 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:35:00 | 342.50 | 344.10 | 0.00 | ORB-short ORB[343.60,345.80] vol=2.1x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:55:00 | 340.52 | 343.20 | 0.00 | T1 1.5R @ 340.52 |
| Target hit | 2024-03-06 12:55:00 | 340.10 | 339.48 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — BUY (started 2024-03-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 10:10:00 | 348.60 | 345.82 | 0.00 | ORB-long ORB[343.05,346.90] vol=7.1x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 10:25:00 | 351.37 | 347.11 | 0.00 | T1 1.5R @ 351.37 |
| Target hit | 2024-03-07 14:35:00 | 351.70 | 351.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — SELL (started 2024-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 09:55:00 | 333.90 | 338.68 | 0.00 | ORB-short ORB[343.00,347.55] vol=2.0x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 10:05:00 | 331.34 | 336.73 | 0.00 | T1 1.5R @ 331.34 |
| Stop hit — per-position SL triggered | 2024-03-13 10:15:00 | 333.90 | 336.42 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-03-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 11:05:00 | 343.00 | 343.25 | 0.00 | ORB-short ORB[344.30,347.90] vol=3.0x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-03-15 11:25:00 | 344.09 | 343.49 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-04-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 09:40:00 | 398.10 | 395.65 | 0.00 | ORB-long ORB[393.60,397.20] vol=3.2x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 09:50:00 | 401.27 | 399.00 | 0.00 | T1 1.5R @ 401.27 |
| Stop hit — per-position SL triggered | 2024-04-18 10:05:00 | 398.10 | 400.81 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-04-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-22 10:10:00 | 395.20 | 397.13 | 0.00 | ORB-short ORB[396.00,401.05] vol=3.3x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 11:00:00 | 392.65 | 396.42 | 0.00 | T1 1.5R @ 392.65 |
| Target hit | 2024-04-22 15:20:00 | 388.30 | 392.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:15:00 | 388.00 | 385.04 | 0.00 | ORB-long ORB[382.00,386.50] vol=2.0x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 10:25:00 | 390.39 | 386.14 | 0.00 | T1 1.5R @ 390.39 |
| Target hit | 2024-04-24 11:15:00 | 390.70 | 391.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — BUY (started 2024-04-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 10:50:00 | 387.25 | 385.54 | 0.00 | ORB-long ORB[382.00,386.00] vol=1.9x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 11:15:00 | 389.24 | 386.70 | 0.00 | T1 1.5R @ 389.24 |
| Target hit | 2024-04-25 15:20:00 | 390.15 | 388.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2024-04-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-26 09:30:00 | 388.45 | 392.39 | 0.00 | ORB-short ORB[391.75,394.80] vol=1.5x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-04-26 09:35:00 | 389.86 | 391.67 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:15:00 | 380.30 | 381.58 | 0.00 | ORB-short ORB[381.10,384.40] vol=4.4x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 10:35:00 | 378.42 | 380.97 | 0.00 | T1 1.5R @ 378.42 |
| Target hit | 2024-05-03 15:20:00 | 369.75 | 373.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2024-05-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 09:55:00 | 376.05 | 376.92 | 0.00 | ORB-short ORB[376.70,381.25] vol=3.2x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-05-09 10:50:00 | 377.33 | 376.75 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-08 09:50:00 | 392.05 | 2023-11-08 10:00:00 | 395.80 | PARTIAL | 0.50 | 0.96% |
| BUY | retest1 | 2023-11-08 09:50:00 | 392.05 | 2023-11-08 15:20:00 | 404.00 | TARGET_HIT | 0.50 | 3.05% |
| SELL | retest1 | 2023-11-10 10:00:00 | 390.95 | 2023-11-10 10:10:00 | 393.73 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest1 | 2023-11-15 09:55:00 | 399.10 | 2023-11-15 10:00:00 | 401.31 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2023-11-17 10:50:00 | 385.45 | 2023-11-17 12:25:00 | 383.22 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2023-11-17 10:50:00 | 385.45 | 2023-11-17 15:20:00 | 381.85 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2023-11-20 09:30:00 | 385.20 | 2023-11-20 10:05:00 | 383.72 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-11-21 09:45:00 | 398.40 | 2023-11-21 11:15:00 | 396.13 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2023-11-24 10:15:00 | 381.10 | 2023-11-24 11:20:00 | 379.27 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-11-24 10:15:00 | 381.10 | 2023-11-24 11:50:00 | 381.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-29 10:05:00 | 369.70 | 2023-11-29 10:25:00 | 368.18 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-11-30 09:45:00 | 357.80 | 2023-11-30 10:10:00 | 359.53 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2023-12-06 10:55:00 | 354.40 | 2023-12-06 11:25:00 | 352.54 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2023-12-06 10:55:00 | 354.40 | 2023-12-06 15:20:00 | 348.45 | TARGET_HIT | 0.50 | 1.68% |
| SELL | retest1 | 2023-12-08 09:35:00 | 345.05 | 2023-12-08 10:15:00 | 342.58 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2023-12-08 09:35:00 | 345.05 | 2023-12-08 10:25:00 | 345.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-15 10:10:00 | 338.80 | 2023-12-15 10:50:00 | 337.34 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-12-19 09:30:00 | 358.00 | 2023-12-19 09:35:00 | 360.57 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2023-12-19 09:30:00 | 358.00 | 2023-12-19 09:40:00 | 358.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-22 09:55:00 | 363.10 | 2023-12-22 10:00:00 | 365.61 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2023-12-22 09:55:00 | 363.10 | 2023-12-22 10:15:00 | 363.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-05 10:45:00 | 378.25 | 2024-01-05 10:50:00 | 380.08 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-02-09 09:40:00 | 348.10 | 2024-02-09 09:50:00 | 345.60 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-02-09 09:40:00 | 348.10 | 2024-02-09 10:10:00 | 347.75 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2024-02-20 09:35:00 | 348.90 | 2024-02-20 12:45:00 | 350.89 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-02-20 09:35:00 | 348.90 | 2024-02-20 15:20:00 | 354.00 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2024-02-21 10:00:00 | 363.50 | 2024-02-21 10:55:00 | 361.29 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2024-03-06 09:35:00 | 342.50 | 2024-03-06 09:55:00 | 340.52 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-03-06 09:35:00 | 342.50 | 2024-03-06 12:55:00 | 340.10 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2024-03-07 10:10:00 | 348.60 | 2024-03-07 10:25:00 | 351.37 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2024-03-07 10:10:00 | 348.60 | 2024-03-07 14:35:00 | 351.70 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2024-03-13 09:55:00 | 333.90 | 2024-03-13 10:05:00 | 331.34 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-03-13 09:55:00 | 333.90 | 2024-03-13 10:15:00 | 333.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-15 11:05:00 | 343.00 | 2024-03-15 11:25:00 | 344.09 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-04-18 09:40:00 | 398.10 | 2024-04-18 09:50:00 | 401.27 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2024-04-18 09:40:00 | 398.10 | 2024-04-18 10:05:00 | 398.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-22 10:10:00 | 395.20 | 2024-04-22 11:00:00 | 392.65 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-04-22 10:10:00 | 395.20 | 2024-04-22 15:20:00 | 388.30 | TARGET_HIT | 0.50 | 1.75% |
| BUY | retest1 | 2024-04-24 10:15:00 | 388.00 | 2024-04-24 10:25:00 | 390.39 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-04-24 10:15:00 | 388.00 | 2024-04-24 11:15:00 | 390.70 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2024-04-25 10:50:00 | 387.25 | 2024-04-25 11:15:00 | 389.24 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-04-25 10:50:00 | 387.25 | 2024-04-25 15:20:00 | 390.15 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-04-26 09:30:00 | 388.45 | 2024-04-26 09:35:00 | 389.86 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-05-03 10:15:00 | 380.30 | 2024-05-03 10:35:00 | 378.42 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-05-03 10:15:00 | 380.30 | 2024-05-03 15:20:00 | 369.75 | TARGET_HIT | 0.50 | 2.77% |
| SELL | retest1 | 2024-05-09 09:55:00 | 376.05 | 2024-05-09 10:50:00 | 377.33 | STOP_HIT | 1.00 | -0.34% |
