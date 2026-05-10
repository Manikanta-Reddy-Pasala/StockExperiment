# REC Ltd. (RECLTD)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 359.30
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 12
- **Target hits / Stop hits / Partials:** 4 / 12 / 6
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 2.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 10 | 62.5% | 4 | 6 | 6 | 0.24% | 3.8% |
| BUY @ 2nd Alert (retest1) | 16 | 10 | 62.5% | 4 | 6 | 6 | 0.24% | 3.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.26% | -1.5% |
| SELL @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.26% | -1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 10 | 45.5% | 4 | 12 | 6 | 0.10% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 342.15 | 344.75 | 0.00 | ORB-short ORB[344.00,348.50] vol=1.8x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 343.27 | 344.23 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:40:00 | 347.40 | 347.26 | 0.00 | ORB-long ORB[343.20,346.60] vol=2.5x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:05:00 | 348.96 | 347.48 | 0.00 | T1 1.5R @ 348.96 |
| Target hit | 2026-02-16 15:20:00 | 353.60 | 349.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:55:00 | 356.40 | 355.07 | 0.00 | ORB-long ORB[352.70,355.10] vol=1.8x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 12:15:00 | 357.55 | 355.83 | 0.00 | T1 1.5R @ 357.55 |
| Stop hit — per-position SL triggered | 2026-02-17 14:15:00 | 356.40 | 356.22 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 363.75 | 362.07 | 0.00 | ORB-long ORB[358.40,363.00] vol=2.2x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:15:00 | 365.11 | 362.58 | 0.00 | T1 1.5R @ 365.11 |
| Stop hit — per-position SL triggered | 2026-02-18 13:50:00 | 363.75 | 363.39 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 348.00 | 349.12 | 0.00 | ORB-short ORB[348.30,351.40] vol=1.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 348.88 | 348.78 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 354.05 | 355.34 | 0.00 | ORB-short ORB[354.75,357.50] vol=1.8x ATR=0.77 |
| Stop hit — per-position SL triggered | 2026-02-25 11:50:00 | 354.82 | 355.01 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:15:00 | 350.10 | 351.76 | 0.00 | ORB-short ORB[352.05,354.20] vol=1.7x ATR=0.79 |
| Stop hit — per-position SL triggered | 2026-02-27 10:20:00 | 350.89 | 351.71 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 336.70 | 332.89 | 0.00 | ORB-long ORB[330.35,333.95] vol=2.4x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:00:00 | 338.58 | 334.02 | 0.00 | T1 1.5R @ 338.58 |
| Target hit | 2026-03-05 14:15:00 | 337.10 | 337.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2026-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:35:00 | 343.15 | 341.98 | 0.00 | ORB-long ORB[339.40,342.90] vol=1.7x ATR=1.23 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 341.92 | 342.46 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:05:00 | 328.45 | 326.52 | 0.00 | ORB-long ORB[323.25,327.65] vol=1.6x ATR=0.84 |
| Stop hit — per-position SL triggered | 2026-03-25 11:25:00 | 327.61 | 326.81 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 358.50 | 356.45 | 0.00 | ORB-long ORB[354.00,357.40] vol=2.0x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-04-16 09:50:00 | 357.44 | 356.75 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:15:00 | 372.00 | 368.51 | 0.00 | ORB-long ORB[362.70,366.70] vol=2.7x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:20:00 | 373.67 | 370.49 | 0.00 | T1 1.5R @ 373.67 |
| Target hit | 2026-04-17 15:20:00 | 373.00 | 372.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-04-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:00:00 | 370.10 | 373.56 | 0.00 | ORB-short ORB[373.35,378.75] vol=1.9x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-04-24 11:30:00 | 371.02 | 373.28 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 380.50 | 377.86 | 0.00 | ORB-long ORB[374.20,379.25] vol=3.5x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-04-28 09:50:00 | 379.09 | 378.61 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:15:00 | 354.80 | 356.26 | 0.00 | ORB-short ORB[356.30,361.10] vol=2.3x ATR=0.98 |
| Stop hit — per-position SL triggered | 2026-04-30 11:45:00 | 355.78 | 356.08 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:40:00 | 361.70 | 360.15 | 0.00 | ORB-long ORB[357.60,361.45] vol=1.6x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:00:00 | 363.42 | 360.97 | 0.00 | T1 1.5R @ 363.42 |
| Target hit | 2026-05-07 10:55:00 | 362.90 | 363.00 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:30:00 | 342.15 | 2026-02-13 09:40:00 | 343.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-16 10:40:00 | 347.40 | 2026-02-16 11:05:00 | 348.96 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-16 10:40:00 | 347.40 | 2026-02-16 15:20:00 | 353.60 | TARGET_HIT | 0.50 | 1.78% |
| BUY | retest1 | 2026-02-17 10:55:00 | 356.40 | 2026-02-17 12:15:00 | 357.55 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-17 10:55:00 | 356.40 | 2026-02-17 14:15:00 | 356.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 10:55:00 | 363.75 | 2026-02-18 11:15:00 | 365.11 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-18 10:55:00 | 363.75 | 2026-02-18 13:50:00 | 363.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 348.00 | 2026-02-24 09:45:00 | 348.88 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-25 11:05:00 | 354.05 | 2026-02-25 11:50:00 | 354.82 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-27 10:15:00 | 350.10 | 2026-02-27 10:20:00 | 350.89 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-03-05 10:45:00 | 336.70 | 2026-03-05 11:00:00 | 338.58 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-03-05 10:45:00 | 336.70 | 2026-03-05 14:15:00 | 337.10 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2026-03-18 09:35:00 | 343.15 | 2026-03-18 09:55:00 | 341.92 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-25 11:05:00 | 328.45 | 2026-03-25 11:25:00 | 327.61 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-16 09:45:00 | 358.50 | 2026-04-16 09:50:00 | 357.44 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-17 11:15:00 | 372.00 | 2026-04-17 12:20:00 | 373.67 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-17 11:15:00 | 372.00 | 2026-04-17 15:20:00 | 373.00 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2026-04-24 11:00:00 | 370.10 | 2026-04-24 11:30:00 | 371.02 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-28 09:30:00 | 380.50 | 2026-04-28 09:50:00 | 379.09 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-30 11:15:00 | 354.80 | 2026-04-30 11:45:00 | 355.78 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-07 09:40:00 | 361.70 | 2026-05-07 10:00:00 | 363.42 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-05-07 09:40:00 | 361.70 | 2026-05-07 10:55:00 | 362.90 | TARGET_HIT | 0.50 | 0.33% |
