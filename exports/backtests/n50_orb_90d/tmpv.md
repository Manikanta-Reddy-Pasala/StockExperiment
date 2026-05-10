# TMPV (TMPV)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 355.50
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 6 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 7
- **Target hits / Stop hits / Partials:** 6 / 7 / 6
- **Avg / median % per leg:** 0.33% / 0.42%
- **Sum % (uncompounded):** 6.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 3 | 5 | 3 | 0.16% | 1.7% |
| BUY @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 3 | 5 | 3 | 0.16% | 1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.56% | 4.5% |
| SELL @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 0.56% | 4.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 12 | 63.2% | 6 | 7 | 6 | 0.33% | 6.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 375.80 | 373.09 | 0.00 | ORB-long ORB[369.60,373.70] vol=1.6x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 12:00:00 | 378.98 | 375.21 | 0.00 | T1 1.5R @ 378.98 |
| Target hit | 2026-02-09 15:20:00 | 377.90 | 376.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 377.10 | 379.38 | 0.00 | ORB-short ORB[377.75,382.50] vol=1.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-02-10 09:35:00 | 378.20 | 379.34 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:45:00 | 385.95 | 384.28 | 0.00 | ORB-long ORB[382.10,385.60] vol=2.4x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-02-11 12:00:00 | 384.80 | 384.70 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 379.00 | 377.03 | 0.00 | ORB-long ORB[373.55,377.00] vol=2.2x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 378.09 | 377.32 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:00:00 | 380.95 | 379.46 | 0.00 | ORB-long ORB[376.10,380.25] vol=4.9x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:35:00 | 382.87 | 379.95 | 0.00 | T1 1.5R @ 382.87 |
| Target hit | 2026-02-25 13:25:00 | 381.65 | 381.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-02-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:55:00 | 385.35 | 383.85 | 0.00 | ORB-long ORB[380.10,383.35] vol=5.6x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:05:00 | 386.82 | 384.30 | 0.00 | T1 1.5R @ 386.82 |
| Target hit | 2026-02-26 13:55:00 | 388.75 | 388.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:15:00 | 339.70 | 343.76 | 0.00 | ORB-short ORB[343.55,347.00] vol=1.8x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 13:00:00 | 338.28 | 342.21 | 0.00 | T1 1.5R @ 338.28 |
| Target hit | 2026-03-11 15:20:00 | 334.95 | 339.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:35:00 | 317.90 | 320.70 | 0.00 | ORB-short ORB[320.05,324.15] vol=1.7x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:45:00 | 316.18 | 319.92 | 0.00 | T1 1.5R @ 316.18 |
| Target hit | 2026-03-13 15:10:00 | 314.35 | 314.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2026-03-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:45:00 | 327.15 | 324.06 | 0.00 | ORB-long ORB[319.90,324.25] vol=2.8x ATR=1.36 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 325.79 | 324.54 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:45:00 | 360.65 | 354.65 | 0.00 | ORB-long ORB[349.25,354.15] vol=2.0x ATR=1.33 |
| Stop hit — per-position SL triggered | 2026-04-15 10:50:00 | 359.32 | 354.97 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 09:30:00 | 352.80 | 355.54 | 0.00 | ORB-short ORB[353.60,358.30] vol=1.8x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-04-20 09:50:00 | 354.14 | 355.00 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:55:00 | 355.80 | 358.10 | 0.00 | ORB-short ORB[356.35,361.20] vol=1.8x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 12:50:00 | 354.07 | 356.86 | 0.00 | T1 1.5R @ 354.07 |
| Target hit | 2026-04-23 15:20:00 | 351.50 | 355.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 343.10 | 341.85 | 0.00 | ORB-long ORB[339.00,342.65] vol=1.6x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-05-05 10:00:00 | 341.98 | 342.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 375.80 | 2026-02-09 12:00:00 | 378.98 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2026-02-09 10:35:00 | 375.80 | 2026-02-09 15:20:00 | 377.90 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2026-02-10 09:30:00 | 377.10 | 2026-02-10 09:35:00 | 378.20 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-11 10:45:00 | 385.95 | 2026-02-11 12:00:00 | 384.80 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-17 10:25:00 | 379.00 | 2026-02-17 10:30:00 | 378.09 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-25 10:00:00 | 380.95 | 2026-02-25 10:35:00 | 382.87 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-25 10:00:00 | 380.95 | 2026-02-25 13:25:00 | 381.65 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2026-02-26 09:55:00 | 385.35 | 2026-02-26 10:05:00 | 386.82 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-26 09:55:00 | 385.35 | 2026-02-26 13:55:00 | 388.75 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2026-03-11 11:15:00 | 339.70 | 2026-03-11 13:00:00 | 338.28 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-11 11:15:00 | 339.70 | 2026-03-11 15:20:00 | 334.95 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2026-03-13 09:35:00 | 317.90 | 2026-03-13 09:45:00 | 316.18 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-13 09:35:00 | 317.90 | 2026-03-13 15:10:00 | 314.35 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2026-03-18 09:45:00 | 327.15 | 2026-03-18 09:55:00 | 325.79 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-15 10:45:00 | 360.65 | 2026-04-15 10:50:00 | 359.32 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-20 09:30:00 | 352.80 | 2026-04-20 09:50:00 | 354.14 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-23 10:55:00 | 355.80 | 2026-04-23 12:50:00 | 354.07 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-23 10:55:00 | 355.80 | 2026-04-23 15:20:00 | 351.50 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2026-05-05 09:40:00 | 343.10 | 2026-05-05 10:00:00 | 341.98 | STOP_HIT | 1.00 | -0.33% |
