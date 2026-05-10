# NTPC Ltd. (NTPC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 402.10
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 14
- **Target hits / Stop hits / Partials:** 3 / 14 / 8
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 3.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 8 | 50.0% | 2 | 8 | 6 | 0.26% | 4.2% |
| BUY @ 2nd Alert (retest1) | 16 | 8 | 50.0% | 2 | 8 | 6 | 0.26% | 4.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.03% | -0.3% |
| SELL @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.03% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 11 | 44.0% | 3 | 14 | 8 | 0.16% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:15:00 | 364.20 | 365.82 | 0.00 | ORB-short ORB[364.65,368.00] vol=2.4x ATR=0.69 |
| Stop hit — per-position SL triggered | 2026-02-13 11:25:00 | 364.89 | 365.64 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:30:00 | 369.05 | 366.87 | 0.00 | ORB-long ORB[362.70,366.35] vol=2.2x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:05:00 | 370.27 | 368.08 | 0.00 | T1 1.5R @ 370.27 |
| Target hit | 2026-02-20 15:20:00 | 373.15 | 371.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:00:00 | 382.05 | 381.67 | 0.00 | ORB-long ORB[376.45,382.00] vol=2.3x ATR=0.96 |
| Stop hit — per-position SL triggered | 2026-03-11 11:15:00 | 381.09 | 381.65 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:10:00 | 383.25 | 380.36 | 0.00 | ORB-long ORB[375.65,380.55] vol=2.9x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:30:00 | 385.26 | 381.95 | 0.00 | T1 1.5R @ 385.26 |
| Target hit | 2026-03-12 15:20:00 | 390.40 | 387.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 381.90 | 383.16 | 0.00 | ORB-short ORB[384.00,386.35] vol=2.9x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:30:00 | 380.73 | 382.90 | 0.00 | T1 1.5R @ 380.73 |
| Target hit | 2026-03-18 13:30:00 | 381.30 | 381.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2026-03-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:55:00 | 371.50 | 374.60 | 0.00 | ORB-short ORB[375.55,379.50] vol=1.5x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:05:00 | 369.75 | 373.56 | 0.00 | T1 1.5R @ 369.75 |
| Stop hit — per-position SL triggered | 2026-03-23 12:35:00 | 371.50 | 373.17 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:00:00 | 371.90 | 374.23 | 0.00 | ORB-short ORB[374.20,378.00] vol=2.3x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-03-24 11:10:00 | 373.18 | 373.89 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:10:00 | 378.45 | 377.17 | 0.00 | ORB-long ORB[375.60,378.05] vol=2.4x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 12:10:00 | 379.71 | 377.62 | 0.00 | T1 1.5R @ 379.71 |
| Stop hit — per-position SL triggered | 2026-03-25 12:50:00 | 378.45 | 377.83 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 11:05:00 | 367.85 | 367.32 | 0.00 | ORB-long ORB[362.25,366.00] vol=3.4x ATR=1.27 |
| Stop hit — per-position SL triggered | 2026-04-07 11:40:00 | 366.58 | 367.36 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 10:50:00 | 381.65 | 380.45 | 0.00 | ORB-long ORB[376.50,381.30] vol=4.4x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:20:00 | 383.30 | 380.81 | 0.00 | T1 1.5R @ 383.30 |
| Stop hit — per-position SL triggered | 2026-04-09 13:10:00 | 381.65 | 381.21 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:40:00 | 380.90 | 378.37 | 0.00 | ORB-long ORB[374.35,379.40] vol=2.7x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-04-13 09:50:00 | 379.37 | 379.03 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 11:00:00 | 397.45 | 395.51 | 0.00 | ORB-long ORB[390.65,395.45] vol=1.5x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 11:05:00 | 398.82 | 396.28 | 0.00 | T1 1.5R @ 398.82 |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 397.45 | 397.96 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 400.95 | 402.71 | 0.00 | ORB-short ORB[402.50,405.90] vol=6.3x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-04-23 11:45:00 | 401.82 | 402.39 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 399.80 | 400.92 | 0.00 | ORB-short ORB[402.40,405.25] vol=2.0x ATR=0.84 |
| Stop hit — per-position SL triggered | 2026-04-24 11:30:00 | 400.64 | 400.84 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:15:00 | 410.60 | 406.76 | 0.00 | ORB-long ORB[402.60,406.80] vol=1.9x ATR=1.17 |
| Stop hit — per-position SL triggered | 2026-04-27 10:25:00 | 409.43 | 407.28 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 396.10 | 398.03 | 0.00 | ORB-short ORB[397.25,400.50] vol=1.8x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 397.00 | 397.91 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:50:00 | 400.00 | 396.85 | 0.00 | ORB-long ORB[393.15,397.05] vol=1.8x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 12:35:00 | 402.17 | 398.43 | 0.00 | T1 1.5R @ 402.17 |
| Stop hit — per-position SL triggered | 2026-05-07 13:15:00 | 400.00 | 398.96 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 11:15:00 | 364.20 | 2026-02-13 11:25:00 | 364.89 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-20 10:30:00 | 369.05 | 2026-02-20 11:05:00 | 370.27 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-20 10:30:00 | 369.05 | 2026-02-20 15:20:00 | 373.15 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2026-03-11 11:00:00 | 382.05 | 2026-03-11 11:15:00 | 381.09 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-03-12 10:10:00 | 383.25 | 2026-03-12 10:30:00 | 385.26 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-12 10:10:00 | 383.25 | 2026-03-12 15:20:00 | 390.40 | TARGET_HIT | 0.50 | 1.87% |
| SELL | retest1 | 2026-03-18 11:00:00 | 381.90 | 2026-03-18 11:30:00 | 380.73 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-03-18 11:00:00 | 381.90 | 2026-03-18 13:30:00 | 381.30 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2026-03-23 10:55:00 | 371.50 | 2026-03-23 12:05:00 | 369.75 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-23 10:55:00 | 371.50 | 2026-03-23 12:35:00 | 371.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-24 11:00:00 | 371.90 | 2026-03-24 11:10:00 | 373.18 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-25 11:10:00 | 378.45 | 2026-03-25 12:10:00 | 379.71 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-03-25 11:10:00 | 378.45 | 2026-03-25 12:50:00 | 378.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-07 11:05:00 | 367.85 | 2026-04-07 11:40:00 | 366.58 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-09 10:50:00 | 381.65 | 2026-04-09 11:20:00 | 383.30 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-04-09 10:50:00 | 381.65 | 2026-04-09 13:10:00 | 381.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 09:40:00 | 380.90 | 2026-04-13 09:50:00 | 379.37 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-20 11:00:00 | 397.45 | 2026-04-20 11:05:00 | 398.82 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-20 11:00:00 | 397.45 | 2026-04-20 15:15:00 | 397.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 11:10:00 | 400.95 | 2026-04-23 11:45:00 | 401.82 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-24 11:10:00 | 399.80 | 2026-04-24 11:30:00 | 400.64 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-04-27 10:15:00 | 410.60 | 2026-04-27 10:25:00 | 409.43 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-05 11:05:00 | 396.10 | 2026-05-05 11:15:00 | 397.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-05-07 10:50:00 | 400.00 | 2026-05-07 12:35:00 | 402.17 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-05-07 10:50:00 | 400.00 | 2026-05-07 13:15:00 | 400.00 | STOP_HIT | 0.50 | 0.00% |
