# Biocon Ltd. (BIOCON)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 378.70
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 16
- **Target hits / Stop hits / Partials:** 3 / 16 / 5
- **Avg / median % per leg:** 0.02% / -0.22%
- **Sum % (uncompounded):** 0.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 8 | 44.4% | 3 | 10 | 5 | 0.13% | 2.3% |
| BUY @ 2nd Alert (retest1) | 18 | 8 | 44.4% | 3 | 10 | 5 | 0.13% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.30% | -1.8% |
| SELL @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.30% | -1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 8 | 33.3% | 3 | 16 | 5 | 0.02% | 0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 374.10 | 370.49 | 0.00 | ORB-long ORB[366.40,369.05] vol=2.4x ATR=1.79 |
| Target hit | 2026-02-09 15:20:00 | 374.55 | 372.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 371.30 | 372.39 | 0.00 | ORB-short ORB[371.45,375.00] vol=1.6x ATR=0.95 |
| Stop hit — per-position SL triggered | 2026-02-10 11:10:00 | 372.25 | 372.35 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:00:00 | 383.20 | 377.05 | 0.00 | ORB-long ORB[371.80,377.40] vol=3.5x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:15:00 | 385.80 | 381.46 | 0.00 | T1 1.5R @ 385.80 |
| Stop hit — per-position SL triggered | 2026-02-12 10:30:00 | 383.20 | 382.48 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 376.90 | 379.68 | 0.00 | ORB-short ORB[378.75,382.90] vol=1.8x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-02-18 11:35:00 | 377.77 | 379.18 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:50:00 | 384.45 | 380.87 | 0.00 | ORB-long ORB[378.50,382.40] vol=5.9x ATR=1.05 |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 383.40 | 382.22 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:35:00 | 387.70 | 384.19 | 0.00 | ORB-long ORB[381.65,384.50] vol=1.9x ATR=1.46 |
| Stop hit — per-position SL triggered | 2026-02-20 10:05:00 | 386.24 | 385.93 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 399.65 | 396.18 | 0.00 | ORB-long ORB[392.30,396.30] vol=1.8x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 398.18 | 396.85 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:05:00 | 390.00 | 390.62 | 0.00 | ORB-short ORB[391.05,395.70] vol=3.4x ATR=1.32 |
| Stop hit — per-position SL triggered | 2026-02-27 10:35:00 | 391.32 | 390.27 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 401.40 | 398.50 | 0.00 | ORB-long ORB[394.45,399.20] vol=2.2x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-03-11 13:45:00 | 400.06 | 400.34 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:25:00 | 375.70 | 382.51 | 0.00 | ORB-short ORB[382.90,387.00] vol=1.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-03-16 10:30:00 | 377.37 | 381.91 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:35:00 | 384.35 | 382.28 | 0.00 | ORB-long ORB[377.00,382.50] vol=1.9x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:40:00 | 385.86 | 383.33 | 0.00 | T1 1.5R @ 385.86 |
| Stop hit — per-position SL triggered | 2026-03-18 11:45:00 | 384.35 | 383.34 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:10:00 | 371.45 | 375.29 | 0.00 | ORB-short ORB[373.80,378.95] vol=2.0x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-03-27 11:35:00 | 372.63 | 374.30 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 354.50 | 352.16 | 0.00 | ORB-long ORB[349.15,352.80] vol=1.7x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:20:00 | 355.72 | 353.26 | 0.00 | T1 1.5R @ 355.72 |
| Target hit | 2026-04-17 15:20:00 | 358.30 | 355.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 364.50 | 362.42 | 0.00 | ORB-long ORB[359.05,364.00] vol=1.6x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-04-21 10:40:00 | 363.39 | 362.83 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 358.75 | 359.42 | 0.00 | ORB-short ORB[358.85,362.50] vol=4.0x ATR=0.80 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 359.55 | 359.26 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:40:00 | 363.30 | 359.74 | 0.00 | ORB-long ORB[355.60,359.50] vol=2.2x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-04-23 10:00:00 | 362.24 | 360.98 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:40:00 | 364.70 | 363.19 | 0.00 | ORB-long ORB[359.20,364.30] vol=1.9x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:30:00 | 366.49 | 363.74 | 0.00 | T1 1.5R @ 366.49 |
| Stop hit — per-position SL triggered | 2026-05-04 12:45:00 | 364.70 | 364.21 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 10:30:00 | 365.05 | 362.42 | 0.00 | ORB-long ORB[359.50,362.30] vol=1.9x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-05-05 10:40:00 | 363.87 | 362.76 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:50:00 | 376.30 | 373.83 | 0.00 | ORB-long ORB[371.05,375.45] vol=1.6x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:55:00 | 378.16 | 375.72 | 0.00 | T1 1.5R @ 378.16 |
| Target hit | 2026-05-06 15:20:00 | 380.10 | 380.26 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 374.10 | 2026-02-09 15:20:00 | 374.55 | TARGET_HIT | 1.00 | 0.12% |
| SELL | retest1 | 2026-02-10 11:00:00 | 371.30 | 2026-02-10 11:10:00 | 372.25 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-12 10:00:00 | 383.20 | 2026-02-12 10:15:00 | 385.80 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-02-12 10:00:00 | 383.20 | 2026-02-12 10:30:00 | 383.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 10:50:00 | 376.90 | 2026-02-18 11:35:00 | 377.77 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-19 10:50:00 | 384.45 | 2026-02-19 11:15:00 | 383.40 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-20 09:35:00 | 387.70 | 2026-02-20 10:05:00 | 386.24 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-26 09:45:00 | 399.65 | 2026-02-26 09:55:00 | 398.18 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-27 10:05:00 | 390.00 | 2026-02-27 10:35:00 | 391.32 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-11 10:55:00 | 401.40 | 2026-03-11 13:45:00 | 400.06 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-16 10:25:00 | 375.70 | 2026-03-16 10:30:00 | 377.37 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-18 10:35:00 | 384.35 | 2026-03-18 11:40:00 | 385.86 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-03-18 10:35:00 | 384.35 | 2026-03-18 11:45:00 | 384.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 11:10:00 | 371.45 | 2026-03-27 11:35:00 | 372.63 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-17 11:00:00 | 354.50 | 2026-04-17 12:20:00 | 355.72 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-04-17 11:00:00 | 354.50 | 2026-04-17 15:20:00 | 358.30 | TARGET_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2026-04-21 10:10:00 | 364.50 | 2026-04-21 10:40:00 | 363.39 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-22 10:45:00 | 358.75 | 2026-04-22 11:05:00 | 359.55 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-23 09:40:00 | 363.30 | 2026-04-23 10:00:00 | 362.24 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-04 10:40:00 | 364.70 | 2026-05-04 11:30:00 | 366.49 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-05-04 10:40:00 | 364.70 | 2026-05-04 12:45:00 | 364.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 10:30:00 | 365.05 | 2026-05-05 10:40:00 | 363.87 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-05-06 09:50:00 | 376.30 | 2026-05-06 09:55:00 | 378.16 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-05-06 09:50:00 | 376.30 | 2026-05-06 15:20:00 | 380.10 | TARGET_HIT | 0.50 | 1.01% |
