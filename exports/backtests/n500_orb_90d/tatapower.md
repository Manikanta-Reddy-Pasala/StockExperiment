# Tata Power Co. Ltd. (TATAPOWER)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 435.50
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
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 16
- **Target hits / Stop hits / Partials:** 3 / 16 / 8
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 4.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 9 | 56.2% | 3 | 7 | 6 | 0.35% | 5.5% |
| BUY @ 2nd Alert (retest1) | 16 | 9 | 56.2% | 3 | 7 | 6 | 0.35% | 5.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 2 | 18.2% | 0 | 9 | 2 | -0.12% | -1.3% |
| SELL @ 2nd Alert (retest1) | 11 | 2 | 18.2% | 0 | 9 | 2 | -0.12% | -1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 11 | 40.7% | 3 | 16 | 8 | 0.15% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:00:00 | 373.55 | 375.31 | 0.00 | ORB-short ORB[374.45,377.40] vol=1.7x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:15:00 | 372.29 | 375.09 | 0.00 | T1 1.5R @ 372.29 |
| Stop hit — per-position SL triggered | 2026-02-12 12:45:00 | 373.55 | 373.92 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:20:00 | 372.35 | 374.12 | 0.00 | ORB-short ORB[374.70,377.80] vol=3.7x ATR=1.08 |
| Stop hit — per-position SL triggered | 2026-02-13 10:35:00 | 373.43 | 373.88 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 377.70 | 376.07 | 0.00 | ORB-long ORB[370.90,376.10] vol=1.5x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:35:00 | 378.93 | 376.49 | 0.00 | T1 1.5R @ 378.93 |
| Stop hit — per-position SL triggered | 2026-02-16 11:50:00 | 377.70 | 376.58 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 373.85 | 371.70 | 0.00 | ORB-long ORB[369.25,372.20] vol=1.7x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 12:40:00 | 375.27 | 372.93 | 0.00 | T1 1.5R @ 375.27 |
| Target hit | 2026-02-20 15:20:00 | 377.40 | 375.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 378.75 | 380.73 | 0.00 | ORB-short ORB[378.90,383.65] vol=1.5x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:30:00 | 377.51 | 380.27 | 0.00 | T1 1.5R @ 377.51 |
| Stop hit — per-position SL triggered | 2026-02-25 13:30:00 | 378.75 | 379.67 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:35:00 | 383.40 | 382.16 | 0.00 | ORB-long ORB[380.05,382.45] vol=5.3x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-02-26 11:35:00 | 382.49 | 382.53 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:05:00 | 380.25 | 378.03 | 0.00 | ORB-long ORB[375.90,379.50] vol=1.8x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 10:25:00 | 382.09 | 379.31 | 0.00 | T1 1.5R @ 382.09 |
| Target hit | 2026-03-10 11:45:00 | 380.45 | 380.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2026-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:35:00 | 397.05 | 393.50 | 0.00 | ORB-long ORB[391.00,394.00] vol=2.0x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 09:40:00 | 399.77 | 395.17 | 0.00 | T1 1.5R @ 399.77 |
| Stop hit — per-position SL triggered | 2026-03-17 09:55:00 | 397.05 | 395.98 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 11:05:00 | 385.50 | 382.55 | 0.00 | ORB-long ORB[378.45,383.00] vol=2.8x ATR=1.14 |
| Stop hit — per-position SL triggered | 2026-04-07 11:10:00 | 384.36 | 382.67 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:50:00 | 419.80 | 414.54 | 0.00 | ORB-long ORB[410.30,416.30] vol=3.2x ATR=1.66 |
| Stop hit — per-position SL triggered | 2026-04-15 09:55:00 | 418.14 | 415.22 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 419.40 | 422.71 | 0.00 | ORB-short ORB[421.75,427.40] vol=1.7x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-04-16 11:40:00 | 420.82 | 422.15 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 426.90 | 427.58 | 0.00 | ORB-short ORB[427.15,432.00] vol=1.9x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-04-17 13:35:00 | 428.05 | 427.33 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 431.50 | 434.23 | 0.00 | ORB-short ORB[432.40,437.80] vol=3.9x ATR=0.95 |
| Stop hit — per-position SL triggered | 2026-04-23 11:25:00 | 432.45 | 434.10 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:05:00 | 427.50 | 429.10 | 0.00 | ORB-short ORB[428.70,433.35] vol=1.6x ATR=1.24 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 428.74 | 429.06 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 440.80 | 438.06 | 0.00 | ORB-long ORB[435.20,439.20] vol=2.0x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 443.20 | 439.50 | 0.00 | T1 1.5R @ 443.20 |
| Target hit | 2026-04-27 15:20:00 | 453.90 | 449.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 459.40 | 457.17 | 0.00 | ORB-long ORB[454.00,458.40] vol=2.5x ATR=1.57 |
| Stop hit — per-position SL triggered | 2026-04-28 10:05:00 | 457.83 | 458.47 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:35:00 | 443.00 | 444.92 | 0.00 | ORB-short ORB[444.00,447.80] vol=1.9x ATR=1.51 |
| Stop hit — per-position SL triggered | 2026-05-04 10:45:00 | 444.51 | 444.70 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:35:00 | 441.80 | 440.35 | 0.00 | ORB-long ORB[437.25,441.40] vol=2.3x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:50:00 | 443.59 | 441.45 | 0.00 | T1 1.5R @ 443.59 |
| Stop hit — per-position SL triggered | 2026-05-05 10:35:00 | 441.80 | 442.61 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:15:00 | 440.15 | 441.61 | 0.00 | ORB-short ORB[442.20,446.00] vol=2.5x ATR=1.16 |
| Stop hit — per-position SL triggered | 2026-05-07 10:25:00 | 441.31 | 441.58 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 11:00:00 | 373.55 | 2026-02-12 11:15:00 | 372.29 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-12 11:00:00 | 373.55 | 2026-02-12 12:45:00 | 373.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:20:00 | 372.35 | 2026-02-13 10:35:00 | 373.43 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-16 11:15:00 | 377.70 | 2026-02-16 11:35:00 | 378.93 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-16 11:15:00 | 377.70 | 2026-02-16 11:50:00 | 377.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:45:00 | 373.85 | 2026-02-20 12:40:00 | 375.27 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-20 10:45:00 | 373.85 | 2026-02-20 15:20:00 | 377.40 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2026-02-25 11:05:00 | 378.75 | 2026-02-25 12:30:00 | 377.51 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-25 11:05:00 | 378.75 | 2026-02-25 13:30:00 | 378.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 10:35:00 | 383.40 | 2026-02-26 11:35:00 | 382.49 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-03-10 10:05:00 | 380.25 | 2026-03-10 10:25:00 | 382.09 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-03-10 10:05:00 | 380.25 | 2026-03-10 11:45:00 | 380.45 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2026-03-17 09:35:00 | 397.05 | 2026-03-17 09:40:00 | 399.77 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-03-17 09:35:00 | 397.05 | 2026-03-17 09:55:00 | 397.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-07 11:05:00 | 385.50 | 2026-04-07 11:10:00 | 384.36 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-15 09:50:00 | 419.80 | 2026-04-15 09:55:00 | 418.14 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-16 11:00:00 | 419.40 | 2026-04-16 11:40:00 | 420.82 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-17 11:00:00 | 426.90 | 2026-04-17 13:35:00 | 428.05 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-23 11:10:00 | 431.50 | 2026-04-23 11:25:00 | 432.45 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-24 11:05:00 | 427.50 | 2026-04-24 11:20:00 | 428.74 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-27 09:45:00 | 440.80 | 2026-04-27 09:50:00 | 443.20 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-27 09:45:00 | 440.80 | 2026-04-27 15:20:00 | 453.90 | TARGET_HIT | 0.50 | 2.97% |
| BUY | retest1 | 2026-04-28 09:30:00 | 459.40 | 2026-04-28 10:05:00 | 457.83 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-05-04 10:35:00 | 443.00 | 2026-05-04 10:45:00 | 444.51 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-05 09:35:00 | 441.80 | 2026-05-05 09:50:00 | 443.59 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-05-05 09:35:00 | 441.80 | 2026-05-05 10:35:00 | 441.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 10:15:00 | 440.15 | 2026-05-07 10:25:00 | 441.31 | STOP_HIT | 1.00 | -0.26% |
