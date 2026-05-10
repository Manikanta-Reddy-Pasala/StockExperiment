# Zydus Wellness Ltd. (ZYDUSWELL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 517.00
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 5
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 3.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.01% | 0.1% |
| BUY @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 1 | 7 | 2 | 0.01% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 5 | 71.4% | 2 | 2 | 3 | 0.55% | 3.8% |
| SELL @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 2 | 2 | 3 | 0.55% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 8 | 47.1% | 3 | 9 | 5 | 0.23% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:50:00 | 414.10 | 410.83 | 0.00 | ORB-long ORB[407.20,411.90] vol=1.6x ATR=1.91 |
| Stop hit — per-position SL triggered | 2026-02-10 10:05:00 | 412.19 | 411.37 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 404.90 | 402.61 | 0.00 | ORB-long ORB[399.25,403.20] vol=1.7x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:55:00 | 406.88 | 403.75 | 0.00 | T1 1.5R @ 406.88 |
| Stop hit — per-position SL triggered | 2026-02-17 10:20:00 | 404.90 | 404.37 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 402.40 | 404.39 | 0.00 | ORB-short ORB[403.20,407.75] vol=2.7x ATR=1.36 |
| Stop hit — per-position SL triggered | 2026-02-18 11:25:00 | 403.76 | 403.17 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:45:00 | 398.35 | 399.13 | 0.00 | ORB-short ORB[398.80,402.45] vol=5.9x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:55:00 | 396.67 | 397.96 | 0.00 | T1 1.5R @ 396.67 |
| Target hit | 2026-02-25 15:20:00 | 393.50 | 395.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:10:00 | 389.75 | 392.90 | 0.00 | ORB-short ORB[392.00,396.50] vol=4.0x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:30:00 | 387.42 | 391.52 | 0.00 | T1 1.5R @ 387.42 |
| Target hit | 2026-02-26 15:20:00 | 385.00 | 387.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-03-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:40:00 | 384.75 | 383.80 | 0.00 | ORB-long ORB[380.25,382.90] vol=1.7x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-03-06 10:45:00 | 383.60 | 383.80 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:40:00 | 380.50 | 377.15 | 0.00 | ORB-long ORB[374.25,378.90] vol=2.2x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-03-10 11:05:00 | 378.87 | 377.39 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 09:35:00 | 388.30 | 386.63 | 0.00 | ORB-long ORB[384.00,387.75] vol=1.5x ATR=1.88 |
| Stop hit — per-position SL triggered | 2026-03-13 09:45:00 | 386.42 | 386.63 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:40:00 | 505.95 | 510.30 | 0.00 | ORB-short ORB[507.95,514.45] vol=1.5x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:45:00 | 502.38 | 509.32 | 0.00 | T1 1.5R @ 502.38 |
| Stop hit — per-position SL triggered | 2026-04-16 11:05:00 | 505.95 | 506.45 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:25:00 | 513.00 | 507.04 | 0.00 | ORB-long ORB[500.00,505.85] vol=2.8x ATR=2.20 |
| Stop hit — per-position SL triggered | 2026-04-23 10:30:00 | 510.80 | 507.29 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:30:00 | 495.45 | 492.51 | 0.00 | ORB-long ORB[488.65,493.60] vol=1.8x ATR=1.93 |
| Stop hit — per-position SL triggered | 2026-04-29 09:35:00 | 493.52 | 492.88 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:40:00 | 508.40 | 504.61 | 0.00 | ORB-long ORB[501.55,507.75] vol=1.5x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:30:00 | 511.16 | 507.76 | 0.00 | T1 1.5R @ 511.16 |
| Target hit | 2026-05-08 15:20:00 | 516.40 | 514.96 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:50:00 | 414.10 | 2026-02-10 10:05:00 | 412.19 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-02-17 09:40:00 | 404.90 | 2026-02-17 09:55:00 | 406.88 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-17 09:40:00 | 404.90 | 2026-02-17 10:20:00 | 404.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 09:55:00 | 402.40 | 2026-02-18 11:25:00 | 403.76 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-25 10:45:00 | 398.35 | 2026-02-25 11:55:00 | 396.67 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-25 10:45:00 | 398.35 | 2026-02-25 15:20:00 | 393.50 | TARGET_HIT | 0.50 | 1.22% |
| SELL | retest1 | 2026-02-26 10:10:00 | 389.75 | 2026-02-26 11:30:00 | 387.42 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-26 10:10:00 | 389.75 | 2026-02-26 15:20:00 | 385.00 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2026-03-06 10:40:00 | 384.75 | 2026-03-06 10:45:00 | 383.60 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-10 10:40:00 | 380.50 | 2026-03-10 11:05:00 | 378.87 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-13 09:35:00 | 388.30 | 2026-03-13 09:45:00 | 386.42 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-04-16 09:40:00 | 505.95 | 2026-04-16 09:45:00 | 502.38 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-16 09:40:00 | 505.95 | 2026-04-16 11:05:00 | 505.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 10:25:00 | 513.00 | 2026-04-23 10:30:00 | 510.80 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-29 09:30:00 | 495.45 | 2026-04-29 09:35:00 | 493.52 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-08 10:40:00 | 508.40 | 2026-05-08 11:30:00 | 511.16 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-05-08 10:40:00 | 508.40 | 2026-05-08 15:20:00 | 516.40 | TARGET_HIT | 0.50 | 1.57% |
