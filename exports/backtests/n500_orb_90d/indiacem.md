# India Cements Ltd. (INDIACEM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 408.00
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
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 12
- **Target hits / Stop hits / Partials:** 1 / 12 / 5
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 2.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 4 | 30.8% | 0 | 9 | 4 | 0.03% | 0.4% |
| BUY @ 2nd Alert (retest1) | 13 | 4 | 30.8% | 0 | 9 | 4 | 0.03% | 0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.35% | 1.7% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.35% | 1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 6 | 33.3% | 1 | 12 | 5 | 0.12% | 2.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:50:00 | 461.25 | 458.89 | 0.00 | ORB-long ORB[456.00,460.05] vol=4.9x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 12:20:00 | 463.35 | 460.79 | 0.00 | T1 1.5R @ 463.35 |
| Stop hit — per-position SL triggered | 2026-02-10 14:00:00 | 461.25 | 461.22 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:05:00 | 431.50 | 434.63 | 0.00 | ORB-short ORB[433.25,437.25] vol=2.8x ATR=1.19 |
| Stop hit — per-position SL triggered | 2026-02-17 11:35:00 | 432.69 | 434.38 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 426.80 | 425.14 | 0.00 | ORB-long ORB[422.75,425.75] vol=1.6x ATR=1.07 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 425.73 | 425.37 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:40:00 | 414.45 | 418.12 | 0.00 | ORB-short ORB[419.40,425.00] vol=6.0x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:55:00 | 412.23 | 415.99 | 0.00 | T1 1.5R @ 412.23 |
| Target hit | 2026-02-27 15:20:00 | 405.55 | 409.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 386.40 | 390.44 | 0.00 | ORB-short ORB[390.00,395.00] vol=1.5x ATR=1.46 |
| Stop hit — per-position SL triggered | 2026-03-06 11:35:00 | 387.86 | 389.78 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:55:00 | 359.70 | 357.24 | 0.00 | ORB-long ORB[353.10,357.00] vol=3.9x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:20:00 | 362.20 | 358.66 | 0.00 | T1 1.5R @ 362.20 |
| Stop hit — per-position SL triggered | 2026-03-17 12:05:00 | 359.70 | 359.62 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:30:00 | 360.55 | 359.18 | 0.00 | ORB-long ORB[356.55,360.45] vol=2.3x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:15:00 | 362.23 | 359.53 | 0.00 | T1 1.5R @ 362.23 |
| Stop hit — per-position SL triggered | 2026-03-20 11:45:00 | 360.55 | 359.77 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 10:15:00 | 358.30 | 355.40 | 0.00 | ORB-long ORB[352.95,358.05] vol=1.8x ATR=1.81 |
| Stop hit — per-position SL triggered | 2026-04-06 10:30:00 | 356.49 | 356.50 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:40:00 | 385.00 | 383.76 | 0.00 | ORB-long ORB[380.00,384.65] vol=4.7x ATR=1.25 |
| Stop hit — per-position SL triggered | 2026-04-10 10:50:00 | 383.75 | 383.83 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:20:00 | 390.00 | 388.04 | 0.00 | ORB-long ORB[385.05,389.80] vol=1.5x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-04-15 10:40:00 | 388.87 | 388.14 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:05:00 | 402.50 | 397.77 | 0.00 | ORB-long ORB[394.10,399.00] vol=1.7x ATR=1.19 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 401.31 | 398.24 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:00:00 | 424.15 | 422.04 | 0.00 | ORB-long ORB[420.05,423.40] vol=3.2x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:05:00 | 426.22 | 423.50 | 0.00 | T1 1.5R @ 426.22 |
| Stop hit — per-position SL triggered | 2026-04-23 10:10:00 | 424.15 | 423.55 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:45:00 | 390.25 | 393.48 | 0.00 | ORB-short ORB[394.40,399.75] vol=1.7x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-05-06 10:50:00 | 391.40 | 393.29 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:50:00 | 461.25 | 2026-02-10 12:20:00 | 463.35 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-10 10:50:00 | 461.25 | 2026-02-10 14:00:00 | 461.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 11:05:00 | 431.50 | 2026-02-17 11:35:00 | 432.69 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-26 10:55:00 | 426.80 | 2026-02-26 11:30:00 | 425.73 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-27 09:40:00 | 414.45 | 2026-02-27 09:55:00 | 412.23 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-27 09:40:00 | 414.45 | 2026-02-27 15:20:00 | 405.55 | TARGET_HIT | 0.50 | 2.15% |
| SELL | retest1 | 2026-03-06 10:45:00 | 386.40 | 2026-03-06 11:35:00 | 387.86 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-17 09:55:00 | 359.70 | 2026-03-17 10:20:00 | 362.20 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-17 09:55:00 | 359.70 | 2026-03-17 12:05:00 | 359.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 10:30:00 | 360.55 | 2026-03-20 11:15:00 | 362.23 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-03-20 10:30:00 | 360.55 | 2026-03-20 11:45:00 | 360.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-06 10:15:00 | 358.30 | 2026-04-06 10:30:00 | 356.49 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-04-10 10:40:00 | 385.00 | 2026-04-10 10:50:00 | 383.75 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-15 10:20:00 | 390.00 | 2026-04-15 10:40:00 | 388.87 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-16 11:05:00 | 402.50 | 2026-04-16 11:25:00 | 401.31 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-23 10:00:00 | 424.15 | 2026-04-23 10:05:00 | 426.22 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-23 10:00:00 | 424.15 | 2026-04-23 10:10:00 | 424.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 10:45:00 | 390.25 | 2026-05-06 10:50:00 | 391.40 | STOP_HIT | 1.00 | -0.30% |
