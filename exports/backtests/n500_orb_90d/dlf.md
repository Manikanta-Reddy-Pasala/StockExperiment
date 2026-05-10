# DLF Ltd. (DLF)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 606.30
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
| ENTRY1 | 21 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 18
- **Target hits / Stop hits / Partials:** 3 / 18 / 6
- **Avg / median % per leg:** 0.14% / -0.25%
- **Sum % (uncompounded):** 3.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 3 | 27.3% | 1 | 8 | 2 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 11 | 3 | 27.3% | 1 | 8 | 2 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.23% | 3.7% |
| SELL @ 2nd Alert (retest1) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.23% | 3.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 9 | 33.3% | 3 | 18 | 6 | 0.14% | 3.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 675.50 | 672.31 | 0.00 | ORB-long ORB[668.65,674.30] vol=6.0x ATR=1.93 |
| Stop hit — per-position SL triggered | 2026-02-10 09:45:00 | 673.57 | 672.52 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 666.00 | 669.39 | 0.00 | ORB-short ORB[668.50,674.40] vol=1.6x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:00:00 | 663.70 | 667.40 | 0.00 | T1 1.5R @ 663.70 |
| Stop hit — per-position SL triggered | 2026-02-11 10:20:00 | 666.00 | 666.71 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:30:00 | 662.60 | 664.19 | 0.00 | ORB-short ORB[662.80,670.25] vol=2.8x ATR=1.72 |
| Stop hit — per-position SL triggered | 2026-02-12 09:55:00 | 664.32 | 663.27 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 637.05 | 640.00 | 0.00 | ORB-short ORB[638.10,644.45] vol=1.6x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:10:00 | 634.65 | 638.90 | 0.00 | T1 1.5R @ 634.65 |
| Target hit | 2026-02-19 15:20:00 | 618.70 | 629.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:35:00 | 625.35 | 627.76 | 0.00 | ORB-short ORB[627.00,634.00] vol=2.6x ATR=1.76 |
| Stop hit — per-position SL triggered | 2026-02-23 10:50:00 | 627.11 | 627.57 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:55:00 | 618.30 | 621.38 | 0.00 | ORB-short ORB[619.65,623.90] vol=2.6x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-02-24 11:05:00 | 619.67 | 621.06 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:50:00 | 615.30 | 618.19 | 0.00 | ORB-short ORB[615.70,623.85] vol=1.5x ATR=2.25 |
| Stop hit — per-position SL triggered | 2026-02-25 10:00:00 | 617.55 | 618.02 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:50:00 | 614.20 | 612.26 | 0.00 | ORB-long ORB[608.75,614.00] vol=1.8x ATR=1.80 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 612.40 | 612.26 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:30:00 | 577.90 | 581.35 | 0.00 | ORB-short ORB[578.95,586.15] vol=1.6x ATR=2.83 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 580.73 | 579.73 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:30:00 | 593.65 | 589.97 | 0.00 | ORB-long ORB[584.60,590.90] vol=2.6x ATR=2.03 |
| Stop hit — per-position SL triggered | 2026-03-11 09:35:00 | 591.62 | 590.69 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:05:00 | 536.20 | 534.55 | 0.00 | ORB-long ORB[529.70,533.30] vol=2.2x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-03-17 11:25:00 | 534.24 | 535.12 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:45:00 | 555.35 | 551.64 | 0.00 | ORB-long ORB[546.15,551.35] vol=1.6x ATR=2.72 |
| Stop hit — per-position SL triggered | 2026-03-20 10:25:00 | 552.63 | 553.50 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:35:00 | 511.55 | 515.66 | 0.00 | ORB-short ORB[512.80,520.40] vol=3.1x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 11:00:00 | 508.01 | 514.93 | 0.00 | T1 1.5R @ 508.01 |
| Target hit | 2026-03-30 15:20:00 | 503.35 | 508.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 570.00 | 567.93 | 0.00 | ORB-long ORB[564.20,569.95] vol=1.6x ATR=2.26 |
| Stop hit — per-position SL triggered | 2026-04-10 09:50:00 | 567.74 | 568.19 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 575.80 | 578.67 | 0.00 | ORB-short ORB[576.30,583.75] vol=1.7x ATR=2.54 |
| Stop hit — per-position SL triggered | 2026-04-15 09:40:00 | 578.34 | 578.67 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:00:00 | 594.10 | 590.45 | 0.00 | ORB-long ORB[587.20,592.70] vol=2.2x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:50:00 | 597.22 | 593.01 | 0.00 | T1 1.5R @ 597.22 |
| Target hit | 2026-04-17 15:20:00 | 602.50 | 597.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2026-04-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:50:00 | 611.60 | 608.89 | 0.00 | ORB-long ORB[605.45,611.40] vol=2.1x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-04-22 11:00:00 | 610.04 | 609.08 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 595.60 | 601.50 | 0.00 | ORB-short ORB[601.00,605.00] vol=2.3x ATR=1.50 |
| Stop hit — per-position SL triggered | 2026-04-23 11:40:00 | 597.10 | 600.57 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-04-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:50:00 | 586.15 | 588.98 | 0.00 | ORB-short ORB[588.15,595.50] vol=1.6x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:00:00 | 583.93 | 588.68 | 0.00 | T1 1.5R @ 583.93 |
| Stop hit — per-position SL triggered | 2026-04-24 11:30:00 | 586.15 | 588.37 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 589.50 | 591.79 | 0.00 | ORB-short ORB[591.05,595.70] vol=1.9x ATR=1.68 |
| Stop hit — per-position SL triggered | 2026-04-28 10:00:00 | 591.18 | 591.39 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 598.95 | 594.21 | 0.00 | ORB-long ORB[589.35,595.20] vol=2.8x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:00:00 | 602.01 | 596.05 | 0.00 | T1 1.5R @ 602.01 |
| Stop hit — per-position SL triggered | 2026-04-29 10:55:00 | 598.95 | 597.42 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 675.50 | 2026-02-10 09:45:00 | 673.57 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-11 09:35:00 | 666.00 | 2026-02-11 10:00:00 | 663.70 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-11 09:35:00 | 666.00 | 2026-02-11 10:20:00 | 666.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 09:30:00 | 662.60 | 2026-02-12 09:55:00 | 664.32 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-19 09:35:00 | 637.05 | 2026-02-19 10:10:00 | 634.65 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-19 09:35:00 | 637.05 | 2026-02-19 15:20:00 | 618.70 | TARGET_HIT | 0.50 | 2.88% |
| SELL | retest1 | 2026-02-23 10:35:00 | 625.35 | 2026-02-23 10:50:00 | 627.11 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-24 10:55:00 | 618.30 | 2026-02-24 11:05:00 | 619.67 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-25 09:50:00 | 615.30 | 2026-02-25 10:00:00 | 617.55 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-26 09:50:00 | 614.20 | 2026-02-26 09:55:00 | 612.40 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-10 09:30:00 | 577.90 | 2026-03-10 10:15:00 | 580.73 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-03-11 09:30:00 | 593.65 | 2026-03-11 09:35:00 | 591.62 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-17 11:05:00 | 536.20 | 2026-03-17 11:25:00 | 534.24 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-20 09:45:00 | 555.35 | 2026-03-20 10:25:00 | 552.63 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-03-30 10:35:00 | 511.55 | 2026-03-30 11:00:00 | 508.01 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-30 10:35:00 | 511.55 | 2026-03-30 15:20:00 | 503.35 | TARGET_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2026-04-10 09:40:00 | 570.00 | 2026-04-10 09:50:00 | 567.74 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-15 09:35:00 | 575.80 | 2026-04-15 09:40:00 | 578.34 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-17 10:00:00 | 594.10 | 2026-04-17 11:50:00 | 597.22 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-17 10:00:00 | 594.10 | 2026-04-17 15:20:00 | 602.50 | TARGET_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2026-04-22 10:50:00 | 611.60 | 2026-04-22 11:00:00 | 610.04 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-23 11:10:00 | 595.60 | 2026-04-23 11:40:00 | 597.10 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-24 10:50:00 | 586.15 | 2026-04-24 11:00:00 | 583.93 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-24 10:50:00 | 586.15 | 2026-04-24 11:30:00 | 586.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 09:40:00 | 589.50 | 2026-04-28 10:00:00 | 591.18 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-29 09:55:00 | 598.95 | 2026-04-29 10:00:00 | 602.01 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-29 09:55:00 | 598.95 | 2026-04-29 10:55:00 | 598.95 | STOP_HIT | 0.50 | 0.00% |
