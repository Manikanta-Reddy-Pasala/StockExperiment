# Jubilant Ingrevia Ltd. (JUBLINGREA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 743.40
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
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 15
- **Target hits / Stop hits / Partials:** 5 / 14 / 10
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 6.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 7 | 58.3% | 2 | 5 | 5 | 0.43% | 5.2% |
| BUY @ 2nd Alert (retest1) | 12 | 7 | 58.3% | 2 | 5 | 5 | 0.43% | 5.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 7 | 41.2% | 3 | 9 | 5 | 0.08% | 1.3% |
| SELL @ 2nd Alert (retest1) | 17 | 7 | 41.2% | 3 | 9 | 5 | 0.08% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 29 | 14 | 48.3% | 5 | 14 | 10 | 0.22% | 6.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 651.75 | 634.07 | 0.00 | ORB-long ORB[620.00,628.00] vol=4.1x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 10:45:00 | 658.07 | 637.89 | 0.00 | T1 1.5R @ 658.07 |
| Stop hit — per-position SL triggered | 2026-02-09 11:20:00 | 651.75 | 646.69 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:20:00 | 672.00 | 664.61 | 0.00 | ORB-long ORB[660.20,667.85] vol=1.7x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:30:00 | 675.68 | 667.30 | 0.00 | T1 1.5R @ 675.68 |
| Stop hit — per-position SL triggered | 2026-02-11 11:20:00 | 672.00 | 671.01 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:25:00 | 648.35 | 650.98 | 0.00 | ORB-short ORB[652.05,659.95] vol=1.9x ATR=2.49 |
| Stop hit — per-position SL triggered | 2026-02-13 10:35:00 | 650.84 | 650.91 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:15:00 | 627.15 | 630.78 | 0.00 | ORB-short ORB[632.50,636.55] vol=2.2x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:25:00 | 623.98 | 628.60 | 0.00 | T1 1.5R @ 623.98 |
| Stop hit — per-position SL triggered | 2026-02-16 12:05:00 | 627.15 | 627.89 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 635.25 | 634.12 | 0.00 | ORB-long ORB[625.00,634.00] vol=1.9x ATR=2.36 |
| Stop hit — per-position SL triggered | 2026-02-17 10:55:00 | 632.89 | 634.16 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:05:00 | 627.95 | 630.46 | 0.00 | ORB-short ORB[629.65,635.65] vol=2.3x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-02-18 13:30:00 | 629.56 | 629.66 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:30:00 | 618.75 | 622.77 | 0.00 | ORB-short ORB[622.00,628.00] vol=1.5x ATR=1.84 |
| Stop hit — per-position SL triggered | 2026-02-19 11:25:00 | 620.59 | 622.19 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 11:10:00 | 604.20 | 608.41 | 0.00 | ORB-short ORB[606.70,614.80] vol=3.3x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:15:00 | 601.26 | 607.47 | 0.00 | T1 1.5R @ 601.26 |
| Stop hit — per-position SL triggered | 2026-02-20 11:20:00 | 604.20 | 607.60 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:35:00 | 598.40 | 600.29 | 0.00 | ORB-short ORB[601.50,608.85] vol=2.1x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:40:00 | 595.30 | 598.80 | 0.00 | T1 1.5R @ 595.30 |
| Target hit | 2026-02-25 15:20:00 | 595.00 | 595.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-02-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:05:00 | 591.50 | 594.60 | 0.00 | ORB-short ORB[593.95,599.45] vol=1.8x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:30:00 | 588.42 | 592.87 | 0.00 | T1 1.5R @ 588.42 |
| Target hit | 2026-02-26 14:10:00 | 588.35 | 588.04 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 563.55 | 567.88 | 0.00 | ORB-short ORB[566.20,572.65] vol=5.1x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 565.29 | 567.84 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:00:00 | 580.80 | 575.54 | 0.00 | ORB-long ORB[567.15,575.75] vol=2.5x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:05:00 | 584.92 | 579.64 | 0.00 | T1 1.5R @ 584.92 |
| Stop hit — per-position SL triggered | 2026-03-11 11:10:00 | 580.80 | 580.04 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:05:00 | 551.75 | 552.74 | 0.00 | ORB-short ORB[552.70,556.15] vol=2.4x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:15:00 | 548.45 | 551.70 | 0.00 | T1 1.5R @ 548.45 |
| Target hit | 2026-03-17 10:15:00 | 552.00 | 551.70 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — SELL (started 2026-03-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:10:00 | 569.40 | 570.19 | 0.00 | ORB-short ORB[569.55,576.90] vol=2.0x ATR=2.05 |
| Stop hit — per-position SL triggered | 2026-03-19 11:20:00 | 571.45 | 570.28 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 11:00:00 | 622.90 | 624.00 | 0.00 | ORB-short ORB[626.85,634.85] vol=4.6x ATR=2.03 |
| Stop hit — per-position SL triggered | 2026-04-09 11:10:00 | 624.93 | 624.11 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:55:00 | 650.80 | 645.79 | 0.00 | ORB-long ORB[641.25,647.80] vol=2.2x ATR=2.95 |
| Stop hit — per-position SL triggered | 2026-04-10 10:45:00 | 647.85 | 646.86 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:50:00 | 656.60 | 650.34 | 0.00 | ORB-long ORB[645.15,653.00] vol=3.5x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:00:00 | 662.08 | 655.93 | 0.00 | T1 1.5R @ 662.08 |
| Target hit | 2026-04-15 15:20:00 | 666.00 | 662.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 720.70 | 718.36 | 0.00 | ORB-long ORB[712.90,719.00] vol=2.3x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:45:00 | 725.11 | 719.90 | 0.00 | T1 1.5R @ 725.11 |
| Target hit | 2026-05-05 15:20:00 | 727.05 | 724.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 743.30 | 747.50 | 0.00 | ORB-short ORB[745.80,753.00] vol=1.6x ATR=3.05 |
| Stop hit — per-position SL triggered | 2026-05-07 09:40:00 | 746.35 | 747.11 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 651.75 | 2026-02-09 10:45:00 | 658.07 | PARTIAL | 0.50 | 0.97% |
| BUY | retest1 | 2026-02-09 10:40:00 | 651.75 | 2026-02-09 11:20:00 | 651.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 10:20:00 | 672.00 | 2026-02-11 10:30:00 | 675.68 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-11 10:20:00 | 672.00 | 2026-02-11 11:20:00 | 672.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:25:00 | 648.35 | 2026-02-13 10:35:00 | 650.84 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-16 10:15:00 | 627.15 | 2026-02-16 11:25:00 | 623.98 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-16 10:15:00 | 627.15 | 2026-02-16 12:05:00 | 627.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:05:00 | 635.25 | 2026-02-17 10:55:00 | 632.89 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-18 11:05:00 | 627.95 | 2026-02-18 13:30:00 | 629.56 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-19 10:30:00 | 618.75 | 2026-02-19 11:25:00 | 620.59 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-20 11:10:00 | 604.20 | 2026-02-20 11:15:00 | 601.26 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-20 11:10:00 | 604.20 | 2026-02-20 11:20:00 | 604.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 10:35:00 | 598.40 | 2026-02-25 11:40:00 | 595.30 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-25 10:35:00 | 598.40 | 2026-02-25 15:20:00 | 595.00 | TARGET_HIT | 0.50 | 0.57% |
| SELL | retest1 | 2026-02-26 10:05:00 | 591.50 | 2026-02-26 11:30:00 | 588.42 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-26 10:05:00 | 591.50 | 2026-02-26 14:10:00 | 588.35 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2026-03-06 10:45:00 | 563.55 | 2026-03-06 10:50:00 | 565.29 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-03-11 11:00:00 | 580.80 | 2026-03-11 11:05:00 | 584.92 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-03-11 11:00:00 | 580.80 | 2026-03-11 11:10:00 | 580.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-17 10:05:00 | 551.75 | 2026-03-17 10:15:00 | 548.45 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-03-17 10:05:00 | 551.75 | 2026-03-17 10:15:00 | 552.00 | TARGET_HIT | 0.50 | -0.05% |
| SELL | retest1 | 2026-03-19 11:10:00 | 569.40 | 2026-03-19 11:20:00 | 571.45 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-09 11:00:00 | 622.90 | 2026-04-09 11:10:00 | 624.93 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-10 09:55:00 | 650.80 | 2026-04-10 10:45:00 | 647.85 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-15 09:50:00 | 656.60 | 2026-04-15 10:00:00 | 662.08 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2026-04-15 09:50:00 | 656.60 | 2026-04-15 15:20:00 | 666.00 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2026-05-05 09:40:00 | 720.70 | 2026-05-05 09:45:00 | 725.11 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-05-05 09:40:00 | 720.70 | 2026-05-05 15:20:00 | 727.05 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2026-05-07 09:35:00 | 743.30 | 2026-05-07 09:40:00 | 746.35 | STOP_HIT | 1.00 | -0.41% |
