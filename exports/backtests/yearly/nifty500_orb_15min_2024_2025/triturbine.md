# Triveni Turbine Ltd. (TRITURBINE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-12-01 15:25:00 (25996 bars)
- **Last close:** 527.00
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
| PARTIAL | 12 |
| TARGET_HIT | 4 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 17
- **Target hits / Stop hits / Partials:** 4 / 17 / 12
- **Avg / median % per leg:** 0.55% / 0.00%
- **Sum % (uncompounded):** 18.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 7 | 38.9% | 2 | 11 | 5 | 0.62% | 11.1% |
| BUY @ 2nd Alert (retest1) | 18 | 7 | 38.9% | 2 | 11 | 5 | 0.62% | 11.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 9 | 60.0% | 2 | 6 | 7 | 0.47% | 7.0% |
| SELL @ 2nd Alert (retest1) | 15 | 9 | 60.0% | 2 | 6 | 7 | 0.47% | 7.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 16 | 48.5% | 4 | 17 | 12 | 0.55% | 18.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-07 10:50:00 | 557.10 | 561.29 | 0.00 | ORB-short ORB[560.15,564.80] vol=2.3x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 11:00:00 | 554.93 | 560.28 | 0.00 | T1 1.5R @ 554.93 |
| Stop hit — per-position SL triggered | 2024-06-07 11:05:00 | 557.10 | 560.11 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-06-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:45:00 | 599.30 | 595.50 | 0.00 | ORB-long ORB[590.00,596.90] vol=3.8x ATR=2.30 |
| Stop hit — per-position SL triggered | 2024-06-21 10:00:00 | 597.00 | 596.87 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-07-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:20:00 | 612.65 | 608.77 | 0.00 | ORB-long ORB[604.00,610.75] vol=1.5x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 10:25:00 | 616.40 | 610.08 | 0.00 | T1 1.5R @ 616.40 |
| Target hit | 2024-07-03 15:20:00 | 647.60 | 632.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2024-07-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:55:00 | 628.05 | 632.39 | 0.00 | ORB-short ORB[631.10,639.45] vol=1.9x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-07-08 10:00:00 | 630.46 | 632.18 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 623.85 | 628.28 | 0.00 | ORB-short ORB[628.10,634.90] vol=3.9x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:10:00 | 619.33 | 627.54 | 0.00 | T1 1.5R @ 619.33 |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 623.85 | 627.28 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-07-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 11:05:00 | 610.20 | 608.05 | 0.00 | ORB-long ORB[603.90,610.00] vol=1.6x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-07-25 11:15:00 | 608.04 | 608.12 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-07-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:00:00 | 610.00 | 606.28 | 0.00 | ORB-long ORB[603.05,609.95] vol=2.1x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:25:00 | 613.14 | 607.88 | 0.00 | T1 1.5R @ 613.14 |
| Stop hit — per-position SL triggered | 2024-07-26 10:45:00 | 610.00 | 608.45 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:20:00 | 614.35 | 611.22 | 0.00 | ORB-long ORB[608.90,613.80] vol=1.5x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-07-29 10:30:00 | 611.89 | 611.35 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-08-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:45:00 | 587.60 | 594.10 | 0.00 | ORB-short ORB[594.85,602.85] vol=2.0x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:55:00 | 584.47 | 592.38 | 0.00 | T1 1.5R @ 584.47 |
| Stop hit — per-position SL triggered | 2024-08-01 11:35:00 | 587.60 | 591.68 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-09-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 11:05:00 | 786.00 | 778.86 | 0.00 | ORB-long ORB[772.45,783.00] vol=5.0x ATR=3.51 |
| Stop hit — per-position SL triggered | 2024-09-16 11:35:00 | 782.49 | 781.32 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-09-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:00:00 | 741.80 | 739.00 | 0.00 | ORB-long ORB[732.15,738.70] vol=2.6x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 10:05:00 | 746.65 | 740.01 | 0.00 | T1 1.5R @ 746.65 |
| Stop hit — per-position SL triggered | 2024-09-23 10:20:00 | 741.80 | 740.75 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-09-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:20:00 | 745.85 | 740.54 | 0.00 | ORB-long ORB[732.30,741.05] vol=2.2x ATR=2.38 |
| Stop hit — per-position SL triggered | 2024-09-24 10:50:00 | 743.47 | 741.56 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-09-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:45:00 | 727.00 | 731.46 | 0.00 | ORB-short ORB[730.05,736.45] vol=2.7x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 10:50:00 | 724.12 | 730.84 | 0.00 | T1 1.5R @ 724.12 |
| Target hit | 2024-09-25 15:20:00 | 713.00 | 719.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2024-09-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 10:50:00 | 708.20 | 712.73 | 0.00 | ORB-short ORB[712.00,721.15] vol=1.8x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 12:05:00 | 704.63 | 710.78 | 0.00 | T1 1.5R @ 704.63 |
| Target hit | 2024-09-27 15:20:00 | 698.15 | 705.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2024-10-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:50:00 | 655.05 | 670.17 | 0.00 | ORB-short ORB[672.00,680.90] vol=1.6x ATR=4.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 11:05:00 | 648.70 | 667.28 | 0.00 | T1 1.5R @ 648.70 |
| Stop hit — per-position SL triggered | 2024-10-07 11:10:00 | 655.05 | 666.79 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-10-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:30:00 | 750.25 | 746.67 | 0.00 | ORB-long ORB[739.20,749.90] vol=2.0x ATR=3.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 09:40:00 | 755.49 | 751.05 | 0.00 | T1 1.5R @ 755.49 |
| Target hit | 2024-10-11 15:20:00 | 793.10 | 791.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:45:00 | 683.20 | 678.71 | 0.00 | ORB-long ORB[673.10,682.95] vol=2.2x ATR=3.32 |
| Stop hit — per-position SL triggered | 2024-10-31 10:15:00 | 679.88 | 679.46 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 752.50 | 754.94 | 0.00 | ORB-short ORB[752.65,759.95] vol=1.6x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:20:00 | 748.37 | 752.99 | 0.00 | T1 1.5R @ 748.37 |
| Stop hit — per-position SL triggered | 2025-01-02 11:35:00 | 752.50 | 752.75 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-02-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-12 10:35:00 | 565.60 | 553.23 | 0.00 | ORB-long ORB[551.00,557.50] vol=1.6x ATR=3.60 |
| Stop hit — per-position SL triggered | 2025-02-12 10:40:00 | 562.00 | 554.08 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:30:00 | 520.60 | 514.08 | 0.00 | ORB-long ORB[507.55,515.00] vol=2.9x ATR=2.20 |
| Stop hit — per-position SL triggered | 2025-05-05 09:35:00 | 518.40 | 515.50 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:35:00 | 528.35 | 522.49 | 0.00 | ORB-long ORB[518.00,525.50] vol=2.7x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 09:40:00 | 531.85 | 524.76 | 0.00 | T1 1.5R @ 531.85 |
| Stop hit — per-position SL triggered | 2025-05-08 09:45:00 | 528.35 | 525.18 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-06-07 10:50:00 | 557.10 | 2024-06-07 11:00:00 | 554.93 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-06-07 10:50:00 | 557.10 | 2024-06-07 11:05:00 | 557.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-21 09:45:00 | 599.30 | 2024-06-21 10:00:00 | 597.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-03 10:20:00 | 612.65 | 2024-07-03 10:25:00 | 616.40 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-07-03 10:20:00 | 612.65 | 2024-07-03 15:20:00 | 647.60 | TARGET_HIT | 0.50 | 5.70% |
| SELL | retest1 | 2024-07-08 09:55:00 | 628.05 | 2024-07-08 10:00:00 | 630.46 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-07-10 10:05:00 | 623.85 | 2024-07-10 10:10:00 | 619.33 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-07-10 10:05:00 | 623.85 | 2024-07-10 10:15:00 | 623.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-25 11:05:00 | 610.20 | 2024-07-25 11:15:00 | 608.04 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-26 10:00:00 | 610.00 | 2024-07-26 10:25:00 | 613.14 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-07-26 10:00:00 | 610.00 | 2024-07-26 10:45:00 | 610.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-29 10:20:00 | 614.35 | 2024-07-29 10:30:00 | 611.89 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-08-01 10:45:00 | 587.60 | 2024-08-01 10:55:00 | 584.47 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-08-01 10:45:00 | 587.60 | 2024-08-01 11:35:00 | 587.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-16 11:05:00 | 786.00 | 2024-09-16 11:35:00 | 782.49 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-09-23 10:00:00 | 741.80 | 2024-09-23 10:05:00 | 746.65 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-09-23 10:00:00 | 741.80 | 2024-09-23 10:20:00 | 741.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-24 10:20:00 | 745.85 | 2024-09-24 10:50:00 | 743.47 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-25 10:45:00 | 727.00 | 2024-09-25 10:50:00 | 724.12 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-09-25 10:45:00 | 727.00 | 2024-09-25 15:20:00 | 713.00 | TARGET_HIT | 0.50 | 1.93% |
| SELL | retest1 | 2024-09-27 10:50:00 | 708.20 | 2024-09-27 12:05:00 | 704.63 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-09-27 10:50:00 | 708.20 | 2024-09-27 15:20:00 | 698.15 | TARGET_HIT | 0.50 | 1.42% |
| SELL | retest1 | 2024-10-07 10:50:00 | 655.05 | 2024-10-07 11:05:00 | 648.70 | PARTIAL | 0.50 | 0.97% |
| SELL | retest1 | 2024-10-07 10:50:00 | 655.05 | 2024-10-07 11:10:00 | 655.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 09:30:00 | 750.25 | 2024-10-11 09:40:00 | 755.49 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-10-11 09:30:00 | 750.25 | 2024-10-11 15:20:00 | 793.10 | TARGET_HIT | 0.50 | 5.71% |
| BUY | retest1 | 2024-10-31 09:45:00 | 683.20 | 2024-10-31 10:15:00 | 679.88 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-01-02 09:30:00 | 752.50 | 2025-01-02 11:20:00 | 748.37 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-02 09:30:00 | 752.50 | 2025-01-02 11:35:00 | 752.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-12 10:35:00 | 565.60 | 2025-02-12 10:40:00 | 562.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2025-05-05 09:30:00 | 520.60 | 2025-05-05 09:35:00 | 518.40 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-05-08 09:35:00 | 528.35 | 2025-05-08 09:40:00 | 531.85 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-05-08 09:35:00 | 528.35 | 2025-05-08 09:45:00 | 528.35 | STOP_HIT | 0.50 | 0.00% |
