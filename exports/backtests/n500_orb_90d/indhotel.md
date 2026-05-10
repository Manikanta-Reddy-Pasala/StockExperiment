# Indian Hotels Co. Ltd. (INDHOTEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 672.00
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
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 12
- **Target hits / Stop hits / Partials:** 4 / 12 / 7
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 3.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 7 | 58.3% | 3 | 5 | 4 | 0.21% | 2.5% |
| BUY @ 2nd Alert (retest1) | 12 | 7 | 58.3% | 3 | 5 | 4 | 0.21% | 2.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.12% | 1.3% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.12% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 11 | 47.8% | 4 | 12 | 7 | 0.16% | 3.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 690.30 | 688.02 | 0.00 | ORB-long ORB[682.95,688.50] vol=1.8x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 12:00:00 | 693.53 | 689.87 | 0.00 | T1 1.5R @ 693.53 |
| Target hit | 2026-02-09 15:20:00 | 696.25 | 693.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 705.45 | 701.02 | 0.00 | ORB-long ORB[693.00,701.80] vol=3.6x ATR=1.97 |
| Stop hit — per-position SL triggered | 2026-02-10 10:05:00 | 703.48 | 702.23 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:40:00 | 699.95 | 702.82 | 0.00 | ORB-short ORB[700.55,706.00] vol=1.5x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-02-11 10:20:00 | 701.53 | 701.55 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 679.70 | 686.86 | 0.00 | ORB-short ORB[689.05,697.00] vol=1.7x ATR=1.85 |
| Stop hit — per-position SL triggered | 2026-02-19 10:40:00 | 681.55 | 686.44 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 672.75 | 670.50 | 0.00 | ORB-long ORB[665.00,672.65] vol=3.3x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:50:00 | 675.86 | 672.04 | 0.00 | T1 1.5R @ 675.86 |
| Target hit | 2026-02-20 15:20:00 | 674.55 | 674.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 673.90 | 676.87 | 0.00 | ORB-short ORB[675.40,682.40] vol=1.6x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-02-23 11:10:00 | 675.43 | 676.53 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 680.30 | 676.59 | 0.00 | ORB-long ORB[672.95,677.00] vol=2.4x ATR=1.33 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 678.97 | 678.49 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:15:00 | 676.20 | 677.27 | 0.00 | ORB-short ORB[678.75,685.00] vol=2.8x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 14:05:00 | 673.89 | 675.97 | 0.00 | T1 1.5R @ 673.89 |
| Target hit | 2026-02-27 15:20:00 | 667.00 | 670.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-03-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:35:00 | 626.05 | 629.96 | 0.00 | ORB-short ORB[635.00,641.85] vol=2.1x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:45:00 | 622.96 | 628.53 | 0.00 | T1 1.5R @ 622.96 |
| Stop hit — per-position SL triggered | 2026-03-05 11:00:00 | 626.05 | 626.62 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:50:00 | 630.00 | 623.91 | 0.00 | ORB-long ORB[616.70,624.70] vol=2.6x ATR=1.69 |
| Stop hit — per-position SL triggered | 2026-03-20 10:55:00 | 628.31 | 624.41 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:10:00 | 625.60 | 626.36 | 0.00 | ORB-short ORB[627.05,635.00] vol=1.7x ATR=2.67 |
| Stop hit — per-position SL triggered | 2026-04-09 10:30:00 | 628.27 | 626.38 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:40:00 | 664.45 | 661.93 | 0.00 | ORB-long ORB[655.75,661.60] vol=1.7x ATR=1.39 |
| Stop hit — per-position SL triggered | 2026-04-21 11:30:00 | 663.06 | 662.44 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:10:00 | 658.60 | 660.88 | 0.00 | ORB-short ORB[659.05,666.05] vol=3.8x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-04-22 10:30:00 | 660.08 | 660.72 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 636.05 | 637.51 | 0.00 | ORB-short ORB[636.10,641.55] vol=3.9x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:45:00 | 633.10 | 637.12 | 0.00 | T1 1.5R @ 633.10 |
| Stop hit — per-position SL triggered | 2026-04-24 14:30:00 | 636.05 | 635.73 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:00:00 | 644.10 | 640.89 | 0.00 | ORB-long ORB[635.35,641.75] vol=1.7x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:30:00 | 647.00 | 642.04 | 0.00 | T1 1.5R @ 647.00 |
| Target hit | 2026-04-27 15:20:00 | 647.45 | 646.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 661.15 | 657.46 | 0.00 | ORB-long ORB[651.00,659.45] vol=1.8x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:20:00 | 663.89 | 660.20 | 0.00 | T1 1.5R @ 663.89 |
| Stop hit — per-position SL triggered | 2026-05-06 12:25:00 | 661.15 | 661.59 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 690.30 | 2026-02-09 12:00:00 | 693.53 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-09 10:35:00 | 690.30 | 2026-02-09 15:20:00 | 696.25 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2026-02-10 09:30:00 | 705.45 | 2026-02-10 10:05:00 | 703.48 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-11 09:40:00 | 699.95 | 2026-02-11 10:20:00 | 701.53 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-19 10:35:00 | 679.70 | 2026-02-19 10:40:00 | 681.55 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-20 10:35:00 | 672.75 | 2026-02-20 10:50:00 | 675.86 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-20 10:35:00 | 672.75 | 2026-02-20 15:20:00 | 674.55 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-23 11:00:00 | 673.90 | 2026-02-23 11:10:00 | 675.43 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-26 10:50:00 | 680.30 | 2026-02-26 11:30:00 | 678.97 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-27 11:15:00 | 676.20 | 2026-02-27 14:05:00 | 673.89 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-27 11:15:00 | 676.20 | 2026-02-27 15:20:00 | 667.00 | TARGET_HIT | 0.50 | 1.36% |
| SELL | retest1 | 2026-03-05 10:35:00 | 626.05 | 2026-03-05 10:45:00 | 622.96 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-05 10:35:00 | 626.05 | 2026-03-05 11:00:00 | 626.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 10:50:00 | 630.00 | 2026-03-20 10:55:00 | 628.31 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-09 10:10:00 | 625.60 | 2026-04-09 10:30:00 | 628.27 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-21 10:40:00 | 664.45 | 2026-04-21 11:30:00 | 663.06 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-04-22 10:10:00 | 658.60 | 2026-04-22 10:30:00 | 660.08 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-24 09:30:00 | 636.05 | 2026-04-24 10:45:00 | 633.10 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-24 09:30:00 | 636.05 | 2026-04-24 14:30:00 | 636.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 10:00:00 | 644.10 | 2026-04-27 10:30:00 | 647.00 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-27 10:00:00 | 644.10 | 2026-04-27 15:20:00 | 647.45 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2026-05-06 09:35:00 | 661.15 | 2026-05-06 10:20:00 | 663.89 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-05-06 09:35:00 | 661.15 | 2026-05-06 12:25:00 | 661.15 | STOP_HIT | 0.50 | 0.00% |
