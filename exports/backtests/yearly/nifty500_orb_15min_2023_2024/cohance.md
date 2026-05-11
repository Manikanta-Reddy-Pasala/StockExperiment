# Cohance Lifesciences Ltd. (COHANCE)

## Backtest Summary

- **Window:** 2024-01-08 09:15:00 → 2026-05-08 15:25:00 (43117 bars)
- **Last close:** 487.90
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
| ENTRY1 | 23 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 7 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 16
- **Target hits / Stop hits / Partials:** 7 / 16 / 10
- **Avg / median % per leg:** 0.29% / 0.18%
- **Sum % (uncompounded):** 9.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.01% | 0.1% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.01% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 22 | 13 | 59.1% | 6 | 9 | 7 | 0.43% | 9.4% |
| SELL @ 2nd Alert (retest1) | 22 | 13 | 59.1% | 6 | 9 | 7 | 0.43% | 9.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 17 | 51.5% | 7 | 16 | 10 | 0.29% | 9.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 10:35:00 | 732.30 | 736.60 | 0.00 | ORB-short ORB[737.65,747.45] vol=4.4x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 12:30:00 | 726.11 | 731.93 | 0.00 | T1 1.5R @ 726.11 |
| Target hit | 2024-01-08 15:20:00 | 720.45 | 727.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2024-01-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 09:40:00 | 733.35 | 735.50 | 0.00 | ORB-short ORB[734.90,739.50] vol=2.5x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-01-11 10:40:00 | 735.42 | 734.53 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-01-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 10:40:00 | 716.75 | 719.41 | 0.00 | ORB-short ORB[718.00,727.80] vol=2.0x ATR=2.35 |
| Stop hit — per-position SL triggered | 2024-01-12 12:40:00 | 719.10 | 717.86 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-01-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 11:10:00 | 682.10 | 683.78 | 0.00 | ORB-short ORB[684.90,694.30] vol=1.8x ATR=2.89 |
| Stop hit — per-position SL triggered | 2024-01-17 11:30:00 | 684.99 | 683.63 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 09:40:00 | 714.00 | 706.12 | 0.00 | ORB-long ORB[690.50,700.45] vol=4.2x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 09:45:00 | 719.33 | 710.81 | 0.00 | T1 1.5R @ 719.33 |
| Stop hit — per-position SL triggered | 2024-01-20 09:50:00 | 714.00 | 711.02 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-02-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 09:45:00 | 624.65 | 620.92 | 0.00 | ORB-long ORB[615.05,622.85] vol=4.6x ATR=4.02 |
| Stop hit — per-position SL triggered | 2024-02-07 09:55:00 | 620.63 | 621.08 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-02-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 09:40:00 | 671.70 | 674.10 | 0.00 | ORB-short ORB[672.40,677.90] vol=1.6x ATR=2.11 |
| Stop hit — per-position SL triggered | 2024-02-20 12:20:00 | 673.81 | 672.16 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 09:40:00 | 670.60 | 675.32 | 0.00 | ORB-short ORB[674.15,683.00] vol=2.1x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 10:05:00 | 667.13 | 670.53 | 0.00 | T1 1.5R @ 667.13 |
| Stop hit — per-position SL triggered | 2024-02-21 10:10:00 | 670.60 | 670.44 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-02-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 09:50:00 | 667.80 | 669.09 | 0.00 | ORB-short ORB[668.85,675.00] vol=4.2x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-22 13:20:00 | 664.62 | 667.58 | 0.00 | T1 1.5R @ 664.62 |
| Target hit | 2024-02-22 15:20:00 | 664.60 | 665.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-02-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-27 10:35:00 | 661.10 | 667.22 | 0.00 | ORB-short ORB[666.10,674.00] vol=2.6x ATR=2.54 |
| Stop hit — per-position SL triggered | 2024-02-27 10:40:00 | 663.64 | 666.86 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-02-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:35:00 | 652.65 | 662.16 | 0.00 | ORB-short ORB[658.55,667.60] vol=2.1x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 10:50:00 | 649.20 | 661.04 | 0.00 | T1 1.5R @ 649.20 |
| Target hit | 2024-02-28 15:20:00 | 642.05 | 643.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-03-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 09:50:00 | 677.05 | 681.72 | 0.00 | ORB-short ORB[681.10,689.40] vol=3.0x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 10:40:00 | 670.99 | 678.99 | 0.00 | T1 1.5R @ 670.99 |
| Target hit | 2024-03-05 15:20:00 | 665.40 | 669.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-03-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 09:50:00 | 655.00 | 657.57 | 0.00 | ORB-short ORB[655.65,665.30] vol=1.7x ATR=3.05 |
| Target hit | 2024-03-11 15:20:00 | 650.60 | 654.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2024-03-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:00:00 | 609.75 | 610.65 | 0.00 | ORB-short ORB[610.70,614.30] vol=3.0x ATR=1.82 |
| Stop hit — per-position SL triggered | 2024-03-20 11:40:00 | 611.57 | 610.03 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-03-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 09:55:00 | 623.25 | 617.70 | 0.00 | ORB-long ORB[614.55,620.45] vol=3.8x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-21 10:00:00 | 627.75 | 624.71 | 0.00 | T1 1.5R @ 627.75 |
| Target hit | 2024-03-21 10:00:00 | 624.35 | 624.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2024-03-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 09:35:00 | 636.85 | 634.73 | 0.00 | ORB-long ORB[628.70,634.55] vol=2.2x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 09:40:00 | 639.98 | 635.80 | 0.00 | T1 1.5R @ 639.98 |
| Stop hit — per-position SL triggered | 2024-03-22 09:50:00 | 636.85 | 636.25 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-03-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 11:05:00 | 656.15 | 650.50 | 0.00 | ORB-long ORB[641.20,650.90] vol=1.6x ATR=2.22 |
| Stop hit — per-position SL triggered | 2024-03-28 11:15:00 | 653.93 | 651.09 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-04-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:30:00 | 658.60 | 662.99 | 0.00 | ORB-short ORB[660.60,668.50] vol=2.4x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 09:40:00 | 653.67 | 659.28 | 0.00 | T1 1.5R @ 653.67 |
| Stop hit — per-position SL triggered | 2024-04-04 09:45:00 | 658.60 | 659.27 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-04-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 10:55:00 | 621.25 | 626.32 | 0.00 | ORB-short ORB[627.40,632.95] vol=4.3x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 13:45:00 | 617.76 | 621.30 | 0.00 | T1 1.5R @ 617.76 |
| Target hit | 2024-04-10 15:20:00 | 614.75 | 619.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-04-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 11:05:00 | 643.75 | 639.71 | 0.00 | ORB-long ORB[639.05,642.00] vol=4.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-04-25 11:10:00 | 642.12 | 639.91 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-05-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-02 11:00:00 | 661.00 | 662.71 | 0.00 | ORB-short ORB[661.60,669.55] vol=4.0x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-05-02 14:35:00 | 663.27 | 662.73 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-05-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 11:05:00 | 660.45 | 657.65 | 0.00 | ORB-long ORB[654.05,659.60] vol=2.4x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-05-09 12:55:00 | 657.98 | 658.15 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-05-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 10:35:00 | 650.85 | 646.58 | 0.00 | ORB-long ORB[640.90,649.00] vol=1.6x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-05-10 12:20:00 | 648.11 | 647.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-01-08 10:35:00 | 732.30 | 2024-01-08 12:30:00 | 726.11 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2024-01-08 10:35:00 | 732.30 | 2024-01-08 15:20:00 | 720.45 | TARGET_HIT | 0.50 | 1.62% |
| SELL | retest1 | 2024-01-11 09:40:00 | 733.35 | 2024-01-11 10:40:00 | 735.42 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-12 10:40:00 | 716.75 | 2024-01-12 12:40:00 | 719.10 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-01-17 11:10:00 | 682.10 | 2024-01-17 11:30:00 | 684.99 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-01-20 09:40:00 | 714.00 | 2024-01-20 09:45:00 | 719.33 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-01-20 09:40:00 | 714.00 | 2024-01-20 09:50:00 | 714.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-07 09:45:00 | 624.65 | 2024-02-07 09:55:00 | 620.63 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2024-02-20 09:40:00 | 671.70 | 2024-02-20 12:20:00 | 673.81 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-02-21 09:40:00 | 670.60 | 2024-02-21 10:05:00 | 667.13 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-02-21 09:40:00 | 670.60 | 2024-02-21 10:10:00 | 670.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-22 09:50:00 | 667.80 | 2024-02-22 13:20:00 | 664.62 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-02-22 09:50:00 | 667.80 | 2024-02-22 15:20:00 | 664.60 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2024-02-27 10:35:00 | 661.10 | 2024-02-27 10:40:00 | 663.64 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-02-28 10:35:00 | 652.65 | 2024-02-28 10:50:00 | 649.20 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-02-28 10:35:00 | 652.65 | 2024-02-28 15:20:00 | 642.05 | TARGET_HIT | 0.50 | 1.62% |
| SELL | retest1 | 2024-03-05 09:50:00 | 677.05 | 2024-03-05 10:40:00 | 670.99 | PARTIAL | 0.50 | 0.89% |
| SELL | retest1 | 2024-03-05 09:50:00 | 677.05 | 2024-03-05 15:20:00 | 665.40 | TARGET_HIT | 0.50 | 1.72% |
| SELL | retest1 | 2024-03-11 09:50:00 | 655.00 | 2024-03-11 15:20:00 | 650.60 | TARGET_HIT | 1.00 | 0.67% |
| SELL | retest1 | 2024-03-20 10:00:00 | 609.75 | 2024-03-20 11:40:00 | 611.57 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-03-21 09:55:00 | 623.25 | 2024-03-21 10:00:00 | 627.75 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-03-21 09:55:00 | 623.25 | 2024-03-21 10:00:00 | 624.35 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2024-03-22 09:35:00 | 636.85 | 2024-03-22 09:40:00 | 639.98 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-03-22 09:35:00 | 636.85 | 2024-03-22 09:50:00 | 636.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-28 11:05:00 | 656.15 | 2024-03-28 11:15:00 | 653.93 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-04-04 09:30:00 | 658.60 | 2024-04-04 09:40:00 | 653.67 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2024-04-04 09:30:00 | 658.60 | 2024-04-04 09:45:00 | 658.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-10 10:55:00 | 621.25 | 2024-04-10 13:45:00 | 617.76 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-04-10 10:55:00 | 621.25 | 2024-04-10 15:20:00 | 614.75 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2024-04-25 11:05:00 | 643.75 | 2024-04-25 11:10:00 | 642.12 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-02 11:00:00 | 661.00 | 2024-05-02 14:35:00 | 663.27 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-05-09 11:05:00 | 660.45 | 2024-05-09 12:55:00 | 657.98 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-05-10 10:35:00 | 650.85 | 2024-05-10 12:20:00 | 648.11 | STOP_HIT | 1.00 | -0.42% |
