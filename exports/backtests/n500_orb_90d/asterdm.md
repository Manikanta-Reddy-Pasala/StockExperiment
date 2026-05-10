# Aster DM Healthcare Ltd. (ASTERDM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 742.00
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 10
- **Target hits / Stop hits / Partials:** 4 / 10 / 4
- **Avg / median % per leg:** 0.14% / -0.30%
- **Sum % (uncompounded):** 2.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 6 | 40.0% | 3 | 9 | 3 | 0.06% | 0.8% |
| BUY @ 2nd Alert (retest1) | 15 | 6 | 40.0% | 3 | 9 | 3 | 0.06% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.57% | 1.7% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 0.57% | 1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 8 | 44.4% | 4 | 10 | 4 | 0.14% | 2.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:25:00 | 578.55 | 573.83 | 0.00 | ORB-long ORB[568.10,576.30] vol=1.9x ATR=1.89 |
| Stop hit — per-position SL triggered | 2026-02-10 11:30:00 | 576.66 | 574.97 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:30:00 | 604.45 | 596.54 | 0.00 | ORB-long ORB[594.00,602.85] vol=1.8x ATR=2.46 |
| Stop hit — per-position SL triggered | 2026-02-13 10:35:00 | 601.99 | 597.85 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 626.30 | 622.50 | 0.00 | ORB-long ORB[617.00,625.80] vol=1.9x ATR=2.70 |
| Stop hit — per-position SL triggered | 2026-02-17 11:45:00 | 623.60 | 625.05 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:05:00 | 621.50 | 623.90 | 0.00 | ORB-short ORB[624.90,631.70] vol=2.0x ATR=1.99 |
| Stop hit — per-position SL triggered | 2026-02-18 10:35:00 | 623.49 | 622.57 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:45:00 | 641.15 | 637.49 | 0.00 | ORB-long ORB[632.30,638.95] vol=1.7x ATR=1.95 |
| Stop hit — per-position SL triggered | 2026-02-23 09:50:00 | 639.20 | 637.92 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:40:00 | 656.50 | 654.20 | 0.00 | ORB-long ORB[648.95,656.00] vol=2.1x ATR=3.13 |
| Stop hit — per-position SL triggered | 2026-03-05 10:10:00 | 653.37 | 654.89 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:10:00 | 667.60 | 663.82 | 0.00 | ORB-long ORB[657.10,665.50] vol=1.9x ATR=2.69 |
| Stop hit — per-position SL triggered | 2026-03-10 14:35:00 | 664.91 | 665.12 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:40:00 | 671.35 | 668.27 | 0.00 | ORB-long ORB[661.85,668.50] vol=1.8x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 09:45:00 | 674.76 | 669.34 | 0.00 | T1 1.5R @ 674.76 |
| Target hit | 2026-03-11 14:40:00 | 685.35 | 686.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2026-03-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:05:00 | 643.35 | 647.20 | 0.00 | ORB-short ORB[645.30,654.45] vol=1.9x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 11:00:00 | 639.07 | 645.81 | 0.00 | T1 1.5R @ 639.07 |
| Target hit | 2026-03-19 15:20:00 | 634.60 | 641.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-04-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:35:00 | 668.05 | 666.20 | 0.00 | ORB-long ORB[661.60,668.00] vol=2.2x ATR=2.00 |
| Stop hit — per-position SL triggered | 2026-04-07 10:50:00 | 666.05 | 666.24 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 10:40:00 | 675.55 | 672.41 | 0.00 | ORB-long ORB[666.30,674.90] vol=2.1x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-04-09 11:50:00 | 673.44 | 673.13 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:05:00 | 679.60 | 676.24 | 0.00 | ORB-long ORB[671.05,677.00] vol=2.6x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:15:00 | 682.67 | 677.35 | 0.00 | T1 1.5R @ 682.67 |
| Target hit | 2026-04-15 14:40:00 | 680.30 | 680.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2026-04-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:00:00 | 684.00 | 679.48 | 0.00 | ORB-long ORB[672.05,679.20] vol=1.8x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 13:30:00 | 687.57 | 682.99 | 0.00 | T1 1.5R @ 687.57 |
| Target hit | 2026-04-17 15:20:00 | 687.75 | 686.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 688.35 | 683.47 | 0.00 | ORB-long ORB[677.65,686.00] vol=5.8x ATR=2.95 |
| Stop hit — per-position SL triggered | 2026-04-21 10:20:00 | 685.40 | 684.95 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:25:00 | 578.55 | 2026-02-10 11:30:00 | 576.66 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-13 10:30:00 | 604.45 | 2026-02-13 10:35:00 | 601.99 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-02-17 09:35:00 | 626.30 | 2026-02-17 11:45:00 | 623.60 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-18 10:05:00 | 621.50 | 2026-02-18 10:35:00 | 623.49 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-23 09:45:00 | 641.15 | 2026-02-23 09:50:00 | 639.20 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-05 09:40:00 | 656.50 | 2026-03-05 10:10:00 | 653.37 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-10 11:10:00 | 667.60 | 2026-03-10 14:35:00 | 664.91 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-03-11 09:40:00 | 671.35 | 2026-03-11 09:45:00 | 674.76 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-11 09:40:00 | 671.35 | 2026-03-11 14:40:00 | 685.35 | TARGET_HIT | 0.50 | 2.09% |
| SELL | retest1 | 2026-03-19 10:05:00 | 643.35 | 2026-03-19 11:00:00 | 639.07 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2026-03-19 10:05:00 | 643.35 | 2026-03-19 15:20:00 | 634.60 | TARGET_HIT | 0.50 | 1.36% |
| BUY | retest1 | 2026-04-07 10:35:00 | 668.05 | 2026-04-07 10:50:00 | 666.05 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-09 10:40:00 | 675.55 | 2026-04-09 11:50:00 | 673.44 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-15 10:05:00 | 679.60 | 2026-04-15 10:15:00 | 682.67 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-15 10:05:00 | 679.60 | 2026-04-15 14:40:00 | 680.30 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2026-04-17 10:00:00 | 684.00 | 2026-04-17 13:30:00 | 687.57 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-17 10:00:00 | 684.00 | 2026-04-17 15:20:00 | 687.75 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-21 10:00:00 | 688.35 | 2026-04-21 10:20:00 | 685.40 | STOP_HIT | 1.00 | -0.43% |
