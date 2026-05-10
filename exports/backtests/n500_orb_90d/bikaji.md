# Bikaji Foods International Ltd. (BIKAJI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 670.20
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
| PARTIAL | 9 |
| TARGET_HIT | 5 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 18
- **Target hits / Stop hits / Partials:** 5 / 18 / 9
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 3.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 7 | 41.2% | 2 | 10 | 5 | 0.10% | 1.6% |
| BUY @ 2nd Alert (retest1) | 17 | 7 | 41.2% | 2 | 10 | 5 | 0.10% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 7 | 46.7% | 3 | 8 | 4 | 0.10% | 1.4% |
| SELL @ 2nd Alert (retest1) | 15 | 7 | 46.7% | 3 | 8 | 4 | 0.10% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 32 | 14 | 43.8% | 5 | 18 | 9 | 0.10% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 663.05 | 656.66 | 0.00 | ORB-long ORB[649.75,656.35] vol=5.8x ATR=3.33 |
| Stop hit — per-position SL triggered | 2026-02-09 11:50:00 | 659.72 | 658.72 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:10:00 | 668.80 | 671.38 | 0.00 | ORB-short ORB[669.50,674.30] vol=1.9x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:50:00 | 665.49 | 670.43 | 0.00 | T1 1.5R @ 665.49 |
| Stop hit — per-position SL triggered | 2026-02-10 12:15:00 | 668.80 | 668.96 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 629.40 | 632.88 | 0.00 | ORB-short ORB[631.00,638.60] vol=2.2x ATR=1.71 |
| Stop hit — per-position SL triggered | 2026-02-23 11:05:00 | 631.11 | 632.69 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:50:00 | 635.00 | 630.05 | 0.00 | ORB-long ORB[625.60,630.00] vol=3.5x ATR=2.03 |
| Stop hit — per-position SL triggered | 2026-02-27 10:00:00 | 632.97 | 630.54 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:35:00 | 620.25 | 622.14 | 0.00 | ORB-short ORB[622.00,629.25] vol=2.4x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:35:00 | 616.89 | 620.00 | 0.00 | T1 1.5R @ 616.89 |
| Target hit | 2026-03-06 12:15:00 | 616.30 | 615.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2026-03-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-09 09:50:00 | 595.50 | 600.34 | 0.00 | ORB-short ORB[599.35,608.10] vol=1.6x ATR=2.84 |
| Stop hit — per-position SL triggered | 2026-03-09 09:55:00 | 598.34 | 600.09 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:15:00 | 613.15 | 616.90 | 0.00 | ORB-short ORB[615.35,623.60] vol=2.3x ATR=2.06 |
| Stop hit — per-position SL triggered | 2026-03-11 11:10:00 | 615.21 | 616.11 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:30:00 | 615.10 | 621.21 | 0.00 | ORB-short ORB[620.05,627.25] vol=2.8x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-03-17 10:55:00 | 617.53 | 620.26 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 621.75 | 619.85 | 0.00 | ORB-long ORB[616.05,621.20] vol=1.6x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 619.51 | 620.22 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:50:00 | 626.15 | 622.28 | 0.00 | ORB-long ORB[616.10,622.00] vol=3.4x ATR=2.63 |
| Stop hit — per-position SL triggered | 2026-03-25 14:05:00 | 623.52 | 624.60 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 11:15:00 | 644.00 | 641.24 | 0.00 | ORB-long ORB[636.75,641.00] vol=3.6x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:30:00 | 646.09 | 642.43 | 0.00 | T1 1.5R @ 646.09 |
| Stop hit — per-position SL triggered | 2026-04-09 14:35:00 | 644.00 | 644.35 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 646.85 | 645.21 | 0.00 | ORB-long ORB[643.00,646.00] vol=2.4x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:30:00 | 648.79 | 646.20 | 0.00 | T1 1.5R @ 648.79 |
| Target hit | 2026-04-17 15:20:00 | 660.00 | 656.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 656.80 | 656.55 | 0.00 | ORB-long ORB[651.55,655.00] vol=14.4x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:30:00 | 660.17 | 658.72 | 0.00 | T1 1.5R @ 660.17 |
| Target hit | 2026-04-21 11:00:00 | 657.95 | 658.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 672.10 | 670.26 | 0.00 | ORB-long ORB[664.00,668.00] vol=2.5x ATR=2.47 |
| Stop hit — per-position SL triggered | 2026-04-22 09:45:00 | 669.63 | 671.84 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:00:00 | 694.55 | 688.31 | 0.00 | ORB-long ORB[683.15,689.85] vol=2.6x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-04-23 10:05:00 | 692.12 | 689.23 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:35:00 | 677.80 | 679.70 | 0.00 | ORB-short ORB[680.05,689.75] vol=9.9x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:10:00 | 674.12 | 679.50 | 0.00 | T1 1.5R @ 674.12 |
| Stop hit — per-position SL triggered | 2026-04-24 11:30:00 | 677.80 | 679.27 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 673.00 | 667.32 | 0.00 | ORB-long ORB[662.40,667.10] vol=2.5x ATR=2.37 |
| Stop hit — per-position SL triggered | 2026-04-27 10:45:00 | 670.63 | 669.79 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 673.65 | 675.58 | 0.00 | ORB-short ORB[675.45,680.50] vol=3.1x ATR=1.57 |
| Stop hit — per-position SL triggered | 2026-04-28 09:45:00 | 675.22 | 676.89 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:50:00 | 685.50 | 683.90 | 0.00 | ORB-long ORB[678.15,685.05] vol=1.6x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:05:00 | 688.24 | 685.01 | 0.00 | T1 1.5R @ 688.24 |
| Stop hit — per-position SL triggered | 2026-04-29 12:00:00 | 685.50 | 685.67 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:15:00 | 683.15 | 680.43 | 0.00 | ORB-long ORB[676.60,683.00] vol=2.3x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 13:10:00 | 686.91 | 682.84 | 0.00 | T1 1.5R @ 686.91 |
| Stop hit — per-position SL triggered | 2026-05-04 14:50:00 | 683.15 | 683.91 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:15:00 | 681.30 | 681.88 | 0.00 | ORB-short ORB[682.00,687.60] vol=6.3x ATR=1.70 |
| Target hit | 2026-05-05 15:20:00 | 680.25 | 680.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2026-05-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:50:00 | 678.55 | 681.75 | 0.00 | ORB-short ORB[681.35,686.45] vol=3.1x ATR=1.40 |
| Stop hit — per-position SL triggered | 2026-05-06 10:55:00 | 679.95 | 681.67 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 679.20 | 682.58 | 0.00 | ORB-short ORB[682.70,687.60] vol=3.5x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:00:00 | 676.30 | 680.39 | 0.00 | T1 1.5R @ 676.30 |
| Target hit | 2026-05-07 15:20:00 | 675.40 | 675.74 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 663.05 | 2026-02-09 11:50:00 | 659.72 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-02-10 10:10:00 | 668.80 | 2026-02-10 10:50:00 | 665.49 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-10 10:10:00 | 668.80 | 2026-02-10 12:15:00 | 668.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 10:55:00 | 629.40 | 2026-02-23 11:05:00 | 631.11 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-27 09:50:00 | 635.00 | 2026-02-27 10:00:00 | 632.97 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-06 09:35:00 | 620.25 | 2026-03-06 10:35:00 | 616.89 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-06 09:35:00 | 620.25 | 2026-03-06 12:15:00 | 616.30 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2026-03-09 09:50:00 | 595.50 | 2026-03-09 09:55:00 | 598.34 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-03-11 10:15:00 | 613.15 | 2026-03-11 11:10:00 | 615.21 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-17 10:30:00 | 615.10 | 2026-03-17 10:55:00 | 617.53 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-03-18 09:30:00 | 621.75 | 2026-03-18 09:55:00 | 619.51 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-25 09:50:00 | 626.15 | 2026-03-25 14:05:00 | 623.52 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-09 11:15:00 | 644.00 | 2026-04-09 11:30:00 | 646.09 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-04-09 11:15:00 | 644.00 | 2026-04-09 14:35:00 | 644.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:05:00 | 646.85 | 2026-04-17 10:30:00 | 648.79 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-04-17 10:05:00 | 646.85 | 2026-04-17 15:20:00 | 660.00 | TARGET_HIT | 0.50 | 2.03% |
| BUY | retest1 | 2026-04-21 10:10:00 | 656.80 | 2026-04-21 10:30:00 | 660.17 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-21 10:10:00 | 656.80 | 2026-04-21 11:00:00 | 657.95 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2026-04-22 09:40:00 | 672.10 | 2026-04-22 09:45:00 | 669.63 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-23 10:00:00 | 694.55 | 2026-04-23 10:05:00 | 692.12 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-24 10:35:00 | 677.80 | 2026-04-24 11:10:00 | 674.12 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-04-24 10:35:00 | 677.80 | 2026-04-24 11:30:00 | 677.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:40:00 | 673.00 | 2026-04-27 10:45:00 | 670.63 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-28 09:30:00 | 673.65 | 2026-04-28 09:45:00 | 675.22 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-29 10:50:00 | 685.50 | 2026-04-29 11:05:00 | 688.24 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-29 10:50:00 | 685.50 | 2026-04-29 12:00:00 | 685.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 10:15:00 | 683.15 | 2026-05-04 13:10:00 | 686.91 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-05-04 10:15:00 | 683.15 | 2026-05-04 14:50:00 | 683.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 11:15:00 | 681.30 | 2026-05-05 15:20:00 | 680.25 | TARGET_HIT | 1.00 | 0.15% |
| SELL | retest1 | 2026-05-06 10:50:00 | 678.55 | 2026-05-06 10:55:00 | 679.95 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-05-07 09:35:00 | 679.20 | 2026-05-07 10:00:00 | 676.30 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-05-07 09:35:00 | 679.20 | 2026-05-07 15:20:00 | 675.40 | TARGET_HIT | 0.50 | 0.56% |
