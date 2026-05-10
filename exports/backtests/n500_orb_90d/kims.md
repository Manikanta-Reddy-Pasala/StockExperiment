# Krishna Institute of Medical Sciences Ltd. (KIMS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 715.60
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
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 5
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 3.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.23% | 2.7% |
| BUY @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.23% | 2.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.07% | 0.4% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.07% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 7 | 38.9% | 2 | 11 | 5 | 0.17% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:35:00 | 668.90 | 659.79 | 0.00 | ORB-long ORB[650.55,659.00] vol=3.2x ATR=3.34 |
| Stop hit — per-position SL triggered | 2026-02-11 10:40:00 | 665.56 | 661.22 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:50:00 | 708.75 | 706.19 | 0.00 | ORB-long ORB[700.40,707.35] vol=2.8x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:40:00 | 712.17 | 707.72 | 0.00 | T1 1.5R @ 712.17 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 708.75 | 709.00 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:45:00 | 727.95 | 727.04 | 0.00 | ORB-long ORB[722.15,727.50] vol=3.4x ATR=1.88 |
| Stop hit — per-position SL triggered | 2026-02-26 10:55:00 | 726.07 | 727.03 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:05:00 | 698.00 | 704.08 | 0.00 | ORB-short ORB[704.60,713.60] vol=7.6x ATR=1.99 |
| Stop hit — per-position SL triggered | 2026-03-05 11:25:00 | 699.99 | 702.94 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 693.90 | 699.73 | 0.00 | ORB-short ORB[698.45,703.60] vol=3.1x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:50:00 | 690.68 | 699.03 | 0.00 | T1 1.5R @ 690.68 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 693.90 | 698.58 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:15:00 | 645.20 | 648.35 | 0.00 | ORB-short ORB[652.00,661.10] vol=10.4x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:30:00 | 641.54 | 648.02 | 0.00 | T1 1.5R @ 641.54 |
| Stop hit — per-position SL triggered | 2026-03-13 11:25:00 | 645.20 | 646.86 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:00:00 | 649.00 | 646.19 | 0.00 | ORB-long ORB[640.80,647.85] vol=2.4x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:05:00 | 651.80 | 646.45 | 0.00 | T1 1.5R @ 651.80 |
| Target hit | 2026-03-17 15:20:00 | 659.60 | 652.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:15:00 | 625.60 | 623.16 | 0.00 | ORB-long ORB[618.55,622.30] vol=2.7x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 10:45:00 | 628.56 | 624.23 | 0.00 | T1 1.5R @ 628.56 |
| Target hit | 2026-04-07 15:20:00 | 637.15 | 633.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-04-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:10:00 | 677.45 | 672.35 | 0.00 | ORB-long ORB[661.45,671.00] vol=4.3x ATR=2.14 |
| Stop hit — per-position SL triggered | 2026-04-17 11:00:00 | 675.31 | 674.77 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 682.95 | 680.71 | 0.00 | ORB-long ORB[675.55,681.30] vol=1.8x ATR=1.97 |
| Stop hit — per-position SL triggered | 2026-04-22 09:50:00 | 680.98 | 680.90 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 676.85 | 679.06 | 0.00 | ORB-short ORB[677.25,683.90] vol=1.7x ATR=2.25 |
| Stop hit — per-position SL triggered | 2026-05-06 11:45:00 | 679.10 | 677.61 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:45:00 | 712.95 | 706.33 | 0.00 | ORB-long ORB[700.30,709.25] vol=1.6x ATR=2.99 |
| Stop hit — per-position SL triggered | 2026-05-07 10:55:00 | 709.96 | 707.05 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 716.30 | 713.00 | 0.00 | ORB-long ORB[705.65,714.85] vol=2.3x ATR=2.55 |
| Stop hit — per-position SL triggered | 2026-05-08 09:35:00 | 713.75 | 713.33 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:35:00 | 668.90 | 2026-02-11 10:40:00 | 665.56 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-02-19 09:50:00 | 708.75 | 2026-02-19 10:40:00 | 712.17 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-19 09:50:00 | 708.75 | 2026-02-19 11:00:00 | 708.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 10:45:00 | 727.95 | 2026-02-26 10:55:00 | 726.07 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-05 11:05:00 | 698.00 | 2026-03-05 11:25:00 | 699.99 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-06 10:45:00 | 693.90 | 2026-03-06 10:50:00 | 690.68 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-06 10:45:00 | 693.90 | 2026-03-06 11:00:00 | 693.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:15:00 | 645.20 | 2026-03-13 10:30:00 | 641.54 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-13 10:15:00 | 645.20 | 2026-03-13 11:25:00 | 645.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 11:00:00 | 649.00 | 2026-03-17 11:05:00 | 651.80 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-03-17 11:00:00 | 649.00 | 2026-03-17 15:20:00 | 659.60 | TARGET_HIT | 0.50 | 1.63% |
| BUY | retest1 | 2026-04-07 10:15:00 | 625.60 | 2026-04-07 10:45:00 | 628.56 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-07 10:15:00 | 625.60 | 2026-04-07 15:20:00 | 637.15 | TARGET_HIT | 0.50 | 1.85% |
| BUY | retest1 | 2026-04-17 10:10:00 | 677.45 | 2026-04-17 11:00:00 | 675.31 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-22 09:40:00 | 682.95 | 2026-04-22 09:50:00 | 680.98 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-05-06 09:35:00 | 676.85 | 2026-05-06 11:45:00 | 679.10 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-07 10:45:00 | 712.95 | 2026-05-07 10:55:00 | 709.96 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-05-08 09:30:00 | 716.30 | 2026-05-08 09:35:00 | 713.75 | STOP_HIT | 1.00 | -0.36% |
