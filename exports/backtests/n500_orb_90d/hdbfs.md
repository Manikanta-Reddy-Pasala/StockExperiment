# HDB Financial Services Ltd. (HDBFS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 700.00
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
| ENTRY1 | 20 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 16
- **Target hits / Stop hits / Partials:** 4 / 16 / 8
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 3.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.15% | 2.4% |
| BUY @ 2nd Alert (retest1) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.15% | 2.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.07% | 0.8% |
| SELL @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.07% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 12 | 42.9% | 4 | 16 | 8 | 0.11% | 3.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:50:00 | 720.65 | 721.04 | 0.00 | ORB-short ORB[721.40,724.00] vol=9.6x ATR=1.49 |
| Stop hit — per-position SL triggered | 2026-02-11 10:35:00 | 722.14 | 721.02 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 706.55 | 711.45 | 0.00 | ORB-short ORB[708.90,717.50] vol=1.8x ATR=1.60 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 708.15 | 710.52 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:10:00 | 714.75 | 710.99 | 0.00 | ORB-long ORB[707.30,711.35] vol=1.5x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:15:00 | 717.00 | 711.14 | 0.00 | T1 1.5R @ 717.00 |
| Stop hit — per-position SL triggered | 2026-02-17 11:40:00 | 714.75 | 712.36 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:15:00 | 724.90 | 721.82 | 0.00 | ORB-long ORB[718.95,724.00] vol=1.7x ATR=1.95 |
| Stop hit — per-position SL triggered | 2026-02-18 10:25:00 | 722.95 | 721.91 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:40:00 | 728.05 | 725.31 | 0.00 | ORB-long ORB[720.95,725.80] vol=1.8x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:30:00 | 730.65 | 726.76 | 0.00 | T1 1.5R @ 730.65 |
| Stop hit — per-position SL triggered | 2026-02-20 13:10:00 | 728.05 | 727.95 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:05:00 | 722.85 | 720.80 | 0.00 | ORB-long ORB[715.00,722.05] vol=2.1x ATR=1.77 |
| Stop hit — per-position SL triggered | 2026-02-24 10:25:00 | 721.08 | 720.93 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:10:00 | 724.30 | 721.42 | 0.00 | ORB-long ORB[716.80,720.50] vol=2.4x ATR=1.44 |
| Stop hit — per-position SL triggered | 2026-02-25 10:35:00 | 722.86 | 721.83 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:45:00 | 719.15 | 721.99 | 0.00 | ORB-short ORB[721.90,725.80] vol=1.5x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:55:00 | 717.25 | 721.11 | 0.00 | T1 1.5R @ 717.25 |
| Target hit | 2026-02-26 15:20:00 | 713.65 | 717.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-02-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:10:00 | 714.35 | 710.29 | 0.00 | ORB-long ORB[708.00,713.60] vol=3.2x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 716.91 | 710.88 | 0.00 | T1 1.5R @ 716.91 |
| Stop hit — per-position SL triggered | 2026-02-27 10:45:00 | 714.35 | 712.60 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:50:00 | 678.50 | 684.61 | 0.00 | ORB-short ORB[684.40,691.00] vol=2.3x ATR=2.54 |
| Stop hit — per-position SL triggered | 2026-03-04 10:05:00 | 681.04 | 682.76 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:50:00 | 671.95 | 674.37 | 0.00 | ORB-short ORB[675.35,683.85] vol=10.2x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 673.56 | 674.35 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:05:00 | 651.15 | 651.94 | 0.00 | ORB-short ORB[654.05,661.50] vol=1.9x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-03-10 11:25:00 | 652.71 | 651.92 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:20:00 | 600.25 | 607.47 | 0.00 | ORB-short ORB[607.00,614.85] vol=1.6x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:30:00 | 596.75 | 604.88 | 0.00 | T1 1.5R @ 596.75 |
| Stop hit — per-position SL triggered | 2026-03-24 12:10:00 | 600.25 | 599.42 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 11:15:00 | 634.50 | 632.73 | 0.00 | ORB-long ORB[627.15,634.45] vol=3.6x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:50:00 | 636.43 | 633.02 | 0.00 | T1 1.5R @ 636.43 |
| Target hit | 2026-04-15 15:20:00 | 643.40 | 637.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:10:00 | 676.40 | 673.02 | 0.00 | ORB-long ORB[668.85,675.55] vol=3.1x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:35:00 | 679.08 | 674.55 | 0.00 | T1 1.5R @ 679.08 |
| Target hit | 2026-04-21 15:20:00 | 683.20 | 681.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 655.65 | 658.65 | 0.00 | ORB-short ORB[656.05,664.90] vol=1.6x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 657.76 | 657.71 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:20:00 | 679.85 | 676.48 | 0.00 | ORB-long ORB[669.05,677.50] vol=4.0x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-04-28 10:35:00 | 678.11 | 676.75 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:15:00 | 661.35 | 663.92 | 0.00 | ORB-short ORB[662.30,666.00] vol=1.6x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:50:00 | 658.79 | 662.34 | 0.00 | T1 1.5R @ 658.79 |
| Target hit | 2026-04-30 13:45:00 | 658.60 | 658.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — BUY (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 695.35 | 690.90 | 0.00 | ORB-long ORB[683.25,690.70] vol=2.2x ATR=3.10 |
| Stop hit — per-position SL triggered | 2026-05-07 09:50:00 | 692.25 | 692.08 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:50:00 | 703.85 | 701.42 | 0.00 | ORB-long ORB[696.00,703.65] vol=2.1x ATR=2.28 |
| Stop hit — per-position SL triggered | 2026-05-08 13:35:00 | 701.57 | 702.89 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:50:00 | 720.65 | 2026-02-11 10:35:00 | 722.14 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-13 09:30:00 | 706.55 | 2026-02-13 09:40:00 | 708.15 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-17 11:10:00 | 714.75 | 2026-02-17 11:15:00 | 717.00 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-02-17 11:10:00 | 714.75 | 2026-02-17 11:40:00 | 714.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 10:15:00 | 724.90 | 2026-02-18 10:25:00 | 722.95 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-20 10:40:00 | 728.05 | 2026-02-20 11:30:00 | 730.65 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-20 10:40:00 | 728.05 | 2026-02-20 13:10:00 | 728.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 10:05:00 | 722.85 | 2026-02-24 10:25:00 | 721.08 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-25 10:10:00 | 724.30 | 2026-02-25 10:35:00 | 722.86 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-26 10:45:00 | 719.15 | 2026-02-26 10:55:00 | 717.25 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-02-26 10:45:00 | 719.15 | 2026-02-26 15:20:00 | 713.65 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2026-02-27 10:10:00 | 714.35 | 2026-02-27 10:15:00 | 716.91 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-27 10:10:00 | 714.35 | 2026-02-27 10:45:00 | 714.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-04 09:50:00 | 678.50 | 2026-03-04 10:05:00 | 681.04 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-06 10:50:00 | 671.95 | 2026-03-06 11:00:00 | 673.56 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-10 11:05:00 | 651.15 | 2026-03-10 11:25:00 | 652.71 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-24 10:20:00 | 600.25 | 2026-03-24 10:30:00 | 596.75 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-24 10:20:00 | 600.25 | 2026-03-24 12:10:00 | 600.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 11:15:00 | 634.50 | 2026-04-15 11:50:00 | 636.43 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-04-15 11:15:00 | 634.50 | 2026-04-15 15:20:00 | 643.40 | TARGET_HIT | 0.50 | 1.40% |
| BUY | retest1 | 2026-04-21 11:10:00 | 676.40 | 2026-04-21 11:35:00 | 679.08 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-21 11:10:00 | 676.40 | 2026-04-21 15:20:00 | 683.20 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2026-04-24 09:35:00 | 655.65 | 2026-04-24 10:00:00 | 657.76 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-28 10:20:00 | 679.85 | 2026-04-28 10:35:00 | 678.11 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-30 10:15:00 | 661.35 | 2026-04-30 10:50:00 | 658.79 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-04-30 10:15:00 | 661.35 | 2026-04-30 13:45:00 | 658.60 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2026-05-07 09:35:00 | 695.35 | 2026-05-07 09:50:00 | 692.25 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-08 09:50:00 | 703.85 | 2026-05-08 13:35:00 | 701.57 | STOP_HIT | 1.00 | -0.32% |
