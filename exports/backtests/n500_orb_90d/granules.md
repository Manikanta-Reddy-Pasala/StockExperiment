# Granules India Ltd. (GRANULES)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 750.10
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 16
- **Target hits / Stop hits / Partials:** 2 / 16 / 5
- **Avg / median % per leg:** -0.07% / -0.26%
- **Sum % (uncompounded):** -1.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 6 | 31.6% | 2 | 13 | 4 | -0.06% | -1.2% |
| BUY @ 2nd Alert (retest1) | 19 | 6 | 31.6% | 2 | 13 | 4 | -0.06% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.10% | -0.4% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.10% | -0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 7 | 30.4% | 2 | 16 | 5 | -0.07% | -1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:45:00 | 564.85 | 569.14 | 0.00 | ORB-short ORB[570.00,577.95] vol=4.7x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:20:00 | 562.57 | 568.40 | 0.00 | T1 1.5R @ 562.57 |
| Stop hit — per-position SL triggered | 2026-02-12 12:30:00 | 564.85 | 567.54 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:40:00 | 569.00 | 565.07 | 0.00 | ORB-long ORB[562.80,568.00] vol=2.5x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-02-13 09:55:00 | 566.76 | 566.39 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 572.85 | 569.71 | 0.00 | ORB-long ORB[564.35,570.80] vol=1.6x ATR=1.87 |
| Stop hit — per-position SL triggered | 2026-02-18 09:50:00 | 570.98 | 570.65 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:25:00 | 577.25 | 575.24 | 0.00 | ORB-long ORB[572.00,576.40] vol=1.8x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:55:00 | 579.59 | 575.56 | 0.00 | T1 1.5R @ 579.59 |
| Stop hit — per-position SL triggered | 2026-02-20 11:45:00 | 577.25 | 575.90 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 597.25 | 595.62 | 0.00 | ORB-long ORB[591.25,596.30] vol=2.5x ATR=2.19 |
| Stop hit — per-position SL triggered | 2026-02-24 09:40:00 | 595.06 | 595.93 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 603.20 | 601.11 | 0.00 | ORB-long ORB[596.20,600.75] vol=9.4x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-02-25 11:30:00 | 601.62 | 601.59 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 600.90 | 599.99 | 0.00 | ORB-long ORB[597.05,600.00] vol=4.2x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-02-26 09:35:00 | 599.72 | 599.81 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:40:00 | 570.45 | 565.93 | 0.00 | ORB-long ORB[560.50,568.30] vol=1.6x ATR=2.70 |
| Stop hit — per-position SL triggered | 2026-03-16 10:20:00 | 567.75 | 568.08 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:25:00 | 577.00 | 581.97 | 0.00 | ORB-short ORB[582.35,588.00] vol=2.0x ATR=2.04 |
| Stop hit — per-position SL triggered | 2026-03-23 10:35:00 | 579.04 | 581.69 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:55:00 | 635.25 | 639.01 | 0.00 | ORB-short ORB[640.90,648.75] vol=1.6x ATR=2.85 |
| Stop hit — per-position SL triggered | 2026-04-09 10:00:00 | 638.10 | 638.83 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 642.30 | 639.81 | 0.00 | ORB-long ORB[634.00,641.00] vol=3.6x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-04-15 09:45:00 | 640.34 | 640.15 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 654.00 | 652.15 | 0.00 | ORB-long ORB[646.85,652.00] vol=3.3x ATR=2.25 |
| Stop hit — per-position SL triggered | 2026-04-17 09:40:00 | 651.75 | 652.16 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 669.40 | 664.93 | 0.00 | ORB-long ORB[661.10,668.70] vol=1.7x ATR=2.04 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 667.36 | 665.81 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 681.00 | 675.59 | 0.00 | ORB-long ORB[666.90,675.40] vol=3.2x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:35:00 | 685.60 | 679.69 | 0.00 | T1 1.5R @ 685.60 |
| Target hit | 2026-04-22 10:50:00 | 681.85 | 682.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 685.25 | 680.99 | 0.00 | ORB-long ORB[670.30,680.40] vol=1.6x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:45:00 | 689.91 | 686.35 | 0.00 | T1 1.5R @ 689.91 |
| Target hit | 2026-04-23 10:05:00 | 686.85 | 687.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 703.50 | 701.57 | 0.00 | ORB-long ORB[695.00,702.00] vol=1.9x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:20:00 | 706.61 | 702.63 | 0.00 | T1 1.5R @ 706.61 |
| Stop hit — per-position SL triggered | 2026-04-28 11:45:00 | 703.50 | 703.16 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 715.55 | 712.50 | 0.00 | ORB-long ORB[706.75,714.00] vol=2.5x ATR=2.79 |
| Stop hit — per-position SL triggered | 2026-04-29 10:05:00 | 712.76 | 712.60 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 723.45 | 718.58 | 0.00 | ORB-long ORB[712.40,720.00] vol=3.7x ATR=2.71 |
| Stop hit — per-position SL triggered | 2026-05-06 09:35:00 | 720.74 | 716.52 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 10:45:00 | 564.85 | 2026-02-12 11:20:00 | 562.57 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-12 10:45:00 | 564.85 | 2026-02-12 12:30:00 | 564.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-13 09:40:00 | 569.00 | 2026-02-13 09:55:00 | 566.76 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-18 09:35:00 | 572.85 | 2026-02-18 09:50:00 | 570.98 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-20 10:25:00 | 577.25 | 2026-02-20 10:55:00 | 579.59 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-20 10:25:00 | 577.25 | 2026-02-20 11:45:00 | 577.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 09:30:00 | 597.25 | 2026-02-24 09:40:00 | 595.06 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-25 10:40:00 | 603.20 | 2026-02-25 11:30:00 | 601.62 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-26 09:30:00 | 600.90 | 2026-02-26 09:35:00 | 599.72 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-03-16 09:40:00 | 570.45 | 2026-03-16 10:20:00 | 567.75 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-03-23 10:25:00 | 577.00 | 2026-03-23 10:35:00 | 579.04 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-09 09:55:00 | 635.25 | 2026-04-09 10:00:00 | 638.10 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-15 09:35:00 | 642.30 | 2026-04-15 09:45:00 | 640.34 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-17 09:30:00 | 654.00 | 2026-04-17 09:40:00 | 651.75 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-21 10:55:00 | 669.40 | 2026-04-21 11:00:00 | 667.36 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-22 09:30:00 | 681.00 | 2026-04-22 09:35:00 | 685.60 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-04-22 09:30:00 | 681.00 | 2026-04-22 10:50:00 | 681.85 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2026-04-23 09:35:00 | 685.25 | 2026-04-23 09:45:00 | 689.91 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-04-23 09:35:00 | 685.25 | 2026-04-23 10:05:00 | 686.85 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2026-04-28 11:05:00 | 703.50 | 2026-04-28 11:20:00 | 706.61 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-04-28 11:05:00 | 703.50 | 2026-04-28 11:45:00 | 703.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:00:00 | 715.55 | 2026-04-29 10:05:00 | 712.76 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-06 09:30:00 | 723.45 | 2026-05-06 09:35:00 | 720.74 | STOP_HIT | 1.00 | -0.37% |
