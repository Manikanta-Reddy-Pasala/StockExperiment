# HDFCLIFE (HDFCLIFE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 619.60
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
| PARTIAL | 13 |
| TARGET_HIT | 7 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 13
- **Target hits / Stop hits / Partials:** 7 / 13 / 13
- **Avg / median % per leg:** 0.27% / 0.32%
- **Sum % (uncompounded):** 8.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 9 | 60.0% | 3 | 6 | 6 | 0.25% | 3.7% |
| BUY @ 2nd Alert (retest1) | 15 | 9 | 60.0% | 3 | 6 | 6 | 0.25% | 3.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 11 | 61.1% | 4 | 7 | 7 | 0.29% | 5.2% |
| SELL @ 2nd Alert (retest1) | 18 | 11 | 61.1% | 4 | 7 | 7 | 0.29% | 5.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 20 | 60.6% | 7 | 13 | 13 | 0.27% | 9.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 707.45 | 708.76 | 0.00 | ORB-short ORB[707.60,713.95] vol=2.1x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:25:00 | 705.19 | 708.16 | 0.00 | T1 1.5R @ 705.19 |
| Target hit | 2026-02-10 15:20:00 | 703.90 | 706.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:20:00 | 699.60 | 701.11 | 0.00 | ORB-short ORB[700.30,705.60] vol=1.6x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-02-11 11:30:00 | 700.88 | 700.71 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:05:00 | 700.15 | 698.33 | 0.00 | ORB-long ORB[693.00,697.70] vol=5.2x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:20:00 | 702.21 | 698.78 | 0.00 | T1 1.5R @ 702.21 |
| Target hit | 2026-02-16 15:20:00 | 706.90 | 702.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:30:00 | 738.20 | 738.87 | 0.00 | ORB-short ORB[738.40,742.80] vol=3.2x ATR=1.39 |
| Stop hit — per-position SL triggered | 2026-02-24 11:15:00 | 739.59 | 738.81 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:00:00 | 722.75 | 727.33 | 0.00 | ORB-short ORB[729.75,736.80] vol=2.2x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:10:00 | 720.47 | 726.74 | 0.00 | T1 1.5R @ 720.47 |
| Target hit | 2026-02-27 15:20:00 | 715.05 | 717.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-03-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 09:40:00 | 705.25 | 698.85 | 0.00 | ORB-long ORB[690.00,698.70] vol=2.2x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-03-04 10:00:00 | 702.11 | 701.35 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 671.55 | 673.51 | 0.00 | ORB-short ORB[672.15,679.20] vol=1.8x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 15:05:00 | 668.19 | 671.53 | 0.00 | T1 1.5R @ 668.19 |
| Target hit | 2026-03-06 15:20:00 | 668.30 | 671.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-03-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:25:00 | 650.95 | 652.32 | 0.00 | ORB-short ORB[651.70,657.50] vol=4.1x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-03-11 10:40:00 | 652.58 | 652.28 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 635.50 | 637.29 | 0.00 | ORB-short ORB[635.70,640.60] vol=2.2x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:40:00 | 633.03 | 636.57 | 0.00 | T1 1.5R @ 633.03 |
| Target hit | 2026-03-13 15:20:00 | 626.30 | 631.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-03-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:40:00 | 619.70 | 623.40 | 0.00 | ORB-short ORB[622.80,628.00] vol=2.7x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:55:00 | 617.24 | 622.90 | 0.00 | T1 1.5R @ 617.24 |
| Stop hit — per-position SL triggered | 2026-03-16 11:00:00 | 619.70 | 622.74 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:40:00 | 632.75 | 628.58 | 0.00 | ORB-long ORB[623.80,628.00] vol=2.8x ATR=1.83 |
| Stop hit — per-position SL triggered | 2026-03-17 11:05:00 | 630.92 | 630.47 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 11:10:00 | 632.65 | 628.62 | 0.00 | ORB-long ORB[624.00,631.50] vol=2.3x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 11:30:00 | 635.34 | 629.01 | 0.00 | T1 1.5R @ 635.34 |
| Stop hit — per-position SL triggered | 2026-03-19 14:10:00 | 632.65 | 632.27 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:50:00 | 613.55 | 603.86 | 0.00 | ORB-long ORB[592.00,600.55] vol=1.8x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:10:00 | 617.45 | 608.53 | 0.00 | T1 1.5R @ 617.45 |
| Target hit | 2026-04-13 15:20:00 | 619.15 | 617.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 605.35 | 606.61 | 0.00 | ORB-short ORB[605.50,610.00] vol=2.5x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:55:00 | 603.35 | 605.71 | 0.00 | T1 1.5R @ 603.35 |
| Stop hit — per-position SL triggered | 2026-04-21 10:05:00 | 605.35 | 605.47 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:35:00 | 606.05 | 610.74 | 0.00 | ORB-short ORB[611.30,617.25] vol=2.1x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-04-22 11:40:00 | 607.53 | 608.41 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:20:00 | 600.15 | 596.32 | 0.00 | ORB-long ORB[590.10,597.15] vol=2.5x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:50:00 | 603.35 | 597.92 | 0.00 | T1 1.5R @ 603.35 |
| Stop hit — per-position SL triggered | 2026-04-27 11:05:00 | 600.15 | 598.26 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 598.20 | 595.44 | 0.00 | ORB-long ORB[589.60,593.15] vol=1.5x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:10:00 | 600.31 | 596.10 | 0.00 | T1 1.5R @ 600.31 |
| Stop hit — per-position SL triggered | 2026-04-29 14:45:00 | 598.20 | 599.84 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:45:00 | 590.95 | 592.86 | 0.00 | ORB-short ORB[591.05,595.80] vol=1.6x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:00:00 | 588.82 | 592.43 | 0.00 | T1 1.5R @ 588.82 |
| Stop hit — per-position SL triggered | 2026-05-04 11:15:00 | 590.95 | 592.20 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 594.35 | 591.79 | 0.00 | ORB-long ORB[586.00,592.00] vol=3.3x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:55:00 | 597.12 | 592.72 | 0.00 | T1 1.5R @ 597.12 |
| Target hit | 2026-05-05 10:30:00 | 595.90 | 596.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — BUY (started 2026-05-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:30:00 | 615.15 | 612.08 | 0.00 | ORB-long ORB[608.85,614.55] vol=2.5x ATR=2.40 |
| Stop hit — per-position SL triggered | 2026-05-07 09:50:00 | 612.75 | 612.39 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:35:00 | 707.45 | 2026-02-10 11:25:00 | 705.19 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-10 10:35:00 | 707.45 | 2026-02-10 15:20:00 | 703.90 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-11 10:20:00 | 699.60 | 2026-02-11 11:30:00 | 700.88 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-16 11:05:00 | 700.15 | 2026-02-16 11:20:00 | 702.21 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2026-02-16 11:05:00 | 700.15 | 2026-02-16 15:20:00 | 706.90 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2026-02-24 10:30:00 | 738.20 | 2026-02-24 11:15:00 | 739.59 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-27 11:00:00 | 722.75 | 2026-02-27 11:10:00 | 720.47 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-27 11:00:00 | 722.75 | 2026-02-27 15:20:00 | 715.05 | TARGET_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2026-03-04 09:40:00 | 705.25 | 2026-03-04 10:00:00 | 702.11 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-03-06 10:45:00 | 671.55 | 2026-03-06 15:05:00 | 668.19 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-06 10:45:00 | 671.55 | 2026-03-06 15:20:00 | 668.30 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2026-03-11 10:25:00 | 650.95 | 2026-03-11 10:40:00 | 652.58 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-13 10:50:00 | 635.50 | 2026-03-13 11:40:00 | 633.03 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-03-13 10:50:00 | 635.50 | 2026-03-13 15:20:00 | 626.30 | TARGET_HIT | 0.50 | 1.45% |
| SELL | retest1 | 2026-03-16 10:40:00 | 619.70 | 2026-03-16 10:55:00 | 617.24 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-03-16 10:40:00 | 619.70 | 2026-03-16 11:00:00 | 619.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:40:00 | 632.75 | 2026-03-17 11:05:00 | 630.92 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-19 11:10:00 | 632.65 | 2026-03-19 11:30:00 | 635.34 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-03-19 11:10:00 | 632.65 | 2026-03-19 14:10:00 | 632.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 09:50:00 | 613.55 | 2026-04-13 10:10:00 | 617.45 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-04-13 09:50:00 | 613.55 | 2026-04-13 15:20:00 | 619.15 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2026-04-21 09:35:00 | 605.35 | 2026-04-21 09:55:00 | 603.35 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-04-21 09:35:00 | 605.35 | 2026-04-21 10:05:00 | 605.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 10:35:00 | 606.05 | 2026-04-22 11:40:00 | 607.53 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-27 10:20:00 | 600.15 | 2026-04-27 10:50:00 | 603.35 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-27 10:20:00 | 600.15 | 2026-04-27 11:05:00 | 600.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 11:00:00 | 598.20 | 2026-04-29 11:10:00 | 600.31 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-29 11:00:00 | 598.20 | 2026-04-29 14:45:00 | 598.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-04 10:45:00 | 590.95 | 2026-05-04 11:00:00 | 588.82 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-05-04 10:45:00 | 590.95 | 2026-05-04 11:15:00 | 590.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:40:00 | 594.35 | 2026-05-05 09:55:00 | 597.12 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-05-05 09:40:00 | 594.35 | 2026-05-05 10:30:00 | 595.90 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2026-05-07 09:30:00 | 615.15 | 2026-05-07 09:50:00 | 612.75 | STOP_HIT | 1.00 | -0.39% |
