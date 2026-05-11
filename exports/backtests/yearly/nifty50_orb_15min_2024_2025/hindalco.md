# HINDALCO (HINDALCO)

## Backtest Summary

- **Window:** 2024-12-09 09:15:00 → 2026-05-08 15:25:00 (24538 bars)
- **Last close:** 1044.50
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
| ENTRY1 | 25 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 20
- **Target hits / Stop hits / Partials:** 5 / 20 / 10
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 3.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 3 | 23.1% | 0 | 10 | 3 | -0.06% | -0.7% |
| BUY @ 2nd Alert (retest1) | 13 | 3 | 23.1% | 0 | 10 | 3 | -0.06% | -0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 22 | 12 | 54.5% | 5 | 10 | 7 | 0.21% | 4.7% |
| SELL @ 2nd Alert (retest1) | 22 | 12 | 54.5% | 5 | 10 | 7 | 0.21% | 4.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 35 | 15 | 42.9% | 5 | 20 | 10 | 0.11% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:40:00 | 675.60 | 671.71 | 0.00 | ORB-long ORB[668.05,672.00] vol=1.5x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 10:00:00 | 678.14 | 674.48 | 0.00 | T1 1.5R @ 678.14 |
| Stop hit — per-position SL triggered | 2024-12-11 10:35:00 | 675.60 | 675.53 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-12-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:05:00 | 656.40 | 659.08 | 0.00 | ORB-short ORB[659.00,666.60] vol=2.2x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 12:15:00 | 654.12 | 657.98 | 0.00 | T1 1.5R @ 654.12 |
| Target hit | 2024-12-16 15:20:00 | 653.05 | 656.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-12-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:40:00 | 646.95 | 649.08 | 0.00 | ORB-short ORB[648.35,653.60] vol=2.0x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-12-17 10:45:00 | 648.29 | 649.04 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:40:00 | 623.05 | 618.89 | 0.00 | ORB-long ORB[614.10,622.95] vol=2.0x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-12-19 09:50:00 | 620.30 | 619.55 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-12-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 10:40:00 | 638.75 | 632.08 | 0.00 | ORB-long ORB[625.20,631.00] vol=1.6x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-12-20 11:05:00 | 636.66 | 633.57 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 11:15:00 | 630.70 | 633.28 | 0.00 | ORB-short ORB[630.90,636.00] vol=5.0x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 12:15:00 | 628.75 | 631.85 | 0.00 | T1 1.5R @ 628.75 |
| Stop hit — per-position SL triggered | 2024-12-24 12:40:00 | 630.70 | 631.42 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-12-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:30:00 | 617.45 | 616.34 | 0.00 | ORB-long ORB[611.15,617.40] vol=1.8x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 10:45:00 | 620.18 | 617.10 | 0.00 | T1 1.5R @ 620.18 |
| Stop hit — per-position SL triggered | 2024-12-30 10:55:00 | 617.45 | 617.17 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-01-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 09:45:00 | 595.75 | 598.26 | 0.00 | ORB-short ORB[596.50,603.75] vol=2.7x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-01-01 10:00:00 | 597.55 | 597.32 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-01-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 09:30:00 | 585.75 | 588.87 | 0.00 | ORB-short ORB[586.15,593.85] vol=1.5x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 09:40:00 | 582.71 | 587.20 | 0.00 | T1 1.5R @ 582.71 |
| Target hit | 2025-01-06 15:20:00 | 573.05 | 578.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2025-01-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:55:00 | 619.45 | 615.60 | 0.00 | ORB-long ORB[612.25,618.90] vol=1.6x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-01-20 11:10:00 | 618.18 | 615.84 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-01-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:10:00 | 601.50 | 606.73 | 0.00 | ORB-short ORB[607.95,614.80] vol=1.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-01-24 10:25:00 | 603.39 | 605.92 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-01-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 11:00:00 | 592.05 | 595.78 | 0.00 | ORB-short ORB[594.35,601.00] vol=1.7x ATR=2.01 |
| Stop hit — per-position SL triggered | 2025-01-27 11:05:00 | 594.06 | 595.74 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-01-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 11:00:00 | 591.50 | 587.88 | 0.00 | ORB-long ORB[584.30,590.85] vol=1.7x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:15:00 | 594.22 | 588.73 | 0.00 | T1 1.5R @ 594.22 |
| Stop hit — per-position SL triggered | 2025-01-31 11:30:00 | 591.50 | 589.12 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-02-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:40:00 | 603.30 | 598.52 | 0.00 | ORB-long ORB[594.00,598.00] vol=2.1x ATR=2.00 |
| Stop hit — per-position SL triggered | 2025-02-07 10:45:00 | 601.30 | 598.75 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-02-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-17 10:30:00 | 597.05 | 599.39 | 0.00 | ORB-short ORB[599.75,607.90] vol=1.6x ATR=2.84 |
| Stop hit — per-position SL triggered | 2025-02-17 10:40:00 | 599.89 | 599.52 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-03-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 10:45:00 | 680.95 | 686.45 | 0.00 | ORB-short ORB[684.15,692.95] vol=1.8x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 11:05:00 | 678.11 | 684.98 | 0.00 | T1 1.5R @ 678.11 |
| Target hit | 2025-03-13 15:20:00 | 677.20 | 678.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-03-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:45:00 | 693.50 | 698.15 | 0.00 | ORB-short ORB[694.20,702.40] vol=1.5x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-03-26 10:00:00 | 695.72 | 697.65 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-03-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 10:55:00 | 685.70 | 687.27 | 0.00 | ORB-short ORB[688.35,697.45] vol=1.6x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 11:10:00 | 682.79 | 687.00 | 0.00 | T1 1.5R @ 682.79 |
| Target hit | 2025-03-28 14:25:00 | 684.75 | 684.66 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — SELL (started 2025-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 11:00:00 | 669.00 | 676.39 | 0.00 | ORB-short ORB[672.00,681.50] vol=1.8x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 11:45:00 | 665.70 | 674.30 | 0.00 | T1 1.5R @ 665.70 |
| Target hit | 2025-04-01 15:20:00 | 663.55 | 669.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2025-04-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 10:55:00 | 651.85 | 652.73 | 0.00 | ORB-short ORB[652.35,658.00] vol=1.6x ATR=2.13 |
| Stop hit — per-position SL triggered | 2025-04-03 11:35:00 | 653.98 | 652.45 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-04-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:35:00 | 619.60 | 625.62 | 0.00 | ORB-short ORB[630.55,634.30] vol=2.2x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:55:00 | 616.29 | 624.14 | 0.00 | T1 1.5R @ 616.29 |
| Stop hit — per-position SL triggered | 2025-04-25 12:10:00 | 619.60 | 622.35 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 11:05:00 | 630.05 | 628.37 | 0.00 | ORB-long ORB[622.40,629.00] vol=1.8x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-04-28 11:10:00 | 628.68 | 628.39 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-04-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 10:45:00 | 632.00 | 627.63 | 0.00 | ORB-long ORB[620.10,627.00] vol=1.8x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-04-30 11:05:00 | 630.45 | 628.33 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-05-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:10:00 | 640.80 | 635.55 | 0.00 | ORB-long ORB[629.55,636.00] vol=2.9x ATR=1.62 |
| Stop hit — per-position SL triggered | 2025-05-05 11:15:00 | 639.18 | 635.74 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 09:50:00 | 628.05 | 630.10 | 0.00 | ORB-short ORB[628.55,636.30] vol=3.5x ATR=1.62 |
| Stop hit — per-position SL triggered | 2025-05-08 10:05:00 | 629.67 | 629.95 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-12-11 09:40:00 | 675.60 | 2024-12-11 10:00:00 | 678.14 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-12-11 09:40:00 | 675.60 | 2024-12-11 10:35:00 | 675.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-16 11:05:00 | 656.40 | 2024-12-16 12:15:00 | 654.12 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-12-16 11:05:00 | 656.40 | 2024-12-16 15:20:00 | 653.05 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2024-12-17 10:40:00 | 646.95 | 2024-12-17 10:45:00 | 648.29 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-12-19 09:40:00 | 623.05 | 2024-12-19 09:50:00 | 620.30 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-12-20 10:40:00 | 638.75 | 2024-12-20 11:05:00 | 636.66 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-24 11:15:00 | 630.70 | 2024-12-24 12:15:00 | 628.75 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-12-24 11:15:00 | 630.70 | 2024-12-24 12:40:00 | 630.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 10:30:00 | 617.45 | 2024-12-30 10:45:00 | 620.18 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-12-30 10:30:00 | 617.45 | 2024-12-30 10:55:00 | 617.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-01 09:45:00 | 595.75 | 2025-01-01 10:00:00 | 597.55 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-06 09:30:00 | 585.75 | 2025-01-06 09:40:00 | 582.71 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-01-06 09:30:00 | 585.75 | 2025-01-06 15:20:00 | 573.05 | TARGET_HIT | 0.50 | 2.17% |
| BUY | retest1 | 2025-01-20 10:55:00 | 619.45 | 2025-01-20 11:10:00 | 618.18 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-01-24 10:10:00 | 601.50 | 2025-01-24 10:25:00 | 603.39 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-27 11:00:00 | 592.05 | 2025-01-27 11:05:00 | 594.06 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-31 11:00:00 | 591.50 | 2025-01-31 11:15:00 | 594.22 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-01-31 11:00:00 | 591.50 | 2025-01-31 11:30:00 | 591.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 10:40:00 | 603.30 | 2025-02-07 10:45:00 | 601.30 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-17 10:30:00 | 597.05 | 2025-02-17 10:40:00 | 599.89 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-03-13 10:45:00 | 680.95 | 2025-03-13 11:05:00 | 678.11 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-03-13 10:45:00 | 680.95 | 2025-03-13 15:20:00 | 677.20 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2025-03-26 09:45:00 | 693.50 | 2025-03-26 10:00:00 | 695.72 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-03-28 10:55:00 | 685.70 | 2025-03-28 11:10:00 | 682.79 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-03-28 10:55:00 | 685.70 | 2025-03-28 14:25:00 | 684.75 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2025-04-01 11:00:00 | 669.00 | 2025-04-01 11:45:00 | 665.70 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-04-01 11:00:00 | 669.00 | 2025-04-01 15:20:00 | 663.55 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2025-04-03 10:55:00 | 651.85 | 2025-04-03 11:35:00 | 653.98 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-04-25 10:35:00 | 619.60 | 2025-04-25 10:55:00 | 616.29 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-04-25 10:35:00 | 619.60 | 2025-04-25 12:10:00 | 619.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-28 11:05:00 | 630.05 | 2025-04-28 11:10:00 | 628.68 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-04-30 10:45:00 | 632.00 | 2025-04-30 11:05:00 | 630.45 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-05-05 11:10:00 | 640.80 | 2025-05-05 11:15:00 | 639.18 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-05-08 09:50:00 | 628.05 | 2025-05-08 10:05:00 | 629.67 | STOP_HIT | 1.00 | -0.26% |
