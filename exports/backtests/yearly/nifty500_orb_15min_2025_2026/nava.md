# Nava Ltd. (NAVA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-10-07 15:25:00 (7800 bars)
- **Last close:** 637.40
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
| ENTRY1 | 30 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 26
- **Target hits / Stop hits / Partials:** 4 / 26 / 11
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 4.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 6 | 25.0% | 1 | 18 | 5 | -0.06% | -1.5% |
| BUY @ 2nd Alert (retest1) | 24 | 6 | 25.0% | 1 | 18 | 5 | -0.06% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 9 | 52.9% | 3 | 8 | 6 | 0.38% | 6.5% |
| SELL @ 2nd Alert (retest1) | 17 | 9 | 52.9% | 3 | 8 | 6 | 0.38% | 6.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 41 | 15 | 36.6% | 4 | 26 | 11 | 0.12% | 5.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-14 10:05:00 | 469.70 | 473.07 | 0.00 | ORB-short ORB[471.45,476.00] vol=5.3x ATR=2.37 |
| Stop hit — per-position SL triggered | 2025-05-14 10:15:00 | 472.07 | 472.84 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 470.35 | 467.88 | 0.00 | ORB-long ORB[463.50,470.00] vol=2.9x ATR=2.02 |
| Stop hit — per-position SL triggered | 2025-05-15 09:35:00 | 468.33 | 467.81 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:40:00 | 481.30 | 478.66 | 0.00 | ORB-long ORB[475.00,481.00] vol=1.6x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-05-16 09:45:00 | 479.38 | 478.71 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 10:45:00 | 455.65 | 457.96 | 0.00 | ORB-short ORB[456.30,462.95] vol=1.6x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 11:30:00 | 453.45 | 457.38 | 0.00 | T1 1.5R @ 453.45 |
| Stop hit — per-position SL triggered | 2025-05-26 12:10:00 | 455.65 | 455.65 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 10:05:00 | 488.90 | 482.77 | 0.00 | ORB-long ORB[476.65,483.95] vol=2.0x ATR=2.78 |
| Stop hit — per-position SL triggered | 2025-05-30 10:10:00 | 486.12 | 483.28 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:45:00 | 532.05 | 527.84 | 0.00 | ORB-long ORB[522.00,529.80] vol=3.4x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 09:50:00 | 535.67 | 529.57 | 0.00 | T1 1.5R @ 535.67 |
| Stop hit — per-position SL triggered | 2025-06-11 09:55:00 | 532.05 | 529.96 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:15:00 | 580.10 | 588.49 | 0.00 | ORB-short ORB[590.30,598.00] vol=1.8x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-06-26 11:30:00 | 582.02 | 588.32 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:30:00 | 601.60 | 598.48 | 0.00 | ORB-long ORB[591.25,597.00] vol=7.1x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:35:00 | 605.81 | 599.30 | 0.00 | T1 1.5R @ 605.81 |
| Stop hit — per-position SL triggered | 2025-06-27 09:40:00 | 601.60 | 599.58 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-07-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-02 11:05:00 | 600.25 | 596.24 | 0.00 | ORB-long ORB[590.05,599.00] vol=6.9x ATR=2.59 |
| Stop hit — per-position SL triggered | 2025-07-02 12:25:00 | 597.66 | 597.16 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:50:00 | 619.00 | 615.69 | 0.00 | ORB-long ORB[610.05,615.00] vol=7.4x ATR=3.24 |
| Stop hit — per-position SL triggered | 2025-07-04 10:05:00 | 615.76 | 615.97 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:05:00 | 588.40 | 591.55 | 0.00 | ORB-short ORB[590.00,595.00] vol=1.6x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 589.86 | 591.49 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:50:00 | 592.70 | 589.85 | 0.00 | ORB-long ORB[586.00,590.95] vol=1.5x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-07-14 10:35:00 | 590.72 | 590.83 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:15:00 | 616.00 | 609.88 | 0.00 | ORB-long ORB[604.50,612.00] vol=2.3x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-07-15 10:30:00 | 613.07 | 610.71 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 615.15 | 611.74 | 0.00 | ORB-long ORB[608.00,614.90] vol=2.4x ATR=2.31 |
| Stop hit — per-position SL triggered | 2025-07-23 09:35:00 | 612.84 | 611.88 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:35:00 | 640.50 | 636.80 | 0.00 | ORB-long ORB[632.00,637.95] vol=4.4x ATR=2.37 |
| Stop hit — per-position SL triggered | 2025-07-24 09:40:00 | 638.13 | 636.76 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:50:00 | 626.55 | 630.22 | 0.00 | ORB-short ORB[629.00,636.00] vol=2.2x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:20:00 | 622.82 | 628.77 | 0.00 | T1 1.5R @ 622.82 |
| Target hit | 2025-07-25 15:20:00 | 610.45 | 617.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2025-07-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:30:00 | 618.45 | 614.98 | 0.00 | ORB-long ORB[608.40,617.50] vol=2.2x ATR=3.59 |
| Stop hit — per-position SL triggered | 2025-07-28 10:10:00 | 614.86 | 615.73 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:10:00 | 622.95 | 625.61 | 0.00 | ORB-short ORB[623.50,631.00] vol=1.7x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 12:00:00 | 618.53 | 623.51 | 0.00 | T1 1.5R @ 618.53 |
| Stop hit — per-position SL triggered | 2025-08-12 13:10:00 | 622.95 | 622.59 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 10:35:00 | 625.85 | 621.14 | 0.00 | ORB-long ORB[615.25,620.80] vol=6.9x ATR=2.18 |
| Stop hit — per-position SL triggered | 2025-08-14 10:45:00 | 623.67 | 621.58 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 11:00:00 | 605.60 | 600.48 | 0.00 | ORB-long ORB[596.00,601.70] vol=2.9x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 11:05:00 | 607.70 | 601.82 | 0.00 | T1 1.5R @ 607.70 |
| Stop hit — per-position SL triggered | 2025-08-21 11:10:00 | 605.60 | 602.53 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:10:00 | 660.50 | 656.90 | 0.00 | ORB-long ORB[653.85,659.85] vol=3.2x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 10:15:00 | 665.09 | 659.58 | 0.00 | T1 1.5R @ 665.09 |
| Target hit | 2025-08-29 11:00:00 | 670.05 | 680.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — BUY (started 2025-09-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:05:00 | 687.35 | 684.31 | 0.00 | ORB-long ORB[676.10,685.95] vol=2.0x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-09-08 15:20:00 | 685.00 | 686.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-09-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:50:00 | 709.00 | 712.70 | 0.00 | ORB-short ORB[711.00,720.10] vol=1.6x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 13:50:00 | 704.03 | 710.21 | 0.00 | T1 1.5R @ 704.03 |
| Stop hit — per-position SL triggered | 2025-09-17 15:05:00 | 709.00 | 708.79 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:55:00 | 711.55 | 708.02 | 0.00 | ORB-long ORB[701.60,709.00] vol=5.3x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 10:10:00 | 715.88 | 708.85 | 0.00 | T1 1.5R @ 715.88 |
| Stop hit — per-position SL triggered | 2025-09-18 11:25:00 | 711.55 | 712.24 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 09:35:00 | 721.40 | 713.97 | 0.00 | ORB-long ORB[709.00,715.00] vol=2.6x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-09-23 09:40:00 | 718.59 | 714.86 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:40:00 | 710.60 | 715.00 | 0.00 | ORB-short ORB[712.05,722.60] vol=2.3x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:45:00 | 707.66 | 714.26 | 0.00 | T1 1.5R @ 707.66 |
| Target hit | 2025-09-24 15:20:00 | 703.95 | 710.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-09-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 09:50:00 | 711.60 | 706.79 | 0.00 | ORB-long ORB[700.10,708.70] vol=5.3x ATR=3.12 |
| Stop hit — per-position SL triggered | 2025-09-25 10:05:00 | 708.48 | 707.54 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 09:40:00 | 659.85 | 663.36 | 0.00 | ORB-short ORB[662.00,670.10] vol=2.3x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 10:15:00 | 654.94 | 660.75 | 0.00 | T1 1.5R @ 654.94 |
| Target hit | 2025-09-30 15:00:00 | 652.65 | 652.46 | 0.00 | Trail-exit close>VWAP |

### Cycle 29 — SELL (started 2025-10-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 10:00:00 | 643.00 | 648.17 | 0.00 | ORB-short ORB[646.70,651.05] vol=1.9x ATR=2.45 |
| Stop hit — per-position SL triggered | 2025-10-01 10:10:00 | 645.45 | 647.73 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:05:00 | 636.90 | 641.29 | 0.00 | ORB-short ORB[640.55,648.25] vol=2.6x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-10-07 11:25:00 | 638.56 | 640.81 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-14 10:05:00 | 469.70 | 2025-05-14 10:15:00 | 472.07 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-05-15 09:30:00 | 470.35 | 2025-05-15 09:35:00 | 468.33 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-05-16 09:40:00 | 481.30 | 2025-05-16 09:45:00 | 479.38 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-05-26 10:45:00 | 455.65 | 2025-05-26 11:30:00 | 453.45 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-05-26 10:45:00 | 455.65 | 2025-05-26 12:10:00 | 455.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-30 10:05:00 | 488.90 | 2025-05-30 10:10:00 | 486.12 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2025-06-11 09:45:00 | 532.05 | 2025-06-11 09:50:00 | 535.67 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-06-11 09:45:00 | 532.05 | 2025-06-11 09:55:00 | 532.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-26 11:15:00 | 580.10 | 2025-06-26 11:30:00 | 582.02 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-27 09:30:00 | 601.60 | 2025-06-27 09:35:00 | 605.81 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-06-27 09:30:00 | 601.60 | 2025-06-27 09:40:00 | 601.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-02 11:05:00 | 600.25 | 2025-07-02 12:25:00 | 597.66 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-07-04 09:50:00 | 619.00 | 2025-07-04 10:05:00 | 615.76 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-07-11 11:05:00 | 588.40 | 2025-07-11 11:10:00 | 589.86 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-14 09:50:00 | 592.70 | 2025-07-14 10:35:00 | 590.72 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-07-15 10:15:00 | 616.00 | 2025-07-15 10:30:00 | 613.07 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-07-23 09:30:00 | 615.15 | 2025-07-23 09:35:00 | 612.84 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-07-24 09:35:00 | 640.50 | 2025-07-24 09:40:00 | 638.13 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-07-25 09:50:00 | 626.55 | 2025-07-25 10:20:00 | 622.82 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-07-25 09:50:00 | 626.55 | 2025-07-25 15:20:00 | 610.45 | TARGET_HIT | 0.50 | 2.57% |
| BUY | retest1 | 2025-07-28 09:30:00 | 618.45 | 2025-07-28 10:10:00 | 614.86 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2025-08-12 10:10:00 | 622.95 | 2025-08-12 12:00:00 | 618.53 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2025-08-12 10:10:00 | 622.95 | 2025-08-12 13:10:00 | 622.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-14 10:35:00 | 625.85 | 2025-08-14 10:45:00 | 623.67 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-08-21 11:00:00 | 605.60 | 2025-08-21 11:05:00 | 607.70 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-08-21 11:00:00 | 605.60 | 2025-08-21 11:10:00 | 605.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-29 10:10:00 | 660.50 | 2025-08-29 10:15:00 | 665.09 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-08-29 10:10:00 | 660.50 | 2025-08-29 11:00:00 | 670.05 | TARGET_HIT | 0.50 | 1.45% |
| BUY | retest1 | 2025-09-08 10:05:00 | 687.35 | 2025-09-08 15:20:00 | 685.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-09-17 09:50:00 | 709.00 | 2025-09-17 13:50:00 | 704.03 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2025-09-17 09:50:00 | 709.00 | 2025-09-17 15:05:00 | 709.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-18 09:55:00 | 711.55 | 2025-09-18 10:10:00 | 715.88 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-09-18 09:55:00 | 711.55 | 2025-09-18 11:25:00 | 711.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-23 09:35:00 | 721.40 | 2025-09-23 09:40:00 | 718.59 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-09-24 10:40:00 | 710.60 | 2025-09-24 10:45:00 | 707.66 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-09-24 10:40:00 | 710.60 | 2025-09-24 15:20:00 | 703.95 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2025-09-25 09:50:00 | 711.60 | 2025-09-25 10:05:00 | 708.48 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-09-30 09:40:00 | 659.85 | 2025-09-30 10:15:00 | 654.94 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2025-09-30 09:40:00 | 659.85 | 2025-09-30 15:00:00 | 652.65 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2025-10-01 10:00:00 | 643.00 | 2025-10-01 10:10:00 | 645.45 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-10-07 11:05:00 | 636.90 | 2025-10-07 11:25:00 | 638.56 | STOP_HIT | 1.00 | -0.26% |
