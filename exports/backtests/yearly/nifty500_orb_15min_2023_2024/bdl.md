# Bharat Dynamics Ltd. (BDL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-06-04 15:25:00 (19704 bars)
- **Last close:** 1436.85
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
| ENTRY1 | 77 |
| ENTRY2 | 0 |
| PARTIAL | 36 |
| TARGET_HIT | 11 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 67
- **Target hits / Stop hits / Partials:** 11 / 66 / 36
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 21.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 24 | 38.7% | 6 | 37 | 19 | 0.15% | 9.2% |
| BUY @ 2nd Alert (retest1) | 62 | 24 | 38.7% | 6 | 37 | 19 | 0.15% | 9.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 51 | 22 | 43.1% | 5 | 29 | 17 | 0.25% | 12.5% |
| SELL @ 2nd Alert (retest1) | 51 | 22 | 43.1% | 5 | 29 | 17 | 0.25% | 12.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 113 | 46 | 40.7% | 11 | 66 | 36 | 0.19% | 21.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 09:35:00 | 509.55 | 506.35 | 0.00 | ORB-long ORB[503.00,508.60] vol=2.1x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-15 09:40:00 | 511.59 | 507.16 | 0.00 | T1 1.5R @ 511.59 |
| Stop hit — per-position SL triggered | 2023-05-15 09:45:00 | 509.55 | 507.40 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:40:00 | 528.98 | 534.18 | 0.00 | ORB-short ORB[533.00,540.92] vol=1.6x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 10:05:00 | 525.51 | 530.72 | 0.00 | T1 1.5R @ 525.51 |
| Stop hit — per-position SL triggered | 2023-05-19 13:25:00 | 528.98 | 528.68 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 09:50:00 | 532.98 | 526.55 | 0.00 | ORB-long ORB[522.13,528.65] vol=2.4x ATR=2.26 |
| Stop hit — per-position SL triggered | 2023-05-22 09:55:00 | 530.72 | 527.15 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:55:00 | 535.58 | 533.52 | 0.00 | ORB-long ORB[529.03,534.90] vol=1.9x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 10:00:00 | 538.33 | 535.63 | 0.00 | T1 1.5R @ 538.33 |
| Stop hit — per-position SL triggered | 2023-05-25 11:00:00 | 535.58 | 536.73 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 09:45:00 | 526.45 | 523.18 | 0.00 | ORB-long ORB[518.50,525.00] vol=1.5x ATR=1.89 |
| Stop hit — per-position SL triggered | 2023-05-30 10:30:00 | 524.56 | 524.02 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 10:55:00 | 558.30 | 555.24 | 0.00 | ORB-long ORB[551.50,558.00] vol=5.6x ATR=1.80 |
| Stop hit — per-position SL triggered | 2023-06-02 11:15:00 | 556.50 | 555.39 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 09:55:00 | 568.00 | 565.73 | 0.00 | ORB-long ORB[562.50,567.45] vol=2.4x ATR=2.02 |
| Stop hit — per-position SL triggered | 2023-06-05 10:25:00 | 565.98 | 566.03 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 10:40:00 | 565.00 | 567.57 | 0.00 | ORB-short ORB[567.38,574.33] vol=2.1x ATR=1.18 |
| Stop hit — per-position SL triggered | 2023-06-13 11:05:00 | 566.18 | 567.30 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:30:00 | 575.00 | 571.26 | 0.00 | ORB-long ORB[565.38,573.00] vol=2.8x ATR=2.47 |
| Stop hit — per-position SL triggered | 2023-06-16 09:35:00 | 572.53 | 571.46 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-19 09:55:00 | 617.80 | 605.58 | 0.00 | ORB-long ORB[593.88,599.38] vol=6.9x ATR=4.32 |
| Stop hit — per-position SL triggered | 2023-06-19 10:00:00 | 613.48 | 607.48 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 10:00:00 | 608.20 | 600.03 | 0.00 | ORB-long ORB[596.48,605.48] vol=2.0x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 10:05:00 | 613.18 | 602.33 | 0.00 | T1 1.5R @ 613.18 |
| Stop hit — per-position SL triggered | 2023-06-20 10:10:00 | 608.20 | 604.23 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-07-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 10:00:00 | 559.67 | 561.75 | 0.00 | ORB-short ORB[560.53,566.50] vol=1.5x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 10:15:00 | 556.13 | 561.18 | 0.00 | T1 1.5R @ 556.13 |
| Stop hit — per-position SL triggered | 2023-07-03 10:30:00 | 559.67 | 560.95 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:30:00 | 557.80 | 554.57 | 0.00 | ORB-long ORB[551.03,555.33] vol=1.9x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 09:35:00 | 560.09 | 555.97 | 0.00 | T1 1.5R @ 560.09 |
| Stop hit — per-position SL triggered | 2023-07-05 09:45:00 | 557.80 | 556.69 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 11:15:00 | 567.50 | 564.24 | 0.00 | ORB-long ORB[559.85,566.48] vol=2.7x ATR=1.46 |
| Stop hit — per-position SL triggered | 2023-07-06 11:20:00 | 566.04 | 564.31 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:50:00 | 600.45 | 596.40 | 0.00 | ORB-long ORB[590.45,597.48] vol=3.7x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-25 09:55:00 | 604.52 | 599.29 | 0.00 | T1 1.5R @ 604.52 |
| Stop hit — per-position SL triggered | 2023-07-25 10:00:00 | 600.45 | 599.48 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 09:35:00 | 598.03 | 602.09 | 0.00 | ORB-short ORB[599.92,607.00] vol=1.5x ATR=1.59 |
| Stop hit — per-position SL triggered | 2023-07-28 09:40:00 | 599.62 | 601.89 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 09:40:00 | 613.13 | 608.76 | 0.00 | ORB-long ORB[601.03,609.85] vol=5.6x ATR=2.08 |
| Stop hit — per-position SL triggered | 2023-07-31 09:45:00 | 611.05 | 608.90 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-08-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 10:35:00 | 596.90 | 601.98 | 0.00 | ORB-short ORB[600.50,608.30] vol=2.4x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 13:35:00 | 593.85 | 599.01 | 0.00 | T1 1.5R @ 593.85 |
| Stop hit — per-position SL triggered | 2023-08-02 15:15:00 | 596.90 | 596.63 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 11:15:00 | 589.10 | 593.70 | 0.00 | ORB-short ORB[592.03,598.05] vol=1.6x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-08-04 11:25:00 | 590.93 | 593.60 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-08-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 10:50:00 | 568.40 | 564.29 | 0.00 | ORB-long ORB[560.55,565.00] vol=2.1x ATR=1.71 |
| Stop hit — per-position SL triggered | 2023-08-09 11:00:00 | 566.69 | 564.50 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-08-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 09:35:00 | 567.42 | 570.80 | 0.00 | ORB-short ORB[568.00,574.30] vol=1.6x ATR=2.28 |
| Stop hit — per-position SL triggered | 2023-08-10 10:00:00 | 569.70 | 570.21 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-08-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 10:55:00 | 569.70 | 565.97 | 0.00 | ORB-long ORB[561.53,567.38] vol=2.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2023-08-11 11:00:00 | 568.07 | 566.07 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 10:00:00 | 560.67 | 564.46 | 0.00 | ORB-short ORB[563.40,569.38] vol=2.6x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 10:10:00 | 557.53 | 561.58 | 0.00 | T1 1.5R @ 557.53 |
| Target hit | 2023-08-18 15:20:00 | 553.50 | 556.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2023-08-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 10:30:00 | 565.30 | 562.44 | 0.00 | ORB-long ORB[557.08,564.45] vol=1.9x ATR=1.39 |
| Stop hit — per-position SL triggered | 2023-08-22 10:40:00 | 563.91 | 562.61 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 09:45:00 | 584.48 | 580.02 | 0.00 | ORB-long ORB[574.13,582.00] vol=1.8x ATR=2.25 |
| Stop hit — per-position SL triggered | 2023-08-23 10:35:00 | 582.23 | 581.66 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 10:55:00 | 579.13 | 584.82 | 0.00 | ORB-short ORB[583.30,590.20] vol=3.2x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 11:40:00 | 576.07 | 583.61 | 0.00 | T1 1.5R @ 576.07 |
| Stop hit — per-position SL triggered | 2023-08-24 11:55:00 | 579.13 | 583.29 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-08-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:25:00 | 569.08 | 575.64 | 0.00 | ORB-short ORB[573.50,580.85] vol=2.5x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 10:30:00 | 566.01 | 574.28 | 0.00 | T1 1.5R @ 566.01 |
| Stop hit — per-position SL triggered | 2023-08-25 10:40:00 | 569.08 | 573.61 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:30:00 | 566.95 | 564.93 | 0.00 | ORB-long ORB[562.40,566.75] vol=2.0x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 09:40:00 | 569.44 | 566.93 | 0.00 | T1 1.5R @ 569.44 |
| Target hit | 2023-08-30 13:00:00 | 572.28 | 573.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2023-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 10:15:00 | 575.53 | 570.67 | 0.00 | ORB-long ORB[564.00,567.73] vol=4.2x ATR=2.08 |
| Stop hit — per-position SL triggered | 2023-09-01 10:20:00 | 573.45 | 571.14 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-09-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 10:10:00 | 589.00 | 582.63 | 0.00 | ORB-long ORB[573.90,581.10] vol=2.1x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 10:20:00 | 592.86 | 586.26 | 0.00 | T1 1.5R @ 592.86 |
| Target hit | 2023-09-04 11:10:00 | 595.25 | 595.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2023-09-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 11:00:00 | 594.00 | 593.36 | 0.00 | ORB-long ORB[585.53,593.25] vol=3.6x ATR=2.28 |
| Stop hit — per-position SL triggered | 2023-09-05 11:30:00 | 591.72 | 593.31 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-09-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-06 11:10:00 | 583.95 | 586.29 | 0.00 | ORB-short ORB[584.53,589.50] vol=2.3x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 12:45:00 | 581.85 | 585.27 | 0.00 | T1 1.5R @ 581.85 |
| Target hit | 2023-09-06 15:20:00 | 574.98 | 580.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2023-09-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 09:50:00 | 606.00 | 600.86 | 0.00 | ORB-long ORB[595.08,603.50] vol=2.5x ATR=2.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 10:10:00 | 610.35 | 603.65 | 0.00 | T1 1.5R @ 610.35 |
| Stop hit — per-position SL triggered | 2023-09-08 10:30:00 | 606.00 | 605.39 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-11 09:30:00 | 592.78 | 598.59 | 0.00 | ORB-short ORB[596.00,603.17] vol=1.6x ATR=2.42 |
| Stop hit — per-position SL triggered | 2023-09-11 09:35:00 | 595.20 | 598.02 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-09-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-15 09:35:00 | 534.65 | 538.04 | 0.00 | ORB-short ORB[535.20,542.40] vol=1.5x ATR=1.74 |
| Stop hit — per-position SL triggered | 2023-09-15 09:40:00 | 536.39 | 537.88 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-09-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 10:20:00 | 506.05 | 512.65 | 0.00 | ORB-short ORB[510.48,517.40] vol=2.6x ATR=2.44 |
| Stop hit — per-position SL triggered | 2023-09-22 10:40:00 | 508.49 | 512.10 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-09-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 10:15:00 | 509.00 | 510.68 | 0.00 | ORB-short ORB[510.15,515.92] vol=1.8x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 13:20:00 | 506.29 | 509.65 | 0.00 | T1 1.5R @ 506.29 |
| Stop hit — per-position SL triggered | 2023-09-28 14:40:00 | 509.00 | 509.10 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-10-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-04 10:55:00 | 503.10 | 505.98 | 0.00 | ORB-short ORB[504.05,510.03] vol=1.7x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 11:15:00 | 501.03 | 505.23 | 0.00 | T1 1.5R @ 501.03 |
| Target hit | 2023-10-04 14:05:00 | 496.45 | 496.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 39 — BUY (started 2023-10-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 09:35:00 | 515.30 | 512.02 | 0.00 | ORB-long ORB[507.50,514.50] vol=3.5x ATR=2.40 |
| Stop hit — per-position SL triggered | 2023-10-10 13:15:00 | 512.90 | 514.22 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-10-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:55:00 | 505.00 | 506.47 | 0.00 | ORB-short ORB[505.55,509.50] vol=2.7x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 11:10:00 | 502.73 | 506.23 | 0.00 | T1 1.5R @ 502.73 |
| Stop hit — per-position SL triggered | 2023-10-18 11:35:00 | 505.00 | 506.04 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-10-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-19 09:35:00 | 500.18 | 502.54 | 0.00 | ORB-short ORB[500.88,506.00] vol=2.0x ATR=1.37 |
| Stop hit — per-position SL triggered | 2023-10-19 09:45:00 | 501.55 | 502.45 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-10-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 09:35:00 | 495.70 | 499.11 | 0.00 | ORB-short ORB[497.30,504.00] vol=2.7x ATR=1.65 |
| Stop hit — per-position SL triggered | 2023-10-23 09:40:00 | 497.35 | 498.88 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-11-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 09:35:00 | 496.55 | 493.27 | 0.00 | ORB-long ORB[488.18,494.40] vol=3.7x ATR=2.08 |
| Stop hit — per-position SL triggered | 2023-11-01 09:50:00 | 494.47 | 493.76 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 11:15:00 | 499.68 | 495.91 | 0.00 | ORB-long ORB[491.33,496.90] vol=10.9x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 11:20:00 | 502.00 | 497.83 | 0.00 | T1 1.5R @ 502.00 |
| Stop hit — per-position SL triggered | 2023-11-02 11:30:00 | 499.68 | 498.34 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 11:15:00 | 509.03 | 505.77 | 0.00 | ORB-long ORB[503.03,508.25] vol=1.8x ATR=1.89 |
| Stop hit — per-position SL triggered | 2023-11-03 11:30:00 | 507.14 | 505.93 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-11-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 10:20:00 | 537.50 | 531.03 | 0.00 | ORB-long ORB[525.38,531.85] vol=7.5x ATR=2.65 |
| Stop hit — per-position SL triggered | 2023-11-13 10:25:00 | 534.85 | 532.08 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-11-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 10:15:00 | 552.75 | 548.32 | 0.00 | ORB-long ORB[543.53,549.15] vol=3.1x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 11:50:00 | 556.01 | 550.92 | 0.00 | T1 1.5R @ 556.01 |
| Stop hit — per-position SL triggered | 2023-11-16 15:10:00 | 552.75 | 553.90 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-11-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 10:40:00 | 558.50 | 554.71 | 0.00 | ORB-long ORB[551.28,557.25] vol=2.4x ATR=2.17 |
| Stop hit — per-position SL triggered | 2023-11-17 11:10:00 | 556.33 | 555.45 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-11-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 11:05:00 | 562.03 | 565.00 | 0.00 | ORB-short ORB[563.20,569.50] vol=2.3x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 11:20:00 | 558.98 | 564.47 | 0.00 | T1 1.5R @ 558.98 |
| Stop hit — per-position SL triggered | 2023-11-20 11:30:00 | 562.03 | 564.34 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-11-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-22 09:40:00 | 560.58 | 561.99 | 0.00 | ORB-short ORB[561.23,565.00] vol=2.4x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 10:00:00 | 556.84 | 561.37 | 0.00 | T1 1.5R @ 556.84 |
| Target hit | 2023-11-22 15:20:00 | 546.00 | 552.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2023-11-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 10:30:00 | 557.23 | 553.13 | 0.00 | ORB-long ORB[547.25,554.98] vol=1.9x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 11:05:00 | 560.60 | 554.29 | 0.00 | T1 1.5R @ 560.60 |
| Stop hit — per-position SL triggered | 2023-11-23 12:20:00 | 557.23 | 554.73 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 09:40:00 | 584.67 | 580.26 | 0.00 | ORB-long ORB[577.00,581.50] vol=1.6x ATR=2.47 |
| Stop hit — per-position SL triggered | 2023-11-30 09:55:00 | 582.20 | 582.06 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 09:30:00 | 615.50 | 619.81 | 0.00 | ORB-short ORB[618.20,625.50] vol=1.7x ATR=2.38 |
| Stop hit — per-position SL triggered | 2023-12-06 09:35:00 | 617.88 | 619.62 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-12-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 10:35:00 | 651.13 | 644.50 | 0.00 | ORB-long ORB[634.63,642.88] vol=3.6x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 10:40:00 | 655.73 | 645.74 | 0.00 | T1 1.5R @ 655.73 |
| Stop hit — per-position SL triggered | 2023-12-07 11:00:00 | 651.13 | 647.34 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-12-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 09:35:00 | 668.93 | 664.88 | 0.00 | ORB-long ORB[657.00,665.70] vol=4.3x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 09:45:00 | 674.84 | 667.66 | 0.00 | T1 1.5R @ 674.84 |
| Target hit | 2023-12-11 10:35:00 | 674.38 | 675.02 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — SELL (started 2023-12-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 10:20:00 | 670.05 | 673.95 | 0.00 | ORB-short ORB[671.00,680.00] vol=2.0x ATR=2.71 |
| Stop hit — per-position SL triggered | 2023-12-12 10:30:00 | 672.76 | 673.89 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 09:30:00 | 691.55 | 685.94 | 0.00 | ORB-long ORB[680.50,688.50] vol=1.8x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 09:40:00 | 697.38 | 692.94 | 0.00 | T1 1.5R @ 697.38 |
| Target hit | 2023-12-13 10:10:00 | 694.03 | 694.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 58 — SELL (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 10:15:00 | 688.48 | 692.20 | 0.00 | ORB-short ORB[693.30,700.45] vol=1.8x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 10:35:00 | 684.37 | 691.48 | 0.00 | T1 1.5R @ 684.37 |
| Stop hit — per-position SL triggered | 2023-12-14 10:40:00 | 688.48 | 691.27 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-12-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 09:55:00 | 698.20 | 693.73 | 0.00 | ORB-long ORB[687.53,694.95] vol=1.8x ATR=2.20 |
| Stop hit — per-position SL triggered | 2023-12-15 10:30:00 | 696.00 | 694.70 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-12-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 10:40:00 | 785.18 | 776.82 | 0.00 | ORB-long ORB[767.95,777.00] vol=4.0x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 10:55:00 | 791.68 | 779.67 | 0.00 | T1 1.5R @ 791.68 |
| Stop hit — per-position SL triggered | 2023-12-22 12:20:00 | 785.18 | 782.03 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 10:10:00 | 872.40 | 857.81 | 0.00 | ORB-long ORB[846.98,859.15] vol=3.0x ATR=6.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 10:20:00 | 881.92 | 863.64 | 0.00 | T1 1.5R @ 881.92 |
| Stop hit — per-position SL triggered | 2023-12-27 10:40:00 | 872.40 | 866.82 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-01-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 09:30:00 | 870.53 | 866.87 | 0.00 | ORB-long ORB[858.50,869.30] vol=2.5x ATR=3.82 |
| Stop hit — per-position SL triggered | 2024-01-01 09:40:00 | 866.71 | 867.04 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 848.80 | 858.40 | 0.00 | ORB-short ORB[855.63,868.40] vol=2.2x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-01-02 10:05:00 | 852.18 | 856.00 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-01-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 11:05:00 | 861.78 | 866.65 | 0.00 | ORB-short ORB[863.50,872.65] vol=2.9x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-01-19 11:15:00 | 864.40 | 866.57 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-01-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 10:20:00 | 868.05 | 860.22 | 0.00 | ORB-long ORB[852.08,862.50] vol=2.6x ATR=3.75 |
| Stop hit — per-position SL triggered | 2024-01-20 10:35:00 | 864.30 | 860.77 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-01-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-30 09:35:00 | 827.00 | 833.74 | 0.00 | ORB-short ORB[830.60,842.23] vol=1.7x ATR=3.53 |
| Stop hit — per-position SL triggered | 2024-01-30 09:50:00 | 830.53 | 832.35 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-02-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 09:35:00 | 863.48 | 858.60 | 0.00 | ORB-long ORB[853.50,861.80] vol=2.4x ATR=4.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 09:40:00 | 869.70 | 863.51 | 0.00 | T1 1.5R @ 869.70 |
| Target hit | 2024-02-02 10:35:00 | 889.35 | 889.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 68 — SELL (started 2024-02-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 09:45:00 | 863.00 | 869.80 | 0.00 | ORB-short ORB[866.78,879.30] vol=1.7x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 09:50:00 | 858.25 | 868.45 | 0.00 | T1 1.5R @ 858.25 |
| Stop hit — per-position SL triggered | 2024-02-21 10:15:00 | 863.00 | 864.76 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 11:15:00 | 903.00 | 911.52 | 0.00 | ORB-short ORB[910.00,921.48] vol=3.5x ATR=2.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-01 11:45:00 | 898.64 | 910.13 | 0.00 | T1 1.5R @ 898.64 |
| Stop hit — per-position SL triggered | 2024-03-01 12:25:00 | 903.00 | 908.91 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-03-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:45:00 | 831.25 | 837.49 | 0.00 | ORB-short ORB[833.75,841.00] vol=2.1x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 09:55:00 | 826.41 | 835.80 | 0.00 | T1 1.5R @ 826.41 |
| Target hit | 2024-03-19 15:20:00 | 811.18 | 820.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — SELL (started 2024-04-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:30:00 | 879.85 | 884.80 | 0.00 | ORB-short ORB[883.00,889.50] vol=2.0x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 09:50:00 | 874.35 | 882.51 | 0.00 | T1 1.5R @ 874.35 |
| Stop hit — per-position SL triggered | 2024-04-04 10:50:00 | 879.85 | 879.83 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-04-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 09:35:00 | 871.70 | 877.83 | 0.00 | ORB-short ORB[874.60,885.00] vol=2.0x ATR=3.41 |
| Stop hit — per-position SL triggered | 2024-04-05 10:30:00 | 875.11 | 876.39 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-04-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 11:05:00 | 871.73 | 874.05 | 0.00 | ORB-short ORB[872.50,878.30] vol=2.0x ATR=2.32 |
| Stop hit — per-position SL triggered | 2024-04-08 11:20:00 | 874.05 | 874.01 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 09:40:00 | 872.40 | 868.75 | 0.00 | ORB-long ORB[864.00,871.95] vol=1.7x ATR=3.27 |
| Stop hit — per-position SL triggered | 2024-04-10 10:05:00 | 869.13 | 869.86 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-04-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 09:30:00 | 888.98 | 884.48 | 0.00 | ORB-long ORB[876.50,887.40] vol=2.0x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 09:40:00 | 894.24 | 886.91 | 0.00 | T1 1.5R @ 894.24 |
| Target hit | 2024-04-12 09:40:00 | 886.00 | 886.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 76 — BUY (started 2024-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 09:45:00 | 933.80 | 928.89 | 0.00 | ORB-long ORB[918.90,928.48] vol=4.9x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 10:00:00 | 940.09 | 931.99 | 0.00 | T1 1.5R @ 940.09 |
| Stop hit — per-position SL triggered | 2024-04-23 10:05:00 | 933.80 | 932.18 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-05-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 09:35:00 | 984.10 | 988.43 | 0.00 | ORB-short ORB[986.03,994.50] vol=1.5x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-05-03 09:50:00 | 986.97 | 987.53 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 09:35:00 | 509.55 | 2023-05-15 09:40:00 | 511.59 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-05-15 09:35:00 | 509.55 | 2023-05-15 09:45:00 | 509.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-19 09:40:00 | 528.98 | 2023-05-19 10:05:00 | 525.51 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2023-05-19 09:40:00 | 528.98 | 2023-05-19 13:25:00 | 528.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-22 09:50:00 | 532.98 | 2023-05-22 09:55:00 | 530.72 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-05-25 09:55:00 | 535.58 | 2023-05-25 10:00:00 | 538.33 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-05-25 09:55:00 | 535.58 | 2023-05-25 11:00:00 | 535.58 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-30 09:45:00 | 526.45 | 2023-05-30 10:30:00 | 524.56 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-06-02 10:55:00 | 558.30 | 2023-06-02 11:15:00 | 556.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-06-05 09:55:00 | 568.00 | 2023-06-05 10:25:00 | 565.98 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-06-13 10:40:00 | 565.00 | 2023-06-13 11:05:00 | 566.18 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-16 09:30:00 | 575.00 | 2023-06-16 09:35:00 | 572.53 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-06-19 09:55:00 | 617.80 | 2023-06-19 10:00:00 | 613.48 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest1 | 2023-06-20 10:00:00 | 608.20 | 2023-06-20 10:05:00 | 613.18 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2023-06-20 10:00:00 | 608.20 | 2023-06-20 10:10:00 | 608.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-03 10:00:00 | 559.67 | 2023-07-03 10:15:00 | 556.13 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2023-07-03 10:00:00 | 559.67 | 2023-07-03 10:30:00 | 559.67 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-05 09:30:00 | 557.80 | 2023-07-05 09:35:00 | 560.09 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-07-05 09:30:00 | 557.80 | 2023-07-05 09:45:00 | 557.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-06 11:15:00 | 567.50 | 2023-07-06 11:20:00 | 566.04 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-07-25 09:50:00 | 600.45 | 2023-07-25 09:55:00 | 604.52 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2023-07-25 09:50:00 | 600.45 | 2023-07-25 10:00:00 | 600.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-28 09:35:00 | 598.03 | 2023-07-28 09:40:00 | 599.62 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-07-31 09:40:00 | 613.13 | 2023-07-31 09:45:00 | 611.05 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-08-02 10:35:00 | 596.90 | 2023-08-02 13:35:00 | 593.85 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-08-02 10:35:00 | 596.90 | 2023-08-02 15:15:00 | 596.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-04 11:15:00 | 589.10 | 2023-08-04 11:25:00 | 590.93 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-08-09 10:50:00 | 568.40 | 2023-08-09 11:00:00 | 566.69 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-08-10 09:35:00 | 567.42 | 2023-08-10 10:00:00 | 569.70 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-08-11 10:55:00 | 569.70 | 2023-08-11 11:00:00 | 568.07 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-08-18 10:00:00 | 560.67 | 2023-08-18 10:10:00 | 557.53 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2023-08-18 10:00:00 | 560.67 | 2023-08-18 15:20:00 | 553.50 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2023-08-22 10:30:00 | 565.30 | 2023-08-22 10:40:00 | 563.91 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-08-23 09:45:00 | 584.48 | 2023-08-23 10:35:00 | 582.23 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-08-24 10:55:00 | 579.13 | 2023-08-24 11:40:00 | 576.07 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-08-24 10:55:00 | 579.13 | 2023-08-24 11:55:00 | 579.13 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-25 10:25:00 | 569.08 | 2023-08-25 10:30:00 | 566.01 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2023-08-25 10:25:00 | 569.08 | 2023-08-25 10:40:00 | 569.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-30 09:30:00 | 566.95 | 2023-08-30 09:40:00 | 569.44 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-08-30 09:30:00 | 566.95 | 2023-08-30 13:00:00 | 572.28 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2023-09-01 10:15:00 | 575.53 | 2023-09-01 10:20:00 | 573.45 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-09-04 10:10:00 | 589.00 | 2023-09-04 10:20:00 | 592.86 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2023-09-04 10:10:00 | 589.00 | 2023-09-04 11:10:00 | 595.25 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2023-09-05 11:00:00 | 594.00 | 2023-09-05 11:30:00 | 591.72 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-09-06 11:10:00 | 583.95 | 2023-09-06 12:45:00 | 581.85 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-09-06 11:10:00 | 583.95 | 2023-09-06 15:20:00 | 574.98 | TARGET_HIT | 0.50 | 1.54% |
| BUY | retest1 | 2023-09-08 09:50:00 | 606.00 | 2023-09-08 10:10:00 | 610.35 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2023-09-08 09:50:00 | 606.00 | 2023-09-08 10:30:00 | 606.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-11 09:30:00 | 592.78 | 2023-09-11 09:35:00 | 595.20 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-09-15 09:35:00 | 534.65 | 2023-09-15 09:40:00 | 536.39 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-09-22 10:20:00 | 506.05 | 2023-09-22 10:40:00 | 508.49 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2023-09-28 10:15:00 | 509.00 | 2023-09-28 13:20:00 | 506.29 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-09-28 10:15:00 | 509.00 | 2023-09-28 14:40:00 | 509.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-04 10:55:00 | 503.10 | 2023-10-04 11:15:00 | 501.03 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-10-04 10:55:00 | 503.10 | 2023-10-04 14:05:00 | 496.45 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2023-10-10 09:35:00 | 515.30 | 2023-10-10 13:15:00 | 512.90 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2023-10-18 10:55:00 | 505.00 | 2023-10-18 11:10:00 | 502.73 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-10-18 10:55:00 | 505.00 | 2023-10-18 11:35:00 | 505.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-19 09:35:00 | 500.18 | 2023-10-19 09:45:00 | 501.55 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-10-23 09:35:00 | 495.70 | 2023-10-23 09:40:00 | 497.35 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-11-01 09:35:00 | 496.55 | 2023-11-01 09:50:00 | 494.47 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-11-02 11:15:00 | 499.68 | 2023-11-02 11:20:00 | 502.00 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-11-02 11:15:00 | 499.68 | 2023-11-02 11:30:00 | 499.68 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-03 11:15:00 | 509.03 | 2023-11-03 11:30:00 | 507.14 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-11-13 10:20:00 | 537.50 | 2023-11-13 10:25:00 | 534.85 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2023-11-16 10:15:00 | 552.75 | 2023-11-16 11:50:00 | 556.01 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2023-11-16 10:15:00 | 552.75 | 2023-11-16 15:10:00 | 552.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-17 10:40:00 | 558.50 | 2023-11-17 11:10:00 | 556.33 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-11-20 11:05:00 | 562.03 | 2023-11-20 11:20:00 | 558.98 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2023-11-20 11:05:00 | 562.03 | 2023-11-20 11:30:00 | 562.03 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-22 09:40:00 | 560.58 | 2023-11-22 10:00:00 | 556.84 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2023-11-22 09:40:00 | 560.58 | 2023-11-22 15:20:00 | 546.00 | TARGET_HIT | 0.50 | 2.60% |
| BUY | retest1 | 2023-11-23 10:30:00 | 557.23 | 2023-11-23 11:05:00 | 560.60 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2023-11-23 10:30:00 | 557.23 | 2023-11-23 12:20:00 | 557.23 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 09:40:00 | 584.67 | 2023-11-30 09:55:00 | 582.20 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-12-06 09:30:00 | 615.50 | 2023-12-06 09:35:00 | 617.88 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-12-07 10:35:00 | 651.13 | 2023-12-07 10:40:00 | 655.73 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2023-12-07 10:35:00 | 651.13 | 2023-12-07 11:00:00 | 651.13 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-11 09:35:00 | 668.93 | 2023-12-11 09:45:00 | 674.84 | PARTIAL | 0.50 | 0.88% |
| BUY | retest1 | 2023-12-11 09:35:00 | 668.93 | 2023-12-11 10:35:00 | 674.38 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2023-12-12 10:20:00 | 670.05 | 2023-12-12 10:30:00 | 672.76 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-12-13 09:30:00 | 691.55 | 2023-12-13 09:40:00 | 697.38 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2023-12-13 09:30:00 | 691.55 | 2023-12-13 10:10:00 | 694.03 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2023-12-14 10:15:00 | 688.48 | 2023-12-14 10:35:00 | 684.37 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2023-12-14 10:15:00 | 688.48 | 2023-12-14 10:40:00 | 688.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-15 09:55:00 | 698.20 | 2023-12-15 10:30:00 | 696.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-12-22 10:40:00 | 785.18 | 2023-12-22 10:55:00 | 791.68 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2023-12-22 10:40:00 | 785.18 | 2023-12-22 12:20:00 | 785.18 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-27 10:10:00 | 872.40 | 2023-12-27 10:20:00 | 881.92 | PARTIAL | 0.50 | 1.09% |
| BUY | retest1 | 2023-12-27 10:10:00 | 872.40 | 2023-12-27 10:40:00 | 872.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-01 09:30:00 | 870.53 | 2024-01-01 09:40:00 | 866.71 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-01-02 09:55:00 | 848.80 | 2024-01-02 10:05:00 | 852.18 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-01-19 11:05:00 | 861.78 | 2024-01-19 11:15:00 | 864.40 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-01-20 10:20:00 | 868.05 | 2024-01-20 10:35:00 | 864.30 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-01-30 09:35:00 | 827.00 | 2024-01-30 09:50:00 | 830.53 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-02-02 09:35:00 | 863.48 | 2024-02-02 09:40:00 | 869.70 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-02-02 09:35:00 | 863.48 | 2024-02-02 10:35:00 | 889.35 | TARGET_HIT | 0.50 | 3.00% |
| SELL | retest1 | 2024-02-21 09:45:00 | 863.00 | 2024-02-21 09:50:00 | 858.25 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-02-21 09:45:00 | 863.00 | 2024-02-21 10:15:00 | 863.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-01 11:15:00 | 903.00 | 2024-03-01 11:45:00 | 898.64 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-03-01 11:15:00 | 903.00 | 2024-03-01 12:25:00 | 903.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-19 09:45:00 | 831.25 | 2024-03-19 09:55:00 | 826.41 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-03-19 09:45:00 | 831.25 | 2024-03-19 15:20:00 | 811.18 | TARGET_HIT | 0.50 | 2.41% |
| SELL | retest1 | 2024-04-04 09:30:00 | 879.85 | 2024-04-04 09:50:00 | 874.35 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-04-04 09:30:00 | 879.85 | 2024-04-04 10:50:00 | 879.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-05 09:35:00 | 871.70 | 2024-04-05 10:30:00 | 875.11 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-04-08 11:05:00 | 871.73 | 2024-04-08 11:20:00 | 874.05 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-04-10 09:40:00 | 872.40 | 2024-04-10 10:05:00 | 869.13 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-04-12 09:30:00 | 888.98 | 2024-04-12 09:40:00 | 894.24 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-04-12 09:30:00 | 888.98 | 2024-04-12 09:40:00 | 886.00 | TARGET_HIT | 0.50 | -0.34% |
| BUY | retest1 | 2024-04-23 09:45:00 | 933.80 | 2024-04-23 10:00:00 | 940.09 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-04-23 09:45:00 | 933.80 | 2024-04-23 10:05:00 | 933.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-03 09:35:00 | 984.10 | 2024-05-03 09:50:00 | 986.97 | STOP_HIT | 1.00 | -0.29% |
