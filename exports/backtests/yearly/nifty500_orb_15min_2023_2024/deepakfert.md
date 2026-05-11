# Deepak Fertilisers & Petrochemicals Corp. Ltd. (DEEPAKFERT)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-02-05 15:25:00 (13738 bars)
- **Last close:** 542.40
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
| ENTRY1 | 88 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 14 |
| STOP_HIT | 74 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 114 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 74
- **Target hits / Stop hits / Partials:** 14 / 74 / 26
- **Avg / median % per leg:** 0.15% / -0.23%
- **Sum % (uncompounded):** 17.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 22 | 34.9% | 8 | 41 | 14 | 0.12% | 7.4% |
| BUY @ 2nd Alert (retest1) | 63 | 22 | 34.9% | 8 | 41 | 14 | 0.12% | 7.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 51 | 18 | 35.3% | 6 | 33 | 12 | 0.19% | 9.8% |
| SELL @ 2nd Alert (retest1) | 51 | 18 | 35.3% | 6 | 33 | 12 | 0.19% | 9.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 114 | 40 | 35.1% | 14 | 74 | 26 | 0.15% | 17.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-15 09:30:00 | 588.90 | 592.51 | 0.00 | ORB-short ORB[592.70,596.85] vol=3.1x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-15 10:50:00 | 586.72 | 589.40 | 0.00 | T1 1.5R @ 586.72 |
| Stop hit — per-position SL triggered | 2023-05-15 12:00:00 | 588.90 | 589.11 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 09:40:00 | 595.85 | 593.53 | 0.00 | ORB-long ORB[589.10,595.50] vol=1.6x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-05-16 10:05:00 | 594.63 | 594.19 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 10:15:00 | 538.80 | 541.23 | 0.00 | ORB-short ORB[540.50,547.35] vol=1.6x ATR=1.37 |
| Stop hit — per-position SL triggered | 2023-05-30 10:40:00 | 540.17 | 541.05 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-06-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 10:10:00 | 555.45 | 552.05 | 0.00 | ORB-long ORB[548.65,554.80] vol=3.5x ATR=1.59 |
| Stop hit — per-position SL triggered | 2023-06-06 10:30:00 | 553.86 | 553.01 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 09:30:00 | 557.40 | 558.86 | 0.00 | ORB-short ORB[558.15,561.95] vol=1.8x ATR=1.89 |
| Stop hit — per-position SL triggered | 2023-06-13 09:40:00 | 559.29 | 558.82 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 09:30:00 | 566.75 | 564.45 | 0.00 | ORB-long ORB[561.00,565.25] vol=3.8x ATR=1.32 |
| Stop hit — per-position SL triggered | 2023-06-14 09:35:00 | 565.43 | 564.69 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 10:40:00 | 575.20 | 572.53 | 0.00 | ORB-long ORB[567.45,574.60] vol=2.9x ATR=1.89 |
| Stop hit — per-position SL triggered | 2023-06-15 10:50:00 | 573.31 | 572.64 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 11:10:00 | 566.70 | 568.50 | 0.00 | ORB-short ORB[567.40,572.80] vol=2.5x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-06-19 11:20:00 | 567.79 | 568.48 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 10:20:00 | 568.45 | 565.11 | 0.00 | ORB-long ORB[562.25,567.85] vol=4.6x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 10:25:00 | 570.80 | 568.30 | 0.00 | T1 1.5R @ 570.80 |
| Target hit | 2023-06-20 14:05:00 | 570.90 | 570.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2023-06-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:10:00 | 568.40 | 570.81 | 0.00 | ORB-short ORB[570.15,573.00] vol=1.5x ATR=1.15 |
| Stop hit — per-position SL triggered | 2023-06-21 11:45:00 | 569.55 | 570.50 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 10:10:00 | 589.25 | 583.75 | 0.00 | ORB-long ORB[580.10,584.90] vol=4.9x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 10:15:00 | 593.20 | 585.95 | 0.00 | T1 1.5R @ 593.20 |
| Stop hit — per-position SL triggered | 2023-06-22 10:30:00 | 589.25 | 587.26 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 09:35:00 | 573.50 | 572.09 | 0.00 | ORB-long ORB[570.10,573.40] vol=2.1x ATR=1.29 |
| Stop hit — per-position SL triggered | 2023-06-27 09:40:00 | 572.21 | 572.14 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 09:40:00 | 565.30 | 567.11 | 0.00 | ORB-short ORB[565.50,571.70] vol=2.1x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-28 10:40:00 | 563.16 | 565.98 | 0.00 | T1 1.5R @ 563.16 |
| Target hit | 2023-06-28 15:20:00 | 560.40 | 563.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2023-06-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-30 10:55:00 | 562.70 | 564.45 | 0.00 | ORB-short ORB[564.00,567.55] vol=2.2x ATR=1.51 |
| Stop hit — per-position SL triggered | 2023-06-30 11:15:00 | 564.21 | 564.27 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 09:40:00 | 610.50 | 605.69 | 0.00 | ORB-long ORB[601.25,608.00] vol=1.7x ATR=2.38 |
| Stop hit — per-position SL triggered | 2023-07-07 09:50:00 | 608.12 | 606.83 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 10:00:00 | 614.70 | 612.72 | 0.00 | ORB-long ORB[607.20,614.50] vol=1.8x ATR=2.31 |
| Stop hit — per-position SL triggered | 2023-07-11 12:30:00 | 612.39 | 614.23 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 09:45:00 | 602.40 | 605.59 | 0.00 | ORB-short ORB[602.45,611.00] vol=1.9x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 11:15:00 | 599.04 | 602.94 | 0.00 | T1 1.5R @ 599.04 |
| Target hit | 2023-07-12 15:20:00 | 596.60 | 599.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2023-07-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 10:25:00 | 606.60 | 603.08 | 0.00 | ORB-long ORB[596.80,605.75] vol=2.7x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 10:30:00 | 609.80 | 603.92 | 0.00 | T1 1.5R @ 609.80 |
| Target hit | 2023-07-13 12:30:00 | 607.55 | 607.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2023-07-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:45:00 | 606.80 | 604.02 | 0.00 | ORB-long ORB[600.10,604.55] vol=1.7x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 09:50:00 | 609.97 | 605.22 | 0.00 | T1 1.5R @ 609.97 |
| Target hit | 2023-07-17 11:40:00 | 608.00 | 608.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — BUY (started 2023-07-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:55:00 | 608.55 | 605.37 | 0.00 | ORB-long ORB[602.30,606.10] vol=1.7x ATR=1.73 |
| Stop hit — per-position SL triggered | 2023-07-19 10:05:00 | 606.82 | 605.82 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-20 09:30:00 | 601.70 | 603.35 | 0.00 | ORB-short ORB[602.40,607.15] vol=3.8x ATR=1.48 |
| Stop hit — per-position SL triggered | 2023-07-20 10:00:00 | 603.18 | 602.92 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:05:00 | 599.30 | 596.18 | 0.00 | ORB-long ORB[591.15,597.00] vol=2.6x ATR=2.04 |
| Stop hit — per-position SL triggered | 2023-07-26 10:15:00 | 597.26 | 596.58 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-07-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 09:30:00 | 537.20 | 540.49 | 0.00 | ORB-short ORB[538.50,545.00] vol=1.5x ATR=1.33 |
| Stop hit — per-position SL triggered | 2023-07-28 09:55:00 | 538.53 | 539.37 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-07-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-31 10:45:00 | 545.85 | 548.54 | 0.00 | ORB-short ORB[545.90,550.95] vol=2.3x ATR=1.75 |
| Stop hit — per-position SL triggered | 2023-07-31 11:20:00 | 547.60 | 548.18 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 09:55:00 | 554.65 | 550.47 | 0.00 | ORB-long ORB[544.65,551.45] vol=2.5x ATR=1.70 |
| Stop hit — per-position SL triggered | 2023-08-01 10:05:00 | 552.95 | 551.54 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 10:30:00 | 553.50 | 557.11 | 0.00 | ORB-short ORB[555.10,559.95] vol=1.6x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 11:00:00 | 551.25 | 556.25 | 0.00 | T1 1.5R @ 551.25 |
| Stop hit — per-position SL triggered | 2023-08-02 12:25:00 | 553.50 | 555.45 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-08-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 10:05:00 | 550.00 | 552.30 | 0.00 | ORB-short ORB[550.55,557.30] vol=2.1x ATR=1.45 |
| Stop hit — per-position SL triggered | 2023-08-04 10:30:00 | 551.45 | 552.03 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 11:10:00 | 549.80 | 551.87 | 0.00 | ORB-short ORB[550.45,554.20] vol=3.5x ATR=1.16 |
| Stop hit — per-position SL triggered | 2023-08-08 11:20:00 | 550.96 | 551.75 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 09:45:00 | 547.95 | 550.04 | 0.00 | ORB-short ORB[549.05,553.80] vol=2.5x ATR=1.42 |
| Stop hit — per-position SL triggered | 2023-08-11 11:05:00 | 549.37 | 548.66 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-08-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 10:35:00 | 551.20 | 548.24 | 0.00 | ORB-long ORB[543.95,550.95] vol=3.1x ATR=1.49 |
| Stop hit — per-position SL triggered | 2023-08-21 11:10:00 | 549.71 | 548.72 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-08-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 09:50:00 | 560.95 | 558.98 | 0.00 | ORB-long ORB[553.90,560.00] vol=5.1x ATR=1.58 |
| Stop hit — per-position SL triggered | 2023-08-22 09:55:00 | 559.37 | 559.02 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-08-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 10:10:00 | 562.00 | 559.09 | 0.00 | ORB-long ORB[555.00,559.00] vol=4.6x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-23 10:15:00 | 564.21 | 561.59 | 0.00 | T1 1.5R @ 564.21 |
| Target hit | 2023-08-23 15:20:00 | 573.50 | 571.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2023-08-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:20:00 | 559.50 | 562.23 | 0.00 | ORB-short ORB[561.25,566.55] vol=2.5x ATR=2.16 |
| Stop hit — per-position SL triggered | 2023-08-25 11:20:00 | 561.66 | 561.28 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 10:10:00 | 563.30 | 558.52 | 0.00 | ORB-long ORB[555.00,559.00] vol=2.3x ATR=1.80 |
| Stop hit — per-position SL triggered | 2023-08-29 10:15:00 | 561.50 | 560.01 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 09:40:00 | 610.90 | 606.40 | 0.00 | ORB-long ORB[601.40,609.75] vol=2.8x ATR=2.29 |
| Stop hit — per-position SL triggered | 2023-09-01 09:45:00 | 608.61 | 607.27 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 11:05:00 | 619.25 | 615.07 | 0.00 | ORB-long ORB[610.00,618.40] vol=3.3x ATR=1.96 |
| Stop hit — per-position SL triggered | 2023-09-04 11:10:00 | 617.29 | 615.20 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-06 09:45:00 | 615.00 | 616.67 | 0.00 | ORB-short ORB[615.05,622.90] vol=2.5x ATR=2.74 |
| Stop hit — per-position SL triggered | 2023-09-06 09:50:00 | 617.74 | 616.73 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-09-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 10:55:00 | 648.35 | 641.08 | 0.00 | ORB-long ORB[632.95,639.50] vol=10.7x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 11:00:00 | 653.41 | 642.79 | 0.00 | T1 1.5R @ 653.41 |
| Target hit | 2023-09-07 15:20:00 | 658.70 | 652.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2023-09-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-11 09:50:00 | 659.70 | 667.57 | 0.00 | ORB-short ORB[666.50,675.00] vol=1.9x ATR=2.55 |
| Stop hit — per-position SL triggered | 2023-09-11 10:05:00 | 662.25 | 665.40 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:50:00 | 647.15 | 641.62 | 0.00 | ORB-long ORB[636.30,643.30] vol=3.4x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 09:55:00 | 651.19 | 643.22 | 0.00 | T1 1.5R @ 651.19 |
| Stop hit — per-position SL triggered | 2023-09-14 10:00:00 | 647.15 | 643.83 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 10:10:00 | 626.85 | 621.84 | 0.00 | ORB-long ORB[617.05,623.25] vol=2.0x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-26 10:15:00 | 629.60 | 634.39 | 0.00 | T1 1.5R @ 629.60 |
| Target hit | 2023-09-26 15:20:00 | 652.00 | 649.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2023-09-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 09:40:00 | 648.10 | 651.01 | 0.00 | ORB-short ORB[649.50,656.00] vol=1.8x ATR=1.90 |
| Stop hit — per-position SL triggered | 2023-09-28 09:45:00 | 650.00 | 650.90 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-10-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 09:30:00 | 649.10 | 646.05 | 0.00 | ORB-long ORB[641.25,648.90] vol=4.4x ATR=2.45 |
| Stop hit — per-position SL triggered | 2023-10-04 09:35:00 | 646.65 | 646.87 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-10-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 10:25:00 | 642.00 | 645.85 | 0.00 | ORB-short ORB[643.00,648.95] vol=3.6x ATR=1.82 |
| Stop hit — per-position SL triggered | 2023-10-05 10:35:00 | 643.82 | 645.72 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-10-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 10:05:00 | 664.90 | 656.82 | 0.00 | ORB-long ORB[649.10,655.75] vol=8.6x ATR=3.33 |
| Stop hit — per-position SL triggered | 2023-10-16 10:10:00 | 661.57 | 658.84 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-10-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 10:25:00 | 689.40 | 684.09 | 0.00 | ORB-long ORB[680.10,688.65] vol=1.6x ATR=2.90 |
| Stop hit — per-position SL triggered | 2023-10-18 10:40:00 | 686.50 | 684.46 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 10:15:00 | 675.50 | 678.65 | 0.00 | ORB-short ORB[677.00,686.00] vol=1.6x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 11:30:00 | 671.51 | 677.46 | 0.00 | T1 1.5R @ 671.51 |
| Target hit | 2023-10-20 15:20:00 | 662.55 | 669.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2023-10-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 09:35:00 | 646.05 | 642.07 | 0.00 | ORB-long ORB[636.00,643.50] vol=3.0x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 09:45:00 | 649.80 | 644.92 | 0.00 | T1 1.5R @ 649.80 |
| Stop hit — per-position SL triggered | 2023-10-31 09:50:00 | 646.05 | 645.07 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-11-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 10:35:00 | 630.10 | 635.50 | 0.00 | ORB-short ORB[637.15,643.90] vol=2.0x ATR=1.85 |
| Stop hit — per-position SL triggered | 2023-11-01 10:45:00 | 631.95 | 634.62 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-11-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 09:35:00 | 613.90 | 609.14 | 0.00 | ORB-long ORB[605.00,612.45] vol=3.9x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 09:40:00 | 618.09 | 610.85 | 0.00 | T1 1.5R @ 618.09 |
| Stop hit — per-position SL triggered | 2023-11-06 09:50:00 | 613.90 | 611.23 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 09:30:00 | 620.05 | 616.68 | 0.00 | ORB-long ORB[610.00,618.90] vol=2.7x ATR=2.21 |
| Stop hit — per-position SL triggered | 2023-11-07 09:40:00 | 617.84 | 616.92 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-11-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-10 10:25:00 | 621.10 | 623.35 | 0.00 | ORB-short ORB[624.00,628.30] vol=1.9x ATR=1.48 |
| Stop hit — per-position SL triggered | 2023-11-10 10:35:00 | 622.58 | 623.20 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-11-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:50:00 | 612.50 | 615.44 | 0.00 | ORB-short ORB[614.25,618.65] vol=2.2x ATR=1.69 |
| Stop hit — per-position SL triggered | 2023-11-13 11:00:00 | 614.19 | 615.36 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 09:35:00 | 618.90 | 615.57 | 0.00 | ORB-long ORB[612.50,617.85] vol=2.6x ATR=1.81 |
| Stop hit — per-position SL triggered | 2023-11-15 09:40:00 | 617.09 | 615.95 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-11-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:00:00 | 621.90 | 617.57 | 0.00 | ORB-long ORB[613.30,619.50] vol=5.0x ATR=1.82 |
| Stop hit — per-position SL triggered | 2023-11-16 11:10:00 | 620.08 | 618.08 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-11-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-17 09:35:00 | 614.40 | 616.36 | 0.00 | ORB-short ORB[615.00,619.65] vol=2.3x ATR=1.81 |
| Stop hit — per-position SL triggered | 2023-11-17 09:50:00 | 616.21 | 616.05 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-11-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 09:40:00 | 603.50 | 606.12 | 0.00 | ORB-short ORB[603.90,612.65] vol=1.6x ATR=1.82 |
| Stop hit — per-position SL triggered | 2023-11-20 10:35:00 | 605.32 | 604.87 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:45:00 | 609.50 | 605.54 | 0.00 | ORB-long ORB[599.25,607.85] vol=2.0x ATR=1.84 |
| Stop hit — per-position SL triggered | 2023-11-22 10:30:00 | 607.66 | 607.56 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-11-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 09:30:00 | 610.45 | 607.60 | 0.00 | ORB-long ORB[599.90,608.95] vol=3.0x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 09:40:00 | 613.74 | 609.92 | 0.00 | T1 1.5R @ 613.74 |
| Stop hit — per-position SL triggered | 2023-11-23 10:25:00 | 610.45 | 611.18 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2023-11-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:50:00 | 603.80 | 607.72 | 0.00 | ORB-short ORB[606.20,612.25] vol=2.1x ATR=1.99 |
| Stop hit — per-position SL triggered | 2023-11-24 10:05:00 | 605.79 | 607.26 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 09:50:00 | 604.80 | 602.56 | 0.00 | ORB-long ORB[600.10,604.40] vol=1.6x ATR=1.64 |
| Stop hit — per-position SL triggered | 2023-11-28 09:55:00 | 603.16 | 602.64 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-11-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 10:10:00 | 607.95 | 606.15 | 0.00 | ORB-long ORB[602.65,607.90] vol=2.1x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 10:25:00 | 610.16 | 606.91 | 0.00 | T1 1.5R @ 610.16 |
| Target hit | 2023-11-29 15:20:00 | 627.30 | 624.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2023-11-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:50:00 | 626.05 | 628.92 | 0.00 | ORB-short ORB[628.00,635.00] vol=1.5x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 09:55:00 | 622.25 | 628.26 | 0.00 | T1 1.5R @ 622.25 |
| Stop hit — per-position SL triggered | 2023-11-30 10:05:00 | 626.05 | 628.07 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2023-12-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 09:35:00 | 641.45 | 637.72 | 0.00 | ORB-long ORB[631.45,640.40] vol=3.3x ATR=2.97 |
| Stop hit — per-position SL triggered | 2023-12-01 09:40:00 | 638.48 | 637.81 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 10:40:00 | 637.30 | 633.28 | 0.00 | ORB-long ORB[628.20,636.00] vol=2.6x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 10:50:00 | 641.01 | 635.17 | 0.00 | T1 1.5R @ 641.01 |
| Stop hit — per-position SL triggered | 2023-12-04 11:25:00 | 637.30 | 635.96 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-12-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 10:00:00 | 651.20 | 648.76 | 0.00 | ORB-long ORB[642.00,650.00] vol=3.6x ATR=3.06 |
| Stop hit — per-position SL triggered | 2023-12-05 10:10:00 | 648.14 | 648.78 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2023-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 09:35:00 | 647.75 | 644.04 | 0.00 | ORB-long ORB[638.25,646.50] vol=3.4x ATR=2.53 |
| Stop hit — per-position SL triggered | 2023-12-06 09:45:00 | 645.22 | 644.41 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2023-12-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 10:20:00 | 666.00 | 663.32 | 0.00 | ORB-long ORB[656.25,665.50] vol=2.0x ATR=2.91 |
| Stop hit — per-position SL triggered | 2023-12-08 10:55:00 | 663.09 | 664.35 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2023-12-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 10:05:00 | 659.05 | 653.33 | 0.00 | ORB-long ORB[646.15,654.85] vol=1.6x ATR=2.97 |
| Stop hit — per-position SL triggered | 2023-12-11 10:15:00 | 656.08 | 653.51 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2023-12-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 09:40:00 | 664.45 | 659.67 | 0.00 | ORB-long ORB[653.55,661.50] vol=2.1x ATR=2.65 |
| Stop hit — per-position SL triggered | 2023-12-18 09:45:00 | 661.80 | 659.83 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2023-12-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 10:10:00 | 704.50 | 708.80 | 0.00 | ORB-short ORB[705.10,712.85] vol=1.6x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 11:30:00 | 700.42 | 707.53 | 0.00 | T1 1.5R @ 700.42 |
| Target hit | 2023-12-20 15:20:00 | 657.90 | 683.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — BUY (started 2023-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:30:00 | 686.10 | 683.04 | 0.00 | ORB-long ORB[677.00,684.85] vol=2.8x ATR=2.70 |
| Stop hit — per-position SL triggered | 2023-12-27 09:55:00 | 683.40 | 684.32 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2023-12-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 09:30:00 | 675.70 | 677.64 | 0.00 | ORB-short ORB[678.00,682.00] vol=6.6x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 09:35:00 | 671.12 | 677.07 | 0.00 | T1 1.5R @ 671.12 |
| Target hit | 2023-12-28 11:30:00 | 674.45 | 673.04 | 0.00 | Trail-exit close>VWAP |

### Cycle 74 — BUY (started 2023-12-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 10:05:00 | 678.80 | 678.16 | 0.00 | ORB-long ORB[673.30,678.45] vol=1.7x ATR=3.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 10:15:00 | 683.71 | 678.74 | 0.00 | T1 1.5R @ 683.71 |
| Target hit | 2023-12-29 11:35:00 | 681.20 | 682.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 75 — BUY (started 2024-01-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 09:35:00 | 693.90 | 684.99 | 0.00 | ORB-long ORB[678.05,685.95] vol=5.5x ATR=3.42 |
| Stop hit — per-position SL triggered | 2024-01-01 09:50:00 | 690.48 | 688.76 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-01-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:40:00 | 686.35 | 689.95 | 0.00 | ORB-short ORB[688.10,695.70] vol=2.9x ATR=2.61 |
| Stop hit — per-position SL triggered | 2024-01-03 09:45:00 | 688.96 | 689.80 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-01-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 10:45:00 | 688.35 | 692.87 | 0.00 | ORB-short ORB[694.50,698.35] vol=3.4x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 10:55:00 | 685.30 | 692.07 | 0.00 | T1 1.5R @ 685.30 |
| Stop hit — per-position SL triggered | 2024-01-05 11:00:00 | 688.35 | 691.90 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-01-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:40:00 | 669.50 | 675.12 | 0.00 | ORB-short ORB[673.70,679.35] vol=1.8x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-01-09 10:50:00 | 672.12 | 672.41 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-01-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-10 10:45:00 | 649.55 | 655.14 | 0.00 | ORB-short ORB[654.55,662.95] vol=1.7x ATR=2.28 |
| Stop hit — per-position SL triggered | 2024-01-10 11:10:00 | 651.83 | 654.32 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-01-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:50:00 | 652.00 | 654.81 | 0.00 | ORB-short ORB[653.15,658.75] vol=3.0x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 10:00:00 | 649.18 | 653.95 | 0.00 | T1 1.5R @ 649.18 |
| Stop hit — per-position SL triggered | 2024-01-15 10:05:00 | 652.00 | 653.78 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-01-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 09:40:00 | 654.50 | 650.94 | 0.00 | ORB-long ORB[647.60,653.35] vol=1.5x ATR=1.93 |
| Stop hit — per-position SL triggered | 2024-01-16 10:00:00 | 652.57 | 652.27 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-01-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:45:00 | 638.50 | 650.77 | 0.00 | ORB-short ORB[650.90,656.00] vol=2.7x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:50:00 | 633.12 | 646.51 | 0.00 | T1 1.5R @ 633.12 |
| Stop hit — per-position SL triggered | 2024-01-18 10:00:00 | 638.50 | 644.19 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-01-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 09:35:00 | 660.65 | 655.42 | 0.00 | ORB-long ORB[650.00,657.95] vol=5.1x ATR=3.76 |
| Stop hit — per-position SL triggered | 2024-01-19 10:00:00 | 656.89 | 657.16 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 09:40:00 | 653.60 | 650.68 | 0.00 | ORB-long ORB[646.00,652.50] vol=2.2x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-01-20 09:45:00 | 651.51 | 650.83 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-01-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 10:00:00 | 633.65 | 636.71 | 0.00 | ORB-short ORB[635.05,639.85] vol=1.8x ATR=3.06 |
| Stop hit — per-position SL triggered | 2024-01-25 12:05:00 | 636.71 | 634.33 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-01-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-29 09:40:00 | 634.35 | 638.78 | 0.00 | ORB-short ORB[638.05,643.30] vol=2.7x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-29 11:25:00 | 630.05 | 635.84 | 0.00 | T1 1.5R @ 630.05 |
| Target hit | 2024-01-29 15:20:00 | 628.40 | 632.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — BUY (started 2024-01-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 09:35:00 | 624.65 | 617.51 | 0.00 | ORB-long ORB[612.75,617.95] vol=1.7x ATR=2.35 |
| Stop hit — per-position SL triggered | 2024-01-31 10:00:00 | 622.30 | 620.07 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-02-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 10:50:00 | 615.50 | 620.34 | 0.00 | ORB-short ORB[617.20,623.75] vol=2.2x ATR=2.18 |
| Stop hit — per-position SL triggered | 2024-02-01 10:55:00 | 617.68 | 619.95 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-15 09:30:00 | 588.90 | 2023-05-15 10:50:00 | 586.72 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-05-15 09:30:00 | 588.90 | 2023-05-15 12:00:00 | 588.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-16 09:40:00 | 595.85 | 2023-05-16 10:05:00 | 594.63 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-05-30 10:15:00 | 538.80 | 2023-05-30 10:40:00 | 540.17 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-06-06 10:10:00 | 555.45 | 2023-06-06 10:30:00 | 553.86 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-06-13 09:30:00 | 557.40 | 2023-06-13 09:40:00 | 559.29 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-06-14 09:30:00 | 566.75 | 2023-06-14 09:35:00 | 565.43 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-06-15 10:40:00 | 575.20 | 2023-06-15 10:50:00 | 573.31 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-06-19 11:10:00 | 566.70 | 2023-06-19 11:20:00 | 567.79 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-06-20 10:20:00 | 568.45 | 2023-06-20 10:25:00 | 570.80 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-06-20 10:20:00 | 568.45 | 2023-06-20 14:05:00 | 570.90 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2023-06-21 11:10:00 | 568.40 | 2023-06-21 11:45:00 | 569.55 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-22 10:10:00 | 589.25 | 2023-06-22 10:15:00 | 593.20 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2023-06-22 10:10:00 | 589.25 | 2023-06-22 10:30:00 | 589.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-27 09:35:00 | 573.50 | 2023-06-27 09:40:00 | 572.21 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-06-28 09:40:00 | 565.30 | 2023-06-28 10:40:00 | 563.16 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-06-28 09:40:00 | 565.30 | 2023-06-28 15:20:00 | 560.40 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2023-06-30 10:55:00 | 562.70 | 2023-06-30 11:15:00 | 564.21 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-07-07 09:40:00 | 610.50 | 2023-07-07 09:50:00 | 608.12 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-07-11 10:00:00 | 614.70 | 2023-07-11 12:30:00 | 612.39 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-07-12 09:45:00 | 602.40 | 2023-07-12 11:15:00 | 599.04 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2023-07-12 09:45:00 | 602.40 | 2023-07-12 15:20:00 | 596.60 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2023-07-13 10:25:00 | 606.60 | 2023-07-13 10:30:00 | 609.80 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-07-13 10:25:00 | 606.60 | 2023-07-13 12:30:00 | 607.55 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2023-07-17 09:45:00 | 606.80 | 2023-07-17 09:50:00 | 609.97 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-07-17 09:45:00 | 606.80 | 2023-07-17 11:40:00 | 608.00 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2023-07-19 09:55:00 | 608.55 | 2023-07-19 10:05:00 | 606.82 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-07-20 09:30:00 | 601.70 | 2023-07-20 10:00:00 | 603.18 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-07-26 10:05:00 | 599.30 | 2023-07-26 10:15:00 | 597.26 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-07-28 09:30:00 | 537.20 | 2023-07-28 09:55:00 | 538.53 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-07-31 10:45:00 | 545.85 | 2023-07-31 11:20:00 | 547.60 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-08-01 09:55:00 | 554.65 | 2023-08-01 10:05:00 | 552.95 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-08-02 10:30:00 | 553.50 | 2023-08-02 11:00:00 | 551.25 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-08-02 10:30:00 | 553.50 | 2023-08-02 12:25:00 | 553.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-04 10:05:00 | 550.00 | 2023-08-04 10:30:00 | 551.45 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-08-08 11:10:00 | 549.80 | 2023-08-08 11:20:00 | 550.96 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-08-11 09:45:00 | 547.95 | 2023-08-11 11:05:00 | 549.37 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-08-21 10:35:00 | 551.20 | 2023-08-21 11:10:00 | 549.71 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-08-22 09:50:00 | 560.95 | 2023-08-22 09:55:00 | 559.37 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-08-23 10:10:00 | 562.00 | 2023-08-23 10:15:00 | 564.21 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-08-23 10:10:00 | 562.00 | 2023-08-23 15:20:00 | 573.50 | TARGET_HIT | 0.50 | 2.05% |
| SELL | retest1 | 2023-08-25 10:20:00 | 559.50 | 2023-08-25 11:20:00 | 561.66 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-08-29 10:10:00 | 563.30 | 2023-08-29 10:15:00 | 561.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-09-01 09:40:00 | 610.90 | 2023-09-01 09:45:00 | 608.61 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-09-04 11:05:00 | 619.25 | 2023-09-04 11:10:00 | 617.29 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-09-06 09:45:00 | 615.00 | 2023-09-06 09:50:00 | 617.74 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-09-07 10:55:00 | 648.35 | 2023-09-07 11:00:00 | 653.41 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2023-09-07 10:55:00 | 648.35 | 2023-09-07 15:20:00 | 658.70 | TARGET_HIT | 0.50 | 1.60% |
| SELL | retest1 | 2023-09-11 09:50:00 | 659.70 | 2023-09-11 10:05:00 | 662.25 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-09-14 09:50:00 | 647.15 | 2023-09-14 09:55:00 | 651.19 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2023-09-14 09:50:00 | 647.15 | 2023-09-14 10:00:00 | 647.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-26 10:10:00 | 626.85 | 2023-09-26 10:15:00 | 629.60 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-09-26 10:10:00 | 626.85 | 2023-09-26 15:20:00 | 652.00 | TARGET_HIT | 0.50 | 4.01% |
| SELL | retest1 | 2023-09-28 09:40:00 | 648.10 | 2023-09-28 09:45:00 | 650.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-10-04 09:30:00 | 649.10 | 2023-10-04 09:35:00 | 646.65 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-10-05 10:25:00 | 642.00 | 2023-10-05 10:35:00 | 643.82 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-10-16 10:05:00 | 664.90 | 2023-10-16 10:10:00 | 661.57 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2023-10-18 10:25:00 | 689.40 | 2023-10-18 10:40:00 | 686.50 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-10-20 10:15:00 | 675.50 | 2023-10-20 11:30:00 | 671.51 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2023-10-20 10:15:00 | 675.50 | 2023-10-20 15:20:00 | 662.55 | TARGET_HIT | 0.50 | 1.92% |
| BUY | retest1 | 2023-10-31 09:35:00 | 646.05 | 2023-10-31 09:45:00 | 649.80 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-10-31 09:35:00 | 646.05 | 2023-10-31 09:50:00 | 646.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-01 10:35:00 | 630.10 | 2023-11-01 10:45:00 | 631.95 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-11-06 09:35:00 | 613.90 | 2023-11-06 09:40:00 | 618.09 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2023-11-06 09:35:00 | 613.90 | 2023-11-06 09:50:00 | 613.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-07 09:30:00 | 620.05 | 2023-11-07 09:40:00 | 617.84 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-11-10 10:25:00 | 621.10 | 2023-11-10 10:35:00 | 622.58 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-11-13 10:50:00 | 612.50 | 2023-11-13 11:00:00 | 614.19 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-11-15 09:35:00 | 618.90 | 2023-11-15 09:40:00 | 617.09 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-11-16 11:00:00 | 621.90 | 2023-11-16 11:10:00 | 620.08 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-17 09:35:00 | 614.40 | 2023-11-17 09:50:00 | 616.21 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-11-20 09:40:00 | 603.50 | 2023-11-20 10:35:00 | 605.32 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-11-22 09:45:00 | 609.50 | 2023-11-22 10:30:00 | 607.66 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-11-23 09:30:00 | 610.45 | 2023-11-23 09:40:00 | 613.74 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2023-11-23 09:30:00 | 610.45 | 2023-11-23 10:25:00 | 610.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-24 09:50:00 | 603.80 | 2023-11-24 10:05:00 | 605.79 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-11-28 09:50:00 | 604.80 | 2023-11-28 09:55:00 | 603.16 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-11-29 10:10:00 | 607.95 | 2023-11-29 10:25:00 | 610.16 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-11-29 10:10:00 | 607.95 | 2023-11-29 15:20:00 | 627.30 | TARGET_HIT | 0.50 | 3.18% |
| SELL | retest1 | 2023-11-30 09:50:00 | 626.05 | 2023-11-30 09:55:00 | 622.25 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2023-11-30 09:50:00 | 626.05 | 2023-11-30 10:05:00 | 626.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-01 09:35:00 | 641.45 | 2023-12-01 09:40:00 | 638.48 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2023-12-04 10:40:00 | 637.30 | 2023-12-04 10:50:00 | 641.01 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-12-04 10:40:00 | 637.30 | 2023-12-04 11:25:00 | 637.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-05 10:00:00 | 651.20 | 2023-12-05 10:10:00 | 648.14 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2023-12-06 09:35:00 | 647.75 | 2023-12-06 09:45:00 | 645.22 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-12-08 10:20:00 | 666.00 | 2023-12-08 10:55:00 | 663.09 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-12-11 10:05:00 | 659.05 | 2023-12-11 10:15:00 | 656.08 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-12-18 09:40:00 | 664.45 | 2023-12-18 09:45:00 | 661.80 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-12-20 10:10:00 | 704.50 | 2023-12-20 11:30:00 | 700.42 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2023-12-20 10:10:00 | 704.50 | 2023-12-20 15:20:00 | 657.90 | TARGET_HIT | 0.50 | 6.61% |
| BUY | retest1 | 2023-12-27 09:30:00 | 686.10 | 2023-12-27 09:55:00 | 683.40 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-12-28 09:30:00 | 675.70 | 2023-12-28 09:35:00 | 671.12 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2023-12-28 09:30:00 | 675.70 | 2023-12-28 11:30:00 | 674.45 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2023-12-29 10:05:00 | 678.80 | 2023-12-29 10:15:00 | 683.71 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2023-12-29 10:05:00 | 678.80 | 2023-12-29 11:35:00 | 681.20 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2024-01-01 09:35:00 | 693.90 | 2024-01-01 09:50:00 | 690.48 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-01-03 09:40:00 | 686.35 | 2024-01-03 09:45:00 | 688.96 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-01-05 10:45:00 | 688.35 | 2024-01-05 10:55:00 | 685.30 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-01-05 10:45:00 | 688.35 | 2024-01-05 11:00:00 | 688.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-09 09:40:00 | 669.50 | 2024-01-09 10:50:00 | 672.12 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-01-10 10:45:00 | 649.55 | 2024-01-10 11:10:00 | 651.83 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-01-15 09:50:00 | 652.00 | 2024-01-15 10:00:00 | 649.18 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-01-15 09:50:00 | 652.00 | 2024-01-15 10:05:00 | 652.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-16 09:40:00 | 654.50 | 2024-01-16 10:00:00 | 652.57 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-01-18 09:45:00 | 638.50 | 2024-01-18 09:50:00 | 633.12 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2024-01-18 09:45:00 | 638.50 | 2024-01-18 10:00:00 | 638.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-19 09:35:00 | 660.65 | 2024-01-19 10:00:00 | 656.89 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-01-20 09:40:00 | 653.60 | 2024-01-20 09:45:00 | 651.51 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-01-25 10:00:00 | 633.65 | 2024-01-25 12:05:00 | 636.71 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-01-29 09:40:00 | 634.35 | 2024-01-29 11:25:00 | 630.05 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-01-29 09:40:00 | 634.35 | 2024-01-29 15:20:00 | 628.40 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2024-01-31 09:35:00 | 624.65 | 2024-01-31 10:00:00 | 622.30 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-02-01 10:50:00 | 615.50 | 2024-02-01 10:55:00 | 617.68 | STOP_HIT | 1.00 | -0.35% |
