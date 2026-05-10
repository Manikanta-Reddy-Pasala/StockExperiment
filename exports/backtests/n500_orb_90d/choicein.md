# Choice International Ltd. (CHOICEIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 686.95
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
| ENTRY1 | 24 |
| ENTRY2 | 0 |
| PARTIAL | 14 |
| TARGET_HIT | 9 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 15
- **Target hits / Stop hits / Partials:** 9 / 15 / 14
- **Avg / median % per leg:** 0.50% / 0.36%
- **Sum % (uncompounded):** 18.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 11 | 52.4% | 4 | 10 | 7 | 0.30% | 6.3% |
| BUY @ 2nd Alert (retest1) | 21 | 11 | 52.4% | 4 | 10 | 7 | 0.30% | 6.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 12 | 70.6% | 5 | 5 | 7 | 0.74% | 12.6% |
| SELL @ 2nd Alert (retest1) | 17 | 12 | 70.6% | 5 | 5 | 7 | 0.74% | 12.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 38 | 23 | 60.5% | 9 | 15 | 14 | 0.50% | 19.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:10:00 | 770.80 | 769.15 | 0.00 | ORB-long ORB[760.10,765.70] vol=3.7x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:25:00 | 774.72 | 769.61 | 0.00 | T1 1.5R @ 774.72 |
| Stop hit — per-position SL triggered | 2026-02-09 11:50:00 | 770.80 | 770.97 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 783.00 | 780.28 | 0.00 | ORB-long ORB[775.00,782.05] vol=1.7x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:25:00 | 786.41 | 783.16 | 0.00 | T1 1.5R @ 786.41 |
| Target hit | 2026-02-10 10:40:00 | 784.00 | 785.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2026-02-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:20:00 | 755.55 | 751.19 | 0.00 | ORB-long ORB[746.00,754.50] vol=3.3x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:40:00 | 758.72 | 752.85 | 0.00 | T1 1.5R @ 758.72 |
| Target hit | 2026-02-16 12:45:00 | 763.10 | 770.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 760.00 | 757.88 | 0.00 | ORB-long ORB[754.35,759.00] vol=4.8x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:40:00 | 762.32 | 759.49 | 0.00 | T1 1.5R @ 762.32 |
| Target hit | 2026-02-18 15:20:00 | 793.50 | 785.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-02-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:25:00 | 807.95 | 800.31 | 0.00 | ORB-long ORB[793.45,801.50] vol=1.8x ATR=3.08 |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 804.87 | 804.49 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:30:00 | 798.75 | 802.20 | 0.00 | ORB-short ORB[799.50,808.90] vol=2.4x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:40:00 | 795.65 | 801.32 | 0.00 | T1 1.5R @ 795.65 |
| Target hit | 2026-02-23 15:20:00 | 767.00 | 782.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-02-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:50:00 | 765.45 | 762.37 | 0.00 | ORB-long ORB[759.00,765.00] vol=5.0x ATR=2.33 |
| Stop hit — per-position SL triggered | 2026-02-25 10:00:00 | 763.12 | 763.39 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:05:00 | 760.00 | 756.45 | 0.00 | ORB-long ORB[752.45,756.05] vol=2.7x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-02-26 10:10:00 | 758.52 | 756.68 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:05:00 | 740.80 | 743.59 | 0.00 | ORB-short ORB[741.30,749.95] vol=3.4x ATR=1.75 |
| Stop hit — per-position SL triggered | 2026-02-27 12:30:00 | 742.55 | 742.85 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-02 11:10:00 | 720.05 | 724.62 | 0.00 | ORB-short ORB[720.50,730.00] vol=1.9x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 11:25:00 | 717.28 | 724.23 | 0.00 | T1 1.5R @ 717.28 |
| Target hit | 2026-03-02 15:20:00 | 708.80 | 717.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:30:00 | 693.00 | 696.19 | 0.00 | ORB-short ORB[694.00,704.10] vol=2.8x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:40:00 | 689.08 | 695.29 | 0.00 | T1 1.5R @ 689.08 |
| Target hit | 2026-03-04 15:20:00 | 682.00 | 684.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-03-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:35:00 | 661.05 | 668.89 | 0.00 | ORB-short ORB[668.30,675.00] vol=2.0x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:30:00 | 656.73 | 666.85 | 0.00 | T1 1.5R @ 656.73 |
| Stop hit — per-position SL triggered | 2026-03-11 12:15:00 | 661.05 | 665.53 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:10:00 | 616.25 | 623.52 | 0.00 | ORB-short ORB[621.10,629.40] vol=1.5x ATR=2.53 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 618.78 | 623.30 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:40:00 | 642.30 | 636.77 | 0.00 | ORB-long ORB[632.65,640.50] vol=2.4x ATR=2.85 |
| Stop hit — per-position SL triggered | 2026-04-07 10:50:00 | 639.45 | 636.90 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 10:15:00 | 683.65 | 678.40 | 0.00 | ORB-long ORB[673.55,682.00] vol=2.8x ATR=2.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 10:50:00 | 687.90 | 681.58 | 0.00 | T1 1.5R @ 687.90 |
| Stop hit — per-position SL triggered | 2026-04-09 13:55:00 | 683.65 | 684.36 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:45:00 | 714.50 | 710.98 | 0.00 | ORB-long ORB[705.10,714.30] vol=2.3x ATR=3.50 |
| Stop hit — per-position SL triggered | 2026-04-15 10:35:00 | 711.00 | 711.88 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 729.00 | 725.49 | 0.00 | ORB-long ORB[721.50,727.85] vol=2.4x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:35:00 | 731.61 | 728.25 | 0.00 | T1 1.5R @ 731.61 |
| Target hit | 2026-04-21 11:45:00 | 730.50 | 730.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — SELL (started 2026-04-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 11:10:00 | 716.50 | 718.33 | 0.00 | ORB-short ORB[718.55,726.40] vol=2.2x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 12:20:00 | 713.92 | 717.61 | 0.00 | T1 1.5R @ 713.92 |
| Target hit | 2026-04-22 15:20:00 | 708.35 | 714.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:15:00 | 688.50 | 689.37 | 0.00 | ORB-short ORB[690.10,695.95] vol=1.5x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:25:00 | 685.94 | 689.28 | 0.00 | T1 1.5R @ 685.94 |
| Target hit | 2026-04-28 15:20:00 | 675.55 | 683.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 675.40 | 681.57 | 0.00 | ORB-short ORB[680.00,689.30] vol=2.0x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:30:00 | 670.64 | 680.77 | 0.00 | T1 1.5R @ 670.64 |
| Stop hit — per-position SL triggered | 2026-04-29 11:00:00 | 675.40 | 679.95 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 673.00 | 669.39 | 0.00 | ORB-long ORB[664.95,671.75] vol=1.7x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:00:00 | 676.63 | 671.66 | 0.00 | T1 1.5R @ 676.63 |
| Stop hit — per-position SL triggered | 2026-05-04 10:10:00 | 673.00 | 671.80 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2026-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:45:00 | 664.00 | 667.07 | 0.00 | ORB-short ORB[666.10,671.45] vol=2.1x ATR=2.00 |
| Stop hit — per-position SL triggered | 2026-05-05 10:00:00 | 666.00 | 666.82 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 673.40 | 669.52 | 0.00 | ORB-long ORB[662.20,670.10] vol=2.9x ATR=2.78 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 670.62 | 671.14 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2026-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:45:00 | 695.00 | 690.79 | 0.00 | ORB-long ORB[687.00,693.00] vol=2.3x ATR=2.46 |
| Stop hit — per-position SL triggered | 2026-05-07 09:50:00 | 692.54 | 691.17 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:10:00 | 770.80 | 2026-02-09 11:25:00 | 774.72 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-09 11:10:00 | 770.80 | 2026-02-09 11:50:00 | 770.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-10 09:30:00 | 783.00 | 2026-02-10 10:25:00 | 786.41 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-02-10 09:30:00 | 783.00 | 2026-02-10 10:40:00 | 784.00 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2026-02-16 10:20:00 | 755.55 | 2026-02-16 10:40:00 | 758.72 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-16 10:20:00 | 755.55 | 2026-02-16 12:45:00 | 763.10 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2026-02-18 10:55:00 | 760.00 | 2026-02-18 11:40:00 | 762.32 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-18 10:55:00 | 760.00 | 2026-02-18 15:20:00 | 793.50 | TARGET_HIT | 0.50 | 4.41% |
| BUY | retest1 | 2026-02-20 10:25:00 | 807.95 | 2026-02-20 11:15:00 | 804.87 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-23 09:30:00 | 798.75 | 2026-02-23 09:40:00 | 795.65 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-23 09:30:00 | 798.75 | 2026-02-23 15:20:00 | 767.00 | TARGET_HIT | 0.50 | 3.97% |
| BUY | retest1 | 2026-02-25 09:50:00 | 765.45 | 2026-02-25 10:00:00 | 763.12 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-26 10:05:00 | 760.00 | 2026-02-26 10:10:00 | 758.52 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-27 11:05:00 | 740.80 | 2026-02-27 12:30:00 | 742.55 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-02 11:10:00 | 720.05 | 2026-03-02 11:25:00 | 717.28 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-03-02 11:10:00 | 720.05 | 2026-03-02 15:20:00 | 708.80 | TARGET_HIT | 0.50 | 1.56% |
| SELL | retest1 | 2026-03-04 09:30:00 | 693.00 | 2026-03-04 09:40:00 | 689.08 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-04 09:30:00 | 693.00 | 2026-03-04 15:20:00 | 682.00 | TARGET_HIT | 0.50 | 1.59% |
| SELL | retest1 | 2026-03-11 10:35:00 | 661.05 | 2026-03-11 11:30:00 | 656.73 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-03-11 10:35:00 | 661.05 | 2026-03-11 12:15:00 | 661.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 11:10:00 | 616.25 | 2026-03-16 11:15:00 | 618.78 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-07 10:40:00 | 642.30 | 2026-04-07 10:50:00 | 639.45 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-09 10:15:00 | 683.65 | 2026-04-09 10:50:00 | 687.90 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-09 10:15:00 | 683.65 | 2026-04-09 13:55:00 | 683.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 09:45:00 | 714.50 | 2026-04-15 10:35:00 | 711.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-04-21 10:10:00 | 729.00 | 2026-04-21 10:35:00 | 731.61 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-04-21 10:10:00 | 729.00 | 2026-04-21 11:45:00 | 730.50 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2026-04-22 11:10:00 | 716.50 | 2026-04-22 12:20:00 | 713.92 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-04-22 11:10:00 | 716.50 | 2026-04-22 15:20:00 | 708.35 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2026-04-28 11:15:00 | 688.50 | 2026-04-28 11:25:00 | 685.94 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-04-28 11:15:00 | 688.50 | 2026-04-28 15:20:00 | 675.55 | TARGET_HIT | 0.50 | 1.88% |
| SELL | retest1 | 2026-04-29 10:20:00 | 675.40 | 2026-04-29 10:30:00 | 670.64 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-29 10:20:00 | 675.40 | 2026-04-29 11:00:00 | 675.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 09:45:00 | 673.00 | 2026-05-04 10:00:00 | 676.63 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-05-04 09:45:00 | 673.00 | 2026-05-04 10:10:00 | 673.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 09:45:00 | 664.00 | 2026-05-05 10:00:00 | 666.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-06 09:30:00 | 673.40 | 2026-05-06 10:05:00 | 670.62 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-05-07 09:45:00 | 695.00 | 2026-05-07 09:50:00 | 692.54 | STOP_HIT | 1.00 | -0.35% |
