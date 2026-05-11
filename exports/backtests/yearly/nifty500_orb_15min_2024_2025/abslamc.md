# Aditya Birla Sun Life AMC Ltd. (ABSLAMC)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1075.00
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
| ENTRY1 | 60 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 9 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 84 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 51
- **Target hits / Stop hits / Partials:** 9 / 51 / 24
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 11.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 17 | 40.5% | 6 | 25 | 11 | 0.14% | 5.7% |
| BUY @ 2nd Alert (retest1) | 42 | 17 | 40.5% | 6 | 25 | 11 | 0.14% | 5.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 16 | 38.1% | 3 | 26 | 13 | 0.14% | 5.9% |
| SELL @ 2nd Alert (retest1) | 42 | 16 | 38.1% | 3 | 26 | 13 | 0.14% | 5.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 84 | 33 | 39.3% | 9 | 51 | 24 | 0.14% | 11.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:25:00 | 530.00 | 531.92 | 0.00 | ORB-short ORB[531.30,539.00] vol=2.4x ATR=1.25 |
| Stop hit — per-position SL triggered | 2024-05-14 10:55:00 | 531.25 | 531.20 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:50:00 | 534.65 | 537.06 | 0.00 | ORB-short ORB[538.45,543.80] vol=1.9x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 11:25:00 | 532.87 | 536.54 | 0.00 | T1 1.5R @ 532.87 |
| Target hit | 2024-05-15 15:20:00 | 530.00 | 533.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 09:30:00 | 533.20 | 534.36 | 0.00 | ORB-short ORB[534.95,538.70] vol=2.5x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 09:35:00 | 530.66 | 533.51 | 0.00 | T1 1.5R @ 530.66 |
| Stop hit — per-position SL triggered | 2024-05-17 09:40:00 | 533.20 | 533.28 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 10:55:00 | 522.85 | 519.95 | 0.00 | ORB-long ORB[514.00,521.35] vol=2.5x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 11:20:00 | 525.31 | 520.54 | 0.00 | T1 1.5R @ 525.31 |
| Target hit | 2024-05-22 15:20:00 | 534.60 | 530.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-05-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 11:10:00 | 531.70 | 532.01 | 0.00 | ORB-short ORB[532.15,536.75] vol=2.1x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-05-23 11:15:00 | 533.42 | 532.15 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:40:00 | 530.05 | 532.11 | 0.00 | ORB-short ORB[532.15,535.00] vol=1.6x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 09:45:00 | 528.43 | 531.83 | 0.00 | T1 1.5R @ 528.43 |
| Stop hit — per-position SL triggered | 2024-05-24 10:25:00 | 530.05 | 530.26 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 09:45:00 | 577.45 | 573.66 | 0.00 | ORB-long ORB[565.90,572.45] vol=5.2x ATR=3.63 |
| Stop hit — per-position SL triggered | 2024-06-06 09:50:00 | 573.82 | 573.74 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 11:05:00 | 613.00 | 607.08 | 0.00 | ORB-long ORB[601.00,607.00] vol=6.7x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 11:10:00 | 616.92 | 607.88 | 0.00 | T1 1.5R @ 616.92 |
| Target hit | 2024-06-11 15:20:00 | 618.20 | 613.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:15:00 | 637.55 | 625.48 | 0.00 | ORB-long ORB[620.00,625.00] vol=2.6x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:20:00 | 641.17 | 625.98 | 0.00 | T1 1.5R @ 641.17 |
| Stop hit — per-position SL triggered | 2024-06-12 11:25:00 | 637.55 | 626.06 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-24 09:35:00 | 649.60 | 655.51 | 0.00 | ORB-short ORB[655.00,660.70] vol=2.4x ATR=2.50 |
| Stop hit — per-position SL triggered | 2024-06-24 09:55:00 | 652.10 | 653.91 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:15:00 | 645.65 | 648.75 | 0.00 | ORB-short ORB[646.05,652.00] vol=1.6x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:20:00 | 643.36 | 648.07 | 0.00 | T1 1.5R @ 643.36 |
| Stop hit — per-position SL triggered | 2024-06-27 10:25:00 | 645.65 | 647.71 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 11:00:00 | 648.65 | 645.22 | 0.00 | ORB-long ORB[636.00,643.90] vol=2.6x ATR=2.24 |
| Stop hit — per-position SL triggered | 2024-07-01 11:05:00 | 646.41 | 645.25 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 11:15:00 | 659.00 | 664.28 | 0.00 | ORB-short ORB[662.00,669.70] vol=2.6x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 11:30:00 | 656.32 | 663.61 | 0.00 | T1 1.5R @ 656.32 |
| Stop hit — per-position SL triggered | 2024-07-09 12:45:00 | 659.00 | 661.37 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:55:00 | 669.00 | 671.00 | 0.00 | ORB-short ORB[670.30,674.95] vol=1.6x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-07-12 10:05:00 | 670.87 | 670.74 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:10:00 | 685.50 | 680.07 | 0.00 | ORB-long ORB[676.40,683.80] vol=3.6x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:15:00 | 689.79 | 681.37 | 0.00 | T1 1.5R @ 689.79 |
| Stop hit — per-position SL triggered | 2024-07-15 10:20:00 | 685.50 | 681.57 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:30:00 | 713.25 | 709.19 | 0.00 | ORB-long ORB[704.40,712.50] vol=2.0x ATR=3.50 |
| Stop hit — per-position SL triggered | 2024-07-16 10:10:00 | 709.75 | 711.48 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 09:40:00 | 711.50 | 716.83 | 0.00 | ORB-short ORB[714.40,723.95] vol=1.5x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-07-29 10:10:00 | 715.78 | 715.88 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 09:35:00 | 710.20 | 713.00 | 0.00 | ORB-short ORB[711.05,720.00] vol=2.2x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 11:30:00 | 706.31 | 710.89 | 0.00 | T1 1.5R @ 706.31 |
| Target hit | 2024-07-30 14:10:00 | 708.65 | 708.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — BUY (started 2024-07-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:55:00 | 707.05 | 705.32 | 0.00 | ORB-long ORB[702.25,706.50] vol=2.2x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-07-31 15:20:00 | 706.35 | 706.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-08-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:50:00 | 701.10 | 697.75 | 0.00 | ORB-long ORB[690.00,698.95] vol=1.8x ATR=3.62 |
| Stop hit — per-position SL triggered | 2024-08-13 10:25:00 | 697.48 | 698.73 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 10:45:00 | 694.90 | 692.29 | 0.00 | ORB-long ORB[688.55,693.30] vol=2.1x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 13:20:00 | 699.50 | 694.77 | 0.00 | T1 1.5R @ 699.50 |
| Target hit | 2024-08-14 15:20:00 | 710.00 | 697.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 713.45 | 710.66 | 0.00 | ORB-long ORB[704.25,713.25] vol=3.5x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-08-19 10:20:00 | 710.62 | 713.03 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:40:00 | 728.95 | 733.75 | 0.00 | ORB-short ORB[730.00,740.50] vol=1.5x ATR=3.58 |
| Stop hit — per-position SL triggered | 2024-08-27 09:55:00 | 732.53 | 732.00 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:45:00 | 716.65 | 724.29 | 0.00 | ORB-short ORB[728.60,738.00] vol=7.3x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-08-29 11:45:00 | 719.84 | 722.25 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:50:00 | 729.15 | 724.10 | 0.00 | ORB-long ORB[720.00,726.70] vol=3.0x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 09:55:00 | 733.23 | 725.90 | 0.00 | T1 1.5R @ 733.23 |
| Target hit | 2024-08-30 15:20:00 | 747.55 | 740.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2024-09-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 09:55:00 | 736.85 | 738.28 | 0.00 | ORB-short ORB[737.05,741.35] vol=3.3x ATR=1.84 |
| Stop hit — per-position SL triggered | 2024-09-05 10:05:00 | 738.69 | 738.31 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 11:10:00 | 731.60 | 739.60 | 0.00 | ORB-short ORB[741.70,749.05] vol=1.6x ATR=2.92 |
| Stop hit — per-position SL triggered | 2024-09-06 11:40:00 | 734.52 | 738.94 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 09:45:00 | 730.10 | 722.35 | 0.00 | ORB-long ORB[715.05,724.95] vol=1.7x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-09-09 09:50:00 | 726.41 | 722.62 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:30:00 | 739.10 | 733.86 | 0.00 | ORB-long ORB[724.05,735.00] vol=2.6x ATR=2.70 |
| Stop hit — per-position SL triggered | 2024-09-10 09:45:00 | 736.40 | 735.32 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 10:40:00 | 735.75 | 737.65 | 0.00 | ORB-short ORB[737.00,743.60] vol=2.3x ATR=2.04 |
| Stop hit — per-position SL triggered | 2024-09-12 10:45:00 | 737.79 | 737.94 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:30:00 | 750.00 | 748.00 | 0.00 | ORB-long ORB[741.05,749.20] vol=4.0x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 09:35:00 | 753.75 | 748.78 | 0.00 | T1 1.5R @ 753.75 |
| Target hit | 2024-09-16 10:00:00 | 751.70 | 752.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2024-09-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:50:00 | 751.00 | 746.08 | 0.00 | ORB-long ORB[740.05,748.00] vol=1.9x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-09-17 10:15:00 | 747.81 | 747.02 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 10:00:00 | 727.50 | 725.59 | 0.00 | ORB-long ORB[715.60,724.85] vol=9.7x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-10-01 10:05:00 | 725.43 | 725.92 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:25:00 | 745.90 | 751.41 | 0.00 | ORB-short ORB[748.00,756.45] vol=2.0x ATR=3.61 |
| Stop hit — per-position SL triggered | 2024-10-22 10:35:00 | 749.51 | 750.51 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:30:00 | 778.40 | 771.96 | 0.00 | ORB-long ORB[765.00,773.70] vol=3.2x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 09:40:00 | 782.77 | 775.51 | 0.00 | T1 1.5R @ 782.77 |
| Target hit | 2024-10-31 10:20:00 | 780.65 | 780.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — SELL (started 2024-11-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 11:00:00 | 816.75 | 819.42 | 0.00 | ORB-short ORB[818.20,828.95] vol=2.6x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-11-06 11:40:00 | 819.95 | 819.06 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-11-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-18 10:05:00 | 815.30 | 803.27 | 0.00 | ORB-long ORB[795.75,806.40] vol=2.3x ATR=5.16 |
| Stop hit — per-position SL triggered | 2024-11-18 10:15:00 | 810.14 | 804.37 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:35:00 | 878.75 | 882.12 | 0.00 | ORB-short ORB[879.00,887.70] vol=2.3x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 11:30:00 | 873.76 | 881.06 | 0.00 | T1 1.5R @ 873.76 |
| Stop hit — per-position SL triggered | 2024-12-03 11:40:00 | 878.75 | 880.83 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:50:00 | 823.75 | 828.39 | 0.00 | ORB-short ORB[825.00,833.15] vol=1.9x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 12:45:00 | 818.58 | 826.59 | 0.00 | T1 1.5R @ 818.58 |
| Stop hit — per-position SL triggered | 2024-12-12 13:20:00 | 823.75 | 825.60 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 09:30:00 | 834.05 | 840.37 | 0.00 | ORB-short ORB[835.00,847.00] vol=1.6x ATR=4.52 |
| Stop hit — per-position SL triggered | 2024-12-16 09:35:00 | 838.57 | 840.01 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:40:00 | 824.10 | 811.01 | 0.00 | ORB-long ORB[804.95,812.50] vol=1.6x ATR=4.38 |
| Stop hit — per-position SL triggered | 2024-12-19 09:45:00 | 819.72 | 812.40 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 09:35:00 | 810.00 | 806.56 | 0.00 | ORB-long ORB[800.05,808.60] vol=3.2x ATR=3.81 |
| Stop hit — per-position SL triggered | 2024-12-26 09:45:00 | 806.19 | 807.02 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:35:00 | 850.30 | 846.10 | 0.00 | ORB-long ORB[840.30,847.40] vol=2.9x ATR=2.90 |
| Stop hit — per-position SL triggered | 2025-01-02 09:40:00 | 847.40 | 846.37 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-01-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 11:05:00 | 794.55 | 799.13 | 0.00 | ORB-short ORB[795.55,806.00] vol=1.6x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-01-09 11:20:00 | 796.63 | 798.78 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-01-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:35:00 | 752.70 | 763.47 | 0.00 | ORB-short ORB[761.00,771.50] vol=1.5x ATR=5.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 10:55:00 | 744.20 | 756.67 | 0.00 | T1 1.5R @ 744.20 |
| Target hit | 2025-01-13 15:20:00 | 723.80 | 742.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2025-01-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 11:05:00 | 768.70 | 773.51 | 0.00 | ORB-short ORB[772.00,782.00] vol=4.1x ATR=3.29 |
| Stop hit — per-position SL triggered | 2025-01-16 11:15:00 | 771.99 | 773.43 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:35:00 | 751.25 | 757.79 | 0.00 | ORB-short ORB[755.65,763.55] vol=2.6x ATR=3.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 12:10:00 | 746.70 | 753.71 | 0.00 | T1 1.5R @ 746.70 |
| Stop hit — per-position SL triggered | 2025-01-20 15:00:00 | 751.25 | 751.90 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-01-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 11:05:00 | 711.80 | 714.44 | 0.00 | ORB-short ORB[718.05,728.00] vol=3.6x ATR=2.71 |
| Stop hit — per-position SL triggered | 2025-01-24 11:10:00 | 714.51 | 714.41 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:45:00 | 678.15 | 664.76 | 0.00 | ORB-long ORB[641.75,651.60] vol=4.7x ATR=4.31 |
| Stop hit — per-position SL triggered | 2025-01-31 10:50:00 | 673.84 | 665.38 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-02-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:55:00 | 710.00 | 706.27 | 0.00 | ORB-long ORB[700.00,706.15] vol=3.1x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 10:05:00 | 714.26 | 710.85 | 0.00 | T1 1.5R @ 714.26 |
| Stop hit — per-position SL triggered | 2025-02-06 10:40:00 | 710.00 | 711.27 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-03-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:50:00 | 605.75 | 601.76 | 0.00 | ORB-long ORB[597.00,605.20] vol=1.6x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:05:00 | 609.60 | 603.43 | 0.00 | T1 1.5R @ 609.60 |
| Stop hit — per-position SL triggered | 2025-03-19 10:20:00 | 605.75 | 603.93 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-03-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:50:00 | 656.95 | 649.36 | 0.00 | ORB-long ORB[640.05,648.00] vol=3.6x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-03-24 11:00:00 | 653.99 | 649.77 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-03-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:30:00 | 640.25 | 644.41 | 0.00 | ORB-short ORB[642.00,650.00] vol=2.4x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 09:40:00 | 636.61 | 642.03 | 0.00 | T1 1.5R @ 636.61 |
| Stop hit — per-position SL triggered | 2025-03-26 09:55:00 | 640.25 | 640.06 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-03-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 11:05:00 | 652.05 | 648.64 | 0.00 | ORB-long ORB[642.15,651.90] vol=2.5x ATR=2.40 |
| Stop hit — per-position SL triggered | 2025-03-28 11:15:00 | 649.65 | 648.86 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-04-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 10:55:00 | 655.00 | 650.00 | 0.00 | ORB-long ORB[643.20,652.30] vol=1.9x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-04-03 11:15:00 | 652.11 | 650.34 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 11:15:00 | 633.65 | 630.53 | 0.00 | ORB-long ORB[623.05,632.40] vol=2.4x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 12:20:00 | 636.40 | 631.85 | 0.00 | T1 1.5R @ 636.40 |
| Stop hit — per-position SL triggered | 2025-04-16 12:30:00 | 633.65 | 631.88 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-04-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 11:10:00 | 678.70 | 665.37 | 0.00 | ORB-long ORB[645.00,652.25] vol=6.7x ATR=3.10 |
| Stop hit — per-position SL triggered | 2025-04-22 11:25:00 | 675.60 | 667.99 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-04-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:05:00 | 661.25 | 670.21 | 0.00 | ORB-short ORB[668.05,673.20] vol=1.6x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:55:00 | 657.02 | 668.03 | 0.00 | T1 1.5R @ 657.02 |
| Stop hit — per-position SL triggered | 2025-04-23 11:10:00 | 661.25 | 667.60 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-04-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:50:00 | 646.75 | 652.86 | 0.00 | ORB-short ORB[654.40,662.95] vol=1.6x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:55:00 | 642.73 | 651.67 | 0.00 | T1 1.5R @ 642.73 |
| Stop hit — per-position SL triggered | 2025-04-25 10:10:00 | 646.75 | 647.48 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 11:15:00 | 643.45 | 637.12 | 0.00 | ORB-long ORB[630.00,639.00] vol=2.9x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-05-07 11:45:00 | 640.58 | 637.56 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:25:00 | 530.00 | 2024-05-14 10:55:00 | 531.25 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-15 10:50:00 | 534.65 | 2024-05-15 11:25:00 | 532.87 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-05-15 10:50:00 | 534.65 | 2024-05-15 15:20:00 | 530.00 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2024-05-17 09:30:00 | 533.20 | 2024-05-17 09:35:00 | 530.66 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-05-17 09:30:00 | 533.20 | 2024-05-17 09:40:00 | 533.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-22 10:55:00 | 522.85 | 2024-05-22 11:20:00 | 525.31 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-05-22 10:55:00 | 522.85 | 2024-05-22 15:20:00 | 534.60 | TARGET_HIT | 0.50 | 2.25% |
| SELL | retest1 | 2024-05-23 11:10:00 | 531.70 | 2024-05-23 11:15:00 | 533.42 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-24 09:40:00 | 530.05 | 2024-05-24 09:45:00 | 528.43 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-05-24 09:40:00 | 530.05 | 2024-05-24 10:25:00 | 530.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-06 09:45:00 | 577.45 | 2024-06-06 09:50:00 | 573.82 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2024-06-11 11:05:00 | 613.00 | 2024-06-11 11:10:00 | 616.92 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-06-11 11:05:00 | 613.00 | 2024-06-11 15:20:00 | 618.20 | TARGET_HIT | 0.50 | 0.85% |
| BUY | retest1 | 2024-06-12 11:15:00 | 637.55 | 2024-06-12 11:20:00 | 641.17 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-12 11:15:00 | 637.55 | 2024-06-12 11:25:00 | 637.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-24 09:35:00 | 649.60 | 2024-06-24 09:55:00 | 652.10 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-06-27 10:15:00 | 645.65 | 2024-06-27 10:20:00 | 643.36 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-06-27 10:15:00 | 645.65 | 2024-06-27 10:25:00 | 645.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-01 11:00:00 | 648.65 | 2024-07-01 11:05:00 | 646.41 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-09 11:15:00 | 659.00 | 2024-07-09 11:30:00 | 656.32 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-07-09 11:15:00 | 659.00 | 2024-07-09 12:45:00 | 659.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 09:55:00 | 669.00 | 2024-07-12 10:05:00 | 670.87 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-15 10:10:00 | 685.50 | 2024-07-15 10:15:00 | 689.79 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-07-15 10:10:00 | 685.50 | 2024-07-15 10:20:00 | 685.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 09:30:00 | 713.25 | 2024-07-16 10:10:00 | 709.75 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-07-29 09:40:00 | 711.50 | 2024-07-29 10:10:00 | 715.78 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2024-07-30 09:35:00 | 710.20 | 2024-07-30 11:30:00 | 706.31 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-07-30 09:35:00 | 710.20 | 2024-07-30 14:10:00 | 708.65 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2024-07-31 09:55:00 | 707.05 | 2024-07-31 15:20:00 | 706.35 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest1 | 2024-08-13 09:50:00 | 701.10 | 2024-08-13 10:25:00 | 697.48 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-08-14 10:45:00 | 694.90 | 2024-08-14 13:20:00 | 699.50 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-08-14 10:45:00 | 694.90 | 2024-08-14 15:20:00 | 710.00 | TARGET_HIT | 0.50 | 2.17% |
| BUY | retest1 | 2024-08-19 09:35:00 | 713.45 | 2024-08-19 10:20:00 | 710.62 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-08-27 09:40:00 | 728.95 | 2024-08-27 09:55:00 | 732.53 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-08-29 10:45:00 | 716.65 | 2024-08-29 11:45:00 | 719.84 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-08-30 09:50:00 | 729.15 | 2024-08-30 09:55:00 | 733.23 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-08-30 09:50:00 | 729.15 | 2024-08-30 15:20:00 | 747.55 | TARGET_HIT | 0.50 | 2.52% |
| SELL | retest1 | 2024-09-05 09:55:00 | 736.85 | 2024-09-05 10:05:00 | 738.69 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-06 11:10:00 | 731.60 | 2024-09-06 11:40:00 | 734.52 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-09-09 09:45:00 | 730.10 | 2024-09-09 09:50:00 | 726.41 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-09-10 09:30:00 | 739.10 | 2024-09-10 09:45:00 | 736.40 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-09-12 10:40:00 | 735.75 | 2024-09-12 10:45:00 | 737.79 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-16 09:30:00 | 750.00 | 2024-09-16 09:35:00 | 753.75 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-09-16 09:30:00 | 750.00 | 2024-09-16 10:00:00 | 751.70 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2024-09-17 09:50:00 | 751.00 | 2024-09-17 10:15:00 | 747.81 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-10-01 10:00:00 | 727.50 | 2024-10-01 10:05:00 | 725.43 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-22 10:25:00 | 745.90 | 2024-10-22 10:35:00 | 749.51 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-10-31 09:30:00 | 778.40 | 2024-10-31 09:40:00 | 782.77 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-10-31 09:30:00 | 778.40 | 2024-10-31 10:20:00 | 780.65 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2024-11-06 11:00:00 | 816.75 | 2024-11-06 11:40:00 | 819.95 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-11-18 10:05:00 | 815.30 | 2024-11-18 10:15:00 | 810.14 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest1 | 2024-12-03 10:35:00 | 878.75 | 2024-12-03 11:30:00 | 873.76 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-12-03 10:35:00 | 878.75 | 2024-12-03 11:40:00 | 878.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 10:50:00 | 823.75 | 2024-12-12 12:45:00 | 818.58 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-12-12 10:50:00 | 823.75 | 2024-12-12 13:20:00 | 823.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-16 09:30:00 | 834.05 | 2024-12-16 09:35:00 | 838.57 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-12-19 09:40:00 | 824.10 | 2024-12-19 09:45:00 | 819.72 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-12-26 09:35:00 | 810.00 | 2024-12-26 09:45:00 | 806.19 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-01-02 09:35:00 | 850.30 | 2025-01-02 09:40:00 | 847.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-09 11:05:00 | 794.55 | 2025-01-09 11:20:00 | 796.63 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-13 09:35:00 | 752.70 | 2025-01-13 10:55:00 | 744.20 | PARTIAL | 0.50 | 1.13% |
| SELL | retest1 | 2025-01-13 09:35:00 | 752.70 | 2025-01-13 15:20:00 | 723.80 | TARGET_HIT | 0.50 | 3.84% |
| SELL | retest1 | 2025-01-16 11:05:00 | 768.70 | 2025-01-16 11:15:00 | 771.99 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-01-20 09:35:00 | 751.25 | 2025-01-20 12:10:00 | 746.70 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-01-20 09:35:00 | 751.25 | 2025-01-20 15:00:00 | 751.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 11:05:00 | 711.80 | 2025-01-24 11:10:00 | 714.51 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-31 10:45:00 | 678.15 | 2025-01-31 10:50:00 | 673.84 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2025-02-06 09:55:00 | 710.00 | 2025-02-06 10:05:00 | 714.26 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-02-06 09:55:00 | 710.00 | 2025-02-06 10:40:00 | 710.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-19 09:50:00 | 605.75 | 2025-03-19 10:05:00 | 609.60 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-03-19 09:50:00 | 605.75 | 2025-03-19 10:20:00 | 605.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-24 10:50:00 | 656.95 | 2025-03-24 11:00:00 | 653.99 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-03-26 09:30:00 | 640.25 | 2025-03-26 09:40:00 | 636.61 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-03-26 09:30:00 | 640.25 | 2025-03-26 09:55:00 | 640.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-28 11:05:00 | 652.05 | 2025-03-28 11:15:00 | 649.65 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-03 10:55:00 | 655.00 | 2025-04-03 11:15:00 | 652.11 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-04-16 11:15:00 | 633.65 | 2025-04-16 12:20:00 | 636.40 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-04-16 11:15:00 | 633.65 | 2025-04-16 12:30:00 | 633.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-22 11:10:00 | 678.70 | 2025-04-22 11:25:00 | 675.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-04-23 10:05:00 | 661.25 | 2025-04-23 10:55:00 | 657.02 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-04-23 10:05:00 | 661.25 | 2025-04-23 11:10:00 | 661.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 09:50:00 | 646.75 | 2025-04-25 09:55:00 | 642.73 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-04-25 09:50:00 | 646.75 | 2025-04-25 10:10:00 | 646.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-07 11:15:00 | 643.45 | 2025-05-07 11:45:00 | 640.58 | STOP_HIT | 1.00 | -0.45% |
