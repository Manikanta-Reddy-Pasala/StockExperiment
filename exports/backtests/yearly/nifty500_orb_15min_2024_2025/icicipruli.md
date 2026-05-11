# ICICI Prudential Life Insurance Company Ltd. (ICICIPRULI)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35296 bars)
- **Last close:** 565.25
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
| ENTRY1 | 83 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 7 |
| STOP_HIT | 76 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 109 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 76
- **Target hits / Stop hits / Partials:** 7 / 76 / 26
- **Avg / median % per leg:** 0.04% / -0.24%
- **Sum % (uncompounded):** 4.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 14 | 25.0% | 3 | 42 | 11 | -0.07% | -3.8% |
| BUY @ 2nd Alert (retest1) | 56 | 14 | 25.0% | 3 | 42 | 11 | -0.07% | -3.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 53 | 19 | 35.8% | 4 | 34 | 15 | 0.15% | 7.9% |
| SELL @ 2nd Alert (retest1) | 53 | 19 | 35.8% | 4 | 34 | 15 | 0.15% | 7.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 109 | 33 | 30.3% | 7 | 76 | 26 | 0.04% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:30:00 | 583.80 | 586.24 | 0.00 | ORB-short ORB[584.10,592.70] vol=2.9x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 09:55:00 | 580.59 | 584.53 | 0.00 | T1 1.5R @ 580.59 |
| Stop hit — per-position SL triggered | 2024-05-21 10:10:00 | 583.80 | 583.90 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 11:05:00 | 586.20 | 585.60 | 0.00 | ORB-long ORB[581.25,585.75] vol=2.5x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-05-24 11:10:00 | 584.79 | 585.59 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 10:25:00 | 577.85 | 579.76 | 0.00 | ORB-short ORB[579.50,583.50] vol=4.1x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-05-27 10:50:00 | 579.71 | 579.54 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:50:00 | 583.70 | 580.45 | 0.00 | ORB-long ORB[573.00,578.50] vol=2.4x ATR=2.29 |
| Stop hit — per-position SL triggered | 2024-05-28 10:10:00 | 581.41 | 582.03 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 10:00:00 | 570.25 | 572.92 | 0.00 | ORB-short ORB[572.00,580.00] vol=1.6x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 10:20:00 | 566.99 | 571.57 | 0.00 | T1 1.5R @ 566.99 |
| Target hit | 2024-05-29 15:20:00 | 557.45 | 559.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-06-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:30:00 | 578.65 | 573.31 | 0.00 | ORB-long ORB[567.10,575.50] vol=1.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-06-10 11:40:00 | 577.02 | 574.73 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:35:00 | 601.50 | 605.25 | 0.00 | ORB-short ORB[606.90,611.55] vol=2.9x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-06-19 10:40:00 | 603.31 | 605.15 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:20:00 | 604.70 | 606.59 | 0.00 | ORB-short ORB[605.70,610.85] vol=3.3x ATR=1.94 |
| Stop hit — per-position SL triggered | 2024-06-25 10:25:00 | 606.64 | 606.56 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:35:00 | 610.00 | 606.00 | 0.00 | ORB-long ORB[601.45,606.75] vol=4.9x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 10:45:00 | 613.40 | 607.86 | 0.00 | T1 1.5R @ 613.40 |
| Stop hit — per-position SL triggered | 2024-06-26 11:40:00 | 610.00 | 610.89 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:55:00 | 602.95 | 599.58 | 0.00 | ORB-long ORB[594.25,602.55] vol=1.7x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-06-27 10:30:00 | 600.74 | 600.25 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 11:15:00 | 604.70 | 604.40 | 0.00 | ORB-long ORB[599.65,603.45] vol=2.3x ATR=1.82 |
| Stop hit — per-position SL triggered | 2024-06-28 11:40:00 | 602.88 | 604.29 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:55:00 | 613.80 | 608.55 | 0.00 | ORB-long ORB[601.20,608.45] vol=1.6x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-07-01 11:30:00 | 612.04 | 609.99 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:55:00 | 639.65 | 636.54 | 0.00 | ORB-long ORB[634.40,639.50] vol=1.9x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:00:00 | 642.58 | 638.28 | 0.00 | T1 1.5R @ 642.58 |
| Stop hit — per-position SL triggered | 2024-07-04 10:10:00 | 639.65 | 639.01 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 638.90 | 642.84 | 0.00 | ORB-short ORB[640.45,647.25] vol=2.1x ATR=1.99 |
| Stop hit — per-position SL triggered | 2024-07-10 10:10:00 | 640.89 | 642.52 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:05:00 | 654.85 | 653.26 | 0.00 | ORB-long ORB[650.25,654.60] vol=4.0x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:15:00 | 657.44 | 653.80 | 0.00 | T1 1.5R @ 657.44 |
| Target hit | 2024-07-15 13:55:00 | 655.50 | 655.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2024-07-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:40:00 | 645.50 | 647.95 | 0.00 | ORB-short ORB[647.45,653.50] vol=1.6x ATR=2.60 |
| Stop hit — per-position SL triggered | 2024-07-18 09:45:00 | 648.10 | 648.05 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 11:10:00 | 717.50 | 722.64 | 0.00 | ORB-short ORB[721.00,729.15] vol=2.2x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-07-29 11:30:00 | 719.53 | 722.34 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:50:00 | 730.80 | 724.64 | 0.00 | ORB-long ORB[719.55,724.00] vol=3.8x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-07-31 10:55:00 | 728.74 | 725.63 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:45:00 | 741.60 | 737.81 | 0.00 | ORB-long ORB[732.70,741.00] vol=1.5x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-08-01 09:55:00 | 738.96 | 738.05 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 10:45:00 | 737.25 | 731.18 | 0.00 | ORB-long ORB[725.00,733.05] vol=1.8x ATR=2.23 |
| Stop hit — per-position SL triggered | 2024-08-02 11:05:00 | 735.02 | 733.04 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:45:00 | 711.05 | 709.12 | 0.00 | ORB-long ORB[705.00,710.60] vol=1.5x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 11:05:00 | 714.64 | 709.80 | 0.00 | T1 1.5R @ 714.64 |
| Stop hit — per-position SL triggered | 2024-08-07 12:10:00 | 711.05 | 711.28 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 11:15:00 | 726.50 | 721.83 | 0.00 | ORB-long ORB[720.65,724.65] vol=3.9x ATR=2.32 |
| Stop hit — per-position SL triggered | 2024-08-08 11:20:00 | 724.18 | 721.94 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 10:00:00 | 728.75 | 731.48 | 0.00 | ORB-short ORB[731.10,739.25] vol=1.6x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-08-12 10:15:00 | 731.21 | 730.97 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 10:45:00 | 724.25 | 719.78 | 0.00 | ORB-long ORB[715.00,720.65] vol=2.7x ATR=1.98 |
| Stop hit — per-position SL triggered | 2024-08-19 10:50:00 | 722.27 | 719.93 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 728.85 | 725.11 | 0.00 | ORB-long ORB[719.40,728.70] vol=3.0x ATR=2.34 |
| Stop hit — per-position SL triggered | 2024-08-20 09:35:00 | 726.51 | 725.30 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:00:00 | 750.90 | 746.79 | 0.00 | ORB-long ORB[739.55,750.00] vol=1.5x ATR=3.55 |
| Stop hit — per-position SL triggered | 2024-08-21 10:10:00 | 747.35 | 747.06 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:50:00 | 730.95 | 726.89 | 0.00 | ORB-long ORB[722.15,726.95] vol=1.6x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 11:00:00 | 734.66 | 727.96 | 0.00 | T1 1.5R @ 734.66 |
| Stop hit — per-position SL triggered | 2024-08-27 11:15:00 | 730.95 | 730.16 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 749.90 | 746.24 | 0.00 | ORB-long ORB[740.10,745.90] vol=2.6x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 09:35:00 | 752.86 | 748.91 | 0.00 | T1 1.5R @ 752.86 |
| Stop hit — per-position SL triggered | 2024-08-29 09:40:00 | 749.90 | 749.04 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:40:00 | 747.00 | 742.65 | 0.00 | ORB-long ORB[740.35,744.40] vol=2.6x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 11:00:00 | 750.09 | 744.37 | 0.00 | T1 1.5R @ 750.09 |
| Stop hit — per-position SL triggered | 2024-08-30 11:30:00 | 747.00 | 745.52 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:35:00 | 751.35 | 755.38 | 0.00 | ORB-short ORB[752.10,760.30] vol=2.2x ATR=2.45 |
| Stop hit — per-position SL triggered | 2024-09-06 10:50:00 | 753.80 | 754.91 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 09:45:00 | 761.65 | 756.53 | 0.00 | ORB-long ORB[749.85,757.70] vol=1.8x ATR=2.69 |
| Stop hit — per-position SL triggered | 2024-09-09 09:50:00 | 758.96 | 757.01 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:25:00 | 757.20 | 749.43 | 0.00 | ORB-long ORB[741.70,751.80] vol=1.6x ATR=2.56 |
| Stop hit — per-position SL triggered | 2024-09-11 11:20:00 | 754.64 | 752.24 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:20:00 | 762.70 | 759.76 | 0.00 | ORB-long ORB[756.55,761.45] vol=3.8x ATR=2.12 |
| Stop hit — per-position SL triggered | 2024-09-13 10:30:00 | 760.58 | 759.97 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 11:05:00 | 749.35 | 751.70 | 0.00 | ORB-short ORB[751.20,756.20] vol=2.3x ATR=1.75 |
| Stop hit — per-position SL triggered | 2024-09-16 11:15:00 | 751.10 | 751.74 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:40:00 | 759.60 | 757.90 | 0.00 | ORB-long ORB[750.40,755.95] vol=2.8x ATR=2.01 |
| Stop hit — per-position SL triggered | 2024-09-18 10:50:00 | 757.59 | 758.03 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 09:45:00 | 780.40 | 775.48 | 0.00 | ORB-long ORB[766.00,772.00] vol=2.5x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 10:00:00 | 784.67 | 777.47 | 0.00 | T1 1.5R @ 784.67 |
| Target hit | 2024-09-23 15:20:00 | 791.00 | 783.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-09-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:50:00 | 779.55 | 784.73 | 0.00 | ORB-short ORB[784.40,795.00] vol=2.5x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 11:05:00 | 776.00 | 783.70 | 0.00 | T1 1.5R @ 776.00 |
| Stop hit — per-position SL triggered | 2024-09-24 14:50:00 | 779.55 | 777.26 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 11:00:00 | 782.50 | 787.40 | 0.00 | ORB-short ORB[784.55,793.80] vol=1.6x ATR=2.13 |
| Stop hit — per-position SL triggered | 2024-09-30 12:25:00 | 784.63 | 785.75 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-08 09:30:00 | 734.20 | 738.25 | 0.00 | ORB-short ORB[736.00,744.00] vol=1.7x ATR=3.48 |
| Stop hit — per-position SL triggered | 2024-10-08 09:45:00 | 737.68 | 737.73 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 09:35:00 | 738.35 | 741.03 | 0.00 | ORB-short ORB[740.10,745.35] vol=2.6x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-10-11 09:55:00 | 740.49 | 740.02 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 10:15:00 | 746.40 | 743.37 | 0.00 | ORB-long ORB[740.00,746.05] vol=1.6x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 10:40:00 | 749.54 | 745.69 | 0.00 | T1 1.5R @ 749.54 |
| Target hit | 2024-10-15 11:45:00 | 750.30 | 751.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — BUY (started 2024-10-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-21 10:20:00 | 753.55 | 750.64 | 0.00 | ORB-long ORB[745.40,752.55] vol=1.9x ATR=2.31 |
| Stop hit — per-position SL triggered | 2024-10-21 10:35:00 | 751.24 | 751.08 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-22 09:30:00 | 759.30 | 756.17 | 0.00 | ORB-long ORB[748.90,759.00] vol=4.6x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-10-22 09:35:00 | 756.55 | 756.31 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:30:00 | 762.30 | 753.02 | 0.00 | ORB-long ORB[744.65,753.05] vol=1.5x ATR=3.39 |
| Stop hit — per-position SL triggered | 2024-10-24 11:30:00 | 758.91 | 757.65 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:45:00 | 762.00 | 765.94 | 0.00 | ORB-short ORB[768.35,773.95] vol=5.2x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:55:00 | 756.87 | 764.35 | 0.00 | T1 1.5R @ 756.87 |
| Target hit | 2024-10-25 15:20:00 | 744.40 | 750.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2024-11-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:55:00 | 734.90 | 739.78 | 0.00 | ORB-short ORB[739.00,744.80] vol=2.0x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 11:10:00 | 730.60 | 738.58 | 0.00 | T1 1.5R @ 730.60 |
| Stop hit — per-position SL triggered | 2024-11-04 11:20:00 | 734.90 | 738.02 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-11-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 09:55:00 | 726.20 | 730.48 | 0.00 | ORB-short ORB[730.15,738.95] vol=1.7x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 10:05:00 | 722.12 | 729.60 | 0.00 | T1 1.5R @ 722.12 |
| Stop hit — per-position SL triggered | 2024-11-05 10:25:00 | 726.20 | 725.80 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-11-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 10:55:00 | 717.30 | 725.61 | 0.00 | ORB-short ORB[726.45,733.90] vol=1.6x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-11-06 11:35:00 | 720.04 | 723.62 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 10:55:00 | 717.30 | 716.37 | 0.00 | ORB-long ORB[709.15,717.15] vol=5.5x ATR=1.92 |
| Stop hit — per-position SL triggered | 2024-11-08 11:25:00 | 715.38 | 716.36 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-14 10:35:00 | 697.30 | 694.79 | 0.00 | ORB-long ORB[688.05,694.35] vol=2.2x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-11-14 11:00:00 | 695.24 | 696.00 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-11-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-19 09:40:00 | 686.65 | 688.52 | 0.00 | ORB-short ORB[687.05,695.95] vol=1.6x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 09:45:00 | 683.52 | 687.24 | 0.00 | T1 1.5R @ 683.52 |
| Stop hit — per-position SL triggered | 2024-11-19 09:55:00 | 686.65 | 686.56 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-11-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:30:00 | 696.55 | 692.01 | 0.00 | ORB-long ORB[687.75,696.05] vol=2.8x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-11-25 09:45:00 | 694.18 | 693.42 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:55:00 | 678.95 | 674.74 | 0.00 | ORB-long ORB[673.40,678.25] vol=2.3x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-12-09 12:00:00 | 677.23 | 676.36 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:50:00 | 679.45 | 683.94 | 0.00 | ORB-short ORB[686.45,692.25] vol=2.7x ATR=2.05 |
| Stop hit — per-position SL triggered | 2024-12-13 11:00:00 | 681.50 | 683.47 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:20:00 | 675.85 | 678.93 | 0.00 | ORB-short ORB[680.20,684.90] vol=8.3x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:30:00 | 673.38 | 678.81 | 0.00 | T1 1.5R @ 673.38 |
| Stop hit — per-position SL triggered | 2024-12-17 10:50:00 | 675.85 | 678.69 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:30:00 | 665.00 | 665.28 | 0.00 | ORB-short ORB[665.15,669.25] vol=3.9x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-12-27 10:35:00 | 666.83 | 665.40 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:15:00 | 657.45 | 661.31 | 0.00 | ORB-short ORB[659.65,665.90] vol=2.5x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:45:00 | 655.06 | 660.82 | 0.00 | T1 1.5R @ 655.06 |
| Stop hit — per-position SL triggered | 2025-01-08 13:20:00 | 657.45 | 658.56 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:45:00 | 637.35 | 641.37 | 0.00 | ORB-short ORB[642.20,647.45] vol=2.5x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:10:00 | 634.45 | 637.14 | 0.00 | T1 1.5R @ 634.45 |
| Stop hit — per-position SL triggered | 2025-01-10 10:45:00 | 637.35 | 636.84 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:35:00 | 633.70 | 637.26 | 0.00 | ORB-short ORB[635.45,642.80] vol=2.1x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 10:05:00 | 630.52 | 634.92 | 0.00 | T1 1.5R @ 630.52 |
| Stop hit — per-position SL triggered | 2025-01-15 10:15:00 | 633.70 | 634.67 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:15:00 | 639.90 | 643.30 | 0.00 | ORB-short ORB[643.00,651.95] vol=1.8x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-01-21 11:35:00 | 641.74 | 643.08 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:00:00 | 590.80 | 592.66 | 0.00 | ORB-short ORB[592.55,598.80] vol=2.5x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-01-24 10:05:00 | 592.41 | 592.60 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:50:00 | 608.45 | 604.34 | 0.00 | ORB-long ORB[598.95,603.25] vol=1.6x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 606.50 | 607.18 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-02-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:50:00 | 624.50 | 620.25 | 0.00 | ORB-long ORB[613.15,618.00] vol=4.8x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-02-01 11:00:00 | 623.06 | 620.74 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 11:00:00 | 576.25 | 581.69 | 0.00 | ORB-short ORB[582.50,590.80] vol=1.6x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-02-11 11:05:00 | 577.85 | 581.48 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-02-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 10:55:00 | 589.85 | 587.14 | 0.00 | ORB-long ORB[578.85,584.85] vol=1.8x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 11:50:00 | 592.54 | 588.27 | 0.00 | T1 1.5R @ 592.54 |
| Stop hit — per-position SL triggered | 2025-02-13 13:50:00 | 589.85 | 589.92 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-02-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-14 09:55:00 | 597.00 | 594.40 | 0.00 | ORB-long ORB[590.85,595.55] vol=2.3x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-02-14 10:05:00 | 595.05 | 594.66 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 11:15:00 | 575.45 | 575.28 | 0.00 | ORB-long ORB[568.25,575.00] vol=8.7x ATR=1.51 |
| Stop hit — per-position SL triggered | 2025-02-19 11:35:00 | 573.94 | 575.23 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-02-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:05:00 | 579.85 | 576.56 | 0.00 | ORB-long ORB[572.55,578.00] vol=4.8x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-02-20 11:10:00 | 578.17 | 576.94 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-03-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-04 10:25:00 | 546.85 | 549.17 | 0.00 | ORB-short ORB[547.80,553.75] vol=2.6x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 11:15:00 | 543.57 | 547.81 | 0.00 | T1 1.5R @ 543.57 |
| Stop hit — per-position SL triggered | 2025-03-04 11:35:00 | 546.85 | 547.72 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-03-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:25:00 | 552.15 | 551.31 | 0.00 | ORB-long ORB[545.00,551.95] vol=6.1x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-03-05 11:10:00 | 549.94 | 551.38 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-03-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 10:50:00 | 547.40 | 550.19 | 0.00 | ORB-short ORB[548.30,554.70] vol=4.1x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-03-07 11:15:00 | 549.03 | 549.11 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-03-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:10:00 | 536.50 | 541.00 | 0.00 | ORB-short ORB[541.75,547.70] vol=2.3x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-03-12 11:20:00 | 538.14 | 540.51 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-03-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 10:30:00 | 544.40 | 541.66 | 0.00 | ORB-long ORB[537.00,543.90] vol=1.9x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-03-17 10:35:00 | 543.00 | 541.80 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:35:00 | 571.65 | 568.78 | 0.00 | ORB-long ORB[565.00,568.70] vol=1.7x ATR=1.50 |
| Stop hit — per-position SL triggered | 2025-03-20 09:45:00 | 570.15 | 569.49 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 11:15:00 | 589.70 | 593.91 | 0.00 | ORB-short ORB[590.45,599.10] vol=1.8x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 11:55:00 | 587.06 | 592.98 | 0.00 | T1 1.5R @ 587.06 |
| Stop hit — per-position SL triggered | 2025-03-26 12:45:00 | 589.70 | 591.73 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-04-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-01 10:50:00 | 572.45 | 569.63 | 0.00 | ORB-long ORB[560.50,565.95] vol=7.7x ATR=2.28 |
| Stop hit — per-position SL triggered | 2025-04-01 10:55:00 | 570.17 | 569.68 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-04-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-02 09:40:00 | 563.60 | 567.05 | 0.00 | ORB-short ORB[565.25,570.60] vol=1.6x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:45:00 | 560.85 | 566.62 | 0.00 | T1 1.5R @ 560.85 |
| Target hit | 2025-04-02 10:55:00 | 562.55 | 562.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 78 — SELL (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-11 10:15:00 | 567.60 | 570.45 | 0.00 | ORB-short ORB[570.50,576.05] vol=2.9x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-11 10:25:00 | 564.73 | 570.02 | 0.00 | T1 1.5R @ 564.73 |
| Target hit | 2025-04-11 15:20:00 | 551.95 | 561.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2025-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:40:00 | 554.00 | 549.67 | 0.00 | ORB-long ORB[545.30,553.15] vol=1.8x ATR=2.53 |
| Stop hit — per-position SL triggered | 2025-04-15 10:00:00 | 551.47 | 550.87 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:15:00 | 590.65 | 595.81 | 0.00 | ORB-short ORB[597.00,603.35] vol=5.7x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-04-23 10:20:00 | 592.55 | 595.15 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2025-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 11:00:00 | 599.70 | 602.45 | 0.00 | ORB-short ORB[603.10,609.10] vol=1.7x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-04-29 11:15:00 | 601.47 | 602.29 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-04-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:45:00 | 610.65 | 609.20 | 0.00 | ORB-long ORB[601.30,609.50] vol=2.8x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 09:50:00 | 613.73 | 610.03 | 0.00 | T1 1.5R @ 613.73 |
| Stop hit — per-position SL triggered | 2025-04-30 10:10:00 | 610.65 | 610.31 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-05-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-05 10:45:00 | 607.80 | 610.40 | 0.00 | ORB-short ORB[608.85,612.75] vol=2.2x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-05-05 14:05:00 | 609.43 | 608.96 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-21 09:30:00 | 583.80 | 2024-05-21 09:55:00 | 580.59 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-05-21 09:30:00 | 583.80 | 2024-05-21 10:10:00 | 583.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-24 11:05:00 | 586.20 | 2024-05-24 11:10:00 | 584.79 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-27 10:25:00 | 577.85 | 2024-05-27 10:50:00 | 579.71 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-05-28 09:50:00 | 583.70 | 2024-05-28 10:10:00 | 581.41 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-05-29 10:00:00 | 570.25 | 2024-05-29 10:20:00 | 566.99 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-05-29 10:00:00 | 570.25 | 2024-05-29 15:20:00 | 557.45 | TARGET_HIT | 0.50 | 2.24% |
| BUY | retest1 | 2024-06-10 10:30:00 | 578.65 | 2024-06-10 11:40:00 | 577.02 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-06-19 10:35:00 | 601.50 | 2024-06-19 10:40:00 | 603.31 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-25 10:20:00 | 604.70 | 2024-06-25 10:25:00 | 606.64 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-06-26 10:35:00 | 610.00 | 2024-06-26 10:45:00 | 613.40 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-06-26 10:35:00 | 610.00 | 2024-06-26 11:40:00 | 610.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 09:55:00 | 602.95 | 2024-06-27 10:30:00 | 600.74 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-28 11:15:00 | 604.70 | 2024-06-28 11:40:00 | 602.88 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-01 10:55:00 | 613.80 | 2024-07-01 11:30:00 | 612.04 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-04 09:55:00 | 639.65 | 2024-07-04 10:00:00 | 642.58 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-04 09:55:00 | 639.65 | 2024-07-04 10:10:00 | 639.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 10:05:00 | 638.90 | 2024-07-10 10:10:00 | 640.89 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-15 10:05:00 | 654.85 | 2024-07-15 10:15:00 | 657.44 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-15 10:05:00 | 654.85 | 2024-07-15 13:55:00 | 655.50 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2024-07-18 09:40:00 | 645.50 | 2024-07-18 09:45:00 | 648.10 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-29 11:10:00 | 717.50 | 2024-07-29 11:30:00 | 719.53 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-31 10:50:00 | 730.80 | 2024-07-31 10:55:00 | 728.74 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-01 09:45:00 | 741.60 | 2024-08-01 09:55:00 | 738.96 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-02 10:45:00 | 737.25 | 2024-08-02 11:05:00 | 735.02 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-07 10:45:00 | 711.05 | 2024-08-07 11:05:00 | 714.64 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-08-07 10:45:00 | 711.05 | 2024-08-07 12:10:00 | 711.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-08 11:15:00 | 726.50 | 2024-08-08 11:20:00 | 724.18 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-08-12 10:00:00 | 728.75 | 2024-08-12 10:15:00 | 731.21 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-19 10:45:00 | 724.25 | 2024-08-19 10:50:00 | 722.27 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-20 09:30:00 | 728.85 | 2024-08-20 09:35:00 | 726.51 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-21 10:00:00 | 750.90 | 2024-08-21 10:10:00 | 747.35 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-08-27 10:50:00 | 730.95 | 2024-08-27 11:00:00 | 734.66 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-08-27 10:50:00 | 730.95 | 2024-08-27 11:15:00 | 730.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-29 09:30:00 | 749.90 | 2024-08-29 09:35:00 | 752.86 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-29 09:30:00 | 749.90 | 2024-08-29 09:40:00 | 749.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-30 10:40:00 | 747.00 | 2024-08-30 11:00:00 | 750.09 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-30 10:40:00 | 747.00 | 2024-08-30 11:30:00 | 747.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 10:35:00 | 751.35 | 2024-09-06 10:50:00 | 753.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-09 09:45:00 | 761.65 | 2024-09-09 09:50:00 | 758.96 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-09-11 10:25:00 | 757.20 | 2024-09-11 11:20:00 | 754.64 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-13 10:20:00 | 762.70 | 2024-09-13 10:30:00 | 760.58 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-16 11:05:00 | 749.35 | 2024-09-16 11:15:00 | 751.10 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-18 10:40:00 | 759.60 | 2024-09-18 10:50:00 | 757.59 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-23 09:45:00 | 780.40 | 2024-09-23 10:00:00 | 784.67 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-09-23 09:45:00 | 780.40 | 2024-09-23 15:20:00 | 791.00 | TARGET_HIT | 0.50 | 1.36% |
| SELL | retest1 | 2024-09-24 10:50:00 | 779.55 | 2024-09-24 11:05:00 | 776.00 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-09-24 10:50:00 | 779.55 | 2024-09-24 14:50:00 | 779.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-30 11:00:00 | 782.50 | 2024-09-30 12:25:00 | 784.63 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-08 09:30:00 | 734.20 | 2024-10-08 09:45:00 | 737.68 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-10-11 09:35:00 | 738.35 | 2024-10-11 09:55:00 | 740.49 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-15 10:15:00 | 746.40 | 2024-10-15 10:40:00 | 749.54 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-10-15 10:15:00 | 746.40 | 2024-10-15 11:45:00 | 750.30 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2024-10-21 10:20:00 | 753.55 | 2024-10-21 10:35:00 | 751.24 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-10-22 09:30:00 | 759.30 | 2024-10-22 09:35:00 | 756.55 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-10-24 10:30:00 | 762.30 | 2024-10-24 11:30:00 | 758.91 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-10-25 09:45:00 | 762.00 | 2024-10-25 09:55:00 | 756.87 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-10-25 09:45:00 | 762.00 | 2024-10-25 15:20:00 | 744.40 | TARGET_HIT | 0.50 | 2.31% |
| SELL | retest1 | 2024-11-04 10:55:00 | 734.90 | 2024-11-04 11:10:00 | 730.60 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-11-04 10:55:00 | 734.90 | 2024-11-04 11:20:00 | 734.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-05 09:55:00 | 726.20 | 2024-11-05 10:05:00 | 722.12 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-11-05 09:55:00 | 726.20 | 2024-11-05 10:25:00 | 726.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-06 10:55:00 | 717.30 | 2024-11-06 11:35:00 | 720.04 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-11-08 10:55:00 | 717.30 | 2024-11-08 11:25:00 | 715.38 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-11-14 10:35:00 | 697.30 | 2024-11-14 11:00:00 | 695.24 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-11-19 09:40:00 | 686.65 | 2024-11-19 09:45:00 | 683.52 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-11-19 09:40:00 | 686.65 | 2024-11-19 09:55:00 | 686.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 09:30:00 | 696.55 | 2024-11-25 09:45:00 | 694.18 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-09 10:55:00 | 678.95 | 2024-12-09 12:00:00 | 677.23 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-13 10:50:00 | 679.45 | 2024-12-13 11:00:00 | 681.50 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-17 10:20:00 | 675.85 | 2024-12-17 10:30:00 | 673.38 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-17 10:20:00 | 675.85 | 2024-12-17 10:50:00 | 675.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-27 10:30:00 | 665.00 | 2024-12-27 10:35:00 | 666.83 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-08 11:15:00 | 657.45 | 2025-01-08 11:45:00 | 655.06 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-01-08 11:15:00 | 657.45 | 2025-01-08 13:20:00 | 657.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-10 09:45:00 | 637.35 | 2025-01-10 10:10:00 | 634.45 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-01-10 09:45:00 | 637.35 | 2025-01-10 10:45:00 | 637.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-15 09:35:00 | 633.70 | 2025-01-15 10:05:00 | 630.52 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-01-15 09:35:00 | 633.70 | 2025-01-15 10:15:00 | 633.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-21 11:15:00 | 639.90 | 2025-01-21 11:35:00 | 641.74 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-24 10:00:00 | 590.80 | 2025-01-24 10:05:00 | 592.41 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-29 09:50:00 | 608.45 | 2025-01-29 11:20:00 | 606.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-02-01 10:50:00 | 624.50 | 2025-02-01 11:00:00 | 623.06 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-02-11 11:00:00 | 576.25 | 2025-02-11 11:05:00 | 577.85 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-02-13 10:55:00 | 589.85 | 2025-02-13 11:50:00 | 592.54 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-02-13 10:55:00 | 589.85 | 2025-02-13 13:50:00 | 589.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-14 09:55:00 | 597.00 | 2025-02-14 10:05:00 | 595.05 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-02-19 11:15:00 | 575.45 | 2025-02-19 11:35:00 | 573.94 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-02-20 11:05:00 | 579.85 | 2025-02-20 11:10:00 | 578.17 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-03-04 10:25:00 | 546.85 | 2025-03-04 11:15:00 | 543.57 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-03-04 10:25:00 | 546.85 | 2025-03-04 11:35:00 | 546.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-05 10:25:00 | 552.15 | 2025-03-05 11:10:00 | 549.94 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-03-07 10:50:00 | 547.40 | 2025-03-07 11:15:00 | 549.03 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-03-12 11:10:00 | 536.50 | 2025-03-12 11:20:00 | 538.14 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-17 10:30:00 | 544.40 | 2025-03-17 10:35:00 | 543.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-03-20 09:35:00 | 571.65 | 2025-03-20 09:45:00 | 570.15 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-03-26 11:15:00 | 589.70 | 2025-03-26 11:55:00 | 587.06 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-03-26 11:15:00 | 589.70 | 2025-03-26 12:45:00 | 589.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-01 10:50:00 | 572.45 | 2025-04-01 10:55:00 | 570.17 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-04-02 09:40:00 | 563.60 | 2025-04-02 09:45:00 | 560.85 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-04-02 09:40:00 | 563.60 | 2025-04-02 10:55:00 | 562.55 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2025-04-11 10:15:00 | 567.60 | 2025-04-11 10:25:00 | 564.73 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-04-11 10:15:00 | 567.60 | 2025-04-11 15:20:00 | 551.95 | TARGET_HIT | 0.50 | 2.76% |
| BUY | retest1 | 2025-04-15 09:40:00 | 554.00 | 2025-04-15 10:00:00 | 551.47 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-04-23 10:15:00 | 590.65 | 2025-04-23 10:20:00 | 592.55 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-29 11:00:00 | 599.70 | 2025-04-29 11:15:00 | 601.47 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-30 09:45:00 | 610.65 | 2025-04-30 09:50:00 | 613.73 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-30 09:45:00 | 610.65 | 2025-04-30 10:10:00 | 610.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-05 10:45:00 | 607.80 | 2025-05-05 14:05:00 | 609.43 | STOP_HIT | 1.00 | -0.27% |
