# CG Power and Industrial Solutions Ltd. (CGPOWER)

## Backtest Summary

- **Window:** 2024-08-09 09:15:00 → 2026-05-08 15:25:00 (30775 bars)
- **Last close:** 875.10
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
| ENTRY1 | 35 |
| ENTRY2 | 0 |
| PARTIAL | 16 |
| TARGET_HIT | 9 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 26
- **Target hits / Stop hits / Partials:** 9 / 26 / 16
- **Avg / median % per leg:** 0.24% / 0.00%
- **Sum % (uncompounded):** 12.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 15 | 45.5% | 5 | 18 | 10 | 0.24% | 8.1% |
| BUY @ 2nd Alert (retest1) | 33 | 15 | 45.5% | 5 | 18 | 10 | 0.24% | 8.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 10 | 55.6% | 4 | 8 | 6 | 0.25% | 4.4% |
| SELL @ 2nd Alert (retest1) | 18 | 10 | 55.6% | 4 | 8 | 6 | 0.25% | 4.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 51 | 25 | 49.0% | 9 | 26 | 16 | 0.24% | 12.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:50:00 | 723.80 | 718.48 | 0.00 | ORB-long ORB[711.05,718.80] vol=3.2x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:00:00 | 727.78 | 723.34 | 0.00 | T1 1.5R @ 727.78 |
| Target hit | 2024-08-20 12:45:00 | 736.00 | 736.59 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2024-08-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:50:00 | 744.30 | 740.76 | 0.00 | ORB-long ORB[737.85,743.20] vol=2.7x ATR=2.25 |
| Stop hit — per-position SL triggered | 2024-08-22 11:15:00 | 742.05 | 741.17 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 722.55 | 726.51 | 0.00 | ORB-short ORB[725.05,730.85] vol=2.4x ATR=2.28 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 724.83 | 725.84 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 11:15:00 | 704.55 | 699.56 | 0.00 | ORB-long ORB[692.60,701.60] vol=1.8x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 11:20:00 | 708.32 | 700.63 | 0.00 | T1 1.5R @ 708.32 |
| Stop hit — per-position SL triggered | 2024-09-12 11:35:00 | 704.55 | 701.05 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-09-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:40:00 | 725.70 | 723.12 | 0.00 | ORB-long ORB[715.45,725.00] vol=1.7x ATR=3.39 |
| Stop hit — per-position SL triggered | 2024-09-17 10:25:00 | 722.31 | 724.51 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-09-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:55:00 | 735.55 | 739.02 | 0.00 | ORB-short ORB[739.20,747.30] vol=7.0x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:15:00 | 731.26 | 738.58 | 0.00 | T1 1.5R @ 731.26 |
| Target hit | 2024-09-19 15:05:00 | 734.90 | 733.48 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — BUY (started 2024-09-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:50:00 | 786.40 | 778.11 | 0.00 | ORB-long ORB[767.35,779.00] vol=2.1x ATR=4.39 |
| Stop hit — per-position SL triggered | 2024-09-24 10:00:00 | 782.01 | 778.91 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-10-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:35:00 | 816.35 | 808.77 | 0.00 | ORB-long ORB[802.95,810.85] vol=1.8x ATR=4.52 |
| Stop hit — per-position SL triggered | 2024-10-10 09:45:00 | 811.83 | 811.10 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-10-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:05:00 | 710.95 | 715.55 | 0.00 | ORB-short ORB[711.90,720.95] vol=2.0x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 11:35:00 | 706.38 | 714.48 | 0.00 | T1 1.5R @ 706.38 |
| Target hit | 2024-10-31 15:20:00 | 702.60 | 709.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2024-11-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 11:00:00 | 720.75 | 715.94 | 0.00 | ORB-long ORB[709.25,717.95] vol=1.5x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 11:05:00 | 724.45 | 717.28 | 0.00 | T1 1.5R @ 724.45 |
| Target hit | 2024-11-06 14:35:00 | 723.60 | 725.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 699.10 | 704.88 | 0.00 | ORB-short ORB[702.55,710.00] vol=1.6x ATR=3.39 |
| Stop hit — per-position SL triggered | 2024-11-13 09:35:00 | 702.49 | 704.45 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-11-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 11:00:00 | 718.90 | 714.66 | 0.00 | ORB-long ORB[708.05,717.10] vol=1.7x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 11:40:00 | 723.01 | 716.05 | 0.00 | T1 1.5R @ 723.01 |
| Target hit | 2024-11-22 15:20:00 | 729.50 | 725.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2024-11-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:40:00 | 749.00 | 745.18 | 0.00 | ORB-long ORB[738.85,747.45] vol=1.9x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 12:25:00 | 753.94 | 748.36 | 0.00 | T1 1.5R @ 753.94 |
| Stop hit — per-position SL triggered | 2024-11-25 15:00:00 | 749.00 | 753.11 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:30:00 | 748.75 | 744.27 | 0.00 | ORB-long ORB[738.10,747.70] vol=1.8x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 09:50:00 | 754.55 | 746.93 | 0.00 | T1 1.5R @ 754.55 |
| Stop hit — per-position SL triggered | 2024-11-27 09:55:00 | 748.75 | 747.02 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:30:00 | 769.00 | 765.73 | 0.00 | ORB-long ORB[758.85,764.95] vol=5.4x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-12-04 09:40:00 | 767.29 | 767.29 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-12-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 10:35:00 | 781.30 | 788.47 | 0.00 | ORB-short ORB[786.80,796.60] vol=1.9x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-12-10 12:00:00 | 783.67 | 786.15 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-12-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:10:00 | 765.60 | 768.74 | 0.00 | ORB-short ORB[768.00,775.60] vol=7.2x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 11:55:00 | 762.17 | 766.49 | 0.00 | T1 1.5R @ 762.17 |
| Stop hit — per-position SL triggered | 2024-12-16 14:50:00 | 765.60 | 765.26 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-12-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 10:45:00 | 776.45 | 771.96 | 0.00 | ORB-long ORB[768.00,775.50] vol=3.1x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:05:00 | 779.88 | 774.06 | 0.00 | T1 1.5R @ 779.88 |
| Stop hit — per-position SL triggered | 2024-12-17 11:45:00 | 776.45 | 776.18 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-12-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 10:10:00 | 719.55 | 721.90 | 0.00 | ORB-short ORB[720.50,727.20] vol=2.9x ATR=2.28 |
| Stop hit — per-position SL triggered | 2024-12-24 10:15:00 | 721.83 | 721.86 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-12-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 11:10:00 | 749.30 | 746.82 | 0.00 | ORB-long ORB[743.00,749.00] vol=1.7x ATR=2.20 |
| Stop hit — per-position SL triggered | 2024-12-27 11:15:00 | 747.10 | 746.84 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-01-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:50:00 | 749.15 | 746.48 | 0.00 | ORB-long ORB[741.95,748.00] vol=1.9x ATR=2.28 |
| Stop hit — per-position SL triggered | 2025-01-02 10:55:00 | 746.87 | 746.60 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-01-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:05:00 | 728.20 | 735.68 | 0.00 | ORB-short ORB[735.25,740.60] vol=2.1x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:25:00 | 724.47 | 732.82 | 0.00 | T1 1.5R @ 724.47 |
| Target hit | 2025-01-06 15:20:00 | 714.55 | 722.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-01-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:00:00 | 713.15 | 719.99 | 0.00 | ORB-short ORB[718.85,728.65] vol=2.0x ATR=3.58 |
| Stop hit — per-position SL triggered | 2025-01-07 10:30:00 | 716.73 | 717.50 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-01-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:35:00 | 669.45 | 673.31 | 0.00 | ORB-short ORB[670.65,678.00] vol=1.6x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:45:00 | 665.59 | 672.00 | 0.00 | T1 1.5R @ 665.59 |
| Stop hit — per-position SL triggered | 2025-01-10 09:55:00 | 669.45 | 671.21 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-02-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 09:30:00 | 619.60 | 615.73 | 0.00 | ORB-long ORB[610.45,618.45] vol=1.9x ATR=2.61 |
| Stop hit — per-position SL triggered | 2025-02-07 09:35:00 | 616.99 | 616.00 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-02-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:50:00 | 609.75 | 605.72 | 0.00 | ORB-long ORB[597.65,605.95] vol=2.2x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-02-25 10:05:00 | 606.61 | 606.08 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-03-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 10:30:00 | 639.85 | 644.04 | 0.00 | ORB-short ORB[644.05,653.00] vol=2.0x ATR=2.94 |
| Stop hit — per-position SL triggered | 2025-03-24 10:45:00 | 642.79 | 643.11 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:30:00 | 648.90 | 645.37 | 0.00 | ORB-long ORB[638.70,647.85] vol=2.0x ATR=2.86 |
| Stop hit — per-position SL triggered | 2025-03-25 09:50:00 | 646.04 | 647.03 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:35:00 | 595.85 | 591.36 | 0.00 | ORB-long ORB[587.00,594.30] vol=1.5x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 09:45:00 | 599.34 | 594.03 | 0.00 | T1 1.5R @ 599.34 |
| Stop hit — per-position SL triggered | 2025-04-16 10:15:00 | 595.85 | 595.05 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 09:50:00 | 604.55 | 600.57 | 0.00 | ORB-long ORB[596.50,602.90] vol=2.7x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-04-17 09:55:00 | 602.60 | 600.87 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:50:00 | 637.50 | 631.39 | 0.00 | ORB-long ORB[624.55,629.80] vol=1.5x ATR=2.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:55:00 | 641.74 | 633.07 | 0.00 | T1 1.5R @ 641.74 |
| Target hit | 2025-04-21 15:20:00 | 651.15 | 642.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2025-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:45:00 | 653.50 | 650.38 | 0.00 | ORB-long ORB[647.50,652.00] vol=1.6x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-04-23 10:00:00 | 651.25 | 650.89 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 631.35 | 640.22 | 0.00 | ORB-short ORB[640.35,649.35] vol=1.6x ATR=2.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:50:00 | 627.26 | 636.15 | 0.00 | T1 1.5R @ 627.26 |
| Target hit | 2025-04-25 12:00:00 | 629.70 | 626.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2025-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:30:00 | 625.55 | 622.91 | 0.00 | ORB-long ORB[618.05,624.90] vol=1.7x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 09:45:00 | 629.17 | 624.73 | 0.00 | T1 1.5R @ 629.17 |
| Target hit | 2025-05-05 15:20:00 | 636.00 | 632.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2025-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 09:30:00 | 646.45 | 642.85 | 0.00 | ORB-long ORB[637.00,645.80] vol=3.9x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-05-06 09:50:00 | 643.66 | 643.76 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-20 09:50:00 | 723.80 | 2024-08-20 10:00:00 | 727.78 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-08-20 09:50:00 | 723.80 | 2024-08-20 12:45:00 | 736.00 | TARGET_HIT | 0.50 | 1.69% |
| BUY | retest1 | 2024-08-22 10:50:00 | 744.30 | 2024-08-22 11:15:00 | 742.05 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-28 09:30:00 | 722.55 | 2024-08-28 09:35:00 | 724.83 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-12 11:15:00 | 704.55 | 2024-09-12 11:20:00 | 708.32 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-09-12 11:15:00 | 704.55 | 2024-09-12 11:35:00 | 704.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-17 09:40:00 | 725.70 | 2024-09-17 10:25:00 | 722.31 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-09-19 10:55:00 | 735.55 | 2024-09-19 11:15:00 | 731.26 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-19 10:55:00 | 735.55 | 2024-09-19 15:05:00 | 734.90 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2024-09-24 09:50:00 | 786.40 | 2024-09-24 10:00:00 | 782.01 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-10-10 09:35:00 | 816.35 | 2024-10-10 09:45:00 | 811.83 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2024-10-31 11:05:00 | 710.95 | 2024-10-31 11:35:00 | 706.38 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-10-31 11:05:00 | 710.95 | 2024-10-31 15:20:00 | 702.60 | TARGET_HIT | 0.50 | 1.17% |
| BUY | retest1 | 2024-11-06 11:00:00 | 720.75 | 2024-11-06 11:05:00 | 724.45 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-11-06 11:00:00 | 720.75 | 2024-11-06 14:35:00 | 723.60 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-11-13 09:30:00 | 699.10 | 2024-11-13 09:35:00 | 702.49 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-11-22 11:00:00 | 718.90 | 2024-11-22 11:40:00 | 723.01 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-11-22 11:00:00 | 718.90 | 2024-11-22 15:20:00 | 729.50 | TARGET_HIT | 0.50 | 1.47% |
| BUY | retest1 | 2024-11-25 09:40:00 | 749.00 | 2024-11-25 12:25:00 | 753.94 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-11-25 09:40:00 | 749.00 | 2024-11-25 15:00:00 | 749.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:30:00 | 748.75 | 2024-11-27 09:50:00 | 754.55 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-11-27 09:30:00 | 748.75 | 2024-11-27 09:55:00 | 748.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-04 09:30:00 | 769.00 | 2024-12-04 09:40:00 | 767.29 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-10 10:35:00 | 781.30 | 2024-12-10 12:00:00 | 783.67 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-16 10:10:00 | 765.60 | 2024-12-16 11:55:00 | 762.17 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-12-16 10:10:00 | 765.60 | 2024-12-16 14:50:00 | 765.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-17 10:45:00 | 776.45 | 2024-12-17 11:05:00 | 779.88 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-12-17 10:45:00 | 776.45 | 2024-12-17 11:45:00 | 776.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-24 10:10:00 | 719.55 | 2024-12-24 10:15:00 | 721.83 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-27 11:10:00 | 749.30 | 2024-12-27 11:15:00 | 747.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-02 10:50:00 | 749.15 | 2025-01-02 10:55:00 | 746.87 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-06 11:05:00 | 728.20 | 2025-01-06 11:25:00 | 724.47 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-01-06 11:05:00 | 728.20 | 2025-01-06 15:20:00 | 714.55 | TARGET_HIT | 0.50 | 1.87% |
| SELL | retest1 | 2025-01-07 10:00:00 | 713.15 | 2025-01-07 10:30:00 | 716.73 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-01-10 09:35:00 | 669.45 | 2025-01-10 09:45:00 | 665.59 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-01-10 09:35:00 | 669.45 | 2025-01-10 09:55:00 | 669.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 09:30:00 | 619.60 | 2025-02-07 09:35:00 | 616.99 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-02-25 09:50:00 | 609.75 | 2025-02-25 10:05:00 | 606.61 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-03-24 10:30:00 | 639.85 | 2025-03-24 10:45:00 | 642.79 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-03-25 09:30:00 | 648.90 | 2025-03-25 09:50:00 | 646.04 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-04-16 09:35:00 | 595.85 | 2025-04-16 09:45:00 | 599.34 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-04-16 09:35:00 | 595.85 | 2025-04-16 10:15:00 | 595.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-17 09:50:00 | 604.55 | 2025-04-17 09:55:00 | 602.60 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-21 09:50:00 | 637.50 | 2025-04-21 09:55:00 | 641.74 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-04-21 09:50:00 | 637.50 | 2025-04-21 15:20:00 | 651.15 | TARGET_HIT | 0.50 | 2.14% |
| BUY | retest1 | 2025-04-23 09:45:00 | 653.50 | 2025-04-23 10:00:00 | 651.25 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-04-25 09:35:00 | 631.35 | 2025-04-25 09:50:00 | 627.26 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-04-25 09:35:00 | 631.35 | 2025-04-25 12:00:00 | 629.70 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-05-05 09:30:00 | 625.55 | 2025-05-05 09:45:00 | 629.17 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-05-05 09:30:00 | 625.55 | 2025-05-05 15:20:00 | 636.00 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2025-05-06 09:30:00 | 646.45 | 2025-05-06 09:50:00 | 643.66 | STOP_HIT | 1.00 | -0.43% |
