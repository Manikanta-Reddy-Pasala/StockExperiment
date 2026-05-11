# Bikaji Foods International Ltd. (BIKAJI)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 670.20
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
| ENTRY1 | 81 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 18 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 63
- **Target hits / Stop hits / Partials:** 18 / 63 / 32
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 11.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 25 | 41.7% | 8 | 35 | 17 | 0.10% | 5.9% |
| BUY @ 2nd Alert (retest1) | 60 | 25 | 41.7% | 8 | 35 | 17 | 0.10% | 5.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 53 | 25 | 47.2% | 10 | 28 | 15 | 0.11% | 5.9% |
| SELL @ 2nd Alert (retest1) | 53 | 25 | 47.2% | 10 | 28 | 15 | 0.11% | 5.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 113 | 50 | 44.2% | 18 | 63 | 32 | 0.10% | 11.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:40:00 | 702.85 | 697.90 | 0.00 | ORB-long ORB[691.20,699.20] vol=1.6x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-05-14 10:00:00 | 700.17 | 698.72 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 11:00:00 | 730.00 | 727.69 | 0.00 | ORB-long ORB[717.45,726.95] vol=9.5x ATR=3.18 |
| Stop hit — per-position SL triggered | 2025-05-26 11:45:00 | 726.82 | 727.94 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:50:00 | 757.70 | 753.59 | 0.00 | ORB-long ORB[745.25,752.25] vol=5.5x ATR=2.60 |
| Stop hit — per-position SL triggered | 2025-05-29 09:55:00 | 755.10 | 753.94 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 768.50 | 765.37 | 0.00 | ORB-long ORB[760.15,767.00] vol=3.5x ATR=2.13 |
| Stop hit — per-position SL triggered | 2025-06-04 09:35:00 | 766.37 | 765.48 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 10:55:00 | 729.25 | 726.41 | 0.00 | ORB-long ORB[719.55,726.60] vol=1.8x ATR=2.02 |
| Stop hit — per-position SL triggered | 2025-06-18 11:10:00 | 727.23 | 726.45 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 11:05:00 | 709.60 | 713.29 | 0.00 | ORB-short ORB[712.85,719.90] vol=1.8x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 13:30:00 | 706.40 | 708.52 | 0.00 | T1 1.5R @ 706.40 |
| Target hit | 2025-06-19 14:05:00 | 706.80 | 706.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2025-07-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 09:50:00 | 733.55 | 737.42 | 0.00 | ORB-short ORB[737.00,743.45] vol=1.8x ATR=2.13 |
| Stop hit — per-position SL triggered | 2025-07-01 10:10:00 | 735.68 | 736.52 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 734.50 | 737.59 | 0.00 | ORB-short ORB[736.80,745.00] vol=2.8x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-07-02 09:50:00 | 736.29 | 737.21 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:40:00 | 726.45 | 729.88 | 0.00 | ORB-short ORB[729.00,735.25] vol=2.9x ATR=1.82 |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 728.27 | 729.35 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 729.00 | 734.66 | 0.00 | ORB-short ORB[731.20,737.80] vol=1.8x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-07-08 11:15:00 | 730.55 | 734.52 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:30:00 | 740.00 | 736.68 | 0.00 | ORB-long ORB[732.00,738.30] vol=2.9x ATR=2.18 |
| Stop hit — per-position SL triggered | 2025-07-09 09:35:00 | 737.82 | 736.92 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:10:00 | 732.00 | 735.32 | 0.00 | ORB-short ORB[735.00,739.70] vol=3.1x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-07-10 11:20:00 | 733.12 | 735.20 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:50:00 | 735.35 | 734.40 | 0.00 | ORB-long ORB[729.75,734.20] vol=16.3x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-07-11 10:00:00 | 733.57 | 734.39 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:55:00 | 764.90 | 757.58 | 0.00 | ORB-long ORB[746.25,757.25] vol=4.3x ATR=2.91 |
| Stop hit — per-position SL triggered | 2025-07-29 10:15:00 | 761.99 | 760.00 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 10:25:00 | 758.15 | 755.28 | 0.00 | ORB-long ORB[746.55,756.70] vol=3.4x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-08-01 10:35:00 | 755.47 | 755.33 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 09:40:00 | 746.95 | 750.74 | 0.00 | ORB-short ORB[747.25,756.20] vol=2.0x ATR=2.88 |
| Stop hit — per-position SL triggered | 2025-08-04 09:50:00 | 749.83 | 750.54 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 10:35:00 | 719.95 | 720.70 | 0.00 | ORB-short ORB[720.80,726.15] vol=1.7x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:10:00 | 717.43 | 720.35 | 0.00 | T1 1.5R @ 717.43 |
| Stop hit — per-position SL triggered | 2025-08-13 11:30:00 | 719.95 | 720.22 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-08-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:05:00 | 785.40 | 780.73 | 0.00 | ORB-long ORB[772.45,783.90] vol=1.7x ATR=3.29 |
| Stop hit — per-position SL triggered | 2025-08-20 10:20:00 | 782.11 | 781.34 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 10:05:00 | 785.65 | 787.19 | 0.00 | ORB-short ORB[786.20,797.00] vol=3.2x ATR=3.30 |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 788.95 | 786.63 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-09-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 11:05:00 | 778.90 | 773.91 | 0.00 | ORB-long ORB[767.00,777.75] vol=2.5x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-09-10 13:45:00 | 776.25 | 774.87 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 10:55:00 | 787.80 | 781.66 | 0.00 | ORB-long ORB[773.00,778.60] vol=1.6x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 11:25:00 | 790.80 | 784.30 | 0.00 | T1 1.5R @ 790.80 |
| Target hit | 2025-09-15 15:20:00 | 792.90 | 788.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2025-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 09:35:00 | 780.80 | 784.38 | 0.00 | ORB-short ORB[784.00,790.00] vol=4.0x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-09-18 10:10:00 | 782.64 | 783.31 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-10-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:05:00 | 725.20 | 727.48 | 0.00 | ORB-short ORB[725.40,730.50] vol=1.8x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 10:40:00 | 722.47 | 726.07 | 0.00 | T1 1.5R @ 722.47 |
| Stop hit — per-position SL triggered | 2025-10-07 11:05:00 | 725.20 | 725.72 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-10-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 10:50:00 | 728.05 | 725.18 | 0.00 | ORB-long ORB[720.70,727.85] vol=1.9x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-10-13 10:55:00 | 726.01 | 725.23 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-10-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:50:00 | 739.50 | 737.56 | 0.00 | ORB-long ORB[731.75,738.20] vol=1.6x ATR=2.31 |
| Stop hit — per-position SL triggered | 2025-10-16 09:55:00 | 737.19 | 737.57 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-10-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 09:50:00 | 731.10 | 732.24 | 0.00 | ORB-short ORB[732.00,735.90] vol=2.5x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-10-20 09:55:00 | 732.80 | 732.23 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-10-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 09:55:00 | 731.20 | 732.41 | 0.00 | ORB-short ORB[732.00,737.00] vol=5.3x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 10:05:00 | 728.31 | 732.13 | 0.00 | T1 1.5R @ 728.31 |
| Target hit | 2025-10-23 15:05:00 | 729.70 | 729.68 | 0.00 | Trail-exit close>VWAP |

### Cycle 28 — BUY (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 10:15:00 | 731.30 | 728.40 | 0.00 | ORB-long ORB[724.55,728.95] vol=1.7x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 10:20:00 | 734.35 | 730.39 | 0.00 | T1 1.5R @ 734.35 |
| Stop hit — per-position SL triggered | 2025-10-24 10:30:00 | 731.30 | 730.49 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-10-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 11:05:00 | 727.00 | 729.33 | 0.00 | ORB-short ORB[727.10,736.45] vol=4.9x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-10-27 11:20:00 | 728.58 | 729.20 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:30:00 | 727.50 | 728.60 | 0.00 | ORB-short ORB[727.65,733.80] vol=1.5x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-10-28 10:45:00 | 728.70 | 728.61 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-10-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 09:40:00 | 733.90 | 731.93 | 0.00 | ORB-long ORB[729.65,733.35] vol=2.0x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-10-30 09:45:00 | 732.49 | 731.97 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:35:00 | 730.30 | 732.04 | 0.00 | ORB-short ORB[731.05,737.80] vol=1.6x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:45:00 | 728.39 | 731.70 | 0.00 | T1 1.5R @ 728.39 |
| Stop hit — per-position SL triggered | 2025-10-31 11:25:00 | 730.30 | 730.69 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:15:00 | 726.15 | 728.64 | 0.00 | ORB-short ORB[726.60,731.60] vol=2.1x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:55:00 | 723.32 | 727.57 | 0.00 | T1 1.5R @ 723.32 |
| Stop hit — per-position SL triggered | 2025-11-06 12:05:00 | 726.15 | 727.51 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:35:00 | 714.50 | 717.17 | 0.00 | ORB-short ORB[715.60,722.20] vol=1.6x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-11-07 09:50:00 | 716.72 | 716.72 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-11-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 10:10:00 | 713.55 | 709.29 | 0.00 | ORB-long ORB[703.50,710.00] vol=4.4x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 11:30:00 | 719.17 | 712.20 | 0.00 | T1 1.5R @ 719.17 |
| Target hit | 2025-11-11 15:20:00 | 723.30 | 721.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2025-11-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 10:35:00 | 710.35 | 715.34 | 0.00 | ORB-short ORB[714.55,725.00] vol=3.9x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 14:00:00 | 707.45 | 713.02 | 0.00 | T1 1.5R @ 707.45 |
| Target hit | 2025-11-13 15:20:00 | 708.30 | 710.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2025-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 09:30:00 | 711.50 | 713.35 | 0.00 | ORB-short ORB[712.05,720.00] vol=2.2x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-11-19 09:40:00 | 713.71 | 713.27 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-11-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 09:30:00 | 725.55 | 722.61 | 0.00 | ORB-long ORB[719.35,723.30] vol=2.6x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 09:40:00 | 728.33 | 723.85 | 0.00 | T1 1.5R @ 728.33 |
| Stop hit — per-position SL triggered | 2025-11-20 10:35:00 | 725.55 | 725.08 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:40:00 | 712.45 | 715.34 | 0.00 | ORB-short ORB[715.15,721.65] vol=1.8x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-11-21 09:45:00 | 714.49 | 715.17 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-12-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 11:00:00 | 711.15 | 705.16 | 0.00 | ORB-long ORB[694.00,701.90] vol=9.4x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 12:40:00 | 714.14 | 708.03 | 0.00 | T1 1.5R @ 714.14 |
| Stop hit — per-position SL triggered | 2025-12-04 15:05:00 | 711.15 | 712.21 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-12-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:40:00 | 718.10 | 712.60 | 0.00 | ORB-long ORB[707.00,714.75] vol=1.8x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-12-05 11:30:00 | 716.21 | 714.55 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-12-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:00:00 | 712.50 | 713.56 | 0.00 | ORB-short ORB[713.25,717.95] vol=1.7x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:20:00 | 709.55 | 712.81 | 0.00 | T1 1.5R @ 709.55 |
| Target hit | 2025-12-08 11:05:00 | 710.65 | 710.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — BUY (started 2025-12-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:50:00 | 730.10 | 725.86 | 0.00 | ORB-long ORB[720.00,726.75] vol=4.6x ATR=1.59 |
| Stop hit — per-position SL triggered | 2025-12-12 11:20:00 | 728.51 | 726.09 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:15:00 | 733.20 | 730.35 | 0.00 | ORB-long ORB[725.55,732.85] vol=1.6x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 10:30:00 | 736.09 | 733.61 | 0.00 | T1 1.5R @ 736.09 |
| Target hit | 2025-12-15 15:20:00 | 740.40 | 739.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2025-12-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 10:10:00 | 744.80 | 738.38 | 0.00 | ORB-long ORB[731.80,739.50] vol=2.0x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 10:35:00 | 748.07 | 741.40 | 0.00 | T1 1.5R @ 748.07 |
| Target hit | 2025-12-16 12:05:00 | 747.00 | 748.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — BUY (started 2025-12-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:55:00 | 746.00 | 742.56 | 0.00 | ORB-long ORB[738.00,745.10] vol=2.4x ATR=2.13 |
| Stop hit — per-position SL triggered | 2025-12-17 10:45:00 | 743.87 | 743.16 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:30:00 | 741.80 | 737.66 | 0.00 | ORB-long ORB[735.55,740.45] vol=2.2x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 10:40:00 | 744.96 | 738.48 | 0.00 | T1 1.5R @ 744.96 |
| Stop hit — per-position SL triggered | 2025-12-18 11:40:00 | 741.80 | 741.37 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:30:00 | 745.15 | 742.78 | 0.00 | ORB-long ORB[740.05,744.90] vol=1.7x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-12-19 09:35:00 | 743.48 | 743.16 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:35:00 | 746.20 | 741.82 | 0.00 | ORB-long ORB[735.00,741.05] vol=2.5x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 12:20:00 | 750.13 | 745.86 | 0.00 | T1 1.5R @ 750.13 |
| Stop hit — per-position SL triggered | 2025-12-26 15:15:00 | 746.20 | 747.65 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-31 10:00:00 | 751.45 | 754.70 | 0.00 | ORB-short ORB[755.45,759.40] vol=3.3x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-12-31 10:05:00 | 753.10 | 754.20 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-01-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 09:50:00 | 730.20 | 732.94 | 0.00 | ORB-short ORB[732.50,739.95] vol=3.9x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 10:00:00 | 726.40 | 731.94 | 0.00 | T1 1.5R @ 726.40 |
| Target hit | 2026-01-05 15:15:00 | 722.50 | 721.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 52 — BUY (started 2026-01-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:30:00 | 720.00 | 716.03 | 0.00 | ORB-long ORB[713.20,719.00] vol=1.5x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 10:35:00 | 722.73 | 716.80 | 0.00 | T1 1.5R @ 722.73 |
| Stop hit — per-position SL triggered | 2026-01-07 10:45:00 | 720.00 | 717.04 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2026-01-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:50:00 | 718.00 | 721.70 | 0.00 | ORB-short ORB[720.10,725.10] vol=1.8x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:10:00 | 715.66 | 721.29 | 0.00 | T1 1.5R @ 715.66 |
| Target hit | 2026-01-08 15:20:00 | 710.70 | 716.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2026-01-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 09:55:00 | 705.65 | 701.11 | 0.00 | ORB-long ORB[695.00,703.40] vol=2.2x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:05:00 | 709.47 | 702.88 | 0.00 | T1 1.5R @ 709.47 |
| Target hit | 2026-01-12 10:20:00 | 709.70 | 709.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 55 — BUY (started 2026-01-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 09:45:00 | 704.30 | 700.80 | 0.00 | ORB-long ORB[698.50,703.20] vol=1.6x ATR=1.70 |
| Stop hit — per-position SL triggered | 2026-01-16 10:00:00 | 702.60 | 701.13 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 11:15:00 | 692.95 | 697.80 | 0.00 | ORB-short ORB[696.70,700.60] vol=1.6x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 11:55:00 | 690.36 | 696.70 | 0.00 | T1 1.5R @ 690.36 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 692.95 | 694.46 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 11:15:00 | 670.05 | 667.71 | 0.00 | ORB-long ORB[662.85,668.00] vol=1.8x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 13:45:00 | 672.39 | 668.62 | 0.00 | T1 1.5R @ 672.39 |
| Target hit | 2026-01-22 15:20:00 | 673.40 | 669.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2026-01-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 10:55:00 | 659.65 | 663.82 | 0.00 | ORB-short ORB[664.70,672.75] vol=6.0x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:35:00 | 657.08 | 662.68 | 0.00 | T1 1.5R @ 657.08 |
| Target hit | 2026-01-23 15:20:00 | 650.50 | 654.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 663.05 | 656.66 | 0.00 | ORB-long ORB[649.75,656.35] vol=4.6x ATR=2.06 |
| Stop hit — per-position SL triggered | 2026-02-09 11:00:00 | 660.99 | 657.88 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-02-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:10:00 | 668.80 | 671.38 | 0.00 | ORB-short ORB[669.50,674.30] vol=1.9x ATR=2.20 |
| Stop hit — per-position SL triggered | 2026-02-10 10:30:00 | 671.00 | 670.97 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 629.40 | 632.88 | 0.00 | ORB-short ORB[631.00,638.60] vol=2.2x ATR=1.71 |
| Stop hit — per-position SL triggered | 2026-02-23 11:05:00 | 631.11 | 632.69 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:50:00 | 635.00 | 630.05 | 0.00 | ORB-long ORB[625.60,630.00] vol=3.5x ATR=2.03 |
| Stop hit — per-position SL triggered | 2026-02-27 10:00:00 | 632.97 | 630.54 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:35:00 | 620.25 | 622.14 | 0.00 | ORB-short ORB[622.00,629.25] vol=2.4x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:35:00 | 616.89 | 620.00 | 0.00 | T1 1.5R @ 616.89 |
| Target hit | 2026-03-06 12:15:00 | 616.30 | 615.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — SELL (started 2026-03-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-09 09:50:00 | 595.50 | 600.34 | 0.00 | ORB-short ORB[599.35,608.10] vol=1.6x ATR=2.84 |
| Stop hit — per-position SL triggered | 2026-03-09 09:55:00 | 598.34 | 600.09 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:15:00 | 613.15 | 616.90 | 0.00 | ORB-short ORB[615.35,623.60] vol=2.3x ATR=2.06 |
| Stop hit — per-position SL triggered | 2026-03-11 11:10:00 | 615.21 | 616.11 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-03-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:30:00 | 615.10 | 621.21 | 0.00 | ORB-short ORB[620.05,627.25] vol=2.8x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-03-17 10:55:00 | 617.53 | 620.26 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 621.75 | 619.85 | 0.00 | ORB-long ORB[616.05,621.20] vol=1.6x ATR=2.24 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 619.51 | 620.22 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-03-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:50:00 | 626.15 | 622.28 | 0.00 | ORB-long ORB[616.10,622.00] vol=3.4x ATR=2.63 |
| Stop hit — per-position SL triggered | 2026-03-25 14:05:00 | 623.52 | 624.60 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 11:15:00 | 644.00 | 641.24 | 0.00 | ORB-long ORB[636.75,641.00] vol=3.6x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:30:00 | 646.09 | 642.43 | 0.00 | T1 1.5R @ 646.09 |
| Stop hit — per-position SL triggered | 2026-04-09 14:35:00 | 644.00 | 644.35 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 646.85 | 645.21 | 0.00 | ORB-long ORB[643.00,646.00] vol=2.4x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:30:00 | 648.79 | 646.20 | 0.00 | T1 1.5R @ 648.79 |
| Target hit | 2026-04-17 15:20:00 | 660.00 | 656.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 656.80 | 656.55 | 0.00 | ORB-long ORB[651.55,655.00] vol=14.4x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:30:00 | 660.17 | 658.72 | 0.00 | T1 1.5R @ 660.17 |
| Target hit | 2026-04-21 11:00:00 | 657.95 | 658.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — BUY (started 2026-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:40:00 | 672.10 | 670.26 | 0.00 | ORB-long ORB[664.00,668.00] vol=2.5x ATR=2.47 |
| Stop hit — per-position SL triggered | 2026-04-22 09:45:00 | 669.63 | 671.84 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:00:00 | 694.55 | 688.31 | 0.00 | ORB-long ORB[683.15,689.85] vol=2.6x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-04-23 10:05:00 | 692.12 | 689.23 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-04-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:35:00 | 677.80 | 679.70 | 0.00 | ORB-short ORB[680.05,689.75] vol=9.9x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:10:00 | 674.12 | 679.50 | 0.00 | T1 1.5R @ 674.12 |
| Stop hit — per-position SL triggered | 2026-04-24 11:30:00 | 677.80 | 679.27 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 673.00 | 667.32 | 0.00 | ORB-long ORB[662.40,667.10] vol=2.5x ATR=2.37 |
| Stop hit — per-position SL triggered | 2026-04-27 10:45:00 | 670.63 | 669.79 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 673.65 | 675.58 | 0.00 | ORB-short ORB[675.45,680.50] vol=3.1x ATR=1.57 |
| Stop hit — per-position SL triggered | 2026-04-28 09:45:00 | 675.22 | 676.89 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:50:00 | 685.50 | 683.90 | 0.00 | ORB-long ORB[678.15,685.05] vol=1.6x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:05:00 | 688.24 | 685.01 | 0.00 | T1 1.5R @ 688.24 |
| Stop hit — per-position SL triggered | 2026-04-29 12:00:00 | 685.50 | 685.67 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:15:00 | 683.15 | 680.43 | 0.00 | ORB-long ORB[676.60,683.00] vol=2.3x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 13:10:00 | 686.91 | 682.84 | 0.00 | T1 1.5R @ 686.91 |
| Stop hit — per-position SL triggered | 2026-05-04 14:50:00 | 683.15 | 683.91 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:15:00 | 681.30 | 681.88 | 0.00 | ORB-short ORB[682.00,687.60] vol=6.3x ATR=1.70 |
| Target hit | 2026-05-05 15:20:00 | 680.25 | 680.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — SELL (started 2026-05-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:50:00 | 678.55 | 681.75 | 0.00 | ORB-short ORB[681.35,686.45] vol=3.1x ATR=1.40 |
| Stop hit — per-position SL triggered | 2026-05-06 10:55:00 | 679.95 | 681.67 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 679.20 | 682.58 | 0.00 | ORB-short ORB[682.70,687.60] vol=3.5x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:00:00 | 676.30 | 680.39 | 0.00 | T1 1.5R @ 676.30 |
| Target hit | 2026-05-07 15:20:00 | 675.40 | 675.74 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:40:00 | 702.85 | 2025-05-14 10:00:00 | 700.17 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-05-26 11:00:00 | 730.00 | 2025-05-26 11:45:00 | 726.82 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-05-29 09:50:00 | 757.70 | 2025-05-29 09:55:00 | 755.10 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-06-04 09:30:00 | 768.50 | 2025-06-04 09:35:00 | 766.37 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-18 10:55:00 | 729.25 | 2025-06-18 11:10:00 | 727.23 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-19 11:05:00 | 709.60 | 2025-06-19 13:30:00 | 706.40 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-06-19 11:05:00 | 709.60 | 2025-06-19 14:05:00 | 706.80 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-01 09:50:00 | 733.55 | 2025-07-01 10:10:00 | 735.68 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-02 09:40:00 | 734.50 | 2025-07-02 09:50:00 | 736.29 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-07 10:40:00 | 726.45 | 2025-07-07 11:15:00 | 728.27 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-08 11:05:00 | 729.00 | 2025-07-08 11:15:00 | 730.55 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-09 09:30:00 | 740.00 | 2025-07-09 09:35:00 | 737.82 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-10 11:10:00 | 732.00 | 2025-07-10 11:20:00 | 733.12 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-07-11 09:50:00 | 735.35 | 2025-07-11 10:00:00 | 733.57 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-29 09:55:00 | 764.90 | 2025-07-29 10:15:00 | 761.99 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-08-01 10:25:00 | 758.15 | 2025-08-01 10:35:00 | 755.47 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-08-04 09:40:00 | 746.95 | 2025-08-04 09:50:00 | 749.83 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-08-13 10:35:00 | 719.95 | 2025-08-13 11:10:00 | 717.43 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-08-13 10:35:00 | 719.95 | 2025-08-13 11:30:00 | 719.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-20 10:05:00 | 785.40 | 2025-08-20 10:20:00 | 782.11 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-08-21 10:05:00 | 785.65 | 2025-08-21 11:15:00 | 788.95 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-09-10 11:05:00 | 778.90 | 2025-09-10 13:45:00 | 776.25 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-09-15 10:55:00 | 787.80 | 2025-09-15 11:25:00 | 790.80 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-09-15 10:55:00 | 787.80 | 2025-09-15 15:20:00 | 792.90 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2025-09-18 09:35:00 | 780.80 | 2025-09-18 10:10:00 | 782.64 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-07 10:05:00 | 725.20 | 2025-10-07 10:40:00 | 722.47 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-07 10:05:00 | 725.20 | 2025-10-07 11:05:00 | 725.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-13 10:50:00 | 728.05 | 2025-10-13 10:55:00 | 726.01 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-16 09:50:00 | 739.50 | 2025-10-16 09:55:00 | 737.19 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-10-20 09:50:00 | 731.10 | 2025-10-20 09:55:00 | 732.80 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-23 09:55:00 | 731.20 | 2025-10-23 10:05:00 | 728.31 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-23 09:55:00 | 731.20 | 2025-10-23 15:05:00 | 729.70 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2025-10-24 10:15:00 | 731.30 | 2025-10-24 10:20:00 | 734.35 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-10-24 10:15:00 | 731.30 | 2025-10-24 10:30:00 | 731.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-27 11:05:00 | 727.00 | 2025-10-27 11:20:00 | 728.58 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-28 10:30:00 | 727.50 | 2025-10-28 10:45:00 | 728.70 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-10-30 09:40:00 | 733.90 | 2025-10-30 09:45:00 | 732.49 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-10-31 10:35:00 | 730.30 | 2025-10-31 10:45:00 | 728.39 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-10-31 10:35:00 | 730.30 | 2025-10-31 11:25:00 | 730.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-06 10:15:00 | 726.15 | 2025-11-06 11:55:00 | 723.32 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-11-06 10:15:00 | 726.15 | 2025-11-06 12:05:00 | 726.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-07 09:35:00 | 714.50 | 2025-11-07 09:50:00 | 716.72 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-11 10:10:00 | 713.55 | 2025-11-11 11:30:00 | 719.17 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2025-11-11 10:10:00 | 713.55 | 2025-11-11 15:20:00 | 723.30 | TARGET_HIT | 0.50 | 1.37% |
| SELL | retest1 | 2025-11-13 10:35:00 | 710.35 | 2025-11-13 14:00:00 | 707.45 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-11-13 10:35:00 | 710.35 | 2025-11-13 15:20:00 | 708.30 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2025-11-19 09:30:00 | 711.50 | 2025-11-19 09:40:00 | 713.71 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-20 09:30:00 | 725.55 | 2025-11-20 09:40:00 | 728.33 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-11-20 09:30:00 | 725.55 | 2025-11-20 10:35:00 | 725.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 09:40:00 | 712.45 | 2025-11-21 09:45:00 | 714.49 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-04 11:00:00 | 711.15 | 2025-12-04 12:40:00 | 714.14 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-12-04 11:00:00 | 711.15 | 2025-12-04 15:05:00 | 711.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-05 10:40:00 | 718.10 | 2025-12-05 11:30:00 | 716.21 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-08 10:00:00 | 712.50 | 2025-12-08 10:20:00 | 709.55 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-12-08 10:00:00 | 712.50 | 2025-12-08 11:05:00 | 710.65 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-12-12 10:50:00 | 730.10 | 2025-12-12 11:20:00 | 728.51 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-15 10:15:00 | 733.20 | 2025-12-15 10:30:00 | 736.09 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-12-15 10:15:00 | 733.20 | 2025-12-15 15:20:00 | 740.40 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2025-12-16 10:10:00 | 744.80 | 2025-12-16 10:35:00 | 748.07 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-12-16 10:10:00 | 744.80 | 2025-12-16 12:05:00 | 747.00 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-17 09:55:00 | 746.00 | 2025-12-17 10:45:00 | 743.87 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-18 10:30:00 | 741.80 | 2025-12-18 10:40:00 | 744.96 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-12-18 10:30:00 | 741.80 | 2025-12-18 11:40:00 | 741.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-19 09:30:00 | 745.15 | 2025-12-19 09:35:00 | 743.48 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-26 09:35:00 | 746.20 | 2025-12-26 12:20:00 | 750.13 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-12-26 09:35:00 | 746.20 | 2025-12-26 15:15:00 | 746.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-31 10:00:00 | 751.45 | 2025-12-31 10:05:00 | 753.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-05 09:50:00 | 730.20 | 2026-01-05 10:00:00 | 726.40 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-01-05 09:50:00 | 730.20 | 2026-01-05 15:15:00 | 722.50 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2026-01-07 10:30:00 | 720.00 | 2026-01-07 10:35:00 | 722.73 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-01-07 10:30:00 | 720.00 | 2026-01-07 10:45:00 | 720.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 10:50:00 | 718.00 | 2026-01-08 11:10:00 | 715.66 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-08 10:50:00 | 718.00 | 2026-01-08 15:20:00 | 710.70 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2026-01-12 09:55:00 | 705.65 | 2026-01-12 10:05:00 | 709.47 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-01-12 09:55:00 | 705.65 | 2026-01-12 10:20:00 | 709.70 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2026-01-16 09:45:00 | 704.30 | 2026-01-16 10:00:00 | 702.60 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-01-19 11:15:00 | 692.95 | 2026-01-19 11:55:00 | 690.36 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-01-19 11:15:00 | 692.95 | 2026-01-19 14:15:00 | 692.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-22 11:15:00 | 670.05 | 2026-01-22 13:45:00 | 672.39 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-01-22 11:15:00 | 670.05 | 2026-01-22 15:20:00 | 673.40 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2026-01-23 10:55:00 | 659.65 | 2026-01-23 11:35:00 | 657.08 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-01-23 10:55:00 | 659.65 | 2026-01-23 15:20:00 | 650.50 | TARGET_HIT | 0.50 | 1.39% |
| BUY | retest1 | 2026-02-09 10:40:00 | 663.05 | 2026-02-09 11:00:00 | 660.99 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-10 10:10:00 | 668.80 | 2026-02-10 10:30:00 | 671.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-23 10:55:00 | 629.40 | 2026-02-23 11:05:00 | 631.11 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-27 09:50:00 | 635.00 | 2026-02-27 10:00:00 | 632.97 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-06 09:35:00 | 620.25 | 2026-03-06 10:35:00 | 616.89 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-06 09:35:00 | 620.25 | 2026-03-06 12:15:00 | 616.30 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2026-03-09 09:50:00 | 595.50 | 2026-03-09 09:55:00 | 598.34 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-03-11 10:15:00 | 613.15 | 2026-03-11 11:10:00 | 615.21 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-17 10:30:00 | 615.10 | 2026-03-17 10:55:00 | 617.53 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-03-18 09:30:00 | 621.75 | 2026-03-18 09:55:00 | 619.51 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-25 09:50:00 | 626.15 | 2026-03-25 14:05:00 | 623.52 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-09 11:15:00 | 644.00 | 2026-04-09 11:30:00 | 646.09 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-04-09 11:15:00 | 644.00 | 2026-04-09 14:35:00 | 644.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:05:00 | 646.85 | 2026-04-17 10:30:00 | 648.79 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-04-17 10:05:00 | 646.85 | 2026-04-17 15:20:00 | 660.00 | TARGET_HIT | 0.50 | 2.03% |
| BUY | retest1 | 2026-04-21 10:10:00 | 656.80 | 2026-04-21 10:30:00 | 660.17 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-21 10:10:00 | 656.80 | 2026-04-21 11:00:00 | 657.95 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2026-04-22 09:40:00 | 672.10 | 2026-04-22 09:45:00 | 669.63 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-23 10:00:00 | 694.55 | 2026-04-23 10:05:00 | 692.12 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-24 10:35:00 | 677.80 | 2026-04-24 11:10:00 | 674.12 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-04-24 10:35:00 | 677.80 | 2026-04-24 11:30:00 | 677.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:40:00 | 673.00 | 2026-04-27 10:45:00 | 670.63 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-28 09:30:00 | 673.65 | 2026-04-28 09:45:00 | 675.22 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-29 10:50:00 | 685.50 | 2026-04-29 11:05:00 | 688.24 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-29 10:50:00 | 685.50 | 2026-04-29 12:00:00 | 685.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 10:15:00 | 683.15 | 2026-05-04 13:10:00 | 686.91 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-05-04 10:15:00 | 683.15 | 2026-05-04 14:50:00 | 683.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 11:15:00 | 681.30 | 2026-05-05 15:20:00 | 680.25 | TARGET_HIT | 1.00 | 0.15% |
| SELL | retest1 | 2026-05-06 10:50:00 | 678.55 | 2026-05-06 10:55:00 | 679.95 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-05-07 09:35:00 | 679.20 | 2026-05-07 10:00:00 | 676.30 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-05-07 09:35:00 | 679.20 | 2026-05-07 15:20:00 | 675.40 | TARGET_HIT | 0.50 | 0.56% |
