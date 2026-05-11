# Action Construction Equipment Ltd. (ACE)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (52193 bars)
- **Last close:** 949.90
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
| ENTRY1 | 46 |
| ENTRY2 | 0 |
| PARTIAL | 16 |
| TARGET_HIT | 9 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 37
- **Target hits / Stop hits / Partials:** 9 / 37 / 16
- **Avg / median % per leg:** 0.24% / 0.00%
- **Sum % (uncompounded):** 14.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 10 | 43.5% | 5 | 13 | 5 | 0.49% | 11.2% |
| BUY @ 2nd Alert (retest1) | 23 | 10 | 43.5% | 5 | 13 | 5 | 0.49% | 11.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 15 | 38.5% | 4 | 24 | 11 | 0.10% | 3.7% |
| SELL @ 2nd Alert (retest1) | 39 | 15 | 38.5% | 4 | 24 | 11 | 0.10% | 3.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 62 | 25 | 40.3% | 9 | 37 | 16 | 0.24% | 14.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-17 10:05:00 | 460.40 | 458.55 | 0.00 | ORB-long ORB[454.65,459.95] vol=1.6x ATR=2.02 |
| Stop hit — per-position SL triggered | 2023-05-17 10:25:00 | 458.38 | 458.71 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:45:00 | 451.75 | 454.28 | 0.00 | ORB-short ORB[454.05,459.50] vol=1.9x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 10:00:00 | 448.55 | 453.62 | 0.00 | T1 1.5R @ 448.55 |
| Stop hit — per-position SL triggered | 2023-05-19 10:20:00 | 451.75 | 452.86 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-24 10:15:00 | 457.90 | 459.48 | 0.00 | ORB-short ORB[457.95,462.75] vol=1.6x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-24 11:30:00 | 455.63 | 458.75 | 0.00 | T1 1.5R @ 455.63 |
| Stop hit — per-position SL triggered | 2023-05-24 12:00:00 | 457.90 | 458.59 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-06-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:35:00 | 490.85 | 484.84 | 0.00 | ORB-long ORB[475.95,482.30] vol=8.6x ATR=2.33 |
| Stop hit — per-position SL triggered | 2023-06-02 09:40:00 | 488.52 | 486.48 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 09:55:00 | 483.00 | 485.14 | 0.00 | ORB-short ORB[483.75,489.20] vol=2.2x ATR=2.01 |
| Stop hit — per-position SL triggered | 2023-06-06 11:25:00 | 485.01 | 484.48 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 09:35:00 | 491.50 | 492.40 | 0.00 | ORB-short ORB[492.10,495.40] vol=1.7x ATR=1.74 |
| Stop hit — per-position SL triggered | 2023-06-08 09:40:00 | 493.24 | 492.42 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:30:00 | 484.40 | 487.67 | 0.00 | ORB-short ORB[485.45,489.90] vol=2.1x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 09:35:00 | 481.74 | 486.54 | 0.00 | T1 1.5R @ 481.74 |
| Stop hit — per-position SL triggered | 2023-06-09 09:40:00 | 484.40 | 486.24 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 10:00:00 | 493.65 | 491.37 | 0.00 | ORB-long ORB[488.90,492.80] vol=1.7x ATR=1.44 |
| Stop hit — per-position SL triggered | 2023-06-14 10:30:00 | 492.21 | 491.69 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 10:00:00 | 477.65 | 479.77 | 0.00 | ORB-short ORB[479.00,483.60] vol=1.6x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 11:30:00 | 475.46 | 478.36 | 0.00 | T1 1.5R @ 475.46 |
| Target hit | 2023-06-22 13:35:00 | 475.00 | 474.90 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — BUY (started 2023-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:30:00 | 458.10 | 456.24 | 0.00 | ORB-long ORB[453.05,457.05] vol=2.9x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 09:35:00 | 460.74 | 456.83 | 0.00 | T1 1.5R @ 460.74 |
| Target hit | 2023-06-26 15:20:00 | 470.05 | 465.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2023-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 09:30:00 | 477.55 | 476.87 | 0.00 | ORB-long ORB[472.85,476.85] vol=7.3x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-27 09:40:00 | 480.21 | 477.89 | 0.00 | T1 1.5R @ 480.21 |
| Target hit | 2023-06-27 15:20:00 | 498.85 | 489.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2023-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 09:40:00 | 488.00 | 491.19 | 0.00 | ORB-short ORB[489.85,496.65] vol=2.4x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 09:50:00 | 485.60 | 490.14 | 0.00 | T1 1.5R @ 485.60 |
| Stop hit — per-position SL triggered | 2023-07-04 11:55:00 | 488.00 | 487.95 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-07-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 11:10:00 | 617.80 | 609.08 | 0.00 | ORB-long ORB[603.05,611.15] vol=3.2x ATR=2.56 |
| Stop hit — per-position SL triggered | 2023-07-17 11:15:00 | 615.24 | 609.37 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 09:40:00 | 701.70 | 686.87 | 0.00 | ORB-long ORB[672.00,679.70] vol=4.0x ATR=4.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 09:55:00 | 709.07 | 699.26 | 0.00 | T1 1.5R @ 709.07 |
| Target hit | 2023-07-26 15:20:00 | 724.45 | 717.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2023-07-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-31 11:10:00 | 692.70 | 694.56 | 0.00 | ORB-short ORB[694.65,699.70] vol=10.4x ATR=1.91 |
| Stop hit — per-position SL triggered | 2023-07-31 11:45:00 | 694.61 | 694.50 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-08-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 10:25:00 | 710.35 | 723.31 | 0.00 | ORB-short ORB[721.50,729.90] vol=1.5x ATR=3.50 |
| Stop hit — per-position SL triggered | 2023-08-01 13:20:00 | 713.85 | 717.73 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-08-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:25:00 | 766.25 | 794.01 | 0.00 | ORB-short ORB[795.00,802.00] vol=1.9x ATR=5.32 |
| Stop hit — per-position SL triggered | 2023-08-10 10:30:00 | 771.57 | 793.06 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-08-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 10:40:00 | 769.10 | 779.45 | 0.00 | ORB-short ORB[778.65,785.00] vol=2.2x ATR=2.57 |
| Stop hit — per-position SL triggered | 2023-08-24 10:50:00 | 771.67 | 777.80 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-08-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:20:00 | 771.30 | 776.27 | 0.00 | ORB-short ORB[775.00,780.00] vol=1.5x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 10:25:00 | 767.75 | 774.83 | 0.00 | T1 1.5R @ 767.75 |
| Target hit | 2023-08-25 10:45:00 | 758.00 | 757.73 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — BUY (started 2023-08-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 09:40:00 | 778.50 | 775.90 | 0.00 | ORB-long ORB[769.95,778.30] vol=1.6x ATR=2.45 |
| Stop hit — per-position SL triggered | 2023-08-29 09:55:00 | 776.05 | 776.11 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:45:00 | 784.20 | 778.54 | 0.00 | ORB-long ORB[775.50,781.85] vol=1.8x ATR=2.59 |
| Stop hit — per-position SL triggered | 2023-08-30 13:25:00 | 781.61 | 782.53 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 09:30:00 | 754.30 | 757.37 | 0.00 | ORB-short ORB[755.05,764.85] vol=1.8x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 09:35:00 | 749.82 | 755.44 | 0.00 | T1 1.5R @ 749.82 |
| Stop hit — per-position SL triggered | 2023-09-04 09:40:00 | 754.30 | 755.31 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-09-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-11 11:05:00 | 708.65 | 713.93 | 0.00 | ORB-short ORB[711.15,719.95] vol=1.8x ATR=1.82 |
| Stop hit — per-position SL triggered | 2023-09-11 11:10:00 | 710.47 | 713.81 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-18 09:30:00 | 718.70 | 718.39 | 0.00 | ORB-long ORB[701.00,709.90] vol=11.9x ATR=4.84 |
| Stop hit — per-position SL triggered | 2023-09-18 09:45:00 | 713.86 | 718.02 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-09-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 10:00:00 | 679.95 | 687.23 | 0.00 | ORB-short ORB[687.05,696.40] vol=2.5x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-22 10:10:00 | 674.44 | 681.77 | 0.00 | T1 1.5R @ 674.44 |
| Target hit | 2023-09-22 10:20:00 | 678.00 | 676.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 26 — BUY (started 2023-09-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 09:55:00 | 711.75 | 706.94 | 0.00 | ORB-long ORB[701.90,709.90] vol=1.9x ATR=2.55 |
| Stop hit — per-position SL triggered | 2023-09-28 10:25:00 | 709.20 | 708.59 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-10-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 10:10:00 | 710.30 | 704.73 | 0.00 | ORB-long ORB[697.80,707.95] vol=1.9x ATR=4.14 |
| Stop hit — per-position SL triggered | 2023-10-05 10:25:00 | 706.16 | 706.60 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-10-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 11:10:00 | 695.85 | 691.42 | 0.00 | ORB-long ORB[687.00,694.95] vol=1.6x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 11:45:00 | 698.92 | 692.39 | 0.00 | T1 1.5R @ 698.92 |
| Target hit | 2023-10-06 14:40:00 | 704.95 | 705.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2023-10-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 11:00:00 | 711.00 | 715.46 | 0.00 | ORB-short ORB[713.00,717.80] vol=1.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2023-10-12 11:15:00 | 712.63 | 715.60 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-10-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 09:40:00 | 710.65 | 707.01 | 0.00 | ORB-long ORB[702.10,710.00] vol=2.2x ATR=3.43 |
| Stop hit — per-position SL triggered | 2023-10-16 13:30:00 | 707.22 | 710.11 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-11-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 10:00:00 | 849.75 | 857.76 | 0.00 | ORB-short ORB[853.65,865.75] vol=2.4x ATR=4.26 |
| Stop hit — per-position SL triggered | 2023-11-21 10:55:00 | 854.01 | 854.90 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-11-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 09:50:00 | 831.90 | 836.17 | 0.00 | ORB-short ORB[832.80,841.90] vol=2.1x ATR=3.97 |
| Stop hit — per-position SL triggered | 2023-11-23 10:00:00 | 835.87 | 836.10 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-11-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:45:00 | 824.85 | 830.98 | 0.00 | ORB-short ORB[827.75,838.65] vol=2.1x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 09:55:00 | 820.14 | 829.50 | 0.00 | T1 1.5R @ 820.14 |
| Stop hit — per-position SL triggered | 2023-11-30 10:15:00 | 824.85 | 828.22 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-12-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 09:40:00 | 816.00 | 807.90 | 0.00 | ORB-long ORB[798.70,809.60] vol=1.7x ATR=4.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 10:20:00 | 822.83 | 812.51 | 0.00 | T1 1.5R @ 822.83 |
| Target hit | 2023-12-07 15:20:00 | 830.85 | 827.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2023-12-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:05:00 | 828.85 | 836.24 | 0.00 | ORB-short ORB[836.00,846.80] vol=1.7x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 11:40:00 | 824.62 | 834.53 | 0.00 | T1 1.5R @ 824.62 |
| Target hit | 2023-12-08 15:20:00 | 813.65 | 824.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2023-12-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 09:35:00 | 810.70 | 816.30 | 0.00 | ORB-short ORB[816.00,821.40] vol=1.9x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 09:40:00 | 805.83 | 813.86 | 0.00 | T1 1.5R @ 805.83 |
| Stop hit — per-position SL triggered | 2023-12-11 09:45:00 | 810.70 | 813.52 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-12-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 10:00:00 | 798.80 | 804.07 | 0.00 | ORB-short ORB[805.00,808.25] vol=1.7x ATR=2.10 |
| Stop hit — per-position SL triggered | 2023-12-13 10:10:00 | 800.90 | 803.58 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 09:35:00 | 828.65 | 838.71 | 0.00 | ORB-short ORB[838.00,850.00] vol=2.7x ATR=3.44 |
| Stop hit — per-position SL triggered | 2023-12-19 09:45:00 | 832.09 | 836.96 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-12-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 09:35:00 | 844.50 | 848.97 | 0.00 | ORB-short ORB[847.10,857.10] vol=1.9x ATR=3.17 |
| Stop hit — per-position SL triggered | 2023-12-27 10:10:00 | 847.67 | 847.23 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-12-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 09:30:00 | 834.80 | 840.82 | 0.00 | ORB-short ORB[837.45,849.70] vol=1.9x ATR=3.11 |
| Stop hit — per-position SL triggered | 2023-12-28 09:40:00 | 837.91 | 839.28 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:30:00 | 912.35 | 920.72 | 0.00 | ORB-short ORB[914.60,927.95] vol=1.8x ATR=4.42 |
| Stop hit — per-position SL triggered | 2024-01-09 09:35:00 | 916.77 | 920.23 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-01-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 09:45:00 | 908.80 | 905.44 | 0.00 | ORB-long ORB[900.05,907.80] vol=1.6x ATR=3.62 |
| Stop hit — per-position SL triggered | 2024-01-11 09:55:00 | 905.18 | 905.43 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-01-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:35:00 | 883.95 | 898.46 | 0.00 | ORB-short ORB[893.80,905.95] vol=1.5x ATR=4.77 |
| Stop hit — per-position SL triggered | 2024-01-18 09:40:00 | 888.72 | 894.49 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-01-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-23 09:30:00 | 929.30 | 919.78 | 0.00 | ORB-long ORB[908.05,918.95] vol=5.8x ATR=4.42 |
| Stop hit — per-position SL triggered | 2024-01-23 09:35:00 | 924.88 | 920.40 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-01-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 09:50:00 | 903.00 | 909.83 | 0.00 | ORB-short ORB[908.50,916.35] vol=1.6x ATR=3.86 |
| Stop hit — per-position SL triggered | 2024-01-25 09:55:00 | 906.86 | 909.58 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 09:35:00 | 1551.45 | 1541.91 | 0.00 | ORB-long ORB[1524.00,1545.00] vol=3.3x ATR=7.70 |
| Stop hit — per-position SL triggered | 2024-04-25 09:45:00 | 1543.75 | 1546.89 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-17 10:05:00 | 460.40 | 2023-05-17 10:25:00 | 458.38 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2023-05-19 09:45:00 | 451.75 | 2023-05-19 10:00:00 | 448.55 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2023-05-19 09:45:00 | 451.75 | 2023-05-19 10:20:00 | 451.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-24 10:15:00 | 457.90 | 2023-05-24 11:30:00 | 455.63 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2023-05-24 10:15:00 | 457.90 | 2023-05-24 12:00:00 | 457.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-02 09:35:00 | 490.85 | 2023-06-02 09:40:00 | 488.52 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2023-06-06 09:55:00 | 483.00 | 2023-06-06 11:25:00 | 485.01 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-06-08 09:35:00 | 491.50 | 2023-06-08 09:40:00 | 493.24 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-06-09 09:30:00 | 484.40 | 2023-06-09 09:35:00 | 481.74 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2023-06-09 09:30:00 | 484.40 | 2023-06-09 09:40:00 | 484.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-14 10:00:00 | 493.65 | 2023-06-14 10:30:00 | 492.21 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-06-22 10:00:00 | 477.65 | 2023-06-22 11:30:00 | 475.46 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-06-22 10:00:00 | 477.65 | 2023-06-22 13:35:00 | 475.00 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2023-06-26 09:30:00 | 458.10 | 2023-06-26 09:35:00 | 460.74 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-06-26 09:30:00 | 458.10 | 2023-06-26 15:20:00 | 470.05 | TARGET_HIT | 0.50 | 2.61% |
| BUY | retest1 | 2023-06-27 09:30:00 | 477.55 | 2023-06-27 09:40:00 | 480.21 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-06-27 09:30:00 | 477.55 | 2023-06-27 15:20:00 | 498.85 | TARGET_HIT | 0.50 | 4.46% |
| SELL | retest1 | 2023-07-04 09:40:00 | 488.00 | 2023-07-04 09:50:00 | 485.60 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-07-04 09:40:00 | 488.00 | 2023-07-04 11:55:00 | 488.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-17 11:10:00 | 617.80 | 2023-07-17 11:15:00 | 615.24 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-07-26 09:40:00 | 701.70 | 2023-07-26 09:55:00 | 709.07 | PARTIAL | 0.50 | 1.05% |
| BUY | retest1 | 2023-07-26 09:40:00 | 701.70 | 2023-07-26 15:20:00 | 724.45 | TARGET_HIT | 0.50 | 3.24% |
| SELL | retest1 | 2023-07-31 11:10:00 | 692.70 | 2023-07-31 11:45:00 | 694.61 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-08-01 10:25:00 | 710.35 | 2023-08-01 13:20:00 | 713.85 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2023-08-10 10:25:00 | 766.25 | 2023-08-10 10:30:00 | 771.57 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest1 | 2023-08-24 10:40:00 | 769.10 | 2023-08-24 10:50:00 | 771.67 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-08-25 10:20:00 | 771.30 | 2023-08-25 10:25:00 | 767.75 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-08-25 10:20:00 | 771.30 | 2023-08-25 10:45:00 | 758.00 | TARGET_HIT | 0.50 | 1.72% |
| BUY | retest1 | 2023-08-29 09:40:00 | 778.50 | 2023-08-29 09:55:00 | 776.05 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-08-30 09:45:00 | 784.20 | 2023-08-30 13:25:00 | 781.61 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-09-04 09:30:00 | 754.30 | 2023-09-04 09:35:00 | 749.82 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2023-09-04 09:30:00 | 754.30 | 2023-09-04 09:40:00 | 754.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-11 11:05:00 | 708.65 | 2023-09-11 11:10:00 | 710.47 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-09-18 09:30:00 | 718.70 | 2023-09-18 09:45:00 | 713.86 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest1 | 2023-09-22 10:00:00 | 679.95 | 2023-09-22 10:10:00 | 674.44 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2023-09-22 10:00:00 | 679.95 | 2023-09-22 10:20:00 | 678.00 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2023-09-28 09:55:00 | 711.75 | 2023-09-28 10:25:00 | 709.20 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-10-05 10:10:00 | 710.30 | 2023-10-05 10:25:00 | 706.16 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2023-10-06 11:10:00 | 695.85 | 2023-10-06 11:45:00 | 698.92 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-10-06 11:10:00 | 695.85 | 2023-10-06 14:40:00 | 704.95 | TARGET_HIT | 0.50 | 1.31% |
| SELL | retest1 | 2023-10-12 11:00:00 | 711.00 | 2023-10-12 11:15:00 | 712.63 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-10-16 09:40:00 | 710.65 | 2023-10-16 13:30:00 | 707.22 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2023-11-21 10:00:00 | 849.75 | 2023-11-21 10:55:00 | 854.01 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2023-11-23 09:50:00 | 831.90 | 2023-11-23 10:00:00 | 835.87 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2023-11-30 09:45:00 | 824.85 | 2023-11-30 09:55:00 | 820.14 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2023-11-30 09:45:00 | 824.85 | 2023-11-30 10:15:00 | 824.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-07 09:40:00 | 816.00 | 2023-12-07 10:20:00 | 822.83 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2023-12-07 09:40:00 | 816.00 | 2023-12-07 15:20:00 | 830.85 | TARGET_HIT | 0.50 | 1.82% |
| SELL | retest1 | 2023-12-08 11:05:00 | 828.85 | 2023-12-08 11:40:00 | 824.62 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-12-08 11:05:00 | 828.85 | 2023-12-08 15:20:00 | 813.65 | TARGET_HIT | 0.50 | 1.83% |
| SELL | retest1 | 2023-12-11 09:35:00 | 810.70 | 2023-12-11 09:40:00 | 805.83 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2023-12-11 09:35:00 | 810.70 | 2023-12-11 09:45:00 | 810.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-13 10:00:00 | 798.80 | 2023-12-13 10:10:00 | 800.90 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-12-19 09:35:00 | 828.65 | 2023-12-19 09:45:00 | 832.09 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-12-27 09:35:00 | 844.50 | 2023-12-27 10:10:00 | 847.67 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-12-28 09:30:00 | 834.80 | 2023-12-28 09:40:00 | 837.91 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-01-09 09:30:00 | 912.35 | 2024-01-09 09:35:00 | 916.77 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-01-11 09:45:00 | 908.80 | 2024-01-11 09:55:00 | 905.18 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-01-18 09:35:00 | 883.95 | 2024-01-18 09:40:00 | 888.72 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-01-23 09:30:00 | 929.30 | 2024-01-23 09:35:00 | 924.88 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-01-25 09:50:00 | 903.00 | 2024-01-25 09:55:00 | 906.86 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-04-25 09:35:00 | 1551.45 | 2024-04-25 09:45:00 | 1543.75 | STOP_HIT | 1.00 | -0.50% |
