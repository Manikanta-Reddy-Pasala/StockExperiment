# DLF Ltd. (DLF)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-10-01 15:25:00 (25983 bars)
- **Last close:** 722.45
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
| ENTRY1 | 85 |
| ENTRY2 | 0 |
| PARTIAL | 36 |
| TARGET_HIT | 17 |
| STOP_HIT | 68 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 121 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 68
- **Target hits / Stop hits / Partials:** 17 / 68 / 36
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 23.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 24 | 38.7% | 7 | 38 | 17 | 0.11% | 6.9% |
| BUY @ 2nd Alert (retest1) | 62 | 24 | 38.7% | 7 | 38 | 17 | 0.11% | 6.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 59 | 29 | 49.2% | 10 | 30 | 19 | 0.28% | 16.7% |
| SELL @ 2nd Alert (retest1) | 59 | 29 | 49.2% | 10 | 30 | 19 | 0.28% | 16.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 121 | 53 | 43.8% | 17 | 68 | 36 | 0.20% | 23.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:40:00 | 836.05 | 833.13 | 0.00 | ORB-long ORB[828.85,834.50] vol=1.9x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:30:00 | 839.94 | 834.82 | 0.00 | T1 1.5R @ 839.94 |
| Stop hit — per-position SL triggered | 2024-05-16 10:45:00 | 836.05 | 835.21 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:50:00 | 851.95 | 846.25 | 0.00 | ORB-long ORB[842.70,849.35] vol=1.7x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-05-17 10:55:00 | 849.56 | 846.47 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 852.45 | 856.70 | 0.00 | ORB-short ORB[853.20,860.55] vol=1.7x ATR=2.80 |
| Stop hit — per-position SL triggered | 2024-05-22 09:50:00 | 855.25 | 856.09 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:35:00 | 837.95 | 843.40 | 0.00 | ORB-short ORB[842.70,851.00] vol=1.5x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-05-23 11:00:00 | 840.59 | 842.15 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 11:15:00 | 836.45 | 840.43 | 0.00 | ORB-short ORB[840.20,847.25] vol=1.6x ATR=2.18 |
| Stop hit — per-position SL triggered | 2024-05-27 13:05:00 | 838.63 | 838.19 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:05:00 | 827.30 | 837.07 | 0.00 | ORB-short ORB[839.50,845.35] vol=1.8x ATR=2.58 |
| Stop hit — per-position SL triggered | 2024-05-28 10:10:00 | 829.88 | 836.42 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:20:00 | 815.90 | 816.68 | 0.00 | ORB-short ORB[816.00,820.95] vol=1.7x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 10:30:00 | 813.27 | 816.33 | 0.00 | T1 1.5R @ 813.27 |
| Stop hit — per-position SL triggered | 2024-05-30 10:35:00 | 815.90 | 816.23 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 11:15:00 | 818.20 | 812.44 | 0.00 | ORB-long ORB[805.35,811.80] vol=1.5x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-05-31 11:55:00 | 815.46 | 813.25 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:55:00 | 866.65 | 862.93 | 0.00 | ORB-long ORB[858.20,864.80] vol=1.7x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 10:00:00 | 870.06 | 864.64 | 0.00 | T1 1.5R @ 870.06 |
| Target hit | 2024-06-13 11:15:00 | 870.55 | 871.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2024-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:30:00 | 882.60 | 878.02 | 0.00 | ORB-long ORB[871.90,880.50] vol=2.9x ATR=2.76 |
| Stop hit — per-position SL triggered | 2024-06-14 09:35:00 | 879.84 | 878.72 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:50:00 | 832.70 | 835.69 | 0.00 | ORB-short ORB[833.35,843.50] vol=1.6x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:10:00 | 829.36 | 834.45 | 0.00 | T1 1.5R @ 829.36 |
| Target hit | 2024-06-25 15:00:00 | 825.75 | 823.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — BUY (started 2024-06-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 10:55:00 | 829.00 | 819.18 | 0.00 | ORB-long ORB[813.00,822.50] vol=2.4x ATR=2.65 |
| Stop hit — per-position SL triggered | 2024-06-28 11:05:00 | 826.35 | 819.85 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:00:00 | 833.75 | 828.62 | 0.00 | ORB-long ORB[824.40,830.00] vol=2.9x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 10:05:00 | 837.12 | 831.97 | 0.00 | T1 1.5R @ 837.12 |
| Stop hit — per-position SL triggered | 2024-07-02 10:45:00 | 833.75 | 834.12 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 10:10:00 | 834.10 | 835.94 | 0.00 | ORB-short ORB[835.55,841.30] vol=1.9x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 11:05:00 | 831.11 | 834.56 | 0.00 | T1 1.5R @ 831.11 |
| Stop hit — per-position SL triggered | 2024-07-05 11:30:00 | 834.10 | 834.12 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 832.20 | 835.36 | 0.00 | ORB-short ORB[832.80,839.75] vol=1.8x ATR=1.82 |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 834.02 | 835.27 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 11:15:00 | 834.90 | 835.72 | 0.00 | ORB-short ORB[835.25,840.80] vol=1.9x ATR=1.85 |
| Stop hit — per-position SL triggered | 2024-07-09 12:05:00 | 836.75 | 835.64 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:55:00 | 831.95 | 833.27 | 0.00 | ORB-short ORB[832.00,837.90] vol=1.5x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:30:00 | 828.88 | 832.32 | 0.00 | T1 1.5R @ 828.88 |
| Target hit | 2024-07-12 15:20:00 | 821.70 | 827.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:15:00 | 844.25 | 839.57 | 0.00 | ORB-long ORB[831.10,842.30] vol=2.2x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 11:10:00 | 847.79 | 842.81 | 0.00 | T1 1.5R @ 847.79 |
| Stop hit — per-position SL triggered | 2024-07-16 14:05:00 | 844.25 | 844.56 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 832.10 | 838.08 | 0.00 | ORB-short ORB[834.00,844.00] vol=1.8x ATR=2.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:40:00 | 827.85 | 835.99 | 0.00 | T1 1.5R @ 827.85 |
| Stop hit — per-position SL triggered | 2024-07-18 09:45:00 | 832.10 | 835.88 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:00:00 | 836.30 | 840.48 | 0.00 | ORB-short ORB[837.30,848.00] vol=1.6x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:10:00 | 831.81 | 839.46 | 0.00 | T1 1.5R @ 831.81 |
| Target hit | 2024-07-19 15:20:00 | 813.00 | 823.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2024-07-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 11:05:00 | 884.00 | 877.00 | 0.00 | ORB-long ORB[870.25,877.70] vol=1.8x ATR=2.50 |
| Stop hit — per-position SL triggered | 2024-07-30 11:50:00 | 881.50 | 878.28 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 09:45:00 | 899.25 | 895.04 | 0.00 | ORB-long ORB[888.00,898.60] vol=3.0x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-08-01 10:05:00 | 896.29 | 896.54 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 09:55:00 | 853.90 | 859.48 | 0.00 | ORB-short ORB[855.40,864.95] vol=2.0x ATR=3.60 |
| Stop hit — per-position SL triggered | 2024-08-02 10:05:00 | 857.50 | 858.97 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:35:00 | 847.85 | 843.75 | 0.00 | ORB-long ORB[839.35,844.80] vol=1.6x ATR=2.12 |
| Stop hit — per-position SL triggered | 2024-08-08 09:50:00 | 845.73 | 845.17 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 09:55:00 | 835.30 | 841.03 | 0.00 | ORB-short ORB[838.50,847.60] vol=1.5x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 10:25:00 | 830.58 | 839.35 | 0.00 | T1 1.5R @ 830.58 |
| Stop hit — per-position SL triggered | 2024-08-09 13:45:00 | 835.30 | 837.00 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:15:00 | 836.40 | 829.43 | 0.00 | ORB-long ORB[822.45,832.50] vol=4.1x ATR=2.51 |
| Stop hit — per-position SL triggered | 2024-08-12 12:15:00 | 833.89 | 832.39 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 10:30:00 | 821.25 | 816.74 | 0.00 | ORB-long ORB[812.20,817.80] vol=2.1x ATR=3.13 |
| Stop hit — per-position SL triggered | 2024-08-14 10:50:00 | 818.12 | 817.02 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:45:00 | 868.40 | 863.75 | 0.00 | ORB-long ORB[860.00,865.70] vol=1.7x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:55:00 | 871.72 | 866.07 | 0.00 | T1 1.5R @ 871.72 |
| Stop hit — per-position SL triggered | 2024-08-20 10:35:00 | 868.40 | 867.44 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 09:30:00 | 864.70 | 866.33 | 0.00 | ORB-short ORB[865.00,871.15] vol=1.5x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 09:40:00 | 862.19 | 865.44 | 0.00 | T1 1.5R @ 862.19 |
| Target hit | 2024-08-21 15:05:00 | 860.45 | 859.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — BUY (started 2024-08-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:55:00 | 853.95 | 848.42 | 0.00 | ORB-long ORB[845.00,851.00] vol=3.0x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:35:00 | 857.42 | 851.48 | 0.00 | T1 1.5R @ 857.42 |
| Stop hit — per-position SL triggered | 2024-08-27 12:00:00 | 853.95 | 854.57 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 840.85 | 844.95 | 0.00 | ORB-short ORB[844.90,849.50] vol=1.5x ATR=2.05 |
| Stop hit — per-position SL triggered | 2024-08-28 09:45:00 | 842.90 | 842.36 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 829.60 | 832.81 | 0.00 | ORB-short ORB[833.00,838.30] vol=4.1x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-08-29 09:50:00 | 831.48 | 831.70 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 858.00 | 854.49 | 0.00 | ORB-long ORB[847.00,855.95] vol=4.1x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-09-03 09:35:00 | 855.81 | 854.67 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:50:00 | 826.20 | 831.87 | 0.00 | ORB-short ORB[832.40,841.00] vol=1.6x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 822.79 | 830.25 | 0.00 | T1 1.5R @ 822.79 |
| Target hit | 2024-09-06 15:20:00 | 814.60 | 818.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2024-09-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:45:00 | 854.20 | 848.93 | 0.00 | ORB-long ORB[836.05,847.80] vol=4.2x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 10:35:00 | 858.76 | 852.92 | 0.00 | T1 1.5R @ 858.76 |
| Target hit | 2024-09-13 15:20:00 | 863.55 | 858.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2024-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:30:00 | 878.10 | 873.15 | 0.00 | ORB-long ORB[865.60,875.90] vol=2.8x ATR=3.11 |
| Stop hit — per-position SL triggered | 2024-09-16 09:35:00 | 874.99 | 873.38 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:50:00 | 863.50 | 859.21 | 0.00 | ORB-long ORB[854.00,862.10] vol=1.7x ATR=2.55 |
| Stop hit — per-position SL triggered | 2024-09-17 09:55:00 | 860.95 | 859.34 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 11:00:00 | 854.85 | 858.02 | 0.00 | ORB-short ORB[855.45,863.05] vol=1.8x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-09-18 11:25:00 | 856.73 | 857.81 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-09-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:35:00 | 865.05 | 869.06 | 0.00 | ORB-short ORB[867.50,874.90] vol=2.1x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:05:00 | 859.97 | 867.33 | 0.00 | T1 1.5R @ 859.97 |
| Target hit | 2024-09-19 15:00:00 | 851.55 | 849.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — BUY (started 2024-09-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:40:00 | 862.70 | 859.08 | 0.00 | ORB-long ORB[853.15,860.95] vol=1.9x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 11:45:00 | 867.08 | 860.77 | 0.00 | T1 1.5R @ 867.08 |
| Target hit | 2024-09-20 15:20:00 | 880.30 | 869.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2024-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:00:00 | 897.20 | 889.23 | 0.00 | ORB-long ORB[880.30,892.00] vol=3.1x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:50:00 | 901.31 | 892.35 | 0.00 | T1 1.5R @ 901.31 |
| Target hit | 2024-09-23 15:20:00 | 907.00 | 904.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2024-09-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:50:00 | 920.05 | 915.01 | 0.00 | ORB-long ORB[910.20,917.50] vol=2.5x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 11:15:00 | 924.76 | 918.25 | 0.00 | T1 1.5R @ 924.76 |
| Stop hit — per-position SL triggered | 2024-09-24 12:10:00 | 920.05 | 918.94 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-09-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:35:00 | 918.45 | 924.04 | 0.00 | ORB-short ORB[920.00,928.80] vol=2.0x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 09:45:00 | 913.29 | 921.92 | 0.00 | T1 1.5R @ 913.29 |
| Target hit | 2024-09-26 14:20:00 | 915.40 | 915.10 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — SELL (started 2024-09-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 11:05:00 | 914.20 | 922.55 | 0.00 | ORB-short ORB[918.00,929.00] vol=2.1x ATR=2.71 |
| Stop hit — per-position SL triggered | 2024-09-27 11:15:00 | 916.91 | 922.17 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:35:00 | 827.05 | 837.81 | 0.00 | ORB-short ORB[841.65,853.75] vol=3.0x ATR=3.59 |
| Stop hit — per-position SL triggered | 2024-10-07 10:40:00 | 830.64 | 837.04 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-10-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:30:00 | 847.80 | 844.03 | 0.00 | ORB-long ORB[840.15,845.90] vol=1.8x ATR=2.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 09:40:00 | 852.04 | 847.05 | 0.00 | T1 1.5R @ 852.04 |
| Stop hit — per-position SL triggered | 2024-10-09 10:00:00 | 847.80 | 849.61 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 11:15:00 | 849.00 | 854.70 | 0.00 | ORB-short ORB[851.60,862.00] vol=2.0x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 11:40:00 | 845.22 | 854.01 | 0.00 | T1 1.5R @ 845.22 |
| Stop hit — per-position SL triggered | 2024-10-11 11:50:00 | 849.00 | 853.47 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-10-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 10:55:00 | 856.20 | 850.89 | 0.00 | ORB-long ORB[844.15,852.05] vol=1.8x ATR=2.60 |
| Stop hit — per-position SL triggered | 2024-10-14 11:10:00 | 853.60 | 851.22 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-10-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:35:00 | 868.95 | 876.12 | 0.00 | ORB-short ORB[874.25,884.40] vol=1.5x ATR=3.33 |
| Stop hit — per-position SL triggered | 2024-10-21 10:00:00 | 872.28 | 873.73 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:15:00 | 840.50 | 834.08 | 0.00 | ORB-long ORB[825.30,836.70] vol=1.7x ATR=2.77 |
| Stop hit — per-position SL triggered | 2024-10-30 10:20:00 | 837.73 | 834.28 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-11-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 09:50:00 | 806.80 | 814.05 | 0.00 | ORB-short ORB[813.80,825.00] vol=1.7x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 10:05:00 | 801.61 | 811.19 | 0.00 | T1 1.5R @ 801.61 |
| Target hit | 2024-11-04 15:20:00 | 789.95 | 797.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 11:15:00 | 792.00 | 799.92 | 0.00 | ORB-short ORB[794.00,805.00] vol=3.4x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 11:30:00 | 788.34 | 798.42 | 0.00 | T1 1.5R @ 788.34 |
| Stop hit — per-position SL triggered | 2024-11-08 12:10:00 | 792.00 | 797.05 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:30:00 | 770.55 | 766.24 | 0.00 | ORB-long ORB[759.60,768.00] vol=1.9x ATR=2.59 |
| Stop hit — per-position SL triggered | 2024-11-19 09:50:00 | 767.96 | 767.68 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-11-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:35:00 | 821.45 | 824.43 | 0.00 | ORB-short ORB[821.80,829.00] vol=3.8x ATR=2.49 |
| Stop hit — per-position SL triggered | 2024-11-27 11:40:00 | 823.94 | 823.24 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:30:00 | 824.75 | 822.01 | 0.00 | ORB-long ORB[816.60,824.10] vol=2.1x ATR=2.17 |
| Stop hit — per-position SL triggered | 2024-11-28 09:35:00 | 822.58 | 822.05 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:30:00 | 832.40 | 827.40 | 0.00 | ORB-long ORB[822.25,828.95] vol=1.7x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-12-02 09:40:00 | 830.01 | 828.22 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-12-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:40:00 | 853.75 | 848.92 | 0.00 | ORB-long ORB[846.35,852.90] vol=2.2x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-12-06 10:50:00 | 850.87 | 849.16 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-12-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:40:00 | 867.45 | 861.56 | 0.00 | ORB-long ORB[855.20,863.65] vol=1.7x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 09:45:00 | 871.36 | 864.01 | 0.00 | T1 1.5R @ 871.36 |
| Stop hit — per-position SL triggered | 2024-12-09 09:50:00 | 867.45 | 864.54 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-12-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:45:00 | 869.30 | 865.72 | 0.00 | ORB-long ORB[860.70,867.50] vol=1.8x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-12-10 09:55:00 | 866.90 | 865.94 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-12-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:30:00 | 876.55 | 873.14 | 0.00 | ORB-long ORB[863.00,875.60] vol=2.3x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-12-11 10:05:00 | 874.36 | 874.68 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-12-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:45:00 | 860.10 | 864.10 | 0.00 | ORB-short ORB[861.30,870.90] vol=1.5x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:10:00 | 855.95 | 862.41 | 0.00 | T1 1.5R @ 855.95 |
| Target hit | 2024-12-13 11:45:00 | 857.40 | 857.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 62 — BUY (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 11:15:00 | 867.30 | 861.18 | 0.00 | ORB-long ORB[853.00,862.20] vol=1.7x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 11:45:00 | 871.25 | 862.01 | 0.00 | T1 1.5R @ 871.25 |
| Stop hit — per-position SL triggered | 2024-12-19 12:30:00 | 867.30 | 862.83 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:15:00 | 849.35 | 842.13 | 0.00 | ORB-long ORB[836.30,846.90] vol=1.6x ATR=2.57 |
| Stop hit — per-position SL triggered | 2024-12-24 11:20:00 | 846.78 | 846.37 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 817.75 | 820.60 | 0.00 | ORB-short ORB[819.00,827.00] vol=2.2x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-01-02 09:40:00 | 819.43 | 819.99 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:10:00 | 826.60 | 833.66 | 0.00 | ORB-short ORB[834.60,839.85] vol=1.7x ATR=2.11 |
| Stop hit — per-position SL triggered | 2025-01-03 10:30:00 | 828.71 | 831.65 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:50:00 | 812.75 | 824.42 | 0.00 | ORB-short ORB[825.00,834.00] vol=2.6x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:05:00 | 808.96 | 821.67 | 0.00 | T1 1.5R @ 808.96 |
| Stop hit — per-position SL triggered | 2025-01-06 12:15:00 | 812.75 | 816.21 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-01-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:45:00 | 735.10 | 738.85 | 0.00 | ORB-short ORB[737.00,747.95] vol=2.3x ATR=3.62 |
| Stop hit — per-position SL triggered | 2025-01-13 09:55:00 | 738.72 | 738.46 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-01-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:05:00 | 757.35 | 753.01 | 0.00 | ORB-long ORB[749.30,754.75] vol=1.6x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 10:10:00 | 760.52 | 754.23 | 0.00 | T1 1.5R @ 760.52 |
| Stop hit — per-position SL triggered | 2025-01-20 10:20:00 | 757.35 | 755.02 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-01-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:05:00 | 739.20 | 753.52 | 0.00 | ORB-short ORB[756.40,764.35] vol=1.6x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-01-21 11:35:00 | 742.16 | 751.65 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:30:00 | 712.50 | 718.44 | 0.00 | ORB-short ORB[715.00,725.00] vol=2.1x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-01-28 09:45:00 | 716.69 | 717.14 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-01-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:40:00 | 740.90 | 733.81 | 0.00 | ORB-long ORB[725.00,732.15] vol=2.5x ATR=2.58 |
| Stop hit — per-position SL triggered | 2025-01-29 10:50:00 | 738.32 | 734.38 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 757.65 | 752.32 | 0.00 | ORB-long ORB[745.40,756.05] vol=2.3x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 09:35:00 | 761.51 | 754.25 | 0.00 | T1 1.5R @ 761.51 |
| Target hit | 2025-01-30 10:50:00 | 760.00 | 760.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — BUY (started 2025-02-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:30:00 | 754.95 | 752.64 | 0.00 | ORB-long ORB[744.80,752.75] vol=2.6x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-02-01 09:40:00 | 752.56 | 752.78 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-02-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 09:30:00 | 760.30 | 763.66 | 0.00 | ORB-short ORB[761.20,768.60] vol=2.2x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 09:45:00 | 756.19 | 762.64 | 0.00 | T1 1.5R @ 756.19 |
| Stop hit — per-position SL triggered | 2025-02-07 10:05:00 | 760.30 | 761.40 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 09:35:00 | 665.45 | 667.76 | 0.00 | ORB-short ORB[666.50,675.20] vol=3.6x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-02-18 09:50:00 | 668.41 | 667.86 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-03-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:30:00 | 678.50 | 673.82 | 0.00 | ORB-long ORB[668.20,676.60] vol=2.8x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-03-07 09:35:00 | 676.11 | 674.04 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:45:00 | 673.25 | 671.65 | 0.00 | ORB-long ORB[666.55,672.80] vol=2.1x ATR=2.07 |
| Stop hit — per-position SL triggered | 2025-03-13 09:50:00 | 671.18 | 671.67 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-03-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:00:00 | 695.25 | 699.05 | 0.00 | ORB-short ORB[696.20,706.50] vol=1.7x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 10:05:00 | 691.23 | 698.68 | 0.00 | T1 1.5R @ 691.23 |
| Stop hit — per-position SL triggered | 2025-03-20 14:25:00 | 695.25 | 696.34 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-03-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 11:00:00 | 685.00 | 681.99 | 0.00 | ORB-long ORB[672.00,682.00] vol=1.6x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-03-27 11:05:00 | 683.20 | 682.01 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-04-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:45:00 | 671.40 | 667.30 | 0.00 | ORB-long ORB[658.05,667.80] vol=2.8x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 10:25:00 | 675.40 | 669.07 | 0.00 | T1 1.5R @ 675.40 |
| Target hit | 2025-04-02 15:20:00 | 683.15 | 675.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — BUY (started 2025-04-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:35:00 | 629.35 | 626.07 | 0.00 | ORB-long ORB[622.35,628.00] vol=1.6x ATR=2.48 |
| Stop hit — per-position SL triggered | 2025-04-11 09:45:00 | 626.87 | 626.34 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 11:00:00 | 653.00 | 658.30 | 0.00 | ORB-short ORB[656.15,665.65] vol=1.9x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-04-16 11:15:00 | 654.93 | 657.80 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-04-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:45:00 | 663.85 | 661.17 | 0.00 | ORB-long ORB[656.50,663.50] vol=2.3x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 11:55:00 | 667.00 | 662.37 | 0.00 | T1 1.5R @ 667.00 |
| Target hit | 2025-04-17 15:20:00 | 670.25 | 665.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 84 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 675.20 | 681.06 | 0.00 | ORB-short ORB[678.00,685.90] vol=2.1x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:35:00 | 671.77 | 679.55 | 0.00 | T1 1.5R @ 671.77 |
| Target hit | 2025-04-25 15:20:00 | 651.45 | 661.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 85 — BUY (started 2025-04-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 10:45:00 | 661.80 | 659.57 | 0.00 | ORB-long ORB[653.00,660.75] vol=2.2x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-04-28 14:35:00 | 659.59 | 661.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 09:40:00 | 836.05 | 2024-05-16 10:30:00 | 839.94 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-05-16 09:40:00 | 836.05 | 2024-05-16 10:45:00 | 836.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-17 10:50:00 | 851.95 | 2024-05-17 10:55:00 | 849.56 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-05-22 09:40:00 | 852.45 | 2024-05-22 09:50:00 | 855.25 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-23 10:35:00 | 837.95 | 2024-05-23 11:00:00 | 840.59 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-27 11:15:00 | 836.45 | 2024-05-27 13:05:00 | 838.63 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-05-28 10:05:00 | 827.30 | 2024-05-28 10:10:00 | 829.88 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-30 10:20:00 | 815.90 | 2024-05-30 10:30:00 | 813.27 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-05-30 10:20:00 | 815.90 | 2024-05-30 10:35:00 | 815.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-31 11:15:00 | 818.20 | 2024-05-31 11:55:00 | 815.46 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-13 09:55:00 | 866.65 | 2024-06-13 10:00:00 | 870.06 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-06-13 09:55:00 | 866.65 | 2024-06-13 11:15:00 | 870.55 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2024-06-14 09:30:00 | 882.60 | 2024-06-14 09:35:00 | 879.84 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-25 09:50:00 | 832.70 | 2024-06-25 10:10:00 | 829.36 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-06-25 09:50:00 | 832.70 | 2024-06-25 15:00:00 | 825.75 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2024-06-28 10:55:00 | 829.00 | 2024-06-28 11:05:00 | 826.35 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-02 10:00:00 | 833.75 | 2024-07-02 10:05:00 | 837.12 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-02 10:00:00 | 833.75 | 2024-07-02 10:45:00 | 833.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-05 10:10:00 | 834.10 | 2024-07-05 11:05:00 | 831.11 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-07-05 10:10:00 | 834.10 | 2024-07-05 11:30:00 | 834.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 11:10:00 | 832.20 | 2024-07-08 11:15:00 | 834.02 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-09 11:15:00 | 834.90 | 2024-07-09 12:05:00 | 836.75 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-12 09:55:00 | 831.95 | 2024-07-12 11:30:00 | 828.88 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-07-12 09:55:00 | 831.95 | 2024-07-12 15:20:00 | 821.70 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2024-07-16 10:15:00 | 844.25 | 2024-07-16 11:10:00 | 847.79 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-07-16 10:15:00 | 844.25 | 2024-07-16 14:05:00 | 844.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-18 09:30:00 | 832.10 | 2024-07-18 09:40:00 | 827.85 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-07-18 09:30:00 | 832.10 | 2024-07-18 09:45:00 | 832.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-19 10:00:00 | 836.30 | 2024-07-19 10:10:00 | 831.81 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-07-19 10:00:00 | 836.30 | 2024-07-19 15:20:00 | 813.00 | TARGET_HIT | 0.50 | 2.79% |
| BUY | retest1 | 2024-07-30 11:05:00 | 884.00 | 2024-07-30 11:50:00 | 881.50 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-01 09:45:00 | 899.25 | 2024-08-01 10:05:00 | 896.29 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-02 09:55:00 | 853.90 | 2024-08-02 10:05:00 | 857.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-08-08 09:35:00 | 847.85 | 2024-08-08 09:50:00 | 845.73 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-09 09:55:00 | 835.30 | 2024-08-09 10:25:00 | 830.58 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-08-09 09:55:00 | 835.30 | 2024-08-09 13:45:00 | 835.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-12 11:15:00 | 836.40 | 2024-08-12 12:15:00 | 833.89 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-14 10:30:00 | 821.25 | 2024-08-14 10:50:00 | 818.12 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-20 09:45:00 | 868.40 | 2024-08-20 09:55:00 | 871.72 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-08-20 09:45:00 | 868.40 | 2024-08-20 10:35:00 | 868.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-21 09:30:00 | 864.70 | 2024-08-21 09:40:00 | 862.19 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-08-21 09:30:00 | 864.70 | 2024-08-21 15:05:00 | 860.45 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2024-08-27 09:55:00 | 853.95 | 2024-08-27 10:35:00 | 857.42 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-27 09:55:00 | 853.95 | 2024-08-27 12:00:00 | 853.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:30:00 | 840.85 | 2024-08-28 09:45:00 | 842.90 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-29 09:30:00 | 829.60 | 2024-08-29 09:50:00 | 831.48 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-03 09:30:00 | 858.00 | 2024-09-03 09:35:00 | 855.81 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-06 09:50:00 | 826.20 | 2024-09-06 10:05:00 | 822.79 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-09-06 09:50:00 | 826.20 | 2024-09-06 15:20:00 | 814.60 | TARGET_HIT | 0.50 | 1.40% |
| BUY | retest1 | 2024-09-13 09:45:00 | 854.20 | 2024-09-13 10:35:00 | 858.76 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-09-13 09:45:00 | 854.20 | 2024-09-13 15:20:00 | 863.55 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2024-09-16 09:30:00 | 878.10 | 2024-09-16 09:35:00 | 874.99 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-09-17 09:50:00 | 863.50 | 2024-09-17 09:55:00 | 860.95 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-18 11:00:00 | 854.85 | 2024-09-18 11:25:00 | 856.73 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-09-19 09:35:00 | 865.05 | 2024-09-19 10:05:00 | 859.97 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-09-19 09:35:00 | 865.05 | 2024-09-19 15:00:00 | 851.55 | TARGET_HIT | 0.50 | 1.56% |
| BUY | retest1 | 2024-09-20 10:40:00 | 862.70 | 2024-09-20 11:45:00 | 867.08 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-09-20 10:40:00 | 862.70 | 2024-09-20 15:20:00 | 880.30 | TARGET_HIT | 0.50 | 2.04% |
| BUY | retest1 | 2024-09-23 11:00:00 | 897.20 | 2024-09-23 11:50:00 | 901.31 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-09-23 11:00:00 | 897.20 | 2024-09-23 15:20:00 | 907.00 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2024-09-24 09:50:00 | 920.05 | 2024-09-24 11:15:00 | 924.76 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-09-24 09:50:00 | 920.05 | 2024-09-24 12:10:00 | 920.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-26 09:35:00 | 918.45 | 2024-09-26 09:45:00 | 913.29 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-09-26 09:35:00 | 918.45 | 2024-09-26 14:20:00 | 915.40 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2024-09-27 11:05:00 | 914.20 | 2024-09-27 11:15:00 | 916.91 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-07 10:35:00 | 827.05 | 2024-10-07 10:40:00 | 830.64 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-10-09 09:30:00 | 847.80 | 2024-10-09 09:40:00 | 852.04 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-10-09 09:30:00 | 847.80 | 2024-10-09 10:00:00 | 847.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-11 11:15:00 | 849.00 | 2024-10-11 11:40:00 | 845.22 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-10-11 11:15:00 | 849.00 | 2024-10-11 11:50:00 | 849.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-14 10:55:00 | 856.20 | 2024-10-14 11:10:00 | 853.60 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-21 09:35:00 | 868.95 | 2024-10-21 10:00:00 | 872.28 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-30 10:15:00 | 840.50 | 2024-10-30 10:20:00 | 837.73 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-11-04 09:50:00 | 806.80 | 2024-11-04 10:05:00 | 801.61 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-11-04 09:50:00 | 806.80 | 2024-11-04 15:20:00 | 789.95 | TARGET_HIT | 0.50 | 2.09% |
| SELL | retest1 | 2024-11-08 11:15:00 | 792.00 | 2024-11-08 11:30:00 | 788.34 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-11-08 11:15:00 | 792.00 | 2024-11-08 12:10:00 | 792.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 09:30:00 | 770.55 | 2024-11-19 09:50:00 | 767.96 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-11-27 10:35:00 | 821.45 | 2024-11-27 11:40:00 | 823.94 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-11-28 09:30:00 | 824.75 | 2024-11-28 09:35:00 | 822.58 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-02 09:30:00 | 832.40 | 2024-12-02 09:40:00 | 830.01 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-06 10:40:00 | 853.75 | 2024-12-06 10:50:00 | 850.87 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-09 09:40:00 | 867.45 | 2024-12-09 09:45:00 | 871.36 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-12-09 09:40:00 | 867.45 | 2024-12-09 09:50:00 | 867.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-10 09:45:00 | 869.30 | 2024-12-10 09:55:00 | 866.90 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-11 09:30:00 | 876.55 | 2024-12-11 10:05:00 | 874.36 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-13 09:45:00 | 860.10 | 2024-12-13 10:10:00 | 855.95 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-12-13 09:45:00 | 860.10 | 2024-12-13 11:45:00 | 857.40 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2024-12-19 11:15:00 | 867.30 | 2024-12-19 11:45:00 | 871.25 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-12-19 11:15:00 | 867.30 | 2024-12-19 12:30:00 | 867.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 10:15:00 | 849.35 | 2024-12-24 11:20:00 | 846.78 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-02 09:30:00 | 817.75 | 2025-01-02 09:40:00 | 819.43 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-01-03 10:10:00 | 826.60 | 2025-01-03 10:30:00 | 828.71 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-01-06 10:50:00 | 812.75 | 2025-01-06 11:05:00 | 808.96 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-01-06 10:50:00 | 812.75 | 2025-01-06 12:15:00 | 812.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-13 09:45:00 | 735.10 | 2025-01-13 09:55:00 | 738.72 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-01-20 10:05:00 | 757.35 | 2025-01-20 10:10:00 | 760.52 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-01-20 10:05:00 | 757.35 | 2025-01-20 10:20:00 | 757.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-21 11:05:00 | 739.20 | 2025-01-21 11:35:00 | 742.16 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-01-28 09:30:00 | 712.50 | 2025-01-28 09:45:00 | 716.69 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2025-01-29 10:40:00 | 740.90 | 2025-01-29 10:50:00 | 738.32 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-30 09:30:00 | 757.65 | 2025-01-30 09:35:00 | 761.51 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-01-30 09:30:00 | 757.65 | 2025-01-30 10:50:00 | 760.00 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2025-02-01 09:30:00 | 754.95 | 2025-02-01 09:40:00 | 752.56 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-02-07 09:30:00 | 760.30 | 2025-02-07 09:45:00 | 756.19 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-02-07 09:30:00 | 760.30 | 2025-02-07 10:05:00 | 760.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-18 09:35:00 | 665.45 | 2025-02-18 09:50:00 | 668.41 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-03-07 09:30:00 | 678.50 | 2025-03-07 09:35:00 | 676.11 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-13 09:45:00 | 673.25 | 2025-03-13 09:50:00 | 671.18 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-03-20 10:00:00 | 695.25 | 2025-03-20 10:05:00 | 691.23 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-03-20 10:00:00 | 695.25 | 2025-03-20 14:25:00 | 695.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-27 11:00:00 | 685.00 | 2025-03-27 11:05:00 | 683.20 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-04-02 09:45:00 | 671.40 | 2025-04-02 10:25:00 | 675.40 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-04-02 09:45:00 | 671.40 | 2025-04-02 15:20:00 | 683.15 | TARGET_HIT | 0.50 | 1.75% |
| BUY | retest1 | 2025-04-11 09:35:00 | 629.35 | 2025-04-11 09:45:00 | 626.87 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-04-16 11:00:00 | 653.00 | 2025-04-16 11:15:00 | 654.93 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-04-17 10:45:00 | 663.85 | 2025-04-17 11:55:00 | 667.00 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-04-17 10:45:00 | 663.85 | 2025-04-17 15:20:00 | 670.25 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2025-04-25 09:30:00 | 675.20 | 2025-04-25 09:35:00 | 671.77 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-04-25 09:30:00 | 675.20 | 2025-04-25 15:20:00 | 651.45 | TARGET_HIT | 0.50 | 3.52% |
| BUY | retest1 | 2025-04-28 10:45:00 | 661.80 | 2025-04-28 14:35:00 | 659.59 | STOP_HIT | 1.00 | -0.33% |
