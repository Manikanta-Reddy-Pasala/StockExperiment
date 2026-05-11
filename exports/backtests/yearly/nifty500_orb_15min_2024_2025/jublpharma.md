# Jubilant Pharmova Ltd. (JUBLPHARMA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1009.00
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
| ENTRY1 | 48 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 8 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 40
- **Target hits / Stop hits / Partials:** 8 / 40 / 24
- **Avg / median % per leg:** 0.31% / 0.00%
- **Sum % (uncompounded):** 21.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 16 | 51.6% | 5 | 15 | 11 | 0.36% | 11.0% |
| BUY @ 2nd Alert (retest1) | 31 | 16 | 51.6% | 5 | 15 | 11 | 0.36% | 11.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 41 | 16 | 39.0% | 3 | 25 | 13 | 0.27% | 10.9% |
| SELL @ 2nd Alert (retest1) | 41 | 16 | 39.0% | 3 | 25 | 13 | 0.27% | 10.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 72 | 32 | 44.4% | 8 | 40 | 24 | 0.31% | 22.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:40:00 | 730.40 | 725.29 | 0.00 | ORB-long ORB[715.00,725.00] vol=4.0x ATR=2.85 |
| Stop hit — per-position SL triggered | 2024-05-17 10:50:00 | 727.55 | 725.54 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 701.20 | 705.85 | 0.00 | ORB-short ORB[704.00,709.90] vol=2.9x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 09:55:00 | 696.98 | 702.41 | 0.00 | T1 1.5R @ 696.98 |
| Stop hit — per-position SL triggered | 2024-05-27 10:05:00 | 701.20 | 701.83 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:40:00 | 708.90 | 702.82 | 0.00 | ORB-long ORB[695.05,704.90] vol=2.0x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 10:00:00 | 713.38 | 707.44 | 0.00 | T1 1.5R @ 713.38 |
| Target hit | 2024-05-29 10:45:00 | 716.05 | 719.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2024-06-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:50:00 | 751.75 | 748.12 | 0.00 | ORB-long ORB[739.35,747.85] vol=3.1x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 09:55:00 | 756.89 | 749.08 | 0.00 | T1 1.5R @ 756.89 |
| Stop hit — per-position SL triggered | 2024-06-11 10:00:00 | 751.75 | 749.12 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:00:00 | 747.00 | 744.07 | 0.00 | ORB-long ORB[738.25,745.95] vol=2.3x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:05:00 | 750.05 | 746.03 | 0.00 | T1 1.5R @ 750.05 |
| Stop hit — per-position SL triggered | 2024-06-12 11:15:00 | 747.00 | 746.32 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:55:00 | 744.60 | 741.32 | 0.00 | ORB-long ORB[735.65,743.15] vol=2.1x ATR=1.98 |
| Stop hit — per-position SL triggered | 2024-06-13 11:05:00 | 742.62 | 741.39 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:35:00 | 759.20 | 750.15 | 0.00 | ORB-long ORB[742.40,752.85] vol=1.6x ATR=4.02 |
| Stop hit — per-position SL triggered | 2024-06-14 14:20:00 | 755.18 | 757.66 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:05:00 | 732.00 | 735.10 | 0.00 | ORB-short ORB[733.65,740.10] vol=1.7x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 10:15:00 | 728.21 | 733.72 | 0.00 | T1 1.5R @ 728.21 |
| Stop hit — per-position SL triggered | 2024-06-21 10:25:00 | 732.00 | 731.39 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 733.50 | 735.54 | 0.00 | ORB-short ORB[734.00,740.45] vol=2.4x ATR=1.58 |
| Stop hit — per-position SL triggered | 2024-06-25 11:25:00 | 735.08 | 735.49 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 11:10:00 | 733.50 | 737.54 | 0.00 | ORB-short ORB[734.00,745.00] vol=1.9x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 11:40:00 | 730.27 | 736.87 | 0.00 | T1 1.5R @ 730.27 |
| Stop hit — per-position SL triggered | 2024-07-02 12:15:00 | 733.50 | 736.34 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 09:50:00 | 732.00 | 734.48 | 0.00 | ORB-short ORB[734.55,739.95] vol=3.2x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 09:55:00 | 729.10 | 733.59 | 0.00 | T1 1.5R @ 729.10 |
| Stop hit — per-position SL triggered | 2024-07-03 10:20:00 | 732.00 | 731.91 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 722.05 | 728.39 | 0.00 | ORB-short ORB[732.85,740.35] vol=3.4x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:35:00 | 716.96 | 725.85 | 0.00 | T1 1.5R @ 716.96 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 722.05 | 725.14 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:50:00 | 724.45 | 726.32 | 0.00 | ORB-short ORB[724.85,729.90] vol=2.8x ATR=2.41 |
| Stop hit — per-position SL triggered | 2024-07-12 09:55:00 | 726.86 | 726.28 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:30:00 | 714.05 | 717.88 | 0.00 | ORB-short ORB[716.20,725.85] vol=2.7x ATR=2.57 |
| Stop hit — per-position SL triggered | 2024-07-15 09:35:00 | 716.62 | 717.39 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 10:25:00 | 712.50 | 715.01 | 0.00 | ORB-short ORB[715.50,725.20] vol=1.7x ATR=3.77 |
| Stop hit — per-position SL triggered | 2024-07-23 11:00:00 | 716.27 | 714.86 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:50:00 | 733.15 | 728.88 | 0.00 | ORB-long ORB[722.35,730.30] vol=2.3x ATR=3.88 |
| Stop hit — per-position SL triggered | 2024-07-24 11:10:00 | 729.27 | 730.58 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 10:05:00 | 715.00 | 718.77 | 0.00 | ORB-short ORB[715.20,724.00] vol=1.6x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-07-25 11:05:00 | 717.39 | 717.21 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-26 10:40:00 | 725.00 | 730.39 | 0.00 | ORB-short ORB[731.80,738.95] vol=1.8x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-07-26 13:05:00 | 727.64 | 727.75 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:55:00 | 898.60 | 900.60 | 0.00 | ORB-short ORB[898.65,911.20] vol=5.9x ATR=5.01 |
| Stop hit — per-position SL triggered | 2024-08-22 10:40:00 | 903.61 | 900.04 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 887.95 | 894.63 | 0.00 | ORB-short ORB[892.00,899.70] vol=1.5x ATR=3.21 |
| Stop hit — per-position SL triggered | 2024-08-23 10:25:00 | 891.16 | 891.95 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:50:00 | 926.90 | 925.10 | 0.00 | ORB-long ORB[919.05,925.75] vol=1.9x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:00:00 | 931.75 | 926.22 | 0.00 | T1 1.5R @ 931.75 |
| Stop hit — per-position SL triggered | 2024-08-29 10:10:00 | 926.90 | 926.34 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 11:15:00 | 913.95 | 918.11 | 0.00 | ORB-short ORB[916.00,924.30] vol=1.6x ATR=2.70 |
| Stop hit — per-position SL triggered | 2024-09-02 11:35:00 | 916.65 | 917.91 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:45:00 | 1038.15 | 1031.39 | 0.00 | ORB-long ORB[1025.35,1036.00] vol=1.6x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 10:50:00 | 1043.57 | 1032.61 | 0.00 | T1 1.5R @ 1043.57 |
| Stop hit — per-position SL triggered | 2024-09-12 10:55:00 | 1038.15 | 1033.48 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:20:00 | 1220.45 | 1211.71 | 0.00 | ORB-long ORB[1208.65,1220.00] vol=1.7x ATR=6.81 |
| Stop hit — per-position SL triggered | 2024-09-23 11:25:00 | 1213.64 | 1215.65 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-10-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:35:00 | 1179.10 | 1175.39 | 0.00 | ORB-long ORB[1165.00,1178.80] vol=2.2x ATR=6.44 |
| Stop hit — per-position SL triggered | 2024-10-10 09:45:00 | 1172.66 | 1175.37 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-10-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:25:00 | 1147.85 | 1157.03 | 0.00 | ORB-short ORB[1150.35,1164.40] vol=3.0x ATR=6.16 |
| Stop hit — per-position SL triggered | 2024-10-15 10:30:00 | 1154.01 | 1156.39 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-10-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:55:00 | 1207.20 | 1215.88 | 0.00 | ORB-short ORB[1217.10,1231.30] vol=1.7x ATR=4.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:20:00 | 1200.77 | 1214.77 | 0.00 | T1 1.5R @ 1200.77 |
| Target hit | 2024-10-17 15:20:00 | 1175.00 | 1195.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2024-10-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-30 11:00:00 | 1083.20 | 1092.45 | 0.00 | ORB-short ORB[1089.05,1102.00] vol=4.5x ATR=4.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 11:10:00 | 1075.75 | 1089.96 | 0.00 | T1 1.5R @ 1075.75 |
| Stop hit — per-position SL triggered | 2024-10-30 12:15:00 | 1083.20 | 1082.22 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:45:00 | 1149.75 | 1142.10 | 0.00 | ORB-long ORB[1124.75,1139.95] vol=1.5x ATR=8.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 11:25:00 | 1162.06 | 1149.87 | 0.00 | T1 1.5R @ 1162.06 |
| Target hit | 2024-10-31 15:20:00 | 1208.50 | 1198.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 10:15:00 | 1228.15 | 1232.68 | 0.00 | ORB-short ORB[1231.20,1243.95] vol=2.0x ATR=5.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 10:35:00 | 1220.02 | 1231.23 | 0.00 | T1 1.5R @ 1220.02 |
| Stop hit — per-position SL triggered | 2024-11-06 10:50:00 | 1228.15 | 1230.50 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-11-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 09:40:00 | 1234.60 | 1226.61 | 0.00 | ORB-long ORB[1218.10,1230.00] vol=1.9x ATR=5.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 10:15:00 | 1243.16 | 1237.34 | 0.00 | T1 1.5R @ 1243.16 |
| Target hit | 2024-11-12 10:30:00 | 1237.65 | 1244.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — SELL (started 2024-11-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 09:55:00 | 1136.95 | 1141.64 | 0.00 | ORB-short ORB[1138.85,1155.00] vol=1.8x ATR=6.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 11:20:00 | 1127.96 | 1138.25 | 0.00 | T1 1.5R @ 1127.96 |
| Stop hit — per-position SL triggered | 2024-11-27 12:35:00 | 1136.95 | 1135.82 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:50:00 | 1151.00 | 1146.95 | 0.00 | ORB-long ORB[1137.05,1150.60] vol=1.5x ATR=4.39 |
| Stop hit — per-position SL triggered | 2024-11-28 10:30:00 | 1146.61 | 1148.04 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-12-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:35:00 | 1230.85 | 1225.07 | 0.00 | ORB-long ORB[1215.50,1227.45] vol=2.0x ATR=3.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 10:45:00 | 1236.58 | 1227.05 | 0.00 | T1 1.5R @ 1236.58 |
| Stop hit — per-position SL triggered | 2024-12-04 11:55:00 | 1230.85 | 1231.18 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 09:30:00 | 1206.95 | 1212.76 | 0.00 | ORB-short ORB[1209.55,1221.00] vol=2.1x ATR=4.56 |
| Stop hit — per-position SL triggered | 2024-12-09 09:35:00 | 1211.51 | 1211.82 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-12-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:50:00 | 1175.40 | 1165.40 | 0.00 | ORB-long ORB[1156.95,1171.10] vol=1.8x ATR=4.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 10:55:00 | 1181.74 | 1166.52 | 0.00 | T1 1.5R @ 1181.74 |
| Stop hit — per-position SL triggered | 2024-12-10 11:25:00 | 1175.40 | 1169.55 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:30:00 | 1064.50 | 1075.61 | 0.00 | ORB-short ORB[1078.00,1091.60] vol=3.1x ATR=4.70 |
| Stop hit — per-position SL triggered | 2024-12-17 10:35:00 | 1069.20 | 1075.23 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:15:00 | 1088.95 | 1093.16 | 0.00 | ORB-short ORB[1094.00,1100.00] vol=2.8x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:20:00 | 1084.47 | 1092.58 | 0.00 | T1 1.5R @ 1084.47 |
| Stop hit — per-position SL triggered | 2024-12-20 10:25:00 | 1088.95 | 1092.00 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:00:00 | 1085.05 | 1088.49 | 0.00 | ORB-short ORB[1090.00,1105.00] vol=6.8x ATR=4.69 |
| Stop hit — per-position SL triggered | 2024-12-26 11:10:00 | 1089.74 | 1088.34 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-01-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:20:00 | 1090.70 | 1088.43 | 0.00 | ORB-long ORB[1081.30,1089.45] vol=1.7x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:50:00 | 1098.27 | 1090.09 | 0.00 | T1 1.5R @ 1098.27 |
| Target hit | 2025-01-02 11:40:00 | 1096.90 | 1098.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2025-01-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:55:00 | 930.30 | 939.99 | 0.00 | ORB-short ORB[938.60,951.50] vol=2.4x ATR=3.39 |
| Stop hit — per-position SL triggered | 2025-01-16 11:05:00 | 933.69 | 939.83 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-01-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:55:00 | 982.80 | 990.24 | 0.00 | ORB-short ORB[987.35,999.70] vol=2.2x ATR=6.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:20:00 | 973.70 | 987.88 | 0.00 | T1 1.5R @ 973.70 |
| Target hit | 2025-01-21 15:20:00 | 943.60 | 971.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:55:00 | 956.95 | 947.94 | 0.00 | ORB-long ORB[930.20,941.40] vol=4.5x ATR=6.13 |
| Stop hit — per-position SL triggered | 2025-01-23 15:20:00 | 954.20 | 956.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2025-01-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:10:00 | 885.35 | 898.73 | 0.00 | ORB-short ORB[905.45,917.00] vol=2.1x ATR=6.53 |
| Stop hit — per-position SL triggered | 2025-01-27 12:20:00 | 891.88 | 891.22 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-03-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-05 10:10:00 | 908.50 | 919.46 | 0.00 | ORB-short ORB[916.00,927.00] vol=1.7x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 10:35:00 | 902.13 | 915.26 | 0.00 | T1 1.5R @ 902.13 |
| Stop hit — per-position SL triggered | 2025-03-05 12:30:00 | 908.50 | 907.87 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 09:30:00 | 882.45 | 882.47 | 0.00 | ORB-short ORB[882.70,891.00] vol=4.9x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 10:15:00 | 875.41 | 881.79 | 0.00 | T1 1.5R @ 875.41 |
| Target hit | 2025-03-12 15:20:00 | 861.70 | 866.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2025-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:45:00 | 932.00 | 923.17 | 0.00 | ORB-long ORB[915.25,926.00] vol=3.7x ATR=3.82 |
| Stop hit — per-position SL triggered | 2025-04-23 09:55:00 | 928.18 | 925.75 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-04-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:10:00 | 941.15 | 929.60 | 0.00 | ORB-long ORB[922.75,935.00] vol=3.8x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 10:20:00 | 946.75 | 932.23 | 0.00 | T1 1.5R @ 946.75 |
| Target hit | 2025-04-24 15:00:00 | 952.15 | 953.16 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-17 10:40:00 | 730.40 | 2024-05-17 10:50:00 | 727.55 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-05-27 09:45:00 | 701.20 | 2024-05-27 09:55:00 | 696.98 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-05-27 09:45:00 | 701.20 | 2024-05-27 10:05:00 | 701.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-29 09:40:00 | 708.90 | 2024-05-29 10:00:00 | 713.38 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-05-29 09:40:00 | 708.90 | 2024-05-29 10:45:00 | 716.05 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2024-06-11 09:50:00 | 751.75 | 2024-06-11 09:55:00 | 756.89 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-06-11 09:50:00 | 751.75 | 2024-06-11 10:00:00 | 751.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-12 11:00:00 | 747.00 | 2024-06-12 11:05:00 | 750.05 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-06-12 11:00:00 | 747.00 | 2024-06-12 11:15:00 | 747.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-13 10:55:00 | 744.60 | 2024-06-13 11:05:00 | 742.62 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-14 09:35:00 | 759.20 | 2024-06-14 14:20:00 | 755.18 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-06-21 10:05:00 | 732.00 | 2024-06-21 10:15:00 | 728.21 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-06-21 10:05:00 | 732.00 | 2024-06-21 10:25:00 | 732.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 11:15:00 | 733.50 | 2024-06-25 11:25:00 | 735.08 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-02 11:10:00 | 733.50 | 2024-07-02 11:40:00 | 730.27 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-07-02 11:10:00 | 733.50 | 2024-07-02 12:15:00 | 733.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-03 09:50:00 | 732.00 | 2024-07-03 09:55:00 | 729.10 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-03 09:50:00 | 732.00 | 2024-07-03 10:20:00 | 732.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 10:10:00 | 722.05 | 2024-07-10 10:35:00 | 716.96 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-07-10 10:10:00 | 722.05 | 2024-07-10 10:55:00 | 722.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 09:50:00 | 724.45 | 2024-07-12 09:55:00 | 726.86 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-15 09:30:00 | 714.05 | 2024-07-15 09:35:00 | 716.62 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-23 10:25:00 | 712.50 | 2024-07-23 11:00:00 | 716.27 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-07-24 09:50:00 | 733.15 | 2024-07-24 11:10:00 | 729.27 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-07-25 10:05:00 | 715.00 | 2024-07-25 11:05:00 | 717.39 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-26 10:40:00 | 725.00 | 2024-07-26 13:05:00 | 727.64 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-22 09:55:00 | 898.60 | 2024-08-22 10:40:00 | 903.61 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2024-08-23 09:30:00 | 887.95 | 2024-08-23 10:25:00 | 891.16 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-29 09:50:00 | 926.90 | 2024-08-29 10:00:00 | 931.75 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-08-29 09:50:00 | 926.90 | 2024-08-29 10:10:00 | 926.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-02 11:15:00 | 913.95 | 2024-09-02 11:35:00 | 916.65 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-12 10:45:00 | 1038.15 | 2024-09-12 10:50:00 | 1043.57 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-09-12 10:45:00 | 1038.15 | 2024-09-12 10:55:00 | 1038.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-23 10:20:00 | 1220.45 | 2024-09-23 11:25:00 | 1213.64 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-10-10 09:35:00 | 1179.10 | 2024-10-10 09:45:00 | 1172.66 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2024-10-15 10:25:00 | 1147.85 | 2024-10-15 10:30:00 | 1154.01 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-10-17 10:55:00 | 1207.20 | 2024-10-17 11:20:00 | 1200.77 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-10-17 10:55:00 | 1207.20 | 2024-10-17 15:20:00 | 1175.00 | TARGET_HIT | 0.50 | 2.67% |
| SELL | retest1 | 2024-10-30 11:00:00 | 1083.20 | 2024-10-30 11:10:00 | 1075.75 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-10-30 11:00:00 | 1083.20 | 2024-10-30 12:15:00 | 1083.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 09:45:00 | 1149.75 | 2024-10-31 11:25:00 | 1162.06 | PARTIAL | 0.50 | 1.07% |
| BUY | retest1 | 2024-10-31 09:45:00 | 1149.75 | 2024-10-31 15:20:00 | 1208.50 | TARGET_HIT | 0.50 | 5.11% |
| SELL | retest1 | 2024-11-06 10:15:00 | 1228.15 | 2024-11-06 10:35:00 | 1220.02 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-11-06 10:15:00 | 1228.15 | 2024-11-06 10:50:00 | 1228.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-12 09:40:00 | 1234.60 | 2024-11-12 10:15:00 | 1243.16 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-11-12 09:40:00 | 1234.60 | 2024-11-12 10:30:00 | 1237.65 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2024-11-27 09:55:00 | 1136.95 | 2024-11-27 11:20:00 | 1127.96 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2024-11-27 09:55:00 | 1136.95 | 2024-11-27 12:35:00 | 1136.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-28 09:50:00 | 1151.00 | 2024-11-28 10:30:00 | 1146.61 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-04 10:35:00 | 1230.85 | 2024-12-04 10:45:00 | 1236.58 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-12-04 10:35:00 | 1230.85 | 2024-12-04 11:55:00 | 1230.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-09 09:30:00 | 1206.95 | 2024-12-09 09:35:00 | 1211.51 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-10 10:50:00 | 1175.40 | 2024-12-10 10:55:00 | 1181.74 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-12-10 10:50:00 | 1175.40 | 2024-12-10 11:25:00 | 1175.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-17 10:30:00 | 1064.50 | 2024-12-17 10:35:00 | 1069.20 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-12-20 10:15:00 | 1088.95 | 2024-12-20 10:20:00 | 1084.47 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-20 10:15:00 | 1088.95 | 2024-12-20 10:25:00 | 1088.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 11:00:00 | 1085.05 | 2024-12-26 11:10:00 | 1089.74 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-01-02 10:20:00 | 1090.70 | 2025-01-02 10:50:00 | 1098.27 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-01-02 10:20:00 | 1090.70 | 2025-01-02 11:40:00 | 1096.90 | TARGET_HIT | 0.50 | 0.57% |
| SELL | retest1 | 2025-01-16 10:55:00 | 930.30 | 2025-01-16 11:05:00 | 933.69 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-01-21 09:55:00 | 982.80 | 2025-01-21 10:20:00 | 973.70 | PARTIAL | 0.50 | 0.93% |
| SELL | retest1 | 2025-01-21 09:55:00 | 982.80 | 2025-01-21 15:20:00 | 943.60 | TARGET_HIT | 0.50 | 3.99% |
| BUY | retest1 | 2025-01-23 09:55:00 | 956.95 | 2025-01-23 15:20:00 | 954.20 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-27 10:10:00 | 885.35 | 2025-01-27 12:20:00 | 891.88 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest1 | 2025-03-05 10:10:00 | 908.50 | 2025-03-05 10:35:00 | 902.13 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2025-03-05 10:10:00 | 908.50 | 2025-03-05 12:30:00 | 908.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-12 09:30:00 | 882.45 | 2025-03-12 10:15:00 | 875.41 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2025-03-12 09:30:00 | 882.45 | 2025-03-12 15:20:00 | 861.70 | TARGET_HIT | 0.50 | 2.35% |
| BUY | retest1 | 2025-04-23 09:45:00 | 932.00 | 2025-04-23 09:55:00 | 928.18 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-04-24 10:10:00 | 941.15 | 2025-04-24 10:20:00 | 946.75 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-04-24 10:10:00 | 941.15 | 2025-04-24 15:00:00 | 952.15 | TARGET_HIT | 0.50 | 1.17% |
