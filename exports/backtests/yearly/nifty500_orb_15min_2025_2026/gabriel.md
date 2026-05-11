# Gabriel India Ltd. (GABRIEL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1136.50
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
| ENTRY1 | 32 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 6 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 26
- **Target hits / Stop hits / Partials:** 6 / 26 / 12
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 11.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 7 | 33.3% | 3 | 14 | 4 | 0.16% | 3.4% |
| BUY @ 2nd Alert (retest1) | 21 | 7 | 33.3% | 3 | 14 | 4 | 0.16% | 3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 23 | 11 | 47.8% | 3 | 12 | 8 | 0.33% | 7.7% |
| SELL @ 2nd Alert (retest1) | 23 | 11 | 47.8% | 3 | 12 | 8 | 0.33% | 7.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 44 | 18 | 40.9% | 6 | 26 | 12 | 0.25% | 11.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 10:25:00 | 650.35 | 644.01 | 0.00 | ORB-long ORB[638.55,646.60] vol=1.7x ATR=2.74 |
| Stop hit — per-position SL triggered | 2025-05-30 12:45:00 | 647.61 | 647.72 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-06-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:45:00 | 637.95 | 634.64 | 0.00 | ORB-long ORB[631.25,636.70] vol=2.9x ATR=2.02 |
| Stop hit — per-position SL triggered | 2025-06-10 09:55:00 | 635.93 | 634.93 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-06-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:00:00 | 632.40 | 636.97 | 0.00 | ORB-short ORB[635.00,643.15] vol=2.1x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:35:00 | 629.78 | 635.55 | 0.00 | T1 1.5R @ 629.78 |
| Stop hit — per-position SL triggered | 2025-06-12 12:40:00 | 632.40 | 634.00 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 10:50:00 | 617.05 | 613.05 | 0.00 | ORB-long ORB[605.25,614.40] vol=1.8x ATR=2.15 |
| Stop hit — per-position SL triggered | 2025-06-13 11:15:00 | 614.90 | 613.72 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:40:00 | 614.75 | 611.37 | 0.00 | ORB-long ORB[606.80,612.30] vol=3.2x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 10:05:00 | 618.23 | 613.21 | 0.00 | T1 1.5R @ 618.23 |
| Target hit | 2025-06-25 15:20:00 | 636.00 | 628.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 11:15:00 | 1261.70 | 1246.06 | 0.00 | ORB-long ORB[1228.00,1244.70] vol=2.2x ATR=4.55 |
| Stop hit — per-position SL triggered | 2025-10-03 11:25:00 | 1257.15 | 1246.94 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-10-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 11:10:00 | 1276.60 | 1287.57 | 0.00 | ORB-short ORB[1280.00,1298.80] vol=2.0x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 12:20:00 | 1268.31 | 1284.59 | 0.00 | T1 1.5R @ 1268.31 |
| Stop hit — per-position SL triggered | 2025-10-10 12:40:00 | 1276.60 | 1283.95 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-10-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 10:05:00 | 1271.00 | 1277.35 | 0.00 | ORB-short ORB[1274.60,1288.00] vol=1.6x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 15:00:00 | 1260.59 | 1270.81 | 0.00 | T1 1.5R @ 1260.59 |
| Target hit | 2025-10-23 15:20:00 | 1245.00 | 1261.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2025-10-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 11:00:00 | 1258.90 | 1246.53 | 0.00 | ORB-long ORB[1236.90,1246.60] vol=1.5x ATR=3.88 |
| Stop hit — per-position SL triggered | 2025-10-28 11:40:00 | 1255.02 | 1250.53 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-10-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:00:00 | 1269.00 | 1277.12 | 0.00 | ORB-short ORB[1278.00,1286.00] vol=1.9x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:10:00 | 1263.20 | 1274.86 | 0.00 | T1 1.5R @ 1263.20 |
| Target hit | 2025-10-30 12:05:00 | 1265.10 | 1265.10 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — SELL (started 2025-11-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 09:50:00 | 1213.70 | 1224.70 | 0.00 | ORB-short ORB[1223.90,1236.00] vol=1.8x ATR=4.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 10:00:00 | 1206.69 | 1221.16 | 0.00 | T1 1.5R @ 1206.69 |
| Target hit | 2025-11-14 15:20:00 | 1167.50 | 1182.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-12-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:40:00 | 1007.00 | 999.54 | 0.00 | ORB-long ORB[990.20,1005.00] vol=1.7x ATR=4.16 |
| Stop hit — per-position SL triggered | 2025-12-04 09:45:00 | 1002.84 | 1000.41 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-12-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 09:45:00 | 958.20 | 961.85 | 0.00 | ORB-short ORB[958.70,969.10] vol=2.4x ATR=3.69 |
| Stop hit — per-position SL triggered | 2025-12-12 09:55:00 | 961.89 | 961.55 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-12-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:45:00 | 979.60 | 965.11 | 0.00 | ORB-long ORB[957.60,967.80] vol=3.2x ATR=4.49 |
| Stop hit — per-position SL triggered | 2025-12-16 09:55:00 | 975.11 | 968.80 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-12-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:30:00 | 946.90 | 952.27 | 0.00 | ORB-short ORB[950.00,960.20] vol=1.6x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:35:00 | 942.34 | 950.47 | 0.00 | T1 1.5R @ 942.34 |
| Stop hit — per-position SL triggered | 2025-12-18 10:15:00 | 946.90 | 946.69 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-12-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-31 10:00:00 | 995.10 | 1002.00 | 0.00 | ORB-short ORB[998.20,1010.80] vol=1.7x ATR=4.70 |
| Stop hit — per-position SL triggered | 2025-12-31 10:35:00 | 999.80 | 1000.21 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-01-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:35:00 | 1058.90 | 1054.17 | 0.00 | ORB-long ORB[1041.00,1055.90] vol=8.3x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:55:00 | 1066.33 | 1058.00 | 0.00 | T1 1.5R @ 1066.33 |
| Target hit | 2026-01-05 11:25:00 | 1066.50 | 1069.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 1021.50 | 1029.56 | 0.00 | ORB-short ORB[1029.00,1040.00] vol=1.8x ATR=3.23 |
| Stop hit — per-position SL triggered | 2026-01-08 11:40:00 | 1024.73 | 1027.39 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-01-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:30:00 | 939.90 | 944.71 | 0.00 | ORB-short ORB[940.50,953.40] vol=2.4x ATR=5.47 |
| Stop hit — per-position SL triggered | 2026-01-14 09:35:00 | 945.37 | 944.59 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-01-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 10:55:00 | 929.70 | 918.95 | 0.00 | ORB-long ORB[911.00,922.90] vol=3.6x ATR=3.26 |
| Stop hit — per-position SL triggered | 2026-01-19 11:00:00 | 926.44 | 919.18 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-01-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:35:00 | 908.50 | 915.87 | 0.00 | ORB-short ORB[913.40,927.00] vol=1.6x ATR=4.16 |
| Stop hit — per-position SL triggered | 2026-01-20 09:40:00 | 912.66 | 915.67 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:45:00 | 1000.80 | 989.14 | 0.00 | ORB-long ORB[970.00,982.15] vol=4.3x ATR=5.56 |
| Stop hit — per-position SL triggered | 2026-02-10 09:50:00 | 995.24 | 989.67 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 967.50 | 972.52 | 0.00 | ORB-short ORB[969.30,978.20] vol=2.0x ATR=2.87 |
| Stop hit — per-position SL triggered | 2026-02-19 09:40:00 | 970.37 | 972.23 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2026-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:40:00 | 978.25 | 972.08 | 0.00 | ORB-long ORB[965.80,973.70] vol=4.2x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 09:45:00 | 983.54 | 974.62 | 0.00 | T1 1.5R @ 983.54 |
| Target hit | 2026-02-25 12:35:00 | 998.35 | 1000.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — BUY (started 2026-03-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:45:00 | 947.90 | 938.95 | 0.00 | ORB-long ORB[933.25,941.70] vol=1.7x ATR=5.22 |
| Stop hit — per-position SL triggered | 2026-03-06 10:20:00 | 942.68 | 941.23 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2026-03-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:55:00 | 916.00 | 905.71 | 0.00 | ORB-long ORB[898.95,911.30] vol=1.5x ATR=3.94 |
| Stop hit — per-position SL triggered | 2026-03-11 10:10:00 | 912.06 | 909.36 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:15:00 | 881.35 | 874.32 | 0.00 | ORB-long ORB[868.85,880.70] vol=2.6x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:40:00 | 886.33 | 876.67 | 0.00 | T1 1.5R @ 886.33 |
| Stop hit — per-position SL triggered | 2026-03-12 11:45:00 | 881.35 | 877.02 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2026-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:55:00 | 861.60 | 867.91 | 0.00 | ORB-short ORB[869.00,878.40] vol=1.6x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:20:00 | 856.44 | 865.94 | 0.00 | T1 1.5R @ 856.44 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 861.60 | 864.16 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 1010.75 | 1019.58 | 0.00 | ORB-short ORB[1015.75,1026.60] vol=3.4x ATR=4.47 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 1015.22 | 1014.74 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2026-04-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:35:00 | 989.90 | 987.43 | 0.00 | ORB-long ORB[976.45,988.90] vol=3.1x ATR=3.94 |
| Stop hit — per-position SL triggered | 2026-04-27 11:00:00 | 985.96 | 987.93 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2026-04-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:25:00 | 1018.55 | 1012.29 | 0.00 | ORB-long ORB[1011.55,1017.00] vol=2.2x ATR=3.31 |
| Stop hit — per-position SL triggered | 2026-04-28 10:30:00 | 1015.24 | 1013.45 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:15:00 | 1022.15 | 1028.32 | 0.00 | ORB-short ORB[1025.60,1037.95] vol=1.8x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:35:00 | 1017.23 | 1027.15 | 0.00 | T1 1.5R @ 1017.23 |
| Stop hit — per-position SL triggered | 2026-04-29 10:50:00 | 1022.15 | 1026.33 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-30 10:25:00 | 650.35 | 2025-05-30 12:45:00 | 647.61 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-06-10 09:45:00 | 637.95 | 2025-06-10 09:55:00 | 635.93 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-06-12 11:00:00 | 632.40 | 2025-06-12 11:35:00 | 629.78 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-06-12 11:00:00 | 632.40 | 2025-06-12 12:40:00 | 632.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-13 10:50:00 | 617.05 | 2025-06-13 11:15:00 | 614.90 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-06-25 09:40:00 | 614.75 | 2025-06-25 10:05:00 | 618.23 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-06-25 09:40:00 | 614.75 | 2025-06-25 15:20:00 | 636.00 | TARGET_HIT | 0.50 | 3.46% |
| BUY | retest1 | 2025-10-03 11:15:00 | 1261.70 | 2025-10-03 11:25:00 | 1257.15 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-10 11:10:00 | 1276.60 | 2025-10-10 12:20:00 | 1268.31 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-10-10 11:10:00 | 1276.60 | 2025-10-10 12:40:00 | 1276.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-23 10:05:00 | 1271.00 | 2025-10-23 15:00:00 | 1260.59 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2025-10-23 10:05:00 | 1271.00 | 2025-10-23 15:20:00 | 1245.00 | TARGET_HIT | 0.50 | 2.05% |
| BUY | retest1 | 2025-10-28 11:00:00 | 1258.90 | 2025-10-28 11:40:00 | 1255.02 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-10-30 10:00:00 | 1269.00 | 2025-10-30 10:10:00 | 1263.20 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-10-30 10:00:00 | 1269.00 | 2025-10-30 12:05:00 | 1265.10 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-11-14 09:50:00 | 1213.70 | 2025-11-14 10:00:00 | 1206.69 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-11-14 09:50:00 | 1213.70 | 2025-11-14 15:20:00 | 1167.50 | TARGET_HIT | 0.50 | 3.81% |
| BUY | retest1 | 2025-12-04 09:40:00 | 1007.00 | 2025-12-04 09:45:00 | 1002.84 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-12-12 09:45:00 | 958.20 | 2025-12-12 09:55:00 | 961.89 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-12-16 09:45:00 | 979.60 | 2025-12-16 09:55:00 | 975.11 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-12-18 09:30:00 | 946.90 | 2025-12-18 09:35:00 | 942.34 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-12-18 09:30:00 | 946.90 | 2025-12-18 10:15:00 | 946.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-31 10:00:00 | 995.10 | 2025-12-31 10:35:00 | 999.80 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-01-05 09:35:00 | 1058.90 | 2026-01-05 09:55:00 | 1066.33 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-01-05 09:35:00 | 1058.90 | 2026-01-05 11:25:00 | 1066.50 | TARGET_HIT | 0.50 | 0.72% |
| SELL | retest1 | 2026-01-08 11:10:00 | 1021.50 | 2026-01-08 11:40:00 | 1024.73 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-14 09:30:00 | 939.90 | 2026-01-14 09:35:00 | 945.37 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-01-19 10:55:00 | 929.70 | 2026-01-19 11:00:00 | 926.44 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-01-20 09:35:00 | 908.50 | 2026-01-20 09:40:00 | 912.66 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-02-10 09:45:00 | 1000.80 | 2026-02-10 09:50:00 | 995.24 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2026-02-19 09:30:00 | 967.50 | 2026-02-19 09:40:00 | 970.37 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-25 09:40:00 | 978.25 | 2026-02-25 09:45:00 | 983.54 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-02-25 09:40:00 | 978.25 | 2026-02-25 12:35:00 | 998.35 | TARGET_HIT | 0.50 | 2.05% |
| BUY | retest1 | 2026-03-06 09:45:00 | 947.90 | 2026-03-06 10:20:00 | 942.68 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2026-03-11 09:55:00 | 916.00 | 2026-03-11 10:10:00 | 912.06 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-12 11:15:00 | 881.35 | 2026-03-12 11:40:00 | 886.33 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-03-12 11:15:00 | 881.35 | 2026-03-12 11:45:00 | 881.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 09:55:00 | 861.60 | 2026-03-13 10:20:00 | 856.44 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-03-13 09:55:00 | 861.60 | 2026-03-13 10:50:00 | 861.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 09:45:00 | 1010.75 | 2026-04-22 11:05:00 | 1015.22 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-27 10:35:00 | 989.90 | 2026-04-27 11:00:00 | 985.96 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-28 10:25:00 | 1018.55 | 2026-04-28 10:30:00 | 1015.24 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-29 10:15:00 | 1022.15 | 2026-04-29 10:35:00 | 1017.23 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-29 10:15:00 | 1022.15 | 2026-04-29 10:50:00 | 1022.15 | STOP_HIT | 0.50 | 0.00% |
