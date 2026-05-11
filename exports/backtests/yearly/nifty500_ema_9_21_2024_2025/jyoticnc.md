# Jyoti CNC Automation Ltd. (JYOTICNC)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 766.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 137 |
| ALERT1 | 102 |
| ALERT2 | 101 |
| ALERT2_SKIP | 57 |
| ALERT3 | 253 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 122 |
| PARTIAL | 12 |
| TARGET_HIT | 23 |
| STOP_HIT | 102 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 85
- **Target hits / Stop hits / Partials:** 23 / 102 / 12
- **Avg / median % per leg:** 1.02% / -0.94%
- **Sum % (uncompounded):** 140.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 23 | 37.1% | 14 | 48 | 0 | 1.48% | 91.8% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.95% | -1.9% |
| BUY @ 3rd Alert (retest2) | 60 | 23 | 38.3% | 14 | 46 | 0 | 1.56% | 93.7% |
| SELL (all) | 75 | 29 | 38.7% | 9 | 54 | 12 | 0.65% | 48.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.22% | -2.2% |
| SELL @ 3rd Alert (retest2) | 74 | 29 | 39.2% | 9 | 53 | 12 | 0.69% | 50.8% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.37% | -4.1% |
| retest2 (combined) | 134 | 52 | 38.8% | 23 | 99 | 12 | 1.08% | 144.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 826.95 | 814.17 | 813.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 846.00 | 822.59 | 817.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 886.60 | 894.04 | 877.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:00:00 | 886.60 | 894.04 | 877.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 896.15 | 901.91 | 894.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 14:45:00 | 893.95 | 901.91 | 894.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 895.95 | 900.72 | 894.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 912.05 | 900.72 | 894.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 11:15:00 | 900.75 | 906.10 | 900.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 11:45:00 | 899.00 | 904.13 | 900.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 12:45:00 | 900.45 | 904.50 | 900.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 899.10 | 903.42 | 900.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:00:00 | 899.10 | 903.42 | 900.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 888.80 | 900.49 | 899.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-21 14:15:00 | 888.80 | 900.49 | 899.51 | SL hit (close<static) qty=1.00 sl=892.90 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 890.00 | 898.40 | 898.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 877.05 | 894.13 | 896.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 877.90 | 869.47 | 877.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 877.90 | 869.47 | 877.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 877.90 | 869.47 | 877.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 892.05 | 869.47 | 877.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 877.50 | 871.08 | 877.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:45:00 | 877.60 | 871.08 | 877.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 878.30 | 872.52 | 877.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:45:00 | 881.45 | 872.52 | 877.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 880.05 | 874.03 | 877.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:00:00 | 880.05 | 874.03 | 877.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 885.95 | 876.41 | 878.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:45:00 | 886.20 | 876.41 | 878.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 15:15:00 | 885.40 | 879.58 | 879.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 10:15:00 | 886.55 | 881.84 | 880.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 14:15:00 | 1011.20 | 1011.83 | 992.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:30:00 | 1014.20 | 1011.83 | 992.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1007.55 | 1018.97 | 1009.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 987.75 | 1018.97 | 1009.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 968.95 | 1008.97 | 1005.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:45:00 | 968.95 | 1008.97 | 1005.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 968.95 | 1000.97 | 1002.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 920.50 | 969.75 | 985.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 962.80 | 941.84 | 959.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 962.80 | 941.84 | 959.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 962.80 | 941.84 | 959.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 959.80 | 941.84 | 959.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 960.05 | 945.48 | 959.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 961.00 | 945.48 | 959.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 960.05 | 948.40 | 959.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:30:00 | 960.80 | 948.40 | 959.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 960.00 | 952.57 | 959.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 965.00 | 952.57 | 959.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 968.90 | 955.84 | 960.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 15:00:00 | 968.90 | 955.84 | 960.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 970.50 | 958.77 | 961.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:15:00 | 1025.45 | 958.77 | 961.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 1048.00 | 976.62 | 969.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 1056.65 | 992.62 | 977.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 1137.65 | 1155.04 | 1117.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 1137.65 | 1155.04 | 1117.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1190.65 | 1171.21 | 1157.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:00:00 | 1236.40 | 1199.52 | 1181.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 12:30:00 | 1221.00 | 1207.39 | 1188.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-19 14:15:00 | 1343.10 | 1290.87 | 1247.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 10:15:00 | 1250.20 | 1285.65 | 1290.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 14:15:00 | 1236.00 | 1260.33 | 1275.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 1289.05 | 1262.01 | 1273.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 1289.05 | 1262.01 | 1273.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 1289.05 | 1262.01 | 1273.32 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 12:15:00 | 1312.40 | 1284.54 | 1281.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 14:15:00 | 1354.95 | 1303.82 | 1291.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 15:15:00 | 1333.00 | 1336.92 | 1319.74 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:15:00 | 1360.00 | 1336.92 | 1319.74 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1361.65 | 1351.78 | 1335.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 1345.45 | 1351.78 | 1335.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1350.15 | 1351.65 | 1338.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 1358.90 | 1351.65 | 1338.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 1349.00 | 1351.12 | 1339.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-28 11:15:00 | 1337.40 | 1348.38 | 1339.33 | SL hit (close<ema400) qty=1.00 sl=1339.33 alert=retest1 |

### Cycle 8 — SELL (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 15:15:00 | 1334.00 | 1337.48 | 1337.82 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 1345.20 | 1339.38 | 1338.64 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 14:15:00 | 1325.95 | 1336.31 | 1337.47 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 1348.15 | 1339.25 | 1338.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 1356.05 | 1343.53 | 1340.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 09:15:00 | 1337.25 | 1344.90 | 1341.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 1337.25 | 1344.90 | 1341.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1337.25 | 1344.90 | 1341.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:00:00 | 1337.25 | 1344.90 | 1341.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 1335.15 | 1342.95 | 1341.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:30:00 | 1335.00 | 1342.95 | 1341.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 12:15:00 | 1331.20 | 1339.16 | 1339.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 09:15:00 | 1328.05 | 1336.34 | 1338.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 09:15:00 | 1218.20 | 1214.50 | 1243.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 1218.20 | 1214.50 | 1243.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1218.20 | 1214.50 | 1243.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 13:45:00 | 1201.45 | 1216.25 | 1235.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 14:30:00 | 1203.00 | 1214.72 | 1232.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 10:00:00 | 1190.25 | 1209.67 | 1220.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 1199.00 | 1210.59 | 1215.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 1209.05 | 1210.28 | 1215.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-15 10:15:00 | 1270.90 | 1222.40 | 1220.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 10:15:00 | 1270.90 | 1222.40 | 1220.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 15:15:00 | 1290.00 | 1255.46 | 1239.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 09:15:00 | 1215.00 | 1247.37 | 1236.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 1215.00 | 1247.37 | 1236.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1215.00 | 1247.37 | 1236.85 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 1213.20 | 1228.34 | 1230.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 1202.55 | 1218.08 | 1224.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 14:15:00 | 1255.00 | 1217.71 | 1221.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 14:15:00 | 1255.00 | 1217.71 | 1221.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1255.00 | 1217.71 | 1221.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 1255.00 | 1217.71 | 1221.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 1235.00 | 1221.17 | 1222.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 1220.00 | 1221.17 | 1222.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 14:15:00 | 1159.00 | 1180.41 | 1199.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 09:15:00 | 1154.00 | 1152.81 | 1170.39 | SL hit (close>ema200) qty=0.50 sl=1152.81 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 1160.10 | 1131.14 | 1129.28 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 11:15:00 | 1129.95 | 1133.54 | 1133.60 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 13:15:00 | 1144.95 | 1134.44 | 1133.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 09:15:00 | 1158.40 | 1141.19 | 1137.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 1144.65 | 1159.11 | 1150.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 1144.65 | 1159.11 | 1150.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1144.65 | 1159.11 | 1150.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 1143.20 | 1159.11 | 1150.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 1148.60 | 1157.01 | 1150.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:00:00 | 1148.60 | 1157.01 | 1150.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 1148.85 | 1155.38 | 1150.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 15:15:00 | 1154.95 | 1152.37 | 1150.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 1108.55 | 1144.02 | 1146.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 1108.55 | 1144.02 | 1146.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 1062.45 | 1117.54 | 1133.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1096.55 | 1094.00 | 1113.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1096.55 | 1094.00 | 1113.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1096.55 | 1094.00 | 1113.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:30:00 | 1082.15 | 1093.70 | 1111.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 1090.00 | 1091.74 | 1106.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:00:00 | 1084.15 | 1084.99 | 1097.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:30:00 | 1087.55 | 1097.80 | 1100.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1080.60 | 1081.13 | 1089.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-09 14:15:00 | 1149.00 | 1094.78 | 1092.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 14:15:00 | 1149.00 | 1094.78 | 1092.38 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 1090.00 | 1096.27 | 1096.91 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 15:15:00 | 1124.00 | 1099.72 | 1096.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 1138.85 | 1107.55 | 1100.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 14:15:00 | 1129.00 | 1129.21 | 1115.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-16 15:00:00 | 1129.00 | 1129.21 | 1115.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 1116.20 | 1127.39 | 1118.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 11:00:00 | 1116.20 | 1127.39 | 1118.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 1115.40 | 1124.99 | 1118.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 11:45:00 | 1116.75 | 1124.99 | 1118.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 1118.95 | 1123.78 | 1118.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 13:30:00 | 1123.40 | 1123.40 | 1118.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 09:15:00 | 1096.10 | 1117.12 | 1116.78 | SL hit (close<static) qty=1.00 sl=1114.90 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 10:15:00 | 1095.75 | 1112.84 | 1114.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 11:15:00 | 1080.15 | 1106.30 | 1111.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 11:15:00 | 1092.40 | 1088.33 | 1097.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 11:15:00 | 1092.40 | 1088.33 | 1097.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 1092.40 | 1088.33 | 1097.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:00:00 | 1092.40 | 1088.33 | 1097.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 1101.70 | 1091.00 | 1097.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:00:00 | 1101.70 | 1091.00 | 1097.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 1125.35 | 1097.87 | 1100.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:00:00 | 1125.35 | 1097.87 | 1100.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 14:15:00 | 1125.75 | 1103.45 | 1102.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 1197.00 | 1125.61 | 1113.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 09:15:00 | 1262.80 | 1278.23 | 1249.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 10:00:00 | 1262.80 | 1278.23 | 1249.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 1270.40 | 1278.68 | 1269.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 1270.40 | 1278.68 | 1269.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 1251.00 | 1273.14 | 1267.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 1266.55 | 1273.14 | 1267.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1270.85 | 1272.68 | 1267.95 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 14:15:00 | 1261.40 | 1265.42 | 1265.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 15:15:00 | 1251.00 | 1262.54 | 1264.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 1215.10 | 1211.98 | 1228.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-03 09:30:00 | 1192.35 | 1211.98 | 1228.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1155.35 | 1164.30 | 1176.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 10:30:00 | 1146.95 | 1163.30 | 1174.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 13:45:00 | 1154.30 | 1159.71 | 1169.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 1130.65 | 1160.03 | 1168.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 14:30:00 | 1139.00 | 1128.34 | 1135.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 1140.25 | 1130.72 | 1136.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 1131.50 | 1130.72 | 1136.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:45:00 | 1135.05 | 1131.78 | 1136.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:45:00 | 1138.50 | 1135.84 | 1137.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:30:00 | 1138.90 | 1135.47 | 1136.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1135.85 | 1135.54 | 1136.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:30:00 | 1136.45 | 1135.54 | 1136.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1129.00 | 1134.24 | 1136.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 1131.70 | 1134.24 | 1136.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1139.00 | 1135.19 | 1136.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:15:00 | 1143.05 | 1135.19 | 1136.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 1144.30 | 1137.01 | 1137.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:30:00 | 1151.00 | 1137.01 | 1137.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-12 11:15:00 | 1152.15 | 1140.04 | 1138.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 11:15:00 | 1152.15 | 1140.04 | 1138.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 12:15:00 | 1177.95 | 1147.62 | 1142.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 1185.60 | 1189.60 | 1176.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 15:00:00 | 1185.60 | 1189.60 | 1176.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 1172.00 | 1186.08 | 1176.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 1172.05 | 1186.08 | 1176.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1182.30 | 1185.32 | 1176.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 14:30:00 | 1200.75 | 1190.42 | 1182.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 15:15:00 | 1178.50 | 1182.76 | 1183.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 1178.50 | 1182.76 | 1183.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 1167.40 | 1179.69 | 1181.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 11:15:00 | 1179.05 | 1163.32 | 1168.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 11:15:00 | 1179.05 | 1163.32 | 1168.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1179.05 | 1163.32 | 1168.86 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 1178.30 | 1171.71 | 1171.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 12:15:00 | 1190.15 | 1178.84 | 1175.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 1166.05 | 1185.43 | 1180.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 1166.05 | 1185.43 | 1180.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1166.05 | 1185.43 | 1180.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 1166.05 | 1185.43 | 1180.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1169.45 | 1182.23 | 1179.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 12:45:00 | 1181.00 | 1180.79 | 1179.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 14:15:00 | 1167.70 | 1176.69 | 1177.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 1167.70 | 1176.69 | 1177.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 15:15:00 | 1163.50 | 1174.05 | 1176.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 10:15:00 | 1170.10 | 1164.88 | 1168.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 10:15:00 | 1170.10 | 1164.88 | 1168.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 1170.10 | 1164.88 | 1168.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 1170.10 | 1164.88 | 1168.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 1165.25 | 1164.96 | 1168.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:30:00 | 1170.50 | 1164.96 | 1168.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 1165.00 | 1164.97 | 1168.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:00:00 | 1160.65 | 1164.10 | 1167.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 1194.65 | 1170.21 | 1169.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 1194.65 | 1170.21 | 1169.83 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 1159.40 | 1168.88 | 1169.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1151.90 | 1162.62 | 1166.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 1164.20 | 1158.30 | 1162.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 1164.20 | 1158.30 | 1162.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1164.20 | 1158.30 | 1162.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 1164.20 | 1158.30 | 1162.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 1156.20 | 1157.88 | 1162.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 14:00:00 | 1147.55 | 1153.95 | 1159.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 1143.00 | 1155.03 | 1158.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 11:15:00 | 1090.17 | 1121.56 | 1132.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 11:15:00 | 1085.85 | 1121.56 | 1132.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-08 09:15:00 | 1032.80 | 1099.28 | 1116.00 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 1123.00 | 1118.36 | 1118.30 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 1118.00 | 1118.93 | 1118.95 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 14:15:00 | 1120.60 | 1118.97 | 1118.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 15:15:00 | 1132.00 | 1121.58 | 1120.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 09:15:00 | 1117.10 | 1120.68 | 1119.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 09:15:00 | 1117.10 | 1120.68 | 1119.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1117.10 | 1120.68 | 1119.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:15:00 | 1112.60 | 1120.68 | 1119.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1116.80 | 1119.90 | 1119.58 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 1116.50 | 1119.22 | 1119.30 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 12:15:00 | 1120.35 | 1119.45 | 1119.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 13:15:00 | 1130.10 | 1121.58 | 1120.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 10:15:00 | 1127.00 | 1127.47 | 1124.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 10:15:00 | 1127.00 | 1127.47 | 1124.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1127.00 | 1127.47 | 1124.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:30:00 | 1126.25 | 1127.47 | 1124.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 1118.00 | 1125.58 | 1123.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:45:00 | 1118.05 | 1125.58 | 1123.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 1106.15 | 1121.69 | 1121.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 09:15:00 | 1100.75 | 1111.76 | 1116.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 13:15:00 | 1117.90 | 1110.26 | 1114.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 13:15:00 | 1117.90 | 1110.26 | 1114.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 1117.90 | 1110.26 | 1114.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:45:00 | 1115.50 | 1110.26 | 1114.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 1130.00 | 1114.21 | 1115.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:45:00 | 1124.70 | 1114.21 | 1115.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 15:15:00 | 1127.35 | 1116.84 | 1116.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 1135.30 | 1120.53 | 1118.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 13:15:00 | 1123.20 | 1123.85 | 1120.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 14:00:00 | 1123.20 | 1123.85 | 1120.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 1117.65 | 1122.61 | 1120.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:15:00 | 1115.75 | 1122.61 | 1120.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 1115.75 | 1121.24 | 1120.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 1111.90 | 1121.24 | 1120.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 1104.95 | 1117.98 | 1118.78 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 14:15:00 | 1144.00 | 1122.44 | 1120.20 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 14:15:00 | 1117.95 | 1120.27 | 1120.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 1107.70 | 1114.70 | 1117.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 15:15:00 | 1023.70 | 1021.10 | 1038.43 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:30:00 | 993.05 | 1015.86 | 1034.47 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 1004.20 | 1001.26 | 1012.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-28 12:15:00 | 1015.05 | 1004.02 | 1013.09 | SL hit (close>ema400) qty=1.00 sl=1013.09 alert=retest1 |

### Cycle 41 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 1021.55 | 1014.42 | 1014.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 1030.45 | 1018.68 | 1016.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 11:15:00 | 1038.15 | 1038.72 | 1030.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:30:00 | 1038.90 | 1038.72 | 1030.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1033.90 | 1037.76 | 1031.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:30:00 | 1030.85 | 1037.76 | 1031.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 1036.90 | 1037.59 | 1031.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:45:00 | 1034.70 | 1037.59 | 1031.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1040.35 | 1047.68 | 1040.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:45:00 | 1040.40 | 1047.68 | 1040.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1054.90 | 1049.12 | 1041.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:00:00 | 1058.55 | 1051.94 | 1044.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 1061.85 | 1053.03 | 1046.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 1064.35 | 1055.13 | 1047.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-07 12:15:00 | 1164.40 | 1134.97 | 1107.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 1116.00 | 1123.41 | 1123.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 1103.10 | 1119.35 | 1121.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1083.25 | 1071.73 | 1085.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1083.25 | 1071.73 | 1085.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1083.25 | 1071.73 | 1085.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:45:00 | 1083.45 | 1071.73 | 1085.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1067.00 | 1070.78 | 1083.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 1085.70 | 1070.78 | 1083.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1099.90 | 1076.56 | 1083.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:45:00 | 1092.05 | 1076.56 | 1083.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 1120.00 | 1085.25 | 1086.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:30:00 | 1123.40 | 1085.25 | 1086.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 15:15:00 | 1100.00 | 1088.20 | 1087.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 1194.00 | 1117.19 | 1104.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 14:15:00 | 1222.05 | 1226.92 | 1196.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-22 15:00:00 | 1222.05 | 1226.92 | 1196.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 1213.90 | 1226.26 | 1213.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:30:00 | 1226.50 | 1224.05 | 1213.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 12:15:00 | 1203.05 | 1215.94 | 1212.21 | SL hit (close<static) qty=1.00 sl=1205.05 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 1322.75 | 1336.24 | 1338.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 1313.95 | 1327.56 | 1333.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 1326.00 | 1318.44 | 1326.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 1326.00 | 1318.44 | 1326.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1326.00 | 1318.44 | 1326.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:30:00 | 1328.15 | 1318.44 | 1326.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 1337.20 | 1322.20 | 1327.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:00:00 | 1337.20 | 1322.20 | 1327.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 1324.70 | 1322.70 | 1327.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 12:30:00 | 1311.60 | 1321.50 | 1326.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:15:00 | 1246.02 | 1271.99 | 1290.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-16 09:15:00 | 1299.90 | 1271.15 | 1281.21 | SL hit (close>ema200) qty=0.50 sl=1271.15 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 1309.75 | 1286.92 | 1286.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 1334.05 | 1302.11 | 1294.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 15:15:00 | 1400.00 | 1404.99 | 1374.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-19 09:15:00 | 1391.30 | 1404.99 | 1374.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1385.35 | 1401.06 | 1375.79 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 1357.95 | 1372.93 | 1373.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 1345.90 | 1367.52 | 1370.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 1364.90 | 1355.34 | 1362.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 10:15:00 | 1364.90 | 1355.34 | 1362.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1364.90 | 1355.34 | 1362.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 1364.90 | 1355.34 | 1362.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 1351.95 | 1354.66 | 1361.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:45:00 | 1340.40 | 1352.29 | 1356.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:00:00 | 1344.35 | 1345.60 | 1351.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 15:00:00 | 1344.70 | 1345.18 | 1350.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 09:30:00 | 1341.00 | 1343.83 | 1348.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 1352.90 | 1345.08 | 1348.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:30:00 | 1349.50 | 1345.08 | 1348.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-27 13:15:00 | 1373.20 | 1350.70 | 1350.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 1373.20 | 1350.70 | 1350.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 1404.00 | 1367.48 | 1358.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 10:15:00 | 1365.05 | 1366.99 | 1359.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 11:00:00 | 1365.05 | 1366.99 | 1359.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 1390.75 | 1371.74 | 1362.08 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 12:15:00 | 1350.00 | 1359.30 | 1360.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 09:15:00 | 1339.90 | 1352.64 | 1356.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 09:15:00 | 1356.65 | 1341.57 | 1347.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 1356.65 | 1341.57 | 1347.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1356.65 | 1341.57 | 1347.25 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 1372.85 | 1354.65 | 1352.38 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 1355.50 | 1361.75 | 1362.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 1344.55 | 1356.95 | 1359.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1379.05 | 1360.26 | 1360.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 1379.05 | 1360.26 | 1360.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1379.05 | 1360.26 | 1360.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 1366.45 | 1360.26 | 1360.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1361.80 | 1360.57 | 1360.76 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 13:15:00 | 1365.00 | 1361.62 | 1361.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 14:15:00 | 1366.60 | 1362.61 | 1361.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 15:15:00 | 1361.00 | 1362.29 | 1361.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 15:15:00 | 1361.00 | 1362.29 | 1361.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 1361.00 | 1362.29 | 1361.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 1350.05 | 1362.29 | 1361.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 1333.00 | 1356.43 | 1359.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 12:15:00 | 1325.80 | 1340.19 | 1348.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 11:15:00 | 1210.25 | 1193.79 | 1229.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 12:00:00 | 1210.25 | 1193.79 | 1229.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1209.60 | 1203.72 | 1221.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 10:15:00 | 1195.00 | 1203.72 | 1221.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:00:00 | 1195.05 | 1201.98 | 1218.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:30:00 | 1193.25 | 1200.41 | 1216.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 1259.55 | 1212.93 | 1214.47 | SL hit (close>static) qty=1.00 sl=1233.30 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1248.50 | 1220.05 | 1217.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 09:15:00 | 1272.15 | 1238.42 | 1229.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 1257.90 | 1263.74 | 1248.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 1257.90 | 1263.74 | 1248.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 1247.50 | 1258.21 | 1250.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:45:00 | 1247.30 | 1258.21 | 1250.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 1258.00 | 1258.17 | 1251.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:30:00 | 1239.20 | 1258.17 | 1251.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 1252.05 | 1256.95 | 1251.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 1224.55 | 1256.95 | 1251.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1232.45 | 1252.05 | 1249.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 1228.00 | 1252.05 | 1249.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 1228.75 | 1247.39 | 1247.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 1192.30 | 1236.37 | 1242.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1241.75 | 1225.73 | 1233.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1241.75 | 1225.73 | 1233.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1241.75 | 1225.73 | 1233.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 1240.30 | 1225.73 | 1233.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1249.30 | 1230.44 | 1234.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 1245.70 | 1230.44 | 1234.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1054.40 | 1017.15 | 1059.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1054.40 | 1017.15 | 1059.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1042.90 | 1022.30 | 1058.21 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 1077.20 | 1067.44 | 1066.24 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 14:15:00 | 1043.30 | 1064.26 | 1065.32 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 1080.30 | 1066.77 | 1066.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 1085.20 | 1070.46 | 1067.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 12:15:00 | 1071.40 | 1072.69 | 1069.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 12:15:00 | 1071.40 | 1072.69 | 1069.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 1071.40 | 1072.69 | 1069.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:45:00 | 1070.10 | 1072.69 | 1069.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 1067.55 | 1071.66 | 1069.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:00:00 | 1067.55 | 1071.66 | 1069.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 1074.55 | 1072.24 | 1069.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 15:15:00 | 1079.00 | 1072.24 | 1069.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 11:15:00 | 1057.15 | 1073.59 | 1071.77 | SL hit (close<static) qty=1.00 sl=1064.15 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1044.95 | 1067.86 | 1069.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 14:15:00 | 1030.00 | 1056.36 | 1063.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1019.95 | 1009.97 | 1028.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 10:15:00 | 1020.30 | 1009.97 | 1028.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 1043.25 | 1015.60 | 1023.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 1043.25 | 1015.60 | 1023.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 1035.55 | 1019.59 | 1024.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 1071.40 | 1019.59 | 1024.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 1085.20 | 1032.71 | 1030.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 1105.00 | 1047.17 | 1037.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 1032.10 | 1118.84 | 1112.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 1032.10 | 1118.84 | 1112.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1032.10 | 1118.84 | 1112.10 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 1073.50 | 1102.43 | 1105.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1054.15 | 1089.18 | 1097.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 1075.75 | 1073.48 | 1085.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 15:00:00 | 1075.75 | 1073.48 | 1085.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1070.20 | 1070.83 | 1081.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 1070.45 | 1070.83 | 1081.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1093.50 | 1075.37 | 1082.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:30:00 | 1086.50 | 1075.37 | 1082.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1071.10 | 1074.51 | 1081.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 13:45:00 | 1056.00 | 1070.47 | 1078.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 15:15:00 | 1055.20 | 1069.95 | 1077.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:15:00 | 1057.00 | 1068.92 | 1073.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 1003.20 | 1048.34 | 1062.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 1002.44 | 1048.34 | 1062.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 1004.15 | 1048.34 | 1062.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-14 13:15:00 | 950.40 | 1010.33 | 1039.92 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 857.95 | 808.58 | 805.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 915.60 | 857.04 | 842.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-11 09:15:00 | 929.70 | 966.26 | 945.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 929.70 | 966.26 | 945.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 929.70 | 966.26 | 945.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:45:00 | 952.00 | 960.37 | 944.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 13:45:00 | 956.15 | 949.68 | 942.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 14:15:00 | 960.80 | 949.68 | 942.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-17 10:15:00 | 1047.20 | 1007.81 | 987.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 1067.75 | 1092.53 | 1095.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 10:15:00 | 1054.85 | 1084.99 | 1091.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 11:15:00 | 1060.60 | 1056.54 | 1069.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 12:00:00 | 1060.60 | 1056.54 | 1069.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1058.60 | 1058.00 | 1067.66 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 13:15:00 | 1075.00 | 1070.05 | 1069.97 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 14:15:00 | 1055.80 | 1067.20 | 1068.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 1051.80 | 1060.81 | 1065.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 13:15:00 | 1069.15 | 1060.87 | 1063.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 13:15:00 | 1069.15 | 1060.87 | 1063.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 1069.15 | 1060.87 | 1063.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:00:00 | 1069.15 | 1060.87 | 1063.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 1068.10 | 1062.31 | 1064.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 1068.10 | 1062.31 | 1064.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 15:15:00 | 1080.00 | 1065.85 | 1065.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 11:15:00 | 1082.80 | 1071.46 | 1068.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 1077.40 | 1078.57 | 1073.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 1077.40 | 1078.57 | 1073.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1077.40 | 1078.57 | 1073.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:15:00 | 1074.90 | 1078.57 | 1073.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 1071.80 | 1077.22 | 1073.40 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 14:15:00 | 1054.05 | 1068.62 | 1070.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 10:15:00 | 1040.50 | 1059.73 | 1065.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 978.45 | 978.04 | 1007.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1016.60 | 978.04 | 1007.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 999.60 | 982.35 | 1006.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 1000.10 | 982.35 | 1006.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1010.70 | 978.32 | 984.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:30:00 | 1019.25 | 978.32 | 984.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 1009.35 | 990.50 | 988.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 15:15:00 | 1012.95 | 1000.16 | 994.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1064.00 | 1070.38 | 1052.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 1064.00 | 1070.38 | 1052.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1063.50 | 1070.75 | 1062.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:45:00 | 1088.40 | 1071.03 | 1065.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 11:15:00 | 1080.70 | 1098.35 | 1097.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 1082.30 | 1095.14 | 1096.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1082.30 | 1095.14 | 1096.21 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 1102.00 | 1093.48 | 1093.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 10:15:00 | 1114.50 | 1097.69 | 1095.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 12:15:00 | 1099.00 | 1099.38 | 1096.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 12:30:00 | 1102.90 | 1099.38 | 1096.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 1091.00 | 1097.70 | 1096.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 14:00:00 | 1091.00 | 1097.70 | 1096.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 1089.80 | 1096.12 | 1095.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 14:30:00 | 1092.00 | 1096.12 | 1095.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 15:15:00 | 1088.00 | 1094.50 | 1094.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 1080.00 | 1091.60 | 1093.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1109.60 | 1086.88 | 1088.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1109.60 | 1086.88 | 1088.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1109.60 | 1086.88 | 1088.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 1109.60 | 1086.88 | 1088.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 1098.10 | 1089.13 | 1089.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 1097.40 | 1089.13 | 1089.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 11:15:00 | 1099.70 | 1091.24 | 1090.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 11:15:00 | 1099.70 | 1091.24 | 1090.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 12:15:00 | 1105.20 | 1094.03 | 1091.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 1138.40 | 1154.65 | 1139.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 14:15:00 | 1138.40 | 1154.65 | 1139.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 1138.40 | 1154.65 | 1139.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 1138.40 | 1154.65 | 1139.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 1130.00 | 1149.72 | 1138.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 1139.50 | 1149.72 | 1138.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 1126.80 | 1145.13 | 1137.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 10:45:00 | 1149.80 | 1149.17 | 1139.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 11:15:00 | 1139.00 | 1164.24 | 1165.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 11:15:00 | 1139.00 | 1164.24 | 1165.04 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 1173.00 | 1161.68 | 1160.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1188.20 | 1169.44 | 1164.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 12:15:00 | 1227.10 | 1246.76 | 1236.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 12:15:00 | 1227.10 | 1246.76 | 1236.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 1227.10 | 1246.76 | 1236.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 13:00:00 | 1227.10 | 1246.76 | 1236.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 1236.80 | 1244.77 | 1236.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 14:30:00 | 1244.60 | 1247.31 | 1238.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 12:00:00 | 1247.50 | 1255.56 | 1250.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 1215.20 | 1241.62 | 1245.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1215.20 | 1241.62 | 1245.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 10:15:00 | 1193.00 | 1224.05 | 1235.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 1210.30 | 1203.61 | 1217.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 1210.30 | 1203.61 | 1217.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1210.30 | 1203.61 | 1217.83 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 1226.40 | 1218.01 | 1217.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 14:15:00 | 1240.10 | 1223.88 | 1220.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 1250.10 | 1250.61 | 1241.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 12:15:00 | 1250.10 | 1250.61 | 1241.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1250.10 | 1250.61 | 1241.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 1238.90 | 1250.61 | 1241.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 1241.30 | 1248.75 | 1241.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 1241.30 | 1248.75 | 1241.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1238.60 | 1246.72 | 1241.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:15:00 | 1235.30 | 1246.72 | 1241.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 1235.30 | 1244.43 | 1240.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 1237.50 | 1244.43 | 1240.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 1252.30 | 1244.02 | 1241.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:30:00 | 1254.70 | 1246.47 | 1242.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 11:15:00 | 1236.00 | 1241.33 | 1241.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 11:15:00 | 1236.00 | 1241.33 | 1241.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 12:15:00 | 1232.40 | 1239.54 | 1240.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 1240.90 | 1239.66 | 1240.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 1240.90 | 1239.66 | 1240.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1240.90 | 1239.66 | 1240.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 1240.90 | 1239.66 | 1240.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1245.00 | 1240.73 | 1241.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1249.20 | 1240.73 | 1241.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 1273.00 | 1247.19 | 1244.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 10:15:00 | 1288.50 | 1255.45 | 1248.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 10:15:00 | 1302.60 | 1304.30 | 1289.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 11:00:00 | 1302.60 | 1304.30 | 1289.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 1300.00 | 1302.18 | 1294.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 1310.30 | 1302.18 | 1294.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 1301.10 | 1304.08 | 1302.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 14:15:00 | 1287.70 | 1300.80 | 1300.80 | SL hit (close<static) qty=1.00 sl=1290.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 15:15:00 | 1294.00 | 1299.44 | 1300.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 1240.00 | 1287.55 | 1294.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 10:15:00 | 1239.90 | 1233.81 | 1255.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 10:30:00 | 1241.60 | 1233.81 | 1255.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 1250.00 | 1239.53 | 1248.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 12:30:00 | 1231.80 | 1241.14 | 1247.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 14:15:00 | 1170.21 | 1200.66 | 1220.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-06-18 09:15:00 | 1108.62 | 1127.94 | 1137.58 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 79 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 1131.80 | 1112.84 | 1112.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 1137.50 | 1117.77 | 1114.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 1114.40 | 1119.66 | 1116.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 1114.40 | 1119.66 | 1116.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1114.40 | 1119.66 | 1116.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 1114.40 | 1119.66 | 1116.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 1116.00 | 1118.93 | 1116.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:30:00 | 1112.40 | 1118.93 | 1116.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 1125.60 | 1120.26 | 1117.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1134.00 | 1117.45 | 1116.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 1135.60 | 1126.35 | 1123.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:00:00 | 1134.00 | 1127.88 | 1124.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 09:15:00 | 1090.00 | 1118.75 | 1121.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1090.00 | 1118.75 | 1121.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 10:15:00 | 1064.00 | 1107.80 | 1116.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 10:15:00 | 1036.40 | 1034.83 | 1046.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:00:00 | 1036.40 | 1034.83 | 1046.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1045.00 | 1036.03 | 1041.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:45:00 | 1029.20 | 1035.31 | 1040.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 13:30:00 | 1031.00 | 1034.45 | 1039.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 14:30:00 | 1030.50 | 1034.14 | 1038.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 1030.00 | 1032.25 | 1037.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1032.10 | 1028.27 | 1031.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:15:00 | 1017.90 | 1030.61 | 1032.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 12:00:00 | 1020.10 | 1023.13 | 1027.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 1015.80 | 1023.15 | 1025.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 1020.00 | 1016.19 | 1018.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1020.00 | 1016.96 | 1018.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 1029.30 | 1016.96 | 1018.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1024.70 | 1018.50 | 1019.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1030.00 | 1020.80 | 1020.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 1030.00 | 1020.80 | 1020.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 1036.10 | 1026.88 | 1023.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 1024.60 | 1026.95 | 1024.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 12:15:00 | 1024.60 | 1026.95 | 1024.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1024.60 | 1026.95 | 1024.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:45:00 | 1031.30 | 1027.20 | 1025.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 1022.30 | 1030.66 | 1030.06 | SL hit (close<static) qty=1.00 sl=1024.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 1024.10 | 1029.35 | 1029.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 1020.60 | 1027.60 | 1028.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 1033.10 | 1026.01 | 1027.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 1033.10 | 1026.01 | 1027.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1033.10 | 1026.01 | 1027.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 1034.90 | 1026.01 | 1027.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 1042.00 | 1029.20 | 1028.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 11:15:00 | 1058.50 | 1035.06 | 1031.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 1056.70 | 1058.85 | 1047.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 10:30:00 | 1060.00 | 1058.85 | 1047.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1042.30 | 1055.19 | 1050.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1042.30 | 1055.19 | 1050.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1042.30 | 1052.61 | 1050.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 1040.50 | 1052.61 | 1050.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 1030.10 | 1045.79 | 1047.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 1028.60 | 1035.23 | 1040.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 09:15:00 | 1032.90 | 1032.61 | 1037.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:00:00 | 1032.90 | 1032.61 | 1037.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1034.00 | 1032.89 | 1036.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:15:00 | 1038.40 | 1032.89 | 1036.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 1030.00 | 1032.31 | 1036.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:30:00 | 1042.00 | 1032.31 | 1036.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1038.10 | 1033.47 | 1036.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:45:00 | 1039.20 | 1033.47 | 1036.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1038.40 | 1034.45 | 1036.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:45:00 | 1041.80 | 1034.45 | 1036.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 1037.10 | 1034.98 | 1036.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:45:00 | 1042.50 | 1034.98 | 1036.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 1035.00 | 1034.99 | 1036.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 1029.40 | 1034.99 | 1036.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1026.20 | 1033.23 | 1035.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 10:45:00 | 1019.70 | 1031.02 | 1034.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:15:00 | 1021.00 | 1031.02 | 1034.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 11:00:00 | 1017.70 | 1021.32 | 1026.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 15:15:00 | 1032.00 | 1029.03 | 1028.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 1032.00 | 1029.03 | 1028.74 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 09:15:00 | 1025.30 | 1028.28 | 1028.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 10:15:00 | 1019.30 | 1026.48 | 1027.60 | Break + close below crossover candle low |

### Cycle 87 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 1037.00 | 1028.59 | 1028.45 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 1021.50 | 1029.81 | 1030.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 1015.10 | 1025.07 | 1028.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 15:15:00 | 1017.70 | 1017.36 | 1022.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 09:15:00 | 1015.20 | 1017.36 | 1022.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1010.30 | 1015.95 | 1021.34 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 1025.50 | 1022.37 | 1022.01 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 1011.90 | 1020.79 | 1021.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 1005.50 | 1014.51 | 1017.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 13:15:00 | 904.50 | 896.43 | 910.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 14:00:00 | 904.50 | 896.43 | 910.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 930.00 | 903.15 | 912.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 930.00 | 903.15 | 912.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 935.10 | 909.54 | 914.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 923.50 | 909.54 | 914.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:15:00 | 924.10 | 914.81 | 916.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 12:15:00 | 921.50 | 917.62 | 917.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 12:15:00 | 921.50 | 917.62 | 917.40 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 911.10 | 917.84 | 917.92 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 924.10 | 918.64 | 918.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 940.00 | 924.60 | 921.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 941.70 | 942.11 | 934.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 12:15:00 | 937.80 | 940.42 | 935.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 937.80 | 940.42 | 935.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 13:45:00 | 941.00 | 940.99 | 936.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 941.00 | 948.09 | 949.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 941.00 | 948.09 | 949.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 930.90 | 945.29 | 947.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 910.20 | 908.34 | 919.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:45:00 | 907.10 | 908.34 | 919.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 915.30 | 910.47 | 915.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 915.30 | 910.47 | 915.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 911.20 | 910.62 | 915.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:30:00 | 906.60 | 910.62 | 915.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 920.00 | 906.72 | 910.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 922.65 | 906.72 | 910.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 915.60 | 908.50 | 911.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 921.00 | 908.50 | 911.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 912.50 | 904.63 | 907.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 913.80 | 904.63 | 907.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 902.05 | 904.12 | 907.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 11:15:00 | 901.00 | 904.12 | 907.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 899.75 | 900.55 | 903.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:00:00 | 901.70 | 895.25 | 896.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 903.45 | 897.12 | 896.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 903.45 | 897.12 | 896.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 11:15:00 | 915.05 | 900.71 | 898.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 911.90 | 914.04 | 909.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 10:15:00 | 912.00 | 913.63 | 909.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 912.00 | 913.63 | 909.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 911.30 | 913.63 | 909.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 911.70 | 913.24 | 910.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 908.10 | 913.24 | 910.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 909.90 | 912.57 | 910.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 909.70 | 912.57 | 910.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 908.45 | 911.75 | 909.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 908.45 | 911.75 | 909.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 907.60 | 910.92 | 909.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 907.60 | 910.92 | 909.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 905.00 | 909.74 | 909.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 908.35 | 909.74 | 909.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 904.85 | 908.76 | 908.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 14:15:00 | 899.50 | 904.48 | 906.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 922.40 | 907.35 | 907.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 922.40 | 907.35 | 907.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 922.40 | 907.35 | 907.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 930.60 | 907.35 | 907.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 922.75 | 910.43 | 908.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 11:15:00 | 933.80 | 915.10 | 911.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 13:15:00 | 931.05 | 933.17 | 925.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 931.05 | 933.17 | 925.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 939.55 | 942.23 | 937.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 939.55 | 942.23 | 937.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 933.00 | 940.39 | 936.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 933.00 | 940.39 | 936.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 935.15 | 939.34 | 936.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 935.15 | 939.34 | 936.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 940.30 | 939.53 | 936.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 946.05 | 939.63 | 937.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 936.90 | 940.69 | 940.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 936.90 | 940.69 | 940.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 920.60 | 935.76 | 938.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 922.35 | 921.09 | 927.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 09:30:00 | 918.75 | 921.09 | 927.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 866.00 | 852.59 | 861.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 865.00 | 852.59 | 861.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 898.95 | 861.86 | 864.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 898.95 | 861.86 | 864.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 898.80 | 869.25 | 867.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 13:15:00 | 920.50 | 896.99 | 884.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 921.80 | 922.03 | 910.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:30:00 | 920.00 | 922.03 | 910.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 922.00 | 926.52 | 922.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 911.10 | 926.52 | 922.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 902.35 | 921.69 | 920.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 902.35 | 921.69 | 920.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 906.45 | 918.64 | 919.17 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 983.10 | 927.71 | 922.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 10:15:00 | 1005.75 | 943.32 | 929.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 944.20 | 957.14 | 944.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 944.20 | 957.14 | 944.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 944.20 | 957.14 | 944.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 937.70 | 957.14 | 944.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 941.00 | 953.91 | 944.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:30:00 | 940.45 | 953.91 | 944.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 941.90 | 951.51 | 944.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:00:00 | 944.15 | 946.08 | 943.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 925.30 | 941.43 | 941.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 925.30 | 941.43 | 941.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 920.50 | 937.25 | 939.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 909.45 | 908.63 | 916.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 09:15:00 | 923.20 | 908.63 | 916.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 923.00 | 911.50 | 917.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:00:00 | 923.00 | 911.50 | 917.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 922.00 | 913.60 | 917.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 921.00 | 913.60 | 917.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 917.00 | 914.28 | 917.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:30:00 | 913.75 | 913.76 | 917.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:30:00 | 914.75 | 914.46 | 916.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 926.20 | 917.70 | 917.93 | SL hit (close>static) qty=1.00 sl=923.25 alert=retest2 |

### Cycle 103 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 925.75 | 919.31 | 918.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 11:15:00 | 928.25 | 921.10 | 919.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 917.50 | 920.38 | 919.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 917.50 | 920.38 | 919.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 917.50 | 920.38 | 919.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 917.50 | 920.38 | 919.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 924.45 | 921.19 | 919.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:45:00 | 927.65 | 922.00 | 920.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:30:00 | 928.65 | 922.78 | 921.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:00:00 | 927.40 | 923.70 | 921.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:45:00 | 927.90 | 924.16 | 921.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 925.00 | 924.33 | 922.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 922.75 | 924.33 | 922.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 929.45 | 930.67 | 927.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 927.50 | 930.67 | 927.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 926.90 | 929.92 | 927.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 926.90 | 929.92 | 927.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 918.55 | 927.64 | 926.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 918.55 | 927.64 | 926.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-23 15:15:00 | 918.05 | 925.72 | 925.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 918.05 | 925.72 | 925.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 912.30 | 923.04 | 924.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 15:15:00 | 872.95 | 870.63 | 877.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 09:15:00 | 880.35 | 870.63 | 877.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 879.30 | 872.36 | 877.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:30:00 | 879.85 | 872.36 | 877.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 868.10 | 871.51 | 876.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:00:00 | 867.50 | 870.71 | 875.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:30:00 | 867.90 | 868.63 | 873.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:00:00 | 862.20 | 868.63 | 873.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 10:15:00 | 882.00 | 868.92 | 872.37 | SL hit (close>static) qty=1.00 sl=880.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 877.35 | 874.00 | 873.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 911.50 | 881.50 | 877.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 10:15:00 | 919.55 | 921.28 | 905.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 11:00:00 | 919.55 | 921.28 | 905.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 906.10 | 917.12 | 905.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:45:00 | 907.85 | 917.12 | 905.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 893.50 | 912.39 | 904.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:00:00 | 893.50 | 912.39 | 904.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 896.75 | 909.27 | 904.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:15:00 | 902.15 | 902.20 | 901.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 894.65 | 900.64 | 900.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 11:15:00 | 894.65 | 900.64 | 900.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 12:15:00 | 889.00 | 898.31 | 899.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 920.90 | 897.95 | 898.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 920.90 | 897.95 | 898.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 920.90 | 897.95 | 898.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 920.90 | 897.95 | 898.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 916.75 | 901.71 | 900.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 10:15:00 | 954.25 | 916.81 | 908.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 962.00 | 973.16 | 955.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:45:00 | 963.10 | 973.16 | 955.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 955.10 | 969.54 | 955.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 955.10 | 969.54 | 955.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 946.30 | 964.90 | 954.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:45:00 | 947.50 | 964.90 | 954.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 939.20 | 959.76 | 953.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 13:00:00 | 939.20 | 959.76 | 953.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 943.00 | 952.00 | 950.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 969.60 | 952.00 | 950.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:30:00 | 961.00 | 953.26 | 951.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 1001.60 | 954.37 | 952.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 13:15:00 | 1016.60 | 1030.45 | 1031.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 1016.60 | 1030.45 | 1031.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 1012.00 | 1026.76 | 1029.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 972.40 | 969.25 | 981.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 972.40 | 969.25 | 981.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 933.40 | 923.76 | 934.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 933.40 | 923.76 | 934.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 935.50 | 926.11 | 934.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 974.80 | 926.11 | 934.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 993.30 | 939.55 | 939.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 993.30 | 939.55 | 939.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 993.80 | 950.40 | 944.59 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 944.30 | 948.57 | 948.64 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 964.30 | 951.72 | 950.06 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 943.80 | 952.75 | 953.12 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 10:15:00 | 957.70 | 952.35 | 952.19 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 941.60 | 950.16 | 951.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 939.00 | 945.44 | 948.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 936.70 | 923.59 | 930.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 936.70 | 923.59 | 930.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 936.70 | 923.59 | 930.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 936.50 | 923.59 | 930.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 936.20 | 926.11 | 931.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 932.10 | 926.11 | 931.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 953.40 | 934.49 | 933.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 953.40 | 934.49 | 933.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 960.00 | 939.59 | 936.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 961.00 | 962.14 | 955.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:15:00 | 977.30 | 962.14 | 955.60 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 985.70 | 989.58 | 982.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:30:00 | 985.00 | 989.58 | 982.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 975.00 | 986.67 | 981.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 975.00 | 986.67 | 981.94 | SL hit (close<ema400) qty=1.00 sl=981.94 alert=retest1 |

### Cycle 116 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 980.75 | 987.62 | 988.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 974.80 | 981.21 | 983.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 10:15:00 | 937.85 | 934.96 | 944.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 10:15:00 | 937.85 | 934.96 | 944.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 937.85 | 934.96 | 944.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 923.40 | 934.12 | 936.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 924.30 | 932.67 | 935.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 877.23 | 910.28 | 918.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 878.08 | 910.28 | 918.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 831.06 | 871.84 | 893.58 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 117 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 840.40 | 822.37 | 820.27 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 802.05 | 816.74 | 818.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 12:15:00 | 799.55 | 813.31 | 816.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 813.00 | 810.65 | 813.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 813.00 | 810.65 | 813.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 813.00 | 810.65 | 813.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 809.90 | 810.65 | 813.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 808.95 | 810.31 | 813.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 814.55 | 810.31 | 813.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 819.50 | 809.39 | 811.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 819.35 | 809.39 | 811.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 818.40 | 811.19 | 811.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 822.15 | 811.19 | 811.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 805.60 | 810.16 | 811.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 789.15 | 804.60 | 808.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 815.20 | 782.58 | 790.34 | SL hit (close>static) qty=1.00 sl=811.85 alert=retest2 |

### Cycle 119 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 827.05 | 796.90 | 795.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 842.00 | 805.92 | 800.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 807.75 | 814.04 | 806.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:00:00 | 807.75 | 814.04 | 806.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 812.85 | 813.80 | 807.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 14:30:00 | 825.40 | 817.08 | 810.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:45:00 | 818.50 | 816.57 | 812.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 821.10 | 817.00 | 813.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 821.65 | 817.92 | 815.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 816.35 | 818.48 | 816.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 816.35 | 818.48 | 816.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 819.65 | 818.71 | 816.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 816.05 | 818.71 | 816.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 821.90 | 819.35 | 816.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:30:00 | 817.30 | 819.35 | 816.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 858.30 | 861.16 | 848.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:30:00 | 864.05 | 861.16 | 848.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 824.55 | 853.84 | 846.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:45:00 | 825.55 | 853.84 | 846.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 828.90 | 848.85 | 844.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 857.85 | 848.85 | 844.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 826.85 | 847.92 | 850.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 826.85 | 847.92 | 850.12 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 841.75 | 838.68 | 838.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 846.80 | 841.76 | 840.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 850.30 | 854.71 | 849.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 850.30 | 854.71 | 849.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 849.95 | 853.75 | 849.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 849.05 | 853.75 | 849.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 841.85 | 851.37 | 848.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:45:00 | 841.85 | 851.37 | 848.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 840.00 | 849.10 | 848.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 840.00 | 849.10 | 848.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 828.85 | 844.39 | 846.13 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 863.00 | 844.22 | 843.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 866.50 | 851.94 | 847.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 843.90 | 853.88 | 850.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 843.90 | 853.88 | 850.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 843.90 | 853.88 | 850.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 846.80 | 853.88 | 850.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 846.00 | 852.30 | 849.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 850.00 | 848.91 | 848.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 845.25 | 848.18 | 848.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 10:15:00 | 845.25 | 848.18 | 848.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 839.15 | 846.18 | 847.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 847.65 | 843.19 | 845.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 847.65 | 843.19 | 845.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 847.65 | 843.19 | 845.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 847.65 | 843.19 | 845.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 841.85 | 842.92 | 845.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 836.85 | 841.78 | 844.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 795.01 | 821.46 | 829.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 14:15:00 | 796.10 | 787.46 | 800.47 | SL hit (close>ema200) qty=0.50 sl=787.46 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 808.35 | 798.94 | 797.97 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 770.65 | 795.56 | 797.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 766.50 | 779.07 | 787.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 782.95 | 778.05 | 784.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 10:15:00 | 782.95 | 778.05 | 784.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 782.95 | 778.05 | 784.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 785.00 | 778.05 | 784.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 774.55 | 777.35 | 783.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:30:00 | 783.80 | 777.35 | 783.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 782.35 | 778.05 | 782.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 15:00:00 | 782.35 | 778.05 | 782.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 786.40 | 779.72 | 782.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 784.10 | 779.72 | 782.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 786.40 | 781.05 | 782.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:15:00 | 781.00 | 781.45 | 782.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 741.95 | 756.89 | 765.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-16 09:15:00 | 702.90 | 723.37 | 741.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 127 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 761.15 | 737.58 | 734.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 767.45 | 743.56 | 737.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 749.90 | 757.45 | 750.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 749.90 | 757.45 | 750.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 749.90 | 757.45 | 750.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:30:00 | 761.25 | 756.99 | 751.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 742.35 | 755.35 | 752.37 | SL hit (close<static) qty=1.00 sl=743.60 alert=retest2 |

### Cycle 128 — SELL (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 12:15:00 | 741.25 | 749.49 | 750.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 716.95 | 740.40 | 745.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 730.75 | 724.75 | 732.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 730.75 | 724.75 | 732.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 730.75 | 724.75 | 732.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 725.60 | 725.63 | 732.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 745.05 | 736.75 | 735.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 745.05 | 736.75 | 735.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 767.55 | 742.91 | 738.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 754.00 | 761.11 | 752.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:00:00 | 754.00 | 761.11 | 752.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 752.85 | 759.46 | 752.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 752.85 | 759.46 | 752.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 751.60 | 757.89 | 752.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:15:00 | 755.30 | 754.35 | 751.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 735.55 | 750.03 | 750.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 735.55 | 750.03 | 750.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 718.60 | 732.87 | 740.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 760.45 | 738.39 | 742.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 760.45 | 738.39 | 742.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 760.45 | 738.39 | 742.42 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 772.25 | 748.39 | 746.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 774.90 | 753.70 | 749.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 744.10 | 756.68 | 752.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 744.10 | 756.68 | 752.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 744.10 | 756.68 | 752.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 744.10 | 756.68 | 752.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 748.40 | 755.03 | 752.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 753.95 | 755.98 | 752.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 10:15:00 | 829.35 | 807.06 | 791.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 707.85 | 798.09 | 802.63 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 746.60 | 721.33 | 718.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 766.40 | 743.05 | 731.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 758.10 | 762.10 | 754.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 758.10 | 762.10 | 754.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 758.10 | 762.10 | 754.61 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 735.60 | 750.26 | 751.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 735.15 | 747.24 | 750.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 758.85 | 743.21 | 746.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 758.85 | 743.21 | 746.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 758.85 | 743.21 | 746.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 758.90 | 743.21 | 746.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 753.00 | 745.17 | 746.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 750.80 | 745.17 | 746.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 753.45 | 747.84 | 747.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 753.45 | 747.84 | 747.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 762.70 | 750.81 | 749.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 761.60 | 763.18 | 757.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 761.60 | 763.18 | 757.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 763.75 | 767.03 | 763.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:15:00 | 765.90 | 767.03 | 763.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 765.90 | 766.80 | 763.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 757.30 | 766.80 | 763.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 748.35 | 763.11 | 762.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 748.35 | 763.11 | 762.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 749.00 | 760.29 | 760.88 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 764.00 | 756.32 | 756.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 768.00 | 758.66 | 757.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 772.00 | 772.08 | 768.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:30:00 | 770.80 | 772.08 | 768.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 768.35 | 771.41 | 768.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 767.30 | 771.41 | 768.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 768.00 | 770.73 | 768.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 767.45 | 770.73 | 768.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 765.95 | 769.77 | 768.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 765.95 | 769.77 | 768.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 766.00 | 769.02 | 767.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:15:00 | 766.00 | 769.02 | 767.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 766.00 | 768.41 | 767.80 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-18 09:15:00 | 912.05 | 2024-05-21 14:15:00 | 888.80 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-05-21 11:15:00 | 900.75 | 2024-05-21 14:15:00 | 888.80 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-05-21 11:45:00 | 899.00 | 2024-05-21 14:15:00 | 888.80 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-05-21 12:45:00 | 900.45 | 2024-05-21 14:15:00 | 888.80 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-06-18 11:00:00 | 1236.40 | 2024-06-19 14:15:00 | 1343.10 | TARGET_HIT | 1.00 | 8.63% |
| BUY | retest2 | 2024-06-18 12:30:00 | 1221.00 | 2024-06-20 09:15:00 | 1360.04 | TARGET_HIT | 1.00 | 11.39% |
| BUY | retest1 | 2024-06-27 09:15:00 | 1360.00 | 2024-06-28 11:15:00 | 1337.40 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-06-28 15:00:00 | 1359.00 | 2024-07-01 09:15:00 | 1333.05 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-07-10 13:45:00 | 1201.45 | 2024-07-15 10:15:00 | 1270.90 | STOP_HIT | 1.00 | -5.78% |
| SELL | retest2 | 2024-07-10 14:30:00 | 1203.00 | 2024-07-15 10:15:00 | 1270.90 | STOP_HIT | 1.00 | -5.64% |
| SELL | retest2 | 2024-07-12 10:00:00 | 1190.25 | 2024-07-15 10:15:00 | 1270.90 | STOP_HIT | 1.00 | -6.78% |
| SELL | retest2 | 2024-07-15 09:15:00 | 1199.00 | 2024-07-15 10:15:00 | 1270.90 | STOP_HIT | 1.00 | -6.00% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1220.00 | 2024-07-19 14:15:00 | 1159.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1220.00 | 2024-07-23 09:15:00 | 1154.00 | STOP_HIT | 0.50 | 5.41% |
| BUY | retest2 | 2024-08-02 15:15:00 | 1154.95 | 2024-08-05 09:15:00 | 1108.55 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2024-08-06 10:30:00 | 1082.15 | 2024-08-09 14:15:00 | 1149.00 | STOP_HIT | 1.00 | -6.18% |
| SELL | retest2 | 2024-08-06 13:30:00 | 1090.00 | 2024-08-09 14:15:00 | 1149.00 | STOP_HIT | 1.00 | -5.41% |
| SELL | retest2 | 2024-08-07 11:00:00 | 1084.15 | 2024-08-09 14:15:00 | 1149.00 | STOP_HIT | 1.00 | -5.98% |
| SELL | retest2 | 2024-08-08 09:30:00 | 1087.55 | 2024-08-09 14:15:00 | 1149.00 | STOP_HIT | 1.00 | -5.65% |
| BUY | retest2 | 2024-08-19 13:30:00 | 1123.40 | 2024-08-20 09:15:00 | 1096.10 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-09-06 10:30:00 | 1146.95 | 2024-09-12 11:15:00 | 1152.15 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-09-06 13:45:00 | 1154.30 | 2024-09-12 11:15:00 | 1152.15 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2024-09-09 09:15:00 | 1130.65 | 2024-09-12 11:15:00 | 1152.15 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-09-10 14:30:00 | 1139.00 | 2024-09-12 11:15:00 | 1152.15 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-09-11 09:15:00 | 1131.50 | 2024-09-12 11:15:00 | 1152.15 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-09-11 09:45:00 | 1135.05 | 2024-09-12 11:15:00 | 1152.15 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-09-11 12:45:00 | 1138.50 | 2024-09-12 11:15:00 | 1152.15 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-09-11 13:30:00 | 1138.90 | 2024-09-12 11:15:00 | 1152.15 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-09-17 14:30:00 | 1200.75 | 2024-09-18 15:15:00 | 1178.50 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-09-25 12:45:00 | 1181.00 | 2024-09-25 14:15:00 | 1167.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-09-27 14:00:00 | 1160.65 | 2024-09-27 14:15:00 | 1194.65 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-10-01 14:00:00 | 1147.55 | 2024-10-07 11:15:00 | 1090.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1143.00 | 2024-10-07 11:15:00 | 1085.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 14:00:00 | 1147.55 | 2024-10-08 09:15:00 | 1032.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1143.00 | 2024-10-08 09:15:00 | 1028.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-10-25 09:30:00 | 993.05 | 2024-10-28 12:15:00 | 1015.05 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-11-04 14:00:00 | 1058.55 | 2024-11-07 12:15:00 | 1164.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-05 09:15:00 | 1061.85 | 2024-11-07 12:15:00 | 1168.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-05 09:45:00 | 1064.35 | 2024-11-07 12:15:00 | 1170.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-26 09:30:00 | 1226.50 | 2024-11-26 12:15:00 | 1203.05 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-11-26 14:45:00 | 1237.00 | 2024-12-05 09:15:00 | 1360.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-27 12:15:00 | 1228.60 | 2024-12-05 09:15:00 | 1351.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-27 13:30:00 | 1228.30 | 2024-12-05 09:15:00 | 1351.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 09:45:00 | 1263.65 | 2024-12-05 10:15:00 | 1390.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 10:30:00 | 1265.00 | 2024-12-05 10:15:00 | 1391.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-03 09:45:00 | 1284.95 | 2024-12-10 09:15:00 | 1322.75 | STOP_HIT | 1.00 | 2.94% |
| SELL | retest2 | 2024-12-11 12:30:00 | 1311.60 | 2024-12-13 10:15:00 | 1246.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 12:30:00 | 1311.60 | 2024-12-16 09:15:00 | 1299.90 | STOP_HIT | 0.50 | 0.89% |
| SELL | retest2 | 2024-12-26 09:45:00 | 1340.40 | 2024-12-27 13:15:00 | 1373.20 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-12-26 13:00:00 | 1344.35 | 2024-12-27 13:15:00 | 1373.20 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-12-26 15:00:00 | 1344.70 | 2024-12-27 13:15:00 | 1373.20 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-12-27 09:30:00 | 1341.00 | 2024-12-27 13:15:00 | 1373.20 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-01-15 10:15:00 | 1195.00 | 2025-01-16 10:15:00 | 1259.55 | STOP_HIT | 1.00 | -5.40% |
| SELL | retest2 | 2025-01-15 11:00:00 | 1195.05 | 2025-01-16 10:15:00 | 1259.55 | STOP_HIT | 1.00 | -5.40% |
| SELL | retest2 | 2025-01-15 11:30:00 | 1193.25 | 2025-01-16 10:15:00 | 1259.55 | STOP_HIT | 1.00 | -5.56% |
| BUY | retest2 | 2025-01-31 15:15:00 | 1079.00 | 2025-02-01 11:15:00 | 1057.15 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-02-12 13:45:00 | 1056.00 | 2025-02-14 10:15:00 | 1003.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 15:15:00 | 1055.20 | 2025-02-14 10:15:00 | 1002.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:15:00 | 1057.00 | 2025-02-14 10:15:00 | 1004.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 13:45:00 | 1056.00 | 2025-02-14 13:15:00 | 950.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-12 15:15:00 | 1055.20 | 2025-02-14 13:15:00 | 951.30 | TARGET_HIT | 0.50 | 9.85% |
| SELL | retest2 | 2025-02-13 15:15:00 | 1057.00 | 2025-02-14 14:15:00 | 949.68 | TARGET_HIT | 0.50 | 10.15% |
| BUY | retest2 | 2025-03-11 10:45:00 | 952.00 | 2025-03-17 10:15:00 | 1047.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-11 13:45:00 | 956.15 | 2025-03-18 10:15:00 | 1051.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-11 14:15:00 | 960.80 | 2025-03-18 10:15:00 | 1056.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-22 10:45:00 | 1088.40 | 2025-04-25 11:15:00 | 1082.30 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-04-25 11:15:00 | 1080.70 | 2025-04-25 11:15:00 | 1082.30 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-05-02 11:15:00 | 1097.40 | 2025-05-02 11:15:00 | 1099.70 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-05-07 10:45:00 | 1149.80 | 2025-05-09 11:15:00 | 1139.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-05-16 14:30:00 | 1244.60 | 2025-05-20 14:15:00 | 1215.20 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-05-20 12:00:00 | 1247.50 | 2025-05-20 14:15:00 | 1215.20 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-05-28 11:30:00 | 1254.70 | 2025-05-29 11:15:00 | 1236.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-06-04 09:15:00 | 1310.30 | 2025-06-05 14:15:00 | 1287.70 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-06-05 13:30:00 | 1301.10 | 2025-06-05 14:15:00 | 1287.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-06-10 12:30:00 | 1231.80 | 2025-06-11 14:15:00 | 1170.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-10 12:30:00 | 1231.80 | 2025-06-18 09:15:00 | 1108.62 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1134.00 | 2025-06-30 09:15:00 | 1090.00 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1135.60 | 2025-06-30 09:15:00 | 1090.00 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2025-06-27 10:00:00 | 1134.00 | 2025-06-30 09:15:00 | 1090.00 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2025-07-07 12:45:00 | 1029.20 | 2025-07-15 10:15:00 | 1030.00 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-07-07 13:30:00 | 1031.00 | 2025-07-15 10:15:00 | 1030.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-07-07 14:30:00 | 1030.50 | 2025-07-15 10:15:00 | 1030.00 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-07-08 09:45:00 | 1030.00 | 2025-07-15 10:15:00 | 1030.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-07-09 13:15:00 | 1017.90 | 2025-07-15 10:15:00 | 1030.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-07-10 12:00:00 | 1020.10 | 2025-07-15 10:15:00 | 1030.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-07-11 10:30:00 | 1015.80 | 2025-07-15 10:15:00 | 1030.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-07-14 15:15:00 | 1020.00 | 2025-07-15 10:15:00 | 1030.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-07-16 13:45:00 | 1031.30 | 2025-07-18 10:15:00 | 1022.30 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-07-28 10:45:00 | 1019.70 | 2025-07-29 15:15:00 | 1032.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-07-28 11:15:00 | 1021.00 | 2025-07-29 15:15:00 | 1032.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-29 11:00:00 | 1017.70 | 2025-07-29 15:15:00 | 1032.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-08-14 09:15:00 | 923.50 | 2025-08-14 12:15:00 | 921.50 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-08-14 11:15:00 | 924.10 | 2025-08-14 12:15:00 | 921.50 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-08-20 13:45:00 | 941.00 | 2025-08-25 11:15:00 | 941.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-09-03 11:15:00 | 901.00 | 2025-09-09 10:15:00 | 903.45 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-09-04 10:30:00 | 899.75 | 2025-09-09 10:15:00 | 903.45 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-09-08 10:00:00 | 901.70 | 2025-09-09 10:15:00 | 903.45 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-09-19 12:00:00 | 946.05 | 2025-09-22 14:15:00 | 936.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-10-13 15:00:00 | 944.15 | 2025-10-14 09:15:00 | 925.30 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-10-16 12:30:00 | 913.75 | 2025-10-17 09:15:00 | 926.20 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-16 14:30:00 | 914.75 | 2025-10-17 09:15:00 | 926.20 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-10-20 09:45:00 | 927.65 | 2025-10-23 15:15:00 | 918.05 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-20 10:30:00 | 928.65 | 2025-10-23 15:15:00 | 918.05 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-20 12:00:00 | 927.40 | 2025-10-23 15:15:00 | 918.05 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-10-20 12:45:00 | 927.90 | 2025-10-23 15:15:00 | 918.05 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-10-31 12:00:00 | 867.50 | 2025-11-03 10:15:00 | 882.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-10-31 14:30:00 | 867.90 | 2025-11-03 10:15:00 | 882.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-10-31 15:00:00 | 862.20 | 2025-11-03 10:15:00 | 882.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-11-07 10:15:00 | 902.15 | 2025-11-07 11:15:00 | 894.65 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-11-14 09:15:00 | 969.60 | 2025-11-28 13:15:00 | 1016.60 | STOP_HIT | 1.00 | 4.85% |
| BUY | retest2 | 2025-11-14 11:30:00 | 961.00 | 2025-11-28 13:15:00 | 1016.60 | STOP_HIT | 1.00 | 5.79% |
| BUY | retest2 | 2025-11-17 09:15:00 | 1001.60 | 2025-11-28 13:15:00 | 1016.60 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-12-19 11:15:00 | 932.10 | 2025-12-19 14:15:00 | 953.40 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest1 | 2025-12-24 09:15:00 | 977.30 | 2025-12-29 12:15:00 | 975.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-12-29 14:15:00 | 981.80 | 2026-01-05 13:15:00 | 980.75 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-12-30 09:30:00 | 995.40 | 2026-01-05 13:15:00 | 980.75 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-12-30 12:15:00 | 983.20 | 2026-01-05 13:15:00 | 980.75 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-01-16 09:15:00 | 923.40 | 2026-01-20 09:15:00 | 877.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 10:15:00 | 924.30 | 2026-01-20 09:15:00 | 878.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 923.40 | 2026-01-20 15:15:00 | 831.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 10:15:00 | 924.30 | 2026-01-20 15:15:00 | 831.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-02 09:15:00 | 789.15 | 2026-02-03 09:15:00 | 815.20 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2026-02-04 14:30:00 | 825.40 | 2026-02-13 09:15:00 | 826.85 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2026-02-05 12:45:00 | 818.50 | 2026-02-13 09:15:00 | 826.85 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2026-02-05 13:30:00 | 821.10 | 2026-02-13 09:15:00 | 826.85 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2026-02-06 11:15:00 | 821.65 | 2026-02-13 09:15:00 | 826.85 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2026-02-11 09:15:00 | 857.85 | 2026-02-13 09:15:00 | 826.85 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2026-02-25 09:30:00 | 850.00 | 2026-02-25 10:15:00 | 845.25 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-02-26 12:45:00 | 836.85 | 2026-03-02 09:15:00 | 795.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:45:00 | 836.85 | 2026-03-04 14:15:00 | 796.10 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2026-03-11 11:15:00 | 781.00 | 2026-03-13 09:15:00 | 741.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:15:00 | 781.00 | 2026-03-16 09:15:00 | 702.90 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-19 11:30:00 | 761.25 | 2026-03-19 14:15:00 | 742.35 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2026-03-24 10:30:00 | 725.60 | 2026-03-24 15:15:00 | 745.05 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2026-03-27 14:15:00 | 755.30 | 2026-03-30 09:15:00 | 735.55 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-04-02 11:30:00 | 753.95 | 2026-04-09 10:15:00 | 829.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 11:15:00 | 750.80 | 2026-04-27 12:15:00 | 753.45 | STOP_HIT | 1.00 | -0.35% |
