# Muthoot Finance Ltd. (MUTHOOTFIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3535.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 235 |
| ALERT1 | 165 |
| ALERT2 | 161 |
| ALERT2_SKIP | 101 |
| ALERT3 | 365 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 152 |
| PARTIAL | 5 |
| TARGET_HIT | 7 |
| STOP_HIT | 147 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 159 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 119
- **Target hits / Stop hits / Partials:** 7 / 147 / 5
- **Avg / median % per leg:** -0.14% / -0.93%
- **Sum % (uncompounded):** -21.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 86 | 30 | 34.9% | 7 | 79 | 0 | 0.34% | 29.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.54% | -2.5% |
| BUY @ 3rd Alert (retest2) | 85 | 30 | 35.3% | 7 | 78 | 0 | 0.37% | 31.5% |
| SELL (all) | 73 | 10 | 13.7% | 0 | 68 | 5 | -0.70% | -50.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.80% | -0.8% |
| SELL @ 3rd Alert (retest2) | 72 | 10 | 13.9% | 0 | 67 | 5 | -0.70% | -50.1% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.67% | -3.3% |
| retest2 (combined) | 157 | 40 | 25.5% | 7 | 145 | 5 | -0.12% | -18.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 09:15:00 | 1080.00 | 1068.68 | 1067.60 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 14:15:00 | 1061.65 | 1067.39 | 1067.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 09:15:00 | 1059.05 | 1065.42 | 1066.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 14:15:00 | 1057.00 | 1056.54 | 1061.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 1054.20 | 1056.24 | 1060.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 1054.20 | 1056.24 | 1060.28 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 09:15:00 | 1107.05 | 1052.12 | 1051.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 11:15:00 | 1116.45 | 1073.39 | 1061.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 09:15:00 | 1115.20 | 1122.30 | 1106.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 10:15:00 | 1104.60 | 1118.76 | 1105.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 10:15:00 | 1104.60 | 1118.76 | 1105.97 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 13:15:00 | 1101.00 | 1102.75 | 1102.77 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 14:15:00 | 1105.65 | 1103.33 | 1103.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 15:15:00 | 1114.00 | 1105.46 | 1104.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 09:15:00 | 1100.20 | 1104.41 | 1103.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 1100.20 | 1104.41 | 1103.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 1100.20 | 1104.41 | 1103.68 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 10:15:00 | 1097.10 | 1102.95 | 1103.08 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 13:15:00 | 1110.10 | 1104.09 | 1103.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 09:15:00 | 1118.40 | 1107.68 | 1105.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 09:15:00 | 1112.05 | 1112.26 | 1109.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 10:15:00 | 1105.10 | 1110.83 | 1109.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 1105.10 | 1110.83 | 1109.01 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 11:15:00 | 1118.20 | 1124.90 | 1124.91 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 15:15:00 | 1126.85 | 1124.76 | 1124.72 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 10:15:00 | 1123.60 | 1124.63 | 1124.67 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 11:15:00 | 1125.20 | 1124.74 | 1124.72 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 13:15:00 | 1116.00 | 1123.07 | 1123.97 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 11:15:00 | 1135.00 | 1126.07 | 1124.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 1153.25 | 1135.06 | 1129.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 09:15:00 | 1145.70 | 1148.66 | 1140.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 09:15:00 | 1145.70 | 1148.66 | 1140.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 1145.70 | 1148.66 | 1140.91 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 15:15:00 | 1291.50 | 1297.46 | 1297.47 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 09:15:00 | 1299.00 | 1297.77 | 1297.61 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 10:15:00 | 1295.05 | 1297.22 | 1297.38 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 13:15:00 | 1301.30 | 1298.09 | 1297.72 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 14:15:00 | 1293.95 | 1297.26 | 1297.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 13:15:00 | 1290.20 | 1294.09 | 1295.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 1290.65 | 1290.55 | 1293.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 1290.65 | 1290.55 | 1293.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 1290.65 | 1290.55 | 1293.14 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 14:15:00 | 1297.35 | 1294.35 | 1294.24 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 10:15:00 | 1288.60 | 1293.84 | 1294.09 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 11:15:00 | 1296.75 | 1294.42 | 1294.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 12:15:00 | 1298.05 | 1295.14 | 1294.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 12:15:00 | 1326.95 | 1327.94 | 1321.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 1325.30 | 1328.22 | 1323.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 1325.30 | 1328.22 | 1323.45 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 13:15:00 | 1337.90 | 1343.46 | 1343.46 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 14:15:00 | 1352.00 | 1341.86 | 1341.86 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 15:15:00 | 1335.00 | 1340.49 | 1341.23 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 1351.90 | 1342.77 | 1342.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 12:15:00 | 1355.90 | 1347.38 | 1344.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 09:15:00 | 1355.90 | 1360.26 | 1355.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 1355.90 | 1360.26 | 1355.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 1355.90 | 1360.26 | 1355.33 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 1246.80 | 1337.69 | 1349.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 12:15:00 | 1235.45 | 1247.72 | 1263.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 09:15:00 | 1249.35 | 1245.56 | 1257.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 10:15:00 | 1259.00 | 1248.25 | 1257.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 1259.00 | 1248.25 | 1257.21 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 13:15:00 | 1266.00 | 1259.85 | 1259.21 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 1258.00 | 1260.76 | 1260.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 11:15:00 | 1254.90 | 1259.59 | 1260.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 13:15:00 | 1261.95 | 1259.56 | 1260.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 13:15:00 | 1261.95 | 1259.56 | 1260.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 13:15:00 | 1261.95 | 1259.56 | 1260.11 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 09:15:00 | 1283.00 | 1264.07 | 1262.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 12:15:00 | 1288.45 | 1273.09 | 1267.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 12:15:00 | 1292.20 | 1295.95 | 1288.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 14:15:00 | 1289.95 | 1294.79 | 1289.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 1289.95 | 1294.79 | 1289.36 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 11:15:00 | 1278.35 | 1286.45 | 1286.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 12:15:00 | 1270.00 | 1283.16 | 1285.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 15:15:00 | 1269.85 | 1265.17 | 1271.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 09:15:00 | 1275.00 | 1267.14 | 1271.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 1275.00 | 1267.14 | 1271.65 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 15:15:00 | 1282.10 | 1272.87 | 1272.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 09:15:00 | 1284.00 | 1275.10 | 1273.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 14:15:00 | 1283.00 | 1283.35 | 1279.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 15:15:00 | 1280.25 | 1282.73 | 1279.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 15:15:00 | 1280.25 | 1282.73 | 1279.11 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-09-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 11:15:00 | 1272.95 | 1278.99 | 1279.05 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 10:15:00 | 1284.80 | 1279.81 | 1279.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 11:15:00 | 1288.45 | 1281.53 | 1279.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 15:15:00 | 1281.00 | 1281.87 | 1280.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 15:15:00 | 1281.00 | 1281.87 | 1280.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 15:15:00 | 1281.00 | 1281.87 | 1280.69 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 1269.65 | 1283.54 | 1283.91 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 1291.80 | 1280.34 | 1279.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 10:15:00 | 1305.50 | 1293.09 | 1287.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 09:15:00 | 1313.25 | 1322.80 | 1313.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 09:15:00 | 1313.25 | 1322.80 | 1313.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 1313.25 | 1322.80 | 1313.71 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 15:15:00 | 1302.40 | 1309.53 | 1310.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 10:15:00 | 1292.15 | 1303.57 | 1307.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-27 10:15:00 | 1250.75 | 1247.63 | 1257.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 14:15:00 | 1258.85 | 1250.58 | 1255.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 14:15:00 | 1258.85 | 1250.58 | 1255.44 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 15:15:00 | 1253.00 | 1247.94 | 1247.45 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 1237.05 | 1245.76 | 1246.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 12:15:00 | 1235.95 | 1242.23 | 1244.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 13:15:00 | 1198.30 | 1198.04 | 1207.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 1196.35 | 1192.79 | 1198.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1196.35 | 1192.79 | 1198.15 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 12:15:00 | 1220.05 | 1202.71 | 1201.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 13:15:00 | 1221.85 | 1206.54 | 1203.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 15:15:00 | 1233.00 | 1234.19 | 1227.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 1232.55 | 1233.87 | 1228.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 1232.55 | 1233.87 | 1228.02 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-10-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 10:15:00 | 1261.50 | 1274.49 | 1274.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 13:15:00 | 1251.15 | 1266.70 | 1270.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 15:15:00 | 1269.00 | 1266.86 | 1270.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 1283.50 | 1270.19 | 1271.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1283.50 | 1270.19 | 1271.38 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 10:15:00 | 1293.00 | 1274.75 | 1273.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 10:15:00 | 1303.70 | 1289.13 | 1282.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 11:15:00 | 1303.50 | 1310.76 | 1300.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 12:15:00 | 1298.25 | 1308.26 | 1300.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 12:15:00 | 1298.25 | 1308.26 | 1300.24 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 13:15:00 | 1318.60 | 1325.23 | 1325.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 11:15:00 | 1315.10 | 1322.56 | 1324.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 09:15:00 | 1286.30 | 1278.19 | 1290.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 10:15:00 | 1287.30 | 1280.01 | 1290.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 1287.30 | 1280.01 | 1290.38 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-11-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 10:15:00 | 1308.40 | 1292.91 | 1292.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 09:15:00 | 1323.45 | 1305.99 | 1299.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 13:15:00 | 1333.70 | 1340.48 | 1332.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 14:15:00 | 1327.20 | 1337.82 | 1331.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 14:15:00 | 1327.20 | 1337.82 | 1331.64 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-11-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 15:15:00 | 1330.45 | 1332.87 | 1332.90 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 09:15:00 | 1333.30 | 1332.95 | 1332.94 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 10:15:00 | 1331.75 | 1332.71 | 1332.83 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 11:15:00 | 1337.00 | 1333.57 | 1333.21 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 13:15:00 | 1326.65 | 1332.74 | 1332.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 14:15:00 | 1325.00 | 1331.19 | 1332.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 10:15:00 | 1335.00 | 1330.64 | 1331.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 10:15:00 | 1335.00 | 1330.64 | 1331.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 1335.00 | 1330.64 | 1331.55 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2023-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 13:15:00 | 1335.35 | 1332.56 | 1332.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 14:15:00 | 1343.50 | 1334.75 | 1333.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 13:15:00 | 1450.35 | 1453.48 | 1427.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 1435.25 | 1456.53 | 1446.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 1435.25 | 1456.53 | 1446.28 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2023-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 14:15:00 | 1436.05 | 1441.10 | 1441.48 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2023-12-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 10:15:00 | 1454.80 | 1440.09 | 1439.08 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 1429.05 | 1440.50 | 1441.67 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 14:15:00 | 1445.90 | 1440.84 | 1440.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 09:15:00 | 1454.60 | 1444.26 | 1442.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 09:15:00 | 1450.70 | 1451.39 | 1447.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 1450.70 | 1451.39 | 1447.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 1450.70 | 1451.39 | 1447.65 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 1442.50 | 1476.38 | 1480.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 13:15:00 | 1439.05 | 1455.34 | 1466.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 15:15:00 | 1457.00 | 1455.47 | 1464.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 1472.00 | 1458.77 | 1465.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 1472.00 | 1458.77 | 1465.17 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2023-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 15:15:00 | 1472.40 | 1467.32 | 1467.25 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2023-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-26 11:15:00 | 1457.90 | 1466.10 | 1466.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-26 12:15:00 | 1450.35 | 1462.95 | 1465.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-27 09:15:00 | 1462.40 | 1458.68 | 1462.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 09:15:00 | 1462.40 | 1458.68 | 1462.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 1462.40 | 1458.68 | 1462.11 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2023-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 13:15:00 | 1470.15 | 1464.31 | 1464.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 14:15:00 | 1476.70 | 1466.79 | 1465.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 1480.55 | 1486.64 | 1479.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 1480.55 | 1486.64 | 1479.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 1480.55 | 1486.64 | 1479.11 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 13:15:00 | 1473.30 | 1476.93 | 1477.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-01 14:15:00 | 1470.35 | 1475.62 | 1476.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 15:15:00 | 1458.00 | 1457.95 | 1464.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 09:15:00 | 1465.00 | 1459.36 | 1464.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 1465.00 | 1459.36 | 1464.56 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 11:15:00 | 1484.25 | 1468.05 | 1467.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 1522.80 | 1483.06 | 1475.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 10:15:00 | 1507.20 | 1508.24 | 1495.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 13:15:00 | 1501.90 | 1509.82 | 1499.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 13:15:00 | 1501.90 | 1509.82 | 1499.81 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 11:15:00 | 1486.85 | 1496.08 | 1496.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 15:15:00 | 1471.40 | 1487.35 | 1490.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 14:15:00 | 1477.20 | 1472.37 | 1480.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 1488.75 | 1475.27 | 1480.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 1488.75 | 1475.27 | 1480.03 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-01-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 12:15:00 | 1496.30 | 1485.40 | 1483.97 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 11:15:00 | 1473.10 | 1482.59 | 1483.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 12:15:00 | 1470.10 | 1480.09 | 1482.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 09:15:00 | 1478.60 | 1476.92 | 1479.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 09:15:00 | 1478.60 | 1476.92 | 1479.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 1478.60 | 1476.92 | 1479.70 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 1401.55 | 1396.59 | 1396.12 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 11:15:00 | 1390.40 | 1395.10 | 1395.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 12:15:00 | 1384.70 | 1393.02 | 1394.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 15:15:00 | 1396.00 | 1391.28 | 1393.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 15:15:00 | 1396.00 | 1391.28 | 1393.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 1396.00 | 1391.28 | 1393.10 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 12:15:00 | 1397.55 | 1394.54 | 1394.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 09:15:00 | 1411.60 | 1399.00 | 1396.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 12:15:00 | 1398.80 | 1399.37 | 1397.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 12:15:00 | 1398.80 | 1399.37 | 1397.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 12:15:00 | 1398.80 | 1399.37 | 1397.39 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 1382.60 | 1394.47 | 1395.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 10:15:00 | 1377.50 | 1387.54 | 1390.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 1382.05 | 1372.96 | 1380.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 1382.05 | 1372.96 | 1380.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 1382.05 | 1372.96 | 1380.56 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 13:15:00 | 1387.25 | 1376.74 | 1376.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 14:15:00 | 1390.75 | 1379.54 | 1377.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 09:15:00 | 1401.25 | 1409.56 | 1398.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 10:15:00 | 1381.20 | 1403.89 | 1396.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 1381.20 | 1403.89 | 1396.85 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-02-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 12:15:00 | 1360.30 | 1389.53 | 1391.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 1351.55 | 1366.80 | 1374.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 1363.60 | 1359.40 | 1366.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 10:15:00 | 1366.90 | 1360.90 | 1366.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 1366.90 | 1360.90 | 1366.90 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 1381.05 | 1366.35 | 1364.98 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-02-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 10:15:00 | 1344.65 | 1362.60 | 1363.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-15 11:15:00 | 1340.95 | 1358.27 | 1361.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 09:15:00 | 1353.95 | 1349.80 | 1355.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 09:15:00 | 1353.95 | 1349.80 | 1355.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 1353.95 | 1349.80 | 1355.11 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 15:15:00 | 1362.00 | 1358.04 | 1357.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 11:15:00 | 1363.85 | 1359.25 | 1358.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 14:15:00 | 1356.00 | 1361.71 | 1359.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 14:15:00 | 1356.00 | 1361.71 | 1359.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 1356.00 | 1361.71 | 1359.94 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 11:15:00 | 1350.10 | 1358.14 | 1358.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 12:15:00 | 1341.00 | 1354.71 | 1357.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 1336.10 | 1331.81 | 1337.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 1336.10 | 1331.81 | 1337.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 1336.10 | 1331.81 | 1337.55 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 1316.60 | 1296.84 | 1294.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 11:15:00 | 1320.00 | 1301.47 | 1297.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-02 09:15:00 | 1311.00 | 1312.72 | 1305.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 09:15:00 | 1311.00 | 1312.72 | 1305.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 1311.00 | 1312.72 | 1305.27 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 11:15:00 | 1357.60 | 1382.77 | 1385.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 13:15:00 | 1354.60 | 1374.11 | 1380.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 1367.65 | 1362.57 | 1372.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 14:15:00 | 1368.00 | 1361.61 | 1368.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 1368.00 | 1361.61 | 1368.54 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 1399.50 | 1347.29 | 1342.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 12:15:00 | 1407.80 | 1373.34 | 1356.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 11:15:00 | 1455.80 | 1456.76 | 1436.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 09:15:00 | 1583.70 | 1563.36 | 1538.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 1583.70 | 1563.36 | 1538.72 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 13:15:00 | 1645.25 | 1659.16 | 1660.11 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 10:15:00 | 1670.90 | 1661.34 | 1660.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 12:15:00 | 1675.20 | 1665.40 | 1662.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 15:15:00 | 1661.00 | 1665.00 | 1663.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 15:15:00 | 1661.00 | 1665.00 | 1663.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 15:15:00 | 1661.00 | 1665.00 | 1663.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:15:00 | 1633.00 | 1665.00 | 1663.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 1657.30 | 1663.46 | 1662.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:30:00 | 1636.80 | 1663.46 | 1662.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 11:15:00 | 1661.15 | 1662.12 | 1662.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 12:30:00 | 1664.00 | 1662.29 | 1662.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 13:15:00 | 1663.95 | 1662.29 | 1662.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 13:15:00 | 1660.20 | 1661.88 | 1662.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 13:15:00 | 1660.20 | 1661.88 | 1662.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 1647.55 | 1659.01 | 1660.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 09:15:00 | 1647.20 | 1641.47 | 1647.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 09:15:00 | 1647.20 | 1641.47 | 1647.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 1647.20 | 1641.47 | 1647.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:45:00 | 1655.25 | 1641.47 | 1647.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 1660.40 | 1645.25 | 1648.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:00:00 | 1660.40 | 1645.25 | 1648.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 1669.90 | 1650.18 | 1650.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 12:00:00 | 1669.90 | 1650.18 | 1650.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 12:15:00 | 1666.55 | 1653.46 | 1652.20 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 1643.75 | 1650.68 | 1651.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-23 09:15:00 | 1592.15 | 1634.19 | 1642.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 15:15:00 | 1621.00 | 1618.33 | 1628.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-24 09:15:00 | 1621.95 | 1618.33 | 1628.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 1649.75 | 1624.61 | 1630.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 10:00:00 | 1649.75 | 1624.61 | 1630.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 10:15:00 | 1658.80 | 1631.45 | 1633.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 10:30:00 | 1666.30 | 1631.45 | 1633.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 12:15:00 | 1651.95 | 1637.53 | 1635.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 09:15:00 | 1699.65 | 1657.79 | 1648.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 10:15:00 | 1708.30 | 1710.13 | 1698.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-02 11:00:00 | 1708.30 | 1710.13 | 1698.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 1695.65 | 1709.96 | 1703.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 10:00:00 | 1695.65 | 1709.96 | 1703.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 1711.70 | 1710.31 | 1704.69 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 13:15:00 | 1690.60 | 1701.00 | 1701.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 14:15:00 | 1677.50 | 1691.48 | 1696.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 14:15:00 | 1674.85 | 1670.20 | 1680.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-07 15:00:00 | 1674.85 | 1670.20 | 1680.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 1671.10 | 1670.38 | 1679.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:15:00 | 1682.80 | 1670.38 | 1679.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 1677.05 | 1671.71 | 1679.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:45:00 | 1683.00 | 1671.71 | 1679.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 1677.10 | 1672.79 | 1679.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:15:00 | 1680.30 | 1672.79 | 1679.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 1690.00 | 1676.23 | 1680.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:00:00 | 1690.00 | 1676.23 | 1680.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 1689.05 | 1678.79 | 1680.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:45:00 | 1690.60 | 1678.79 | 1680.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 1661.05 | 1675.46 | 1679.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 15:00:00 | 1661.05 | 1675.46 | 1679.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 1650.45 | 1624.84 | 1640.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:00:00 | 1650.45 | 1624.84 | 1640.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 1649.90 | 1629.85 | 1641.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 12:00:00 | 1649.90 | 1629.85 | 1641.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 10:15:00 | 1662.95 | 1648.45 | 1646.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 11:15:00 | 1680.30 | 1654.82 | 1649.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 13:15:00 | 1667.55 | 1679.85 | 1670.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 13:15:00 | 1667.55 | 1679.85 | 1670.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 1667.55 | 1679.85 | 1670.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 14:00:00 | 1667.55 | 1679.85 | 1670.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 1676.00 | 1679.08 | 1670.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 15:15:00 | 1680.00 | 1679.08 | 1670.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 1699.55 | 1674.17 | 1672.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:45:00 | 1680.70 | 1684.37 | 1681.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 13:00:00 | 1679.90 | 1683.48 | 1681.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 1676.85 | 1682.15 | 1681.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:30:00 | 1675.60 | 1682.15 | 1681.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 1676.00 | 1680.82 | 1680.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 1705.00 | 1680.82 | 1680.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 12:00:00 | 1685.25 | 1692.37 | 1688.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 13:00:00 | 1685.15 | 1690.93 | 1687.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 13:45:00 | 1685.05 | 1690.34 | 1687.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 1703.95 | 1693.06 | 1689.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 10:45:00 | 1708.70 | 1698.94 | 1693.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 14:15:00 | 1708.80 | 1701.12 | 1695.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:00:00 | 1709.05 | 1702.71 | 1696.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 10:45:00 | 1709.00 | 1704.55 | 1699.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1690.00 | 1708.02 | 1704.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 1690.00 | 1708.02 | 1704.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1699.65 | 1706.35 | 1703.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 12:30:00 | 1707.60 | 1706.52 | 1704.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 14:15:00 | 1694.45 | 1703.08 | 1703.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 1694.45 | 1703.08 | 1703.10 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 11:15:00 | 1718.55 | 1703.30 | 1702.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 09:15:00 | 1722.85 | 1709.58 | 1706.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 13:15:00 | 1734.00 | 1739.55 | 1730.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 13:15:00 | 1734.00 | 1739.55 | 1730.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 1734.00 | 1739.55 | 1730.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:00:00 | 1734.00 | 1739.55 | 1730.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 1738.65 | 1739.37 | 1730.86 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 10:15:00 | 1686.90 | 1723.63 | 1725.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 1677.75 | 1714.45 | 1721.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 1693.30 | 1692.05 | 1704.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 1682.70 | 1690.14 | 1699.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1682.70 | 1690.14 | 1699.62 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 1738.60 | 1705.70 | 1704.70 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1640.40 | 1694.16 | 1700.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1613.10 | 1677.95 | 1692.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 1689.25 | 1667.92 | 1680.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 1689.25 | 1667.92 | 1680.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 1689.25 | 1667.92 | 1680.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 1689.25 | 1667.92 | 1680.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 1726.15 | 1679.57 | 1684.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 1726.15 | 1679.57 | 1684.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 1721.45 | 1687.94 | 1688.22 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 1724.00 | 1695.15 | 1691.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 13:15:00 | 1738.85 | 1703.89 | 1695.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 09:15:00 | 1748.10 | 1748.42 | 1731.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 09:45:00 | 1749.90 | 1748.42 | 1731.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 1753.75 | 1764.46 | 1757.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 1735.25 | 1759.70 | 1755.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 1749.60 | 1757.68 | 1755.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:00:00 | 1754.95 | 1757.13 | 1755.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:45:00 | 1753.05 | 1755.71 | 1754.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 14:15:00 | 1755.00 | 1754.00 | 1753.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 14:15:00 | 1770.95 | 1778.37 | 1778.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 14:15:00 | 1770.95 | 1778.37 | 1778.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 10:15:00 | 1765.45 | 1772.90 | 1775.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 10:15:00 | 1758.55 | 1757.92 | 1765.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-19 11:00:00 | 1758.55 | 1757.92 | 1765.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1749.40 | 1750.99 | 1758.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:45:00 | 1755.95 | 1750.99 | 1758.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1732.60 | 1741.71 | 1749.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 1743.85 | 1741.71 | 1749.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1740.15 | 1729.69 | 1736.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 1740.15 | 1729.69 | 1736.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 1736.80 | 1731.11 | 1736.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 1737.00 | 1731.11 | 1736.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 1737.30 | 1732.35 | 1736.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 14:15:00 | 1734.55 | 1733.88 | 1737.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 14:45:00 | 1734.15 | 1733.56 | 1736.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:45:00 | 1733.85 | 1736.76 | 1737.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:30:00 | 1734.80 | 1736.82 | 1737.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 1765.70 | 1742.59 | 1740.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 12:15:00 | 1765.70 | 1742.59 | 1740.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 13:15:00 | 1769.00 | 1747.87 | 1742.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 15:15:00 | 1769.10 | 1772.70 | 1762.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 09:15:00 | 1768.00 | 1772.70 | 1762.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1802.30 | 1778.62 | 1766.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 10:30:00 | 1817.15 | 1792.30 | 1773.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 14:45:00 | 1824.50 | 1809.03 | 1788.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 13:15:00 | 1792.50 | 1801.04 | 1801.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 13:15:00 | 1792.50 | 1801.04 | 1801.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-01 14:15:00 | 1790.00 | 1798.83 | 1800.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 13:15:00 | 1793.05 | 1786.98 | 1792.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 13:15:00 | 1793.05 | 1786.98 | 1792.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 1793.05 | 1786.98 | 1792.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:30:00 | 1795.00 | 1786.98 | 1792.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 1796.05 | 1788.79 | 1792.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 14:45:00 | 1797.80 | 1788.79 | 1792.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1796.80 | 1792.52 | 1793.73 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 1798.00 | 1794.64 | 1794.55 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 15:15:00 | 1788.00 | 1794.18 | 1794.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 09:15:00 | 1785.05 | 1792.35 | 1793.69 | Break + close below crossover candle low |

### Cycle 95 — BUY (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 11:15:00 | 1813.80 | 1795.74 | 1794.94 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 1772.65 | 1795.34 | 1798.13 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 14:15:00 | 1808.75 | 1794.93 | 1794.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 12:15:00 | 1813.50 | 1802.64 | 1798.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 1843.85 | 1844.01 | 1833.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 09:45:00 | 1837.70 | 1844.01 | 1833.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 1837.70 | 1842.75 | 1833.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 1834.00 | 1842.75 | 1833.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 1835.00 | 1841.20 | 1834.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:45:00 | 1833.30 | 1841.20 | 1834.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 1834.40 | 1839.84 | 1834.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:30:00 | 1833.65 | 1839.84 | 1834.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 1833.85 | 1838.64 | 1834.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:45:00 | 1832.45 | 1838.64 | 1834.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 1835.75 | 1838.06 | 1834.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:45:00 | 1848.80 | 1841.86 | 1836.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 13:30:00 | 1840.75 | 1841.95 | 1838.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 14:00:00 | 1843.80 | 1841.95 | 1838.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 1865.40 | 1838.77 | 1837.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 1835.00 | 1847.51 | 1844.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 1822.75 | 1847.51 | 1844.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 1828.70 | 1843.75 | 1842.89 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 1828.70 | 1843.75 | 1842.89 | SL hit (close<static) qty=1.00 sl=1830.90 alert=retest2 |

### Cycle 98 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 1823.15 | 1839.63 | 1841.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 12:15:00 | 1816.95 | 1829.26 | 1834.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 14:15:00 | 1743.45 | 1736.32 | 1753.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-25 15:00:00 | 1743.45 | 1736.32 | 1753.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 1767.30 | 1744.05 | 1754.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 1767.30 | 1744.05 | 1754.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 1774.35 | 1750.11 | 1756.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 1774.35 | 1750.11 | 1756.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 1781.40 | 1760.80 | 1760.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 1784.80 | 1765.60 | 1762.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 1779.75 | 1781.92 | 1774.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 1779.75 | 1781.92 | 1774.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1782.05 | 1782.28 | 1776.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:15:00 | 1798.00 | 1782.28 | 1776.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 12:15:00 | 1817.25 | 1843.64 | 1845.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 1817.25 | 1843.64 | 1845.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 1807.15 | 1824.62 | 1832.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 1839.05 | 1815.93 | 1824.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 10:15:00 | 1839.05 | 1815.93 | 1824.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 1839.05 | 1815.93 | 1824.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:00:00 | 1839.05 | 1815.93 | 1824.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 1829.15 | 1818.58 | 1824.84 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 1857.75 | 1832.38 | 1830.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 1867.55 | 1850.03 | 1842.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 1879.80 | 1885.19 | 1874.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 1879.80 | 1885.19 | 1874.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1879.80 | 1885.19 | 1874.03 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 15:15:00 | 1852.60 | 1867.73 | 1869.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 1791.00 | 1852.39 | 1862.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 12:15:00 | 1821.85 | 1820.65 | 1835.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 13:00:00 | 1821.85 | 1820.65 | 1835.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 1834.90 | 1824.32 | 1834.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:45:00 | 1836.20 | 1824.32 | 1834.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 1834.00 | 1826.26 | 1834.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 1840.05 | 1826.26 | 1834.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1853.05 | 1831.62 | 1836.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:45:00 | 1850.25 | 1831.62 | 1836.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 1853.35 | 1835.96 | 1837.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 11:15:00 | 1858.15 | 1835.96 | 1837.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 1853.00 | 1839.37 | 1839.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 10:15:00 | 1866.90 | 1848.93 | 1844.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 12:15:00 | 1902.95 | 1908.95 | 1893.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 13:00:00 | 1902.95 | 1908.95 | 1893.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1902.45 | 1911.65 | 1901.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 12:30:00 | 1908.45 | 1909.70 | 1902.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 13:45:00 | 1907.45 | 1909.22 | 1902.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 1909.75 | 1906.59 | 1902.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 15:15:00 | 1957.20 | 1966.19 | 1966.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 15:15:00 | 1957.20 | 1966.19 | 1966.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 13:15:00 | 1952.65 | 1961.14 | 1963.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 14:15:00 | 1975.95 | 1964.11 | 1964.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 1975.95 | 1964.11 | 1964.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1975.95 | 1964.11 | 1964.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 1975.95 | 1964.11 | 1964.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1969.00 | 1965.08 | 1965.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 1979.90 | 1965.08 | 1965.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 1979.65 | 1968.00 | 1966.57 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 11:15:00 | 1957.90 | 1965.70 | 1965.76 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 1988.00 | 1969.10 | 1966.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 12:15:00 | 1993.10 | 1973.90 | 1969.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 1976.75 | 1980.52 | 1974.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 1976.75 | 1980.52 | 1974.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1976.75 | 1980.52 | 1974.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 1976.75 | 1980.52 | 1974.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 1973.70 | 1979.16 | 1974.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 12:45:00 | 1987.25 | 1980.61 | 1977.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 14:30:00 | 1988.45 | 1982.95 | 1978.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 1966.75 | 1981.46 | 1978.96 | SL hit (close<static) qty=1.00 sl=1969.40 alert=retest2 |

### Cycle 108 — SELL (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 10:15:00 | 1957.60 | 1976.69 | 1977.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 1949.25 | 1957.04 | 1963.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 1970.65 | 1958.03 | 1962.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 1970.65 | 1958.03 | 1962.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1970.65 | 1958.03 | 1962.26 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 1988.00 | 1968.50 | 1966.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 2060.00 | 1992.63 | 1978.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 2012.15 | 2013.55 | 1996.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 15:00:00 | 2012.15 | 2013.55 | 1996.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 2009.80 | 2013.20 | 2006.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:00:00 | 2026.80 | 2015.92 | 2008.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 13:45:00 | 2023.75 | 2025.77 | 2020.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 15:00:00 | 2019.90 | 2024.59 | 2020.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 2053.90 | 2022.48 | 2019.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 2021.45 | 2026.65 | 2022.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:45:00 | 2014.85 | 2026.65 | 2022.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 2025.90 | 2026.50 | 2023.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 1986.15 | 2021.21 | 2022.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 09:15:00 | 1986.15 | 2021.21 | 2022.09 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 2029.70 | 2014.12 | 2012.99 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 10:15:00 | 2001.25 | 2012.62 | 2012.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 14:15:00 | 1992.05 | 2005.71 | 2009.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 09:15:00 | 2006.95 | 2003.87 | 2007.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 2006.95 | 2003.87 | 2007.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 2006.95 | 2003.87 | 2007.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:15:00 | 2027.45 | 2003.87 | 2007.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 2031.00 | 2009.30 | 2009.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 2037.65 | 2009.30 | 2009.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 11:15:00 | 2030.95 | 2013.63 | 2011.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 12:15:00 | 2037.25 | 2018.35 | 2014.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 2018.95 | 2026.55 | 2020.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 2018.95 | 2026.55 | 2020.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 2018.95 | 2026.55 | 2020.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 2019.65 | 2026.55 | 2020.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 2028.00 | 2026.84 | 2020.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:00:00 | 2040.00 | 2027.19 | 2022.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 14:45:00 | 2038.10 | 2043.21 | 2041.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 09:15:00 | 1958.70 | 2024.55 | 2033.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 09:15:00 | 1958.70 | 2024.55 | 2033.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 15:15:00 | 1952.40 | 1975.10 | 2000.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 15:15:00 | 1969.95 | 1965.87 | 1982.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 09:15:00 | 1948.10 | 1965.87 | 1982.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1951.10 | 1962.91 | 1979.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:00:00 | 1932.45 | 1954.68 | 1971.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 1928.60 | 1943.60 | 1961.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:30:00 | 1930.30 | 1913.38 | 1915.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 12:15:00 | 1940.95 | 1918.90 | 1917.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 1940.95 | 1918.90 | 1917.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 13:15:00 | 1950.80 | 1925.28 | 1920.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 13:15:00 | 1943.50 | 1945.79 | 1939.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 13:45:00 | 1944.00 | 1945.79 | 1939.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1945.00 | 1944.51 | 1940.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 1945.00 | 1944.51 | 1940.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1951.35 | 1945.88 | 1941.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:30:00 | 1942.35 | 1945.88 | 1941.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 1949.95 | 1946.75 | 1942.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 1956.00 | 1947.02 | 1943.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 11:30:00 | 1952.80 | 1952.24 | 1947.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:00:00 | 1956.50 | 1954.73 | 1952.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 12:15:00 | 1951.40 | 1954.59 | 1953.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 1962.85 | 1956.24 | 1954.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 1941.55 | 1951.31 | 1952.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 1941.55 | 1951.31 | 1952.33 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 11:15:00 | 1960.50 | 1952.99 | 1952.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 13:15:00 | 1973.60 | 1957.67 | 1955.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 10:15:00 | 1958.85 | 1961.08 | 1957.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 1958.85 | 1961.08 | 1957.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 1958.85 | 1961.08 | 1957.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:45:00 | 1956.50 | 1961.08 | 1957.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 1948.05 | 1958.47 | 1957.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:45:00 | 1952.20 | 1958.47 | 1957.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 1946.15 | 1956.01 | 1956.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 1941.00 | 1949.63 | 1952.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 09:15:00 | 1960.20 | 1951.74 | 1953.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 09:15:00 | 1960.20 | 1951.74 | 1953.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 1960.20 | 1951.74 | 1953.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 1960.20 | 1951.74 | 1953.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1944.10 | 1950.21 | 1952.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 11:30:00 | 1926.10 | 1946.64 | 1950.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 12:30:00 | 1928.05 | 1942.36 | 1948.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:15:00 | 1930.50 | 1931.98 | 1937.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:00:00 | 1929.95 | 1934.32 | 1935.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 1936.90 | 1933.98 | 1935.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:15:00 | 1936.75 | 1933.98 | 1935.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1923.70 | 1931.92 | 1934.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:15:00 | 1906.35 | 1931.92 | 1934.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 11:00:00 | 1919.05 | 1929.35 | 1933.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 12:15:00 | 1921.80 | 1928.64 | 1932.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:00:00 | 1921.95 | 1927.30 | 1931.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 1920.50 | 1922.63 | 1927.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:15:00 | 1929.05 | 1922.63 | 1927.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 1930.95 | 1924.29 | 1928.22 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-29 12:15:00 | 1963.00 | 1935.46 | 1931.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 1963.00 | 1935.46 | 1931.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 1986.85 | 1955.24 | 1943.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 1956.40 | 1964.83 | 1955.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 1956.40 | 1964.83 | 1955.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1956.40 | 1964.83 | 1955.51 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 1923.10 | 1946.86 | 1948.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 15:15:00 | 1915.20 | 1934.52 | 1942.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 18:15:00 | 1935.80 | 1934.43 | 1940.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 1935.80 | 1934.43 | 1940.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1903.85 | 1928.32 | 1937.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:30:00 | 1901.55 | 1921.36 | 1933.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 11:45:00 | 1900.55 | 1915.91 | 1929.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 12:45:00 | 1897.00 | 1913.49 | 1927.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 13:15:00 | 1897.55 | 1913.49 | 1927.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 1917.90 | 1892.34 | 1906.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 1917.90 | 1892.34 | 1906.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 1913.65 | 1896.60 | 1906.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 1913.65 | 1896.60 | 1906.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1919.50 | 1902.56 | 1907.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 14:30:00 | 1901.05 | 1903.72 | 1907.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 11:15:00 | 1806.47 | 1858.79 | 1883.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 11:15:00 | 1805.52 | 1858.79 | 1883.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 11:15:00 | 1802.15 | 1858.79 | 1883.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 11:15:00 | 1802.67 | 1858.79 | 1883.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 11:15:00 | 1806.00 | 1858.79 | 1883.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-11 10:15:00 | 1815.55 | 1810.14 | 1832.63 | SL hit (close>ema200) qty=0.50 sl=1810.14 alert=retest2 |

### Cycle 121 — BUY (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 09:15:00 | 1875.80 | 1797.06 | 1793.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 12:15:00 | 1912.00 | 1846.55 | 1819.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 12:15:00 | 1879.55 | 1882.28 | 1855.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 13:00:00 | 1879.55 | 1882.28 | 1855.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1931.55 | 1946.92 | 1936.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 1930.55 | 1946.92 | 1936.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 1923.55 | 1942.25 | 1935.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 1923.45 | 1942.25 | 1935.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1925.90 | 1934.61 | 1933.69 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 1913.20 | 1930.33 | 1931.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 1901.80 | 1922.36 | 1927.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 11:15:00 | 1913.70 | 1913.47 | 1920.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 12:00:00 | 1913.70 | 1913.47 | 1920.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 1913.90 | 1913.55 | 1919.51 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 1936.95 | 1920.59 | 1920.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 12:15:00 | 1941.00 | 1924.67 | 1922.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 11:15:00 | 1928.55 | 1930.39 | 1927.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 11:45:00 | 1929.75 | 1930.39 | 1927.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1932.00 | 1932.15 | 1929.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:15:00 | 1930.60 | 1932.15 | 1929.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 1933.20 | 1932.36 | 1929.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 13:30:00 | 1939.90 | 1934.73 | 1931.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:00:00 | 1942.60 | 1937.65 | 1934.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 1938.00 | 1939.16 | 1936.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-12 14:15:00 | 2131.80 | 2091.45 | 2061.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 2088.55 | 2104.90 | 2106.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 2072.95 | 2098.51 | 2103.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 2095.40 | 2093.45 | 2100.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 10:00:00 | 2095.40 | 2093.45 | 2100.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 2058.60 | 2045.06 | 2056.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 2058.60 | 2045.06 | 2056.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 2051.10 | 2046.27 | 2055.58 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 2071.05 | 2059.99 | 2059.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 2096.80 | 2068.96 | 2063.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 2242.60 | 2242.70 | 2218.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 2242.60 | 2242.70 | 2218.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2208.70 | 2234.21 | 2218.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 2206.95 | 2234.21 | 2218.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 2174.60 | 2222.28 | 2214.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 2174.60 | 2222.28 | 2214.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 2186.25 | 2215.08 | 2211.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 2169.40 | 2215.08 | 2211.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 2186.85 | 2209.43 | 2209.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 2177.30 | 2203.01 | 2206.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 15:15:00 | 2199.90 | 2197.98 | 2203.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:15:00 | 2196.00 | 2197.98 | 2203.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 2197.70 | 2197.93 | 2202.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 2210.00 | 2197.93 | 2202.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 2197.00 | 2197.74 | 2202.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:45:00 | 2201.00 | 2197.74 | 2202.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 2207.25 | 2199.64 | 2202.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 2210.25 | 2199.64 | 2202.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 2210.85 | 2201.88 | 2203.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:30:00 | 2213.85 | 2201.88 | 2203.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 13:15:00 | 2216.90 | 2204.89 | 2204.76 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 2180.00 | 2200.60 | 2203.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 2167.00 | 2193.88 | 2199.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 15:15:00 | 2141.00 | 2139.10 | 2153.26 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:15:00 | 2118.35 | 2139.10 | 2153.26 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 2113.80 | 2112.41 | 2129.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 2127.35 | 2112.41 | 2129.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 2135.40 | 2117.01 | 2130.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 2135.40 | 2117.01 | 2130.01 | SL hit (close>ema400) qty=1.00 sl=2130.01 alert=retest1 |

### Cycle 129 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 2163.40 | 2134.35 | 2133.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 11:15:00 | 2174.00 | 2142.28 | 2137.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 14:15:00 | 2179.90 | 2183.42 | 2168.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 15:00:00 | 2179.90 | 2183.42 | 2168.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 2168.80 | 2181.07 | 2170.16 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 14:15:00 | 2147.90 | 2163.21 | 2164.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 09:15:00 | 2145.75 | 2158.22 | 2162.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 11:15:00 | 2157.35 | 2147.03 | 2152.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 11:15:00 | 2157.35 | 2147.03 | 2152.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 2157.35 | 2147.03 | 2152.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:45:00 | 2165.00 | 2147.03 | 2152.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 2168.80 | 2151.38 | 2153.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:00:00 | 2168.80 | 2151.38 | 2153.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 13:15:00 | 2176.90 | 2156.49 | 2155.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-22 09:15:00 | 2185.55 | 2165.52 | 2160.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-22 12:15:00 | 2166.95 | 2170.30 | 2164.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 12:15:00 | 2166.95 | 2170.30 | 2164.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 2166.95 | 2170.30 | 2164.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 13:00:00 | 2166.95 | 2170.30 | 2164.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 13:15:00 | 2166.50 | 2169.54 | 2164.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 13:30:00 | 2156.65 | 2169.54 | 2164.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 2200.10 | 2175.65 | 2167.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 09:30:00 | 2203.10 | 2187.10 | 2174.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 10:00:00 | 2215.80 | 2187.10 | 2174.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 09:15:00 | 2164.50 | 2185.10 | 2180.18 | SL hit (close<static) qty=1.00 sl=2167.15 alert=retest2 |

### Cycle 132 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 2173.50 | 2179.67 | 2179.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 2143.55 | 2172.44 | 2176.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 2152.40 | 2136.73 | 2151.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 12:15:00 | 2152.40 | 2136.73 | 2151.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 2152.40 | 2136.73 | 2151.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 2152.40 | 2136.73 | 2151.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 2145.60 | 2138.50 | 2150.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:00:00 | 2135.40 | 2137.88 | 2149.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 2182.40 | 2146.33 | 2151.00 | SL hit (close>static) qty=1.00 sl=2153.25 alert=retest2 |

### Cycle 133 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 2189.15 | 2154.90 | 2154.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 12:15:00 | 2217.25 | 2173.14 | 2163.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 10:15:00 | 2243.75 | 2244.92 | 2226.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 11:00:00 | 2243.75 | 2244.92 | 2226.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 2211.80 | 2238.30 | 2225.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 2211.80 | 2238.30 | 2225.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 2232.85 | 2237.21 | 2225.81 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 2184.00 | 2217.48 | 2218.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 2169.55 | 2207.89 | 2214.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 15:15:00 | 2180.05 | 2178.44 | 2193.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:15:00 | 2219.20 | 2178.44 | 2193.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 2237.35 | 2190.22 | 2197.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 2237.35 | 2190.22 | 2197.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 2219.40 | 2196.06 | 2199.47 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 2215.65 | 2203.65 | 2202.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 13:15:00 | 2236.60 | 2210.24 | 2205.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 13:15:00 | 2253.80 | 2254.55 | 2234.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 14:00:00 | 2253.80 | 2254.55 | 2234.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 2242.05 | 2255.16 | 2245.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 2242.05 | 2255.16 | 2245.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 2231.95 | 2250.51 | 2244.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 2231.95 | 2250.51 | 2244.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 2231.10 | 2246.63 | 2242.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 2242.10 | 2246.63 | 2242.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 2258.50 | 2249.68 | 2244.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 2251.00 | 2249.68 | 2244.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 2254.25 | 2250.59 | 2245.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:30:00 | 2251.80 | 2250.59 | 2245.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 2241.10 | 2248.40 | 2245.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 2241.10 | 2248.40 | 2245.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 2242.10 | 2247.14 | 2245.29 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 2201.65 | 2237.65 | 2241.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 2170.00 | 2203.39 | 2219.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 11:15:00 | 2200.75 | 2200.47 | 2215.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 12:00:00 | 2200.75 | 2200.47 | 2215.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 2194.15 | 2181.46 | 2195.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 2194.15 | 2181.46 | 2195.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 2207.15 | 2186.60 | 2196.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 2207.90 | 2186.60 | 2196.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 2170.15 | 2183.31 | 2194.44 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 2317.65 | 2209.39 | 2203.50 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 10:15:00 | 2224.00 | 2242.49 | 2244.19 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 2255.50 | 2241.13 | 2240.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 10:15:00 | 2264.50 | 2245.80 | 2242.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 15:15:00 | 2257.65 | 2257.66 | 2250.49 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:15:00 | 2287.15 | 2257.66 | 2250.49 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 2258.70 | 2267.35 | 2258.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 12:45:00 | 2258.65 | 2267.35 | 2258.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 2273.65 | 2268.61 | 2259.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 14:45:00 | 2280.15 | 2270.55 | 2261.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 2229.00 | 2263.38 | 2259.91 | SL hit (close<ema400) qty=1.00 sl=2259.91 alert=retest1 |

### Cycle 140 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 2216.85 | 2254.08 | 2255.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 11:15:00 | 2198.20 | 2242.90 | 2250.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 14:15:00 | 2190.95 | 2188.23 | 2209.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 14:30:00 | 2193.70 | 2188.23 | 2209.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 2220.60 | 2194.53 | 2208.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:00:00 | 2220.60 | 2194.53 | 2208.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 2206.00 | 2196.82 | 2208.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:30:00 | 2224.30 | 2196.82 | 2208.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 2194.85 | 2196.43 | 2207.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 12:30:00 | 2191.65 | 2195.86 | 2205.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 13:15:00 | 2190.55 | 2195.86 | 2205.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 14:00:00 | 2180.80 | 2192.85 | 2203.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 2221.00 | 2195.31 | 2201.76 | SL hit (close>static) qty=1.00 sl=2208.10 alert=retest2 |

### Cycle 141 — BUY (started 2025-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 14:15:00 | 2211.10 | 2204.73 | 2204.60 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 2144.95 | 2192.81 | 2199.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 2126.95 | 2163.97 | 2183.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 2154.35 | 2141.86 | 2159.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 12:45:00 | 2149.05 | 2141.86 | 2159.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 2152.25 | 2143.94 | 2158.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 2132.95 | 2145.23 | 2156.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 10:00:00 | 2144.80 | 2145.14 | 2155.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 10:30:00 | 2136.35 | 2143.49 | 2153.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 14:30:00 | 2143.90 | 2142.65 | 2150.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 2145.10 | 2143.14 | 2149.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 2129.65 | 2143.14 | 2149.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 2132.40 | 2140.99 | 2148.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 2162.55 | 2138.25 | 2142.26 | SL hit (close>static) qty=1.00 sl=2160.70 alert=retest2 |

### Cycle 143 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 2174.35 | 2145.47 | 2145.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 12:15:00 | 2177.25 | 2156.87 | 2150.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 2160.45 | 2165.58 | 2159.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 2160.45 | 2165.58 | 2159.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 2165.05 | 2165.47 | 2159.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 13:15:00 | 2172.25 | 2165.47 | 2159.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:00:00 | 2176.30 | 2167.64 | 2161.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 09:45:00 | 2171.40 | 2185.76 | 2178.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-19 10:15:00 | 2389.48 | 2337.89 | 2296.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 13:15:00 | 2353.40 | 2360.57 | 2360.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 09:15:00 | 2342.35 | 2354.25 | 2357.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-25 13:15:00 | 2349.00 | 2346.09 | 2351.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 13:15:00 | 2349.00 | 2346.09 | 2351.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 2349.00 | 2346.09 | 2351.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:45:00 | 2353.50 | 2346.09 | 2351.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 2343.50 | 2345.57 | 2351.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 2343.50 | 2345.57 | 2351.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 2340.25 | 2344.42 | 2349.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 2336.40 | 2344.42 | 2349.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:15:00 | 2332.05 | 2343.68 | 2348.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 15:15:00 | 2321.00 | 2321.22 | 2328.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 10:15:00 | 2360.00 | 2333.27 | 2332.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 2360.00 | 2333.27 | 2332.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 11:15:00 | 2390.60 | 2344.74 | 2337.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 2352.70 | 2363.24 | 2351.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 2352.70 | 2363.24 | 2351.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 2352.70 | 2363.24 | 2351.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 2352.70 | 2363.24 | 2351.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 2342.40 | 2359.07 | 2350.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 2342.40 | 2359.07 | 2350.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 2357.10 | 2358.68 | 2351.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 2343.00 | 2358.68 | 2351.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 2353.45 | 2357.63 | 2351.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:45:00 | 2352.15 | 2357.63 | 2351.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 2344.40 | 2354.98 | 2350.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:00:00 | 2344.40 | 2354.98 | 2350.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 2336.10 | 2351.21 | 2349.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:30:00 | 2337.00 | 2351.21 | 2349.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 2328.00 | 2346.57 | 2347.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 2315.05 | 2340.26 | 2344.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 2343.70 | 2337.41 | 2342.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 11:15:00 | 2343.70 | 2337.41 | 2342.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 2343.70 | 2337.41 | 2342.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 2343.70 | 2337.41 | 2342.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 2347.75 | 2339.48 | 2342.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:30:00 | 2352.75 | 2339.48 | 2342.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 2335.25 | 2338.63 | 2342.09 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 2370.45 | 2347.60 | 2345.52 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 2257.40 | 2334.21 | 2343.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 10:15:00 | 2063.40 | 2235.59 | 2271.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 14:15:00 | 2050.40 | 2041.15 | 2084.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-15 14:45:00 | 2046.20 | 2041.15 | 2084.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 2064.90 | 2047.79 | 2080.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 10:45:00 | 2048.80 | 2048.95 | 2077.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 11:15:00 | 2048.30 | 2048.95 | 2077.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 12:00:00 | 2050.80 | 2049.32 | 2075.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 13:15:00 | 2051.60 | 2051.12 | 2073.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 14:15:00 | 2079.80 | 2058.92 | 2073.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-16 15:00:00 | 2079.80 | 2058.92 | 2073.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 15:15:00 | 2079.00 | 2062.93 | 2074.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:15:00 | 2107.30 | 2062.93 | 2074.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 2106.10 | 2071.57 | 2077.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-17 09:15:00 | 2106.10 | 2071.57 | 2077.02 | SL hit (close>static) qty=1.00 sl=2090.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 11:15:00 | 2109.00 | 2085.84 | 2082.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 13:15:00 | 2125.50 | 2098.07 | 2089.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 2207.10 | 2246.09 | 2207.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 2207.10 | 2246.09 | 2207.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 2207.10 | 2246.09 | 2207.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 2207.10 | 2246.09 | 2207.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 2209.90 | 2238.85 | 2207.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 2207.40 | 2238.85 | 2207.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 2212.00 | 2233.48 | 2207.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 2212.00 | 2233.48 | 2207.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 2189.00 | 2224.58 | 2206.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:00:00 | 2189.00 | 2224.58 | 2206.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 2189.20 | 2217.51 | 2204.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:15:00 | 2184.90 | 2217.51 | 2204.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 10:15:00 | 2179.60 | 2197.09 | 2197.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 13:15:00 | 2156.20 | 2181.56 | 2190.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 10:15:00 | 2115.00 | 2114.35 | 2137.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:45:00 | 2133.40 | 2114.35 | 2137.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 2130.90 | 2117.75 | 2135.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 2131.80 | 2117.75 | 2135.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 2137.00 | 2121.60 | 2135.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:00:00 | 2137.00 | 2121.60 | 2135.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 2134.50 | 2124.18 | 2135.44 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 13:15:00 | 2160.80 | 2141.75 | 2139.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 2175.00 | 2150.70 | 2144.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 13:15:00 | 2161.50 | 2163.59 | 2153.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 13:45:00 | 2163.70 | 2163.59 | 2153.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 2161.50 | 2164.79 | 2156.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 2150.00 | 2164.79 | 2156.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 2136.10 | 2159.05 | 2154.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:45:00 | 2134.00 | 2159.05 | 2154.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 2163.50 | 2159.94 | 2155.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 13:00:00 | 2170.10 | 2161.31 | 2156.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 14:00:00 | 2176.10 | 2164.27 | 2158.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 14:30:00 | 2170.50 | 2166.57 | 2159.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 2205.30 | 2256.07 | 2258.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 2205.30 | 2256.07 | 2258.91 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2297.30 | 2254.56 | 2251.26 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 09:15:00 | 2223.30 | 2251.66 | 2251.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 2211.40 | 2233.98 | 2242.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 2255.90 | 2231.52 | 2238.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 2255.90 | 2231.52 | 2238.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 2255.90 | 2231.52 | 2238.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:30:00 | 2251.60 | 2231.52 | 2238.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 2248.00 | 2234.82 | 2238.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-14 12:30:00 | 2238.10 | 2241.36 | 2241.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 13:15:00 | 2255.10 | 2244.11 | 2242.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 13:15:00 | 2255.10 | 2244.11 | 2242.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 15:15:00 | 2266.00 | 2250.26 | 2245.82 | Break + close above crossover candle high |

### Cycle 156 — SELL (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 09:15:00 | 2148.00 | 2229.81 | 2236.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 10:15:00 | 2131.00 | 2210.05 | 2227.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2065.50 | 2052.68 | 2077.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 2065.50 | 2052.68 | 2077.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 2072.40 | 2058.24 | 2073.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:30:00 | 2067.50 | 2058.24 | 2073.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 2082.50 | 2063.09 | 2074.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:00:00 | 2082.50 | 2063.09 | 2074.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 2090.00 | 2068.47 | 2075.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:45:00 | 2087.40 | 2068.47 | 2075.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 2092.00 | 2078.80 | 2079.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:45:00 | 2101.00 | 2078.80 | 2079.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 2097.70 | 2082.58 | 2080.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 2103.80 | 2092.21 | 2088.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 2096.00 | 2096.26 | 2092.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 15:15:00 | 2096.00 | 2096.26 | 2092.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 2096.00 | 2096.26 | 2092.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 2102.20 | 2096.26 | 2092.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2081.40 | 2093.29 | 2091.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 2081.40 | 2093.29 | 2091.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 2072.00 | 2089.03 | 2090.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 2069.00 | 2082.14 | 2086.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 2075.20 | 2073.38 | 2080.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 10:00:00 | 2075.20 | 2073.38 | 2080.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 2091.70 | 2077.04 | 2081.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 2091.70 | 2077.04 | 2081.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 2097.70 | 2081.18 | 2082.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 2097.70 | 2081.18 | 2082.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 2099.10 | 2084.76 | 2084.28 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 2064.00 | 2084.64 | 2084.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 2061.50 | 2072.56 | 2078.44 | Break + close below crossover candle low |

### Cycle 161 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 2150.20 | 2085.65 | 2082.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 10:15:00 | 2204.80 | 2109.48 | 2093.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 2540.60 | 2542.62 | 2494.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:30:00 | 2526.90 | 2542.62 | 2494.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 2549.00 | 2558.63 | 2540.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 2575.00 | 2558.63 | 2540.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2568.00 | 2560.51 | 2542.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:00:00 | 2587.50 | 2566.32 | 2548.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 2598.90 | 2622.93 | 2625.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 2598.90 | 2622.93 | 2625.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 2589.10 | 2616.16 | 2622.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 14:15:00 | 2602.50 | 2602.07 | 2609.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 14:45:00 | 2596.00 | 2602.07 | 2609.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 2609.70 | 2603.60 | 2609.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 2586.90 | 2603.60 | 2609.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2553.20 | 2593.52 | 2604.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 10:15:00 | 2550.00 | 2593.52 | 2604.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:00:00 | 2548.80 | 2562.16 | 2579.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:30:00 | 2550.00 | 2555.49 | 2571.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 13:30:00 | 2551.50 | 2555.49 | 2569.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 2574.00 | 2559.19 | 2570.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 2574.00 | 2559.19 | 2570.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 2590.00 | 2565.35 | 2572.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 2590.30 | 2565.35 | 2572.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 2560.80 | 2564.44 | 2571.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 10:45:00 | 2557.50 | 2560.53 | 2568.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 2541.90 | 2569.27 | 2570.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:00:00 | 2553.10 | 2560.86 | 2565.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 09:15:00 | 2613.20 | 2575.62 | 2570.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 09:15:00 | 2613.20 | 2575.62 | 2570.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 11:15:00 | 2647.50 | 2596.42 | 2581.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 2606.80 | 2613.36 | 2597.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:15:00 | 2605.50 | 2613.36 | 2597.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 2613.90 | 2613.47 | 2598.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 2600.00 | 2613.47 | 2598.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 2619.70 | 2626.28 | 2615.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 2619.70 | 2626.28 | 2615.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 2619.50 | 2624.93 | 2616.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 2608.60 | 2624.93 | 2616.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 2625.60 | 2625.06 | 2617.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:15:00 | 2611.30 | 2625.06 | 2617.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 2611.30 | 2622.31 | 2616.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 2636.90 | 2622.31 | 2616.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 15:00:00 | 2635.30 | 2634.62 | 2626.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:45:00 | 2634.00 | 2635.00 | 2628.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 2635.10 | 2649.91 | 2648.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 13:15:00 | 2636.70 | 2645.68 | 2646.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 13:15:00 | 2636.70 | 2645.68 | 2646.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 11:15:00 | 2634.90 | 2642.65 | 2644.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 12:15:00 | 2645.00 | 2643.12 | 2644.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 12:15:00 | 2645.00 | 2643.12 | 2644.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 2645.00 | 2643.12 | 2644.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:00:00 | 2645.00 | 2643.12 | 2644.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 2646.00 | 2643.69 | 2644.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:45:00 | 2649.90 | 2643.69 | 2644.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 2645.50 | 2644.06 | 2644.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:00:00 | 2645.50 | 2644.06 | 2644.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 2631.50 | 2641.54 | 2643.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 2648.20 | 2641.54 | 2643.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 2646.50 | 2642.54 | 2643.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:00:00 | 2639.30 | 2641.89 | 2643.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 2626.80 | 2640.46 | 2641.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 2627.60 | 2629.49 | 2632.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 2661.80 | 2635.95 | 2635.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 2661.80 | 2635.95 | 2635.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 10:15:00 | 2670.50 | 2642.86 | 2638.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 10:15:00 | 2655.90 | 2658.22 | 2650.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:15:00 | 2657.10 | 2658.22 | 2650.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 2650.30 | 2656.23 | 2651.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 2650.30 | 2656.23 | 2651.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 2646.70 | 2654.33 | 2650.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:00:00 | 2646.70 | 2654.33 | 2650.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 2657.80 | 2655.02 | 2651.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:45:00 | 2641.70 | 2655.02 | 2651.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 2645.00 | 2653.02 | 2650.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 2645.30 | 2653.02 | 2650.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 2645.40 | 2651.49 | 2650.27 | EMA400 retest candle locked (from upside) |

### Cycle 166 — SELL (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 11:15:00 | 2641.10 | 2649.00 | 2649.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 2631.30 | 2644.07 | 2646.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 11:15:00 | 2650.50 | 2644.78 | 2646.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 11:15:00 | 2650.50 | 2644.78 | 2646.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 2650.50 | 2644.78 | 2646.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 2650.50 | 2644.78 | 2646.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 2647.30 | 2645.29 | 2646.49 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 14:15:00 | 2659.00 | 2649.07 | 2648.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 09:15:00 | 2665.30 | 2656.71 | 2653.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 2663.60 | 2672.67 | 2665.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 2663.60 | 2672.67 | 2665.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 2663.60 | 2672.67 | 2665.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 2657.90 | 2672.67 | 2665.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 2659.80 | 2670.10 | 2664.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 2673.40 | 2670.30 | 2665.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:00:00 | 2670.50 | 2674.46 | 2670.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 2674.50 | 2675.31 | 2670.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 2663.00 | 2675.31 | 2675.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 2663.00 | 2675.31 | 2675.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 2638.60 | 2655.80 | 2662.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 14:15:00 | 2634.00 | 2615.90 | 2625.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 14:15:00 | 2634.00 | 2615.90 | 2625.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 2634.00 | 2615.90 | 2625.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 2634.00 | 2615.90 | 2625.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 2627.00 | 2618.12 | 2626.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:30:00 | 2625.30 | 2614.89 | 2623.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 2612.60 | 2613.68 | 2620.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:45:00 | 2619.90 | 2613.68 | 2620.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 2619.20 | 2614.78 | 2620.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 2619.20 | 2614.78 | 2620.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 2600.10 | 2612.11 | 2618.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 15:15:00 | 2582.50 | 2605.28 | 2612.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 2637.00 | 2607.98 | 2612.12 | SL hit (close>static) qty=1.00 sl=2630.90 alert=retest2 |

### Cycle 169 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 2638.80 | 2618.60 | 2616.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 12:15:00 | 2649.20 | 2624.72 | 2619.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 2628.00 | 2641.47 | 2630.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 2628.00 | 2641.47 | 2630.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 2628.00 | 2641.47 | 2630.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 2628.00 | 2641.47 | 2630.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 2642.60 | 2641.70 | 2631.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 15:00:00 | 2644.30 | 2638.59 | 2633.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 2622.90 | 2634.04 | 2632.29 | SL hit (close<static) qty=1.00 sl=2623.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 2616.00 | 2630.73 | 2631.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 2603.30 | 2623.96 | 2628.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 2633.60 | 2620.31 | 2624.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 2633.60 | 2620.31 | 2624.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2633.60 | 2620.31 | 2624.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 2633.60 | 2620.31 | 2624.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2636.10 | 2623.47 | 2625.59 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 2644.10 | 2629.60 | 2628.13 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 2600.10 | 2626.90 | 2628.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 2570.00 | 2615.52 | 2622.97 | Break + close below crossover candle low |

### Cycle 173 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 2786.10 | 2569.59 | 2563.62 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 11:15:00 | 2680.00 | 2714.14 | 2714.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 2666.00 | 2682.20 | 2692.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 2690.30 | 2683.82 | 2692.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 2690.30 | 2683.82 | 2692.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 2690.30 | 2683.82 | 2692.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 2690.30 | 2683.82 | 2692.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 2685.70 | 2684.74 | 2691.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:45:00 | 2690.90 | 2684.74 | 2691.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 2692.70 | 2686.33 | 2691.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:45:00 | 2691.80 | 2686.33 | 2691.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 2686.70 | 2686.40 | 2691.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:30:00 | 2694.80 | 2686.40 | 2691.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 2695.90 | 2685.27 | 2689.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 2668.50 | 2688.59 | 2689.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:00:00 | 2676.30 | 2686.63 | 2688.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:30:00 | 2675.30 | 2683.25 | 2686.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 11:15:00 | 2675.60 | 2682.21 | 2685.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 2674.40 | 2680.64 | 2684.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:00:00 | 2665.90 | 2677.70 | 2682.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:30:00 | 2670.10 | 2675.68 | 2681.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 2670.40 | 2658.02 | 2668.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 2706.90 | 2665.21 | 2665.88 | SL hit (close>static) qty=1.00 sl=2685.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 2714.10 | 2674.99 | 2670.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 2747.10 | 2706.79 | 2689.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 2712.00 | 2713.59 | 2697.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 13:00:00 | 2712.00 | 2713.59 | 2697.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 2693.90 | 2709.65 | 2696.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 2693.90 | 2709.65 | 2696.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 2699.70 | 2707.66 | 2697.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 2693.70 | 2707.66 | 2697.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 2697.00 | 2705.53 | 2697.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 2754.70 | 2705.53 | 2697.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 2879.40 | 2892.77 | 2893.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 2879.40 | 2892.77 | 2893.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 2875.00 | 2889.22 | 2891.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 2898.40 | 2889.90 | 2891.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 2898.40 | 2889.90 | 2891.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 2898.40 | 2889.90 | 2891.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 2898.40 | 2889.90 | 2891.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 2890.70 | 2890.06 | 2891.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 2885.80 | 2889.65 | 2891.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 2915.90 | 2894.90 | 2893.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 2915.90 | 2894.90 | 2893.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 14:15:00 | 2926.20 | 2903.93 | 2897.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 10:15:00 | 2949.00 | 2964.79 | 2948.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 10:15:00 | 2949.00 | 2964.79 | 2948.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 2949.00 | 2964.79 | 2948.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 2949.00 | 2964.79 | 2948.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 2958.20 | 2963.47 | 2948.95 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 2923.10 | 2940.18 | 2942.45 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 2955.00 | 2943.66 | 2942.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 2963.30 | 2947.59 | 2944.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 2946.00 | 2947.27 | 2944.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 11:00:00 | 2946.00 | 2947.27 | 2944.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 2927.20 | 2943.26 | 2943.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 2927.20 | 2943.26 | 2943.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 2926.40 | 2939.88 | 2941.61 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 3008.80 | 2952.89 | 2946.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 13:15:00 | 3022.50 | 2991.46 | 2969.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 09:15:00 | 3048.10 | 3067.79 | 3048.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 3048.10 | 3067.79 | 3048.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 3048.10 | 3067.79 | 3048.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 3054.20 | 3067.79 | 3048.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 3047.00 | 3063.63 | 3047.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:45:00 | 3042.70 | 3063.63 | 3047.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 3044.70 | 3059.85 | 3047.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:15:00 | 3038.30 | 3059.85 | 3047.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 3066.10 | 3061.10 | 3049.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:30:00 | 3058.60 | 3061.10 | 3049.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 3048.80 | 3059.76 | 3050.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 3048.80 | 3059.76 | 3050.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 3045.00 | 3056.81 | 3050.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 3040.20 | 3056.81 | 3050.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 3047.30 | 3054.91 | 3050.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 3061.60 | 3054.91 | 3050.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 3049.90 | 3053.90 | 3050.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 3048.90 | 3053.90 | 3050.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 3025.00 | 3048.12 | 3047.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 3025.20 | 3048.12 | 3047.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 3043.80 | 3047.26 | 3047.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 3018.70 | 3041.55 | 3044.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 3036.80 | 3034.63 | 3040.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 3036.80 | 3034.63 | 3040.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 3036.80 | 3034.63 | 3040.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 3034.90 | 3034.63 | 3040.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 3033.70 | 3034.44 | 3039.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 3033.00 | 3034.44 | 3039.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 3049.20 | 3036.32 | 3039.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 3049.20 | 3036.32 | 3039.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 3072.70 | 3043.59 | 3042.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 09:15:00 | 3077.80 | 3055.31 | 3048.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 13:15:00 | 3061.00 | 3061.88 | 3054.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 14:00:00 | 3061.00 | 3061.88 | 3054.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 3076.20 | 3064.74 | 3056.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 09:30:00 | 3094.40 | 3073.74 | 3062.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 3181.80 | 3227.54 | 3228.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 10:15:00 | 3181.80 | 3227.54 | 3228.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 3163.80 | 3205.25 | 3217.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 10:15:00 | 3191.90 | 3183.17 | 3199.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 10:45:00 | 3190.40 | 3183.17 | 3199.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 3195.30 | 3185.60 | 3199.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 3198.30 | 3185.60 | 3199.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 3200.10 | 3188.50 | 3199.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 3200.10 | 3188.50 | 3199.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 3208.40 | 3192.48 | 3200.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:45:00 | 3210.00 | 3192.48 | 3200.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 3215.20 | 3197.02 | 3201.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 3215.20 | 3197.02 | 3201.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 3264.10 | 3213.80 | 3208.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3351.90 | 3280.45 | 3259.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 3293.20 | 3319.29 | 3296.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 3293.20 | 3319.29 | 3296.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 3293.20 | 3319.29 | 3296.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 14:30:00 | 3337.90 | 3312.42 | 3300.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 3278.00 | 3306.91 | 3300.05 | SL hit (close<static) qty=1.00 sl=3282.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 3167.60 | 3273.46 | 3285.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 3118.40 | 3220.66 | 3258.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 10:15:00 | 3207.20 | 3192.06 | 3223.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 11:00:00 | 3207.20 | 3192.06 | 3223.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 3173.40 | 3153.96 | 3171.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 3173.40 | 3153.96 | 3171.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 3201.00 | 3163.36 | 3173.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:45:00 | 3212.00 | 3163.36 | 3173.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 3200.10 | 3170.71 | 3176.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 3173.20 | 3170.71 | 3176.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 3182.00 | 3175.34 | 3174.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 09:15:00 | 3182.00 | 3175.34 | 3174.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 3192.00 | 3179.99 | 3177.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 3176.00 | 3193.91 | 3188.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 3176.00 | 3193.91 | 3188.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 3176.00 | 3193.91 | 3188.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 3176.00 | 3193.91 | 3188.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 3191.00 | 3193.33 | 3188.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 3175.40 | 3193.33 | 3188.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 3194.30 | 3193.52 | 3189.05 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 3182.00 | 3187.36 | 3187.99 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 3199.30 | 3189.75 | 3189.02 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 3183.00 | 3188.44 | 3188.55 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 15:15:00 | 3191.00 | 3188.95 | 3188.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 3210.00 | 3193.16 | 3190.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 10:15:00 | 3177.80 | 3190.09 | 3189.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 10:15:00 | 3177.80 | 3190.09 | 3189.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 3177.80 | 3190.09 | 3189.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 3177.80 | 3190.09 | 3189.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 3181.80 | 3188.43 | 3188.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 3169.00 | 3181.83 | 3185.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 3190.40 | 3183.54 | 3185.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 3190.40 | 3183.54 | 3185.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 3190.40 | 3183.54 | 3185.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 3188.00 | 3183.54 | 3185.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 3183.90 | 3183.61 | 3185.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 3185.50 | 3183.61 | 3185.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3198.80 | 3186.65 | 3186.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 3203.30 | 3186.65 | 3186.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 3222.00 | 3193.72 | 3190.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 3243.00 | 3212.43 | 3200.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 3309.90 | 3338.94 | 3300.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 3309.90 | 3338.94 | 3300.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 3357.50 | 3333.44 | 3314.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:30:00 | 3382.10 | 3343.51 | 3320.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-14 09:15:00 | 3720.31 | 3442.68 | 3380.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 3663.30 | 3684.63 | 3686.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 3635.00 | 3666.03 | 3677.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 3656.90 | 3634.08 | 3649.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 3656.90 | 3634.08 | 3649.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 3656.90 | 3634.08 | 3649.41 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 3684.00 | 3660.05 | 3658.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 15:15:00 | 3697.90 | 3670.45 | 3663.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 12:15:00 | 3739.30 | 3750.65 | 3731.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 13:00:00 | 3739.30 | 3750.65 | 3731.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 3746.30 | 3749.78 | 3732.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:15:00 | 3752.90 | 3749.78 | 3732.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 3785.80 | 3746.70 | 3734.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 3727.60 | 3776.25 | 3774.90 | SL hit (close<static) qty=1.00 sl=3731.50 alert=retest2 |

### Cycle 196 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 3717.10 | 3764.42 | 3769.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 14:15:00 | 3705.90 | 3732.01 | 3748.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 3741.90 | 3728.87 | 3743.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 3741.90 | 3728.87 | 3743.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 3741.90 | 3728.87 | 3743.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 3741.90 | 3728.87 | 3743.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 3756.60 | 3734.41 | 3744.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 3756.60 | 3734.41 | 3744.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 3760.80 | 3739.69 | 3746.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:30:00 | 3773.40 | 3739.69 | 3746.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 3787.00 | 3754.91 | 3752.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 3803.80 | 3764.69 | 3757.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 3773.00 | 3773.66 | 3763.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 11:00:00 | 3773.00 | 3773.66 | 3763.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 3768.40 | 3772.61 | 3764.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:15:00 | 3761.60 | 3772.61 | 3764.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 3775.90 | 3773.27 | 3765.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:30:00 | 3767.80 | 3773.27 | 3765.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 3773.50 | 3773.32 | 3765.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 3774.10 | 3773.32 | 3765.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 3770.90 | 3772.83 | 3766.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:30:00 | 3772.90 | 3772.83 | 3766.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 3763.10 | 3770.89 | 3766.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 3733.10 | 3770.89 | 3766.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 3737.50 | 3764.21 | 3763.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 3737.50 | 3764.21 | 3763.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 3736.60 | 3758.69 | 3761.08 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 3770.40 | 3755.75 | 3755.70 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 13:15:00 | 3734.40 | 3751.97 | 3754.21 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 3784.60 | 3759.63 | 3756.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 3795.00 | 3766.70 | 3760.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 3824.10 | 3839.90 | 3816.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 3824.10 | 3839.90 | 3816.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3836.70 | 3839.49 | 3827.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 3836.70 | 3839.49 | 3827.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 3835.80 | 3838.75 | 3828.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 3831.90 | 3838.75 | 3828.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 3800.90 | 3831.18 | 3825.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:00:00 | 3800.90 | 3831.18 | 3825.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 3778.90 | 3820.72 | 3821.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 3766.10 | 3802.68 | 3812.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 3760.20 | 3753.87 | 3769.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 14:00:00 | 3760.20 | 3753.87 | 3769.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 3780.10 | 3759.12 | 3770.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 3780.10 | 3759.12 | 3770.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 3767.70 | 3760.84 | 3770.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:30:00 | 3764.80 | 3763.17 | 3770.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:30:00 | 3767.00 | 3762.57 | 3769.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:00:00 | 3760.20 | 3762.57 | 3769.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 14:15:00 | 3789.80 | 3775.62 | 3774.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 3789.80 | 3775.62 | 3774.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 12:15:00 | 3801.00 | 3787.91 | 3781.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 3837.10 | 3840.28 | 3816.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 3837.10 | 3840.28 | 3816.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 3809.40 | 3834.10 | 3815.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 3809.40 | 3834.10 | 3815.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 3796.70 | 3826.62 | 3814.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 3842.90 | 3826.62 | 3814.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:15:00 | 3814.90 | 3822.07 | 3814.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:45:00 | 3815.40 | 3818.30 | 3814.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 3781.70 | 3806.02 | 3809.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 3781.70 | 3806.02 | 3809.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 3767.30 | 3794.56 | 3803.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 3767.70 | 3760.10 | 3775.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 12:15:00 | 3767.70 | 3760.10 | 3775.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 3767.70 | 3760.10 | 3775.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:45:00 | 3772.90 | 3760.10 | 3775.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 3798.20 | 3767.72 | 3777.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:30:00 | 3788.30 | 3767.72 | 3777.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 3801.30 | 3774.44 | 3779.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 3801.30 | 3774.44 | 3779.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 3794.70 | 3778.49 | 3781.06 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 3847.20 | 3792.23 | 3787.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 3852.40 | 3827.58 | 3813.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 3817.70 | 3839.05 | 3825.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 13:15:00 | 3817.70 | 3839.05 | 3825.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 3817.70 | 3839.05 | 3825.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 3817.70 | 3839.05 | 3825.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 3820.10 | 3835.26 | 3824.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:45:00 | 3802.80 | 3835.26 | 3824.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 3827.00 | 3833.61 | 3824.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 3889.10 | 3833.61 | 3824.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 3840.10 | 3901.50 | 3909.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 3840.10 | 3901.50 | 3909.70 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 3911.00 | 3878.83 | 3874.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 10:15:00 | 3940.00 | 3895.89 | 3883.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 3908.60 | 3909.58 | 3897.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 3908.60 | 3909.58 | 3897.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 3908.60 | 3909.58 | 3897.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 3943.80 | 3917.25 | 3902.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 14:30:00 | 3940.40 | 3930.76 | 3913.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 3940.00 | 3930.76 | 3913.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 3950.10 | 3932.37 | 3917.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 3939.60 | 3938.57 | 3926.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 3936.00 | 3938.57 | 3926.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 3926.30 | 3936.12 | 3926.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 3963.60 | 3936.12 | 3926.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:15:00 | 3945.00 | 3935.68 | 3928.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:00:00 | 3946.70 | 3937.89 | 3930.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 11:15:00 | 3903.30 | 3930.06 | 3930.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 3903.30 | 3930.06 | 3930.58 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 09:15:00 | 3953.30 | 3930.72 | 3929.71 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 14:15:00 | 3905.80 | 3930.89 | 3931.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 09:15:00 | 3825.00 | 3906.40 | 3919.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 13:15:00 | 3870.60 | 3865.63 | 3892.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 14:00:00 | 3870.60 | 3865.63 | 3892.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 3906.50 | 3872.16 | 3888.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 3876.00 | 3879.94 | 3889.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 3946.30 | 3884.99 | 3877.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 3946.30 | 3884.99 | 3877.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 3956.00 | 3918.73 | 3897.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 3965.90 | 4034.81 | 3988.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 3965.90 | 4034.81 | 3988.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 3965.90 | 4034.81 | 3988.78 | EMA400 retest candle locked (from upside) |

### Cycle 212 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 3863.80 | 3953.24 | 3961.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 14:15:00 | 3828.60 | 3928.31 | 3949.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 3539.50 | 3515.63 | 3627.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 3539.50 | 3515.63 | 3627.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 3603.80 | 3541.96 | 3620.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:15:00 | 3602.10 | 3569.23 | 3620.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:30:00 | 3602.80 | 3589.82 | 3621.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 11:15:00 | 3684.00 | 3639.75 | 3636.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — BUY (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 11:15:00 | 3684.00 | 3639.75 | 3636.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 12:15:00 | 3711.80 | 3654.16 | 3643.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 3585.30 | 3668.56 | 3657.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 3585.30 | 3668.56 | 3657.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 3585.30 | 3668.56 | 3657.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 3583.70 | 3668.56 | 3657.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 3560.40 | 3646.93 | 3648.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 3491.60 | 3581.22 | 3611.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 3582.90 | 3568.01 | 3599.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 12:00:00 | 3582.90 | 3568.01 | 3599.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 3626.10 | 3579.63 | 3601.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 3642.70 | 3579.63 | 3601.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 3648.20 | 3593.35 | 3605.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 3648.20 | 3593.35 | 3605.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 3704.50 | 3615.58 | 3614.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 3785.00 | 3663.77 | 3637.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 3622.10 | 3919.55 | 3898.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 3622.10 | 3919.55 | 3898.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 3622.10 | 3919.55 | 3898.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:45:00 | 3592.20 | 3919.55 | 3898.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 3585.40 | 3852.72 | 3869.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 3576.10 | 3702.76 | 3784.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 11:15:00 | 3509.90 | 3508.09 | 3595.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 12:00:00 | 3509.90 | 3508.09 | 3595.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 3450.90 | 3420.85 | 3450.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 3450.90 | 3420.85 | 3450.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 3462.00 | 3429.08 | 3451.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 3462.00 | 3429.08 | 3451.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 3461.00 | 3435.46 | 3452.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 3505.40 | 3435.46 | 3452.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 3472.40 | 3461.09 | 3461.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:45:00 | 3471.40 | 3461.09 | 3461.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 3471.30 | 3463.13 | 3462.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 10:15:00 | 3483.20 | 3469.29 | 3465.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 3475.90 | 3487.33 | 3479.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 12:15:00 | 3475.90 | 3487.33 | 3479.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 3475.90 | 3487.33 | 3479.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:45:00 | 3473.10 | 3487.33 | 3479.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 3493.20 | 3488.50 | 3480.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 14:15:00 | 3496.40 | 3488.50 | 3480.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 3444.40 | 3480.34 | 3480.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 3444.40 | 3480.34 | 3480.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 3426.10 | 3469.49 | 3475.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 09:15:00 | 3424.10 | 3386.61 | 3413.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 3424.10 | 3386.61 | 3413.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 3424.10 | 3386.61 | 3413.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:30:00 | 3427.90 | 3386.61 | 3413.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 3433.00 | 3395.89 | 3415.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 11:00:00 | 3433.00 | 3395.89 | 3415.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 11:15:00 | 3417.50 | 3400.21 | 3415.54 | EMA400 retest candle locked (from downside) |

### Cycle 219 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 3472.10 | 3428.74 | 3425.76 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 3373.50 | 3423.57 | 3424.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 3345.00 | 3407.86 | 3417.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 3306.90 | 3301.36 | 3338.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:15:00 | 3318.40 | 3301.36 | 3338.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 3249.40 | 3223.42 | 3253.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 3287.20 | 3223.42 | 3253.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 3296.80 | 3238.10 | 3257.36 | EMA400 retest candle locked (from downside) |

### Cycle 221 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 3280.30 | 3267.53 | 3266.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 3297.00 | 3273.43 | 3269.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 3248.10 | 3268.36 | 3267.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 3248.10 | 3268.36 | 3267.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 3248.10 | 3268.36 | 3267.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 3249.70 | 3268.36 | 3267.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 3208.50 | 3256.39 | 3262.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 3187.40 | 3242.59 | 3255.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 3189.60 | 3188.05 | 3214.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 3189.60 | 3188.05 | 3214.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 3244.10 | 3202.18 | 3216.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 3240.70 | 3202.18 | 3216.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 3245.50 | 3210.84 | 3219.51 | EMA400 retest candle locked (from downside) |

### Cycle 223 — BUY (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 09:15:00 | 3280.00 | 3229.18 | 3226.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-13 10:15:00 | 3314.50 | 3246.24 | 3234.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 3353.80 | 3408.09 | 3386.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 3353.80 | 3408.09 | 3386.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3353.80 | 3408.09 | 3386.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 3353.70 | 3408.09 | 3386.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 3360.00 | 3398.47 | 3383.89 | EMA400 retest candle locked (from upside) |

### Cycle 224 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 3314.40 | 3375.15 | 3375.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 3310.10 | 3362.14 | 3369.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 3317.10 | 3312.78 | 3333.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 15:00:00 | 3317.10 | 3312.78 | 3333.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 3135.40 | 3122.04 | 3164.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 3290.60 | 3122.04 | 3164.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 3293.10 | 3156.25 | 3175.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 3293.10 | 3156.25 | 3175.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 3263.70 | 3195.81 | 3191.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 3290.90 | 3214.83 | 3200.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 10:15:00 | 3236.80 | 3259.62 | 3233.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 10:15:00 | 3236.80 | 3259.62 | 3233.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 3236.80 | 3259.62 | 3233.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 3249.20 | 3259.62 | 3233.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 3210.00 | 3249.70 | 3230.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 3210.00 | 3249.70 | 3230.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 3239.50 | 3247.66 | 3231.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:30:00 | 3242.30 | 3246.13 | 3232.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:00:00 | 3260.70 | 3249.04 | 3235.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 12:15:00 | 3177.60 | 3220.16 | 3225.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 226 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 3177.60 | 3220.16 | 3225.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 3166.90 | 3203.47 | 3216.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3253.30 | 3211.28 | 3217.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3253.30 | 3211.28 | 3217.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3253.30 | 3211.28 | 3217.89 | EMA400 retest candle locked (from downside) |

### Cycle 227 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 3257.40 | 3228.11 | 3224.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 13:15:00 | 3280.00 | 3245.83 | 3233.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 3155.60 | 3230.41 | 3230.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 3155.60 | 3230.41 | 3230.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 3155.60 | 3230.41 | 3230.13 | EMA400 retest candle locked (from upside) |

### Cycle 228 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 3180.80 | 3220.49 | 3225.65 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 3273.50 | 3220.84 | 3216.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 3286.60 | 3233.99 | 3222.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 3249.00 | 3252.18 | 3235.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 3249.00 | 3252.18 | 3235.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 3238.30 | 3247.98 | 3236.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:30:00 | 3240.10 | 3247.98 | 3236.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 3242.90 | 3246.97 | 3236.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 3421.10 | 3245.24 | 3237.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 3502.40 | 3567.46 | 3570.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 230 — SELL (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 11:15:00 | 3502.40 | 3567.46 | 3570.47 | EMA200 below EMA400 |

### Cycle 231 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 3596.20 | 3559.22 | 3558.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 11:15:00 | 3610.80 | 3569.53 | 3563.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 3590.70 | 3595.97 | 3581.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 3590.70 | 3595.97 | 3581.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 3590.70 | 3595.97 | 3581.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 13:00:00 | 3640.00 | 3607.37 | 3590.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 3572.80 | 3601.89 | 3591.16 | SL hit (close<static) qty=1.00 sl=3577.00 alert=retest2 |

### Cycle 232 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 3546.20 | 3585.70 | 3586.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 3541.10 | 3563.62 | 3573.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 3509.40 | 3504.26 | 3524.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 12:45:00 | 3510.50 | 3504.26 | 3524.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 3521.90 | 3507.79 | 3523.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 3521.90 | 3507.79 | 3523.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 3498.40 | 3505.91 | 3521.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 3490.00 | 3505.91 | 3521.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 3490.20 | 3501.47 | 3515.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:30:00 | 3488.50 | 3498.31 | 3506.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 3489.70 | 3496.69 | 3504.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 3491.60 | 3447.35 | 3461.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 3502.00 | 3447.35 | 3461.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 3478.60 | 3453.60 | 3463.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 3485.00 | 3453.60 | 3463.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-04 14:15:00 | 3491.00 | 3472.34 | 3470.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 3491.00 | 3472.34 | 3470.17 | EMA200 above EMA400 |

### Cycle 234 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 3423.70 | 3466.40 | 3468.08 | EMA200 below EMA400 |

### Cycle 235 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 3480.70 | 3462.51 | 3461.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 3491.50 | 3470.99 | 3465.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 3532.50 | 3557.95 | 3529.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 3532.50 | 3557.95 | 3529.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 3532.50 | 3557.95 | 3529.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 3520.10 | 3557.95 | 3529.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 3535.20 | 3553.40 | 3529.63 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-15 12:30:00 | 1664.00 | 2024-04-15 13:15:00 | 1660.20 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-04-15 13:15:00 | 1663.95 | 2024-04-15 13:15:00 | 1660.20 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-05-14 15:15:00 | 1680.00 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2024-05-16 09:15:00 | 1699.55 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-05-17 11:45:00 | 1680.70 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2024-05-17 13:00:00 | 1679.90 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2024-05-18 09:15:00 | 1705.00 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-05-21 12:00:00 | 1685.25 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2024-05-21 13:00:00 | 1685.15 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2024-05-21 13:45:00 | 1685.05 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2024-05-22 10:45:00 | 1708.70 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-05-22 14:15:00 | 1708.80 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-05-22 15:00:00 | 1709.05 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-05-23 10:45:00 | 1709.00 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-05-24 12:30:00 | 1707.60 | 2024-05-24 14:15:00 | 1694.45 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-06-11 12:00:00 | 1754.95 | 2024-06-14 14:15:00 | 1770.95 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2024-06-11 12:45:00 | 1753.05 | 2024-06-14 14:15:00 | 1770.95 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2024-06-11 14:15:00 | 1755.00 | 2024-06-14 14:15:00 | 1770.95 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2024-06-24 14:15:00 | 1734.55 | 2024-06-25 12:15:00 | 1765.70 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-06-24 14:45:00 | 1734.15 | 2024-06-25 12:15:00 | 1765.70 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-06-25 10:45:00 | 1733.85 | 2024-06-25 12:15:00 | 1765.70 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-06-25 11:30:00 | 1734.80 | 2024-06-25 12:15:00 | 1765.70 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-06-27 10:30:00 | 1817.15 | 2024-07-01 13:15:00 | 1792.50 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-06-27 14:45:00 | 1824.50 | 2024-07-01 13:15:00 | 1792.50 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-07-16 09:45:00 | 1848.80 | 2024-07-19 09:15:00 | 1828.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-07-16 13:30:00 | 1840.75 | 2024-07-19 09:15:00 | 1828.70 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-07-16 14:00:00 | 1843.80 | 2024-07-19 09:15:00 | 1828.70 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-07-18 09:15:00 | 1865.40 | 2024-07-19 09:15:00 | 1828.70 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-07-30 10:15:00 | 1798.00 | 2024-08-05 12:15:00 | 1817.25 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2024-08-23 12:30:00 | 1908.45 | 2024-09-02 15:15:00 | 1957.20 | STOP_HIT | 1.00 | 2.55% |
| BUY | retest2 | 2024-08-23 13:45:00 | 1907.45 | 2024-09-02 15:15:00 | 1957.20 | STOP_HIT | 1.00 | 2.61% |
| BUY | retest2 | 2024-08-26 09:15:00 | 1909.75 | 2024-09-02 15:15:00 | 1957.20 | STOP_HIT | 1.00 | 2.48% |
| BUY | retest2 | 2024-09-09 12:45:00 | 1987.25 | 2024-09-10 09:15:00 | 1966.75 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-09-09 14:30:00 | 1988.45 | 2024-09-10 09:15:00 | 1966.75 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-09-17 11:00:00 | 2026.80 | 2024-09-20 09:15:00 | 1986.15 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-09-18 13:45:00 | 2023.75 | 2024-09-20 09:15:00 | 1986.15 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-09-18 15:00:00 | 2019.90 | 2024-09-20 09:15:00 | 1986.15 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-09-19 09:15:00 | 2053.90 | 2024-09-20 09:15:00 | 1986.15 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2024-09-26 14:00:00 | 2040.00 | 2024-10-01 09:15:00 | 1958.70 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2024-09-30 14:45:00 | 2038.10 | 2024-10-01 09:15:00 | 1958.70 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2024-10-04 13:00:00 | 1932.45 | 2024-10-09 12:15:00 | 1940.95 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-10-07 09:15:00 | 1928.60 | 2024-10-09 12:15:00 | 1940.95 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-10-09 11:30:00 | 1930.30 | 2024-10-09 12:15:00 | 1940.95 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-10-15 09:15:00 | 1956.00 | 2024-10-18 09:15:00 | 1941.55 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-10-15 11:30:00 | 1952.80 | 2024-10-18 09:15:00 | 1941.55 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-10-16 14:00:00 | 1956.50 | 2024-10-18 09:15:00 | 1941.55 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-10-17 12:15:00 | 1951.40 | 2024-10-18 09:15:00 | 1941.55 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-10-22 11:30:00 | 1926.10 | 2024-10-29 12:15:00 | 1963.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-10-22 12:30:00 | 1928.05 | 2024-10-29 12:15:00 | 1963.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-10-23 14:15:00 | 1930.50 | 2024-10-29 12:15:00 | 1963.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-10-24 14:00:00 | 1929.95 | 2024-10-29 12:15:00 | 1963.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-10-25 10:15:00 | 1906.35 | 2024-10-29 12:15:00 | 1963.00 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2024-10-25 11:00:00 | 1919.05 | 2024-10-29 12:15:00 | 1963.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-10-25 12:15:00 | 1921.80 | 2024-10-29 12:15:00 | 1963.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-10-25 13:00:00 | 1921.95 | 2024-10-29 12:15:00 | 1963.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-11-04 10:30:00 | 1901.55 | 2024-11-07 11:15:00 | 1806.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-04 11:45:00 | 1900.55 | 2024-11-07 11:15:00 | 1805.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-04 12:45:00 | 1897.00 | 2024-11-07 11:15:00 | 1802.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-04 13:15:00 | 1897.55 | 2024-11-07 11:15:00 | 1802.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 14:30:00 | 1901.05 | 2024-11-07 11:15:00 | 1806.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-04 10:30:00 | 1901.55 | 2024-11-11 10:15:00 | 1815.55 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2024-11-04 11:45:00 | 1900.55 | 2024-11-11 10:15:00 | 1815.55 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2024-11-04 12:45:00 | 1897.00 | 2024-11-11 10:15:00 | 1815.55 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2024-11-04 13:15:00 | 1897.55 | 2024-11-11 10:15:00 | 1815.55 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2024-11-06 14:30:00 | 1901.05 | 2024-11-11 10:15:00 | 1815.55 | STOP_HIT | 0.50 | 4.50% |
| BUY | retest2 | 2024-12-04 13:30:00 | 1939.90 | 2024-12-12 14:15:00 | 2131.80 | TARGET_HIT | 1.00 | 9.89% |
| BUY | retest2 | 2024-12-05 12:00:00 | 1942.60 | 2024-12-19 14:15:00 | 2133.89 | TARGET_HIT | 1.00 | 9.85% |
| BUY | retest2 | 2024-12-05 15:00:00 | 1938.00 | 2024-12-19 14:15:00 | 2136.86 | TARGET_HIT | 1.00 | 10.26% |
| SELL | retest1 | 2025-01-13 09:15:00 | 2118.35 | 2025-01-14 10:15:00 | 2135.40 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-01-14 14:30:00 | 2121.30 | 2025-01-15 10:15:00 | 2163.40 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-01-23 09:30:00 | 2203.10 | 2025-01-24 09:15:00 | 2164.50 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-01-23 10:00:00 | 2215.80 | 2025-01-24 09:15:00 | 2164.50 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-01-28 15:00:00 | 2135.40 | 2025-01-29 09:15:00 | 2182.40 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest1 | 2025-02-20 09:15:00 | 2287.15 | 2025-02-21 09:15:00 | 2229.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-02-20 14:45:00 | 2280.15 | 2025-02-21 09:15:00 | 2229.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-02-25 12:30:00 | 2191.65 | 2025-02-27 09:15:00 | 2221.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-02-25 13:15:00 | 2190.55 | 2025-02-27 09:15:00 | 2221.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-02-25 14:00:00 | 2180.80 | 2025-02-27 09:15:00 | 2221.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-03-04 09:15:00 | 2132.95 | 2025-03-06 09:15:00 | 2162.55 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-03-04 10:00:00 | 2144.80 | 2025-03-06 09:15:00 | 2162.55 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-03-04 10:30:00 | 2136.35 | 2025-03-06 09:15:00 | 2162.55 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-03-04 14:30:00 | 2143.90 | 2025-03-06 09:15:00 | 2162.55 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-03-07 13:15:00 | 2172.25 | 2025-03-19 10:15:00 | 2389.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-07 14:00:00 | 2176.30 | 2025-03-19 10:15:00 | 2393.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-11 09:45:00 | 2171.40 | 2025-03-19 10:15:00 | 2388.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-26 10:15:00 | 2336.40 | 2025-03-28 10:15:00 | 2360.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-03-26 11:15:00 | 2332.05 | 2025-03-28 10:15:00 | 2360.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-03-27 15:15:00 | 2321.00 | 2025-03-28 10:15:00 | 2360.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-04-16 10:45:00 | 2048.80 | 2025-04-17 09:15:00 | 2106.10 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-04-16 11:15:00 | 2048.30 | 2025-04-17 09:15:00 | 2106.10 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2025-04-16 12:00:00 | 2050.80 | 2025-04-17 09:15:00 | 2106.10 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-04-16 13:15:00 | 2051.60 | 2025-04-17 09:15:00 | 2106.10 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-05-02 13:00:00 | 2170.10 | 2025-05-09 09:15:00 | 2205.30 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest2 | 2025-05-02 14:00:00 | 2176.10 | 2025-05-09 09:15:00 | 2205.30 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-05-02 14:30:00 | 2170.50 | 2025-05-09 09:15:00 | 2205.30 | STOP_HIT | 1.00 | 1.60% |
| SELL | retest2 | 2025-05-14 12:30:00 | 2238.10 | 2025-05-14 13:15:00 | 2255.10 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-06-13 12:00:00 | 2587.50 | 2025-06-20 11:15:00 | 2598.90 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-06-24 10:15:00 | 2550.00 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-06-25 10:00:00 | 2548.80 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-06-25 12:30:00 | 2550.00 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-06-25 13:30:00 | 2551.50 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-06-26 10:45:00 | 2557.50 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-06-27 09:15:00 | 2541.90 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-06-27 11:00:00 | 2553.10 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-07-03 09:15:00 | 2636.90 | 2025-07-08 13:15:00 | 2636.70 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-07-03 15:00:00 | 2635.30 | 2025-07-08 13:15:00 | 2636.70 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-07-04 09:45:00 | 2634.00 | 2025-07-08 13:15:00 | 2636.70 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-07-08 11:30:00 | 2635.10 | 2025-07-08 13:15:00 | 2636.70 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-07-10 11:00:00 | 2639.30 | 2025-07-14 09:15:00 | 2661.80 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-07-11 09:15:00 | 2626.80 | 2025-07-14 09:15:00 | 2661.80 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-07-14 09:15:00 | 2627.60 | 2025-07-14 09:15:00 | 2661.80 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-22 11:30:00 | 2673.40 | 2025-07-24 14:15:00 | 2663.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-07-23 10:00:00 | 2670.50 | 2025-07-24 14:15:00 | 2663.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-07-23 10:30:00 | 2674.50 | 2025-07-24 14:15:00 | 2663.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-08-01 15:15:00 | 2582.50 | 2025-08-04 09:15:00 | 2637.00 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-08-05 15:00:00 | 2644.30 | 2025-08-06 10:15:00 | 2622.90 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-08-06 12:15:00 | 2645.20 | 2025-08-06 15:15:00 | 2616.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-08-06 13:15:00 | 2643.60 | 2025-08-06 15:15:00 | 2616.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-08-26 09:15:00 | 2668.50 | 2025-09-01 10:15:00 | 2706.90 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-08-26 14:00:00 | 2676.30 | 2025-09-01 10:15:00 | 2706.90 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-08-26 14:30:00 | 2675.30 | 2025-09-01 10:15:00 | 2706.90 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-08-28 11:15:00 | 2675.60 | 2025-09-01 11:15:00 | 2714.10 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-08-28 13:00:00 | 2665.90 | 2025-09-01 11:15:00 | 2714.10 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-08-28 13:30:00 | 2670.10 | 2025-09-01 11:15:00 | 2714.10 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-08-29 12:15:00 | 2670.40 | 2025-09-01 11:15:00 | 2714.10 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-09-03 09:15:00 | 2754.70 | 2025-09-11 13:15:00 | 2879.40 | STOP_HIT | 1.00 | 4.53% |
| SELL | retest2 | 2025-09-12 11:45:00 | 2885.80 | 2025-09-12 12:15:00 | 2915.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-10-01 09:30:00 | 3094.40 | 2025-10-10 10:15:00 | 3181.80 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2025-10-20 14:30:00 | 3337.90 | 2025-10-21 13:15:00 | 3278.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-10-28 14:15:00 | 3173.20 | 2025-10-30 09:15:00 | 3182.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-11-13 10:30:00 | 3382.10 | 2025-11-14 09:15:00 | 3720.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-28 14:15:00 | 3752.90 | 2025-12-03 09:15:00 | 3727.60 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-12-01 09:15:00 | 3785.80 | 2025-12-03 09:15:00 | 3727.60 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-12-22 09:30:00 | 3764.80 | 2025-12-22 14:15:00 | 3789.80 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-12-22 10:30:00 | 3767.00 | 2025-12-22 14:15:00 | 3789.80 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-12-22 11:00:00 | 3760.20 | 2025-12-22 14:15:00 | 3789.80 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-12-26 09:15:00 | 3842.90 | 2025-12-29 09:15:00 | 3781.70 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-12-26 11:15:00 | 3814.90 | 2025-12-29 09:15:00 | 3781.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-12-26 13:45:00 | 3815.40 | 2025-12-29 09:15:00 | 3781.70 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-05 09:15:00 | 3889.10 | 2026-01-08 11:15:00 | 3840.10 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-01-14 12:15:00 | 3943.80 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-01-14 14:30:00 | 3940.40 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-01-14 15:00:00 | 3940.00 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-01-16 10:15:00 | 3950.10 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-01-19 09:15:00 | 3963.60 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-01-19 12:15:00 | 3945.00 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-01-19 13:00:00 | 3946.70 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-23 12:00:00 | 3876.00 | 2026-01-28 10:15:00 | 3946.30 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-02-03 12:15:00 | 3602.10 | 2026-02-04 11:15:00 | 3684.00 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2026-02-03 13:30:00 | 3602.80 | 2026-02-04 11:15:00 | 3684.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-02-25 14:15:00 | 3496.40 | 2026-02-26 11:15:00 | 3444.40 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-03-27 13:30:00 | 3242.30 | 2026-03-30 12:15:00 | 3177.60 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-03-27 15:00:00 | 3260.70 | 2026-03-30 12:15:00 | 3177.60 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2026-04-08 09:15:00 | 3421.10 | 2026-04-17 11:15:00 | 3502.40 | STOP_HIT | 1.00 | 2.38% |
| BUY | retest2 | 2026-04-22 13:00:00 | 3640.00 | 2026-04-22 14:15:00 | 3572.80 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-04-27 15:15:00 | 3490.00 | 2026-05-04 14:15:00 | 3491.00 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2026-04-28 11:15:00 | 3490.20 | 2026-05-04 14:15:00 | 3491.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2026-04-29 09:30:00 | 3488.50 | 2026-05-04 14:15:00 | 3491.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2026-04-29 10:30:00 | 3489.70 | 2026-05-04 14:15:00 | 3491.00 | STOP_HIT | 1.00 | -0.04% |
