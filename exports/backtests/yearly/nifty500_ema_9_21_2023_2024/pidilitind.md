# Pidilite Industries Ltd. (PIDILITIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1472.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 248 |
| ALERT1 | 162 |
| ALERT2 | 159 |
| ALERT2_SKIP | 107 |
| ALERT3 | 342 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 148 |
| PARTIAL | 11 |
| TARGET_HIT | 0 |
| STOP_HIT | 149 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 160 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 116
- **Target hits / Stop hits / Partials:** 0 / 149 / 11
- **Avg / median % per leg:** 0.10% / -0.50%
- **Sum % (uncompounded):** 15.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 7 | 11.1% | 0 | 63 | 0 | -0.52% | -32.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 63 | 7 | 11.1% | 0 | 63 | 0 | -0.52% | -32.6% |
| SELL (all) | 97 | 37 | 38.1% | 0 | 86 | 11 | 0.49% | 47.9% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 2.04% | 2.0% |
| SELL @ 3rd Alert (retest2) | 96 | 36 | 37.5% | 0 | 85 | 11 | 0.48% | 45.8% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 2.04% | 2.0% |
| retest2 (combined) | 159 | 43 | 27.0% | 0 | 148 | 11 | 0.08% | 13.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 15:15:00 | 1274.45 | 1278.90 | 1279.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 10:15:00 | 1268.75 | 1276.09 | 1277.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 15:15:00 | 1277.60 | 1272.54 | 1274.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 15:15:00 | 1277.60 | 1272.54 | 1274.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 15:15:00 | 1277.60 | 1272.54 | 1274.92 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 11:15:00 | 1280.00 | 1276.76 | 1276.47 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 1274.58 | 1276.49 | 1276.57 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 10:15:00 | 1280.13 | 1277.22 | 1276.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 11:15:00 | 1282.25 | 1278.23 | 1277.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-22 09:15:00 | 1276.97 | 1280.45 | 1279.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 1276.97 | 1280.45 | 1279.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 1276.97 | 1280.45 | 1279.05 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 11:15:00 | 1266.43 | 1276.67 | 1277.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-22 13:15:00 | 1264.50 | 1272.57 | 1275.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 10:15:00 | 1275.28 | 1272.01 | 1274.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 10:15:00 | 1275.28 | 1272.01 | 1274.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 10:15:00 | 1275.28 | 1272.01 | 1274.07 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 10:15:00 | 1276.90 | 1264.70 | 1264.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 11:15:00 | 1279.50 | 1267.66 | 1265.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 15:15:00 | 1293.70 | 1294.30 | 1288.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 11:15:00 | 1312.80 | 1318.99 | 1312.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 11:15:00 | 1312.80 | 1318.99 | 1312.20 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 13:15:00 | 1310.25 | 1315.09 | 1315.30 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 1326.38 | 1316.55 | 1315.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 10:15:00 | 1331.53 | 1319.54 | 1317.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 10:15:00 | 1322.25 | 1324.09 | 1321.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 10:15:00 | 1322.25 | 1324.09 | 1321.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 10:15:00 | 1322.25 | 1324.09 | 1321.26 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 14:15:00 | 1317.00 | 1319.77 | 1319.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 10:15:00 | 1312.93 | 1317.84 | 1318.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 11:15:00 | 1320.88 | 1312.77 | 1314.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 11:15:00 | 1320.88 | 1312.77 | 1314.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 11:15:00 | 1320.88 | 1312.77 | 1314.78 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 13:15:00 | 1326.18 | 1316.67 | 1316.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 1360.83 | 1326.96 | 1321.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 15:15:00 | 1340.00 | 1341.80 | 1333.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 12:15:00 | 1343.30 | 1347.11 | 1342.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 12:15:00 | 1343.30 | 1347.11 | 1342.62 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 13:15:00 | 1335.00 | 1341.70 | 1342.46 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 1346.50 | 1342.13 | 1341.90 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 13:15:00 | 1339.80 | 1341.78 | 1341.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 14:15:00 | 1337.00 | 1340.82 | 1341.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 09:15:00 | 1340.60 | 1339.93 | 1340.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 1340.60 | 1339.93 | 1340.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 1340.60 | 1339.93 | 1340.84 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 11:15:00 | 1304.00 | 1296.70 | 1296.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 09:15:00 | 1318.33 | 1304.22 | 1300.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 10:15:00 | 1318.48 | 1319.08 | 1311.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 12:15:00 | 1314.85 | 1317.77 | 1312.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 1314.85 | 1317.77 | 1312.34 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 10:15:00 | 1298.10 | 1308.42 | 1309.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 11:15:00 | 1292.63 | 1305.27 | 1307.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 1313.18 | 1302.94 | 1305.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 1313.18 | 1302.94 | 1305.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 1313.18 | 1302.94 | 1305.20 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 1341.48 | 1310.65 | 1308.50 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 10:15:00 | 1294.53 | 1310.92 | 1311.22 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 09:15:00 | 1323.50 | 1311.48 | 1310.81 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 15:15:00 | 1309.50 | 1310.83 | 1310.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 09:15:00 | 1301.08 | 1308.88 | 1310.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 12:15:00 | 1307.22 | 1307.17 | 1308.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 12:15:00 | 1307.22 | 1307.17 | 1308.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 1307.22 | 1307.17 | 1308.85 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 1321.23 | 1311.26 | 1310.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 12:15:00 | 1328.98 | 1319.68 | 1316.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 09:15:00 | 1336.00 | 1336.34 | 1329.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 09:15:00 | 1336.63 | 1337.92 | 1333.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 1336.63 | 1337.92 | 1333.79 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 09:15:00 | 1327.83 | 1333.71 | 1333.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 12:15:00 | 1320.48 | 1327.86 | 1330.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 11:15:00 | 1312.65 | 1310.01 | 1316.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 1306.30 | 1308.94 | 1313.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 1306.30 | 1308.94 | 1313.32 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 13:15:00 | 1316.25 | 1309.23 | 1308.49 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-08-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 11:15:00 | 1305.90 | 1309.86 | 1310.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 12:15:00 | 1294.65 | 1306.82 | 1308.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 1303.03 | 1300.38 | 1304.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 1303.03 | 1300.38 | 1304.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 1303.03 | 1300.38 | 1304.35 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 13:15:00 | 1307.15 | 1305.42 | 1305.22 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 10:15:00 | 1302.70 | 1304.67 | 1304.92 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 12:15:00 | 1312.35 | 1306.42 | 1305.69 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 11:15:00 | 1301.25 | 1305.24 | 1305.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 09:15:00 | 1286.97 | 1299.94 | 1302.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 13:15:00 | 1273.33 | 1272.82 | 1282.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 09:15:00 | 1271.10 | 1269.12 | 1273.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 1271.10 | 1269.12 | 1273.80 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 09:15:00 | 1269.50 | 1250.23 | 1249.26 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 14:15:00 | 1256.80 | 1259.55 | 1259.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 15:15:00 | 1255.50 | 1258.74 | 1259.26 | Break + close below crossover candle low |

### Cycle 30 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 1273.00 | 1261.59 | 1260.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 10:15:00 | 1283.45 | 1265.96 | 1262.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 1285.93 | 1286.03 | 1278.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 1279.65 | 1285.47 | 1279.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 1279.65 | 1285.47 | 1279.95 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 15:15:00 | 1252.53 | 1274.46 | 1276.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-04 09:15:00 | 1238.50 | 1252.01 | 1261.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 09:15:00 | 1246.90 | 1243.24 | 1251.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 10:15:00 | 1249.88 | 1244.57 | 1251.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 10:15:00 | 1249.88 | 1244.57 | 1251.20 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-09-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 11:15:00 | 1252.10 | 1249.98 | 1249.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 12:15:00 | 1255.03 | 1250.99 | 1250.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 09:15:00 | 1249.90 | 1251.67 | 1250.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 1249.90 | 1251.67 | 1250.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 1249.90 | 1251.67 | 1250.94 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 1244.38 | 1251.45 | 1252.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 1235.90 | 1245.07 | 1248.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 14:15:00 | 1243.25 | 1240.52 | 1244.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 14:15:00 | 1243.25 | 1240.52 | 1244.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 1243.25 | 1240.52 | 1244.45 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 11:15:00 | 1248.68 | 1243.51 | 1243.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 12:15:00 | 1250.93 | 1244.99 | 1243.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 13:15:00 | 1244.60 | 1244.91 | 1244.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 13:15:00 | 1244.60 | 1244.91 | 1244.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 13:15:00 | 1244.60 | 1244.91 | 1244.03 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 11:15:00 | 1245.90 | 1250.07 | 1250.52 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 13:15:00 | 1251.05 | 1249.99 | 1249.95 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 09:15:00 | 1242.55 | 1248.68 | 1249.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 09:15:00 | 1241.10 | 1247.01 | 1248.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 14:15:00 | 1205.00 | 1204.83 | 1212.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 1215.50 | 1206.75 | 1211.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 1215.50 | 1206.75 | 1211.93 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 1218.97 | 1214.21 | 1213.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 14:15:00 | 1227.63 | 1220.32 | 1217.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 1208.33 | 1219.15 | 1217.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 1208.33 | 1219.15 | 1217.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 1208.33 | 1219.15 | 1217.40 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 13:15:00 | 1212.10 | 1215.79 | 1216.22 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 11:15:00 | 1220.20 | 1216.06 | 1216.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 12:15:00 | 1221.95 | 1217.24 | 1216.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 09:15:00 | 1239.93 | 1240.16 | 1235.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 1239.93 | 1240.16 | 1235.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 1239.93 | 1240.16 | 1235.50 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 15:15:00 | 1229.47 | 1233.61 | 1233.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 09:15:00 | 1222.35 | 1231.36 | 1232.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 09:15:00 | 1216.00 | 1215.45 | 1219.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 1166.18 | 1156.43 | 1164.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1166.18 | 1156.43 | 1164.45 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 1200.58 | 1165.45 | 1162.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 10:15:00 | 1226.78 | 1177.71 | 1168.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 10:15:00 | 1212.93 | 1213.89 | 1196.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 1200.55 | 1207.50 | 1200.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 1200.55 | 1207.50 | 1200.50 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 10:15:00 | 1214.88 | 1225.39 | 1225.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 11:15:00 | 1211.43 | 1216.98 | 1220.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 1220.43 | 1215.74 | 1218.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 1220.43 | 1215.74 | 1218.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 1220.43 | 1215.74 | 1218.13 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-11-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 14:15:00 | 1229.85 | 1221.12 | 1219.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 15:15:00 | 1234.08 | 1223.71 | 1221.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 09:15:00 | 1246.95 | 1249.22 | 1240.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 10:15:00 | 1241.97 | 1247.77 | 1240.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 1241.97 | 1247.77 | 1240.80 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 11:15:00 | 1243.60 | 1247.25 | 1247.46 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 12:15:00 | 1255.00 | 1248.36 | 1247.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 11:15:00 | 1264.78 | 1253.18 | 1250.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 12:15:00 | 1276.00 | 1278.17 | 1271.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 1283.75 | 1278.92 | 1274.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 1283.75 | 1278.92 | 1274.12 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 1278.00 | 1289.69 | 1290.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 11:15:00 | 1273.00 | 1280.58 | 1285.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 09:15:00 | 1281.50 | 1277.08 | 1281.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 1281.50 | 1277.08 | 1281.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 1281.50 | 1277.08 | 1281.15 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 11:15:00 | 1301.97 | 1284.38 | 1283.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 14:15:00 | 1304.25 | 1293.55 | 1288.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 09:15:00 | 1292.22 | 1294.57 | 1290.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 1292.22 | 1294.57 | 1290.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 1292.22 | 1294.57 | 1290.07 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 1293.58 | 1316.30 | 1319.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 1290.00 | 1311.04 | 1316.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 1310.10 | 1303.67 | 1309.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 14:15:00 | 1310.10 | 1303.67 | 1309.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 1310.10 | 1303.67 | 1309.24 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 12:15:00 | 1323.00 | 1312.71 | 1311.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 11:15:00 | 1328.50 | 1320.90 | 1316.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 12:15:00 | 1360.18 | 1360.23 | 1351.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 13:15:00 | 1349.55 | 1358.09 | 1350.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 13:15:00 | 1349.55 | 1358.09 | 1350.94 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 1339.98 | 1349.79 | 1350.55 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 12:15:00 | 1361.00 | 1351.93 | 1351.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-02 13:15:00 | 1367.53 | 1355.05 | 1352.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 09:15:00 | 1379.38 | 1382.57 | 1376.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 09:15:00 | 1379.38 | 1382.57 | 1376.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 1379.38 | 1382.57 | 1376.72 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 13:15:00 | 1359.50 | 1373.40 | 1373.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 09:15:00 | 1352.50 | 1365.39 | 1369.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-08 11:15:00 | 1363.33 | 1362.12 | 1367.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 1361.00 | 1360.94 | 1364.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 1361.00 | 1360.94 | 1364.75 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-01-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 13:15:00 | 1368.13 | 1361.58 | 1361.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 13:15:00 | 1372.30 | 1366.00 | 1363.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 15:15:00 | 1364.63 | 1366.26 | 1364.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 15:15:00 | 1364.63 | 1366.26 | 1364.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 15:15:00 | 1364.63 | 1366.26 | 1364.49 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 12:15:00 | 1367.08 | 1375.06 | 1375.25 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 15:15:00 | 1382.00 | 1376.12 | 1375.61 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 1343.35 | 1369.56 | 1372.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 1295.00 | 1333.72 | 1343.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 1294.45 | 1288.25 | 1310.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 1294.45 | 1288.25 | 1310.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 1294.45 | 1288.25 | 1310.87 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 14:15:00 | 1278.33 | 1268.38 | 1268.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 09:15:00 | 1291.72 | 1274.64 | 1271.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 10:15:00 | 1325.00 | 1331.48 | 1315.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 12:15:00 | 1314.00 | 1326.31 | 1315.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 12:15:00 | 1314.00 | 1326.31 | 1315.74 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 15:15:00 | 1325.00 | 1330.15 | 1330.76 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 1336.78 | 1331.48 | 1331.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 10:15:00 | 1357.75 | 1348.95 | 1341.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 13:15:00 | 1366.25 | 1368.60 | 1362.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 14:15:00 | 1367.23 | 1368.33 | 1362.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 1367.23 | 1368.33 | 1362.56 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 15:15:00 | 1356.53 | 1360.44 | 1360.75 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 12:15:00 | 1369.65 | 1362.00 | 1361.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 13:15:00 | 1374.93 | 1364.58 | 1362.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 09:15:00 | 1359.53 | 1364.55 | 1363.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 09:15:00 | 1359.53 | 1364.55 | 1363.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 1359.53 | 1364.55 | 1363.13 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 12:15:00 | 1357.15 | 1361.30 | 1361.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 15:15:00 | 1356.00 | 1359.58 | 1360.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 09:15:00 | 1365.70 | 1360.80 | 1361.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 1365.70 | 1360.80 | 1361.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 1365.70 | 1360.80 | 1361.29 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 10:15:00 | 1367.80 | 1362.20 | 1361.88 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 13:15:00 | 1357.78 | 1361.14 | 1361.45 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 14:15:00 | 1366.83 | 1362.28 | 1361.94 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 09:15:00 | 1357.08 | 1361.45 | 1361.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 10:15:00 | 1353.50 | 1359.86 | 1360.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 10:15:00 | 1353.70 | 1353.09 | 1356.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 11:15:00 | 1361.78 | 1354.83 | 1356.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 1361.78 | 1354.83 | 1356.69 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-02-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 13:15:00 | 1367.25 | 1358.56 | 1358.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 09:15:00 | 1378.00 | 1364.48 | 1361.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-02 09:15:00 | 1373.85 | 1374.24 | 1368.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 11:15:00 | 1370.58 | 1373.51 | 1369.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 1370.58 | 1373.51 | 1369.01 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 11:15:00 | 1358.60 | 1365.90 | 1366.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 14:15:00 | 1350.75 | 1360.42 | 1363.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 09:15:00 | 1361.45 | 1358.88 | 1362.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 09:15:00 | 1361.45 | 1358.88 | 1362.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 1361.45 | 1358.88 | 1362.29 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 11:15:00 | 1384.00 | 1363.64 | 1361.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-06 12:15:00 | 1389.43 | 1368.80 | 1364.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 09:15:00 | 1441.25 | 1444.79 | 1429.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 14:15:00 | 1429.45 | 1436.78 | 1431.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 14:15:00 | 1429.45 | 1436.78 | 1431.11 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 12:15:00 | 1413.83 | 1427.15 | 1428.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 14:15:00 | 1410.23 | 1421.58 | 1425.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 11:15:00 | 1430.58 | 1419.82 | 1423.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 11:15:00 | 1430.58 | 1419.82 | 1423.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 1430.58 | 1419.82 | 1423.03 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-03-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 15:15:00 | 1435.00 | 1424.74 | 1424.44 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-03-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 09:15:00 | 1420.00 | 1423.79 | 1424.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 10:15:00 | 1412.00 | 1421.43 | 1422.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 11:15:00 | 1421.65 | 1421.48 | 1422.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 12:15:00 | 1429.28 | 1423.04 | 1423.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 12:15:00 | 1429.28 | 1423.04 | 1423.41 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-03-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 13:15:00 | 1433.58 | 1425.14 | 1424.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 09:15:00 | 1440.95 | 1429.85 | 1426.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 09:15:00 | 1447.80 | 1453.11 | 1442.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 09:15:00 | 1447.80 | 1453.11 | 1442.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 1447.80 | 1453.11 | 1442.96 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 10:15:00 | 1421.93 | 1438.51 | 1440.34 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 14:15:00 | 1450.38 | 1439.93 | 1438.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 15:15:00 | 1453.00 | 1442.54 | 1439.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 14:15:00 | 1494.90 | 1503.06 | 1490.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 15:15:00 | 1505.00 | 1503.45 | 1491.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 1505.00 | 1503.45 | 1491.60 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 14:15:00 | 1497.05 | 1506.57 | 1507.32 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 1515.00 | 1504.58 | 1503.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 10:15:00 | 1519.25 | 1507.51 | 1505.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 09:15:00 | 1514.23 | 1515.33 | 1510.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 09:15:00 | 1514.23 | 1515.33 | 1510.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 1514.23 | 1515.33 | 1510.87 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 14:15:00 | 1506.53 | 1508.61 | 1508.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 10:15:00 | 1501.13 | 1506.15 | 1507.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 10:15:00 | 1505.48 | 1502.74 | 1504.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 10:15:00 | 1505.48 | 1502.74 | 1504.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 1505.48 | 1502.74 | 1504.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 10:45:00 | 1508.00 | 1502.74 | 1504.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 11:15:00 | 1501.93 | 1502.58 | 1504.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 12:15:00 | 1499.60 | 1502.58 | 1504.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 1424.62 | 1439.56 | 1450.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-22 09:15:00 | 1425.43 | 1421.57 | 1433.42 | SL hit (close>ema200) qty=0.50 sl=1421.57 alert=retest2 |

### Cycle 80 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 1449.53 | 1439.37 | 1438.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 15:15:00 | 1452.08 | 1441.91 | 1440.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 10:15:00 | 1442.05 | 1443.41 | 1441.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 10:15:00 | 1442.05 | 1443.41 | 1441.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 1442.05 | 1443.41 | 1441.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:45:00 | 1441.65 | 1443.41 | 1441.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 1441.60 | 1443.05 | 1441.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 12:30:00 | 1444.33 | 1444.26 | 1441.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 13:15:00 | 1490.98 | 1502.42 | 1502.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 13:15:00 | 1490.98 | 1502.42 | 1502.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 1481.73 | 1494.65 | 1498.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 15:15:00 | 1477.90 | 1476.76 | 1483.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-07 09:15:00 | 1476.43 | 1476.76 | 1483.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 1476.98 | 1476.80 | 1482.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:30:00 | 1479.05 | 1476.80 | 1482.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 1470.65 | 1475.57 | 1481.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 1412.40 | 1475.11 | 1479.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-10 12:15:00 | 1454.30 | 1437.61 | 1437.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-05-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 12:15:00 | 1454.30 | 1437.61 | 1437.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 09:15:00 | 1469.50 | 1451.38 | 1444.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 1487.18 | 1488.32 | 1477.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 10:00:00 | 1487.18 | 1488.32 | 1477.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 1489.88 | 1488.29 | 1479.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:30:00 | 1480.18 | 1488.29 | 1479.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 1499.03 | 1501.82 | 1496.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 11:45:00 | 1496.40 | 1501.82 | 1496.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 1498.83 | 1501.22 | 1496.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 12:30:00 | 1498.68 | 1501.22 | 1496.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 1495.83 | 1500.14 | 1496.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:45:00 | 1495.95 | 1500.14 | 1496.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 1495.15 | 1499.14 | 1496.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 1495.15 | 1499.14 | 1496.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 1495.00 | 1498.32 | 1496.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:45:00 | 1499.85 | 1498.62 | 1496.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 1509.13 | 1500.72 | 1497.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 1509.00 | 1500.72 | 1497.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1512.03 | 1504.40 | 1499.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 1525.03 | 1509.19 | 1504.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 11:15:00 | 1490.50 | 1503.20 | 1504.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 1490.50 | 1503.20 | 1504.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 1478.30 | 1494.41 | 1499.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 10:15:00 | 1498.63 | 1495.25 | 1499.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 10:15:00 | 1498.63 | 1495.25 | 1499.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1498.63 | 1495.25 | 1499.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:00:00 | 1498.63 | 1495.25 | 1499.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 1489.10 | 1494.02 | 1498.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 12:15:00 | 1488.88 | 1494.02 | 1498.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 12:15:00 | 1502.48 | 1495.58 | 1496.07 | SL hit (close>static) qty=1.00 sl=1502.45 alert=retest2 |

### Cycle 84 — BUY (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 13:15:00 | 1513.98 | 1499.26 | 1497.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 09:15:00 | 1526.00 | 1507.78 | 1502.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 09:15:00 | 1511.78 | 1516.73 | 1510.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 1511.78 | 1516.73 | 1510.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1511.78 | 1516.73 | 1510.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 1511.78 | 1516.73 | 1510.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 1510.05 | 1515.39 | 1510.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:45:00 | 1507.93 | 1515.39 | 1510.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 1518.50 | 1516.02 | 1511.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:30:00 | 1510.85 | 1516.02 | 1511.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 1512.50 | 1516.28 | 1512.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 1510.20 | 1516.28 | 1512.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1507.20 | 1514.47 | 1512.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:45:00 | 1506.55 | 1514.47 | 1512.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 1502.50 | 1512.07 | 1511.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:00:00 | 1502.50 | 1512.07 | 1511.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 1510.00 | 1511.36 | 1511.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:15:00 | 1505.03 | 1511.36 | 1511.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 13:15:00 | 1498.93 | 1508.88 | 1510.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 1493.98 | 1505.90 | 1508.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 1485.58 | 1481.21 | 1492.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 15:00:00 | 1485.58 | 1481.21 | 1492.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1502.58 | 1484.31 | 1491.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 1473.30 | 1493.69 | 1494.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 1509.60 | 1496.87 | 1495.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 09:15:00 | 1509.60 | 1496.87 | 1495.81 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1474.50 | 1492.40 | 1493.87 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 13:15:00 | 1525.85 | 1498.47 | 1496.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 09:15:00 | 1581.50 | 1523.93 | 1509.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 1563.65 | 1571.38 | 1547.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 09:45:00 | 1562.33 | 1571.38 | 1547.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 1546.60 | 1558.15 | 1549.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 14:30:00 | 1544.23 | 1558.15 | 1549.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 1546.00 | 1555.72 | 1548.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 1558.50 | 1555.72 | 1548.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 11:15:00 | 1541.75 | 1568.05 | 1570.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 11:15:00 | 1541.75 | 1568.05 | 1570.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 13:15:00 | 1537.50 | 1557.77 | 1565.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 15:15:00 | 1544.50 | 1541.71 | 1550.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-14 09:15:00 | 1552.50 | 1541.71 | 1550.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1551.50 | 1543.66 | 1550.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:30:00 | 1558.93 | 1543.66 | 1550.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 1554.45 | 1545.82 | 1550.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:30:00 | 1555.48 | 1545.82 | 1550.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 12:15:00 | 1550.00 | 1547.17 | 1550.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 12:45:00 | 1550.00 | 1547.17 | 1550.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 1550.00 | 1547.73 | 1550.35 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 09:15:00 | 1564.35 | 1552.96 | 1552.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 12:15:00 | 1571.98 | 1565.22 | 1561.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 14:15:00 | 1566.23 | 1566.64 | 1563.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 14:15:00 | 1566.23 | 1566.64 | 1563.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 1566.23 | 1566.64 | 1563.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:30:00 | 1570.40 | 1566.64 | 1563.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 1560.68 | 1565.45 | 1562.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 1573.15 | 1565.45 | 1562.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 14:45:00 | 1570.95 | 1568.57 | 1565.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 1559.63 | 1566.24 | 1565.31 | SL hit (close<static) qty=1.00 sl=1560.63 alert=retest2 |

### Cycle 91 — SELL (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 12:15:00 | 1553.10 | 1563.35 | 1564.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 15:15:00 | 1550.58 | 1558.05 | 1561.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 1564.93 | 1559.43 | 1561.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 1564.93 | 1559.43 | 1561.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 1564.93 | 1559.43 | 1561.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:15:00 | 1566.80 | 1559.43 | 1561.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 1568.55 | 1561.25 | 1562.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:45:00 | 1568.70 | 1561.25 | 1562.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 11:15:00 | 1569.88 | 1562.98 | 1562.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 1576.03 | 1568.68 | 1566.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 10:15:00 | 1569.68 | 1581.41 | 1576.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 10:15:00 | 1569.68 | 1581.41 | 1576.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 1569.68 | 1581.41 | 1576.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:00:00 | 1569.68 | 1581.41 | 1576.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 1577.60 | 1580.65 | 1576.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:30:00 | 1570.00 | 1580.65 | 1576.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 1580.00 | 1580.52 | 1576.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:45:00 | 1575.68 | 1580.52 | 1576.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1572.18 | 1578.85 | 1576.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 1572.18 | 1578.85 | 1576.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1571.23 | 1577.33 | 1575.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 1571.23 | 1577.33 | 1575.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 1573.63 | 1576.59 | 1575.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:30:00 | 1588.45 | 1579.26 | 1576.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 13:15:00 | 1580.25 | 1580.77 | 1578.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 13:45:00 | 1580.03 | 1580.55 | 1578.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 14:45:00 | 1580.00 | 1580.24 | 1578.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 1577.85 | 1579.76 | 1578.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 1571.50 | 1579.76 | 1578.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1580.98 | 1580.01 | 1578.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 1578.08 | 1580.01 | 1578.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1575.33 | 1579.07 | 1578.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 11:00:00 | 1575.33 | 1579.07 | 1578.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-01 11:15:00 | 1570.50 | 1577.36 | 1577.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 11:15:00 | 1570.50 | 1577.36 | 1577.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-01 12:15:00 | 1563.00 | 1574.49 | 1576.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 1559.08 | 1551.19 | 1559.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 1559.08 | 1551.19 | 1559.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1559.08 | 1551.19 | 1559.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:45:00 | 1556.50 | 1551.19 | 1559.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 1559.53 | 1552.86 | 1559.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 11:15:00 | 1554.13 | 1552.86 | 1559.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 15:15:00 | 1553.13 | 1550.19 | 1555.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 10:00:00 | 1555.45 | 1551.71 | 1555.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 12:00:00 | 1555.93 | 1553.08 | 1555.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 1554.50 | 1553.37 | 1555.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:15:00 | 1557.03 | 1553.37 | 1555.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 1554.28 | 1553.55 | 1555.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:45:00 | 1557.25 | 1553.55 | 1555.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 1548.23 | 1552.48 | 1554.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 11:00:00 | 1547.20 | 1550.80 | 1553.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 11:30:00 | 1546.55 | 1549.90 | 1552.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 13:00:00 | 1545.60 | 1549.04 | 1551.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 13:45:00 | 1546.53 | 1548.59 | 1551.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 1549.93 | 1548.86 | 1551.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 15:00:00 | 1549.93 | 1548.86 | 1551.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 1547.53 | 1548.60 | 1550.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 1548.35 | 1548.60 | 1550.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1544.33 | 1547.74 | 1550.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 11:00:00 | 1537.95 | 1545.78 | 1549.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 1557.00 | 1543.03 | 1544.80 | SL hit (close>static) qty=1.00 sl=1555.05 alert=retest2 |

### Cycle 94 — BUY (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 12:15:00 | 1555.23 | 1547.70 | 1546.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 09:15:00 | 1584.50 | 1558.56 | 1552.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 09:15:00 | 1575.13 | 1575.19 | 1566.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 1575.13 | 1575.19 | 1566.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1575.13 | 1575.19 | 1566.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 1570.68 | 1575.19 | 1566.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 1564.98 | 1573.15 | 1565.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:00:00 | 1564.98 | 1573.15 | 1565.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 1567.75 | 1572.07 | 1566.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 12:15:00 | 1570.43 | 1572.07 | 1566.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 12:15:00 | 1563.93 | 1583.15 | 1585.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 1563.93 | 1583.15 | 1585.06 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 12:15:00 | 1583.48 | 1578.73 | 1578.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 13:15:00 | 1596.25 | 1582.24 | 1579.85 | Break + close above crossover candle high |

### Cycle 97 — SELL (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 09:15:00 | 1560.55 | 1579.12 | 1579.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 10:15:00 | 1557.60 | 1574.82 | 1577.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 14:15:00 | 1554.85 | 1553.64 | 1560.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 1573.48 | 1557.74 | 1561.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 1573.48 | 1557.74 | 1561.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 1573.48 | 1557.74 | 1561.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 1567.90 | 1559.78 | 1561.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 11:15:00 | 1565.55 | 1559.78 | 1561.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 12:00:00 | 1565.48 | 1560.92 | 1562.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 12:45:00 | 1563.78 | 1562.80 | 1563.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 13:45:00 | 1562.50 | 1562.75 | 1562.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 14:15:00 | 1568.98 | 1563.99 | 1563.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 14:15:00 | 1568.98 | 1563.99 | 1563.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 1574.03 | 1566.56 | 1564.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 12:15:00 | 1581.43 | 1582.49 | 1575.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 13:00:00 | 1581.43 | 1582.49 | 1575.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 1584.53 | 1582.90 | 1576.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 1594.78 | 1580.97 | 1576.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:30:00 | 1591.23 | 1592.97 | 1586.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 13:15:00 | 1572.23 | 1586.19 | 1585.29 | SL hit (close<static) qty=1.00 sl=1576.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 14:15:00 | 1559.58 | 1580.87 | 1582.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 1540.95 | 1562.22 | 1570.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1559.70 | 1548.13 | 1557.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1559.70 | 1548.13 | 1557.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1559.70 | 1548.13 | 1557.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 1559.70 | 1548.13 | 1557.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 1552.35 | 1548.97 | 1557.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:15:00 | 1544.85 | 1550.64 | 1556.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 1572.48 | 1555.35 | 1554.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 1572.48 | 1555.35 | 1554.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 1576.40 | 1559.56 | 1556.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 1578.98 | 1581.61 | 1570.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:00:00 | 1578.98 | 1581.61 | 1570.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1577.45 | 1582.98 | 1575.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:00:00 | 1577.45 | 1582.98 | 1575.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 1568.15 | 1580.02 | 1574.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:30:00 | 1569.48 | 1580.02 | 1574.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 1563.28 | 1576.67 | 1573.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:15:00 | 1563.45 | 1576.67 | 1573.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 14:15:00 | 1567.00 | 1570.54 | 1571.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 1533.45 | 1562.26 | 1567.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1517.53 | 1516.05 | 1524.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1517.53 | 1516.05 | 1524.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1517.53 | 1516.05 | 1524.86 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 1530.00 | 1526.70 | 1526.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 1564.98 | 1536.86 | 1532.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 1549.65 | 1552.07 | 1543.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 1549.65 | 1552.07 | 1543.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1543.50 | 1550.36 | 1543.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 1543.50 | 1550.36 | 1543.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 1546.28 | 1549.54 | 1544.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:45:00 | 1551.20 | 1545.86 | 1544.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 13:45:00 | 1550.38 | 1546.27 | 1544.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 14:15:00 | 1551.80 | 1546.27 | 1544.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 1550.55 | 1548.52 | 1546.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1551.40 | 1549.09 | 1546.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-27 13:15:00 | 1541.93 | 1546.64 | 1546.19 | SL hit (close<static) qty=1.00 sl=1543.50 alert=retest2 |

### Cycle 103 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 1541.95 | 1546.07 | 1546.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 1534.28 | 1540.39 | 1542.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 1540.40 | 1539.10 | 1541.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 1540.40 | 1539.10 | 1541.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 1545.00 | 1540.28 | 1541.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 1553.25 | 1540.28 | 1541.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 1555.60 | 1543.35 | 1543.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 1563.00 | 1552.67 | 1548.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 1586.58 | 1587.21 | 1576.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 15:00:00 | 1586.58 | 1587.21 | 1576.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 1595.00 | 1603.04 | 1594.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:00:00 | 1595.00 | 1603.04 | 1594.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 1606.78 | 1603.79 | 1595.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 12:15:00 | 1611.58 | 1603.79 | 1595.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 1615.63 | 1603.66 | 1598.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 1610.85 | 1626.04 | 1626.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 1610.85 | 1626.04 | 1626.07 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 11:15:00 | 1630.60 | 1626.82 | 1626.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 1634.03 | 1628.61 | 1627.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 15:15:00 | 1627.43 | 1628.38 | 1627.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 09:15:00 | 1626.15 | 1628.38 | 1627.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1629.85 | 1628.67 | 1627.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:15:00 | 1642.40 | 1630.14 | 1628.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 13:15:00 | 1643.05 | 1645.35 | 1640.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:45:00 | 1643.75 | 1647.00 | 1643.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 14:15:00 | 1634.88 | 1641.13 | 1641.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 14:15:00 | 1634.88 | 1641.13 | 1641.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 1627.05 | 1637.89 | 1639.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 12:15:00 | 1606.50 | 1605.03 | 1616.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 12:45:00 | 1604.55 | 1605.03 | 1616.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 1623.20 | 1609.45 | 1616.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 1623.20 | 1609.45 | 1616.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 1616.48 | 1610.86 | 1616.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 1623.85 | 1610.86 | 1616.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 1637.58 | 1619.55 | 1619.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 1642.50 | 1627.02 | 1623.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 13:15:00 | 1641.23 | 1644.90 | 1639.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 13:15:00 | 1641.23 | 1644.90 | 1639.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 1641.23 | 1644.90 | 1639.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:00:00 | 1641.23 | 1644.90 | 1639.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 1632.88 | 1642.50 | 1639.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:45:00 | 1632.43 | 1642.50 | 1639.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 1632.50 | 1640.50 | 1638.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 1622.50 | 1640.50 | 1638.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 1612.00 | 1634.80 | 1636.26 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 1642.40 | 1633.33 | 1633.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 1658.08 | 1643.20 | 1638.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 14:15:00 | 1679.30 | 1682.40 | 1669.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 15:00:00 | 1679.30 | 1682.40 | 1669.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1675.13 | 1680.56 | 1670.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:45:00 | 1672.70 | 1680.56 | 1670.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 1671.55 | 1678.76 | 1670.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:00:00 | 1671.55 | 1678.76 | 1670.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 1673.75 | 1677.76 | 1671.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:30:00 | 1673.30 | 1677.76 | 1671.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 1667.50 | 1675.71 | 1670.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:00:00 | 1667.50 | 1675.71 | 1670.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 1666.03 | 1673.77 | 1670.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:30:00 | 1666.58 | 1673.77 | 1670.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 1669.33 | 1672.88 | 1670.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:30:00 | 1667.20 | 1672.88 | 1670.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 1667.50 | 1671.81 | 1670.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 1658.15 | 1671.81 | 1670.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 1652.93 | 1668.03 | 1668.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 1604.55 | 1635.89 | 1649.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 14:15:00 | 1593.50 | 1578.92 | 1595.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 15:00:00 | 1593.50 | 1578.92 | 1595.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1595.95 | 1582.32 | 1595.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1608.60 | 1582.32 | 1595.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1615.95 | 1589.05 | 1597.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 1615.95 | 1589.05 | 1597.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 13:15:00 | 1589.83 | 1595.15 | 1598.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:15:00 | 1587.03 | 1595.15 | 1598.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 15:00:00 | 1586.03 | 1593.32 | 1597.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 10:00:00 | 1585.80 | 1590.86 | 1595.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 09:45:00 | 1584.03 | 1568.92 | 1570.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 11:15:00 | 1580.10 | 1572.40 | 1571.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 11:15:00 | 1580.10 | 1572.40 | 1571.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 12:15:00 | 1592.80 | 1576.48 | 1573.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 10:15:00 | 1579.28 | 1581.16 | 1577.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 10:15:00 | 1579.28 | 1581.16 | 1577.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 1579.28 | 1581.16 | 1577.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:15:00 | 1571.50 | 1581.16 | 1577.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 1573.00 | 1579.53 | 1577.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:30:00 | 1573.00 | 1579.53 | 1577.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 1581.55 | 1579.93 | 1577.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:15:00 | 1583.95 | 1580.37 | 1578.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 10:15:00 | 1584.50 | 1583.45 | 1580.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 1615.98 | 1582.51 | 1581.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 10:45:00 | 1582.73 | 1591.76 | 1591.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 11:15:00 | 1572.65 | 1587.94 | 1589.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 1572.65 | 1587.94 | 1589.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 1568.80 | 1577.85 | 1582.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 1571.93 | 1571.47 | 1577.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 09:45:00 | 1570.35 | 1571.47 | 1577.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 1586.25 | 1564.02 | 1569.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:30:00 | 1583.33 | 1564.02 | 1569.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 1556.25 | 1562.47 | 1568.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:30:00 | 1552.83 | 1558.82 | 1566.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 13:45:00 | 1554.75 | 1555.92 | 1563.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 1589.90 | 1564.76 | 1565.81 | SL hit (close>static) qty=1.00 sl=1588.60 alert=retest2 |

### Cycle 114 — BUY (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 10:15:00 | 1589.03 | 1569.61 | 1567.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 11:15:00 | 1596.28 | 1586.57 | 1579.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-28 13:15:00 | 1586.00 | 1588.91 | 1581.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-28 14:00:00 | 1586.00 | 1588.91 | 1581.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 1583.00 | 1587.73 | 1581.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 15:00:00 | 1583.00 | 1587.73 | 1581.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 1586.10 | 1587.40 | 1582.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:15:00 | 1560.00 | 1587.40 | 1582.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 1555.38 | 1581.00 | 1579.78 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 1547.55 | 1574.31 | 1576.85 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 14:15:00 | 1578.43 | 1571.74 | 1571.07 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 1553.45 | 1569.34 | 1571.27 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 1580.00 | 1564.61 | 1563.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 13:15:00 | 1592.23 | 1570.13 | 1565.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1576.23 | 1576.73 | 1570.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 1576.23 | 1576.73 | 1570.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1576.23 | 1576.73 | 1570.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1576.23 | 1576.73 | 1570.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1573.98 | 1576.62 | 1571.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:45:00 | 1569.03 | 1576.62 | 1571.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 1574.03 | 1576.10 | 1571.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 1574.03 | 1576.10 | 1571.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 1574.63 | 1575.81 | 1571.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:30:00 | 1570.38 | 1575.81 | 1571.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1579.98 | 1576.82 | 1573.37 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 14:15:00 | 1561.98 | 1569.97 | 1570.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1550.05 | 1565.22 | 1568.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1509.18 | 1505.54 | 1521.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 1509.18 | 1505.54 | 1521.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1511.38 | 1509.73 | 1518.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:30:00 | 1508.88 | 1512.72 | 1517.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 1508.33 | 1513.10 | 1516.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 15:00:00 | 1508.60 | 1512.20 | 1515.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 09:15:00 | 1494.00 | 1511.76 | 1515.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1509.00 | 1508.94 | 1513.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:45:00 | 1508.78 | 1508.94 | 1513.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 1505.50 | 1506.93 | 1511.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:45:00 | 1509.90 | 1506.93 | 1511.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 1506.30 | 1504.82 | 1508.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 12:45:00 | 1486.95 | 1498.04 | 1504.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 09:45:00 | 1484.78 | 1480.77 | 1486.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 11:15:00 | 1514.00 | 1490.05 | 1490.16 | SL hit (close>static) qty=1.00 sl=1510.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 12:15:00 | 1504.18 | 1492.88 | 1491.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 10:15:00 | 1520.00 | 1508.29 | 1502.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 11:15:00 | 1508.63 | 1517.31 | 1511.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 11:15:00 | 1508.63 | 1517.31 | 1511.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1508.63 | 1517.31 | 1511.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:00:00 | 1508.63 | 1517.31 | 1511.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 1501.88 | 1514.22 | 1511.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 1501.88 | 1514.22 | 1511.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 1524.98 | 1515.76 | 1512.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:45:00 | 1526.48 | 1519.24 | 1514.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:30:00 | 1526.50 | 1521.52 | 1516.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:45:00 | 1527.35 | 1523.17 | 1517.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 1560.50 | 1580.26 | 1582.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 1560.50 | 1580.26 | 1582.17 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 1605.10 | 1584.81 | 1582.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 1611.65 | 1601.30 | 1593.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 11:15:00 | 1601.83 | 1602.25 | 1595.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 11:45:00 | 1602.05 | 1602.25 | 1595.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 1594.30 | 1600.15 | 1595.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:00:00 | 1594.30 | 1600.15 | 1595.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 1593.30 | 1598.78 | 1595.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:45:00 | 1589.40 | 1598.78 | 1595.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1582.75 | 1596.17 | 1594.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 1582.75 | 1596.17 | 1594.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1585.08 | 1593.96 | 1593.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 11:15:00 | 1590.75 | 1593.96 | 1593.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 1593.50 | 1593.86 | 1593.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 1593.50 | 1593.86 | 1593.91 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 12:15:00 | 1595.50 | 1594.19 | 1594.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 13:15:00 | 1596.95 | 1594.74 | 1594.32 | Break + close above crossover candle high |

### Cycle 125 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 1579.53 | 1592.33 | 1593.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 10:15:00 | 1574.10 | 1588.68 | 1591.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 1590.20 | 1586.86 | 1589.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 13:15:00 | 1590.20 | 1586.86 | 1589.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 1590.20 | 1586.86 | 1589.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 1590.20 | 1586.86 | 1589.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1592.50 | 1587.98 | 1590.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 1592.50 | 1587.98 | 1590.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 1592.00 | 1588.79 | 1590.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 1592.63 | 1588.79 | 1590.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 10:15:00 | 1597.53 | 1591.54 | 1591.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 11:15:00 | 1602.63 | 1593.76 | 1592.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 15:15:00 | 1596.50 | 1596.95 | 1594.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 09:15:00 | 1581.50 | 1596.95 | 1594.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 127 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 1570.73 | 1591.71 | 1592.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 10:15:00 | 1556.63 | 1584.69 | 1589.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 11:15:00 | 1507.45 | 1504.61 | 1520.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 12:00:00 | 1507.45 | 1504.61 | 1520.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 1477.63 | 1484.38 | 1490.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 11:15:00 | 1475.73 | 1482.79 | 1489.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:00:00 | 1473.33 | 1473.44 | 1480.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 14:15:00 | 1463.83 | 1450.60 | 1449.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 14:15:00 | 1463.83 | 1450.60 | 1449.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 11:15:00 | 1468.48 | 1457.90 | 1453.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 1452.00 | 1460.02 | 1456.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 1452.00 | 1460.02 | 1456.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1452.00 | 1460.02 | 1456.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 1452.18 | 1460.02 | 1456.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1442.50 | 1456.52 | 1455.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1442.50 | 1456.52 | 1455.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 1437.50 | 1452.71 | 1453.57 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 1462.50 | 1453.86 | 1453.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 13:15:00 | 1464.70 | 1457.21 | 1455.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 09:15:00 | 1455.80 | 1458.83 | 1456.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 1455.80 | 1458.83 | 1456.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1455.80 | 1458.83 | 1456.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:45:00 | 1456.28 | 1458.83 | 1456.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 1452.35 | 1457.54 | 1456.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:00:00 | 1452.35 | 1457.54 | 1456.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 1443.68 | 1454.77 | 1454.99 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 1466.85 | 1456.62 | 1455.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 10:15:00 | 1478.85 | 1461.07 | 1457.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 1462.38 | 1470.81 | 1465.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 1462.38 | 1470.81 | 1465.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1462.38 | 1470.81 | 1465.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 1462.38 | 1470.81 | 1465.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 1471.73 | 1470.99 | 1465.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:15:00 | 1474.35 | 1470.99 | 1465.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 13:15:00 | 1456.80 | 1466.13 | 1464.79 | SL hit (close<static) qty=1.00 sl=1461.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 14:15:00 | 1450.85 | 1463.08 | 1463.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1428.00 | 1453.97 | 1459.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 10:15:00 | 1415.00 | 1408.00 | 1418.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 10:15:00 | 1415.00 | 1408.00 | 1418.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 1415.00 | 1408.00 | 1418.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 1415.50 | 1408.00 | 1418.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1400.88 | 1403.92 | 1411.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:45:00 | 1395.05 | 1402.03 | 1410.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 15:15:00 | 1407.95 | 1404.89 | 1404.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 15:15:00 | 1407.95 | 1404.89 | 1404.65 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 1398.35 | 1403.58 | 1404.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 14:15:00 | 1392.20 | 1400.38 | 1402.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1379.58 | 1377.09 | 1384.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 1379.58 | 1377.09 | 1384.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 136 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 1488.15 | 1398.97 | 1392.88 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 11:15:00 | 1425.50 | 1435.77 | 1436.87 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 1443.00 | 1438.36 | 1437.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 12:15:00 | 1483.03 | 1448.10 | 1442.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 12:15:00 | 1463.03 | 1467.12 | 1457.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 12:15:00 | 1463.03 | 1467.12 | 1457.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 1463.03 | 1467.12 | 1457.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:00:00 | 1463.03 | 1467.12 | 1457.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1460.00 | 1470.33 | 1462.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 1460.00 | 1470.33 | 1462.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1465.28 | 1469.32 | 1462.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:45:00 | 1461.18 | 1469.32 | 1462.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 1470.63 | 1468.68 | 1463.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 14:00:00 | 1471.68 | 1469.28 | 1464.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 14:15:00 | 1463.03 | 1468.03 | 1464.16 | SL hit (close<static) qty=1.00 sl=1463.18 alert=retest2 |

### Cycle 139 — SELL (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 13:15:00 | 1453.08 | 1462.60 | 1462.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 14:15:00 | 1445.20 | 1459.12 | 1461.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 14:15:00 | 1448.50 | 1448.45 | 1453.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-06 15:00:00 | 1448.50 | 1448.45 | 1453.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1434.85 | 1446.34 | 1451.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 13:45:00 | 1428.10 | 1439.37 | 1446.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:30:00 | 1431.50 | 1434.50 | 1441.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 14:00:00 | 1431.43 | 1433.29 | 1439.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 14:45:00 | 1431.45 | 1433.35 | 1439.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1418.73 | 1418.37 | 1423.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:15:00 | 1404.00 | 1417.31 | 1421.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 13:15:00 | 1405.05 | 1391.95 | 1390.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 1405.05 | 1391.95 | 1390.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 1412.95 | 1396.15 | 1392.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1393.13 | 1398.00 | 1393.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 1393.13 | 1398.00 | 1393.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1393.13 | 1398.00 | 1393.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 1393.13 | 1398.00 | 1393.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1398.85 | 1398.17 | 1394.39 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 1376.05 | 1391.31 | 1392.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 11:15:00 | 1370.13 | 1387.08 | 1390.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 15:15:00 | 1382.00 | 1381.61 | 1386.42 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 09:15:00 | 1365.23 | 1381.61 | 1386.42 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 1337.33 | 1328.88 | 1335.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-03 14:15:00 | 1337.33 | 1328.88 | 1335.16 | SL hit (close>ema400) qty=1.00 sl=1335.16 alert=retest1 |

### Cycle 142 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 1348.63 | 1339.07 | 1337.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 1356.70 | 1346.28 | 1341.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 14:15:00 | 1346.55 | 1352.09 | 1346.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 14:15:00 | 1346.55 | 1352.09 | 1346.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 1346.55 | 1352.09 | 1346.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-05 15:00:00 | 1346.55 | 1352.09 | 1346.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 15:15:00 | 1346.50 | 1350.97 | 1346.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 09:15:00 | 1362.23 | 1350.97 | 1346.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 10:15:00 | 1358.58 | 1367.65 | 1367.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 1358.58 | 1367.65 | 1367.87 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 10:15:00 | 1377.20 | 1367.56 | 1366.97 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 1360.50 | 1368.02 | 1368.53 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 1373.03 | 1369.03 | 1368.94 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 12:15:00 | 1364.98 | 1368.17 | 1368.58 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 14:15:00 | 1374.00 | 1369.13 | 1368.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 15:15:00 | 1376.43 | 1370.59 | 1369.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 12:15:00 | 1368.95 | 1371.85 | 1370.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 12:15:00 | 1368.95 | 1371.85 | 1370.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 1368.95 | 1371.85 | 1370.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:45:00 | 1365.78 | 1371.85 | 1370.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 1365.63 | 1370.61 | 1370.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:00:00 | 1365.63 | 1370.61 | 1370.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 14:15:00 | 1363.05 | 1369.10 | 1369.58 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 1374.40 | 1370.14 | 1369.97 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 14:15:00 | 1365.18 | 1369.13 | 1369.66 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 11:15:00 | 1382.95 | 1371.50 | 1370.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 1408.00 | 1383.29 | 1376.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 10:15:00 | 1402.78 | 1405.09 | 1394.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 10:30:00 | 1402.83 | 1405.09 | 1394.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1429.53 | 1413.83 | 1407.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 11:15:00 | 1434.50 | 1416.60 | 1409.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 15:15:00 | 1434.70 | 1419.64 | 1415.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:30:00 | 1434.15 | 1426.74 | 1423.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 13:15:00 | 1416.78 | 1421.30 | 1421.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 13:15:00 | 1416.78 | 1421.30 | 1421.60 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 1425.73 | 1421.06 | 1420.73 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 10:15:00 | 1418.15 | 1420.27 | 1420.47 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 1424.50 | 1421.11 | 1420.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 1430.05 | 1424.23 | 1422.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1402.10 | 1431.16 | 1428.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 1402.10 | 1431.16 | 1428.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1402.10 | 1431.16 | 1428.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 10:15:00 | 1405.08 | 1431.16 | 1428.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 1409.00 | 1426.73 | 1427.05 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 09:15:00 | 1441.03 | 1426.31 | 1425.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 10:15:00 | 1454.13 | 1431.87 | 1428.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 09:15:00 | 1463.33 | 1467.90 | 1457.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 10:00:00 | 1463.33 | 1467.90 | 1457.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 1507.70 | 1513.60 | 1506.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 11:00:00 | 1507.70 | 1513.60 | 1506.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 1502.20 | 1511.32 | 1505.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 12:00:00 | 1502.20 | 1511.32 | 1505.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 1507.50 | 1510.56 | 1505.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 13:15:00 | 1516.90 | 1510.56 | 1505.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 1503.80 | 1524.56 | 1524.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1503.80 | 1524.56 | 1524.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 11:15:00 | 1499.95 | 1519.64 | 1522.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1511.00 | 1509.90 | 1515.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 1511.00 | 1509.90 | 1515.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1511.00 | 1509.90 | 1515.61 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 1529.90 | 1517.17 | 1516.32 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 14:15:00 | 1505.15 | 1515.22 | 1516.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 15:15:00 | 1501.30 | 1512.44 | 1514.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 13:15:00 | 1513.45 | 1508.33 | 1511.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 13:15:00 | 1513.45 | 1508.33 | 1511.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 1513.45 | 1508.33 | 1511.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 1513.45 | 1508.33 | 1511.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 1519.15 | 1510.49 | 1512.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 15:00:00 | 1519.15 | 1510.49 | 1512.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 1515.00 | 1511.39 | 1512.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 1503.70 | 1511.39 | 1512.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:30:00 | 1507.50 | 1508.52 | 1510.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 1526.20 | 1509.45 | 1509.77 | SL hit (close>static) qty=1.00 sl=1520.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 1519.10 | 1511.38 | 1510.62 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 1493.50 | 1508.54 | 1510.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 1492.45 | 1505.32 | 1508.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 1494.50 | 1491.96 | 1497.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 09:15:00 | 1482.50 | 1491.96 | 1497.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1477.95 | 1489.16 | 1496.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 10:15:00 | 1475.40 | 1489.16 | 1496.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 10:45:00 | 1474.50 | 1485.94 | 1493.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 1505.50 | 1485.46 | 1489.43 | SL hit (close>static) qty=1.00 sl=1497.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1526.40 | 1496.13 | 1493.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1533.70 | 1512.66 | 1502.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 11:15:00 | 1549.00 | 1549.33 | 1536.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:45:00 | 1550.50 | 1549.33 | 1536.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1545.45 | 1555.33 | 1551.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:45:00 | 1539.95 | 1555.33 | 1551.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 1549.75 | 1554.22 | 1551.36 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 15:15:00 | 1537.95 | 1548.65 | 1549.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 10:15:00 | 1536.75 | 1545.56 | 1547.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 1498.00 | 1497.04 | 1508.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 11:00:00 | 1498.00 | 1497.04 | 1508.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1501.80 | 1497.99 | 1503.94 | EMA400 retest candle locked (from downside) |

### Cycle 166 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 1519.25 | 1507.99 | 1506.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1533.45 | 1513.08 | 1509.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 1520.30 | 1520.55 | 1515.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 1515.00 | 1520.55 | 1515.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1509.85 | 1518.41 | 1515.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1509.85 | 1518.41 | 1515.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1514.20 | 1517.57 | 1515.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:45:00 | 1515.60 | 1517.48 | 1515.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 1516.25 | 1514.46 | 1514.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1507.15 | 1513.28 | 1513.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 1507.15 | 1513.28 | 1513.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 11:15:00 | 1503.20 | 1509.84 | 1512.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 1505.80 | 1504.86 | 1508.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1505.80 | 1504.86 | 1508.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1505.80 | 1504.86 | 1508.26 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 1513.80 | 1510.20 | 1509.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 15:15:00 | 1521.95 | 1512.55 | 1511.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 12:15:00 | 1543.30 | 1545.87 | 1535.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 13:00:00 | 1543.30 | 1545.87 | 1535.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1538.55 | 1543.12 | 1536.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:30:00 | 1533.90 | 1543.12 | 1536.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1543.55 | 1544.52 | 1539.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 1543.55 | 1544.52 | 1539.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 1542.10 | 1545.19 | 1542.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:45:00 | 1545.45 | 1545.19 | 1542.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 1544.75 | 1545.10 | 1542.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:30:00 | 1549.40 | 1546.14 | 1542.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 15:00:00 | 1546.75 | 1546.70 | 1543.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 1528.55 | 1545.22 | 1544.62 | SL hit (close<static) qty=1.00 sl=1540.10 alert=retest2 |

### Cycle 169 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1529.35 | 1542.04 | 1543.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 15:15:00 | 1523.00 | 1527.43 | 1533.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 10:15:00 | 1524.90 | 1524.56 | 1530.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 1524.90 | 1524.56 | 1530.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 1528.90 | 1526.17 | 1529.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:45:00 | 1528.20 | 1526.17 | 1529.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 1532.50 | 1527.44 | 1529.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 1539.70 | 1527.44 | 1529.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 1546.55 | 1531.26 | 1531.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 1546.55 | 1531.26 | 1531.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 1540.50 | 1533.11 | 1532.16 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 1531.75 | 1533.16 | 1533.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1524.95 | 1530.60 | 1531.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 1509.45 | 1506.78 | 1515.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 1519.35 | 1506.78 | 1515.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1512.50 | 1507.92 | 1514.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 14:45:00 | 1500.55 | 1504.66 | 1510.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:00:00 | 1500.45 | 1502.67 | 1508.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 1500.45 | 1502.78 | 1507.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 1498.55 | 1505.77 | 1507.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1477.00 | 1482.61 | 1489.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1473.05 | 1478.76 | 1484.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 12:45:00 | 1473.90 | 1475.67 | 1480.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1504.60 | 1481.40 | 1481.77 | SL hit (close>static) qty=1.00 sl=1490.80 alert=retest2 |

### Cycle 172 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1503.45 | 1485.81 | 1483.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 1512.50 | 1491.15 | 1486.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 1501.75 | 1505.95 | 1498.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 13:15:00 | 1501.75 | 1505.95 | 1498.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 1501.75 | 1505.95 | 1498.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:45:00 | 1494.25 | 1505.95 | 1498.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 1505.25 | 1505.81 | 1499.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:45:00 | 1500.55 | 1505.81 | 1499.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 1499.45 | 1504.54 | 1499.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:30:00 | 1509.10 | 1504.77 | 1499.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 1506.25 | 1504.68 | 1501.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 1518.95 | 1536.63 | 1538.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 1518.95 | 1536.63 | 1538.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 1516.10 | 1527.50 | 1530.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 10:15:00 | 1533.75 | 1527.05 | 1529.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 10:15:00 | 1533.75 | 1527.05 | 1529.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1533.75 | 1527.05 | 1529.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 1533.75 | 1527.05 | 1529.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1532.80 | 1528.20 | 1529.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:45:00 | 1540.50 | 1528.20 | 1529.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 1536.05 | 1530.12 | 1530.06 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 1527.05 | 1529.76 | 1529.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 1513.00 | 1526.41 | 1528.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 1491.35 | 1488.09 | 1498.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 10:45:00 | 1491.65 | 1488.09 | 1498.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 1491.00 | 1489.92 | 1496.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 1489.65 | 1492.92 | 1497.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:30:00 | 1489.90 | 1491.29 | 1495.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:15:00 | 1489.05 | 1491.29 | 1495.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 13:00:00 | 1490.10 | 1491.05 | 1494.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1495.05 | 1491.88 | 1494.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 1495.00 | 1491.88 | 1494.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1493.90 | 1492.29 | 1494.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 1497.50 | 1492.29 | 1494.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1502.00 | 1494.23 | 1495.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 1502.00 | 1494.23 | 1495.11 | SL hit (close>static) qty=1.00 sl=1497.85 alert=retest2 |

### Cycle 176 — BUY (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 09:15:00 | 1465.10 | 1438.99 | 1437.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 11:15:00 | 1482.95 | 1453.62 | 1445.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 13:15:00 | 1518.65 | 1519.08 | 1505.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 13:30:00 | 1517.65 | 1519.08 | 1505.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1540.65 | 1546.52 | 1539.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 1540.65 | 1546.52 | 1539.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 1540.45 | 1545.30 | 1540.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:45:00 | 1540.50 | 1545.30 | 1540.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 1535.50 | 1543.34 | 1539.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:30:00 | 1536.25 | 1543.34 | 1539.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 1529.60 | 1540.59 | 1538.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 1529.60 | 1540.59 | 1538.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 15:15:00 | 1529.50 | 1536.41 | 1537.11 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 13:15:00 | 1545.00 | 1536.92 | 1536.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 1547.75 | 1541.51 | 1539.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 15:15:00 | 1546.00 | 1546.80 | 1543.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:15:00 | 1544.55 | 1546.80 | 1543.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1547.35 | 1546.91 | 1544.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 1545.00 | 1546.91 | 1544.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1542.20 | 1546.37 | 1545.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 1542.20 | 1546.37 | 1545.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 1542.50 | 1545.60 | 1544.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 1548.95 | 1545.60 | 1544.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:30:00 | 1544.95 | 1545.24 | 1544.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 1541.15 | 1544.42 | 1544.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 1541.15 | 1544.42 | 1544.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 13:15:00 | 1537.85 | 1542.45 | 1543.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 11:15:00 | 1539.95 | 1539.82 | 1541.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 11:15:00 | 1539.95 | 1539.82 | 1541.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1539.95 | 1539.82 | 1541.63 | EMA400 retest candle locked (from downside) |

### Cycle 180 — BUY (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 14:15:00 | 1544.00 | 1542.93 | 1542.80 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 1540.00 | 1542.35 | 1542.55 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 1546.50 | 1543.18 | 1542.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 1564.35 | 1547.41 | 1544.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 14:15:00 | 1553.50 | 1553.79 | 1549.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 15:00:00 | 1553.50 | 1553.79 | 1549.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1552.00 | 1553.43 | 1549.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1541.00 | 1553.43 | 1549.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1543.75 | 1551.49 | 1549.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1545.50 | 1551.49 | 1549.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1545.40 | 1550.28 | 1548.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 11:30:00 | 1549.50 | 1550.41 | 1548.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 15:15:00 | 1551.25 | 1553.75 | 1551.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 11:30:00 | 1548.45 | 1549.99 | 1549.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 12:15:00 | 1540.15 | 1548.03 | 1549.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 1540.15 | 1548.03 | 1549.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 1534.50 | 1545.32 | 1547.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1549.50 | 1531.64 | 1535.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1549.50 | 1531.64 | 1535.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1549.50 | 1531.64 | 1535.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1549.50 | 1531.64 | 1535.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1565.00 | 1538.31 | 1538.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 1567.40 | 1538.31 | 1538.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1565.25 | 1543.70 | 1540.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1568.05 | 1556.67 | 1549.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 1561.60 | 1564.71 | 1559.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 1561.60 | 1564.71 | 1559.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 1553.55 | 1562.48 | 1558.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 1553.55 | 1562.48 | 1558.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1555.70 | 1561.12 | 1558.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 15:15:00 | 1556.35 | 1561.12 | 1558.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 1557.55 | 1561.75 | 1559.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 1551.85 | 1558.29 | 1559.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 1551.85 | 1558.29 | 1559.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 13:15:00 | 1548.10 | 1554.92 | 1557.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 13:15:00 | 1556.75 | 1552.54 | 1554.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 13:15:00 | 1556.75 | 1552.54 | 1554.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 1556.75 | 1552.54 | 1554.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 1556.75 | 1552.54 | 1554.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 1550.95 | 1552.22 | 1554.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:30:00 | 1560.25 | 1552.22 | 1554.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1563.55 | 1554.05 | 1554.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 1563.55 | 1554.05 | 1554.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 1558.35 | 1554.91 | 1554.91 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 1550.95 | 1554.48 | 1554.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 10:15:00 | 1537.50 | 1550.34 | 1552.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 12:15:00 | 1540.95 | 1537.79 | 1542.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 1540.95 | 1537.79 | 1542.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1541.20 | 1538.75 | 1542.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:30:00 | 1541.35 | 1538.75 | 1542.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1545.00 | 1540.36 | 1542.51 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 1544.25 | 1543.77 | 1543.75 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 14:15:00 | 1539.40 | 1542.93 | 1543.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 15:15:00 | 1534.65 | 1539.32 | 1541.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 1541.35 | 1539.72 | 1541.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1541.35 | 1539.72 | 1541.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1541.35 | 1539.72 | 1541.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 1541.35 | 1539.72 | 1541.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1543.30 | 1540.44 | 1541.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 1543.30 | 1540.44 | 1541.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1543.40 | 1541.03 | 1541.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:30:00 | 1545.00 | 1541.03 | 1541.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 1534.85 | 1540.10 | 1541.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 1531.60 | 1540.10 | 1541.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:00:00 | 1528.50 | 1537.35 | 1539.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:45:00 | 1531.85 | 1534.65 | 1537.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:15:00 | 1531.90 | 1535.51 | 1537.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1533.35 | 1531.25 | 1533.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:30:00 | 1526.75 | 1529.99 | 1533.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 10:00:00 | 1527.05 | 1526.82 | 1529.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:30:00 | 1525.80 | 1528.55 | 1529.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 1455.02 | 1462.80 | 1470.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 1452.08 | 1462.80 | 1470.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 1455.26 | 1462.80 | 1470.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 1455.31 | 1462.80 | 1470.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 1450.41 | 1462.80 | 1470.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 1450.70 | 1462.80 | 1470.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 1449.51 | 1462.80 | 1470.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 1468.50 | 1462.12 | 1467.68 | SL hit (close>ema200) qty=0.50 sl=1462.12 alert=retest2 |

### Cycle 190 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 1475.70 | 1470.65 | 1470.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 1483.40 | 1474.73 | 1472.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 1486.00 | 1486.01 | 1481.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 1479.70 | 1486.01 | 1481.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1480.00 | 1484.81 | 1481.50 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 1470.60 | 1478.98 | 1479.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1465.40 | 1474.83 | 1477.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 11:15:00 | 1487.70 | 1477.35 | 1478.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 11:15:00 | 1487.70 | 1477.35 | 1478.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1487.70 | 1477.35 | 1478.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:00:00 | 1487.70 | 1477.35 | 1478.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 12:15:00 | 1490.20 | 1479.92 | 1479.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 14:15:00 | 1494.30 | 1484.62 | 1481.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 12:15:00 | 1505.30 | 1507.73 | 1499.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 13:00:00 | 1505.30 | 1507.73 | 1499.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1513.20 | 1509.00 | 1502.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:00:00 | 1514.90 | 1510.33 | 1504.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:00:00 | 1515.10 | 1512.12 | 1506.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 1517.50 | 1512.07 | 1507.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 1494.80 | 1505.98 | 1505.85 | SL hit (close<static) qty=1.00 sl=1497.70 alert=retest2 |

### Cycle 193 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 1488.50 | 1502.49 | 1504.27 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 1512.50 | 1502.24 | 1501.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 1520.20 | 1506.93 | 1504.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 11:15:00 | 1533.00 | 1534.14 | 1526.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 12:00:00 | 1533.00 | 1534.14 | 1526.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 1526.70 | 1532.70 | 1527.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 1526.70 | 1532.70 | 1527.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 1528.90 | 1531.94 | 1527.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 1532.80 | 1532.55 | 1528.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1515.40 | 1528.71 | 1527.20 | SL hit (close<static) qty=1.00 sl=1523.40 alert=retest2 |

### Cycle 195 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 1516.00 | 1525.16 | 1526.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 1513.00 | 1522.73 | 1525.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 1511.70 | 1511.69 | 1517.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 09:30:00 | 1511.80 | 1511.69 | 1517.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1517.00 | 1512.75 | 1517.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 1517.00 | 1512.75 | 1517.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 1506.80 | 1511.56 | 1516.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:30:00 | 1520.40 | 1511.56 | 1516.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1507.00 | 1498.55 | 1501.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 1506.00 | 1498.55 | 1501.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 1507.00 | 1500.24 | 1502.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 1506.30 | 1500.24 | 1502.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1500.10 | 1500.17 | 1501.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:15:00 | 1502.30 | 1500.17 | 1501.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1496.70 | 1499.48 | 1501.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 14:15:00 | 1492.80 | 1498.89 | 1500.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1462.10 | 1454.84 | 1454.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 1462.10 | 1454.84 | 1454.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 1470.70 | 1462.13 | 1459.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 1481.00 | 1483.04 | 1476.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 1481.00 | 1483.04 | 1476.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1471.00 | 1480.14 | 1476.32 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 1462.80 | 1472.82 | 1473.59 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 1479.80 | 1472.94 | 1472.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 09:15:00 | 1490.60 | 1478.75 | 1475.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 10:15:00 | 1481.60 | 1487.40 | 1483.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 10:15:00 | 1481.60 | 1487.40 | 1483.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1481.60 | 1487.40 | 1483.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 1482.70 | 1487.40 | 1483.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1481.60 | 1486.24 | 1482.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:45:00 | 1482.10 | 1485.47 | 1482.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 1487.80 | 1482.01 | 1481.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 10:00:00 | 1486.60 | 1482.93 | 1482.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 1473.20 | 1485.80 | 1484.88 | SL hit (close<static) qty=1.00 sl=1473.60 alert=retest2 |

### Cycle 199 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1474.60 | 1483.56 | 1483.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 10:15:00 | 1466.80 | 1475.07 | 1479.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 15:15:00 | 1472.50 | 1467.15 | 1472.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 15:15:00 | 1472.50 | 1467.15 | 1472.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1472.50 | 1467.15 | 1472.82 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1481.90 | 1473.05 | 1472.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 1489.70 | 1476.38 | 1473.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 1477.20 | 1480.17 | 1477.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 11:15:00 | 1477.20 | 1480.17 | 1477.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1477.20 | 1480.17 | 1477.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 1477.90 | 1480.17 | 1477.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1472.20 | 1478.58 | 1476.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 1472.20 | 1478.58 | 1476.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1470.00 | 1476.86 | 1476.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 1470.00 | 1476.86 | 1476.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 1470.80 | 1475.19 | 1475.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 12:15:00 | 1461.80 | 1467.49 | 1470.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 1468.70 | 1466.84 | 1469.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 14:15:00 | 1468.70 | 1466.84 | 1469.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1468.70 | 1466.84 | 1469.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 1468.70 | 1466.84 | 1469.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1468.60 | 1467.19 | 1469.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 1460.40 | 1467.19 | 1469.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:45:00 | 1461.30 | 1466.13 | 1468.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 11:15:00 | 1462.00 | 1465.76 | 1468.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:00:00 | 1463.30 | 1464.57 | 1467.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1470.00 | 1465.28 | 1467.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1470.00 | 1465.28 | 1467.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1470.60 | 1466.34 | 1467.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1470.60 | 1466.34 | 1467.49 | SL hit (close>static) qty=1.00 sl=1470.00 alert=retest2 |

### Cycle 202 — BUY (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 09:15:00 | 1479.90 | 1469.05 | 1468.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 12:15:00 | 1485.90 | 1475.73 | 1472.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 1470.00 | 1478.72 | 1476.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1470.00 | 1478.72 | 1476.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1470.00 | 1478.72 | 1476.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 1469.40 | 1478.72 | 1476.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1471.20 | 1477.22 | 1476.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 1471.10 | 1477.22 | 1476.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1469.90 | 1475.76 | 1475.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 1468.70 | 1475.76 | 1475.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1476.60 | 1476.32 | 1475.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:45:00 | 1473.10 | 1476.32 | 1475.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1475.00 | 1476.05 | 1475.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 1472.30 | 1476.05 | 1475.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 15:15:00 | 1473.20 | 1475.48 | 1475.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 1468.90 | 1474.17 | 1475.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1458.00 | 1457.17 | 1463.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 1458.00 | 1457.17 | 1463.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 1458.10 | 1457.55 | 1462.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 1459.00 | 1457.55 | 1462.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1464.90 | 1459.63 | 1462.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 1464.90 | 1459.63 | 1462.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1456.90 | 1459.09 | 1461.88 | EMA400 retest candle locked (from downside) |

### Cycle 204 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 1464.40 | 1461.21 | 1461.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1471.50 | 1465.01 | 1463.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 1475.50 | 1480.06 | 1476.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 12:15:00 | 1475.50 | 1480.06 | 1476.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 1475.50 | 1480.06 | 1476.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 1475.50 | 1480.06 | 1476.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 1470.90 | 1478.22 | 1475.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 1470.90 | 1478.22 | 1475.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1475.60 | 1477.70 | 1475.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:45:00 | 1470.90 | 1477.70 | 1475.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1469.40 | 1476.04 | 1475.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1464.80 | 1476.04 | 1475.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 1461.60 | 1473.15 | 1473.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 1454.60 | 1469.44 | 1472.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1454.00 | 1452.68 | 1459.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 1454.00 | 1452.68 | 1459.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1451.60 | 1450.87 | 1456.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 1449.30 | 1450.87 | 1456.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1457.80 | 1451.83 | 1454.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 1457.80 | 1451.83 | 1454.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1460.10 | 1453.48 | 1454.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 1456.90 | 1453.48 | 1454.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:00:00 | 1456.90 | 1454.16 | 1455.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:00:00 | 1456.10 | 1454.55 | 1455.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 12:15:00 | 1459.70 | 1455.78 | 1455.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 1459.70 | 1455.78 | 1455.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 14:15:00 | 1462.00 | 1457.84 | 1456.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 1454.30 | 1457.74 | 1457.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 10:15:00 | 1454.30 | 1457.74 | 1457.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1454.30 | 1457.74 | 1457.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 1454.30 | 1457.74 | 1457.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1454.50 | 1457.09 | 1456.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:45:00 | 1456.40 | 1456.85 | 1456.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 1451.50 | 1455.78 | 1456.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 1451.50 | 1455.78 | 1456.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 1447.00 | 1454.03 | 1455.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 1455.70 | 1452.69 | 1454.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 10:15:00 | 1455.70 | 1452.69 | 1454.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1455.70 | 1452.69 | 1454.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1455.70 | 1452.69 | 1454.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1460.30 | 1454.21 | 1454.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 1460.30 | 1454.21 | 1454.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 1461.50 | 1455.67 | 1455.47 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 1446.20 | 1454.58 | 1455.25 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 13:15:00 | 1461.00 | 1456.41 | 1455.93 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 1448.60 | 1454.30 | 1455.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 1441.00 | 1451.64 | 1453.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1451.60 | 1447.30 | 1450.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 1451.60 | 1447.30 | 1450.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1451.60 | 1447.30 | 1450.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1451.60 | 1447.30 | 1450.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1457.00 | 1449.24 | 1451.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1454.20 | 1449.24 | 1451.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1460.40 | 1451.47 | 1452.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 1460.40 | 1451.47 | 1452.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 1465.30 | 1454.24 | 1453.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 1472.50 | 1457.89 | 1454.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 14:15:00 | 1470.80 | 1472.34 | 1467.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 14:30:00 | 1468.00 | 1472.34 | 1467.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1477.00 | 1472.64 | 1468.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 1494.40 | 1477.44 | 1473.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 1490.80 | 1498.66 | 1499.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 1490.80 | 1498.66 | 1499.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 1485.80 | 1495.16 | 1497.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 1495.00 | 1491.04 | 1494.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 1495.00 | 1491.04 | 1494.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1495.00 | 1491.04 | 1494.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:00:00 | 1495.00 | 1491.04 | 1494.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 1491.70 | 1491.17 | 1494.11 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 1501.90 | 1495.72 | 1495.32 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 1491.20 | 1495.24 | 1495.24 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 12:15:00 | 1495.30 | 1495.25 | 1495.25 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 13:15:00 | 1490.10 | 1494.22 | 1494.78 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 1497.40 | 1495.56 | 1495.32 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 13:15:00 | 1492.40 | 1494.90 | 1495.06 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 14:15:00 | 1496.50 | 1495.22 | 1495.19 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 15:15:00 | 1493.30 | 1494.84 | 1495.02 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 1497.50 | 1495.37 | 1495.25 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 1484.10 | 1493.12 | 1494.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 1472.90 | 1486.91 | 1490.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 14:15:00 | 1475.90 | 1475.66 | 1481.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 15:00:00 | 1475.90 | 1475.66 | 1481.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1433.80 | 1433.31 | 1442.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:15:00 | 1445.00 | 1433.31 | 1442.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1458.00 | 1438.25 | 1443.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 1458.00 | 1438.25 | 1443.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1453.30 | 1441.26 | 1444.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:30:00 | 1443.90 | 1444.38 | 1445.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:00:00 | 1447.90 | 1444.38 | 1445.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 10:15:00 | 1459.50 | 1447.41 | 1446.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 1459.50 | 1447.41 | 1446.86 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1439.20 | 1445.59 | 1446.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 1428.70 | 1441.37 | 1443.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 11:15:00 | 1445.00 | 1440.04 | 1442.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 11:15:00 | 1445.00 | 1440.04 | 1442.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1445.00 | 1440.04 | 1442.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:45:00 | 1450.50 | 1440.04 | 1442.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 1454.20 | 1442.87 | 1443.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:45:00 | 1457.00 | 1442.87 | 1443.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 226 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 1463.60 | 1448.08 | 1446.08 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 1419.40 | 1441.31 | 1443.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1411.40 | 1422.46 | 1427.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1415.50 | 1412.77 | 1419.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 1415.50 | 1412.77 | 1419.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1415.50 | 1412.77 | 1419.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1415.50 | 1412.77 | 1419.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1415.00 | 1413.22 | 1419.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1429.60 | 1413.22 | 1419.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1431.40 | 1416.86 | 1420.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 1418.90 | 1416.86 | 1420.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 11:15:00 | 1424.10 | 1419.16 | 1421.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:15:00 | 1420.90 | 1421.70 | 1422.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 1429.00 | 1423.16 | 1422.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 1429.00 | 1423.16 | 1422.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1469.50 | 1434.08 | 1427.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 12:15:00 | 1458.80 | 1460.04 | 1448.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 13:00:00 | 1458.80 | 1460.04 | 1448.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1477.00 | 1483.03 | 1477.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 1477.00 | 1483.03 | 1477.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1476.60 | 1481.75 | 1477.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:30:00 | 1474.80 | 1481.75 | 1477.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1474.90 | 1480.38 | 1477.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:00:00 | 1474.90 | 1480.38 | 1477.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 1475.00 | 1477.88 | 1476.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 1489.60 | 1477.88 | 1476.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:45:00 | 1482.10 | 1480.06 | 1478.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:00:00 | 1480.20 | 1480.09 | 1479.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 10:30:00 | 1480.30 | 1484.44 | 1483.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1485.60 | 1484.27 | 1483.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 1478.10 | 1482.62 | 1482.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 229 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 1478.10 | 1482.62 | 1482.63 | EMA200 below EMA400 |

### Cycle 230 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 1490.20 | 1483.78 | 1482.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 1494.80 | 1489.41 | 1486.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 12:15:00 | 1488.80 | 1493.49 | 1490.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 12:15:00 | 1488.80 | 1493.49 | 1490.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 1488.80 | 1493.49 | 1490.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 1488.80 | 1493.49 | 1490.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 1491.70 | 1493.13 | 1490.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:15:00 | 1487.80 | 1493.13 | 1490.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1486.10 | 1491.73 | 1490.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:30:00 | 1484.60 | 1491.73 | 1490.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 1489.80 | 1491.34 | 1490.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 1476.60 | 1491.34 | 1490.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 231 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 1472.30 | 1487.53 | 1488.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1464.60 | 1480.43 | 1484.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 12:15:00 | 1468.80 | 1468.50 | 1474.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:30:00 | 1472.90 | 1468.50 | 1474.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1475.20 | 1468.19 | 1472.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 1477.10 | 1468.19 | 1472.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1473.70 | 1469.29 | 1472.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:15:00 | 1475.70 | 1469.29 | 1472.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 1478.70 | 1471.17 | 1473.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 1478.70 | 1471.17 | 1473.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 1475.60 | 1472.06 | 1473.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:15:00 | 1482.10 | 1472.06 | 1473.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 232 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 1483.10 | 1474.27 | 1474.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 11:15:00 | 1484.80 | 1478.87 | 1476.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 14:15:00 | 1479.20 | 1479.77 | 1477.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 1479.20 | 1479.77 | 1477.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 1479.20 | 1479.77 | 1477.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 1479.20 | 1479.77 | 1477.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1489.30 | 1503.46 | 1499.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 1489.30 | 1503.46 | 1499.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1486.50 | 1500.07 | 1498.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 1464.10 | 1500.07 | 1498.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 233 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1464.50 | 1492.96 | 1495.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 10:15:00 | 1426.50 | 1438.73 | 1454.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 1438.30 | 1436.81 | 1450.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:00:00 | 1438.30 | 1436.81 | 1450.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1442.00 | 1438.52 | 1448.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 1452.90 | 1438.52 | 1448.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1453.10 | 1441.43 | 1448.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 1449.80 | 1441.43 | 1448.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1450.80 | 1443.31 | 1448.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:00:00 | 1446.90 | 1444.02 | 1448.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:00:00 | 1446.20 | 1444.46 | 1448.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:30:00 | 1443.90 | 1444.31 | 1447.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1374.56 | 1427.87 | 1439.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1373.89 | 1427.87 | 1439.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1371.70 | 1427.87 | 1439.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 1399.80 | 1394.02 | 1411.87 | SL hit (close>ema200) qty=0.50 sl=1394.02 alert=retest2 |

### Cycle 234 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 1428.60 | 1414.76 | 1414.76 | EMA200 above EMA400 |

### Cycle 235 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 1408.90 | 1413.95 | 1414.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 1401.20 | 1411.40 | 1413.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1355.00 | 1347.96 | 1361.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1355.00 | 1347.96 | 1361.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1355.00 | 1347.96 | 1361.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 1359.00 | 1347.96 | 1361.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1354.70 | 1350.26 | 1360.05 | EMA400 retest candle locked (from downside) |

### Cycle 236 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 1369.80 | 1362.53 | 1361.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1383.90 | 1368.37 | 1364.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1335.60 | 1365.86 | 1364.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1335.60 | 1365.86 | 1364.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1335.60 | 1365.86 | 1364.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1335.60 | 1365.86 | 1364.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 237 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1335.80 | 1359.85 | 1362.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 1327.00 | 1349.03 | 1356.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1334.80 | 1331.82 | 1344.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 1345.00 | 1334.45 | 1344.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1345.00 | 1334.45 | 1344.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 1345.00 | 1334.45 | 1344.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1348.80 | 1337.32 | 1344.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 1346.00 | 1337.32 | 1344.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 1349.00 | 1339.66 | 1345.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:45:00 | 1351.50 | 1339.66 | 1345.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 1336.40 | 1339.01 | 1344.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 1334.50 | 1339.85 | 1344.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1309.80 | 1341.66 | 1344.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:00:00 | 1331.70 | 1327.52 | 1328.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1370.90 | 1337.39 | 1332.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 238 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1370.90 | 1337.39 | 1332.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 1375.20 | 1354.43 | 1342.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1335.80 | 1356.47 | 1348.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1335.80 | 1356.47 | 1348.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1335.80 | 1356.47 | 1348.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1335.80 | 1356.47 | 1348.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1335.90 | 1352.36 | 1346.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 1331.90 | 1352.36 | 1346.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 239 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 1326.10 | 1341.17 | 1342.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1313.60 | 1335.65 | 1340.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1309.70 | 1300.36 | 1313.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1309.70 | 1300.36 | 1313.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1309.70 | 1300.36 | 1313.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:30:00 | 1299.90 | 1299.09 | 1311.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:00:00 | 1304.80 | 1307.57 | 1312.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1280.10 | 1307.46 | 1312.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 15:15:00 | 1299.90 | 1288.37 | 1287.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 240 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 1299.90 | 1288.37 | 1287.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 1362.60 | 1303.21 | 1294.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 10:15:00 | 1344.10 | 1345.87 | 1327.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 11:00:00 | 1344.10 | 1345.87 | 1327.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 1351.00 | 1347.48 | 1338.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:30:00 | 1336.70 | 1347.48 | 1338.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1335.80 | 1349.54 | 1343.28 | EMA400 retest candle locked (from upside) |

### Cycle 241 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 1331.30 | 1338.61 | 1339.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 1325.80 | 1336.05 | 1338.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1336.10 | 1334.85 | 1337.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1336.10 | 1334.85 | 1337.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1336.10 | 1334.85 | 1337.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:45:00 | 1321.50 | 1328.58 | 1332.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 12:30:00 | 1324.90 | 1327.09 | 1330.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 1371.00 | 1337.18 | 1334.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 242 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 1371.00 | 1337.18 | 1334.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 10:15:00 | 1376.90 | 1345.12 | 1338.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 1389.00 | 1389.09 | 1375.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 1392.10 | 1389.09 | 1375.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1406.30 | 1410.76 | 1401.69 | EMA400 retest candle locked (from upside) |

### Cycle 243 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 1392.80 | 1400.01 | 1400.62 | EMA200 below EMA400 |

### Cycle 244 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1399.60 | 1399.07 | 1399.04 | EMA200 above EMA400 |

### Cycle 245 — SELL (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 15:15:00 | 1397.00 | 1398.66 | 1398.86 | EMA200 below EMA400 |

### Cycle 246 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1402.20 | 1399.37 | 1399.16 | EMA200 above EMA400 |

### Cycle 247 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 1393.10 | 1398.11 | 1398.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 1388.50 | 1395.28 | 1396.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 1378.60 | 1376.06 | 1383.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 15:15:00 | 1378.60 | 1376.06 | 1383.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1378.60 | 1376.06 | 1383.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 1387.60 | 1376.06 | 1383.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1379.00 | 1376.65 | 1382.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 1383.70 | 1376.65 | 1382.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1383.10 | 1377.94 | 1382.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:45:00 | 1380.90 | 1377.94 | 1382.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1379.00 | 1378.15 | 1382.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 1385.00 | 1378.15 | 1382.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1392.70 | 1369.58 | 1371.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:15:00 | 1403.10 | 1369.58 | 1371.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 248 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 1409.00 | 1377.47 | 1374.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 1418.70 | 1385.71 | 1378.94 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 12:15:00 | 1499.60 | 2024-04-19 09:15:00 | 1424.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 12:15:00 | 1499.60 | 2024-04-22 09:15:00 | 1425.43 | STOP_HIT | 0.50 | 4.95% |
| BUY | retest2 | 2024-04-23 12:30:00 | 1444.33 | 2024-05-02 13:15:00 | 1490.98 | STOP_HIT | 1.00 | 3.23% |
| SELL | retest2 | 2024-05-08 09:15:00 | 1412.40 | 2024-05-10 12:15:00 | 1454.30 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2024-05-22 09:45:00 | 1525.03 | 2024-05-23 11:15:00 | 1490.50 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-05-24 12:15:00 | 1488.88 | 2024-05-27 12:15:00 | 1502.48 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1473.30 | 2024-06-04 09:15:00 | 1509.60 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2024-06-07 09:15:00 | 1558.50 | 2024-06-12 11:15:00 | 1541.75 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-06-21 09:15:00 | 1573.15 | 2024-06-24 09:15:00 | 1559.63 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-06-21 14:45:00 | 1570.95 | 2024-06-24 09:15:00 | 1559.63 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-06-28 09:30:00 | 1588.45 | 2024-07-01 11:15:00 | 1570.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-06-28 13:15:00 | 1580.25 | 2024-07-01 11:15:00 | 1570.50 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-06-28 13:45:00 | 1580.03 | 2024-07-01 11:15:00 | 1570.50 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-06-28 14:45:00 | 1580.00 | 2024-07-01 11:15:00 | 1570.50 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-07-03 11:15:00 | 1554.13 | 2024-07-09 10:15:00 | 1557.00 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-07-03 15:15:00 | 1553.13 | 2024-07-09 10:15:00 | 1557.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-07-04 10:00:00 | 1555.45 | 2024-07-09 10:15:00 | 1557.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-07-04 12:00:00 | 1555.93 | 2024-07-09 10:15:00 | 1557.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-07-05 11:00:00 | 1547.20 | 2024-07-09 10:15:00 | 1557.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-07-05 11:30:00 | 1546.55 | 2024-07-09 12:15:00 | 1555.23 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-07-05 13:00:00 | 1545.60 | 2024-07-09 12:15:00 | 1555.23 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-07-05 13:45:00 | 1546.53 | 2024-07-09 12:15:00 | 1555.23 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-07-08 11:00:00 | 1537.95 | 2024-07-09 12:15:00 | 1555.23 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-07-11 12:15:00 | 1570.43 | 2024-07-19 12:15:00 | 1563.93 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-07-26 11:15:00 | 1565.55 | 2024-07-26 14:15:00 | 1568.98 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-07-26 12:00:00 | 1565.48 | 2024-07-26 14:15:00 | 1568.98 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-07-26 12:45:00 | 1563.78 | 2024-07-26 14:15:00 | 1568.98 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-07-26 13:45:00 | 1562.50 | 2024-07-26 14:15:00 | 1568.98 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-07-31 09:15:00 | 1594.78 | 2024-08-01 13:15:00 | 1572.23 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-08-01 09:30:00 | 1591.23 | 2024-08-01 13:15:00 | 1572.23 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-08-06 13:15:00 | 1544.85 | 2024-08-07 13:15:00 | 1572.48 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-08-26 11:45:00 | 1551.20 | 2024-08-27 13:15:00 | 1541.93 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-08-26 13:45:00 | 1550.38 | 2024-08-27 13:15:00 | 1541.93 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-08-26 14:15:00 | 1551.80 | 2024-08-27 13:15:00 | 1541.93 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-08-27 09:15:00 | 1550.55 | 2024-08-27 13:15:00 | 1541.93 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-09-05 12:15:00 | 1611.58 | 2024-09-12 09:15:00 | 1610.85 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-09-06 09:15:00 | 1615.63 | 2024-09-12 09:15:00 | 1610.85 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-09-13 11:15:00 | 1642.40 | 2024-09-17 14:15:00 | 1634.88 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-09-16 13:15:00 | 1643.05 | 2024-09-17 14:15:00 | 1634.88 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-09-17 09:45:00 | 1643.75 | 2024-09-17 14:15:00 | 1634.88 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-10-09 14:15:00 | 1587.03 | 2024-10-15 11:15:00 | 1580.10 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-10-09 15:00:00 | 1586.03 | 2024-10-15 11:15:00 | 1580.10 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2024-10-10 10:00:00 | 1585.80 | 2024-10-15 11:15:00 | 1580.10 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2024-10-15 09:45:00 | 1584.03 | 2024-10-15 11:15:00 | 1580.10 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2024-10-16 14:15:00 | 1583.95 | 2024-10-21 11:15:00 | 1572.65 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-10-17 10:15:00 | 1584.50 | 2024-10-21 11:15:00 | 1572.65 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-10-18 09:15:00 | 1615.98 | 2024-10-21 11:15:00 | 1572.65 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2024-10-21 10:45:00 | 1582.73 | 2024-10-21 11:15:00 | 1572.65 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-10-24 11:30:00 | 1552.83 | 2024-10-25 09:15:00 | 1589.90 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-10-24 13:45:00 | 1554.75 | 2024-10-25 09:15:00 | 1589.90 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-11-18 10:30:00 | 1508.88 | 2024-11-25 11:15:00 | 1514.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-11-18 14:15:00 | 1508.33 | 2024-11-25 11:15:00 | 1514.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-11-18 15:00:00 | 1508.60 | 2024-11-25 12:15:00 | 1504.18 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2024-11-19 09:15:00 | 1494.00 | 2024-11-25 12:15:00 | 1504.18 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-11-21 12:45:00 | 1486.95 | 2024-11-25 12:15:00 | 1504.18 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-11-25 09:45:00 | 1484.78 | 2024-11-25 12:15:00 | 1504.18 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-11-29 10:45:00 | 1526.48 | 2024-12-09 09:15:00 | 1560.50 | STOP_HIT | 1.00 | 2.23% |
| BUY | retest2 | 2024-11-29 12:30:00 | 1526.50 | 2024-12-09 09:15:00 | 1560.50 | STOP_HIT | 1.00 | 2.23% |
| BUY | retest2 | 2024-11-29 13:45:00 | 1527.35 | 2024-12-09 09:15:00 | 1560.50 | STOP_HIT | 1.00 | 2.17% |
| BUY | retest2 | 2024-12-12 11:15:00 | 1590.75 | 2024-12-12 11:15:00 | 1593.50 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2024-12-26 11:15:00 | 1475.73 | 2025-01-02 14:15:00 | 1463.83 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2024-12-27 10:00:00 | 1473.33 | 2025-01-02 14:15:00 | 1463.83 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-01-10 11:15:00 | 1474.35 | 2025-01-10 13:15:00 | 1456.80 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-01-16 10:45:00 | 1395.05 | 2025-01-17 15:15:00 | 1407.95 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-02-04 14:00:00 | 1471.68 | 2025-02-04 14:15:00 | 1463.03 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-02-07 13:45:00 | 1428.10 | 2025-02-20 13:15:00 | 1405.05 | STOP_HIT | 1.00 | 1.61% |
| SELL | retest2 | 2025-02-10 11:30:00 | 1431.50 | 2025-02-20 13:15:00 | 1405.05 | STOP_HIT | 1.00 | 1.85% |
| SELL | retest2 | 2025-02-10 14:00:00 | 1431.43 | 2025-02-20 13:15:00 | 1405.05 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest2 | 2025-02-10 14:45:00 | 1431.45 | 2025-02-20 13:15:00 | 1405.05 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest2 | 2025-02-13 13:15:00 | 1404.00 | 2025-02-20 13:15:00 | 1405.05 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest1 | 2025-02-25 09:15:00 | 1365.23 | 2025-03-03 14:15:00 | 1337.33 | STOP_HIT | 1.00 | 2.04% |
| SELL | retest2 | 2025-03-04 09:15:00 | 1322.68 | 2025-03-04 14:15:00 | 1348.63 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-03-06 09:15:00 | 1362.23 | 2025-03-11 10:15:00 | 1358.58 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-03-26 11:15:00 | 1434.50 | 2025-04-01 13:15:00 | 1416.78 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-03-27 15:15:00 | 1434.70 | 2025-04-01 13:15:00 | 1416.78 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-04-01 09:30:00 | 1434.15 | 2025-04-01 13:15:00 | 1416.78 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-04-21 13:15:00 | 1516.90 | 2025-04-25 10:15:00 | 1503.80 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-05-02 09:15:00 | 1503.70 | 2025-05-05 09:15:00 | 1526.20 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-05-02 11:30:00 | 1507.50 | 2025-05-05 09:15:00 | 1526.20 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-05-08 10:15:00 | 1475.40 | 2025-05-09 09:15:00 | 1505.50 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-05-08 10:45:00 | 1474.50 | 2025-05-09 09:15:00 | 1505.50 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-05-27 11:45:00 | 1515.60 | 2025-05-28 09:15:00 | 1507.15 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-05-27 15:15:00 | 1516.25 | 2025-05-28 09:15:00 | 1507.15 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-04 12:30:00 | 1549.40 | 2025-06-05 12:15:00 | 1528.55 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-06-04 15:00:00 | 1546.75 | 2025-06-05 12:15:00 | 1528.55 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-06-16 14:45:00 | 1500.55 | 2025-06-24 09:15:00 | 1504.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-06-17 10:00:00 | 1500.45 | 2025-06-24 09:15:00 | 1504.60 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-06-17 11:45:00 | 1500.45 | 2025-06-24 10:15:00 | 1503.45 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-06-18 10:15:00 | 1498.55 | 2025-06-24 10:15:00 | 1503.45 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1473.05 | 2025-06-24 10:15:00 | 1503.45 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-06-23 12:45:00 | 1473.90 | 2025-06-24 10:15:00 | 1503.45 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-06-26 09:30:00 | 1509.10 | 2025-07-07 11:15:00 | 1518.95 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2025-06-26 13:00:00 | 1506.25 | 2025-07-07 11:15:00 | 1518.95 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2025-07-16 09:15:00 | 1489.65 | 2025-07-17 09:15:00 | 1502.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-07-16 11:30:00 | 1489.90 | 2025-07-17 09:15:00 | 1502.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-07-16 12:15:00 | 1489.05 | 2025-07-17 09:15:00 | 1502.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-07-16 13:00:00 | 1490.10 | 2025-07-17 09:15:00 | 1502.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-07-18 09:15:00 | 1492.00 | 2025-08-04 09:15:00 | 1465.10 | STOP_HIT | 1.00 | 1.80% |
| BUY | retest2 | 2025-08-21 09:15:00 | 1548.95 | 2025-08-21 11:15:00 | 1541.15 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-08-21 10:30:00 | 1544.95 | 2025-08-21 11:15:00 | 1541.15 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-08-26 11:30:00 | 1549.50 | 2025-08-28 12:15:00 | 1540.15 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-08-26 15:15:00 | 1551.25 | 2025-08-28 12:15:00 | 1540.15 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-08-28 11:30:00 | 1548.45 | 2025-08-28 12:15:00 | 1540.15 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-09-03 15:15:00 | 1556.35 | 2025-09-05 11:15:00 | 1551.85 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-09-04 09:30:00 | 1557.55 | 2025-09-05 11:15:00 | 1551.85 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-09-16 14:15:00 | 1531.60 | 2025-10-01 09:15:00 | 1455.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 13:00:00 | 1528.50 | 2025-10-01 09:15:00 | 1452.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 09:45:00 | 1531.85 | 2025-10-01 09:15:00 | 1455.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 12:15:00 | 1531.90 | 2025-10-01 09:15:00 | 1455.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 10:30:00 | 1526.75 | 2025-10-01 09:15:00 | 1450.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 10:00:00 | 1527.05 | 2025-10-01 09:15:00 | 1450.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:30:00 | 1525.80 | 2025-10-01 09:15:00 | 1449.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 14:15:00 | 1531.60 | 2025-10-01 13:15:00 | 1468.50 | STOP_HIT | 0.50 | 4.12% |
| SELL | retest2 | 2025-09-17 13:00:00 | 1528.50 | 2025-10-01 13:15:00 | 1468.50 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2025-09-18 09:45:00 | 1531.85 | 2025-10-01 13:15:00 | 1468.50 | STOP_HIT | 0.50 | 4.14% |
| SELL | retest2 | 2025-09-18 12:15:00 | 1531.90 | 2025-10-01 13:15:00 | 1468.50 | STOP_HIT | 0.50 | 4.14% |
| SELL | retest2 | 2025-09-19 10:30:00 | 1526.75 | 2025-10-01 13:15:00 | 1468.50 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-09-22 10:00:00 | 1527.05 | 2025-10-01 13:15:00 | 1468.50 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2025-09-22 13:30:00 | 1525.80 | 2025-10-01 13:15:00 | 1468.50 | STOP_HIT | 0.50 | 3.76% |
| BUY | retest2 | 2025-10-13 12:00:00 | 1514.90 | 2025-10-14 12:15:00 | 1494.80 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-10-13 15:00:00 | 1515.10 | 2025-10-14 12:15:00 | 1494.80 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-14 09:15:00 | 1517.50 | 2025-10-14 12:15:00 | 1494.80 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-10-21 13:45:00 | 1532.80 | 2025-10-23 09:15:00 | 1515.40 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-23 11:00:00 | 1530.50 | 2025-10-23 14:15:00 | 1515.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-10-30 14:15:00 | 1492.80 | 2025-11-10 10:15:00 | 1462.10 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2025-11-19 12:45:00 | 1482.10 | 2025-11-21 09:15:00 | 1473.20 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-11-20 09:15:00 | 1487.80 | 2025-11-21 09:15:00 | 1473.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-11-20 10:00:00 | 1486.60 | 2025-11-21 09:15:00 | 1473.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-12-02 09:15:00 | 1460.40 | 2025-12-02 15:15:00 | 1470.60 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-12-02 09:45:00 | 1461.30 | 2025-12-02 15:15:00 | 1470.60 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-12-02 11:15:00 | 1462.00 | 2025-12-02 15:15:00 | 1470.60 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-12-02 13:00:00 | 1463.30 | 2025-12-02 15:15:00 | 1470.60 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-12-22 09:15:00 | 1456.90 | 2025-12-22 12:15:00 | 1459.70 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-12-22 10:00:00 | 1456.90 | 2025-12-22 12:15:00 | 1459.70 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-12-22 11:00:00 | 1456.10 | 2025-12-22 12:15:00 | 1459.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-12-24 12:45:00 | 1456.40 | 2025-12-24 13:15:00 | 1451.50 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-01-05 10:15:00 | 1494.40 | 2026-01-09 10:15:00 | 1490.80 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2026-01-23 09:30:00 | 1443.90 | 2026-01-23 10:15:00 | 1459.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-23 10:00:00 | 1447.90 | 2026-01-23 10:15:00 | 1459.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-02-03 10:15:00 | 1418.90 | 2026-02-03 13:15:00 | 1429.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-02-03 11:15:00 | 1424.10 | 2026-02-03 13:15:00 | 1429.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2026-02-03 13:15:00 | 1420.90 | 2026-02-03 13:15:00 | 1429.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-02-11 09:15:00 | 1489.60 | 2026-02-13 14:15:00 | 1478.10 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-02-12 09:45:00 | 1482.10 | 2026-02-13 14:15:00 | 1478.10 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-02-12 11:00:00 | 1480.20 | 2026-02-13 14:15:00 | 1478.10 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-02-13 10:30:00 | 1480.30 | 2026-02-13 14:15:00 | 1478.10 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2026-03-06 12:00:00 | 1446.90 | 2026-03-09 09:15:00 | 1374.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:00:00 | 1446.20 | 2026-03-09 09:15:00 | 1373.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:30:00 | 1443.90 | 2026-03-09 09:15:00 | 1371.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:00:00 | 1446.90 | 2026-03-10 09:15:00 | 1399.80 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2026-03-06 13:00:00 | 1446.20 | 2026-03-10 09:15:00 | 1399.80 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2026-03-06 13:30:00 | 1443.90 | 2026-03-10 09:15:00 | 1399.80 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2026-03-20 14:30:00 | 1334.50 | 2026-03-25 09:15:00 | 1370.90 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1309.80 | 2026-03-25 09:15:00 | 1370.90 | STOP_HIT | 1.00 | -4.66% |
| SELL | retest2 | 2026-03-24 15:00:00 | 1331.70 | 2026-03-25 09:15:00 | 1370.90 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2026-04-01 10:30:00 | 1299.90 | 2026-04-07 15:15:00 | 1299.90 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2026-04-01 15:00:00 | 1304.80 | 2026-04-07 15:15:00 | 1299.90 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1280.10 | 2026-04-07 15:15:00 | 1299.90 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-16 09:45:00 | 1321.50 | 2026-04-17 09:15:00 | 1371.00 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2026-04-16 12:30:00 | 1324.90 | 2026-04-17 09:15:00 | 1371.00 | STOP_HIT | 1.00 | -3.48% |
