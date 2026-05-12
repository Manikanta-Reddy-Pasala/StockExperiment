# APL Apollo Tubes Ltd. (APLAPOLLO)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 1950.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 209 |
| ALERT1 | 143 |
| ALERT2 | 143 |
| ALERT2_SKIP | 85 |
| ALERT3 | 398 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 198 |
| PARTIAL | 24 |
| TARGET_HIT | 11 |
| STOP_HIT | 195 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 224 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 84 / 140
- **Target hits / Stop hits / Partials:** 8 / 195 / 21
- **Avg / median % per leg:** 0.52% / -0.52%
- **Sum % (uncompounded):** 115.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 95 | 34 | 35.8% | 8 | 87 | 0 | 0.64% | 60.7% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 0 | 5 | 0 | 0.11% | 0.6% |
| BUY @ 3rd Alert (retest2) | 90 | 30 | 33.3% | 8 | 82 | 0 | 0.67% | 60.1% |
| SELL (all) | 129 | 50 | 38.8% | 0 | 108 | 21 | 0.43% | 55.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 129 | 50 | 38.8% | 0 | 108 | 21 | 0.43% | 55.1% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 5 | 0 | 0.11% | 0.6% |
| retest2 (combined) | 219 | 80 | 36.5% | 8 | 190 | 21 | 0.53% | 115.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 11:15:00 | 1137.05 | 1115.21 | 1113.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 12:15:00 | 1140.50 | 1120.27 | 1116.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 09:15:00 | 1145.70 | 1147.67 | 1138.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 1145.70 | 1147.67 | 1138.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 1145.70 | 1147.67 | 1138.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:30:00 | 1139.05 | 1147.67 | 1138.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 1138.55 | 1145.84 | 1138.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:45:00 | 1138.00 | 1145.84 | 1138.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 1144.65 | 1145.61 | 1138.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 12:45:00 | 1149.00 | 1147.28 | 1140.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-26 12:15:00 | 1134.90 | 1142.43 | 1141.48 | SL hit (close<static) qty=1.00 sl=1137.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 13:15:00 | 1130.35 | 1140.02 | 1140.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-30 09:15:00 | 1120.90 | 1131.92 | 1134.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 13:15:00 | 1129.40 | 1114.96 | 1120.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 13:15:00 | 1129.40 | 1114.96 | 1120.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 13:15:00 | 1129.40 | 1114.96 | 1120.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 14:00:00 | 1129.40 | 1114.96 | 1120.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 1139.95 | 1119.95 | 1122.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 14:30:00 | 1140.00 | 1119.95 | 1122.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 1123.10 | 1121.55 | 1122.96 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-06-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 11:15:00 | 1133.25 | 1125.51 | 1124.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 12:15:00 | 1147.40 | 1129.89 | 1126.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 09:15:00 | 1128.50 | 1135.43 | 1130.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 09:15:00 | 1128.50 | 1135.43 | 1130.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 1128.50 | 1135.43 | 1130.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 10:00:00 | 1128.50 | 1135.43 | 1130.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 10:15:00 | 1129.30 | 1134.21 | 1130.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 10:30:00 | 1126.50 | 1134.21 | 1130.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 11:15:00 | 1132.00 | 1133.76 | 1130.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 13:30:00 | 1138.00 | 1134.26 | 1131.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 09:15:00 | 1141.60 | 1134.30 | 1132.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-12 10:15:00 | 1251.80 | 1231.60 | 1210.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 14:15:00 | 1297.30 | 1304.27 | 1304.32 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 09:15:00 | 1313.25 | 1305.61 | 1304.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 12:15:00 | 1322.85 | 1312.78 | 1308.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 10:15:00 | 1328.00 | 1333.24 | 1324.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-21 10:45:00 | 1330.70 | 1333.24 | 1324.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 11:15:00 | 1330.00 | 1332.59 | 1325.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 11:45:00 | 1325.45 | 1332.59 | 1325.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 14:15:00 | 1338.65 | 1336.31 | 1329.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 14:45:00 | 1327.15 | 1336.31 | 1329.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 1363.90 | 1341.66 | 1332.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 10:30:00 | 1373.95 | 1349.00 | 1336.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-23 10:30:00 | 1372.05 | 1371.43 | 1357.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-23 15:15:00 | 1330.10 | 1354.11 | 1354.10 | SL hit (close<static) qty=1.00 sl=1330.30 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 09:15:00 | 1317.15 | 1346.72 | 1350.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 13:15:00 | 1304.95 | 1325.01 | 1337.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 1346.55 | 1325.13 | 1334.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 1346.55 | 1325.13 | 1334.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 1346.55 | 1325.13 | 1334.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:45:00 | 1348.20 | 1325.13 | 1334.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 1352.00 | 1330.50 | 1336.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:00:00 | 1352.00 | 1330.50 | 1336.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 14:15:00 | 1339.00 | 1338.22 | 1338.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 15:15:00 | 1333.90 | 1338.22 | 1338.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 15:15:00 | 1333.90 | 1337.36 | 1338.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 09:15:00 | 1346.50 | 1337.36 | 1338.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 1339.10 | 1337.70 | 1338.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 10:30:00 | 1332.35 | 1336.94 | 1337.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 12:00:00 | 1334.15 | 1323.12 | 1324.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-03 13:15:00 | 1326.80 | 1325.22 | 1325.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 13:15:00 | 1326.80 | 1325.22 | 1325.08 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 13:15:00 | 1317.55 | 1325.03 | 1325.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 14:15:00 | 1310.75 | 1322.17 | 1324.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 1326.70 | 1321.93 | 1323.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 09:15:00 | 1326.70 | 1321.93 | 1323.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 1326.70 | 1321.93 | 1323.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-05 14:15:00 | 1312.00 | 1320.42 | 1322.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 09:30:00 | 1310.85 | 1316.91 | 1320.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 13:45:00 | 1312.10 | 1313.64 | 1317.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 14:15:00 | 1313.10 | 1313.64 | 1317.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 1321.00 | 1315.11 | 1317.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 15:00:00 | 1321.00 | 1315.11 | 1317.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 15:15:00 | 1328.05 | 1317.70 | 1318.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-07-07 09:15:00 | 1341.30 | 1322.42 | 1320.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 09:15:00 | 1341.30 | 1322.42 | 1320.68 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 15:15:00 | 1317.00 | 1320.53 | 1320.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 1310.00 | 1318.43 | 1319.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 1330.85 | 1315.10 | 1316.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 1330.85 | 1315.10 | 1316.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 1330.85 | 1315.10 | 1316.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 11:00:00 | 1318.00 | 1315.68 | 1316.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 12:00:00 | 1319.50 | 1316.45 | 1316.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 12:45:00 | 1319.85 | 1317.16 | 1317.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-11 13:15:00 | 1321.45 | 1318.01 | 1317.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 13:15:00 | 1321.45 | 1318.01 | 1317.63 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 09:15:00 | 1309.95 | 1316.23 | 1316.89 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 09:15:00 | 1328.85 | 1315.71 | 1315.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 10:15:00 | 1354.00 | 1323.37 | 1319.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-14 12:15:00 | 1342.95 | 1343.23 | 1335.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-14 13:00:00 | 1342.95 | 1343.23 | 1335.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 1344.85 | 1343.55 | 1336.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 09:30:00 | 1347.55 | 1342.66 | 1337.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 15:15:00 | 1348.00 | 1355.70 | 1353.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-24 09:15:00 | 1482.31 | 1428.82 | 1412.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-08-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 10:15:00 | 1525.00 | 1563.76 | 1566.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 12:15:00 | 1515.50 | 1547.80 | 1558.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 15:15:00 | 1538.00 | 1527.03 | 1536.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 15:15:00 | 1538.00 | 1527.03 | 1536.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 15:15:00 | 1538.00 | 1527.03 | 1536.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 09:15:00 | 1507.05 | 1527.03 | 1536.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 1472.75 | 1516.18 | 1530.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 11:15:00 | 1460.90 | 1506.75 | 1525.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 09:15:00 | 1457.30 | 1485.00 | 1505.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-09 14:15:00 | 1524.80 | 1502.84 | 1500.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 14:15:00 | 1524.80 | 1502.84 | 1500.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 15:15:00 | 1538.15 | 1509.90 | 1503.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 10:15:00 | 1556.90 | 1557.05 | 1539.55 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 13:00:00 | 1561.25 | 1557.82 | 1542.93 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 09:15:00 | 1562.85 | 1555.82 | 1545.66 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 1569.00 | 1558.46 | 1547.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 11:15:00 | 1574.80 | 1561.54 | 1550.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 13:15:00 | 1579.20 | 1566.58 | 1554.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 14:15:00 | 1574.60 | 1568.06 | 1556.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-17 09:15:00 | 1573.80 | 1582.24 | 1574.56 | SL hit (close<ema400) qty=1.00 sl=1574.56 alert=retest1 |

### Cycle 16 — SELL (started 2023-08-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 14:15:00 | 1559.95 | 1569.43 | 1570.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 15:15:00 | 1554.30 | 1566.40 | 1569.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 10:15:00 | 1566.15 | 1563.98 | 1567.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 10:15:00 | 1566.15 | 1563.98 | 1567.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 1566.15 | 1563.98 | 1567.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:45:00 | 1567.05 | 1563.98 | 1567.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 11:15:00 | 1569.00 | 1564.99 | 1567.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 11:45:00 | 1569.45 | 1564.99 | 1567.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 12:15:00 | 1549.15 | 1561.82 | 1565.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 14:15:00 | 1547.45 | 1559.46 | 1564.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 14:45:00 | 1547.80 | 1557.56 | 1563.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 10:00:00 | 1544.15 | 1553.35 | 1560.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-22 09:15:00 | 1584.45 | 1549.03 | 1552.87 | SL hit (close>static) qty=1.00 sl=1569.90 alert=retest2 |

### Cycle 17 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 1565.00 | 1555.58 | 1555.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 13:15:00 | 1573.05 | 1560.58 | 1557.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 14:15:00 | 1560.25 | 1560.51 | 1558.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 14:15:00 | 1560.25 | 1560.51 | 1558.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 14:15:00 | 1560.25 | 1560.51 | 1558.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 15:00:00 | 1560.25 | 1560.51 | 1558.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 15:15:00 | 1568.00 | 1562.01 | 1558.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 09:15:00 | 1579.95 | 1562.01 | 1558.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 11:15:00 | 1627.25 | 1655.56 | 1656.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2023-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 11:15:00 | 1627.25 | 1655.56 | 1656.83 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 14:15:00 | 1696.00 | 1647.34 | 1643.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 09:15:00 | 1698.60 | 1663.62 | 1652.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 15:15:00 | 1716.20 | 1727.48 | 1705.58 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 09:15:00 | 1761.90 | 1727.48 | 1705.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 1743.05 | 1772.13 | 1747.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-06 09:15:00 | 1743.05 | 1772.13 | 1747.44 | SL hit (close<ema400) qty=1.00 sl=1747.44 alert=retest1 |

### Cycle 20 — SELL (started 2023-09-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 13:15:00 | 1737.00 | 1750.56 | 1752.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 14:15:00 | 1723.95 | 1745.24 | 1749.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 09:15:00 | 1742.95 | 1727.16 | 1734.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 1742.95 | 1727.16 | 1734.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 1742.95 | 1727.16 | 1734.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 09:30:00 | 1696.55 | 1719.00 | 1727.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 15:15:00 | 1611.72 | 1661.17 | 1690.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-13 15:15:00 | 1638.00 | 1633.25 | 1657.61 | SL hit (close>ema200) qty=0.50 sl=1633.25 alert=retest2 |

### Cycle 21 — BUY (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 10:15:00 | 1607.70 | 1564.07 | 1563.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 11:15:00 | 1618.25 | 1574.90 | 1568.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 14:15:00 | 1620.65 | 1622.07 | 1605.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-28 14:45:00 | 1609.65 | 1622.07 | 1605.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 1624.60 | 1627.44 | 1616.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 15:00:00 | 1624.60 | 1627.44 | 1616.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 1644.70 | 1631.60 | 1620.33 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 13:15:00 | 1608.10 | 1621.58 | 1622.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 14:15:00 | 1602.25 | 1617.72 | 1620.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 1640.20 | 1619.52 | 1620.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 1640.20 | 1619.52 | 1620.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 1640.20 | 1619.52 | 1620.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 10:00:00 | 1640.20 | 1619.52 | 1620.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-10-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 10:15:00 | 1637.60 | 1623.14 | 1622.36 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-10-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 15:15:00 | 1608.00 | 1621.38 | 1622.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 1586.95 | 1614.51 | 1618.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 12:15:00 | 1594.10 | 1580.56 | 1591.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 12:15:00 | 1594.10 | 1580.56 | 1591.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 1594.10 | 1580.56 | 1591.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:30:00 | 1597.50 | 1580.56 | 1591.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 13:15:00 | 1610.30 | 1586.51 | 1593.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 14:00:00 | 1610.30 | 1586.51 | 1593.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 14:15:00 | 1616.95 | 1592.60 | 1595.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 15:15:00 | 1614.00 | 1592.60 | 1595.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 15:15:00 | 1614.00 | 1596.88 | 1597.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:15:00 | 1639.35 | 1596.88 | 1597.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 1653.15 | 1608.13 | 1602.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 10:15:00 | 1671.55 | 1620.82 | 1608.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 11:15:00 | 1732.00 | 1753.00 | 1736.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 11:15:00 | 1732.00 | 1753.00 | 1736.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 1732.00 | 1753.00 | 1736.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 12:00:00 | 1732.00 | 1753.00 | 1736.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 1756.75 | 1753.75 | 1738.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 14:15:00 | 1762.45 | 1754.36 | 1739.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 14:45:00 | 1769.15 | 1756.48 | 1742.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-19 09:15:00 | 1722.85 | 1751.43 | 1742.37 | SL hit (close<static) qty=1.00 sl=1731.60 alert=retest2 |

### Cycle 26 — SELL (started 2023-10-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 12:15:00 | 1708.90 | 1733.25 | 1735.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 14:15:00 | 1697.00 | 1722.23 | 1729.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 14:15:00 | 1705.00 | 1700.70 | 1712.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 14:15:00 | 1705.00 | 1700.70 | 1712.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 14:15:00 | 1705.00 | 1700.70 | 1712.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 15:00:00 | 1705.00 | 1700.70 | 1712.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 1727.95 | 1706.15 | 1714.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 09:15:00 | 1683.00 | 1706.15 | 1714.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 1598.85 | 1625.45 | 1648.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 1625.25 | 1609.15 | 1626.37 | SL hit (close>ema200) qty=0.50 sl=1609.15 alert=retest2 |

### Cycle 27 — BUY (started 2023-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 11:15:00 | 1580.35 | 1555.91 | 1555.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 12:15:00 | 1585.00 | 1561.73 | 1558.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 09:15:00 | 1566.15 | 1573.96 | 1566.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 09:15:00 | 1566.15 | 1573.96 | 1566.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 1566.15 | 1573.96 | 1566.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 10:00:00 | 1566.15 | 1573.96 | 1566.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 1569.00 | 1572.96 | 1566.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 12:45:00 | 1576.00 | 1572.13 | 1567.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 1586.00 | 1569.82 | 1567.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-20 11:15:00 | 1674.35 | 1675.60 | 1675.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 11:15:00 | 1674.35 | 1675.60 | 1675.71 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 14:15:00 | 1681.65 | 1675.57 | 1675.53 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 13:15:00 | 1664.00 | 1674.90 | 1675.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 14:15:00 | 1651.20 | 1670.16 | 1673.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 09:15:00 | 1668.50 | 1666.81 | 1671.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 1668.50 | 1666.81 | 1671.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 1668.50 | 1666.81 | 1671.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:45:00 | 1666.15 | 1666.81 | 1671.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 1672.70 | 1667.99 | 1671.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 10:30:00 | 1670.15 | 1667.99 | 1671.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 11:15:00 | 1662.60 | 1666.91 | 1670.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 13:00:00 | 1652.95 | 1664.12 | 1668.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 15:00:00 | 1656.60 | 1660.80 | 1666.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 10:15:00 | 1682.60 | 1666.60 | 1667.76 | SL hit (close>static) qty=1.00 sl=1675.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 11:15:00 | 1676.95 | 1668.67 | 1668.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 14:15:00 | 1685.45 | 1675.38 | 1671.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 15:15:00 | 1693.00 | 1694.08 | 1686.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-28 09:15:00 | 1691.20 | 1694.08 | 1686.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 1667.00 | 1688.66 | 1684.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:00:00 | 1667.00 | 1688.66 | 1684.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 1667.60 | 1684.45 | 1682.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:30:00 | 1668.05 | 1684.45 | 1682.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 11:15:00 | 1661.60 | 1679.88 | 1680.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 12:15:00 | 1658.50 | 1675.61 | 1678.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 10:15:00 | 1667.85 | 1663.91 | 1671.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-29 11:00:00 | 1667.85 | 1663.91 | 1671.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 1663.50 | 1663.83 | 1670.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 12:45:00 | 1658.20 | 1662.90 | 1669.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 13:15:00 | 1654.85 | 1662.90 | 1669.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 10:15:00 | 1688.75 | 1667.17 | 1668.28 | SL hit (close>static) qty=1.00 sl=1679.95 alert=retest2 |

### Cycle 33 — BUY (started 2023-11-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 11:15:00 | 1701.95 | 1674.13 | 1671.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 15:15:00 | 1727.00 | 1689.32 | 1679.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 09:15:00 | 1678.90 | 1687.24 | 1679.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 09:15:00 | 1678.90 | 1687.24 | 1679.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 1678.90 | 1687.24 | 1679.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 10:00:00 | 1678.90 | 1687.24 | 1679.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 10:15:00 | 1674.15 | 1684.62 | 1679.16 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-01 13:15:00 | 1662.45 | 1674.67 | 1675.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-01 14:15:00 | 1641.95 | 1668.13 | 1672.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 14:15:00 | 1599.10 | 1595.78 | 1611.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-06 15:00:00 | 1599.10 | 1595.78 | 1611.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 1599.50 | 1596.55 | 1608.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-07 13:15:00 | 1595.95 | 1597.43 | 1606.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-07 14:00:00 | 1594.30 | 1596.81 | 1605.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 12:45:00 | 1595.90 | 1596.70 | 1601.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 14:15:00 | 1591.30 | 1597.37 | 1601.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 14:15:00 | 1602.55 | 1598.40 | 1601.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 15:00:00 | 1602.55 | 1598.40 | 1601.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 15:15:00 | 1599.80 | 1598.68 | 1601.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 09:15:00 | 1605.00 | 1598.68 | 1601.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 1599.75 | 1598.90 | 1601.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-12-11 14:15:00 | 1632.80 | 1606.47 | 1603.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2023-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 14:15:00 | 1632.80 | 1606.47 | 1603.81 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 10:15:00 | 1615.95 | 1621.89 | 1622.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 14:15:00 | 1607.15 | 1617.17 | 1619.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 09:15:00 | 1620.00 | 1616.49 | 1618.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 1620.00 | 1616.49 | 1618.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 1620.00 | 1616.49 | 1618.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-19 12:00:00 | 1605.50 | 1615.47 | 1618.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-19 12:30:00 | 1604.80 | 1612.02 | 1616.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 10:15:00 | 1595.20 | 1589.70 | 1589.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 1595.20 | 1589.70 | 1589.60 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 14:15:00 | 1579.15 | 1589.19 | 1590.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 09:15:00 | 1577.70 | 1586.14 | 1588.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-01 09:15:00 | 1549.00 | 1540.65 | 1553.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 09:15:00 | 1549.00 | 1540.65 | 1553.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 1549.00 | 1540.65 | 1553.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-01 12:15:00 | 1541.00 | 1543.63 | 1552.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-01 13:15:00 | 1541.85 | 1543.31 | 1551.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-01 13:45:00 | 1538.20 | 1540.96 | 1549.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 10:15:00 | 1463.95 | 1479.07 | 1496.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 10:15:00 | 1464.76 | 1479.07 | 1496.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 10:15:00 | 1461.29 | 1479.07 | 1496.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-05 09:15:00 | 1505.05 | 1467.54 | 1480.32 | SL hit (close>ema200) qty=0.50 sl=1467.54 alert=retest2 |

### Cycle 39 — BUY (started 2024-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 11:15:00 | 1543.95 | 1489.74 | 1488.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 12:15:00 | 1546.50 | 1501.09 | 1493.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 11:15:00 | 1510.75 | 1515.15 | 1505.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 11:15:00 | 1510.75 | 1515.15 | 1505.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 1510.75 | 1515.15 | 1505.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:45:00 | 1504.00 | 1515.15 | 1505.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 1501.25 | 1512.37 | 1505.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 13:00:00 | 1501.25 | 1512.37 | 1505.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 13:15:00 | 1504.65 | 1510.82 | 1505.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 13:45:00 | 1497.55 | 1510.82 | 1505.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 1513.20 | 1511.30 | 1505.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:45:00 | 1519.10 | 1515.87 | 1508.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 12:15:00 | 1551.35 | 1558.69 | 1558.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 12:15:00 | 1551.35 | 1558.69 | 1558.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 13:15:00 | 1539.00 | 1554.75 | 1556.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 14:15:00 | 1559.80 | 1555.76 | 1557.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 14:15:00 | 1559.80 | 1555.76 | 1557.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 14:15:00 | 1559.80 | 1555.76 | 1557.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 15:00:00 | 1559.80 | 1555.76 | 1557.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 15:15:00 | 1552.00 | 1555.01 | 1556.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:15:00 | 1544.25 | 1555.01 | 1556.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 1546.70 | 1553.35 | 1555.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 09:15:00 | 1525.05 | 1553.04 | 1554.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-19 14:15:00 | 1547.70 | 1528.10 | 1526.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 14:15:00 | 1547.70 | 1528.10 | 1526.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 15:15:00 | 1551.50 | 1532.78 | 1528.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 11:15:00 | 1536.95 | 1539.21 | 1532.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 11:45:00 | 1538.00 | 1539.21 | 1532.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 1538.50 | 1539.96 | 1535.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 1529.35 | 1539.96 | 1535.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 1528.70 | 1537.71 | 1534.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 1528.70 | 1537.71 | 1534.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 1519.70 | 1534.11 | 1533.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:45:00 | 1522.00 | 1534.11 | 1533.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 1490.80 | 1525.45 | 1529.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 12:15:00 | 1482.00 | 1516.76 | 1525.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 14:15:00 | 1504.50 | 1503.51 | 1511.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 15:00:00 | 1504.50 | 1503.51 | 1511.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 1488.80 | 1500.81 | 1508.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 11:30:00 | 1486.00 | 1496.04 | 1505.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 12:45:00 | 1486.05 | 1494.14 | 1503.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 10:00:00 | 1484.15 | 1491.29 | 1499.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 14:15:00 | 1533.90 | 1504.78 | 1503.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 14:15:00 | 1533.90 | 1504.78 | 1503.09 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 14:15:00 | 1498.90 | 1504.35 | 1504.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 15:15:00 | 1491.00 | 1501.68 | 1503.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 14:15:00 | 1502.15 | 1488.74 | 1493.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 14:15:00 | 1502.15 | 1488.74 | 1493.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 1502.15 | 1488.74 | 1493.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 15:00:00 | 1502.15 | 1488.74 | 1493.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 15:15:00 | 1512.00 | 1493.39 | 1495.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 09:30:00 | 1500.00 | 1494.45 | 1495.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 11:00:00 | 1500.00 | 1495.56 | 1496.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 14:15:00 | 1425.00 | 1458.28 | 1471.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 14:15:00 | 1425.00 | 1458.28 | 1471.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-08 09:15:00 | 1400.75 | 1393.62 | 1411.10 | SL hit (close>ema200) qty=0.50 sl=1393.62 alert=retest2 |

### Cycle 45 — BUY (started 2024-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 13:15:00 | 1373.40 | 1356.00 | 1354.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 1386.95 | 1362.19 | 1357.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 14:15:00 | 1426.50 | 1427.20 | 1408.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 15:00:00 | 1426.50 | 1427.20 | 1408.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 1418.60 | 1428.80 | 1421.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:45:00 | 1419.50 | 1428.80 | 1421.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 11:15:00 | 1409.45 | 1424.93 | 1420.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:45:00 | 1405.50 | 1424.93 | 1420.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 1417.00 | 1421.75 | 1419.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:00:00 | 1417.00 | 1421.75 | 1419.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 13:15:00 | 1427.85 | 1434.24 | 1428.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:00:00 | 1427.85 | 1434.24 | 1428.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 1433.00 | 1433.99 | 1429.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:45:00 | 1424.80 | 1433.99 | 1429.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 1424.10 | 1432.34 | 1429.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:00:00 | 1424.10 | 1432.34 | 1429.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 1424.25 | 1430.72 | 1428.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:30:00 | 1425.00 | 1430.72 | 1428.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 1438.45 | 1431.30 | 1429.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:45:00 | 1425.60 | 1431.30 | 1429.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 1423.70 | 1429.78 | 1428.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:00:00 | 1423.70 | 1429.78 | 1428.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 1422.65 | 1428.36 | 1428.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 1422.65 | 1428.36 | 1428.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 15:15:00 | 1418.95 | 1426.47 | 1427.43 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 09:15:00 | 1437.65 | 1428.71 | 1428.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 10:15:00 | 1447.55 | 1432.48 | 1430.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 10:15:00 | 1450.35 | 1451.17 | 1442.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-26 10:45:00 | 1453.05 | 1451.17 | 1442.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 1465.80 | 1468.38 | 1460.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:30:00 | 1461.90 | 1468.38 | 1460.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 1452.75 | 1464.65 | 1459.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 12:00:00 | 1452.75 | 1464.65 | 1459.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 1444.85 | 1460.69 | 1458.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 13:00:00 | 1444.85 | 1460.69 | 1458.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 13:15:00 | 1460.80 | 1460.71 | 1458.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 14:15:00 | 1465.50 | 1460.71 | 1458.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 09:15:00 | 1466.45 | 1460.73 | 1459.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 09:45:00 | 1465.95 | 1461.38 | 1459.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 10:15:00 | 1468.70 | 1461.38 | 1459.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 1548.90 | 1539.24 | 1532.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 15:00:00 | 1556.20 | 1543.17 | 1537.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 09:30:00 | 1565.15 | 1551.50 | 1542.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-03-13 09:15:00 | 1612.05 | 1596.62 | 1587.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 14:15:00 | 1484.65 | 1566.75 | 1577.37 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 15:15:00 | 1549.90 | 1519.33 | 1515.25 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 14:15:00 | 1502.70 | 1523.29 | 1525.11 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 12:15:00 | 1529.85 | 1518.14 | 1517.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 13:15:00 | 1532.60 | 1521.03 | 1519.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 1519.20 | 1520.67 | 1519.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 14:15:00 | 1519.20 | 1520.67 | 1519.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 14:15:00 | 1519.20 | 1520.67 | 1519.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 15:00:00 | 1519.20 | 1520.67 | 1519.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 1515.95 | 1519.72 | 1518.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:15:00 | 1505.90 | 1519.72 | 1518.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 09:15:00 | 1492.15 | 1514.21 | 1516.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 13:15:00 | 1488.20 | 1501.15 | 1508.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 12:15:00 | 1485.35 | 1477.10 | 1486.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 12:15:00 | 1485.35 | 1477.10 | 1486.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 12:15:00 | 1485.35 | 1477.10 | 1486.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 12:45:00 | 1483.00 | 1477.10 | 1486.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 13:15:00 | 1496.35 | 1480.95 | 1487.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 14:00:00 | 1496.35 | 1480.95 | 1487.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 1499.60 | 1484.68 | 1488.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 14:30:00 | 1501.65 | 1484.68 | 1488.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 1500.60 | 1488.02 | 1489.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 09:30:00 | 1508.00 | 1488.02 | 1489.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 11:15:00 | 1495.15 | 1489.87 | 1489.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 12:15:00 | 1504.00 | 1492.70 | 1491.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-01 13:15:00 | 1486.50 | 1491.46 | 1490.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 13:15:00 | 1486.50 | 1491.46 | 1490.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 13:15:00 | 1486.50 | 1491.46 | 1490.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-01 13:45:00 | 1486.80 | 1491.46 | 1490.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 14:15:00 | 1507.70 | 1494.71 | 1492.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 09:15:00 | 1536.00 | 1496.18 | 1493.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-05 14:15:00 | 1548.55 | 1563.05 | 1563.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 14:15:00 | 1548.55 | 1563.05 | 1563.72 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 1591.70 | 1567.49 | 1565.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 11:15:00 | 1603.65 | 1576.41 | 1570.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 13:15:00 | 1588.05 | 1590.60 | 1583.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 13:15:00 | 1588.05 | 1590.60 | 1583.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 1588.05 | 1590.60 | 1583.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 14:00:00 | 1588.05 | 1590.60 | 1583.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 1588.55 | 1592.90 | 1586.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:15:00 | 1579.35 | 1592.90 | 1586.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 1565.00 | 1587.32 | 1584.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 11:00:00 | 1565.00 | 1587.32 | 1584.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 1568.20 | 1583.50 | 1583.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 12:15:00 | 1571.00 | 1583.50 | 1583.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-10 12:15:00 | 1572.50 | 1581.30 | 1582.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 12:15:00 | 1572.50 | 1581.30 | 1582.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 11:15:00 | 1566.30 | 1573.41 | 1577.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 1556.15 | 1549.26 | 1556.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 1556.15 | 1549.26 | 1556.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1556.15 | 1549.26 | 1556.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:45:00 | 1556.70 | 1549.26 | 1556.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 1557.45 | 1550.90 | 1556.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:30:00 | 1563.00 | 1550.90 | 1556.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 1554.75 | 1551.67 | 1556.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:30:00 | 1558.05 | 1551.67 | 1556.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 1548.95 | 1547.86 | 1552.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 1542.60 | 1548.95 | 1552.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 11:15:00 | 1544.40 | 1545.08 | 1545.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 13:15:00 | 1556.45 | 1547.87 | 1546.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 1556.45 | 1547.87 | 1546.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 11:15:00 | 1562.00 | 1553.43 | 1550.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 09:15:00 | 1568.50 | 1575.09 | 1567.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 1568.50 | 1575.09 | 1567.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 1568.50 | 1575.09 | 1567.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:45:00 | 1566.00 | 1575.09 | 1567.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 1563.00 | 1572.67 | 1567.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 11:00:00 | 1563.00 | 1572.67 | 1567.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 1558.75 | 1569.89 | 1566.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 13:45:00 | 1566.70 | 1565.06 | 1564.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 14:15:00 | 1546.30 | 1561.31 | 1563.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 14:15:00 | 1546.30 | 1561.31 | 1563.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 09:15:00 | 1542.95 | 1552.00 | 1556.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 11:15:00 | 1553.10 | 1551.41 | 1555.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-29 11:15:00 | 1553.10 | 1551.41 | 1555.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 11:15:00 | 1553.10 | 1551.41 | 1555.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 11:45:00 | 1558.30 | 1551.41 | 1555.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 14:15:00 | 1558.75 | 1550.67 | 1553.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 15:00:00 | 1558.75 | 1550.67 | 1553.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 1550.70 | 1550.67 | 1553.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:15:00 | 1563.00 | 1550.67 | 1553.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 1576.95 | 1555.93 | 1555.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 12:15:00 | 1584.40 | 1572.87 | 1566.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 11:15:00 | 1576.25 | 1585.59 | 1577.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 11:15:00 | 1576.25 | 1585.59 | 1577.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 1576.25 | 1585.59 | 1577.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:00:00 | 1576.25 | 1585.59 | 1577.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 1578.60 | 1584.19 | 1577.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 14:30:00 | 1587.80 | 1584.59 | 1578.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 15:00:00 | 1589.70 | 1584.59 | 1578.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 09:15:00 | 1600.65 | 1584.04 | 1578.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 09:45:00 | 1592.75 | 1582.41 | 1578.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 1570.10 | 1579.94 | 1577.89 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-06 10:15:00 | 1570.10 | 1579.94 | 1577.89 | SL hit (close<static) qty=1.00 sl=1574.80 alert=retest2 |

### Cycle 60 — SELL (started 2024-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 11:15:00 | 1558.10 | 1575.58 | 1576.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 12:15:00 | 1550.50 | 1570.56 | 1573.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 10:15:00 | 1545.50 | 1545.42 | 1553.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 11:15:00 | 1550.50 | 1545.42 | 1553.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 1546.50 | 1545.64 | 1553.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:30:00 | 1552.25 | 1545.64 | 1553.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 1547.20 | 1545.95 | 1552.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:30:00 | 1553.85 | 1545.95 | 1552.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 1544.65 | 1545.69 | 1551.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 14:00:00 | 1544.65 | 1545.69 | 1551.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 1561.65 | 1548.88 | 1552.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 15:00:00 | 1561.65 | 1548.88 | 1552.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 15:15:00 | 1561.35 | 1551.38 | 1553.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 1551.05 | 1551.38 | 1553.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 09:15:00 | 1473.50 | 1531.23 | 1538.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-13 11:15:00 | 1530.00 | 1527.44 | 1535.67 | SL hit (close>ema200) qty=0.50 sl=1527.44 alert=retest2 |

### Cycle 61 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 1550.50 | 1540.78 | 1539.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 1580.40 | 1563.11 | 1552.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 13:15:00 | 1576.00 | 1580.40 | 1568.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 13:15:00 | 1576.00 | 1580.40 | 1568.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 1576.00 | 1580.40 | 1568.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:45:00 | 1579.25 | 1580.40 | 1568.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 1580.05 | 1580.33 | 1569.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:30:00 | 1569.00 | 1580.33 | 1569.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 1708.15 | 1710.35 | 1694.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 1708.15 | 1710.35 | 1694.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1701.00 | 1708.75 | 1696.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:15:00 | 1693.80 | 1708.75 | 1696.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 1692.90 | 1705.58 | 1695.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 1687.50 | 1705.58 | 1695.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 1698.00 | 1704.06 | 1696.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 12:30:00 | 1699.50 | 1702.91 | 1696.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 13:45:00 | 1699.60 | 1701.20 | 1696.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 14:45:00 | 1698.70 | 1700.76 | 1696.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 09:15:00 | 1713.30 | 1699.41 | 1696.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1704.65 | 1700.45 | 1696.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 10:30:00 | 1722.85 | 1702.15 | 1698.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 12:15:00 | 1687.30 | 1697.60 | 1696.60 | SL hit (close<static) qty=1.00 sl=1690.60 alert=retest2 |

### Cycle 62 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 1685.30 | 1694.31 | 1695.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 12:15:00 | 1680.40 | 1691.52 | 1693.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 1540.15 | 1524.05 | 1557.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 1540.15 | 1524.05 | 1557.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1533.60 | 1517.93 | 1545.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 1516.60 | 1549.96 | 1552.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:45:00 | 1521.95 | 1535.71 | 1545.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 1440.77 | 1518.06 | 1536.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 1445.85 | 1518.06 | 1536.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 1479.15 | 1478.98 | 1507.44 | SL hit (close>ema200) qty=0.50 sl=1478.98 alert=retest2 |

### Cycle 63 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 1581.50 | 1505.88 | 1504.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 1588.70 | 1560.11 | 1537.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 1629.30 | 1629.80 | 1598.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 1623.30 | 1629.80 | 1598.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 1614.50 | 1630.12 | 1618.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:30:00 | 1617.70 | 1630.12 | 1618.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 1603.50 | 1624.79 | 1616.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:00:00 | 1603.50 | 1624.79 | 1616.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 1594.50 | 1618.73 | 1614.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:00:00 | 1594.50 | 1618.73 | 1614.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2024-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 13:15:00 | 1585.50 | 1607.01 | 1609.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 14:15:00 | 1582.50 | 1602.11 | 1607.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 10:15:00 | 1562.50 | 1558.30 | 1575.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-14 10:30:00 | 1556.35 | 1558.30 | 1575.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1568.55 | 1559.74 | 1568.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 1568.55 | 1559.74 | 1568.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 1578.95 | 1563.58 | 1569.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:00:00 | 1578.95 | 1563.58 | 1569.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 1584.75 | 1567.81 | 1570.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:00:00 | 1584.75 | 1567.81 | 1570.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 13:15:00 | 1582.05 | 1572.38 | 1572.32 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 12:15:00 | 1568.40 | 1572.69 | 1572.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 09:15:00 | 1564.75 | 1569.84 | 1571.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 12:15:00 | 1581.05 | 1570.36 | 1571.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 12:15:00 | 1581.05 | 1570.36 | 1571.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 1581.05 | 1570.36 | 1571.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 13:00:00 | 1581.05 | 1570.36 | 1571.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 13:15:00 | 1582.95 | 1572.88 | 1572.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 14:15:00 | 1588.40 | 1575.98 | 1573.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 1584.85 | 1588.02 | 1581.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 12:15:00 | 1584.85 | 1588.02 | 1581.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 1584.85 | 1588.02 | 1581.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 1584.85 | 1588.02 | 1581.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 1647.45 | 1599.72 | 1587.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 1654.00 | 1599.72 | 1587.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 12:15:00 | 1601.60 | 1609.25 | 1609.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 12:15:00 | 1601.60 | 1609.25 | 1609.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 1592.65 | 1604.76 | 1607.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 14:15:00 | 1551.25 | 1548.02 | 1567.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 15:00:00 | 1551.25 | 1548.02 | 1567.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1621.00 | 1564.53 | 1572.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:00:00 | 1621.00 | 1564.53 | 1572.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 1628.45 | 1577.32 | 1577.20 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 11:15:00 | 1576.70 | 1591.98 | 1592.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 12:15:00 | 1574.80 | 1588.55 | 1590.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 14:15:00 | 1597.00 | 1587.47 | 1589.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 14:15:00 | 1597.00 | 1587.47 | 1589.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 1597.00 | 1587.47 | 1589.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 1597.00 | 1587.47 | 1589.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 1599.60 | 1589.90 | 1590.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 1588.00 | 1589.90 | 1590.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 11:15:00 | 1569.00 | 1560.67 | 1559.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 11:15:00 | 1569.00 | 1560.67 | 1559.83 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 11:15:00 | 1549.10 | 1559.14 | 1559.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 13:15:00 | 1546.40 | 1555.11 | 1557.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 10:15:00 | 1565.00 | 1556.38 | 1557.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 10:15:00 | 1565.00 | 1556.38 | 1557.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 1565.00 | 1556.38 | 1557.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:00:00 | 1565.00 | 1556.38 | 1557.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 1539.95 | 1553.09 | 1555.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:45:00 | 1537.50 | 1549.77 | 1554.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 14:45:00 | 1536.50 | 1546.15 | 1551.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 12:45:00 | 1535.95 | 1548.83 | 1551.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 15:15:00 | 1553.00 | 1551.07 | 1551.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 15:15:00 | 1553.00 | 1551.07 | 1551.04 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 1546.80 | 1550.22 | 1550.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 13:15:00 | 1523.50 | 1539.27 | 1545.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 11:15:00 | 1484.45 | 1483.50 | 1496.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 12:00:00 | 1484.45 | 1483.50 | 1496.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1495.20 | 1484.97 | 1493.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 1495.20 | 1484.97 | 1493.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 1497.75 | 1487.52 | 1493.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 1487.55 | 1487.52 | 1493.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1485.95 | 1487.21 | 1493.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:15:00 | 1476.95 | 1486.35 | 1492.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:15:00 | 1472.05 | 1488.08 | 1490.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 09:15:00 | 1487.45 | 1483.09 | 1482.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 1487.45 | 1483.09 | 1482.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 1502.90 | 1487.05 | 1484.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 1495.50 | 1498.97 | 1492.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 1495.50 | 1498.97 | 1492.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1495.50 | 1498.97 | 1492.86 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 13:15:00 | 1474.30 | 1488.04 | 1489.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 1424.85 | 1464.20 | 1472.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1454.15 | 1434.60 | 1448.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1454.15 | 1434.60 | 1448.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1454.15 | 1434.60 | 1448.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:00:00 | 1411.40 | 1432.54 | 1443.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:15:00 | 1419.00 | 1427.34 | 1437.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:30:00 | 1418.00 | 1418.03 | 1427.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 1443.35 | 1426.44 | 1426.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 1443.35 | 1426.44 | 1426.41 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 10:15:00 | 1421.95 | 1429.62 | 1430.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 11:15:00 | 1410.10 | 1425.72 | 1428.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 11:15:00 | 1419.00 | 1415.38 | 1420.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 12:00:00 | 1419.00 | 1415.38 | 1420.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 1417.55 | 1415.82 | 1420.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:45:00 | 1423.40 | 1415.82 | 1420.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 1422.15 | 1417.08 | 1420.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:00:00 | 1422.15 | 1417.08 | 1420.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 1422.00 | 1418.07 | 1420.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:30:00 | 1421.00 | 1418.07 | 1420.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 1415.85 | 1417.62 | 1420.02 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 1433.70 | 1423.16 | 1422.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 1434.05 | 1425.34 | 1423.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 14:15:00 | 1419.10 | 1425.65 | 1423.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 14:15:00 | 1419.10 | 1425.65 | 1423.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 1419.10 | 1425.65 | 1423.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 1419.10 | 1425.65 | 1423.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 15:15:00 | 1403.95 | 1421.31 | 1421.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 09:15:00 | 1390.25 | 1415.10 | 1419.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 13:15:00 | 1352.50 | 1352.08 | 1365.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-21 14:00:00 | 1352.50 | 1352.08 | 1365.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 1347.05 | 1353.60 | 1362.96 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 13:15:00 | 1401.00 | 1368.85 | 1367.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 14:15:00 | 1418.85 | 1378.85 | 1372.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 12:15:00 | 1416.70 | 1417.20 | 1404.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 13:00:00 | 1416.70 | 1417.20 | 1404.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 1453.90 | 1483.75 | 1475.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 1453.90 | 1483.75 | 1475.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 1462.00 | 1479.40 | 1474.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 1429.25 | 1479.40 | 1474.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 09:15:00 | 1432.70 | 1470.06 | 1470.30 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 12:15:00 | 1458.15 | 1447.67 | 1447.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 10:15:00 | 1470.30 | 1454.32 | 1451.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 14:15:00 | 1438.75 | 1454.98 | 1453.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 14:15:00 | 1438.75 | 1454.98 | 1453.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 1438.75 | 1454.98 | 1453.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 1438.75 | 1454.98 | 1453.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 1443.05 | 1452.59 | 1452.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 1458.25 | 1452.59 | 1452.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 1439.70 | 1450.01 | 1450.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 1439.70 | 1450.01 | 1450.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 10:15:00 | 1434.15 | 1446.84 | 1449.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1418.05 | 1407.83 | 1418.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1418.05 | 1407.83 | 1418.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1418.05 | 1407.83 | 1418.84 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 1444.90 | 1422.08 | 1421.13 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 13:15:00 | 1419.50 | 1424.89 | 1424.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 14:15:00 | 1415.05 | 1422.92 | 1424.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 15:15:00 | 1430.70 | 1424.48 | 1424.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 15:15:00 | 1430.70 | 1424.48 | 1424.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 1430.70 | 1424.48 | 1424.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 1430.80 | 1424.48 | 1424.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 1439.90 | 1427.56 | 1426.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 10:15:00 | 1451.25 | 1432.30 | 1428.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 10:15:00 | 1450.25 | 1453.00 | 1443.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 11:00:00 | 1450.25 | 1453.00 | 1443.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 1446.05 | 1451.61 | 1443.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 1446.05 | 1451.61 | 1443.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 1446.20 | 1450.53 | 1444.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 13:15:00 | 1456.40 | 1450.53 | 1444.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 15:00:00 | 1453.55 | 1451.58 | 1445.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 1437.25 | 1449.26 | 1445.75 | SL hit (close<static) qty=1.00 sl=1442.55 alert=retest2 |

### Cycle 88 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 1432.50 | 1444.26 | 1444.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 12:15:00 | 1428.45 | 1435.91 | 1439.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1440.50 | 1431.84 | 1435.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 10:15:00 | 1440.50 | 1431.84 | 1435.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1440.50 | 1431.84 | 1435.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 1440.50 | 1431.84 | 1435.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1439.50 | 1433.37 | 1435.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:30:00 | 1438.95 | 1433.37 | 1435.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 1439.80 | 1434.65 | 1436.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:00:00 | 1439.80 | 1434.65 | 1436.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 1431.70 | 1434.06 | 1435.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:30:00 | 1426.50 | 1435.65 | 1436.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 09:45:00 | 1427.65 | 1434.06 | 1435.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 10:30:00 | 1427.20 | 1432.50 | 1434.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 11:00:00 | 1426.25 | 1432.50 | 1434.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 1425.00 | 1424.33 | 1428.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:15:00 | 1434.30 | 1424.33 | 1428.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1427.70 | 1425.00 | 1428.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-24 12:15:00 | 1459.80 | 1434.66 | 1432.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 12:15:00 | 1459.80 | 1434.66 | 1432.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 13:15:00 | 1469.15 | 1441.56 | 1435.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 15:15:00 | 1600.00 | 1608.52 | 1591.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 09:15:00 | 1578.20 | 1608.52 | 1591.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1614.05 | 1609.63 | 1593.65 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 1545.00 | 1594.96 | 1595.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 1540.70 | 1584.10 | 1590.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 1552.85 | 1549.19 | 1560.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 09:15:00 | 1552.85 | 1549.19 | 1560.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1552.85 | 1549.19 | 1560.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:45:00 | 1538.65 | 1551.93 | 1558.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 10:15:00 | 1543.05 | 1549.58 | 1556.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:00:00 | 1544.20 | 1548.50 | 1555.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 14:15:00 | 1574.90 | 1555.55 | 1556.27 | SL hit (close>static) qty=1.00 sl=1573.70 alert=retest2 |

### Cycle 91 — BUY (started 2024-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 15:15:00 | 1568.00 | 1558.04 | 1557.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 09:15:00 | 1575.20 | 1561.47 | 1558.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 1559.10 | 1561.00 | 1558.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 10:45:00 | 1560.05 | 1561.00 | 1558.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 1565.45 | 1561.89 | 1559.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:00:00 | 1569.15 | 1564.06 | 1560.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 1540.30 | 1561.80 | 1563.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 09:15:00 | 1540.30 | 1561.80 | 1563.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 10:15:00 | 1536.20 | 1556.68 | 1561.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 14:15:00 | 1552.80 | 1549.94 | 1556.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 14:15:00 | 1552.80 | 1549.94 | 1556.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 1552.80 | 1549.94 | 1556.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 1552.80 | 1549.94 | 1556.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1577.05 | 1555.69 | 1557.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 1581.00 | 1555.69 | 1557.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 10:15:00 | 1578.60 | 1560.27 | 1559.56 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 1564.90 | 1570.43 | 1570.87 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 1580.50 | 1571.18 | 1570.75 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 1553.35 | 1567.38 | 1569.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 1539.75 | 1558.88 | 1564.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 14:15:00 | 1538.40 | 1530.92 | 1542.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-22 15:00:00 | 1538.40 | 1530.92 | 1542.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 1525.00 | 1529.73 | 1540.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 09:15:00 | 1518.90 | 1529.73 | 1540.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 10:00:00 | 1520.05 | 1527.80 | 1538.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 13:45:00 | 1518.00 | 1517.48 | 1529.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 1442.95 | 1481.61 | 1499.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 1444.05 | 1481.61 | 1499.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 1442.10 | 1481.61 | 1499.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 14:15:00 | 1477.45 | 1475.16 | 1490.18 | SL hit (close>ema200) qty=0.50 sl=1475.16 alert=retest2 |

### Cycle 97 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 1484.70 | 1468.40 | 1466.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 1523.65 | 1490.79 | 1480.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1523.30 | 1523.68 | 1508.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 1523.30 | 1523.68 | 1508.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1509.80 | 1520.91 | 1508.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 1509.80 | 1520.91 | 1508.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1509.55 | 1518.64 | 1508.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:30:00 | 1508.45 | 1518.64 | 1508.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 1505.70 | 1516.05 | 1508.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:00:00 | 1515.00 | 1514.33 | 1508.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 1536.60 | 1519.56 | 1516.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 10:15:00 | 1511.95 | 1525.66 | 1527.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 1511.95 | 1525.66 | 1527.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 1490.05 | 1518.53 | 1524.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 14:15:00 | 1519.35 | 1512.07 | 1519.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 14:15:00 | 1519.35 | 1512.07 | 1519.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 1519.35 | 1512.07 | 1519.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:30:00 | 1524.80 | 1512.07 | 1519.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 1510.00 | 1511.65 | 1518.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 1534.45 | 1511.65 | 1518.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 1519.75 | 1513.27 | 1518.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:30:00 | 1535.10 | 1513.27 | 1518.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 1526.60 | 1515.94 | 1519.14 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 14:15:00 | 1524.90 | 1521.54 | 1521.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 09:15:00 | 1535.00 | 1524.62 | 1522.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 11:15:00 | 1520.20 | 1525.10 | 1523.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 11:15:00 | 1520.20 | 1525.10 | 1523.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 1520.20 | 1525.10 | 1523.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:00:00 | 1520.20 | 1525.10 | 1523.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 12:15:00 | 1510.00 | 1522.08 | 1522.10 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 15:15:00 | 1530.00 | 1522.70 | 1522.24 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 1503.70 | 1518.90 | 1520.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 15:15:00 | 1495.00 | 1504.28 | 1511.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1511.45 | 1505.71 | 1511.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1511.45 | 1505.71 | 1511.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1511.45 | 1505.71 | 1511.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:30:00 | 1479.00 | 1496.48 | 1505.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:15:00 | 1487.50 | 1480.54 | 1485.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 09:15:00 | 1405.05 | 1441.62 | 1457.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 09:15:00 | 1413.12 | 1441.62 | 1457.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 1458.95 | 1432.63 | 1442.98 | SL hit (close>ema200) qty=0.50 sl=1432.63 alert=retest2 |

### Cycle 103 — BUY (started 2024-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 13:15:00 | 1474.85 | 1453.12 | 1450.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 15:15:00 | 1480.00 | 1459.62 | 1453.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 14:15:00 | 1479.50 | 1483.94 | 1471.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 15:00:00 | 1479.50 | 1483.94 | 1471.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1474.70 | 1481.65 | 1472.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 1482.80 | 1481.65 | 1472.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 1473.40 | 1480.00 | 1472.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 1473.00 | 1480.00 | 1472.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1472.75 | 1480.45 | 1476.21 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 1459.75 | 1471.96 | 1472.99 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 09:15:00 | 1498.20 | 1475.80 | 1474.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 10:15:00 | 1517.70 | 1484.18 | 1478.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 1507.25 | 1508.95 | 1496.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 1507.25 | 1508.95 | 1496.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1507.25 | 1508.95 | 1496.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 1492.75 | 1508.95 | 1496.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 1494.75 | 1506.11 | 1496.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:45:00 | 1494.70 | 1506.11 | 1496.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 1497.80 | 1504.45 | 1496.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 12:15:00 | 1494.95 | 1504.45 | 1496.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 1499.10 | 1503.38 | 1496.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 13:45:00 | 1500.85 | 1502.62 | 1496.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 14:30:00 | 1500.80 | 1503.34 | 1497.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 1589.55 | 1603.27 | 1603.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 1589.55 | 1603.27 | 1603.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 10:15:00 | 1581.40 | 1598.90 | 1601.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 1599.20 | 1596.67 | 1599.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 13:15:00 | 1599.20 | 1596.67 | 1599.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 1599.20 | 1596.67 | 1599.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 1599.20 | 1596.67 | 1599.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1596.90 | 1596.71 | 1599.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 1596.90 | 1596.71 | 1599.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 1595.65 | 1596.50 | 1599.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 1587.95 | 1596.50 | 1599.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1593.10 | 1595.82 | 1598.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:45:00 | 1573.45 | 1589.02 | 1593.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 10:45:00 | 1575.55 | 1586.53 | 1591.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 11:30:00 | 1576.35 | 1585.22 | 1590.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 12:15:00 | 1576.25 | 1585.22 | 1590.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 1584.10 | 1572.84 | 1578.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:45:00 | 1581.80 | 1572.84 | 1578.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 1587.70 | 1575.81 | 1579.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 1585.05 | 1575.81 | 1579.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 1590.10 | 1579.90 | 1580.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:00:00 | 1590.10 | 1579.90 | 1580.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-20 11:15:00 | 1594.70 | 1582.86 | 1581.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 1594.70 | 1582.86 | 1581.95 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 1576.20 | 1581.43 | 1581.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 1561.90 | 1577.52 | 1579.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 15:15:00 | 1525.00 | 1503.26 | 1518.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 15:15:00 | 1525.00 | 1503.26 | 1518.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1525.00 | 1503.26 | 1518.53 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 1530.80 | 1521.78 | 1520.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 10:15:00 | 1554.80 | 1528.16 | 1524.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 15:15:00 | 1583.00 | 1583.27 | 1572.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:15:00 | 1591.65 | 1583.27 | 1572.15 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1591.95 | 1605.21 | 1593.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 1591.95 | 1605.21 | 1593.94 | SL hit (close<ema400) qty=1.00 sl=1593.94 alert=retest1 |

### Cycle 110 — SELL (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 11:15:00 | 1584.40 | 1592.04 | 1592.06 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 12:15:00 | 1592.45 | 1592.12 | 1592.10 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 1562.05 | 1586.21 | 1589.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 1550.80 | 1579.13 | 1585.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 13:15:00 | 1545.95 | 1541.21 | 1555.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 14:00:00 | 1545.95 | 1541.21 | 1555.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1476.35 | 1461.35 | 1484.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:45:00 | 1474.75 | 1461.35 | 1484.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1499.65 | 1471.77 | 1480.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:00:00 | 1499.65 | 1471.77 | 1480.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 1510.00 | 1479.42 | 1483.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 1509.60 | 1479.42 | 1483.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 12:15:00 | 1517.45 | 1491.48 | 1488.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 1520.85 | 1505.37 | 1496.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 14:15:00 | 1586.30 | 1586.45 | 1565.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 14:45:00 | 1594.30 | 1586.45 | 1565.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1583.05 | 1590.39 | 1572.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 1583.05 | 1590.39 | 1572.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 1585.60 | 1591.70 | 1579.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 1577.30 | 1591.70 | 1579.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 1586.50 | 1590.66 | 1579.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 1558.70 | 1590.66 | 1579.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1553.35 | 1583.20 | 1577.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 1561.40 | 1583.20 | 1577.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 1548.25 | 1576.21 | 1574.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:45:00 | 1549.10 | 1576.21 | 1574.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 1537.05 | 1568.38 | 1571.30 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 1595.95 | 1573.71 | 1571.29 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 1560.45 | 1572.04 | 1573.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1508.20 | 1557.36 | 1566.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 14:15:00 | 1494.05 | 1491.88 | 1515.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 14:30:00 | 1503.55 | 1491.88 | 1515.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1516.45 | 1496.49 | 1513.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1516.45 | 1496.49 | 1513.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1527.20 | 1502.63 | 1514.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 1527.20 | 1502.63 | 1514.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 1524.40 | 1506.99 | 1515.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:15:00 | 1529.30 | 1506.99 | 1515.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 1523.30 | 1510.25 | 1516.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:30:00 | 1521.05 | 1512.20 | 1516.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 14:45:00 | 1520.60 | 1512.96 | 1516.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:15:00 | 1520.75 | 1516.41 | 1517.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 1445.00 | 1489.36 | 1494.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 1444.57 | 1489.36 | 1494.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-03 09:15:00 | 1444.71 | 1489.36 | 1494.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-04 14:15:00 | 1435.90 | 1432.49 | 1450.94 | SL hit (close>ema200) qty=0.50 sl=1432.49 alert=retest2 |

### Cycle 117 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 1347.25 | 1330.66 | 1328.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 1372.00 | 1338.92 | 1332.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 11:15:00 | 1479.00 | 1479.27 | 1452.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 12:00:00 | 1479.00 | 1479.27 | 1452.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1470.35 | 1474.75 | 1460.23 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 13:15:00 | 1442.00 | 1456.64 | 1458.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 15:15:00 | 1436.00 | 1450.07 | 1455.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 1440.25 | 1427.45 | 1438.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 1440.25 | 1427.45 | 1438.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 1440.25 | 1427.45 | 1438.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 14:45:00 | 1444.65 | 1427.45 | 1438.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 1430.45 | 1428.05 | 1437.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 1408.25 | 1428.05 | 1437.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 14:30:00 | 1426.25 | 1419.77 | 1427.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 1426.15 | 1423.64 | 1426.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:00:00 | 1425.50 | 1423.64 | 1426.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 1427.05 | 1424.33 | 1426.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:30:00 | 1427.45 | 1424.33 | 1426.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 1428.10 | 1425.08 | 1426.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 1464.20 | 1432.97 | 1430.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 1464.20 | 1432.97 | 1430.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 1467.85 | 1439.95 | 1433.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 1464.95 | 1467.78 | 1457.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 14:45:00 | 1467.00 | 1467.78 | 1457.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 1462.00 | 1466.62 | 1458.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 1465.00 | 1466.62 | 1458.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 11:15:00 | 1452.15 | 1465.15 | 1459.93 | SL hit (close<static) qty=1.00 sl=1455.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 1451.65 | 1457.43 | 1457.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 1445.80 | 1454.23 | 1456.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 1447.35 | 1445.84 | 1451.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 10:00:00 | 1447.35 | 1445.84 | 1451.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 1465.50 | 1449.77 | 1452.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 1465.50 | 1449.77 | 1452.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 1458.10 | 1451.44 | 1453.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:30:00 | 1439.95 | 1443.61 | 1448.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 15:15:00 | 1367.95 | 1385.13 | 1404.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1406.30 | 1389.36 | 1404.56 | SL hit (close>ema200) qty=0.50 sl=1389.36 alert=retest2 |

### Cycle 121 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 1424.80 | 1412.03 | 1410.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 1439.95 | 1417.61 | 1413.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 12:15:00 | 1499.05 | 1500.87 | 1482.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 13:00:00 | 1499.05 | 1500.87 | 1482.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1516.80 | 1526.30 | 1513.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 1517.85 | 1526.30 | 1513.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1515.10 | 1524.06 | 1514.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 1515.00 | 1524.06 | 1514.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1513.55 | 1521.96 | 1513.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:15:00 | 1514.95 | 1521.96 | 1513.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1514.10 | 1520.38 | 1514.00 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 1499.95 | 1510.14 | 1511.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 1493.00 | 1506.71 | 1509.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 1513.70 | 1502.38 | 1504.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 1513.70 | 1502.38 | 1504.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1513.70 | 1502.38 | 1504.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 1511.70 | 1502.38 | 1504.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 1534.55 | 1508.82 | 1507.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 1545.60 | 1518.16 | 1512.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 1520.95 | 1521.67 | 1515.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 12:15:00 | 1520.95 | 1521.67 | 1515.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 1520.95 | 1521.67 | 1515.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:30:00 | 1521.40 | 1521.67 | 1515.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 1518.75 | 1521.08 | 1515.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 1518.75 | 1521.08 | 1515.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 1527.75 | 1522.42 | 1516.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 1532.05 | 1521.98 | 1517.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 10:15:00 | 1505.15 | 1519.17 | 1516.78 | SL hit (close<static) qty=1.00 sl=1516.55 alert=retest2 |

### Cycle 124 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 1533.45 | 1544.89 | 1545.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 1523.70 | 1540.65 | 1543.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 1453.65 | 1451.03 | 1479.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 1453.65 | 1451.03 | 1479.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1485.90 | 1460.31 | 1478.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:00:00 | 1485.90 | 1460.31 | 1478.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1475.40 | 1463.32 | 1478.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1443.95 | 1467.02 | 1477.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1516.15 | 1474.35 | 1473.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1516.15 | 1474.35 | 1473.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 1529.00 | 1485.28 | 1478.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 1607.90 | 1618.90 | 1604.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 1607.90 | 1618.90 | 1604.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1607.90 | 1618.90 | 1604.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 1607.90 | 1618.90 | 1604.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1604.00 | 1615.92 | 1604.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1602.50 | 1615.92 | 1604.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 1629.70 | 1618.67 | 1606.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:30:00 | 1631.20 | 1619.94 | 1608.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:15:00 | 1632.90 | 1622.77 | 1612.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:00:00 | 1635.00 | 1625.22 | 1614.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 13:00:00 | 1630.60 | 1626.72 | 1617.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1602.30 | 1622.37 | 1618.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1602.30 | 1622.37 | 1618.84 | SL hit (close<static) qty=1.00 sl=1604.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1594.20 | 1612.44 | 1614.67 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 1623.90 | 1616.34 | 1615.80 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 11:15:00 | 1607.00 | 1615.14 | 1615.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 12:15:00 | 1600.70 | 1612.25 | 1614.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 15:15:00 | 1610.00 | 1607.84 | 1611.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-30 09:15:00 | 1617.50 | 1607.84 | 1611.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1633.10 | 1612.89 | 1613.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:45:00 | 1626.90 | 1612.89 | 1613.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 1632.90 | 1616.89 | 1615.29 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 1589.00 | 1611.28 | 1614.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 12:15:00 | 1583.50 | 1605.73 | 1611.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1614.50 | 1605.37 | 1609.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 1614.50 | 1605.37 | 1609.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1614.50 | 1605.37 | 1609.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1614.50 | 1605.37 | 1609.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1609.60 | 1606.22 | 1609.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 1616.90 | 1606.22 | 1609.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1613.80 | 1607.73 | 1609.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1613.80 | 1607.73 | 1609.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1618.30 | 1609.85 | 1610.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:00:00 | 1618.30 | 1609.85 | 1610.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 1618.60 | 1611.60 | 1611.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 10:15:00 | 1629.90 | 1617.47 | 1614.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 12:15:00 | 1614.90 | 1619.12 | 1615.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 12:15:00 | 1614.90 | 1619.12 | 1615.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 1614.90 | 1619.12 | 1615.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 1614.90 | 1619.12 | 1615.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 1616.40 | 1618.58 | 1615.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:15:00 | 1616.90 | 1618.58 | 1615.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 1613.30 | 1617.52 | 1615.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:45:00 | 1611.70 | 1617.52 | 1615.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 1612.00 | 1616.42 | 1615.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 1629.00 | 1616.42 | 1615.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1705.00 | 1659.52 | 1641.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:45:00 | 1723.40 | 1672.64 | 1663.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 11:00:00 | 1711.30 | 1680.38 | 1667.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 1830.00 | 1834.37 | 1834.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 1830.00 | 1834.37 | 1834.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 11:15:00 | 1822.00 | 1831.90 | 1833.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 1829.80 | 1827.25 | 1830.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 1829.80 | 1827.25 | 1830.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1829.80 | 1827.25 | 1830.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:45:00 | 1826.00 | 1827.25 | 1830.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1807.30 | 1822.90 | 1827.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 1801.60 | 1822.90 | 1827.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:30:00 | 1802.20 | 1812.65 | 1821.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 13:00:00 | 1801.30 | 1812.65 | 1821.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:00:00 | 1802.60 | 1810.64 | 1819.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1811.60 | 1810.83 | 1819.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 1811.60 | 1810.83 | 1819.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1807.90 | 1803.29 | 1809.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 1807.90 | 1803.29 | 1809.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1825.10 | 1808.18 | 1810.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 1825.00 | 1813.90 | 1813.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 1825.00 | 1813.90 | 1813.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 1861.30 | 1825.37 | 1819.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 1902.10 | 1904.35 | 1883.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:45:00 | 1905.50 | 1904.35 | 1883.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1893.30 | 1907.32 | 1895.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:00:00 | 1893.30 | 1907.32 | 1895.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 1890.70 | 1904.00 | 1895.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:00:00 | 1890.70 | 1904.00 | 1895.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 1885.70 | 1900.34 | 1894.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:45:00 | 1885.00 | 1900.34 | 1894.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 1893.00 | 1893.16 | 1892.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 1893.00 | 1893.16 | 1892.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1895.50 | 1893.63 | 1892.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 15:00:00 | 1895.80 | 1894.32 | 1893.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 15:15:00 | 1887.40 | 1892.94 | 1892.71 | SL hit (close<static) qty=1.00 sl=1888.60 alert=retest2 |

### Cycle 134 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 1888.40 | 1893.01 | 1893.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 15:15:00 | 1881.50 | 1890.71 | 1892.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1846.20 | 1845.20 | 1858.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1846.20 | 1845.20 | 1858.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1853.30 | 1842.75 | 1849.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1853.30 | 1842.75 | 1849.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1852.00 | 1844.60 | 1849.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1859.30 | 1844.60 | 1849.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1847.50 | 1845.18 | 1849.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:30:00 | 1843.10 | 1844.17 | 1848.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1794.90 | 1789.70 | 1789.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1794.90 | 1789.70 | 1789.03 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 1778.60 | 1790.66 | 1791.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 12:15:00 | 1762.80 | 1781.83 | 1786.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 12:15:00 | 1742.00 | 1737.04 | 1750.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 12:15:00 | 1742.00 | 1737.04 | 1750.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1742.00 | 1737.04 | 1750.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:30:00 | 1750.00 | 1737.04 | 1750.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 1762.30 | 1743.91 | 1751.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 1762.30 | 1743.91 | 1751.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1761.70 | 1747.47 | 1752.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1746.60 | 1747.47 | 1752.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 1747.00 | 1742.19 | 1746.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 1758.70 | 1742.19 | 1746.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1737.20 | 1741.19 | 1745.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:15:00 | 1731.90 | 1740.81 | 1744.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 1734.00 | 1739.45 | 1743.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:30:00 | 1734.40 | 1738.20 | 1742.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:00:00 | 1733.20 | 1738.20 | 1742.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1724.60 | 1732.14 | 1738.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 1720.60 | 1727.89 | 1735.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 11:15:00 | 1738.80 | 1730.98 | 1730.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 11:15:00 | 1738.80 | 1730.98 | 1730.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 1741.50 | 1735.45 | 1733.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 10:15:00 | 1726.30 | 1734.90 | 1733.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 10:15:00 | 1726.30 | 1734.90 | 1733.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1726.30 | 1734.90 | 1733.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 1726.30 | 1734.90 | 1733.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 11:15:00 | 1722.20 | 1732.36 | 1732.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 1705.00 | 1726.89 | 1730.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 11:15:00 | 1719.50 | 1716.46 | 1722.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:00:00 | 1719.50 | 1716.46 | 1722.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1722.30 | 1717.63 | 1722.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:30:00 | 1723.10 | 1717.63 | 1722.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1724.70 | 1719.04 | 1722.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 1723.60 | 1719.04 | 1722.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1718.30 | 1718.89 | 1722.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 15:15:00 | 1715.00 | 1718.89 | 1722.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:45:00 | 1715.00 | 1716.67 | 1720.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 1719.20 | 1707.49 | 1706.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 1719.20 | 1707.49 | 1706.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 1721.70 | 1710.33 | 1708.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1711.00 | 1713.72 | 1710.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1711.00 | 1713.72 | 1710.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1711.00 | 1713.72 | 1710.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 1705.50 | 1713.72 | 1710.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1710.30 | 1713.03 | 1710.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 1710.30 | 1713.03 | 1710.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1712.90 | 1713.01 | 1710.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:45:00 | 1717.20 | 1713.74 | 1711.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 15:15:00 | 1718.70 | 1713.74 | 1711.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:15:00 | 1717.90 | 1714.98 | 1712.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 1705.90 | 1713.17 | 1712.02 | SL hit (close<static) qty=1.00 sl=1706.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 1702.00 | 1710.93 | 1711.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 1693.40 | 1705.21 | 1708.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 1684.90 | 1674.97 | 1685.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 10:15:00 | 1684.90 | 1674.97 | 1685.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1684.90 | 1674.97 | 1685.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 1684.90 | 1674.97 | 1685.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1695.40 | 1679.06 | 1686.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:45:00 | 1696.80 | 1679.06 | 1686.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1696.20 | 1682.49 | 1687.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:15:00 | 1687.80 | 1684.03 | 1687.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 1688.70 | 1686.04 | 1688.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 1669.70 | 1687.81 | 1688.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1693.70 | 1676.01 | 1675.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 1693.70 | 1676.01 | 1675.31 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1608.40 | 1668.81 | 1673.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 1573.10 | 1628.56 | 1652.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 1522.20 | 1516.27 | 1549.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 1522.20 | 1516.27 | 1549.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1552.00 | 1524.31 | 1542.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 1552.00 | 1524.31 | 1542.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1561.40 | 1531.73 | 1544.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:30:00 | 1560.00 | 1531.73 | 1544.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1566.50 | 1550.84 | 1550.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 15:15:00 | 1568.90 | 1554.45 | 1552.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 1586.80 | 1598.51 | 1586.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 1586.80 | 1598.51 | 1586.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1586.80 | 1598.51 | 1586.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 1586.80 | 1598.51 | 1586.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 1594.10 | 1597.63 | 1587.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 1593.00 | 1597.63 | 1587.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1604.20 | 1598.94 | 1588.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 1584.10 | 1598.94 | 1588.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1588.60 | 1596.65 | 1590.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1588.60 | 1596.65 | 1590.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1583.50 | 1594.02 | 1589.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:30:00 | 1585.80 | 1594.02 | 1589.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1587.00 | 1592.62 | 1589.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 1608.70 | 1591.27 | 1588.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 11:15:00 | 1579.50 | 1597.64 | 1597.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 1579.50 | 1597.64 | 1597.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 1568.40 | 1586.55 | 1592.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 1570.50 | 1569.66 | 1579.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:00:00 | 1570.50 | 1569.66 | 1579.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1596.70 | 1575.07 | 1580.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1596.70 | 1575.07 | 1580.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1591.30 | 1578.32 | 1581.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1602.50 | 1578.32 | 1581.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 1592.30 | 1584.71 | 1584.40 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 1581.00 | 1584.76 | 1584.80 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 1597.10 | 1587.23 | 1585.92 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 1576.30 | 1585.33 | 1585.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 10:15:00 | 1567.80 | 1581.82 | 1584.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 12:15:00 | 1580.80 | 1580.54 | 1583.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 13:00:00 | 1580.80 | 1580.54 | 1583.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 1584.90 | 1581.41 | 1583.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 1584.90 | 1581.41 | 1583.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 1580.50 | 1581.23 | 1582.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:30:00 | 1585.10 | 1581.23 | 1582.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1600.10 | 1585.13 | 1584.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 1605.00 | 1589.10 | 1586.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 12:15:00 | 1589.20 | 1589.30 | 1586.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 12:30:00 | 1591.50 | 1589.30 | 1586.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1585.10 | 1588.46 | 1586.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:45:00 | 1584.20 | 1588.46 | 1586.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1582.60 | 1587.29 | 1586.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 1582.60 | 1587.29 | 1586.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1583.00 | 1586.43 | 1586.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 1593.80 | 1586.43 | 1586.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1594.20 | 1587.98 | 1586.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:30:00 | 1601.50 | 1594.93 | 1590.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 1631.20 | 1642.16 | 1642.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 1631.20 | 1642.16 | 1642.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 1623.90 | 1634.98 | 1638.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1619.50 | 1610.41 | 1620.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 1619.50 | 1610.41 | 1620.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 1619.50 | 1610.41 | 1620.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 1619.50 | 1610.41 | 1620.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1627.00 | 1613.73 | 1621.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1631.00 | 1613.73 | 1621.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1617.40 | 1614.46 | 1620.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:30:00 | 1623.90 | 1614.46 | 1620.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1608.10 | 1613.19 | 1619.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:45:00 | 1616.30 | 1613.19 | 1619.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1624.00 | 1612.26 | 1617.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1624.00 | 1612.26 | 1617.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1635.00 | 1616.81 | 1619.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1635.00 | 1616.81 | 1619.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1629.10 | 1619.27 | 1619.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 1635.00 | 1619.27 | 1619.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 1631.00 | 1621.61 | 1620.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 1634.20 | 1624.13 | 1622.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1653.90 | 1654.41 | 1644.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 1653.90 | 1654.41 | 1644.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1662.40 | 1668.85 | 1663.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 1662.40 | 1668.85 | 1663.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1653.30 | 1665.74 | 1662.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1653.30 | 1665.74 | 1662.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1653.60 | 1663.31 | 1661.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:30:00 | 1652.50 | 1663.31 | 1661.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 1664.40 | 1664.25 | 1662.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 1664.40 | 1664.25 | 1662.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1677.10 | 1667.42 | 1664.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 13:45:00 | 1684.80 | 1677.10 | 1670.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:00:00 | 1685.80 | 1682.11 | 1676.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:00:00 | 1685.00 | 1689.85 | 1684.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 1682.90 | 1683.86 | 1683.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 1682.90 | 1683.86 | 1683.97 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 1707.10 | 1687.56 | 1685.52 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 1691.20 | 1693.97 | 1693.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 1679.60 | 1690.30 | 1692.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 15:15:00 | 1690.00 | 1689.99 | 1691.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 15:15:00 | 1690.00 | 1689.99 | 1691.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1690.00 | 1689.99 | 1691.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1693.70 | 1689.99 | 1691.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1686.50 | 1689.29 | 1691.26 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 1695.80 | 1692.36 | 1692.20 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 1687.40 | 1691.69 | 1691.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 10:15:00 | 1673.70 | 1688.10 | 1690.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 13:15:00 | 1688.00 | 1684.60 | 1687.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 13:15:00 | 1688.00 | 1684.60 | 1687.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1688.00 | 1684.60 | 1687.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 1688.00 | 1684.60 | 1687.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1691.00 | 1685.88 | 1688.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1691.00 | 1685.88 | 1688.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1694.00 | 1687.50 | 1688.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:30:00 | 1694.50 | 1687.46 | 1688.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1691.50 | 1688.27 | 1688.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 1691.50 | 1688.27 | 1688.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 1686.00 | 1687.82 | 1688.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:15:00 | 1680.60 | 1686.10 | 1687.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:30:00 | 1674.40 | 1680.60 | 1684.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 1678.30 | 1679.85 | 1681.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 1676.10 | 1681.21 | 1682.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1685.90 | 1682.15 | 1682.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 1685.90 | 1682.15 | 1682.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1685.20 | 1682.76 | 1682.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 1683.10 | 1682.76 | 1682.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1676.50 | 1681.51 | 1682.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 1673.10 | 1680.63 | 1681.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 1689.40 | 1681.18 | 1681.76 | SL hit (close>static) qty=1.00 sl=1689.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 1688.50 | 1682.64 | 1682.38 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 1679.60 | 1682.03 | 1682.12 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 12:15:00 | 1683.70 | 1682.37 | 1682.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 13:15:00 | 1691.00 | 1684.09 | 1683.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 10:15:00 | 1683.10 | 1686.95 | 1685.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 10:15:00 | 1683.10 | 1686.95 | 1685.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1683.10 | 1686.95 | 1685.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 1683.10 | 1686.95 | 1685.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 1672.60 | 1684.08 | 1683.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:45:00 | 1669.00 | 1684.08 | 1683.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 1670.00 | 1681.26 | 1682.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 1667.90 | 1678.59 | 1681.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1685.00 | 1674.42 | 1678.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1685.00 | 1674.42 | 1678.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1685.00 | 1674.42 | 1678.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 1685.50 | 1674.42 | 1678.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1685.00 | 1676.54 | 1678.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 1685.80 | 1676.54 | 1678.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 12:15:00 | 1697.70 | 1683.24 | 1681.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 15:15:00 | 1702.00 | 1690.18 | 1685.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 09:15:00 | 1688.80 | 1689.90 | 1685.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 10:00:00 | 1688.80 | 1689.90 | 1685.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 1683.70 | 1688.16 | 1685.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 1683.70 | 1688.16 | 1685.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1690.50 | 1688.63 | 1686.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:30:00 | 1691.20 | 1688.63 | 1686.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1687.90 | 1688.49 | 1686.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 1687.90 | 1688.49 | 1686.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1683.30 | 1687.45 | 1686.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 09:30:00 | 1696.00 | 1687.76 | 1686.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 1696.10 | 1687.76 | 1686.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 11:00:00 | 1696.50 | 1689.51 | 1687.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1724.20 | 1732.43 | 1732.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1724.20 | 1732.43 | 1732.60 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 1742.80 | 1732.77 | 1732.62 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1721.40 | 1734.69 | 1736.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 1707.00 | 1729.15 | 1733.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1730.30 | 1725.99 | 1729.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1730.30 | 1725.99 | 1729.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1730.30 | 1725.99 | 1729.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 1728.90 | 1725.99 | 1729.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1742.40 | 1729.27 | 1730.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 1742.40 | 1729.27 | 1730.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1736.80 | 1730.78 | 1731.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 1735.40 | 1730.78 | 1731.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:45:00 | 1733.70 | 1731.04 | 1731.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 13:15:00 | 1735.60 | 1731.95 | 1731.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 1735.60 | 1731.95 | 1731.84 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 09:15:00 | 1722.20 | 1730.27 | 1731.13 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 1735.30 | 1730.32 | 1730.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 12:15:00 | 1753.00 | 1735.91 | 1732.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 13:15:00 | 1747.60 | 1750.14 | 1744.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 14:00:00 | 1747.60 | 1750.14 | 1744.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 1745.50 | 1749.21 | 1744.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 15:15:00 | 1754.00 | 1749.21 | 1744.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:45:00 | 1749.80 | 1753.71 | 1751.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:45:00 | 1753.20 | 1752.97 | 1751.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 1750.50 | 1750.91 | 1750.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 1754.00 | 1751.53 | 1750.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 1750.00 | 1751.53 | 1750.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1769.80 | 1756.01 | 1752.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 11:15:00 | 1776.80 | 1759.09 | 1754.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:45:00 | 1775.50 | 1765.26 | 1760.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 1780.00 | 1792.40 | 1783.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 10:30:00 | 1781.90 | 1791.92 | 1784.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 1781.20 | 1790.44 | 1784.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 1781.20 | 1790.44 | 1784.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1784.00 | 1789.16 | 1784.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 1778.60 | 1789.16 | 1784.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 1783.50 | 1788.02 | 1784.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1797.00 | 1787.22 | 1784.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 1778.90 | 1792.57 | 1793.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1778.90 | 1792.57 | 1793.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 1765.60 | 1787.18 | 1791.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 1793.90 | 1784.40 | 1787.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 1793.90 | 1784.40 | 1787.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1793.90 | 1784.40 | 1787.56 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 11:15:00 | 1807.00 | 1792.04 | 1790.68 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 1783.20 | 1790.38 | 1790.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1779.60 | 1788.22 | 1789.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 1790.70 | 1788.72 | 1789.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 1790.70 | 1788.72 | 1789.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1790.70 | 1788.72 | 1789.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 1790.70 | 1788.72 | 1789.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1793.60 | 1789.70 | 1790.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 1797.80 | 1789.70 | 1790.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 1801.40 | 1792.13 | 1791.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 14:15:00 | 1804.00 | 1794.51 | 1792.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 09:15:00 | 1791.60 | 1795.25 | 1793.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 1791.60 | 1795.25 | 1793.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1791.60 | 1795.25 | 1793.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 1791.80 | 1795.25 | 1793.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1796.20 | 1795.44 | 1793.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 11:30:00 | 1801.50 | 1797.13 | 1794.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 1790.00 | 1794.68 | 1795.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 14:15:00 | 1790.00 | 1794.68 | 1795.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 10:15:00 | 1781.70 | 1791.29 | 1793.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 1796.30 | 1791.45 | 1792.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 12:15:00 | 1796.30 | 1791.45 | 1792.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 1796.30 | 1791.45 | 1792.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:45:00 | 1795.20 | 1791.45 | 1792.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 1792.70 | 1791.70 | 1792.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:45:00 | 1796.10 | 1791.70 | 1792.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 1795.60 | 1792.48 | 1793.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 1795.60 | 1792.48 | 1793.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 1796.90 | 1793.36 | 1793.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 1791.50 | 1793.36 | 1793.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 1795.70 | 1793.83 | 1793.72 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 11:15:00 | 1790.10 | 1793.18 | 1793.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 12:15:00 | 1785.60 | 1791.66 | 1792.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 11:15:00 | 1773.80 | 1771.45 | 1780.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 12:15:00 | 1776.60 | 1771.45 | 1780.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1783.70 | 1772.35 | 1777.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 1786.00 | 1772.35 | 1777.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1775.00 | 1772.88 | 1776.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:30:00 | 1771.30 | 1773.22 | 1775.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 12:30:00 | 1768.50 | 1772.11 | 1774.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 1728.90 | 1718.77 | 1718.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1728.90 | 1718.77 | 1718.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 1734.10 | 1721.84 | 1720.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 1717.40 | 1722.42 | 1720.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 1717.40 | 1722.42 | 1720.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1717.40 | 1722.42 | 1720.71 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 1716.00 | 1721.44 | 1721.87 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 1731.20 | 1722.16 | 1721.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 11:15:00 | 1747.40 | 1730.90 | 1728.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 1769.00 | 1770.37 | 1761.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 09:15:00 | 1766.50 | 1770.37 | 1761.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1748.90 | 1766.08 | 1760.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1748.90 | 1766.08 | 1760.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1747.50 | 1762.36 | 1759.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 1747.50 | 1762.36 | 1759.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 1738.30 | 1757.55 | 1757.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 1738.30 | 1757.55 | 1757.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1732.10 | 1752.46 | 1754.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 1722.10 | 1734.27 | 1739.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 1727.30 | 1725.44 | 1730.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 1727.30 | 1725.44 | 1730.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1727.30 | 1725.44 | 1730.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 1728.50 | 1725.44 | 1730.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1722.60 | 1724.87 | 1730.13 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 1736.00 | 1732.56 | 1732.41 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1730.10 | 1732.07 | 1732.20 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 1741.00 | 1732.55 | 1732.09 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 1721.60 | 1731.00 | 1731.84 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 10:15:00 | 1740.90 | 1732.98 | 1732.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 12:15:00 | 1770.90 | 1741.37 | 1736.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 1861.30 | 1865.25 | 1850.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:45:00 | 1858.90 | 1865.25 | 1850.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1855.20 | 1862.41 | 1851.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 1852.60 | 1862.41 | 1851.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1861.40 | 1862.21 | 1852.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 1853.80 | 1862.21 | 1852.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1863.30 | 1860.95 | 1853.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 1871.10 | 1860.95 | 1853.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:00:00 | 1876.60 | 1864.08 | 1855.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 1931.40 | 1944.32 | 1945.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 1931.40 | 1944.32 | 1945.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 1925.00 | 1940.46 | 1943.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 1906.50 | 1885.52 | 1899.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 14:15:00 | 1906.50 | 1885.52 | 1899.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1906.50 | 1885.52 | 1899.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 1906.50 | 1885.52 | 1899.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1905.00 | 1889.41 | 1899.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1905.00 | 1889.41 | 1899.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 1900.40 | 1891.87 | 1899.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 1900.40 | 1891.87 | 1899.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 1884.40 | 1890.37 | 1897.72 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 1921.50 | 1902.85 | 1901.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 11:15:00 | 1943.80 | 1916.14 | 1908.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1936.50 | 1938.73 | 1927.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:30:00 | 1932.00 | 1938.73 | 1927.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1934.10 | 1938.61 | 1930.62 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 1921.50 | 1927.01 | 1927.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1911.60 | 1920.83 | 1923.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1946.20 | 1898.13 | 1904.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1946.20 | 1898.13 | 1904.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1946.20 | 1898.13 | 1904.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1953.10 | 1898.13 | 1904.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1935.60 | 1905.63 | 1907.69 | EMA400 retest candle locked (from downside) |

### Cycle 187 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 1947.90 | 1914.08 | 1911.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 1954.70 | 1926.18 | 1917.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 1997.80 | 1998.45 | 1970.97 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:15:00 | 2052.70 | 1998.45 | 1970.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 2054.90 | 2073.61 | 2059.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 2054.90 | 2073.61 | 2059.30 | SL hit (close<ema400) qty=1.00 sl=2059.30 alert=retest1 |

### Cycle 188 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 2049.90 | 2054.72 | 2054.92 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 12:15:00 | 2066.20 | 2055.05 | 2054.36 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 2046.10 | 2053.25 | 2053.66 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 2080.20 | 2058.70 | 2055.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 2112.90 | 2072.66 | 2062.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 2159.50 | 2160.72 | 2133.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 2159.50 | 2160.72 | 2133.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 2272.80 | 2278.70 | 2261.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 2269.40 | 2278.70 | 2261.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 2239.40 | 2268.65 | 2260.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 2239.40 | 2268.65 | 2260.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 2249.20 | 2264.76 | 2259.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 2257.90 | 2263.75 | 2259.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 2257.80 | 2258.40 | 2258.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 2248.70 | 2256.46 | 2257.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 2248.70 | 2256.46 | 2257.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 2231.20 | 2251.41 | 2254.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 2224.30 | 2219.74 | 2230.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 2224.30 | 2219.74 | 2230.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 2249.70 | 2226.06 | 2231.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 2261.10 | 2226.06 | 2231.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 2249.30 | 2230.71 | 2233.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 2249.30 | 2230.71 | 2233.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 2250.00 | 2237.57 | 2236.28 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 2208.60 | 2231.40 | 2234.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 2203.50 | 2218.77 | 2226.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 2218.50 | 2214.74 | 2223.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 2218.50 | 2214.74 | 2223.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 2218.50 | 2214.74 | 2223.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:15:00 | 2224.90 | 2214.74 | 2223.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 2218.60 | 2215.51 | 2222.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:30:00 | 2227.00 | 2215.51 | 2222.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 2211.00 | 2214.61 | 2221.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 12:30:00 | 2203.10 | 2212.39 | 2220.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 14:15:00 | 2202.80 | 2212.31 | 2219.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 2206.20 | 2201.00 | 2211.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 2204.40 | 2194.73 | 2202.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 2204.70 | 2196.72 | 2202.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:30:00 | 2203.90 | 2196.72 | 2202.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 2201.00 | 2197.58 | 2202.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:15:00 | 2199.10 | 2197.58 | 2202.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 2222.30 | 2201.48 | 2201.67 | SL hit (close>static) qty=1.00 sl=2205.30 alert=retest2 |

### Cycle 195 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 2224.40 | 2206.06 | 2203.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 2231.70 | 2212.62 | 2207.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 2214.00 | 2216.04 | 2210.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 2214.00 | 2216.04 | 2210.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 2214.00 | 2216.04 | 2210.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 2209.50 | 2216.04 | 2210.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 2214.20 | 2215.68 | 2210.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 2212.00 | 2215.68 | 2210.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 2217.00 | 2215.94 | 2211.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 2212.80 | 2215.94 | 2211.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2236.60 | 2233.37 | 2225.87 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 2164.50 | 2213.99 | 2219.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 2127.80 | 2186.23 | 2205.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 2160.50 | 2149.18 | 2176.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 2160.50 | 2149.18 | 2176.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 2172.80 | 2155.43 | 2166.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 2172.90 | 2155.43 | 2166.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2181.50 | 2160.64 | 2167.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 2161.00 | 2164.75 | 2168.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 10:15:00 | 2052.95 | 2130.20 | 2150.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 2152.10 | 2112.15 | 2128.65 | SL hit (close>ema200) qty=0.50 sl=2112.15 alert=retest2 |

### Cycle 197 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1994.70 | 1949.53 | 1946.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 2005.20 | 1968.64 | 1956.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1962.90 | 1987.82 | 1972.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1962.90 | 1987.82 | 1972.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1962.90 | 1987.82 | 1972.86 | EMA400 retest candle locked (from upside) |

### Cycle 198 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1935.60 | 1965.02 | 1965.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1926.90 | 1957.40 | 1962.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1981.80 | 1958.77 | 1961.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1981.80 | 1958.77 | 1961.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1981.80 | 1958.77 | 1961.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 1983.00 | 1958.77 | 1961.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 1985.00 | 1964.02 | 1963.92 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1932.40 | 1965.44 | 1966.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 1913.40 | 1947.65 | 1957.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1962.10 | 1924.00 | 1935.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 1962.10 | 1924.00 | 1935.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1962.10 | 1924.00 | 1935.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 1962.10 | 1924.00 | 1935.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1982.30 | 1935.66 | 1939.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 1977.00 | 1935.66 | 1939.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 1971.90 | 1942.91 | 1942.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 2012.80 | 1963.62 | 1952.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1976.80 | 1989.86 | 1975.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1976.80 | 1989.86 | 1975.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1976.80 | 1989.86 | 1975.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 12:15:00 | 2004.70 | 1992.02 | 1978.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 1957.20 | 1987.37 | 1982.12 | SL hit (close<static) qty=1.00 sl=1965.70 alert=retest2 |

### Cycle 202 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 1936.90 | 1977.28 | 1978.01 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1991.10 | 1974.70 | 1973.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 2000.30 | 1979.82 | 1976.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 13:15:00 | 1955.00 | 1974.86 | 1974.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 13:15:00 | 1955.00 | 1974.86 | 1974.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1955.00 | 1974.86 | 1974.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:30:00 | 1927.10 | 1974.86 | 1974.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — SELL (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 14:15:00 | 1930.70 | 1966.03 | 1970.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 1846.30 | 1937.44 | 1956.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 15:15:00 | 1907.00 | 1903.70 | 1926.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 09:15:00 | 1878.20 | 1903.70 | 1926.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 1913.00 | 1902.44 | 1916.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:15:00 | 1922.80 | 1902.44 | 1916.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 1917.30 | 1905.41 | 1916.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 1885.00 | 1907.73 | 1916.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1983.80 | 1910.51 | 1909.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1983.80 | 1910.51 | 1909.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 1997.60 | 1927.93 | 1917.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 10:15:00 | 2028.00 | 2029.04 | 2003.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 11:00:00 | 2028.00 | 2029.04 | 2003.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1999.40 | 2040.61 | 2022.91 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 1980.70 | 2012.32 | 2014.48 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 12:15:00 | 2033.00 | 2017.05 | 2015.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 2041.90 | 2023.63 | 2018.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 2025.90 | 2030.60 | 2024.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 11:15:00 | 2025.90 | 2030.60 | 2024.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 2025.90 | 2030.60 | 2024.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:00:00 | 2025.90 | 2030.60 | 2024.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 2036.70 | 2031.82 | 2025.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:30:00 | 2022.30 | 2031.82 | 2025.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 2102.40 | 2120.84 | 2104.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:00:00 | 2102.40 | 2120.84 | 2104.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 2104.30 | 2117.54 | 2104.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 2102.70 | 2117.54 | 2104.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 2105.50 | 2115.13 | 2104.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:30:00 | 2104.10 | 2115.13 | 2104.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 2114.00 | 2114.90 | 2105.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 2113.60 | 2114.90 | 2105.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 2118.80 | 2115.68 | 2106.35 | EMA400 retest candle locked (from upside) |

### Cycle 208 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 2048.20 | 2096.75 | 2103.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 2044.90 | 2074.12 | 2090.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 2016.90 | 2016.16 | 2040.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 2016.90 | 2016.16 | 2040.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 2016.90 | 2016.16 | 2040.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 2002.90 | 2015.43 | 2037.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:15:00 | 1902.76 | 1945.10 | 1968.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 15:15:00 | 1877.00 | 1876.45 | 1907.56 | SL hit (close>ema200) qty=0.50 sl=1876.45 alert=retest2 |

### Cycle 209 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 1918.00 | 1894.89 | 1892.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 1957.10 | 1907.33 | 1898.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 1948.70 | 1953.88 | 1936.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 1948.70 | 1953.88 | 1936.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-25 12:45:00 | 1149.00 | 2023-05-26 12:15:00 | 1134.90 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-06-02 13:30:00 | 1138.00 | 2023-06-12 10:15:00 | 1251.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-05 09:15:00 | 1141.60 | 2023-06-12 10:15:00 | 1255.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-22 10:30:00 | 1373.95 | 2023-06-23 15:15:00 | 1330.10 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2023-06-23 10:30:00 | 1372.05 | 2023-06-23 15:15:00 | 1330.10 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2023-06-28 10:30:00 | 1332.35 | 2023-07-03 13:15:00 | 1326.80 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2023-07-03 12:00:00 | 1334.15 | 2023-07-03 13:15:00 | 1326.80 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2023-07-05 14:15:00 | 1312.00 | 2023-07-07 09:15:00 | 1341.30 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2023-07-06 09:30:00 | 1310.85 | 2023-07-07 09:15:00 | 1341.30 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2023-07-06 13:45:00 | 1312.10 | 2023-07-07 09:15:00 | 1341.30 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2023-07-06 14:15:00 | 1313.10 | 2023-07-07 09:15:00 | 1341.30 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2023-07-11 11:00:00 | 1318.00 | 2023-07-11 13:15:00 | 1321.45 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2023-07-11 12:00:00 | 1319.50 | 2023-07-11 13:15:00 | 1321.45 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2023-07-11 12:45:00 | 1319.85 | 2023-07-11 13:15:00 | 1321.45 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2023-07-17 09:30:00 | 1347.55 | 2023-07-24 09:15:00 | 1482.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-18 15:15:00 | 1348.00 | 2023-07-24 09:15:00 | 1482.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-07 11:15:00 | 1460.90 | 2023-08-09 14:15:00 | 1524.80 | STOP_HIT | 1.00 | -4.37% |
| SELL | retest2 | 2023-08-08 09:15:00 | 1457.30 | 2023-08-09 14:15:00 | 1524.80 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest1 | 2023-08-11 13:00:00 | 1561.25 | 2023-08-17 09:15:00 | 1573.80 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest1 | 2023-08-14 09:15:00 | 1562.85 | 2023-08-17 09:15:00 | 1573.80 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2023-08-14 11:15:00 | 1574.80 | 2023-08-17 14:15:00 | 1559.95 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-08-14 13:15:00 | 1579.20 | 2023-08-17 14:15:00 | 1559.95 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2023-08-14 14:15:00 | 1574.60 | 2023-08-17 14:15:00 | 1559.95 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-08-17 10:30:00 | 1574.50 | 2023-08-17 14:15:00 | 1559.95 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-08-18 14:15:00 | 1547.45 | 2023-08-22 09:15:00 | 1584.45 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2023-08-18 14:45:00 | 1547.80 | 2023-08-22 09:15:00 | 1584.45 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2023-08-21 10:00:00 | 1544.15 | 2023-08-22 09:15:00 | 1584.45 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2023-08-23 09:15:00 | 1579.95 | 2023-08-29 11:15:00 | 1627.25 | STOP_HIT | 1.00 | 2.99% |
| BUY | retest1 | 2023-09-05 09:15:00 | 1761.90 | 2023-09-06 09:15:00 | 1743.05 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-09-06 11:45:00 | 1779.80 | 2023-09-07 10:15:00 | 1733.75 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2023-09-06 13:30:00 | 1777.85 | 2023-09-07 10:15:00 | 1733.75 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2023-09-06 14:15:00 | 1779.65 | 2023-09-07 10:15:00 | 1733.75 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2023-09-12 09:30:00 | 1696.55 | 2023-09-12 15:15:00 | 1611.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-12 09:30:00 | 1696.55 | 2023-09-13 15:15:00 | 1638.00 | STOP_HIT | 0.50 | 3.45% |
| BUY | retest2 | 2023-10-18 14:15:00 | 1762.45 | 2023-10-19 09:15:00 | 1722.85 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2023-10-18 14:45:00 | 1769.15 | 2023-10-19 09:15:00 | 1722.85 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2023-10-23 09:15:00 | 1683.00 | 2023-10-26 09:15:00 | 1598.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 09:15:00 | 1683.00 | 2023-10-27 09:15:00 | 1625.25 | STOP_HIT | 0.50 | 3.43% |
| BUY | retest2 | 2023-11-07 12:45:00 | 1576.00 | 2023-11-20 11:15:00 | 1674.35 | STOP_HIT | 1.00 | 6.24% |
| BUY | retest2 | 2023-11-08 09:15:00 | 1586.00 | 2023-11-20 11:15:00 | 1674.35 | STOP_HIT | 1.00 | 5.57% |
| SELL | retest2 | 2023-11-22 13:00:00 | 1652.95 | 2023-11-23 10:15:00 | 1682.60 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2023-11-22 15:00:00 | 1656.60 | 2023-11-23 10:15:00 | 1682.60 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2023-11-29 12:45:00 | 1658.20 | 2023-11-30 10:15:00 | 1688.75 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2023-11-29 13:15:00 | 1654.85 | 2023-11-30 10:15:00 | 1688.75 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2023-12-07 13:15:00 | 1595.95 | 2023-12-11 14:15:00 | 1632.80 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2023-12-07 14:00:00 | 1594.30 | 2023-12-11 14:15:00 | 1632.80 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2023-12-08 12:45:00 | 1595.90 | 2023-12-11 14:15:00 | 1632.80 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2023-12-08 14:15:00 | 1591.30 | 2023-12-11 14:15:00 | 1632.80 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2023-12-19 12:00:00 | 1605.50 | 2023-12-26 10:15:00 | 1595.20 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2023-12-19 12:30:00 | 1604.80 | 2023-12-26 10:15:00 | 1595.20 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2024-01-01 12:15:00 | 1541.00 | 2024-01-04 10:15:00 | 1463.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-01 13:15:00 | 1541.85 | 2024-01-04 10:15:00 | 1464.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-01 13:45:00 | 1538.20 | 2024-01-04 10:15:00 | 1461.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-01 12:15:00 | 1541.00 | 2024-01-05 09:15:00 | 1505.05 | STOP_HIT | 0.50 | 2.33% |
| SELL | retest2 | 2024-01-01 13:15:00 | 1541.85 | 2024-01-05 09:15:00 | 1505.05 | STOP_HIT | 0.50 | 2.39% |
| SELL | retest2 | 2024-01-01 13:45:00 | 1538.20 | 2024-01-05 09:15:00 | 1505.05 | STOP_HIT | 0.50 | 2.16% |
| BUY | retest2 | 2024-01-09 09:45:00 | 1519.10 | 2024-01-15 12:15:00 | 1551.35 | STOP_HIT | 1.00 | 2.12% |
| SELL | retest2 | 2024-01-17 09:15:00 | 1525.05 | 2024-01-19 14:15:00 | 1547.70 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-01-25 11:30:00 | 1486.00 | 2024-01-29 14:15:00 | 1533.90 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2024-01-25 12:45:00 | 1486.05 | 2024-01-29 14:15:00 | 1533.90 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2024-01-29 10:00:00 | 1484.15 | 2024-01-29 14:15:00 | 1533.90 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2024-02-01 09:30:00 | 1500.00 | 2024-02-05 14:15:00 | 1425.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-01 11:00:00 | 1500.00 | 2024-02-05 14:15:00 | 1425.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-01 09:30:00 | 1500.00 | 2024-02-08 09:15:00 | 1400.75 | STOP_HIT | 0.50 | 6.62% |
| SELL | retest2 | 2024-02-01 11:00:00 | 1500.00 | 2024-02-08 09:15:00 | 1400.75 | STOP_HIT | 0.50 | 6.62% |
| BUY | retest2 | 2024-02-28 14:15:00 | 1465.50 | 2024-03-13 09:15:00 | 1612.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-29 09:15:00 | 1466.45 | 2024-03-13 09:15:00 | 1613.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-29 09:45:00 | 1465.95 | 2024-03-13 09:15:00 | 1612.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-29 10:15:00 | 1468.70 | 2024-03-13 09:15:00 | 1615.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-06 15:00:00 | 1556.20 | 2024-03-13 14:15:00 | 1484.65 | STOP_HIT | 1.00 | -4.60% |
| BUY | retest2 | 2024-03-07 09:30:00 | 1565.15 | 2024-03-13 14:15:00 | 1484.65 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest2 | 2024-04-02 09:15:00 | 1536.00 | 2024-04-05 14:15:00 | 1548.55 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2024-04-10 12:15:00 | 1571.00 | 2024-04-10 12:15:00 | 1572.50 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2024-04-18 13:15:00 | 1542.60 | 2024-04-22 13:15:00 | 1556.45 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-04-22 11:15:00 | 1544.40 | 2024-04-22 13:15:00 | 1556.45 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-04-25 13:45:00 | 1566.70 | 2024-04-25 14:15:00 | 1546.30 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-05-03 14:30:00 | 1587.80 | 2024-05-06 10:15:00 | 1570.10 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-05-03 15:00:00 | 1589.70 | 2024-05-06 10:15:00 | 1570.10 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-05-06 09:15:00 | 1600.65 | 2024-05-06 10:15:00 | 1570.10 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-05-06 09:45:00 | 1592.75 | 2024-05-06 10:15:00 | 1570.10 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-05-09 09:15:00 | 1551.05 | 2024-05-13 09:15:00 | 1473.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 09:15:00 | 1551.05 | 2024-05-13 11:15:00 | 1530.00 | STOP_HIT | 0.50 | 1.36% |
| SELL | retest2 | 2024-05-13 15:15:00 | 1550.50 | 2024-05-13 15:15:00 | 1550.50 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-05-23 12:30:00 | 1699.50 | 2024-05-24 12:15:00 | 1687.30 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-05-23 13:45:00 | 1699.60 | 2024-05-24 12:15:00 | 1687.30 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-05-23 14:45:00 | 1698.70 | 2024-05-24 12:15:00 | 1687.30 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-05-24 09:15:00 | 1713.30 | 2024-05-24 12:15:00 | 1687.30 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-05-24 10:30:00 | 1722.85 | 2024-05-24 12:15:00 | 1687.30 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1516.60 | 2024-06-04 11:15:00 | 1440.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 10:45:00 | 1521.95 | 2024-06-04 11:15:00 | 1445.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1516.60 | 2024-06-05 09:15:00 | 1479.15 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2024-06-04 10:45:00 | 1521.95 | 2024-06-05 09:15:00 | 1479.15 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2024-06-06 09:30:00 | 1524.15 | 2024-06-06 10:15:00 | 1581.50 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2024-06-21 15:15:00 | 1654.00 | 2024-06-26 12:15:00 | 1601.60 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2024-07-04 09:15:00 | 1588.00 | 2024-07-10 11:15:00 | 1569.00 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2024-07-12 12:45:00 | 1537.50 | 2024-07-16 15:15:00 | 1553.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-07-12 14:45:00 | 1536.50 | 2024-07-16 15:15:00 | 1553.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-07-15 12:45:00 | 1535.95 | 2024-07-16 15:15:00 | 1553.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-07-24 11:15:00 | 1476.95 | 2024-07-29 09:15:00 | 1487.45 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-07-25 09:15:00 | 1472.05 | 2024-07-29 09:15:00 | 1487.45 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-08-06 15:00:00 | 1411.40 | 2024-08-09 09:15:00 | 1443.35 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-08-07 11:15:00 | 1419.00 | 2024-08-09 09:15:00 | 1443.35 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-08-08 09:30:00 | 1418.00 | 2024-08-09 09:15:00 | 1443.35 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-09-06 09:15:00 | 1458.25 | 2024-09-06 09:15:00 | 1439.70 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-09-16 13:15:00 | 1456.40 | 2024-09-17 09:15:00 | 1437.25 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-09-16 15:00:00 | 1453.55 | 2024-09-17 09:15:00 | 1437.25 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-09-17 10:15:00 | 1448.50 | 2024-09-18 10:15:00 | 1432.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-09-17 13:15:00 | 1449.50 | 2024-09-18 10:15:00 | 1432.50 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-09-20 14:30:00 | 1426.50 | 2024-09-24 12:15:00 | 1459.80 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-09-23 09:45:00 | 1427.65 | 2024-09-24 12:15:00 | 1459.80 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-09-23 10:30:00 | 1427.20 | 2024-09-24 12:15:00 | 1459.80 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-09-23 11:00:00 | 1426.25 | 2024-09-24 12:15:00 | 1459.80 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-10-09 14:45:00 | 1538.65 | 2024-10-10 14:15:00 | 1574.90 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-10-10 10:15:00 | 1543.05 | 2024-10-10 14:15:00 | 1574.90 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-10-10 11:00:00 | 1544.20 | 2024-10-10 14:15:00 | 1574.90 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-10-11 14:00:00 | 1569.15 | 2024-10-15 09:15:00 | 1540.30 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-10-23 09:15:00 | 1518.90 | 2024-10-25 10:15:00 | 1442.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 10:00:00 | 1520.05 | 2024-10-25 10:15:00 | 1444.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 13:45:00 | 1518.00 | 2024-10-25 10:15:00 | 1442.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 09:15:00 | 1518.90 | 2024-10-25 14:15:00 | 1477.45 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2024-10-23 10:00:00 | 1520.05 | 2024-10-25 14:15:00 | 1477.45 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2024-10-23 13:45:00 | 1518.00 | 2024-10-25 14:15:00 | 1477.45 | STOP_HIT | 0.50 | 2.67% |
| BUY | retest2 | 2024-11-04 15:00:00 | 1515.00 | 2024-11-08 10:15:00 | 1511.95 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1536.60 | 2024-11-08 10:15:00 | 1511.95 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-11-14 13:30:00 | 1479.00 | 2024-11-22 09:15:00 | 1405.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 12:15:00 | 1487.50 | 2024-11-22 09:15:00 | 1413.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-14 13:30:00 | 1479.00 | 2024-11-25 09:15:00 | 1458.95 | STOP_HIT | 0.50 | 1.36% |
| SELL | retest2 | 2024-11-19 12:15:00 | 1487.50 | 2024-11-25 09:15:00 | 1458.95 | STOP_HIT | 0.50 | 1.92% |
| BUY | retest2 | 2024-12-02 13:45:00 | 1500.85 | 2024-12-13 09:15:00 | 1589.55 | STOP_HIT | 1.00 | 5.91% |
| BUY | retest2 | 2024-12-02 14:30:00 | 1500.80 | 2024-12-13 09:15:00 | 1589.55 | STOP_HIT | 1.00 | 5.91% |
| SELL | retest2 | 2024-12-18 09:45:00 | 1573.45 | 2024-12-20 11:15:00 | 1594.70 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-12-18 10:45:00 | 1575.55 | 2024-12-20 11:15:00 | 1594.70 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-12-18 11:30:00 | 1576.35 | 2024-12-20 11:15:00 | 1594.70 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-12-18 12:15:00 | 1576.25 | 2024-12-20 11:15:00 | 1594.70 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest1 | 2025-01-03 09:15:00 | 1591.65 | 2025-01-06 10:15:00 | 1591.95 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-01-29 13:30:00 | 1521.05 | 2025-02-03 09:15:00 | 1445.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-29 14:45:00 | 1520.60 | 2025-02-03 09:15:00 | 1444.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-30 10:15:00 | 1520.75 | 2025-02-03 09:15:00 | 1444.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-29 13:30:00 | 1521.05 | 2025-02-04 14:15:00 | 1435.90 | STOP_HIT | 0.50 | 5.60% |
| SELL | retest2 | 2025-01-29 14:45:00 | 1520.60 | 2025-02-04 14:15:00 | 1435.90 | STOP_HIT | 0.50 | 5.57% |
| SELL | retest2 | 2025-01-30 10:15:00 | 1520.75 | 2025-02-04 14:15:00 | 1435.90 | STOP_HIT | 0.50 | 5.58% |
| SELL | retest2 | 2025-03-03 09:15:00 | 1408.25 | 2025-03-05 09:15:00 | 1464.20 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-03-03 14:30:00 | 1426.25 | 2025-03-05 09:15:00 | 1464.20 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-03-04 11:30:00 | 1426.15 | 2025-03-05 09:15:00 | 1464.20 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-03-04 12:00:00 | 1425.50 | 2025-03-05 09:15:00 | 1464.20 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-03-07 09:15:00 | 1465.00 | 2025-03-07 11:15:00 | 1452.15 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-03-10 09:30:00 | 1466.45 | 2025-03-10 10:15:00 | 1451.55 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-03-12 09:30:00 | 1439.95 | 2025-03-13 15:15:00 | 1367.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-12 09:30:00 | 1439.95 | 2025-03-17 09:15:00 | 1406.30 | STOP_HIT | 0.50 | 2.34% |
| BUY | retest2 | 2025-04-01 09:15:00 | 1532.05 | 2025-04-01 10:15:00 | 1505.15 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-04-02 09:30:00 | 1533.65 | 2025-04-04 12:15:00 | 1533.45 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-04-04 11:00:00 | 1534.50 | 2025-04-04 12:15:00 | 1533.45 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1443.95 | 2025-04-11 09:15:00 | 1516.15 | STOP_HIT | 1.00 | -5.00% |
| BUY | retest2 | 2025-04-23 12:30:00 | 1631.20 | 2025-04-25 09:15:00 | 1602.30 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-04-24 09:15:00 | 1632.90 | 2025-04-25 09:15:00 | 1602.30 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-04-24 10:00:00 | 1635.00 | 2025-04-25 09:15:00 | 1602.30 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-04-24 13:00:00 | 1630.60 | 2025-04-25 09:15:00 | 1602.30 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-05-12 09:45:00 | 1723.40 | 2025-05-29 10:15:00 | 1830.00 | STOP_HIT | 1.00 | 6.19% |
| BUY | retest2 | 2025-05-12 11:00:00 | 1711.30 | 2025-05-29 10:15:00 | 1830.00 | STOP_HIT | 1.00 | 6.94% |
| SELL | retest2 | 2025-05-30 10:15:00 | 1801.60 | 2025-06-03 11:15:00 | 1825.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-05-30 12:30:00 | 1802.20 | 2025-06-03 11:15:00 | 1825.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-05-30 13:00:00 | 1801.30 | 2025-06-03 11:15:00 | 1825.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-05-30 14:00:00 | 1802.60 | 2025-06-03 11:15:00 | 1825.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-06-10 15:00:00 | 1895.80 | 2025-06-10 15:15:00 | 1887.40 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1906.50 | 2025-06-11 13:15:00 | 1887.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-06-11 12:00:00 | 1896.80 | 2025-06-11 13:15:00 | 1887.50 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-06-11 12:30:00 | 1896.00 | 2025-06-11 13:15:00 | 1887.50 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-06-17 10:30:00 | 1843.10 | 2025-06-24 09:15:00 | 1794.90 | STOP_HIT | 1.00 | 2.62% |
| SELL | retest2 | 2025-07-03 11:15:00 | 1731.90 | 2025-07-08 11:15:00 | 1738.80 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-03 12:00:00 | 1734.00 | 2025-07-08 11:15:00 | 1738.80 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-07-03 12:30:00 | 1734.40 | 2025-07-08 11:15:00 | 1738.80 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-07-03 13:00:00 | 1733.20 | 2025-07-08 11:15:00 | 1738.80 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-07-04 11:30:00 | 1720.60 | 2025-07-08 11:15:00 | 1738.80 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-07-10 15:15:00 | 1715.00 | 2025-07-15 12:15:00 | 1719.20 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-07-11 09:45:00 | 1715.00 | 2025-07-15 12:15:00 | 1719.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-07-16 14:45:00 | 1717.20 | 2025-07-17 10:15:00 | 1705.90 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-16 15:15:00 | 1718.70 | 2025-07-17 10:15:00 | 1705.90 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-17 10:15:00 | 1717.90 | 2025-07-17 10:15:00 | 1705.90 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-07-21 14:15:00 | 1687.80 | 2025-07-24 09:15:00 | 1693.70 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-07-21 14:45:00 | 1688.70 | 2025-07-24 09:15:00 | 1693.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-07-22 09:15:00 | 1669.70 | 2025-07-24 09:15:00 | 1693.70 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-08-05 09:15:00 | 1608.70 | 2025-08-06 11:15:00 | 1579.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-08-14 14:30:00 | 1601.50 | 2025-08-26 10:15:00 | 1631.20 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest2 | 2025-09-08 13:45:00 | 1684.80 | 2025-09-11 15:15:00 | 1682.90 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-09 13:00:00 | 1685.80 | 2025-09-11 15:15:00 | 1682.90 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-09-10 14:00:00 | 1685.00 | 2025-09-11 15:15:00 | 1682.90 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-09-22 14:15:00 | 1680.60 | 2025-09-25 09:15:00 | 1689.40 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-09-23 09:30:00 | 1674.40 | 2025-09-25 10:15:00 | 1688.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1678.30 | 2025-09-25 10:15:00 | 1688.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-24 11:15:00 | 1676.10 | 2025-09-25 10:15:00 | 1688.50 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-24 15:15:00 | 1673.10 | 2025-09-25 10:15:00 | 1688.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-10-01 09:30:00 | 1696.00 | 2025-10-08 14:15:00 | 1724.20 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2025-10-01 10:15:00 | 1696.10 | 2025-10-08 14:15:00 | 1724.20 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2025-10-01 11:00:00 | 1696.50 | 2025-10-08 14:15:00 | 1724.20 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-10-15 12:15:00 | 1735.40 | 2025-10-15 13:15:00 | 1735.60 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-10-15 12:45:00 | 1733.70 | 2025-10-15 13:15:00 | 1735.60 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-10-20 15:15:00 | 1754.00 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2025-10-24 10:45:00 | 1749.80 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2025-10-24 11:45:00 | 1753.20 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2025-10-24 14:00:00 | 1750.50 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest2 | 2025-10-27 11:15:00 | 1776.80 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-10-28 09:45:00 | 1775.50 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-10-30 09:30:00 | 1780.00 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-10-30 10:30:00 | 1781.90 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1797.00 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-11-10 11:30:00 | 1801.50 | 2025-11-11 14:15:00 | 1790.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-18 09:30:00 | 1771.30 | 2025-11-26 13:15:00 | 1728.90 | STOP_HIT | 1.00 | 2.39% |
| SELL | retest2 | 2025-11-18 12:30:00 | 1768.50 | 2025-11-26 13:15:00 | 1728.90 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2025-12-26 10:15:00 | 1871.10 | 2026-01-08 09:15:00 | 1931.40 | STOP_HIT | 1.00 | 3.22% |
| BUY | retest2 | 2025-12-26 11:00:00 | 1876.60 | 2026-01-08 09:15:00 | 1931.40 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest1 | 2026-01-27 09:15:00 | 2052.70 | 2026-01-29 13:15:00 | 2054.90 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2026-02-13 11:45:00 | 2257.90 | 2026-02-16 09:15:00 | 2248.70 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2026-02-16 09:15:00 | 2257.80 | 2026-02-16 09:15:00 | 2248.70 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-02-20 12:30:00 | 2203.10 | 2026-02-25 10:15:00 | 2222.30 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-02-20 14:15:00 | 2202.80 | 2026-02-25 11:15:00 | 2224.40 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-02-23 09:30:00 | 2206.20 | 2026-02-25 11:15:00 | 2224.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-24 09:30:00 | 2204.40 | 2026-02-25 11:15:00 | 2224.40 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-02-24 12:15:00 | 2199.10 | 2026-02-25 11:15:00 | 2224.40 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-03-06 14:45:00 | 2161.00 | 2026-03-09 10:15:00 | 2052.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 2161.00 | 2026-03-10 09:15:00 | 2152.10 | STOP_HIT | 0.50 | 0.41% |
| BUY | retest2 | 2026-03-27 12:15:00 | 2004.70 | 2026-03-30 09:15:00 | 1957.20 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-04-07 09:15:00 | 1885.00 | 2026-04-08 09:15:00 | 1983.80 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2026-04-27 11:15:00 | 2002.90 | 2026-04-30 11:15:00 | 1902.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 11:15:00 | 2002.90 | 2026-05-04 15:15:00 | 1877.00 | STOP_HIT | 0.50 | 6.29% |
