# Cholamandalam Financial Holdings Ltd. (CHOLAHLDNG)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1785.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 171 |
| ALERT1 | 109 |
| ALERT2 | 108 |
| ALERT2_SKIP | 64 |
| ALERT3 | 270 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 110 |
| PARTIAL | 12 |
| TARGET_HIT | 4 |
| STOP_HIT | 109 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 124 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 82
- **Target hits / Stop hits / Partials:** 4 / 108 / 12
- **Avg / median % per leg:** 0.14% / -1.16%
- **Sum % (uncompounded):** 17.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 13 | 28.9% | 4 | 41 | 0 | -0.15% | -6.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 45 | 13 | 28.9% | 4 | 41 | 0 | -0.15% | -6.8% |
| SELL (all) | 79 | 29 | 36.7% | 0 | 67 | 12 | 0.30% | 23.9% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.61% | 7.8% |
| SELL @ 3rd Alert (retest2) | 76 | 27 | 35.5% | 0 | 65 | 11 | 0.21% | 16.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.61% | 7.8% |
| retest2 (combined) | 121 | 40 | 33.1% | 4 | 106 | 11 | 0.08% | 9.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 13:15:00 | 1081.15 | 1078.23 | 1078.09 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 1075.85 | 1077.76 | 1077.89 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 15:15:00 | 1081.00 | 1078.40 | 1078.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 1097.95 | 1082.31 | 1079.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 1112.10 | 1119.98 | 1111.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 1112.10 | 1119.98 | 1111.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1112.10 | 1119.98 | 1111.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 11:15:00 | 1124.85 | 1119.83 | 1112.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 12:30:00 | 1130.20 | 1121.72 | 1114.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 1122.55 | 1127.84 | 1120.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 10:15:00 | 1106.05 | 1121.50 | 1121.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 1106.05 | 1121.50 | 1121.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 1098.55 | 1114.44 | 1117.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 11:15:00 | 1123.60 | 1113.61 | 1116.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 11:15:00 | 1123.60 | 1113.61 | 1116.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 1123.60 | 1113.61 | 1116.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:00:00 | 1123.60 | 1113.61 | 1116.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 1124.95 | 1115.88 | 1117.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:30:00 | 1127.15 | 1115.88 | 1117.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 1120.50 | 1117.88 | 1118.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 1129.90 | 1117.88 | 1118.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 09:15:00 | 1120.60 | 1118.43 | 1118.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 11:15:00 | 1143.25 | 1124.20 | 1121.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 13:15:00 | 1121.15 | 1125.27 | 1122.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 13:15:00 | 1121.15 | 1125.27 | 1122.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 1121.15 | 1125.27 | 1122.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:30:00 | 1123.15 | 1125.27 | 1122.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 1120.00 | 1124.22 | 1121.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 14:45:00 | 1120.05 | 1124.22 | 1121.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 1120.60 | 1123.49 | 1121.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 1141.45 | 1123.49 | 1121.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 10:15:00 | 1118.45 | 1124.11 | 1122.52 | SL hit (close<static) qty=1.00 sl=1120.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 1111.10 | 1119.68 | 1120.72 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 09:15:00 | 1126.85 | 1121.21 | 1121.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 14:15:00 | 1138.00 | 1124.71 | 1122.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 1118.90 | 1124.71 | 1123.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 1118.90 | 1124.71 | 1123.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1118.90 | 1124.71 | 1123.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:30:00 | 1119.00 | 1124.71 | 1123.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 1113.45 | 1122.46 | 1122.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:00:00 | 1113.45 | 1122.46 | 1122.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 1111.05 | 1120.18 | 1121.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 1102.90 | 1114.96 | 1118.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 1093.50 | 1091.65 | 1100.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 1093.50 | 1091.65 | 1100.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1093.50 | 1091.65 | 1100.96 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 09:15:00 | 1117.30 | 1105.05 | 1104.36 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1091.15 | 1102.27 | 1103.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1076.20 | 1097.06 | 1100.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 12:15:00 | 1102.90 | 1098.22 | 1100.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 1102.90 | 1098.22 | 1100.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1102.90 | 1098.22 | 1100.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:00:00 | 1102.90 | 1098.22 | 1100.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 1096.90 | 1097.96 | 1100.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 14:00:00 | 1096.90 | 1097.96 | 1100.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 1095.05 | 1097.38 | 1100.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 1095.05 | 1097.38 | 1100.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 1126.50 | 1102.34 | 1101.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1161.05 | 1130.42 | 1118.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 1229.95 | 1231.66 | 1206.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:45:00 | 1228.15 | 1231.66 | 1206.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 1237.85 | 1247.04 | 1234.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 1254.15 | 1245.62 | 1236.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 11:15:00 | 1279.95 | 1285.09 | 1285.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 11:15:00 | 1279.95 | 1285.09 | 1285.68 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 15:15:00 | 1293.90 | 1286.32 | 1285.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 1307.70 | 1290.60 | 1287.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 13:15:00 | 1292.00 | 1295.54 | 1291.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 13:15:00 | 1292.00 | 1295.54 | 1291.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 1292.00 | 1295.54 | 1291.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:00:00 | 1292.00 | 1295.54 | 1291.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 1299.35 | 1296.30 | 1292.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 10:45:00 | 1308.55 | 1296.99 | 1293.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 12:15:00 | 1290.40 | 1295.69 | 1293.55 | SL hit (close<static) qty=1.00 sl=1292.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 1278.75 | 1292.05 | 1292.90 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 1383.00 | 1310.01 | 1300.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 10:15:00 | 1462.80 | 1340.57 | 1315.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 15:15:00 | 1552.00 | 1558.18 | 1511.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 09:15:00 | 1534.10 | 1558.18 | 1511.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 1494.00 | 1532.10 | 1513.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:00:00 | 1494.00 | 1532.10 | 1513.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 1474.95 | 1520.67 | 1509.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:00:00 | 1474.95 | 1520.67 | 1509.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 15:15:00 | 1468.00 | 1501.71 | 1502.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 11:15:00 | 1459.50 | 1482.25 | 1492.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 09:15:00 | 1476.20 | 1471.40 | 1482.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 1476.20 | 1471.40 | 1482.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1476.20 | 1471.40 | 1482.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 15:15:00 | 1452.00 | 1465.14 | 1474.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 09:30:00 | 1453.20 | 1461.01 | 1471.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 10:15:00 | 1449.65 | 1461.01 | 1471.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 10:45:00 | 1451.90 | 1458.81 | 1469.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1445.00 | 1447.76 | 1458.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 1489.00 | 1456.76 | 1456.78 | SL hit (close>static) qty=1.00 sl=1486.60 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 10:15:00 | 1467.00 | 1458.81 | 1457.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 1510.00 | 1488.16 | 1479.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 14:15:00 | 1490.65 | 1495.16 | 1487.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 14:15:00 | 1490.65 | 1495.16 | 1487.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1490.65 | 1495.16 | 1487.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 1490.65 | 1495.16 | 1487.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 1488.00 | 1493.73 | 1487.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 1508.95 | 1493.73 | 1487.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 10:15:00 | 1474.80 | 1491.52 | 1491.00 | SL hit (close<static) qty=1.00 sl=1482.80 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 1475.00 | 1488.22 | 1489.55 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 1523.45 | 1492.69 | 1490.63 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 1491.80 | 1499.26 | 1500.20 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 1552.65 | 1500.02 | 1498.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 1568.80 | 1529.90 | 1516.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 1529.55 | 1535.41 | 1522.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 12:15:00 | 1529.55 | 1535.41 | 1522.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 1529.55 | 1535.41 | 1522.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:45:00 | 1531.60 | 1535.41 | 1522.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 1529.00 | 1534.13 | 1523.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:00:00 | 1529.00 | 1534.13 | 1523.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 1520.00 | 1531.30 | 1523.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:30:00 | 1520.05 | 1531.30 | 1523.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 1518.00 | 1528.64 | 1522.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:15:00 | 1513.00 | 1528.64 | 1522.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1519.95 | 1526.90 | 1522.46 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 13:15:00 | 1499.70 | 1516.40 | 1518.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 15:15:00 | 1491.30 | 1508.52 | 1514.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 09:15:00 | 1526.90 | 1512.19 | 1515.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 09:15:00 | 1526.90 | 1512.19 | 1515.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 1526.90 | 1512.19 | 1515.58 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 09:15:00 | 1558.60 | 1524.89 | 1520.46 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 1496.05 | 1530.49 | 1531.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 15:15:00 | 1457.75 | 1489.13 | 1507.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 1461.10 | 1456.53 | 1478.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 09:45:00 | 1463.35 | 1456.53 | 1478.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 1471.00 | 1459.43 | 1477.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 1470.95 | 1459.43 | 1477.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 1483.85 | 1468.05 | 1474.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:00:00 | 1483.85 | 1468.05 | 1474.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 1490.00 | 1472.44 | 1475.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:45:00 | 1489.90 | 1472.44 | 1475.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 1478.00 | 1474.91 | 1476.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 14:30:00 | 1470.00 | 1474.33 | 1475.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 15:15:00 | 1461.10 | 1474.33 | 1475.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 1505.20 | 1478.38 | 1477.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 1505.20 | 1478.38 | 1477.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 10:15:00 | 1523.90 | 1487.49 | 1481.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 15:15:00 | 1525.00 | 1527.65 | 1507.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 09:15:00 | 1508.55 | 1527.65 | 1507.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1501.55 | 1522.43 | 1506.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:30:00 | 1542.55 | 1510.96 | 1505.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:00:00 | 1543.95 | 1510.96 | 1505.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 1549.55 | 1515.26 | 1511.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-23 13:15:00 | 1696.81 | 1673.62 | 1651.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 1636.50 | 1668.14 | 1669.61 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 1680.00 | 1668.96 | 1668.67 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 12:15:00 | 1656.65 | 1666.50 | 1667.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 09:15:00 | 1619.20 | 1641.90 | 1651.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 1636.95 | 1621.20 | 1633.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 09:15:00 | 1636.95 | 1621.20 | 1633.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 1636.95 | 1621.20 | 1633.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:45:00 | 1636.95 | 1621.20 | 1633.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 1635.00 | 1623.96 | 1633.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:45:00 | 1636.20 | 1623.96 | 1633.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 1618.00 | 1622.77 | 1632.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:30:00 | 1631.75 | 1622.77 | 1632.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 1677.55 | 1632.38 | 1634.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:00:00 | 1677.55 | 1632.38 | 1634.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 15:15:00 | 1663.00 | 1638.50 | 1636.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 1705.70 | 1651.94 | 1642.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 15:15:00 | 1677.00 | 1677.22 | 1662.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 09:15:00 | 1717.25 | 1677.22 | 1662.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1721.90 | 1686.15 | 1667.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 10:45:00 | 1735.80 | 1697.16 | 1674.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:45:00 | 1730.65 | 1720.31 | 1694.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 1770.55 | 1721.25 | 1697.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 11:15:00 | 1784.85 | 1801.98 | 1802.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 11:15:00 | 1784.85 | 1801.98 | 1802.17 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 10:15:00 | 1814.25 | 1800.19 | 1799.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 11:15:00 | 1840.00 | 1808.15 | 1803.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 12:15:00 | 1954.70 | 1954.99 | 1931.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 13:00:00 | 1954.70 | 1954.99 | 1931.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1938.70 | 1950.56 | 1937.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 1938.70 | 1950.56 | 1937.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1939.55 | 1948.36 | 1937.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 10:00:00 | 1965.20 | 1947.41 | 1941.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 1993.20 | 2037.49 | 2040.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 1993.20 | 2037.49 | 2040.49 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 09:15:00 | 2081.05 | 2040.70 | 2036.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 10:15:00 | 2124.80 | 2057.52 | 2044.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 14:15:00 | 2087.65 | 2093.03 | 2068.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 15:00:00 | 2087.65 | 2093.03 | 2068.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 2041.40 | 2083.66 | 2068.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:00:00 | 2041.40 | 2083.66 | 2068.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 2032.80 | 2073.49 | 2065.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:45:00 | 2033.40 | 2073.49 | 2065.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 2027.20 | 2056.53 | 2058.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 2010.40 | 2039.05 | 2049.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 14:15:00 | 1971.00 | 1952.46 | 1984.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 14:15:00 | 1971.00 | 1952.46 | 1984.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 1971.00 | 1952.46 | 1984.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 15:00:00 | 1971.00 | 1952.46 | 1984.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 15:15:00 | 1935.00 | 1948.97 | 1980.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 09:30:00 | 1924.15 | 1942.59 | 1974.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:00:00 | 1917.10 | 1942.59 | 1974.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 12:00:00 | 1929.40 | 1939.48 | 1967.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 14:30:00 | 1922.50 | 1936.52 | 1959.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1960.80 | 1940.97 | 1957.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 09:30:00 | 1927.15 | 1943.33 | 1952.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 13:15:00 | 1928.85 | 1914.28 | 1913.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 13:15:00 | 1928.85 | 1914.28 | 1913.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 14:15:00 | 1971.05 | 1934.39 | 1925.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 10:15:00 | 1973.50 | 1979.12 | 1961.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 11:00:00 | 1973.50 | 1979.12 | 1961.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 1987.00 | 1981.98 | 1968.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:30:00 | 1978.85 | 1981.98 | 1968.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1983.00 | 1981.55 | 1970.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 11:30:00 | 2004.35 | 1986.26 | 1974.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 2008.65 | 1991.38 | 1981.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 10:00:00 | 2005.10 | 1994.12 | 1983.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 1998.30 | 1995.39 | 1985.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 1990.00 | 1995.55 | 1989.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 1954.90 | 1995.55 | 1989.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 1943.60 | 1985.16 | 1984.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-22 09:15:00 | 1943.60 | 1985.16 | 1984.93 | SL hit (close<static) qty=1.00 sl=1960.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 1949.35 | 1978.00 | 1981.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 10:15:00 | 1934.00 | 1952.26 | 1964.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 09:15:00 | 1693.25 | 1693.19 | 1732.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 09:15:00 | 1693.25 | 1693.19 | 1732.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1693.25 | 1693.19 | 1732.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 11:30:00 | 1687.05 | 1690.84 | 1724.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 11:45:00 | 1684.90 | 1688.66 | 1706.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 13:30:00 | 1686.60 | 1688.33 | 1702.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-01 18:15:00 | 1733.00 | 1709.80 | 1709.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 1733.00 | 1709.80 | 1709.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 10:15:00 | 1761.60 | 1725.16 | 1716.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 12:15:00 | 1718.90 | 1725.24 | 1718.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 12:15:00 | 1718.90 | 1725.24 | 1718.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 1718.90 | 1725.24 | 1718.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:45:00 | 1719.90 | 1725.24 | 1718.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 1719.90 | 1724.17 | 1718.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:15:00 | 1742.95 | 1724.17 | 1718.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 14:15:00 | 1713.00 | 1736.36 | 1738.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 14:15:00 | 1713.00 | 1736.36 | 1738.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 10:15:00 | 1691.10 | 1718.51 | 1728.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 14:15:00 | 1684.95 | 1683.04 | 1698.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 15:00:00 | 1684.95 | 1683.04 | 1698.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 1650.00 | 1639.12 | 1652.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 1591.90 | 1639.12 | 1652.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 10:15:00 | 1512.31 | 1555.57 | 1579.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 1580.15 | 1541.25 | 1559.18 | SL hit (close>ema200) qty=0.50 sl=1541.25 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 12:15:00 | 1547.30 | 1531.37 | 1530.57 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 13:15:00 | 1513.05 | 1527.70 | 1528.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-25 15:15:00 | 1500.00 | 1519.79 | 1525.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 10:15:00 | 1521.45 | 1518.18 | 1523.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 10:15:00 | 1521.45 | 1518.18 | 1523.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 1521.45 | 1518.18 | 1523.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 11:00:00 | 1521.45 | 1518.18 | 1523.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 1526.60 | 1519.87 | 1523.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 13:30:00 | 1517.50 | 1521.05 | 1523.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 15:15:00 | 1537.00 | 1525.15 | 1525.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 15:15:00 | 1537.00 | 1525.15 | 1525.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 15:15:00 | 1544.80 | 1534.47 | 1530.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 1566.45 | 1574.51 | 1558.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 1566.45 | 1574.51 | 1558.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1575.50 | 1574.71 | 1559.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:00:00 | 1593.75 | 1576.98 | 1563.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 09:15:00 | 1523.25 | 1569.12 | 1573.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 09:15:00 | 1523.25 | 1569.12 | 1573.91 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 1625.35 | 1574.52 | 1571.03 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 1537.85 | 1570.32 | 1571.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 10:15:00 | 1531.05 | 1562.47 | 1568.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 09:15:00 | 1529.20 | 1523.10 | 1534.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 09:15:00 | 1529.20 | 1523.10 | 1534.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1529.20 | 1523.10 | 1534.08 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 13:15:00 | 1549.95 | 1539.40 | 1539.14 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 1521.90 | 1537.17 | 1538.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 09:15:00 | 1515.30 | 1532.76 | 1536.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 12:15:00 | 1532.95 | 1530.34 | 1534.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 12:15:00 | 1532.95 | 1530.34 | 1534.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 1532.95 | 1530.34 | 1534.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:30:00 | 1556.60 | 1530.34 | 1534.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 1531.55 | 1530.58 | 1533.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 1521.55 | 1529.27 | 1532.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 1539.50 | 1528.98 | 1531.84 | SL hit (close>static) qty=1.00 sl=1537.90 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 13:15:00 | 1542.00 | 1535.03 | 1534.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 15:15:00 | 1549.95 | 1537.63 | 1535.48 | Break + close above crossover candle high |

### Cycle 48 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 1511.25 | 1532.35 | 1533.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 10:15:00 | 1495.70 | 1525.02 | 1529.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 15:15:00 | 1515.00 | 1513.40 | 1521.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 09:15:00 | 1522.40 | 1513.40 | 1521.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1552.65 | 1521.25 | 1523.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 1552.65 | 1521.25 | 1523.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1544.00 | 1525.80 | 1525.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 1553.65 | 1525.80 | 1525.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 1541.80 | 1529.00 | 1527.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 14:15:00 | 1554.90 | 1538.37 | 1532.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 12:15:00 | 1539.75 | 1544.69 | 1538.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 12:15:00 | 1539.75 | 1544.69 | 1538.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 1539.75 | 1544.69 | 1538.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:45:00 | 1542.25 | 1544.69 | 1538.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 1522.80 | 1540.31 | 1536.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:30:00 | 1518.95 | 1540.31 | 1536.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 1503.60 | 1532.97 | 1533.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 12:15:00 | 1501.40 | 1513.81 | 1522.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 15:15:00 | 1507.00 | 1506.77 | 1516.84 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-19 09:15:00 | 1476.45 | 1506.77 | 1516.84 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1461.15 | 1473.49 | 1490.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:15:00 | 1455.90 | 1470.03 | 1486.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:00:00 | 1453.55 | 1466.73 | 1483.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 12:15:00 | 1402.63 | 1431.06 | 1455.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 11:15:00 | 1383.11 | 1407.85 | 1431.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 11:15:00 | 1380.87 | 1407.85 | 1431.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 1399.65 | 1393.24 | 1407.07 | SL hit (close>ema200) qty=0.50 sl=1393.24 alert=retest1 |

### Cycle 51 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1434.00 | 1409.93 | 1408.73 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 11:15:00 | 1405.05 | 1408.71 | 1408.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 12:15:00 | 1393.80 | 1405.73 | 1407.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 14:15:00 | 1399.95 | 1396.16 | 1400.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-01 15:00:00 | 1399.95 | 1396.16 | 1400.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 1392.00 | 1395.33 | 1399.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:15:00 | 1417.35 | 1395.33 | 1399.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1417.90 | 1399.84 | 1401.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:30:00 | 1408.05 | 1399.84 | 1401.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 1426.40 | 1405.15 | 1403.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 12:15:00 | 1470.35 | 1421.13 | 1411.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 1531.05 | 1534.83 | 1499.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 1531.05 | 1534.83 | 1499.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1537.00 | 1539.13 | 1525.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:45:00 | 1524.50 | 1539.13 | 1525.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1535.60 | 1538.12 | 1527.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:30:00 | 1512.30 | 1538.12 | 1527.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 1536.00 | 1537.70 | 1528.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:30:00 | 1527.40 | 1537.70 | 1528.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 1526.75 | 1534.09 | 1528.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:00:00 | 1526.75 | 1534.09 | 1528.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 1529.95 | 1533.26 | 1528.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:45:00 | 1537.95 | 1534.09 | 1529.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 1546.95 | 1532.48 | 1529.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 1521.95 | 1560.99 | 1551.51 | SL hit (close<static) qty=1.00 sl=1522.80 alert=retest2 |

### Cycle 54 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 1516.15 | 1540.03 | 1543.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 1500.25 | 1532.07 | 1539.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 1511.65 | 1485.83 | 1502.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 1511.65 | 1485.83 | 1502.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1511.65 | 1485.83 | 1502.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:00:00 | 1511.65 | 1485.83 | 1502.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1522.30 | 1493.13 | 1504.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 1522.30 | 1493.13 | 1504.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1477.20 | 1489.94 | 1501.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:30:00 | 1466.70 | 1484.27 | 1497.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 15:00:00 | 1466.55 | 1478.16 | 1492.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:30:00 | 1459.65 | 1476.43 | 1487.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:00:00 | 1465.35 | 1466.70 | 1479.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1448.95 | 1460.72 | 1474.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:15:00 | 1445.65 | 1460.72 | 1474.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:45:00 | 1434.85 | 1455.22 | 1470.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 13:45:00 | 1448.15 | 1449.91 | 1464.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 14:15:00 | 1447.65 | 1449.91 | 1464.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 1445.40 | 1448.25 | 1459.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:15:00 | 1433.70 | 1448.25 | 1459.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:30:00 | 1429.05 | 1433.75 | 1444.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 1433.85 | 1445.39 | 1446.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 09:15:00 | 1462.10 | 1448.73 | 1448.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 09:15:00 | 1462.10 | 1448.73 | 1448.13 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 1435.80 | 1446.06 | 1447.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 12:15:00 | 1433.10 | 1440.40 | 1443.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1440.45 | 1434.39 | 1438.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1440.45 | 1434.39 | 1438.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1440.45 | 1434.39 | 1438.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 1440.10 | 1434.39 | 1438.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1443.25 | 1436.16 | 1439.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 1443.25 | 1436.16 | 1439.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 1436.50 | 1436.42 | 1438.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:30:00 | 1436.00 | 1436.42 | 1438.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 1442.95 | 1437.73 | 1438.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:45:00 | 1449.60 | 1437.73 | 1438.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 1428.80 | 1435.94 | 1438.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 1445.95 | 1435.94 | 1438.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1412.10 | 1431.17 | 1435.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 1407.90 | 1424.68 | 1429.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 15:15:00 | 1403.25 | 1405.85 | 1415.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 14:15:00 | 1436.45 | 1418.15 | 1415.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 14:15:00 | 1436.45 | 1418.15 | 1415.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 10:15:00 | 1468.60 | 1431.43 | 1422.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 1522.90 | 1525.80 | 1504.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 14:00:00 | 1522.90 | 1525.80 | 1504.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 1514.75 | 1523.59 | 1505.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:30:00 | 1515.00 | 1523.59 | 1505.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1504.35 | 1519.79 | 1506.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:00:00 | 1504.35 | 1519.79 | 1506.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 1526.20 | 1521.07 | 1508.56 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 15:15:00 | 1505.00 | 1513.19 | 1513.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 15:15:00 | 1497.50 | 1507.18 | 1510.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 14:15:00 | 1494.85 | 1484.20 | 1494.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 14:15:00 | 1494.85 | 1484.20 | 1494.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1494.85 | 1484.20 | 1494.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 1494.85 | 1484.20 | 1494.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 1504.50 | 1488.26 | 1495.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 10:15:00 | 1474.65 | 1489.71 | 1495.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 14:30:00 | 1469.40 | 1489.46 | 1493.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 12:45:00 | 1479.00 | 1453.76 | 1461.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 14:15:00 | 1508.70 | 1468.31 | 1467.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-11 14:15:00 | 1508.70 | 1468.31 | 1467.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 12:15:00 | 1525.50 | 1498.39 | 1486.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 09:15:00 | 1479.65 | 1499.78 | 1491.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 09:15:00 | 1479.65 | 1499.78 | 1491.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1479.65 | 1499.78 | 1491.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:45:00 | 1491.00 | 1499.78 | 1491.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 1464.70 | 1492.76 | 1489.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 1464.70 | 1492.76 | 1489.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 12:15:00 | 1461.70 | 1484.84 | 1485.95 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 14:15:00 | 1502.15 | 1486.83 | 1486.57 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 10:15:00 | 1458.50 | 1481.73 | 1484.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 11:15:00 | 1455.30 | 1476.45 | 1482.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 09:15:00 | 1512.30 | 1470.71 | 1475.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 09:15:00 | 1512.30 | 1470.71 | 1475.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1512.30 | 1470.71 | 1475.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:45:00 | 1513.00 | 1470.71 | 1475.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 1492.00 | 1474.97 | 1476.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:00:00 | 1492.00 | 1474.97 | 1476.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 11:15:00 | 1497.85 | 1479.55 | 1478.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 1504.90 | 1489.20 | 1484.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 1493.50 | 1503.66 | 1494.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 09:15:00 | 1493.50 | 1503.66 | 1494.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 1493.50 | 1503.66 | 1494.50 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 15:15:00 | 1472.35 | 1488.13 | 1489.77 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 09:15:00 | 1502.70 | 1489.90 | 1488.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 13:15:00 | 1511.15 | 1498.24 | 1493.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 15:15:00 | 1481.20 | 1495.78 | 1493.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 15:15:00 | 1481.20 | 1495.78 | 1493.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 1481.20 | 1495.78 | 1493.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 1535.00 | 1495.78 | 1493.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 1556.45 | 1588.97 | 1591.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 09:15:00 | 1556.45 | 1588.97 | 1591.53 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 1610.00 | 1592.76 | 1591.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1627.50 | 1600.87 | 1595.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 1600.20 | 1603.54 | 1598.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 11:15:00 | 1600.20 | 1603.54 | 1598.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 1600.20 | 1603.54 | 1598.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:45:00 | 1601.50 | 1603.54 | 1598.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1610.10 | 1607.56 | 1602.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:30:00 | 1607.35 | 1607.56 | 1602.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 1602.10 | 1606.47 | 1602.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:45:00 | 1602.30 | 1606.47 | 1602.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 1601.25 | 1605.42 | 1602.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 1601.25 | 1605.42 | 1602.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 1601.95 | 1604.73 | 1602.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:45:00 | 1601.95 | 1604.73 | 1602.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 1602.10 | 1604.20 | 1602.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 1600.00 | 1604.20 | 1602.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 1602.10 | 1603.20 | 1602.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 1587.45 | 1603.20 | 1602.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1599.05 | 1602.37 | 1601.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 1591.05 | 1602.37 | 1601.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1636.65 | 1609.22 | 1604.96 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 1602.40 | 1609.88 | 1609.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 1598.80 | 1606.07 | 1608.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 12:15:00 | 1605.65 | 1603.61 | 1606.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 13:00:00 | 1605.65 | 1603.61 | 1606.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 1608.40 | 1604.57 | 1606.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:00:00 | 1608.40 | 1604.57 | 1606.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1596.30 | 1602.91 | 1605.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:30:00 | 1585.50 | 1598.81 | 1602.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 1556.40 | 1598.05 | 1601.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 10:15:00 | 1642.70 | 1592.03 | 1591.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 1642.70 | 1592.03 | 1591.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 14:15:00 | 1690.95 | 1640.74 | 1633.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 09:15:00 | 1752.85 | 1788.57 | 1751.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 1752.85 | 1788.57 | 1751.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1752.85 | 1788.57 | 1751.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 1752.70 | 1788.57 | 1751.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 1741.05 | 1779.07 | 1750.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 1741.05 | 1779.07 | 1750.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 1737.00 | 1770.66 | 1748.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 1737.00 | 1770.66 | 1748.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 1736.00 | 1763.72 | 1747.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:00:00 | 1736.00 | 1763.72 | 1747.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 1703.95 | 1739.40 | 1739.61 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 1788.85 | 1746.57 | 1741.12 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 1698.00 | 1740.24 | 1742.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 11:15:00 | 1684.15 | 1729.02 | 1737.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 15:15:00 | 1726.00 | 1720.56 | 1729.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 09:15:00 | 1719.00 | 1720.56 | 1729.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1721.95 | 1720.83 | 1729.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:15:00 | 1743.00 | 1720.83 | 1729.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1760.20 | 1728.71 | 1731.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:45:00 | 1758.80 | 1728.71 | 1731.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 1768.00 | 1736.57 | 1735.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 1784.75 | 1754.65 | 1744.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 15:15:00 | 1754.00 | 1754.52 | 1745.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 15:15:00 | 1754.00 | 1754.52 | 1745.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 1754.00 | 1754.52 | 1745.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 1749.30 | 1754.52 | 1745.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1733.00 | 1750.22 | 1744.30 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1692.60 | 1736.52 | 1740.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1553.55 | 1675.63 | 1705.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1692.40 | 1615.97 | 1650.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 1692.40 | 1615.97 | 1650.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1692.40 | 1615.97 | 1650.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:00:00 | 1692.40 | 1615.97 | 1650.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 1689.95 | 1630.77 | 1654.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:30:00 | 1683.00 | 1630.77 | 1654.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 1708.75 | 1666.78 | 1665.60 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 10:15:00 | 1638.80 | 1662.87 | 1664.48 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 1723.30 | 1673.56 | 1667.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 1750.95 | 1689.04 | 1674.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 1785.60 | 1793.59 | 1759.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 10:15:00 | 1778.40 | 1793.59 | 1759.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1947.60 | 1951.29 | 1924.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:15:00 | 1953.30 | 1951.29 | 1924.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 11:30:00 | 1951.90 | 1951.93 | 1939.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 15:15:00 | 1956.60 | 1948.59 | 1940.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:00:00 | 1951.50 | 1945.38 | 1941.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1912.30 | 1953.53 | 1948.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-28 09:15:00 | 1912.30 | 1953.53 | 1948.88 | SL hit (close<static) qty=1.00 sl=1921.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 10:15:00 | 1905.20 | 1943.87 | 1944.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 11:15:00 | 1892.70 | 1933.63 | 1940.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 1924.10 | 1912.64 | 1925.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 1924.10 | 1912.64 | 1925.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1924.10 | 1912.64 | 1925.09 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 15:15:00 | 1960.00 | 1927.25 | 1926.39 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 1911.50 | 1924.40 | 1925.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 1867.80 | 1913.08 | 1920.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1830.10 | 1829.04 | 1858.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 10:00:00 | 1830.10 | 1829.04 | 1858.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 1856.00 | 1837.27 | 1853.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 1856.00 | 1837.27 | 1853.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 1857.30 | 1841.27 | 1853.44 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 10:15:00 | 1890.10 | 1860.35 | 1859.81 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 14:15:00 | 1865.10 | 1868.59 | 1868.83 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 1874.00 | 1869.67 | 1869.30 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 10:15:00 | 1851.60 | 1865.60 | 1867.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 11:15:00 | 1845.40 | 1861.56 | 1865.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1818.70 | 1768.64 | 1798.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1818.70 | 1768.64 | 1798.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1818.70 | 1768.64 | 1798.96 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 1846.70 | 1798.03 | 1796.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 1858.00 | 1836.57 | 1820.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 13:15:00 | 1864.10 | 1869.01 | 1853.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 14:00:00 | 1864.10 | 1869.01 | 1853.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1871.80 | 1868.55 | 1857.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 1856.30 | 1868.55 | 1857.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1870.20 | 1880.56 | 1871.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 1873.30 | 1880.56 | 1871.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1862.00 | 1876.85 | 1870.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 1862.00 | 1876.85 | 1870.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 1850.00 | 1871.48 | 1868.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:00:00 | 1850.00 | 1871.48 | 1868.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1844.70 | 1866.12 | 1866.39 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 1881.80 | 1865.60 | 1865.20 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 14:15:00 | 1858.00 | 1865.41 | 1865.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 1845.00 | 1861.26 | 1863.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 1841.50 | 1840.64 | 1850.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 14:15:00 | 1841.50 | 1840.64 | 1850.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1841.50 | 1840.64 | 1850.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 1841.50 | 1840.64 | 1850.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 1840.00 | 1840.52 | 1849.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 1822.10 | 1840.52 | 1849.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 1817.70 | 1828.28 | 1831.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1861.40 | 1835.34 | 1833.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 1861.40 | 1835.34 | 1833.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 1892.60 | 1855.45 | 1843.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 1916.50 | 1933.23 | 1902.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 10:00:00 | 1916.50 | 1933.23 | 1902.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1896.10 | 1918.46 | 1909.27 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 1868.10 | 1898.71 | 1901.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 13:15:00 | 1862.80 | 1891.53 | 1898.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 15:15:00 | 1889.90 | 1889.41 | 1895.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 1900.80 | 1884.62 | 1888.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1900.80 | 1884.62 | 1888.39 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 1904.10 | 1891.55 | 1891.07 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 1878.90 | 1889.02 | 1889.96 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 14:15:00 | 1895.10 | 1890.73 | 1890.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 1910.80 | 1897.57 | 1893.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 11:15:00 | 1895.60 | 1897.18 | 1894.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 11:15:00 | 1895.60 | 1897.18 | 1894.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 1895.60 | 1897.18 | 1894.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 1898.00 | 1897.18 | 1894.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 1884.10 | 1894.56 | 1893.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 1884.10 | 1894.56 | 1893.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 13:15:00 | 1839.10 | 1883.47 | 1888.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 14:15:00 | 1828.40 | 1872.46 | 1882.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 10:15:00 | 1884.90 | 1821.80 | 1837.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 10:15:00 | 1884.90 | 1821.80 | 1837.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1884.90 | 1821.80 | 1837.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 1884.90 | 1821.80 | 1837.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 1913.00 | 1840.04 | 1844.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 1913.00 | 1840.04 | 1844.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 1939.40 | 1859.91 | 1853.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 10:15:00 | 1997.10 | 1923.40 | 1889.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 1988.50 | 1993.79 | 1973.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:45:00 | 1990.30 | 1993.79 | 1973.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1967.00 | 1986.83 | 1973.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 1985.80 | 1989.66 | 1975.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 12:15:00 | 1962.70 | 1983.56 | 1976.52 | SL hit (close<static) qty=1.00 sl=1965.10 alert=retest2 |

### Cycle 96 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 1960.00 | 1972.19 | 1972.48 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 13:15:00 | 1996.90 | 1975.83 | 1973.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 2027.30 | 1998.88 | 1989.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 13:15:00 | 2011.10 | 2012.95 | 2000.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 14:00:00 | 2011.10 | 2012.95 | 2000.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2004.20 | 2011.55 | 2002.90 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 1981.30 | 1997.48 | 1997.94 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 14:15:00 | 2020.20 | 2000.97 | 1999.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 09:15:00 | 2036.00 | 2011.02 | 2004.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 2001.90 | 2014.01 | 2007.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 11:15:00 | 2001.90 | 2014.01 | 2007.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 2001.90 | 2014.01 | 2007.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:30:00 | 2007.20 | 2014.01 | 2007.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 2000.00 | 2011.21 | 2006.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:30:00 | 1987.50 | 2011.21 | 2006.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 2000.70 | 2009.11 | 2006.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 2001.10 | 2009.11 | 2006.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 1977.10 | 2002.70 | 2003.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 1955.90 | 1993.34 | 1999.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 2001.20 | 1994.92 | 1999.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 2001.20 | 1994.92 | 1999.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 2001.20 | 1994.92 | 1999.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:30:00 | 2001.80 | 1994.92 | 1999.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 2001.20 | 1996.17 | 1999.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 1997.20 | 1996.17 | 1999.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1986.40 | 1994.22 | 1998.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 1976.10 | 1994.22 | 1998.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:00:00 | 1979.50 | 1991.27 | 1996.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:45:00 | 1980.50 | 1989.64 | 1995.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 2017.80 | 1995.27 | 1997.38 | SL hit (close>static) qty=1.00 sl=2005.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 2038.00 | 2003.82 | 2001.07 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 12:15:00 | 1990.50 | 2000.15 | 2000.37 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 2004.90 | 2001.10 | 2000.78 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 14:15:00 | 1994.30 | 1999.74 | 2000.19 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 2061.00 | 2011.20 | 2005.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 2074.70 | 2032.70 | 2016.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 15:15:00 | 2035.00 | 2042.78 | 2027.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 15:15:00 | 2035.00 | 2042.78 | 2027.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 2035.00 | 2042.78 | 2027.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:15:00 | 2047.30 | 2042.78 | 2027.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 2058.50 | 2045.92 | 2030.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:45:00 | 2076.10 | 2057.38 | 2045.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 2101.20 | 2075.06 | 2061.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 11:15:00 | 2102.10 | 2122.25 | 2123.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 2102.10 | 2122.25 | 2123.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 12:15:00 | 2083.20 | 2114.44 | 2120.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 2099.30 | 2090.81 | 2100.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 2099.30 | 2090.81 | 2100.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 2099.30 | 2090.81 | 2100.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 2099.30 | 2090.81 | 2100.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 2098.00 | 2092.25 | 2100.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 2055.20 | 2092.25 | 2100.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 2087.20 | 2075.77 | 2075.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 2087.20 | 2075.77 | 2075.67 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 2061.10 | 2075.34 | 2076.07 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 2094.80 | 2079.23 | 2077.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 13:15:00 | 2111.90 | 2087.67 | 2082.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 11:15:00 | 2088.00 | 2099.73 | 2091.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 11:15:00 | 2088.00 | 2099.73 | 2091.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 2088.00 | 2099.73 | 2091.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:30:00 | 2089.20 | 2099.73 | 2091.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 2087.90 | 2097.36 | 2091.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:00:00 | 2087.90 | 2097.36 | 2091.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 2069.70 | 2091.83 | 2089.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 2069.70 | 2091.83 | 2089.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 2064.80 | 2084.76 | 2086.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 2043.50 | 2076.50 | 2082.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 2079.80 | 2057.60 | 2065.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 10:15:00 | 2079.80 | 2057.60 | 2065.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 2079.80 | 2057.60 | 2065.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 2079.80 | 2057.60 | 2065.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 2072.70 | 2060.62 | 2066.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 2086.20 | 2060.62 | 2066.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 2077.80 | 2064.06 | 2067.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 2077.80 | 2064.06 | 2067.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 2077.70 | 2066.79 | 2068.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:15:00 | 2076.80 | 2066.79 | 2068.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 2076.00 | 2070.37 | 2069.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 2099.10 | 2076.12 | 2072.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 2101.50 | 2123.38 | 2110.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 2101.50 | 2123.38 | 2110.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 2101.50 | 2123.38 | 2110.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:15:00 | 2099.80 | 2123.38 | 2110.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 2106.60 | 2120.03 | 2110.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 2098.00 | 2120.03 | 2110.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 2111.80 | 2118.38 | 2110.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:30:00 | 2125.30 | 2116.16 | 2110.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 2031.20 | 2100.08 | 2104.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 2031.20 | 2100.08 | 2104.84 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 2140.00 | 2106.00 | 2103.98 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 2071.30 | 2099.06 | 2101.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 14:15:00 | 2064.50 | 2080.57 | 2088.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 15:15:00 | 2083.80 | 2081.22 | 2088.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:15:00 | 2081.60 | 2081.22 | 2088.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 2080.50 | 2081.08 | 2087.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:15:00 | 2047.50 | 2068.87 | 2079.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 1945.12 | 1995.74 | 2027.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 1918.70 | 1913.58 | 1946.54 | SL hit (close>ema200) qty=0.50 sl=1913.58 alert=retest2 |

### Cycle 115 — BUY (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 13:15:00 | 1838.30 | 1816.23 | 1815.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 14:15:00 | 1849.30 | 1822.84 | 1818.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 1846.10 | 1849.70 | 1839.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 1846.10 | 1849.70 | 1839.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1846.10 | 1849.70 | 1839.06 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 11:15:00 | 1832.70 | 1839.75 | 1840.14 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 13:15:00 | 1844.00 | 1840.62 | 1840.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 15:15:00 | 1860.00 | 1846.38 | 1843.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 1846.80 | 1848.28 | 1844.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 11:00:00 | 1846.80 | 1848.28 | 1844.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 1864.00 | 1851.42 | 1846.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:45:00 | 1867.60 | 1856.51 | 1849.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:15:00 | 1871.00 | 1856.51 | 1849.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 13:15:00 | 1846.30 | 1894.70 | 1896.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 1846.30 | 1894.70 | 1896.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 09:15:00 | 1842.90 | 1871.29 | 1884.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 1883.30 | 1867.82 | 1874.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 1883.30 | 1867.82 | 1874.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1883.30 | 1867.82 | 1874.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1890.00 | 1867.82 | 1874.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1880.60 | 1870.38 | 1875.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:30:00 | 1874.40 | 1871.00 | 1875.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:45:00 | 1879.10 | 1872.44 | 1875.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 14:15:00 | 1885.30 | 1875.10 | 1876.23 | SL hit (close>static) qty=1.00 sl=1885.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 12:15:00 | 1761.90 | 1758.80 | 1758.42 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 09:15:00 | 1749.80 | 1756.96 | 1757.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 10:15:00 | 1742.40 | 1754.05 | 1756.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 1773.60 | 1743.98 | 1748.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 10:15:00 | 1773.60 | 1743.98 | 1748.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1773.60 | 1743.98 | 1748.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 1773.60 | 1743.98 | 1748.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 1754.50 | 1746.08 | 1748.72 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 1769.90 | 1753.01 | 1751.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 1772.80 | 1756.97 | 1753.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 1868.50 | 1873.47 | 1849.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 1868.50 | 1873.47 | 1849.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 1861.00 | 1870.97 | 1850.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 1850.20 | 1870.97 | 1850.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1862.80 | 1869.34 | 1852.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:45:00 | 1870.10 | 1869.55 | 1853.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 1939.90 | 1968.79 | 1971.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 1939.90 | 1968.79 | 1971.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 13:15:00 | 1920.80 | 1959.19 | 1967.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 1870.80 | 1832.88 | 1851.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 13:15:00 | 1870.80 | 1832.88 | 1851.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 1870.80 | 1832.88 | 1851.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 1870.80 | 1832.88 | 1851.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1949.40 | 1856.19 | 1860.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 1949.40 | 1856.19 | 1860.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 15:15:00 | 1930.80 | 1871.11 | 1866.96 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 1829.70 | 1876.23 | 1877.66 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1910.90 | 1881.94 | 1879.39 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 10:15:00 | 1851.00 | 1874.49 | 1876.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 11:15:00 | 1836.80 | 1866.95 | 1873.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 09:15:00 | 1855.40 | 1850.88 | 1861.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 1855.40 | 1850.88 | 1861.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1855.40 | 1850.88 | 1861.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 1855.40 | 1850.88 | 1861.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1871.80 | 1855.06 | 1862.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 1871.80 | 1855.06 | 1862.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1875.00 | 1859.05 | 1863.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 1882.80 | 1859.05 | 1863.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 1889.50 | 1870.36 | 1867.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 11:15:00 | 1893.40 | 1880.21 | 1873.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 13:15:00 | 1878.80 | 1880.76 | 1875.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 13:15:00 | 1878.80 | 1880.76 | 1875.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1878.80 | 1880.76 | 1875.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 1878.80 | 1880.76 | 1875.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1888.80 | 1882.37 | 1876.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 1888.80 | 1882.37 | 1876.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1864.00 | 1880.70 | 1876.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 1864.00 | 1880.70 | 1876.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1855.10 | 1875.58 | 1874.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1855.10 | 1875.58 | 1874.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 1856.60 | 1871.78 | 1873.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 1842.00 | 1863.09 | 1868.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 1835.30 | 1827.53 | 1843.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 1835.30 | 1827.53 | 1843.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1835.30 | 1827.53 | 1843.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1835.30 | 1827.53 | 1843.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1846.30 | 1832.77 | 1843.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 1856.20 | 1832.77 | 1843.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1851.00 | 1836.41 | 1843.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 1850.10 | 1836.41 | 1843.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 1835.30 | 1838.48 | 1843.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:30:00 | 1843.80 | 1838.48 | 1843.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1815.10 | 1832.48 | 1839.11 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 1859.50 | 1837.52 | 1835.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 1878.30 | 1845.68 | 1839.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 11:15:00 | 1880.40 | 1885.10 | 1866.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 12:00:00 | 1880.40 | 1885.10 | 1866.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1855.20 | 1881.56 | 1872.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 1855.20 | 1881.56 | 1872.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1866.50 | 1878.55 | 1871.52 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1852.80 | 1865.85 | 1867.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 11:15:00 | 1842.90 | 1858.23 | 1863.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 1859.90 | 1854.55 | 1859.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 1859.90 | 1854.55 | 1859.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1859.90 | 1854.55 | 1859.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1855.80 | 1854.55 | 1859.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1855.00 | 1854.64 | 1858.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 1855.00 | 1854.64 | 1858.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1854.60 | 1854.63 | 1858.39 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 1893.30 | 1862.37 | 1861.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 1936.60 | 1894.41 | 1879.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 12:15:00 | 1969.40 | 1974.90 | 1944.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-27 13:00:00 | 1969.40 | 1974.90 | 1944.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1932.90 | 1957.51 | 1948.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 1932.90 | 1957.51 | 1948.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1943.20 | 1954.65 | 1948.00 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 1908.80 | 1939.66 | 1942.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 10:15:00 | 1904.30 | 1932.59 | 1939.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 11:15:00 | 1915.20 | 1914.83 | 1923.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 12:00:00 | 1915.20 | 1914.83 | 1923.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 1915.10 | 1914.88 | 1922.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:30:00 | 1916.40 | 1914.88 | 1922.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1919.50 | 1915.81 | 1922.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:00:00 | 1919.50 | 1915.81 | 1922.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 1921.60 | 1916.97 | 1922.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:30:00 | 1920.00 | 1916.97 | 1922.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 1915.00 | 1916.57 | 1921.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 1924.90 | 1916.57 | 1921.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1922.30 | 1917.72 | 1921.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:30:00 | 1926.80 | 1917.72 | 1921.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1920.00 | 1918.17 | 1921.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 1920.40 | 1918.17 | 1921.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1920.00 | 1918.54 | 1921.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:00:00 | 1899.90 | 1915.31 | 1919.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 1962.30 | 1922.87 | 1922.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1962.30 | 1922.87 | 1922.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 10:15:00 | 1978.70 | 1934.03 | 1927.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 13:15:00 | 1971.90 | 2011.29 | 1997.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 13:15:00 | 1971.90 | 2011.29 | 1997.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 1971.90 | 2011.29 | 1997.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:00:00 | 1971.90 | 2011.29 | 1997.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1978.80 | 2004.79 | 1995.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:15:00 | 1961.50 | 2004.79 | 1995.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 1932.50 | 1983.41 | 1987.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 13:15:00 | 1900.00 | 1944.08 | 1965.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 14:15:00 | 1912.00 | 1910.17 | 1931.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 15:00:00 | 1912.00 | 1910.17 | 1931.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1928.50 | 1915.51 | 1930.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:45:00 | 1930.60 | 1915.51 | 1930.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1934.20 | 1919.25 | 1930.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 1934.20 | 1919.25 | 1930.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 1938.10 | 1923.02 | 1931.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:00:00 | 1938.10 | 1923.02 | 1931.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1932.90 | 1924.99 | 1931.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 1923.10 | 1929.23 | 1932.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 1950.40 | 1934.91 | 1933.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 1950.40 | 1934.91 | 1933.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 1960.60 | 1942.27 | 1937.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 1940.50 | 1944.86 | 1940.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1940.50 | 1944.86 | 1940.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1940.50 | 1944.86 | 1940.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 1938.90 | 1944.86 | 1940.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1951.20 | 1946.13 | 1941.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:30:00 | 1962.50 | 1949.06 | 1943.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 14:15:00 | 1960.90 | 1954.24 | 1950.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 10:30:00 | 1962.50 | 1959.73 | 1955.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:15:00 | 1964.00 | 1957.96 | 1954.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 1951.60 | 1957.02 | 1954.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:00:00 | 1951.60 | 1957.02 | 1954.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1973.70 | 1960.35 | 1956.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-18 12:15:00 | 1927.40 | 1950.09 | 1953.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 1927.40 | 1950.09 | 1953.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 13:15:00 | 1923.30 | 1944.73 | 1950.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 11:15:00 | 1894.10 | 1893.54 | 1909.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 12:00:00 | 1894.10 | 1893.54 | 1909.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1845.90 | 1832.27 | 1843.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 1845.90 | 1832.27 | 1843.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1856.00 | 1837.02 | 1844.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 1867.30 | 1837.02 | 1844.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1857.20 | 1841.05 | 1845.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 1858.10 | 1841.05 | 1845.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1855.60 | 1843.96 | 1846.46 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1856.50 | 1849.02 | 1848.49 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 13:15:00 | 1842.30 | 1847.68 | 1847.92 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 1853.50 | 1848.97 | 1848.48 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 1828.90 | 1844.96 | 1846.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 1803.10 | 1833.15 | 1840.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 10:15:00 | 1817.40 | 1814.58 | 1826.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 10:15:00 | 1817.40 | 1814.58 | 1826.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1817.40 | 1814.58 | 1826.03 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 1853.90 | 1832.61 | 1830.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 10:15:00 | 1873.80 | 1840.85 | 1834.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 1861.60 | 1877.35 | 1860.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 1861.60 | 1877.35 | 1860.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1861.60 | 1877.35 | 1860.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 1856.40 | 1877.35 | 1860.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1865.50 | 1874.98 | 1860.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 1880.00 | 1874.70 | 1865.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 1829.40 | 1865.32 | 1862.43 | SL hit (close<static) qty=1.00 sl=1856.30 alert=retest2 |

### Cycle 142 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 1878.80 | 1891.97 | 1892.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 1854.40 | 1884.45 | 1888.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 1886.30 | 1884.82 | 1888.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 1886.30 | 1884.82 | 1888.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1909.30 | 1889.72 | 1890.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 1909.30 | 1889.72 | 1890.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 1921.80 | 1896.14 | 1893.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 13:15:00 | 1935.20 | 1903.95 | 1897.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 12:15:00 | 1925.70 | 1929.90 | 1915.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 13:00:00 | 1925.70 | 1929.90 | 1915.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1916.80 | 1927.47 | 1917.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 1916.80 | 1927.47 | 1917.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1912.10 | 1924.40 | 1916.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1908.60 | 1924.40 | 1916.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1922.40 | 1924.00 | 1917.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 1903.60 | 1924.00 | 1917.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1912.10 | 1921.62 | 1916.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 1912.10 | 1921.62 | 1916.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1918.30 | 1920.95 | 1916.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 1918.30 | 1920.95 | 1916.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1931.30 | 1926.00 | 1921.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 12:00:00 | 1946.80 | 1931.12 | 1924.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 1894.30 | 1928.09 | 1926.33 | SL hit (close<static) qty=1.00 sl=1917.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 1911.10 | 1924.69 | 1924.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 1884.20 | 1899.10 | 1909.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1825.30 | 1810.10 | 1828.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:45:00 | 1820.20 | 1810.10 | 1828.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1830.90 | 1815.16 | 1823.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 1831.30 | 1815.16 | 1823.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1829.70 | 1818.07 | 1824.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 1808.00 | 1818.07 | 1824.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:30:00 | 1818.50 | 1819.01 | 1823.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:30:00 | 1820.80 | 1820.94 | 1823.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 1844.80 | 1822.79 | 1823.58 | SL hit (close>static) qty=1.00 sl=1831.20 alert=retest2 |

### Cycle 145 — BUY (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 10:15:00 | 1870.00 | 1832.23 | 1827.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 11:15:00 | 1901.50 | 1846.09 | 1834.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 10:15:00 | 1877.80 | 1883.47 | 1869.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 1877.80 | 1883.47 | 1869.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 1871.10 | 1879.60 | 1870.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 1871.10 | 1879.60 | 1870.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 1861.60 | 1876.00 | 1869.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 1861.60 | 1876.00 | 1869.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1857.90 | 1872.38 | 1868.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 1858.30 | 1872.38 | 1868.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 1841.30 | 1864.34 | 1865.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 1815.20 | 1841.15 | 1850.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 1841.90 | 1824.13 | 1831.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 1841.90 | 1824.13 | 1831.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1841.90 | 1824.13 | 1831.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 1852.50 | 1824.13 | 1831.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1854.40 | 1830.18 | 1833.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 1854.40 | 1830.18 | 1833.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 1856.70 | 1835.49 | 1835.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 1869.90 | 1842.37 | 1838.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 1893.60 | 1914.36 | 1899.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 13:15:00 | 1893.60 | 1914.36 | 1899.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1893.60 | 1914.36 | 1899.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 1893.60 | 1914.36 | 1899.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1930.90 | 1917.67 | 1901.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 1890.10 | 1917.67 | 1901.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1896.90 | 1913.11 | 1902.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 1891.90 | 1913.11 | 1902.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1876.50 | 1905.78 | 1900.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 1876.50 | 1905.78 | 1900.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1887.50 | 1902.13 | 1899.02 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 1879.30 | 1894.21 | 1895.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 1873.70 | 1887.16 | 1891.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 14:15:00 | 1864.10 | 1862.95 | 1876.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 15:00:00 | 1864.10 | 1862.95 | 1876.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1825.90 | 1817.13 | 1832.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1831.00 | 1817.13 | 1832.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1813.90 | 1816.48 | 1830.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 1796.60 | 1816.01 | 1824.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 1803.50 | 1811.38 | 1819.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 1801.00 | 1813.77 | 1818.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 1706.77 | 1750.04 | 1769.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 1713.32 | 1750.04 | 1769.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 1710.95 | 1750.04 | 1769.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1680.90 | 1663.29 | 1692.48 | SL hit (close>ema200) qty=0.50 sl=1663.29 alert=retest2 |

### Cycle 149 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 1641.90 | 1627.74 | 1627.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 1650.00 | 1632.19 | 1629.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1639.00 | 1647.28 | 1640.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 1639.00 | 1647.28 | 1640.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1639.00 | 1647.28 | 1640.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 1636.20 | 1647.28 | 1640.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1668.90 | 1651.61 | 1643.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:30:00 | 1643.90 | 1651.61 | 1643.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1683.20 | 1659.81 | 1648.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1647.30 | 1659.81 | 1648.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1631.90 | 1672.02 | 1660.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 1631.90 | 1672.02 | 1660.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 1617.50 | 1661.11 | 1656.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 1617.50 | 1661.11 | 1656.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1613.10 | 1651.51 | 1652.25 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1680.30 | 1653.27 | 1651.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 1732.80 | 1679.30 | 1664.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 1702.20 | 1707.88 | 1688.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 1702.20 | 1707.88 | 1688.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1713.50 | 1706.35 | 1692.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:30:00 | 1691.30 | 1706.35 | 1692.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1687.20 | 1705.41 | 1695.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 1687.20 | 1705.41 | 1695.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1680.10 | 1700.35 | 1694.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 1680.10 | 1700.35 | 1694.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 1670.80 | 1690.06 | 1690.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 1663.80 | 1684.81 | 1688.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 15:15:00 | 1660.00 | 1659.44 | 1670.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 09:15:00 | 1703.00 | 1659.44 | 1670.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1707.50 | 1669.05 | 1673.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 1708.50 | 1669.05 | 1673.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1726.90 | 1680.62 | 1678.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 12:15:00 | 1738.80 | 1712.86 | 1700.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 1749.50 | 1754.40 | 1736.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 1749.50 | 1754.40 | 1736.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1749.50 | 1754.40 | 1736.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 1737.00 | 1754.40 | 1736.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1723.00 | 1749.52 | 1742.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 1721.80 | 1749.52 | 1742.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1721.80 | 1743.97 | 1740.99 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 1710.20 | 1733.43 | 1736.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 13:15:00 | 1707.50 | 1728.24 | 1733.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 1729.50 | 1721.42 | 1727.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 11:15:00 | 1729.50 | 1721.42 | 1727.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 1729.50 | 1721.42 | 1727.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 1729.50 | 1721.42 | 1727.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 1737.70 | 1724.68 | 1728.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 1737.70 | 1724.68 | 1728.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 1735.50 | 1731.33 | 1730.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 10:15:00 | 1757.60 | 1736.59 | 1733.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 1751.00 | 1752.58 | 1744.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 1751.00 | 1752.58 | 1744.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 1748.50 | 1751.72 | 1746.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:30:00 | 1745.30 | 1751.72 | 1746.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1755.70 | 1752.51 | 1747.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:30:00 | 1748.20 | 1752.51 | 1747.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1732.80 | 1748.42 | 1746.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 1736.10 | 1748.42 | 1746.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 1725.80 | 1743.90 | 1744.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 1713.00 | 1734.70 | 1739.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 15:15:00 | 1690.00 | 1683.79 | 1695.74 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1641.00 | 1683.79 | 1695.74 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1679.90 | 1668.19 | 1679.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 1679.90 | 1668.19 | 1679.25 | SL hit (close>ema400) qty=1.00 sl=1679.25 alert=retest1 |

### Cycle 157 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 1583.90 | 1570.97 | 1570.24 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1504.10 | 1559.37 | 1565.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 1489.20 | 1528.42 | 1546.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 13:15:00 | 1515.70 | 1500.75 | 1521.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 13:15:00 | 1515.70 | 1500.75 | 1521.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 1515.70 | 1500.75 | 1521.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:00:00 | 1515.70 | 1500.75 | 1521.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 1557.10 | 1512.02 | 1524.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 1557.10 | 1512.02 | 1524.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 1550.00 | 1519.62 | 1526.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 1507.90 | 1519.62 | 1526.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 12:45:00 | 1544.30 | 1526.66 | 1527.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 13:15:00 | 1555.50 | 1532.42 | 1529.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 13:15:00 | 1555.50 | 1532.42 | 1529.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 14:15:00 | 1571.00 | 1540.14 | 1533.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 09:15:00 | 1536.30 | 1544.15 | 1536.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 1536.30 | 1544.15 | 1536.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1536.30 | 1544.15 | 1536.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 1529.80 | 1544.15 | 1536.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1542.70 | 1543.86 | 1537.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 1528.40 | 1543.86 | 1537.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 1529.70 | 1541.03 | 1536.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:00:00 | 1529.70 | 1541.03 | 1536.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 1532.30 | 1539.28 | 1536.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:30:00 | 1530.70 | 1539.28 | 1536.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1523.20 | 1535.08 | 1534.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 15:00:00 | 1523.20 | 1535.08 | 1534.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 15:15:00 | 1515.00 | 1531.06 | 1533.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 10:15:00 | 1509.50 | 1525.09 | 1529.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 15:15:00 | 1370.00 | 1355.39 | 1391.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 09:15:00 | 1355.00 | 1355.39 | 1391.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1315.20 | 1347.35 | 1384.55 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1412.00 | 1383.59 | 1381.91 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1368.20 | 1384.13 | 1384.72 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 1389.30 | 1384.10 | 1383.64 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 14:15:00 | 1361.60 | 1379.60 | 1381.64 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 1407.00 | 1382.23 | 1381.41 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1332.80 | 1374.57 | 1378.61 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 1397.40 | 1379.70 | 1378.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1417.20 | 1391.16 | 1384.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 1405.10 | 1407.31 | 1396.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 1405.10 | 1407.31 | 1396.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 1384.00 | 1401.93 | 1395.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:00:00 | 1384.00 | 1401.93 | 1395.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 1388.50 | 1399.24 | 1395.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:45:00 | 1385.60 | 1399.24 | 1395.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 1408.80 | 1401.15 | 1396.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:30:00 | 1379.10 | 1401.15 | 1396.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 1406.50 | 1402.22 | 1397.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1478.00 | 1402.22 | 1397.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 13:15:00 | 1625.80 | 1520.71 | 1463.77 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 1572.00 | 1599.38 | 1603.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 1562.30 | 1580.63 | 1591.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 11:15:00 | 1578.00 | 1577.60 | 1587.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 12:00:00 | 1578.00 | 1577.60 | 1587.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 1604.00 | 1582.88 | 1589.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:30:00 | 1606.50 | 1582.88 | 1589.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 1593.50 | 1585.01 | 1589.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:45:00 | 1598.50 | 1585.01 | 1589.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 1590.00 | 1585.08 | 1588.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 1588.50 | 1585.08 | 1588.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1590.70 | 1586.20 | 1589.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 15:00:00 | 1574.10 | 1585.20 | 1587.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 1540.50 | 1584.34 | 1587.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 15:00:00 | 1569.60 | 1559.66 | 1569.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:15:00 | 1571.30 | 1563.74 | 1568.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1581.60 | 1567.31 | 1569.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:45:00 | 1581.30 | 1567.31 | 1569.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-27 13:15:00 | 1587.50 | 1571.35 | 1571.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1587.50 | 1571.35 | 1571.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 1589.10 | 1574.90 | 1572.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 1563.00 | 1574.46 | 1573.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 1563.00 | 1574.46 | 1573.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1563.00 | 1574.46 | 1573.11 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 10:15:00 | 1555.80 | 1570.73 | 1571.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 11:15:00 | 1543.00 | 1565.18 | 1568.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 15:15:00 | 1552.00 | 1547.25 | 1553.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 15:15:00 | 1552.00 | 1547.25 | 1553.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1552.00 | 1547.25 | 1553.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1542.70 | 1547.25 | 1553.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1540.20 | 1545.84 | 1552.50 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1663.00 | 1575.36 | 1564.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1726.20 | 1605.53 | 1578.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 12:15:00 | 1711.60 | 1715.39 | 1683.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 12:45:00 | 1706.00 | 1715.39 | 1683.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-14 09:15:00 | 1085.00 | 2024-05-15 13:15:00 | 1081.15 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2024-05-21 11:15:00 | 1124.85 | 2024-05-23 10:15:00 | 1106.05 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-05-21 12:30:00 | 1130.20 | 2024-05-23 10:15:00 | 1106.05 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-05-22 09:45:00 | 1122.55 | 2024-05-23 10:15:00 | 1106.05 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-05-28 09:15:00 | 1141.45 | 2024-05-28 10:15:00 | 1118.45 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-06-13 09:15:00 | 1254.15 | 2024-06-21 11:15:00 | 1279.95 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2024-06-26 10:45:00 | 1308.55 | 2024-06-26 12:15:00 | 1290.40 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-06-27 09:45:00 | 1308.20 | 2024-06-27 13:15:00 | 1278.75 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-07-05 15:15:00 | 1452.00 | 2024-07-10 09:15:00 | 1489.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-07-08 09:30:00 | 1453.20 | 2024-07-10 09:15:00 | 1489.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-07-08 10:15:00 | 1449.65 | 2024-07-10 09:15:00 | 1489.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-07-08 10:45:00 | 1451.90 | 2024-07-10 09:15:00 | 1489.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2024-07-18 09:15:00 | 1508.95 | 2024-07-19 10:15:00 | 1474.80 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-08-08 14:30:00 | 1470.00 | 2024-08-09 09:15:00 | 1505.20 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-08-08 15:15:00 | 1461.10 | 2024-08-09 09:15:00 | 1505.20 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2024-08-13 09:30:00 | 1542.55 | 2024-08-23 13:15:00 | 1696.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-13 10:00:00 | 1543.95 | 2024-08-23 13:15:00 | 1698.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-14 09:15:00 | 1549.55 | 2024-08-23 13:15:00 | 1704.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-04 10:45:00 | 1735.80 | 2024-09-11 11:15:00 | 1784.85 | STOP_HIT | 1.00 | 2.83% |
| BUY | retest2 | 2024-09-04 14:45:00 | 1730.65 | 2024-09-11 11:15:00 | 1784.85 | STOP_HIT | 1.00 | 3.13% |
| BUY | retest2 | 2024-09-05 09:15:00 | 1770.55 | 2024-09-11 11:15:00 | 1784.85 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2024-09-20 10:00:00 | 1965.20 | 2024-09-27 14:15:00 | 1993.20 | STOP_HIT | 1.00 | 1.42% |
| SELL | retest2 | 2024-10-08 09:30:00 | 1924.15 | 2024-10-14 13:15:00 | 1928.85 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-10-08 10:00:00 | 1917.10 | 2024-10-14 13:15:00 | 1928.85 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-10-08 12:00:00 | 1929.40 | 2024-10-14 13:15:00 | 1928.85 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-10-08 14:30:00 | 1922.50 | 2024-10-14 13:15:00 | 1928.85 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-10-10 09:30:00 | 1927.15 | 2024-10-14 13:15:00 | 1928.85 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2024-10-18 11:30:00 | 2004.35 | 2024-10-22 09:15:00 | 1943.60 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-10-21 09:15:00 | 2008.65 | 2024-10-22 09:15:00 | 1943.60 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2024-10-21 10:00:00 | 2005.10 | 2024-10-22 09:15:00 | 1943.60 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2024-10-21 12:00:00 | 1998.30 | 2024-10-22 09:15:00 | 1943.60 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2024-10-30 11:30:00 | 1687.05 | 2024-11-01 18:15:00 | 1733.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2024-10-31 11:45:00 | 1684.90 | 2024-11-01 18:15:00 | 1733.00 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2024-10-31 13:30:00 | 1686.60 | 2024-11-01 18:15:00 | 1733.00 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2024-11-04 14:15:00 | 1742.95 | 2024-11-06 14:15:00 | 1713.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-11-13 09:15:00 | 1591.90 | 2024-11-18 10:15:00 | 1512.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-13 09:15:00 | 1591.90 | 2024-11-19 09:15:00 | 1580.15 | STOP_HIT | 0.50 | 0.74% |
| SELL | retest2 | 2024-11-26 13:30:00 | 1517.50 | 2024-11-26 15:15:00 | 1537.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-11-29 13:00:00 | 1593.75 | 2024-12-03 09:15:00 | 1523.25 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest2 | 2024-12-12 09:15:00 | 1521.55 | 2024-12-12 10:15:00 | 1539.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest1 | 2024-12-19 09:15:00 | 1476.45 | 2024-12-23 12:15:00 | 1402.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:15:00 | 1455.90 | 2024-12-24 11:15:00 | 1383.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 13:00:00 | 1453.55 | 2024-12-24 11:15:00 | 1380.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-12-19 09:15:00 | 1476.45 | 2024-12-26 14:15:00 | 1399.65 | STOP_HIT | 0.50 | 5.20% |
| SELL | retest2 | 2024-12-20 12:15:00 | 1455.90 | 2024-12-26 14:15:00 | 1399.65 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2024-12-20 13:00:00 | 1453.55 | 2024-12-26 14:15:00 | 1399.65 | STOP_HIT | 0.50 | 3.71% |
| BUY | retest2 | 2025-01-08 14:45:00 | 1537.95 | 2025-01-10 09:15:00 | 1521.95 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-01-09 09:15:00 | 1546.95 | 2025-01-10 09:15:00 | 1521.95 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-01-14 12:30:00 | 1466.70 | 2025-01-21 09:15:00 | 1462.10 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-01-14 15:00:00 | 1466.55 | 2025-01-21 09:15:00 | 1462.10 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-01-15 11:30:00 | 1459.65 | 2025-01-21 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-01-15 15:00:00 | 1465.35 | 2025-01-21 09:15:00 | 1462.10 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-01-16 10:15:00 | 1445.65 | 2025-01-21 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-01-16 10:45:00 | 1434.85 | 2025-01-21 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-01-16 13:45:00 | 1448.15 | 2025-01-21 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-01-16 14:15:00 | 1447.65 | 2025-01-21 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-01-17 10:15:00 | 1433.70 | 2025-01-21 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-01-20 09:30:00 | 1429.05 | 2025-01-21 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-01-21 09:15:00 | 1433.85 | 2025-01-21 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-01-27 09:15:00 | 1407.90 | 2025-01-28 14:15:00 | 1436.45 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-01-27 15:15:00 | 1403.25 | 2025-01-28 14:15:00 | 1436.45 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-02-07 10:15:00 | 1474.65 | 2025-02-11 14:15:00 | 1508.70 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-02-07 14:30:00 | 1469.40 | 2025-02-11 14:15:00 | 1508.70 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-02-11 12:45:00 | 1479.00 | 2025-02-11 14:15:00 | 1508.70 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-02-25 09:15:00 | 1535.00 | 2025-03-05 09:15:00 | 1556.45 | STOP_HIT | 1.00 | 1.40% |
| SELL | retest2 | 2025-03-13 14:30:00 | 1585.50 | 2025-03-18 10:15:00 | 1642.70 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2025-03-17 09:15:00 | 1556.40 | 2025-03-18 10:15:00 | 1642.70 | STOP_HIT | 1.00 | -5.54% |
| BUY | retest2 | 2025-04-23 11:15:00 | 1953.30 | 2025-04-28 09:15:00 | 1912.30 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-04-24 11:30:00 | 1951.90 | 2025-04-28 09:15:00 | 1912.30 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-04-24 15:15:00 | 1956.60 | 2025-04-28 09:15:00 | 1912.30 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-04-25 12:00:00 | 1951.50 | 2025-04-28 09:15:00 | 1912.30 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-05-23 09:15:00 | 1822.10 | 2025-05-27 11:15:00 | 1861.40 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-05-27 09:15:00 | 1817.70 | 2025-05-27 11:15:00 | 1861.40 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-06-12 09:30:00 | 1985.80 | 2025-06-12 12:15:00 | 1962.70 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-20 12:15:00 | 1976.10 | 2025-06-20 14:15:00 | 2017.80 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-06-20 13:00:00 | 1979.50 | 2025-06-20 14:15:00 | 2017.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-06-20 13:45:00 | 1980.50 | 2025-06-20 14:15:00 | 2017.80 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-06-27 10:45:00 | 2076.10 | 2025-07-03 11:15:00 | 2102.10 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2025-06-30 09:15:00 | 2101.20 | 2025-07-03 11:15:00 | 2102.10 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-07-07 09:15:00 | 2055.20 | 2025-07-09 11:15:00 | 2087.20 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-07-18 13:30:00 | 2125.30 | 2025-07-21 09:15:00 | 2031.20 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2025-07-24 14:15:00 | 2047.50 | 2025-07-28 09:15:00 | 1945.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 14:15:00 | 2047.50 | 2025-07-29 14:15:00 | 1918.70 | STOP_HIT | 0.50 | 6.29% |
| BUY | retest2 | 2025-08-13 13:45:00 | 1867.60 | 2025-08-19 13:15:00 | 1846.30 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-08-13 14:15:00 | 1871.00 | 2025-08-19 13:15:00 | 1846.30 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-08-21 11:30:00 | 1874.40 | 2025-08-21 14:15:00 | 1885.30 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-08-21 12:45:00 | 1879.10 | 2025-08-21 14:15:00 | 1885.30 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-08-21 15:15:00 | 1874.00 | 2025-08-26 15:15:00 | 1780.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 15:15:00 | 1874.00 | 2025-08-29 11:15:00 | 1760.10 | STOP_HIT | 0.50 | 6.08% |
| BUY | retest2 | 2025-09-15 10:45:00 | 1870.10 | 2025-09-23 12:15:00 | 1939.90 | STOP_HIT | 1.00 | 3.73% |
| SELL | retest2 | 2025-10-31 15:00:00 | 1899.90 | 2025-11-03 09:15:00 | 1962.30 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-11-11 15:15:00 | 1923.10 | 2025-11-12 11:15:00 | 1950.40 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-11-13 11:30:00 | 1962.50 | 2025-11-18 12:15:00 | 1927.40 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-11-14 14:15:00 | 1960.90 | 2025-11-18 12:15:00 | 1927.40 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-11-17 10:30:00 | 1962.50 | 2025-11-18 12:15:00 | 1927.40 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-11-17 12:15:00 | 1964.00 | 2025-11-18 12:15:00 | 1927.40 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-12-02 15:00:00 | 1880.00 | 2025-12-03 09:15:00 | 1829.40 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-12-03 12:15:00 | 1880.10 | 2025-12-08 15:15:00 | 1878.80 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-12-04 09:30:00 | 1882.80 | 2025-12-08 15:15:00 | 1878.80 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-12-04 13:45:00 | 1881.90 | 2025-12-08 15:15:00 | 1878.80 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-12-12 12:00:00 | 1946.80 | 2025-12-15 09:15:00 | 1894.30 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-12-22 09:15:00 | 1808.00 | 2025-12-23 09:15:00 | 1844.80 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-12-22 10:30:00 | 1818.50 | 2025-12-23 09:15:00 | 1844.80 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-12-22 12:30:00 | 1820.80 | 2025-12-23 09:15:00 | 1844.80 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-14 09:15:00 | 1796.60 | 2026-01-20 10:15:00 | 1706.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:15:00 | 1803.50 | 2026-01-20 10:15:00 | 1713.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:15:00 | 1801.00 | 2026-01-20 10:15:00 | 1710.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:15:00 | 1796.60 | 2026-01-22 09:15:00 | 1680.90 | STOP_HIT | 0.50 | 6.44% |
| SELL | retest2 | 2026-01-14 12:15:00 | 1803.50 | 2026-01-22 09:15:00 | 1680.90 | STOP_HIT | 0.50 | 6.80% |
| SELL | retest2 | 2026-01-14 15:15:00 | 1801.00 | 2026-01-22 09:15:00 | 1680.90 | STOP_HIT | 0.50 | 6.67% |
| SELL | retest1 | 2026-02-24 09:15:00 | 1641.00 | 2026-02-24 15:15:00 | 1679.90 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-02-25 13:45:00 | 1663.80 | 2026-03-02 09:15:00 | 1580.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:15:00 | 1623.90 | 2026-03-02 09:15:00 | 1574.43 | PARTIAL | 0.50 | 3.05% |
| SELL | retest2 | 2026-02-25 13:45:00 | 1663.80 | 2026-03-04 12:15:00 | 1604.40 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2026-02-26 10:15:00 | 1623.90 | 2026-03-04 12:15:00 | 1604.40 | STOP_HIT | 0.50 | 1.20% |
| SELL | retest2 | 2026-02-27 09:15:00 | 1657.30 | 2026-03-09 09:15:00 | 1542.70 | PARTIAL | 0.50 | 6.91% |
| SELL | retest2 | 2026-02-27 09:15:00 | 1657.30 | 2026-03-10 10:15:00 | 1547.20 | STOP_HIT | 0.50 | 6.64% |
| SELL | retest2 | 2026-03-16 09:15:00 | 1507.90 | 2026-03-16 13:15:00 | 1555.50 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2026-03-16 12:45:00 | 1544.30 | 2026-03-16 13:15:00 | 1555.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1478.00 | 2026-04-08 13:15:00 | 1625.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-23 15:00:00 | 1574.10 | 2026-04-27 13:15:00 | 1587.50 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-04-24 09:15:00 | 1540.50 | 2026-04-27 13:15:00 | 1587.50 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2026-04-24 15:00:00 | 1569.60 | 2026-04-27 13:15:00 | 1587.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-04-27 12:15:00 | 1571.30 | 2026-04-27 13:15:00 | 1587.50 | STOP_HIT | 1.00 | -1.03% |
