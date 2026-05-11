# Global Health Ltd. (MEDANTA)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1202.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 136 |
| ALERT1 | 97 |
| ALERT2 | 97 |
| ALERT2_SKIP | 56 |
| ALERT3 | 246 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 100 |
| PARTIAL | 24 |
| TARGET_HIT | 6 |
| STOP_HIT | 98 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 63 / 65
- **Target hits / Stop hits / Partials:** 6 / 98 / 24
- **Avg / median % per leg:** 1.43% / -0.04%
- **Sum % (uncompounded):** 183.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 16 | 29.6% | 5 | 47 | 2 | 0.20% | 10.7% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 3.86% | 27.0% |
| BUY @ 3rd Alert (retest2) | 47 | 12 | 25.5% | 3 | 44 | 0 | -0.35% | -16.2% |
| SELL (all) | 74 | 47 | 63.5% | 1 | 51 | 22 | 2.34% | 172.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 74 | 47 | 63.5% | 1 | 51 | 22 | 2.34% | 172.9% |
| retest1 (combined) | 7 | 4 | 57.1% | 2 | 3 | 2 | 3.86% | 27.0% |
| retest2 (combined) | 121 | 59 | 48.8% | 4 | 95 | 22 | 1.29% | 156.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 11:15:00 | 1401.45 | 1405.73 | 1406.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 14:15:00 | 1399.90 | 1403.31 | 1404.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 09:15:00 | 1399.65 | 1397.97 | 1401.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 1399.65 | 1397.97 | 1401.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 1399.65 | 1397.97 | 1401.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:45:00 | 1400.00 | 1397.97 | 1401.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 1404.55 | 1399.15 | 1401.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:30:00 | 1405.95 | 1399.15 | 1401.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 12:15:00 | 1411.20 | 1401.56 | 1402.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 13:00:00 | 1411.20 | 1401.56 | 1402.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 1407.45 | 1402.74 | 1403.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 14:00:00 | 1407.45 | 1402.74 | 1403.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 1412.20 | 1404.63 | 1403.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 1419.90 | 1409.38 | 1406.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 11:15:00 | 1407.10 | 1409.02 | 1406.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 11:15:00 | 1407.10 | 1409.02 | 1406.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 1407.10 | 1409.02 | 1406.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:00:00 | 1407.10 | 1409.02 | 1406.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 1404.10 | 1408.03 | 1406.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:00:00 | 1404.10 | 1408.03 | 1406.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 1401.95 | 1406.82 | 1406.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:30:00 | 1402.05 | 1406.82 | 1406.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 1400.05 | 1405.46 | 1405.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 09:15:00 | 1395.50 | 1403.40 | 1404.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 13:15:00 | 1403.00 | 1399.54 | 1401.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 13:15:00 | 1403.00 | 1399.54 | 1401.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 1403.00 | 1399.54 | 1401.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:00:00 | 1403.00 | 1399.54 | 1401.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 1397.10 | 1399.05 | 1401.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 1391.35 | 1399.05 | 1401.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 10:15:00 | 1406.60 | 1399.72 | 1401.02 | SL hit (close>static) qty=1.00 sl=1403.80 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 11:15:00 | 1419.45 | 1403.66 | 1402.70 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 1354.80 | 1400.00 | 1403.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 1244.35 | 1335.24 | 1365.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 15:15:00 | 1222.00 | 1213.14 | 1255.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-24 09:15:00 | 1195.75 | 1213.14 | 1255.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1204.05 | 1188.53 | 1216.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:30:00 | 1208.10 | 1188.53 | 1216.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 1221.00 | 1195.03 | 1216.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:00:00 | 1221.00 | 1195.03 | 1216.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 1220.95 | 1200.21 | 1216.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:45:00 | 1225.90 | 1200.21 | 1216.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 10:15:00 | 1239.30 | 1225.26 | 1223.71 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 1206.15 | 1224.58 | 1224.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 13:15:00 | 1193.90 | 1209.86 | 1217.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 1155.00 | 1153.93 | 1171.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 1155.00 | 1153.93 | 1171.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1192.80 | 1161.71 | 1173.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 1192.80 | 1161.71 | 1173.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 1185.00 | 1166.37 | 1174.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 1190.30 | 1166.37 | 1174.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 1166.00 | 1167.30 | 1173.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:30:00 | 1166.85 | 1167.30 | 1173.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 1167.00 | 1166.53 | 1171.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 14:30:00 | 1168.45 | 1166.53 | 1171.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1121.35 | 1157.64 | 1166.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:30:00 | 1107.95 | 1138.79 | 1156.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 1052.55 | 1113.77 | 1141.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 11:15:00 | 1113.00 | 1096.75 | 1118.29 | SL hit (close>ema200) qty=0.50 sl=1096.75 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 1179.80 | 1133.95 | 1129.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 12:15:00 | 1182.50 | 1150.94 | 1138.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 1208.40 | 1208.46 | 1185.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 15:15:00 | 1228.00 | 1212.41 | 1202.93 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 14:15:00 | 1232.40 | 1220.76 | 1212.04 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 09:15:00 | 1289.40 | 1271.25 | 1247.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 09:15:00 | 1294.02 | 1271.25 | 1247.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-06-14 11:15:00 | 1350.80 | 1298.68 | 1264.72 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 9 — SELL (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 11:15:00 | 1307.35 | 1330.55 | 1332.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 13:15:00 | 1300.15 | 1310.20 | 1313.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 11:15:00 | 1307.00 | 1300.76 | 1306.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 11:15:00 | 1307.00 | 1300.76 | 1306.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 1307.00 | 1300.76 | 1306.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:00:00 | 1307.00 | 1300.76 | 1306.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 1309.45 | 1302.50 | 1306.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:45:00 | 1307.50 | 1302.50 | 1306.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 1310.90 | 1304.18 | 1307.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:00:00 | 1310.90 | 1304.18 | 1307.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 1309.65 | 1305.27 | 1307.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:45:00 | 1309.40 | 1305.27 | 1307.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 1302.50 | 1304.72 | 1306.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 1307.55 | 1304.72 | 1306.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1288.95 | 1301.57 | 1305.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 14:45:00 | 1279.80 | 1294.62 | 1300.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 10:00:00 | 1284.30 | 1290.29 | 1297.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 1252.00 | 1298.29 | 1298.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 14:15:00 | 1285.20 | 1276.21 | 1279.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 1275.85 | 1276.14 | 1279.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-04 13:15:00 | 1284.80 | 1277.14 | 1276.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2024-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 13:15:00 | 1284.80 | 1277.14 | 1276.93 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 14:15:00 | 1271.25 | 1275.96 | 1276.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 15:15:00 | 1270.00 | 1274.77 | 1275.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 11:15:00 | 1275.70 | 1273.80 | 1275.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 11:15:00 | 1275.70 | 1273.80 | 1275.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1275.70 | 1273.80 | 1275.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:45:00 | 1276.00 | 1273.80 | 1275.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 1276.50 | 1274.34 | 1275.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:30:00 | 1275.75 | 1274.34 | 1275.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 1272.30 | 1273.93 | 1274.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 15:00:00 | 1266.10 | 1272.37 | 1274.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 11:00:00 | 1266.00 | 1270.16 | 1272.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 12:45:00 | 1267.75 | 1268.96 | 1271.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 09:15:00 | 1202.79 | 1222.77 | 1236.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 09:15:00 | 1202.70 | 1222.77 | 1236.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 09:15:00 | 1204.36 | 1222.77 | 1236.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-15 14:15:00 | 1211.25 | 1211.15 | 1224.11 | SL hit (close>ema200) qty=0.50 sl=1211.15 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 15:15:00 | 1209.65 | 1194.37 | 1193.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 1226.35 | 1203.75 | 1200.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 1244.30 | 1251.19 | 1240.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 14:15:00 | 1244.30 | 1251.19 | 1240.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1244.30 | 1251.19 | 1240.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 1244.30 | 1251.19 | 1240.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 1243.65 | 1249.68 | 1240.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 1234.20 | 1249.68 | 1240.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 1229.05 | 1245.55 | 1239.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:30:00 | 1228.00 | 1245.55 | 1239.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 1226.80 | 1241.80 | 1238.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:30:00 | 1226.05 | 1241.80 | 1238.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 1223.25 | 1235.02 | 1235.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 1214.10 | 1225.61 | 1230.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 10:15:00 | 1179.80 | 1170.39 | 1187.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:45:00 | 1180.95 | 1170.39 | 1187.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 14:15:00 | 1183.10 | 1177.33 | 1186.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 15:00:00 | 1183.10 | 1177.33 | 1186.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1180.00 | 1177.49 | 1184.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:30:00 | 1169.85 | 1183.44 | 1185.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:45:00 | 1173.70 | 1181.35 | 1184.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:45:00 | 1172.05 | 1176.69 | 1181.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 15:15:00 | 1160.95 | 1177.09 | 1180.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 1160.95 | 1173.86 | 1179.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:15:00 | 1158.35 | 1173.86 | 1179.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1151.80 | 1169.45 | 1176.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:30:00 | 1149.55 | 1165.76 | 1174.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:30:00 | 1148.20 | 1161.64 | 1171.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1111.36 | 1133.26 | 1152.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1115.02 | 1133.26 | 1152.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1113.45 | 1133.26 | 1152.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1102.90 | 1133.26 | 1152.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1092.07 | 1133.26 | 1152.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1090.79 | 1133.26 | 1152.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 1091.10 | 1090.78 | 1115.30 | SL hit (close>ema200) qty=0.50 sl=1090.78 alert=retest2 |

### Cycle 14 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 1130.20 | 1098.41 | 1094.51 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 11:15:00 | 1083.50 | 1099.31 | 1099.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 12:15:00 | 1079.85 | 1095.42 | 1097.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 11:15:00 | 1087.90 | 1076.84 | 1080.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 11:15:00 | 1087.90 | 1076.84 | 1080.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 1087.90 | 1076.84 | 1080.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:00:00 | 1087.90 | 1076.84 | 1080.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 1081.30 | 1077.73 | 1080.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 15:00:00 | 1074.70 | 1077.51 | 1079.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 1075.05 | 1075.59 | 1078.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 1070.60 | 1071.73 | 1071.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 12:00:00 | 1073.70 | 1071.86 | 1071.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 13:15:00 | 1080.45 | 1073.17 | 1072.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 13:15:00 | 1080.45 | 1073.17 | 1072.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 14:15:00 | 1090.95 | 1076.73 | 1074.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 10:15:00 | 1074.30 | 1077.75 | 1075.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 10:15:00 | 1074.30 | 1077.75 | 1075.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1074.30 | 1077.75 | 1075.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 1074.20 | 1077.75 | 1075.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 1097.75 | 1081.75 | 1077.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:30:00 | 1071.80 | 1081.75 | 1077.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 1126.25 | 1141.85 | 1127.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:00:00 | 1126.25 | 1141.85 | 1127.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 1131.00 | 1139.68 | 1127.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 1157.45 | 1139.68 | 1127.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 13:15:00 | 1130.00 | 1146.87 | 1147.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 13:15:00 | 1130.00 | 1146.87 | 1147.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 10:15:00 | 1125.05 | 1140.42 | 1144.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1112.30 | 1107.84 | 1117.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1112.30 | 1107.84 | 1117.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1112.30 | 1107.84 | 1117.72 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 15:15:00 | 1128.95 | 1121.25 | 1121.07 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 1116.65 | 1121.26 | 1121.81 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 11:15:00 | 1125.65 | 1122.17 | 1122.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 1135.25 | 1126.13 | 1124.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 12:15:00 | 1125.00 | 1126.80 | 1125.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 12:15:00 | 1125.00 | 1126.80 | 1125.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 1125.00 | 1126.80 | 1125.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:45:00 | 1125.35 | 1126.80 | 1125.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 1121.35 | 1125.71 | 1124.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 14:00:00 | 1121.35 | 1125.71 | 1124.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 1124.15 | 1125.40 | 1124.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:15:00 | 1120.00 | 1125.40 | 1124.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 1120.00 | 1124.32 | 1124.26 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 1120.70 | 1123.60 | 1123.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 10:15:00 | 1117.45 | 1122.37 | 1123.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 11:15:00 | 1124.55 | 1122.80 | 1123.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 11:15:00 | 1124.55 | 1122.80 | 1123.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 1124.55 | 1122.80 | 1123.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:45:00 | 1125.55 | 1122.80 | 1123.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 1126.45 | 1123.53 | 1123.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:00:00 | 1126.45 | 1123.53 | 1123.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 13:15:00 | 1127.55 | 1124.34 | 1124.07 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 15:15:00 | 1123.00 | 1125.18 | 1125.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 1112.05 | 1121.22 | 1123.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 13:15:00 | 1097.60 | 1097.24 | 1103.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 14:15:00 | 1099.95 | 1097.24 | 1103.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 1119.75 | 1101.74 | 1104.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 1119.75 | 1101.74 | 1104.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 1115.00 | 1104.39 | 1105.67 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 1122.85 | 1108.08 | 1107.23 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 14:15:00 | 1099.65 | 1107.12 | 1107.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 15:15:00 | 1095.00 | 1104.70 | 1106.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 13:15:00 | 1099.20 | 1098.95 | 1102.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-24 13:30:00 | 1098.95 | 1098.95 | 1102.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 1095.50 | 1098.26 | 1101.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:45:00 | 1099.65 | 1098.26 | 1101.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1030.70 | 1041.46 | 1056.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 13:15:00 | 1024.55 | 1036.11 | 1050.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 15:15:00 | 1020.00 | 1031.88 | 1045.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 13:15:00 | 973.32 | 990.81 | 1007.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 986.95 | 984.76 | 999.87 | SL hit (close>ema200) qty=0.50 sl=984.76 alert=retest2 |

### Cycle 26 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 994.90 | 983.05 | 981.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 1004.50 | 991.30 | 986.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 15:15:00 | 998.00 | 998.26 | 992.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 09:15:00 | 996.35 | 998.26 | 992.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 1004.40 | 999.49 | 993.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 11:30:00 | 1011.50 | 1005.47 | 997.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-16 10:15:00 | 1112.65 | 1078.91 | 1059.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 1066.75 | 1077.32 | 1077.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 12:15:00 | 1060.70 | 1068.33 | 1072.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 1038.00 | 1033.23 | 1045.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 10:15:00 | 1038.00 | 1033.23 | 1045.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 1038.00 | 1033.23 | 1045.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:45:00 | 1040.00 | 1033.23 | 1045.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 1045.45 | 1035.37 | 1043.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:00:00 | 1045.45 | 1035.37 | 1043.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 1045.55 | 1037.40 | 1043.51 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 11:15:00 | 1056.95 | 1047.29 | 1046.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 12:15:00 | 1064.90 | 1050.81 | 1048.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 1056.00 | 1058.24 | 1053.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 1056.00 | 1058.24 | 1053.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1056.00 | 1058.24 | 1053.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:30:00 | 1057.05 | 1058.24 | 1053.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 1050.40 | 1056.68 | 1053.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:00:00 | 1050.40 | 1056.68 | 1053.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 1057.15 | 1056.77 | 1053.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:30:00 | 1053.60 | 1056.77 | 1053.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 1063.00 | 1058.02 | 1054.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 10:30:00 | 1067.40 | 1062.06 | 1057.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 14:15:00 | 1072.65 | 1086.57 | 1088.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 14:15:00 | 1072.65 | 1086.57 | 1088.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 1063.45 | 1079.93 | 1084.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 1072.05 | 1066.87 | 1073.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 1072.05 | 1066.87 | 1073.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1072.05 | 1066.87 | 1073.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 10:30:00 | 1065.55 | 1068.26 | 1073.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 15:15:00 | 1066.80 | 1071.79 | 1073.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 09:15:00 | 1095.45 | 1075.72 | 1075.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 09:15:00 | 1095.45 | 1075.72 | 1075.32 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 14:15:00 | 1065.50 | 1077.05 | 1078.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 1054.50 | 1069.14 | 1072.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 13:15:00 | 1064.45 | 1039.03 | 1046.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 13:15:00 | 1064.45 | 1039.03 | 1046.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1064.45 | 1039.03 | 1046.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:00:00 | 1064.45 | 1039.03 | 1046.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 1076.60 | 1046.54 | 1049.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:45:00 | 1078.15 | 1046.54 | 1049.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 15:15:00 | 1072.00 | 1051.63 | 1051.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 1083.00 | 1070.24 | 1062.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 1086.25 | 1086.58 | 1075.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 15:15:00 | 1086.25 | 1086.58 | 1075.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 1086.25 | 1086.58 | 1075.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 1061.35 | 1081.54 | 1074.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 1064.95 | 1078.22 | 1073.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 12:15:00 | 1077.50 | 1076.58 | 1073.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 13:00:00 | 1074.70 | 1076.20 | 1073.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 14:30:00 | 1074.50 | 1074.61 | 1073.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 1083.10 | 1073.77 | 1072.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1087.45 | 1076.50 | 1074.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 11:00:00 | 1094.85 | 1080.17 | 1076.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 12:30:00 | 1100.75 | 1087.48 | 1080.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-27 10:15:00 | 1081.95 | 1096.10 | 1097.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 10:15:00 | 1081.95 | 1096.10 | 1097.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 11:15:00 | 1075.20 | 1091.92 | 1095.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 13:15:00 | 1073.90 | 1072.53 | 1080.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-28 14:00:00 | 1073.90 | 1072.53 | 1080.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1080.00 | 1075.40 | 1080.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 1080.00 | 1075.40 | 1080.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1081.75 | 1076.67 | 1080.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:30:00 | 1080.30 | 1076.67 | 1080.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 1076.00 | 1076.54 | 1079.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 12:15:00 | 1071.80 | 1076.54 | 1079.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 13:30:00 | 1071.80 | 1074.90 | 1078.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 1102.50 | 1080.46 | 1080.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 1102.50 | 1080.46 | 1080.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 1141.50 | 1103.57 | 1093.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 10:15:00 | 1138.35 | 1171.00 | 1167.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 10:15:00 | 1138.35 | 1171.00 | 1167.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 1138.35 | 1171.00 | 1167.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:00:00 | 1138.35 | 1171.00 | 1167.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 1146.60 | 1166.12 | 1166.04 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 12:15:00 | 1143.00 | 1161.50 | 1163.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 13:15:00 | 1140.00 | 1157.20 | 1161.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 1158.00 | 1152.66 | 1158.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 1158.00 | 1152.66 | 1158.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1158.00 | 1152.66 | 1158.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:00:00 | 1158.00 | 1152.66 | 1158.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 1142.00 | 1150.53 | 1156.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 12:00:00 | 1130.00 | 1146.42 | 1154.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 12:30:00 | 1135.05 | 1145.22 | 1152.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 13:15:00 | 1133.20 | 1145.22 | 1152.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 12:15:00 | 1121.90 | 1119.17 | 1118.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 12:15:00 | 1121.90 | 1119.17 | 1118.96 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 1108.30 | 1117.00 | 1117.99 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 10:15:00 | 1132.05 | 1117.68 | 1117.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 11:15:00 | 1137.10 | 1121.56 | 1119.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 1123.50 | 1127.44 | 1123.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 09:15:00 | 1123.50 | 1127.44 | 1123.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1123.50 | 1127.44 | 1123.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 10:15:00 | 1130.85 | 1127.44 | 1123.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:15:00 | 1132.50 | 1127.67 | 1124.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 13:00:00 | 1131.65 | 1129.25 | 1125.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 1130.75 | 1128.07 | 1126.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 1133.35 | 1129.13 | 1126.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-20 13:15:00 | 1114.75 | 1124.51 | 1125.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 1114.75 | 1124.51 | 1125.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 1094.25 | 1118.46 | 1122.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 10:15:00 | 1102.75 | 1101.90 | 1108.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:30:00 | 1103.05 | 1101.90 | 1108.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 1101.25 | 1101.77 | 1107.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 14:15:00 | 1099.80 | 1101.98 | 1106.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:00:00 | 1097.70 | 1101.12 | 1106.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 1113.95 | 1105.45 | 1104.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1113.95 | 1105.45 | 1104.99 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 11:15:00 | 1095.45 | 1103.24 | 1104.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 14:15:00 | 1091.55 | 1098.84 | 1101.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 09:15:00 | 1083.85 | 1083.78 | 1090.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 1083.85 | 1083.78 | 1090.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1083.85 | 1083.78 | 1090.49 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 1102.95 | 1092.92 | 1091.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 1104.55 | 1095.24 | 1092.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 14:15:00 | 1103.45 | 1103.68 | 1099.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 15:00:00 | 1103.45 | 1103.68 | 1099.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 1094.00 | 1101.49 | 1099.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:15:00 | 1089.30 | 1101.49 | 1099.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 1094.85 | 1100.16 | 1098.99 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 1092.30 | 1097.73 | 1098.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 15:15:00 | 1088.00 | 1094.32 | 1096.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 1081.50 | 1068.84 | 1074.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 1081.50 | 1068.84 | 1074.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1081.50 | 1068.84 | 1074.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:00:00 | 1081.50 | 1068.84 | 1074.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 1081.70 | 1071.41 | 1075.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:15:00 | 1075.35 | 1071.41 | 1075.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 14:15:00 | 1083.00 | 1078.18 | 1077.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 14:15:00 | 1083.00 | 1078.18 | 1077.68 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 13:15:00 | 1072.80 | 1077.39 | 1077.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 15:15:00 | 1069.00 | 1074.78 | 1076.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 10:15:00 | 1073.55 | 1072.32 | 1074.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 11:00:00 | 1073.55 | 1072.32 | 1074.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 1074.20 | 1072.70 | 1074.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:00:00 | 1074.20 | 1072.70 | 1074.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 1069.40 | 1072.04 | 1074.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:30:00 | 1075.10 | 1072.04 | 1074.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 1067.70 | 1070.12 | 1073.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 15:00:00 | 1067.70 | 1070.12 | 1073.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 15:15:00 | 1070.50 | 1070.20 | 1072.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 1057.60 | 1070.20 | 1072.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 11:15:00 | 1004.72 | 1016.74 | 1023.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-21 09:15:00 | 1044.45 | 1015.22 | 1019.09 | SL hit (close>ema200) qty=0.50 sl=1015.22 alert=retest2 |

### Cycle 46 — BUY (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 11:15:00 | 1053.00 | 1025.86 | 1023.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 15:15:00 | 1059.00 | 1043.02 | 1033.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-22 09:15:00 | 1041.75 | 1042.77 | 1034.01 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 10:30:00 | 1053.05 | 1043.21 | 1035.01 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1061.00 | 1046.38 | 1039.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 11:45:00 | 1064.55 | 1053.25 | 1044.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 09:15:00 | 1039.80 | 1058.73 | 1051.55 | SL hit (close<ema400) qty=1.00 sl=1051.55 alert=retest1 |

### Cycle 47 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 1021.20 | 1045.77 | 1048.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 11:15:00 | 1010.05 | 1035.80 | 1042.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 1034.75 | 1025.30 | 1032.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 1034.75 | 1025.30 | 1032.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 1034.75 | 1025.30 | 1032.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 1034.75 | 1025.30 | 1032.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 1031.95 | 1026.63 | 1032.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:30:00 | 1023.75 | 1025.91 | 1031.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 12:15:00 | 1043.55 | 1030.97 | 1031.63 | SL hit (close>static) qty=1.00 sl=1039.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 1043.85 | 1033.55 | 1032.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 1050.15 | 1039.98 | 1036.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 14:15:00 | 1046.95 | 1047.46 | 1041.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 15:00:00 | 1046.95 | 1047.46 | 1041.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 1045.90 | 1047.67 | 1043.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 1045.90 | 1047.67 | 1043.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 1052.35 | 1048.60 | 1044.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 12:45:00 | 1055.80 | 1051.01 | 1045.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 14:15:00 | 1039.55 | 1049.72 | 1046.14 | SL hit (close<static) qty=1.00 sl=1042.80 alert=retest2 |

### Cycle 49 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 1037.80 | 1051.34 | 1051.59 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 1105.35 | 1058.02 | 1053.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 1116.65 | 1069.75 | 1059.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 11:15:00 | 1147.70 | 1157.41 | 1135.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 12:00:00 | 1147.70 | 1157.41 | 1135.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1142.30 | 1156.53 | 1143.39 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 15:15:00 | 1134.55 | 1138.16 | 1138.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1112.75 | 1133.08 | 1135.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 10:15:00 | 1105.80 | 1097.18 | 1107.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 10:15:00 | 1105.80 | 1097.18 | 1107.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1105.80 | 1097.18 | 1107.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 1105.80 | 1097.18 | 1107.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 1130.50 | 1103.84 | 1109.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:00:00 | 1130.50 | 1103.84 | 1109.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 1134.15 | 1109.90 | 1111.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:45:00 | 1134.65 | 1109.90 | 1111.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 13:15:00 | 1134.30 | 1114.78 | 1113.51 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 1096.35 | 1111.91 | 1114.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 10:15:00 | 1085.60 | 1106.65 | 1111.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 1131.05 | 1106.68 | 1109.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 14:15:00 | 1131.05 | 1106.68 | 1109.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1131.05 | 1106.68 | 1109.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:45:00 | 1134.90 | 1106.68 | 1109.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 1126.10 | 1110.57 | 1110.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 1110.35 | 1110.57 | 1110.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 10:15:00 | 1117.40 | 1111.22 | 1111.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 11:15:00 | 1124.55 | 1113.88 | 1112.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 15:15:00 | 1189.90 | 1216.77 | 1204.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 15:15:00 | 1189.90 | 1216.77 | 1204.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 1189.90 | 1216.77 | 1204.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 12:30:00 | 1227.65 | 1215.62 | 1207.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 12:15:00 | 1190.85 | 1204.73 | 1205.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 12:15:00 | 1190.85 | 1204.73 | 1205.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 13:15:00 | 1186.35 | 1201.06 | 1203.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 10:15:00 | 1161.00 | 1158.36 | 1174.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 10:45:00 | 1162.65 | 1158.36 | 1174.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 1173.45 | 1161.42 | 1171.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:00:00 | 1173.45 | 1161.42 | 1171.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 1225.60 | 1174.26 | 1176.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 1225.60 | 1174.26 | 1176.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 1210.15 | 1181.44 | 1179.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 1229.00 | 1203.99 | 1198.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 1219.30 | 1227.25 | 1216.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 1219.30 | 1227.25 | 1216.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 1225.00 | 1226.93 | 1220.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:30:00 | 1231.00 | 1227.18 | 1220.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:00:00 | 1230.50 | 1227.84 | 1221.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 1208.75 | 1226.47 | 1223.33 | SL hit (close<static) qty=1.00 sl=1220.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 1180.00 | 1214.10 | 1218.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 12:15:00 | 1163.80 | 1170.39 | 1179.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 13:15:00 | 1180.20 | 1172.35 | 1179.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 13:15:00 | 1180.20 | 1172.35 | 1179.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 1180.20 | 1172.35 | 1179.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:00:00 | 1180.20 | 1172.35 | 1179.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 1177.80 | 1173.44 | 1179.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 1177.80 | 1173.44 | 1179.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1178.50 | 1175.34 | 1179.43 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 1222.10 | 1187.86 | 1183.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 1225.40 | 1200.59 | 1190.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 11:15:00 | 1252.95 | 1254.68 | 1244.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 12:00:00 | 1252.95 | 1254.68 | 1244.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 1231.40 | 1249.42 | 1244.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 15:00:00 | 1231.40 | 1249.42 | 1244.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 1230.25 | 1245.58 | 1243.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 1259.95 | 1245.58 | 1243.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 1238.15 | 1246.10 | 1247.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 1238.15 | 1246.10 | 1247.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 1229.90 | 1242.86 | 1245.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 1226.10 | 1212.43 | 1219.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 1226.10 | 1212.43 | 1219.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1226.10 | 1212.43 | 1219.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 1226.10 | 1212.43 | 1219.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1221.70 | 1214.29 | 1219.58 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 1231.45 | 1223.96 | 1223.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 09:15:00 | 1243.75 | 1227.29 | 1224.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 1242.55 | 1243.18 | 1236.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 1242.55 | 1243.18 | 1236.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1242.55 | 1243.18 | 1236.03 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1210.20 | 1233.38 | 1234.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1172.45 | 1217.79 | 1225.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1209.55 | 1199.76 | 1210.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 15:15:00 | 1209.55 | 1199.76 | 1210.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 1209.55 | 1199.76 | 1210.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 1225.65 | 1199.76 | 1210.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1246.65 | 1209.14 | 1213.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 10:00:00 | 1246.65 | 1209.14 | 1213.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 1238.80 | 1219.43 | 1217.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 1250.70 | 1232.67 | 1228.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 10:15:00 | 1257.50 | 1258.71 | 1246.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 10:30:00 | 1256.40 | 1258.71 | 1246.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 1271.40 | 1279.19 | 1272.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:00:00 | 1271.40 | 1279.19 | 1272.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 1270.00 | 1277.35 | 1271.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:45:00 | 1269.00 | 1277.35 | 1271.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 1265.60 | 1275.00 | 1271.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 1265.60 | 1275.00 | 1271.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 1270.00 | 1274.00 | 1271.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 1263.20 | 1274.00 | 1271.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1268.00 | 1272.80 | 1270.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:30:00 | 1275.50 | 1272.80 | 1270.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 1268.00 | 1271.84 | 1270.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 10:30:00 | 1265.00 | 1271.84 | 1270.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 11:15:00 | 1247.60 | 1266.99 | 1268.56 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 14:15:00 | 1273.00 | 1266.06 | 1265.75 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 1250.60 | 1263.66 | 1264.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 10:15:00 | 1244.40 | 1259.81 | 1262.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 13:15:00 | 1256.00 | 1256.00 | 1260.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-23 14:00:00 | 1256.00 | 1256.00 | 1260.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1262.00 | 1254.69 | 1257.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:00:00 | 1262.00 | 1254.69 | 1257.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 1254.20 | 1254.59 | 1257.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 12:15:00 | 1243.40 | 1254.59 | 1257.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 13:15:00 | 1181.23 | 1197.63 | 1211.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 1195.60 | 1191.23 | 1204.89 | SL hit (close>ema200) qty=0.50 sl=1191.23 alert=retest2 |

### Cycle 66 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 1217.80 | 1206.49 | 1205.04 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 1192.70 | 1207.87 | 1208.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 1185.40 | 1203.37 | 1206.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 1203.90 | 1198.82 | 1202.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 1203.90 | 1198.82 | 1202.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1203.90 | 1198.82 | 1202.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:45:00 | 1206.00 | 1198.82 | 1202.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1205.20 | 1200.10 | 1202.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:30:00 | 1210.10 | 1200.10 | 1202.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 1203.50 | 1200.78 | 1202.93 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 1215.70 | 1206.39 | 1205.20 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 1191.50 | 1204.50 | 1205.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 1185.00 | 1200.60 | 1203.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 1178.90 | 1176.80 | 1186.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 1211.00 | 1176.80 | 1186.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1222.10 | 1185.86 | 1190.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 1222.10 | 1185.86 | 1190.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1231.50 | 1194.98 | 1193.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1243.10 | 1222.51 | 1210.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 15:15:00 | 1240.00 | 1241.82 | 1233.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:15:00 | 1248.90 | 1241.82 | 1233.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 12:15:00 | 1245.40 | 1243.15 | 1236.08 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 1240.00 | 1242.52 | 1236.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:45:00 | 1239.70 | 1242.52 | 1236.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1236.20 | 1241.25 | 1236.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 1236.20 | 1241.25 | 1236.41 | SL hit (close<ema400) qty=1.00 sl=1236.41 alert=retest1 |

### Cycle 71 — SELL (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 09:15:00 | 1191.10 | 1232.06 | 1233.48 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 1210.90 | 1192.30 | 1190.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 14:15:00 | 1219.90 | 1197.82 | 1193.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 1205.40 | 1207.53 | 1202.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 1207.80 | 1207.53 | 1202.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1198.70 | 1205.76 | 1201.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:00:00 | 1210.50 | 1206.00 | 1202.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 13:30:00 | 1211.90 | 1207.61 | 1203.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:00:00 | 1211.00 | 1208.96 | 1205.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:15:00 | 1210.60 | 1209.19 | 1205.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1206.60 | 1209.30 | 1207.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-29 11:15:00 | 1197.40 | 1206.55 | 1206.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 11:15:00 | 1197.40 | 1206.55 | 1206.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 1192.10 | 1201.94 | 1204.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 15:15:00 | 1194.90 | 1190.74 | 1195.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 09:15:00 | 1191.40 | 1190.74 | 1195.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1206.70 | 1193.93 | 1196.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 1206.70 | 1193.93 | 1196.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1212.30 | 1197.60 | 1197.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:45:00 | 1215.80 | 1197.60 | 1197.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 1207.00 | 1199.48 | 1198.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 11:15:00 | 1220.00 | 1210.36 | 1205.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 10:15:00 | 1213.00 | 1215.23 | 1210.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 11:00:00 | 1213.00 | 1215.23 | 1210.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 1207.00 | 1213.58 | 1210.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:00:00 | 1207.00 | 1213.58 | 1210.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 1203.10 | 1211.48 | 1209.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:45:00 | 1205.00 | 1211.48 | 1209.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 1197.40 | 1207.47 | 1208.10 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 1212.00 | 1207.74 | 1207.71 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1204.40 | 1207.08 | 1207.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 14:15:00 | 1200.00 | 1205.66 | 1206.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 1205.80 | 1204.96 | 1206.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 1205.80 | 1204.96 | 1206.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1205.80 | 1204.96 | 1206.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 1205.70 | 1204.96 | 1206.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1210.00 | 1205.97 | 1206.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 1213.00 | 1205.97 | 1206.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 1210.40 | 1206.85 | 1206.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:30:00 | 1209.90 | 1206.85 | 1206.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 1217.70 | 1209.02 | 1207.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 1218.80 | 1210.98 | 1208.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 12:15:00 | 1207.60 | 1214.76 | 1212.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 12:15:00 | 1207.60 | 1214.76 | 1212.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 1207.60 | 1214.76 | 1212.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:45:00 | 1209.20 | 1214.76 | 1212.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 1198.30 | 1211.47 | 1210.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 1198.30 | 1211.47 | 1210.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 14:15:00 | 1198.50 | 1208.87 | 1209.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 1195.90 | 1200.30 | 1203.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 10:15:00 | 1204.80 | 1199.53 | 1201.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 10:15:00 | 1204.80 | 1199.53 | 1201.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1204.80 | 1199.53 | 1201.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 1204.80 | 1199.53 | 1201.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1196.90 | 1199.00 | 1201.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 12:30:00 | 1194.30 | 1197.68 | 1200.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 11:15:00 | 1205.10 | 1190.29 | 1190.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1205.10 | 1190.29 | 1190.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 1209.70 | 1194.17 | 1191.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 10:15:00 | 1200.00 | 1205.47 | 1199.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 10:15:00 | 1200.00 | 1205.47 | 1199.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1200.00 | 1205.47 | 1199.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:00:00 | 1200.00 | 1205.47 | 1199.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1198.40 | 1204.06 | 1199.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:45:00 | 1198.80 | 1204.06 | 1199.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 1194.20 | 1202.09 | 1198.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 1194.00 | 1202.09 | 1198.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 1189.90 | 1199.65 | 1198.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:00:00 | 1189.90 | 1199.65 | 1198.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 15:15:00 | 1189.00 | 1195.74 | 1196.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 1173.10 | 1186.01 | 1191.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 1186.80 | 1181.04 | 1186.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 1186.80 | 1181.04 | 1186.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1186.80 | 1181.04 | 1186.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 1190.40 | 1181.04 | 1186.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1183.50 | 1181.53 | 1186.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 1177.00 | 1181.53 | 1186.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 09:15:00 | 1118.15 | 1139.70 | 1155.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1141.90 | 1140.14 | 1154.02 | SL hit (close>ema200) qty=0.50 sl=1140.14 alert=retest2 |

### Cycle 82 — BUY (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 09:15:00 | 1148.40 | 1135.54 | 1134.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 12:15:00 | 1160.80 | 1149.39 | 1144.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 1180.90 | 1182.03 | 1169.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 09:45:00 | 1177.50 | 1182.03 | 1169.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1171.70 | 1178.81 | 1171.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:00:00 | 1171.70 | 1178.81 | 1171.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 1177.00 | 1178.45 | 1171.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:30:00 | 1175.50 | 1178.45 | 1171.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1172.90 | 1179.00 | 1173.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 1172.90 | 1179.00 | 1173.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1184.80 | 1180.16 | 1174.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 12:45:00 | 1191.30 | 1182.65 | 1176.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-09 11:15:00 | 1310.43 | 1245.60 | 1218.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1321.00 | 1331.75 | 1332.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 11:15:00 | 1313.20 | 1321.91 | 1326.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1311.50 | 1296.65 | 1304.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1311.50 | 1296.65 | 1304.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1311.50 | 1296.65 | 1304.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 1304.20 | 1296.65 | 1304.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1306.20 | 1298.56 | 1304.84 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 1333.10 | 1312.21 | 1309.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 09:15:00 | 1353.50 | 1332.07 | 1325.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 10:15:00 | 1337.10 | 1348.13 | 1339.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 10:15:00 | 1337.10 | 1348.13 | 1339.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1337.10 | 1348.13 | 1339.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 1337.10 | 1348.13 | 1339.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 1338.50 | 1346.20 | 1339.12 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 09:15:00 | 1315.60 | 1333.25 | 1334.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 10:15:00 | 1307.90 | 1328.18 | 1332.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 09:15:00 | 1321.30 | 1314.10 | 1321.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1321.30 | 1314.10 | 1321.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1321.30 | 1314.10 | 1321.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:30:00 | 1320.60 | 1314.10 | 1321.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1315.90 | 1314.46 | 1321.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 1313.50 | 1316.48 | 1320.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:00:00 | 1307.10 | 1314.60 | 1319.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:45:00 | 1311.00 | 1312.82 | 1317.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 10:15:00 | 1326.80 | 1315.62 | 1318.46 | SL hit (close>static) qty=1.00 sl=1325.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 12:15:00 | 1348.70 | 1325.24 | 1322.52 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 1308.30 | 1323.89 | 1325.45 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 1328.10 | 1325.69 | 1325.62 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 1324.40 | 1325.45 | 1325.53 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 14:15:00 | 1341.00 | 1328.29 | 1326.71 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 1314.80 | 1325.43 | 1325.88 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 1419.70 | 1342.80 | 1332.95 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 1363.00 | 1377.50 | 1378.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 14:15:00 | 1353.70 | 1367.23 | 1372.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 1371.30 | 1366.09 | 1371.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 1371.30 | 1366.09 | 1371.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1371.30 | 1366.09 | 1371.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:45:00 | 1349.60 | 1361.75 | 1366.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 13:15:00 | 1372.10 | 1365.68 | 1365.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 13:15:00 | 1372.10 | 1365.68 | 1365.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 1397.90 | 1373.87 | 1369.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 12:15:00 | 1373.50 | 1377.24 | 1372.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 12:15:00 | 1373.50 | 1377.24 | 1372.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 1373.50 | 1377.24 | 1372.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 1373.50 | 1377.24 | 1372.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 1381.10 | 1378.01 | 1373.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:45:00 | 1386.70 | 1380.54 | 1375.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 12:30:00 | 1389.40 | 1383.93 | 1378.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:30:00 | 1386.00 | 1384.05 | 1381.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 1385.00 | 1383.98 | 1381.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1384.10 | 1392.43 | 1389.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 1382.50 | 1392.43 | 1389.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1388.80 | 1391.70 | 1389.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 11:30:00 | 1392.30 | 1391.32 | 1389.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 12:15:00 | 1392.30 | 1391.32 | 1389.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 13:00:00 | 1392.50 | 1391.56 | 1389.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:30:00 | 1394.40 | 1397.44 | 1393.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 1413.60 | 1400.67 | 1395.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 1388.70 | 1396.12 | 1396.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 13:15:00 | 1388.70 | 1396.12 | 1396.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 14:15:00 | 1376.00 | 1392.09 | 1394.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 12:15:00 | 1397.80 | 1387.97 | 1391.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 12:15:00 | 1397.80 | 1387.97 | 1391.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1397.80 | 1387.97 | 1391.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 1397.80 | 1387.97 | 1391.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1399.10 | 1390.20 | 1391.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 1399.00 | 1390.20 | 1391.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1397.50 | 1393.25 | 1392.93 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 1387.10 | 1392.49 | 1392.88 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1419.80 | 1397.95 | 1395.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 10:15:00 | 1433.30 | 1405.02 | 1398.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 1407.60 | 1413.25 | 1404.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 1407.60 | 1413.25 | 1404.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1423.80 | 1425.37 | 1417.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:45:00 | 1420.10 | 1425.37 | 1417.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1428.30 | 1425.96 | 1418.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1412.30 | 1425.96 | 1418.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1408.80 | 1422.52 | 1417.89 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 1406.00 | 1414.17 | 1414.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 09:15:00 | 1394.30 | 1407.09 | 1410.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 15:15:00 | 1332.00 | 1329.94 | 1340.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 09:15:00 | 1367.20 | 1329.94 | 1340.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1367.00 | 1337.35 | 1343.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 1366.60 | 1337.35 | 1343.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1363.30 | 1342.54 | 1344.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 1371.80 | 1342.54 | 1344.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 1358.70 | 1347.75 | 1346.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 13:15:00 | 1362.50 | 1350.70 | 1348.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 15:15:00 | 1371.50 | 1371.95 | 1363.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 09:15:00 | 1371.60 | 1371.95 | 1363.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1368.00 | 1372.73 | 1366.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 1368.00 | 1372.73 | 1366.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1365.00 | 1371.19 | 1366.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 1365.80 | 1371.19 | 1366.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1366.50 | 1370.25 | 1366.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:15:00 | 1372.30 | 1370.25 | 1366.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 1363.90 | 1369.31 | 1366.75 | SL hit (close<static) qty=1.00 sl=1365.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 1359.20 | 1364.57 | 1364.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 1350.20 | 1360.01 | 1362.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 12:15:00 | 1324.00 | 1323.11 | 1331.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 13:00:00 | 1324.00 | 1323.11 | 1331.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1328.50 | 1323.68 | 1328.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 1328.50 | 1323.68 | 1328.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1327.90 | 1324.52 | 1328.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1294.40 | 1326.63 | 1328.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 1336.10 | 1317.12 | 1315.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 1336.10 | 1317.12 | 1315.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 10:15:00 | 1339.10 | 1321.51 | 1317.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 12:15:00 | 1321.60 | 1321.82 | 1318.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 12:15:00 | 1321.60 | 1321.82 | 1318.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1321.60 | 1321.82 | 1318.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:45:00 | 1317.80 | 1321.82 | 1318.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1308.80 | 1319.21 | 1317.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 1308.80 | 1319.21 | 1317.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1312.30 | 1317.83 | 1317.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:30:00 | 1314.00 | 1317.83 | 1317.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 1316.70 | 1317.60 | 1317.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 1314.00 | 1317.60 | 1317.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1320.50 | 1318.18 | 1317.51 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 1311.50 | 1316.85 | 1316.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 14:15:00 | 1303.60 | 1312.49 | 1314.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 09:15:00 | 1317.10 | 1312.86 | 1314.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 1317.10 | 1312.86 | 1314.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1317.10 | 1312.86 | 1314.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:00:00 | 1317.10 | 1312.86 | 1314.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 1326.40 | 1315.57 | 1315.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 1330.90 | 1322.12 | 1318.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 12:15:00 | 1359.40 | 1359.92 | 1347.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 13:00:00 | 1359.40 | 1359.92 | 1347.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1347.20 | 1357.38 | 1347.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:30:00 | 1345.70 | 1357.38 | 1347.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1345.80 | 1355.06 | 1347.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1365.90 | 1353.93 | 1347.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 1349.70 | 1366.97 | 1367.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 1349.70 | 1366.97 | 1367.50 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 1374.50 | 1357.52 | 1357.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 1381.50 | 1362.31 | 1359.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 1381.20 | 1385.18 | 1375.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:45:00 | 1380.60 | 1385.18 | 1375.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1373.10 | 1382.06 | 1376.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 1371.90 | 1382.06 | 1376.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1372.20 | 1380.09 | 1376.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 1372.20 | 1380.09 | 1376.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1371.90 | 1377.92 | 1375.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:45:00 | 1372.30 | 1377.92 | 1375.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 1378.10 | 1377.96 | 1376.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:15:00 | 1393.90 | 1378.03 | 1376.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1367.70 | 1382.50 | 1380.52 | SL hit (close<static) qty=1.00 sl=1371.10 alert=retest2 |

### Cycle 107 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 1364.80 | 1378.96 | 1379.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 12:15:00 | 1358.00 | 1372.15 | 1375.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 12:15:00 | 1349.90 | 1345.90 | 1353.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 12:15:00 | 1349.90 | 1345.90 | 1353.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 1349.90 | 1345.90 | 1353.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:30:00 | 1352.70 | 1345.90 | 1353.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1351.50 | 1346.81 | 1351.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1368.50 | 1346.81 | 1351.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1362.10 | 1349.87 | 1352.69 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 1365.80 | 1355.67 | 1354.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 1372.90 | 1362.41 | 1358.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1367.40 | 1368.79 | 1363.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 09:45:00 | 1365.70 | 1368.79 | 1363.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1371.40 | 1369.33 | 1365.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 1364.20 | 1369.33 | 1365.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1366.30 | 1369.84 | 1366.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 1366.30 | 1369.84 | 1366.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 1337.10 | 1363.29 | 1363.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 1328.90 | 1356.41 | 1360.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 10:15:00 | 1265.20 | 1263.54 | 1282.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 11:00:00 | 1265.20 | 1263.54 | 1282.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 1280.00 | 1266.83 | 1282.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 1280.00 | 1266.83 | 1282.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1280.50 | 1269.56 | 1282.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:30:00 | 1281.80 | 1269.56 | 1282.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 1282.80 | 1272.21 | 1282.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:30:00 | 1280.10 | 1272.21 | 1282.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1280.90 | 1273.95 | 1282.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:30:00 | 1281.10 | 1273.95 | 1282.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1284.00 | 1275.96 | 1282.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 1265.50 | 1275.96 | 1282.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 1202.22 | 1244.79 | 1261.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 13:15:00 | 1168.30 | 1166.73 | 1185.47 | SL hit (close>ema200) qty=0.50 sl=1166.73 alert=retest2 |

### Cycle 110 — BUY (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 13:15:00 | 1200.90 | 1191.70 | 1190.65 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1181.80 | 1190.29 | 1190.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 1176.10 | 1185.66 | 1188.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1201.50 | 1184.66 | 1186.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1201.50 | 1184.66 | 1186.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1201.50 | 1184.66 | 1186.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:45:00 | 1201.40 | 1184.66 | 1186.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1198.10 | 1187.35 | 1187.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 1203.90 | 1187.35 | 1187.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 1197.80 | 1189.44 | 1188.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 1200.00 | 1194.10 | 1191.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1182.20 | 1193.82 | 1191.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1182.20 | 1193.82 | 1191.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1182.20 | 1193.82 | 1191.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1182.20 | 1193.82 | 1191.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1184.10 | 1191.87 | 1190.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1178.30 | 1191.87 | 1190.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 1186.00 | 1189.76 | 1190.08 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 1193.10 | 1190.45 | 1190.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 14:15:00 | 1202.30 | 1195.56 | 1192.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 1194.00 | 1196.42 | 1193.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1194.00 | 1196.42 | 1193.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1194.00 | 1196.42 | 1193.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 1195.70 | 1196.42 | 1193.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1200.70 | 1197.28 | 1194.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:00:00 | 1204.60 | 1198.74 | 1195.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 1245.90 | 1247.17 | 1247.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 1245.90 | 1247.17 | 1247.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 1236.20 | 1244.53 | 1245.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 1248.10 | 1242.65 | 1243.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 14:15:00 | 1248.10 | 1242.65 | 1243.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1248.10 | 1242.65 | 1243.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 1248.10 | 1242.65 | 1243.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1244.00 | 1242.92 | 1243.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 1232.40 | 1242.92 | 1243.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 1170.78 | 1191.01 | 1203.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 15:15:00 | 1170.80 | 1170.24 | 1185.56 | SL hit (close>ema200) qty=0.50 sl=1170.24 alert=retest2 |

### Cycle 116 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1158.00 | 1143.07 | 1141.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 1172.00 | 1154.63 | 1147.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 10:15:00 | 1149.40 | 1158.36 | 1152.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 10:15:00 | 1149.40 | 1158.36 | 1152.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 1149.40 | 1158.36 | 1152.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 1149.40 | 1158.36 | 1152.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 1161.00 | 1158.89 | 1153.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 12:15:00 | 1163.50 | 1158.89 | 1153.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 1159.10 | 1176.58 | 1177.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 1159.10 | 1176.58 | 1177.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 1155.50 | 1164.23 | 1170.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1156.80 | 1155.65 | 1161.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 1156.80 | 1155.65 | 1161.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1156.80 | 1155.65 | 1161.03 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1184.00 | 1165.75 | 1164.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1193.50 | 1174.38 | 1168.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1224.90 | 1226.20 | 1213.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:30:00 | 1223.70 | 1226.20 | 1213.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1210.70 | 1220.27 | 1215.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 1210.70 | 1220.27 | 1215.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 1209.80 | 1218.18 | 1214.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 1222.00 | 1218.18 | 1214.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1223.70 | 1230.20 | 1224.77 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 1208.90 | 1221.47 | 1222.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 1198.40 | 1216.86 | 1219.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 1184.70 | 1184.29 | 1196.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 1184.70 | 1184.29 | 1196.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1192.60 | 1186.22 | 1194.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 1192.60 | 1186.22 | 1194.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1195.00 | 1187.97 | 1194.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1186.10 | 1187.97 | 1194.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1193.60 | 1189.10 | 1194.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:15:00 | 1200.50 | 1189.10 | 1194.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 1197.50 | 1190.78 | 1194.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 1199.10 | 1190.78 | 1194.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 1182.00 | 1189.02 | 1193.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 1177.70 | 1186.08 | 1191.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 14:15:00 | 1118.82 | 1131.19 | 1146.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 09:15:00 | 1059.93 | 1083.36 | 1108.38 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 120 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 1062.60 | 1047.40 | 1045.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 12:15:00 | 1070.00 | 1051.92 | 1047.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 13:15:00 | 1048.50 | 1051.24 | 1047.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 14:00:00 | 1048.50 | 1051.24 | 1047.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1054.50 | 1051.89 | 1048.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:30:00 | 1044.40 | 1051.89 | 1048.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 1059.90 | 1053.49 | 1049.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 1068.00 | 1053.49 | 1049.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 12:15:00 | 1093.40 | 1107.39 | 1108.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 1093.40 | 1107.39 | 1108.67 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 09:15:00 | 1117.60 | 1109.15 | 1108.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 1141.50 | 1121.40 | 1115.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 1146.10 | 1151.51 | 1139.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 10:00:00 | 1146.10 | 1151.51 | 1139.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 1140.10 | 1147.84 | 1141.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:30:00 | 1142.20 | 1147.84 | 1141.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 1138.90 | 1146.06 | 1141.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 1138.90 | 1146.06 | 1141.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 1143.00 | 1145.44 | 1141.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 1158.50 | 1145.44 | 1141.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:15:00 | 1149.90 | 1146.34 | 1143.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 1145.00 | 1143.58 | 1143.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 1127.10 | 1142.21 | 1142.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1127.10 | 1142.21 | 1142.96 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 13:15:00 | 1144.20 | 1140.46 | 1140.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 1153.40 | 1144.28 | 1142.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 15:15:00 | 1167.10 | 1169.70 | 1161.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 09:15:00 | 1152.70 | 1169.70 | 1161.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1151.10 | 1165.98 | 1160.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 1150.40 | 1165.98 | 1160.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1147.30 | 1162.25 | 1159.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 1149.20 | 1162.25 | 1159.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 1149.90 | 1156.81 | 1157.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 1140.80 | 1152.92 | 1155.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 15:15:00 | 1130.80 | 1130.40 | 1139.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 09:15:00 | 1140.30 | 1130.40 | 1139.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1154.80 | 1135.28 | 1140.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 1154.80 | 1135.28 | 1140.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1157.90 | 1139.80 | 1142.35 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1161.90 | 1144.22 | 1144.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 1169.20 | 1153.94 | 1149.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 1144.30 | 1153.48 | 1149.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 1144.30 | 1153.48 | 1149.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1144.30 | 1153.48 | 1149.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1143.20 | 1153.48 | 1149.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 1146.30 | 1152.04 | 1149.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:45:00 | 1147.20 | 1152.04 | 1149.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1145.60 | 1150.75 | 1149.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 1145.60 | 1150.75 | 1149.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1151.00 | 1149.59 | 1148.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 1155.20 | 1149.59 | 1148.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1140.20 | 1148.61 | 1148.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 10:15:00 | 1140.20 | 1148.61 | 1148.61 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 1160.00 | 1150.89 | 1149.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 1165.50 | 1155.44 | 1152.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1159.30 | 1159.79 | 1156.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 1159.30 | 1159.79 | 1156.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1159.30 | 1159.79 | 1156.44 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 1144.50 | 1152.70 | 1153.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 1137.70 | 1148.16 | 1151.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1109.80 | 1106.63 | 1115.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 1109.80 | 1106.63 | 1115.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1112.00 | 1107.71 | 1115.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 1117.40 | 1107.71 | 1115.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1114.90 | 1109.15 | 1115.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1086.80 | 1110.29 | 1113.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 1108.70 | 1105.23 | 1105.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1108.70 | 1105.23 | 1105.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1115.80 | 1107.39 | 1106.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 1106.30 | 1110.95 | 1108.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 13:15:00 | 1106.30 | 1110.95 | 1108.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1106.30 | 1110.95 | 1108.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 1106.60 | 1110.95 | 1108.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1105.00 | 1109.76 | 1108.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1105.00 | 1109.76 | 1108.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1097.90 | 1107.39 | 1107.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1070.20 | 1107.39 | 1107.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1093.90 | 1104.69 | 1106.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 1055.80 | 1076.03 | 1087.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1059.30 | 1044.89 | 1054.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 1059.30 | 1044.89 | 1054.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1059.30 | 1044.89 | 1054.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 1064.00 | 1044.89 | 1054.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1053.10 | 1046.53 | 1054.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 11:15:00 | 1049.00 | 1046.53 | 1054.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 11:45:00 | 1049.90 | 1047.61 | 1054.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:15:00 | 1050.20 | 1047.61 | 1054.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 13:45:00 | 1049.50 | 1048.62 | 1053.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 996.55 | 1006.61 | 1020.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 997.41 | 1006.61 | 1020.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 997.69 | 1006.61 | 1020.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 997.02 | 1006.61 | 1020.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 991.60 | 983.18 | 998.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 991.60 | 983.18 | 998.84 | SL hit (close>ema200) qty=0.50 sl=983.18 alert=retest2 |

### Cycle 132 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1033.50 | 1001.01 | 998.11 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 986.60 | 998.61 | 999.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 984.10 | 993.92 | 997.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 983.80 | 975.96 | 983.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 983.80 | 975.96 | 983.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 983.80 | 975.96 | 983.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:45:00 | 963.60 | 979.67 | 982.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 1001.85 | 981.62 | 982.09 | SL hit (close>static) qty=1.00 sl=996.15 alert=retest2 |

### Cycle 134 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 1002.35 | 985.77 | 983.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 1007.70 | 992.30 | 987.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 1051.95 | 1059.80 | 1046.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 1051.45 | 1059.80 | 1046.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 1052.95 | 1058.43 | 1047.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 1052.95 | 1058.43 | 1047.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1053.75 | 1063.29 | 1057.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1058.50 | 1063.29 | 1057.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-27 14:15:00 | 1164.35 | 1138.10 | 1122.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1115.70 | 1137.84 | 1138.02 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 1176.40 | 1142.26 | 1137.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 1187.40 | 1174.35 | 1164.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 1191.80 | 1202.23 | 1192.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 10:15:00 | 1191.80 | 1202.23 | 1192.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1191.80 | 1202.23 | 1192.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 1191.80 | 1202.23 | 1192.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 1191.30 | 1200.05 | 1191.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 1187.00 | 1200.05 | 1191.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1191.30 | 1198.30 | 1191.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 1191.30 | 1198.30 | 1191.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1191.50 | 1196.94 | 1191.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:30:00 | 1202.00 | 1197.69 | 1192.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-16 15:15:00 | 1391.35 | 2024-05-17 10:15:00 | 1406.60 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-06-04 10:30:00 | 1107.95 | 2024-06-04 12:15:00 | 1052.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 10:30:00 | 1107.95 | 2024-06-05 11:15:00 | 1113.00 | STOP_HIT | 0.50 | -0.46% |
| SELL | retest2 | 2024-06-05 14:15:00 | 1114.15 | 2024-06-06 10:15:00 | 1179.80 | STOP_HIT | 1.00 | -5.89% |
| BUY | retest1 | 2024-06-11 15:15:00 | 1228.00 | 2024-06-14 09:15:00 | 1289.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-12 14:15:00 | 1232.40 | 2024-06-14 09:15:00 | 1294.02 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-11 15:15:00 | 1228.00 | 2024-06-14 11:15:00 | 1350.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-06-12 14:15:00 | 1232.40 | 2024-06-14 11:15:00 | 1355.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-27 14:45:00 | 1279.80 | 2024-07-04 13:15:00 | 1284.80 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-06-28 10:00:00 | 1284.30 | 2024-07-04 13:15:00 | 1284.80 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-07-01 09:15:00 | 1252.00 | 2024-07-04 13:15:00 | 1284.80 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-07-02 14:15:00 | 1285.20 | 2024-07-04 13:15:00 | 1284.80 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-07-05 15:00:00 | 1266.10 | 2024-07-15 09:15:00 | 1202.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-08 11:00:00 | 1266.00 | 2024-07-15 09:15:00 | 1202.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-08 12:45:00 | 1267.75 | 2024-07-15 09:15:00 | 1204.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-05 15:00:00 | 1266.10 | 2024-07-15 14:15:00 | 1211.25 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2024-07-08 11:00:00 | 1266.00 | 2024-07-15 14:15:00 | 1211.25 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2024-07-08 12:45:00 | 1267.75 | 2024-07-15 14:15:00 | 1211.25 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2024-08-08 09:30:00 | 1169.85 | 2024-08-12 09:15:00 | 1111.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 10:45:00 | 1173.70 | 2024-08-12 09:15:00 | 1115.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 13:45:00 | 1172.05 | 2024-08-12 09:15:00 | 1113.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 15:15:00 | 1160.95 | 2024-08-12 09:15:00 | 1102.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 10:30:00 | 1149.55 | 2024-08-12 09:15:00 | 1092.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 11:30:00 | 1148.20 | 2024-08-12 09:15:00 | 1090.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 09:30:00 | 1169.85 | 2024-08-13 10:15:00 | 1091.10 | STOP_HIT | 0.50 | 6.73% |
| SELL | retest2 | 2024-08-08 10:45:00 | 1173.70 | 2024-08-13 10:15:00 | 1091.10 | STOP_HIT | 0.50 | 7.04% |
| SELL | retest2 | 2024-08-08 13:45:00 | 1172.05 | 2024-08-13 10:15:00 | 1091.10 | STOP_HIT | 0.50 | 6.91% |
| SELL | retest2 | 2024-08-08 15:15:00 | 1160.95 | 2024-08-13 10:15:00 | 1091.10 | STOP_HIT | 0.50 | 6.02% |
| SELL | retest2 | 2024-08-09 10:30:00 | 1149.55 | 2024-08-13 10:15:00 | 1091.10 | STOP_HIT | 0.50 | 5.08% |
| SELL | retest2 | 2024-08-09 11:30:00 | 1148.20 | 2024-08-13 10:15:00 | 1091.10 | STOP_HIT | 0.50 | 4.97% |
| SELL | retest2 | 2024-08-23 15:00:00 | 1074.70 | 2024-08-28 13:15:00 | 1080.45 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-08-26 09:30:00 | 1075.05 | 2024-08-28 13:15:00 | 1080.45 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-08-28 09:15:00 | 1070.60 | 2024-08-28 13:15:00 | 1080.45 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-08-28 12:00:00 | 1073.70 | 2024-08-28 13:15:00 | 1080.45 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-09-03 09:15:00 | 1157.45 | 2024-09-05 13:15:00 | 1130.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-09-30 13:15:00 | 1024.55 | 2024-10-03 13:15:00 | 973.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 13:15:00 | 1024.55 | 2024-10-04 09:15:00 | 986.95 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2024-09-30 15:15:00 | 1020.00 | 2024-10-04 09:15:00 | 969.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 15:15:00 | 1020.00 | 2024-10-04 09:15:00 | 986.95 | STOP_HIT | 0.50 | 3.24% |
| BUY | retest2 | 2024-10-10 11:30:00 | 1011.50 | 2024-10-16 10:15:00 | 1112.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-28 10:30:00 | 1067.40 | 2024-11-04 14:15:00 | 1072.65 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2024-11-06 10:30:00 | 1065.55 | 2024-11-07 09:15:00 | 1095.45 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2024-11-06 15:15:00 | 1066.80 | 2024-11-07 09:15:00 | 1095.45 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-11-21 12:15:00 | 1077.50 | 2024-11-27 10:15:00 | 1081.95 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2024-11-21 13:00:00 | 1074.70 | 2024-11-27 10:15:00 | 1081.95 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2024-11-21 14:30:00 | 1074.50 | 2024-11-27 10:15:00 | 1081.95 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2024-11-22 09:15:00 | 1083.10 | 2024-11-27 10:15:00 | 1081.95 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-11-22 11:00:00 | 1094.85 | 2024-11-27 10:15:00 | 1081.95 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-11-22 12:30:00 | 1100.75 | 2024-11-27 10:15:00 | 1081.95 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-11-29 12:15:00 | 1071.80 | 2024-12-02 09:15:00 | 1102.50 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2024-11-29 13:30:00 | 1071.80 | 2024-12-02 09:15:00 | 1102.50 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2024-12-11 12:00:00 | 1130.00 | 2024-12-17 12:15:00 | 1121.90 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest2 | 2024-12-11 12:30:00 | 1135.05 | 2024-12-17 12:15:00 | 1121.90 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2024-12-11 13:15:00 | 1133.20 | 2024-12-17 12:15:00 | 1121.90 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2024-12-19 10:15:00 | 1130.85 | 2024-12-20 13:15:00 | 1114.75 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-12-19 11:15:00 | 1132.50 | 2024-12-20 13:15:00 | 1114.75 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-12-19 13:00:00 | 1131.65 | 2024-12-20 13:15:00 | 1114.75 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-12-20 09:30:00 | 1130.75 | 2024-12-20 13:15:00 | 1114.75 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-12-24 14:15:00 | 1099.80 | 2024-12-27 09:15:00 | 1113.95 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-12-24 15:00:00 | 1097.70 | 2024-12-27 09:15:00 | 1113.95 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-01-08 11:15:00 | 1075.35 | 2025-01-08 14:15:00 | 1083.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-01-13 09:15:00 | 1057.60 | 2025-01-20 11:15:00 | 1004.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-13 09:15:00 | 1057.60 | 2025-01-21 09:15:00 | 1044.45 | STOP_HIT | 0.50 | 1.24% |
| BUY | retest1 | 2025-01-22 10:30:00 | 1053.05 | 2025-01-24 09:15:00 | 1039.80 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-01-23 11:45:00 | 1064.55 | 2025-01-27 09:15:00 | 1021.20 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2025-01-28 14:30:00 | 1023.75 | 2025-01-29 12:15:00 | 1043.55 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-01-31 12:45:00 | 1055.80 | 2025-01-31 14:15:00 | 1039.55 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-02-01 09:15:00 | 1055.25 | 2025-02-01 12:15:00 | 1037.80 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-02-01 11:30:00 | 1055.50 | 2025-02-01 12:15:00 | 1037.80 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-02-01 14:00:00 | 1090.65 | 2025-02-03 12:15:00 | 1041.35 | STOP_HIT | 1.00 | -4.52% |
| BUY | retest2 | 2025-02-25 12:30:00 | 1227.65 | 2025-02-27 12:15:00 | 1190.85 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-03-10 09:30:00 | 1231.00 | 2025-03-10 14:15:00 | 1208.75 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-03-10 11:00:00 | 1230.50 | 2025-03-10 14:15:00 | 1208.75 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-03-25 09:15:00 | 1259.95 | 2025-03-26 12:15:00 | 1238.15 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-04-24 12:15:00 | 1243.40 | 2025-04-29 13:15:00 | 1181.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-24 12:15:00 | 1243.40 | 2025-04-30 09:15:00 | 1195.60 | STOP_HIT | 0.50 | 3.84% |
| BUY | retest1 | 2025-05-15 09:15:00 | 1248.90 | 2025-05-15 13:15:00 | 1236.20 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest1 | 2025-05-15 12:15:00 | 1245.40 | 2025-05-15 13:15:00 | 1236.20 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-05-15 15:15:00 | 1247.90 | 2025-05-16 09:15:00 | 1191.10 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2025-05-27 12:00:00 | 1210.50 | 2025-05-29 11:15:00 | 1197.40 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-05-27 13:30:00 | 1211.90 | 2025-05-29 11:15:00 | 1197.40 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-05-28 10:00:00 | 1211.00 | 2025-05-29 11:15:00 | 1197.40 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-05-28 11:15:00 | 1210.60 | 2025-05-29 11:15:00 | 1197.40 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-06-12 12:30:00 | 1194.30 | 2025-06-16 11:15:00 | 1205.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-19 11:15:00 | 1177.00 | 2025-06-23 09:15:00 | 1118.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-19 11:15:00 | 1177.00 | 2025-06-23 10:15:00 | 1141.90 | STOP_HIT | 0.50 | 2.98% |
| BUY | retest2 | 2025-07-07 12:45:00 | 1191.30 | 2025-07-09 11:15:00 | 1310.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-31 13:45:00 | 1313.50 | 2025-08-01 10:15:00 | 1326.80 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-31 15:00:00 | 1307.10 | 2025-08-01 10:15:00 | 1326.80 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-08-01 09:45:00 | 1311.00 | 2025-08-01 10:15:00 | 1326.80 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-08-18 12:45:00 | 1349.60 | 2025-08-19 13:15:00 | 1372.10 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-08-21 10:45:00 | 1386.70 | 2025-08-29 13:15:00 | 1388.70 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-08-21 12:30:00 | 1389.40 | 2025-08-29 13:15:00 | 1388.70 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-08-22 13:30:00 | 1386.00 | 2025-08-29 13:15:00 | 1388.70 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-08-22 15:15:00 | 1385.00 | 2025-08-29 13:15:00 | 1388.70 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2025-08-26 11:30:00 | 1392.30 | 2025-08-29 13:15:00 | 1388.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-08-26 12:15:00 | 1392.30 | 2025-08-29 13:15:00 | 1388.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-08-26 13:00:00 | 1392.50 | 2025-08-29 13:15:00 | 1388.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-08-28 09:30:00 | 1394.40 | 2025-08-29 13:15:00 | 1388.70 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-09-18 15:15:00 | 1372.30 | 2025-09-19 09:15:00 | 1363.90 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1294.40 | 2025-09-30 09:15:00 | 1336.10 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-10-08 09:15:00 | 1365.90 | 2025-10-10 11:15:00 | 1349.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-20 12:15:00 | 1393.90 | 2025-10-23 09:15:00 | 1367.70 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-11-07 09:15:00 | 1265.50 | 2025-11-10 09:15:00 | 1202.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-07 09:15:00 | 1265.50 | 2025-11-12 13:15:00 | 1168.30 | STOP_HIT | 0.50 | 7.68% |
| BUY | retest2 | 2025-11-20 12:00:00 | 1204.60 | 2025-11-28 11:15:00 | 1245.90 | STOP_HIT | 1.00 | 3.43% |
| SELL | retest2 | 2025-12-02 09:15:00 | 1232.40 | 2025-12-05 09:15:00 | 1170.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 09:15:00 | 1232.40 | 2025-12-05 15:15:00 | 1170.80 | STOP_HIT | 0.50 | 5.00% |
| BUY | retest2 | 2025-12-22 12:15:00 | 1163.50 | 2025-12-29 09:15:00 | 1159.10 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-01-13 12:30:00 | 1177.70 | 2026-01-19 14:15:00 | 1118.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:30:00 | 1177.70 | 2026-01-21 09:15:00 | 1059.93 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-01 09:15:00 | 1068.00 | 2026-02-05 12:15:00 | 1093.40 | STOP_HIT | 1.00 | 2.38% |
| BUY | retest2 | 2026-02-11 09:15:00 | 1158.50 | 2026-02-13 09:15:00 | 1127.10 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2026-02-11 13:15:00 | 1149.90 | 2026-02-13 09:15:00 | 1127.10 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-02-12 11:15:00 | 1145.00 | 2026-02-13 09:15:00 | 1127.10 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-02-25 09:15:00 | 1155.20 | 2026-02-25 10:15:00 | 1140.20 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1086.80 | 2026-03-10 13:15:00 | 1108.70 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-03-18 11:15:00 | 1049.00 | 2026-03-23 09:15:00 | 996.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 11:45:00 | 1049.90 | 2026-03-23 09:15:00 | 997.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 12:15:00 | 1050.20 | 2026-03-23 09:15:00 | 997.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 13:45:00 | 1049.50 | 2026-03-23 09:15:00 | 997.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 11:15:00 | 1049.00 | 2026-03-24 09:15:00 | 991.60 | STOP_HIT | 0.50 | 5.47% |
| SELL | retest2 | 2026-03-18 11:45:00 | 1049.90 | 2026-03-24 09:15:00 | 991.60 | STOP_HIT | 0.50 | 5.55% |
| SELL | retest2 | 2026-03-18 12:15:00 | 1050.20 | 2026-03-24 09:15:00 | 991.60 | STOP_HIT | 0.50 | 5.58% |
| SELL | retest2 | 2026-03-18 13:45:00 | 1049.50 | 2026-03-24 09:15:00 | 991.60 | STOP_HIT | 0.50 | 5.52% |
| SELL | retest2 | 2026-03-24 13:15:00 | 975.00 | 2026-03-25 09:15:00 | 1018.20 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2026-04-02 09:45:00 | 963.60 | 2026-04-02 13:15:00 | 1001.85 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1058.50 | 2026-04-27 14:15:00 | 1164.35 | TARGET_HIT | 1.00 | 10.00% |
