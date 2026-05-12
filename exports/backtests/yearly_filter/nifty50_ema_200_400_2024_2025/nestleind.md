# NESTLEIND (NESTLEIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1475.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 66 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 51 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 45
- **Target hits / Stop hits / Partials:** 2 / 49 / 2
- **Avg / median % per leg:** -0.73% / -1.18%
- **Sum % (uncompounded):** -38.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 2 | 11.8% | 2 | 15 | 0 | -0.14% | -2.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 2 | 11.8% | 2 | 15 | 0 | -0.14% | -2.5% |
| SELL (all) | 36 | 6 | 16.7% | 0 | 34 | 2 | -1.00% | -36.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 6 | 16.7% | 0 | 34 | 2 | -1.00% | -36.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 8 | 15.1% | 2 | 49 | 2 | -0.73% | -38.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 13:15:00 | 1255.47 | 1252.14 | 1252.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 1261.95 | 1252.37 | 1252.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 09:15:00 | 1264.00 | 1282.81 | 1271.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 1264.00 | 1282.81 | 1271.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1264.00 | 1282.81 | 1271.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 1265.00 | 1282.81 | 1271.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1269.72 | 1282.68 | 1271.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 12:15:00 | 1272.70 | 1282.53 | 1271.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 14:00:00 | 1273.00 | 1282.36 | 1271.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 15:15:00 | 1275.50 | 1282.25 | 1271.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 10:15:00 | 1253.03 | 1281.79 | 1271.65 | SL hit (close<static) qty=1.00 sl=1263.63 alert=retest2 |

### Cycle 2 — SELL (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 10:15:00 | 1257.65 | 1263.66 | 1263.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 1251.50 | 1263.35 | 1263.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 14:15:00 | 1263.65 | 1256.80 | 1259.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 14:15:00 | 1263.65 | 1256.80 | 1259.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 1263.65 | 1256.80 | 1259.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 1263.65 | 1256.80 | 1259.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 1262.00 | 1256.85 | 1259.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 1251.78 | 1256.85 | 1259.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 1260.58 | 1256.87 | 1259.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 12:00:00 | 1260.58 | 1256.87 | 1259.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 1261.50 | 1256.91 | 1259.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 12:45:00 | 1261.47 | 1256.91 | 1259.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 1250.97 | 1256.89 | 1259.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 10:30:00 | 1250.15 | 1260.10 | 1261.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 13:30:00 | 1249.75 | 1259.84 | 1260.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 14:00:00 | 1249.80 | 1259.84 | 1260.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:00:00 | 1250.25 | 1259.48 | 1260.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1264.50 | 1258.40 | 1260.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 1264.50 | 1258.40 | 1260.01 | SL hit (close>static) qty=1.00 sl=1261.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 09:15:00 | 1280.75 | 1261.06 | 1260.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 11:15:00 | 1289.22 | 1261.57 | 1261.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 1307.10 | 1308.16 | 1289.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 14:00:00 | 1307.10 | 1308.16 | 1289.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 1287.50 | 1307.48 | 1289.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 14:45:00 | 1285.97 | 1307.48 | 1289.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 15:15:00 | 1285.00 | 1307.26 | 1289.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:15:00 | 1284.40 | 1307.26 | 1289.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 1284.00 | 1306.79 | 1289.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:45:00 | 1278.72 | 1306.79 | 1289.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1289.13 | 1305.86 | 1289.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1280.50 | 1305.86 | 1289.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1235.88 | 1305.16 | 1289.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 1235.88 | 1305.16 | 1289.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1260.65 | 1304.72 | 1289.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 11:30:00 | 1261.97 | 1304.30 | 1288.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:45:00 | 1263.55 | 1302.23 | 1288.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 13:30:00 | 1262.05 | 1300.53 | 1287.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 1222.50 | 1289.34 | 1283.24 | SL hit (close<static) qty=1.00 sl=1235.05 alert=retest2 |

### Cycle 4 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 1171.00 | 1277.63 | 1277.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 13:15:00 | 1167.50 | 1256.53 | 1266.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 1113.50 | 1106.45 | 1137.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 11:00:00 | 1113.50 | 1106.45 | 1137.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1128.22 | 1106.27 | 1133.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:45:00 | 1129.78 | 1106.27 | 1133.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1129.60 | 1107.62 | 1133.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:45:00 | 1135.97 | 1107.62 | 1133.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 1134.25 | 1107.88 | 1133.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 11:00:00 | 1134.25 | 1107.88 | 1133.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 1135.97 | 1108.16 | 1133.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 11:30:00 | 1134.63 | 1108.16 | 1133.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 1127.15 | 1108.35 | 1133.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:30:00 | 1131.35 | 1108.35 | 1133.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1124.45 | 1102.55 | 1119.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:00:00 | 1124.45 | 1102.55 | 1119.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 1136.97 | 1102.90 | 1120.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 1136.97 | 1102.90 | 1120.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 1124.03 | 1116.05 | 1124.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 14:45:00 | 1124.85 | 1116.05 | 1124.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 1124.72 | 1116.13 | 1124.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:15:00 | 1124.43 | 1116.13 | 1124.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1117.25 | 1116.15 | 1124.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 10:15:00 | 1115.15 | 1116.15 | 1124.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 14:00:00 | 1114.65 | 1116.25 | 1124.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 14:45:00 | 1115.68 | 1116.29 | 1124.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 1111.20 | 1116.35 | 1124.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1116.20 | 1116.04 | 1124.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 1116.20 | 1116.04 | 1124.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 1116.50 | 1109.01 | 1117.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:00:00 | 1116.50 | 1109.01 | 1117.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 1114.50 | 1109.06 | 1117.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:30:00 | 1116.05 | 1109.06 | 1117.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1117.93 | 1109.20 | 1117.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:30:00 | 1121.25 | 1109.20 | 1117.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 1122.93 | 1109.34 | 1117.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:15:00 | 1125.15 | 1109.34 | 1117.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-27 14:15:00 | 1139.50 | 1111.50 | 1118.01 | SL hit (close>static) qty=1.00 sl=1134.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 10:15:00 | 1169.47 | 1116.34 | 1116.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 11:15:00 | 1176.85 | 1116.95 | 1116.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 1165.00 | 1165.27 | 1146.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 13:00:00 | 1165.00 | 1165.27 | 1146.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1149.05 | 1166.46 | 1149.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 1149.05 | 1166.46 | 1149.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 1155.25 | 1166.35 | 1149.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:15:00 | 1156.35 | 1166.35 | 1149.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 14:15:00 | 1156.65 | 1166.03 | 1149.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1142.95 | 1207.29 | 1197.00 | SL hit (close<static) qty=1.00 sl=1147.50 alert=retest2 |

### Cycle 6 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 1115.85 | 1187.85 | 1187.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1111.90 | 1169.11 | 1177.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1161.20 | 1146.25 | 1163.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1161.20 | 1146.25 | 1163.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1161.20 | 1146.25 | 1163.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1161.20 | 1146.25 | 1163.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1166.00 | 1146.45 | 1163.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:00:00 | 1166.00 | 1146.45 | 1163.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 1160.90 | 1146.59 | 1163.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:45:00 | 1158.20 | 1146.87 | 1163.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1157.60 | 1146.97 | 1162.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 1171.60 | 1147.22 | 1162.88 | SL hit (close>static) qty=1.00 sl=1167.80 alert=retest2 |

### Cycle 7 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1206.00 | 1171.03 | 1170.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 1216.20 | 1171.86 | 1171.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1186.50 | 1187.13 | 1180.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:45:00 | 1186.50 | 1187.13 | 1180.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1171.90 | 1187.00 | 1180.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 1171.90 | 1187.00 | 1180.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1168.30 | 1186.81 | 1180.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 1168.30 | 1186.81 | 1180.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1182.60 | 1185.53 | 1179.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 1178.00 | 1185.53 | 1179.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 1177.50 | 1185.42 | 1179.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 1177.50 | 1185.42 | 1179.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1180.00 | 1185.36 | 1179.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 1183.50 | 1185.36 | 1179.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 11:15:00 | 1173.30 | 1185.13 | 1179.90 | SL hit (close<static) qty=1.00 sl=1176.30 alert=retest2 |

### Cycle 8 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 1234.50 | 1278.90 | 1279.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1221.30 | 1277.41 | 1278.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 11:30:00 | 1218.90 | 1222.43 | 1242.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 1215.40 | 1222.04 | 1242.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:00:00 | 1220.10 | 1221.97 | 1241.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:45:00 | 1220.00 | 1221.97 | 1241.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 1240.40 | 1222.74 | 1241.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:00:00 | 1240.40 | 1222.74 | 1241.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 1248.60 | 1223.00 | 1241.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 15:00:00 | 1248.60 | 1223.00 | 1241.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 1250.00 | 1223.26 | 1241.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 1237.70 | 1223.26 | 1241.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 13:15:00 | 1253.60 | 1224.69 | 1241.22 | SL hit (close>static) qty=1.00 sl=1250.60 alert=retest2 |

### Cycle 9 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1408.00 | 1253.92 | 1253.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 1416.30 | 1264.47 | 1259.24 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-14 09:15:00 | 1247.50 | 2024-05-31 14:15:00 | 1185.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-18 12:15:00 | 1250.00 | 2024-05-31 14:15:00 | 1187.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-14 09:15:00 | 1247.50 | 2024-06-05 09:15:00 | 1271.95 | STOP_HIT | 0.50 | -1.96% |
| SELL | retest2 | 2024-05-18 12:15:00 | 1250.00 | 2024-06-05 09:15:00 | 1271.95 | STOP_HIT | 0.50 | -1.76% |
| SELL | retest2 | 2024-06-05 12:30:00 | 1251.47 | 2024-06-10 09:15:00 | 1264.53 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-06-21 09:30:00 | 1261.63 | 2024-06-25 13:15:00 | 1255.47 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2024-06-24 10:45:00 | 1263.05 | 2024-06-25 13:15:00 | 1255.47 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2024-06-24 12:30:00 | 1262.50 | 2024-06-25 13:15:00 | 1255.47 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2024-06-24 13:00:00 | 1262.78 | 2024-06-25 13:15:00 | 1255.47 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2024-07-24 12:15:00 | 1272.70 | 2024-07-25 10:15:00 | 1253.03 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-07-24 14:00:00 | 1273.00 | 2024-07-25 10:15:00 | 1253.03 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-07-24 15:15:00 | 1275.50 | 2024-07-25 10:15:00 | 1253.03 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-08-28 10:30:00 | 1250.15 | 2024-09-03 09:15:00 | 1264.50 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-08-28 13:30:00 | 1249.75 | 2024-09-03 09:15:00 | 1264.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-08-28 14:00:00 | 1249.80 | 2024-09-03 09:15:00 | 1264.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-08-29 10:00:00 | 1250.25 | 2024-09-03 09:15:00 | 1264.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-09-12 11:15:00 | 1252.15 | 2024-09-13 09:15:00 | 1275.78 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-10-09 11:30:00 | 1261.97 | 2024-10-16 09:15:00 | 1222.50 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2024-10-10 09:45:00 | 1263.55 | 2024-10-16 09:15:00 | 1222.50 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2024-10-10 13:30:00 | 1262.05 | 2024-10-16 09:15:00 | 1222.50 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-02-06 10:15:00 | 1115.15 | 2025-02-27 14:15:00 | 1139.50 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-02-06 14:00:00 | 1114.65 | 2025-02-27 14:15:00 | 1139.50 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-02-06 14:45:00 | 1115.68 | 2025-02-27 14:15:00 | 1139.50 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-02-07 09:15:00 | 1111.20 | 2025-02-27 14:15:00 | 1139.50 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-03-12 10:15:00 | 1107.35 | 2025-03-21 09:15:00 | 1121.95 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-03-20 14:30:00 | 1110.38 | 2025-03-21 09:15:00 | 1121.95 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-03-20 15:00:00 | 1109.90 | 2025-03-21 09:15:00 | 1121.95 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-04-01 13:00:00 | 1110.40 | 2025-04-03 13:15:00 | 1119.05 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-04-02 09:15:00 | 1088.93 | 2025-04-03 14:15:00 | 1124.08 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-04-07 09:15:00 | 1114.18 | 2025-04-07 13:15:00 | 1118.70 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-04-07 12:00:00 | 1113.72 | 2025-04-07 13:15:00 | 1118.70 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-05-09 11:15:00 | 1156.35 | 2025-07-25 09:15:00 | 1142.95 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-05-09 14:15:00 | 1156.65 | 2025-07-25 09:15:00 | 1142.95 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-08-18 13:45:00 | 1158.20 | 2025-08-20 09:15:00 | 1171.60 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1157.60 | 2025-08-20 09:15:00 | 1171.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-08-22 15:00:00 | 1160.40 | 2025-08-26 09:15:00 | 1169.90 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-26 15:15:00 | 1159.20 | 2025-09-01 11:15:00 | 1168.80 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-08-29 10:00:00 | 1150.50 | 2025-09-01 11:15:00 | 1168.80 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-08-29 13:45:00 | 1154.70 | 2025-09-01 11:15:00 | 1168.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-09-25 09:15:00 | 1183.50 | 2025-09-25 11:15:00 | 1173.30 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-06 13:00:00 | 1180.60 | 2025-10-07 14:15:00 | 1176.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-10-06 14:30:00 | 1180.20 | 2025-10-07 14:15:00 | 1176.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-10-07 09:30:00 | 1181.00 | 2025-10-07 14:15:00 | 1176.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-10-07 12:15:00 | 1185.00 | 2025-10-08 09:15:00 | 1166.40 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-10-09 14:00:00 | 1185.00 | 2025-10-14 12:15:00 | 1172.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-13 12:30:00 | 1185.20 | 2025-10-14 12:15:00 | 1172.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1188.70 | 2025-10-17 09:15:00 | 1307.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 10:15:00 | 1201.30 | 2026-01-06 09:15:00 | 1321.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-08 11:30:00 | 1218.90 | 2026-04-15 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2026-04-09 09:45:00 | 1215.40 | 2026-04-15 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2026-04-09 12:00:00 | 1220.10 | 2026-04-15 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2026-04-09 12:45:00 | 1220.00 | 2026-04-15 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1237.70 | 2026-04-15 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -1.28% |
