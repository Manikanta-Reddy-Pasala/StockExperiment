# Lodha Developers Ltd. (LODHA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 960.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 19 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 18
- **Target hits / Stop hits / Partials:** 3 / 20 / 5
- **Avg / median % per leg:** 0.20% / -2.42%
- **Sum % (uncompounded):** 5.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 1 | 8.3% | 1 | 11 | 0 | -2.32% | -27.8% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.35% | -21.4% |
| BUY @ 3rd Alert (retest2) | 8 | 1 | 12.5% | 1 | 7 | 0 | -0.80% | -6.4% |
| SELL (all) | 16 | 9 | 56.2% | 2 | 9 | 5 | 2.10% | 33.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 9 | 56.2% | 2 | 9 | 5 | 2.10% | 33.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.35% | -21.4% |
| retest2 (combined) | 24 | 10 | 41.7% | 3 | 16 | 5 | 1.13% | 27.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 15:15:00 | 1220.00 | 1364.25 | 1364.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 1210.40 | 1362.72 | 1363.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 09:15:00 | 1285.05 | 1246.94 | 1283.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 1285.05 | 1246.94 | 1283.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 1285.05 | 1246.94 | 1283.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 1285.05 | 1246.94 | 1283.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 1276.60 | 1247.24 | 1283.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 12:00:00 | 1264.20 | 1247.40 | 1283.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 12:30:00 | 1264.00 | 1247.54 | 1283.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 1242.25 | 1248.35 | 1283.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 1298.80 | 1253.20 | 1283.42 | SL hit (close>static) qty=1.00 sl=1292.85 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 1306.00 | 1234.08 | 1234.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 10:15:00 | 1317.85 | 1238.95 | 1236.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 1358.35 | 1370.32 | 1322.34 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 14:30:00 | 1398.50 | 1370.62 | 1323.68 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 12:00:00 | 1397.15 | 1370.99 | 1327.91 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 13:15:00 | 1404.95 | 1371.21 | 1328.23 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 14:15:00 | 1401.40 | 1371.45 | 1328.56 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1346.30 | 1371.10 | 1330.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 1346.30 | 1371.10 | 1330.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1325.50 | 1370.32 | 1330.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-07 11:15:00 | 1325.50 | 1370.32 | 1330.50 | SL hit (close<ema400) qty=1.00 sl=1330.50 alert=retest1 |

### Cycle 3 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 1174.00 | 1303.32 | 1303.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 1168.55 | 1291.91 | 1297.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 11:15:00 | 1230.80 | 1219.08 | 1254.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-31 12:00:00 | 1230.80 | 1219.08 | 1254.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1237.45 | 1218.56 | 1252.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 1260.10 | 1218.56 | 1252.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1259.00 | 1218.97 | 1252.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:00:00 | 1259.00 | 1218.97 | 1252.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 1277.80 | 1219.55 | 1252.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 15:00:00 | 1277.80 | 1219.55 | 1252.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 15:15:00 | 1263.80 | 1219.99 | 1252.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:30:00 | 1252.20 | 1220.35 | 1252.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 1285.10 | 1220.99 | 1252.93 | SL hit (close>static) qty=1.00 sl=1284.25 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 11:15:00 | 1352.60 | 1196.96 | 1196.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 13:15:00 | 1357.70 | 1200.09 | 1198.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1241.80 | 1258.82 | 1233.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 1241.80 | 1258.82 | 1233.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1241.80 | 1258.82 | 1233.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 1241.80 | 1258.82 | 1233.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 1229.00 | 1258.53 | 1233.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:30:00 | 1228.70 | 1258.53 | 1233.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 11:15:00 | 1241.60 | 1258.36 | 1233.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 1302.80 | 1257.41 | 1233.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-20 09:15:00 | 1433.08 | 1290.42 | 1256.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 1244.30 | 1373.35 | 1373.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 13:15:00 | 1236.10 | 1371.99 | 1373.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 13:15:00 | 1299.80 | 1285.48 | 1318.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:00:00 | 1299.80 | 1285.48 | 1318.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1305.40 | 1285.93 | 1318.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1320.00 | 1285.93 | 1318.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1202.80 | 1175.70 | 1202.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 1202.80 | 1175.70 | 1202.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1190.40 | 1175.84 | 1201.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1184.50 | 1198.78 | 1206.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 15:15:00 | 1125.27 | 1182.78 | 1196.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 14:15:00 | 1066.05 | 1160.77 | 1182.12 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-16 12:00:00 | 1264.20 | 2024-09-19 09:15:00 | 1298.80 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2024-09-16 12:30:00 | 1264.00 | 2024-09-19 09:15:00 | 1298.80 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-09-17 09:15:00 | 1242.25 | 2024-09-19 09:15:00 | 1298.80 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2024-09-19 11:45:00 | 1263.10 | 2024-09-19 14:15:00 | 1305.90 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1275.50 | 2024-10-03 09:15:00 | 1211.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1275.50 | 2024-10-04 09:15:00 | 1147.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-12 10:30:00 | 1267.30 | 2024-11-13 12:15:00 | 1203.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:30:00 | 1267.30 | 2024-11-13 12:15:00 | 1198.25 | STOP_HIT | 0.50 | 5.45% |
| SELL | retest2 | 2024-11-19 09:30:00 | 1275.00 | 2024-11-22 09:15:00 | 1211.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 09:30:00 | 1275.00 | 2024-11-22 09:15:00 | 1275.90 | STOP_HIT | 0.50 | -0.07% |
| SELL | retest2 | 2024-11-26 13:30:00 | 1278.10 | 2024-11-28 09:15:00 | 1308.45 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest1 | 2024-12-31 14:30:00 | 1398.50 | 2025-01-07 11:15:00 | 1325.50 | STOP_HIT | 1.00 | -5.22% |
| BUY | retest1 | 2025-01-03 12:00:00 | 1397.15 | 2025-01-07 11:15:00 | 1325.50 | STOP_HIT | 1.00 | -5.13% |
| BUY | retest1 | 2025-01-03 13:15:00 | 1404.95 | 2025-01-07 11:15:00 | 1325.50 | STOP_HIT | 1.00 | -5.66% |
| BUY | retest1 | 2025-01-03 14:15:00 | 1401.40 | 2025-01-07 11:15:00 | 1325.50 | STOP_HIT | 1.00 | -5.42% |
| BUY | retest2 | 2025-01-07 13:15:00 | 1345.75 | 2025-01-09 14:15:00 | 1305.95 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-01-08 09:30:00 | 1345.00 | 2025-01-09 14:15:00 | 1305.95 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-01-08 12:15:00 | 1338.55 | 2025-01-09 14:15:00 | 1305.95 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-01-09 10:00:00 | 1338.30 | 2025-01-09 14:15:00 | 1305.95 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-02-03 09:30:00 | 1252.20 | 2025-02-03 10:15:00 | 1285.10 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-02-06 09:15:00 | 1231.50 | 2025-02-11 09:15:00 | 1169.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 1231.50 | 2025-02-21 10:15:00 | 1211.40 | STOP_HIT | 0.50 | 1.63% |
| BUY | retest2 | 2025-05-12 09:15:00 | 1302.80 | 2025-05-20 09:15:00 | 1433.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-28 10:30:00 | 1244.00 | 2025-07-28 13:15:00 | 1213.40 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-07-28 11:15:00 | 1249.50 | 2025-07-28 13:15:00 | 1213.40 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-07-29 09:15:00 | 1248.80 | 2025-07-30 12:15:00 | 1244.30 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1184.50 | 2025-12-01 15:15:00 | 1125.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1184.50 | 2025-12-08 14:15:00 | 1066.05 | TARGET_HIT | 0.50 | 10.00% |
