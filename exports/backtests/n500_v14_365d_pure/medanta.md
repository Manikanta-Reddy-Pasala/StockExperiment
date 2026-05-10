# Global Health Ltd. (MEDANTA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1202.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 38 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 36
- **Target hits / Stop hits / Partials:** 1 / 37 / 1
- **Avg / median % per leg:** -0.96% / -1.18%
- **Sum % (uncompounded):** -37.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 1 | 2.9% | 0 | 35 | 0 | -1.39% | -48.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 35 | 1 | 2.9% | 0 | 35 | 0 | -1.39% | -48.6% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.82% | 11.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.82% | 11.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 39 | 3 | 7.7% | 1 | 37 | 1 | -0.96% | -37.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 1139.60 | 1198.63 | 1198.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 1130.90 | 1197.96 | 1198.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1185.50 | 1172.30 | 1183.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 1185.50 | 1172.30 | 1183.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1185.50 | 1172.30 | 1183.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1185.50 | 1172.30 | 1183.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1190.00 | 1172.48 | 1183.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1184.10 | 1173.29 | 1183.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 13:15:00 | 1204.70 | 1174.35 | 1183.62 | SL hit (close>static) qty=1.00 sl=1192.30 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 1300.00 | 1192.16 | 1191.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 13:15:00 | 1309.20 | 1197.59 | 1194.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 1364.80 | 1370.62 | 1329.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 1364.80 | 1370.62 | 1329.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 1332.30 | 1368.55 | 1330.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:45:00 | 1330.80 | 1368.55 | 1330.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1322.60 | 1368.09 | 1330.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 1322.60 | 1368.09 | 1330.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 1321.00 | 1367.62 | 1330.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 1320.70 | 1367.62 | 1330.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 1327.90 | 1366.29 | 1330.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 1329.40 | 1365.91 | 1330.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 1332.00 | 1365.14 | 1330.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 11:15:00 | 1319.20 | 1362.21 | 1335.34 | SL hit (close<static) qty=1.00 sl=1321.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 11:15:00 | 1319.20 | 1362.21 | 1335.34 | SL hit (close<static) qty=1.00 sl=1321.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 14:45:00 | 1329.90 | 1356.14 | 1334.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 1302.80 | 1355.33 | 1334.16 | SL hit (close<static) qty=1.00 sl=1321.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 1332.70 | 1349.21 | 1332.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 1323.30 | 1348.85 | 1332.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 1323.30 | 1348.85 | 1332.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1321.60 | 1348.58 | 1332.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:45:00 | 1317.80 | 1348.58 | 1332.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-30 13:15:00 | 1308.80 | 1348.19 | 1332.23 | SL hit (close<static) qty=1.00 sl=1321.10 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 1332.30 | 1343.78 | 1331.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 1339.00 | 1343.78 | 1331.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 12:15:00 | 1321.10 | 1357.62 | 1346.28 | SL hit (close<static) qty=1.00 sl=1324.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:45:00 | 1332.80 | 1357.31 | 1346.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 1319.30 | 1356.93 | 1346.05 | SL hit (close<static) qty=1.00 sl=1324.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 1247.50 | 1336.19 | 1336.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 1189.30 | 1333.17 | 1334.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 11:15:00 | 1260.80 | 1260.61 | 1289.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 11:30:00 | 1263.80 | 1260.61 | 1289.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1227.50 | 1188.25 | 1220.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 1228.00 | 1188.25 | 1220.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1227.90 | 1188.65 | 1220.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:45:00 | 1228.70 | 1188.65 | 1220.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1225.10 | 1189.37 | 1220.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 15:00:00 | 1210.70 | 1192.37 | 1221.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 1234.80 | 1192.96 | 1221.06 | SL hit (close>static) qty=1.00 sl=1227.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 13:00:00 | 1214.20 | 1196.43 | 1221.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 1153.49 | 1192.23 | 1215.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 1092.78 | 1184.32 | 1209.92 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 12:15:00 | 1186.30 | 1101.69 | 1101.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 1216.10 | 1105.27 | 1103.41 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 1211.00 | 2025-05-19 09:15:00 | 1179.00 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-05-16 12:30:00 | 1195.50 | 2025-05-21 11:15:00 | 1181.40 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-05-20 11:00:00 | 1196.20 | 2025-05-21 11:15:00 | 1181.40 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-05-21 10:45:00 | 1194.70 | 2025-05-27 09:15:00 | 1198.70 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-05-23 13:00:00 | 1200.20 | 2025-05-27 09:15:00 | 1198.70 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-05-23 15:00:00 | 1219.90 | 2025-05-27 09:15:00 | 1198.70 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-05-26 09:30:00 | 1212.70 | 2025-05-27 09:15:00 | 1198.70 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-05-26 10:30:00 | 1216.20 | 2025-05-29 13:15:00 | 1192.10 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-05-26 11:00:00 | 1212.60 | 2025-05-29 13:15:00 | 1192.10 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-27 12:00:00 | 1210.50 | 2025-05-29 13:15:00 | 1192.10 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-05-27 13:30:00 | 1211.90 | 2025-05-29 13:15:00 | 1192.10 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-05-28 10:00:00 | 1211.00 | 2025-05-30 10:15:00 | 1181.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-05-28 11:15:00 | 1210.60 | 2025-06-04 14:15:00 | 1197.40 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-06-03 12:00:00 | 1220.00 | 2025-06-04 14:15:00 | 1197.40 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-06-03 12:30:00 | 1222.70 | 2025-06-04 14:15:00 | 1197.40 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-06-03 13:15:00 | 1219.90 | 2025-06-04 14:15:00 | 1197.40 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-06-04 09:15:00 | 1221.20 | 2025-06-09 13:15:00 | 1198.30 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-06-04 14:30:00 | 1204.90 | 2025-06-09 13:15:00 | 1198.30 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-06-05 09:45:00 | 1204.30 | 2025-06-10 11:15:00 | 1197.10 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-05 10:30:00 | 1206.90 | 2025-06-11 12:15:00 | 1198.80 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-06-05 11:00:00 | 1205.20 | 2025-06-11 14:15:00 | 1194.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-06-06 09:15:00 | 1203.50 | 2025-06-11 14:15:00 | 1194.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-06 09:45:00 | 1205.70 | 2025-06-11 14:15:00 | 1194.50 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-06-10 09:15:00 | 1215.90 | 2025-06-11 14:15:00 | 1194.50 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1204.60 | 2025-06-12 11:15:00 | 1196.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-06-12 11:00:00 | 1204.80 | 2025-06-17 11:15:00 | 1198.40 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-06-16 10:30:00 | 1206.60 | 2025-06-17 11:15:00 | 1198.40 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-06-16 11:45:00 | 1206.00 | 2025-06-17 11:15:00 | 1198.40 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-06-16 12:45:00 | 1216.60 | 2025-06-20 11:15:00 | 1139.60 | STOP_HIT | 1.00 | -6.33% |
| SELL | retest2 | 2025-07-04 09:15:00 | 1184.10 | 2025-07-07 13:15:00 | 1204.70 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-09-15 13:15:00 | 1329.40 | 2025-09-23 11:15:00 | 1319.20 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-09-15 15:15:00 | 1332.00 | 2025-09-23 11:15:00 | 1319.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-09-25 14:45:00 | 1329.90 | 2025-09-26 09:15:00 | 1302.80 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-09-30 09:30:00 | 1332.70 | 2025-09-30 13:15:00 | 1308.80 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-10-06 09:15:00 | 1339.00 | 2025-10-31 12:15:00 | 1321.10 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-31 13:45:00 | 1332.80 | 2025-10-31 14:15:00 | 1319.30 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-01-06 15:00:00 | 1210.70 | 2026-01-07 09:15:00 | 1234.80 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-01-08 13:00:00 | 1214.20 | 2026-01-16 09:15:00 | 1153.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 13:00:00 | 1214.20 | 2026-01-20 09:15:00 | 1092.78 | TARGET_HIT | 0.50 | 10.00% |
