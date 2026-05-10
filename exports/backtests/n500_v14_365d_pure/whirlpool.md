# Whirlpool of India Ltd. (WHIRLPOOL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 954.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 17
- **Target hits / Stop hits / Partials:** 2 / 19 / 4
- **Avg / median % per leg:** -0.08% / -1.68%
- **Sum % (uncompounded):** -1.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.98% | -13.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.98% | -13.9% |
| SELL (all) | 18 | 8 | 44.4% | 2 | 12 | 4 | 0.66% | 11.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 8 | 44.4% | 2 | 12 | 4 | 0.66% | 11.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 8 | 32.0% | 2 | 19 | 4 | -0.08% | -2.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 1304.20 | 1188.53 | 1188.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 1357.00 | 1232.76 | 1216.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 12:15:00 | 1339.40 | 1341.08 | 1296.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 12:45:00 | 1340.90 | 1341.08 | 1296.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1343.10 | 1379.09 | 1340.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:45:00 | 1341.20 | 1379.09 | 1340.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1345.80 | 1378.76 | 1340.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:45:00 | 1343.40 | 1378.76 | 1340.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1340.00 | 1378.05 | 1340.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 1340.00 | 1378.05 | 1340.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1334.50 | 1377.61 | 1340.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:30:00 | 1323.90 | 1377.61 | 1340.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1336.00 | 1377.20 | 1340.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1339.90 | 1377.20 | 1340.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1323.90 | 1376.17 | 1339.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 1323.00 | 1376.17 | 1339.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1340.00 | 1375.43 | 1339.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 1340.00 | 1375.43 | 1339.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1332.60 | 1375.00 | 1339.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 1332.60 | 1375.00 | 1339.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1313.90 | 1374.39 | 1339.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 1313.90 | 1374.39 | 1339.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 1339.60 | 1372.66 | 1339.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:45:00 | 1336.50 | 1372.66 | 1339.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1340.00 | 1372.34 | 1339.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1340.00 | 1372.34 | 1339.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1338.40 | 1372.00 | 1339.51 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1287.00 | 1319.14 | 1319.22 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 10:15:00 | 1369.60 | 1318.68 | 1318.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 1378.00 | 1323.82 | 1321.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 1328.50 | 1332.85 | 1326.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 14:15:00 | 1328.50 | 1332.85 | 1326.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1328.50 | 1332.85 | 1326.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 1330.00 | 1332.85 | 1326.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1338.50 | 1332.91 | 1326.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:15:00 | 1347.10 | 1332.91 | 1326.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:45:00 | 1345.70 | 1334.10 | 1327.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 1355.00 | 1334.18 | 1327.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 12:15:00 | 1344.40 | 1337.14 | 1329.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1331.30 | 1337.33 | 1329.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:30:00 | 1328.40 | 1337.33 | 1329.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1326.70 | 1337.22 | 1329.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 1329.60 | 1337.22 | 1329.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1336.00 | 1337.21 | 1329.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 1329.10 | 1337.21 | 1329.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1333.50 | 1337.14 | 1329.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 14:45:00 | 1345.40 | 1336.99 | 1330.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 1345.80 | 1337.10 | 1330.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 1325.90 | 1336.84 | 1330.18 | SL hit (close<static) qty=1.00 sl=1326.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 1325.90 | 1336.84 | 1330.18 | SL hit (close<static) qty=1.00 sl=1326.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 15:15:00 | 1320.00 | 1335.93 | 1329.98 | SL hit (close<static) qty=1.00 sl=1321.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 15:15:00 | 1320.00 | 1335.93 | 1329.98 | SL hit (close<static) qty=1.00 sl=1321.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 15:15:00 | 1320.00 | 1335.93 | 1329.98 | SL hit (close<static) qty=1.00 sl=1321.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 15:15:00 | 1320.00 | 1335.93 | 1329.98 | SL hit (close<static) qty=1.00 sl=1321.10 alert=retest2 |

### Cycle 4 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 1233.50 | 1324.06 | 1324.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 10:15:00 | 1222.10 | 1321.18 | 1322.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 1239.70 | 1238.29 | 1271.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 15:00:00 | 1239.70 | 1238.29 | 1271.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1321.30 | 1239.15 | 1272.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 1314.90 | 1239.15 | 1272.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1331.50 | 1240.07 | 1272.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 1334.60 | 1240.07 | 1272.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 1407.30 | 1297.64 | 1297.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 1423.00 | 1302.06 | 1299.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1314.00 | 1320.43 | 1309.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:00:00 | 1314.00 | 1320.43 | 1309.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1306.40 | 1320.29 | 1309.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 1306.40 | 1320.29 | 1309.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1315.30 | 1320.24 | 1309.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 13:15:00 | 1338.90 | 1320.19 | 1309.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 1304.00 | 1320.61 | 1310.32 | SL hit (close<static) qty=1.00 sl=1306.30 alert=retest2 |

### Cycle 6 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1252.20 | 1301.47 | 1301.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 15:15:00 | 1241.30 | 1298.69 | 1300.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 839.85 | 838.07 | 924.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:30:00 | 840.10 | 838.07 | 924.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 898.25 | 843.11 | 922.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 14:15:00 | 890.15 | 845.30 | 921.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:00:00 | 891.80 | 851.91 | 920.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 889.35 | 853.10 | 920.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 888.75 | 856.52 | 919.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 915.85 | 860.00 | 918.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 916.55 | 860.00 | 918.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 926.00 | 861.21 | 918.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 926.00 | 861.21 | 918.25 | SL hit (close>static) qty=1.00 sl=922.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 926.00 | 861.21 | 918.25 | SL hit (close>static) qty=1.00 sl=922.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 926.00 | 861.21 | 918.25 | SL hit (close>static) qty=1.00 sl=922.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 926.00 | 861.21 | 918.25 | SL hit (close>static) qty=1.00 sl=922.40 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-17 13:00:00 | 926.00 | 861.21 | 918.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 933.00 | 861.93 | 918.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 933.00 | 861.93 | 918.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 918.10 | 869.03 | 918.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 912.10 | 869.46 | 918.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 10:00:00 | 915.15 | 871.64 | 918.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 10:15:00 | 930.50 | 872.23 | 918.87 | SL hit (close>static) qty=1.00 sl=923.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 10:15:00 | 930.50 | 872.23 | 918.87 | SL hit (close>static) qty=1.00 sl=923.85 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:15:00 | 916.55 | 879.67 | 919.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:15:00 | 908.70 | 880.08 | 919.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 901.95 | 880.30 | 919.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:30:00 | 897.50 | 884.36 | 918.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 14:15:00 | 900.80 | 884.58 | 918.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 14:45:00 | 900.50 | 884.76 | 918.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 13:15:00 | 921.15 | 886.48 | 918.65 | SL hit (close>static) qty=1.00 sl=920.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 13:15:00 | 921.15 | 886.48 | 918.65 | SL hit (close>static) qty=1.00 sl=920.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 13:15:00 | 921.15 | 886.48 | 918.65 | SL hit (close>static) qty=1.00 sl=920.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 10:00:00 | 900.45 | 887.28 | 918.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 915.30 | 890.43 | 916.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 913.60 | 890.43 | 916.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 915.60 | 890.68 | 916.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 898.00 | 890.68 | 916.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 870.72 | 890.75 | 916.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 863.26 | 890.75 | 916.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 895.80 | 888.47 | 913.44 | SL hit (close>ema200) qty=0.50 sl=888.47 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 895.80 | 888.47 | 913.44 | SL hit (close>ema200) qty=0.50 sl=888.47 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 15:15:00 | 855.43 | 887.53 | 911.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 853.10 | 887.05 | 911.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-16 10:15:00 | 810.41 | 882.59 | 907.82 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-19 14:15:00 | 808.20 | 869.97 | 898.14 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 912.45 | 850.95 | 868.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 933.25 | 851.77 | 868.94 | SL hit (close>static) qty=1.00 sl=918.20 alert=retest2 |

### Cycle 7 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 996.50 | 884.31 | 883.77 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-12 10:15:00 | 1347.10 | 2025-09-22 14:15:00 | 1325.90 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-09-15 12:45:00 | 1345.70 | 2025-09-22 14:15:00 | 1325.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-15 15:15:00 | 1355.00 | 2025-09-23 15:15:00 | 1320.00 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-09-17 12:15:00 | 1344.40 | 2025-09-23 15:15:00 | 1320.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-09-19 14:45:00 | 1345.40 | 2025-09-23 15:15:00 | 1320.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-09-22 09:30:00 | 1345.80 | 2025-09-23 15:15:00 | 1320.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-11-07 13:15:00 | 1338.90 | 2025-11-10 09:15:00 | 1304.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-02-10 14:15:00 | 890.15 | 2026-02-17 12:15:00 | 926.00 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2026-02-12 12:00:00 | 891.80 | 2026-02-17 12:15:00 | 926.00 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2026-02-12 14:45:00 | 889.35 | 2026-02-17 12:15:00 | 926.00 | STOP_HIT | 1.00 | -4.12% |
| SELL | retest2 | 2026-02-16 09:15:00 | 888.75 | 2026-02-17 12:15:00 | 926.00 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2026-02-19 12:00:00 | 912.10 | 2026-02-20 10:15:00 | 930.50 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-02-20 10:00:00 | 915.15 | 2026-02-20 10:15:00 | 930.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-02-24 12:15:00 | 916.55 | 2026-02-27 13:15:00 | 921.15 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-02-24 13:15:00 | 908.70 | 2026-02-27 13:15:00 | 921.15 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-02-26 12:30:00 | 897.50 | 2026-02-27 13:15:00 | 921.15 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2026-02-26 14:15:00 | 900.80 | 2026-03-09 09:15:00 | 870.72 | PARTIAL | 0.50 | 3.34% |
| SELL | retest2 | 2026-02-26 14:45:00 | 900.50 | 2026-03-09 09:15:00 | 863.26 | PARTIAL | 0.50 | 4.13% |
| SELL | retest2 | 2026-02-26 14:15:00 | 900.80 | 2026-03-11 09:15:00 | 895.80 | STOP_HIT | 0.50 | 0.56% |
| SELL | retest2 | 2026-02-26 14:45:00 | 900.50 | 2026-03-11 09:15:00 | 895.80 | STOP_HIT | 0.50 | 0.52% |
| SELL | retest2 | 2026-03-02 10:00:00 | 900.45 | 2026-03-12 15:15:00 | 855.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 12:15:00 | 898.00 | 2026-03-13 09:15:00 | 853.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 10:00:00 | 900.45 | 2026-03-16 10:15:00 | 810.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-06 12:15:00 | 898.00 | 2026-03-19 14:15:00 | 808.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-22 09:15:00 | 912.45 | 2026-04-22 09:15:00 | 933.25 | STOP_HIT | 1.00 | -2.28% |
