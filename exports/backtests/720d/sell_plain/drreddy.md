# DRREDDY (DRREDDY)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1306.40
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 5 |
| ALERT3 | 5 |
| PENDING | 22 |
| PENDING_CANCEL | 11 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 6
- **Target hits / Stop hits / Partials:** 0 / 11 / 0
- **Avg / median % per leg:** -0.28% / -2.02%
- **Sum % (uncompounded):** -3.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 5 | 45.5% | 0 | 11 | 0 | -0.28% | -3.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 5 | 45.5% | 0 | 11 | 0 | -0.28% | -3.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 5 | 45.5% | 0 | 11 | 0 | -0.28% | -3.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 12:15:00 | 1268.55 | 1329.98 | 1330.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 09:15:00 | 1242.95 | 1327.49 | 1328.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 10:15:00 | 1320.60 | 1311.14 | 1319.70 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 10:15:00 | 1320.60 | 1311.14 | 1319.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1320.60 | 1311.14 | 1319.70 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-07 09:15:00 | 1287.10 | 1310.47 | 1319.11 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 11:15:00 | 1291.05 | 1310.03 | 1318.81 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-12-19 12:15:00 | 1327.35 | 1251.31 | 1268.45 | SL hit (close>static) qty=1.00 sl=1321.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-17 09:15:00 | 1295.00 | 1326.09 | 1310.79 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-17 11:15:00 | 1302.35 | 1325.57 | 1310.69 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-01-20 09:15:00 | 1296.80 | 1324.52 | 1310.52 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-20 10:15:00 | 1303.10 | 1324.30 | 1310.48 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-20 11:15:00 | 1297.40 | 1324.04 | 1310.42 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-20 12:15:00 | 1299.05 | 1323.79 | 1310.36 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-21 10:15:00 | 1296.75 | 1322.70 | 1310.14 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-21 11:15:00 | 1300.40 | 1322.48 | 1310.09 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-21 14:15:00 | 1288.60 | 1321.69 | 1309.88 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-22 09:15:00 | 1302.15 | 1321.20 | 1309.75 | ENTRY2 sustain failed after 1140m |
| Cross detected — sustain check pending | 2025-01-22 11:15:00 | 1297.50 | 1320.80 | 1309.67 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 13:15:00 | 1297.95 | 1320.31 | 1309.53 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 1182.00 | 1299.54 | 1300.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 12:15:00 | 1182.00 | 1299.54 | 1300.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 09:15:00 | 1172.25 | 1238.86 | 1261.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 1164.55 | 1162.32 | 1202.11 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 12:15:00 | 1196.10 | 1165.14 | 1199.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 1196.10 | 1165.14 | 1199.42 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-25 09:15:00 | 1175.35 | 1168.75 | 1199.44 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 11:15:00 | 1173.10 | 1168.86 | 1199.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-24 09:15:00 | 1201.70 | 1153.69 | 1174.47 | SL hit (close>static) qty=1.00 sl=1200.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-24 12:15:00 | 1190.70 | 1154.94 | 1174.79 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-24 14:15:00 | 1198.40 | 1155.71 | 1174.98 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-04-25 09:15:00 | 1180.10 | 1156.44 | 1175.15 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 11:15:00 | 1168.60 | 1156.66 | 1175.08 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-28 09:15:00 | 1202.20 | 1157.75 | 1175.17 | SL hit (close>static) qty=1.00 sl=1200.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-28 13:15:00 | 1193.80 | 1159.28 | 1175.60 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-28 14:15:00 | 1198.00 | 1159.67 | 1175.71 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-29 09:15:00 | 1179.80 | 1160.29 | 1175.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:15:00 | 1172.00 | 1160.61 | 1175.87 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-12 15:15:00 | 1193.60 | 1164.19 | 1174.07 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-13 09:15:00 | 1236.10 | 1164.91 | 1174.38 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2025-05-13 09:15:00 | 1236.10 | 1164.91 | 1174.38 | SL hit (close>static) qty=1.00 sl=1200.95 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-06 10:15:00 | 1193.20 | 1263.91 | 1260.91 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-06 11:15:00 | 1196.40 | 1263.24 | 1260.59 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-06 15:15:00 | 1193.50 | 1260.59 | 1259.30 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:15:00 | 1190.10 | 1259.89 | 1258.96 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-08-07 12:15:00 | 1188.80 | 1257.75 | 1257.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 1188.80 | 1257.75 | 1257.89 | EMA200 below EMA400 |

### Cycle 4 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1251.50 | 1265.94 | 1265.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1250.10 | 1265.79 | 1265.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 1268.60 | 1259.68 | 1262.56 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 1268.60 | 1259.68 | 1262.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1268.60 | 1259.68 | 1262.56 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-29 13:15:00 | 1252.00 | 1267.38 | 1266.26 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 15:15:00 | 1258.40 | 1267.16 | 1266.16 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 1199.10 | 1265.09 | 1265.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 1199.10 | 1265.09 | 1265.13 | EMA200 below EMA400 |

### Cycle 6 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 1247.10 | 1252.53 | 1252.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 1245.00 | 1263.62 | 1259.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 1263.20 | 1261.77 | 1259.27 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 1263.20 | 1261.77 | 1259.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1263.20 | 1261.77 | 1259.27 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-07 11:15:00 | 1244.60 | 1261.57 | 1259.19 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-07 12:15:00 | 1247.30 | 1261.43 | 1259.13 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-07 13:15:00 | 1245.10 | 1261.27 | 1259.06 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 15:15:00 | 1244.90 | 1260.93 | 1258.91 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 1210.20 | 1256.67 | 1256.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1210.20 | 1256.67 | 1256.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1204.20 | 1254.40 | 1255.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.12 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.12 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-28 10:15:00 | 1224.00 | 1227.53 | 1239.03 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-28 11:15:00 | 1226.40 | 1227.52 | 1238.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-28 12:15:00 | 1223.70 | 1227.48 | 1238.90 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 14:15:00 | 1223.30 | 1227.39 | 1238.73 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-02-01 12:15:00 | 1221.60 | 1225.10 | 1236.49 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:15:00 | 1181.00 | 1224.40 | 1236.03 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-02-05 12:15:00 | 1248.00 | 1224.13 | 1234.36 | SL hit (close>static) qty=1.00 sl=1247.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 12:15:00 | 1248.00 | 1224.13 | 1234.36 | SL hit (close>static) qty=1.00 sl=1247.20 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-01 13:15:00 | 1220.80 | 1279.53 | 1270.88 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 1207.50 | 1278.12 | 1270.26 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1187.90 | 1263.24 | 1263.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 09:15:00 | 1187.90 | 1263.24 | 1263.25 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-07 11:15:00 | 1291.05 | 2024-12-19 12:15:00 | 1327.35 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-01-22 13:15:00 | 1297.95 | 2025-01-28 12:15:00 | 1182.00 | STOP_HIT | 1.00 | 8.93% |
| SELL | retest2 | 2025-03-25 11:15:00 | 1173.10 | 2025-04-24 09:15:00 | 1201.70 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-04-25 11:15:00 | 1168.60 | 2025-04-28 09:15:00 | 1202.20 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-04-29 11:15:00 | 1172.00 | 2025-05-13 09:15:00 | 1236.10 | STOP_HIT | 1.00 | -5.47% |
| SELL | retest2 | 2025-08-07 09:15:00 | 1190.10 | 2025-08-07 12:15:00 | 1188.80 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-10-29 15:15:00 | 1258.40 | 2025-10-30 11:15:00 | 1199.10 | STOP_HIT | 1.00 | 4.71% |
| SELL | retest2 | 2026-01-07 15:15:00 | 1244.90 | 2026-01-09 11:15:00 | 1210.20 | STOP_HIT | 1.00 | 2.79% |
| SELL | retest2 | 2026-01-28 14:15:00 | 1223.30 | 2026-02-05 12:15:00 | 1248.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-02-01 14:15:00 | 1181.00 | 2026-02-05 12:15:00 | 1248.00 | STOP_HIT | 1.00 | -5.67% |
| SELL | retest2 | 2026-04-01 15:15:00 | 1207.50 | 2026-04-08 09:15:00 | 1187.90 | STOP_HIT | 1.00 | 1.62% |
