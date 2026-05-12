# Lloyds Metals And Energy Ltd. (LLOYDSME)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-11 15:15:00 (3605 bars)
- **Last close:** 1755.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 11 |
| TARGET_HIT | 9 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 7
- **Target hits / Stop hits / Partials:** 6 / 12 / 11
- **Avg / median % per leg:** 3.75% / 5.00%
- **Sum % (uncompounded):** 108.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 29 | 22 | 75.9% | 6 | 12 | 11 | 3.75% | 108.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 22 | 75.9% | 6 | 12 | 11 | 3.75% | 108.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 22 | 75.9% | 6 | 12 | 11 | 3.75% | 108.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 1355.40 | 1435.62 | 1435.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 1339.60 | 1434.67 | 1435.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 15:15:00 | 1315.00 | 1314.21 | 1351.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:15:00 | 1324.80 | 1314.21 | 1351.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1345.80 | 1313.32 | 1346.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:00:00 | 1336.40 | 1313.91 | 1346.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:00:00 | 1340.00 | 1314.84 | 1346.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:45:00 | 1339.50 | 1315.11 | 1346.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 1339.60 | 1315.11 | 1346.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 1342.60 | 1315.87 | 1346.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:45:00 | 1343.90 | 1315.87 | 1346.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1345.80 | 1316.43 | 1346.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1341.20 | 1316.43 | 1346.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:45:00 | 1341.90 | 1318.20 | 1344.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 15:15:00 | 1338.50 | 1318.70 | 1344.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:00:00 | 1339.60 | 1319.11 | 1344.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1321.60 | 1319.52 | 1343.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 1335.90 | 1319.52 | 1343.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1350.70 | 1319.98 | 1342.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 1350.70 | 1319.98 | 1342.47 | SL hit (close>static) qty=1.00 sl=1348.70 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 1388.30 | 1298.83 | 1298.44 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 15:15:00 | 1236.00 | 1300.75 | 1300.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 10:15:00 | 1228.90 | 1299.38 | 1300.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1314.00 | 1201.33 | 1240.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 1314.00 | 1201.33 | 1240.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1314.00 | 1201.33 | 1240.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 1314.00 | 1201.33 | 1240.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1251.70 | 1210.69 | 1242.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 1251.70 | 1210.69 | 1242.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 1256.70 | 1211.14 | 1242.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 1256.70 | 1211.14 | 1242.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1233.00 | 1219.09 | 1244.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 10:45:00 | 1231.00 | 1219.18 | 1244.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:15:00 | 1228.70 | 1220.46 | 1243.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 15:15:00 | 1169.45 | 1214.49 | 1237.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 1220.00 | 1214.26 | 1237.44 | SL hit (close>ema200) qty=0.50 sl=1214.26 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 1351.00 | 1226.04 | 1225.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 1353.30 | 1227.31 | 1226.21 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-09 14:00:00 | 1336.40 | 2025-10-21 13:15:00 | 1350.70 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-10 10:00:00 | 1340.00 | 2025-10-21 13:15:00 | 1350.70 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-10-10 10:45:00 | 1339.50 | 2025-10-21 13:15:00 | 1350.70 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-10 11:15:00 | 1339.60 | 2025-10-21 13:15:00 | 1350.70 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1341.20 | 2025-11-13 13:15:00 | 1269.58 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2025-10-15 12:45:00 | 1341.90 | 2025-11-13 13:15:00 | 1273.00 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2025-10-15 15:15:00 | 1338.50 | 2025-11-13 13:15:00 | 1272.52 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2025-10-16 10:00:00 | 1339.60 | 2025-11-13 13:15:00 | 1272.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 13:30:00 | 1321.00 | 2025-11-13 13:15:00 | 1254.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-27 09:15:00 | 1329.70 | 2025-11-13 13:15:00 | 1263.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1341.20 | 2025-11-24 11:15:00 | 1202.76 | TARGET_HIT | 0.50 | 10.32% |
| SELL | retest2 | 2025-10-15 12:45:00 | 1341.90 | 2025-11-24 11:15:00 | 1206.00 | TARGET_HIT | 0.50 | 10.13% |
| SELL | retest2 | 2025-10-15 15:15:00 | 1338.50 | 2025-11-24 11:15:00 | 1205.55 | TARGET_HIT | 0.50 | 9.93% |
| SELL | retest2 | 2025-10-16 10:00:00 | 1339.60 | 2025-11-24 11:15:00 | 1205.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-24 13:30:00 | 1321.00 | 2025-11-24 11:15:00 | 1196.73 | TARGET_HIT | 0.50 | 9.41% |
| SELL | retest2 | 2025-10-27 09:15:00 | 1329.70 | 2025-11-24 14:15:00 | 1188.90 | TARGET_HIT | 0.50 | 10.59% |
| SELL | retest2 | 2026-02-11 10:45:00 | 1231.00 | 2026-02-17 15:15:00 | 1169.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 10:45:00 | 1231.00 | 2026-02-18 11:15:00 | 1220.00 | STOP_HIT | 0.50 | 0.89% |
| SELL | retest2 | 2026-02-12 14:15:00 | 1228.70 | 2026-02-19 10:15:00 | 1167.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 14:15:00 | 1228.70 | 2026-02-25 09:15:00 | 1217.50 | STOP_HIT | 0.50 | 0.91% |
| SELL | retest2 | 2026-02-27 15:00:00 | 1230.20 | 2026-03-04 09:15:00 | 1168.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1218.00 | 2026-03-04 10:15:00 | 1157.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 11:15:00 | 1223.10 | 2026-03-04 10:15:00 | 1161.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:00:00 | 1230.20 | 2026-03-05 14:15:00 | 1203.80 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1218.00 | 2026-03-05 14:15:00 | 1203.80 | STOP_HIT | 0.50 | 1.17% |
| SELL | retest2 | 2026-03-02 11:15:00 | 1223.10 | 2026-03-05 14:15:00 | 1203.80 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2026-03-17 14:00:00 | 1224.60 | 2026-03-18 10:15:00 | 1252.80 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1218.50 | 2026-03-20 09:15:00 | 1252.80 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1196.40 | 2026-03-24 12:15:00 | 1258.30 | STOP_HIT | 1.00 | -5.17% |
