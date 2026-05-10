# United Spirits Ltd. (UNITDSPR)

## Backtest Summary

- **Window:** 2024-07-09 09:15:00 → 2026-05-08 15:15:00 (3168 bars)
- **Last close:** 1284.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 6 / 14
- **Target hits / Stop hits / Partials:** 2 / 14 / 4
- **Avg / median % per leg:** 1.06% / -0.78%
- **Sum % (uncompounded):** 21.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.33% | -5.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.33% | -5.3% |
| SELL (all) | 16 | 6 | 37.5% | 2 | 10 | 4 | 1.65% | 26.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 6 | 37.5% | 2 | 10 | 4 | 1.65% | 26.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 6 | 30.0% | 2 | 14 | 4 | 1.06% | 21.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 1452.50 | 1498.82 | 1499.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 1441.60 | 1497.33 | 1498.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 1334.00 | 1328.83 | 1364.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 09:45:00 | 1333.50 | 1328.83 | 1364.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1342.50 | 1329.94 | 1362.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:15:00 | 1340.10 | 1329.94 | 1362.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:15:00 | 1339.10 | 1322.60 | 1349.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 1341.50 | 1324.83 | 1347.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:30:00 | 1338.20 | 1324.99 | 1347.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1348.30 | 1326.07 | 1347.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1348.30 | 1326.07 | 1347.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1345.30 | 1326.26 | 1347.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:30:00 | 1342.80 | 1328.20 | 1347.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:00:00 | 1339.60 | 1328.20 | 1347.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 1338.80 | 1324.06 | 1342.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 14:00:00 | 1342.70 | 1324.72 | 1342.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 1345.90 | 1324.93 | 1342.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:45:00 | 1345.50 | 1324.93 | 1342.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1347.50 | 1325.15 | 1342.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 1349.40 | 1325.15 | 1342.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1352.50 | 1325.43 | 1342.82 | SL hit (close>static) qty=1.00 sl=1352.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1352.50 | 1325.43 | 1342.82 | SL hit (close>static) qty=1.00 sl=1352.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1352.50 | 1325.43 | 1342.82 | SL hit (close>static) qty=1.00 sl=1352.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1352.50 | 1325.43 | 1342.82 | SL hit (close>static) qty=1.00 sl=1352.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 1350.50 | 1325.68 | 1342.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:30:00 | 1352.10 | 1325.68 | 1342.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 1352.00 | 1328.39 | 1343.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 1354.30 | 1328.39 | 1343.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 1358.60 | 1328.70 | 1343.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:30:00 | 1362.80 | 1328.70 | 1343.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1339.80 | 1330.91 | 1343.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 14:00:00 | 1331.60 | 1331.61 | 1343.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1332.80 | 1331.69 | 1343.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 1347.50 | 1329.36 | 1340.69 | SL hit (close>static) qty=1.00 sl=1343.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 1347.50 | 1329.36 | 1340.69 | SL hit (close>static) qty=1.00 sl=1343.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1365.80 | 1333.13 | 1341.87 | SL hit (close>static) qty=1.00 sl=1364.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1365.80 | 1333.13 | 1341.87 | SL hit (close>static) qty=1.00 sl=1364.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1365.80 | 1333.13 | 1341.87 | SL hit (close>static) qty=1.00 sl=1364.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1365.80 | 1333.13 | 1341.87 | SL hit (close>static) qty=1.00 sl=1364.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 10:15:00 | 1459.90 | 1348.20 | 1348.14 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 11:15:00 | 1318.50 | 1401.31 | 1401.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 1281.10 | 1377.35 | 1388.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 1360.90 | 1360.18 | 1376.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 14:00:00 | 1360.90 | 1360.18 | 1376.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1374.10 | 1357.51 | 1373.08 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 11:15:00 | 1406.00 | 1381.97 | 1381.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 13:15:00 | 1418.80 | 1383.81 | 1382.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 1386.40 | 1388.48 | 1385.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 1386.40 | 1388.48 | 1385.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1386.40 | 1388.48 | 1385.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:30:00 | 1386.40 | 1388.48 | 1385.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1377.80 | 1388.37 | 1385.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:45:00 | 1379.00 | 1388.37 | 1385.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 1380.60 | 1388.29 | 1385.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 1381.20 | 1388.29 | 1385.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1375.50 | 1388.10 | 1385.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 1375.50 | 1388.10 | 1385.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1372.90 | 1387.95 | 1385.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 1372.90 | 1387.95 | 1385.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 1386.80 | 1387.83 | 1385.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:15:00 | 1388.10 | 1387.83 | 1385.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 1380.10 | 1387.75 | 1385.24 | SL hit (close<static) qty=1.00 sl=1384.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:45:00 | 1391.00 | 1387.75 | 1385.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 1380.10 | 1387.67 | 1385.21 | SL hit (close<static) qty=1.00 sl=1384.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 14:15:00 | 1314.00 | 1382.78 | 1382.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 1312.90 | 1379.15 | 1381.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 12:15:00 | 1392.20 | 1376.77 | 1379.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 12:15:00 | 1392.20 | 1376.77 | 1379.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 1392.20 | 1376.77 | 1379.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:00:00 | 1392.20 | 1376.77 | 1379.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1380.30 | 1376.80 | 1379.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1368.00 | 1377.05 | 1379.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 1352.00 | 1377.61 | 1379.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 14:15:00 | 1299.60 | 1366.05 | 1373.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 14:15:00 | 1284.40 | 1358.18 | 1368.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-30 10:15:00 | 1231.20 | 1336.77 | 1355.43 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-30 14:15:00 | 1216.80 | 1332.47 | 1352.89 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:00:00 | 1376.50 | 1295.04 | 1320.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:45:00 | 1375.00 | 1306.50 | 1324.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 1329.20 | 1324.62 | 1332.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:30:00 | 1334.00 | 1324.62 | 1332.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 1325.60 | 1324.63 | 1332.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:45:00 | 1329.20 | 1324.63 | 1332.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1324.80 | 1324.66 | 1332.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 1332.00 | 1324.66 | 1332.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1328.80 | 1324.74 | 1331.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 1322.40 | 1324.74 | 1331.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:30:00 | 1323.40 | 1324.50 | 1331.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 13:15:00 | 1307.67 | 1323.98 | 1331.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1306.25 | 1323.37 | 1330.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-30 15:00:00 | 1518.40 | 2025-06-11 10:15:00 | 1488.60 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-06-02 12:00:00 | 1519.00 | 2025-06-11 10:15:00 | 1488.60 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-09-04 10:15:00 | 1340.10 | 2025-10-03 09:15:00 | 1352.50 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-09-17 13:15:00 | 1339.10 | 2025-10-03 09:15:00 | 1352.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-22 13:45:00 | 1341.50 | 2025-10-03 09:15:00 | 1352.50 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-09-22 14:30:00 | 1338.20 | 2025-10-03 09:15:00 | 1352.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-09-25 11:30:00 | 1342.80 | 2025-10-16 09:15:00 | 1347.50 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-09-25 12:00:00 | 1339.60 | 2025-10-16 09:15:00 | 1347.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-10-01 09:45:00 | 1338.80 | 2025-10-20 09:15:00 | 1365.80 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-10-01 14:00:00 | 1342.70 | 2025-10-20 09:15:00 | 1365.80 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-10-10 14:00:00 | 1331.60 | 2025-10-20 09:15:00 | 1365.80 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1332.80 | 2025-10-20 09:15:00 | 1365.80 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-02-27 14:15:00 | 1388.10 | 2026-02-27 14:15:00 | 1380.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-02-27 14:45:00 | 1391.00 | 2026-02-27 15:15:00 | 1380.10 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1368.00 | 2026-03-17 14:15:00 | 1299.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 09:15:00 | 1352.00 | 2026-03-19 14:15:00 | 1284.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1368.00 | 2026-03-30 10:15:00 | 1231.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-12 09:15:00 | 1352.00 | 2026-03-30 14:15:00 | 1216.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-22 11:00:00 | 1376.50 | 2026-05-05 13:15:00 | 1307.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 09:45:00 | 1375.00 | 2026-05-06 09:15:00 | 1306.25 | PARTIAL | 0.50 | 5.00% |
