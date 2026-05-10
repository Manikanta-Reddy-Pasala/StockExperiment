# Rainbow Childrens Medicare Ltd. (RAINBOW)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1311.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 2
- **Target hits / Stop hits / Partials:** 4 / 6 / 8
- **Avg / median % per leg:** 4.38% / 5.00%
- **Sum % (uncompounded):** 78.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.55% | -3.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.55% | -3.1% |
| SELL (all) | 16 | 16 | 100.0% | 4 | 4 | 8 | 5.12% | 81.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 16 | 100.0% | 4 | 4 | 8 | 5.12% | 81.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 16 | 88.9% | 4 | 6 | 8 | 4.38% | 78.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 1426.60 | 1385.73 | 1385.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 12:15:00 | 1445.80 | 1392.16 | 1389.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 1519.40 | 1519.95 | 1484.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 1519.40 | 1519.95 | 1484.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1493.00 | 1519.34 | 1487.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 1493.00 | 1519.34 | 1487.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1492.30 | 1519.07 | 1487.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 1485.70 | 1519.07 | 1487.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 1484.40 | 1518.73 | 1487.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 1484.40 | 1518.73 | 1487.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1482.00 | 1518.36 | 1487.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:45:00 | 1481.10 | 1518.36 | 1487.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1479.10 | 1517.47 | 1487.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 1479.10 | 1517.47 | 1487.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1480.10 | 1517.09 | 1487.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:30:00 | 1480.20 | 1517.09 | 1487.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1481.90 | 1515.40 | 1487.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 1478.50 | 1515.40 | 1487.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1466.70 | 1514.91 | 1487.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 1466.70 | 1514.91 | 1487.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 1480.00 | 1513.63 | 1487.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:45:00 | 1477.70 | 1513.63 | 1487.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1484.60 | 1513.34 | 1487.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:15:00 | 1479.90 | 1513.34 | 1487.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1469.30 | 1512.16 | 1487.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 15:15:00 | 1488.90 | 1510.17 | 1486.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:00:00 | 1481.70 | 1509.67 | 1486.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 1462.20 | 1511.75 | 1503.57 | SL hit (close<static) qty=1.00 sl=1465.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 1462.20 | 1511.75 | 1503.57 | SL hit (close<static) qty=1.00 sl=1465.20 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 1450.30 | 1496.51 | 1496.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 1430.50 | 1494.93 | 1495.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1378.30 | 1375.49 | 1412.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:30:00 | 1385.60 | 1375.49 | 1412.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1368.00 | 1349.88 | 1369.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:45:00 | 1368.90 | 1349.88 | 1369.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1378.50 | 1350.17 | 1369.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 1378.50 | 1350.17 | 1369.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 1377.00 | 1350.43 | 1369.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 1375.20 | 1350.43 | 1369.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1354.00 | 1352.88 | 1369.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 1363.10 | 1352.88 | 1369.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1349.90 | 1353.00 | 1369.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:00:00 | 1342.30 | 1352.89 | 1369.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 1340.00 | 1345.54 | 1362.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 10:00:00 | 1346.00 | 1347.20 | 1362.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 11:00:00 | 1346.10 | 1347.19 | 1362.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 09:15:00 | 1278.70 | 1331.69 | 1349.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 09:15:00 | 1278.79 | 1331.69 | 1349.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 1275.18 | 1330.70 | 1348.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 1273.00 | 1330.70 | 1348.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 13:15:00 | 1208.07 | 1301.09 | 1328.37 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 13:15:00 | 1206.00 | 1301.09 | 1328.37 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 13:15:00 | 1211.40 | 1301.09 | 1328.37 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 13:15:00 | 1211.49 | 1301.09 | 1328.37 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1185.90 | 1204.67 | 1235.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 1180.50 | 1204.67 | 1235.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1178.80 | 1203.72 | 1234.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 15:15:00 | 1182.10 | 1202.86 | 1233.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 15:15:00 | 1176.30 | 1200.64 | 1231.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 1121.47 | 1192.56 | 1222.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 1119.86 | 1192.56 | 1222.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 1122.99 | 1192.56 | 1222.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 1117.48 | 1192.56 | 1222.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1173.90 | 1164.67 | 1200.31 | SL hit (close>ema200) qty=0.50 sl=1164.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1173.90 | 1164.67 | 1200.31 | SL hit (close>ema200) qty=0.50 sl=1164.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1173.90 | 1164.67 | 1200.31 | SL hit (close>ema200) qty=0.50 sl=1164.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1173.90 | 1164.67 | 1200.31 | SL hit (close>ema200) qty=0.50 sl=1164.67 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1193.10 | 1163.82 | 1193.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 10:00:00 | 1193.10 | 1163.82 | 1193.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 1195.60 | 1164.14 | 1193.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 10:30:00 | 1195.10 | 1164.14 | 1193.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 1207.30 | 1164.57 | 1193.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:00:00 | 1207.30 | 1164.57 | 1193.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 1302.20 | 1213.02 | 1212.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 10:15:00 | 1331.70 | 1241.28 | 1229.46 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-08 15:15:00 | 1488.90 | 2025-09-12 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-08-11 10:00:00 | 1481.70 | 2025-09-12 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-12-16 11:00:00 | 1342.30 | 2026-01-08 09:15:00 | 1278.70 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2025-12-23 09:15:00 | 1340.00 | 2026-01-08 09:15:00 | 1278.79 | PARTIAL | 0.50 | 4.57% |
| SELL | retest2 | 2025-12-26 10:00:00 | 1346.00 | 2026-01-08 11:15:00 | 1275.18 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2025-12-26 11:00:00 | 1346.10 | 2026-01-08 11:15:00 | 1273.00 | PARTIAL | 0.50 | 5.43% |
| SELL | retest2 | 2025-12-16 11:00:00 | 1342.30 | 2026-01-20 13:15:00 | 1208.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 09:15:00 | 1340.00 | 2026-01-20 13:15:00 | 1206.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-26 10:00:00 | 1346.00 | 2026-01-20 13:15:00 | 1211.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-26 11:00:00 | 1346.10 | 2026-01-20 13:15:00 | 1211.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-05 10:15:00 | 1180.50 | 2026-03-16 09:15:00 | 1121.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:45:00 | 1178.80 | 2026-03-16 09:15:00 | 1119.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 15:15:00 | 1182.10 | 2026-03-16 09:15:00 | 1122.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 15:15:00 | 1176.30 | 2026-03-16 09:15:00 | 1117.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 10:15:00 | 1180.50 | 2026-03-25 09:15:00 | 1173.90 | STOP_HIT | 0.50 | 0.56% |
| SELL | retest2 | 2026-03-06 10:45:00 | 1178.80 | 2026-03-25 09:15:00 | 1173.90 | STOP_HIT | 0.50 | 0.42% |
| SELL | retest2 | 2026-03-06 15:15:00 | 1182.10 | 2026-03-25 09:15:00 | 1173.90 | STOP_HIT | 0.50 | 0.69% |
| SELL | retest2 | 2026-03-09 15:15:00 | 1176.30 | 2026-03-25 09:15:00 | 1173.90 | STOP_HIT | 0.50 | 0.20% |
