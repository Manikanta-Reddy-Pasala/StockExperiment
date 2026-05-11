# United Spirits Ltd. (UNITDSPR)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1284.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 8 |
| TARGET_HIT | 6 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 14 / 19
- **Target hits / Stop hits / Partials:** 6 / 19 / 8
- **Avg / median % per leg:** 2.11% / -0.59%
- **Sum % (uncompounded):** 69.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.33% | -5.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.33% | -5.3% |
| SELL (all) | 29 | 14 | 48.3% | 6 | 15 | 8 | 2.58% | 74.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 14 | 48.3% | 6 | 15 | 8 | 2.58% | 74.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 33 | 14 | 42.4% | 6 | 19 | 8 | 2.11% | 69.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 1452.20 | 1517.82 | 1517.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 1451.80 | 1517.17 | 1517.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 12:15:00 | 1486.30 | 1480.26 | 1496.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 13:00:00 | 1486.30 | 1480.26 | 1496.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 1500.00 | 1480.54 | 1496.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:45:00 | 1488.05 | 1480.82 | 1496.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 1485.75 | 1480.82 | 1496.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 11:00:00 | 1486.95 | 1480.88 | 1496.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:30:00 | 1484.50 | 1480.91 | 1496.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 1492.10 | 1481.02 | 1496.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:30:00 | 1492.65 | 1481.02 | 1496.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 14:15:00 | 1413.65 | 1475.96 | 1492.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 14:15:00 | 1412.60 | 1475.96 | 1492.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 15:15:00 | 1411.46 | 1475.36 | 1491.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 15:15:00 | 1410.27 | 1475.36 | 1491.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-17 09:15:00 | 1339.24 | 1448.07 | 1473.77 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 12:15:00 | 1521.00 | 1413.16 | 1412.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 1536.30 | 1424.11 | 1418.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 1529.10 | 1531.58 | 1497.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 13:00:00 | 1529.10 | 1531.58 | 1497.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 1513.50 | 1529.51 | 1499.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 15:00:00 | 1518.40 | 1529.40 | 1499.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:00:00 | 1519.00 | 1529.26 | 1500.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 10:15:00 | 1488.60 | 1553.01 | 1519.75 | SL hit (close<static) qty=1.00 sl=1498.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 1452.50 | 1498.82 | 1499.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 1441.60 | 1497.33 | 1498.30 | Break + close below crossover candle low |
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
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:30:00 | 1342.80 | 1328.20 | 1347.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:00:00 | 1339.60 | 1328.20 | 1347.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 1338.80 | 1324.06 | 1342.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 14:00:00 | 1342.70 | 1324.72 | 1342.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 1345.90 | 1324.93 | 1342.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:45:00 | 1345.50 | 1324.93 | 1342.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1347.50 | 1325.15 | 1342.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 1349.40 | 1325.15 | 1342.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1352.50 | 1325.43 | 1342.82 | SL hit (close>static) qty=1.00 sl=1352.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 10:15:00 | 1459.90 | 1348.20 | 1348.14 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 11:15:00 | 1318.50 | 1401.31 | 1401.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 1281.10 | 1377.35 | 1388.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 1360.90 | 1360.18 | 1376.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 14:00:00 | 1360.90 | 1360.18 | 1376.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1374.10 | 1357.51 | 1373.08 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-02-19 11:15:00)

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

### Cycle 7 — SELL (started 2026-03-04 14:15:00)

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


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-03 09:45:00 | 1488.05 | 2025-02-06 14:15:00 | 1413.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-03 10:15:00 | 1485.75 | 2025-02-06 14:15:00 | 1412.60 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-02-03 11:00:00 | 1486.95 | 2025-02-06 15:15:00 | 1411.46 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2025-02-04 10:30:00 | 1484.50 | 2025-02-06 15:15:00 | 1410.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-03 09:45:00 | 1488.05 | 2025-02-17 09:15:00 | 1339.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-03 10:15:00 | 1485.75 | 2025-02-17 09:15:00 | 1338.26 | TARGET_HIT | 0.50 | 9.93% |
| SELL | retest2 | 2025-02-03 11:00:00 | 1486.95 | 2025-02-18 10:15:00 | 1337.17 | TARGET_HIT | 0.50 | 10.07% |
| SELL | retest2 | 2025-02-04 10:30:00 | 1484.50 | 2025-02-18 10:15:00 | 1336.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-24 13:00:00 | 1387.35 | 2025-04-02 09:15:00 | 1423.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-03-24 13:45:00 | 1389.35 | 2025-04-02 09:15:00 | 1423.00 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-03-24 14:15:00 | 1388.00 | 2025-04-02 09:15:00 | 1423.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-03-25 10:00:00 | 1386.40 | 2025-04-02 09:15:00 | 1423.00 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-04-07 10:00:00 | 1382.90 | 2025-04-07 13:15:00 | 1402.60 | STOP_HIT | 1.00 | -1.42% |
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
