# Info Edge (India) Ltd. (NAUKRI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 978.40
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
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 63 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 28
- **Target hits / Stop hits / Partials:** 2 / 33 / 7
- **Avg / median % per leg:** -0.12% / -1.00%
- **Sum % (uncompounded):** -5.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 14 | 33.3% | 2 | 33 | 7 | -0.12% | -5.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 14 | 33.3% | 2 | 33 | 7 | -0.12% | -5.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 14 | 33.3% | 2 | 33 | 7 | -0.12% | -5.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 1498.49 | 1628.78 | 1629.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 1491.00 | 1626.09 | 1627.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 15:15:00 | 1562.40 | 1561.85 | 1589.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 09:15:00 | 1560.38 | 1561.85 | 1589.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 1590.17 | 1560.55 | 1586.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:00:00 | 1590.17 | 1560.55 | 1586.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 1590.12 | 1560.84 | 1586.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 13:45:00 | 1559.06 | 1560.86 | 1586.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 09:15:00 | 1618.20 | 1561.95 | 1586.58 | SL hit (close>static) qty=1.00 sl=1597.60 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 1490.50 | 1429.23 | 1428.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 1502.90 | 1435.79 | 1432.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 1452.00 | 1456.68 | 1444.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 1452.00 | 1456.68 | 1444.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1452.00 | 1456.68 | 1444.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:30:00 | 1438.80 | 1456.68 | 1444.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1467.40 | 1478.27 | 1461.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1467.40 | 1478.27 | 1461.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1463.50 | 1478.12 | 1461.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 1461.90 | 1478.12 | 1461.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1454.90 | 1477.89 | 1461.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 1454.90 | 1477.89 | 1461.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1454.30 | 1477.65 | 1461.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:00:00 | 1454.30 | 1477.65 | 1461.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1452.40 | 1475.29 | 1460.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:45:00 | 1449.60 | 1475.29 | 1460.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 1454.20 | 1475.08 | 1460.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 1452.10 | 1474.94 | 1460.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1457.40 | 1474.76 | 1460.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 1458.90 | 1474.76 | 1460.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 1457.70 | 1474.59 | 1460.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:30:00 | 1459.20 | 1474.59 | 1460.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 1455.90 | 1474.41 | 1460.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 1452.60 | 1474.41 | 1460.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1405.80 | 1473.32 | 1460.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 1405.80 | 1473.32 | 1460.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1422.00 | 1472.81 | 1460.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 1412.00 | 1472.81 | 1460.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1464.40 | 1470.86 | 1460.16 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 1400.70 | 1451.73 | 1451.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 10:15:00 | 1394.50 | 1449.67 | 1450.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1455.90 | 1434.38 | 1442.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 1455.90 | 1434.38 | 1442.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1455.90 | 1434.38 | 1442.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 1455.90 | 1434.38 | 1442.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1450.70 | 1434.54 | 1442.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:45:00 | 1458.40 | 1434.54 | 1442.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1461.20 | 1434.91 | 1442.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 1461.20 | 1434.91 | 1442.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1422.50 | 1383.25 | 1406.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 1422.50 | 1383.25 | 1406.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1431.70 | 1383.73 | 1406.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 1432.80 | 1383.73 | 1406.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1419.10 | 1385.06 | 1406.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 1405.50 | 1385.23 | 1406.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 1407.00 | 1385.67 | 1405.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:45:00 | 1407.20 | 1385.87 | 1405.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 13:45:00 | 1407.90 | 1386.29 | 1405.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1407.20 | 1386.50 | 1405.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:45:00 | 1407.30 | 1386.50 | 1405.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1404.30 | 1386.67 | 1405.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1391.40 | 1386.67 | 1405.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1393.60 | 1386.74 | 1405.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 15:00:00 | 1383.10 | 1387.31 | 1404.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 1382.30 | 1387.31 | 1404.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:45:00 | 1381.70 | 1387.12 | 1404.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 14:15:00 | 1335.22 | 1380.40 | 1398.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 14:15:00 | 1336.65 | 1380.40 | 1398.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 14:15:00 | 1336.84 | 1380.40 | 1398.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 14:15:00 | 1337.51 | 1380.40 | 1398.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1388.20 | 1373.95 | 1393.00 | SL hit (close>ema200) qty=0.50 sl=1373.95 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-05 13:45:00 | 1559.06 | 2025-02-06 09:15:00 | 1618.20 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-02-07 11:45:00 | 1579.02 | 2025-02-11 14:15:00 | 1500.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 11:45:00 | 1579.02 | 2025-02-25 09:15:00 | 1421.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-22 09:30:00 | 1405.50 | 2025-09-04 14:15:00 | 1335.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 11:15:00 | 1407.00 | 2025-09-04 14:15:00 | 1336.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 11:45:00 | 1407.20 | 2025-09-04 14:15:00 | 1336.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 13:45:00 | 1407.90 | 2025-09-04 14:15:00 | 1337.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 09:30:00 | 1405.50 | 2025-09-10 09:15:00 | 1388.20 | STOP_HIT | 0.50 | 1.23% |
| SELL | retest2 | 2025-08-25 11:15:00 | 1407.00 | 2025-09-10 09:15:00 | 1388.20 | STOP_HIT | 0.50 | 1.34% |
| SELL | retest2 | 2025-08-25 11:45:00 | 1407.20 | 2025-09-10 09:15:00 | 1388.20 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2025-08-25 13:45:00 | 1407.90 | 2025-09-10 09:15:00 | 1388.20 | STOP_HIT | 0.50 | 1.40% |
| SELL | retest2 | 2025-08-28 15:00:00 | 1383.10 | 2025-09-22 10:15:00 | 1397.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-08-29 09:15:00 | 1382.30 | 2025-09-22 10:15:00 | 1397.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-08-29 12:45:00 | 1381.70 | 2025-09-22 10:15:00 | 1397.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-09-10 10:30:00 | 1383.20 | 2025-09-22 10:15:00 | 1397.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-10 13:30:00 | 1378.70 | 2025-09-22 11:15:00 | 1414.40 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-09-10 14:15:00 | 1378.80 | 2025-09-22 11:15:00 | 1414.40 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-09-11 09:15:00 | 1371.90 | 2025-09-22 11:15:00 | 1414.40 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-09-17 11:15:00 | 1377.20 | 2025-09-22 11:15:00 | 1414.40 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-10-09 10:15:00 | 1343.80 | 2025-10-24 09:15:00 | 1409.50 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2025-10-09 11:30:00 | 1343.20 | 2025-10-24 09:15:00 | 1409.50 | STOP_HIT | 1.00 | -4.94% |
| SELL | retest2 | 2025-10-09 12:45:00 | 1342.50 | 2025-10-24 09:15:00 | 1409.50 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2025-10-09 14:00:00 | 1341.80 | 2025-10-24 09:15:00 | 1409.50 | STOP_HIT | 1.00 | -5.05% |
| SELL | retest2 | 2025-11-03 09:15:00 | 1358.10 | 2025-12-01 14:15:00 | 1368.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-11-13 09:15:00 | 1348.90 | 2025-12-01 14:15:00 | 1368.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-20 10:45:00 | 1365.80 | 2025-12-01 14:15:00 | 1368.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-11-20 14:15:00 | 1367.00 | 2025-12-01 14:15:00 | 1368.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-11-24 15:15:00 | 1338.00 | 2025-12-04 10:15:00 | 1397.80 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2025-11-25 09:30:00 | 1340.50 | 2025-12-04 10:15:00 | 1397.80 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2025-11-25 10:45:00 | 1339.40 | 2025-12-04 10:15:00 | 1397.80 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2025-11-26 13:00:00 | 1340.50 | 2025-12-04 10:15:00 | 1397.80 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1351.30 | 2025-12-23 13:15:00 | 1367.70 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-12-16 10:15:00 | 1353.40 | 2025-12-23 13:15:00 | 1367.70 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-12-17 10:00:00 | 1351.80 | 2025-12-23 13:15:00 | 1367.70 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-23 10:45:00 | 1358.40 | 2025-12-23 13:15:00 | 1367.70 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-12-29 13:15:00 | 1352.60 | 2025-12-29 14:15:00 | 1359.40 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-12-30 09:15:00 | 1346.60 | 2026-01-02 10:15:00 | 1362.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1348.00 | 2026-01-07 09:15:00 | 1374.30 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1351.00 | 2026-01-21 09:15:00 | 1283.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1351.00 | 2026-01-21 12:15:00 | 1339.60 | STOP_HIT | 0.50 | 0.84% |
| SELL | retest2 | 2026-01-22 10:15:00 | 1326.80 | 2026-01-27 13:15:00 | 1260.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:15:00 | 1326.80 | 2026-02-04 09:15:00 | 1194.12 | TARGET_HIT | 0.50 | 10.00% |
